import asyncio
import json
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.requests import Request

import backend.storage.db as storage_db
import web.routes as web_routes
from backend.storage.models import (
    Base,
    Bucket,
    City,
    Event,
    MarketSnapshot,
    ModelSnapshot,
    Signal,
)


def _run(coro):
    return asyncio.run(coro)


async def _setup_test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "web_routes_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(storage_db, "_engine", engine)
    monkeypatch.setattr(storage_db, "_session_factory", session_factory)
    return engine, session_factory


async def _create_city(session_factory, slug: str = "la") -> City:
    async with session_factory() as session:
        city = City(
            city_slug=slug,
            display_name="Los Angeles",
            metar_station="KCQT",
            enabled=True,
            is_us=True,
            unit="F",
            tz="America/Los_Angeles",
            lat=34.05,
            lon=-118.24,
        )
        session.add(city)
        await session.commit()
        await session.refresh(city)
        return city


def test_city_page_uses_signal_model_prob_over_raw_snapshot_prob(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    async def _empty_bins(city_id: int, days_back: int = 30):
        return []

    async def _stub_settlement_result(*args, **kwargs):
        return {"high_f": 69.1, "source_used": "wu_history", "obs_time": None}

    monkeypatch.setattr(web_routes, "get_reliability_metrics", _empty_bins)
    monkeypatch.setattr(web_routes, "_resolve_realized_high_with_source", _stub_settlement_result)

    date_et = "2026-04-16"
    computed_at = datetime(2026, 4, 16, 22, 5, tzinfo=timezone.utc)

    async def seed():
        async with session_factory() as session:
            event = Event(
                city_id=city.id,
                date_et=date_et,
                status="ok",
                forecast_quality="ok",
                gamma_slug=f"{city.city_slug}-{date_et}",
            )
            session.add(event)
            await session.flush()

            bucket_a = Bucket(event_id=event.id, bucket_idx=0, label="68-69", low_f=68.0, high_f=69.0)
            bucket_b = Bucket(event_id=event.id, bucket_idx=1, label="70-71", low_f=70.0, high_f=71.0)
            session.add_all([bucket_a, bucket_b])
            await session.flush()

            snapshot = ModelSnapshot(
                event_id=event.id,
                computed_at=computed_at,
                mu=69.8,
                sigma=1.2,
                probs_json=json.dumps([0.12, 0.88]),
                inputs_json=json.dumps({}),
                forecast_quality="ok",
            )
            session.add(snapshot)
            await session.flush()

            session.add_all(
                [
                    Signal(
                        bucket_id=bucket_a.id,
                        model_snapshot_id=snapshot.id,
                        computed_at=computed_at,
                        model_prob=0.93,
                        mkt_prob=0.50,
                        raw_edge=0.43,
                        exec_cost=0.01,
                        true_edge=0.42,
                        reason_json="{}",
                        gate_failures_json="[]",
                    ),
                    Signal(
                        bucket_id=bucket_b.id,
                        model_snapshot_id=snapshot.id,
                        computed_at=computed_at,
                        model_prob=0.07,
                        mkt_prob=0.50,
                        raw_edge=-0.43,
                        exec_cost=0.01,
                        true_edge=-0.44,
                        reason_json="{}",
                        gate_failures_json="[]",
                    ),
                    MarketSnapshot(
                        bucket_id=bucket_a.id,
                        fetched_at=computed_at,
                        yes_bid=0.48,
                        yes_ask=0.52,
                        yes_mid=0.50,
                        yes_bid_depth=250.0,
                        yes_ask_depth=250.0,
                        spread=0.04,
                    ),
                    MarketSnapshot(
                        bucket_id=bucket_b.id,
                        fetched_at=computed_at,
                        yes_bid=0.48,
                        yes_ask=0.52,
                        yes_mid=0.50,
                        yes_bid_depth=250.0,
                        yes_ask_depth=250.0,
                        spread=0.04,
                    ),
                ]
            )

            await session.commit()

    async def exercise():
        await seed()
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": f"/city/{city.city_slug}",
                "headers": [],
                "query_string": b"",
            }
        )
        return await web_routes.city_detail(request, city.city_slug, date_et)

    response = _run(exercise())

    buckets = response.context["buckets"]
    model_ctx = response.context["model"]

    assert buckets[0]["model_prob"] == 0.93
    assert buckets[1]["model_prob"] == 0.07
    assert json.loads(model_ctx["probs_json"]) == [0.93, 0.07]
    assert json.loads(model_ctx["raw_probs_json"]) == [0.12, 0.88]

    _run(engine.dispose())
