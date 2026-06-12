import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.backtesting.flow_validation import (
    FlowValidationParams,
    build_flow_validation_report,
    evaluate_market_flow_features,
)
from backend.storage.models import (
    Base,
    Bucket,
    City,
    Event,
    MarketFlowFeature,
    MarketSnapshot,
)


def _run(coro):
    return asyncio.run(coro)


def test_flow_validation_empty_report_stays_shadow_only():
    report = build_flow_validation_report(
        [],
        params=FlowValidationParams(min_samples=2),
    )
    assert report["n_flow_rows"] == 0
    assert report["promotion"]["recommendation"] == "keep_shadow_only"
    assert "price_move_samples_below_min:0<2" in report["promotion"]["blockers"]


def test_flow_validation_detects_predictive_signed_flow(tmp_path):
    async def scenario():
        db_path = tmp_path / "flow_validation.db"
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        session_factory = async_sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        t0 = datetime(2026, 6, 11, 12, tzinfo=timezone.utc)
        async with session_factory() as session:
            city = City(city_slug="atlanta", display_name="Atlanta", enabled=True)
            session.add(city)
            await session.flush()
            event = Event(city_id=city.id, date_et="2026-06-11", winning_bucket_idx=0)
            session.add(event)
            await session.flush()
            buckets = [
                Bucket(event_id=event.id, bucket_idx=0, label="hot"),
                Bucket(event_id=event.id, bucket_idx=1, label="cold"),
                Bucket(event_id=event.id, bucket_idx=2, label="flat-hot"),
                Bucket(event_id=event.id, bucket_idx=3, label="flat-cold"),
            ]
            session.add_all(buckets)
            await session.flush()
            cases = [
                (buckets[0].id, 100.0, 0.8, 0.40, 0.47),
                (buckets[1].id, -100.0, -0.8, 0.60, 0.53),
                (buckets[2].id, 50.0, 0.5, 0.30, 0.34),
                (buckets[3].id, -50.0, -0.5, 0.70, 0.66),
            ]
            for bucket_id, signed, imbalance, now_price, future_price in cases:
                session.add_all([
                    MarketSnapshot(
                        bucket_id=bucket_id,
                        fetched_at=t0 - timedelta(minutes=1),
                        yes_mid=now_price,
                    ),
                    MarketSnapshot(
                        bucket_id=bucket_id,
                        fetched_at=t0 + timedelta(minutes=15),
                        yes_mid=future_price,
                    ),
                    MarketFlowFeature(
                        bucket_id=bucket_id,
                        computed_at=t0,
                        window_minutes=15,
                        signed_net_notional=signed,
                        buy_notional=max(signed, 0.0),
                        sell_notional=max(-signed, 0.0),
                        imbalance=imbalance,
                        vpin=abs(imbalance),
                        toxicity_score=abs(imbalance),
                        top_wallet_weighted_flow=signed,
                        direction_source="data_api_side",
                        direction_confidence=0.9,
                    ),
                ])
            await session.commit()

        async with session_factory() as session:
            report = await evaluate_market_flow_features(
                session,
                FlowValidationParams(days_back=1, min_samples=4, max_rows=20),
                as_of=t0 + timedelta(hours=1),
            )

        await engine.dispose()
        return report

    report = _run(scenario())
    assert report["n_price_move_samples"] == 4
    signed = next(
        item for item in report["directional_scores"]
        if item["score"] == "signed_net_notional"
    )
    assert signed["auc"] == 1.0
    assert signed["directional"]["hit_rate"] == 1.0
    assert report["promotion"]["recommendation"] == "candidate_for_shadow_promotion"
    assert report["promotion"]["allowed_for_execution"] is False


def test_flow_validation_blocks_low_sample_promotion():
    samples = [
        {
            "signed_net_notional": 100.0,
            "imbalance": 1.0,
            "top_wallet_weighted_flow": 100.0,
            "vpin": 1.0,
            "toxicity_score": 1.0,
            "price_delta": 0.02,
            "next_up": 1,
            "final_yes": 1,
        }
    ]
    report = build_flow_validation_report(
        samples,
        params=FlowValidationParams(min_samples=5),
    )
    assert report["promotion"]["recommendation"] == "keep_shadow_only"
    assert "price_move_samples_below_min:1<5" in report["promotion"]["blockers"]
