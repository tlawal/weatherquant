import asyncio
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.storage.models import Base, Bucket, City, Event, ModelSnapshot, Signal
from backend.storage.repos import get_dashboard_signal_rows


def _run(coro):
    return asyncio.run(coro)


def test_dashboard_signal_rows_only_use_latest_snapshot(tmp_path):
    async def scenario():
        db_path = tmp_path / "dashboard.db"
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        session_factory = async_sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            city = City(
                city_slug="atlanta",
                display_name="Atlanta",
                enabled=True,
                metar_station="KATL",
            )
            session.add(city)
            await session.flush()
            event = Event(city_id=city.id, date_et="2026-06-11")
            session.add(event)
            await session.flush()
            bucket = Bucket(event_id=event.id, bucket_idx=0, label="80-81", low_f=80, high_f=81)
            session.add(bucket)
            await session.flush()
            old_model = ModelSnapshot(
                event_id=event.id,
                mu=80.0,
                sigma=1.0,
                probs_json="[0.4]",
                computed_at=datetime(2026, 6, 11, 12, tzinfo=timezone.utc),
            )
            new_model = ModelSnapshot(
                event_id=event.id,
                mu=81.0,
                sigma=1.0,
                probs_json="[0.6]",
                computed_at=datetime(2026, 6, 11, 13, tzinfo=timezone.utc),
            )
            session.add_all([old_model, new_model])
            await session.flush()
            session.add_all([
                Signal(
                    bucket_id=bucket.id,
                    model_snapshot_id=old_model.id,
                    model_prob=0.4,
                    mkt_prob=0.3,
                    raw_edge=0.1,
                    exec_cost=0.01,
                    true_edge=0.09,
                    computed_at=datetime(2026, 6, 11, 12, 1, tzinfo=timezone.utc),
                ),
                Signal(
                    bucket_id=bucket.id,
                    model_snapshot_id=new_model.id,
                    model_prob=0.6,
                    mkt_prob=0.3,
                    raw_edge=0.3,
                    exec_cost=0.01,
                    true_edge=0.29,
                    computed_at=datetime(2026, 6, 11, 13, 1, tzinfo=timezone.utc),
                ),
            ])
            await session.commit()

        async with session_factory() as session:
            rows = await get_dashboard_signal_rows(
                session,
                date_et="2026-06-11",
                limit=20,
            )

        await engine.dispose()
        return rows

    rows = _run(scenario())
    assert len(rows) == 1
    assert rows[0]["city"].city_slug == "atlanta"
    assert rows[0]["bucket"].label == "80-81"
    assert rows[0]["signal"].model_prob == 0.6
