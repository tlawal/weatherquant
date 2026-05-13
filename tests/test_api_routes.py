import asyncio
import inspect
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import backend.api.routes as api_routes
import backend.notifications.telegram as telegram_mod
import backend.storage.db as storage_db
import web.routes as web_routes
from backend.execution.position_sync import sync_positions_from_chain
from backend.storage.models import (
    Base,
    AuditLog,
    Bucket,
    City,
    Event,
    ForecastObs,
    Fill,
    MetarObs,
    ModelSnapshot,
    Order,
    Position,
    Signal,
    WorkerHeartbeat,
)
from backend.tz_utils import city_local_date


def _run(coro):
    return asyncio.run(coro)


async def _setup_test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "api_routes_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(storage_db, "_engine", engine)
    monkeypatch.setattr(storage_db, "_session_factory", session_factory)
    return engine, session_factory


async def _create_city(session_factory, slug: str = "atlanta") -> City:
    async with session_factory() as session:
        city = City(
            city_slug=slug,
            display_name="Atlanta",
            metar_station="KATL",
            enabled=True,
            is_us=True,
            unit="F",
            tz="America/New_York",
        )
        session.add(city)
        await session.commit()
        await session.refresh(city)
        return city


def test_unredeemed_wins_skips_events_without_positions(tmp_path, monkeypatch):
    """Regression for the phantom $0 unredeemed-wins panel: a resolved event
    with condition_ids but no Position rows must not surface as a claimable
    winning."""
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    async def seed():
        async with session_factory() as session:
            event = Event(
                city_id=city.id,
                date_et="2026-04-10",
                status="ok",
                trading_enabled=True,
                resolved_at=datetime.utcnow(),
                winning_bucket_idx=1,
            )
            session.add(event)
            await session.flush()

            # Two buckets, both with condition_ids (resolvable), no positions.
            for idx, (lo, hi) in enumerate([(76.0, 77.0), (78.0, 79.0)]):
                session.add(
                    Bucket(
                        event_id=event.id,
                        bucket_idx=idx,
                        label=f"{int(lo)}-{int(hi)}°F",
                        low_f=lo,
                        high_f=hi,
                        yes_token_id=f"yes-{idx}",
                        no_token_id=f"no-{idx}",
                        condition_id=f"cond-{idx}",
                    )
                )
            await session.commit()

    _run(seed())
    result = _run(api_routes.unredeemed_wins())
    assert result == {"unredeemed": []}

    _run(engine.dispose())


def test_unredeemed_wins_includes_events_with_winning_position(tmp_path, monkeypatch):
    """Counterpart: when a real winning Position exists, the event surfaces."""
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    async def seed():
        async with session_factory() as session:
            event = Event(
                city_id=city.id,
                date_et="2026-04-10",
                status="ok",
                trading_enabled=True,
                resolved_at=datetime.utcnow(),
                winning_bucket_idx=1,
            )
            session.add(event)
            await session.flush()

            winning_bucket = None
            for idx, (lo, hi) in enumerate([(76.0, 77.0), (78.0, 79.0)]):
                b = Bucket(
                    event_id=event.id,
                    bucket_idx=idx,
                    label=f"{int(lo)}-{int(hi)}°F",
                    low_f=lo,
                    high_f=hi,
                    yes_token_id=f"yes-{idx}",
                    no_token_id=f"no-{idx}",
                    condition_id=f"cond-{idx}",
                )
                session.add(b)
                if idx == 1:
                    winning_bucket = b
            await session.flush()

            assert winning_bucket is not None
            session.add(
                Position(
                    bucket_id=winning_bucket.id,
                    side="YES",
                    net_qty=10.0,
                    avg_cost=0.42,
                    realized_pnl=0.0,
                    unrealized_pnl=0.0,
                    last_mkt_price=0.99,
                )
            )
            await session.commit()

    _run(seed())
    result = _run(api_routes.unredeemed_wins())
    assert len(result["unredeemed"]) == 1
    entry = result["unredeemed"][0]
    assert entry["winning_bucket_idx"] == 1
    assert entry["total_expected_payout"] == 10.0

    _run(engine.dispose())


def test_sync_positions_matches_yes_no_and_condition_ids(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    async def seed():
        async with session_factory() as session:
            event = Event(city_id=city.id, date_et="2026-05-07", status="ok")
            session.add(event)
            await session.flush()
            session.add_all(
                [
                    Bucket(
                        event_id=event.id,
                        bucket_idx=0,
                        label="Atlanta 80-81F",
                        yes_token_id="yes-token",
                        no_token_id="no-token",
                        condition_id="cond-token",
                    ),
                    Bucket(
                        event_id=event.id,
                        bucket_idx=1,
                        label="Atlanta 82-83F",
                        yes_token_id="yes-existing",
                        no_token_id="no-existing",
                        condition_id="cond-existing",
                    ),
                ]
            )
            await session.flush()
            session.add(
                Position(
                    bucket_id=2,
                    side="yes",
                    net_qty=2.0,
                    avg_cost=0.20,
                    last_mkt_price=0.20,
                )
            )
            await session.commit()

    _run(seed())
    api_positions = [
        {
            "asset": "no-token",
            "conditionId": "cond-token",
            "outcome": "No",
            "size": 7,
            "avgPrice": 0.31,
            "curPrice": 0.36,
            "title": "Will the highest temperature in Atlanta be 80-81F?",
        },
        {
            "asset": "missing-asset",
            "conditionId": "cond-existing",
            "outcome": "Yes",
            "size": 5,
            "avgPrice": 0.44,
            "curPrice": 0.40,
            "title": "Will the highest temperature in Atlanta be 82-83F?",
        },
    ]

    res = _run(sync_positions_from_chain(api_positions=api_positions))
    assert res["ok"] is True
    assert res["synced"] == 2

    async def check():
        from sqlalchemy import select

        async with session_factory() as session:
            p_new = (await session.execute(
                select(Position).where(Position.bucket_id == 1)
            )).scalar_one()
            p_existing = (await session.execute(
                select(Position).where(Position.bucket_id == 2)
            )).scalar_one()
            return p_existing, p_new

    p_existing, p_new = _run(check())
    assert p_new.side == "no"
    assert p_new.net_qty == 7
    assert p_new.avg_cost == 0.31
    assert p_existing.net_qty == 5
    assert p_existing.avg_cost == 0.44

    _run(engine.dispose())


def test_manual_market_trade_triggers_position_sync(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))
    calls = {"sync": 0, "retry": 0}

    async def seed():
        async with session_factory() as session:
            event = Event(city_id=city.id, date_et=city_local_date(city), status="ok")
            session.add(event)
            await session.flush()
            session.add(
                Bucket(
                    event_id=event.id,
                    bucket_idx=0,
                    label="Will the highest temperature in Atlanta be 80-81F?",
                    yes_token_id="yes-token",
                    no_token_id="no-token",
                    condition_id="cond-token",
                )
            )
            await session.commit()

    class FakeClob:
        can_trade = True

        async def get_balance(self):
            return 10.0

    async def fake_execute_signal(*args, **kwargs):
        return {"status": "filled"}

    async def fake_sync_positions_from_chain(*args, **kwargs):
        calls["sync"] += 1
        return {"ok": True, "synced": 1}

    async def fake_schedule_position_sync_retries(*args, **kwargs):
        calls["retry"] += 1

    _run(seed())
    monkeypatch.setattr("backend.ingestion.polymarket_clob.get_clob", lambda: FakeClob())
    monkeypatch.setattr("backend.execution.trader.execute_signal", fake_execute_signal)
    monkeypatch.setattr("backend.execution.position_sync.sync_positions_from_chain", fake_sync_positions_from_chain)
    monkeypatch.setattr("backend.execution.position_sync.schedule_position_sync_retries", fake_schedule_position_sync_retries)

    res = _run(api_routes.manual_trade(api_routes.ManualTradeRequest(
        city_slug=city.city_slug,
        bucket_id=1,
        side="buy_yes",
        qty=5,
        order_type="market",
    ), actor="test"))

    assert res["status"] == "filled"
    assert res["position_sync"] == {"ok": True, "synced": 1}
    assert calls["sync"] == 1

    _run(engine.dispose())


def test_manual_market_open_trade_sends_late_fill_alert_from_sync(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))
    sent = []

    async def seed():
        async with session_factory() as session:
            event = Event(city_id=city.id, date_et=city_local_date(city), status="ok")
            session.add(event)
            await session.flush()
            bucket = Bucket(
                event_id=event.id,
                bucket_idx=0,
                label="Will the highest temperature in Atlanta be 80-81F?",
                yes_token_id="yes-token",
                no_token_id="no-token",
                condition_id="cond-token",
            )
            session.add(bucket)
            await session.flush()
            order = Order(
                bucket_id=bucket.id,
                side="buy_yes",
                qty=5,
                limit_price=0.25,
                status="open",
                gates_json="{}",
            )
            session.add(order)
            await session.commit()
            return bucket.id, order.id

    class FakeClob:
        can_trade = True

        async def get_balance(self):
            return 10.0

    bucket_id, order_id = _run(seed())

    async def fake_execute_signal(*args, **kwargs):
        return {"status": "open", "order_id": order_id}

    async def fake_sync_positions_from_chain(*args, **kwargs):
        return {
            "ok": True,
            "synced": 1,
            "corrections": [{
                "bucket_id": bucket_id,
                "old_qty": 0,
                "new_qty": 5,
                "avg_price": 0.25,
            }],
        }

    async def fake_notify_trade_filled(**kwargs):
        sent.append(kwargs)
        return True

    monkeypatch.setattr("backend.ingestion.polymarket_clob.get_clob", lambda: FakeClob())
    monkeypatch.setattr("backend.execution.trader.execute_signal", fake_execute_signal)
    monkeypatch.setattr("backend.execution.position_sync.sync_positions_from_chain", fake_sync_positions_from_chain)
    monkeypatch.setattr("backend.notifications.telegram.notify_trade_filled", fake_notify_trade_filled)

    res = _run(api_routes.manual_trade(api_routes.ManualTradeRequest(
        city_slug=city.city_slug,
        bucket_id=bucket_id,
        side="buy_yes",
        qty=5,
        order_type="market",
    ), actor="test"))

    assert res["status"] == "open"
    assert res["late_fill_alert_sent"] is True
    assert len(sent) == 1
    assert sent[0]["shares"] == 5
    assert sent[0]["price"] == 0.25

    async def check():
        async with session_factory() as session:
            order = await session.get(Order, order_id)
            fills = (await session.execute(
                select(Fill).where(Fill.order_id == order_id)
            )).scalars().all()
            return order, fills

    order, fills = _run(check())
    assert order.status == "filled"
    assert order.fill_qty == 5
    assert order.fill_price == 0.25
    assert len(fills) == 1

    _run(engine.dispose())


def test_late_fill_alert_ignores_non_matching_sync_correction(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))
    sent = []

    async def seed():
        async with session_factory() as session:
            event = Event(city_id=city.id, date_et=city_local_date(city), status="ok")
            session.add(event)
            await session.flush()
            bucket = Bucket(event_id=event.id, bucket_idx=0, label="A", condition_id="cond")
            session.add(bucket)
            await session.flush()
            order = Order(bucket_id=bucket.id, side="buy_yes", qty=5, limit_price=0.25, status="open")
            session.add(order)
            await session.commit()
            return bucket.id, order.id

    bucket_id, order_id = _run(seed())

    async def fake_notify_trade_filled(**kwargs):
        sent.append(kwargs)
        return True

    monkeypatch.setattr("backend.notifications.telegram.notify_trade_filled", fake_notify_trade_filled)
    alerted = _run(api_routes._notify_late_manual_fill_if_synced(
        sync_res={"corrections": [{"bucket_id": bucket_id + 999, "old_qty": 0, "new_qty": 5, "avg_price": 0.25}]},
        order_id=order_id,
        bucket_id=bucket_id,
        city_slug=city.city_slug,
        bucket_label="A",
        side="BUY",
        edge=0.1,
    ))

    assert alerted is False
    assert sent == []

    _run(engine.dispose())


def test_late_fill_alert_is_idempotent_for_filled_order(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))
    sent = []

    async def seed():
        async with session_factory() as session:
            event = Event(city_id=city.id, date_et=city_local_date(city), status="ok")
            session.add(event)
            await session.flush()
            bucket = Bucket(event_id=event.id, bucket_idx=0, label="A", condition_id="cond")
            session.add(bucket)
            await session.flush()
            order = Order(bucket_id=bucket.id, side="buy_yes", qty=5, limit_price=0.25, status="filled", fill_qty=5, fill_price=0.25)
            session.add(order)
            await session.flush()
            session.add(Fill(order_id=order.id, qty=5, price=0.25))
            await session.commit()
            return bucket.id, order.id

    bucket_id, order_id = _run(seed())

    async def fake_notify_trade_filled(**kwargs):
        sent.append(kwargs)
        return True

    monkeypatch.setattr("backend.notifications.telegram.notify_trade_filled", fake_notify_trade_filled)
    alerted = _run(api_routes._notify_late_manual_fill_if_synced(
        sync_res={"corrections": [{"bucket_id": bucket_id, "old_qty": 0, "new_qty": 5, "avg_price": 0.25}]},
        order_id=order_id,
        bucket_id=bucket_id,
        city_slug=city.city_slug,
        bucket_label="A",
        side="BUY",
        edge=0.1,
    ))

    assert alerted is False
    assert sent == []

    _run(engine.dispose())


def test_notify_trade_filled_returns_send_result(monkeypatch):
    calls = []

    async def fake_send_telegram(message, parse_mode="HTML"):
        calls.append((message, parse_mode))
        return True

    monkeypatch.setattr(telegram_mod, "send_telegram", fake_send_telegram)
    ok = _run(telegram_mod.notify_trade_filled(
        city_slug="atlanta",
        bucket_label="80-81F",
        side="BUY",
        shares=5,
        price=0.25,
        edge=0.1,
    ))

    assert ok is True
    assert calls and "atlanta" in calls[0][0]


def test_redemptions_can_include_api_position_without_admin_sync(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    async def seed():
        async with session_factory() as session:
            event = Event(city_id=city.id, date_et="2026-05-07", status="ok")
            session.add(event)
            await session.flush()
            session.add(
                Bucket(
                    event_id=event.id,
                    bucket_idx=0,
                    label="Will the highest temperature in Atlanta be 80-81F?",
                    yes_token_id="yes-token",
                    no_token_id="no-token",
                    condition_id="cond-token",
                )
            )
            await session.commit()

    class FakeResp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return {"result": "0x1"}

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            return FakeResp()

        def post(self, *args, **kwargs):
            return FakeResp()

    async def fake_fetch_wallet_api_positions(*args, **kwargs):
        return ([{
            "asset": "yes-token",
            "conditionId": "cond-token",
            "outcome": "Yes",
            "size": 3,
            "avgPrice": 0.25,
            "curPrice": 0.30,
            "title": "Will the highest temperature in Atlanta be 80-81F?",
        }], "0xwallet")

    _run(seed())
    monkeypatch.setattr(api_routes, "_fetch_wallet_api_positions", fake_fetch_wallet_api_positions)
    monkeypatch.setattr("aiohttp.ClientSession", FakeSession)

    res = _run(api_routes.redemptions_live())
    bucket = res["events"][0]["buckets"][0]
    assert bucket["net_qty"] == 3
    assert bucket["avg_cost"] == 0.25
    assert bucket["entry_type"] == "MANUAL"

    _run(engine.dispose())


def test_redemptions_marks_db_only_position_missing_on_chain(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    async def seed():
        async with session_factory() as session:
            event = Event(city_id=city.id, date_et="2026-05-07", status="ok")
            session.add(event)
            await session.flush()
            bucket = Bucket(
                event_id=event.id,
                bucket_idx=0,
                label="Will the highest temperature in Atlanta be 70-71F?",
                yes_token_id="yes-token",
                no_token_id="no-token",
                condition_id="cond-stale",
            )
            session.add(bucket)
            await session.flush()
            session.add(Position(
                bucket_id=bucket.id,
                side="yes",
                net_qty=2.0,
                avg_cost=0.25,
                last_mkt_price=0.20,
                entry_type="MANUAL",
            ))
            await session.commit()

    async def fake_fetch_wallet_api_positions(*args, **kwargs):
        return ([], "0xwallet")

    _run(seed())
    monkeypatch.setattr(api_routes, "_fetch_wallet_api_positions", fake_fetch_wallet_api_positions)

    res = _run(api_routes.redemptions_list())
    bucket = res["events"][0]["buckets"][0]
    assert bucket["sync_status"] == "missing_on_chain"
    assert bucket["requires_action"] is True
    assert res["summary"]["db_only_exposure"] == 0.5
    assert res["summary"]["stale_rows"] == 1

    _run(engine.dispose())


def test_redemptions_wallet_timeout_still_returns_db_rows(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    async def seed():
        async with session_factory() as session:
            event = Event(
                city_id=city.id,
                date_et="2026-05-07",
                status="ok",
                resolved_at=datetime.utcnow(),
            )
            session.add(event)
            await session.flush()
            bucket = Bucket(
                event_id=event.id,
                bucket_idx=0,
                label="Will the highest temperature in Atlanta be 70-71F?",
                yes_token_id="yes-token",
                no_token_id="no-token",
                condition_id="cond-timeout",
            )
            session.add(bucket)
            await session.flush()
            session.add(Position(
                bucket_id=bucket.id,
                side="yes",
                net_qty=1.0,
                avg_cost=0.40,
                entry_type="MANUAL",
            ))
            await session.commit()

    async def fake_fetch_wallet_api_positions(*args, **kwargs):
        raise TimeoutError("wallet slow")

    async def fail_onchain(*args, **kwargs):
        raise AssertionError("default redemptions endpoint must not call on-chain checks")

    _run(seed())
    monkeypatch.setattr(api_routes, "_fetch_wallet_api_positions", fake_fetch_wallet_api_positions)
    monkeypatch.setattr(api_routes, "_fetch_onchain_determined_map", fail_onchain)

    res = _run(api_routes.redemptions_list())
    assert res["events"][0]["buckets"][0]["net_qty"] == 1.0
    assert "wallet_positions_unavailable" in res["timing"]["degraded_reason"]

    _run(engine.dispose())


def test_mark_position_closed_zeroes_stale_position_and_audits(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))
    ids = {}

    async def seed():
        async with session_factory() as session:
            event = Event(city_id=city.id, date_et="2026-05-07", status="ok")
            session.add(event)
            await session.flush()
            bucket = Bucket(
                event_id=event.id,
                bucket_idx=0,
                label="Will the highest temperature in Atlanta be 70-71F?",
                yes_token_id="yes-token",
                condition_id="cond-close",
            )
            session.add(bucket)
            await session.flush()
            pos = Position(
                bucket_id=bucket.id,
                side="yes",
                net_qty=3.0,
                avg_cost=0.20,
                realized_pnl=0.0,
                unrealized_pnl=0.15,
                entry_type="MANUAL",
            )
            session.add(pos)
            await session.commit()
            ids["position_id"] = pos.id

    async def verify():
        async with session_factory() as session:
            pos = await session.get(Position, ids["position_id"])
            audits = (await session.execute(
                select(AuditLog).where(AuditLog.action == "position_mark_closed")
            )).scalars().all()
            assert pos.net_qty == 0.0
            assert pos.unrealized_pnl == 0.0
            assert "Closed locally" in pos.current_exit_status
            assert len(audits) == 1

    _run(seed())
    res = _run(api_routes.mark_position_closed(
        ids["position_id"],
        api_routes.MarkClosedRequest(reason="stale DB cleanup"),
        actor="test",
    ))
    assert res["ok"] is True
    assert res["old_qty"] == 3.0
    _run(verify())

    _run(engine.dispose())


def test_redemptions_template_wraps_market_titles():
    html = open("web/templates/redemptions.html", encoding="utf-8").read()
    assert "whitespace-normal break-words leading-snug" in html
    assert 'text-cyan-600 font-semibold mt-0.5 truncate' not in html
    assert "STALE DB POSITION" in html
    assert "Mark Closed" in html
    assert "Check UMA" in html
    assert "Quick Exit" in html
    assert "REDEEM" in html


def test_position_sync_does_not_cancel_db_work_with_global_timeout():
    sig = inspect.signature(sync_positions_from_chain)
    assert "total_timeout_s" not in sig.parameters
    src = inspect.getsource(sync_positions_from_chain)
    assert "wait_for" not in src


def test_strategies_page_bulk_loads_heatmap_without_per_signal_sessions(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))
    session_entries = {"count": 0}

    async def seed():
        async with session_factory() as session:
            event = Event(
                city_id=city.id,
                date_et=city_local_date(city),
                status="ok",
            )
            session.add(event)
            await session.flush()
            snapshot = ModelSnapshot(
                event_id=event.id,
                mu=82.0,
                sigma=2.5,
                probs_json="[]",
            )
            session.add(snapshot)
            await session.flush()

            for idx in range(25):
                bucket = Bucket(
                    event_id=event.id,
                    bucket_idx=idx,
                    label=f"Atlanta bucket {idx}",
                    low_f=70.0 + idx,
                    high_f=71.0 + idx,
                    yes_token_id=f"yes-{idx}",
                    no_token_id=f"no-{idx}",
                    condition_id=f"cond-{idx}",
                )
                session.add(bucket)
                await session.flush()
                session.add(
                    Signal(
                        bucket_id=bucket.id,
                        model_snapshot_id=snapshot.id,
                        model_prob=0.20,
                        mkt_prob=0.10,
                        raw_edge=0.10,
                        exec_cost=0.01,
                        true_edge=0.09,
                    )
                )
            await session.commit()

    _run(seed())

    original_get_session = storage_db.get_session

    def counting_get_session():
        session_entries["count"] += 1
        return original_get_session()

    async def fake_fetch_openmeteo_metadata(key):
        return {}

    def fake_template_response(template_name, context):
        return {
            "template_name": template_name,
            "context": context,
        }

    monkeypatch.setattr(storage_db, "get_session", counting_get_session)
    monkeypatch.setattr(web_routes, "fetch_openmeteo_metadata", fake_fetch_openmeteo_metadata)
    monkeypatch.setattr(web_routes.templates, "TemplateResponse", fake_template_response)

    response = _run(web_routes.strategies_page(request=None))

    assert response["template_name"] == "strategies.html"
    assert len(response["context"]["heatmap_data"][city.city_slug]["buckets"]) == 25
    assert session_entries["count"] == 1

    _run(engine.dispose())


def test_health_handles_naive_heartbeat_timestamps(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))

    async def seed():
        async with session_factory() as session:
            session.add(
                WorkerHeartbeat(
                    job_name="scheduler_alive",
                    last_run_at=datetime.utcnow().replace(microsecond=0),
                    run_count=1,
                )
            )
            await session.commit()

    _run(seed())
    response = _run(api_routes.health())

    assert response["status"] == "ok"
    assert response["worker_heartbeat_age_s"] is not None

    _run(engine.dispose())


def test_get_city_state_handles_naive_forecast_timestamps(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    async def seed():
        fetched_at = datetime.utcnow().replace(microsecond=0)
        date_et = city_local_date(city)

        async with session_factory() as session:
            session.add(
                MetarObs(
                    city_id=city.id,
                    metar_station=city.metar_station,
                    observed_at=fetched_at,
                    fetched_at=fetched_at,
                    temp_c=20.0,
                    temp_f=68.0,
                    daily_high_f=70.0,
                )
            )
            session.add_all(
                [
                    ForecastObs(
                        city_id=city.id,
                        source="nws",
                        date_et=date_et,
                        fetched_at=fetched_at,
                        high_f=71.0,
                        raw_payload_hash="nws",
                        raw_json="{}",
                    ),
                    ForecastObs(
                        city_id=city.id,
                        source="wu_hourly",
                        date_et=date_et,
                        fetched_at=fetched_at,
                        high_f=73.0,
                        raw_payload_hash="wu_hourly",
                        raw_json="{}",
                    ),
                ]
            )
            await session.commit()

    _run(seed())
    response = _run(api_routes.get_city_state(city.city_slug))

    assert response["city_slug"] == city.city_slug
    assert response["metar_age_s"] is not None
    assert response["forecasts"]["nws"]["age_s"] is not None
    assert response["forecasts"]["wu_hourly"]["age_s"] is not None
    assert response["daily_high_f"] == 70.0

    _run(engine.dispose())


def test_get_city_state_exposes_hotter_bucket_fields(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    async def seed():
        fetched_at = datetime.utcnow().replace(microsecond=0)
        date_et = city_local_date(city)

        async with session_factory() as session:
            event = Event(
                city_id=city.id,
                date_et=date_et,
                status="ok",
                forecast_quality="ok",
                gamma_slug=f"atlanta-{date_et}",
            )
            session.add(event)
            await session.flush()

            session.add(
                MetarObs(
                    city_id=city.id,
                    metar_station=city.metar_station,
                    observed_at=fetched_at,
                    fetched_at=fetched_at,
                    temp_c=19.0,
                    temp_f=66.2,
                    daily_high_f=69.0,
                )
            )
            session.add_all(
                [
                    ForecastObs(
                        city_id=city.id,
                        source="nws",
                        date_et=date_et,
                        fetched_at=fetched_at,
                        high_f=72.0,
                        raw_payload_hash="nws",
                        raw_json="{}",
                    ),
                    ForecastObs(
                        city_id=city.id,
                        source="wu_hourly",
                        date_et=date_et,
                        fetched_at=fetched_at,
                        high_f=66.0,
                        raw_payload_hash="wu_hourly",
                        raw_json="{}",
                    ),
                ]
            )
            session.add(
                ModelSnapshot(
                    event_id=event.id,
                    mu=69.0,
                    sigma=0.15,
                    probs_json="[0.999, 0.001]",
                    inputs_json='{"prob_new_high":0.001,"prob_hotter_bucket":0.001,"prob_new_high_raw":0.49,"lock_regime":true,"observed_bucket_idx":0,"observed_bucket_upper_f":70.0,"city_state":"resolved"}',
                    forecast_quality="ok",
                )
            )
            await session.commit()

    _run(seed())
    response = _run(api_routes.get_city_state(city.city_slug))

    assert response["prob_new_high"] == 0.001
    assert response["prob_hotter_bucket"] == 0.001
    assert response["prob_new_high_raw"] == 0.49
    assert response["lock_regime"] is True
    assert response["observed_bucket_idx"] == 0
    assert response["observed_bucket_upper_f"] == 70.0
    assert response["city_state"] == "resolved"

    _run(engine.dispose())


def test_get_city_signals_exposes_hotter_bucket_fields(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    async def seed():
        fetched_at = datetime.utcnow().replace(microsecond=0)
        date_et = city_local_date(city)

        async with session_factory() as session:
            event = Event(
                city_id=city.id,
                date_et=date_et,
                status="ok",
                forecast_quality="ok",
                gamma_slug=f"atlanta-{date_et}",
            )
            session.add(event)
            await session.flush()

            bucket = Bucket(
                event_id=event.id,
                bucket_idx=0,
                label="68-69",
                low_f=68.0,
                high_f=69.0,
            )
            session.add(bucket)
            await session.flush()

            session.add(
                Signal(
                    bucket_id=bucket.id,
                    model_prob=0.999,
                    mkt_prob=0.995,
                    raw_edge=0.004,
                    exec_cost=0.006,
                    true_edge=-0.002,
                    reason_json='{"prob_new_high":0.001,"prob_hotter_bucket":0.001,"prob_new_high_raw":0.49,"lock_regime":true,"observed_bucket_idx":0,"observed_bucket_upper_f":70.0,"city_state":"resolved"}',
                    gate_failures_json="[]",
                    computed_at=fetched_at,
                )
            )
            await session.commit()

    _run(seed())
    response = _run(api_routes.get_city_signals(city.city_slug))

    assert len(response) == 1
    assert response[0]["prob_new_high"] == 0.001
    assert response[0]["prob_hotter_bucket"] == 0.001
    assert response[0]["prob_new_high_raw"] == 0.49
    assert response[0]["lock_regime"] is True
    assert response[0]["observed_bucket_idx"] == 0
    assert response[0]["observed_bucket_upper_f"] == 70.0
    assert response[0]["city_state"] == "resolved"

    _run(engine.dispose())
