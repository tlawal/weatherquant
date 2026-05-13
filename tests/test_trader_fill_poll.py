import asyncio

from backend.execution import trader


def _run(coro):
    return asyncio.run(coro)


def test_poll_for_fill_uses_clob_open_orders_wrapper(monkeypatch):
    calls = []

    class FakeClob:
        _client = object()

        async def get_open_orders(self, market: str):
            calls.append(market)
            return [{"id": "order-1"}]

    monkeypatch.setattr(trader, "get_clob", lambda: FakeClob())

    result = _run(
        trader._poll_for_fill(
            order_id=1,
            clob_order_id="order-1",
            token_id="token-1",
            expected_size=5.0,
            expected_price=0.25,
            condition_id="condition-1",
            timeout_s=0.02,
            poll_interval_s=0.001,
        )
    )

    assert result is None
    assert calls
    assert set(calls) == {"condition-1"}
