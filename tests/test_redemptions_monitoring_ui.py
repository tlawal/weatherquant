from pathlib import Path


def test_redemptions_page_surfaces_exit_plan_and_performance_ledger():
    html = Path("web/templates/redemptions.html").read_text()
    assert "Closed trade analytics" in html
    assert "/api/performance/summary" in html
    assert "Exit plan" in html
    assert "HOLD WINNER" in html or "hold_winner" in html
    assert "Sell now P&L" in html
    assert "Hold-if-win P&L" in html


def test_active_trading_templates_use_estimated_edge_labels():
    redemptions = Path("web/templates/redemptions.html").read_text()
    signals = Path("web/templates/partials/signals_table.html").read_text()
    city = Path("web/templates/city.html").read_text()

    combined = "\n".join([redemptions, signals, city])
    assert "Estimated Model Edge" in combined
    assert "Hold Edge @ Bid" in combined
    assert "True Edge" not in combined
