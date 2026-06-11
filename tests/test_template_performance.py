from pathlib import Path


TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "web" / "templates"


def test_plotly_not_loaded_globally():
    base = (TEMPLATE_DIR / "base.html").read_text()
    assert "cdn.plot.ly" not in base


def test_plotly_loaded_only_on_chart_pages():
    for name in ("city.html", "calibration_edge.html", "backtest.html"):
        text = (TEMPLATE_DIR / name).read_text()
        assert "cdn.plot.ly" in text

    for name in ("redemptions.html", "db_admin.html", "cities_admin.html"):
        text = (TEMPLATE_DIR / name).read_text()
        assert "cdn.plot.ly" not in text


def test_db_admin_template_uses_safe_admin_endpoints():
    text = (TEMPLATE_DIR / "db_admin.html").read_text()
    assert "/api/admin/db/size-report" in text
    assert "/api/admin/db/maintenance/prune" in text
    assert "/api/admin/db/cold-export" in text
    assert "dry_run: String(dryRun)" in text
    assert "forecast_obs_days: String(this.policy.forecast_obs_days)" in text
    assert "wallet_trade_days: String(this.policy.wallet_trade_days)" in text
    assert "model_input_days: String(this.policy.model_input_days)" in text
    assert "Download Gzip JSONL" in text
    assert "Execute Prune" in text
    assert "Protected History" in text


def test_dashboard_lazy_loads_signal_table():
    text = (TEMPLATE_DIR / "dashboard.html").read_text()
    assert 'hx-get="/htmx/signals-table"' in text
    assert 'hx-trigger="load, every 15s"' in text
    assert 'hx-get="/htmx/redeem-summary"' in text
    assert 'hx-trigger="load, every 60s"' in text
