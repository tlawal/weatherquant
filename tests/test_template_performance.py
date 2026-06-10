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
    assert "dry_run: String(dryRun)" in text
    assert "Execute Prune" in text
    assert "Protected History" in text
