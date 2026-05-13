from pathlib import Path


def test_source_exclusion_alert_stays_inside_forecast_sources_card():
    template = Path("web/templates/city.html").read_text()

    grid_start = template.index("TEMPERATURE SNAPSHOT ROW")
    forecast_sources = template.index("<!-- Forecast Sources Summary -->")
    grid_prefix = template[grid_start:forecast_sources]
    forecast_card = template[forecast_sources:]

    assert "Forecast Source Excluded" not in grid_prefix
    assert forecast_card.index("Forecast Sources</div>") < forecast_card.index(
        "Forecast Source Excluded"
    )
