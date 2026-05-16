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
    assert forecast_card.index("HRRR 15min") < forecast_card.index(
        "Forecast Source Excluded"
    )
    assert forecast_card.rindex("<details", 0, forecast_card.index("Forecast Source Excluded")) > 0


def test_station_calibration_card_separates_overall_and_current_lead_skill():
    template = Path("web/templates/city.html").read_text()

    assert "Best overall:" in template
    assert "Best at current lead:" in template
    assert "n < 30 are provisional" in template


def test_buckets_table_surfaces_execution_microstructure():
    template = Path("web/templates/city.html").read_text()

    assert "Est. Model Edge" in template
    assert "Intraday" in template
    assert "Legacy Daily-High PDF" in template
    assert "Intraday Threshold" in template
    assert "settles below" in template
    assert "Max Size" in template
    assert "Trade State" in template
    assert "After-Cost Edge" in template
    assert "Est. Fill Cost" in template
    assert "why_not_tradable" in template


def test_wallet_leaderboard_disclaimer_is_read_only():
    template = Path("web/templates/city.html").read_text()

    assert "Top Weather Wallets" in template
    assert "Wallet leaderboard is read-only public-market analytics" in template
    assert "does not trigger automated trades" in template


def test_city_page_surfaces_obs_proximity_readout():
    template = Path("web/templates/city.html").read_text()

    assert "OBS Proximity Exit" in template
    assert "Next Obs" in template
    assert "Nearest Boundary" in template
    assert "Reference Bucket" in template
    assert "Armed" in template
