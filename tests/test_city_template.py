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
    assert "23:59:59 local time" in template
    assert "Lower MAE is better" in template
    assert "Latest scored error" in template
    assert "Forecast - observed at the station-calibration checkpoint" in template
    assert "latest forecast available by ~6 AM local" in template


def test_buckets_table_surfaces_execution_microstructure():
    template = Path("web/templates/city.html").read_text()

    assert "Est. Model Edge" in template
    assert "Intraday" in template
    assert "Legacy Daily-High PDF" in template
    assert "Intraday Threshold" in template
    assert "Intraday Threshold (shadow)" in template
    assert "shadow threshold-crossing model" in template
    assert "settles below" in template
    assert "Max Size" in template
    assert "Trade State" in template
    assert "After-Cost Edge" in template
    assert "Est. Fill Cost" in template
    assert "why_not_tradable" in template


def test_wallet_leaderboard_disclaimer_is_read_only():
    template = Path("web/templates/city.html").read_text()

    assert "Weather Smart Money" in template
    assert "REFRESH FLOW" in template
    assert "refreshSmartMoney" in template
    assert "/api/wallet-rankings/refresh" in template
    assert "CURRENT MARKET" in template
    assert "GLOBAL LEADERS" in template
    assert "CITY LEADERS" in template
    assert "smartOpen" in template
    assert "weather-smart-money-panel" in template
    assert "Current rows" in template
    assert "Exposure rows" in template
    assert "Skill rows" in template
    assert "Last refresh" in template
    assert "wallets long" in template
    assert "Skill source" in template
    assert "Wallet leaderboard is read-only public-market analytics" in template
    assert "does not trigger automated trades" in template
    assert "No wallet trades have been stored for this city/date yet." in template


def test_wallet_bucket_consensus_titles_wrap_instead_of_truncate():
    template = Path("web/templates/city.html").read_text()

    bucket_section = template[template.index("Bucket Consensus"):template.index("CURRENT MARKET")]
    assert "whitespace-normal break-words leading-snug" in bucket_section
    assert "min-h-[2.5rem]" in bucket_section
    assert " text-[11px] truncate" not in bucket_section


def test_city_page_surfaces_obs_proximity_readout():
    template = Path("web/templates/city.html").read_text()

    assert "OBS Proximity Exit" in template
    assert "Next Obs" in template
    assert "Nearest Boundary" in template
    assert "Reference Bucket" in template
    assert "Armed" in template


def test_redemptions_quick_exit_uses_market_sell():
    template = Path("web/templates/redemptions.html").read_text()

    assert "SELL ${qty.toFixed(1)} shares with a FOK market exit" in template
    assert "order_type: 'market'" in template
    assert "order_type: 'limit'" not in template[template.index("async function marketSell"):]
