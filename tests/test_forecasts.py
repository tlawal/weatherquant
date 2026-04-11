"""Regression tests for `_scrape_wu_daily`.

The WU daily page renders today's forecast high as `--` at night between
the "peak locked in" window and the next morning's forecast publication.
The scraper must return None in that state — the previous max-sweep
fallback silently returned an unrelated number from the page (current
temp, an hourly strip value, a different day's low) and poisoned the
model downstream.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import backend.ingestion.forecasts as forecasts
from backend.storage.models import City


def _run(coro):
    return asyncio.run(coro)


def _make_city(slug: str = "houston", station: str = "KHOU") -> City:
    return City(
        city_slug=slug,
        display_name=slug.title(),
        metar_station=station,
        enabled=True,
        is_us=True,
        unit="F",
        tz="America/Chicago",
        wu_state="tx",
        wu_city=slug,
    )


def _install_fake_html(monkeypatch, html: str) -> None:
    async def _fake_fetch_html(url: str, retries: int = 3) -> Optional[str]:
        return html

    monkeypatch.setattr(forecasts, "_fetch_html", _fake_fetch_html)


def test_scrape_wu_daily_returns_none_on_double_dash(monkeypatch):
    """WU is explicitly showing `--` in the structured selector — no guessing."""
    html = """
    <html><body>
      <span data-test="daily-temperature-high">--</span>
      <span>55°</span>
      <span>78°</span>
    </body></html>
    """
    _install_fake_html(monkeypatch, html)
    result = _run(forecasts._scrape_wu_daily(_make_city()))
    assert result is None


def test_scrape_wu_daily_returns_none_on_em_dash(monkeypatch):
    """Em-dash (—) variant of the same rollover-window state."""
    html = """
    <html><body>
      <span data-test="daily-temperature-high">—°F</span>
    </body></html>
    """
    _install_fake_html(monkeypatch, html)
    result = _run(forecasts._scrape_wu_daily(_make_city()))
    assert result is None


def test_scrape_wu_daily_returns_none_when_only_unrelated_numbers(monkeypatch):
    """No trusted selector, no scoped regex — the old max-sweep fallback
    used to return 78.0 here. Now we return None (silence beats garbage)."""
    html = """
    <html><body>
      <div>Current conditions</div>
      <span>55°</span>
      <span>62°</span>
      <span>78°</span>
      <div>Humidity 88%</div>
    </body></html>
    """
    _install_fake_html(monkeypatch, html)
    result = _run(forecasts._scrape_wu_daily(_make_city()))
    assert result is None


def test_scrape_wu_daily_parses_structured_high(monkeypatch):
    """Happy path: structured selector carries a numeric degree value."""
    html = """
    <html><body>
      <span data-test="daily-temperature-high">87°F</span>
    </body></html>
    """
    _install_fake_html(monkeypatch, html)
    result = _run(forecasts._scrape_wu_daily(_make_city()))
    assert result == 87.0


def test_scrape_wu_daily_parses_regex_high(monkeypatch):
    """Selectors missing but the body contains a clean `Today High 87°F`."""
    html = """
    <html><body>
      <p>Today High 87°F overnight low 64°F.</p>
    </body></html>
    """
    _install_fake_html(monkeypatch, html)
    result = _run(forecasts._scrape_wu_daily(_make_city()))
    assert result == 87.0


def test_scrape_wu_daily_ignores_record_high_phrase(monkeypatch):
    """The scoped regex must not match "Record high 112°F set in 1998"."""
    html = """
    <html><body>
      <p>Record high 112°F set in 1998. No current forecast available.</p>
    </body></html>
    """
    _install_fake_html(monkeypatch, html)
    result = _run(forecasts._scrape_wu_daily(_make_city()))
    assert result is None


def test_scrape_wu_daily_ignores_high_pressure_phrase(monkeypatch):
    """The scoped regex must not match "high pressure 1020 hPa"."""
    html = """
    <html><body>
      <p>high pressure 1020 hPa dominates the region.</p>
    </body></html>
    """
    _install_fake_html(monkeypatch, html)
    result = _run(forecasts._scrape_wu_daily(_make_city()))
    assert result is None
