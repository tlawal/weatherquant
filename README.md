# WeatherQuant

Quantitative prediction market trading system for weather events on Polymarket.

## 🏗️ Architecture

- **Signal Engine**: Fuses NWS, Weather Underground, and METAR data into a probabilistic distribution.
- **Calibration Engine**: Monitors model reliability and remaps probabilities based on historical hit rates (Brier Score optimization).
- **Execution Layer**: Interfaces with Polymarket CLOB, implementing Kelly Criterion sizing and automated risk gates.
- **Dashboard**: HTMX-powered real-time monitoring of edges, calibration, and portfolio state.

## 🚀 Deployment (Railway)

### 1. Persistence (Critical)
The system uses SQLite (`state.db`) and local logs. On Railway, you **must** mount a Volume to `/app/data` to prevent data loss on redeploys.
- Create a Railway Volume named `weatherquant-data`.
- Mount it to path: `/app/data`.
- Ensure `DATABASE_URL` in `.env` points to `/app/data/state.db`.

### 2. Environment Variables
- `POLY_API_KEY`: Polymarket API key.
- `POLY_SECRET`: Polymarket secret.
- `POLY_PASSPHRASE`: Polymarket passphrase.
- `ADMIN_TOKEN`: Bearer token for dashboard trade execution.
- `KELLY_FRACTION`: Default 0.10 (Conservative 1/10th Kelly).
- `MAX_POSITION_PCT`: Max bankroll % per single trade (default 0.05).

## 🛡️ Risk Management

- **Kelly Criterion**: Position sizes are scaled by `(Edge / Odds) * FractionalMult`.
- **Daily Loss Limit**: Automated halt if realized daily loss exceeds 2% of bankroll.
- **Gating**: System prevents trading on "degraded" forecast quality or thin liquidity (<100 shares).

## 📊 Evaluation

The dashboard includes a **Reliability Diagram** (Calibration Curve). 
- A perfect diagonal line represents perfect model calibration.
- Points above the line indicate underconfidence (the outcome occurs more than predicted).
- Points below the line indicate overconfidence.

## 🛠️ Local Development

1. Install dependencies: `pip install -r requirements.txt`
2. Initialize DB: `python -m backend.scripts.init_db`
3. Run dev server: `python -m web.app`
4. Run trading engine: `python -m backend.main`

---

## External Data Sources

### Weather Underground (WU)
Official settlement resolution source. We scrape:
- **Daily Summary**: `https://www.wunderground.com/weather/{METAR}`
- **Hourly Projections**: `https://www.wunderground.com/hourly/{METAR}`

### METAR (Aviation Weather)
Official real-time temperature observations (ground truth).
- `https://aviationweather.gov/api/data/metar?ids={METAR}&format=json&latest=1`

### National Weather Service (NWS)
Baseline gridpoint forecasts from `api.weather.gov`.

### Open-Meteo & OpenWeatherMap (International Fallbacks)
Provides global current weather and forecast models used entirely for non-US markets.
- **Open-Meteo Documentation:** [https://open-meteo.com/en/docs](https://open-meteo.com/en/docs?daily=temperature_2m_max&latitude=33.641&longitude=-84.4227&timezone=America%2FNew_York&forecast_days=1&temperature_unit=fahrenheit)
 *(Helpful to pull `temperature_2m_max` from `daily` parameter endpoints)*
- **OpenWeatherMap Backup**: Used dynamically when the primary Open-Meteo endpoint enforces rate limits on Railway's shared IPs.
