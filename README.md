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

## 📡 Data Ingestion & Observation Routing

### Weather Underground (WU)
Official settlement resolution source. We scrape:
- **Daily Summary**: `https://www.wunderground.com/weather/{METAR}`
- **Hourly Projections**: `https://www.wunderground.com/hourly/{METAR}`
- **Historical Observations**: We parse daily high temperatures directly from the station's raw historical JSON payload, systematically capturing off-pattern, mid-hour observations to strictly align with Polymarket's granular resolution rules.

### National Weather Service (NWS) & METAR
- **Live Ground Truth**: We poll `api.weather.gov/stations/{METAR}/observations/latest` as the primary real-time temperature source for **all** US and International cities possessing a valid ICAO METAR station code.
- **METAR Fallback**: `https://aviationweather.gov/api/data/metar?ids={METAR}&format=json&latest=1` serves as a highly reliable secondary fallback for US and global aviation stations.
- **NWS Forecasts**: Baseline gridpoint forecasts from `api.weather.gov`. The dynamic continuous fusion of these forecasts tracks intraday cloud cover and humidity trends, which meteorological research indicates are primary regulators of diurnal temperature peak timing and magnitude (as solar insolation is directly impeded by cloud masking).

### Polymarket Gamma API Routing & Web UI
- Custom slug routing gracefully maps UI abbreviations (e.g., `la`, `sf`) to full, hypenated names (`los-angeles`, `san-francisco`) for accurate Polymarket Gamma API event matching. 
- Timezone-aware date rollovers automatically transition city dashboard links to the next day's active market after the 8:00 PM local time daily market close cutoff, effectively preventing dead links to resolved events.

### Open-Meteo & OpenWeatherMap (International Fallbacks)
Provides global current weather and forecast models used strictly for non-US markets without explicit ICAO METAR assignments.
- **Open-Meteo Documentation:** [https://open-meteo.com/en/docs](https://open-meteo.com/en/docs)
- **OpenWeatherMap Backup**: Used dynamically when the primary Open-Meteo endpoint enforces rate limits on Railway's shared IPs.

---

## 🧠 Model Forecast μ Calculation

The `Model Forecast μ` represents the system's "best guess" for the true projected daily high, acting as the mean of our probabilistic temperature distribution. It dynamically fuses three key components:

1. **Base Forecast Fusion (`mu_forecast`)**: The model takes the NWS Daily High, the Weather Underground (WU) Daily High, and the WU Hourly Peak. It applies historically calibrated biases and weights to each source (derived via the Brier Score calibration engine) to compute a weighted, bias-corrected baseline prediction.
2. **Static Diurnal "Remaining Rise" Table**: Based entirely on the current local hour, the model looks up an expected `remaining_rise` table (e.g., at 6 AM it expects +11°F remaining, at 1 PM it expects +2°F). 
3. **Live METAR Observation Blending**: As the day progresses, the model calculates a `projected_high` which equals the `max(daily_high_observed, current_temp + remaining_rise)`. Finally, it utilizes a time-of-day weighting factor (`w_metar`) to dynamically mix the Base Forecast and the Projected High. At midnight, `w_metar` is 0.0 (100% forecasting). By 8 PM local, it reaches 0.99 (99% reliant on live observation trends).

### ML Residual Tracker (Dynamic Remaining Rise)

The static "Remaining Rise" table (step 2) is being replaced by a `GradientBoostingRegressor` trained on historical METAR observations. It uses the following features:

| Feature | Description |
|---|---|
| `hour_local` | Current local hour of day |
| `temp_f` | Current observed temperature |
| `temp_slope_3h` | Temperature change over the past 3 hours (real-time momentum) |
| `avg_peak_timing_mins` | Rolling 3-day average of when the daily high was reached (minutes since midnight) |
| `day_of_year` | Seasonal solar angle proxy |

> [!IMPORTANT]
> **Zero-risk deployment**: Until the model is trained, the system falls back automatically to the old static table — no regression risk.
>
> To activate ML predictions after deploying to Railway:
> ```bash
> python -m backend.modeling.ml_trainer
> ```
> This trains on live `metar_obs` data, prints MAE improvement vs. the static baseline, and saves `backend/modeling/residual_model.pkl`. The signal engine loads it automatically on next restart.
