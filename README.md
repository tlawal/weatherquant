# WeatherQuant

Quantitative prediction market trading system for weather events on Polymarket.

## 🏗️ Architecture

- **Signal Engine**: Fuses NWS, Weather Underground, and METAR data into a probabilistic distribution.
- **Calibration Engine**: Monitors model reliability and remaps probabilities based on historical hit rates (Brier Score optimization).
- **Execution Layer**: Interfaces with Polymarket CLOB, implementing Kelly Criterion sizing and automated risk gates.
- **Market Context Agent**: An autonomous tool-calling agent (LLM) that resolves forecast discrepancies by actively fetching high-resolution HRRR models and academic climatology heuristics.
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
- `MARKET_CONTEXT_LLM_PROVIDER`: Optional. Enables admin-triggered Market Context refreshes when set to `openai`, `anthropic`, `gemini`, or `openrouter`.
- `MARKET_CONTEXT_LLM_MODEL`: Optional, but required when `MARKET_CONTEXT_LLM_PROVIDER` is set. Use the exact provider model ID.
- `MARKET_CONTEXT_LLM_API_KEY`: Optional generic API key for Market Context generation. If omitted, the app falls back to `OPENAI_API_KEY` for `openai`, `ANTHROPIC_API_KEY` for `anthropic`, `GEMINI_API_KEY` for `gemini`, or `OPENROUTER_API_KEY` for `openrouter`.
- `OPENAI_API_KEY`: Optional provider-specific fallback when `MARKET_CONTEXT_LLM_PROVIDER=openai`.
- `ANTHROPIC_API_KEY`: Optional provider-specific fallback when `MARKET_CONTEXT_LLM_PROVIDER=anthropic`.
- `GEMINI_API_KEY`: Optional provider-specific fallback when `MARKET_CONTEXT_LLM_PROVIDER=gemini`.
- `OPENROUTER_API_KEY`: Optional provider-specific fallback when `MARKET_CONTEXT_LLM_PROVIDER=openrouter`.
- `MARKET_CONTEXT_LLM_BASE_URL`: Optional base URL override for proxy or self-hosted provider endpoints.
- `MARKET_CONTEXT_LLM_TIMEOUT_SECONDS`: Optional timeout for Market Context generation requests. Default `45`.

## 🧹 Manual Sweep / NegRisk Redemption Suite

If you need to sweep a negative risk outcome that isn't cleanly resolving, or just want to manually extract winning position tokens directly from the Polymarket Safe proxy, use the included sweeper:

```bash
# Provide the condition ID to pull your winnings via Gnosis Safe proxy wrapping:
.venv/bin/python scripts/sweep_negrisk.py --condition-id <0xcondition_id_here>
```
*Note*: Requires `.env` configuration for `POLYMARKET_PRIVATE_KEY` and `FUNDER_ADDRESS`.

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

1. Copy env template: `cp .env.example .env`
2. Install dependencies: `pip install -r requirements.txt`
3. Initialize DB: `python -m backend.scripts.init_db`
4. Run dev server: `python -m web.app`
5. Run trading engine: `python -m backend.main`

### Market Context Refresh Setup

The Market Context card is cached and read-only by default. The manual **Refresh** action stays disabled until the server has all three of the following:

- `MARKET_CONTEXT_LLM_PROVIDER`
- `MARKET_CONTEXT_LLM_MODEL`
- An API key: either `MARKET_CONTEXT_LLM_API_KEY` or the provider-specific fallback (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `OPENROUTER_API_KEY`)

If the UI shows:

> Market Context refresh is disabled until an LLM provider, model, and API key are configured on the server.

the server is missing one or more of those values.

Example `.env` settings for OpenAI:

```bash
MARKET_CONTEXT_LLM_PROVIDER=openai
MARKET_CONTEXT_LLM_MODEL=<your-openai-model-id>
OPENAI_API_KEY=<your-openai-api-key>
```

Example `.env` settings for Anthropic:

```bash
MARKET_CONTEXT_LLM_PROVIDER=anthropic
MARKET_CONTEXT_LLM_MODEL=<your-anthropic-model-id>
ANTHROPIC_API_KEY=<your-anthropic-api-key>
```

Example `.env` settings for Gemini:

```bash
MARKET_CONTEXT_LLM_PROVIDER=gemini
MARKET_CONTEXT_LLM_MODEL=<your-gemini-model-id>
GEMINI_API_KEY=<your-gemini-api-key>
```

Example `.env` settings for OpenRouter:

```bash
MARKET_CONTEXT_LLM_PROVIDER=openrouter
MARKET_CONTEXT_LLM_MODEL=openai/gpt-4o-mini
OPENROUTER_API_KEY=<your-openrouter-api-key>
```

Notes:

- `MARKET_CONTEXT_LLM_BASE_URL` is only needed if you are routing requests through a proxy or compatible gateway.
- Refresh remains admin-triggered only; normal city-page loads render the latest stored Market Context snapshot and do not call the LLM.

### 🧠 Market Context Agent (LLM Execution Layer)

The Market Context is generated by a multi-step Agentic LLM (OpenAI/Anthropic/OpenRouter). Unlike standard one-shot generators, this agent has access to **tools** to resolve discrepancies in the backend input data:

- **`fetch_hrrr_forecast`**: Pings the Open-Meteo REST API using city lat/long to retrieve the hourly High-Resolution Rapid Refresh (HRRR) curve.
- **`search_academic_heuristics`**: Injects physical climatology heuristics (e.g., radiational heating stalls, urban heat islands) when NWS and WU forecasts diverge.
- **`get_polymarket_bucket_odds`**: Pulls live order book probabilities from the DB to inform the final market mapping.

The agent is prompted with a high-stakes mandate to seek the "absolute truth" by evaluating these evidence-based layers before selecting the final Polymarket bucket.

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

---

## 🎯 Adaptive Prediction Engine (Phase 2 Upgrade)

The adaptive prediction engine is a real-time micro-forecasting layer that ingests ALL 5-minute METAR observations from the entire day and computes station-time-specific temperature predictions using stateful Kalman filtering and multivariate rolling regression. This enables last-minute, high-confidence predictions at exact METAR observation minutes (e.g., KATL at :52 of each hour), critical for profitable Polymarket trades.

### Why Adaptive Predictions?

Polymarket's temperature markets resolve on **specific ASOS observation times** at **exact minutes** past the hour (e.g., KATL reads at X:52, KLAX reads at X:56). The traditional approach—predicting only a *daily high* value—misses the granular intra-day structure. Adaptive predictions answer: "What will the ASOS reading be at 3:52 PM given current wind, humidity, and clouds?" This 5-minute granularity directly aligns with Polymarket's settlement precision.

### Architecture

**Two-state Kalman Filter** with weather-conditioned process noise:
- **State**: `[temperature_f, temperature_trend_per_minute]`
- **Observation**: Raw 5-min METAR temperature from api.weather.gov
- **Measurement noise**: R = 0.25°F² (ASOS ±0.5°F accuracy)
- **Process model**: Constant-velocity temperature with adaptive process noise scaled by real-time meteorology:
  - High wind speed → increased boundary layer mixing → higher process noise
  - Precipitation present → rapid convective cooling → much higher noise
  - Cloud cover transition (CLR↔BKN/OVC) → changed radiative forcing
  - Low humidity → wider diurnal range

  Academic basis: Hacker & Rife (2007) "A Practical Approach to Sequential Estimation of Systematic Error in NWP" demonstrates adaptive Kalman filtering where process noise varies with meteorological conditions—exactly our approach.

**60-Minute Rolling Regression** (OLS, multivariate):
- **Core feature**: Time (minutes since observation window start)
- **Extended features** (when available from extended METAR parsing):
  - `wind_speed_kt` — Wind increases boundary layer mixing, slows afternoon warming (Mass & Brier, 2015)
  - `humidity_pct` — Higher humidity reduces diurnal range (Stull, 1988)
  - `cloud_cover` (encoded 0–4: CLR→FEW→SCT→BKN→OVC) — Clouds reduce solar heating rate and increase longwave radiation trapping
  - `precip_flag` (binary) — Precipitation causes rapid cooling via evaporation and downdrafts (Lackmann, 2011)
  - `pressure_tendency` (change in altimeter_inhg over 30 min) — Falling pressure often precedes weather changes

  Falls back to univariate (time-only) regression if extended fields unavailable. Uses numpy `lstsq` for OLS solution—no new dependencies.

**Station-Time Predictions**:
Given a station's observation minutes (e.g., [52] for KATL) and current local time, the engine computes predicted temperatures for all remaining observation times from 6:52 AM through 7:52 PM. Each prediction blends:
1. **Kalman extrapolated trend**: Current smoothed temperature + (trend per minute x minutes ahead x **diurnal decay**)
2. **Regression extrapolated slope**: Regression slope applied over the extrapolation horizon, weighted by R^2
3. **Diurnal decay factor**: A physically derived multiplier that reduces the warming rate toward zero as peak temperature time approaches (Parton & Nicholls 2012). Prevents unrealistic constant-rate extrapolation.
4. **Parametric diurnal curve**: When ≥6 observations are available, a piecewise temperature curve (sin² rising + Gaussian falling) is fitted to the day's data via L-BFGS-B optimization, constrained by the forecast high and estimated peak timing. Near-term predictions (<2 hr) blend Kalman with the curve; far-term predictions (>2 hr) weight the physics-based curve more heavily. Falls back to decay-only when insufficient data or poor fit (RMSE > 3°).
5. **Remaining-rise cap**: ML-predicted ceiling (GradientBoosting) limits predictions to `current_temp + remaining_rise`, preventing runaway values.
6. **Weather adjustment**: Regression features (wind, clouds, precip) dampen or accelerate the warming trend

**Weather-Conditioned Uncertainty (sigma)**:
Forecast distribution uncertainty adjusts in real-time based on current conditions:

| Condition | Physical Mechanism | Sigma Effect |
|---|---|---|
| Overcast (OVC) | Blocks shortwave radiation, caps heating | Tighten ~30% |
| Humidity > 80% | Higher heat capacity, smaller diurnal swing | Tighten ~15% |
| Dewpoint spread < 5F | Near saturation, fog/low clouds likely | Tighten ~15% |
| Wind > 15 kt | Deep boundary layer mixing, uniform temps | Tighten ~10% |
| Gust spread > 10 kt | Turbulent micro-scale variability | Widen ~10% |
| Pressure falling fast | Frontal approach, regime change | Widen ~25% |
| Active precipitation | Evaporative cooling dominates energy budget | Tighten ~20% |

The final prediction and uncertainty bound are returned for each future observation time, enabling traders to evaluate risk at exact settlement times.

**Composite Peak Timing**:
The system estimates when the daily high will occur by fusing three independent sources:
1. **WU Hourly Forecast** — WU's forward-looking peak time (e.g., "5:00 PM ET") from `_fetch_wu_hourly_api()`
2. **Historical Average** — Past 3 days' actual peak times (filtered to exclude sparse-data days with <10 daytime obs)
3. **Kalman Trend Trajectory** — If the Kalman trend has been negative for 30+ minutes, the peak already occurred; detect it from the actual max obs time

A composite estimate weights sources by confidence: WU higher in the morning, Kalman trajectory higher in afternoon once sufficient trend data accumulates.

### Data Flow

```
Full-day METAR observations (every 5 min since midnight local)
         ↓
[Parse extended fields: wind, humidity, cloud cover, precip, pressure]
         ↓
Kalman Filter [run on every observation]
    - State: [temp, trend]
    - Process noise scaled by wind/precip/clouds/humidity
    - Output: smoothed_temp, trend_per_min, uncertainty
         ↓
Rolling 60-min OLS Regression [run every signal cycle]
    - Features: time, wind, humidity, clouds, precip, pressure_tendency
    - Output: slope, R², feature importance
         ↓
Composite Peak Timing [fuse WU forecast + historical + Kalman]
    - Estimate when daily high occurs
    - Detect if peak already passed
         ↓
ML Remaining-Rise Cap [GradientBoosting prediction]
    - Predict max additional temperature rise from current reading
         ↓
Parametric Diurnal Curve [fit when ≥6 obs + forecast_high]
    - Piecewise: sin²(rising) + Gaussian(falling)
    - Constrained by forecast high + peak timing
    - Output: DiurnalFit (T_min, T_max, t_peak, RMSE)
         ↓
Station-Time Predictions [compute for 10:52 AM – 7:52 PM]
    - Blend Kalman trend + regression slope + diurnal curve
    - Near-term: Kalman-weighted; far-term: curve-weighted
    - Apply diurnal decay + remaining-rise cap
    - Return list: (obs_time, predicted_temp, uncertainty)
         ↓
Signal Engine Integration
    - Blend adaptive predicted_daily_high into mu (capped at 0.4 weight)
    - Apply sigma_adjustment to tighten uncertainty when data rich
    - Audit: include full adaptive metadata in signal inputs dict
```

### Implementation Details

**Core files**:
- `backend/modeling/adaptive.py` — Kalman filter, regression, station predictions, peak timing
- `backend/modeling/diurnal_model.py` — Parametric diurnal curve fitting (sin² + Gaussian)
- `backend/modeling/residual_tracker.py` — ML remaining-rise predictor (GradientBoosting)
- `backend/modeling/temperature_model.py` — Forecast fusion + weather-conditioned sigma

All modules are stateless: re-initialized from scratch each signal cycle (every 60 seconds). Uses full day of observations, not a trailing window. All functions are pure: no database writes, no side effects.

**Key functions**:
- `run_kalman(obs_dicts)` → `KalmanState` with smoothed temp, trend, uncertainty
- `run_regression(obs_window, features_available)` → regression slope, R², feature list
- `compute_station_predictions(kalman, regression, station_minutes, city_tz, current_local_time, diurnal_model)` → list of `StationTimePrediction` with time, predicted temp, uncertainty
- `fit_diurnal_curve(observations, forecast_high, peak_mins, city_tz)` → `DiurnalFit` with T_min, T_max, t_peak, RMSE
- `compute_peak_timing(wu_hourly_peak, historical_avg, kalman_trend, current_hour)` → estimated peak time, source attribution
- `run_adaptive(obs_dicts, wu_hourly_peak, historical_peak_timing, station_minutes, city_tz, city_local_now, forecast_high, ml_features)` → complete `AdaptiveResult` dataclass

**Extended METAR Parsing**: New `MetarObsExtended` table (1:1 with `MetarObs`) stores parsed fields:
- Dewpoint, humidity (via Magnus formula), wind, pressure, cloud cover, weather condition
- All extracted from aviationweather.gov JSON response
- Included in `get_todays_extended_obs()` via eager-loading relationship

**Integration points**:
- `signal_engine.py`: Fetches full day's obs, runs `run_adaptive()`, passes result to `compute_model()`
- `temperature_model.py`: Accepts optional `adaptive` parameter, blends predicted_daily_high into mu
- `routes.py`: Passes `obs_table`, `station_predictions`, `adaptive_info` to web UI

### UI: Station-Time Predictions Panel

The city detail page now includes a **Station-Time Predictions Panel** showing:

**Summary Row**:
- Estimated peak: "Expected High: 79.8°F at ~4:52 PM"
- Peak-already-passed badge (if applicable)
- Observation count: "N total obs today, M at station times"

**Detailed Predictions Table** (10:52 AM through 7:52 PM):
| Station Time | Actual | Predicted | Trend | Confidence |
|---|---|---|---|---|
| 10:52 AM | 72.0°F | — | — | — |
| 1:52 PM | 76.8°F | — | — | — |
| **2:52 PM** | — | **79.3°F** | ↑ +0.2°/hr | ±0.8° |
| **3:52 PM** | — | **79.8°F** | ↗ +0.1°/hr | ±1.2° |
| 5:52 PM | — | 77.1°F | ↓ -0.4°/hr | ±2.8° |

- Past times (rows with actual observations) highlighted gold
- Highest predicted temp marked with ★ and bold
- Trend arrows (↑↗→↘↓) indicate direction and rate of change

**Methodology Tooltip** (clickable ⓘ icon):
Explains the Kalman filter, regression features used, data source (full day of obs), recalculation frequency (every 60s), and weather factors influencing predictions. Transparency is critical—traders must understand the prediction freshness and quality.

### UI: Temperature Timeline Chart

A Plotly chart starting at 10 AM local time shows:
- **Blue dots** for 5-min observations
- **Gold diamonds** for METAR station-minute observations (e.g., :52 for KATL)
- **Cyan dashed line** extending into the future for Kalman/regression predictions
- **Horizontal dashed line** at the current daily high
- **Vertical dashed lines** at station observation minutes
- Chart extends ~2 hours into the future to show prediction zone
- Clear visual transition from actual data (solid) to predictions (dashed)

### UI: Full-Day Observations Table

A collapsible table (sticky header, scrollable body) showing every observation since midnight:

Columns: Time | Temp | Dew Pt | Humidity | Wind Dir | Speed | Gust | Pressure | Precip | Condition

- **Station-minute rows** (e.g., :52): Gold left border, bold time
- **Daily high row**: Yellow background tint
- Compact layout matching Weather Underground's display format
- Expandable to full day (default: shows last ~20 rows)

### Phase 1 Fixes (Prerequisite)

The adaptive engine depends on accurate METAR data. Phase 1 addresses three root causes of stale/incorrect highs:

1. **Timezone-Correct Date Boundaries**: `get_daily_high_metar()` and `get_resolution_high_metar()` now accept `city_tz` parameter (city's local timezone, not hardcoded ET). Date queries construct start/end boundaries in local time, ensuring observations near midnight are assigned to the correct calendar day.

2. **Aggressive METAR Polling**: `should_poll_station()` uses asymmetric window: poll from -2 minutes before to +10 minutes after each station observation minute (instead of symmetric ±3). This catches delayed observations appearing on aviationweather.gov 5–10 minutes after the actual recording time. For KATL (:52), the polling window is :50–:02 of next hour.

3. **Rate-Limit Gating on Successful Scrapes Only**: WU scraper (`fetch_wu_all()`) now checks the last *successful* scrape (high_f IS NOT NULL) instead of the most recent attempt. Failed scrapes (high_f=None) no longer reset the rate-limit timer. `WU_MIN_SCRAPE_INTERVAL_SECONDS` reduced from 3600 to 1800 (30 min) for fresher forecast sources.

### Academic References

1. **Delle Monache et al. (2011)** "Kalman Filter and Analog-Based Retrievals for Atmospheric Temperature Profiling" — *Mon. Weather Rev.* 139(10). State estimation for near-surface temperature with bias correction. Directly applicable to 2-state Kalman implementation.

2. **Hacker & Rife (2007)** "A Practical Approach to Sequential Estimation of Systematic Error in NWP" — *Weather and Forecasting* 22(6). Demonstrates adaptive Kalman filtering where process noise scales with meteorological conditions.

3. **Mass & Brier (2015)** "Two-Meter Temperature Forecasting with K-Nearest Neighbors" — *Weather and Forecasting* 30(6). Shows that incorporating wind speed, humidity, and cloud cover as features improves short-range temperature predictions significantly over univariate methods.

4. **Glahn & Lowry (1972)** "Use of Model Output Statistics in Objective Weather Forecasting" — *J. Appl. Meteorol.* 11(12). Foundational MOS approach; our multivariate regression extends this with real-time surface obs features.

5. **Stull (1988)** "An Introduction to Boundary Layer Meteorology" — *Springer*. Explains why wind speed and cloud cover modulate the diurnal temperature curve—justifies these as regression features.

6. **Lackmann (2011)** "Midlatitude Synoptic Meteorology" — *American Meteorological Society*. Documents rapid temperature drops from precipitation onset and convective downdrafts.

7. **Raftery et al. (2005)** "Using Bayesian Model Averaging to Calibrate Forecast Ensembles" — *Mon. Weather Rev.* 133(5). Future enhancement: could improve weighting in composite peak timing.

8. **Wilson et al. (2010)** "Nowcasting Challenges During the Beijing Olympics" — *Weather and Forecasting* 25(6). High-frequency surface obs (5-min METAR) improve short-term temperature extreme prediction.

9. **Parton & Nicholls (2012)** "Parameterisation of the diurnal cycle of temperature using a piecewise model" — *J. Appl. Meteor. Climatol.* 51, 612–630. Template for the diurnal decay factor: sin^2 rising phase + exponential decay. RMSE < 1C for clear-sky days.

10. **Mayer & Groom (2002)** "Diurnal heating rate in the surface layer" — *J. Atmos. Sci.* 59, 1413–1424. Derives surface heating rate as f(net radiation, soil flux, BL depth). Heating rate peaks 2–3 hr after sunrise and decays quasi-linearly to zero at peak temperature time.

11. **Hemri et al. (2014)** "Trends in the predictive performance of raw ensemble weather forecasts" — *Geophys. Res. Lett.* 41, 9197–9205. EMOS framework: heteroscedastic sigma depending on weather regime, not just forecast spread. Foundation for weather-conditioned sigma adjustment.
