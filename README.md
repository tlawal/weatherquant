# WeatherQuant

Quantitative trading system for daily-high-temperature prediction markets on Polymarket. Fuses **10** numerical and AI weather models, a per-minute METAR nowcast, and a Bayesian-mixture predictive distribution (with offline + online EM weight updates) into bucket-level edge signals; sizes positions via Kelly; executes through Polymarket CLOB v2 with a tiered exit cascade. Independent LLM **Market Context Agent** runs alongside with a 10-source encyclopedia + adversarial reasoning. Real-time **alpha-vs-market** dashboard at `/calibration/edge` measures Brier(model) − Brier(market) and CRPS distribution error for promotion decisions.

---

## Architecture at a glance

```
┌──────────────────────────┐    ┌────────────────────────────────────────┐
│    Forecast ingestion    │    │            Modeling layer              │
│  ─────────────────────   │    │  ────────────────────────────────────  │
│  NWS, WU                 │    │  Per-source bias correction (EWMA)     │
│  HRRR, NBM (Open-Meteo)  │ ─▶ │  Lead-time skill weighting             │
│  IFS, AIFS (Herbie)      │    │  Freshness decay on model_run_at       │
│  GraphCast (Open-Meteo)  │    │  Weighted-mean μ + ensemble-σ (legacy) │
│  Pangu, FCN-v2, Aurora   │    │  BMA mixture predictive — Phase 1/2/3  │
│  (NOAA AIWP S3)          │    │  + offline EM + online EM on settle    │
│  METAR, MADIS, TGFTP     │    │  Kalman+regression nowcast (METAR)     │
└──────────────────────────┘    └────────────────────────────────────────┘
                                                 │
                                                 ▼
┌──────────────────────────┐    ┌────────────────────────────────────────┐
│       Execution          │    │                  UI                    │
│  ─────────────────────   │    │  ────────────────────────────────────  │
│  Kelly sizing            │    │  /city/<slug>: legacy + BMA shadow,    │
│  Quick Flip exits        │ ◀─ │    Buckets & Edges, Refresh Skills btn │
│  Ladder scaling          │    │  /calibration/edge: alpha-vs-market    │
│  Emergency cascades      │    │  /backtest: walk-forward harness       │
│  Night Owl orchestrator  │    │  /redemptions: alpha card + outcomes   │
│  Auto-redeem             │    │  Market Context Agent — 10-source enc. │
└──────────────────────────┘    └────────────────────────────────────────┘
```

Top-level layout:

| Path | Purpose |
|---|---|
| `backend/ingestion/` | Forecast + observation fetchers (HTTP, S3, Herbie/cfgrib) |
| `backend/modeling/` | Bias correction, Kalman, BMA, calibration, ML residual model |
| `backend/engine/signal_engine.py` | Per-event fusion → bucket signals + persistence |
| `backend/execution/` | CLOB orders, exit engine cascade, position sync, redeemer |
| `backend/strategy/night_owl.py` | Overnight bulk-limit orchestrator |
| `backend/worker/scheduler.py` | APScheduler — 30+ jobs covering ingestion, modeling, execution |
| `backend/api/routes.py` | Admin/trade REST API |
| `web/routes.py`, `web/templates/` | Operator dashboard |
| `backend/backtesting/` | Walk-forward backtest engine + Optuna (planned) |

---

## Modeling roadmap (Path 1)

The system's competitive edge is **per-station ensemble post-processing of a 10-model panel for a binary outcome**. We don't try to beat ECMWF at global weather forecasting — we combine their forecasts (and 9 others) optimally for our specific 12 stations and our specific bucket payoff. This is "Path 1" in the project's strategic plan; see `/Users/larry/.claude/plans/here-s-a-rough-draft-ethereal-lark.md` for the full reasoning.

### Status check (May 2026)

| Stage | What ships | State |
|---|---|---|
| Ingestion of 10 models | NWS / WU / HRRR / NBM / IFS / AIFS / GraphCast / Pangu / FCN-v2 / **Aurora** | ✓ live |
| Adaptive Kalman | Innovation-covariance-driven Q (Mehra 1972), Joseph-form update | ✓ live |
| Lead-time σ growth | NOAA-empirical 1.5 + 0.05·L °F added in quadrature | ✓ live |
| Regime σ inflation | CALM/NORMAL/VOLATILE multiplier ∈ [1.0, 2.0] | ✓ live |
| Per-regime backtest breakouts | CALM vs VOLATILE Brier/Sharpe split | ✓ live |
| Manual override (Q8) | `POST /api/trade-override` bypasses gates with audit row | ✓ live |
| Lead-Time Skill scheduler + diagnostic endpoint | `job_refresh_lead_time_skills` (6h) + `POST /api/admin/recompute-lead-time-skills` for on-demand probe | ✓ live |
| **M1 BMA Phase 1**       | **Mixture predictive computed every minute, persisted in `inputs.bma_shadow`** | **✓ live (shadow)** |
| **M1 BMA Phase 1.5**     | **Side-by-side BMA vs legacy on city-page UI (Model Forecast box + Buckets column + multimodal chip)** | **✓ live** |
| **M1 BMA Phase 2** | **Offline EM weight fitter — `BMAWeights` table populated nightly** | **✓ live** |
| **M1 BMA Phase 3** | **Online-EM updates on settlement (every newly-resolved event nudges weights with lr=0.05)** | **✓ live** |
| **M1 BMA intraday conditioning** | **Conditional BMA probabilities on observed high floor (fixes intraday low-tail mass display bug)** | **✓ live** |
| **Intraday threshold shadow model** | **Deterministic threshold-crossing probabilities with monotone survival-to-bucket conversion, shadow-only** | **✓ live (shadow)** |
| **OBS_PROXIMITY exit layer** | **Pre-observation profit protection for fragile buckets near scheduled station observations, with UI controls in strategies page** | **✓ live** |
| **Wallet tracker read-only analytics** | **Public Polymarket wallet analytics with scoring, strategy inference, leaderboard, and Smart Money vs Model divergence** | **✓ live (disabled by default)** |
| **Alpha dashboard** | **`/calibration/edge` — Brier(model) − Brier(market) plus CRPS distribution error, per-city/per-day, plus chip on `/` and card on `/redemptions`** | **✓ live** |
| **Market Context Agent rewrite** | **10-source encyclopedia, calibration MAE per source, adversarial reasoning + trigger conditions sections** | **✓ live** |
| **AIWP probe** | **Weekly check of NOAA S3 for new AI weather model prefixes (FCN3 watch). One-line integration when detected.** | **✓ live** |
| M1 BMA promotion | Swap `mu/sigma/probs` to drive trades after ≥14d CRPS comparison | **gated on data** (need ~14 settled days post-deploy) |
| **CRPS comparator** | Continuous ranked probability score alongside Brier in `/calibration/edge` | **✓ live** |
| **M5 — posterior-aware Kelly** | Component-level BMA Kelly distribution; auto-sizing uses conservative weighted-median component Kelly | **✓ live** |
| **M7 — reliability-driven gate** | Daily-miscalibration kill switch (Bröcker & Smith 2007) | queued — needs 30d data first |
| **M3 — NIG conjugate σ + bias updates** | Online closed-form Bayesian replaces nightly EWMA | queued (~150 LOC) |
| **Dynamic exits** | Time-decay, vol-aware sizing, regime-conditional trailing stops, slippage modeling | queued (~150 LOC) |
| **EMOS as third path** | Linear EMOS predictive (Gneiting 2005) alongside BMA + legacy | queued (~300 LOC) |
| **MS4 — Optuna nightly hyperparameter search** | Wraps backtester in Optuna; tunes 12 hand-tuned constants | queued (~250 LOC) |
| **M6 — BOCPD regime change-point detection** | Continuous regime posterior replaces discrete label | queued (~200 LOC) |
| **Neural EMOS** (Rasp & Lerch 2018) | NN replacement for `build_bma_predictive` | future |
| **First international city (London/EGLL)** | Foundation phase per international-roadmap design doc | gated on US alpha proving out |
| **CorrDiff downscaling** | NVIDIA diffusion-based 2km downscaling | future, gated on M1 saturation |
| **FCN3 self-host** (Modal) | Failed 10-attempt validation in May 2026; AIWP probe watches for NOAA-hosted version | deferred — see plan §20.x |

## Wallet Tracker

The wallet tracker is read-only public-market analytics for daily high-temperature Polymarket markets. It never places orders, sizes positions, arms trading, or feeds the execution engine.

Wallet data is stored in two layers:

- `wallet_stats` is the compatibility read model used by the original leaderboard. City pages keep this as the fallback source so the panel still renders while V2 normalized tables are empty or only partially backfilled.
- `wallet_trades`, `wallet_market_exposures`, and `wallet_skill_scores` are the V2 normalized tables. They dedupe public trades, summarize per-wallet bucket exposure, and rank wallets globally and by city when enough resolved history exists.

On `/city/<slug>`, Weather Smart Money first uses V2 exposure and skill rows when available. If those rows are missing, it falls back to `wallet_stats` for the selected city/date. Ranked flow is the net displayed smart-wallet exposure by bucket, weighted by wallet skill when available and by the v1 consistency score in fallback mode. The Smart Money vs Model badge compares the model-favored bucket against the strongest ranked-flow bucket.

Bucket Consensus tiles are generated from the current city/date buckets shown on the page. Each tile uses the stored market bucket label, displays the full title with wrapping, and shows ranked long wallet count, net flow, and average entry when available.

Manual refresh is exposed as `POST /api/wallet-rankings/refresh` and the city-page `REFRESH FLOW` control. It refreshes public trades for the exact selected city/date event, writes `wallet_trades` and `wallet_market_exposures`, and updates city-specific weather-wallet skill without overwriting global rankings from a one-city slice. The Polymarket data-api is queried one condition at a time by default (`WALLET_TRACKER_CONDITION_CHUNK_SIZE=1`) because multi-condition comma batches can time out on active weather markets. The scheduler also runs `update_wallet_rankings` on worker startup and then at `WALLET_TRACKER_UPDATE_INTERVAL_MINUTES`.

Known limitations: the tracker depends on public Polymarket API availability and scheduler ingestion; V2 skill quality depends on historical resolved markets; normalized trade backfill may lag the v1 read model; PnL estimates are analytics approximations rather than execution ledger accounting; and wallet output is corroborating context only, not a copy-trading signal.

### Why BMA matters here

The legacy single-Gaussian summary `N(μ, σ²)` captures only between-source variance. Each forecaster's own residual variance σᵢ is dropped, so tail-bucket probabilities are systematically underestimated — exactly the buckets Polymarket prices most aggressively. BMA's mixture variance includes both terms:

```
var(mixture) = Σwᵢσᵢ²   +   Σwᵢ(μᵢ − μ̄)²
              └─ within ─┘   └── between ──┘
```

Empirical example on a 5-source panel with 4°F panel spread:

```
Legacy:  μ=82.15  σ=1.07  probs=[0.023, 0.973, 0.004]   ← 97% in middle bucket
BMA:     μ=82.15  σ=2.31  probs=[0.191, 0.697, 0.113]   ← 30% in tails
```

Same μ, same per-source weights, but BMA correctly assigns 30% of mass to the tails because between-source variance is 42% of mixture total. On front-passage days (IFS vs HRRR diverging by 5°+), the mismatch is much bigger — that's the alpha.

---

## Forecast sources

| Source | Cadence | Latency | Method |
|---|---|---|---|
| **NWS** | hourly | minutes | `api.weather.gov` gridpoints |
| **WU** | 30 min | minutes | scrape Weather Underground daily summary + hourly |
| **HRRR** | hourly (~45 min) | ~1h | Open-Meteo `gfs_hrrr` + Herbie side-channel for raw |
| **NBM** | hourly | ~2h | Open-Meteo `ncep_nbm_conus` + Herbie |
| **ECMWF IFS** | 4×/day | ~6h | Herbie + cfgrib from ECMWF open data |
| **ECMWF AIFS** | 4×/day | ~6h | Herbie + cfgrib (experimental*) |
| **GraphCast** | 4×/day | ~6h | Open-Meteo `gfs_graphcast025` (experimental*) |
| **Pangu-Weather** | 2×/day (00z/12z) | ~8h | NOAA AIWP S3 archive (experimental*) |
| **FourCastNet v2-small** | 2×/day | ~8h | NOAA AIWP S3 archive (experimental*) |
| **METAR** | per-station obs minute | seconds | aviationweather.gov + TGFTP + MADIS HFMETAR |

`*experimental` sources get a 0.35–0.40 base weight (lower than NWS/HRRR/IFS until they accumulate per-source skill data). All weights then multiply by lead-skill factor and freshness decay; final weights are normalized.

NOAA AIWP integration (`backend/ingestion/aiwp.py`) downloads the ~3 GB NetCDF file once per cycle into a `tempfile.TemporaryDirectory`, extracts daily-high values for all enabled cities, and auto-deletes the file when the context exits. Idempotent: re-fetches skip if the same `(source, model_run_at)` is already in DB.

---

## Model forecast μ — the legacy path

The legacy path produces a single Normal `N(μ, σ²)` over the daily-high temperature, integrates against bucket boundaries via the standard CDF, and drives trade decisions today. BMA Phase 1 runs alongside in shadow mode (see roadmap above).

Components fused into μ:

1. **Per-source weighted mean** of `{NWS, HRRR, NBM, IFS, AIFS, GraphCast, Pangu, FCN-v2, WU hourly peak}` after EWMA bias correction. Weights are `base × lead-skill × freshness`, then re-normalized.
2. **Kalman + regression nowcast** from same-day 5-min METAR. Two-state filter `[temp, trend]` with adaptive process noise (Mehra 1972 innovation-covariance estimator). Joseph-form update for numerical stability. Blended into μ at up to 30% weight, gated to a ±2h tent around the predicted peak with a 6°F divergence cap.
3. **METAR intraday projection**. `projected_high = max(daily_high_so_far, current_temp + remaining_rise)`, where `remaining_rise` comes from a `GradientBoostingRegressor` trained on hour-of-day, current temp, 3-hour slope, peak-timing features, and day-of-year. Falls back to a static lookup table when `residual_model.pkl` is absent. Time-of-day weight `w_metar(hour_local)` interpolates between forecast and projection: 0.0 at midnight → 0.99 by 8 PM local.
4. **Late-day lock**. After 6 PM with the day's high firmly above current temp and a negative Kalman trend, `remaining_rise` collapses to 0 and probability mass is locked into the observed bucket (caps at 2/5/10% remaining tail mass depending on conditions).

σ assembly:

| Component | What it adds |
|---|---|
| Weight-aware ensemble σ | √(Σwᵢ(μᵢ − μ̄)² / Σwᵢ) when ≥3 sources, else max(spread/2, floor) |
| Lead-time growth (Q4) | √(σ²_ensemble + σ²_lead), σ_lead(L) = 1.5 + 0.05·L °F |
| Regime inflation (Q6) | × multiplier ∈ [1.0, 2.0] from `regime_sigma_inflation()` |
| Weather-conditioned (EMOS) | Hemri 2014 — tighten on overcast/high-humidity, widen on falling pressure |
| Sigma floor | max with 1.0°F (or 5/9°C) |

---

## Adaptive nowcast (Kalman + regression + diurnal curve)

The adaptive engine ingests every 5-minute METAR observation since local midnight and produces station-time-specific predictions for each remaining ASOS observation minute through the day. This matches Polymarket's settlement granularity (e.g. KATL resolves on the :52 reading).

```
Full-day METAR (5-min) ─▶ Kalman filter ─▶ 60-min OLS regression ─▶
   │                       (adaptive Q)      (wind/RH/clouds/precip)
   │                                                │
   ├─ML remaining-rise ─▶ Diurnal curve fit ─▶ Station-time predictions
   │ (GradientBoosting)    (sin² + Gaussian)        │
   │                                                │
   └─ Composite peak timing ◀───────────────────────┘
      (WU forecast + historical avg + Kalman trend)
```

Core files: `backend/modeling/adaptive.py`, `backend/modeling/diurnal_model.py`, `backend/modeling/residual_tracker.py`. All stateless — re-initialized from scratch each signal-engine cycle (60s).

UI surfaces: a station-time predictions table, a Plotly timeline chart with observations as gold diamonds and predictions as a cyan dashed line, and a full-day observations table.

---

## Station calibration

Per-station rolling 30-day MAE/bias/RMSE, refreshed every 6h (`job_refresh_station_calibrations`). Persisted in `StationCalibration` and `StationSourceWeight`. Drives:

- **Per-source bias correction** at debias time
- **Per-source weighting** via NIG empirical-Bayes shrinkage (`station_weights.py:58-90` — the one genuinely Bayesian component in the legacy stack: `λ = n / (n+7), mse = λ·m̂se + (1−λ)·9.0`)
- **Tradeability flag** GREEN (<1.5°F MAE) / AMBER (1.5–3.0°F) / RED (>3.0°F) — informational today, M7 plan promotes it to a hard signal-engine gate

Station/source scoring is checkpoint-safe. `ForecastDailyError`, `StationSourceWeight`, and the station-card per-source MAE use the latest forecast available by a fixed morning checkpoint, currently `18h` before the `23:59:59` local settlement time (~6 AM local), rather than late-night revisions after the daily high has likely occurred. This applies across the ensemble sources (`nws`, `open_meteo`, `wu_hourly`, HRRR, NBM, IFS/AIFS, GraphCast, Pangu, FCN-v2, Aurora). Rows with parse errors or missing target dates are skipped for calibration.

NWS target-date handling is strict: `api.weather.gov` daytime periods are parsed into the city timezone and must match the requested market date. If a late-night NWS response only contains tomorrow's daytime period, the system stores a failed `ForecastObs` with `parse_error="target_date_not_in_nws_periods"` instead of writing tomorrow's high under today's `date_et`. Open-Meteo rows also store requested-date and available-date metadata; missing target dates are marked `target_date_not_in_source_payload`.

The station calibration diagnostics endpoint reports suspicious late-day target-date leakage counts across all forecast sources:

```text
GET /api/station-calibrations/diagnostics
```

After deploying forecast-date fixes, rebuild derived calibration rows so old late-night rows stop affecting live weights:

```text
POST /api/admin/recompute-forecast-daily-errors?max_days=60
```

Sortable analytics + Leaflet station-health map at `/calibration`.

The complementary **Lead-Time Skill** table (`SourceLeadTimeSkill`) is refreshed on its own 6h job. It tracks `MAE_f` and `bias_f` per `(city, source, lead_bucket)` ∈ {0, 1, 3, 6, 12, 18, 24, 36, 48, 72h}. Surfaced on city pages and used immediately with empirical-Bayes shrinkage: `0 < n_obs < 30` partially blends observed MAE into BMA σᵢ and source-weight factors, while `n_obs ≥ 30` uses the lead-bucket MAE at full strength.

---

## Risk + execution

### Sizing

- **Kelly criterion**, fractional (default 0.10 from `KELLY_FRACTION` env var), capped at `MAX_POSITION_PCT` (default 0.10) of bankroll per single trade. Auto sizing respects Polymarket's `$1` notional floor: it may bump a positive-Kelly order up to the legal minimum only when position cap, remaining bankroll, top-book liquidity, and `MIN_NOTIONAL_BUMP_MAX_KELLY_MULTIPLE` all permit it.
- **Daily loss limit**: automated halt when realized daily loss exceeds 2% of bankroll.
- **Signal gates**: `min_true_edge`, `max_entry_price`, `max_spread`, `min_liquidity_shares`. Bypassable via `POST /api/trade-override` (audit-logged with operator-supplied reason; the `⚡` button on city-page bucket rows triggers this).

### Exit cascade (5-min sweep)

Priority gate evaluated by `run_exit_engine`:

1. **Emergency** — observed high has already busted the held bucket, or a deep miss is confirmed late day.
2. **EDGE_DECAY** — exits only when `ev_at_bid` is debounced below threshold, the stored entry EV has materially deteriorated, and the model/source thesis has actually worsened since entry. Pure market repricing is suppressed with `blocked_reason="no_model_deterioration"` and is handled by profit or hold logic instead.
3. **Urgent** — debounced model consensus shifted to a different bucket, with spread, confidence, depth, adjacent-bucket, and EV corroboration guards.
4. **OBS_PROXIMITY** — pre-observation profit protection for fragile buckets near scheduled station observations.
5. **Profit / Ladder** — tier-1 50% at +8¢, tier-2 25% at +15¢, trailing stop after tier-1, and legacy Quick Flip for uninitialized positions.
6. **Expiry** — late-day likely winners hold to redeem or passively offer near par; risk exits are capped by positive-EV and P&L guards. Near-par CLOB limit prices are normalized to the legal `[0.01, 0.99]` range while preserving the reference bid in diagnostics.

Tiered exits track `original_qty`, `tier_1_exited`, `tier_2_exited`, `moon_bag_qty`, `trailing_stop_price`, `max_bid_seen` per position. EDGE_DECAY diagnostics include entry/current EV, model-probability delta, market-probability delta, bid delta, and per-source forecast deterioration.

### Night Owl

`backend/strategy/night_owl.py` runs 23:00–06:00 ET. Polymarket orderbooks stagnate overnight while NWP models continue producing fresh forecasts (00z and early 06z). When orderbook depth on a high-edge bucket drops below $2,000, the orchestrator places bulk limit orders. This is when most structural pricing inefficiencies open up.

### Auto-redeem

`run_auto_redeem` runs every 12h. Sweeps resolved markets where the operator (or auto-trader) holds winning tokens and submits redemption transactions. NegRisk markets that don't cleanly resolve are handled via `scripts/sweep_negrisk.py`:

```bash
.venv/bin/python scripts/sweep_negrisk.py --condition-id <0xcondition_id_here>
```

---

## Dashboard

### City page (`/city/<slug>`)

- **Model Forecast box** — legacy μ ± σ. Below it, a yellow **BMA shadow** row shows mixture μ ± σ, between-source variance share, fallback indicator, Δμ/Δσ vs legacy, and (click ⓘ) a per-component breakdown (source · μ · σ · weight · n_obs).
- **Forecast Sources box** — per-source value, model_run_at age, lead time, asterisk badge for experimental sources (AIFS, GraphCast, Pangu, FCN-v2). Per-station skill block shows weight, MAE 7d/30d.
- **Lead-Time Skill panel** — collapsed by default when populated; expands to show per-source/lead MAE, bias, and sample count. Empty state stays open with the manual refresh diagnostic.
- **Buckets & Edges table** — Range / Mkt / Model / **BMA** / EV / Kelly / actions. The BMA column shows mixture probability with an inline ±pp delta when |BMA − Model| > 5pp. A `multimodal` chip appears in the table header when between-source variance share > 50%. Action buttons: ▶ for normal-gate trade, ⚡ for override-trade (writes audit row).
- **Station-time predictions table** — predicted ASOS reading at each remaining observation minute, with Kalman trend arrow and ±confidence.
- **Temperature timeline chart** — Plotly. Blue dots = 5-min obs, gold diamonds = station-minute obs, cyan dashed = future predictions.
- **Full-day observations table** — every METAR since local midnight. Station-minute rows have a gold left border; the daily-high row has a yellow tint.

### City-grouped signals (dashboard `/`)

Each city header shows TWE, consensus temperature, Kalman divergence, METAR anomaly badges (Rain / Snow / Blizzard), and a compact probability sparkline. Per-city groups expand into bucket-level rows (model vs market prob, diff, action).

**TWE (Total Weighted Edge)** is a city-level *display* metric: the sum of positive after-cost edges across liquid buckets where `mkt_prob ≥ 5%`. **TWE does not gate execution** — per-bucket `true_edge` is the operational metric.

### Reliability (`/calibration`)

Calibration curve (model probability vs realized frequency, 10 bins). Diagonal = perfect calibration. Above-diagonal = underconfident, below = overconfident.

### Backtester (`/backtest`)

Walk-forward replay (default 21d train / 7d test). Optimizes `kelly_fraction`, `max_entry_price`, `quick_flip_target` per training fold; tests on the held-out window. Computes equity curve, Sharpe, Brier + Brier skill score, max drawdown, reliability diagram, **per-regime breakouts** (CALM vs VOLATILE).

`min_true_edge` is excluded from optimization because the probability calibration engine adapts the model surface — searching it would double-count.

What-if mode: change any slider and re-run, dashboard renders side-by-side new vs old.

---

## Market Context Agent (LLM)

Independent autonomous agent that runs alongside the algorithmic stack. Uses 5 tools to research weather conditions and produce an independent assessment that flags disagreements with the deterministic model.

| Tool | Source | Purpose |
|---|---|---|
| `fetch_hrrr_forecast` | Open-Meteo `gfs_hrrr` | Hourly HRRR-blended curve |
| `fetch_nbm_forecast` | Open-Meteo `ncep_nbm_conus` | NCEP NBM hourly curve with peak detection |
| `search_academic_climatology` | Semantic Scholar | Peer-reviewed papers on cold air damming, UHI, model biases (with topic-aware fallback library on rate-limit) |
| `fetch_nws_discussion` | `api.weather.gov` AFD | NWS Area Forecast Discussion text |
| `get_polymarket_bucket_odds` | internal DB | Live order-book probabilities |

The LLM provider is operator-configurable (`MARKET_CONTEXT_LLM_PROVIDER` ∈ `{openai, anthropic, gemini, openrouter}`). Read-only by default; the **Refresh** button on the city page is admin-triggered and disabled unless provider + model + API key are all set. Errors return `500 detail="<ClassName>: <message>"` with a full audit row.

---

## NWP cycle reference

| Cycle | UTC | ET | Typical availability (HRRR / GFS / NBM / IFS) |
|---|---|---|---|
| 00z | 00:00 | 8 PM | 1 AM / 3:30 AM / 2 AM / 6 AM |
| 06z | 06:00 | 2 AM | 7 AM / 9:30 AM / 8 AM / 12 PM |
| 12z | 12:00 | 8 AM | 1 PM / 3:30 PM / 2 PM / 6 PM |
| 18z | 18:00 | 2 PM | 7 PM / 9:30 PM / 8 PM / 12 AM |

00z is the highest-alpha window because Polymarket prices don't reprice until 8–10 AM ET. If both 00z and 06z agree on a temperature shift, confidence is high — Night Owl exploits exactly this.

NOAA AIWP files (Pangu, FCN-v2) land ~5h post-init for GFS-initialized variants and ~8h for IFS-initialized. We use IFS-initialized for both to match our IFS/AIFS preference.

---

## Deployment (Railway)

### Persistence

Mount a Railway Volume to `/app/data`:

```bash
DATABASE_URL=sqlite+aiosqlite:////app/data/weatherquant.db
```

The boot path **fail-fast aborts** if `DATABASE_URL` is missing on Railway and falls through to the `/tmp` ephemeral default — set `ALLOW_SQLITE_IN_PROD=1` to override (only do this if you know you're using `/tmp` deliberately).

### Environment variables

**Polymarket / wallet**
- `POLYMARKET_PRIVATE_KEY` — ECDSA key for the trading EOA. Used to mint API credentials at boot via `py-clob-client-v2`.
- `FUNDER_ADDRESS` — Wallet that holds collateral (USDC.e) and conditional tokens.
- `PROXY_ADDRESS` — Optional. Preferred over `FUNDER_ADDRESS` for on-chain balance queries.
- `POLYMARKET_HOST` — Defaults to `https://clob.polymarket.com`. Do not point at the deprecated `clob-v2.polymarket.com` staging host.
- `CHAIN_ID` — `137` (Polygon mainnet).
- `POLYGON_RPC_URL` — Direct `eth_sendRawTransaction` for approvals + redemptions.
- `WALLET_TRACKER_ENABLED` — Enables read-only public wallet analytics. Default `false`.
- `WALLET_TRACKER_CONDITION_CHUNK_SIZE` — Number of condition IDs per Polymarket data-api trade request. Default `1`; keep this at `1` unless data-api multi-market batching is verified healthy.
- `WALLET_TRACKER_LOOKBACK_DAYS` — Background scheduler lookback for skill history. Manual city refreshes scan the selected event date exactly.
- `WALLET_TRACKER_MIN_RESOLVED_MARKETS`, `WALLET_TRACKER_MIN_VOLUME_USD`, `WALLET_TRACKER_MIN_ACTIVE_DAYS` — Weather-wallet skill quality gates.

**Trading**
- `ADMIN_TOKEN` — Bearer for admin/trade endpoints.
- `KELLY_FRACTION` — Default `0.10` (1/10th Kelly).
- `MAX_POSITION_PCT` — Default `0.10`.
- `BANKROLL_CAP` — Optional override; otherwise read from on-chain balance.
- `MIN_ORDER_NOTIONAL_DOLLARS` — Default `1.00`; Polymarket's executable notional floor for dust rejection.
- `MIN_NOTIONAL_BUMP_MAX_KELLY_MULTIPLE` — Default `3.00`; maximum allowed multiple from desired Kelly dollars to a legal minimum-size order.
- `EDGE_DECAY_REQUIRE_MODEL_DETERIORATION` — Default `true`; prevents EDGE_DECAY exits caused only by bid movement.
- `EDGE_DECAY_MIN_MODEL_PROB_DROP` — Default `0.03`; minimum bucket model-probability drop from entry to qualify as thesis deterioration.
- `EDGE_DECAY_MIN_SOURCE_TEMP_DETERIORATION_F` — Default `0.75`; minimum source forecast movement farther outside the held bucket.
- `EDGE_DECAY_MIN_EV_DROP` — Default `0.03`; minimum entry-to-current EV deterioration before EDGE_DECAY can fire.

**Market Context Agent (optional)**
- `MARKET_CONTEXT_LLM_PROVIDER` ∈ `{openai, anthropic, gemini, openrouter}`
- `MARKET_CONTEXT_LLM_MODEL` — provider-specific model ID
- `MARKET_CONTEXT_LLM_API_KEY` — generic, or use provider-specific fallback (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`)
- `MARKET_CONTEXT_LLM_BASE_URL` — only for proxy/self-hosted gateways
- `MARKET_CONTEXT_LLM_TIMEOUT_SECONDS` — default `45`

The Refresh button stays disabled until provider + model + key are all present.

---

## Local development

```bash
cp .env.example .env             # then fill in
pip install -r requirements.txt
python -m backend.scripts.init_db
python -m web.app                # API + dashboard
python -m backend.main           # full stack incl. scheduler
PYTHONPATH=. .venv/bin/pytest tests/ -q --ignore=tests/test_market_context.py
```

Train the ML residual tracker after data has accumulated. The trainer builds a remaining-rise model from METAR observations, uses a chronological date-grouped holdout to avoid same-day leakage, and requires at least 50 usable daytime samples:

```bash
python -m backend.modeling.ml_trainer
```

By default this is shadow-only and writes `backend/modeling/residual_model_shadow.pkl` plus `backend/modeling/residual_model_shadow_meta.json`. The metadata records train/test MAE, baseline MAE, date split, sample count, city count, and whether promotion is ready.

Use the live Railway database from inside the deployed service network:

```bash
railway ssh "python -m backend.modeling.ml_trainer"
```

`railway run` only injects Railway environment variables into a local process; it does not join Railway's private network. If `DATABASE_URL` uses `postgres.railway.internal`, run through `railway ssh` or the trainer will fail DNS resolution from your laptop.

Promote only when the chronological holdout beats the static table by at least `0.20°F`:

```bash
railway ssh "PROMOTE_RESIDUAL_ML=1 python -m backend.modeling.ml_trainer"
```

Promotion writes `backend/modeling/residual_model.pkl` plus `backend/modeling/residual_model_meta.json`; otherwise it still saves the shadow model and logs why promotion was blocked. The signal engine loads a promoted model on restart. Until a promoted model exists, `residual_tracker.py` uses the static remaining-rise lookup table.

Operator prompt for a shadow-only production check:

```text
Run the WeatherQuant residual ML trainer in shadow mode on the live Railway database from inside the service network. Do not promote or change live inference. Report train MAE, test MAE, static-table baseline MAE, improvement in degrees F, sample count, city count, chronological train/test dates, and promotion_ready. If the trainer fails, patch the trainer first, rerun it in shadow, then summarize the exact error and fix.
```

---

## Academic references

**Modeling foundations**
1. Raftery, Gneiting, Balabdaoui, Polakowski (2005). *Using Bayesian Model Averaging to Calibrate Forecast Ensembles.* MWR 133:1155. — basis for M1 BMA.
2. Gneiting, Raftery (2007). *Strictly Proper Scoring Rules.* JASA 102:359. — promotion gate uses CRPS.
3. Mehra (1972). *Approaches to adaptive filtering.* IEEE TAC. — adaptive Q in `backend/modeling/adaptive.py`.
4. Hacker & Rife (2007). *A Practical Approach to Sequential Estimation of Systematic Error in NWP.* WAF 22(6). — weather-conditioned process noise.
5. Hemri et al. (2014). *Trends in the predictive performance of raw ensemble weather forecasts.* GRL 41. — heteroscedastic σ → weather-conditioned EMOS.

**Adaptive engine + diurnal**
6. Delle Monache et al. (2011). *Kalman Filter and Analog-Based Retrievals.* MWR 139.
7. Mass & Brier (2015). *Two-Meter Temperature Forecasting with K-Nearest Neighbors.* WAF 30(6).
8. Glahn & Lowry (1972). *Use of Model Output Statistics in Objective Weather Forecasting.* JAM 11. — foundational MOS.
9. Stull (1988). *An Introduction to Boundary Layer Meteorology.* Springer.
10. Lackmann (2011). *Midlatitude Synoptic Meteorology.* AMS.
11. Parton & Nicholls (2012). *Parameterisation of the diurnal cycle.* JAMC 51:612.
12. Mayer & Groom (2002). *Diurnal heating rate in the surface layer.* JAS 59:1413.

**AI weather models (the ensemble we postprocess)**
13. Lam et al. (2023). *GraphCast.* Science 382:1416.
14. Bi et al. (2023). *Pangu-Weather.* Nature 619:533.
15. Pathak et al. (2022/2024). *FourCastNet.*
16. Lang et al. (2024). *AIFS — ECMWF's data-driven forecasting system.* arXiv.
17. Radford, Ebert-Uphoff, Stewart et al. (2025). *NOAA AIWP archive.* BAMS 106:E68. — the data source for our Pangu + FCN-v2 ingestion.

**Decision theory**
18. MacLean, Thorp, Ziemba (2010). *The Kelly Capital Growth Investment Criterion.* World Scientific. — fractional Kelly + posterior-aware sizing (M5 plan).
19. Bröcker & Smith (2007). *Increasing the reliability of reliability diagrams.* WAF 22:651.

---

## Critical files map

| File | Purpose |
|---|---|
| `backend/modeling/temperature_model.py` | `compute_model` — fuses everything into `ModelResult` |
| `backend/modeling/bma.py` | BMA mixture predictive (Phase 1) |
| `backend/modeling/adaptive.py` | Kalman + regression + station-time predictions |
| `backend/modeling/distribution.py` | Single-Gaussian bucket integration (legacy path) |
| `backend/modeling/calibration_engine.py` | Brier-driven probability calibration + lead-time skill computation |
| `backend/modeling/station_calibration.py` | 30-day rolling MAE/bias per station using checkpoint-safe source selection |
| `backend/modeling/station_weights.py` | Checkpoint-safe `ForecastDailyError` rebuild + NIG empirical-Bayes per-source weights |
| `backend/modeling/ml_trainer.py` | Remaining-rise ML trainer; shadow by default, promotion gated by chronological holdout MAE |
| `backend/modeling/regime.py` | CALM/NORMAL/VOLATILE label + σ multiplier |
| `backend/engine/signal_engine.py` | Per-event orchestration → bucket signals + persistence |
| `backend/ingestion/aiwp.py` | NOAA AIWP S3 fetcher (Pangu + FCN-v2) |
| `backend/ingestion/herbie_side_channel.py` | HRRR/NBM/IFS/AIFS via Herbie + cfgrib |
| `backend/ingestion/forecasts.py` | Open-Meteo + WU + NWS gridpoints |
| `backend/execution/exit_engine.py` | 4-level exit cascade |
| `backend/strategy/night_owl.py` | Overnight orchestrator |
| `backend/worker/scheduler.py` | APScheduler — all jobs |
| `backend/storage/db.py` | Async engine, fail-fast DATABASE_URL guard, AsyncAdaptedQueuePool tuning |
| `web/routes.py`, `web/templates/city.html` | Dashboard + city detail page (incl. BMA shadow surfaces) |
| `backend/backtesting/engine.py` | Walk-forward backtest harness |
