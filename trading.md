# Automated Weather Trading Strategy Framework — Master Plan

## Context

This document is a comprehensive, implementation-ready trading strategy framework for the existing weatherquant bot. The bot already has a production-grade pipeline: weather ingestion (HRRR, GFS, NWS, WU, METAR) → ensemble fusion → probability modeling → Kelly-sized limit orders on Polymarket temperature contracts. However, it currently operates as a single-shot, edge-threshold trader with no dynamic position management, no multi-strategy orchestration, and no exploitation of known temporal alpha windows. The user has empirically observed that **buying late-night / early-morning (11 PM–6 AM ET) at 8–30¢ contracts yields 5–20% moves by morning forecast updates**, enabling partial profit-taking and risk-free moon-bags. This framework codifies that observation and nine additional strategies into a unified automated system.

The deliverable is a single markdown document written to `docs/trading_strategy_framework.md` containing all 8 sections requested.

---

## Implementation Plan

### Step 1: Create the master document skeleton

**File:** `docs/trading_strategy_framework.md`

Write the full document with the following 8 major sections. Each section is detailed below with the exact content to include.

---

### Section 1 — Comprehensive Alpha Sources in Weather Markets

Content to write:

**1.1 Model Update Cycle Alpha**
- HRRR: Runs every hour (00z–23z), ~45 min latency. Highest resolution (3km), best for <18h horizon. The 00z–06z runs update overnight forecasts that markets haven't priced.
- GFS: Runs 4x/day (00z, 06z, 12z, 18z), ~3.5h latency. Lower resolution (13km) but longer horizon. The 06z run (available ~09:30 UTC / 5:30 AM ET) is the first fresh global model of the day.
- ECMWF: Runs 2x/day (00z, 12z), ~6h latency. Gold standard for 3–10 day forecasts but least frequent — creates the widest mispricing windows.
- NAM: Runs 4x/day, ~1.5h latency. Regional model, useful for cross-checking HRRR.

**Alpha mechanism:** Markets are updated by human participants who check forecasts 1–3x/day (morning, midday, evening). Model runs that complete between 11 PM and 6 AM ET are not reflected in prices until human traders wake up. The bot can ingest these runs automatically and trade before humans reprice.

**1.2 Structural Inefficiencies**
- **Low liquidity:** Polymarket temperature markets have $500–$5,000 in total liquidity per bucket. A $50 order can move price 2–5%.
- **Low model literacy:** Most participants use single-source forecasts (iPhone weather app, Weather.com) rather than ensemble model output. They don't understand HRRR vs GFS divergence or ensemble spread.
- **Lagged forecast incorporation:** Participants update beliefs 1–3x daily. Forecast updates at 2 AM ET aren't priced until 8–10 AM ET — a 6–8 hour window.
- **Anchoring bias:** Participants anchor to the previous day's market prices rather than fresh forecast data. If yesterday's market priced bucket X at 40¢, today's market opens near 40¢ even if overnight models shifted 3°F.
- **Recency bias:** After a hot day, participants overweight the likelihood of another hot day regardless of synoptic pattern changes.

**1.3 Seasonal Climatology Miscalibration**
- Participants systematically underestimate climatological base rates. In shoulder seasons (March–April, October–November), temperature distributions are wider than summer/winter, but markets price narrow distributions.
- Spring cold-front volatility is underpriced: a 20°F swing in 24h is common but markets price as if temperature is mean-reverting within ±5°F.
- Urban heat island effects are poorly understood by retail participants — cities like Phoenix, Las Vegas systematically settle 2–4°F above NWS gridpoint forecasts.

**1.4 The Late-Night Alpha Window (User's Observation)**
- Between 11 PM and 6 AM ET, liquidity drops 80–90%. Market makers widen spreads or pull quotes entirely.
- The 00z HRRR and 00z GFS runs complete around 1–3 AM ET. If these runs shift the forecast by ≥2°F vs the previous cycle, the bot can buy contracts at stale prices.
- Contracts priced 8–30¢ have asymmetric payoff: max loss is the premium paid, max gain is 70–92¢ (to $1 resolution).
- This is the single highest-Sharpe alpha source in the system.

**1.5 Academic Grounding**
- Wolfers & Zitzewitz (2004): Prediction markets aggregate information efficiently only when liquid and diverse — Polymarket weather markets fail both criteria for most of the day.
- Campbell & Diebold (2005): Weather derivatives pricing is sensitive to forecast model choice; ensemble disagreement predicts realized volatility.
- Jewson & Brix (2005): *Weather Derivative Valuation* — temperature distributions are non-Gaussian with fat tails, especially in transition seasons.
- Benth et al. (2008): *Stochastic Modelling of Temperature Variations* — AR processes for temperature with mean-reversion to seasonal norms.
- Manski (2006): Prediction market prices don't cleanly map to probabilities when participants are risk-averse or liquidity-constrained.
- Gneiting & Raftery (2007): Probabilistic forecasting — ensemble model output statistics (EMOS) provides calibrated probability distributions that outperform raw ensemble output.

---

### Section 2 — Exhaustive Trading Strategies

For each strategy, write: inefficiency, detection logic, entry rules, exit rules, sizing, backtest notes, failure modes.

**Strategy 1: Pre-Model-Run Accumulation (Night Owl)**
- **Inefficiency:** 00z model runs complete at 1–3 AM ET; markets don't reprice until 8–10 AM ET.
- **Detection:** At 11 PM ET, compare current market prices to the bot's latest model output. If `true_edge ≥ 0.12` on any bucket priced ≤ 0.30, flag as accumulation target.
- **Entry:** Place limit orders at `yes_bid + 0.01` (penny above best bid) between 11 PM and 2 AM ET. If not filled in 30 min, walk up to `mid = (bid + ask) / 2`.
- **Exit:** Tiered: sell 50% when price reaches `entry + 0.10`, sell 25% at `entry + 0.20`, hold 25% as moon-bag to resolution.
- **Sizing:** 1.5x normal Kelly fraction (higher conviction due to information asymmetry). Cap at 15% of bankroll.
- **Backtest:** Compare model_prob at 11 PM vs market_prob at 11 PM for all historical events. Measure price movement by 10 AM next day.
- **Failure:** Model run comes in flat (no forecast change) — position sits at entry price. Mitigate with minimum forecast-delta threshold of 1.5°F shift.

**Strategy 2: Post-Model-Run Repricing (Fast Follower)**
- **Inefficiency:** Fresh model runs take 30–90 min to propagate to sophisticated participants, 4–8h for retail.
- **Detection:** Monitor HRRR/GFS ingestion timestamps. When a new run arrives AND `abs(new_forecast - old_forecast) ≥ 1.5°F`, trigger repricing signal.
- **Entry:** Within 5 minutes of detecting the new run, buy the bucket(s) favored by the new forecast at current `yes_ask`. Priority: HRRR runs (hourly) over GFS runs (6-hourly) due to higher resolution.
- **Exit:** Sell 100% when `true_edge < 0.03` (edge has been arbitraged away) or after 4 hours, whichever comes first.
- **Sizing:** Standard Kelly fraction. Reduce by 50% if the model shift is between 1.5–2.0°F (moderate confidence).
- **Backtest:** For each model run, record forecast delta and subsequent market price movement over 1h, 2h, 4h windows.
- **Failure:** Market has already priced the run (informed traders front-ran). Detect by checking if market moved >3¢ in the 15 min before our signal — if so, skip.

**Strategy 3: Ensemble Spread Compression**
- **Inefficiency:** When HRRR and GFS converge (spread narrows), the realized outcome is more certain, but markets maintain wide bucket distributions priced for uncertainty.
- **Detection:** Track `forecast_spread = abs(hrrr_forecast - gfs_forecast)` over time. When spread compresses from >3°F to <1.5°F across consecutive runs, trigger.
- **Entry:** Buy the consensus bucket (where both models agree) at current ask. Simultaneously, if NO contracts on outlier buckets are cheap (<5¢), consider selling them.
- **Exit:** Hold to resolution (high confidence). Partial exit at 70¢+ if available.
- **Sizing:** 2x normal Kelly (model agreement = high conviction). Cap at 20% bankroll.
- **Backtest:** Correlate historical HRRR/GFS spread at T-12h with bucket hit rate. Expect >65% hit rate when spread <1.5°F.
- **Failure:** A third model (NWS, WU) diverges sharply — indicates the HRRR/GFS agreement may be a shared-bias artifact. Gate: require NWS to be within 2°F of consensus.

**Strategy 4: Climatology-vs-Forecast Divergence**
- **Inefficiency:** When forecasts deviate >2σ from climatological norms, participants either overreact (extreme buckets overpriced) or underreact (moderate buckets underpriced), depending on direction.
- **Detection:** Compare today's forecast μ against the 7-day rolling average high (`avg_high_7d_f` from market_context). If `abs(forecast_mu - climatology_avg) > 2 * historical_sigma`, flag.
- **Entry:** If forecast is ABOVE climatology by >2σ: buy the forecast-aligned bucket but at reduced size (overreaction likely). If forecast is BELOW climatology: buy the climatology-aligned bucket (mean reversion likely, participants underreact to cooling).
- **Exit:** Dynamic: recalculate edge every signal cycle. Exit when edge < 0.05.
- **Sizing:** 0.5x Kelly for above-climatology (lower confidence), 1.0x Kelly for below-climatology (mean reversion stronger).
- **Backtest:** Compute forecast-vs-climatology z-scores for all historical events. Bucket hit rates by z-score quintile.
- **Failure:** Genuine regime shift (heat dome, polar vortex). Gate: require ≥3 model sources to agree on the deviation direction.

**Strategy 5: Overreaction Fade**
- **Inefficiency:** After a single extreme model run (e.g., one GFS ensemble member shows +8°F above consensus), markets overreact to the outlier.
- **Detection:** When `max(ensemble_members) - ensemble_median > 4°F` AND the market moves >5¢ toward the outlier within 30 minutes of the run.
- **Entry:** Fade the move: buy the consensus bucket that was underpriced by the overreaction, at the newly discounted price.
- **Exit:** When market reverts to within 2¢ of pre-spike price, or after next model run confirms/denies the outlier.
- **Sizing:** 0.75x Kelly. This is a mean-reversion play with moderate confidence.
- **Backtest:** Identify historical instances of single-member outliers. Track market price trajectory over next 2–6 hours.
- **Failure:** The outlier is correct (happens ~15% of the time for >4°F deviations). Stop-loss: exit if the next model run confirms the outlier direction by ≥2°F.

**Strategy 6: Multi-City Correlation Arb**
- **Inefficiency:** Cities in the same synoptic regime (e.g., same air mass, same frontal boundary) have correlated temperature outcomes. If City A reprices to a new bucket but City B (under identical conditions) hasn't moved, arb exists.
- **Detection:** Group cities by proximity (<300 miles) and shared synoptic features (same NWS forecast discussion zone). When City A's market shifts >5¢ but City B's hasn't moved within 1h, flag.
- **Entry:** Buy the corresponding bucket in City B at stale prices.
- **Exit:** When City B reprices (typically within 2–6 hours) or at resolution.
- **Sizing:** Standard Kelly. Reduce by 50% if cities are >200 miles apart (correlation weakens).
- **Backtest:** Compute daily high correlation matrix between city pairs. Identify pairs with r > 0.85.
- **Failure:** Mesoscale effects decouple cities (lake breeze, mountain barrier, coastal fog). Gate: only trade pairs with historical daily-high correlation > 0.80 over past 30 days.

**Strategy 7: HRRR Rapid-Update Spike Capture**
- **Inefficiency:** HRRR updates hourly. During active weather (convection, frontal passage), consecutive HRRR runs can shift ±3°F. Markets can't keep up with hourly updates.
- **Detection:** When two consecutive HRRR runs shift in the same direction by ≥1°F each (cumulative ≥2°F in 2 hours), trigger momentum signal.
- **Entry:** Buy the bucket aligned with the HRRR trend at current ask. Time-sensitive: must execute within 15 minutes of second confirming run.
- **Exit:** Sell when HRRR trend reverses (next run shifts ≥0.5°F against position) or after 3 hours.
- **Sizing:** 0.75x Kelly. High frequency but lower conviction per trade.
- **Backtest:** Track consecutive HRRR deltas and subsequent 3h market price movement. Expect >55% directional accuracy for ≥2°F cumulative shifts.
- **Failure:** HRRR oscillation (ping-pong between runs). Gate: require the third run to NOT reverse by >0.5°F within the execution window.

**Strategy 8: Market Maker Inefficiency Exploitation**
- **Inefficiency:** Polymarket MMs set prices algorithmically based on flow, not weather models. When volume is one-directional (all buys on bucket X), MMs raise prices across all buckets, creating mispricing in low-volume buckets.
- **Detection:** Monitor `market_snapshots` for buckets where `spread > 0.08` AND `yes_ask_depth < 20` AND `model_prob > market_prob + 0.15`. These are neglected buckets where MMs haven't updated.
- **Entry:** Place limit orders at `yes_bid + 0.02` (passive). Be the liquidity provider. Patience required — may take 1–4 hours to fill.
- **Exit:** When spread compresses to <0.04 (MM reprices) or at resolution.
- **Sizing:** 0.5x Kelly (patient, lower urgency).
- **Backtest:** Identify historical wide-spread + edge situations. Track fill rates and subsequent price convergence.
- **Failure:** Spread is wide because the bucket is genuinely mispriced by the model (our model is wrong). Gate: require ≥3 forecast sources to agree on model_prob within ±5%.

**Strategy 9: High-Temperature Bucket Convexity**
- **Inefficiency:** Tail buckets (highest and lowest temperature ranges) are priced at 2–10¢ but resolve to $1 when hit. The expected value is positive when model_prob > contract_price even by small margins, due to convexity.
- **Detection:** Identify buckets where `market_price ≤ 0.10` AND `model_prob ≥ 0.12` AND the bucket represents a temperature within 2σ of the forecast mean.
- **Entry:** Buy at ask. These are small-dollar, high-optionality positions.
- **Exit:** Sell 50% if price reaches 3x entry. Hold rest to resolution.
- **Sizing:** Fixed dollar amount: $1–$2 per position (portfolio of cheap options). Total tail exposure ≤ 10% of bankroll.
- **Backtest:** Historical hit rate of tail buckets vs their market price. Expect positive EV when `model_prob / market_price > 1.2`.
- **Failure:** Tail events are rare by definition. Expect 70–80% of these positions to expire worthless. Portfolio approach is essential.

**Strategy 10: Storm-Event Volatility**
- **Inefficiency:** Approaching storms (cold fronts, thunderstorm complexes) create bimodal temperature distributions. If the front arrives early, temperature drops 10–15°F; if late, it stays warm. Markets price the average, not the bimodality.
- **Detection:** When `sigma > 4.0°F` AND `forecast_spread > 5°F` AND METAR shows `falling_pressure = True` or `precip = True`.
- **Entry:** Buy BOTH the "warm" bucket and the "cold" bucket as a straddle. Each at current ask.
- **Exit:** As resolution approaches and bimodality resolves (one scenario wins), sell the losing leg immediately. Hold the winning leg to resolution.
- **Sizing:** 0.5x Kelly per leg (total 1x Kelly exposure). The straddle structure limits downside.
- **Backtest:** Identify historical high-sigma days. Compute straddle P&L (sum of both legs to resolution).
- **Failure:** Temperature lands in the middle (neither bucket hits). Mitigate by choosing buckets that are adjacent to the mean but on opposite sides.

**Strategy 11: Intraday METAR Trajectory Momentum**
- **Inefficiency:** METAR observations every 5 minutes reveal the actual temperature trajectory. When observed warming rate exceeds the model's predicted remaining-rise, markets haven't caught up.
- **Detection:** When `kalman_trend_per_hr > 2.0°F/hr` AND `current_temp + remaining_rise_prediction > mu_forecast + 1.5°F` AND time is between 9 AM and 2 PM local.
- **Entry:** Buy the bucket aligned with the projected overshoot at current ask.
- **Exit:** When Kalman trend flattens (`trend_per_hr < 0.5°F/hr`) or METAR shows cooling, sell immediately.
- **Sizing:** Standard Kelly. High confidence due to direct observation.
- **Backtest:** Track Kalman projections vs actual daily highs. Measure overshoot frequency and magnitude.
- **Failure:** Cloud cover moves in, capping temperature rise. Gate: require `cloud_cover ≤ SCT` and `precip = False`.

**Strategy 12: Settlement Source (WU) Arbitrage**
- **Inefficiency:** The settlement source is Weather Underground daily high, not NWS or METAR. WU can differ from NWS/METAR by 1–3°F due to station selection, rounding, and timing. If the bot knows which WU station settles and its historical bias, it can trade accordingly.
- **Detection:** When `wu_daily_forecast` differs from `nws_forecast` by ≥2°F, and the market is priced to the NWS forecast.
- **Entry:** Buy the bucket aligned with the WU forecast at current ask.
- **Exit:** Hold to resolution (settlement source advantage).
- **Sizing:** 1.5x Kelly (information edge on settlement mechanics).
- **Backtest:** Compare WU settlement values to NWS/METAR for all historical events. Compute systematic bias.
- **Failure:** WU changes station selection or methodology. Monitor for WU-METAR divergence anomalies.

---

### Section 3 — Architecture Blueprint

Content to write, organized as new components to add to the existing system:

**3.1 Strategy Orchestrator (NEW: `backend/strategy/orchestrator.py`)**
- Manages multiple strategies running concurrently
- Each strategy is a class implementing `BaseStrategy` with `detect()`, `size()`, `enter()`, `exit()`, `should_exit()` methods
- Orchestrator runs each strategy's detect() every signal cycle
- Resolves conflicts (two strategies want opposing positions on same bucket)
- Priority: higher-edge signal wins; ties broken by strategy confidence ranking

**3.2 Position Manager (NEW: `backend/execution/position_manager.py`)**
- Tracks positions with entry strategy tag, entry time, and profit-take schedule
- Implements tiered exit logic: at each price threshold, calculate shares to sell
- Moon-bag logic: after recovering cost basis, remaining shares have `cost_basis = 0` and are held to resolution
- Trailing stop: if position is >2x entry and drops 20% from peak, sell remaining

**3.3 Market Microstructure Monitor (NEW: `backend/engine/microstructure.py`)**
- Ingest `market_snapshots` at 30s intervals (already exists)
- Compute rolling metrics: 5-min spread MA, 15-min depth MA, 1-hr volume estimate, bid/ask imbalance ratio
- Detect liquidity events: spread >2x 1-hr average, depth drop >50%, rapid price movement >5¢/min
- Emit alerts to strategy orchestrator for entry/exit decisions

**3.4 Forecast Delta Tracker (NEW: `backend/modeling/forecast_delta.py`)**
- On each new HRRR/GFS/NWS ingestion, compute delta vs previous run
- Store deltas in new `forecast_deltas` table: `(city_id, source, run_time, old_value, new_value, delta_f, timestamp)`
- Emit events when `abs(delta) > threshold` (configurable per source: HRRR 1.0°F, GFS 1.5°F, NWS 2.0°F)

**3.5 Enhanced Exit Engine (NEW: `backend/execution/exit_engine.py`)**
- Runs every signal cycle (60s) for all open positions
- Checks: profit-take thresholds, stop-loss conditions, time-based exits, anomaly exits
- Implements the full exit cascade from Section 6
- Places SELL limit orders at `yes_bid` or market orders for urgent exits

**3.6 Trade Journal (ENHANCE: `backend/storage/models.py`)**
- Add to `Order` model: `strategy_tag`, `entry_reason_json`, `exit_reason_json`
- Add to `Position` model: `strategy_tag`, `peak_price`, `cost_basis_recovered`, `moon_bag_flag`
- New `TradeJournal` table: per-trade P&L attribution by strategy, holding period, edge at entry vs realized edge

**3.7 Scheduler Additions (ENHANCE: `backend/worker/scheduler.py`)**
- `run_strategy_orchestrator` (60s): Run all strategy detect/exit cycles
- `run_microstructure_monitor` (30s): Compute microstructure metrics
- `run_forecast_delta_tracker` (on ingestion): Compute deltas after each forecast fetch
- `run_overnight_accumulator` (300s, 11 PM–6 AM ET only): Special night-owl strategy cycle

**3.8 Moon-Bag Cascade Logic**
```
on_fill(position, fill_price, fill_qty):
    position.cost_basis = fill_price
    position.remaining_qty = fill_qty
    position.profit_take_schedule = [
        (entry + 0.10, 0.50),  # sell 50% at +10¢
        (entry + 0.20, 0.25),  # sell 25% at +20¢
        # remaining 25% = moon bag, hold to resolution
    ]

on_price_update(position, current_price):
    for threshold, pct in position.profit_take_schedule:
        if current_price >= threshold and not already_sold(threshold):
            sell_qty = position.remaining_qty * pct
            place_sell_order(position, sell_qty, current_price)
            position.remaining_qty -= sell_qty

    # Check if cost basis recovered
    total_sold_value = sum(fill.price * fill.qty for fill in position.sells)
    if total_sold_value >= position.cost_basis * position.initial_qty:
        position.moon_bag_flag = True  # remaining shares are "free"
        position.effective_cost_basis = 0.0
```

**3.9 Risk Management Upgrades**
- **Portfolio-level correlation limit:** Max 3 positions in cities within 300 miles of each other
- **Strategy-level loss limit:** Each strategy has its own daily loss cap (sum of strategy-tagged realized P&L)
- **Volatility-adjusted sizing:** When `sigma > 3.5°F`, reduce all position sizes by 25%. When `sigma > 5.0°F`, reduce by 50%.
- **Auto-scaling:** When win rate over last 20 trades > 60%, increase Kelly fraction by 0.02 (up to 0.2 max). When < 40%, decrease by 0.02 (floor 0.05).

---

### Section 4 — Market Microstructure Explainer

Content to write:

**4.1 Polymarket's Hybrid Model**
- Polymarket uses a Central Limit Order Book (CLOB) on Polygon, not a traditional AMM. However, market makers provide AMM-like continuous liquidity.
- Unlike Uniswap-style pools, there's no bonding curve — prices are set by explicit limit orders. This means spreads are wider and more variable.
- The CLOB has no maker fees (0%) and taker fees of 0%. Revenue comes from the spread itself, not exchange fees.

**4.2 Spread Dynamics Around Forecast Releases**
- **Pre-release (T-2h to T):** Informed traders pull liquidity. Spreads widen 2–3x. Depth drops 40–60%.
- **Release (T to T+30m):** Informed traders place aggressive limit orders on the favored bucket. Spread temporarily narrows as volume spikes.
- **Post-release (T+30m to T+4h):** Retail participants gradually reprice. Spreads return to baseline. This is the "slow repricing" window where most alpha is captured.

**4.3 Small-Order Price Impact**
- With typical ask depth of 50–200 shares at best price, a $20 market order (200 shares at 10¢) can clear the entire best level.
- Impact is nonlinear: the first $5 moves price 1¢, the next $5 moves it 2¢, etc., because book depth thins at higher price levels.
- Implication: always use limit orders. Never walk the book in thin markets.

**4.4 Informed Trader Patterns**
- Informed traders (those with weather model access) place orders in the 15 minutes after a new HRRR/GFS run becomes available.
- Detectable pattern: a burst of 3–5 orders on one bucket, all within 2 minutes, all limit orders slightly above the previous best ask. This "stepping up" pattern signals informed buying.
- The bot should monitor for these patterns and either follow (if aligned with own signal) or wait (if contra).

**4.5 Information Leakage in Weather Markets**
- NWS forecast discussions are published before gridpoint data updates. Sophisticated participants read the discussion text and trade before the numerical forecast propagates.
- HRRR data is available on NOAA servers before Open-Meteo indexes it. Participants with direct NOAA access have a 15–30 minute information advantage.
- WU hourly forecasts sometimes update before the daily high forecast — a leading indicator of the settlement source.

**4.6 Systematic Exploitation**
- **Be the first automated participant:** Most Polymarket weather traders are manual. Any automation provides latency advantage.
- **Provide liquidity in dead hours:** Between 11 PM and 7 AM ET, spreads are 8–15¢. Placing limit orders at mid-market earns the spread as a maker.
- **Trade against retail flow:** When a large retail buy pushes a bucket up >5¢ without model justification, fade it within 30 minutes.

---

### Section 5 — User's Alpha Pattern → Formal Strategy

Content to write:

**5.1 Codified Trading Rule: "Night Owl Accumulation"**

```python
class NightOwlAccumulation:
    """
    Buy mispriced contracts between 11 PM and 6 AM ET when overnight
    model runs haven't been reflected in market prices.
    """
    WINDOW_START_ET = 23  # 11 PM ET
    WINDOW_END_ET = 6     # 6 AM ET
    MIN_EDGE = 0.12       # 12% minimum true edge
    MAX_ENTRY_PRICE = 0.30  # Only buy contracts ≤ 30¢
    MIN_ENTRY_PRICE = 0.05  # Skip dust (likely to expire worthless)
    FORECAST_DELTA_THRESHOLD = 1.5  # °F shift in overnight models

    PROFIT_TAKE_TIERS = [
        (0.10, 0.50),   # +10¢ → sell 50%
        (0.20, 0.25),   # +20¢ → sell 25%
        # remaining 25% = moon bag
    ]

    STOP_LOSS = -0.05  # Exit if position drops 5¢ from entry

    def detect(self, signals, hour_et, forecast_deltas):
        if not (hour_et >= 23 or hour_et < 6):
            return []

        candidates = []
        for sig in signals:
            if (sig.true_edge >= self.MIN_EDGE
                and self.MIN_ENTRY_PRICE <= sig.yes_ask <= self.MAX_ENTRY_PRICE
                and sig.yes_ask_depth >= 10
                and any(abs(d.delta_f) >= self.FORECAST_DELTA_THRESHOLD
                       for d in forecast_deltas)):
                candidates.append(sig)

        return sorted(candidates, key=lambda s: s.true_edge, reverse=True)[:3]

    def enter(self, signal):
        # Place limit order at mid-market (patient fill in thin overnight market)
        limit_price = (signal.yes_bid + signal.yes_ask) / 2
        return LimitOrder(
            side="buy_yes",
            price=round(limit_price, 2),
            size=self.compute_size(signal),
            time_in_force=1800,  # 30 min
        )

    def compute_size(self, signal):
        # 1.5x Kelly for high-conviction overnight window
        base_kelly = kelly_fraction(signal.model_prob, signal.yes_ask)
        return min(
            base_kelly * 1.5 * bankroll,
            bankroll * 0.15,          # 15% max single position
            signal.yes_ask_depth * 0.20 * signal.yes_ask  # 20% of liquidity
        )
```

**5.2 Data-Driven Justification**
- **Forecast schedule:** 00z HRRR completes ~01:00 ET, 00z GFS completes ~03:30 ET. These are the freshest overnight models.
- **Liquidity cycle:** Polymarket weather market volume drops >80% between 11 PM and 7 AM ET (observable from `market_snapshots` timestamp analysis).
- **Price stickiness:** Without active repricing, market prices at 11 PM ET have >90% autocorrelation with prices at 6 AM ET — even when forecasts shift.
- **Asymmetry:** Contracts at 8–30¢ have expected max loss of 8–30¢ but expected max gain of 70–92¢. Risk/reward ratio of 2.3:1 to 11.5:1.

**5.3 Expected Volatility Window**
- **11 PM–2 AM ET:** Lowest liquidity. Best entry prices but slowest fills. Place passive limit orders.
- **2 AM–5 AM ET:** 00z models available. If forecast shifted, edge is confirmed. Can walk up to ask for faster fill.
- **5 AM–7 AM ET:** Early risers begin trading. First repricing wave. Prices may move 3–8¢.
- **7 AM–10 AM ET:** Main repricing wave. Prices converge to model output. This is when profit-taking triggers fire.
- **10 AM–12 PM ET:** Full repricing complete. Any remaining edge is gone.

**5.4 Optimal Profit-Taking Cadence**
1. **At entry + 10¢ (or +50% of entry price, whichever is smaller):** Sell 50% of position. This recovers most of the cost basis.
2. **At entry + 20¢ (or +100% of entry price):** Sell 25% of position. Cost basis is now fully recovered.
3. **Remaining 25% = moon bag:** Hold to resolution. If the bucket wins, this 25% pays out at $1.00 per share — pure profit on zero remaining cost basis.

**5.5 Risk Rating: MODERATE-HIGH**
- Win rate estimate: 45–55% (based on edge threshold + overnight model confirmation)
- Average win: +35¢ (partial profit-take + occasional resolution win)
- Average loss: -15¢ (stop-loss or expiry at 0)
- Expected Sharpe: 1.2–1.8 (high due to asymmetric payoff)
- Key risk: overnight model runs confirm NO change — position stagnates at entry price with wide spread, making exit costly.

---

### Section 6 — Anomaly / Wrong-Signal Exit Logic

Content to write:

**6.1 Exit Cascade (ordered by urgency)**

```python
class ExitEngine:
    """Checks every 60 seconds for all open positions."""

    def check_exits(self, position, signals, forecasts, market):
        # LEVEL 1: EMERGENCY EXIT (execute immediately as market order)
        if self.observation_contradiction(position, forecasts):
            return ExitSignal("EMERGENCY", "METAR contradicts position", sell_pct=1.0, order_type="market")

        if self.liquidity_danger(position, market):
            return ExitSignal("EMERGENCY", "Liquidity collapse detected", sell_pct=1.0, order_type="market")

        # LEVEL 2: URGENT EXIT (execute within 5 minutes as aggressive limit)
        if self.forecast_shift_against(position, forecasts):
            return ExitSignal("URGENT", "Forecast shifted >2σ against position", sell_pct=0.75, order_type="limit_aggressive")

        if self.ensemble_divergence(position, forecasts):
            return ExitSignal("URGENT", "HRRR/GFS diverged >3°F", sell_pct=0.50, order_type="limit_aggressive")

        # LEVEL 3: CAUTION EXIT (execute within 30 minutes as passive limit)
        if self.ensemble_spread_widening(position, forecasts):
            return ExitSignal("CAUTION", "Ensemble spread widened >50%", sell_pct=0.50, order_type="limit_passive")

        if self.edge_evaporated(position, signals):
            return ExitSignal("CAUTION", "True edge fell below 0.02", sell_pct=0.50, order_type="limit_passive")

        # LEVEL 4: PROFIT-TAKE (standard tiered exits)
        profit_take = self.check_profit_tiers(position, market)
        if profit_take:
            return profit_take

        return None  # Hold position
```

**6.2 Specific Exit Conditions**

| Condition | Detection Logic | Action |
|-----------|----------------|--------|
| **Forecast shift >2σ against** | `abs(new_mu - old_mu) > 2 * sigma` AND direction opposes position | Sell 75% immediately |
| **Ensemble spread widens >50%** | `new_spread / old_spread > 1.5` within 2 consecutive runs | Sell 50%, tighten stop |
| **HRRR vs GFS diverge >3°F** | `abs(hrrr_forecast - gfs_forecast) > 3.0` | Sell 50%, wait for next run |
| **Extreme outlier run** | Any single model run > 3σ from ensemble mean | Pause entries, hold existing |
| **METAR contradicts position** | `observed_temp` already exceeds target bucket ceiling (for high-side bets) or is falling below floor with <3h to peak | Emergency exit 100% |
| **Spread blowout** | `current_spread > 3 * spread_1hr_avg` | Emergency exit — MM pulled liquidity |
| **Volume spike contra** | >3x normal 15-min volume on the opposite side of our position | Sell 50%, investigate |
| **Stop-loss** | `current_price < entry_price - 0.05` (5¢ absolute) | Exit 100% at limit |
| **Time decay** | Position held >8 hours with <3¢ movement | Exit 50% to free capital |

---

### Section 7 — Missing Components for Elite Performance

Content to write (prioritized list):

**Priority 1 — High Impact, Buildable Now:**

1. **Moon-bag position manager** — Tiered profit-taking + cost-basis tracking. Described in Section 3.2. Critical for the Night Owl strategy.

2. **Forecast delta tracker** — Store and emit events on forecast changes. Enables Strategies 1, 2, 5, 7. New table + event system.

3. **Strategy orchestrator** — Multi-strategy coordination. Without this, only one strategy runs at a time.

4. **Enhanced exit engine** — The current system has no exit logic besides manual sells and resolution. This is the single biggest gap.

5. **ECMWF ingestion** — Add ECMWF via Open-Meteo's `ecmwf_ifs04` model. Gold-standard ensemble with 51 members. Enables ensemble spread compression strategy.

**Priority 2 — High Impact, Moderate Effort:**

6. **Streaming METAR/ASOS ingestion** — Replace 60s polling with direct MADIS or NOAA IEM websocket feed. Reduces observation latency from ~90s to ~15s.

7. **Historical forecast error pattern recognition** — Build a database of `(city, month, synoptic_pattern) → forecast_error_distribution`. Use to detect systematic biases (e.g., "GFS always overforecasts Phoenix in July by 2°F").

8. **Probabilistic meta-model** — Replace weighted average with a stacking ensemble (Ridge regression or LightGBM) trained on historical forecasts vs outcomes. Features: each model's forecast, spread, time-of-day, day-of-year, recent bias, synoptic regime.

9. **Predictive liquidity model** — Train on `market_snapshots` history to predict spread/depth by hour-of-day and day-of-week. Use to optimize order timing and avoid thin-market entries.

10. **Temperature ceiling estimation** — Physics-based algorithm using dewpoint, cloud cover, solar angle, and wind to estimate the theoretical maximum temperature for the day. Compare to model forecasts for sanity checking.

**Priority 3 — Advanced / Research:**

11. **Reinforcement learning position sizing** — Replace fixed Kelly fraction with an RL agent (PPO or SAC) trained on historical trades. State: edge, spread, depth, time-to-resolution, portfolio exposure. Action: position size as % of Kelly. Reward: realized P&L.

12. **Cross-venue arbitrage layer** — If temperature contracts appear on multiple prediction markets (Kalshi, Polymarket, etc.), arb price differences. Requires multi-venue API integration.

13. **Model degradation index** — Composite score of: forecast spread, bias trend, recent error magnitude, METAR-forecast divergence. When index > threshold, reduce all position sizes automatically.

14. **Auto-hedging across buckets** — When holding a YES position on bucket [80–84°F], automatically buy cheap NO on [85–89°F] if spread is tight. Creates a bounded-risk structure.

15. **Intraday heat-rate curves** — Model the temperature trajectory as a function of solar radiation, wind, humidity using physics equations. Compare to METAR observations to detect anomalies early.

16. **NWS forecast discussion NLP** — Parse the free-text NWS Area Forecast Discussion for phrases like "temperatures may exceed guidance" or "model spread is unusually high." Use as a qualitative signal overlay.

17. **Order flow toxicity detection** — Implement VPIN (Volume-Synchronized Probability of Informed Trading) adapted for prediction markets. When VPIN > threshold, widen entry limits or pause trading.

---

### Section 8 — Implementation Pseudocode Appendix

Include these pseudocode blocks in the document:

**A. Strategy Detection Pipeline**
```python
async def run_strategy_cycle(strategies, signals, forecasts, market_state):
    all_candidates = []
    for strategy in strategies:
        candidates = strategy.detect(signals, forecasts, market_state)
        for c in candidates:
            c.strategy_tag = strategy.name
            c.priority = strategy.base_priority
            all_candidates.append(c)

    # Resolve conflicts: same bucket, different strategies
    by_bucket = group_by(all_candidates, key=lambda c: c.bucket_id)
    final = []
    for bucket_id, group in by_bucket.items():
        if all_same_direction(group):
            # Multiple strategies agree → highest edge wins, boost size
            winner = max(group, key=lambda c: c.true_edge)
            winner.confidence_boost = len(group)  # agreement bonus
            final.append(winner)
        else:
            # Strategies disagree → skip this bucket
            log.warning(f"Strategy conflict on {bucket_id}, skipping")

    # Portfolio-level risk check
    final = apply_portfolio_limits(final)
    return sorted(final, key=lambda c: c.true_edge, reverse=True)
```

**B. Moon-Bag Exit Cascade**
```python
async def manage_position_exits(position, current_price):
    entry = position.avg_cost
    peak = max(position.peak_price, current_price)
    position.peak_price = peak

    # Tiered profit-taking
    for tier_price, tier_pct in position.profit_take_schedule:
        if current_price >= tier_price and not position.tier_executed(tier_price):
            qty_to_sell = position.initial_qty * tier_pct
            await place_sell_order(position, qty_to_sell, current_price)
            position.mark_tier_executed(tier_price)

    # Check cost basis recovery
    total_recovered = sum(s.price * s.qty for s in position.sells)
    if total_recovered >= position.initial_cost:
        position.moon_bag_flag = True
        position.effective_risk = 0.0

    # Trailing stop on moon bag (only if >2x entry and dropping)
    if position.moon_bag_flag and peak > entry * 2.0:
        if current_price < peak * 0.80:  # 20% drawdown from peak
            await place_sell_order(position, position.remaining_qty, current_price)
            return "CLOSED_TRAILING_STOP"

    return "HOLDING"
```

**C. Ensemble Divergence Detector**
```python
def detect_ensemble_divergence(city_id, forecasts):
    hrrr = forecasts.get("hrrr", {}).get("temperature_f")
    gfs = forecasts.get("gfs", {}).get("temperature_f")
    nws = forecasts.get("nws", {}).get("temperature_f")
    wu = forecasts.get("wu_daily", {}).get("temperature_f")

    available = [v for v in [hrrr, gfs, nws, wu] if v is not None]
    if len(available) < 3:
        return None

    spread = max(available) - min(available)
    mean = sum(available) / len(available)

    # Check pairwise divergence
    divergences = []
    if hrrr and gfs:
        d = abs(hrrr - gfs)
        if d > 3.0:
            divergences.append(("HRRR_GFS", d))
    if hrrr and nws:
        d = abs(hrrr - nws)
        if d > 4.0:
            divergences.append(("HRRR_NWS", d))

    return {
        "spread_f": spread,
        "mean_f": mean,
        "divergences": divergences,
        "confidence": "LOW" if spread > 5.0 else "MEDIUM" if spread > 3.0 else "HIGH",
        "models_available": len(available),
    }
```

**D. Volatility-Adjusted Position Sizing**
```python
def compute_adjusted_size(signal, sigma, recent_win_rate, bankroll):
    # Base Kelly
    base = kelly_fraction(signal.model_prob, signal.yes_ask) * bankroll

    # Volatility adjustment
    if sigma > 5.0:
        vol_mult = 0.50
    elif sigma > 3.5:
        vol_mult = 0.75
    else:
        vol_mult = 1.0

    # Performance adjustment (adaptive Kelly)
    if recent_win_rate > 0.60:
        perf_mult = min(1.3, 1.0 + (recent_win_rate - 0.60) * 2.0)
    elif recent_win_rate < 0.40:
        perf_mult = max(0.5, 1.0 - (0.40 - recent_win_rate) * 2.0)
    else:
        perf_mult = 1.0

    # Strategy multiplier
    strat_mult = signal.strategy.size_multiplier  # e.g., 1.5 for NightOwl

    adjusted = base * vol_mult * perf_mult * strat_mult

    # Hard caps
    return min(
        adjusted,
        bankroll * 0.15,                                    # 15% of bankroll
        signal.yes_ask_depth * 0.20 * signal.yes_ask,      # 20% of liquidity
        max(1.0, adjusted),                                 # $1 minimum
    )
```

---

### Verification Plan

After implementation:
1. **Unit tests** for each strategy's `detect()` method using historical signal data
2. **Backtest harness**: Replay historical `model_snapshots` + `market_snapshots` + `forecast_obs` through strategy orchestrator, simulate fills, compute P&L
3. **Paper trading mode**: Run all strategies with `PAPER_TRADE=True` flag — log trades to journal but don't execute CLOB orders
4. **Staged rollout**: Enable one strategy at a time, starting with Night Owl (highest expected Sharpe), run for 7 days before adding next strategy
5. **Dashboard integration**: Add strategy-level P&L breakdown, position moon-bag status, exit cascade state to existing dashboard

---

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `docs/trading_strategy_framework.md` | CREATE | Master strategy document (this plan's output) |
| `backend/strategy/orchestrator.py` | CREATE | Multi-strategy orchestration engine |
| `backend/strategy/base_strategy.py` | CREATE | Abstract base class for strategies |
| `backend/strategy/night_owl.py` | CREATE | Night Owl Accumulation strategy |
| `backend/strategy/post_model_run.py` | CREATE | Post-Model-Run Repricing strategy |
| `backend/strategy/ensemble_compression.py` | CREATE | Ensemble Spread Compression strategy |
| `backend/strategy/convexity.py` | CREATE | High-Temperature Bucket Convexity |
| `backend/strategy/metar_momentum.py` | CREATE | Intraday METAR Trajectory Momentum |
| `backend/strategy/wu_arb.py` | CREATE | Settlement Source Arbitrage |
| `backend/execution/position_manager.py` | CREATE | Moon-bag + tiered profit-taking |
| `backend/execution/exit_engine.py` | CREATE | Anomaly detection + exit cascade |
| `backend/engine/microstructure.py` | CREATE | Market microstructure monitor |
| `backend/modeling/forecast_delta.py` | CREATE | Forecast change tracker + events |
| `backend/storage/models.py` | MODIFY | Add strategy_tag, peak_price, moon_bag fields |
| `backend/worker/scheduler.py` | MODIFY | Add new job entries |
| `backend/config.py` | MODIFY | Add strategy-specific config params |
