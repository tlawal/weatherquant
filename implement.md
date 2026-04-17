 # WeatherQuant — WU Forecast Removal, TGFTP Settlement Primary, MADIS Benchmark, Stations Page Fix

## Context

Four user-reported items, bundled into two PRs (user chose "Fix #4 first, then #1–#3 together"):

1. `wu_daily` (HTML scrape of the Wunderground daily forecast page) is unreliable — past afternoon it draws the overnight-low value because the page rolls over to the night forecast card. Drop it from the ensemble. Keep `wu_hourly` (weather.com API) and `wu_history` (settlement source).
2. `WU Settlement High` (backed by `wu_history` weather.com API) is slow. Make **TGFTP METAR** (`https://tgftp.nws.noaa.gov/data/observations/metar/stations/{STATION}.TXT`) the primary signal for the settlement-high card, with `wu_history` as fallback. Display temp in °F for US cities, obs time in city's TZ, and a source badge.
3. Add a **second current-temp reading** for US cities via MADIS HFMETAR (netCDF over HTTPS) so we can measure whether MADIS or TGFTP updates faster. This is **benchmarking-only** — it does not feed downstream trading logic.
4. `https://weatherquant.up.railway.app/stations` is stuck on "Computing 30-day station calibration…" and the Diagnostics panel shows `—`. Root cause is a FastAPI route-ordering bug plus a likely data-availability issue that the UI fails to surface.

---

## PR 1 — Fix /stations page (ship first, verify in prod, then start PR 2)

### Root cause #1 — route ordering shadow
In `backend/api/routes.py`, the GET catch-all `/api/station-calibrations/{station_id}` is registered at **line 2090**, but these sibling routes are registered AFTER it:

- `/api/station-calibrations/diagnostics` (GET) — line 2129 ← **shadowed**, always 404s as `station_id="diagnostics"`
- `/api/station-calibrations/auto-refresh` (POST) — line 2160 ← safe (different method)
- `/api/station-calibrations/refresh` (POST) — line 2183 ← safe (different method)

Only the GET `/diagnostics` is broken. That's why the JS `fetchDiagnostics()` (`web/templates/stations.html:395-400`) never populates `this.diag` and the panel stays at `—`.

### Root cause #2 — UI can't distinguish "broken" from "0 stations qualify"
`refresh_all_station_calibrations()` (`backend/modeling/station_calibration.py:286-316`) silently returns `0` when no city has (a) `enabled=True`, (b) `metar_station`, and (c) ≥3 days of observed daily highs (check at line 185). The frontend (`stations.html:366-389`) polls every 5s indefinitely with no timeout and no error surfacing — the page is stuck forever when the refresh legitimately completes with 0 rows.

### Changes

**`backend/api/routes.py`**
- Move the handler at line 2090 (`get_station_calibration_detail` for `/{station_id}`) to be registered **after** `/diagnostics`, `/auto-refresh`, `/refresh`, and `/csv`. This is a pure reordering — no logic change.
- In `/api/station-calibrations/diagnostics` (line 2129), add two fields so the UI can explain emptiness:
  - `enabled_cities_count` — `SELECT COUNT(*) FROM cities WHERE enabled=True`
  - `cities_with_station_count` — `SELECT COUNT(*) FROM cities WHERE enabled=True AND metar_station IS NOT NULL`
  - Keep the existing `row_count`, `last_updated`, `refresh_in_progress`, `error` fields.

**`web/templates/stations.html`**
- After the `auto-refresh` POST, stop polling after **60 seconds with no new data** and show an explanatory message using the new diagnostics fields — e.g. "0 of N cities have sufficient 30-day history" or "0 cities are enabled." Use `setTimeout` alongside the existing `setInterval` and `clearInterval` on either success or timeout (lines 366-389).
- Update the Diagnostics `<details>` block (lines 73-76 region) to render the new counts.

**Verification (PR 1)**
- `curl -s localhost:8000/api/station-calibrations/diagnostics | jq` returns JSON with all fields populated (not 404).
- Load `/stations` in a browser: Diagnostics panel shows real values within 2s.
- If `row_count=0`, the UI stops polling after 60s and explains why.

### Critical files (PR 1)
- `backend/api/routes.py` — lines 2019-2194 (the station-calibrations route group)
- `web/templates/stations.html` — `init()` at 349, `fetchData()` at 355, `fetchDiagnostics()` at 395, Diagnostics panel HTML near line 73

### Note on multi-worker state
`_cal_refresh_in_progress` at `routes.py:2126` is a module-global and is not shared across Railway workers. Not a blocker for PR 1, but document in the final plan that the "already_running" guard only holds within a single process.

---

## PR 2 — Bundled: remove wu_daily, add TGFTP primary settlement, add MADIS benchmark

Internally structured as three sub-parts (2A, 2B, 2C) that can be landed as separate commits on one branch. Recommend reviewing in this order because 2B+2C depend on an indexed `source` column we add once.

### 2A — Remove `wu_daily` (HTML scrape) from ensemble and everywhere downstream

#### Ingestion
**`backend/ingestion/forecasts.py`**
- Delete `_scrape_wu_daily()` (lines 617-693).
- In `_scrape_wu_city()` (lines 440-612): remove the `skip_daily` parameter, the `daily_high` variable and its try/except, the floor-against-wu_daily logic (lines 517-541), and the `insert_forecast_obs(source="wu_daily", ...)` call (lines 543-586 region). Keep the `wu_hourly` and `wu_history` inserts.
- In `fetch_wu_all()` (lines 364-437): remove `last_daily` rate-limit check (lines 389, 393, 407-410, 427). Rate-limit gating now looks at `wu_hourly` and `wu_history` only; verify the cadence is still appropriate (the current `WU_MIN_SCRAPE_INTERVAL_SECONDS=1800` applies whenever any of the checked sources is fresh — dropping daily makes the scraper slightly more likely to skip, which is fine).
- Remove WU HTML-scrape headers if unused — check whether `_WU_HEADERS` at lines 50-61 is still needed after dropping daily. `wu_hourly` and `wu_history` hit weather.com APIs, not wunderground.com HTML, so `_WU_HEADERS` becomes dead.
- Delete `_WU_HEADERS` and the `from bs4 import BeautifulSoup` import if nothing else uses them.

#### Model / calibration / engine
**`backend/engine/signal_engine.py`** — 5 references:
- Line 192: delete the `wu_daily_obs = await get_latest_forecast(...)` fetch.
- Line 279: remove `"bias_wu_daily": cal.bias_wu_daily` from the calibration dict.
- Line 284: remove `"weight_wu_daily": cal.weight_wu_daily`.
- Line 343: update the max-over-observations list from `[nws_obs, wu_daily_obs, wu_hourly_obs]` → `[nws_obs, wu_hourly_obs]`.
- Line 392: delete `wu_daily_high=wu_daily_obs.high_f if wu_daily_obs else None` from the `update_calibration` call.

**`backend/modeling/temperature_model.py`**
- Remove wu_daily from the ensemble description (line 6-9, 281).
- Delete the `_drop_wu_daily` physics-floor block and references (lines 313, 335, 339-340, 351, 354, 369-398, 403-404, 611) — these only mitigated the HTML-scrape drift we're removing.
- Rebalance default weights: if the default was 1/3 each across `{nws, wu_daily, wu_hourly}`, the new default is 1/2 each for `{nws, wu_hourly}`. Normalize in code so existing `CalibrationParams` rows (which have `weight_wu_daily≈0.334` baked in) get renormalized at read time rather than doing a destructive DB migration.

**`backend/modeling/calibration.py`**
- Remove `bias_wu_daily` and `weight_wu_daily` from the defaults dict at lines 25-26.
- Remove the getter line at 44-46.
- In `update_calibration()` (lines 87-95), drop the `wu_daily_forecast` kwarg and its EWMA update. Callers in `signal_engine.py:392` are updated in lockstep.

**`backend/modeling/station_calibration.py`**
- Line 190: remove `"wu_daily"` from the `sources` list used in the per-source MAE comparison.

#### Storage
**`backend/storage/models.py`**
- `CalibrationParams` table (lines 446-474): **leave `bias_wu_daily` and `weight_wu_daily` columns in place** but stop reading/writing them. Dropping columns on Postgres is irreversible-on-rollback; SQLite can't drop at all without table rebuild. Marking them "deprecated" in a code comment is sufficient — can be cleaned up in a dedicated migration PR later.
- Update the `ForecastObs` docstring/comment at lines 207, 215 to remove `wu_daily` from the source enumeration.

**`backend/storage/db.py`**
- Leave the existing schema migrations for `bias_wu_daily`/`weight_wu_daily` alone — they no-op on existing deploys.

#### API/web
**`backend/api/routes.py`** — remove wu_daily from API outputs:
- Lines 338, 350 (wu_daily fetch for dashboard) — delete the fetch.
- Lines 369-371 (all-three-WU-sources fetch) — reduce to wu_hourly + wu_history.
- Lines 403-410 — remove the `"wu_daily": { ... }` block from the `/city/<slug>` response.
- Lines 1487-1491 — remove `bias_wu_daily` / `weight_wu_daily` from the calibration response (if present).

**`web/routes.py`**
- Lines 324-326 — remove the wu_daily fetch in the batch.
- Lines 663-681 — remove the `"wu_daily"` entry from the `forecasts` dict passed to `city.html`.

**`web/templates/city.html`**
- Line 281 — update the Model popover text: drop "WU Daily" from the "weighted ensemble of 5 forecast sources" list; update to 4.

#### Tests (all updated in the same commit to avoid a red state on master)
- `tests/test_forecasts.py` — delete all `_scrape_wu_daily`-specific tests (~30+ cases). Keep wu_hourly and wu_history tests.
- `tests/test_api_routes.py` — remove assertions on `wu_daily` keys in response shape.
- `tests/test_temperature_model.py` — update ensemble fixtures; remove wu_daily inputs.
- `tests/test_market_context.py` — same.

#### Docs
- `README.md` — lines 207-258 on WU Forecast: prune the `wu_daily` description; keep `wu_hourly` and `wu_history`. Line 473 rate-limiting note updates.
- `trading.md` — grep and update.

---

### 2B — TGFTP METAR as primary for the "Settlement High" card

#### New ingestion source
**New file: `backend/ingestion/tgftp_metar.py`** — follows the pattern of `backend/ingestion/metar.py`:
- `fetch_tgftp_all()`: iterates enabled cities with `metar_station`. Staggers requests (e.g. `asyncio.sleep(1)` between cities) to be polite to NOAA — TGFTP has no formal rate-limit header.
- For each city: `GET https://tgftp.nws.noaa.gov/data/observations/metar/stations/{STATION}.TXT` with `User-Agent: WeatherQuant/1.0 (contact@weatherquant.local)` (same as `metar.py:34`).
- Parse the response:
  - Discard the first line (the date header like `2026/04/14 18:53`).
  - Parse the **last** METAR line (TGFTP sometimes appends SPECI reports — the last is freshest).
  - Reuse the existing regex `r"\b(M?)(\d{2})/(M?\d{2})\b"` (already in `metar.py:_parse_temp` at line 86) to extract temp in °C; convert to °F via `_c_to_f`.
  - Extract the `DDHHMMZ` token (e.g. `141853Z`) → UTC datetime. Handle month rollover: if the parsed day > today's UTC day, assume it belongs to the previous month.
- Write to `MetarObs` with `source="tgftp"` (see schema change below). Dedupe using existing `get_metar_obs_by_key(session, metar_station, observed_at)` from `repos.py:23` import list — only insert if no row exists for the same `(station, observed_at)`.
- Timeout 15s, 3 retries with `sleep(2**attempt)` backoff, matching the existing pattern in `forecasts.py:828-861`.

**New scheduler job in `backend/worker/scheduler.py`** (around line 245):
- `add(job_fetch_tgftp_metar, seconds=60, name="fetch_tgftp_metar")`
- Wrap in `_run_with_heartbeat(...)` (same pattern as existing jobs, lines 23-38).

#### Schema change (one-time migration, shared by 2B + 2C's benchmarking table)
**`backend/storage/models.py`** — `MetarObs` at line 137-158:
- Add `source: Mapped[str] = mapped_column(String(16), default="aviation", nullable=False, index=True)` — indexed so per-source queries are fast.
- Add an index `Index("ix_metar_source_station_ts", "source", "metar_station", "observed_at")`.

**`backend/storage/db.py`** — add idempotent DDL for Postgres + SQLite:
- `ALTER TABLE metar_obs ADD COLUMN source VARCHAR(16) NOT NULL DEFAULT 'aviation'`
- Backfill: existing rows auto-get `'aviation'` via the default.
- Create the new index if not exists.

**`backend/storage/repos.py`**
- `insert_metar_obs()` — accept and persist `source` kwarg.
- Add `get_daily_high_metar(sess, city_id, date_et, city_tz, source=None)` — when `source` is provided, filter on it; when None, aggregate across all sources (preserves current behavior).
- Existing aviation polling passes `source="aviation"`; new TGFTP polling passes `source="tgftp"`.

#### Settlement-high resolution flip
**`backend/market_context/service.py:672-700`** — `_resolve_realized_high()`:
- New fallback chain:
  1. **Primary**: `get_daily_high_metar(sess, city.id, date_et, city_tz, source="tgftp")` — use max TGFTP-sourced METAR temp for the day
  2. **Fallback 1** (current primary): `get_latest_successful_forecast(sess, city.id, "wu_history", date_et)`
  3. **Fallback 2**: `get_resolution_high_metar(...)` (unchanged)
  4. **Fallback 3**: `get_daily_high_metar(sess, city.id, date_et, city_tz)` (unsourced — aggregate all METAR)
- Return tuple `(high_f, source_used, obs_time)` so the UI can render the source badge and time.

**Feature-flag primary source via env var** — add `SETTLEMENT_HIGH_PRIMARY` config (default `"tgftp"`, fallback to `"wu_history"` if unset or set to `wu_history`). This gives a cheap rollback without a code change if TGFTP temperatures diverge from WU in unexpected ways during the first few trading days.

#### UI changes
**`web/templates/city.html:209-259`** — the "WU Settlement High" card:
- Rename card title to **"Settlement High"** (source-agnostic).
- Main value: show the resolved high in `°F` (US) or `°C` per `city.unit`. Render `city.unit` directly rather than hardcoding F.
- Below the value, render:
  - Source badge: `TGFTP` / `WU` / `Aviation` — color-coded (green for TGFTP, amber for WU, grey for Aviation fallback).
  - Obs time: format in `{city.tz}` via a new Jinja filter or an existing util. Show as `HH:MM zzz` (e.g. `1:53 PM EDT`).
- Keep the info popover but update it to describe the new fallback chain in plain language.

**`web/routes.py`** — when building the context dict for `city.html`:
- Call `_resolve_realized_high` (or a new thin wrapper) and pass `{high_f, source_used, obs_time_local}` to the template as `settlement_high`.

#### Verification (2B)
- `curl -s "https://tgftp.nws.noaa.gov/data/observations/metar/stations/KATL.TXT"` returns raw text — parse correctly in `pytest tests/test_tgftp_metar.py`.
- After one hour of running locally: `SELECT source, COUNT(*) FROM metar_obs GROUP BY source;` shows both `aviation` and `tgftp` rows.
- `/city/atlanta` shows the new Settlement High card with "TGFTP" badge.
- Kill TGFTP (block via `/etc/hosts`) — the card should degrade to "WU" source.

---

### 2C — MADIS HFMETAR as a benchmark-only second reading

#### Deployment spike (do this FIRST before writing code — answer before committing to netCDF4)
- `netCDF4` Python requires `libhdf5-dev`, `libnetcdf-dev`, and a C compiler. Current `Dockerfile` (lines 4-5) only installs `libxml2-dev libxslt-dev gcc`. Adding HDF5/netCDF adds ~150-200MB to the image.
- **Action**: add to Dockerfile and rebuild locally to verify image builds and boots. If the image blows past Railway's layer-size limit or pushes startup memory over the plan's budget, fall back to the **text/XML sfcdumpguest** interface (user's original second option) as a scoped pivot. Document the fallback in the final PR description.

Assuming the spike succeeds:

#### New benchmark table (do NOT mix with MetarObs)
**`backend/storage/models.py`** — new model:

```python
class MadisObs(Base):
    __tablename__ = "madis_obs"
    __table_args__ = (
        Index("ix_madis_station_ts", "metar_station", "observed_at"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    metar_station: Mapped[str] = mapped_column(String(8), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    temp_c: Mapped[Optional[float]] = mapped_column(Float)
    temp_f: Mapped[Optional[float]] = mapped_column(Float)
    source_file: Mapped[Optional[str]] = mapped_column(String(64))  # e.g. "20260414_2000.gz"
```

Add DDL + index creation in `backend/storage/db.py`.

#### New ingestion
**New file: `backend/ingestion/madis_hfmetar.py`**:
- `fetch_madis_latest()`:
  - Compute wallclock UTC rounded to nearest 5-minute mark (MADIS HFMETAR is at 5-min intervals, 12/hr).
  - Build filename `YYYYMMDD_HHMM.gz`.
  - URL: `https://madis-data.ncep.noaa.gov/madisPublic1/data/LDAD/hfmetar/netCDF/{filename}`.
  - `GET` with `aiohttp`. If 404, step back 5 min; retry up to 6 times (30 min back). Cache the last-successful filename so repeated failures don't re-probe.
  - Gunzip to a `tempfile.NamedTemporaryFile` (MADIS files are ~1-5 MB).
  - Open with `netCDF4.Dataset(path, "r")`.
  - Extract `stationName` (char array — `.tobytes().decode().strip()`), `temperature` (Kelvin), `observationTime` (epoch seconds).
  - Filter to our city metar stations (`get_all_cities(enabled_only=True)`), matching on `metar_station`.
  - Dedupe on `(station, observed_at)` — MADIS files can have multiple obs per station (5-min resolution across a 1-hr window, depending on file).
  - Insert into `madis_obs` via a new `insert_madis_obs()` in `repos.py`.

**New scheduler job** — every 5 minutes: `add(job_fetch_madis, seconds=300, name="fetch_madis")`.

#### Requirements
**`requirements.txt`**:
- Add `netCDF4==1.7.2` (or whatever's current & Python 3.12 compatible).

**`Dockerfile`** (line 4-5):
- Add `libhdf5-dev libnetcdf-dev` to the `apt-get install` line.

#### UI — side-by-side comparison
**`web/templates/city.html`** — new "Current Temp" panel (or augment existing one) showing TWO readings side-by-side for US cities:
- Left: TGFTP (value °F, obs time in city TZ, age-seconds badge)
- Right: MADIS (value °F, obs time in city TZ, age-seconds badge)
- The one with the newest `observed_at` gets a "FASTER" badge.

**`web/routes.py`**:
- Fetch latest MadisObs + latest TGFTP MetarObs per city; pass both to the template.

#### Verification (2C)
- After one cycle: `SELECT COUNT(*) FROM madis_obs WHERE metar_station='KATL';` returns ≥1.
- On `/city/atlanta`: both TGFTP and MADIS readings render with observation timestamps, and the "FASTER" badge flips between them over time as new data arrives.
- Log a per-cycle summary: `madis: fetched N observations for M stations (file=YYYYMMDD_HHMM.gz)`.

---

## Rollout order

1. **Land PR 1.** Deploy to Railway. Verify `/stations` shows real data (or renders an informative emptiness message). Do NOT start PR 2 until `/stations` is green — it's our signal for whether the ensemble changes in PR 2 break anything downstream.
2. **PR 2, commit by commit on one branch:**
   - Commit 2A (wu_daily removal + weight renormalization) — run full test suite locally before push.
   - Commit 2B (TGFTP + `source` column migration) with feature flag `SETTLEMENT_HIGH_PRIMARY=wu_history` initially.
   - Deploy, monitor logs for TGFTP success/failure rate for 24h.
   - Flip `SETTLEMENT_HIGH_PRIMARY=tgftp` (or remove env var so it uses the default `"tgftp"`).
   - Commit 2C (MADIS + Dockerfile deps) — only if the deployment spike confirmed netCDF4 works on Railway.

---

## Critical files

- `backend/api/routes.py` — route ordering + diagnostics enrichment (PR 1)
- `web/templates/stations.html` — poll timeout + empty-state messaging (PR 1)
- `backend/ingestion/forecasts.py` — wu_daily removal + rate-limit gating (2A)
- `backend/engine/signal_engine.py` — 5 wu_daily refs (2A)
- `backend/modeling/temperature_model.py` — ensemble rebalance + drop physics-floor (2A)
- `backend/modeling/calibration.py` — `update_calibration` signature + defaults (2A)
- `backend/modeling/station_calibration.py` — sources list (2A)
- `backend/storage/models.py` — `MetarObs.source` column, new `MadisObs` table (2B + 2C)
- `backend/storage/db.py` — DDL migrations (2B + 2C)
- `backend/storage/repos.py` — sourced `get_daily_high_metar`, `insert_madis_obs` (2B + 2C)
- `backend/ingestion/tgftp_metar.py` — NEW (2B)
- `backend/ingestion/madis_hfmetar.py` — NEW (2C)
- `backend/worker/scheduler.py` — two new jobs (2B + 2C)
- `backend/market_context/service.py` — `_resolve_realized_high` new fallback chain (2B)
- `web/templates/city.html` — Settlement High card redesign + current-temp dual-read panel (2B + 2C)
- `web/routes.py` — context dict for the new template data (2B + 2C)
- `Dockerfile` — HDF5/netCDF4 system deps (2C)
- `requirements.txt` — `netCDF4` (2C)
- Tests across `tests/test_forecasts.py`, `tests/test_api_routes.py`, `tests/test_temperature_model.py`, `tests/test_market_context.py` — wu_daily removal (2A)
- NEW: `tests/test_tgftp_metar.py`, `tests/test_madis_hfmetar.py`
- `README.md`, `trading.md` — docs cleanup (2A)

---

## Reused utilities (do not reinvent)

- `backend/tz_utils.py` — `city_local_date`, `city_local_now` for time display in city TZ (2B + 2C)
- `backend/ingestion/metar.py:_parse_temp` and `_c_to_f` — METAR temp regex + Celsius→Fahrenheit (2B)
- `backend/storage/repos.py:get_metar_obs_by_key` — dedupe primitive (2B)
- `backend/worker/scheduler.py:_run_with_heartbeat` — error-tracking wrapper for new jobs (2B + 2C)

---

## Open items / risks to revisit during implementation

- **Route-fix alone may not make /stations populate** — `refresh_all_station_calibrations` can legitimately return 0 stations. PR 1's added diagnostics fields + UI timeout surface this. If production shows 0 enabled cities, that's a separate investigation (Railway DB state, `cities_admin` page).
- **Weight renormalization in `temperature_model.py`** — existing `CalibrationParams` rows have baked-in `weight_wu_daily≈0.334`. Code must renormalize on read, not rely on a migration. Spot-check `signal_engine.py:343` (the max-over-observations list) handles single-source gracefully.
- **Kalshi/Polymarket settlement source wording** — renaming "WU Settlement High" → "Settlement High" is user-facing. Confirm the contract language still reads WU (it does per `polymarket_gamma.py`'s `_is_wu_source`), and keep the info popover explaining the resolution source chain.
- **Railway Dockerfile rebuild** — the 2C spike is a gate. If HDF5 deps bloat the image or Railway's build fails, defer MADIS to a follow-up PR and ship just 2A+2B.
- **Multi-worker note (`_cal_refresh_in_progress`)** — not fixed in PR 1, but documented for later hardening.