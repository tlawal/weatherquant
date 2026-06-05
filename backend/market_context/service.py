"""
Market Context analytics builder, deterministic selection engine, and snapshot orchestration.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from backend.city_registry import CITY_REGISTRY_BY_SLUG
from backend.config import Config
from backend.market_context.adapter import (
    MarketContextLLMAdapter,
    MarketContextLLMError,
    market_context_provider_ready,
)
from backend.market_context.types import (
    MarketContextInput,
    MarketContextOutput,
    MarketContextSelection,
    SECTION_ORDER,
    SECTION_KEYS,
)
from backend.modeling.calibration_engine import get_reliability_metrics, remap_probability
from backend.modeling.settlement import bucket_upper_bound, canonical_bucket_ranges, hotter_bucket_floor
from backend.storage.db import get_session
from backend.storage.models import City, MarketContextSnapshot
from backend.storage.repos import (
    get_avg_peak_timing,
    get_buckets_for_event,
    get_calibration,
    get_city_by_slug,
    get_daily_high_metar,
    get_event,
    get_latest_forecast,
    get_latest_market_snapshot,
    get_latest_model_snapshot,
    get_latest_signal_for_bucket,
    get_latest_successful_forecast,
    get_market_context_snapshot,
    get_market_snapshots_for_bucket,
    get_recent_events_for_city,
    get_resolution_high_metar,
    get_station_profile,
    get_todays_extended_obs,
    upsert_market_context_snapshot,
)
from backend.tz_utils import city_local_date

log = logging.getLogger(__name__)


# Forecast sources we surface to the LLM. Order matters for human readability
# in prompt context (physical NWP first, then AI-NWP, then live obs).
_ENSEMBLE_SOURCES: tuple[str, ...] = (
    "nws", "hrrr", "nbm", "ecmwf_ifs", "ecmwf_aifs",
    "gfs_graphcast", "pangu_weather", "fourcastnet_v2", "aurora",
    "wu_hourly",
)


# Source metadata encyclopedia — included in the system prompt so the LLM
# reasons about model strengths/weaknesses from first principles instead
# of decorating output with vague "NBM is good" hand-waves.
#
# Each entry deliberately calls out:
#   - HOW the model is formed (physics? statistical blend? AI architecture?)
#   - HOW the temperature value is calculated (gridpoint extract, blend, etc.)
#   - WHEN it's run (cadence + typical post-init availability)
#   - PROS / CONS in operational use for daily-high prediction
#
# When new sources are added (e.g. FCN3 from §20.6 probe), append here.
MODEL_ENCYCLOPEDIA: dict[str, dict[str, str]] = {
    "nws": {
        "full_name": "NWS National Blend of Models gridpoint",
        "type": "Statistical blend of physics NWP",
        "calculation": "NOAA's National Blend of Models (NBM) gridpoint forecast queried via api.weather.gov; blends GFS, NAM, HRRR, ECMWF, GEFS, others using decaying-average + quantile mapping postprocess.",
        "cadence": "Hourly updates; gridpoint refresh ~every 60 min",
        "typical_age_at_trade_h": "0.5–2",
        "pros": "US-only but excellent for it; runs continuously so always fresh; bias-corrected against ASOS observations.",
        "cons": "Not available outside US. Lower spatial resolution than HRRR (~2.5km vs 3km). Lags behind raw HRRR by 30-60min.",
    },
    "hrrr": {
        "full_name": "High-Resolution Rapid Refresh",
        "type": "Convection-allowing physical NWP",
        "calculation": "NOAA's 3km CONUS regional model; non-hydrostatic equations, hourly cycling with radar data assimilation. We query via Open-Meteo `gfs_hrrr` blend.",
        "cadence": "Hourly init; output available ~45-60 min post-init",
        "typical_age_at_trade_h": "1–3",
        "pros": "Highest spatial resolution we have for US. Excellent at convective cooling, sea-breeze fronts, urban heat island, terrain effects.",
        "cons": "US CONUS only. Known cool bias overnight, warm bias in afternoon convective regimes (Glahn/MDL studies). Drifts at lead > 18h.",
    },
    "nbm": {
        "full_name": "NCEP National Blend of Models — full MDL output",
        "type": "Statistical blend of 50+ physics NWP models",
        "calculation": "NOAA MDL blends GFS, NAM, HRRR, ECMWF, GEFS, SREF, NAEFS, MOS via dynamic MAE-weighted fusion + quantile mapping for bias. We pull via Open-Meteo `ncep_nbm_conus`.",
        "cadence": "Hourly init; output ~90 min post-init",
        "typical_age_at_trade_h": "1.5–3",
        "pros": "10–20% MAE reduction over raw GFS for temperature (NWS MDL verification). Wisdom-of-crowds via dynamic re-weighting. v4.2+ has wind quantile mapping.",
        "cons": "US CONUS only. Slightly stale relative to raw HRRR. Less responsive to sudden mesoscale events because it's a blend.",
    },
    "ecmwf_ifs": {
        "full_name": "ECMWF Integrated Forecasting System",
        "type": "Physical NWP — global gold-standard",
        "calculation": "ECMWF's flagship model: spectral primitive equations, 9km resolution, TCo1279 grid, full 4D-Var data assimilation. We extract 2m-temperature at city lat/lon via Herbie cfgrib + ECMWF open data.",
        "cadence": "4 runs/day (00z, 06z, 12z, 18z); IFS available ~6h post-init",
        "typical_age_at_trade_h": "6–18",
        "pros": "Globally available. Best long-range deterministic skill in the world (consistently beats GFS at lead > 24h). Strong on synoptic-scale features.",
        "cons": "Lower resolution than HRRR for fine-scale convection. Latency: ~6h post-init means stalest of our short-range sources at trade time.",
    },
    "ecmwf_aifs": {
        "full_name": "ECMWF AI Forecasting System",
        "type": "AI-NWP — graph neural network",
        "calculation": "ECMWF's transformer-based AI model trained on ERA5 + IFS analyses. 0.25° single-level. Same physical IC as IFS. Herbie cfgrib + ECMWF AIFS open data.",
        "cadence": "4 runs/day matching IFS schedule; ~6h post-init",
        "typical_age_at_trade_h": "6–18",
        "pros": "Architecturally diverse from IFS (different inductive bias). Strong on global patterns. Trains on full ERA5 history.",
        "cons": "Newer than IFS, less validation in operational use. May still inherit IFS biases via training data.",
    },
    "gfs_graphcast": {
        "full_name": "DeepMind GraphCast (GFS-init)",
        "type": "AI-NWP — graph neural network",
        "calculation": "DeepMind's GNN trained on ERA5 (Lam et al. 2023, Science 382:1416). 0.25° resolution, autoregressive 6h steps. Open-Meteo blends GFS init + GraphCast forward integration.",
        "cadence": "4 runs/day on GFS init schedule (00z/06z/12z/18z); ~5h post-init",
        "typical_age_at_trade_h": "5–17",
        "pros": "Beats GFS on most lead times in WeatherBench-2. Free via Open-Meteo. Architecturally distinct from FCN.",
        "cons": "Initialized on GFS analysis (not IFS) → inherits GFS biases at t=0. Lower fidelity on convective events than HRRR.",
    },
    "pangu_weather": {
        "full_name": "Huawei Pangu-Weather",
        "type": "AI-NWP — 3D Earth-specific transformer",
        "calculation": "Hierarchical 3D transformer with Earth-specific positional encodings (Bi et al. 2023, Nature 619:533). Trained on ERA5. We pull from NOAA AIWP S3 archive (IFS-initialized, 6h timestep, 10-day horizon).",
        "cadence": "2 runs/day (00z, 12z) on AIWP cadence; IFS-init ~8h post-init",
        "typical_age_at_trade_h": "8–20",
        "pros": "One of the strongest AI-NWP models. IFS analysis as IC = high-fidelity start. 3D Earth-specific transformer captures vertical structure well.",
        "cons": "6-hour timestep — true peak T may sit between forecast steps (small under-estimation possible). Older than newer FCN3, may be superseded.",
    },
    "fourcastnet_v2": {
        "full_name": "NVIDIA FourCastNet v2-small",
        "type": "AI-NWP — adaptive Fourier neural operator",
        "calculation": "NVIDIA's spherical FNO (Pathak et al. 2022, 2024 update). 75M parameters. Operates in spectral domain — fast, naturally handles spherical geometry. We pull from NOAA AIWP S3 (IFS-initialized).",
        "cadence": "2 runs/day (00z, 12z); IFS-init ~8h post-init",
        "typical_age_at_trade_h": "8–20",
        "pros": "Strong on tail events. Very fast inference. Architecturally diverse from transformers (FNO operates in spectral domain).",
        "cons": "6-hour timestep limitation same as Pangu. Pre-FCN3 generation; less skill than FCN3 would have at long lead.",
    },
    "aurora": {
        "full_name": "Microsoft Aurora",
        "type": "AI-NWP — Swin transformer foundation model",
        "calculation": "Microsoft's Swin transformer trained on multi-resolution Earth data (Bodnar et al. 2024). Foundation-model architecture lets it generalize across atmospheric tasks. We pull from NOAA AIWP S3 (IFS-initialized).",
        "cadence": "2 runs/day (00z, 12z); IFS-init ~8h post-init",
        "typical_age_at_trade_h": "8–20",
        "pros": "4th distinct AI inductive bias in our ensemble (alongside FCN's spherical FNO, Pangu's 3D transformer, GraphCast's GNN). Foundation-model approach is cutting-edge.",
        "cons": "Newest of our AI sources, least operational track record. Same 6h timestep limitation as other AIWP models.",
    },
    "wu_hourly": {
        "full_name": "Weather Underground hourly forecast",
        "type": "Aggregated commercial blend",
        "calculation": "WU/IBM blends multiple NWP sources + their proprietary microclimate model. Hourly forecasts scraped from weather.com.",
        "cadence": "Updates ~30 min; available continuously",
        "typical_age_at_trade_h": "0.5–1",
        "pros": "Always fresh. Microclimate-aware (urban vs airport). International coverage when others fail.",
        "cons": "Black-box methodology. Inconsistent quality across stations. Has historically had 1–3°F bias on summer afternoons.",
    },
}


class MarketContextBuildError(RuntimeError):
    pass


def serialize_market_context_snapshot(
    snapshot: Optional[MarketContextSnapshot],
) -> Optional[dict[str, Any]]:
    if snapshot is None:
        return None
    return {
        "id": snapshot.id,
        "city_id": snapshot.city_id,
        "date_et": snapshot.date_et,
        "generation_status": snapshot.generation_status,
        "sections": json.loads(snapshot.sections_json) if snapshot.sections_json else None,
        "selection": json.loads(snapshot.selection_json) if snapshot.selection_json else None,
        "source_context": json.loads(snapshot.source_context_json) if snapshot.source_context_json else None,
        "section_order": [
            {"key": key, "label": label}
            for key, label in SECTION_ORDER
        ],
        "provider": snapshot.provider,
        "model_name": snapshot.model_name,
        "generated_at": snapshot.generated_at.isoformat() if snapshot.generated_at else None,
        "freshness_at": snapshot.freshness_at.isoformat() if snapshot.freshness_at else None,
        "last_error": snapshot.last_error,
    }


async def get_market_context_snapshot_payload(
    city_slug: str,
    date_et: str,
) -> Optional[dict[str, Any]]:
    async with get_session() as sess:
        city = await get_city_by_slug(sess, city_slug)
        if city is None:
            return None
        snapshot = await get_market_context_snapshot(sess, city.id, date_et)
    return serialize_market_context_snapshot(snapshot)


async def refresh_market_context_snapshot(
    city_slug: str,
    date_et: str,
) -> dict[str, Any]:
    if not market_context_provider_ready():
        raise MarketContextLLMError("Market Context provider is not configured")

    async with get_session() as sess:
        city = await get_city_by_slug(sess, city_slug)
        if city is None:
            raise MarketContextBuildError(f"Unknown city: {city_slug}")

    source_context = await build_market_context_input(city, date_et)

    adapter = MarketContextLLMAdapter()
    provider = adapter.provider
    model_name = adapter.model
    now = datetime.now(timezone.utc)

    async with get_session() as sess:
        await upsert_market_context_snapshot(
            sess,
            city_id=city.id,
            date_et=date_et,
            generation_status="pending",
            provider=provider,
            model_name=model_name,
            source_context_json=json.dumps(source_context.model_dump()),
            selection_json=json.dumps(source_context.final_selection.model_dump()),
            freshness_at=now,
            last_error=None,
        )

    try:
        generated = await _generate_market_context_output(source_context)
    except Exception as exc:
        async with get_session() as sess:
            snapshot = await upsert_market_context_snapshot(
                sess,
                city_id=city.id,
                date_et=date_et,
                generation_status="failed",
                provider=provider,
                model_name=model_name,
                source_context_json=json.dumps(source_context.model_dump()),
                selection_json=json.dumps(source_context.final_selection.model_dump()),
                freshness_at=now,
                last_error=str(exc),
            )
        return serialize_market_context_snapshot(snapshot) or {}

    async with get_session() as sess:
        snapshot = await upsert_market_context_snapshot(
            sess,
            city_id=city.id,
            date_et=date_et,
            generation_status="success",
            sections_json=json.dumps(generated.sections),
            selection_json=json.dumps(source_context.final_selection.model_dump()),
            source_context_json=json.dumps(source_context.model_dump()),
            provider=provider,
            model_name=model_name,
            generated_at=now,
            freshness_at=now,
            last_error=None,
        )

    return serialize_market_context_snapshot(snapshot) or {}


async def build_market_context_input(city: City, date_et: str) -> MarketContextInput:
    city_tz = ZoneInfo(getattr(city, "tz", "America/New_York"))
    today_local = city_local_date(city)
    target_is_today = date_et == today_local
    now_utc = datetime.now(timezone.utc)

    async with get_session() as sess:
        event = await get_event(sess, city.id, date_et)
        if event is None:
            raise MarketContextBuildError(f"No event found for {city.city_slug} on {date_et}")

        buckets = await get_buckets_for_event(sess, event.id)
        if not buckets:
            raise MarketContextBuildError(f"No buckets found for {city.city_slug} on {date_et}")
        canonical_ranges = canonical_bucket_ranges([(bucket.low_f, bucket.high_f) for bucket in buckets])

        model = await get_latest_model_snapshot(sess, event.id)
        if model is None:
            raise MarketContextBuildError(f"No model snapshot found for {city.city_slug} on {date_et}")

        station_profile = await get_station_profile(sess, city.metar_station) if city.metar_station else None
        observation_minutes = _json_list(station_profile.observation_minutes) if station_profile and station_profile.observation_minutes else []
        obs_rows = await get_todays_extended_obs(sess, city.id, date_et, city_tz=getattr(city, "tz", "America/New_York"))
        latest_obs = obs_rows[-1] if obs_rows else None
        obs_high_f = await get_daily_high_metar(sess, city.id, date_et, city_tz=getattr(city, "tz", "America/New_York"))
        resolution_high_f = None
        if observation_minutes:
            resolution_high_f = await get_resolution_high_metar(
                sess,
                city.id,
                date_et,
                observation_minutes,
                city_tz=getattr(city, "tz", "America/New_York"),
            )

        primary_source = "nws" if city.is_us else "open_meteo"
        primary_fc = await get_latest_forecast(sess, city.id, primary_source, date_et)
        wu_hourly = await get_latest_successful_forecast(sess, city.id, "wu_hourly", date_et)
        wu_history = await get_latest_successful_forecast(sess, city.id, "wu_history", date_et)
        hrrr_fc = await get_latest_successful_forecast(sess, city.id, "hrrr", date_et)
        nbm_fc = await get_latest_successful_forecast(sess, city.id, "nbm", date_et)
        # Surface the full 10-source ensemble to the LLM. Previously only
        # NWS/HRRR/NBM/WU were visible; the LLM had no way to reason about
        # IFS/AIFS/GraphCast/Pangu/FCN/Aurora because those names never
        # appeared in its context.
        ecmwf_ifs_fc    = await get_latest_successful_forecast(sess, city.id, "ecmwf_ifs", date_et)
        ecmwf_aifs_fc   = await get_latest_successful_forecast(sess, city.id, "ecmwf_aifs", date_et)
        gfs_graphcast_fc = await get_latest_successful_forecast(sess, city.id, "gfs_graphcast", date_et)
        pangu_weather_fc = await get_latest_successful_forecast(sess, city.id, "pangu_weather", date_et)
        fourcastnet_v2_fc = await get_latest_successful_forecast(sess, city.id, "fourcastnet_v2", date_et)
        aurora_fc       = await get_latest_successful_forecast(sess, city.id, "aurora", date_et)
        calibration = await get_calibration(sess, city.id)
        recent_events = await get_recent_events_for_city(sess, city.id, before_or_on_date_et=date_et, limit=16)
        avg_peak_timing = await get_avg_peak_timing(sess, city.id, days_back=7, et_tz=city_tz)

        model_probs = json.loads(model.probs_json) if model.probs_json else []
        model_inputs = json.loads(model.inputs_json) if model.inputs_json else {}
        adaptive_inputs = model_inputs.get("adaptive") or {}

        reliability_bins = await get_reliability_metrics(city.id)

        bucket_rows: list[dict[str, Any]] = []
        for bucket in buckets:
            signal = await get_latest_signal_for_bucket(sess, bucket.id)
            market = await get_latest_market_snapshot(sess, bucket.id)
            raw_model_prob = 0.0
            if bucket.bucket_idx < len(model_probs):
                raw_model_prob = float(model_probs[bucket.bucket_idx])
            calibrated_prob = float(remap_probability(raw_model_prob, reliability_bins))
            market_prob = market.yes_mid if market and market.yes_mid is not None else None
            true_edge = signal.true_edge if signal is not None else (
                calibrated_prob - market_prob if market_prob is not None else None
            )

            snapshots_since_start = await get_market_snapshots_for_bucket(
                sess,
                bucket.id,
                since=_start_of_local_day(date_et, city_tz),
            )
            movement = _summarize_bucket_market_movement(snapshots_since_start)

            bucket_rows.append(
                {
                    "bucket_id": bucket.id,
                    "bucket_idx": bucket.bucket_idx,
                    "label": bucket.label or _bucket_label_from_bounds(bucket.low_f, bucket.high_f),
                    "low_f": bucket.low_f,
                    "high_f": bucket.high_f,
                    "settlement_upper_f": bucket_upper_bound(canonical_ranges, bucket.bucket_idx),
                    "next_hotter_floor_f": hotter_bucket_floor(canonical_ranges, bucket.bucket_idx),
                    "raw_model_prob": raw_model_prob,
                    "calibrated_prob": calibrated_prob,
                    "market_prob": market_prob,
                    "true_edge": true_edge,
                    "spread": market.spread if market else None,
                    "bid_depth": market.yes_bid_depth if market else None,
                    "ask_depth": market.yes_ask_depth if market else None,
                    "movement": movement,
                }
            )

        error_summary, last_settled = await _compute_recent_error_summary(
            sess,
            city=city,
            recent_events=recent_events,
            observation_minutes=observation_minutes,
            reference_date_et=date_et,
        )

        recent_highs = await _compute_recent_high_summary(
            sess,
            city=city,
            recent_events=recent_events,
            observation_minutes=observation_minutes,
            reference_date_et=date_et,
        )

    latest_ext = latest_obs.extended if latest_obs else None
    latest_temp_f = latest_obs.temp_f if latest_obs else None
    latest_dewpoint_f = latest_ext.dewpoint_f if latest_ext else None
    dewpoint_spread_f = None
    if latest_temp_f is not None and latest_dewpoint_f is not None:
        dewpoint_spread_f = round(latest_temp_f - latest_dewpoint_f, 1)

    temp_change_1h = _temperature_change(obs_rows, minutes=60)
    temp_change_3h = _temperature_change(obs_rows, minutes=180)
    warming_accel = _warming_acceleration(obs_rows)
    pressure_tendency = _pressure_tendency(obs_rows, minutes=180)
    wind_shift_deg = _wind_shift(obs_rows, minutes=180)
    cloud_trend = _cloud_trend(obs_rows, minutes=180)
    resolution_mismatch = None
    if obs_high_f is not None and resolution_high_f is not None:
        mismatch = round(obs_high_f - resolution_high_f, 1)
        if mismatch >= 1.0:
            resolution_mismatch = mismatch

    latest_obs_time_local = _fmt_local(latest_obs.observed_at, city_tz) if latest_obs else None
    metar_age_s = _age_seconds(latest_obs.observed_at) if (latest_obs and target_is_today) else None
    sources = _summarize_sources(
        primary_fc=primary_fc,
        wu_hourly=wu_hourly,
        hrrr_fc=hrrr_fc,
        nbm_fc=nbm_fc,
        adaptive_inputs=adaptive_inputs,
        target_is_today=target_is_today,
        now_utc=now_utc,
    )

    selection = _select_bucket(
        bucket_rows=bucket_rows,
        current_temp_f=latest_temp_f,
        obs_high_f=obs_high_f,
        forecast_spread_f=model_inputs.get("spread"),
        model_sigma=model.sigma,
        metar_age_s=metar_age_s,
        model_inputs=model_inputs,
        adaptive_inputs=adaptive_inputs,
        source_summary=sources,
        resolution_mismatch=resolution_mismatch,
        cloud_trend=cloud_trend,
    )

    smart_money = await _build_smart_money_context(
        city_slug=city.city_slug,
        date_et=date_et,
        bucket_rows=bucket_rows,
        selected_bucket_idx=selection.bucket_idx,
    )

    availability = {
        "latest_observations": bool(obs_rows),
        "nws_available": primary_fc is not None if city.is_us else False,
        "wu_hourly_available": wu_hourly is not None,
        "wu_history_available": wu_history is not None,
        "adaptive_available": bool(adaptive_inputs),
        "station_pattern_available": bool(observation_minutes),
        "hrrr_available": hrrr_fc is not None,
        "nbm_available": nbm_fc is not None,
        "nam_available": False,
        "rap_available": False,
        "ecmwf_available": False,
        "climatology_available": False,
        "microclimate_memory_available": False,
        "wallet_tracker_available": smart_money.get("status") == "ok",
    }

    return MarketContextInput(
        city_slug=city.city_slug,
        city_display=city.display_name,
        date_et=date_et,
        unit=city.unit or "F",
        availability=availability,
        current_observations={
            "station": city.metar_station,
            "observed_at_local": latest_obs_time_local,
            "current_temp_f": latest_temp_f,
            "observed_high_f": obs_high_f,
            "resolution_high_f": resolution_high_f,
            "resolution_mismatch_f": resolution_mismatch,
            "dewpoint_f": latest_dewpoint_f,
            "dewpoint_spread_f": dewpoint_spread_f,
            "humidity_pct": round(latest_ext.humidity_pct) if latest_ext and latest_ext.humidity_pct is not None else None,
            "wind_dir_deg": latest_ext.wind_dir_deg if latest_ext else None,
            "wind_speed_kt": latest_ext.wind_speed_kt if latest_ext else None,
            "wind_gust_kt": latest_ext.wind_gust_kt if latest_ext else None,
            "cloud_cover": latest_ext.cloud_cover if latest_ext else None,
            "cloud_base_ft": latest_ext.cloud_base_ft if latest_ext else None,
            "condition": latest_ext.condition if latest_ext else None,
            "pressure_tendency_inhg_3h": pressure_tendency,
            "temp_change_1h_f": temp_change_1h,
            "temp_change_3h_f": temp_change_3h,
            "warming_acceleration_f_per_hr_delta": warming_accel,
            "wind_shift_deg_3h": wind_shift_deg,
            "cloud_trend": cloud_trend,
            "metar_age_s": metar_age_s,
            "observation_minutes": observation_minutes,
            "latest_matches_station_pattern": _matches_station_pattern(latest_obs, observation_minutes, city_tz),
        },
        short_range_models=_build_short_range_models(
            model=model,
            model_inputs=model_inputs,
            error_summary=error_summary,
            sources_legacy=sources,
            source_forecasts={
                "nws": primary_fc,
                "hrrr": hrrr_fc,
                "nbm": nbm_fc,
                "ecmwf_ifs": ecmwf_ifs_fc,
                "ecmwf_aifs": ecmwf_aifs_fc,
                "gfs_graphcast": gfs_graphcast_fc,
                "pangu_weather": pangu_weather_fc,
                "fourcastnet_v2": fourcastnet_v2_fc,
                "aurora": aurora_fc,
                "wu_hourly": wu_hourly,
            },
            city_unit=city.unit or "F",
            settlement_at_utc=_settlement_at_utc_for(date_et, city_tz),
            now_utc=now_utc,
        ),
        historical_context={
            "avg_peak_time_local_7d": avg_peak_timing,
            "recent_realized_highs": recent_highs["recent_days"],
            "avg_high_7d_f": recent_highs["avg_7d"],
            "avg_high_prev_7d_f": recent_highs["avg_prev_7d"],
            "trend_delta_f": recent_highs["trend_delta"],
            "last_settled_date_et": last_settled["date_et"] if last_settled else None,
            "last_settled_realized_high_f": last_settled["realized_high_f"] if last_settled else None,
            "last_settled_errors_f": last_settled["errors"] if last_settled else {},
            # Calibration biases for the full ensemble — previously only
            # exposed nws + wu_hourly. Now the LLM can reason about every
            # source's drift direction (e.g. HRRR running +1.2°F warm in
            # last 30 days, IFS running −0.4°F cool).
            "calibration_biases_f": {
                src: (error_summary.get(src) or {}).get("bias_f")
                for src in _ENSEMBLE_SOURCES
            },
            "calibration_mae_30d_f": {
                src: (error_summary.get(src) or {}).get("mae_f")
                for src in _ENSEMBLE_SOURCES
            },
            "calibration_samples_30d": {
                src: (error_summary.get(src) or {}).get("samples", 0)
                for src in _ENSEMBLE_SOURCES
            },
            "best_recent_source": _best_recent_source(error_summary),
            "worst_recent_source": _worst_recent_source(error_summary),
            "calibration_window": error_summary.get("window"),
            "climatology_status": "not_ingested",
            "microclimate_status": "not_ingested",
        },
        market_pricing={
            "bucket_table": bucket_rows,
            "market_consensus_bucket": _top_bucket(bucket_rows, "market_prob"),
            "model_consensus_bucket": _top_bucket(bucket_rows, "calibrated_prob"),
            "market_runner_up": _runner_up_bucket(bucket_rows, "market_prob"),
            "model_runner_up": _runner_up_bucket(bucket_rows, "calibrated_prob"),
            "overpriced_buckets": _sort_edge_buckets(bucket_rows, overpriced=True),
            "underpriced_buckets": _sort_edge_buckets(bucket_rows, overpriced=False),
            "market_history_status": "current bucket snapshot history only",
            "consensus_spread_pts": _consensus_spread_points(bucket_rows),
        },
        smart_money=smart_money,
        diagnostics={
            "cloud_trend": cloud_trend,
            "wind_shift_deg_3h": wind_shift_deg,
            "pressure_tendency_inhg_3h": pressure_tendency,
            "resolution_mismatch_f": resolution_mismatch,
            "peak_already_passed": adaptive_inputs.get("peak_already_passed"),
            "kalman_trend_per_hr": adaptive_inputs.get("kalman_trend_per_hr"),
            "regression_r2": adaptive_inputs.get("regression_r2"),
            "remaining_rise_f": _round_float(model_inputs.get("remaining_rise")),
            "projected_high_f": _round_float(model_inputs.get("projected_high")),
            "peak_time_local": adaptive_inputs.get("composite_peak_timing"),
            "peak_timing_source": adaptive_inputs.get("peak_timing_source"),
            "metar_weight_pct": round(float(model_inputs.get("w_metar", 0.0)) * 100),
            "forecast_quality": event.forecast_quality,
            "utc_offset": CITY_REGISTRY_BY_SLUG.get(city.city_slug, {}).get("utc_offset"),
        },
        final_selection=selection,
    )


async def _build_smart_money_context(
    *,
    city_slug: str,
    date_et: str,
    bucket_rows: list[dict[str, Any]],
    selected_bucket_idx: int,
) -> dict[str, Any]:
    """Build compact wallet-positioning context for the LLM.

    The wallet tracker is read-only analytics. This helper deliberately
    summarizes already-stored public-wallet rows instead of asking the LLM to
    infer wallet quality from raw trade lists.
    """
    wallet_buckets = [
        {
            "bucket_idx": row.get("bucket_idx"),
            "label": row.get("label"),
            "model_prob": row.get("calibrated_prob"),
            "market_prob": row.get("market_prob"),
        }
        for row in bucket_rows
    ]
    try:
        from backend.market_context.wallet_tracker import get_weather_smart_money_payload

        payload = await get_weather_smart_money_payload(
            city_slug,
            date_et,
            buckets=wallet_buckets,
            limit=Config.WALLET_TRACKER_DISPLAY_LIMIT,
        )
    except Exception as exc:
        log.warning("market_context: smart-money payload failed for %s %s: %s", city_slug, date_et, exc)
        return {
            "enabled": Config.WALLET_TRACKER_ENABLED,
            "status": "error",
            "reason": "wallet_tracker_load_error",
            "message": "Wallet tracker context unavailable during Market Context build.",
            "recommendation": {
                "bucket_idx": None,
                "bucket_label": None,
                "stance": "NO_SMART_MONEY_SIGNAL",
                "rationale": "Wallet tracker load failed; do not infer wallet flow.",
            },
            "data_quality_notes": [str(exc)],
        }

    return _summarize_smart_money_payload(
        payload,
        selected_bucket_idx=selected_bucket_idx,
    )


def _summarize_smart_money_payload(
    payload: dict[str, Any],
    *,
    selected_bucket_idx: int,
) -> dict[str, Any]:
    current_rows = list(payload.get("current_market") or payload.get("rows") or [])
    global_by_wallet = _leader_by_wallet(payload.get("global_leaders") or [])
    city_by_wallet = _leader_by_wallet(payload.get("city_leaders") or [])
    long_rows = [row for row in current_rows if _wallet_row_is_long(row)]

    elite_rows: list[dict[str, Any]] = []
    elite_tags_by_wallet: dict[str, list[str]] = {}
    for row in long_rows:
        address = str(row.get("wallet_address") or "").lower()
        tags = _elite_wallet_tags(row, global_by_wallet.get(address), city_by_wallet.get(address))
        if not tags:
            continue
        elite_tags_by_wallet[address] = tags
        elite_rows.append(
            _compact_wallet_row(
                row,
                global_skill=global_by_wallet.get(address),
                city_skill=city_by_wallet.get(address),
                tags=tags,
            )
        )
    elite_rows.sort(
        key=lambda row: (
            "credible_perfect_city_record" in row.get("tags", []),
            "credible_perfect_weather_record" in row.get("tags", []),
            row.get("alpha_score") or 0.0,
            abs(float(row.get("net_notional_usd") or 0.0)),
        ),
        reverse=True,
    )

    rows_by_bucket: dict[int, list[dict[str, Any]]] = {}
    for row in long_rows:
        idx = row.get("bucket_idx")
        if idx is None:
            continue
        try:
            rows_by_bucket.setdefault(int(idx), []).append(row)
        except (TypeError, ValueError):
            continue

    clusters: list[dict[str, Any]] = []
    for bucket in payload.get("bucket_consensus") or []:
        idx = bucket.get("bucket_idx")
        if idx is None:
            continue
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            continue
        bucket_rows = rows_by_bucket.get(idx_int, [])
        wallet_tags_for_bucket = [
            (
                str(row.get("wallet_address") or "").lower(),
                elite_tags_by_wallet.get(str(row.get("wallet_address") or "").lower(), []),
            )
            for row in bucket_rows
        ]
        sample_wallets = [
            _compact_wallet_row(
                row,
                global_skill=global_by_wallet.get(str(row.get("wallet_address") or "").lower()),
                city_skill=city_by_wallet.get(str(row.get("wallet_address") or "").lower()),
                tags=elite_tags_by_wallet.get(str(row.get("wallet_address") or "").lower(), []),
            )
            for row in sorted(
                bucket_rows,
                key=lambda r: (r.get("alpha_score") or 0.0, abs(float(r.get("net_notional_usd") or 0.0))),
                reverse=True,
            )[:5]
        ]
        clusters.append(
            {
                "bucket_idx": idx_int,
                "bucket_label": bucket.get("bucket_label"),
                "wallets_long": bucket.get("wallets_long", 0) or 0,
                "ranked_wallets_long": bucket.get("ranked_wallets_long", 0) or 0,
                "elite_wallets_long": sum(1 for _, tags in wallet_tags_for_bucket if tags),
                "perfect_wallets_long": sum(
                    1 for _, tags in wallet_tags_for_bucket
                    if any(tag.startswith("perfect_") for tag in tags)
                ),
                "credible_perfect_wallets_long": sum(
                    1 for _, tags in wallet_tags_for_bucket
                    if any(tag.startswith("credible_perfect_") for tag in tags)
                ),
                "extremely_profitable_wallets_long": sum(
                    1 for _, tags in wallet_tags_for_bucket
                    if "extremely_profitable_weather_record" in tags
                ),
                "net_notional_usd": bucket.get("net_notional_usd"),
                "net_position_qty": bucket.get("net_position_qty"),
                "weighted_flow": bucket.get("weighted_flow"),
                "avg_entry_price": bucket.get("avg_entry_price"),
                "sample_wallets": sample_wallets,
            }
        )

    strongest = _strongest_wallet_cluster(clusters)
    selected_cluster = next(
        (cluster for cluster in clusters if cluster.get("bucket_idx") == selected_bucket_idx),
        None,
    )
    recommendation = _smart_money_recommendation(
        payload=payload,
        strongest=strongest,
        selected_bucket_idx=selected_bucket_idx,
    )

    data_quality_notes: list[str] = []
    reason = payload.get("reason")
    if reason:
        data_quality_notes.append(str(reason))
    coverage = payload.get("coverage") or {}
    if (coverage.get("current_market_wallets") or 0) == 0:
        data_quality_notes.append("No current wallet exposure rows for this city/date.")
    tiny_perfect = [
        row for row in elite_rows
        if any(tag.startswith("perfect_") for tag in row.get("tags", []))
        and not any(tag.startswith("credible_perfect_") for tag in row.get("tags", []))
    ]
    if tiny_perfect:
        data_quality_notes.append(
            "Some perfect records have tiny samples; cite resolved_markets and Wilson rate before trusting them."
        )

    return {
        "enabled": payload.get("enabled"),
        "status": payload.get("status"),
        "reason": payload.get("reason"),
        "message": payload.get("message"),
        "current_source": payload.get("current_source"),
        "coverage": {
            "current_market_wallets": coverage.get("current_market_wallets", 0),
            "wallet_trade_rows": coverage.get("wallet_trade_rows", 0),
            "wallets_scanned": coverage.get("wallets_scanned", 0),
            "condition_ids_scanned": coverage.get("condition_ids_scanned", 0),
            "exposure_rows": coverage.get("exposure_rows", 0),
            "global_skill_rows": coverage.get("global_skill_rows", 0),
            "city_skill_rows": coverage.get("city_skill_rows", 0),
            "window_days": coverage.get("window_days"),
            "last_refresh": coverage.get("last_refresh"),
        },
        "confluence": payload.get("confluence") or {},
        "bucket_clusters": clusters,
        "strongest_wallet_bucket": strongest,
        "selected_bucket_wallet_support": selected_cluster,
        "elite_wallets_current": elite_rows[:10],
        "elite_wallet_count_current": len(elite_rows),
        "recommendation": recommendation,
        "data_quality_notes": data_quality_notes,
        "disclaimer": payload.get("disclaimer"),
    }


def _leader_by_wallet(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("wallet_address") or "").lower(): row
        for row in rows
        if row.get("wallet_address")
    }


def _wallet_row_is_long(row: dict[str, Any]) -> bool:
    direction = str(row.get("direction") or "").upper()
    net_qty = _float_or_none(row.get("net_position_qty")) or 0.0
    net_notional = _float_or_none(row.get("net_notional_usd") or row.get("net_flow_usd")) or 0.0
    return direction == "LONG" or net_qty > 0 or net_notional > 0


def _elite_wallet_tags(
    current_row: dict[str, Any],
    global_skill: dict[str, Any] | None,
    city_skill: dict[str, Any] | None,
) -> list[str]:
    tags: list[str] = []
    min_resolved = max(1, Config.WALLET_TRACKER_MIN_RESOLVED_MARKETS)

    global_win = _float_or_none((global_skill or current_row).get("win_rate"))
    global_resolved = _int_or_zero((global_skill or current_row).get("resolved_markets"))
    if global_win is not None and global_win >= 0.999 and global_resolved > 0:
        tags.append("perfect_weather_record")
        if global_resolved >= min_resolved:
            tags.append("credible_perfect_weather_record")

    city_win = _float_or_none((city_skill or {}).get("win_rate"))
    city_resolved = _int_or_zero((city_skill or {}).get("resolved_markets"))
    if city_win is not None and city_win >= 0.999 and city_resolved > 0:
        tags.append("perfect_city_record")
        if city_resolved >= min_resolved:
            tags.append("credible_perfect_city_record")

    realized = _float_or_none((global_skill or {}).get("realized_pnl"))
    roi = _float_or_none((global_skill or current_row).get("roi"))
    profit_factor = _float_or_none((global_skill or current_row).get("profit_factor"))
    adjusted_score = _float_or_none(current_row.get("global_score") or current_row.get("city_score"))
    if (
        (realized is not None and realized >= 100.0)
        or (roi is not None and roi >= 0.50)
        or (profit_factor is not None and profit_factor >= 3.0)
        or (adjusted_score is not None and adjusted_score >= 0.75)
    ):
        tags.append("extremely_profitable_weather_record")

    if current_row.get("is_ranked"):
        tags.append("ranked_current_position")

    return list(dict.fromkeys(tags))


def _compact_wallet_row(
    row: dict[str, Any],
    *,
    global_skill: dict[str, Any] | None,
    city_skill: dict[str, Any] | None,
    tags: list[str],
) -> dict[str, Any]:
    global_source = global_skill or row
    return {
        "wallet_address": row.get("wallet_address"),
        "display_address": row.get("display_address"),
        "profile_url": row.get("profile_url"),
        "bucket_idx": row.get("bucket_idx"),
        "bucket_label": row.get("bucket_label"),
        "net_notional_usd": row.get("net_notional_usd") or row.get("net_flow_usd"),
        "net_position_qty": row.get("net_position_qty"),
        "avg_entry_price": row.get("avg_entry_price"),
        "last_trade_age_min": row.get("last_trade_age_min"),
        "alpha_score": row.get("alpha_score"),
        "skill_source": row.get("skill_source") or row.get("source"),
        "global_rank": row.get("global_rank") or (global_skill or {}).get("rank"),
        "city_rank": row.get("city_rank") or (city_skill or {}).get("rank"),
        "global_win_rate": _round_float(global_source.get("win_rate"), 4),
        "global_wilson_win_rate": _round_float(global_source.get("wilson_win_rate"), 4),
        "global_resolved_markets": _int_or_zero(global_source.get("resolved_markets")),
        "global_roi": _round_float(global_source.get("roi"), 4),
        "global_profit_factor": _round_float(global_source.get("profit_factor"), 4),
        "city_win_rate": _round_float((city_skill or {}).get("win_rate"), 4),
        "city_wilson_win_rate": _round_float((city_skill or {}).get("wilson_win_rate"), 4),
        "city_resolved_markets": _int_or_zero((city_skill or {}).get("resolved_markets")),
        "city_roi": _round_float((city_skill or {}).get("roi"), 4),
        "tags": tags,
    }


def _strongest_wallet_cluster(clusters: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        cluster for cluster in clusters
        if (cluster.get("ranked_wallets_long") or 0) > 0
        and (cluster.get("weighted_flow") or 0.0) > 0
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda cluster: (
            cluster.get("credible_perfect_wallets_long") or 0,
            cluster.get("elite_wallets_long") or 0,
            cluster.get("ranked_wallets_long") or 0,
            cluster.get("weighted_flow") or 0.0,
            cluster.get("net_notional_usd") or 0.0,
        ),
    )


def _smart_money_recommendation(
    *,
    payload: dict[str, Any],
    strongest: dict[str, Any] | None,
    selected_bucket_idx: int,
) -> dict[str, Any]:
    if not payload.get("enabled"):
        return {
            "bucket_idx": None,
            "bucket_label": None,
            "stance": "NO_SMART_MONEY_SIGNAL",
            "strength": "none",
            "rationale": "Wallet tracker is disabled.",
        }
    if strongest is None:
        return {
            "bucket_idx": None,
            "bucket_label": None,
            "stance": "NO_SMART_MONEY_SIGNAL",
            "strength": "none",
            "rationale": "No positive ranked wallet flow is available for this market.",
        }

    smart_idx = strongest.get("bucket_idx")
    distance = abs(int(smart_idx) - int(selected_bucket_idx)) if smart_idx is not None else None
    if distance == 0:
        stance = "CONFIRMS_SELECTED_BUCKET"
    elif distance == 1:
        stance = "ADJACENT_SMART_MONEY_CAUTION"
    else:
        stance = "DIVERGES_FROM_SELECTED_BUCKET"

    credible_perfect = strongest.get("credible_perfect_wallets_long") or 0
    elite = strongest.get("elite_wallets_long") or 0
    ranked = strongest.get("ranked_wallets_long") or 0
    net_notional = abs(float(strongest.get("net_notional_usd") or 0.0))
    if credible_perfect >= 1 or elite >= 3 or (ranked >= 3 and net_notional >= 250.0):
        strength = "high"
    elif elite >= 1 or ranked >= 2 or net_notional >= 100.0:
        strength = "medium"
    else:
        strength = "low"

    return {
        "bucket_idx": smart_idx,
        "bucket_label": strongest.get("bucket_label"),
        "stance": stance,
        "strength": strength,
        "ranked_wallets_long": ranked,
        "elite_wallets_long": elite,
        "credible_perfect_wallets_long": credible_perfect,
        "net_notional_usd": strongest.get("net_notional_usd"),
        "weighted_flow": strongest.get("weighted_flow"),
        "rationale": (
            "Use this as corroborating/tie-breaker evidence; do not override the weather model "
            "unless the wallet cluster is high strength and the meteorological evidence is ambiguous."
        ),
    }


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


async def _generate_market_context_output(
    context: MarketContextInput,
) -> MarketContextOutput:
    from backend.market_context.tools import TOOL_DEFINITIONS
    adapter = MarketContextLLMAdapter()
    system_prompt, user_prompt = _build_prompts(context)
    last_error: Exception | None = None

    for attempt in range(2):
        try:
            payload = await adapter.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt if attempt == 0 else (
                    user_prompt
                    + "\n\nThe previous response failed validation. Return valid JSON only. "
                    + f"Keep bucket_idx={context.final_selection.bucket_idx}, label={json.dumps(context.final_selection.label)}, "
                    + f"confidence_pct={context.final_selection.confidence_pct} exactly."
                ),
                tools=TOOL_DEFINITIONS,
            )
            llm_selection = MarketContextSelection(**payload.get("final_selection", {}))
            _validate_authoritative_selection(llm_selection, context.final_selection)
            parsed = MarketContextOutput(
                sections=payload.get("sections", {}),
                final_selection=llm_selection,
            )
            return parsed
        except Exception as exc:
            last_error = exc

    raise MarketContextLLMError(f"Market Context generation failed validation: {last_error}")


def _build_prompts(context: MarketContextInput) -> tuple[str, str]:
    """Build system + user prompts for the Market Context LLM.

    Major changes from earlier prompt versions:
      - System prompt now includes the MODEL_ENCYCLOPEDIA so the LLM can
        reason about each source's formation method, cadence, strengths,
        and weaknesses from first principles instead of generic phrases.
      - Source list expanded from {NWS, HRRR, NBM, WU} to the full
        10-source ensemble (NWS, HRRR, NBM, IFS, AIFS, GraphCast, Pangu,
        FCN-v2, Aurora, WU). Each source surfaced with model_run_at,
        age, lead, MAE/bias, and effective weight.
      - New required section: ADVERSARIAL REASONING — "why might this
        trade lose? what would invalidate the call?". Forces the LLM
        to commit to falsifiable predictions instead of decorating
        the precomputed call.
      - New required section: TRIGGER CONDITIONS — specific observable
        events that flip the analysis (e.g. "if 14:00 METAR < 73°F,
        the call is wrong").
      - Calibration MAE block surfaced explicitly with per-source
        rolling 30d MAE/bias.
      - "Disagreement diagnosis" requires NAMING the specific outlier
        sources from panel_disagreement.outliers_high/low.
    """
    system_prompt = (
        "You are an autonomous quantitative weather-derivatives analyst producing a concise, "
        "evidence-dense Market Context report for a Polymarket temperature-bucket prediction. "
        "Your output drives real capital allocation. Capital lost to hand-wavy reasoning is "
        "permanent. Be specific or be silent.\n\n"
        "OPERATING PRINCIPLES:\n"
        "1. ACTIVELY USE YOUR TOOLS. Before writing, call fetch_nws_discussion for synoptic context, "
        "search_academic_climatology for peer-reviewed heuristics, and fetch_nbm_forecast / "
        "fetch_hrrr_forecast for hourly curves. Tool results are CRITICAL.\n"
        "2. NAME EVERY SOURCE you reference. Don't say 'the models' — say 'IFS, AIFS, and HRRR'. "
        "Don't say 'an outlier' — say 'NBM is +2.1°F above panel mean'.\n"
        "3. CITE QUANTITATIVE FACTS only. If you reference an academic paper, name (year, journal) "
        "and quote a specific quantitative finding. Vague references waste tokens.\n"
        "4. APPLY BIAS CORRECTIONS EXPLICITLY. State the raw forecast AND the bias-adjusted value, "
        "where bias comes from `historical_context.calibration_biases_f`. A source running "
        "+1.0°F warm in the last 30d should have its forecast adjusted down before fusion.\n"
        "5. WEIGHT BY FRESHNESS. Older runs lose information. A 24h-stale IFS forecast carries "
        "less authority than a 2h-fresh HRRR run for tomorrow's high. Surface this explicitly.\n"
        "6. THE PRECOMPUTED BUCKET is a baseline, not a verdict. In diagnostic_reasoning, "
        "adversarial_reasoning, and independent_assessment, state whether you AGREE or DISAGREE "
        "and why. If your tool-informed analysis points to a different bucket, explicitly flag "
        "the discrepancy with reasoning.\n"
        "7. Be terse. Each section: 2-4 evidence-dense sentences. Total output must fit 2800 tokens.\n"
        f"8. Return ONLY valid JSON with keys `sections` ({len(SECTION_KEYS)} string values) and `final_selection` "
        "(echo pre-computed values exactly).\n\n"
        "════════════════════════════════════════════════════════════════════════════\n"
        "ENSEMBLE SOURCE ENCYCLOPEDIA — reason about these models from first principles\n"
        "════════════════════════════════════════════════════════════════════════════\n"
    )
    # Embed the encyclopedia compactly. Each source = one paragraph.
    for src, meta in MODEL_ENCYCLOPEDIA.items():
        system_prompt += (
            f"\n{src.upper()} ({meta['full_name']}): {meta['type']}. "
            f"Calculation: {meta['calculation']} "
            f"Run cadence: {meta['cadence']}. Typical age at trade: {meta['typical_age_at_trade_h']}h. "
            f"PROS: {meta['pros']} CONS: {meta['cons']}\n"
        )
    system_prompt += (
        "\n════════════════════════════════════════════════════════════════════════════\n\n"
        "Use the encyclopedia to reason ABOUT each source — not just to repeat its forecast value. "
        "A 6h-timestep AI model (Pangu, FCN, Aurora) cannot resolve true peak temperature between "
        "steps. A frontal passage day favors physics models (HRRR, IFS) over AI models trained on "
        "ERA5 climatology. WU has historical 1-3°F summer afternoon bias. Apply these tradeoffs.\n"
    )

    avail = context.availability
    unavailable = [
        k.replace("_available", "").upper()
        for k, v in avail.items()
        if k.endswith("_available") and not v
    ]
    availability_note = (
        f"Unavailable sources: {', '.join(unavailable)}. Do not reference these as if present."
        if unavailable else "All configured data sources are available."
    )

    section_keys_extended = list(SECTION_KEYS)
    for k in ("independent_assessment", "adversarial_reasoning", "trigger_conditions"):
        if k not in section_keys_extended:
            section_keys_extended.append(k)

    user_prompt = (
        f"Market Context report: {context.city_display} on {context.date_et}. Return JSON only.\n\n"
        "BEFORE WRITING — USE YOUR TOOLS:\n"
        f"  1. fetch_nws_discussion(city_slug='{context.city_slug}') for synoptic analysis\n"
        f"  2. search_academic_climatology with a query relevant to today's regime\n"
        f"  3. fetch_nbm_forecast(city_slug='{context.city_slug}') for the NBM hourly curve\n"
        f"  4. fetch_hrrr_forecast(city_slug='{context.city_slug}') for the HRRR hourly curve\n"
        "Tool results should inform every section.\n\n"
        "STRUCTURE: `sections` must contain exactly these keys: "
        + ", ".join(section_keys_extended) + "\n"
        "Each value: a single markdown prose string (3-5 dense sentences). No nested JSON.\n\n"
        "SECTION REQUIREMENTS:\n\n"
        "1. current_observations -- current_temp_f, observed_high_f, resolution_high_f. Report "
        "dewpoint_spread_f and its implication for remaining heating potential. Cite "
        "temp_change_1h_f, warming_acceleration_f_per_hr_delta, cloud_trend. Characterize the "
        "temperature trajectory: accelerating / decelerating / stalling. Flag metar_age_s > 1200.\n\n"
        "2. short_range_model_landscape -- Walk through `short_range_models.ensemble_sources` by "
        "name. For each available source, cite the high_f, age_hours, lead_hours, and bias-adjusted "
        "value (= high_f − bias_30d_f). Identify panel_disagreement.outliers_high and "
        "outliers_low BY NAME. Diagnose >2°F disagreements using the encyclopedia + the NWS "
        "discussion (e.g. 'HRRR is 2.5°F above the panel; HRRR has a known summer-afternoon warm "
        "bias and the AFD mentions weakening synoptic forcing — discount HRRR'). Note "
        "panel_disagreement.missing_sources as info gaps.\n\n"
        "3. calibration_landscape -- This is a NEW required section. Report "
        "historical_context.calibration_mae_30d_f for each source. State best_recent_source and "
        "worst_recent_source. Compare each forecast's bias_30d_f to its own MAE — sources with "
        "|bias|/MAE > 0.5 are systematically biased and need correction, not just averaging. "
        "Identify which source's forecast you trust MOST today given freshness × MAE × regime. "
        "Reference the bma_shadow.between_share if available — high values (>0.5) mean the panel "
        "is multimodal and the legacy single-Gaussian summary may understate tail risk.\n\n"
        "4. historical_climatology_perspective -- Report avg_high_7d_f, avg_high_prev_7d_f, "
        "trend_delta_f. Compare projected_high_f to the 7-day realized average. If "
        "last_settled_errors_f exist, state which source nailed yesterday and which missed. "
        "Reference academic papers from your search_academic_climatology call by (author, year, "
        "journal) with a SPECIFIC quantitative finding relevant to today's regime — e.g. cold air "
        "damming, urban heat island, lake breeze, transition-season fat tails.\n\n"
        "5. market_pricing_analysis -- model_consensus_bucket vs market_consensus_bucket. If they "
        "diverge, explain true_edge on the selected bucket. Report consensus_spread_pts. Top "
        "overpriced_buckets and underpriced_buckets by true_edge magnitude. State selected bucket "
        "calibrated_prob vs market_prob.\n\n"
        "6. smart_money_analysis -- NEW required section. Analyze `smart_money` as read-only "
        "public-wallet context, not as an automated copy-trading signal. State the wallet-only "
        "recommended bucket from smart_money.recommendation, its strength, whether it confirms / "
        "is adjacent to / diverges from the precomputed bucket, and the ranked_wallets_long + "
        "net_notional_usd behind it. If elite_wallets_current includes wallets tagged "
        "credible_perfect_weather_record, credible_perfect_city_record, perfect_weather_record, "
        "perfect_city_record, or extremely_profitable_weather_record, name the display addresses, "
        "bucket labels, and counts by bucket. If a wallet is perfect but has a tiny sample, say "
        "the sample count and Wilson rate instead of treating it as certain. End with one "
        "succinct recommendation: overweight / neutral / underweight the precomputed bucket "
        "based on wallet evidence alone.\n\n"
        "7. diagnostic_reasoning -- Build a causal chain using BOTH DB context AND tool results: "
        "(a) peak_already_passed? If yes, anchor on remaining_rise_f and projected_high_f. "
        "(b) kalman_trend_per_hr + regression_r2: is the intraday trend reliable? "
        "(c) pressure_tendency_inhg_3h + wind_shift_deg_3h: synoptic shift? Reference NWS AFD. "
        "(d) cloud_trend + resolution_mismatch_f: observational red flags? "
        "(e) What does the NWS forecaster say about confidence and pitfalls? "
        "Synthesize a single verdict: projected high robust or vulnerable, and why.\n\n"
        "8. final_high_stakes_selection -- Restate the pre-computed bucket label + confidence_pct. "
        "Decompose confidence via confidence_components (top 2 boosters, top 2 penalties by "
        "magnitude). State flip_signals verbatim. Declare peak_time and life_or_death_call. Fold "
        "in the smart_money.recommendation as one named corroborating or cautionary input, but do "
        "not change the echoed final_selection fields.\n\n"
        "9. adversarial_reasoning -- This is a NEW required section. WHY MIGHT THIS TRADE LOSE? "
        "Generate the strongest counter-case: which 2-3 things, if they happen, would make the "
        "precomputed bucket wrong? Be specific (e.g. 'cumulus build by 14:00 caps heating 2°F "
        "below NBM peak'). What's the historical analog where a setup like today went the OTHER "
        "way? Don't be polite — your job here is to find the case against the trade.\n\n"
        "10. trigger_conditions -- This is a NEW required section. List 2-4 SPECIFIC OBSERVABLE "
        "events between now and resolution that would invalidate or confirm the call. Format: "
        "'IF <observation> BY <time>, THEN <action>'. Examples: 'IF 14:00 EDT METAR < 73°F, "
        "the warm-bucket call is wrong — exit'. 'IF wind shifts north > 15° before peak, "
        "discount HRRR'. Make the LLM commit to falsifiable predictions.\n\n"
        "11. independent_assessment -- YOUR OWN expert opinion based on tool results + the "
        "encyclopedia + calibration MAEs. State your independent projected high and which bucket "
        "you believe is most likely. If this AGREES with the precomputed selection, say so and "
        "explain converging evidence. If it DISAGREES, explicitly flag: your preferred bucket, "
        "your projected high, and the key evidence (NWS AFD phrase, specific source values, "
        "calibration drift, smart-money cluster, academic heuristic) that justifies divergence. THIS SECTION DRIVES "
        "TRADING DECISIONS — its value comes from disagreement when warranted, not from "
        "rubber-stamping the precomputed call.\n\n"
        "CONSISTENCY RULES:\n"
        "- Probability figures must match context to 1 decimal place. Do NOT alter "
        "calibrated_prob, market_prob, or true_edge.\n"
        "- final_selection fields bucket_id, bucket_idx, label, confidence_pct: echo EXACTLY.\n"
        "- In independent_assessment + adversarial_reasoning, clearly separate your opinion from "
        "the algorithmic baseline.\n"
        "- Reference each forecast source by NAME at least once across the report.\n\n"
        f"AVAILABILITY: {availability_note}\n\n"
        "Pre-computed baseline selection (echo exactly in final_selection):\n"
        f"{json.dumps(context.final_selection.model_dump(), indent=2)}\n\n"
        "Structured backend context (includes ensemble_sources with per-model run times, MAE, "
        "bias, weights; calibration_mae_30d_f; bma_shadow; panel_disagreement; smart_money "
        "wallet clusters, elite wallet tags, and wallet-only recommendation):\n"
        f"{json.dumps(context.model_dump(), indent=2)}"
    )
    return system_prompt, user_prompt


def _validate_authoritative_selection(
    llm_selection: MarketContextSelection,
    authoritative: MarketContextSelection,
) -> None:
    if llm_selection.bucket_id != authoritative.bucket_id:
        raise MarketContextLLMError("LLM changed authoritative bucket_id")
    if llm_selection.bucket_idx != authoritative.bucket_idx:
        raise MarketContextLLMError("LLM changed authoritative bucket_idx")
    if llm_selection.label != authoritative.label:
        raise MarketContextLLMError("LLM changed authoritative bucket label")
    if llm_selection.confidence_pct != authoritative.confidence_pct:
        raise MarketContextLLMError("LLM changed authoritative confidence_pct")


def _settlement_at_utc_for(date_et: str, city_tz: str) -> Optional[datetime]:
    """Compute the local-day end-of-settlement timestamp in UTC.

    Polymarket temperature events resolve at 23:59:59 local time on the
    market's date. We use this for the per-source `lead_hours` calc
    (time from model_run_at to event end).
    """
    try:
        local_eod = datetime.strptime(date_et, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=ZoneInfo(city_tz),
        )
        return local_eod.astimezone(timezone.utc)
    except Exception:
        return None


def _worst_recent_source(error_summary: dict) -> Optional[str]:
    """Highest-MAE source in the last 30 days. Counterpart to
    `_best_recent_source` so the LLM can frame trustworthiness explicitly:
    'HRRR has been the most accurate this month; WU has been the worst.'
    """
    candidates: list[tuple[str, float]] = []
    for src in _ENSEMBLE_SOURCES:
        item = error_summary.get(src) or {}
        mae = item.get("mae_f")
        if mae is not None and (item.get("samples") or 0) >= 5:
            candidates.append((src, float(mae)))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])[0]


def _build_short_range_models(
    *,
    model,
    model_inputs: dict,
    error_summary: dict,
    sources_legacy: dict,
    source_forecasts: dict,
    city_unit: str,
    settlement_at_utc: Optional[datetime],
    now_utc: datetime,
) -> dict:
    """Build the rich short_range_models payload that the LLM reasons over.

    Centralizes:
      - The legacy `sources` summary (kept for back-compat with
        deterministic _select_bucket).
      - Per-source dict (high, run time, age, lead, MAE, bias, weight,
        encyclopedia metadata) for the 10-source ensemble.
      - Mixture σ/μ from the legacy single-Gaussian path.
      - BMA shadow predictive output if compute_model surfaced it.
      - Forecast panel disagreement summary so the LLM can call out
        specific outliers.
    """
    # Live per-source weights from compute_model's ensemble_breakdown.
    ensemble_breakdown = model_inputs.get("ensemble_breakdown") or {}
    weight_factors = ensemble_breakdown.get("weight_factors") or {}

    per_source = _build_per_source_summary(
        city_unit=city_unit,
        source_forecasts=source_forecasts,
        settlement_at_utc=settlement_at_utc,
        error_summary_30d=error_summary,
        weight_factors=weight_factors,
        now_utc=now_utc,
    )

    # Panel disagreement: max-min spread + outlier identification. Lets
    # the LLM call out specific sources by name without re-deriving.
    highs = [
        (src, entry["high_f"])
        for src, entry in per_source.items()
        if entry.get("high_f") is not None
    ]
    spread_f: Optional[float] = None
    outliers_high: list[str] = []
    outliers_low: list[str] = []
    if len(highs) >= 2:
        vals = [v for _, v in highs]
        mn, mx = min(vals), max(vals)
        spread_f = round(mx - mn, 2)
        # Calls out sources that are >1°F from the panel mean as
        # outliers — these are what the LLM should diagnose first.
        mean_high = sum(vals) / len(vals)
        for src, v in highs:
            if v - mean_high > 1.0:
                outliers_high.append(src)
            elif mean_high - v > 1.0:
                outliers_low.append(src)

    legacy_missing_external_models = [
        label
        for src, label in (
            ("hrrr", "HRRR"),
            ("nbm", "NBM"),
            ("nam", "NAM"),
            ("rap", "RAP"),
            ("ecmwf_ifs", "ECMWF"),
        )
        if source_forecasts.get(src) is None
    ]

    return {
        # Legacy fields kept for back-compat with deterministic engine
        # and any older prompt path.
        "stored_sources": sources_legacy,
        "forecast_spread_f": _round_float(model_inputs.get("spread")),
        "mu_f": _round_float(model.mu),
        "sigma_f": _round_float(model.sigma),
        "projected_high_f": _round_float(model_inputs.get("projected_high")),
        "prob_new_high": _round_float(model_inputs.get("prob_new_high"), 4),
        "recent_error_summary": error_summary,
        "best_recent_source": _best_recent_source(error_summary),
        "worst_recent_source": _worst_recent_source(error_summary),
        "missing_external_models": legacy_missing_external_models,
        # NEW: full per-source ensemble view with metadata.
        "ensemble_sources": per_source,
        "panel_disagreement": {
            "spread_f": spread_f,
            "n_active_sources": len(highs),
            "outliers_high": outliers_high,
            "outliers_low": outliers_low,
            # Sources currently MISSING from the panel today (no forecast
            # row) — operator + LLM should call these out as info gaps.
            "missing_sources": [
                src for src, entry in per_source.items()
                if entry.get("high_f") is None
            ],
        },
        # BMA shadow predictive (M1 Phase 1) — gives LLM the mixture
        # variance share so it can reason about whether the panel is
        # multimodal (between-source variance dominates).
        "bma_shadow": model_inputs.get("bma_shadow"),
    }


def _build_per_source_summary(
    *,
    city_unit: str,
    source_forecasts: dict,
    settlement_at_utc: Optional[datetime],
    error_summary_30d: dict,
    weight_factors: dict,
    now_utc: datetime,
) -> dict[str, dict]:
    """Build the rich per-source dict the LLM reasons over.

    Each source entry includes:
      - high_f: forecast value (None when source missing today)
      - model_run_at: ISO timestamp of init time
      - age_hours: time since init
      - lead_hours: time from init to event settlement
      - mae_30d_f: rolling 30-day MAE from settled events
      - bias_30d_f: rolling 30-day bias (forecast minus realized)
      - effective_weight: live ensemble weight from compute_model
      - meta: encyclopedia entry (full_name, type, pros, cons, etc.)

    Sources missing forecast data still appear with `high_f=None` so the
    LLM can comment on absences (e.g. HRRR didn't run today → discount
    that signal).
    """
    out: dict[str, dict] = {}
    for src in _ENSEMBLE_SOURCES:
        fc = source_forecasts.get(src)
        meta = MODEL_ENCYCLOPEDIA.get(src, {})
        entry: dict = {
            "high_f": _round_float(fc.high_f) if fc and fc.high_f is not None else None,
            "model_run_at": fc.model_run_at.isoformat() if fc and getattr(fc, "model_run_at", None) else None,
            "fetched_at": fc.fetched_at.isoformat() if fc and getattr(fc, "fetched_at", None) else None,
            "age_hours": None,
            "lead_hours": None,
            "mae_30d_f": None,
            "bias_30d_f": None,
            "effective_weight": None,
            "meta": meta,
        }
        if fc and getattr(fc, "model_run_at", None):
            mr = fc.model_run_at
            if mr.tzinfo is None:
                mr = mr.replace(tzinfo=timezone.utc)
            entry["age_hours"] = round((now_utc - mr).total_seconds() / 3600.0, 2)
            if settlement_at_utc:
                entry["lead_hours"] = round(
                    (settlement_at_utc - mr).total_seconds() / 3600.0, 2,
                )
        # Calibration stats from the rolling settled-event window.
        err = error_summary_30d.get(src) or {}
        entry["mae_30d_f"] = err.get("mae_f")
        entry["bias_30d_f"] = err.get("bias_f")
        entry["n_samples"] = err.get("samples")
        # Live ensemble weight from compute_model's weight_factors block
        # (signal_engine writes this into model.inputs.ensemble_breakdown).
        wf = (weight_factors or {}).get(src) or {}
        entry["effective_weight"] = wf.get("effective")
        entry["lead_skill_factor"] = wf.get("lead_factor")
        entry["freshness_factor"] = wf.get("freshness_factor")
        out[src] = entry
    return out


async def _compute_recent_error_summary(
    sess,
    *,
    city: City,
    recent_events: list,
    observation_minutes: list[int],
    reference_date_et: str,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    city_tz = getattr(city, "tz", "America/New_York")
    summary: dict[str, Any] = {}
    # Track signed errors (forecast − realized) per source so we can
    # report both MAE (mean of |err|) and bias (mean of err) in one pass.
    # Expanded from {nws, wu_hourly} to the full ensemble — the LLM
    # needs the same 30-day calibration view for IFS/AIFS/AIWP models
    # to make trustworthy weighting calls.
    errors_by_source: dict[str, list[float]] = {s: [] for s in _ENSEMBLE_SOURCES}
    last_settled: Optional[dict[str, Any]] = None

    for event in recent_events:
        if event.date_et >= reference_date_et:
            continue
        realized_high = await _resolve_realized_high(
            sess,
            city=city,
            date_et=event.date_et,
            observation_minutes=observation_minutes,
        )
        if realized_high is None:
            continue

        date_errors: dict[str, float] = {}
        for source in _ENSEMBLE_SOURCES:
            fc = await get_latest_successful_forecast(sess, city.id, source, event.date_et)
            if fc and fc.high_f is not None:
                err = round(fc.high_f - realized_high, 1)
                errors_by_source[source].append(err)
                date_errors[source] = err

        if last_settled is None:
            last_settled = {
                "date_et": event.date_et,
                "realized_high_f": realized_high,
                "errors": date_errors,
            }

        # Bound walk so we don't scan unbounded history. ~30 settled days
        # × 8-source coverage ≈ 240 points is plenty for 30d MAE.
        if sum(len(v) for v in errors_by_source.values()) >= 60:
            break

    for source, signed_errors in errors_by_source.items():
        if not signed_errors:
            summary[source] = {
                "mae_f": None, "bias_f": None, "samples": 0,
            }
            continue
        n = len(signed_errors)
        mae = sum(abs(e) for e in signed_errors) / n
        bias = sum(signed_errors) / n
        summary[source] = {
            "mae_f": round(mae, 2),
            "bias_f": round(bias, 2),
            "samples": n,
            # Backward-compat alias; old callers still read this key.
            "avg_abs_error_f": round(mae, 2),
        }
    summary["window"] = "rolling 30-day settled sample"
    return summary, last_settled


async def _compute_recent_high_summary(
    sess,
    *,
    city: City,
    recent_events: list,
    observation_minutes: list[int],
    reference_date_et: str,
) -> dict[str, Any]:
    realized_days: list[dict[str, Any]] = []
    for event in recent_events:
        if event.date_et >= reference_date_et:
            continue
        realized_high = await _resolve_realized_high(
            sess,
            city=city,
            date_et=event.date_et,
            observation_minutes=observation_minutes,
        )
        if realized_high is None:
            continue
        realized_days.append({"date_et": event.date_et, "high_f": realized_high})
        if len(realized_days) >= 14:
            break

    highs = [row["high_f"] for row in realized_days]
    first7 = highs[:7]
    second7 = highs[7:14]
    avg_7d = round(sum(first7) / len(first7), 1) if first7 else None
    avg_prev_7d = round(sum(second7) / len(second7), 1) if second7 else None
    trend_delta = round(avg_7d - avg_prev_7d, 1) if avg_7d is not None and avg_prev_7d is not None else None
    return {
        "recent_days": realized_days,
        "avg_7d": avg_7d,
        "avg_prev_7d": avg_prev_7d,
        "trend_delta": trend_delta,
    }


async def _resolve_realized_high_with_source(
    sess,
    *,
    city: City,
    date_et: str,
    observation_minutes: list[int],
) -> dict[str, Any]:
    """Resolve the realized daily high with source tracking.

    Returns dict with keys: high_f (Optional[float]), source_used (Optional[str]),
    obs_time (Optional[datetime]).

    Fallback chain is controlled by Config.SETTLEMENT_HIGH_PRIMARY:
      - "tgftp" (default): TGFTP daily high → wu_history → resolution_metar → raw_metar
      - "wu_history": wu_history → resolution_metar → raw_metar (legacy behavior)
    """
    primary = Config.SETTLEMENT_HIGH_PRIMARY
    city_tz = getattr(city, "tz", "America/New_York")
    tz = ZoneInfo(city_tz)

    def _local_date_of(dt: Optional[datetime]) -> Optional[str]:
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(tz).strftime("%Y-%m-%d")

    # ── TGFTP primary path ────────────────────────────────────────────────
    if primary == "tgftp":
        from backend.storage.repos import get_daily_high_metar_obs
        tgftp_row = await get_daily_high_metar_obs(
            sess, city.id, date_et, city_tz=city_tz, source="tgftp",
        )
        if (
            tgftp_row is not None
            and tgftp_row.temp_f is not None
            and _local_date_of(tgftp_row.observed_at) == date_et
        ):
            return {"high_f": float(tgftp_row.temp_f), "source_used": "tgftp", "obs_time": tgftp_row.observed_at}

    # ── WU history (always available as fallback or primary) ──────────────
    wu_history = await get_latest_successful_forecast(sess, city.id, "wu_history", date_et)
    if wu_history and wu_history.high_f is not None:
        wu_raw = json.loads(wu_history.raw_json) if wu_history.raw_json else {}
        obs_time_str = wu_raw.get("obs_time")
        obs_time = None
        if obs_time_str:
            try:
                obs_time = datetime.fromisoformat(str(obs_time_str).rstrip("Z")).replace(tzinfo=timezone.utc)
            except Exception:
                pass
        # Freshness gate: the wu_history row is keyed by date_et but its
        # reported high can carry over the prior local date's peak when WU
        # has not yet updated. Reject if obs_time resolves to a different
        # local date.
        if obs_time is None or _local_date_of(obs_time) == date_et:
            return {"high_f": float(wu_history.high_f), "source_used": "wu_history", "obs_time": obs_time}

    # ── Resolution METAR (valid observation minutes only) ─────────────────
    if observation_minutes:
        resolution_high = await get_resolution_high_metar(
            sess,
            city.id,
            date_et,
            observation_minutes,
            city_tz=city_tz,
        )
        if resolution_high is not None:
            return {"high_f": float(resolution_high), "source_used": "resolution_metar", "obs_time": None}

    # ── Raw METAR (aggregate all sources) ─────────────────────────────────
    daily_high = await get_daily_high_metar(
        sess,
        city.id,
        date_et,
        city_tz=city_tz,
    )
    if daily_high is not None:
        return {"high_f": float(daily_high), "source_used": "raw_metar", "obs_time": None}

    return {"high_f": None, "source_used": None, "obs_time": None}


async def _resolve_realized_high(
    sess,
    *,
    city: City,
    date_et: str,
    observation_minutes: list[int],
) -> Optional[float]:
    """Backward-compatible wrapper — returns just the float."""
    result = await _resolve_realized_high_with_source(
        sess,
        city=city,
        date_et=date_et,
        observation_minutes=observation_minutes,
    )
    return result["high_f"]


def _summarize_sources(
    *,
    primary_fc,
    wu_hourly,
    hrrr_fc=None,
    nbm_fc=None,
    adaptive_inputs: dict[str, Any],
    target_is_today: bool,
    now_utc: datetime,
) -> dict[str, Any]:
    def _build_source(name: str, forecast_obj, ttl: int) -> dict[str, Any]:
        if forecast_obj is None:
            return {
                "name": name,
                "high_f": None,
                "age_s": None,
                "stale": True if target_is_today else False,
            }
        age_s = _age_seconds(forecast_obj.fetched_at, now_utc=now_utc)
        return {
            "name": name,
            "high_f": _round_float(forecast_obj.high_f),
            "age_s": age_s,
            "stale": bool(target_is_today and age_s is not None and age_s > ttl),
        }

    return {
        "primary": _build_source("primary", primary_fc, 3600),
        "wu_hourly": _build_source("wu_hourly", wu_hourly, Config.WU_STALE_TTL_SECONDS),
        "hrrr": _build_source("hrrr", hrrr_fc, 3600),
        "nbm": _build_source("nbm", nbm_fc, 3600),
        "adaptive": {
            "predicted_daily_high_f": _round_float(adaptive_inputs.get("predicted_daily_high")),
            "peak_time_local": adaptive_inputs.get("composite_peak_timing"),
            "peak_already_passed": adaptive_inputs.get("peak_already_passed"),
            "kalman_trend_per_hr": _round_float(adaptive_inputs.get("kalman_trend_per_hr")),
            "regression_r2": adaptive_inputs.get("regression_r2"),
        },
    }


def _select_bucket(
    *,
    bucket_rows: list[dict[str, Any]],
    current_temp_f: Optional[float],
    obs_high_f: Optional[float],
    forecast_spread_f: Optional[float],
    model_sigma: float,
    metar_age_s: Optional[float],
    model_inputs: dict[str, Any],
    adaptive_inputs: dict[str, Any],
    source_summary: dict[str, Any],
    resolution_mismatch: Optional[float],
    cloud_trend: str,
) -> MarketContextSelection:
    ranked = sorted(
        bucket_rows,
        key=lambda row: (
            row.get("calibrated_prob") or 0.0,
            row.get("raw_model_prob") or 0.0,
            -(abs((row.get("low_f") or row.get("high_f") or 0.0) - (model_inputs.get("projected_high") or 0.0))),
        ),
        reverse=True,
    )
    selected = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    selected_prob = float(selected.get("calibrated_prob") or 0.0)
    runner_up_prob = float(runner_up.get("calibrated_prob") or 0.0) if runner_up else 0.0
    gap = max(0.0, selected_prob - runner_up_prob)
    source_stale_count = sum(
        1
        for key in ("primary", "wu_hourly")
        if source_summary.get(key, {}).get("stale")
    )

    confidence_components = {
        "base_prob": round(selected_prob * 55, 2),
        "runner_up_gap": round(gap * 95, 2),
        "metar_bonus": 6.0 if (metar_age_s is not None and metar_age_s <= Config.METAR_STALE_TTL_SECONDS) else -4.0,
        "metar_weight_bonus": round(float(model_inputs.get("w_metar", 0.0)) * 8, 2),
        "peak_passed_bonus": 5.0 if adaptive_inputs.get("peak_already_passed") else 0.0,
        "sigma_penalty": round(min(14.0, float(model_sigma) * 3.0), 2),
        "spread_penalty": round(min(10.0, abs(float(forecast_spread_f or 0.0)) * 2.0), 2),
        "stale_penalty": float(source_stale_count * 4),
        "mismatch_penalty": 6.0 if resolution_mismatch else 0.0,
        "cloud_penalty": 3.0 if cloud_trend == "worsening" else 0.0,
    }

    raw_confidence = (
        22
        + confidence_components["base_prob"]
        + confidence_components["runner_up_gap"]
        + confidence_components["metar_bonus"]
        + confidence_components["metar_weight_bonus"]
        + confidence_components["peak_passed_bonus"]
        - confidence_components["sigma_penalty"]
        - confidence_components["spread_penalty"]
        - confidence_components["stale_penalty"]
        - confidence_components["mismatch_penalty"]
        - confidence_components["cloud_penalty"]
    )
    confidence_pct = max(18, min(97, int(round(raw_confidence))))

    flip_signals = _build_flip_signals(
        selected=selected,
        current_temp_f=current_temp_f,
        obs_high_f=obs_high_f,
        projected_high_f=model_inputs.get("projected_high"),
        temp_change_1h_f=_round_float(model_inputs.get("remaining_rise")),
    )

    rationale = (
        f"{selected['label']} leads at {selected_prob * 100:.1f}% calibrated probability "
        f"versus market {((selected.get('market_prob') or 0.0) * 100):.1f}%, "
        f"with runner-up {runner_up['label'] if runner_up else 'none'} at {runner_up_prob * 100:.1f}% "
        f"and modeled high {float(model_inputs.get('projected_high') or model_inputs.get('mu_forecast') or 0.0):.1f}°F."
    )
    life_or_death_call = (
        f"If life depended on being correct, I would select {selected['label']} because "
        f"the calibrated distribution centers on that bucket at {selected_prob * 100:.1f}% "
        f"while the market is at {((selected.get('market_prob') or 0.0) * 100):.1f}%."
    )

    return MarketContextSelection(
        bucket_id=int(selected["bucket_id"]),
        bucket_idx=int(selected["bucket_idx"]),
        label=str(selected["label"]),
        low_f=selected.get("low_f"),
        high_f=selected.get("high_f"),
        calibrated_prob=selected_prob,
        raw_model_prob=float(selected.get("raw_model_prob") or 0.0),
        market_prob=selected.get("market_prob"),
        true_edge=selected.get("true_edge"),
        confidence_pct=confidence_pct,
        rationale=rationale,
        flip_signals=flip_signals,
        life_or_death_call=life_or_death_call,
        most_likely_peak_time=adaptive_inputs.get("composite_peak_timing"),
        confidence_components=confidence_components,
    )


def _build_flip_signals(
    *,
    selected: dict[str, Any],
    current_temp_f: Optional[float],
    obs_high_f: Optional[float],
    projected_high_f: Optional[float],
    temp_change_1h_f: Optional[float],
) -> list[str]:
    triggers: list[str] = []
    low_f = selected.get("low_f")
    settlement_upper_f = selected.get("settlement_upper_f")

    if settlement_upper_f is not None:
        triggers.append(
            f"If the next official reading reaches {float(settlement_upper_f):.1f}°F or higher, the hotter neighboring bucket takes control."
        )
    if low_f is not None:
        triggers.append(
            f"If the observed high stalls below {float(low_f):.1f}°F and the warming rate fades, the cooler neighboring bucket gains probability."
        )
    if projected_high_f is not None and current_temp_f is not None:
        triggers.append(
            f"If projected high slips from {float(projected_high_f):.1f}°F toward {float(current_temp_f):.1f}°F, confidence in {selected['label']} should be cut."
        )
    if temp_change_1h_f is not None:
        triggers.append(
            f"If the remaining-rise model compresses under {float(temp_change_1h_f):.1f}°F, the upside path weakens materially."
        )
    cleaned = []
    seen = set()
    for trigger in triggers:
        if trigger not in seen:
            seen.add(trigger)
            cleaned.append(trigger)
    return cleaned[:3]


def _summarize_bucket_market_movement(snapshots: list[Any]) -> dict[str, Any]:
    mids = [float(s.yes_mid) for s in snapshots if s.yes_mid is not None]
    if not mids:
        return {
            "samples": 0,
            "open_mid": None,
            "latest_mid": None,
            "change_pts": None,
            "high_mid": None,
            "low_mid": None,
            "reversal": False,
        }

    open_mid = mids[0]
    latest_mid = mids[-1]
    high_mid = max(mids)
    low_mid = min(mids)
    total_range = high_mid - low_mid
    retrace = abs(high_mid - latest_mid) if latest_mid >= open_mid else abs(latest_mid - low_mid)
    reversal = bool(total_range >= 0.05 and retrace >= total_range * 0.45)
    return {
        "samples": len(mids),
        "open_mid": round(open_mid, 4),
        "latest_mid": round(latest_mid, 4),
        "change_pts": round((latest_mid - open_mid) * 100, 1),
        "high_mid": round(high_mid, 4),
        "low_mid": round(low_mid, 4),
        "reversal": reversal,
    }


def _top_bucket(bucket_rows: list[dict[str, Any]], field: str) -> Optional[dict[str, Any]]:
    candidates = [row for row in bucket_rows if row.get(field) is not None]
    if not candidates:
        return None
    row = max(candidates, key=lambda item: item.get(field) or 0.0)
    return {
        "bucket_idx": row["bucket_idx"],
        "label": row["label"],
        "prob": row.get(field),
    }


def _runner_up_bucket(bucket_rows: list[dict[str, Any]], field: str) -> Optional[dict[str, Any]]:
    candidates = [row for row in bucket_rows if row.get(field) is not None]
    if len(candidates) < 2:
        return None
    ranked = sorted(candidates, key=lambda item: item.get(field) or 0.0, reverse=True)
    row = ranked[1]
    return {
        "bucket_idx": row["bucket_idx"],
        "label": row["label"],
        "prob": row.get(field),
    }


def _sort_edge_buckets(
    bucket_rows: list[dict[str, Any]],
    *,
    overpriced: bool,
) -> list[dict[str, Any]]:
    rows = []
    for row in bucket_rows:
        calibrated_prob = row.get("calibrated_prob")
        market_prob = row.get("market_prob")
        if calibrated_prob is None or market_prob is None:
            continue
        mispricing = round((calibrated_prob - market_prob) * 100, 1)
        if overpriced and mispricing < -2.0:
            rows.append(
                {
                    "bucket_idx": row["bucket_idx"],
                    "label": row["label"],
                    "mispricing_pts": mispricing,
                    "confidence": min(95, int(abs(mispricing) * 2 + 35)),
                }
            )
        elif not overpriced and mispricing > 2.0:
            rows.append(
                {
                    "bucket_idx": row["bucket_idx"],
                    "label": row["label"],
                    "mispricing_pts": mispricing,
                    "confidence": min(95, int(abs(mispricing) * 2 + 35)),
                }
            )
    rows.sort(key=lambda item: abs(item["mispricing_pts"]), reverse=True)
    return rows[:3]


def _consensus_spread_points(bucket_rows: list[dict[str, Any]]) -> Optional[float]:
    market_rows = [
        row for row in bucket_rows
        if row.get("market_prob") is not None
    ]
    if len(market_rows) < 2:
        return None
    ranked = sorted(market_rows, key=lambda row: row["market_prob"], reverse=True)
    return round((ranked[0]["market_prob"] - ranked[1]["market_prob"]) * 100, 1)


def _best_recent_source(error_summary: dict[str, Any]) -> Optional[dict[str, Any]]:
    ranked = []
    for source in ("nws", "wu_hourly"):
        item = error_summary.get(source) or {}
        mae = item.get("avg_abs_error_f")
        samples = item.get("samples", 0)
        if mae is None or samples <= 0:
            continue
        ranked.append((mae, -samples, source, item))
    if not ranked:
        return None
    _, _, source, item = min(ranked)
    return {
        "source": source,
        "avg_abs_error_f": item["avg_abs_error_f"],
        "samples": item["samples"],
    }


def _start_of_local_day(date_et: str, tz: ZoneInfo) -> datetime:
    return datetime.strptime(date_et, "%Y-%m-%d").replace(tzinfo=tz)


def _temperature_change(obs_rows: list[Any], *, minutes: int) -> Optional[float]:
    if not obs_rows:
        return None
    end = obs_rows[-1]
    window_start = end.observed_at - timedelta(minutes=minutes)
    eligible = [row for row in obs_rows if row.observed_at >= window_start and row.temp_f is not None]
    if len(eligible) < 2:
        return None
    return round(float(eligible[-1].temp_f) - float(eligible[0].temp_f), 1)


def _warming_acceleration(obs_rows: list[Any]) -> Optional[float]:
    if len(obs_rows) < 4:
        return None
    recent = _temperature_change(obs_rows, minutes=60)
    previous_end = obs_rows[-1].observed_at - timedelta(minutes=60)
    previous = [row for row in obs_rows if previous_end - timedelta(minutes=60) <= row.observed_at <= previous_end and row.temp_f is not None]
    if recent is None or len(previous) < 2:
        return None
    prev_delta = float(previous[-1].temp_f) - float(previous[0].temp_f)
    return round(recent - prev_delta, 1)


def _pressure_tendency(obs_rows: list[Any], *, minutes: int) -> Optional[float]:
    if not obs_rows:
        return None
    end = obs_rows[-1].observed_at
    window_start = end - timedelta(minutes=minutes)
    eligible = [
        row for row in obs_rows
        if row.observed_at >= window_start
        and row.extended is not None
        and row.extended.altimeter_inhg is not None
    ]
    if len(eligible) < 2:
        return None
    return round(float(eligible[-1].extended.altimeter_inhg) - float(eligible[0].extended.altimeter_inhg), 2)


def _wind_shift(obs_rows: list[Any], *, minutes: int) -> Optional[int]:
    if not obs_rows:
        return None
    end = obs_rows[-1].observed_at
    window_start = end - timedelta(minutes=minutes)
    eligible = [
        row for row in obs_rows
        if row.observed_at >= window_start
        and row.extended is not None
        and row.extended.wind_dir_deg is not None
    ]
    if len(eligible) < 2:
        return None
    start_dir = int(eligible[0].extended.wind_dir_deg)
    end_dir = int(eligible[-1].extended.wind_dir_deg)
    raw = abs(end_dir - start_dir)
    return min(raw, 360 - raw)


def _cloud_trend(obs_rows: list[Any], *, minutes: int) -> str:
    if not obs_rows:
        return "unknown"
    end = obs_rows[-1].observed_at
    window_start = end - timedelta(minutes=minutes)
    eligible = [
        row for row in obs_rows
        if row.observed_at >= window_start
        and row.extended is not None
        and row.extended.cloud_cover is not None
    ]
    if len(eligible) < 2:
        return "steady"
    start_val = _cloud_cover_value(eligible[0].extended.cloud_cover)
    end_val = _cloud_cover_value(eligible[-1].extended.cloud_cover)
    delta = end_val - start_val
    if delta >= 1:
        return "worsening"
    if delta <= -1:
        return "improving"
    return "steady"


def _cloud_cover_value(cover: Optional[str]) -> int:
    return {
        "CLR": 0,
        "SKC": 0,
        "FEW": 1,
        "SCT": 2,
        "BKN": 3,
        "OVC": 4,
    }.get((cover or "").upper(), 2)


def _matches_station_pattern(obs_row: Any, observation_minutes: list[int], tz: ZoneInfo) -> bool:
    if not obs_row or not observation_minutes:
        return False
    dt_local = obs_row.observed_at.astimezone(tz)
    for minute in observation_minutes:
        diff = abs(dt_local.minute - minute)
        if diff <= 1 or diff >= 59:
            return True
    return False


def _bucket_label_from_bounds(low_f: Optional[float], high_f: Optional[float]) -> str:
    if low_f is not None and high_f is not None:
        return f"{int(low_f)}–{int(high_f)}°"
    if low_f is not None:
        return f"≥{int(low_f)}°"
    if high_f is not None:
        return f"<{int(high_f)}°"
    return "Unknown"


def _json_list(raw: str) -> list[int]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [int(v) for v in parsed]
    except Exception:
        pass
    return []


def _fmt_local(dt: Optional[datetime], tz: ZoneInfo) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz).strftime("%-I:%M %p %Z")


def _age_seconds(dt: Optional[datetime], *, now_utc: Optional[datetime] = None) -> Optional[float]:
    if dt is None:
        return None
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return round((now_utc - dt.astimezone(timezone.utc)).total_seconds(), 0)


def _round_float(value: Any, digits: int = 1) -> Optional[float]:
    try:
        if value is None:
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None
