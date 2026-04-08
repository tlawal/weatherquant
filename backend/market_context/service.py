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
    SECTION_KEYS,
)
from backend.modeling.calibration_engine import get_reliability_metrics, remap_probability
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
        wu_daily = await get_latest_successful_forecast(sess, city.id, "wu_daily", date_et)
        wu_hourly = await get_latest_successful_forecast(sess, city.id, "wu_hourly", date_et)
        wu_history = await get_latest_successful_forecast(sess, city.id, "wu_history", date_et)
        hrrr_fc = await get_latest_successful_forecast(sess, city.id, "hrrr", date_et)
        nbm_fc = await get_latest_successful_forecast(sess, city.id, "nbm", date_et)
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
        wu_daily=wu_daily,
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

    availability = {
        "latest_observations": bool(obs_rows),
        "nws_available": primary_fc is not None if city.is_us else False,
        "wu_daily_available": wu_daily is not None,
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
        short_range_models={
            "stored_sources": sources,
            "forecast_spread_f": _round_float(model_inputs.get("spread")),
            "mu_f": _round_float(model.mu),
            "sigma_f": _round_float(model.sigma),
            "projected_high_f": _round_float(model_inputs.get("projected_high")),
            "prob_new_high": _round_float(model_inputs.get("prob_new_high"), 4),
            "best_recent_source": _best_recent_source(error_summary),
            "recent_error_summary": error_summary,
            "hrrr_high_f": _round_float(hrrr_fc.high_f) if hrrr_fc else None,
            "nbm_high_f": _round_float(nbm_fc.high_f) if nbm_fc else None,
            "missing_external_models": [
                m for m, avail in [
                    ("HRRR", hrrr_fc is not None),
                    ("NBM", nbm_fc is not None),
                    ("NAM", False),
                    ("RAP", False),
                    ("ECMWF", False),
                ] if not avail
            ],
        },
        historical_context={
            "avg_peak_time_local_7d": avg_peak_timing,
            "recent_realized_highs": recent_highs["recent_days"],
            "avg_high_7d_f": recent_highs["avg_7d"],
            "avg_high_prev_7d_f": recent_highs["avg_prev_7d"],
            "trend_delta_f": recent_highs["trend_delta"],
            "last_settled_date_et": last_settled["date_et"] if last_settled else None,
            "last_settled_realized_high_f": last_settled["realized_high_f"] if last_settled else None,
            "last_settled_errors_f": last_settled["errors"] if last_settled else {},
            "calibration_biases_f": {
                "nws": _round_float(calibration.bias_nws) if calibration else 0.0,
                "wu_daily": _round_float(calibration.bias_wu_daily) if calibration else 0.0,
                "wu_hourly": _round_float(calibration.bias_wu_hourly) if calibration else 0.0,
            },
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
            parsed = MarketContextOutput(
                sections=payload.get("sections", {}),
                final_selection=context.final_selection,
            )
            return parsed
        except Exception as exc:
            last_error = exc

    raise MarketContextLLMError(f"Market Context generation failed validation: {last_error}")


def _build_prompts(context: MarketContextInput) -> tuple[str, str]:
    system_prompt = (
        "You are an autonomous quantitative weather-derivatives analyst producing a concise, "
        "evidence-dense Market Context report for a Polymarket temperature-bucket prediction. "
        "Your output drives real capital allocation.\n\n"
        "Operating principles:\n"
        "1. ACTIVELY USE YOUR TOOLS. Before writing the report, call fetch_nws_discussion to read "
        "the NWS Area Forecast Discussion for synoptic context. Call search_academic_climatology "
        "for relevant peer-reviewed heuristics. Call fetch_nbm_forecast and/or fetch_hrrr_forecast "
        "for the latest hourly temperature curves. These external sources are CRITICAL.\n"
        "2. Synthesize ALL available information — DB context, tool-call results, NWS discussions, "
        "academic heuristics, and your own meteorological knowledge — into a unified analysis.\n"
        "3. When model forecasts disagree, diagnose WHY using synoptic regime analysis from the "
        "NWS Area Forecast Discussion, initialization timing, known biases, and academic literature.\n"
        "4. Apply bias corrections explicitly: state the raw forecast AND the bias-adjusted value.\n"
        "5. Quantify uncertainty via sigma_f, forecast_spread_f, and tool-derived ensemble spread.\n"
        "6. The final_selection is a pre-computed baseline from the deterministic algorithm. "
        "Echo it exactly in the JSON output. However, in your diagnostic_reasoning and "
        "independent_assessment sections, state whether you AGREE or DISAGREE with it and why. "
        "If your tool-informed analysis points to a different bucket, explicitly flag the discrepancy "
        "with reasoning. This independent assessment is extremely valuable for trading decisions.\n"
        "7. Be terse. Each section: 2-4 sentences max. Total output must fit 1800 tokens.\n"
        "8. Return ONLY valid JSON with keys `sections` (7 string values) and `final_selection` "
        "(echo pre-computed values exactly)."
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

    section_keys_extended = list(SECTION_KEYS) + (
        ["independent_assessment"] if "independent_assessment" not in SECTION_KEYS else []
    )

    user_prompt = (
        f"Market Context report: {context.city_display} on {context.date_et}. Return JSON only.\n\n"
        "IMPORTANT: Before writing your analysis, USE YOUR TOOLS:\n"
        f"  1. Call fetch_nws_discussion(city_slug='{context.city_slug}') for synoptic analysis\n"
        f"  2. Call search_academic_climatology with a query relevant to today's weather pattern\n"
        f"  3. Call fetch_nbm_forecast(city_slug='{context.city_slug}') for the latest NBM hourly curve\n"
        f"  4. Call fetch_hrrr_forecast(city_slug='{context.city_slug}') for the latest HRRR curve\n"
        "These tool results should inform every section of your analysis.\n\n"
        "STRUCTURE: `sections` must contain exactly these keys: "
        + ", ".join(section_keys_extended)
        + "\n"
        "Each value: a single markdown prose string (2-4 dense sentences). No nested JSON objects or lists.\n\n"
        "SECTION REQUIREMENTS:\n\n"
        "1. current_observations -- State current_temp_f, observed_high_f, resolution_high_f. "
        "Report dewpoint_spread_f and its implication for remaining heating potential. "
        "Cite temp_change_1h_f, warming_acceleration_f_per_hr_delta, and cloud_trend to characterize the "
        "temperature trajectory (accelerating, decelerating, or stalling). Flag if metar_age_s > 1200.\n\n"
        "2. short_range_model_landscape -- Report mu_f +/- sigma_f as distribution center/width. "
        "State forecast_spread_f across sources. "
        "Compare hrrr_high_f vs nbm_high_f; diagnose >1F disagreements using known model biases "
        "AND the NWS discussion (from your tool call). "
        "Apply calibration_biases_f to each source to produce bias-adjusted forecasts. "
        "Name best_recent_source and its MAE from recent_error_summary. Note missing_external_models. "
        "Reference the HRRR and NBM hourly curves from your tool calls to identify peak timing.\n\n"
        "3. historical_climatology_perspective -- Report avg_high_7d_f, avg_high_prev_7d_f, trend_delta_f. "
        "Compare projected_high_f to the 7-day realized average. "
        "If last_settled_errors_f exist, state which source was most accurate yesterday. "
        "Quantify calibration_biases_f for NWS, WU daily, WU hourly. "
        "Reference academic climatology papers (from your tool call) for relevant phenomena "
        "(cold air damming, urban heat island, lake breeze, transition-season fat tails, etc.).\n\n"
        "4. market_pricing_analysis -- State model_consensus_bucket vs market_consensus_bucket. "
        "If they diverge, explain true_edge on the selected bucket. Report consensus_spread_pts. "
        "Identify top overpriced_buckets and underpriced_buckets by true_edge magnitude. "
        "State selected bucket calibrated_prob vs market_prob.\n\n"
        "5. diagnostic_reasoning -- Build a causal chain using BOTH DB context AND tool results: "
        "(a) peak_already_passed? If yes, anchor on remaining_rise_f and projected_high_f. "
        "(b) kalman_trend_per_hr + regression_r2: is the intraday trend reliable? "
        "(c) pressure_tendency_inhg_3h + wind_shift_deg_3h: synoptic shift? Reference NWS discussion. "
        "(d) cloud_trend + resolution_mismatch_f: observational red flags? "
        "(e) What does the NWS forecaster say about confidence and potential pitfalls? "
        "Synthesize a single verdict: projected high robust or vulnerable, and why.\n\n"
        "6. final_high_stakes_selection -- Restate the pre-computed bucket label and confidence_pct. "
        "Decompose confidence via confidence_components (top 2 boosters, top 2 penalties by magnitude). "
        "State flip_signals verbatim. Declare peak_time and life_or_death_call.\n\n"
        "7. independent_assessment -- YOUR OWN expert opinion based on tool results. "
        "State your independent projected high temperature and which bucket you believe is most likely. "
        "If this AGREES with the pre-computed selection, say so and explain the converging evidence. "
        "If this DISAGREES, explicitly flag it: state your preferred bucket, your projected high, "
        "and the key evidence (NWS discussion phrase, NBM peak, HRRR curve shape, academic heuristic) "
        "that justifies divergence. This section is the most valuable for trading decisions.\n\n"
        "CONSISTENCY RULES:\n"
        "- Probability figures must match context to 1 decimal place. Do not alter calibrated_prob, market_prob, or true_edge.\n"
        "- final_selection fields bucket_id, bucket_idx, label, confidence_pct: echo EXACTLY as provided.\n"
        "- In independent_assessment, clearly separate your opinion from the algorithmic baseline.\n\n"
        f"AVAILABILITY: {availability_note}\n\n"
        "Pre-computed baseline selection (echo exactly in final_selection):\n"
        f"{json.dumps(context.final_selection.model_dump(), indent=2)}\n\n"
        "Structured backend context:\n"
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
    errors_by_source: dict[str, list[float]] = {"nws": [], "wu_daily": [], "wu_hourly": []}
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
        for source in ("nws", "wu_daily", "wu_hourly"):
            fc = await get_latest_successful_forecast(sess, city.id, source, event.date_et)
            if fc and fc.high_f is not None:
                err = round(fc.high_f - realized_high, 1)
                errors_by_source[source].append(abs(err))
                date_errors[source] = err

        if last_settled is None:
            last_settled = {
                "date_et": event.date_et,
                "realized_high_f": realized_high,
                "errors": date_errors,
            }

        if sum(len(v) for v in errors_by_source.values()) >= 6:
            break

    for source, values in errors_by_source.items():
        summary[source] = {
            "avg_abs_error_f": round(sum(values) / len(values), 2) if values else None,
            "samples": len(values),
        }
    summary["window"] = "last 48h settled sample"
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


async def _resolve_realized_high(
    sess,
    *,
    city: City,
    date_et: str,
    observation_minutes: list[int],
) -> Optional[float]:
    wu_history = await get_latest_successful_forecast(sess, city.id, "wu_history", date_et)
    if wu_history and wu_history.high_f is not None:
        return float(wu_history.high_f)

    if observation_minutes:
        resolution_high = await get_resolution_high_metar(
            sess,
            city.id,
            date_et,
            observation_minutes,
            city_tz=getattr(city, "tz", "America/New_York"),
        )
        if resolution_high is not None:
            return float(resolution_high)

    daily_high = await get_daily_high_metar(
        sess,
        city.id,
        date_et,
        city_tz=getattr(city, "tz", "America/New_York"),
    )
    return float(daily_high) if daily_high is not None else None


def _summarize_sources(
    *,
    primary_fc,
    wu_daily,
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
        "wu_daily": _build_source("wu_daily", wu_daily, Config.WU_STALE_TTL_SECONDS),
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
        for key in ("primary", "wu_daily", "wu_hourly")
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
    high_f = selected.get("high_f")

    if high_f is not None:
        triggers.append(
            f"If the next official reading reaches {float(high_f):.1f}°F or higher, the hotter neighboring bucket takes control."
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
    for source in ("nws", "wu_daily", "wu_hourly"):
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
