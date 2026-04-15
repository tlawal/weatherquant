"""
SQLAlchemy 2.0 ORM models for WeatherQuant.

All tables use integer primary keys for simplicity; UUIDs for external IDs.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Index, Integer,
    String, Text, UniqueConstraint, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ─── City Registry ────────────────────────────────────────────────────────────

class City(Base):
    """Registry of all cities with temperature markets on Polymarket."""
    __tablename__ = "cities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city_slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    metar_station: Mapped[Optional[str]] = mapped_column(String(8))
    nws_office: Mapped[Optional[str]] = mapped_column(String(8))
    nws_grid_x: Mapped[Optional[int]] = mapped_column(Integer)
    nws_grid_y: Mapped[Optional[int]] = mapped_column(Integer)
    wu_state: Mapped[Optional[str]] = mapped_column(String(32))
    wu_city: Mapped[Optional[str]] = mapped_column(String(64))
    lat: Mapped[Optional[float]] = mapped_column(Float)
    lon: Mapped[Optional[float]] = mapped_column(Float)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_us: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # "F" or "C"
    unit: Mapped[str] = mapped_column(String(1), default="F", nullable=False)
    # IANA timezone (e.g., "America/New_York", "Asia/Tokyo")
    tz: Mapped[str] = mapped_column(String(64), default="America/New_York", nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    events: Mapped[list["Event"]] = relationship("Event", back_populates="city_ref")
    metar_obs: Mapped[list["MetarObs"]] = relationship("MetarObs", back_populates="city_ref")
    forecast_obs: Mapped[list["ForecastObs"]] = relationship(
        "ForecastObs", back_populates="city_ref"
    )
    calibration: Mapped[Optional["CalibrationParams"]] = relationship(
        "CalibrationParams", back_populates="city_ref", uselist=False
    )
    market_context_snapshots: Mapped[list["MarketContextSnapshot"]] = relationship(
        "MarketContextSnapshot", back_populates="city_ref", cascade="all, delete-orphan"
    )


# ─── Events & Buckets ─────────────────────────────────────────────────────────

class Event(Base):
    """One Polymarket event per city per ET calendar day."""
    __tablename__ = "events"
    __table_args__ = (UniqueConstraint("city_id", "date_et", name="uq_event_city_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city_id: Mapped[int] = mapped_column(Integer, ForeignKey("cities.id"), nullable=False)
    date_et: Mapped[str] = mapped_column(String(10), nullable=False)  # YYYY-MM-DD
    gamma_event_id: Mapped[Optional[str]] = mapped_column(String(128))
    gamma_slug: Mapped[Optional[str]] = mapped_column(String(256))
    settlement_source: Mapped[Optional[str]] = mapped_column(String(256))
    settlement_source_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    resolution_source_url: Mapped[Optional[str]] = mapped_column(Text)
    resolution_station_id: Mapped[Optional[str]] = mapped_column(String(16))
    # ok | degraded | no_event | bad_buckets
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    forecast_quality: Mapped[str] = mapped_column(String(16), default="ok", nullable=False)
    wu_scrape_error: Mapped[Optional[str]] = mapped_column(Text)
    trading_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    winning_bucket_idx: Mapped[Optional[int]] = mapped_column(Integer)
    redeemed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    city_ref: Mapped[City] = relationship("City", back_populates="events")
    buckets: Mapped[list["Bucket"]] = relationship(
        "Bucket", back_populates="event", cascade="all, delete-orphan"
    )
    model_snapshots: Mapped[list["ModelSnapshot"]] = relationship(
        "ModelSnapshot", back_populates="event"
    )


class Bucket(Base):
    """One YES/NO temperature bucket market within an Event."""
    __tablename__ = "buckets"
    __table_args__ = (UniqueConstraint("event_id", "bucket_idx", name="uq_bucket_event_idx"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(Integer, ForeignKey("events.id"), nullable=False)
    bucket_idx: Mapped[int] = mapped_column(Integer, nullable=False)  # 0-based
    label: Mapped[Optional[str]] = mapped_column(String(256))
    low_f: Mapped[Optional[float]] = mapped_column(Float)   # None = -inf (below bucket)
    high_f: Mapped[Optional[float]] = mapped_column(Float)  # None = +inf (above bucket)
    yes_token_id: Mapped[Optional[str]] = mapped_column(String(128))
    no_token_id: Mapped[Optional[str]] = mapped_column(String(128))
    condition_id: Mapped[Optional[str]] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    event: Mapped[Event] = relationship("Event", back_populates="buckets")
    market_snapshots: Mapped[list["MarketSnapshot"]] = relationship(
        "MarketSnapshot", back_populates="bucket"
    )
    signals: Mapped[list["Signal"]] = relationship("Signal", back_populates="bucket")
    orders: Mapped[list["Order"]] = relationship("Order", back_populates="bucket")
    position: Mapped[Optional["Position"]] = relationship(
        "Position", back_populates="bucket", uselist=False
    )


# ─── Ingestion Snapshots ──────────────────────────────────────────────────────

class MetarObs(Base):
    """Raw METAR observation per station per poll."""
    __tablename__ = "metar_obs"
    __table_args__ = (Index("ix_metar_station_ts", "metar_station", "observed_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city_id: Mapped[int] = mapped_column(Integer, ForeignKey("cities.id"), nullable=False)
    metar_station: Mapped[str] = mapped_column(String(8), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    temp_c: Mapped[Optional[float]] = mapped_column(Float)
    temp_f: Mapped[Optional[float]] = mapped_column(Float)
    daily_high_f: Mapped[Optional[float]] = mapped_column(Float)
    report_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    raw_text: Mapped[Optional[str]] = mapped_column(Text)
    raw_json: Mapped[Optional[str]] = mapped_column(Text)

    city_ref: Mapped[City] = relationship("City", back_populates="metar_obs")
    extended: Mapped[Optional["MetarObsExtended"]] = relationship(
        "MetarObsExtended", back_populates="metar_obs_ref", uselist=False,
        cascade="all, delete-orphan",
    )


class MetarObsExtended(Base):
    """Extended METAR fields parsed from raw_json — 1:1 with MetarObs."""
    __tablename__ = "metar_obs_extended"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    metar_obs_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("metar_obs.id"), unique=True, nullable=False
    )
    dewpoint_c: Mapped[Optional[float]] = mapped_column(Float)
    dewpoint_f: Mapped[Optional[float]] = mapped_column(Float)
    humidity_pct: Mapped[Optional[float]] = mapped_column(Float)
    wind_dir_deg: Mapped[Optional[int]] = mapped_column(Integer)
    wind_speed_kt: Mapped[Optional[float]] = mapped_column(Float)
    wind_gust_kt: Mapped[Optional[float]] = mapped_column(Float)
    altimeter_inhg: Mapped[Optional[float]] = mapped_column(Float)
    sea_level_pressure_mb: Mapped[Optional[float]] = mapped_column(Float)
    visibility_sm: Mapped[Optional[float]] = mapped_column(Float)
    cloud_cover: Mapped[Optional[str]] = mapped_column(String(4))   # CLR/FEW/SCT/BKN/OVC
    cloud_base_ft: Mapped[Optional[int]] = mapped_column(Integer)
    wx_string: Mapped[Optional[str]] = mapped_column(String(64))    # -RA, TS, etc.
    precip_in: Mapped[Optional[float]] = mapped_column(Float)
    condition: Mapped[Optional[str]] = mapped_column(String(32))    # Fair, Cloudy, Rain, etc.

    metar_obs_ref: Mapped[MetarObs] = relationship("MetarObs", back_populates="extended")


class StationProfile(Base):
    """Detected observation pattern for a METAR station."""
    __tablename__ = "station_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    metar_station: Mapped[str] = mapped_column(String(8), unique=True, nullable=False)
    # JSON array e.g. "[52]" or "[0,30]"
    observation_minutes: Mapped[Optional[str]] = mapped_column(Text)
    # "hourly" | "half_hourly" | "irregular"
    observation_frequency: Mapped[Optional[str]] = mapped_column(String(16))
    samples_analyzed: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    last_analyzed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class ForecastObs(Base):
    """Single forecast reading from one source (nws / wu_hourly / wu_history / hrrr / nbm / ecmwf_ifs / open_meteo)."""
    __tablename__ = "forecast_obs"
    __table_args__ = (
        Index("ix_forecast_city_source_ts", "city_id", "source", "fetched_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city_id: Mapped[int] = mapped_column(Integer, ForeignKey("cities.id"), nullable=False)
    # "nws" | "wu_hourly" | "wu_history" | "hrrr" | "nbm" | "ecmwf_ifs" | "open_meteo"
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    date_et: Mapped[str] = mapped_column(String(10), nullable=False)  # YYYY-MM-DD
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    high_f: Mapped[Optional[float]] = mapped_column(Float)
    raw_payload_hash: Mapped[Optional[str]] = mapped_column(String(64))
    raw_json: Mapped[Optional[str]] = mapped_column(Text)
    parse_error: Mapped[Optional[str]] = mapped_column(Text)

    city_ref: Mapped[City] = relationship("City", back_populates="forecast_obs")


# ─── Market Data ──────────────────────────────────────────────────────────────

class MarketSnapshot(Base):
    """CLOB orderbook snapshot for one bucket at one point in time."""
    __tablename__ = "market_snapshots"
    __table_args__ = (Index("ix_mktsnap_bucket_ts", "bucket_id", "fetched_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bucket_id: Mapped[int] = mapped_column(Integer, ForeignKey("buckets.id"), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    yes_bid: Mapped[Optional[float]] = mapped_column(Float)
    yes_ask: Mapped[Optional[float]] = mapped_column(Float)
    yes_mid: Mapped[Optional[float]] = mapped_column(Float)
    yes_bid_depth: Mapped[Optional[float]] = mapped_column(Float)
    yes_ask_depth: Mapped[Optional[float]] = mapped_column(Float)
    spread: Mapped[Optional[float]] = mapped_column(Float)

    bucket: Mapped[Bucket] = relationship("Bucket", back_populates="market_snapshots")


# ─── Model Output ─────────────────────────────────────────────────────────────

class ModelSnapshot(Base):
    """Temperature distribution model output for one event at one point in time."""
    __tablename__ = "model_snapshots"
    __table_args__ = (Index("ix_modelsnap_event_ts", "event_id", "computed_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(Integer, ForeignKey("events.id"), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    mu: Mapped[float] = mapped_column(Float, nullable=False)
    sigma: Mapped[float] = mapped_column(Float, nullable=False)
    # JSON array of bucket probabilities (float), length == number of buckets
    probs_json: Mapped[str] = mapped_column(Text, nullable=False)
    # JSON dict of inputs used: nws_val, wu_hourly_val,
    #   metar_high_so_far, w_metar, projected_high, mu_forecast, spread, etc.
    inputs_json: Mapped[Optional[str]] = mapped_column(Text)
    forecast_quality: Mapped[str] = mapped_column(String(16), default="ok")

    event: Mapped[Event] = relationship("Event", back_populates="model_snapshots")


class MarketContextSnapshot(Base):
    """Stored Market Context narrative and structured source payload for a city/date."""
    __tablename__ = "market_context_snapshots"
    __table_args__ = (
        UniqueConstraint("city_id", "date_et", name="uq_market_context_city_date"),
        Index("ix_market_context_city_date", "city_id", "date_et"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city_id: Mapped[int] = mapped_column(Integer, ForeignKey("cities.id"), nullable=False)
    date_et: Mapped[str] = mapped_column(String(10), nullable=False)
    generation_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    sections_json: Mapped[Optional[str]] = mapped_column(Text)
    selection_json: Mapped[Optional[str]] = mapped_column(Text)
    source_context_json: Mapped[Optional[str]] = mapped_column(Text)
    provider: Mapped[Optional[str]] = mapped_column(String(32))
    model_name: Mapped[Optional[str]] = mapped_column(String(128))
    generated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    freshness_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    city_ref: Mapped[City] = relationship("City", back_populates="market_context_snapshots")


# ─── Signals ──────────────────────────────────────────────────────────────────

class Signal(Base):
    """Computed edge for one bucket at one point in time."""
    __tablename__ = "signals"
    __table_args__ = (
        Index("ix_signal_bucket_ts", "bucket_id", "computed_at"),
        Index("ix_signal_snapshot", "model_snapshot_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bucket_id: Mapped[int] = mapped_column(Integer, ForeignKey("buckets.id"), nullable=False)
    # Generation tag — every Signal row is produced inside a single
    # _compute_city_signals() pass alongside one ModelSnapshot. Stamping the
    # snapshot id lets reads filter to "rows from the latest generation only",
    # so the dashboard never mixes pre/post-rerun signals for the same event.
    model_snapshot_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("model_snapshots.id"), nullable=True
    )
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    model_prob: Mapped[float] = mapped_column(Float, nullable=False)
    mkt_prob: Mapped[float] = mapped_column(Float, nullable=False)
    raw_edge: Mapped[float] = mapped_column(Float, nullable=False)
    exec_cost: Mapped[float] = mapped_column(Float, nullable=False)
    true_edge: Mapped[float] = mapped_column(Float, nullable=False)
    # JSON with full reasoning: mu, sigma, nws_val, spread, depth, etc.
    reason_json: Mapped[Optional[str]] = mapped_column(Text)
    # gate failures preventing execution (JSON list of strings)
    gate_failures_json: Mapped[Optional[str]] = mapped_column(Text)

    bucket: Mapped[Bucket] = relationship("Bucket", back_populates="signals")


# ─── Execution ────────────────────────────────────────────────────────────────

class Order(Base):
    """One limit order placed or attempted."""
    __tablename__ = "orders"
    __table_args__ = (Index("ix_order_bucket_ts", "bucket_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bucket_id: Mapped[int] = mapped_column(Integer, ForeignKey("buckets.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    # "buy_yes" | "buy_no"
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    limit_price: Mapped[float] = mapped_column(Float, nullable=False)
    # "pending" | "open" | "filled" | "cancelled" | "rejected" | "timeout"
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    clob_order_id: Mapped[Optional[str]] = mapped_column(String(128))
    fill_price: Mapped[Optional[float]] = mapped_column(Float)
    fill_qty: Mapped[Optional[float]] = mapped_column(Float)
    cancel_reason: Mapped[Optional[str]] = mapped_column(String(256))
    signal_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("signals.id"))
    # JSON of gate check results at time of order
    gates_json: Mapped[Optional[str]] = mapped_column(Text)

    bucket: Mapped[Bucket] = relationship("Bucket", back_populates="orders")
    fills: Mapped[list["Fill"]] = relationship(
        "Fill", back_populates="order", cascade="all, delete-orphan"
    )


class Fill(Base):
    """Individual fill event for an order."""
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(Integer, ForeignKey("orders.id"), nullable=False)
    filled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    fee: Mapped[float] = mapped_column(Float, default=0.0)

    order: Mapped[Order] = relationship("Order", back_populates="fills")


class Position(Base):
    """Aggregate position per bucket (net shares, avg cost, PnL)."""
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bucket_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("buckets.id"), unique=True, nullable=False
    )
    # "yes" | "no"
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    net_qty: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    avg_cost: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_mkt_price: Mapped[Optional[float]] = mapped_column(Float)
    
    # Traceability and execution logging
    entry_type: Mapped[Optional[str]] = mapped_column(String(16)) # "AUTOMATIC" | "MANUAL"
    strategy: Mapped[Optional[str]] = mapped_column(String(64))
    governing_exit_conditions: Mapped[Optional[str]] = mapped_column(Text)
    current_exit_status: Mapped[Optional[str]] = mapped_column(String(256))
    entry_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    entry_price: Mapped[Optional[float]] = mapped_column(Float)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    bucket: Mapped[Bucket] = relationship("Bucket", back_populates="position")


# ─── Audit & Config ───────────────────────────────────────────────────────────

class AuditLog(Base):
    """Append-only audit trail of all system decisions."""
    __tablename__ = "audit_log"
    __table_args__ = (Index("ix_audit_ts", "ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    actor: Mapped[str] = mapped_column(String(64), nullable=False)  # "system" | "user" | IP
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    # JSON payload — full context for the action
    payload_json: Mapped[Optional[str]] = mapped_column(Text)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    error_msg: Mapped[Optional[str]] = mapped_column(Text)


class ArmingState(Base):
    """Singleton arming state machine. Always exactly 1 row (id=1)."""
    __tablename__ = "arming_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    # "DISARMED" | "ARMING_PENDING" | "ARMED"
    state: Mapped[str] = mapped_column(String(16), default="DISARMED", nullable=False)
    auto_redeem_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    arming_token: Mapped[Optional[str]] = mapped_column(String(64))
    token_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    armed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    armed_by: Mapped[Optional[str]] = mapped_column(String(64))
    disarmed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    disarmed_reason: Mapped[Optional[str]] = mapped_column(String(256))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class CalibrationParams(Base):
    """Per-city forecast bias corrections and ensemble weights."""
    __tablename__ = "calibration_params"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cities.id"), unique=True, nullable=False
    )
    # Additive bias corrections (degrees F)
    bias_nws: Mapped[float] = mapped_column(Float, default=0.0)
    # DEPRECATED: wu_daily ensemble source was removed (HTML scrape unreliable).
    # Column retained to avoid destructive migration; no code reads/writes it.
    bias_wu_daily: Mapped[float] = mapped_column(Float, default=0.0)
    bias_wu_hourly: Mapped[float] = mapped_column(Float, default=0.0)
    bias_hrrr: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    bias_nbm: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    # Ensemble weights (must sum to 1.0)
    weight_nws: Mapped[float] = mapped_column(Float, default=0.333)
    # DEPRECATED: wu_daily ensemble source was removed.
    weight_wu_daily: Mapped[float] = mapped_column(Float, default=0.334)
    weight_wu_hourly: Mapped[float] = mapped_column(Float, default=0.333)
    weight_hrrr: Mapped[Optional[float]] = mapped_column(Float, default=0.5)
    weight_nbm: Mapped[Optional[float]] = mapped_column(Float, default=0.2)
    # Metadata
    n_samples: Mapped[int] = mapped_column(Integer, default=0)
    last_realized_high: Mapped[Optional[float]] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    city_ref: Mapped[City] = relationship("City", back_populates="calibration")


class WorkerHeartbeat(Base):
    """Worker liveness check — one row per named job, updated periodically."""
    __tablename__ = "worker_heartbeats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    last_run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_success_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[Optional[str]] = mapped_column(Text)
    run_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)


# ─── Backtesting ─────────────────────────────────────────────────────────────

class BacktestRun(Base):
    """One complete backtest execution with parameters and summary metrics."""
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    # Parameters used (JSON dict of BacktestParams)
    params_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Date range covered
    start_date: Mapped[str] = mapped_column(String(10), nullable=False)
    end_date: Mapped[str] = mapped_column(String(10), nullable=False)

    # Summary metrics
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)
    total_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float)
    brier_score: Mapped[Optional[float]] = mapped_column(Float)
    brier_skill_score: Mapped[Optional[float]] = mapped_column(Float)
    win_rate: Mapped[Optional[float]] = mapped_column(Float)
    avg_edge: Mapped[Optional[float]] = mapped_column(Float)

    # Full results: equity_curve, daily_pnl, per_city_stats, reliability_bins
    results_json: Mapped[Optional[str]] = mapped_column(Text)

    # "running" | "completed" | "failed"
    status: Mapped[str] = mapped_column(String(16), default="running", nullable=False)
    error_msg: Mapped[Optional[str]] = mapped_column(Text)

    trades: Mapped[list["BacktestTrade"]] = relationship(
        "BacktestTrade", back_populates="run", cascade="all, delete-orphan"
    )


class BacktestTrade(Base):
    """One simulated trade within a backtest run."""
    __tablename__ = "backtest_trades"
    __table_args__ = (Index("ix_bt_trade_run", "run_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("backtest_runs.id"), nullable=False)

    city_slug: Mapped[str] = mapped_column(String(64), nullable=False)
    date_et: Mapped[str] = mapped_column(String(10), nullable=False)
    bucket_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    bucket_label: Mapped[Optional[str]] = mapped_column(String(256))

    # Signal at entry time
    model_prob: Mapped[float] = mapped_column(Float, nullable=False)
    mkt_prob: Mapped[float] = mapped_column(Float, nullable=False)
    true_edge: Mapped[float] = mapped_column(Float, nullable=False)

    # Simulated execution
    side: Mapped[str] = mapped_column(String(8), nullable=False)   # "buy_yes" | "buy_no"
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    shares: Mapped[float] = mapped_column(Float, nullable=False)
    cost: Mapped[float] = mapped_column(Float, nullable=False)

    # Outcome
    won: Mapped[Optional[bool]] = mapped_column(Boolean)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)
    # "resolved_win" | "resolved_loss" | "quick_flip" | "expiry"
    exit_reason: Mapped[Optional[str]] = mapped_column(String(32))

    run: Mapped["BacktestRun"] = relationship("BacktestRun", back_populates="trades")


class StationCalibration(Base):
    """Per-station 30-day rolling forecast calibration metrics.

    Compares fused ensemble forecast vs. observed METAR daily highs
    to track MAE, bias, and tradeability per ICAO station.
    """
    __tablename__ = "station_calibrations"
    __table_args__ = (
        UniqueConstraint("station_id", name="uq_station_cal_station"),
        Index("ix_station_cal_tradeability", "tradeability"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    station_id: Mapped[str] = mapped_column(String(8), nullable=False)   # ICAO e.g. "KATL"
    city_slug: Mapped[str] = mapped_column(String(64), nullable=False)
    city_name: Mapped[str] = mapped_column(String(128), nullable=False)
    lat: Mapped[Optional[float]] = mapped_column(Float)
    lon: Mapped[Optional[float]] = mapped_column(Float)
    # 30-day rolling metrics
    mae_f: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    bias_f: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    rmse_f: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    n_samples: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    pct_days_traded: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    # "GREEN" (<1.5°F MAE), "AMBER" (1.5-3.0), "RED" (>3.0)
    tradeability: Mapped[str] = mapped_column(String(8), default="RED", nullable=False)
    # Best individual forecast source and its MAE
    best_source: Mapped[Optional[str]] = mapped_column(String(16))
    best_source_mae: Mapped[Optional[float]] = mapped_column(Float)
    # Per-source MAE breakdown (JSON dict)
    source_mae_json: Mapped[Optional[str]] = mapped_column(Text)
    # Per-model 30-day MAE for side-by-side comparison
    mae_ecmwf_f: Mapped[Optional[float]] = mapped_column(Float)       # ECMWF IFS
    mae_gfs_hrrr_f: Mapped[Optional[float]] = mapped_column(Float)    # GFS+HRRR blend
    mae_nws_f: Mapped[Optional[float]] = mapped_column(Float)         # NWS WFO official
    winner: Mapped[Optional[str]] = mapped_column(String(10))          # "ECMWF"|"GFS_HRRR"|"NWS"|"TIE"
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
