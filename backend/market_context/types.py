"""
Typed contracts for Market Context analytics and LLM output validation.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


SECTION_ORDER: list[tuple[str, str]] = [
    ("current_observations", "1. Current Observations:"),
    ("short_range_model_landscape", "2. Short-Range Model Landscape:"),
    ("historical_climatology_perspective", "3. Historical & Climatology Perspective:"),
    ("market_pricing_analysis", "4. Market Pricing Analysis:"),
    ("diagnostic_reasoning", "5. Diagnostic Reasoning:"),
    ("final_high_stakes_selection", "6. Final, High-Stakes Selection:"),
    ("independent_assessment", "7. Independent Assessment:"),
]
SECTION_KEYS = [key for key, _ in SECTION_ORDER]
SECTION_LABELS = {key: label for key, label in SECTION_ORDER}

# Sections that are allowed but not required (for backward compat with older snapshots)
OPTIONAL_SECTION_KEYS = {"independent_assessment"}


class MarketContextSelection(BaseModel):
    bucket_id: int
    bucket_idx: int
    label: str
    low_f: Optional[float] = None
    high_f: Optional[float] = None
    calibrated_prob: float = Field(ge=0.0, le=1.0)
    raw_model_prob: float = Field(ge=0.0, le=1.0)
    market_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    true_edge: Optional[float] = None
    confidence_pct: int = Field(ge=0, le=100)
    rationale: str = Field(min_length=1)
    flip_signals: list[str] = Field(default_factory=list)
    life_or_death_call: str = Field(min_length=1)
    most_likely_peak_time: Optional[str] = None
    confidence_components: dict[str, float] = Field(default_factory=dict)

    @field_validator("flip_signals")
    @classmethod
    def _validate_flip_signals(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("flip_signals must include at least one concrete trigger")
        return cleaned


class MarketContextInput(BaseModel):
    city_slug: str
    city_display: str
    date_et: str
    unit: str = "F"
    availability: dict[str, bool] = Field(default_factory=dict)
    current_observations: dict[str, Any] = Field(default_factory=dict)
    short_range_models: dict[str, Any] = Field(default_factory=dict)
    historical_context: dict[str, Any] = Field(default_factory=dict)
    market_pricing: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    final_selection: MarketContextSelection


class MarketContextOutput(BaseModel):
    sections: dict[str, str]
    final_selection: MarketContextSelection

    @field_validator("sections")
    @classmethod
    def _validate_sections(cls, value: dict[str, str]) -> dict[str, str]:
        required = [key for key in SECTION_KEYS if key not in OPTIONAL_SECTION_KEYS]
        missing = [key for key in required if not value.get(key)]
        if missing:
            raise ValueError(f"Missing required Market Context sections: {', '.join(missing)}")
        extra = [key for key in value.keys() if key not in SECTION_KEYS]
        if extra:
            raise ValueError(f"Unexpected Market Context sections: {', '.join(extra)}")

        cleaned: dict[str, str] = {}
        for key in SECTION_KEYS:
            text = (value.get(key) or "").strip()
            if text:
                cleaned[key] = text
            elif key not in OPTIONAL_SECTION_KEYS:
                raise ValueError(f"Section {key} must not be blank")
        return cleaned
