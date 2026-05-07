"""Bayesian Model Averaging (BMA) for ensemble forecast post-processing.

Reference: Raftery, A. E., Gneiting, T., Balabdaoui, F., Polakowski, M. (2005).
*Using Bayesian Model Averaging to Calibrate Forecast Ensembles.* MWR 133:1155.

The predictive PDF for the daily-high temperature T is a finite mixture of
Gaussians, one kernel per ensemble member:

    p(T | μ_NWS, μ_HRRR, ...) = Σᵢ wᵢ · N(T | aᵢ + bᵢ·μᵢ, σᵢ²)

For Phase 1 (shadow mode) we set bᵢ = 1 and let aᵢ absorb per-source bias
(equivalent to the existing `_debias` correction; we deliberately keep the bias
adjustment in temperature_model.py so the comparison against the legacy path is
apples-to-apples). In Phase 2 we'll fit (aᵢ, bᵢ) via offline EM.

Why a mixture instead of a single Gaussian:

- Captures within-source AND between-source uncertainty:
      var(mixture) = Σwᵢσᵢ² + Σwᵢ(μᵢ - μ̄)²
                     └─ within ─┘   └── between ──┘
  Today's `_ensemble_sigma` only has the second term, so it systematically
  understates uncertainty when sources individually have large σᵢ.
- Bimodal predictive PDF on front-passage days where IFS says 79°F and HRRR
  says 85°F. Polymarket prices that bimodality; a single-Gaussian fit doesn't
  see it.

Phase 1 scope: predictive PDF, mixture mean/variance, mixture CDF/bucket
probabilities, and a constructor that assembles per-source σᵢ from
SourceLeadTimeSkill (with safe fallbacks for cold-start). NO trade decisions
change in Phase 1 — the mixture output is computed alongside the legacy
single-Gaussian path and surfaced in ModelResult for shadow comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from scipy.stats import norm


# Minimum number of residual observations before we trust a per-source σ
# estimate from SourceLeadTimeSkill. Below this we fall back to a wider prior.
BMA_MIN_N_FOR_SIGMA = 30

# Cold-start prior σ when no per-source skill data exists yet (°F).
# Roughly the climatological MAE of a free NWP forecast at 24h lead.
BMA_PRIOR_SIGMA_F = 3.0

# Hard sigma floor (°F). A degenerate σ → 0 would collapse the mixture to a
# point mass and explode the bucket integration. 0.5°F ≈ ASOS resolution.
BMA_SIGMA_FLOOR_F = 0.5


@dataclass
class BMAComponent:
    """One Gaussian kernel in the BMA mixture, corresponding to one ensemble member."""
    source: str
    mu: float          # bias-corrected forecast from this source (°F)
    sigma: float       # per-source residual std (°F), σᵢ
    weight: float      # mixture weight wᵢ; weights across all components sum to 1
    n_obs: int = 0     # number of residual obs backing the σᵢ estimate (audit only)


@dataclass
class BMAPredictive:
    """A BMA predictive distribution — finite mixture of Gaussians."""
    components: list[BMAComponent]
    # Provenance for audit/UI:
    sigma_unit_mult: float = 1.0       # 1.0 for °F, 5/9 for °C
    fallback_used: bool = False        # True if any component used the prior σ
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Normalize weights defensively. Caller may pass un-normalized.
        total = sum(c.weight for c in self.components)
        if total > 0:
            for c in self.components:
                c.weight = c.weight / total

    # ─────────────────────── Predictive PDF / CDF ────────────────────────────
    def pdf(self, t: float) -> float:
        """Mixture PDF at temperature t."""
        return sum(c.weight * float(norm.pdf(t, c.mu, c.sigma)) for c in self.components)

    def cdf(self, t: float) -> float:
        """Mixture CDF at temperature t."""
        return sum(c.weight * float(norm.cdf(t, c.mu, c.sigma)) for c in self.components)

    # ───────────────────── Summary moments ───────────────────────────────────
    @property
    def mean(self) -> float:
        """Mixture mean: μ̄ = Σ wᵢ μᵢ."""
        return sum(c.weight * c.mu for c in self.components)

    @property
    def variance(self) -> float:
        """Mixture variance: within-source + between-source.

        var(mixture) = Σwᵢσᵢ² + Σwᵢ(μᵢ - μ̄)²

        The first term is the weighted average of per-source variances (what
        each forecaster says they don't know); the second is the spread of
        forecaster means around their consensus (how much the forecasters
        disagree). Today's `_ensemble_sigma` is only the second term.
        """
        if not self.components:
            return 0.0
        mu_bar = self.mean
        within = sum(c.weight * c.sigma * c.sigma for c in self.components)
        between = sum(c.weight * (c.mu - mu_bar) ** 2 for c in self.components)
        return within + between

    @property
    def sigma(self) -> float:
        """Mixture std (sqrt of variance) — useful as a single-number summary."""
        return math.sqrt(max(self.variance, 0.0))

    @property
    def is_bimodal_indicator(self) -> float:
        """Ratio of between-source to total variance. >0.5 ⇒ disagreement
        dominates. UI uses this to flag rows where the mixture is meaningfully
        non-Gaussian and the single-Gaussian summary is misleading."""
        if not self.components:
            return 0.0
        mu_bar = self.mean
        within = sum(c.weight * c.sigma * c.sigma for c in self.components)
        between = sum(c.weight * (c.mu - mu_bar) ** 2 for c in self.components)
        total = within + between
        return between / total if total > 0 else 0.0


# ──────────────────────── Bucket integration ─────────────────────────────────

def bma_bucket_probabilities(
    predictive: BMAPredictive,
    buckets: list[tuple[Optional[float], Optional[float]]],
) -> list[float]:
    """Compute mixture bucket probabilities.

    P(T ∈ [lo, hi)) = Σᵢ wᵢ · [Φ((hi − μᵢ)/σᵢ) − Φ((lo − μᵢ)/σᵢ)]

    Drop-in shape-compatible with `distribution.bucket_probabilities` so the
    shadow path can swap in. Open buckets (lo or hi = None) handled the same
    way (−∞ / +∞ tails).
    """
    if not buckets:
        return []
    if not predictive.components:
        # Degenerate: no components → uniform over buckets, with the same
        # error semantics as the legacy single-Gaussian path expects.
        n = len(buckets)
        return [1.0 / n] * n

    probs: list[float] = []
    for lo, hi in buckets:
        prob = 0.0
        for c in predictive.components:
            lo_cdf = 0.0 if lo is None else float(norm.cdf(lo, c.mu, c.sigma))
            hi_cdf = 1.0 if hi is None else float(norm.cdf(hi, c.mu, c.sigma))
            prob += c.weight * max(0.0, hi_cdf - lo_cdf)
        probs.append(prob)

    # Normalize — should already be ~1.0 by construction; renormalize for
    # float-rounding drift, matching the legacy bucket_probabilities behavior.
    total = sum(probs)
    if total > 0 and abs(total - 1.0) < 0.05:
        probs = [p / total for p in probs]
    return probs


# ───────────────────── Predictive constructor ───────────────────────────────

def build_bma_predictive(
    calibrated_means: dict[str, float],
    weights_by_source: dict[str, float],
    lead_skill_mae_by_source: Optional[dict[str, float]] = None,
    lead_skill_n_obs_by_source: Optional[dict[str, int]] = None,
    sigma_unit_mult: float = 1.0,
) -> BMAPredictive:
    """Assemble a BMAPredictive from the same inputs `compute_model` already has.

    Args:
        calibrated_means: {source: bias-corrected daily-high forecast (°F)}.
            Bias is applied upstream by `_debias` so this matches the existing
            single-Gaussian input — keeps shadow comparison apples-to-apples.
        weights_by_source: {source: w_i}. Pre-normalization OK; weights are
            normalized in BMAPredictive.__post_init__. Caller passes the same
            base × lead-factor × freshness-factor product the legacy path uses.
        lead_skill_mae_by_source: {source: MAE °F at the relevant lead bucket}
            from SourceLeadTimeSkill. MAE is the natural σ proxy: under a
            zero-mean Laplace residual model, σ ≈ MAE · √2 / √π · √2 ≈ MAE.
            We use σᵢ = MAE directly — slightly conservative for Gaussian, but
            BMA is robust to small misspecification of within-source σ once
            mixture variance is dominated by the between-source term.
        lead_skill_n_obs_by_source: {source: n}. Sources with n < BMA_MIN_N_FOR_SIGMA
            fall back to BMA_PRIOR_SIGMA_F.
        sigma_unit_mult: 1.0 for °F outputs, 5/9 for °C. Applied to the prior σ
            and floor; MAE values are already in display units from the caller.
    """
    mae_map = lead_skill_mae_by_source or {}
    n_map = lead_skill_n_obs_by_source or {}
    components: list[BMAComponent] = []
    fallback_used = False
    notes: list[str] = []

    for src, mu in calibrated_means.items():
        w = float(weights_by_source.get(src, 0.0))
        if w <= 0.0:
            # Source is in the panel but weights downscaled it to zero — skip;
            # mixture should not include it. Edge case: all weights zero →
            # caller will land on degenerate path, but legacy path would too.
            continue

        mae = mae_map.get(src)
        n = int(n_map.get(src, 0))
        if mae is not None and n >= BMA_MIN_N_FOR_SIGMA and mae > 0:
            sigma = float(mae)
        else:
            sigma = BMA_PRIOR_SIGMA_F * sigma_unit_mult
            fallback_used = True
            if mae is None:
                notes.append(f"{src}: no SourceLeadTimeSkill row, σ=prior")
            elif n < BMA_MIN_N_FOR_SIGMA:
                notes.append(f"{src}: n={n}<{BMA_MIN_N_FOR_SIGMA}, σ=prior")

        sigma = max(sigma, BMA_SIGMA_FLOOR_F * sigma_unit_mult)
        components.append(BMAComponent(
            source=src, mu=float(mu), sigma=sigma, weight=w, n_obs=n,
        ))

    return BMAPredictive(
        components=components,
        sigma_unit_mult=sigma_unit_mult,
        fallback_used=fallback_used,
        notes=notes,
    )


# ───────────────────── Audit / dict export ──────────────────────────────────

def predictive_to_dict(p: BMAPredictive) -> dict:
    """Serialize a BMAPredictive for ModelResult.inputs and UI rendering."""
    return {
        "mean": round(p.mean, 3) if p.components else None,
        "sigma": round(p.sigma, 3) if p.components else None,
        "variance": round(p.variance, 3) if p.components else None,
        "between_share": round(p.is_bimodal_indicator, 3) if p.components else None,
        "fallback_used": p.fallback_used,
        "n_components": len(p.components),
        "components": [
            {
                "source": c.source,
                "mu": round(c.mu, 2),
                "sigma": round(c.sigma, 3),
                "weight": round(c.weight, 4),
                "n_obs": c.n_obs,
            }
            for c in p.components
        ],
        "notes": list(p.notes),
    }
