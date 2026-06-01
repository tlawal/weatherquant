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


# Minimum number of residual observations before we fully trust a per-source σ
# estimate from SourceLeadTimeSkill. Below this we shrink the observed MAE
# toward the wider prior instead of discarding it entirely.
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


def bma_conditional_bucket_probabilities(
    predictive: BMAPredictive,
    buckets: list[tuple[Optional[float], Optional[float]]],
    floor: float,
) -> list[float]:
    """Compute mixture P(T in bucket | T >= floor).

    Same observed-high semantics as
    `distribution.conditional_bucket_probabilities`, but integrating the BMA
    mixture instead of a single Gaussian. This matters intraday: once the
    settlement high has reached `floor`, BMA shadow probabilities must not
    continue displaying unconditional low-tail mass.
    """
    if not buckets:
        return []
    if not predictive.components:
        probs = [
            0.0 if hi is not None and floor >= hi else 1.0
            for _, hi in buckets
        ]
        total = sum(probs)
        return [p / total for p in probs] if total > 0 else probs

    probs: list[float] = []
    for lo, hi in buckets:
        if hi is not None and floor >= hi:
            probs.append(0.0)
            continue

        effective_lo = max(lo, floor) if lo is not None else floor
        prob = 0.0
        for c in predictive.components:
            lo_cdf = float(norm.cdf(effective_lo, c.mu, c.sigma))
            hi_cdf = 1.0 if hi is None else float(norm.cdf(hi, c.mu, c.sigma))
            prob += c.weight * max(0.0, hi_cdf - lo_cdf)
        probs.append(prob)

    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    return probs


# ───────────────────── Predictive constructor ───────────────────────────────

def build_bma_predictive(
    calibrated_means: dict[str, float],
    weights_by_source: dict[str, float],
    lead_skill_mae_by_source: Optional[dict[str, float]] = None,
    lead_skill_n_obs_by_source: Optional[dict[str, int]] = None,
    sigma_unit_mult: float = 1.0,
    fitted_weights_by_source: Optional[dict[str, float]] = None,
) -> BMAPredictive:
    """Assemble a BMAPredictive from the same inputs `compute_model` already has.

    Args:
        calibrated_means: {source: bias-corrected daily-high forecast (°F)}.
            Bias is applied upstream by `_debias` so this matches the existing
            single-Gaussian input — keeps shadow comparison apples-to-apples.
        weights_by_source: {source: w_i}. Pre-normalization OK; weights are
            normalized in BMAPredictive.__post_init__. Caller passes the same
            base × lead-factor × freshness-factor product the legacy path uses.
            Used when `fitted_weights_by_source` is None or a source is missing
            from the fitted set (graceful fallback).
        lead_skill_mae_by_source: {source: MAE °F at the relevant lead bucket}
            from SourceLeadTimeSkill. MAE is the natural σ proxy: under a
            zero-mean Laplace residual model, σ ≈ MAE · √2 / √π · √2 ≈ MAE.
            We use σᵢ = MAE directly — slightly conservative for Gaussian, but
            BMA is robust to small misspecification of within-source σ once
            mixture variance is dominated by the between-source term.
        lead_skill_n_obs_by_source: {source: n}. Sources with 0 < n <
            BMA_MIN_N_FOR_SIGMA use empirical-Bayes shrinkage toward
            BMA_PRIOR_SIGMA_F; n >= BMA_MIN_N_FOR_SIGMA uses MAE directly.
        sigma_unit_mult: 1.0 for °F outputs, 5/9 for °C. Applied to the prior σ
            and floor; MAE values are already in display units from the caller.
        fitted_weights_by_source: optional EM-fit weights from BMAWeights
            (M1 Phase 2). When present, takes precedence over `weights_by_source`
            for sources in the fitted set; sources missing from the fitted set
            fall back to `weights_by_source` so a brand-new source isn't
            dropped from the mixture mid-deploy. None = use legacy weights only.
    """
    mae_map = lead_skill_mae_by_source or {}
    n_map = lead_skill_n_obs_by_source or {}
    fitted = fitted_weights_by_source or {}
    components: list[BMAComponent] = []
    fallback_used = False
    notes: list[str] = []
    fitted_used = False

    for src, mu in calibrated_means.items():
        # Prefer the fitted EM weight; fall back to the legacy lead-skill ×
        # freshness weight when the source isn't in the fitted set yet.
        if src in fitted:
            w = float(fitted[src])
            fitted_used = True
        else:
            w = float(weights_by_source.get(src, 0.0))
        if w <= 0.0:
            # Source is in the panel but weights downscaled it to zero — skip;
            # mixture should not include it. Edge case: all weights zero →
            # caller will land on degenerate path, but legacy path would too.
            continue

        mae = mae_map.get(src)
        n = int(n_map.get(src, 0))
        if mae is not None and mae > 0 and n >= BMA_MIN_N_FOR_SIGMA:
            sigma = float(mae)
        elif mae is not None and mae > 0 and n > 0:
            prior_sigma = BMA_PRIOR_SIGMA_F * sigma_unit_mult
            confidence = min(1.0, n / BMA_MIN_N_FOR_SIGMA)
            sigma = (1.0 - confidence) * prior_sigma + confidence * float(mae)
            fallback_used = True
            notes.append(
                f"{src}: n={n}<{BMA_MIN_N_FOR_SIGMA}, "
                f"σ=shrinkage({sigma:.2f}; prior={prior_sigma:.2f}, mae={float(mae):.2f})"
            )
        else:
            sigma = BMA_PRIOR_SIGMA_F * sigma_unit_mult
            fallback_used = True
            if mae is None:
                notes.append(f"{src}: no SourceLeadTimeSkill row, σ=prior")
            elif n <= 0:
                notes.append(f"{src}: n=0, σ=prior")
            elif n < BMA_MIN_N_FOR_SIGMA:
                notes.append(f"{src}: n={n}<{BMA_MIN_N_FOR_SIGMA}, σ=prior")

        sigma = max(sigma, BMA_SIGMA_FLOOR_F * sigma_unit_mult)
        components.append(BMAComponent(
            source=src, mu=float(mu), sigma=sigma, weight=w, n_obs=n,
        ))

    if fitted_used:
        notes.append("M1 Phase 2: EM-fitted weights in use")
    return BMAPredictive(
        components=components,
        sigma_unit_mult=sigma_unit_mult,
        fallback_used=fallback_used,
        notes=notes,
    )


# ───────────────────── Offline EM weight fitter (Phase 2) ──────────────────
#
# Reference: Raftery, Gneiting, Balabdaoui, Polakowski (2005) §3.
# Given training data (per-source forecast means, observed value), the EM
# algorithm finds the mixing weights wₖ that maximize the likelihood under:
#
#     y_i ~ Σₖ wₖ · N(μ_ik, σₖ²)
#
# We hold σₖ fixed at the SourceLeadTimeSkill MAE so Phase 2 only fits weights
# (Phase 3 will update σ online via NIG conjugates per Section 6 Layer 2). The
# E-step computes responsibilities; the M-step replaces wₖ with the average
# responsibility across observations. Iterates until log-likelihood plateaus.

EM_MAX_ITER_DEFAULT = 200
EM_LL_TOL_DEFAULT = 1e-6
EM_MIN_TRAINING_OBS = 30      # below this, return uniform weights (cold start)
EM_WEIGHT_FLOOR = 1e-4         # don't let any source go to zero — trapped sources
                               # can't recover when new evidence favors them


@dataclass
class BMAFitResult:
    weights: dict[str, float]    # fitted wₖ (sum to 1)
    log_likelihood: float        # final log-likelihood
    n_iter: int                  # iterations used
    converged: bool              # True if Δll < tol before max_iter
    n_obs: int                   # training set size
    sources: list[str]           # ordered key list (for stable serialization)


def _gaussian_pdf(y: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    z = (y - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def _log_likelihood(
    training: list[tuple[dict[str, float], float]],
    weights: dict[str, float],
    sigma_by_source: dict[str, float],
    sources: list[str],
) -> float:
    """Σ_i log Σₖ wₖ · N(y_i | μ_ik, σₖ).

    Floors per-obs likelihood at 1e-300 to keep log finite when a single
    forecast is wildly off (e.g. 30°F miss). Such observations contribute a
    near-constant penalty to LL so they don't dominate convergence.
    """
    ll = 0.0
    for forecasts, y in training:
        per_obs = 0.0
        for src in sources:
            if src not in forecasts:
                continue
            per_obs += weights[src] * _gaussian_pdf(y, forecasts[src], sigma_by_source[src])
        ll += math.log(max(per_obs, 1e-300))
    return ll


def fit_bma_weights_em(
    training: list[tuple[dict[str, float], float]],
    sigma_by_source: dict[str, float],
    init_weights: Optional[dict[str, float]] = None,
    max_iter: int = EM_MAX_ITER_DEFAULT,
    tol: float = EM_LL_TOL_DEFAULT,
) -> BMAFitResult:
    """Fit BMA mixture weights via EM (Raftery et al. 2005).

    Args:
        training: list of `(forecasts, observed_y)` pairs. `forecasts` is a
            dict {source: μ_ik (°F)}; sources may be missing on some obs (e.g.
            HRRR didn't run on day 5) and the algorithm handles that — those
            obs contribute zero responsibility to the absent source's weight.
        sigma_by_source: {source: σₖ (°F)}, held fixed during the fit. Use
            SourceLeadTimeSkill MAE per source × lead-bucket. Sources missing
            from this map default to BMA_PRIOR_SIGMA_F.
        init_weights: starting wₖ. Defaults to uniform 1/K. Must sum > 0.
        max_iter, tol: convergence controls.

    Returns:
        BMAFitResult with fitted weights (sum to 1), final log-likelihood,
        iteration count, and convergence flag.

    Cold-start guard: if `len(training) < EM_MIN_TRAINING_OBS`, returns
    uniform weights without iterating. Caller decides whether to use the
    result based on `n_obs`.
    """
    # Discover the union of sources across training + sigma_by_source. Sources
    # in only one but not the other are problematic; we exclude any source
    # missing a σ since the E-step would crash.
    all_sources = set(sigma_by_source.keys())
    for forecasts, _ in training:
        all_sources.update(forecasts.keys())
    sources = sorted(s for s in all_sources if s in sigma_by_source)

    if not sources:
        return BMAFitResult(
            weights={}, log_likelihood=float("-inf"),
            n_iter=0, converged=False, n_obs=len(training), sources=[],
        )

    # Cold start
    if len(training) < EM_MIN_TRAINING_OBS:
        uniform = {s: 1.0 / len(sources) for s in sources}
        ll = _log_likelihood(training, uniform, sigma_by_source, sources) if training else float("-inf")
        return BMAFitResult(
            weights=uniform, log_likelihood=ll, n_iter=0,
            converged=False, n_obs=len(training), sources=sources,
        )

    # Initialize weights
    if init_weights:
        total = sum(init_weights.get(s, 0.0) for s in sources)
        if total <= 0:
            weights = {s: 1.0 / len(sources) for s in sources}
        else:
            weights = {s: init_weights.get(s, 0.0) / total for s in sources}
    else:
        weights = {s: 1.0 / len(sources) for s in sources}

    prev_ll = float("-inf")
    converged = False
    n_iter = 0

    n_total = len(training)

    for n_iter in range(1, max_iter + 1):
        # ── E-step ──────────────────────────────────────────────────────
        # responsibility_sum[k] accumulates Σ_i r_ik over ALL training obs.
        # Sources missing on a given obs contribute 0 there — that's the
        # availability-aware behavior we want: a source that misses 25% of
        # days gets ~25% less weight automatically, and at fuse time the
        # downstream renormalization among available sources still produces
        # sensible probabilities.
        responsibility_sum = {s: 0.0 for s in sources}
        for forecasts, y in training:
            # Per-source numerator wₖ · N(y | μ_ik, σₖ)
            numerator = {}
            denom = 0.0
            for s in sources:
                if s not in forecasts:
                    continue
                pdf = _gaussian_pdf(y, forecasts[s], sigma_by_source[s])
                contribution = weights[s] * pdf
                numerator[s] = contribution
                denom += contribution
            if denom <= 0:
                # Pathological obs — every forecast far enough from y that the
                # mixture density underflows. Skip; these contribute nothing
                # to the M-step but stay in the LL computation as a small
                # constant penalty.
                continue
            for s, num in numerator.items():
                responsibility_sum[s] += num / denom

        # ── M-step ──────────────────────────────────────────────────────
        # Standard Raftery 2005: wₖ = (1/N) Σ_i r_ik.
        new_weights: dict[str, float] = {
            s: responsibility_sum[s] / n_total for s in sources
        }

        # Normalize first (handles obs we skipped above where every source's
        # density underflowed — they'd leave new_weights summing < 1).
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {s: w / total for s, w in new_weights.items()}

        # Apply floor with redistribution: clamp tiny weights to EM_WEIGHT_FLOOR
        # and subtract the deficit from the largest weight. This preserves
        # sum=1 exactly (no second normalization needed) and guarantees the
        # floor invariant — protecting trapped sources without slipping back
        # below threshold via float rounding. The largest weight always has
        # enough headroom to absorb at most (K−1) × floor of deficit.
        floor_deficit = 0.0
        for s, w in new_weights.items():
            if w < EM_WEIGHT_FLOOR:
                floor_deficit += (EM_WEIGHT_FLOOR - w)
                new_weights[s] = EM_WEIGHT_FLOOR
        if floor_deficit > 0:
            largest = max(new_weights, key=new_weights.get)
            new_weights[largest] -= floor_deficit

        # ── Convergence ────────────────────────────────────────────────
        ll = _log_likelihood(training, new_weights, sigma_by_source, sources)
        if abs(ll - prev_ll) < tol:
            weights = new_weights
            converged = True
            break
        weights = new_weights
        prev_ll = ll

    return BMAFitResult(
        weights=weights,
        log_likelihood=ll,
        n_iter=n_iter,
        converged=converged,
        n_obs=len(training),
        sources=sources,
    )


# ───────────────────── Online EM weight update (Phase 3) ───────────────────
#
# Reference: Cappé, O. & Moulines, E. (2009). On-line expectation–maximisation
# algorithm for latent data models. JRSS B 71(3):593–613.
#
# The offline EM in `fit_bma_weights_em` reprocesses the entire training
# window each night. That's robust but slow to react: a regime shift today
# only moves weights tomorrow. Online EM applies one stochastic E+M step per
# new observation, using a small learning rate `lr` to keep the trajectory
# stable. With lr ≈ 0.05 and ~50 settled events flowing in per month, the
# online updater closes ~30% of the gap to the new equilibrium between
# nightly batch refits — fast enough to catch front-passage clusters while
# slow enough that one anomalous resolution can't crash the weights.
#
# Update rule (single new observation `(forecasts, y)`):
#
#     E:  rₖ = wₖ · N(y | μ_k, σ_k²)  /  Σⱼ wⱼ · N(y | μ_j, σ_j²)
#     M:  wₖ_new = (1 − lr) · wₖ + lr · rₖ                   for sources present
#         wₖ_new = wₖ                                          for sources absent
#
# Sources absent from this observation aren't updated — the offline batch
# refit (every 24h) handles availability penalties via the standard Raftery
# /N normalization. Online and offline thus play complementary roles:
# online = fast adjuster within the day; offline = correct anchor.

ONLINE_EM_LR_DEFAULT = 0.05


def online_em_step(
    current_weights: dict[str, float],
    forecasts: dict[str, float],
    observed_y: float,
    sigma_by_source: dict[str, float],
    lr: float = ONLINE_EM_LR_DEFAULT,
) -> dict[str, float]:
    """Apply one online-EM update to BMA mixture weights given a new settled
    observation.

    Args:
        current_weights: {source: wₖ} from the most recent offline fit (or
            from a prior online step). Must sum > 0; renormalized at end so
            the input doesn't have to be exactly normalized.
        forecasts: {source: μ_ik} for this newly-settled event. Sources may
            be missing (e.g. HRRR didn't run that day); missing sources
            are left unchanged.
        observed_y: realized daily-high temperature for this event (°F).
        sigma_by_source: {source: σₖ} held fixed (matches offline-EM Phase 2).
        lr: learning rate. Smaller = slower to react, more stable.

    Returns:
        Updated {source: weight} dict, summing to 1, all entries above
        EM_WEIGHT_FLOOR. If the per-obs density underflows (every forecast
        far from y), returns `current_weights` unchanged.
    """
    # E-step on this single observation
    numerator: dict[str, float] = {}
    denom = 0.0
    for s, w in current_weights.items():
        if s not in forecasts or s not in sigma_by_source:
            continue
        pdf = _gaussian_pdf(observed_y, forecasts[s], sigma_by_source[s])
        contribution = w * pdf
        numerator[s] = contribution
        denom += contribution

    if denom <= 0:
        # Pathological: every forecast was so far from y that the mixture
        # density underflowed. No-op rather than corrupt the weights with a
        # division-by-zero or with an arbitrary fallback.
        return dict(current_weights)

    # M-step: nudge each present source toward its responsibility; leave
    # absent sources untouched. lr ≈ 0.05 keeps any single observation from
    # dominating, even ones with extreme PDF ratios.
    new_weights: dict[str, float] = {}
    for s, w in current_weights.items():
        if s in numerator:
            r = numerator[s] / denom
            new_weights[s] = (1.0 - lr) * w + lr * r
        else:
            new_weights[s] = w

    # Floor with redistribution (same algorithm as offline-EM M-step) so the
    # weight invariants from Phase 2 hold at every intermediate online step.
    total = sum(new_weights.values())
    if total > 0:
        new_weights = {s: w / total for s, w in new_weights.items()}

    floor_deficit = 0.0
    for s, w in new_weights.items():
        if w < EM_WEIGHT_FLOOR:
            floor_deficit += (EM_WEIGHT_FLOOR - w)
            new_weights[s] = EM_WEIGHT_FLOOR
    if floor_deficit > 0:
        largest = max(new_weights, key=new_weights.get)
        new_weights[largest] -= floor_deficit

    return new_weights


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
