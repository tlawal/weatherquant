"""
Parametric diurnal temperature curve fitting.

Fits a piecewise model (rising sinusoidal + decaying Gaussian) to today's
METAR observations, constrained by forecast high and estimated peak timing.

The fitted curve provides:
  1. Station-time temperature predictions that respect diurnal physics
  2. Peak temperature and timing estimates
  3. Confidence metrics (RMSE, number of observations used)

Physical basis:
  The daytime heating phase follows a sin^2 curve because the surface
  energy balance produces a warming rate proportional to sin(solar elevation).
  After peak temperature, the cooling phase follows a Gaussian/exponential
  decay as net radiation becomes negative and the boundary layer collapses.

References:
  - Parton & Nicholls (2012) "Parameterisation of the diurnal cycle of
    temperature" J. Appl. Meteor. Climatol. 51, 612-630.
  - Mayer & Groom (2002) "Diurnal heating rate in the surface layer"
    J. Atmos. Sci. 59, 1413-1424.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger(__name__)

# Approximate sunrise hours by month (mid-latitude US, ~33-42N)
_SUNRISE_BY_MONTH = {
    1: 7.2, 2: 6.9, 3: 6.3, 4: 5.7, 5: 5.3, 6: 5.1,
    7: 5.3, 8: 5.6, 9: 6.0, 10: 6.4, 11: 6.8, 12: 7.2,
}


@dataclass
class DiurnalFit:
    """Result of fitting a diurnal temperature curve."""
    T_min: float           # fitted morning minimum (deg)
    T_max: float           # fitted daily high (deg)
    t_sunrise: float       # sunrise hour (fractional)
    t_peak: float          # peak hour (fractional)
    tau_decay: float       # afternoon cooling time constant (hours)
    rmse: float            # fit RMSE against observations (deg)
    n_obs_used: int        # number of observations in fit
    forecast_high: Optional[float]  # the constraint used

    def predict(self, t_hours: float) -> float:
        """Predict temperature at fractional hour since midnight."""
        return _diurnal_curve(
            t_hours, self.T_min, self.T_max,
            self.t_sunrise, self.t_peak, self.tau_decay,
        )

    def predict_rate(self, t_hours: float) -> float:
        """Predict rate of temperature change (deg/hr) via numerical derivative."""
        dt = 1.0 / 60.0  # 1 minute in hours
        return (self.predict(t_hours + dt) - self.predict(t_hours - dt)) / (2 * dt)

    @property
    def is_reliable(self) -> bool:
        """True if the fit is good enough to use for predictions."""
        return self.rmse < 3.0 and self.n_obs_used >= 5


def _diurnal_curve(
    t_hours: float,
    T_min: float,
    T_max: float,
    t_sunrise: float,
    t_peak: float,
    tau_decay: float,
) -> float:
    """Piecewise diurnal temperature model.

    Rising phase (sunrise to peak):
        T_min + A * sin^2(pi * phase / 2)
        where phase = (t - t_sunrise) / (t_peak - t_sunrise)

    Falling phase (after peak):
        T_min + A * exp(-(t - t_peak)^2 / (2 * tau^2))

    Properties:
      - Rate is zero at sunrise (smooth start)
      - Rate peaks at (t_sunrise + t_peak) / 2 (mid-morning)
      - Rate returns to zero at t_peak (smooth transition)
      - Gaussian decay after peak (natural cooling)
    """
    A = T_max - T_min
    if A <= 0:
        return T_min

    if t_hours <= t_sunrise:
        return T_min
    elif t_hours <= t_peak:
        heating_window = t_peak - t_sunrise
        if heating_window <= 0:
            return T_max
        phase = (t_hours - t_sunrise) / heating_window  # 0 to 1
        return T_min + A * math.sin(math.pi * phase / 2.0) ** 2
    else:
        dt = t_hours - t_peak
        if tau_decay <= 0:
            return T_min
        return T_min + A * math.exp(-dt ** 2 / (2.0 * tau_decay ** 2))


def _obs_to_hour_frac(obs: dict, city_tz: str, dt_key: str = "observed_at") -> float:
    """Convert observation timestamp to fractional hours since midnight, local time."""
    dt = obs[dt_key]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    tz = ZoneInfo(city_tz)
    local_dt = dt.astimezone(tz)
    return local_dt.hour + local_dt.minute / 60.0 + local_dt.second / 3600.0


def fit_diurnal_curve(
    observations: list[dict],
    forecast_high: Optional[float],
    peak_mins: Optional[float] = None,
    city_tz: str = "America/New_York",
    dt_key: str = "observed_at",
    temp_key: str = "temp_f",
    forecast_weight: float = 2.0,
) -> Optional[DiurnalFit]:
    """Fit a parametric diurnal curve to today's observations.

    Args:
        observations: list of observation dicts with dt_key and temp_key
        forecast_high: NWS/WU fused daily high forecast (soft constraint)
        peak_mins: estimated peak time in minutes since midnight
        city_tz: IANA timezone string
        forecast_weight: weight of forecast_high penalty in cost function
                        (higher = trust forecast more over observations)

    Returns:
        DiurnalFit or None if insufficient data or optimization failure.
    """
    if len(observations) < 4:
        return None

    # Extract daytime observation hours and temperatures
    obs_hours = []
    obs_temps = []
    for obs in observations:
        try:
            h = _obs_to_hour_frac(obs, city_tz, dt_key)
            t = obs.get(temp_key)
            if t is not None and 5.0 <= h <= 22.0:
                obs_hours.append(h)
                obs_temps.append(float(t))
        except Exception:
            continue

    if len(obs_hours) < 4:
        return None

    obs_hours_arr = np.array(obs_hours)
    obs_temps_arr = np.array(obs_temps)

    # Initial guesses
    month = datetime.now().month
    t_sunrise_init = _SUNRISE_BY_MONTH.get(month, 6.5)
    t_peak_init = (peak_mins or 900) / 60.0
    T_min_init = float(np.min(obs_temps_arr))
    T_max_init = float(forecast_high) if forecast_high else float(np.max(obs_temps_arr)) + 3.0
    tau_init = 2.5

    # Parameter bounds
    bounds = [
        (T_min_init - 8, T_min_init + 5),          # T_min
        (float(np.max(obs_temps_arr)) - 2,          # T_max lower: near observed max
         float(np.max(obs_temps_arr)) + 20),         # T_max upper: allow room
        (12.0, 18.0),                                # t_peak
        (1.5, 4.0),                                  # tau_decay
    ]

    def cost(params):
        T_min, T_max, t_peak, tau = params
        # Observation residuals
        preds = np.array([
            _diurnal_curve(h, T_min, T_max, t_sunrise_init, t_peak, tau)
            for h in obs_hours_arr
        ])
        obs_cost = float(np.sum((obs_temps_arr - preds) ** 2))

        # Forecast high soft constraint
        fc_cost = 0.0
        if forecast_high is not None:
            fc_cost = forecast_weight * (T_max - forecast_high) ** 2

        # Peak timing soft constraint
        peak_cost = 0.0
        if peak_mins is not None:
            peak_cost = 0.5 * (t_peak - peak_mins / 60.0) ** 2

        # Regularization: prefer tau near 2.5
        reg_cost = 0.3 * (tau - 2.5) ** 2

        return obs_cost + fc_cost + peak_cost + reg_cost

    x0 = [T_min_init, T_max_init, t_peak_init, tau_init]

    try:
        result = minimize(
            cost, x0, bounds=bounds, method="L-BFGS-B",
            options={"maxiter": 100, "ftol": 1e-6},
        )
        T_min_fit, T_max_fit, t_peak_fit, tau_fit = result.x
    except Exception:
        log.debug("diurnal_model: optimization failed")
        return None

    # Compute RMSE
    preds = np.array([
        _diurnal_curve(h, T_min_fit, T_max_fit, t_sunrise_init, t_peak_fit, tau_fit)
        for h in obs_hours_arr
    ])
    rmse = float(np.sqrt(np.mean((obs_temps_arr - preds) ** 2)))

    return DiurnalFit(
        T_min=float(T_min_fit),
        T_max=float(T_max_fit),
        t_sunrise=t_sunrise_init,
        t_peak=float(t_peak_fit),
        tau_decay=float(tau_fit),
        rmse=rmse,
        n_obs_used=len(obs_hours),
        forecast_high=forecast_high,
    )
