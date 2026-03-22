"""
Kelly Criterion sizing engine for WeatherQuant.
Calculates optimal position size based on model probability (p), market odds (b), and fractional Kelly constraints.
"""
import logging
from typing import Optional

log = logging.getLogger(__name__)

def calculate_kelly_fraction(
    model_prob: float,
    yes_price: float,
    fractional_kelly: float = 0.1,
    max_position_size: float = 0.25,
) -> float:
    """
    Calculate the Kelly fraction for a binary outcome.
    
    Formula: f* = (b*p - q) / b
    where:
      f* is the fraction of the account to wager
      p is the probability of winning (model_prob)
      q is the probability of losing (1 - p)
      b is the net odds received on the wager (odds are (1 / yes_price) - 1)
      
    Simplified: f* = p - (1-p) / ((1/yes_price) - 1)
    """
    if not (0 < yes_price < 1):
        return 0.0
    
    if model_prob <= 0:
        return 0.0

    # Net odds b
    # If price is 0.50, odds are (1/0.50) - 1 = 1 (even money)
    # If price is 0.25, odds are (1/0.25) - 1 = 3 (3-to-1)
    b = (1.0 / yes_price) - 1.0
    
    if b <= 0:
        return 0.0

    q = 1.0 - model_prob
    
    # Raw Kelly fraction
    f_star = (b * model_prob - q) / b
    
    if f_star <= 0:
        return 0.0
        
    # Apply fractional Kelly (e.g., 0.1 for 1/10th Kelly) to be conservative
    suggested_fraction = f_star * fractional_kelly
    
    # Cap at absolute maximum position size
    final_fraction = min(suggested_fraction, max_position_size)
    
    return round(final_fraction, 4)

def calculate_expected_value(model_prob: float, yes_price: float) -> float:
    """
    Calculate simple EV: (model_p * (1 - market_p)) - ((1 - model_p) * market_p)
    where market_p is the yes_price.
    """
    if not (0 < yes_price < 1):
        return 0.0
    
    # EV = (WinProb * WinAmt) - (LossProb * LossAmt)
    # If we bet $1 at price $P, we win $(1/P - 1) if right, lose $1 if wrong.
    # EV per $1 wagered = (p * (1/P - 1)) - ((1-p) * 1)
    ev = (model_prob * ((1.0 / yes_price) - 1.0)) - (1.0 - model_prob)
    return round(ev, 4)
