from typing import Dict, List, Literal, Union
from numpy import ndarray

from utils import (
    brzycki,
    inverse_brzycki,
    lombardi,
    inverse_lombardi,
    oconner,
    inverse_oconner,
    wathen,
    inverse_wathen,
    round_to_closest
)
from .config import Config
from .constants import SMALLEST_WEIGHT_STEP


ONE_REP_MAX = {
    "brzycki": brzycki,
    "lombardi": lombardi,
    "oconner": oconner,
    "wathen": wathen,
}
INVERSE_MAX = {
    "brzycki": inverse_brzycki,
    "lombardi": inverse_lombardi,
    "oconner": inverse_oconner,
    "wathen": inverse_wathen,
}


class Exercise(Config):
    # Exercise name.
    name: Literal["squat", "bench press", "deadlift"]
    # Number of days a week to train
    frequency: int = 3
    # Current 1RM of the exercise.
    one_rm: float
    # Minimum significant weight (suggest: 50%-65% 1RM).
    min_weight: float
    # The proportion of 1RM you could lift on a bad day. Recommended to be conservative.
    # e.g. 85% of 1RM at 100kg:  0.85 * 100kg = 85kg  ==>  insert 0.85 here
    one_rm_low_cap: float = 0.80
    # Mean rep describe the average rep set. E.g. if mean 4, the rep/weight distribution will be centered at 4
    mean_reps: int = 4
    # (Optional: leave blank to fill with the median estimated weight - between upper and lower limit calculated based on
    # parameters one_rm and (one_rm * one_rm_low_cap), respectively). Set a mean weight at the above number of reps.
    mean_weight_at_reps: float = None
    # Sigma parameter describe the smoothing of the mesh grid.
    sigma: float = 2
    # Adds a buffer to all rep-weight combinations in a reasonable range.
    buffer: float = 0.01
    # Optimal estimator. Don't change!
    optimal_estimator: Literal["brzycki", "lombardi", "oconner", "wathen"] = None
    # Upper limit (r-RM). Don't change!
    upper_limit: List[float] = None
    # Linear dampening. Don't change
    dampening: bool = False
    # Dampening strength. Don't change
    dampening_decay: float = None
    # L1 regularisation parameter on the dampening. 0 for none. Recommended range between 0 and 1.
    l1_regularisation_param: float = 0.

    @staticmethod
    def get_estimator(estimator_string, inverse = False):
        if not inverse:
            return ONE_REP_MAX[estimator_string]
        else:
            return INVERSE_MAX[estimator_string]
    
    @staticmethod
    def get_rep_max(one_rm, r, rep_max_estimator = inverse_brzycki, step = SMALLEST_WEIGHT_STEP):
        if (step is not None) and (step > 0):
            return round_to_closest(rep_max_estimator(one_rm, r), step)
        else:
            if step < 0: print(f"Step size must be equal or greater than 0. Returned with no rounding (equal to step = 0).")
            return rep_max_estimator(one_rm, r)
