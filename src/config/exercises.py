from typing import Dict, Literal, Union

from utils import (
    brzycki,
    inverse_brzycki,
    inverse_lombardi,
    inverse_oconner,
    round_to_closest
)
from .config import Config
from .constants import SMALLEST_WEIGHT_STEP


class Exercise(Config):
    # Exercise name.
    name: Literal["squat", "bench press", "deadlift"]
    # Current 1RM of the exercise.
    one_rm: float
    # Minimum significant weight (suggest: 50%-65% 1RM).
    min_weight: float
    # The proportion of 1RM you could lift on a bad day. Recommended to be conservative.
    # e.g. 85% of 1RM at 100kg:  0.85 * 100kg = 85kg  ==>  insert 0.85 here
    one_rm_low_cap: float = 0.85
    # Mean rep describe the average rep set. E.g. if mean 4, the rep/weight distribution will be centered at 4
    mean_reps: int = 4
    # (Optional: leave blank to fill with the median estimated weight - between upper and lower limit calculated based on
    # parameters one_rm and (one_rm * one_rm_low_cap), respectively). Set a mean weight at the above number of reps.
    mean_weight_at_reps: float = None
    # Sigma parameter describe the smoothing of the mesh grid.
    sigma: float = 2
    # Adds a buffer to all rep-weight combinations in a reasonable range.
    buffer: float = 0.01
    
    @staticmethod
    def get_rep_max(one_rm, r, rep_max_estimator = inverse_brzycki):
        return round_to_closest(rep_max_estimator(one_rm, r), SMALLEST_WEIGHT_STEP)
