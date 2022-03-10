from typing import Dict, Literal, Union

from utils import brzycki, inverse_brzycki, round_to_closest
from .config import Config
from .constants import SMALLEST_WEIGHT_STEP


class Exercise(Config):
    # Exercise name.
    name: Literal["squat", "bench press", "deadlift"]
    # Current 1RM of the exercise.
    one_rm: float
    # Minimum significant weight (suggest: 50%-65% 1RM).
    min_weight: float
    # Sigma parameter describe the smoothing of the mesh grid.
    sigma: float = 2
    # Adds a buffer to all rep-weight combinations less than the estimated N rep max (N-RM).
    buffer: float = 0.01
    
    @staticmethod
    def get_rep_max(one_rm, r):
        return round_to_closest(inverse_brzycki(one_rm, r), SMALLEST_WEIGHT_STEP)
