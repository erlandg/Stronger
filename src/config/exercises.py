from typing import Dict, Literal, Union

from utils import brzycki, inverse_brzycki, round_to_closest
from .config import Config
from .constants import SMALLEST_WEIGHT_STEP


class Exercise(Config):
    # Exercise name
    name: Literal["squat", "bench press", "deadlift"]
    # Current 1RM of the exercise
    one_rm: float
    # Minimum significant weight (suggest: 50%-65% 1RM)
    min_weight: float
    
    @staticmethod
    def get_rep_max(one_rm, r):
        return round_to_closest(inverse_brzycki(one_rm, r), SMALLEST_WEIGHT_STEP)[::-1]
