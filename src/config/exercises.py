from typing import Literal, Union

from .config import Config


class Exercise(Config):
    # Exercise name
    name: Literal["squat", "bench press", "deadlift"]
    # Current 1RM of the exercise
    one_rm: float
    # Minimum significant weight (suggest: 50%-65% 1RM)
    min_weight: float
    
