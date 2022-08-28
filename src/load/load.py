from typing import Dict, Union
import numpy as np

import config
from utils import get_yaml, round_to_closest


SCALE_MIN_WEIGHT = 0.5


def parse_exercise_configs(config_path):
    cfg = config.Exercise
    config_dict = get_yaml(config_path)
    cfgs = []
    for exercise, exercise_dict in config_dict.items():
        exercise_dict["name"] = exercise
        if ("min_weight" not in exercise_dict) or (exercise_dict["min_weight"] is None):
            exercise_dict["min_weight"] = round_to_closest(exercise_dict["one_rm"] * SCALE_MIN_WEIGHT, config.WEIGHT_STEP)
        cfgs.append(cfg(**exercise_dict))
    return tuple(cfgs)


def parse_program_config(config_path):
    cfg = config.Program
    config_dict = get_yaml(config_path)
    return cfg(**config_dict)