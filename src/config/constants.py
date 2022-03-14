import os
from pathlib import Path
from datetime import date
from typing import Literal

WEIGHT_UNIT: Literal["kg", "lbs"] = "kg"

MAX_REP_COUNT: int = 12 # cap number of reps
SMALLEST_WEIGHT_STEP: float = 0.5 # The smallest weight step allowed
WEIGHT_STEP: float = 2.5 # in kg

MESHGRID_SIGMA_MULTIPLIER: float = 5. # Decides the blurring on the mesh grid.

DATETIME = date.today()
DATE_LIMIT: int = 6 # filter dataset to the last N months

PROJECT_ROOT = Path(os.path.abspath(__file__)).parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_ROOT = PROJECT_ROOT / "config"
SAVE_DIR = PROJECT_ROOT / "out"
