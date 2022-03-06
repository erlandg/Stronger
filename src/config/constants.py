import os
from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(__file__)).parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_ROOT = PROJECT_ROOT / "config"
SAVE_DIR = PROJECT_ROOT / "out"

WEIGHT_STEP: float = 2.5 # in kg