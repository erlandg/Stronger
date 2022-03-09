import os
from pathlib import Path
from datetime import date

DATETIME = date.today()
DATE_LIMIT: int = 6 # filter dataset to the last N months

PROJECT_ROOT = Path(os.path.abspath(__file__)).parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_ROOT = PROJECT_ROOT / "config"
SAVE_DIR = PROJECT_ROOT / "out"

WEIGHT_STEP: float = 2.5 # in kg