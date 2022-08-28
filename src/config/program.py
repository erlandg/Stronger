
from .config import Config


class Program(Config):
    # Number of training sessions per week
    n_training_sessions: int = 4
    # Minimum number of total sets for any exercise (for programming purposes)
    min_sets: int = 2
    # Duration of progressive overload (PO) (in weeks)
    progressive_overload_period: int = 4
    # Rate of increase per PO cycle, i.e. how much to increase your theoretical max per cycle (in %)
    increase_rate: float = .025
    # For scaling the sampling range such that intensity ranges overlap
    intensity_sampling_range_scaling: float = 1.0