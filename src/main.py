import click
from pathlib import PosixPath, Path

import config
from load import parse_config
from utils import get_yaml


@click.command()
@click.option(
    "--config_path", "--config", "-c",
    default = config.CONFIG_ROOT / "config.yaml",
    type = click.Path(exists=True),
    help = "Path to config file of YAML-format corresponding to config objects found in src/config/exercise.py"
)
def main(config_path):
    squat_cfg, bench_cfg, deadlift_cfg = parse_config(
        config_path,
        {
            "squat": config.Squat,
            "bench press": config.BenchPress,
            "deadlift": config.Deadlift
        }
    )
    print()



if __name__ == "__main__":
    main()
