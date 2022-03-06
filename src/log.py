import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from utils import ensure_no_zeros, round_to_closest
from load import parse_config


MAX_REP_COUNT: int = 12 # cap number of reps
TIME_LIMIT: int = 6 # in months


def count_instances(df, weight, rep, rpe=None, r_max=None):
    if r_max is not None:
        df["Reps"] = df["Reps"].clip(upper=r_max)

    df["Weight"] = round_to_closest(df["Weight"], config.WEIGHT_STEP)
    overlap = df[df["Weight"] == weight].isin(df[df["Reps"] == rep])
    if rpe is not None:
        overlap = overlap.isin(df["RPE"] == rpe)
    return overlap["Exercise Name"].sum() # Count instances of overlapping required fields "Exercise Name".


def plot_heatmap(xs, ys, zs, save_path=None, title=None, xlabel=None, ylabel=None, **kwargs):
    fig, ax = plt.subplots()
    c = ax.pcolormesh(xs, ys, zs, **kwargs)
    if title is not None: ax.set_title(title)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    ax.axis([xs.min(), xs.max(), ys.min(), ys.max()])
    fig.colorbar(c, ax = ax)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        return fig, ax


def create_plots(exercise, exercise_df, cfg = None):
    # Most common variation
    exercise_counts = exercise_df.groupby(["Date", "Exercise Name"])["Exercise Name"].nunique().reset_index(name="counts")["Exercise Name"].value_counts()
    print(f"{exercise} exercise counts:\n{exercise_counts}")
    main_exercise = exercise_counts.index[0]
    main_df = exercise_df[exercise_df["Exercise Name"] == main_exercise]

    if cfg is not None:
        min_weight = cfg.min_weight
        main_df = main_df[main_df["Weight"] >= min_weight]
    else:
        min_weight = 0

    # Create heatmap
    xs, ys = np.meshgrid(
        np.arange(
            round_to_closest(min_weight, config.WEIGHT_STEP),
            round_to_closest(main_df["Weight"].max(), config.WEIGHT_STEP) + config.WEIGHT_STEP,
            config.WEIGHT_STEP
        ),
        np.arange(0, MAX_REP_COUNT + 1, 1),
        sparse = True
    )
    zs = np.array([[count_instances(
        main_df, weight, rep, r_max = MAX_REP_COUNT
    ) for weight in xs[0,:]] for rep in ys[:,0]])
    zs = zs / zs.sum()
    plot_heatmap(
        xs,
        ys,
        zs,
        save_path = f"{str(config.SAVE_DIR / exercise)}-heatmap.png",
        title = f"{exercise} rep frequency",
        xlabel = "Weight",
        ylabel = "Reps",
        cmap = "hot",
    )



@click.command()
@click.argument(
    "filepath",
    type = click.Path(),
)
@click.option(
    "--config_path", "--config", "-c",
    default = config.CONFIG_ROOT / "config.yaml",
    type = click.Path(exists=True),
    help = "Path to config file of YAML-format corresponding to config objects found in src/config/exercise.py"
)
@click.option(
    "--analysis", "-a",
    is_flag = True,
    default = False,
    help = "Whether to graphically analyse the data."
)
def log_data(filepath, config_path, analysis):
    filepath = config.DATA_DIR / filepath
    file_df = pd.read_csv(filepath, header = 0)
    file_df["Exercise Name"] = file_df["Exercise Name"].str.lower()

    exercise_df = {
        "Squat": file_df[file_df["Exercise Name"].str.contains("squat")],
        "Bench Press": file_df[file_df["Exercise Name"].str.contains("bench press")],
        "Deadlift": file_df[file_df["Exercise Name"].str.contains("deadlift")],
    }

    if config_path is not None:
        cfgs = parse_config(
            config_path,
            config.Exercise
        )
    else:
        cfgs = [None for _ in range(len(exercise_df))]

    if analysis:
        for (exercise, df), cfg in zip(exercise_df.items(), cfgs):
            create_plots(exercise, df, cfg)
    print()



if __name__ == "__main__":
    log_data()
