import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import (
    gaussian_filter,
    maximum_filter,
    minimum_filter,
    median_filter,
    percentile_filter,
    rank_filter,
    uniform_filter,
)

from main import date_filter
import config
from utils import (
    ensure_no_zeros,
    round_to_closest
)
from load import parse_config


def add_buffer(zs, weight_ax, buffer, limits=None):
    for row in range(zs.shape[0]):
        zs[row,:][weight_ax <= limits[row]] += buffer
    return zs


def count_instances(df, weight, rep, rpe=None, r_max=None):
    if r_max is not None:
        df["Reps"] = df["Reps"].clip(upper=r_max)

    df["Weight"] = round_to_closest(df["Weight"], config.WEIGHT_STEP)
    overlap = df[df["Weight"] == weight].isin(df[df["Reps"] == rep])
    if rpe is not None:
        overlap = overlap.isin(df["RPE"] == rpe)
    return overlap["Exercise Name"].sum() # Count instances of overlapping required fields "Exercise Name".


def plot_heatmap(xs, ys, zs, filter=None, save_path=None, title=None, xlabel=None, ylabel=None, cmap=None, **filter_kwargs):
    if filter is not None:
        zs = filter(zs, **filter_kwargs)
    zs = zs / zs.sum()

    fig, ax = plt.subplots()
    c = ax.pcolormesh(xs, ys, zs, cmap=cmap, vmin=0)
    fig.colorbar(c, ax = ax)
    ax.axis([xs.min(), xs.max(), ys.min(), ys.max()])

    if title is not None: ax.set_title(title)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        return fig, ax


def extract_meshgrid(exercise, exercise_df, cfg, return_df = False):
    # Finds most common variation by # days trained (not # sets).
    exercise_counts = exercise_df.groupby(["Date", "Exercise Name"])["Exercise Name"].nunique().reset_index(name="counts")["Exercise Name"].value_counts()
    print(f"{exercise} exercise counts:\n{exercise_counts}")
    main_exercise = exercise_counts.index[0]
    main_df = exercise_df[exercise_df["Exercise Name"] == main_exercise]

    min_weight = cfg.min_weight
    main_df = main_df[main_df["Weight"] >= min_weight]

    # Create heatmap
    xs, ys = np.meshgrid(
        np.arange(
            round_to_closest(min_weight, config.WEIGHT_STEP),
            round_to_closest(cfg.one_rm, config.WEIGHT_STEP) + config.WEIGHT_STEP,
            config.WEIGHT_STEP
        ),
        np.arange(1, config.MAX_REP_COUNT + 1, 1),
        sparse = True
    )
    zs = np.array([[count_instances(
        main_df, weight, rep, r_max = config.MAX_REP_COUNT
    ) for weight in xs[0,:]] for rep in ys[:,0]], dtype=np.float64)

    zs = add_buffer(
        zs,
        weight_ax = xs[0,:],
        buffer = cfg.buffer * exercise_counts.iloc[0],
        limits = cfg.get_rep_max(cfg.one_rm, ys[:,0])
    )
    return xs, ys, zs


def create_plots(exercise, xs, ys, zs, cfg, filter=None, **filter_kwargs):
    if (filter == 'gauss') or (filter == 'gaussian'):
        sigma = cfg.sigma * np.array(zs.shape)/sum(zs.shape)
        filter = lambda array: gaussian_filter(array, sigma, **filter_kwargs)
    elif (filter == 'max') or (filter == 'maximum'):
        filter = lambda array: maximum_filter(array, **filter_kwargs)
    elif (filter == 'min') or (filter == 'minimum'):
        filter = lambda array: minimum_filter(array, **filter_kwargs)
    elif filter == 'uniform':
        filter = lambda array: uniform_filter(array, **filter_kwargs)
    elif filter == 'median':
        filter = lambda array: median_filter(array, **filter_kwargs)
    elif filter == 'percentile':
        filter = lambda array: percentile_filter(array, **filter_kwargs)
    elif filter == 'rank':
        filter = lambda array: rank_filter(array, **filter_kwargs)

    plot_heatmap(
        xs,
        ys,
        zs,
        filter = filter,
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
    # Remove old data.
    file_df = date_filter(file_df).reset_index()

    exercise_dfs = {
        "Squat": file_df[file_df["Exercise Name"].str.contains("squat")],
        "Bench Press": file_df[file_df["Exercise Name"].str.contains("bench press")],
        "Deadlift": file_df[file_df["Exercise Name"].str.contains("deadlift")],
    }

    cfgs = parse_config(
        config_path,
        config.Exercise
    )

    meshes = {
        exercise: extract_meshgrid(exercise, df, cfg) for (exercise, df), cfg in zip(exercise_dfs, cfgs)
    }

    if analysis:
        for (exercise, (xs, ys, zs)), cfg in zip(meshes.items(), cfgs):
            create_plots(exercise, xs, ys, zs, cfg, filter=gaussian_filter)



if __name__ == "__main__":
    log_data()
