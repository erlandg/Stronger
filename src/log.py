from audioop import mul
from itertools import count
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

from main import date_filter
import config
from utils import (
    ensure_no_zeros,
    round_to_closest,
    inverse_lombardi,
    inverse_oconner,
    inverse_brzycki,
    inverse_wathen,
    gridshape_to_datapoints,
    get_median,
)
from load import parse_config


def column_normalise(array):
    col_sums = array.sum(0)
    for col, col_sum in zip(range(array.shape[1]), col_sums):
        array[:,col] = array[:,col] / col_sum
    return array


def add_buffer(zs, weight_ax, buffer, lower_limit=None, upper_limit=None, return_mask=False):
    for row in range(zs.shape[0]):
        if upper_limit is not None:
            max = upper_limit[row]
        else:
            max = weight_ax.max()
        if lower_limit is not None:
            min = lower_limit[row]
        else:
            min = weight_ax.min()
        zs[row,:][weight_ax <= max] += buffer
        zs[row,:][weight_ax <= min] = 0 # Remove junk volume
    if not return_mask:
        return zs
    else:
        return zs, zs != 0


def get_filter(filter, **filter_kwargs):
    if (filter == 'gauss') or (filter == 'gaussian'):
        from scipy.ndimage import gaussian_filter
        filter = lambda array: gaussian_filter(array, **filter_kwargs)
    elif (filter == 'max') or (filter == 'maximum'):
        from scipy.ndimage import maximum_filter
        filter = lambda array: maximum_filter(array, **filter_kwargs)
    elif (filter == 'min') or (filter == 'minimum'):
        from scipy.ndimage import minimum_filter
        filter = lambda array: minimum_filter(array, **filter_kwargs)
    elif filter == 'uniform':
        from scipy.ndimage import uniform_filter
        filter = lambda array: uniform_filter(array, **filter_kwargs)
    elif filter == 'median':
        from scipy.ndimage import median_filter
        filter = lambda array: median_filter(array, **filter_kwargs)
    elif filter == 'percentile':
        from scipy.ndimage import percentile_filter
        filter = lambda array: percentile_filter(array, **filter_kwargs)
    elif filter == 'rank':
        from scipy.ndimage import rank_filter
        filter = lambda array: rank_filter(array, **filter_kwargs)
    else:
        print(f"Filter {filter} not recognised.")
        return
    return filter


def count_instances(df, weight, rep, rpe=None, r_max=None):
    if r_max is not None:
        df["Reps"] = df["Reps"].clip(upper=r_max)

    df["Weight"] = round_to_closest(df["Weight"], config.WEIGHT_STEP)
    overlap = df[df["Weight"] == weight].isin(df[df["Reps"] == rep])
    if rpe is not None:
        overlap = overlap.isin(df["RPE"] == rpe)
    return overlap["Exercise Name"].sum() # Count instances of overlapping required fields "Exercise Name".


def plot_heatmap(xs, ys, zs, save_path=None, title=None, xlabel=None, ylabel=None, cmap=None, **mesh_kwargs):
    zs = zs / zs.sum()

    fig, ax = plt.subplots()
    c = ax.pcolormesh(xs, ys, zs, cmap=cmap, **mesh_kwargs)
    fig.colorbar(c, ax = ax)
    ax.axis([xs.min(), xs.max(), ys.min(), ys.max()])

    if title is not None: ax.set_title(title)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        return fig, ax


def masked_multinormal_distribution(xs, ys, mask = None, reps=None, weight=None):
    assert (reps is not None) or (weight is not None), "Either reps or weight argument must be given.."
    guess_weight, guess_reps = weight is None, reps is None

    if mask is None:
        mask = np.ones(len(ys), len(xs))

    datapoints = gridshape_to_datapoints(mask, xs, ys)
    if guess_weight:
        mean_reps = reps
        mean_weight = round_to_closest(
            get_median(mask, xs, ys, reps = mean_reps),
            config.WEIGHT_STEP
        )
    elif guess_reps:
        mean_weight = weight
        mean_reps = round_to_closest(
            get_median(mask, xs, ys, weight = mean_weight),
            config.WEIGHT_STEP
        )

    covariance = np.cov(datapoints.T)
    multinormal = multivariate_normal([mean_weight, mean_reps], covariance)

    all_datapoints = gridshape_to_datapoints(np.ones_like(mask), xs, ys)
    multi_w = multinormal.pdf(all_datapoints)
    pdf_mesh = np.array([[
        multi_w[np.bitwise_and(all_datapoints[:,0] == w, all_datapoints[:,1] == r)] for w in xs
    ] for r in ys])[:,:,0]
    pdf_mesh[~mask] = 0
    # plot_heatmap(xs, ys, pdf_mesh, save_path="test.png") # For testing
    return pdf_mesh



def extract_meshgrid(exercise, exercise_df, cfg, filter=None, return_df = False, **filter_kwargs):
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
    # # Add recorded sets
    # zs = np.array([[count_instances(
    #     main_df, weight, rep, r_max = config.MAX_REP_COUNT
    # ) for weight in xs[0,:]] for rep in ys[:,0]], dtype=np.float64)
    zs = np.zeros((ys.shape[0], xs.shape[1]))

    # Add a buffer and remove points below lower threshold
    _, mask = add_buffer(
        zs.copy(),
        weight_ax = xs[0,:],
        buffer = cfg.buffer * exercise_counts.iloc[0],
        lower_limit = cfg.get_rep_max(cfg.one_rm_low_cap * cfg.one_rm, ys[:,0], rep_max_estimator = inverse_brzycki),
        upper_limit = cfg.get_rep_max(cfg.one_rm, ys[:,0], rep_max_estimator = inverse_brzycki),
        return_mask = True
    )

    # Add gaussian distribution around mean weight @ mean reps
    zs += masked_multinormal_distribution(xs[0,:], ys[:,0], mask, reps = cfg.mean_reps, weight = cfg.mean_weight_at_reps)

    # Get and apply filter
    filter_ = get_filter(
        filter,
        sigma = cfg.sigma * np.array(zs.shape)/sum(zs.shape),
        **filter_kwargs
    )
    zs = filter_(zs)

    if not return_df:
        return xs, ys, zs
    else:
        return (xs, ys, zs), main_df    



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
    # Remove out-of-date data. Based on DATE_LIMIT in config.constants.
    file_df = date_filter(file_df).reset_index()

    exercise_dfs = {
        "Squat": file_df[file_df["Exercise Name"].str.contains("squat")],
        "Bench Press": file_df[file_df["Exercise Name"].str.contains("bench press")],
        "Deadlift": file_df[file_df["Exercise Name"].str.contains("deadlift")],
    }

    cfgs = parse_config(config_path, config.Exercise)

    meshes = {
        exercise: extract_meshgrid(exercise, df, cfg, filter="gauss") for (exercise, df), cfg in zip(exercise_dfs.items(), cfgs)
    }

    if analysis:
        for exercise, (xs, ys, zs) in meshes.items():
            plot_heatmap(
                xs,
                ys,
                zs,
                save_path = f"{str(config.SAVE_DIR / exercise)}-heatmap.png",
                title = f"{exercise} rep frequency",
                xlabel = "Weight",
                ylabel = "Reps",
                cmap = "hot",
                vmin = 0,
            )


if __name__ == "__main__":
    log_data()
