from audioop import mul
from itertools import count
import click
from defer import return_value
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

from datetime import timedelta
from dateutil.relativedelta import relativedelta
import config
from utils import (
    ensure_no_zeros,
    round_to_closest,
    inverse_lombardi,
    inverse_oconner,
    inverse_brzycki,
    inverse_wathen,
    mse,
    gridshape_to_datapoints,
    get_median,
    apply_dampening,
)
from training_program import TrainingProgram
from load import parse_exercise_configs, parse_program_config



def date_filter(df):
    dates = df["Date"]
    date_lim = str(config.DATETIME + relativedelta(months=-config.DATE_LIMIT))
    return df[dates > date_lim]



def column_normalise(array):
    col_sums = array.sum(0)
    for col, col_sum in zip(range(array.shape[1]), col_sums):
        array[:,col] = array[:,col] / col_sum
    return array



def add_buffer(mesh, weight_ax, buffer, lower_limit=None, upper_limit=None, return_mask=False):
    for row in range(mesh.shape[0]):
        if upper_limit is not None:
            max = upper_limit[row]
        else:
            max = weight_ax.max()
        if lower_limit is not None:
            min = lower_limit[row]
        else:
            min = weight_ax.min()
        mesh[row,:][weight_ax <= max] += buffer
        mesh[row,:][weight_ax <= min] = 0 # Remove junk volume
    if not return_mask:
        return mesh
    else:
        return mesh, mesh != 0



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



def fit_estimator(cfg, recorded_sets, weights, reps, cost = "mse", l1_dampening_param = 0):
    assert recorded_sets.shape == (reps.shape[0], weights.shape[0]), "Incorrect shapes"
    if cost == "mse":
        cost_func = mse
    else:
        print(f"Cost function {cost} not recognised")

    maxes = np.zeros(reps.shape)
    for i, rep_count in enumerate(recorded_sets):
        if np.sum(rep_count) == 0:
            maxes[i] = np.nan
        else:
            max_weight_idx = np.where(rep_count != 0)[0].max()
            maxes[i] = weights[max_weight_idx]

    estimator_loss = {estimator: [0, 1] for estimator in config.ONE_REP_MAX.keys()}
    for estimator in estimator_loss.keys():
        estimated_rep_maxes = cfg.get_rep_max(cfg.one_rm, reps, rep_max_estimator = config.INVERSE_MAX[estimator])

        if cfg.dampening:
            best_loss = np.inf
            for dampening in np.linspace(0, .025, 26):
                loss = 0

                for i, max in enumerate(maxes):
                    if np.isnan(max):
                        continue
                    rep_max = round_to_closest(estimated_rep_maxes[i] * (1 - dampening)**i, config.SMALLEST_WEIGHT_STEP)
                    loss += cost_func(max/cfg.one_rm, rep_max/cfg.one_rm) + l1_dampening_param * dampening
                
                if loss < best_loss:
                    best_loss = loss
                    best_dampening = dampening
        else:
            best_loss = 0
            best_dampening = None
            for i, max in enumerate(maxes):
                if np.isnan(max):
                    continue
                rep_max = estimated_rep_maxes[i]
                best_loss += cost_func(max/cfg.one_rm, rep_max/cfg.one_rm)

        estimator_loss[estimator] = [best_loss, best_dampening]


    optimal_estimator = min(estimator_loss, key=estimator_loss.get)
    return optimal_estimator, estimator_loss[optimal_estimator][-1]



def crop(mesh, weight_ax, lower_limit = None, upper_limit = None):
    assert mesh.shape == (lower_limit.shape[0], weight_ax.shape[0]), f"Shapes do not match"
    for row in range(mesh.shape[0]):
        if upper_limit is not None:
            max = upper_limit[row]
        else:
            max = weight_ax.max()
        if lower_limit is not None:
            min = lower_limit[row]
        else:
            min = weight_ax.min()
        mesh[row,:][weight_ax > max] = 0
        mesh[row,:][weight_ax <= min] = 0 # Remove junk volume
    return mesh



def get_upper_limit(mesh, weight_ax, upper_limit):
    updated_limit = np.zeros_like(upper_limit)
    for i, (r_rm, obs) in enumerate(zip(upper_limit, mesh)): # Ascending
        if not obs.any():
            new_max = r_rm
        else:
            max_weight_idx = np.where(obs != 0)[0].max()
            new_max = max(weight_ax[max_weight_idx], r_rm)

        if (updated_limit[:i] < new_max).any():
            # If new r-RM max is higher than any (l < r) l-RM, increase l-RM to r-RM.
            updated_limit[:i][updated_limit[:i] < new_max] = new_max

        updated_limit[i] = new_max
    return updated_limit



def count_instances(df, weight, rep, rpe=None, r_max=None):
    if r_max is not None:
        df["Reps"] = df["Reps"].clip(upper=r_max)

    df["Weight"] = round_to_closest(df["Weight"], config.WEIGHT_STEP)
    overlap = df[df["Weight"] == weight].isin(df[df["Reps"] == rep])
    if rpe is not None:
        overlap = overlap.isin(df["RPE"] == rpe)
    return overlap["Exercise Name"].sum() # Count instances of overlapping required fields "Exercise Name".



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
    else:
        mean_weight, mean_reps = weight, reps
    print(f"Estimated mean set: {mean_reps} at {mean_weight}{config.WEIGHT_UNIT}")
    print(f"""If this estimate is too high (or low):
- adjust parameter 'one_rm_low_cap' down (or up),
- or set your desired weight at {mean_reps} reps with 'mean_weight_at_reps'""")

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
    print(f"{50*'*'}\nAnalysing {exercise}:")
    # Finds most common variation by # days trained (not # sets).
    exercise_counts = exercise_df.groupby(["Date", "Exercise Name"])["Exercise Name"].nunique().reset_index(name="counts")["Exercise Name"].value_counts()
    # print(f"{exercise} exercise counts:\n{exercise_counts}")
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
    # Add recorded sets
    recorded_sets = np.array([[count_instances(
        main_df, weight, rep, r_max = config.MAX_REP_COUNT
    ) for weight in xs[0,:]] for rep in ys[:,0]], dtype=np.float64)
    recorded_sets = crop(
        recorded_sets,
        xs[0,:],
        lower_limit = cfg.get_rep_max(cfg.one_rm_low_cap * cfg.one_rm, ys[:,0], rep_max_estimator = inverse_brzycki),
    )

    zs = np.zeros((ys.shape[0], xs.shape[1]))
    
    # Find the optimal 1RM estimator
    if cfg.optimal_estimator is None:
        estimator_string, dampening_decay = fit_estimator(
            cfg,
            recorded_sets,
            xs[0,:],
            ys[:,0],
            cost = "mse",
            l1_dampening_param = cfg.l1_regularisation_param
        )
        setattr(cfg, "optimal_estimator", estimator_string)
        if not cfg.dampening:
            dampening_decay = 0
        setattr(cfg, "dampening_decay", dampening_decay)

    if cfg.upper_limit is None:
        upper_limit = get_upper_limit(
            recorded_sets,
            xs[0,:],
            apply_dampening(
                cfg.get_rep_max(
                    cfg.one_rm, ys[:,0], rep_max_estimator = cfg.get_estimator(cfg.optimal_estimator, inverse = True)
                ),
                dampening = cfg.dampening_decay
            )
        )
        setattr(cfg, "upper_limit", upper_limit.tolist())
    lower_limit = apply_dampening(
        cfg.get_rep_max(cfg.one_rm_low_cap * cfg.one_rm, ys[:,0], rep_max_estimator = inverse_brzycki),
        dampening = cfg.dampening_decay
    )

    # Add a buffer and remove points below lower threshold
    _, mask = add_buffer(
        zs.copy(),
        weight_ax = xs[0,:],
        buffer = cfg.buffer * exercise_counts.iloc[0],
        lower_limit = lower_limit,
        upper_limit = cfg.upper_limit,
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
    zs = crop(
        zs,
        xs[0,:],
        lower_limit = lower_limit,
        upper_limit = cfg.upper_limit,

    )

    zs = zs / zs.sum()
    if not return_df:
        return xs, ys, zs
    else:
        return (xs, ys, zs), main_df    



def plot_heatmap(xs, ys, zs, save_path=None, title=None, xlabel=None, ylabel=None, cmap=None, **mesh_kwargs):
    zs = zs / zs.sum()

    if xs.shape[1] == zs.shape[1]:
        xs = np.concatenate((xs, [[xs[0,-1]+config.WEIGHT_STEP]]), axis=1)
    if ys.shape[0] == zs.shape[0]:
        ys = np.concatenate((ys, [[ys[-1,0]+1]]), axis=0)

    fig, ax = plt.subplots()
    c = ax.pcolormesh(xs-config.WEIGHT_STEP/2, ys-.5, zs, cmap=cmap, shading="flat", **mesh_kwargs)
    fig.colorbar(c, ax = ax)
    ax.set_yticks(ys[:-1,0])

    if title is not None: ax.set_title(title)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        return fig, ax



@click.command()
@click.argument(
    "filepath",
    type = click.Path(),
)
@click.option(
    "--exercise-config-path", "--exercise-config", "-ec",
    default = config.CONFIG_ROOT / "exercise_config.yaml",
    type = click.Path(exists=True),
    help = "Path to config file of YAML-format corresponding to config objects found in src/config/exercise.py"
)
@click.option(
    "--program-config-path", "--program-config", "-pc",
    default = config.CONFIG_ROOT / "program_config.yaml",
    type = click.Path(exists=True),
    help = "Path to config file of YAML-format corresponding to config objects found in src/config/program.py"
)
@click.option(
    "--analysis", "-a",
    is_flag = True,
    default = False,
    help = "Whether to graphically analyse the data."
)
def main(filepath, exercise_config_path, program_config_path, analysis):
    filepath = config.DATA_DIR / filepath
    file_df = pd.read_csv(filepath, header = 0)
    file_df["Exercise Name"] = file_df["Exercise Name"].str.lower()
    # Remove out-of-date data. Based on DATE_LIMIT in config.constants.
    if config.DATE_LIMIT is not None:
        file_df = date_filter(file_df).reset_index()

    exercise_dfs = {
        "Squat": file_df[file_df["Exercise Name"].str.contains("squat")],
        "Bench Press": file_df[file_df["Exercise Name"].str.contains("bench press")],
        "Deadlift": file_df[file_df["Exercise Name"].str.contains("deadlift")],
    }

    cfgs = parse_exercise_configs(exercise_config_path)

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


    program_cfg = parse_program_config(program_config_path)
    # Sample program
    training_program = TrainingProgram(program_cfg, cfgs, meshes)
    training_program.sample_program(
        fname = f"{str(config.SAVE_DIR)}-program.png"
    )



if __name__ == "__main__":
    main()
