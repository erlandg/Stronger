import yaml
import numpy as np


# See https://en.wikipedia.org/wiki/One-repetition_maximum for one-rep max estimates.

def brzycki(w, r):
    """
    Realistic for low reps. Conservative for high reps.
    1RM estimate based on weight and reps of a set.
    """
    return w * 36 / (37 - r)


def inverse_brzycki(one_rm, r):
    """
    Realistic for low reps. Conservative for high reps.
    r-RM estimate based on 1RM.
    """
    return (37 - r) / 36 * one_rm


def lombardi(w, r):
    """
    Conservative for low reps. Optimistic for high reps.
    """
    return w * r**0.10


def inverse_lombardi(one_rm, r):
    """
    Conservative for low reps. Optimistic for high reps.
    """
    return one_rm * r**(-0.10)


def oconner(w, r):
    """
    Optimistic for low reps. Between Lombardi and Brzycki for high reps.
    """
    return w * (1 + r/40)


def inverse_oconner(one_rm, r):
    """
    Optimistic for low reps. Between Lombardi and Brzycki for high reps.
    """
    return one_rm / (1 + r/40)


def wathen(w, r):
    """
    Realistic for low reps. More conservative than O'Conner et al. for high reps.
    """
    return 100 * w / (48.8 + 53.8 * np.exp(-0.075 * r))


def inverse_wathen(one_rm, r):
    """
    Realistic for low reps. More conservative than O'Conner et al. for high reps.
    """
    return (48.8 + 53.8 * np.exp(-0.075 * r)) * one_rm / 100


def mse(observed, target):
    return np.mean((observed - target)**2)


def get_yaml(filepath):
    with open(filepath, "r") as f:
        yamlfile = yaml.safe_load(f)
    return yamlfile


def ensure_no_zeros(array, fill_value = 1):
    array[array == 0] = fill_value
    return array


def round_to_closest(x, step):
    if type(x) == np.ndarray:
        return step * (x / step).round()
    else:
        return step * round(x / step)


def gridshape_to_datapoints(mask, index_0, index_1):
    datapoints = []
    for j, y in enumerate(index_1):
        for i, x in enumerate(index_0):
            if mask[j,i]:
                datapoints.append((x, y))
    return np.array(datapoints)


def get_median(mask, xs, ys, reps=None, weight=None):
    assert (reps is not None) or (weight is not None), "Either reps or weight argument must be given.."
    assert not ((reps is not None) and (weight is not None)), "Reps and weight arguments cannot both be given."
    if reps is not None:
        median = np.where(mask[ys == reps])[-1]
        median = np.median(xs[median])
    elif weight is not None:
        median = np.where(mask[xs == weight])[-1]
        median = np.median(ys[median])
    return median
