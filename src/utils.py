import yaml
from numpy import ndarray


def brzycki(w, r):
    """
    Estimates 1RM based on weight and reps of a set
    """
    return w * 36 / (37 - r)


def inverse_brzycki(one_rm, r):
    """
    Estimates r-RM based on 1RM
    """
    return (37 - r) / 36 * one_rm


def get_yaml(filepath):
    with open(filepath, "r") as f:
        yamlfile = yaml.safe_load(f)
    return yamlfile


def ensure_no_zeros(array, fill_value = 1):
    array[array == 0] = fill_value
    return array


def round_to_closest(x, step):
    if type(x) == ndarray:
        return step * (x / step).round()
    else:
        return step * round(x / step)
