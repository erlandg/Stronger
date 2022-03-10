import yaml
from numpy import ndarray


def brzycki(w, r):
    """
    (Conservative for high reps) 1RM estimate based on weight and reps of a set
    """
    return w * 36 / (37 - r)


def inverse_brzycki(one_rm, r):
    """
    (Conservative for high reps) r-RM estimate based on 1RM
    """
    return (37 - r) / 36 * one_rm


def lombardi(w, r):
    """
    (Optimistic for high reps) 1RM estimate based on weight and reps of a set
    """
    return w * r**0.10


def inverse_lombardi(one_rm, r):
    """
    (Optimistic for high reps) r-RM estimate based on 1RM
    """
    return one_rm * r**(-0.10)


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
