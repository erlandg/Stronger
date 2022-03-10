import yaml
from numpy import ndarray

import config


def brzycki(w, r):
    return w * 36 / (37 - r)


def inverse_brzycki(one_rm, r):
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
