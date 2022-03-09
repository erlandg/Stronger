import yaml
from datetime import timedelta
from dateutil.relativedelta import relativedelta

import config


def get_yaml(filepath):
    with open(filepath, "r") as f:
        yamlfile = yaml.safe_load(f)
    return yamlfile


def ensure_no_zeros(array, fill_value = 1):
    array[array == 0] = fill_value
    return array


def round_to_closest(x, step):
    return step * round(x / step)


def date_filter(df):
    dates = df["Date"]
    six_month = str(config.DATETIME + relativedelta(months=-config.DATE_LIMIT))
    return df[dates > six_month]
