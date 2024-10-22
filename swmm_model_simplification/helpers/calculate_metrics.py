import numpy as np
import pandas as pd

"""
Calculate regression metrics with a default value if all values are zero.
"""


def all_values_are_zero(s: (np.array or pd.Series)) -> bool:
    """True if all values are zero."""
    # s.round(1).eq(0).all()
    return np.all(np.round(s, 1) == 0)


def both_arrays_are_all_zero(o: (np.array or pd.Series), s: (np.array or pd.Series)) -> bool:
    """True if all values of both arrays are zero."""
    return all_values_are_zero(o) and all_values_are_zero(s)


def calculate_nse(o, s):
    """Calculate Nash-Sutcliffe efficiency. 1 if for both arrays all values are zero."""
    if both_arrays_are_all_zero(o, s):
        return 1
    divisor = ((o - o.mean()) ** 2).sum()
    if divisor == 0:
        return np.nan
    return 1 - ((o - s) ** 2).sum() / divisor


def pae(o, s):
    """Calculate absolute peak error. 0 if for both arrays all values are zero."""
    if both_arrays_are_all_zero(o, s):
        return 0
    return abs(s.max() - o.max()) / o.max()


def peak_error(o, s):
    """Calculate peak error. 0 if for both arrays all values are zero."""
    if both_arrays_are_all_zero(o, s):
        return 0
    max_obs = np.nanmax(o)
    if max_obs == 0:
        return 2
    return (np.nanmax(s) - max_obs) / max_obs


def volume_error(o, s):
    """Calculate volume error. 0 if for both arrays all values are zero."""
    if both_arrays_are_all_zero(o, s):
        return 0
    sum_obs = o.sum()
    if sum_obs == 0:
        return np.nan
    return (s.sum() - sum_obs) / sum_obs
