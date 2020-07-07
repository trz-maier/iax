import numpy as np
from math import floor


def progress_bar(current, maximum, length: int = 50) -> str:
    """
    Function returns a progress bar in text form
    Minimum position is assummed at 0
    :param current: current position
    :param maximum: maximum position
    :param length: total length of the bar
    :return: progress bar, i.e. [######    ] 60%
    """
    progress = floor(current/maximum * 100)
    l1 = floor(progress/(100/length))
    l2 = length - l1
    return "[%s%s] %s%%" % ('#'*l1, ' '*l2, str(progress))


def box_out_mask(mask: np.array) -> np.array():
    """
    Function boxes out the mask overs its far-most edges
    :param mask: boolean mask in form of a 2-d array
    :return: new mask as a 2-d array
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    out = mask.copy()
    out[r_min:r_max+1, c_min:c_max+1] = True
    return out
