import numpy.typing as npt
import numpy as np


# TODO: raise errors instead?
def quality_check(x: npt.NDArray) -> bool:
    """
    Verifies that the data has no nan values and on strings.
    Return False if there are issues with the data.
    """

    # input array should not contain strings
    status: bool = False

    # string: return false
    if x.dtype.type == np.object_:
        return False

    # no nan values
    if np.isnan(np.sum(x)) == np.True_:
        return False

    return True


def negative_value_check(x: npt.NDArray) -> bool:
    # log transformation are impossible on negative values
    return bool((x < 0).any())


def get_look_back_window_length(x: npt.NDArray, timestamp_column: int=0):
    """
    Computes the look back window length for the input dataset.
    By default, it is assumed that the timestamp is the first column of the 2D array.
    """
    # TODO: implement timestamp based assessment with temporal frequency

    # value index assessment
    # 1. zero-crossing
    value_col = x[:, -1]
    value_col -= np.mean(value_col);

    # the bit sign is an array of booleans; do a diff (x[i+1] - x[i]) and find indices of true
    zero_crossing_idxs = np.nonzero(np.diff(np.signbit(value_col)))[0]
    zero_crossing_mean = float(np.mean(zero_crossing_idxs))

    # TODO: 2. spectral analysis

    # select look-back window
    look_backs: list[float] = [candidate for candidate in [zero_crossing_mean] if candidate > len(x)]

    look_back = 0.0

    if len(look_backs) > 1:
        # which look back to keep?
        look_back = select_look_back(look_backs)
    elif len(look_backs) == 1:
        look_back = look_backs[0]
    else:
        look_back = 8 # default value

    return look_back


def select_look_back(look_backs: list[float]) -> float:
    # TODO: implement the look back selection as described in the paper
    return look_backs[0]
