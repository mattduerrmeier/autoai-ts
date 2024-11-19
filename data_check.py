from scipy.signal import find_peaks
import pandas as pd
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
    """
    Check for negatives values in the input array.
    Return true if there are negative values, false otherwise.
    """
    # log transformation are impossible on negative values
    return bool((x < 0).any())


def compute_look_back_window(x: npt.NDArray, timestamp_column: int=0,
                             max_look_back: int | None=None
                             ) -> int:
    """
    Computes the look back window length for the input dataset.
    By default, it is assumed that the timestamp is the first column of the 2D array.
    """
    ### Timestamps assessment
    timestamps = x[:, timestamp_column]
    timestamps_candidates = _timestamp_analysis(timestamps)

    ### value index assessment
    # 1. zero-crossing
    value_col = x[:, 1].astype(float)
    value_col -= np.mean(value_col)

    # the bit sign is an array of booleans; do a diff (x[i+1] - x[i]) and find indices of true
    zero_crossing_idxs = np.nonzero(np.diff(np.signbit(value_col)))[0]
    zero_crossing_mean = int(np.mean(zero_crossing_idxs))

    # 2. spectral analysis
    spectral_analysis_candidates = _spectral_analysis(value_col)

    # flatten the list of lists
    look_backs: list[int] = [
        candidate
        for candidates in [timestamps_candidates, [zero_crossing_mean], spectral_analysis_candidates]
        for candidate in candidates
    ]
    look_back = _select_look_back(look_backs, len(x), max_look_back)
    return look_back


def _timestamp_analysis(timestamps: npt.NDArray) -> list[int]:
    frequency = pd.infer_freq(timestamps)

    possible_seasonal_periods: list[int]
    if frequency is None:
        possible_seasonal_periods = []
    elif frequency == "min":
        possible_seasonal_periods = [1, 60]
    elif frequency == "h":
        possible_seasonal_periods = [1, 60, 3600]
    elif frequency == "D":
        possible_seasonal_periods = [1, 24, 1440, 86400]
    elif frequency == "W":
        possible_seasonal_periods = [1, 7, 168, 10080, 604800]
    elif frequency[0] == "M": # MS or MY (month start or month end)
        possible_seasonal_periods = [1, 4, 30, 720, 43200, 2592000]
    elif frequency[0] == "Y": # YS or YE (year start / year end)
        possible_seasonal_periods = [1, 12, 52, 365, 8766, 525960, 31557600]

    return possible_seasonal_periods

def _spectral_analysis(values :npt.NDArray) -> list[float]:
    # https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html#numpy.fft.fftfreq
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#find-peaks

    # TODO: what's the frequency of the data for the find peaks and fft?
    fft_magnitude = np.abs(np.fft.fft(values)) # this should be y I think?
    peaks, _ = find_peaks(fft_magnitude)
    return peaks


def _select_look_back(look_backs: list[int],
                      len_x: int,
                      max_look_back: int | None=None
                      ) -> int:
    look_backs = [
        lb
        for lb in look_backs
        # discard values longer than the dataset
        if lb <= len_x
        # we discard 0 and 1 values
        and lb > 1
        # if max_look_back is not None, we check the condition
        and (max_look_back is None or lb <= max_look_back)
    ]

    look_back: int
    if len(look_backs) > 1:
        # TODO: influence vector: how can to implement this?
        # for now, we use the mean value
        look_back = int(np.mean(look_backs))
    elif len(look_backs) == 1:
        look_back = look_backs[0]
    else:
        # default value
        look_back = 8

    return look_back

def to_supervised(X: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    y = X[1:]
    return X[:-1], y
