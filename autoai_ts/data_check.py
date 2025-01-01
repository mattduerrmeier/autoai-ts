import pandas as pd
import numpy as np
import numpy.typing as npt

"""
This Module performs the data quality check and the look-back window computation in AutoAI-TS,
the first two steps of this model selection techniques.
"""


def quality_check(X: npt.NDArray) -> None:
    """
    Verifies that the data has no NaN values or strings.
    Raises an error if there are issues with the data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to verify.
    """

    # input array should not contain strings
    if X.dtype.type == np.object_:
        raise TypeError("AutoAI-TS cannot accept data with strings")

    # input array should not contain nan values
    if np.isnan(np.sum(X)) == np.True_:
        raise TypeError("AutoAI-TS cannot accept data with NaN values")


def negative_value_check(x: npt.NDArray) -> bool:
    """
    Check for negatives values in the input array.
    For example, log transformation are impossible on negative values.
    Return true if there are negative values, false otherwise.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to verify.

    Returns
    -------
    bool
        True if the are negative values in the data, false otherwise.
    """
    return bool((x < 0).any())


def compute_look_back_window(
    x: npt.NDArray,
    timestamps: npt.NDArray | None = None,
    max_look_back: int | None = None,
) -> int:
    """
    Computes the look back window length for the input dataset.
    Timestamps must be passed explicitly to be used.
    Currently works only with univariate datasets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data used to compute the look-back window.
        Value index assessment is performed on these values.

    timestamps: array-like of shape (n_samples), default None
        Optional timestamps or index of the data.
        Timestamps assessment is performed on these values.

    max_look_back: int, default None
        Maximum value for the look-back window.
        Used during the look-back candidate selection.

    Returns
    -------
    int
        The optimal look-back window for X.
    """
    look_backs: list[int] = []

    ### Timestamps assessment
    # We may skip this analysis in no timestamps are provided (synthetic datasets, for example)
    if timestamps is not None:
        timestamps_candidates: list[int] = _timestamp_analysis(timestamps)
        look_backs = timestamps_candidates

    ### value index assessment
    # 1. zero-crossing
    value_col = x.flatten().copy()
    value_col = value_col - np.mean(value_col)

    # the bit sign is an array of booleans; do a diff (x[i+1] - x[i]) and find indices of true
    zero_crossing_idxs = np.nonzero(np.diff(np.signbit(value_col)))[0]
    zero_crossing_mean = int(np.mean(zero_crossing_idxs))

    # 2. spectral analysis
    spectral_analysis_candidate = _spectral_analysis(value_col)

    # flatten the list of lists
    look_backs = look_backs + [zero_crossing_mean, spectral_analysis_candidate]

    look_back = _select_look_back(look_backs, len(x), max_look_back)
    return look_back


def _timestamp_analysis(timestamps: npt.NDArray[np.datetime64]) -> list[int]:
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
    elif frequency[0] == "M":  # MS or MY (month start or month end)
        possible_seasonal_periods = [1, 4, 30, 720, 43200, 2592000]
    elif frequency[0] == "Y":  # YS or YE (year start / year end)
        possible_seasonal_periods = [1, 12, 52, 365, 8766, 525960, 31557600]

    return possible_seasonal_periods


def _spectral_analysis(values: npt.NDArray) -> int:
    # transform to the frequency domain
    fft = np.fft.fft(values)
    # find the peak in this domain
    peak = int(np.argmax(fft))
    return peak


def _select_look_back(
    look_backs: list[int], len_x: int, max_look_back: int | None = None
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
        # influence vector: how can to implement this?
        # for now, we use the median value
        look_back = int(np.median(look_backs))
    elif len(look_backs) == 1:
        look_back = look_backs[0]
    else:
        # default value
        look_back = 8

    return look_back
