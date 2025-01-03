from sklearn.linear_model import LinearRegression  # TODO: replace with numpy polyfit
import pandas as pd
import numpy as np
import numpy.typing as npt

"""
This Module performs the data quality check and the look-back window computation in AutoAI-TS,
the first two steps of this model selection technique.
"""


def train_test_split(
    X: npt.NDArray,
    y: npt.NDArray,
    test_size: float = 0.2,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Split the time series data into a train and a test split.
    The data is not shuffled.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Train data to split.

    y : array-like of shape (n_samples, n_targets)
        Target data to split.

    test_size : float, default 0.2
        Proportion of the data to use in the test set.
        Must be between 0.0 and 1.0.

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
        `X_train, X_test, y_train, y_test` splits.
    """
    split_point = len(X) - int(np.ceil(len(X) * test_size))

    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    return X_train, X_test, y_train, y_test


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
    X: npt.NDArray,
    y: npt.NDArray,
    timestamps: pd.DatetimeIndex | None = None,
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

    y : array-like of shape (n_samples, n_targets)
        Target values.
        Used for the selection of a window candidate.

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
    ### timestamps assessment
    # skip if no timestamps are provided (synthetic dataset)
    timestamps_candidates: list[int] = []
    if timestamps is not None:
        timestamps_candidates = _timestamp_analysis(timestamps)

    ### value index assessment
    # 1. zero-crossing
    value_col = X.flatten().copy()
    value_col = value_col - np.mean(value_col)

    # the bit sign is an array of booleans; do a diff (x[i+1] - x[i]) and find indices of true
    zero_crossing_idxs = np.nonzero(np.diff(np.signbit(value_col)))[0]
    zero_crossing_mean = int(np.mean(zero_crossing_idxs))

    # 2. spectral analysis
    spectral_analysis_candidates: list[int] = [
        int(_spectral_analysis(value_col[:ts], ts))
        for ts in timestamps_candidates
        if ts < len(value_col)
    ]

    # combine into a single flat list
    look_backs: list[int] = (
        timestamps_candidates + [zero_crossing_mean] + spectral_analysis_candidates
    )
    return _select_look_back(X, y, look_backs, len(X), max_look_back)


def _timestamp_analysis(
    timestamps: pd.DatetimeIndex,
) -> list[int]:
    frequency = pd.infer_freq(timestamps)

    # Frequency strings: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-period-aliases
    possible_seasonal_periods: list[int]
    if frequency is None:
        possible_seasonal_periods = []
    elif frequency == "s":
        possible_seasonal_periods = [60, 3600, 86400, 604800, 2592000, 31557600]
    elif frequency == "min":
        possible_seasonal_periods = [1, 60, 1440, 10080, 43200, 525960]
    elif frequency == "h":
        possible_seasonal_periods = [1, 24, 168, 720, 8766]
    elif frequency == "D":
        possible_seasonal_periods = [1, 7, 30, 365]
    elif frequency == "W":
        possible_seasonal_periods = [1, 4, 52]
    elif frequency[0] == "M":  # MS or MY (month start or month end)
        possible_seasonal_periods = [1, 12]
    elif frequency[0] == "Y":  # YS or YE (year start / year end)
        possible_seasonal_periods = [1]

    return possible_seasonal_periods


def _spectral_analysis(values_window: npt.NDArray, ts_window: int) -> float:
    # transform to the frequency domain
    # TODO: implement periodigram analysis
    fft = np.fft.fft(values_window)
    peak = int(np.argmax(fft))
    # peak = np.argmax(fft).item()
    # fft_freq = np.fft.fftfreq(ts_window)
    return peak


def _select_look_back(
    X: npt.NDArray,
    y: npt.NDArray,
    look_backs: list[int],
    len_X: int,
    max_look_back: int | None = None,
) -> int:
    look_backs = [
        lb
        for lb in look_backs
        # discard values longer than the dataset
        if lb <= len_X
        # we discard 0 and 1 values
        and lb > 1
        # if max_look_back is not None, we check the condition
        and (max_look_back is None or lb <= max_look_back)
    ]

    look_back: int
    if len(look_backs) > 1:
        # TODO: implement influence score
        # test the different split with a polyfit function and measure accuracy
        scores: list[float] = []
        for lb in look_backs:
            reg = LinearRegression().fit(X[-lb:], y[-lb:])
            scores.append(reg.score(X[-lb:], y[-lb:]))

        max_score = np.argmax(scores).item()
        look_back = look_backs[max_score]
    elif len(look_backs) == 1:
        look_back = look_backs[0]
    else:
        # default value
        look_back = 8

    return look_back
