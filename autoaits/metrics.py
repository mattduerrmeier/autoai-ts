import numpy as np
import numpy.typing as npt

def smape(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    abs_errors = np.abs(y_true - y_pred)
    smape = np.mean(200 * abs_errors / (np.abs(y_true) + np.abs(y_pred)))
    return smape

def mape(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    abs_errors = np.abs(y_true - y_pred)
    mape = 100 * np.mean(abs_errors / np.abs(y_true))
    return mape
