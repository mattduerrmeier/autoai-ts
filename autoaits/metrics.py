import numpy as np
import numpy.typing as npt

def smape(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    epsilon = np.finfo(np.float64).eps
    smape = np.mean(
        200 * np.abs(y_true - y_pred)
        / np.maximum((np.abs(y_true) + np.abs(y_pred)), epsilon)
    )
    return smape

def mape(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    mape = 100 * np.mean(
        np.abs(y_true - y_pred)
        / np.abs(y_true)
    )
    return mape
