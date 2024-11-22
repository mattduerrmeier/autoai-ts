import numpy as np

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    abs_errors = np.abs(y_true - y_pred)
    smape = np.mean(200 * abs_errors / (np.abs(y_true) + np.abs(y_pred)))
    return smape
