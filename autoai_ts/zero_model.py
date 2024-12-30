from autoai_ts.model import Model
import numpy as np
import numpy.typing as npt


class ZeroModel(Model):
    """
    The Zero Model is used as a baseline in AutoAI-TS.
    It always returns the most recent value of the time series as prediction.
    """

    def __init__(self) -> None:
        self.pred: float = 0.0

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> "ZeroModel":
        self.pred = y[-1]
        return self

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        return np.array([self.pred] * len(X))

    def score(self, X: npt.NDArray, y: npt.NDArray) -> float:
        def r2_score(y, y_pred):
            return (
                1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)).item()
            )

        y_pred = self.predict(X)
        return r2_score(y, y_pred)
