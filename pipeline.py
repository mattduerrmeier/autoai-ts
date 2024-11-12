from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import numpy as np
import numpy.typing as npt
from model import Model
from typing import Callable

# Model to find:
# - HW (Holt-Winter) Additive
# - HW (Holt-Winter) Multiplicative
# - AutoEnsembler: are these variations of XGBoost?
def create_pipelines() -> list[Model]:
    # Stat Models
    # TODO: add them

    # Zero Model
    zm = ZeroModel()

    # ML Models
    svr = SVR()
    rfr = RandomForestRegressor()
    xgb = XGBRegressor()

    # Transformers
    log_transformer = FunctionTransformer(np.log, validate=True)

    return [zm, svr, rfr, xgb]


class ZeroModel(Model):
    """
    the Zero model is used as the baseline by AutoAI-TS.
    This model always returns the most recent value of the time series as prediction.
    """
    def __init__(self) -> None:
        self.pred: float = 0.

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> 'ZeroModel':
        self.pred = X[-1, -1].item()
        return self

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        return np.array([self.pred] * len(X))

    def score(self, X: npt.NDArray, y: npt.NDArray) -> float:
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
