from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import numpy as np
import numpy.typing as npt
from model import Model
from typing import Callable

def create_pipelines(log_transform: bool) -> list[Model]:
    # Zero Model
    zm = ZeroModel()

    # Stat Models
    # bats = None # TODO: what to use for BATS?
    arima = SMWrapper(ARIMA)
    hw_add = SMWrapper(ExponentialSmoothing, **{"seasonal": "add", "seasonal_periods": 4})
    hw_mult = SMWrapper(ExponentialSmoothing, **{"seasonal": "Multiplicative", "seasonal_periods": 4})

    # ML Models
    svr = SVR()
    rfr = RandomForestRegressor()
    xgb = XGBRegressor()
    # AutoEnsembler: are these variations of XGBoost?
    model_list = [zm, arima, svr, rfr, xgb]

    # Transformers: do we use it?
    if log_transform == False:
        log_transformer = FunctionTransformer(np.log, validate=True)
        model_list.append(log_transformer)

    return model_list


class SMWrapper(Model):
    """
    A Wrapper for statsmodels regressor with true scikit-learn API.
    Required because statsmodel needs the data to initialize the model, but we want this to happen when we call fit.
    """
    def __init__(self, model_class, **kwargs):
        self.model_class = model_class
        self.kwargs = kwargs

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> 'SMWrapper':
        m = self.model_class(y, **self.kwargs)
        self.model = m.fit()
        return self

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        return self.model.forecast(len(X))

    def score(self, X: npt.NDArray, y: npt.NDArray) -> float:
        from sklearn.metrics import r2_score
        y_pred = self.model.forecast(len(X))
        return r2_score(y, y_pred)


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
