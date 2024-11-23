from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import numpy as np
import numpy.typing as npt
from model import Model
from typing import Callable

def create_pipelines(contains_neg_values: bool=True, random_state: int=42) -> list[Model]:
    """
    Initialize the model pipelines.
    Pipelines are made of statistical models, machine learning model and Gradient Boosted methods.
    """
    # Zero Model
    zm = ZeroModel()

    # Stat Models
    arima = StatsModelWrapper(ARIMA)
    hw_add = StatsModelWrapper(ExponentialSmoothing, **{"seasonal": "add", "seasonal_periods": 4})
    # currently impossible to install BATS because of pmdarima
    # see this issue: https://github.com/alkaline-ml/pmdarima/issues/577
    # bats = None

    # ML Models
    lr = LinearRegression()
    svr = SVR()
    rfr = RandomForestRegressor(random_state=random_state)
    # AutoEnsembler: use XGBoost instead
    xgb = XGBRegressor(random_state=random_state)

    model_list = [zm, arima, hw_add, svr, rfr, xgb]

    # include models that work with negative
    if contains_neg_values == False:
        hw_mult = StatsModelWrapper(ExponentialSmoothing, **{"seasonal": "Multiplicative", "seasonal_periods": 4})
        model_list.append(hw_mult)

        # Transformer should be combined with other models => which one?
        # log_transformer = FunctionTransformer(np.log, validate=True)
        # model_list.append(log_transformer)

    return model_list


class StatsModelWrapper(Model):
    """
    A Wrapper for statsmodels regressor with true scikit-learn API.
    Required because statsmodel needs the data to initialize the model, but we want this to happen when we call fit instead.
    """
    def __init__(self, model_class, **kwargs):
        self.model_class = model_class
        self.kwargs = kwargs

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> 'StatsModelWrapper':
        m = self.model_class(X, **self.kwargs)
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
        self.pred = X[-1, -1]
        return self

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        return np.array([self.pred] * len(X))

    def score(self, X: npt.NDArray, y: npt.NDArray) -> float:
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
