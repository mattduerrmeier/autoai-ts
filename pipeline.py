from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy.typing as npt
from autoai_ts.model import Model


def create_pipelines(random_state: int = 42) -> list[Model]:
    """
    Initialize the model pipelines used in T-Daub.
    Pipelines are made of statistical models, machine learning model and Gradient Boosted methods.
    """
    # Stat Models
    arima = StatsModelWrapper(ARIMA, **{"order": (1, 0, 1)})
    hw_add = StatsModelWrapper(
        ExponentialSmoothing, **{"seasonal": "add", "seasonal_periods": 4}
    )
    hw_mult = StatsModelWrapper(
        ExponentialSmoothing,
        **{"seasonal": "Multiplicative", "seasonal_periods": 4},
    )
    # currently impossible to install BATS because of pmdarima
    # see this issue: https://github.com/alkaline-ml/pmdarima/issues/577
    # bats = None

    # ML Models
    lr = LinearRegression()
    svr = SVR()
    rfr = RandomForestRegressor(random_state=random_state)
    # AutoEnsembler: use XGBoost instead
    xgb = XGBRegressor(random_state=random_state)

    model_list = [arima, hw_add, hw_mult, lr, svr, rfr, xgb]

    return model_list


class StatsModelWrapper(Model):
    """
    A Wrapper with scikit-learn API for statsmodels regressor.
    Statsmodels need the data to initialize the model.
    To conform to the scikit-learn API, this should happen we call fit instead.
    """

    def __init__(self, model_class, **kwargs) -> None:
        self.model_class = model_class
        self.kwargs = kwargs

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> "StatsModelWrapper":
        m = self.model_class(endog=y, **self.kwargs)
        self.model = m.fit()
        return self

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        return self.model.forecast(len(X))

    def score(self, X: npt.NDArray, y: npt.NDArray) -> float:
        from sklearn.metrics import r2_score

        y_pred = self.model.forecast(len(X))
        return r2_score(y, y_pred)
