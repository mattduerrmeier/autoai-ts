from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from model import Model
from typing import Callable
import numpy.typing as npt

# Model to find:
# - HW (Holt-Winter) Additive
# - HW (Holt-Winter) Multiplicative
# - AutoEnsembler: are these variations of XGBoost?
def create_pipelines() -> list[Model]:
    svr = SVR()
    rfr = RandomForestRegressor()
    xgb = XGBRegressor()
    return [svr, rfr, xgb]

def zero_model() -> Callable[[npt.NDArray], npt.DTypeLike]:
    """
    Returns the zero model, used as baseline by AutoAI-TS.
    This model always predicts the most recent value of the time serie.
    """
    return lambda x: x[-1]
