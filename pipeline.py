from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from model import Model

# Model to find:
# - HW (Holt-Winter) Additive
# - HW (Holt-Winter) Multiplicative
# - AutoEnsembler: are these variations of XGBoost?
def create_pipelines() -> list[Model]:
    svr = SVR()
    rfr = RandomForestRegressor()
    xgb = XGBRegressor()
    return [svr, rfr, xgb]
