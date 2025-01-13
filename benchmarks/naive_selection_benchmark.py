from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from autoai_ts.metrics import smape
import dataset
from pipeline import create_pipelines
from autoai_ts.data_check import negative_value_check
import pandas as pd
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")


def regressor_pipelines(random_state: int = 42):
    # ML Models
    lr = LinearRegression()
    svr = SVR()
    rfr = RandomForestRegressor(random_state=random_state)
    # AutoEnsembler: use XGBoost instead
    xgb = XGBRegressor(random_state=random_state)
    return [lr, svr, rfr, xgb]


def train_pipelines(
    X: np.ndarray, y: np.ndarray, metric: Callable = smape
) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    contains_neg_values: bool = negative_value_check(X)
    pipelines = create_pipelines(contains_neg_values=contains_neg_values)

    scores: dict[str, float] = {}

    for i, p in enumerate(pipelines):
        print(".", end="", flush=True)
        p.fit(X_train, y_train)

        y_pred = p.predict(X_test)
        scores[type(p).__name__] = metric(y_test, y_pred)

    print(" -> Done!")

    return pd.DataFrame(scores.items(), columns=["model", "smape"])


def plot_save_results(show: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(12, 9), layout="tight")
    ax.set_yscale("log")

    sns.barplot(scores, x="dataset", y="smape", hue="model", ax=ax)
    ax.set_ylabel("SMAPE")
    ax.set_xlabel("Dataset")
    ax.legend(title="Model")
    if show:
        plt.show()

    fig.savefig("model-selection.png", dpi=200, format="png")


### air quality dataset
df = dataset.get_air_quality()
X = df.drop(columns="AH").to_numpy()
y = df["AH"].to_numpy()
air_scores = train_pipelines(X, y)

# NASA Global average Temperature
df = dataset.get_nasa_gistemp()
X, y = dataset.to_supervised(df.to_numpy())
nasa_scores = train_pipelines(X, y)

### Bern Bundesplatz Temperature
df = dataset.get_bundesplatz_temperature()
X = df["temperature"].to_numpy()
X, y = dataset.to_supervised(X)
bern_scores = train_pipelines(X, y)

### Flight dataset
df = dataset.get_flights_dataset()
X = df.to_numpy()
X, y = dataset.to_supervised(X)
flight_scores = train_pipelines(X, y)

### Ozone dataset
df = dataset.get_ozone_dataset()
X = df.to_numpy()
X, y = dataset.to_supervised(X)
ozone_scores = train_pipelines(X, y)

### Walmart
df = dataset.get_walmart_dataset()
X = df.drop(columns=["Weekly_Sales"]).to_numpy()
y = df["Weekly_Sales"].to_numpy()
walmart_scores = train_pipelines(X, y)

### Influenza
df = dataset.get_influenza_cases()
X = df.to_numpy()
X, y = dataset.to_supervised(X)
influenza_scores = train_pipelines(X, y)

### Beijing
df = dataset.get_beijing_pm25()
X = df.drop(columns=["pm2.5"]).to_numpy()
y = df["pm2.5"].to_numpy()
beijing_scores = train_pipelines(X, y)

scores = (
    pd.concat(
        [
            air_scores,
            nasa_scores,
            bern_scores,
            flight_scores,
            ozone_scores,
            walmart_scores,
            influenza_scores,
            beijing_scores,
        ],
        keys=[
            "Air Quality",
            "NASA GISTEMP",
            "Bundesplatz Temperature",
            "Passenger Flights",
            "Ozone",
            "Walmart",
            "Influenza",
            "Beijing",
        ],
        names=["dataset"],
    )
    .reset_index(level=0)
    .reset_index(drop=True)
)

best = scores.iloc[scores.groupby(["dataset"], sort=False)["smape"].idxmin()]
best.loc[:, "smape"] = best.loc[:, "smape"].round(2)
print(best)
