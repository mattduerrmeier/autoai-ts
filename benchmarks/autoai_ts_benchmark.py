from sklearn.model_selection import train_test_split
from pipeline import create_pipelines
from autoai_ts import AutoAITS
import dataset
import pandas as pd
import numpy as np


def train_autoai_ts(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    pipelines = create_pipelines()
    model = AutoAITS(pipelines, positive_idx=[2])
    model.fit(
        X_train,
        y_train,
        allocation_size=(len(X_train) // 10),
        test_size=0.1,
        verbose=False,
    )

    scores = model.score(X_test, y_test)
    print(scores)

    best_models = [type(p).__name__ for p in model.pipelines]
    df = pd.DataFrame(zip(best_models, scores), columns=["model", "smape"])
    return df


### air quality dataset
df = dataset.get_air_quality()
X = df.drop(columns="AH").to_numpy()
y = df["AH"].to_numpy()
air_scores = train_autoai_ts(X, y)

# NASA Global average Temperature
df = dataset.get_nasa_gistemp()
X, y = dataset.to_supervised(df.to_numpy())
nasa_scores = train_autoai_ts(X, y)

### Bern Bundesplatz Temperature
df = dataset.get_bundesplatz_temperature()
X = df["temperature"].to_numpy()
X, y = dataset.to_supervised(X)
bern_scores = train_autoai_ts(X, y)

### Flight dataset
df = dataset.get_flights_dataset()
X = df.to_numpy()
X, y = dataset.to_supervised(X)
flight_scores = train_autoai_ts(X, y)

### Ozone dataset
df = dataset.get_ozone_dataset()
X = df.to_numpy()
X, y = dataset.to_supervised(X)
ozone_scores = train_autoai_ts(X, y)

### Walmart
df = dataset.get_walmart_dataset()
X = df.drop(columns=["Weekly_Sales"]).to_numpy()
y = df["Weekly_Sales"].to_numpy()
walmart_scores = train_autoai_ts(X, y)

### Influenza
df = dataset.get_influenza_cases()
X = df.to_numpy()
X, y = dataset.to_supervised(X)
influenza_scores = train_autoai_ts(X, y)

### Beijing
df = dataset.get_beijing_pm25()
X = df.drop(columns=["pm2.5"]).to_numpy()
y = df["pm2.5"].to_numpy()
beijing_scores = train_autoai_ts(X, y)

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
