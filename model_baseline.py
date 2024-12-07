from sklearn.model_selection import train_test_split
from autoaits.pipeline import create_pipelines
from autoaits.metrics import smape
from autoaits.model import Model
import dataset
import pandas as pd
import numpy as np
from typing import Callable


def train_pipelines(X: np.ndarray, y: np.ndarray, metric: Callable) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    pipelines = create_pipelines()
    scores: dict[str, float] = {}

    for i, p in enumerate(pipelines):
        print(".", end="", flush=True)
        p.fit(X_train, y_train)

        y_pred = p.predict(X_test)
        scores[type(p).__name__] = metric(y_test, y_pred)

    print(" -> Done!")

    return pd.DataFrame(scores.items(), columns=["model", "smape"])


### air quality dataset
df = dataset.get_air_quality()
X = df.drop(columns="AH").to_numpy()
y = df["AH"].to_numpy()
air_scores = train_pipelines(X, y, smape)

# NASA Global average Temperature
df = dataset.get_nasa_gistemp()
X, y = dataset.to_supervised(df.to_numpy())
nasa_scores = train_pipelines(X, y, smape)

### Bern Bundesplatz Temperature
df = dataset.get_bundesplatz_temperature()
X = df["temperature"].to_numpy()
X, y = dataset.to_supervised(X)
bern_scores = train_pipelines(X, y, smape)

### Flight dataset
df = dataset.get_flights_dataset()
X = df.to_numpy()
X, y = dataset.to_supervised(X)
flight_scores = train_pipelines(X, y, smape)

### Ozone dataset
df = dataset.get_ozone_dataset()
X = df.to_numpy()
X, y = dataset.to_supervised(X)
ozone_scores = train_pipelines(X, y, smape)

### visualize scores
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

scores = pd.concat(
    [air_scores, nasa_scores, bern_scores, flight_scores, ozone_scores],
    keys=["Air Quality", "NASA GISTEMP", "Bundesplatz Temperature", "Passenger Flights", "Ozone"],
    names=["dataset"],
).reset_index(level=0).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 9), layout="tight")
ax.set_yscale("log")

sns.barplot(scores, x="dataset", y="smape", hue="model", ax=ax)
ax.set_ylabel("SMAPE")
ax.set_xlabel("Dataset")
ax.legend(title="Model")
plt.show()

fig.savefig("model-selection.png", dpi=200, format="png")
