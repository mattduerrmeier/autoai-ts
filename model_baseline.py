from sklearn.model_selection import train_test_split
from autoaits.pipeline import create_pipelines
from autoaits.metrics import mape
from autoaits.model import Model
import dataset
import pandas as pd
import numpy as np
from typing import Callable


def train_pipelines(X: np.ndarray, y: np.ndarray, metric: Callable) -> tuple[pd.DataFrame, list[Model]]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    pipelines = create_pipelines()
    scores: dict[str, float] = {}

    for i, p in enumerate(pipelines):
        print(".", end="", flush=True)
        p.fit(X_train, y_train)

        y_pred = p.predict(X_test)
        scores[type(p).__name__] = metric(y_test, y_pred)

    print(" -> Done!")

    df_scores = pd.DataFrame(scores.items(), columns=["model", "mape"])
    return df_scores, pipelines


### air quality dataset
df = dataset.get_air_quality()
X = df.drop(columns="AH").to_numpy()
y = df["AH"].to_numpy()
scores, p = train_pipelines(X, y, mape)

# NASA Global average Temperature
df = dataset.get_nasa_gistemp()
X, y = dataset.to_supervised(df.to_numpy())
scores, p = train_pipelines(X, y, mape)

### Bern Bundesplatz Temperature
df = dataset.get_bundesplatz_temperature()
X = df["temperature"].to_numpy()
X, y = dataset.to_supervised(X)
scores, p = train_pipelines(X, y, mape)

### Flight dataset
df = dataset.get_flights_dataset()
X = df.to_numpy()
X, y = dataset.to_supervised(X)
scores, p = train_pipelines(X, y, mape)

### Ozone dataset
df = dataset.get_ozone_dataset()
X = df.to_numpy()
X, y = dataset.to_supervised(X)
scores, p = train_pipelines(X, y, mape)

### visualize scores
import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(scores, y="mape", hue="model")
plt.show()
