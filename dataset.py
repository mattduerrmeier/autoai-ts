import pandas as pd
import numpy as np
import numpy.typing as npt

def get_flights_dataset() -> pd.DataFrame:
    df = pd.read_csv("data/flights.csv")

    month_map = {
        "January": 1, "February": 2, "March": 3,
        "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9,
        "October": 10, "November": 11, "December": 12
    }

    df["month"] = df["month"].map(month_map)

    # a day is needed for the to_datetime function
    df["day"] = 1
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.drop(columns=["month", "year", "day"])
    # reorder the columns
    df = df[["date", "passengers"]]

    df = df.set_index("date")
    return df


def get_cosine_function(freq=0.01, time=2000) -> npt.NDArray:
    t = np.arange(0, time)
    amp = np.linspace(0, 1000, t.size)
    return amp * np.sin(2*np.pi * freq * t + np.pi/2)


def to_supervised(X: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    y = X[1:].reshape(X.size-1)
    return X[:-1].reshape(-1, 1), y
