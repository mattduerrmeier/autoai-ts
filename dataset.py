import pandas as pd
import numpy as np
import numpy.typing as npt


def get_flights_dataset() -> pd.DataFrame:
    df = pd.read_csv("data/flights.csv")

    month_map = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
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


def get_ozone_dataset() -> pd.DataFrame:
    df = pd.read_csv("data/ozone.csv", parse_dates=["Month"], index_col="Month")
    return df


def get_nasa_gistemp() -> pd.DataFrame:
    df = pd.read_csv("data/nasa-gistemp.csv")

    df["date"] = pd.to_datetime(df["year"], format="%Y")
    # for now drop such that it's univariate
    df = df.drop(columns=["no-smoothing", "year"])
    df = df.set_index("date")
    return df


def get_air_quality() -> pd.DataFrame:
    df = pd.read_csv("data/air-quality.csv", sep=";", decimal=",")
    df = df.drop(columns=["Unnamed: 15", "Unnamed: 16"])
    df = df.dropna()

    df["date"] = df["Date"].astype(str) + " " + df["Time"].astype(str)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H.%M.%S")
    df = df.set_index("date")

    df = df.drop(columns=["Date", "Time"])
    return df


def get_bundesplatz_temperature() -> pd.DataFrame:
    df = pd.read_csv("data/bundesplatz-2024.csv", sep=";")
    df["date"] = pd.to_datetime(df["dateObserved"])
    df = df.set_index("date")
    df = df.drop(columns=["dateObserved", "relativeHumidity"])
    return df


def get_walmart_dataset() -> pd.DataFrame:
    df = pd.read_csv(
        "data/walmart.csv",
        parse_dates=["Date"],
        index_col="Date",
        date_format="%d-%m-%Y",
    )
    return df


def get_influenza_cases() -> pd.DataFrame:
    df = pd.read_csv(
        "data/influenza-fohp-oblig.csv",
        parse_dates=["Date"],
        date_format="%d.%m.%Y",
        index_col="Date",
    )
    return df


def get_beijing_pm25() -> pd.DataFrame:
    df = pd.read_csv("data/beijing-pm25.csv")
    df["date"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.drop(columns=["year", "month", "day", "hour", "No"])
    df = df.set_index("date")

    df = df.iloc[24:].ffill()
    df["cbwd"] = df["cbwd"].astype("category").cat.codes

    return df


def get_cosine_function(freq: float = 0.01, time: int = 2000) -> npt.NDArray:
    t = np.arange(0, time)
    amp = np.linspace(0, 1000, t.size)
    return amp * np.sin(2 * np.pi * freq * t + np.pi / 2)


def to_supervised[T: (npt.NDArray, pd.DataFrame)](X: T) -> tuple[T, T]:
    """
    Transform a univariate time series into a supervised problem.
    This is done by creating a lag feature, shifting the observations by 1 step.
    """
    y: np.ndarray | pd.DataFrame
    if isinstance(X, np.ndarray):
        y = X[1:].flatten()
        return X[:-1].reshape(-1, 1), y
    if isinstance(X, pd.DataFrame):
        # must fill with a 0 otherwise it changes the data type
        y = X.shift(-1, fill_value=0).iloc[:-1]
        return X.iloc[:-1], y
    else:
        msg = f"{type(X)} is not supported"
        raise TypeError(msg)
