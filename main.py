from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from data_check import quality_check, negative_value_check, compute_look_back_window, to_supervised
from pipeline import create_pipelines
from model import Model


df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv')
# convert months to numbers since our pipeline requires number only
month_map = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
df["month"] = df["month"].map(month_map)
# we need a day for the to_datetime function
df["day"] = 1
df["date"] = pd.to_datetime(df[["year", "month", "day"]])
df = df.drop(columns=["month", "year", "day"])
# reorder the columns
df = df[["date", "passengers"]]

# train-test split 80/20; not shuffled because time series
train, test = train_test_split(df, test_size=0.2, shuffle=False)

# perform quality check of the data
quality = quality_check(train["passengers"].to_numpy())
assert quality, "There are issues with the data: string or nan values"

look_back = compute_look_back_window(train.to_numpy())
print(look_back)

pipelines: list[Model] = create_pipelines()
X, y = to_supervised(train.to_numpy())
