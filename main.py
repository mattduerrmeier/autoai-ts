from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from data_check import quality_check, negative_value_check
from pipeline import create_pipelines
from model import Model


df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv')
# convert months to numbers since our pipeline requires number only
month_map = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
df["month"] = df["month"].map(month_map)

# train-test split 80/20; not shuffled because time series
train, test = train_test_split(df.to_numpy(), test_size=0.2, shuffle=False)

# perform quality check of the data
quality = quality_check(train)
assert quality, "There are issues with the data: string or nan values"

log_transform = negative_value_check(train)

pipelines: list[Model] = create_pipelines()
