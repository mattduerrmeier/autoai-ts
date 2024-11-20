from sklearn.model_selection import train_test_split
from pipeline import create_pipelines
import data_check
import dataset
from model import Model
from t_daub import t_daub_algorithm

df = dataset.get_flights_dataset()

# train-test split 80/20; not shuffled because time series
train, test = train_test_split(df, test_size=0.2, shuffle=False)

# perform quality check of the data
quality = data_check.quality_check(train["passengers"].to_numpy())
assert quality, "There are issues with the data: string or nan values"

look_back = data_check.compute_look_back_window(train.to_numpy(), train.index.to_numpy())

pipelines: list[Model] = create_pipelines()
X, y = dataset.to_supervised(train.to_numpy())

top_pipelines = t_daub_algorithm(pipelines, X, y, min_allocation_size=8, allocation_size=8, geo_increment_size=2)

print(top_pipelines)
