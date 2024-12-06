from sklearn.model_selection import train_test_split
import dataset
from autoaits.pipeline import create_pipelines
from autoaits import data_check
from autoaits.model import Model
from autoaits.t_daub import TDaub

arr = dataset.get_flights_dataset()
idx = arr.index.to_numpy()

X, y = dataset.to_supervised(arr.to_numpy())

# train-test split 80/20; not shuffled because time series
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# perform quality check of the data
data_check.quality_check(X_train)
contains_negative_value = data_check.negative_value_check(X_train)

look_back = data_check.compute_look_back_window(X_train, idx)
print("look back: \t", look_back)
print("array length: \t", len(arr))

pipelines: list[Model] = create_pipelines(contains_negative_value)

tdaub = TDaub(pipelines)
scoring = "smape"
tdaub.fit(X_train, y_train, allocation_size=8, scoring=scoring, verbose=True)

print("Evaluation: ", tdaub.score(X_test, y_test, scoring=scoring))
