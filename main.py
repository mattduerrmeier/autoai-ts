from sklearn.model_selection import train_test_split
import dataset
from typing import Callable
from autoai_ts import data_check
from autoai_ts.t_daub import TDaub
from autoai_ts.metrics import smape

### Passenger flights
df = dataset.get_flights_dataset()
X = df.to_numpy()
X, y = dataset.to_supervised(df.to_numpy())

# train-test split 80/20; not shuffled because time series
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

look_back = data_check.compute_look_back_window(X_train, df.index)
print("look back: \t", look_back)
print("array length: \t", len(df))


# data check
data_check.quality_check(X)
contains_neg_values = data_check.negative_value_check(X)

metric: Callable = smape

tdaub = TDaub()
tdaub.fit(X_train, y_train, allocation_size=look_back, metric=metric, verbose=True)

scores = tdaub.score(X_test, y_test, metric=metric)
print("Evaluation: ", scores)
