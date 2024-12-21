from sklearn.model_selection import train_test_split
from autoaits.metrics import smape
import dataset
from autoaits import data_check
from autoaits.model import Model
from autoaits.t_daub import TDaub

### Passenger flights
df = dataset.get_flights_dataset()
X = df.to_numpy()
X, y = dataset.to_supervised(df.to_numpy())

### Bern Bundesplatz Temperature
# df = dataset.get_bundesplatz_temperature()
# X = df["temperature"].to_numpy()
# X, y = dataset.to_supervised(X)

### NASA
# df = dataset.get_nasa_gistemp()
# X, y = dataset.to_supervised(df.to_numpy())

### ozone
# df = dataset.get_ozone_dataset()
# X = df.to_numpy()
# X, y = dataset.to_supervised(X)

### Air Quality
# df = dataset.get_air_quality()
# X = df.drop(columns="AH").to_numpy()
# y = df["AH"].to_numpy()

# train-test split 80/20; not shuffled because time series
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

look_back = data_check.compute_look_back_window(X_train, df.index)
print("look back: \t", look_back)
print("array length: \t", len(df))


# data check
data_check.quality_check(X)
contains_neg_values = data_check.negative_value_check(X)

metric: callable = smape

tdaub = TDaub()
tdaub.fit(X_train, y_train, allocation_size=look_back, metric=metric, verbose=True)

scores = tdaub.score(X_test, y_test, metric=metric)
print("Evaluation: ", scores)
