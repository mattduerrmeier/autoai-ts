from sklearn.model_selection import train_test_split
from pipeline import create_pipelines
import data_check
import dataset
from model import Model
from t_daub import TDaub

arr = dataset.get_cosine_function()

X, y = dataset.to_supervised(arr)
# train-test split 80/20; not shuffled because time series
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# perform quality check of the data
data_check.quality_check(X_train)
contains_negative_value = data_check.negative_value_check(X_train)

look_back = data_check.compute_look_back_window(X_train)
print("look back: ", look_back)
print("array length: ", len(arr))

pipelines: list[Model] = create_pipelines(contains_negative_value)

tdaub = TDaub(pipelines)
tdaub.fit(X_train, y_train, allocation_size=100, geo_increment_size=5)

print(tdaub.evaluate(X_test, y_test))
