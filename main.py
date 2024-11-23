from sklearn.model_selection import train_test_split
from pipeline import create_pipelines
import data_check
import dataset
from model import Model
from t_daub import t_daub_algorithm, evaluate_performance

arr = dataset.get_cosine_function()

# train-test split 80/20; not shuffled because time series
train, test = train_test_split(arr, test_size=0.2, shuffle=False)

# perform quality check of the data
data_check.quality_check(train)

look_back = data_check.compute_look_back_window(train)
print("look back: ", look_back)
print("array length: ", len(arr))

contains_negative_value = data_check.negative_value_check(train)

pipelines: list[Model] = create_pipelines(contains_negative_value)
X_train, y_train = dataset.to_supervised(train)

top_pipelines = t_daub_algorithm(pipelines, X_train, y_train,
                                 allocation_size=look_back,
                                 geo_increment_size=5)

print(top_pipelines)

X_test, y_test = dataset.to_supervised(test)
print(evaluate_performance(top_pipelines, X_test, y_test))
