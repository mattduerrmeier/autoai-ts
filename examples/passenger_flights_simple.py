from autoai_ts import AutoAITS
from sklearn.model_selection import train_test_split
from sklearn import linear_model, svm, ensemble
from xgboost import XGBRegressor
import dataset

### Passenger flights
df = dataset.get_flights_dataset()
X, y = dataset.to_supervised(df)

# train-test split 80/20; not shuffled because time series
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

pipelines = [
    linear_model.LinearRegression(),
    svm.SVR(),
    ensemble.RandomForestRegressor(),
    XGBRegressor(),
]

model = AutoAITS(pipelines)
model.fit(X_train, y_train)

scores = model.score(X_test, y_test)
print("Evaluation: ", scores)
