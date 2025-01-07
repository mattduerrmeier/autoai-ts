from autoai_ts import AutoAITS
from pipeline import create_pipelines
from sklearn.model_selection import train_test_split
import dataset

### Passenger flights
df = dataset.get_flights_dataset()
X, y = dataset.to_supervised(df)

# train-test split 80/20; not shuffled because time series
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

pipelines = create_pipelines()

model = AutoAITS(pipelines, positive_idx=[2])
model.fit(X_train, y_train)

scores = model.score(X_test, y_test)
print("Evaluation: ", scores)
