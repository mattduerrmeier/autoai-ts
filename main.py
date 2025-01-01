from sklearn.model_selection import train_test_split
from autoai_ts.t_daub import TDaub
from pipeline import create_pipelines
import dataset

### Passenger flights
df = dataset.get_flights_dataset()
X = df.to_numpy()
X, y = dataset.to_supervised(df.to_numpy())

# train-test split 80/20; not shuffled because time series
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

pipelines = create_pipelines()

tdaub = TDaub(pipelines, positive_idx=[2])
tdaub.fit(X_train, y_train, allocation_size=10)

scores = tdaub.score(X_test, y_test)
print("Evaluation: ", scores)
