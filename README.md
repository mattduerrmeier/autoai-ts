# AutoAI-TS

An opens-source implementation of the AutoAI-TS model selection framework with minimal dependencies.

AutoAI-TS is a *zero-config* model selection framework for time series.
It performs data validation, automatic look-back computation, and model selection with the T-Daub algorithm.

For more details on AutoAI-TS, read the original paper: [AutoAI-TS: AutoAI for Time Series Forecasting](https://dl.acm.org/doi/10.1145/3448016.3457557).

## Installation

This project uses Python 3.13.
We recommend you install the conda package manager to install the correct Python version and packages.

The model selection implementation only depends on `numpy` and `pandas`.
In most cases, your ML project should have these libraries already installed.

There are two version of the environment: standard and minimal.

### Standard Installation (Recommended)

The standard environment installs the model selection dependencies and the following ML libraries: `scikit-learn`, `statsmodels`, and `xgboost`.

```sh
conda env create -f environment.yml
conda activate autoai-ts
```

### Minimal Installation

If you want to use your own set of models, you can install the minimal environment.
This will **only install** `numpy` and `pandas`:

```sh
conda env create -f minimal-environment.yml
conda activate autoai-ts
```

## Example

With the training data as `X` and your targets as `y`.
The data can be a pandas DataFrame or a Numpy array.

```py
from autoai_ts import AutoAITS
from sklearn import linear_model, svm, ensemble

# you need a set of models to use for model selection
pipelines = [linear_model.LinearRegression(), svm.SVR(), ensemble.RandomForestRegressor()]

model = AutoAITS(pipelines)
model.fit(X, y)

model.score(X, y)
```

If you are using the standard environment, a set of pipeline is available through the `pipeline.py` module.
`dataset.py` contains some time series dataset that you can use to test AutoAI-TS.
Here is a more complete example on the Passenger flights dataset with our set of pipelines:

```py
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
```

## License

Licensed under the [GNU AGPLv3](LICENSE) license.
