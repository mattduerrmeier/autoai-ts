# AutoAI-TS

An opens-source implementation of the AutoAI-TS model selection framework.

AutoAI-TS is a *zero-config* model selection techniques for time series.
The framework performs data validation, automatic look-back computation, and model selection with the T-Daub algorithm.

For more details on AutoAI-TS, read the original paper: [AutoAI-TS: AutoAI for Time Series Forecasting](https://dl.acm.org/doi/10.1145/3448016.3457557).

## Installation

This project uses Python 3.13.
We recommend you install the conda package manager to install the correct Python version and packages.
There are two version of the environment: standard and minimal.

The model selection implementation only depends on `numpy` and `pandas`.
In most cases, your ML project should have these libraries already installed.


### Recommended Installation

The standard environment installs the model selection dependencies and the following ML libraries: `scikit-learn` and `xgboost`.

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
