from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

class TDaub:
    def __init__(self, pipelines):
        self.pipelines = pipelines


    def fit(X, y):
        estimators = [p.fit(X, y) for p in self.pipelines]
        return estimators


    def predict(X):
        preds = [p.predict(X) for p in self.pipelines]
        return preds


    def score(X, y):
        scores = [p.score(X, y) for p in self.pipelines]
        return scores


def t_daub_algorithm(pipelines, X, y,
                     min_allocation_size, fixed_allocation_cutoff,
                     geo_increment_size, run_to_completion,
                     test_size=0.2):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    l = len(X_train)

    # Fixed allocation: run on fixed size data
    num_fix_runs = int(fixed_allocation_cutoff / min_allocation_size)
    scores: dict[str, list[float]] = {k: [] for k in p}

    for i in range(num_fix_runs):
        for p in pipelines:
            p.fit(
                X_train[L - min_allocation_size*i:L],
                y_train[L - min_allocation_size*i:L]
            )

            score = p.score(X_test, y_test)
            scores[p].append(score)


    for p in pipelines:
        X = scores[p]
        y = [*range(len(X))]

        reg = LinearRegression().fit(X, y)
        future_scores = X[-1] + 1
        reg.predict(future_scores)

    # allocation acceleration


    # scoring
    # # TODO: select the top pipelines; define top
    top_pipelines = []
    for top_p in top_pipelines:
        top_p.fit(X_train, y_train)
        top_p.score(X_test, y_test)

    return pipelines
