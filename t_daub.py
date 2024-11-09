from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from model import Model
import numpy.typing as npt

class TDaub():
    def __init__(self, pipelines: list[Model]):
        self.pipelines = pipelines


    def fit(self, X: npt.NDArray, y: npt.NDArray) -> list[Model]:
        estimators = [p.fit(X, y) for p in self.pipelines]
        return estimators


    def predict(self, X: npt.NDArray) -> list[npt.NDArray]:
        preds = [p.predict(X) for p in self.pipelines]
        return preds


    def score(self, X: npt.NDArray, y: npt.NDArray) -> list[float]:
        pipeline_scores = [p.score(X, y) for p in self.pipelines]
        return pipeline_scores


def t_daub_algorithm(pipelines: list[Model], X: npt.NDArray, y: npt.NDArray,
                     min_allocation_size: int, fixed_allocation_cutoff: int,
                     geo_increment_size: int, run_to_completion: int,
                     test_size: float = 0.2):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    L = len(X_train)

    # 1. Fixed allocation: run on fixed size data
    num_fix_runs = int(fixed_allocation_cutoff / min_allocation_size)
    pipeline_scores: dict[int, list[float]] = {p_idx: [] for p_idx, _ in enumerate(pipelines)}

    for i in range(num_fix_runs):
        for p_idx, p in enumerate(pipelines):
            p.fit(
                X_train[L - min_allocation_size*i:L],
                y_train[L - min_allocation_size*i:L]
            )

            score = p.score(X_test, y_test)
            pipeline_scores[p_idx].append(score)


    for p_idx, p in enumerate(pipelines):
        X_score = pipeline_scores[p_idx]
        y_score = [*range(len(X))]

        reg = LinearRegression().fit(X_score, y_score)
        future_pipeline_scores = X_score[-1] + 1
        score_preds = reg.predict(future_pipeline_scores)

        pipeline_scores[p_idx].append(score_preds)


    # sort the pipelines based on their average score
    pipeline_scores_sorted = sorted(pipeline_scores.items(), key=lambda x: float(np.mean(x[1])), reverse=True)

    # 2. Allocation acceleration

    # select the top pipelines
    # TODO: define n (for now top 3 or len(pipelines) if smaller)
    top_n = 3 if len(pipelines) >= 3 else len(pipelines)
    top_pipelines: list[Model] = [pipelines[p_idx] for p_idx, _ in pipeline_scores_sorted[0:3]]

    for top_p in top_pipelines:
        top_p.fit(X_train, y_train)
        top_p.score(X_test, y_test)

    return pipelines
