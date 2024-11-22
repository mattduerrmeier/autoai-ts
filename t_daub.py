from metrics import smape
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


def t_daub_algorithm(pipelines: list[Model],
                     X: npt.NDArray, y: npt.NDArray,
                     # min_allocation_size: int,
                     allocation_size: int, # TODO: default value for allocation size?
                     geo_increment_size: int=2,
                     fixed_allocation_cutoff: int | None = None,
                     run_to_completion: int = 3,
                     test_size: float = 0.2
                     ) -> list[Model]:

    if fixed_allocation_cutoff is None:
        fixed_allocation_cutoff = 5 * allocation_size

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    L = len(X_train)

    ### 1. Fixed allocation: run on fixed size data
    num_fix_runs = int(fixed_allocation_cutoff / allocation_size)
    pipeline_scores: dict[int, list[float]] = {p_idx: [] for p_idx, _ in enumerate(pipelines)}

    for i in range(num_fix_runs):
        for p_idx, p in enumerate(pipelines):
            p.fit(
                X_train[L - allocation_size*(i+1):L],
                y_train[L - allocation_size*(i+1):L],
            )

            score = p.score(X_test, y_test)
            pipeline_scores[p_idx].append(score)


    for p_idx, p in enumerate(pipelines):
        y_score = pipeline_scores[p_idx]
        X_score = np.arange(0, len(y_score)).reshape(-1, 1)

        reg = LinearRegression().fit(X_score, y_score)
        future_pipeline_score = np.array([X_score[-1]+1])
        score_preds = reg.predict(future_pipeline_score)

        pipeline_scores[p_idx].append(score_preds.item())


    # sort the pipelines based on their average score
    pipeline_scores_sorted = sorted(pipeline_scores.items(), key=lambda x: float(np.mean(x[1])), reverse=True)

    ### 2. Allocation acceleration
    l = L - allocation_size * num_fix_runs
    last_allocation_size = num_fix_runs * allocation_size

    next_allocation = int(last_allocation_size * geo_increment_size * allocation_size**(-1)) * allocation_size
    l = l + next_allocation
    while l < L:
        top_p_idx = pipeline_scores_sorted[0][0]
        p = pipelines[top_p_idx]
        p.fit(
            X_train[l+1:L],
            y_train[l+1:L],
        )

        score = p.score(X_test, y_test)
        pipeline_scores[top_p_idx].append(score_preds)
        # re-rank based on new score
        pipeline_scores_sorted = sorted(pipeline_scores.items(), key=lambda x: float(np.mean(x[1])), reverse=True)

        next_allocation = int(last_allocation_size * geo_increment_size * allocation_size**(-1)) * allocation_size
        l = l + next_allocation


    ### T-Daub Scoring: select the top n pipelines (n = run_to_completion)
    # TODO: define a default run_to_completion value (for now 3)
    top_pipelines: list[Model] = [pipelines[p_idx] for p_idx, _ in pipeline_scores_sorted[0:run_to_completion]]

    top_scores: list[float] = []
    for top_p in top_pipelines:
        top_p.fit(X_train, y_train)
        score = top_p.score(X_test, y_test)
        top_scores.append(score)

    top_pipelines_sorted = [top_p for (top_p, _) in sorted(zip(top_pipelines, top_scores), key=lambda x: x[1], reverse=True)]
    return top_pipelines_sorted

def evaluate_performance(pipelines: list[Model],
                         X_test: npt.NDArray,
                         y_test: npt.NDArray,
                         evaluation_metric="smape"
                         ) -> list[float]:

    results: list[float] = []
    for p in pipelines:
        y_preds = p.predict(X_test)
        score = smape(y_preds, y_test)
        results.append(score)

    return results
