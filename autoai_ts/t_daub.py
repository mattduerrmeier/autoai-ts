from sklearn.linear_model import LinearRegression  # TODO: replace with numpy polyfit
from sklearn.model_selection import train_test_split  # TODO: replace with our own
import numpy as np
from autoai_ts.zero_model import ZeroModel
from autoai_ts.metrics import smape
from autoai_ts.model import Model
from typing import Callable
import numpy.typing as npt


class TDaub:
    def __init__(self, pipelines: list[Model], use_zm: bool = True):
        self.pipelines = pipelines
        if use_zm:
            zm = ZeroModel()
            self.pipelines.append(zm)

    def fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        allocation_size: int = 8,
        geo_increment_size: float = 2,
        fixed_allocation_cutoff: int | None = None,
        run_to_completion: int = 3,
        test_size: float = 0.2,
        metric: Callable = smape,
        verbose: bool = True,
    ) -> list[Model]:
        """
        Execute the T-Daub algorithm on the pipelines.
        Returns on the top `run_to_completion` pipelines.
        Modifies the object's state as well.
        """

        if fixed_allocation_cutoff is None:
            fixed_allocation_cutoff = 5 * allocation_size

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        L = len(X_train)

        ### 1. Fixed allocation: run on fixed size data
        num_fix_runs = int(fixed_allocation_cutoff / allocation_size)
        pipeline_scores: dict[int, list[float]] = {
            p_idx: [] for p_idx, _ in enumerate(self.pipelines)
        }

        for i in range(num_fix_runs):
            if verbose:
                print(f"Fixed allocation: [{L - allocation_size*(i+1)}|{L}]")

            for p_idx, p in enumerate(self.pipelines):
                p.fit(
                    X_train[L - allocation_size * (i + 1) : L],
                    y_train[L - allocation_size * (i + 1) : L],
                )
                y_pred = p.predict(X_test)
                score = metric(y_test, y_pred)
                pipeline_scores[p_idx].append(float(score))

        regression_scores: dict[int, float] = {
            p_idx: 0.0 for p_idx, _ in enumerate(self.pipelines)
        }

        for p_idx, p in enumerate(self.pipelines):
            y_score = pipeline_scores[p_idx]
            X_score = np.arange(0, len(y_score)).reshape(-1, 1)

            reg = LinearRegression().fit(X_score, y_score)
            future_pipeline_score = np.array([X_score[-1] + 1])  # how to make this L?
            score_pred = reg.predict(future_pipeline_score)
            regression_scores[p_idx] = score_pred.item()

        # sort the regression score based on future best score
        regression_scores = dict(
            sorted(regression_scores.items(), key=lambda x: x[1], reverse=False)
        )

        ### 2. Allocation acceleration
        # TODO: fix the allocation acceleration
        l = L - allocation_size * num_fix_runs
        last_allocation_size = num_fix_runs * allocation_size

        next_allocation = (
            int(last_allocation_size * geo_increment_size * allocation_size ** (-1))
            * allocation_size
        )
        l = l + next_allocation
        while l < L:
            if verbose:
                print(f"Accel allocation: [{l}|{L}]")

            top_p_idx = next(iter(regression_scores))
            p = self.pipelines[top_p_idx]
            p.fit(
                X_train[l + 1 : L],
                y_train[l + 1 : L],
            )

            y_pred = p.predict(X_test)
            score = metric(y_test, y_pred)
            pipeline_scores[top_p_idx].append(score)

            y_score = pipeline_scores[top_p_idx]
            X_score = np.arange(0, len(y_score)).reshape(-1, 1)
            reg = LinearRegression().fit(X_score, y_score)
            future_pipeline_score = np.array([[10]])  # how to make this L?
            score_preds = reg.predict(future_pipeline_score)

            regression_scores[top_p_idx] = score_preds

            # re-rank based on new score
            regression_scores = dict(
                sorted(regression_scores.items(), key=lambda x: x[1], reverse=False)
            )

            next_allocation = (
                int(last_allocation_size * geo_increment_size * allocation_size ** (-1))
                * allocation_size
            )
            l = l + next_allocation

        ### 3. T-Daub Scoring: select the top n pipelines (n = run_to_completion)
        top_pipelines: list[Model] = [
            self.pipelines[p_idx]
            for p_idx in list(regression_scores)[:run_to_completion]
        ]

        top_scores: list[float] = []
        for top_p in top_pipelines:
            top_p.fit(X_train, y_train)
            y_pred = top_p.predict(X_test)
            score = metric(y_test, y_pred)
            top_scores.append(score)

        top_pipelines = [
            top_p
            for (top_p, _) in sorted(
                zip(top_pipelines, top_scores), key=lambda x: x[1], reverse=False
            )
        ]

        if verbose:
            print(f"Best models: {[type(p).__name__ for p in top_pipelines]}")

        self.pipelines = top_pipelines
        return top_pipelines

    def predict(self, X: npt.NDArray) -> list[npt.NDArray]:
        preds = [p.predict(X) for p in self.pipelines]
        return preds

    def score(
        self, X_test: npt.NDArray, y_test: npt.NDArray, metric: Callable = smape
    ) -> list[float]:
        results: list[float] = []
        for p in self.pipelines:
            y_preds = p.predict(X_test)
            score = metric(y_preds, y_test)
            results.append(score)

        return results
