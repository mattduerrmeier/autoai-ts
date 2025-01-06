from autoai_ts import data_check
from autoai_ts.zero_model import ZeroModel
from autoai_ts.metrics import smape
from autoai_ts.model import Model
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Callable


class TDaub:
    def __init__(
        self,
        pipelines: list[Model],
        positive_idx: list[int] | None = None,
        use_zm: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        pipelines : list
            List of pipelines to use in AutoAI-TS.

        positive_idx : list, default None
            List of the indexes of pipelines that only work on positive values.
            During the data quality check, the first step of AutoAI-TS,
            these pipelines will be deactivated if the dataset contains negative values.

        use_zm : bool, default True
            If True, adds the Zero Model to the list of pipelines.
        """
        self.pipelines = pipelines
        self.positive_idx = positive_idx
        if use_zm:
            zm = ZeroModel()
            self.pipelines.append(zm)

    def fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        timestamps: pd.DatetimeIndex | None = None,
        max_look_back: int | None = None,
        allocation_size: int | None = None,
        geo_increment_size: float = 0.5,
        fixed_allocation_cutoff: int | None = None,
        run_to_completion: int = 3,
        test_size: float = 0.2,
        metric: Callable = smape,
        verbose: bool = True,
    ) -> list[Model]:
        """
        Performs AutoAI-TS model selection.
        Verifies the integrity of the data, deactivate some pipelines
        on negative values, And compute the look-back window.
        Then, execute the T-Daub algorithm on the list of pipelines.
        Returns the top `run_to_completion` pipelines.

        Note that only the top pipelines are preserved after T-Daub.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples, n_targets)
            Target values.

        timestamps : array-like of shape (n_samples), default None
            Optional timestamps or index of the data.
            Used for the automatic look-back window computation.
            If X is a pandas DataFrame, this value is inferred automatically.

        max_look_back : int, default None
            Optional maximum size for the look-back windows.
            Used for the automatic look-back window computation.
            If set, only `lb_candidates <= max_look_back` will be kept.

        allocation_size : int, default None
            Slice of data to use during the fixed allocation.
            If None, the look-back window will be computed automatically.

        geo_increment_size : float, default 0.5
            Size of the geo-increment to use during the allocation acceleration.

        fixed_allocation_cutoff : int, default None

        run_to_completion : int, default 3
            Number of models to select from the list of top performers
            at the end of T-Daub.

        test_size : float, default 0.2
            Proportion of the data to use in the validation set.
            Must be between 0.0 and 1.0.

        metric : Callable, default smape
            Function used to compute the score on the validation set.

        verbose : bool, default True
            If True, prints information during model selection.

        Returns
        -------
        list[Model]
            A list of the top `run_to_completion` models
            selected by the T-Daub algorithm.
        """
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
            # set the timestamps to the index of the dataframe before casting to ndarray
            if timestamps is None:
                timestamps = X.index

            X = X.to_numpy()
            y = y.to_numpy().flatten()

        # 1. Data quality check
        data_check.quality_check(X)

        # deactivate pipelines that do not work on negative values
        contains_neg_values = data_check.negative_value_check(X)
        if (
            self.positive_idx is not None
            and len(self.positive_idx) > 0
            and contains_neg_values
        ):
            self.pipelines = [
                p
                for p_idx, p in enumerate(self.pipelines)
                if p_idx not in self.positive_idx
            ]

            if verbose:
                print(
                    f"Negative values: deactivating models at index {self.positive_idx}"
                )

        # 2. Look-back window computation
        if allocation_size is None:
            allocation_size = data_check.compute_look_back_window(
                X,
                timestamps=timestamps,
                max_look_back=max_look_back,
                verbose=verbose,
            )

        if verbose:
            print(
                f"Look-back window selected: {allocation_size}; Dataset length: {len(X)}"
            )

        # 3. T-Daub model selection
        return self.t_daub(
            X,
            y,
            allocation_size=allocation_size,
            geo_increment_size=geo_increment_size,
            fixed_allocation_cutoff=fixed_allocation_cutoff,
            run_to_completion=run_to_completion,
            test_size=test_size,
            metric=metric,
            verbose=verbose,
        )

    def t_daub(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        allocation_size: int = 8,
        geo_increment_size: float = 0.5,
        fixed_allocation_cutoff: int | None = None,
        run_to_completion: int = 3,
        test_size: float = 0.2,
        metric: Callable = smape,
        verbose: bool = True,
    ) -> list[Model]:
        # TODO: document T-Daub
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
            X = X.to_numpy()
            y = y.to_numpy().flatten()

        X_train, X_test, y_train, y_test = data_check.train_test_split(
            X, y, test_size=test_size
        )
        L = len(X_train)

        if fixed_allocation_cutoff is None:
            # the cutoff must be smaller or equal to the dataset
            fixed_allocation_cutoff = (
                5 * allocation_size if 5 * allocation_size < L else L
            )
            if verbose:
                print("Allocation cutoff: ", fixed_allocation_cutoff)

        ### 1. Fixed allocation
        num_fix_runs = int(np.ceil(fixed_allocation_cutoff / allocation_size).item())
        last_allocation_size = 0

        pipeline_scores: dict[int, list[float]] = {
            p_idx: [] for p_idx, _ in enumerate(self.pipelines)
        }

        for i in range(num_fix_runs):
            start_idx = (
                L - allocation_size * (i + 1) if allocation_size * (i + 1) < L else 0
            )
            if verbose:
                print(f"Fixed allocation: [{start_idx}|{L}]")

            for p_idx, p in enumerate(self.pipelines):
                p.fit(
                    X_train[start_idx:L],
                    y_train[start_idx:L],
                )
                y_pred = p.predict(X_test)
                score = metric(y_test, y_pred)
                pipeline_scores[p_idx].append(score)

            last_allocation_size = allocation_size * (i + 1)

        regression_scores: dict[int, float] = {
            p_idx: 0.0 for p_idx, _ in enumerate(self.pipelines)
        }

        for p_idx, p in enumerate(self.pipelines):
            if num_fix_runs == 1:
                # we can't fit on 1 point: use the pipeline_scores from the single run
                regression_scores[p_idx] = pipeline_scores[p_idx][0]
            else:
                # fit a linear regression on the scores
                y_score = pipeline_scores[p_idx]
                X_score = np.arange(0, len(y_score))
                reg = np.poly1d(np.polyfit(X_score, y_score, 1))

                future_pipeline_score = np.array(
                    [X_score[-1] + 1]
                )  # TODO: how to make this L?
                score_pred = reg(future_pipeline_score)
                regression_scores[p_idx] = score_pred.item()

        # sort the regression score based on future best score
        regression_scores = dict(
            sorted(regression_scores.items(), key=lambda x: x[1], reverse=False)
        )

        ### 2. Allocation acceleration
        if verbose:
            print("--------------------------")

        l = L - allocation_size * num_fix_runs

        next_allocation = (
            int(last_allocation_size * geo_increment_size * (1 / allocation_size))
            * allocation_size
        )
        l = l + next_allocation
        while l < L:
            if verbose:
                print(f"Accel allocation: [{L - l}|{L}]")

            top_p_idx = next(iter(regression_scores))
            p = self.pipelines[top_p_idx]
            p.fit(
                X_train[L - l : L],
                y_train[L - l : L],
            )

            y_pred = p.predict(X_test)
            score = metric(y_test, y_pred)
            pipeline_scores[top_p_idx].append(score)

            # fit a linear regression with the new score
            y_score = pipeline_scores[top_p_idx]
            X_score = np.arange(0, len(y_score))
            reg = np.poly1d(np.polyfit(X_score, y_score, 1))

            future_pipeline_score = np.array(
                X_score[-1] + 1
            )  # TODO: how to make this L?
            score_pred = reg(future_pipeline_score)
            regression_scores[top_p_idx] = score_pred.item()

            # re-rank based on new score
            regression_scores = dict(
                sorted(regression_scores.items(), key=lambda x: x[1], reverse=False)
            )

            # not specified in the algorithm, but this is probably how the geometric increase should behave
            last_allocation_size = l
            next_allocation = (
                int(last_allocation_size * geo_increment_size * (1 / allocation_size))
                * allocation_size
            )
            l = l + next_allocation

        ### 3. T-Daub scoring: select the top n pipelines (n = run_to_completion)
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
        """
        Predict using each of the pipelines selected by T-Daub.

        Parameters
        ----------
        X : array-like of shape(n_samples, n_features)
            Samples to predict.

        Returns
        -------
        list[np.NDArray] : list of size `run_to_completion`
            A list containing the predicted samples.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        preds = [p.predict(X) for p in self.pipelines]
        return preds

    def score(
        self, X: npt.NDArray, y: npt.NDArray, metric: Callable = smape
    ) -> list[float]:
        """
        Score each of the pipelines selected by T-Daub.

        Parameters
        ----------
        X : array-like of shape(n_samples, n_features)
            Samples to predict.

        y : array-like of shape (n_samples, n_targets)
            Target values.

        metric : Callable, default smape
            Function to compute the score.

        Returns
        -------
        list[float]
            A list containing the scores.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            y = y.to_numpy().flatten()

        results: list[float] = []
        for p in self.pipelines:
            y_preds = p.predict(X)
            score = metric(y_preds, y)
            results.append(score)

        return results
