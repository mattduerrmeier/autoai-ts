from autoai_ts import data_check
from autoai_ts.zero_model import ZeroModel
from autoai_ts.metrics import smape
from autoai_ts.model import Model
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Callable


class AutoAITS:
    def __init__(
        self,
        pipelines: list[Model],
        positive_idx: list[int] | None = None,
        use_zm: bool = True,
    ) -> None:
        """
        AutoAI-TS model selection framework.
        Initialize AutoAI-TS with a set of pipelines.

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
        Execute AutoAI-TS model selection.
        Verifies the integrity of the data, check for negative values
        and deactivate pipelines that only work on positives values,
        and compute the look-back window.
        Then, execute the T-Daub algorithm on the list of pipelines.
        T-Daub consists of 2 parts:
            1. Fixed allocation: test all pipelines on
                fixed size data allocation.
            2. Allocation acceleration: test only the top pipeline on
                data allocation increasing geometrically.

        Train and score the top `run_to_completion` pipelines on the
        full dataset.
        Sort according the performance, and returns the top pipelines.

        Note that after T-Daub, the list of pipelines is updated
        with the top pipelines only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples, n_targets)
            Target values.

        timestamps : array-like of shape (n_samples), default None
            Optional timestamps or index of the data.
            Used for the automatic look-back window computation.
            If X is a pandas DataFrame, this value is automatically
            set to `df.index`.

        max_look_back : int, default None
            Maximum size of the look-back window.
            Used during the automatic look-back window computation.
            If set, only `lb_candidates <= max_look_back` will be kept.

        allocation_size : int, default None
            Slice of data to use during the fixed allocation.
            If None, the look-back window will be computed automatically.

        geo_increment_size : float, default 0.5
            Size of the geo-increment to use during the allocation
            acceleration.
            Smaller values will increase the number of iterations.

        fixed_allocation_cutoff : int, default None
            Cutoff point where the fixed allocation must stop.
            Must be smaller than `len(X_train)`.
            If None, set to 5 times the allocation_size.

        run_to_completion : int, default 3
            Number of models to select from the list of top performers
            at the end of T-Daub.

        test_size : float, default 0.2
            Proportion of the data to use as validation set.
            Must be between 0.0 and 1.0.

        metric : Callable, default smape
            Function used to compute the score on the validation set.

        verbose : bool, default True
            If True, prints information during execution.

        Returns
        -------
        list[Model]
            A list of the top `run_to_completion` models selected by
            the T-Daub algorithm.
        """
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
            # set the timestamps to the index of the DataFrame before casting to ndarray
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
        """
        Execute the T-Daub algorithm.
        T-Daub consists of 2 parts:
            1. Fixed allocation: test all pipelines on
                fixed size data allocation.
            2. Allocation acceleration: test only the top pipeline on
                data allocation increasing geometrically.

        Train and score the top `run_to_completion` pipelines on the
        full dataset.
        Sort according the performance, and returns the top pipelines.

        Note that after T-Daub, the list of pipelines is updated
        with the top pipelines only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples, n_targets)
            Target values.

        allocation_size : int, default 8
            Slice of data to use during the fixed allocation.
            Default is 8 as recommended in the paper.

        geo_increment_size : float, default 0.5
            Size of the geo-increment to use during the allocation
            acceleration.
            Smaller values will increase the number of iterations.

        fixed_allocation_cutoff : int, default None
            Cutoff point where the fixed allocation must stop.
            Must be smaller than `len(X_train)`.
            If None, set to 5 times the allocation_size.

        run_to_completion : int, default 3
            Number of models to select from the list of top performers
            at the end of T-Daub.

        test_size : float, default 0.2
            Proportion of the data to use as validation set.
            Must be between 0.0 and 1.0.

        metric : Callable, default smape
            Function used to compute the score on the validation set.

        verbose : bool, default True
            If True, prints information during execution.

        Returns
        -------
        list[Model]
            A list of the top `run_to_completion` models selected by
            the T-Daub algorithm.
        """
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
        # push the num_fix_runs to the next value, such that we train on to the full dataset
        # example: 30 / 20 = 1.5 -> 2, such that we train on {20, 30}
        num_fix_runs = int(np.ceil(fixed_allocation_cutoff / allocation_size))
        list_allocation_sizes: list[int] = []

        pipeline_scores: dict[int, list[float]] = {
            p_idx: [] for p_idx, _ in enumerate(self.pipelines)
        }

        for i in range(num_fix_runs):
            # to avoid indexing out of the range of X_train
            alloc = min(allocation_size * (i + 1), L)
            if verbose:
                print(f"Fixed allocation: [{L - alloc}|{L}]")

            for p_idx, p in enumerate(self.pipelines):
                p.fit(
                    X_train[L - alloc : L],
                    y_train[L - alloc : L],
                )
                y_pred = p.predict(X_test)
                score = metric(y_test, y_pred)
                pipeline_scores[p_idx].append(score)

            list_allocation_sizes.append(alloc)

        last_allocation_size = list_allocation_sizes[-1]

        regression_scores: dict[int, float] = {
            p_idx: 0.0 for p_idx, _ in enumerate(self.pipelines)
        }

        for p_idx, p in enumerate(self.pipelines):
            # we can't fit on 1 point: use the pipeline_scores from the single run
            # similarly, if we already trained on L, we can use this score instead
            if num_fix_runs == 1 or last_allocation_size == L:
                if verbose and p_idx == 0:  # show this message only once
                    if num_fix_runs == 1:
                        print("Not enough data points -> skipping regression!")
                    else:
                        print("Already trained on L -> skipping regression!")

                regression_scores[p_idx] = pipeline_scores[p_idx][-1]
            else:
                # fit a linear regression on the scores and predict for L
                y_score = pipeline_scores[p_idx]
                X_score = list_allocation_sizes
                reg = np.poly1d(np.polyfit(X_score, y_score, 1))

                score_pred = reg(L)
                regression_scores[p_idx] = score_pred.item()

        # sort the regression score based on future best score
        regression_scores = dict(
            sorted(regression_scores.items(), key=lambda x: x[1], reverse=False)
        )

        ### 2. Allocation acceleration
        if verbose:
            print("--------------------------")

        # start the allocation where we stopped in phase 1
        # not exactly like the algorithm, but more correct: original algorithm does not always work!
        l_accel = last_allocation_size

        # the next_allocation can be 0: use 4 instead (arbitrary but reasonable increase)
        next_allocation = max(
            4,
            (
                int(last_allocation_size * geo_increment_size * (1 / allocation_size))
                * allocation_size
            ),
        )
        l_accel = l_accel + next_allocation

        # since we switch between the top pipeline in the allocation acceleration,
        # every pipeline needs its own list of allocation_size
        pipeline_allocation_sizes: dict[int, list[int]] = {
            p_idx: list_allocation_sizes.copy()
            for p_idx, _ in enumerate(self.pipelines)
        }

        while l_accel < L:
            if verbose:
                print(f"Accel allocation: [{L-l_accel}|{L}]")

            top_p_idx = next(iter(regression_scores))
            p = self.pipelines[top_p_idx]
            p.fit(
                X_train[L - l_accel : L],
                y_train[L - l_accel : L],
            )

            y_pred = p.predict(X_test)
            score = metric(y_test, y_pred)
            pipeline_scores[top_p_idx].append(score)
            pipeline_allocation_sizes[top_p_idx].append(l_accel)

            # fit a linear regression with the new score and predict for L
            y_score = pipeline_scores[top_p_idx]
            X_score = pipeline_allocation_sizes[top_p_idx]
            reg = np.poly1d(np.polyfit(X_score, y_score, 1))

            score_pred = reg(L)
            regression_scores[top_p_idx] = score_pred.item()

            # re-rank based on the new predicted score
            regression_scores = dict(
                sorted(regression_scores.items(), key=lambda x: x[1], reverse=False)
            )

            next_allocation = max(
                4,
                (
                    int(
                        last_allocation_size
                        * geo_increment_size
                        * (1 / allocation_size)
                    )
                    * allocation_size
                ),
            )
            l_accel = l_accel + next_allocation

        ### 3. T-Daub scoring: select the top run_to_completion pipelines
        top_pipelines: list[Model] = [
            self.pipelines[p_idx]
            for p_idx in list(regression_scores)[:run_to_completion]
        ]

        # train and the full dataset, score and rank the pipelines
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
        list[np.NDArray]
            A list containing the predicted samples.
            The list is of size `run_to_completion * len(X)`.
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
