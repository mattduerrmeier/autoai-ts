from typing import Protocol, Self
import numpy.typing as npt


class Model(Protocol):
    def fit(self, X: npt.NDArray, y: npt.NDArray) -> Self: ...
    def predict(self, X: npt.NDArray) -> npt.NDArray: ...
    def score(self, X: npt.NDArray, y: npt.NDArray) -> float: ...
