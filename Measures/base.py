# Metrics/base.py

import abc


class BaseMeasure(abc.ABC):
    type: str

    @abc.abstractmethod
    def __call__(self, prediction, mask) -> object:
        pass
