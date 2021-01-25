#!/usr/bin/env python3.9

from typing import Any, Callable, Tuple
from operator import add

from utils import map_, uc_


class DummyScheduler(object):
    def __call__(self, epoch: int, optimizer: Any, loss_fns: list[list[Callable]], loss_weights: list[list[float]]) \
            -> Tuple[float, list[list[Callable]], list[list[float]]]:
        return optimizer, loss_fns, loss_weights


class AddWeightLoss():
    def __init__(self, to_add: list[float]):
        self.to_add: list[float] = to_add

    def __call__(self, epoch: int, optimizer: Any, loss_fns: list[list[Callable]], loss_weights: list[list[float]]) \
            -> Tuple[float, list[list[Callable]], list[list[float]]]:
        assert len(self.to_add) == len(loss_weights[0])
        if len(loss_weights) > 1:
            raise NotImplementedError
        new_weights: list[list[float]] = map_(lambda w: map_(uc_(add), zip(w, self.to_add)), loss_weights)

        print(f"Loss weights went from {loss_weights} to {new_weights}")

        return optimizer, loss_fns, new_weights


class StealWeight():
    def __init__(self, to_steal: float):
        self.to_steal: float = to_steal

    def __call__(self, epoch: int, optimizer: Any, loss_fns: list[list[Callable]], loss_weights: list[list[float]]) \
            -> Tuple[float, list[list[Callable]], list[list[float]]]:
        new_weights: list[list[float]] = [[max(0.1, a - self.to_steal), b + self.to_steal] for a, b in loss_weights]

        print(f"Loss weights went from {loss_weights} to {new_weights}")

        return optimizer, loss_fns, new_weights
