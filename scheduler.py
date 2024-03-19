#!/usr/bin/env python3.9

# MIT License

# Copyright (c) 2023 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
