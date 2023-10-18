#!/usr/bin/env python3.6

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

import argparse
from typing import List
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import map_


def run(args: argparse.Namespace) -> None:
    plt.rc('font', size=args.fontsize)
    # if len(args.columns) > 1:
    #     raise NotImplementedError("Only 1 columns at a time is handled for now")

    paths: List[Path] = [Path(f, args.filename) for f in args.folders]
    arrays: List[np.ndarray] = map_(np.load, paths)
    metric_name: str = paths[0].stem

    assert len(set(a.shape for a in arrays)) == 1  # All arrays should have the same shape
    if len(arrays[0].shape) == 2:
        arrays = map_(lambda a: a[..., np.newaxis], arrays)  # Add an extra dimension for column selection

    fig = plt.figure(figsize=args.figsize)
    ax = fig.gca()

    ymin, ymax = args.ylim  # Tuple[int, int]
    ax.set_ylim(ymin, ymax)
    yrange: int = ymax - ymin
    ystep: float = yrange / 10
    ax.set_yticks(np.mgrid[ymin:ymax + ystep:ystep])

    if not args.xlabel:
        ax.set_xlabel(metric_name)
    else:
        ax.set_xlabel(args.xlabel)

    if not args.ylabel:
        ax.set_ylabel("Percentage")
    else:
        ax.set_ylabel(args.ylabel)

    ax.grid(True, axis='y')
    if not args.title:
        ax.set_title(f"{metric_name} moustaches")
    else:
        ax.set_title(args.title)

    # bins = np.linspace(0, 1, args.nbins)
    pos = 0
    for i, (a, p) in enumerate(zip(arrays, paths)):
        for k in args.columns:
            mean_a = a[..., k].mean(axis=1)
            best_epoch: int = np.argmax(mean_a)

            # values = a[args.epc, :, k]
            values = a[best_epoch, :, k]

            ax.boxplot(values, positions=[pos + 1], manage_ticks=False, showmeans=True, meanline=True, whis=[5, 95])
            print(f"{p.parent.stem:10}: min {values.min():.03f} 25{np.percentile(values, 25):.03f} "
                  + f"avg {values.mean():.03f} 75 {np.percentile(values, 75):.03f} max {values.max():.03f} at epc {best_epoch}")

            pos += 1
    # ax.legend()

    if not args.labels:
        ax.set_xticklabels([""] + [f"{p.parent.stem}-{k}" for p in paths for k in range(len(args.columns))],
                           rotation=60)
    else:
        if len(args.columns):
            ax.set_xticklabels([""] + [f"{l}-{k}" for l in args.labels for k in range(len(args.columns))],
                               rotation=60)
        else:
            ax.set_xticklabels([""] + [f"{l}" for l in args.labels],
                               rotation=60)

    ax.set_xticks(np.mgrid[0:len(args.folders) * len(args.columns) + 1])

    fig.tight_layout()
    if args.savefig:
        fig.savefig(args.savefig)

    if not args.headless:
        plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot data over time')
    parser.add_argument('--folders', type=str, required=True, nargs='+', help="The folders containing the file")
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--columns', type=int, nargs='+', default=0, help="Which columns of the third axis to plot")
    parser.add_argument("--savefig", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--nbins", type=int, default=100)
    parser.add_argument("--epc", type=int, required=True)

    parser.add_argument("--ylim", type=float, nargs=2, default=[0, 1])

    parser.add_argument("--xlabel", type=str, default='')
    parser.add_argument("--ylabel", type=str, default='')
    parser.add_argument("--labels", type=str, nargs='*')
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--loc", type=str, default=None)
    parser.add_argument("--figsize", type=int, nargs='*', default=[14, 9])
    parser.add_argument("--fontsize", type=int, default=10, help="Dummy opt for compatibility")

    # Dummies
    parser.add_argument("--debug", action="store_true", help="Dummy for compatibility")
    parser.add_argument("--cpu", action="store_true", help="Dummy for compatibility")
    parser.add_argument("--save_csv", action="store_true", help="Dummy for compatibility")
    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    run(get_args())
