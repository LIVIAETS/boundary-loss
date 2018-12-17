#!/usr/bin/env python3.6

import argparse
from typing import List
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import map_


def run(args: argparse.Namespace) -> None:
    colors = ["c", "r", "g", "b", "m", 'y', 'k', 'chartreuse', 'coral']
    assert len(args.folders) <= len(colors)

    if len(args.columns) > 1:
        raise NotImplementedError("Only 1 columns at a time is handled for now")

    paths: List[Path] = [Path(f, args.filename) for f in args.folders]
    arrays: List[np.ndarray] = map_(np.load, paths)
    metric_name: str = paths[0].stem

    if len(arrays[0].shape) == 2:
        arrays = map_(lambda a: a[..., np.newaxis], arrays)
    epoch, _, class_ = arrays[0].shape
    for a in arrays[1:]:
        ea, _, ca = a.shape
        assert epoch == ea and class_ == ca

    fig = plt.figure(figsize=(14, 9))
    ax = fig.gca()
    # ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Percentage")
    ax.grid(True, axis='y')
    ax.set_title(f"{metric_name} histograms")

    bins = np.linspace(0, 1, args.nbins)
    for a, c, p in zip(arrays, colors, paths):
        mean_a = a.mean(axis=1).mean(axis=1)
        best_epoch: int = np.argmax(mean_a)

        for k in args.columns:
            values = a[best_epoch, :, k]

            ax.hist(values, bins, alpha=0.5, label=f"{p.parent.name}-{k}", color=c)
    ax.legend()

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
    parser.add_argument("--smooth", action="store_true",
                        help="Help for compatibility with other plotting functions, does not do anything.")
    parser.add_argument("--nbins", type=int, default=100)
    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    run(get_args())
