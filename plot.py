#!/usr/bin/env python3.6

import argparse
from typing import List
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import spline

from utils import map_


def run(args: argparse.Namespace) -> None:
    plt.rc('font', size=args.fontsize)

    colors = ["c", "r", "g", "b", "m", 'y', 'k', 'chartreuse', 'coral']
    styles = ['--', '-.', ':']
    assert len(args.folders) <= len(colors)
    assert len(args.columns) <= len(styles)

    paths: List[Path] = [Path(f, args.filename) for f in args.folders]
    arrays: List[np.ndarray] = map_(np.load, paths)

    if len(arrays[0].shape) == 2:
        arrays = map_(lambda a: a[..., np.newaxis], arrays)
    epoch, _, class_ = arrays[0].shape
    for a in arrays[1:]:
        ea, _, ca = a.shape
        assert epoch == ea and class_ == ca

    n_epoch = arrays[0].shape[0]

    fig = plt.figure(figsize=args.figsize)
    ax = fig.gca()
    ax.set_ylim(args.ylim)
    ax.set_xlim([0, n_epoch - 2])
    ax.set_xlabel("Epoch")
    if args.ylabel:
        ax.set_ylabel(args.ylabel)
    else:
        ax.set_ylabel(Path(args.filename).stem)
    ax.grid(True, axis='y')
    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title(f"{paths[0].stem} over epochs")

    if args.labels:
        labels = args.labels
    else:
        labels = [p.parent.name for p in paths]

    xnew = np.linspace(0, n_epoch, n_epoch * 4)
    epcs = np.arange(n_epoch)
    for a, c, p, l in zip(arrays, colors, paths, labels):
        mean_a = a.mean(axis=1)

        if len(args.columns) > 1:
            mean_column = mean_a[:, args.columns].mean(axis=1)
            ax.plot(epcs, mean_column, color=c, linestyle='-', label=f"{l}-mean", linewidth=2)

        for k, s in zip(args.columns, styles):
            values = mean_a[..., k]

            if args.smooth:
                smoothed = spline(epcs, values, xnew)
                x, y = xnew, smoothed
            else:
                x, y = epcs, values

            lab = l if len(args.columns) == 1 else f"{l}-{k}"
            sty = '-' if len(args.columns) == 1 else s
            ax.plot(x, y, linestyle=sty, color=c, label=lab, linewidth=1.5)
            if args.min:
                print(f"{Path(p).parents[0]}, class {k}: {values.min():.04f}")
            else:
                print(f"{Path(p).parents[0]}, class {k}: {values.max():.04f}")

    if args.hline:
        for v, l, s in zip(args.hline, args.l_line, styles):
            ax.plot([0, n_epoch], [v, v], linestyle=s, linewidth=1, color='green', label=l)

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

    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--min", action="store_true", help="Display the min of each file instead of maximum")

    parser.add_argument("--savefig", type=str, default=None)
    parser.add_argument('--columns', type=int, nargs='+', default=0, help="Which columns of the third axis to plot")
    parser.add_argument("--hline", type=float, nargs='*')
    parser.add_argument("--ylim", type=int, nargs=2, default=[0, 1])

    parser.add_argument("--l_line", type=str, nargs='*')
    parser.add_argument("--title", type=str, default='')
    parser.add_argument("--ylabel", type=str, default='')
    parser.add_argument("--labels", type=str, nargs='*')
    parser.add_argument("--figsize", type=int, nargs='*', default=[14, 9])
    parser.add_argument("--fontsize", type=int, default=10)
    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    run(get_args())
