#!/usr/bin/env python3.9

import argparse
from typing import List
from pathlib import Path
from subprocess import call
from itertools import cycle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from utils import map_, colors as util_colors


def run(args: argparse.Namespace) -> None:
    plt.rc('font', size=args.fontsize)

    colors: List[str] = args.colors if args.colors else util_colors

    styles = ['--', '-.', ':', '-']
    if len(args.folders) > len(colors):
        print("Warning: more folders than colors")
    assert len(args.columns) <= len(styles)

    paths: List[Path] = [Path(f, args.filename) for f in args.folders]
    arrays: List[np.ndarray] = map_(np.load, paths)

    if len(arrays[0].shape) == 2:
        arrays = map_(lambda a: a[..., np.newaxis], arrays)
    epoch, _, class_ = arrays[0].shape
    if args.n_epoch:
        epoch = min(epoch, args.n_epoch)
    for a in arrays[1:]:
        ea, _, ca = a.shape
        assert ea <= epoch, (epoch, class_, a.shape)

        if not args.dynamic_third_axis:  # Useful for when trainings don't have same number of losses
            assert class_ == ca, (epoch, class_, a.shape)

    n_epoch = arrays[0].shape[0] if not args.n_epoch else args.n_epoch

    fig = plt.figure(figsize=args.figsize)
    ax = fig.gca()
    ax.set_xlim([0, n_epoch - 2])

    ymin, ymax = args.ylim  # Tuple[int, int]
    ax.set_ylim(ymin, ymax)
    yrange: int = ymax - ymin
    ystep: float = yrange / 10
    yticks = np.mgrid[ymin:ymax + ystep:ystep] if not args.yticks else args.yticks

    ax.set_yticks(yticks)

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

    xnew = np.linspace(0, n_epoch - 1, int(n_epoch * args.sampling_factor))
    epcs = np.arange(n_epoch)
    for i, (a, c, p, l) in enumerate(zip(arrays, cycle(colors), paths, labels)):
        mean_a = a.mean(axis=1)

        _, n_col = mean_a.shape
        # For when more args.columns than columns (weird case with varying multiple losses)
        allowed_cols: List[int] = list(set(args.columns).intersection(set(range(n_col))))

        if len(allowed_cols) > 1 and not args.no_mean:
            mean_column = mean_a[:, allowed_cols].mean(axis=1)
            lw: float = 2 if not args.only_mean else 1.5
            lab: str = f"{l}-mean" if not args.only_mean else l
            ax.plot(epcs, mean_column[:n_epoch], color=c, linestyle='-', label=lab, linewidth=lw)

        if not args.only_mean:
            for k, s in zip(allowed_cols, styles):
                values = mean_a[..., k]

                if args.smooth:
                    # smoothed = spline(epcs, values, xnew)
                    inter_fn = interp1d(epcs, values[:n_epoch], kind='slinear')
                    smoothed = inter_fn(xnew)
                    x, y = xnew, smoothed
                else:
                    x, y = epcs, values[:n_epoch]

                lab = l if len(args.columns) == 1 else f"{l}-{k}"

                sty: str
                if len(args.columns) == 1:
                    if args.curves_styles:
                        sty = args.curves_styles[i][1:]  # Have to remove the extra space
                    else:
                        sty = '-'
                else:
                    sty = s

                ax.plot(x, y, linestyle=sty, color=c, label=lab, linewidth=1.5)
                if args.min:
                    print(f"{Path(p).parents[0]}, class {k}: {values.min():.04f}")
                else:
                    print(f"{Path(p).parents[0]}, class {k}: {values.max():.04f}")

    if args.hline:
        for v, l, s in zip(args.hline, args.l_line, styles):
            ax.plot([0, n_epoch], [v, v], linestyle=s, linewidth=1, color='green', label=l)

    ax.legend(loc=args.loc)

    fig.tight_layout()
    if args.savefig:
        fig.savefig(args.savefig)
        if args.trim:
            call(["mogrify", "-trim", args.savefig])

    if not args.headless:
        plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot data over time')
    parser.add_argument('--folders', type=str, required=True, nargs='+', help="The folders containing the file")
    parser.add_argument('--filename', type=str, required=True)

    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--trim", action="store_true", help="Remove the whitespaces around the figure")
    parser.add_argument("--min", action="store_true", help="Display the min of each file instead of maximum")
    parser.add_argument("--debug", action="store_true", help="Dummy for compatibility")
    parser.add_argument("--cpu", action="store_true", help="Dummy for compatibility")
    parser.add_argument("--no_mean", action="store_true", help="Don't plot the mean line")
    parser.add_argument("--only_mean", action="store_true", help="Plot only the mean line")
    parser.add_argument("--dynamic_third_axis", action="store_true",
                        help="Allow the third axis of the arguments to be of varying size")

    parser.add_argument("--savefig", type=str, default=None)
    parser.add_argument('--columns', type=int, nargs='+', default=0, help="Which columns of the third axis to plot")
    parser.add_argument("--hline", type=float, nargs='*')
    parser.add_argument("--ylim", type=float, nargs=2, default=[0, 1])

    parser.add_argument("--l_line", type=str, nargs='*')
    parser.add_argument("--title", type=str, default='')
    parser.add_argument("--ylabel", type=str, default='')
    parser.add_argument("--labels", type=str, nargs='*')
    parser.add_argument("--colors", type=str, nargs='*')
    parser.add_argument("--figsize", type=int, nargs='*', default=[14, 9])
    parser.add_argument("--yticks", type=float, nargs='*')
    parser.add_argument("--fontsize", type=int, default=10)
    parser.add_argument("--sampling_factor", type=float, default=4)
    parser.add_argument("--n_epoch", type=int, default=None)
    parser.add_argument("--curves_styles", type=str, nargs='*', choices=[' -', ' --', ' -.', ' :'],
                        help="Careful: put an extra space at the beginning of the string, to avoid a parsing error.")
    parser.add_argument("--loc", type=str, default=None, choices=matplotlib.legend.Legend.codes.copy())
    parser.add_argument("--epc", type=int, help="Dummy to maintain call compatibility with hist.py and moustache.py")

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    run(get_args())
