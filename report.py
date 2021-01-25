#!/usr/bin/env python3.9

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np


def main(args) -> None:
    print(f"Reporting on {len(args.folders)} folders.")

    main_metric: str = args.metrics[0]

    best_epoch: List[int] = display_metric(main_metric, args.folders, args.axises)
    for metric in args.metrics[1:]:
        display_metric(metric, args.folders, args.axises, best_epoch)


def display_metric(metric: str, folders: List[str], axises: Tuple[int], best_epoch: List[int] = None):
    print(f"{metric} (classes {axises})")

    if not best_epoch:
        get_epoch = True
        best_epoch = [0] * len(folders)
    else:
        get_epoch = False

    for i, folder in enumerate(folders):
        file: Path = Path(folder, metric).with_suffix(".npy")
        data: np.ndarray = np.load(file)[:, :, axises]  # Epoch, sample, classes
        averages: np.ndarray = data.mean(axis=(1, 2))
        stds: np.ndarray = data.std(axis=(1, 2))

        if get_epoch:
            best_epoch[i] = np.argmax(averages)
            # print(np.argmax(data))
        val: float = averages[best_epoch[i]]
        val_std: float = stds[best_epoch[i]]

        print(f"\t{Path(folder).name}: {val:.4f} ({val_std:.4f}) at epoch {best_epoch[i]}")

    return best_epoch


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot data over time')
    parser.add_argument('--folders', type=str, required=True, nargs='+', help="The folders containing the file")
    parser.add_argument('--metrics', type=str, required=True, nargs='+')
    parser.add_argument('--axises', type=int, required=True, nargs='+')

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
