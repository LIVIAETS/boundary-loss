#!/usr/bin/env python3.6

import warnings
from sys import argv
from typing import Dict, Iterable
from pathlib import Path
from functools import partial

import numpy as np
from skimage.io import imread, imsave
from utils import mmap_


def remap(changes: Dict[int, int], filename: str):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        acc = imread(filename)
        assert set(np.unique(acc)).issubset(changes), (set(changes), np.unique(acc))

        for a, b in changes.items():
            acc[acc == a] = b

        imsave(filename, acc)


def main():
    assert len(argv) == 3

    folder = Path(argv[1])
    changes = eval(argv[2])
    remap_ = partial(remap, changes)

    targets: Iterable[str] = map(str, folder.glob("*.png"))
    mmap_(remap_, targets)
    # for filename in targets:
    #     remap_(filename)


if __name__ == "__main__":
    main()
