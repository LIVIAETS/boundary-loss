#!/usr/bin/env python3.6

import argparse
from typing import List
from pathlib import Path
from functools import partial

from tqdm import tqdm
from PIL import Image

from utils import augment, map_, mmap_


def main(args: argparse.Namespace) -> None:
    print(f'>>> Starting data augmentation (original + {args.n_aug} new images)')

    root_dir: str = args.root_dir
    dest_dir: str = args.dest_dir

    folders: List[Path] = list(Path(root_dir).glob("*"))
    dest_folders: List[Path] = [Path(dest_dir, p.name) for p in folders]
    print(f"Will augment data from {len(folders)} folders ({map_(str, folders)})")

    # Create all the destination folders
    for d_folder in dest_folders:
        d_folder.mkdir(parents=True, exist_ok=True)

    names: List[str] = map_(lambda p: str(p.name), folders[0].glob("*.png"))

    partial_process = partial(process_name, folders=folders, dest_folders=dest_folders, n_aug=args.n_aug)
    mmap_(partial_process, names)
    # for name in tqdm(names, ncols=75):
    #     partial_process(name)


def process_name(name: str, folders: List[Path], dest_folders: List[Path], n_aug: int) -> None:
    images: List[Image.Image] = [Image.open(Path(folder, name)).convert('L') for folder in folders]

    stem: str = Path(name).stem

    # Save the unmodified images as _0
    save(stem, 0, images, dest_folders)
    for i in range(1, n_aug + 1):
        augmented: List[Image.Image] = augment(*images)
        save(stem, i, augmented, dest_folders)


def save(stem: str, n: int, imgs: List[Image.Image], dest_folders: List[Path]):
    assert len(imgs) == len(dest_folders)

    for img, folder in zip(imgs, dest_folders):
        img.save(Path(folder, f"{n}_{stem}").with_suffix(".png"))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Data augmentation parameters')
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--n_aug', type=int, required=True)
    args = parser.parse_args()

    print(args)

    return args


if __name__ == '__main__':
    main(get_args())
