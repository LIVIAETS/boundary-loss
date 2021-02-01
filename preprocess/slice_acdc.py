#!/usr/bin/env python3.6

import re
import random
import argparse
import warnings
from pathlib import Path
from pprint import pprint
from functools import partial
from typing import Any, Callable, List, Tuple

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy import unique as uniq
from skimage.io import imsave
from skimage.transform import resize
# from PIL import Image

from utils import mmap_, uc_, map_, augment, flatten_


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    return res.astype(np.uint8)


def get_frame(filename: str, regex: str = ".*_frame(\d+)(_gt)?\.nii.*") -> str:
    matched = re.match(regex, filename)

    if matched:
        return matched.group(1)
    raise ValueError(regex, filename)


def get_p_id(path: Path) -> str:
    '''
    The patient ID, for the ACDC dataset, is the folder containing the data.
    '''
    res = path.parent.name

    assert "patient" in res, res
    return res


def save_slices(img_p: Path, gt_p: Path,
                dest_dir: Path, shape: Tuple[int, int], n_augment: int,
                img_dir: str = "img", gt_dir: str = "gt") -> Tuple[Any, Any, Any, Any]:
    p_id: str = get_p_id(img_p)
    assert "patient" in p_id
    assert p_id == get_p_id(gt_p)

    f_id: str = get_frame(img_p.name)
    assert f_id == get_frame(gt_p.name)

    # Load the data
    dx, dy, dz = nib.load(str(img_p)).header.get_zooms()
    assert dz in [5, 6.5, 7, 10], dz
    img = np.asarray(nib.load(str(img_p)).dataobj)
    gt = np.asarray(nib.load(str(gt_p)).dataobj)

    nx, ny = shape
    fx = nx / img.shape[0]
    fy = ny / img.shape[1]
    # print(f"Before dx {dx:.04f}, dy {dy:.04f}")
    dx /= fx
    dy /= fy
    # print(f"After dx {dx:.04f}, dy {dy:.04f}")

    # print(dx, dy, dz)
    pixel_surface: float = dx * dy
    voxel_volume: float = dx * dy * dz

    assert img.shape == gt.shape
    # assert img.shape[:-1] == shape
    assert img.dtype in [np.uint8, np.int16, np.float32]

    # Normalize and check data content
    norm_img = norm_arr(img)  # We need to normalize the whole 3d img, not 2d slices
    assert 0 == norm_img.min() and norm_img.max() == 255, (norm_img.min(), norm_img.max())
    assert gt.dtype == norm_img.dtype == np.uint8

    resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)

    save_dir_img: Path = Path(dest_dir, img_dir)
    save_dir_gt: Path = Path(dest_dir, gt_dir)
    sizes_2d: np.ndarray = np.zeros(img.shape[-1])
    for j in range(img.shape[-1]):
        img_s = norm_img[:, :, j]
        gt_s = gt[:, :, j]
        assert img_s.shape == gt_s.shape
        assert gt_s.dtype == np.uint8

        # Resize and check the data are still what we expect
        r_img: np.ndarray = resize_(img_s, shape).astype(np.uint8)
        r_gt: np.ndarray = resize_(gt_s, shape, order=0)
        # r_gt: np.ndarray = np.array(Image.fromarray(gt_s, mode='L').resize(shape))
        assert set(uniq(r_gt)).issubset(set(uniq(gt))), (r_gt.dtype, uniq(r_gt))
        r_gt = r_gt.astype(np.uint8)
        assert r_img.dtype == r_gt.dtype == np.uint8
        assert 0 <= r_img.min() and r_img.max() <= 255  # The range might be smaller
        sizes_2d[j] = (r_gt == 3).astype(np.int64).sum()

        for k in range(n_augment + 1):
            if k == 0:
                a_img, a_gt = r_img, r_gt
            else:
                a_img, a_gt = map_(np.asarray, augment(r_img, r_gt))

            for save_dir, data in zip([save_dir_img, save_dir_gt], [a_img, a_gt]):
                filename = f"{p_id}_{f_id}_{k}_{j}.png"
                save_dir.mkdir(parents=True, exist_ok=True)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    imsave(str(Path(save_dir, filename)), data)

    lv_gt = (gt == 3).astype(np.uint8)
    assert set(np.unique(lv_gt)) <= set([0, 1])
    assert lv_gt.shape == gt.shape

    lv_gt = resize_(lv_gt, (*shape, img.shape[-1]), order=0)
    assert set(np.unique(lv_gt)) <= set([0, 1])

    slices_sizes_px = np.einsum("xyz->z", lv_gt.astype(np.int64))
    assert np.array_equal(slices_sizes_px, sizes_2d), (slices_sizes_px, sizes_2d)
    # slices_sizes_px = sizes_2d[...]
    slices_sizes_px = slices_sizes_px[slices_sizes_px > 0]
    slices_sizes_mm2 = slices_sizes_px * pixel_surface

    # volume_size_px = np.einsum("xyz->", lv_gt)
    volume_size_px = slices_sizes_px.sum()
    volume_size_mm3 = volume_size_px * voxel_volume

    # print(f"{slices_sizes_px.mean():.0f}, {volume_size_px}")

    return slices_sizes_px, slices_sizes_mm2, volume_size_px, volume_size_mm3


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the cleaning up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    # Get all the file names, avoid the temporal ones
    nii_paths: List[Path] = [p for p in src_path.rglob('*.nii.gz') if "_4d" not in str(p)]
    assert len(nii_paths) % 2 == 0, "Uneven number of .nii, one+ pair is broken"

    # We sort now, but also id matching is checked while iterating later on
    img_nii_paths: List[Path] = sorted(p for p in nii_paths if "_gt" not in str(p))
    gt_nii_paths: List[Path] = sorted(p for p in nii_paths if "_gt" in str(p))
    assert len(img_nii_paths) == len(gt_nii_paths) == 200
    paths: List[Tuple[Path, Path]] = list(zip(img_nii_paths, gt_nii_paths))

    print(f"Found {len(img_nii_paths)} pairs in total")
    pprint(paths[:5])

    pids: List[str] = sorted(set(map_(get_p_id, img_nii_paths)))
    assert len(pids) == (len(img_nii_paths) // 2), (len(pids), len(img_nii_paths))

    # validation_pids: List[str] = random.sample(pids, args.retains)
    random.shuffle(pids)  # Shuffle before to avoid any problem if the patients are sorted in any way
    validation_slice = slice(args.fold * args.retains, (args.fold + 1) * args.retains)
    validation_pids: List[str] = pids[validation_slice]
    assert len(validation_pids) == args.retains

    validation_paths: List[Tuple[Path, Path]] = [p for p in paths if get_p_id(p[0]) in validation_pids]
    training_paths: List[Tuple[Path, Path]] = [p for p in paths if get_p_id(p[0]) not in validation_pids]
    assert set(validation_paths).isdisjoint(set(training_paths))
    assert len(paths) == (len(validation_paths) + len(training_paths))
    assert len(validation_paths) == 2 * args.retains
    assert len(training_paths) == (len(paths) - 2 * args.retains)

    for mode, _paths, n_augment in zip(["train", "val"], [training_paths, validation_paths], [args.n_augment, 0]):
        img_paths, gt_paths = zip(*_paths)  # type: Tuple[Any, Any]

        dest_dir = Path(dest_path, mode)
        print(f"Slicing {len(img_paths)} pairs to {dest_dir}")
        assert len(img_paths) == len(gt_paths)

        pfun = partial(save_slices, dest_dir=dest_dir, shape=args.shape, n_augment=n_augment)
        all_sizes = mmap_(uc_(pfun), zip(img_paths, gt_paths))
        # for paths in tqdm(list(zip(img_paths, gt_paths)), ncols=50):
        #     uc_(pfun)(paths)

        all_slices_sizes_px, all_slices_sizes_mm2, all_volume_size_px, all_volume_size_mm3 = zip(*all_sizes)

        flat_sizes_px = flatten_(all_slices_sizes_px)
        flat_sizes_mm2 = flatten_(all_slices_sizes_mm2)
        print("px", len(flat_sizes_px), min(flat_sizes_px), max(flat_sizes_px))
        print('\t', "px 5/95", np.percentile(flat_sizes_px, 5), np.percentile(flat_sizes_px, 95))
        print('\t', "mm2", f"{min(flat_sizes_mm2):.02f}", f"{max(flat_sizes_mm2):.02f}")

        _, axes = plt.subplots(nrows=2, ncols=2)
        axes = axes.flatten()

        axes[0].set_title("Slice surface (pixel)")
        axes[0].boxplot(all_slices_sizes_px, whis=[0, 100])

        axes[1].set_title("Slice surface (mm2)")
        axes[1].boxplot(all_slices_sizes_mm2, whis=[0, 100])

        axes[2].set_title("LV volume (pixel)")
        axes[2].hist(all_volume_size_px, bins=len(all_volume_size_px) // 2)

        axes[3].set_title("LV volume (mm3)")
        axes[3].hist(all_volume_size_mm3, bins=len(all_volume_size_px) // 2)

        # plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--img_dir', type=str, default="IMG")
    parser.add_argument('--gt_dir', type=str, default="GT")
    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retains', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--n_augment', type=int, default=0,
                        help="Number of augmentation to create per image, only for the training set")
    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)

    main(args)
