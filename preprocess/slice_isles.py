 #!/usr/bin/env python3.9

import pickle
import random
import argparse
import warnings
from pathlib import Path
from pprint import pprint
from functools import partial
from typing import Dict, List, Tuple

import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from torch import Tensor
from tqdm import tqdm
from skimage.io import imsave

from utils import mmap_, uc_, map_, augment_arr
from utils import class2one_hot, one_hot2dist


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = norm

    return res


def get_p_id(path: Path) -> str:
    '''
    The patient ID, for the ACDC dataset, is the folder containing the data.
    '''
    res = path.parents[1].name

    assert "case_" in res, res
    return res


def save_slices(ct_paths, cbf_paths, cbv_paths, mtt_paths, tmax_paths, gt_paths,
                dest_dir: Path, shape: Tuple[int], n_augment: int,
                ct_dir: str = "ct", cbf_dir="cbf", cbv_dir="cbv", mtt_dir="mtt", tmax_dir="tmax",
                gt_dir: str = "gt", in_npy_dir="in_npy", gt_npy_dir='gt_npy') -> Tuple[int,
                                                                                       int,
                                                                                       Dict,
                                                                                       Tuple[float, float, float]]:
    p_id: str = get_p_id(ct_paths)
    assert len(set(map_(get_p_id, [ct_paths, cbf_paths, cbv_paths, mtt_paths, tmax_paths, gt_paths]))) == 1

    space_dict: Dict[str, Tuple[float, float]] = {}

    # Load the data
    dx, dy, dz = nib.load(str(ct_paths)).header.get_zooms()
    assert dx == dy
    ct = np.asarray(nib.load(str(ct_paths)).dataobj)
    cbf = np.asarray(nib.load(str(cbf_paths)).dataobj)
    cbv = np.asarray(nib.load(str(cbv_paths)).dataobj)
    mtt = np.asarray(nib.load(str(mtt_paths)).dataobj)
    tmax = np.asarray(nib.load(str(tmax_paths)).dataobj)
    gt = np.asarray(nib.load(str(gt_paths)).dataobj)

    assert len(set(map_(np.shape, [ct, cbf, cbv, mtt, tmax, gt]))) == 1
    assert ct.dtype in [np.int32], ct.dtype
    assert cbf.dtype in [np.uint16], cbf.dtype
    assert cbv.dtype in [np.uint16], cbv.dtype
    assert mtt.dtype in [np.float64], mtt.dtype
    assert tmax.dtype in [np.float64], tmax.dtype
    assert gt.dtype in [np.uint8], gt.dtype

    pos: int = (gt == 1).sum()
    neg: int = (gt == 0).sum()

    x, y, z = ct.shape

    # Normalize and check data content
    norm_ct = norm_arr(ct)  # We need to normalize the whole 3d img, not 2d slices
    norm_cbf = norm_arr(cbf)
    norm_cbv = norm_arr(cbv)
    norm_mtt = norm_arr(mtt)
    norm_tmax = norm_arr(tmax)
    assert 0 == norm_ct.min() and norm_ct.max() == 1, (norm_ct.min(), norm_ct.max())
    assert 0 == norm_cbf.min() and norm_cbf.max() == 1, (norm_cbf.min(), norm_cbf.max())
    assert 0 == norm_cbv.min() and norm_cbv.max() == 1, (norm_cbv.min(), norm_cbv.max())
    assert 0 == norm_mtt.min() and norm_mtt.max() == 1, (norm_mtt.min(), norm_mtt.max())
    assert 0 == norm_tmax.min() and norm_tmax.max() == 1, (norm_tmax.min(), norm_tmax.max())

    one_hot_gt: Tensor = class2one_hot(torch.tensor(gt[None, ...], dtype=torch.int64), K=2)[0]
    assert one_hot_gt.shape == (2, 256, 256, z), one_hot_gt.shape
    distmap: np.ndarray = one_hot2dist(one_hot_gt.numpy(),
                                       resolution=(dx, dy, dz),
                                       dtype=np.float32)

    save_dir_ct: Path = Path(dest_dir, ct_dir)
    save_dir_cbf: Path = Path(dest_dir, cbf_dir)
    save_dir_cbv: Path = Path(dest_dir, cbv_dir)
    save_dir_mtt: Path = Path(dest_dir, mtt_dir)
    save_dir_tmax: Path = Path(dest_dir, tmax_dir)
    save_dir_gt: Path = Path(dest_dir, gt_dir)
    save_dir_in_npy: Path = Path(dest_dir, in_npy_dir)
    save_dir_gt_npy: Path = Path(dest_dir, gt_npy_dir)
    save_dir_distmap_npy: Path = Path(dest_dir, "3d_distmap")
    save_dirs = [save_dir_ct, save_dir_cbf, save_dir_cbv, save_dir_mtt, save_dir_tmax, save_dir_gt]

    for j in range(ct.shape[-1]):
        ct_s = norm_ct[:, :, j]
        cbf_s = norm_cbf[:, :, j]
        cbv_s = norm_cbv[:, :, j]
        mtt_s = norm_mtt[:, :, j]
        tmax_s = norm_tmax[:, :, j]
        gt_s = gt[:, :, j]
        dist_s = distmap[:, :, :, j]
        slices = [ct_s, cbf_s, cbv_s, mtt_s, tmax_s, gt_s]
        assert ct_s.shape == cbf_s.shape == cbv_s.shape, mtt_s.shape == tmax_s.shape == gt_s.shape
        assert gt_s.shape == dist_s[0, ...].shape, ((x, y, z), gt_s.shape, dist_s.shape)
        assert set(np.unique(gt_s)).issubset([0, 1])

        # if gt_s.sum() > 0:
        #     print(f"{dist_s[1].min()=} {dist_s[1].max()=}")
        #     _, axes = plt.subplots(nrows=1, ncols=3)
        #     axes[0].imshow(gt_s)
        #     axes[0].set_title("GT")

        #     tmp = axes[1].imshow(dist_s[1, ...], cmap='rainbow')
        #     axes[1].set_title("Signed distance map")
        #     plt.colorbar(tmp, ax=axes[1])

        #     tmp = axes[2].imshow(np.abs(dist_s[1, ...]), cmap='rainbow')
        #     axes[2].set_title("Abs distance map")
        #     plt.colorbar(tmp, ax=axes[2])
        #     plt.show()

        for k in range(n_augment + 1):
            if k == 0:
                to_save = slices
            else:
                to_save = map_(np.asarray, augment_arr(*slices))
                assert to_save[0].shape == slices[0].shape, (to_save[0].shape, slices[0].shape)

            filename = f"{p_id}_{k:02d}_{j:04d}"
            space_dict[filename] = (dx, dy)
            for save_dir, data in zip(save_dirs, to_save):
                save_dir.mkdir(parents=True, exist_ok=True)

                if "gt" not in str(save_dir):
                    img = (data * 255).astype(np.uint8)
                else:
                    img = data.astype(np.uint8)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    imsave(str(Path(save_dir, filename).with_suffix(".png")), img)

            multimodal = np.stack(to_save[:-1])  # Do not include the ground truth
            assert 0 <= multimodal.min() and multimodal.max() <= 1
            save_dir_in_npy.mkdir(parents=True, exist_ok=True)
            save_dir_gt_npy.mkdir(parents=True, exist_ok=True)
            np.save(Path(save_dir_in_npy, filename).with_suffix(".npy"), multimodal)
            np.save(Path(save_dir_gt_npy, filename).with_suffix(".npy"), to_save[-1])

            save_dir_distmap_npy.mkdir(parents=True, exist_ok=True)
            np.save(Path(save_dir_distmap_npy, filename).with_suffix(".npy"), dist_s)

    return neg, pos, space_dict, (dx, dy, dz)


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the cleaning up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    # Get all the file names, avoid the temporal ones
    all_paths: List[Path] = list(src_path.rglob('*.nii'))
    nii_paths: List[Path] = [p for p in all_paths if "_4D" not in str(p)]
    assert len(nii_paths) % 6 == 0, "Number of .nii not multiple of 6, some pairs are broken"

    # We sort now, but also id matching is checked while iterating later on
    CT_nii_paths: List[Path] = sorted(p for p in nii_paths if "CT." in str(p))
    CBF_nii_paths: List[Path] = sorted(p for p in nii_paths if "CT_CBF" in str(p))
    CBV_nii_paths: List[Path] = sorted(p for p in nii_paths if "CT_CBV" in str(p))
    MTT_nii_paths: List[Path] = sorted(p for p in nii_paths if "CT_MTT" in str(p))
    Tmax_nii_paths: List[Path] = sorted(p for p in nii_paths if "CT_Tmax" in str(p))
    gt_nii_paths: List[Path] = sorted(p for p in nii_paths if "OT" in str(p))
    assert len(CT_nii_paths) == len(CBF_nii_paths) == len(CBV_nii_paths) == len(MTT_nii_paths) \
        == len(Tmax_nii_paths) == len(gt_nii_paths)
    paths: List[Tuple[Path, ...]] = list(zip(CT_nii_paths, CBF_nii_paths, CBV_nii_paths, MTT_nii_paths,
                                             Tmax_nii_paths, gt_nii_paths))

    print(f"Found {len(CT_nii_paths)} pairs in total")
    pprint(paths[:2])

    resolution_dict: Dict[str, Tuple[float, float, float]] = {}

    validation_paths: List[Tuple[Path, ...]] = random.sample(paths, args.retain)
    training_paths: List[Tuple[Path, ...]] = [p for p in paths if p not in validation_paths]
    assert set(validation_paths).isdisjoint(set(training_paths))
    assert len(paths) == (len(validation_paths) + len(training_paths))

    for mode, _paths, n_augment in zip(["train", "val"], [training_paths, validation_paths], [args.n_augment, 0]):
        # ct_paths, cbf_paths, cbv_paths, mtt_paths, tmax_paths, gt_paths = zip(*_paths)
        six_paths = list(zip(*_paths))

        dest_dir = Path(dest_path, mode)
        print(f"Slicing {len(six_paths[0])} pairs to {dest_dir}")
        assert len(set(map_(len, six_paths))) == 1

        pfun = partial(save_slices, dest_dir=dest_dir, shape=args.shape, n_augment=n_augment)
        resolutions: List[Tuple[float, float, float]]
        all_neg, all_pos, space_dicts, resolutions = zip(*mmap_(uc_(pfun), zip(*six_paths)))
        neg, pos = sum(all_neg), sum(all_pos)
        ratio = pos / neg
        print(f"Ratio between pos/neg: {ratio} ({pos}/{neg})")
        # space_dicts, resolutions = zip(*map_(uc_(pfun), zip(*six_paths)))
        # for case_paths in tqdm(list(zip(*six_paths)), ncols=50):
        #     uc_(pfun)(case_paths)

        final_dict = {k: v for space_dict in space_dicts for k, v in space_dict.items()}

        for key, val in zip(map_(get_p_id, six_paths[0]), resolutions):
            resolution_dict[key] = val

        with open(Path(dest_dir, "spacing.pkl"), 'wb') as f:
            pickle.dump(final_dict, f, pickle.HIGHEST_PROTOCOL)
            print(f"Saved spacing dictionnary to {f}")

    assert len(resolution_dict.keys()) == len(CT_nii_paths)
    pprint(resolution_dict)

    with open(dest_path / "spacing_3d.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--img_dir', type=str, default="IMG")
    parser.add_argument('--gt_dir', type=str, default="GT")
    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retain', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
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
