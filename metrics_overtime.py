#!/usr/bin/env python3.8

import re
import pickle
import argparse
from pathlib import Path
from functools import partial
from multiprocessing import cpu_count, Pool
from typing import Dict, List, Match, Optional, Pattern, Tuple

import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor, einsum
from torch.utils.data import DataLoader
from medpy.metric.binary import hd, hd95

from utils import map_, starmmap_
from utils import dice_batch, hausdorff
from dataloader import SliceDataset, PatientSampler, custom_collate
from dataloader import png_transform, gt_transform, dist_map_transform


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compute metrics over time on saved predictions')
    parser.add_argument('--basefolder', type=str, required=True, help="The folder containing the predicted epochs")
    parser.add_argument('--gt_folder', type=str)
    parser.add_argument('--spacing', type=str, default='')
    parser.add_argument('--metrics', type=str, nargs='+', required=True,
                        choices=['3d_dsc', '3d_hausdorff', '3d_hd95', 'hausdorff', 'boundary'])
    parser.add_argument("--grp_regex", type=str, required=True)
    parser.add_argument("--resolution_regex", type=str, default=None)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument("--debug", action="store_true", help="Dummy for compatibility")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--n_epoch", type=int, default=-1)
    args = parser.parse_args()

    print(args)

    return args


def main() -> None:
    args = get_args()

    cpu: bool = args.cpu or not torch.cuda.is_available()
    device = torch.device("cpu") if cpu else torch.device("cuda")

    base_path: Path = Path(args.basefolder)

    iterations_paths: List[Path] = sorted(base_path.glob("iter*"))
    # print(iterations_paths)
    print(f">>> Found {len(iterations_paths)} epoch folders")

    # Handle gracefully if not all folders are there (early stop)
    EPC: int = args.n_epoch if args.n_epoch >= 0 else len(iterations_paths)
    K: int = args.num_classes

    # Get the patient number, and image names, from the GT folder
    gt_path: Path = Path(args.gt_folder)
    names: List[str] = map_(lambda p: str(p.name), gt_path.glob("*"))
    n_img: int = len(names)

    resolution_regex: Pattern = re.compile(args.resolution_regex if args.resolution_regex else args.grp_regex)
    spacing_dict: Dict[str, Tuple[float, float, float]]
    spacing_dict = pickle.load(open(args.spacing, 'rb')) if args.spacing else None

    grouping_regex: Pattern = re.compile(args.grp_regex)
    stems: List[str] = [Path(filename).stem for filename in names]  # avoid matching the extension
    matches: List[Match] = map_(grouping_regex.match, stems)  # type: ignore
    patients: List[str] = [match.group(1) for match in matches]

    unique_patients: List[str] = list(set(patients))
    n_patients: int = len(unique_patients)

    print(f">>> Found {len(unique_patients)} unique patients out of {n_img} images ; regex: {args.grp_regex}")
    # from pprint import pprint
    # pprint(unique_patients)

    # First, quickly assert all folders have the same numbers of predited images
    n_img_epoc: List[int] = [len(list((p / "val").glob("*.png"))) for p in iterations_paths]
    assert len(set(n_img_epoc)) == 1
    assert all(len(list((p / "val").glob("*.png"))) == n_img for p in iterations_paths)

    metrics: Dict['str', Tensor] = {}
    if '3d_dsc' in args.metrics:
        metrics['3d_dsc'] = torch.zeros((EPC, n_patients, K), dtype=torch.float32)
        print(f">> Will compute {'3d_dsc'} metric")
    if '3d_hausdorff' in args.metrics:
        metrics['3d_hausdorff'] = torch.zeros((EPC, n_patients, K), dtype=torch.float32)
        print(f">> Will compute {'3d_hausdorff'} metric")
    if '3d_hd95' in args.metrics:
        metrics['3d_hd95'] = torch.zeros((EPC, n_patients, K), dtype=torch.float32)
        print(f">> Will compute {'3d_hd95'} metric")
    if 'hausdorff' in args.metrics:
        metrics['hausdorff'] = torch.zeros((EPC, n_img, K), dtype=torch.float32)
        print(f">> Will compute {'hausdorff'} metric")
    if 'boundary' in args.metrics:
        metrics['boundary'] = torch.zeros((EPC, n_img, K), dtype=torch.float32)
        print(f">> Will compute {'boundary'} metric")

    gen_dataset = partial(SliceDataset,
                          transforms=[png_transform, gt_transform, gt_transform, dist_map_transform],
                          are_hots=[False, True, True, False],
                          K=K,
                          in_memory=False,
                          dimensions=2)
    data_loader = partial(DataLoader,
                          num_workers=cpu_count(),
                          pin_memory=False,
                          collate_fn=custom_collate)

    # Will replace live dataset.folders and call again load_images to update dataset.files
    print(gt_path, gt_path, Path(iterations_paths[0], 'val'))
    dataset: SliceDataset = gen_dataset(names, [gt_path, gt_path, Path(iterations_paths[0], 'val'), gt_path])
    sampler: PatientSampler = PatientSampler(dataset, args.grp_regex, shuffle=False)
    dataloader: DataLoader = data_loader(dataset, batch_sampler=sampler)

    current_path: Path
    for e, current_path in enumerate(iterations_paths):
        pool = Pool()
        dataset.folders = [gt_path, gt_path, Path(current_path, 'val'), gt_path]
        dataset.files = SliceDataset.load_images(dataset.folders, dataset.filenames, False)

        print(f">>> Doing epoch {str(current_path)}")

        done_img: int = 0
        for i, data in enumerate(tqdm(dataloader, leave=None)):
            target: Tensor = data["gt"]
            prediction: Tensor = data["labels"][0]
            B, *_ = target.shape
            # slice_names: Tensor = data['filenames']

            # assert len(slice_names) == target.shape[0]
            # print(slice_names)

            if (match := resolution_regex.match(data['filenames'][0])):
                pid: str = match.group(1)
            else:
                raise ValueError

            voxelspacing: Optional[Tuple[float, float, float]]
            if spacing_dict:
                voxelspacing = spacing_dict[pid]
                # Need to go from (dx, dy, dz) to (dz, dx, dy) (z is on the batch axis now)
                voxelspacing = (voxelspacing[2], voxelspacing[0], voxelspacing[1])
                assert len(voxelspacing) == 3
            else:
                voxelspacing = None
            # print(f"{pid=} {voxelspacing=}")

            assert target.shape == prediction.shape

            if 'hausdorff' in args.metrics:
                hausdorff_res: Tensor = hausdorff(prediction, target,
                                                  data["spacings"])
                assert hausdorff_res.shape == (B, K)
                metrics['hausdorff'][e, done_img:done_img + B, ...] = hausdorff_res[...]

            if 'boundary' in args.metrics:
                distmap: Tensor = data["labels"][1]
                bd: Tensor = einsum("bkwh,bkwh->bk", prediction.type(torch.float32), distmap)

                metrics['boundary'][e, done_img:done_img + B, ...] = bd

            if '3d_dsc' in args.metrics:
                dsc: Tensor = dice_batch(target.to(device), prediction.to(device))
                assert dsc.shape == (K,)

                metrics['3d_dsc'][e, i, :] = dsc.cpu()

            np_pred: np.ndarray
            np_target: np.ndarray
            if '3d_hausdorff' or '3d_hd95' in args.metrics:
                np_pred = prediction.numpy().astype(np.uint8)
                np_target = target.numpy().astype(np.uint8)

                list_float: List[float]
                if '3d_hausdorff' in args.metrics:
                    def cb_1(r):
                        metrics["3d_hausdorff"][e, i, 1:] = torch.tensor(r)
                    pool.starmap_async(partial(get_hd_thing,
                                               fn=hd,
                                               voxelspacing=voxelspacing),
                                       ((np_pred[:, k, :, :], np_target[:, k, :, :])
                                        for k in range(1, K)),
                                       callback=cb_1)
                if '3d_hd95' in args.metrics:
                    def cb_2(r):
                        metrics["3d_hd95"][e, i, 1:] = torch.tensor(r)
                    pool.starmap_async(partial(get_hd_thing,
                                               fn=hd95,
                                               voxelspacing=voxelspacing),
                                       ((np_pred[:, k, :, :], np_target[:, k, :, :])
                                        for k in range(1, K)),
                                       callback=cb_2)

        pool.close()
        pool.join()

        for metric in args.metrics:
            # For now, hardcode the fact we care about class 1 only
            print(f">> {metric}: {metrics[metric][e].mean(dim=0)[1]:.04f}")

    key: str
    el: Tensor
    for key, el in metrics.items():
        np.save(Path(args.basefolder, f"val_{key}.npy"), el.cpu().numpy())


def get_hd_thing(np_pred: np.ndarray, np_target: np.ndarray, fn, voxelspacing):
    hd_thing: float
    if np_pred.sum() > 0:
        hd_thing = fn(np_pred, np_target, voxelspacing=voxelspacing)
    else:
        x, y, z = np_pred.shape
        dx, dy, dz = voxelspacing if voxelspacing else (1, 1, 1)
        hd_thing = ((dx * x)**2 + (dy * y)**2 + (dz * z)**2)**0.5

    return hd_thing


if __name__ == '__main__':
    main()
