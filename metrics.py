#!/usr/bin/env python3.6

import argparse
from typing import List
from pathlib import Path
from functools import partial
from operator import itemgetter

import torch
import numpy as np
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from dataloader import SliceDataset, PatientSampler
from utils import map_, tqdm_, dice_batch, dice_coef, class2one_hot, simplex, sset


def runInference(args: argparse.Namespace, pred_folder: str):
    # print('>>> Loading the data')
    device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
    C: int = args.num_classes

    # Let's just reuse some code
    png_transform = transforms.Compose([
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])
    gt_transform = transforms.Compose([
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: torch.tensor(nd, dtype=torch.int64),
        partial(class2one_hot, C=C),
        itemgetter(0)
    ])

    bounds_gen = [(lambda *a: torch.zeros(C, 1, 2)) for _ in range(2)]

    folders: List[Path] = [Path(pred_folder), Path(pred_folder), Path(args.gt_folder)]  # First one is dummy
    names: List[str] = map_(lambda p: str(p.name), folders[0].glob("*.png"))
    are_hots = [False, True, True]

    dt_set = SliceDataset(names,
                          folders,
                          transforms=[png_transform, gt_transform, gt_transform],
                          debug=False,
                          C=C,
                          are_hots=are_hots,
                          in_memory=False,
                          bounds_generators=bounds_gen)
    sampler = PatientSampler(dt_set, args.grp_regex)
    loader = DataLoader(dt_set,
                        batch_sampler=sampler,
                        num_workers=11)

    # print('>>> Computing the metrics')
    total_iteration, total_images = len(loader), len(loader.dataset)
    metrics = {"all_dices": torch.zeros((total_images, C), dtype=torch.float64, device=device),
               "batch_dices": torch.zeros((total_iteration, C), dtype=torch.float64, device=device),
               "sizes": torch.zeros((total_images, 1), dtype=torch.float64, device=device)
               }

    desc = f">> Computing"
    tq_iter = tqdm_(enumerate(loader), total=total_iteration, desc=desc)
    done: int = 0
    for j, (filenames, _, pred, gt, _) in tq_iter:
        B = len(pred)
        pred = pred.to(device)
        gt = gt.to(device)
        assert simplex(pred) and sset(pred, [0, 1])
        assert simplex(gt) and sset(gt, [0, 1])

        dices: Tensor = dice_coef(pred, gt)
        b_dices: Tensor = dice_batch(pred, gt)
        assert dices.shape == (B, C)
        assert b_dices.shape == (C,), b_dices.shape

        sm_slice = slice(done, done + B)  # Values only for current batch
        metrics["all_dices"][sm_slice, ...] = dices
        metrics["sizes"][sm_slice, :] = torch.einsum("bwh->b", gt[:, 1, ...])[..., None]
        metrics["batch_dices"][j] = b_dices
        done += B

    print(f">>> {pred_folder}")
    for key, v in metrics.items():
        print(key, map_("{:.4f}".format, v.mean(dim=0)))

    # savedir: Path = Path(args.save_folder)
    # for k, e in metrics.items():
    #     np.save(Path(savedir, f"{k}.npy"), e.cpu().numpy())


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compute metrics for a list of images')
    parser.add_argument('--pred_folders', type=str, nargs='+', help="The folder containing the predicted masks")
    parser.add_argument('--gt_folder', type=str, required=True)
    # parser.add_argument('--save_folder', type=str, required=True, help="The folder to save the metrics")
    parser.add_argument("--grp_regex", type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)

    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    print(args)

    return args


def main() -> None:
    args = get_args()
    for pred_folder in args.pred_folders:
        runInference(args, pred_folder)


if __name__ == '__main__':
    main()
