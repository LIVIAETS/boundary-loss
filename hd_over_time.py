#!/usr/bin/env python3.6

import pickle
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
import matplotlib.pyplot as plt

from dataloader import SliceDataset, PatientSampler
from utils import map_, tqdm_, dice_batch, dice_coef, class2one_hot, simplex, sset, haussdorf


def runInference(args: argparse.Namespace):
    # print('>>> Loading the data')
    # device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
    device = torch.device("cpu")
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
    metrics = None

    pred_folders = sorted(list(Path(args.pred_root).glob('iter*')))
    assert len(pred_folders) == args.epochs, (len(pred_folders), args.epochs)
    for epoch, pred_folder in enumerate(pred_folders):
        if args.do_only and epoch not in args.do_only:
            continue

        # First one is dummy:
        folders: List[Path] = [Path(pred_folder, 'val'), Path(pred_folder, 'val'), Path(args.gt_folder)]
        names: List[str] = map_(lambda p: str(p.name), folders[0].glob("*.png"))
        are_hots = [False, True, True]

        # spacing_dict = pickle.load(open(Path(args.gt_folder, "..", "spacing.pkl"), 'rb'))
        spacing_dict = None

        dt_set = SliceDataset(names,
                              folders,
                              transforms=[png_transform, gt_transform, gt_transform],
                              debug=False,
                              C=C,
                              are_hots=are_hots,
                              in_memory=False,
                              spacing_dict=spacing_dict,
                              bounds_generators=bounds_gen,
                              quiet=True)
        loader = DataLoader(dt_set,
                            num_workers=2)

        # print('>>> Computing the metrics')
        total_iteration, total_images = len(loader), len(loader.dataset)
        if not metrics:
            metrics = {"all_dices": torch.zeros((args.epochs, total_images, C), dtype=torch.float64, device=device),
                       "hausdorff": torch.zeros((args.epochs, total_images, C), dtype=torch.float64, device=device)}

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
            assert dices.shape == (B, C)

            haussdorf_res: Tensor = haussdorf(pred, gt)
            assert haussdorf_res.shape == (B, C)

            sm_slice = slice(done, done + B)  # Values only for current batch
            metrics["all_dices"][epoch, sm_slice, ...] = dices
            metrics["hausdorff"][epoch, sm_slice, ...] = haussdorf_res
            done += B

        for key, v in metrics.items():
            print(epoch, key, map_("{:.4f}".format, v[epoch].mean(dim=0)))

    if metrics:
        savedir: Path = Path(args.save_folder)
        for k, e in metrics.items():
            np.save(Path(savedir, f"{k}.npy"), e.cpu().numpy())


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compute metrics for a list of images')
    parser.add_argument('--pred_root', type=str, help="The folder containing the predicted masks")
    parser.add_argument('--gt_folder', type=str, required=True)
    parser.add_argument('--save_folder', type=str, required=True, help="The folder to save the metrics")
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--do_only', type=int, nargs='*')

    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    print(args)

    return args


def main() -> None:
    with torch.no_grad():
        runInference(get_args())


if __name__ == '__main__':
    main()
