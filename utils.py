#!/usr/bin/env python3.9

import argparse
from pathlib import Path
from operator import add
from multiprocessing.pool import Pool
from random import random, uniform, randint
from functools import partial

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast

import torch
import numpy as np
import torch.sparse
from tqdm import tqdm
from torch import einsum
from torch import Tensor
from skimage.io import imsave
from PIL import Image, ImageOps
from medpy.metric.binary import hd
from scipy.ndimage import distance_transform_edt as eucl_distance


colors = ["c", "r", "g", "b", "m", 'y', 'k', 'chartreuse', 'coral', 'gold', 'lavender',
          'silver', 'tan', 'teal', 'wheat', 'orchid', 'orange', 'tomato']

# functions redefinitions
tqdm_ = partial(tqdm, dynamic_ncols=True,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [' '{rate_fmt}{postfix}]')

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def starmmap_(fn: Callable[[Tuple[A]], B], iter: Iterable[Tuple[A]]) -> List[B]:
    return Pool().starmap(fn, iter)


def uc_(fn: Callable) -> Callable:
    return partial(uncurry, fn)


def uncurry(fn: Callable, args: List[Any]) -> Any:
    return fn(*args)


def id_(x):
    return x


def flatten_(to_flat: Iterable[Iterable[A]]) -> List[A]:
    return [e for l in to_flat for e in l]


def flatten__(to_flat):
    if type(to_flat) != list:
        return [to_flat]

    return [e for l in to_flat for e in flatten__(l)]


def depth(e: List) -> int:
    """
    Compute the depth of nested lists
    """
    if type(e) == list and e:
        return 1 + depth(e[0])

    return 0


# fns
def soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bk...->bk", a)[..., None]


def batch_soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bk...->k", a)[..., None]


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


# # Metrics and shitz
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bk...->bk")
dice_batch = partial(meta_dice, "bk...->k")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a & b
    assert sset(res, [0, 1])

    return res


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a | b
    assert sset(res, [0, 1])

    return res


def inter_sum(a: Tensor, b: Tensor) -> Tensor:
    return einsum("bk...->bk", intersection(a, b).type(torch.float32))


def union_sum(a: Tensor, b: Tensor) -> Tensor:
    return einsum("bk...->bk", union(a, b).type(torch.float32))


def hausdorff(preds: Tensor, target: Tensor, spacing: Tensor = None) -> Tensor:
    assert preds.shape == target.shape
    assert one_hot(preds)
    assert one_hot(target)

    B, K, *img_shape = preds.shape

    if spacing is None:
        D: int = len(img_shape)
        spacing = torch.ones((B, D), dtype=torch.float32)

    assert spacing.shape == (B, len(img_shape))

    res = torch.zeros((B, K), dtype=torch.float32, device=preds.device)
    n_pred = preds.cpu().numpy()
    n_target = target.cpu().numpy()
    n_spacing = spacing.cpu().numpy()

    for b in range(B):
        # print(spacing[b])
        # if K == 2:
        #     res[b, :] = hd(n_pred[b, 1], n_target[b, 1], voxelspacing=n_spacing[b])
        #     continue

        for k in range(K):
            if not n_target[b, k].any():  # No object to predict
                if n_pred[b, k].any():  # Predicted something nonetheless
                    res[b, k] = sum((dd * d)**2 for (dd, d) in zip(n_spacing[b], img_shape)) ** 0.5
                    continue
                else:
                    res[b, k] = 0
                    continue
            if not n_pred[b, k].any():
                if n_target[b, k].any():
                    res[b, k] = sum((dd * d)**2 for (dd, d) in zip(n_spacing[b], img_shape)) ** 0.5
                    continue
                else:
                    res[b, k] = 0
                    continue

            res[b, k] = hd(n_pred[b, k], n_target[b, k], voxelspacing=n_spacing[b])

    return res


# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, *img_shape = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, *img_shape)

    return res


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape  # type: Tuple[int, ...]

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res


def np_class2one_hot(seg: np.ndarray, K: int) -> np.ndarray:
    return class2one_hot(torch.from_numpy(seg.copy()).type(torch.int64), K).numpy()


def probs2one_hot(probs: Tensor) -> Tensor:
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                 dtype=None) -> np.ndarray:
    assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res


def one_hot2hd_dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                    dtype=None) -> np.ndarray:
    """
    Used for https://arxiv.org/pdf/1904.10030.pdf,
    implementation from https://github.com/JunMa11/SegWithDistMap
    """
    # Relasx the assertion to allow computation live on only a
    # subset of the classes
    # assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            res[k] = eucl_distance(posmask, sampling=resolution)

    return res


# Misc utils
def save_images(segs: Tensor, names: Iterable[str], root: str, mode: str, iter: int) -> None:
    for seg, name in zip(segs, names):
        save_path = Path(root, f"iter{iter:03d}", mode, name).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if len(seg.shape) == 2:
            imsave(str(save_path), seg.detach().cpu().numpy().astype(np.uint8))
        elif len(seg.shape) == 3:
            np.save(str(save_path), seg.detach().cpu().numpy())
        else:
            raise ValueError("How did you get here")


def augment(*arrs: Union[np.ndarray, Image.Image], rotate_angle: float = 45,
            flip: bool = True, mirror: bool = True,
            rotate: bool = True, scale: bool = False) -> List[Image.Image]:
    imgs: List[Image.Image] = map_(Image.fromarray, arrs) if isinstance(arrs[0], np.ndarray) else list(arrs)

    if flip and random() > 0.5:
        imgs = map_(ImageOps.flip, imgs)
    if mirror and random() > 0.5:
        imgs = map_(ImageOps.mirror, imgs)
    if rotate and random() > 0.5:
        angle: float = uniform(-rotate_angle, rotate_angle)
        imgs = map_(lambda e: e.rotate(angle), imgs)
    if scale and random() > 0.5:
        scale_factor: float = uniform(1, 1.2)
        w, h = imgs[0].size  # Tuple[int, int]
        nw, nh = int(w * scale_factor), int(h * scale_factor)  # Tuple[int, int]

        # Resize
        imgs = map_(lambda i: i.resize((nw, nh)), imgs)

        # Now need to crop to original size
        bw, bh = randint(0, nw - w), randint(0, nh - h)  # Tuple[int, int]

        imgs = map_(lambda i: i.crop((bw, bh, bw + w, bh + h)), imgs)
        assert all(i.size == (w, h) for i in imgs)

    return imgs


def augment_arr(*arrs_a: np.ndarray, rotate_angle: float = 45,
                flip: bool = True, mirror: bool = True,
                rotate: bool = True, scale: bool = False,
                noise: bool = False, noise_loc: float = 0.5, noise_lambda: float = 0.1) -> List[np.ndarray]:
    arrs = list(arrs_a)  # manoucherie type check

    if flip and random() > 0.5:
        arrs = map_(np.flip, arrs)
    if mirror and random() > 0.5:
        arrs = map_(np.fliplr, arrs)
    if noise and random() > 0.5:
        mask: np.ndarray = np.random.laplace(noise_loc, noise_lambda, arrs[0].shape)
        arrs = map_(partial(add, mask), arrs)
        arrs = map_(lambda e: (e - e.min()) / (e.max() - e.min()), arrs)
    # if random() > 0.5:
    #     orig_shape = arrs[0].shape

    #     angle = random() * 90 - 45
    #     arrs = map_(lambda e: sp.ndimage.rotate(e, angle, order=1), arrs)

    #     arrs = get_center(orig_shape, *arrs)

    return arrs


def get_center(shape: Tuple, *arrs: np.ndarray) -> List[np.ndarray]:
    """ center cropping """
    def g_center(arr):
        if arr.shape == shape:
            return arr

        offsets: List[int] = [(arrs - s) // 2 for (arrs, s) in zip(arr.shape, shape)]

        if 0 in offsets:
            return arr[[slice(0, s) for s in shape]]

        res = arr[[slice(d, -d) for d in offsets]][[slice(0, s) for s in shape]]  # Deal with off-by-one errors
        assert res.shape == shape, (res.shape, shape, offsets)

        return res

    return [g_center(arr) for arr in arrs]


def center_pad(arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    assert len(arr.shape) == len(target_shape)

    diff: List[int] = [(nx - x) for (x, nx) in zip(arr.shape, target_shape)]
    pad_width: List[Tuple[int, int]] = [(w // 2, w - (w // 2)) for w in diff]

    res = np.pad(arr, pad_width)
    assert res.shape == target_shape, (res.shape, target_shape)

    return res
