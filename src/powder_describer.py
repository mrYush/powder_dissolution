import copy
import hashlib
import math
from typing import Optional, List, Dict, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from constants import PIC_TO_DATETIME, MAX_INTENSITY


def get_coeff(diff, time, radius: float, n: int = 10) -> float:
    ns = np.arange(1, n + 1)
    power = -diff * (math.pi ** 2) * (ns ** 2) * time / (radius ** 2)
    tmp_sum = sum((1 / ns ** 2) * np.exp(power))
    mult = 6 / math.pi ** 2
    return 1 - mult * tmp_sum


class OnePic:
    def __init__(self, img, pred, thr: float = 0.5):
        self.img = img
        self.image_g_channel = self.img[:, :, 1]
        self.pred = {key: val.cpu().numpy() for key, val in pred.items()}
        self.thr = thr
        self.masks = self.pred['masks'].reshape((
            self.pred['masks'].shape[0], 600, 600
        ))[self.pred['scores'] > self.thr]
        self.joint_masks = self.masks.mean(axis=0) > \
            self.masks.mean()
        self.background = 1 - self.joint_masks

        self._flat_background: Optional[np.ndarray] = None
        self._boxes: Optional[np.ndarray] = None
        self._particles: Optional[List[Particle]] = None

    @property
    def flat_background(self):
        if self._flat_background is None:
            flat_background_all = np.where(
                self.background > 0,
                self.image_g_channel,
                np.nan
            ).flatten()
            self._flat_background = pd.Series(
                flat_background_all[~np.isnan(flat_background_all)]
            )
        return self._flat_background

    @property
    def boxes(self):
        if self._boxes is None:
            self._boxes = self.pred['boxes'][self.pred['scores'] > self.thr]
        return self._boxes

    @property
    def particles(self):
        if self._particles is None:
            self._particles = [
                Particle(img=self.img, mask=cur_mask, box=cur_box, score=score)
                for cur_mask, cur_box, score in
                zip(self.masks, self.boxes,
                    self.pred['scores'][self.pred['scores'] > self.thr])
            ]
        return self._particles

    @property
    def particles_dict(self):
        return {part.ori_hash: part for part in self.particles}


class Particle:
    def __init__(self, img: np.ndarray,
                 mask: np.ndarray, box: np.ndarray,
                 score: Optional[float] = None):
        self.source_img = img
        self.mask = mask
        self.box = box
        join_hash = (hashlib.sha256(self.source_img).hexdigest() +
                     hashlib.sha256(self.box).hexdigest() +
                     hashlib.sha256(self.mask).hexdigest())
        self.hash = hashlib.sha256(join_hash.encode()).hexdigest()[:7]

        self._flatten: Optional[pd.Series] = None
        self._ori_hash: Optional[str] = None
        self.renamed = False
        self.score = score

    @property
    def ori_hash(self):
        if self._ori_hash is None:
            return self.hash
        else:
            return self._ori_hash

    @ori_hash.setter
    def ori_hash(self, new_hash: str):
        self.renamed = True
        self._ori_hash = new_hash

    def plot_particle(self, ax=None):
        box = self.box.astype(int)
        if ax is not None:
            ax.imshow(self.source_img[box[1]: box[3],
                                      box[0]: box[2]],
                      cmap='gray')
        plt.imshow(self.source_img[box[1]: box[3],
                                   box[0]: box[2]],
                   cmap='gray')

    def plot_diff_mask(self, thr):
        box = self.box.astype(int)
        img = copy.deepcopy(self.source_img)
        core_mask = np.where(self.mask > 0.5, img[:, :, 1] < thr,
                             False).astype(int)
        print(core_mask.sum())
        img[:, :, 0] = core_mask * 200
        img[:, :, 2] = (self.mask > 0.5).astype(int) * 100
        plt.imshow(img[box[1]: box[3], box[0]: box[2]],
                   cmap='gray')

    def plot_img_with_mask(self):
        mask = self.mask
        img = copy.deepcopy(self.source_img)
        img[:, :, 0] = ((mask > 0.5) * 230).astype(int)
        plt.imshow(img)

    @property
    def flatten(self) -> pd.Series:
        if self._flatten is None:
            one_part = np.where(
                self.mask > 0.5,
                self.source_img[:, :, 1],
                np.nan).flatten()
            self._flatten = pd.Series(
                one_part[~np.isnan(one_part)])
        return self._flatten

    def int_rel(self):
        return ((self.flatten - self.flatten.min()) /
                (self.flatten.max() - self.flatten.min())).mean()

    def int_bg(self, q: float = 0.05):
        box = self.box.astype(int)
        bounded_pic = self.source_img[box[1]: box[3],
                                      box[0]: box[2]]
        bouonded_mask = self.mask[box[1]: box[3],
                                  box[0]: box[2]]
        flatten_bg_bounded = np.where(bouonded_mask < 0.5,
                                      bounded_pic[:, :, 1], np.nan).flatten()
        flatten_bg_bounded = flatten_bg_bounded[~np.isnan(flatten_bg_bounded)]
        q = 0.05
        ser_bg = pd.Series(flatten_bg_bounded)
        return ((ser_bg - ser_bg.quantile(q)) /
                (ser_bg.quantile(1 - q) - ser_bg.quantile(q))).mean()


def get_particles_hist(
    all_particals: Dict[str, Particle], pictures: Dict[str, OnePic],
    part_hash: str
) -> Dict[pd.Timestamp, Dict[str, Union[float, str]]]:
    part_int_hist = {}
    for img_name, part in all_particals[part_hash].items():
        min_img = min(PIC_TO_DATETIME.keys() & all_particals[part_hash].keys())
        min_int = all_particals[part_hash][min_img].flatten.min()
        sof_norm_flatten = (part.flatten - min_int) / (MAX_INTENSITY - min_int)
        normed_flatten = (
            (part.flatten - pictures[img_name].flat_background.min()) /
            (MAX_INTENSITY - pictures[img_name].flat_background.min())
        )
        mean_int = normed_flatten.mean()
        sof_int = sof_norm_flatten.mean()
        sof_int_min = sof_norm_flatten.min()
        qs = part.flatten.quantile([0, 0.01, 0.5, 0.8]).to_dict()
        qs = {f'q{int(k * 100)}': val for k, val in qs.items()}
        qs.update({'mean_int': mean_int, 'sof_int': sof_int,
                   'sof_int_min': sof_int_min,
                   'std': part.flatten.std(),
                   'std_normed': normed_flatten.std(),
                   'img_name': img_name})
        part_int_hist[pd.Timestamp(PIC_TO_DATETIME[img_name])] = qs
    return part_int_hist
