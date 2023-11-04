import json
import logging
import random
from hashlib import sha256
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from torchvision.models.detection import maskrcnn_resnet50_fpn, faster_rcnn
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def load_model(caches_model_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Load model from caches_model_path or from pretrained
    Parameters
    ----------
    caches_model_path: Path
        Path to model
    device: torch.device
        Device for model

    Returns
    -------
    torch.nn.Module
        Model
    """
    model = maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
        in_features, num_classes=2)
    if caches_model_path is not None:
        model.load_state_dict(torch.load(caches_model_path))
    return model.to(device)


class PowderNN:
    def __init__(self, annotation_path: Path, dataset_path: Path,
                 image_size: Tuple[int, int], seed: int,
                 #  epochs: int,
                 batch_size: int,
                 val_file_names=tuple(),
                 caches_model_path: Optional[Path] = None):
        self.annotation_path = annotation_path
        self.dataset_path = dataset_path
        self.coco = COCO(annotation_path)
        self.seed = seed
        # self.epochs = epochs
        self.cat_ids = self.coco.getCatIds()
        self._train_images_info = None
        self._val_images_info = None
        self.batch_size = batch_size
        self.image_size = image_size
        self.val_file_names = val_file_names

        self.device = (torch.device('cuda') if torch.cuda.is_available()
                       else torch.device('cpu'))

        self._model = None
        self.caches_model_path = caches_model_path
        self._optimizer = None
        self.train_looses = []
        self.train_looses = []
        self.val_looses = []
        self.cur_epoch = 0

    @property
    def train_images_info(self):
        if self._train_images_info is None:
            self._train_images_info = pd.DataFrame.from_dict(
                self.coco.imgs, orient='index')
            self._train_images_info = self._train_images_info[
                ~self._train_images_info['file_name'].isin(self.val_file_names)
            ]
        return self._train_images_info
    
    @property
    def val_images_info(self):
        if self._val_images_info is None:
            self._val_images_info = pd.DataFrame.from_dict(
                self.coco.imgs, orient='index')
            self._val_images_info = self._val_images_info[
                self._val_images_info['file_name'].isin(self.val_file_names)
            ]
        return self._val_images_info

    @property
    def model(self):
        """
        Get model, if model is None load model from caches_model_path or
        from pretrained
        Returns
        -------

        """
        if self._model is None:
            # load an instance segmentation model pre-trained
            # pre-trained on COCO
            model = load_model(caches_model_path=self.caches_model_path,
                               device=self.device)
            self._optimizer = torch.optim.AdamW(params=model.parameters(),
                                                lr=1e-5)
            self._model = model
        return self._model

    def one_epoch_data_gen(self, l_seed: int, val_mode: bool = False,
                           images_info: Optional[pd.DataFrame] = None):
        if images_info is not None:
            imgs = images_info
        else:
            if val_mode:
                imgs = self.val_images_info.copy().sample(
                        frac=1, random_state=l_seed)
            else:
                imgs = self.train_images_info.copy().sample(
                        frac=1, random_state=l_seed)
        batch_size = imgs.shape[0] if val_mode else self.batch_size
        if batch_size == 0:
            return Tuple[np.ndarray, np.ndarray, np.ndarray]
        for start_index in range(0, imgs.shape[0], batch_size):
            batch_imgs = []
            batch_data = []  # load images and masks
            file_names = []
            ldf = imgs.iloc[start_index: start_index + batch_size]
            for idx, row in ldf.iterrows():
                cur_path = self.dataset_path / row['file_name']
                if not cur_path.exists():
                    raise KeyError(f"file {cur_path} doesnt exist")
                img = cv2.imread(str(cur_path))
                img = cv2.resize(img, self.image_size, cv2.INTER_LINEAR)
                anns_ids = self.coco.getAnnIds(
                    imgIds=row['id'], catIds=self.cat_ids,
                    iscrowd=None
                )
                anns = self.coco.loadAnns(anns_ids)
                boxes = []
                masks = []
                # labels = []
                for ann in anns:
                    if ann['category_id'] != 2:
                        continue
                    ves_mask = cv2.resize(self.coco.annToMask(ann),
                                          self.image_size,
                                          cv2.INTER_NEAREST)
                    masks.append(ves_mask)
                    x, y, w, h = cv2.boundingRect(ves_mask)
                    boxes.append(np.array([x, y, x + w, y + h]))
                    # labels.append(ann['category_id'])
                # print(labels)
                data = dict(
                    masks=torch.as_tensor(np.array(masks), dtype=torch.uint8),
                    boxes=torch.as_tensor(np.array(boxes), dtype=torch.float32),
                    # labels=torch.as_tensor(np.array(labels),
                    #                        dtype=torch.int64),
                    labels=torch.ones((len(boxes),), dtype=torch.int64),
                )
                batch_imgs.append(torch.as_tensor(img, dtype=torch.float32))
                batch_data.append(data)
                file_names.append(row['file_name'])
            batch_imgs = torch.stack(
                [torch.as_tensor(d) for d in batch_imgs], 0
            )
            batch_imgs = batch_imgs.swapaxes(1, 3).swapaxes(2, 3)
            yield batch_imgs, batch_data, file_names

    def one_epoch_fit(self, l_seed: int):
        # torch.cuda.empty_cache()
        self.model.train()
        epoch_losses = {}
        genirator = self.one_epoch_data_gen(l_seed=l_seed, val_mode=False)
        for batch_imgs, batch_data, file_names in genirator:
            files_json = json.dumps(file_names).encode()
            images = list(image.to(self.device) for image in batch_imgs)
            targets = [{k: v.to(self.device) for k, v in t.items()}
                       for t in batch_data]
            self._optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            self._optimizer.step()
            self.train_looses.append({
                'epoch': self.cur_epoch,
                'file_names': file_names,
                'losses': {k: v.cpu().item() for k, v in loss_dict.items()},
                'batch_hash': sha256(files_json).hexdigest()
            })
            epoch_losses[self.cur_epoch] = losses.item()
        return epoch_losses

    def validate(self):
        # torch.cuda.empty_cache()
        self.model.eval()
        val_losses = {}
        genirator = self.one_epoch_data_gen(l_seed=self.seed, val_mode=True)
        for batch_imgs, batch_data, file_names in genirator:
            files_json = json.dumps(file_names).encode()
            images = list(image.to(self.device) for image in batch_imgs)
            targets = [{k: v.to(self.device) for k, v in t.items()}
                       for t in batch_data]
            # self._optimizer.zero_grad()
            losses = self.model(images, targets)
            losses_full = np.mean([
                loss['scores'].mean().cpu().detach().numpy().mean()
                for loss in losses
            ])
            self.val_looses.append({
                'epoch': self.cur_epoch,
                'file_names': file_names,
                'losses': [{k: val.cpu().detach() for k, val in loss.items()}
                           for loss in losses],
                'batch_hash': sha256(files_json).hexdigest()
            })
            val_losses[self.cur_epoch] = losses_full
        return val_losses

    def fit(self, epochs: int):
        self.model.train()
        np.random.seed(self.seed)
        seeds = np.random.randint(low=0, high=epochs, size=epochs)
        for epoch in tqdm(range(epochs)):
            self.one_epoch_fit(l_seed=seeds[epoch])
            # val_losses = self.validate()
            # LOGGER.info(f"epoch {self.cur_epoch}, val_losses: {val_losses}")
            self.cur_epoch += 1
        torch.cuda.empty_cache()


def get_next_batch(dataset_path: Path, image_size: Tuple[int, int],
                   epochs: int, batch_size: int, annotation_path: Path,
                   random_state: Optional[int] = None):
    coco = COCO(annotation_path)
    cat_ids = coco.getCatIds()
    np.random.seed(random_state)
    random_states = np.random.randint(low=0, high=epochs, size=epochs)
    for epoch, rs in zip(range(epochs), random_states):
        print(epoch)
        imgs = pd.DataFrame.from_dict(
            coco.imgs, orient='index').sample(
                frac=1, random_state=rs)
        for start_index in range(0, imgs.shape[0], batch_size):
            batch_imgs = []
            batch_data = []  # load images and masks
            ldf = imgs.iloc[start_index: start_index + batch_size]
            for idx, row in ldf.iterrows():
                cur_path = dataset_path / row['file_name']
                if not cur_path.exists():
                    raise KeyError(f"file {cur_path} doesnt exist")
                img = cv2.imread(str(cur_path))
                img = cv2.resize(img, image_size, cv2.INTER_LINEAR)
                anns_ids = coco.getAnnIds(imgIds=row['id'], catIds=cat_ids,
                                          iscrowd=None)
                anns = coco.loadAnns(anns_ids)
                boxes = []
                masks = []
                # labels = []
                for ann in anns:
                    if ann['category_id'] != 2:
                        continue
                    ves_mask = cv2.resize(coco.annToMask(ann), image_size,
                                          cv2.INTER_NEAREST)
                    masks.append(ves_mask)
                    x, y, w, h = cv2.boundingRect(ves_mask)
                    boxes.append(np.array([x, y, x + w, y + h]))
                    # labels.append(ann['category_id'])
                # print(labels)
                data = dict(
                    masks=torch.as_tensor(masks, dtype=torch.uint8),
                    boxes=torch.as_tensor(boxes, dtype=torch.float32),
                    # labels=torch.as_tensor(np.array(labels),
                    #                        dtype=torch.int64),
                    labels=torch.ones((len(boxes),), dtype=torch.int64),
                )
                batch_imgs.append(torch.as_tensor(img, dtype=torch.float32))
                batch_data.append(data)
            batch_imgs = torch.stack(
                [torch.as_tensor(d) for d in batch_imgs], 0
            )
            batch_imgs = batch_imgs.swapaxes(1, 3).swapaxes(2, 3)
            yield batch_imgs, batch_data


def load_data(batch_size: int, image_size: Tuple[int, int],
              imgs: List[Path]):
    batch_imgs = []
    batch_data = []  # load images and masks
    for i in range(batch_size):
        idx = random.randint(0, len(imgs)-1)
        img = cv2.imread(str(imgs[idx] / "Image.jpg"))
        img = cv2.resize(img, image_size, cv2.INTER_LINEAR)
        mask_dir = imgs[idx] / "Vessels"
        masks = []
        masks_paths = [p for p in mask_dir.glob("*.png")]
        num_objs = len(masks_paths)

        if num_objs == 0:
            # if image have no objects just load another
            return load_data(batch_size=batch_size, image_size=image_size,
                             imgs=imgs)

        boxes = torch.zeros([num_objs, 4], dtype=torch.float32)
        for row_n, msk_name in enumerate(masks_paths):
            ves_mask = (cv2.imread(str(mask_dir / msk_name), 0) > 0).astype(
                np.uint8)  # Read vesse instance mask
            ves_mask = cv2.resize(ves_mask, image_size, cv2.INTER_NEAREST)
            # get bounding box coordinates for each mask
            masks.append(ves_mask)

            x, y, w, h = cv2.boundingRect(ves_mask)
            boxes[row_n] = torch.tensor([x, y, x + w, y + h])
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        img = torch.as_tensor(img, dtype=torch.float32)
        data = dict(
            boxes=boxes,
            # there is only one class
            labels=torch.ones((num_objs,), dtype=torch.int64),
            masks=masks
        )
        batch_imgs.append(img)
        batch_data.append(data)  # load images and masks
    batch_imgs = torch.stack([torch.as_tensor(d) for d in batch_imgs], 0)
    batch_imgs = batch_imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_imgs, batch_data
