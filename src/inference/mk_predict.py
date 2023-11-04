import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from constants import IMG_SIZE
from data_loader import load_model
from powder_describer import OnePic
from scrip_utils import get_kwargs, get_logger, get_yaml_config

if __name__ == '__main__':
    default_config_path = (Path(__file__).parent /
                           f'{Path(__file__).stem}_config.yml')
    kwargs = get_kwargs(default_config_path=default_config_path).parse_args()
    LOGGER = get_logger(logger_name=Path(__file__).stem,
                        level=kwargs.logger_level)
    LOGGER.info('start training with config: %s', kwargs.config_path)
    config = get_yaml_config(kwargs.config_path)
    # prepare data
    dataset_path = Path(config['dataset_path'])
    imgs = []
    file_paths_all = [p for p in dataset_path.glob("*.jpg")]
    LOGGER.info("found %s images", len(file_paths_all))
    file_paths_all.sort()
    # file_paths = file_paths_all[17:]  # file_paths_all[:17]
    for cur_path in file_paths_all:
        img: np.ndarray = cv2.imread(str(cur_path))
        img = cv2.resize(img, np.array(IMG_SIZE), cv2.INTER_LINEAR)
        imgs.append(torch.as_tensor(img, dtype=torch.float32))
    imgs = torch.stack(
        [torch.as_tensor(d) for d in imgs], 0
    )
    imgs = imgs.swapaxes(1, 3).swapaxes(2, 3)

    model_path = Path(config['model_path'])
    device = torch.device("cuda" if config['use_cuda'] else "cpu")
    model = load_model(caches_model_path=model_path, device=device)
    model.eval()
    with torch.no_grad():
        local_imgs = list(image.to(device) for image in imgs)
        local_names = [p.stem for p in file_paths_all]
        preds = model(local_imgs)
    pictures = {}
    for t_num, l_name in enumerate(local_names):
        image = cv2.resize(cv2.imread(str(file_paths_all[t_num])), IMG_SIZE,
                           cv2.INTER_LINEAR)
        hash_id = hashlib.sha256(image).hexdigest()[:5]
        pictures[f"{l_name}_{hash_id}"] = OnePic(img=image, pred=preds[t_num],
                                                 thr=0.97)
    pd.to_pickle(pictures, Path(config['destionation_path']) / "pictures.pkl")
    LOGGER.info("finish. Find pictures.pkl in %s", config['destionation_path'])
