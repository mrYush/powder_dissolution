from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from scrip_utils import get_kwargs, get_yaml_config, get_logger


def augmentation(frame: np.ndarray, random_state: int) -> np.ndarray:
    """
    Augmentation function for images
    Parameters
    ----------
    frame: np.ndarray
        image
    random_state: int
        random state

    Returns
    -------
    np.ndarray
        augmented image
    """
    if random_state % 2 == 1:
        # flip image horizontally in 50% of cases
        frame = frame[::-1]
    rows, cols, dim = frame.shape

    # add padding to image
    top = rows * 2
    bottom = rows * 2
    left = cols * 2
    right = cols * 2
    padding_image = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    np.random.seed(random_state)
    # random rotation
    trans_matrix = np.float32([[np.random.normal(1, 0.5), np.random.uniform(low=-0.5, high=0.5), 0],
                               [np.random.uniform(low=-0.5, high=0.5), np.random.normal(1, 0.5), 0],
                               [0, 0, 1]])
    translated_img = cv2.warpPerspective(padding_image, trans_matrix, padding_image.shape[:-1][::-1])
    my = (translated_img != 0).any(axis=2).any(axis=1)
    mx = (translated_img != 0).any(axis=2).any(axis=0)
    translated_img_cut = translated_img[my, :, :][:, mx,:]
    return translated_img_cut


if __name__ == "__main__":
    cur_path = Path(__file__)
    default_config_path = (cur_path.parent / f"{cur_path.stem}_config.yml")
    kwargs = get_kwargs(default_config_path=default_config_path).parse_args()
    LOGGER = get_logger(logger_name=cur_path.stem, level=kwargs.logger_level)
    LOGGER.info("start augmentation whith config: %s", kwargs.config_path)
    config = get_yaml_config(kwargs.config_path)
    source_path = Path(config["source_path"])
    destionation_path = Path(config["destination_path"])
    for one_img_path in tqdm(source_path.glob("*.jpg")):
        image = cv2.imread(str(one_img_path))
        for random_state in list(np.random.randint(low=0, high=100, size=2)):
            new_frame = augmentation(frame=image, random_state=random_state)
            if new_frame.shape[0] == 0:
                continue
            new_file_name = f"{one_img_path.stem}_aug_{random_state}.jpg"
            plt.imsave(fname=destionation_path / new_file_name, arr=new_frame)
    LOGGER.info("Augmentation finished. Find results in %s", destionation_path)
