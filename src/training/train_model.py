from datetime import datetime
from pathlib import Path

import torch

from constants import IMG_SIZE
from data_loader import PowderNN
from scrip_utils import get_kwargs, get_logger, get_yaml_config

if __name__ == '__main__':
    default_config_path = (Path(__file__).parent /
                           f'{Path(__file__).stem}_config.yml')
    kwargs = get_kwargs(default_config_path=default_config_path).parse_args()
    LOGGER = get_logger(logger_name=Path(__file__).stem,
                        level=kwargs.logger_level)
    LOGGER.info('start training with config: %s', kwargs.config_path)
    config = get_yaml_config(kwargs.config_path)
    dataset_path = Path(config['dataset_path'])
    annotation_path = Path(config['annotation_path'])
    validation_file_names: list[str] = config['validation_file_names']

    destionation_path = Path(config.get('destionation_path', "."))
    LOGGER.info("destionation_path: %s", destionation_path)

    caches_model_path = config.get('caches_model_path')
    caches_model_path = Path(caches_model_path) if caches_model_path else None

    seed = config['seed']
    batch_size = config['batch_size']
    epochs = config['epochs']
    torch.cuda.empty_cache()
    LOGGER.info("spawning PowderNN")
    pnn = PowderNN(
        annotation_path=annotation_path,
        dataset_path=dataset_path,
        image_size=IMG_SIZE,
        seed=seed,
        batch_size=batch_size,
        val_file_names=validation_file_names,
        caches_model_path=caches_model_path
    )
    LOGGER.info("start training")
    pnn.fit(epochs=epochs)
    LOGGER.info("training finished")
    new_name = f"powder_{datetime.now().strftime('%Y%m%d_%H%M')}.torch"
    torch.save(pnn.model.state_dict(),
               destionation_path / new_name)
    LOGGER.info("model saved as %s", new_name)
    LOGGER.info("finish")
