from datetime import datetime
from pathlib import Path

import pandas as pd

from constants import BAD_PARTS
from powder_describer import get_particles_hist
from scrip_utils import get_kwargs, get_logger, get_yaml_config

if __name__ == "__main__":
    default_config_path = (Path(__file__).parent /
                           f'{Path(__file__).stem}_config.yml')
    kwargs = get_kwargs(default_config_path=default_config_path).parse_args()
    LOGGER = get_logger(logger_name=Path(__file__).stem, level=kwargs.logger_level)
    config = get_yaml_config(kwargs.config_path)
    LOGGER.info("start calculation with config: %s", kwargs.config_path)
    data_path = Path(config["data_path"])
    if data_path.suffix != ".pickle":
        raise ValueError("data_path should be .pickle file")
    pictures = pd.read_pickle(data_path)
    img_seq = pd.DataFrame([pictures.keys()], index=['first_img']).T
    img_seq['sec_img'] = img_seq['first_img'].shift(-1)
    LOGGER.info("there are %s images in sequence", img_seq.shape[0])

    destination_path = Path(config["destination_path"])
    all_particles = {}
    for img_name, picture in pictures.items():
        for part_hash, part in picture.particles_dict.items():
            cur_ser = all_particles.get(part_hash, {})
            cur_ser[img_name] = part
            all_particles[part_hash] = cur_ser

    result = {}
    for part_hash in all_particles.keys() - BAD_PARTS:
        part_int_hist = get_particles_hist(all_particals=all_particles,
                                           pictures=pictures,
                                           part_hash=part_hash)
        df = pd.DataFrame.from_dict(part_int_hist, orient='index')
        df['peak'] = (
                ((df['q1'].diff().diff().ffill().shift(-1) < -4.99)
                 & (df['q1'] > 80)
                 ) |
                (df['q1'] > 100) |
                ((df['std_normed'] < 0.18) & (
                            df['std_normed'].diff().abs() < 0.0001) & (
                             df['q1'] > 80))
        )
        img = pictures[df.sort_index().iloc[0]['img_name']]
        square = img.particles_dict[part_hash].mask.mean()
        time_end = pd.NaT
        picture_name = None
        if df['peak'].any():
            time_end = df.loc[df['peak']].iloc[0].name
            picture_name = df.loc[df['peak'], 'img_name'].iloc[0]
        result[part_hash] = {'square': square, 'time_end': time_end,
                             'last_pic_name': picture_name}
    result_df = pd.DataFrame(result).T
    result_df["square_mkm"] = result_df["square"] * config["photo_square"]
    result_df["radius_mkm"] = result_df["square_mkm"].apply(
        lambda x: np.sqrt(x / np.pi)
    )
    result_df["time_end"] = pd.to_datetime(result_df["time_end"])
    result_df["duration"] = result_df["time_end"] - datetime.strptime(
        config["start_time"], "%Y-%m-%d %H:%M:%S"
    )
    result_df["diffusion_(mkm2/sec)"] = (
            result_df["square_mkm"] / result_df["duration"].dt.total_seconds()
    )
    result_df["diffusion_(m2/sec)"] = result_df["diffusion_(mkm2/sec)"] * 1e-12
    result_df.to_excel(destination_path / "result.xlsx")

