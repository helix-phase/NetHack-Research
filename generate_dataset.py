import os
import h5py
import numpy as np
import nle.dataset as nld

import zipfile
import random
import pyrallis
from dataclasses import dataclass
from typing import Optional

from tqdm.auto import tqdm
from collections import defaultdict


@dataclass
class Config:
    data_path: str = "/code/nld-nao/nld-nao-unzipped"
    save_path: str = "/code/NetHack-Research/data/raw/converted_data"
    race: Optional[str] = None
    role: Optional[str] = None
    alignment: Optional[str] = None
    gender: Optional[str] = None
    sampling: Optional[str] = None
    num_episodes: Optional[int] = None
    num_bins: int = 50
    random_seed: int = 32
    clean_db_after: bool = False
    compress_after: bool = False


def stratified_sample(x, scores, num_samples, num_bins=100):
    num_total = len(x)

    bins, edges = np.histogram(scores, bins=num_bins)
    assert sum(bins) == num_total, "change number of bins"
    n_strat_samples = [int(num_samples * (num_bin / num_total)) for num_bin in bins]

    bin_ids = np.digitize(scores, edges)

    sampled_ids = []
    for sample_size, bin_id in zip(n_strat_samples, range(1, num_bins + 1)):
        sample = np.random.choice(x[bin_ids == bin_id], size=sample_size, replace=False)
        assert sample.shape[0] == sample_size

        sampled_ids.extend(sample.tolist())

    return np.array(sampled_ids)


def load_game(dataset, game_id):
    raw_data = defaultdict(list)
    # Get the ttyrec data
    ttyrec_data = dataset.get_ttyrec(game_id, 1)[:-1]

    for step in ttyrec_data:
        raw_data["tty_chars"].append(step["tty_chars"].squeeze())
        raw_data["tty_colors"].append(step["tty_colors"].squeeze())
        raw_data["tty_cursor"].append(step["tty_cursor"].squeeze())
        raw_data["timestamps"].append(step["timestamps"].item() if "timestamps" in step else 0)
        raw_data["dones"].append(step["done"].item() if "done" in step else False)

    # Create the data dictionary
    data = {
        "tty_chars": np.stack(raw_data["tty_chars"]),
        "tty_colors": np.stack(raw_data["tty_colors"]),
        "tty_cursor": np.stack(raw_data["tty_cursor"]),
        "timestamps": np.array(raw_data["timestamps"]).astype(np.int64),
        "dones": np.array(raw_data["dones"]).astype(bool)
    }

    # Make sure the last step is marked as done
    if len(data["dones"]) > 0:
        data["dones"][-1] = True

    return data


def optional_eq(x, cond):
    if cond is not None:
        return x == cond
    return True


def name(role, race, align, gender):
    return f"{role or 'any'}-{race or 'any'}-{align or 'any'}-{gender or 'any'}"


@pyrallis.wrap()
def main(config: Config):
    os.makedirs(config.save_path, exist_ok=True)

    dbfilename = "/code/NetHack-Research/data/raw/nld-nao.db"
    if not nld.db.exists(dbfilename):
        nld.db.create(dbfilename)
        nld.add_altorg_directory(config.data_path, "nld-nao", dbfilename)

    dataset = nld.TtyrecDataset(
        "nld-nao",
        batch_size=1,
        seq_length=1,
        dbfilename=dbfilename,
    )
    # retrieving and filtering metadata from the dataset
    metadata = {game_id: dict(dataset.get_meta(game_id)) for game_id in dataset._gameids}
    metadata = {
        k: v for k, v in metadata.items() if (
                optional_eq(v.get("role", "").lower(), config.role) and
                optional_eq(v.get("race", "").lower(), config.race) and
                optional_eq(v.get("align", "").lower(), config.alignment) and
                optional_eq(v.get("gender", "").lower(), config.gender)
        )
    }
    file_name = name(config.role, config.race, config.alignment, config.gender)

    game_ids = np.array(list(metadata.keys()))
    assert len(game_ids) != 0, "dataset does not have episodes with such configuration"

    if config.sampling is not None and config.num_episodes is not None:
        scores = np.array([metadata[game_id].get("points", 0) for game_id in game_ids])

        if config.sampling == "stratify":
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

            game_ids = stratified_sample(game_ids, scores, config.num_episodes, num_bins=config.num_bins)
            print(f"Sampled {len(game_ids)} episodes with stratified sampling!")
        elif config.sampling == "sort":
            game_ids = game_ids[np.argsort(scores)][-config.num_episodes:]
            mean_score = np.mean(np.sort(scores)[-config.num_episodes:])
            print(f"Sampled episodes with top {config.num_episodes} scores. Mean score: {mean_score}")
        else:
            raise RuntimeError("Unknown sampling type")

    # saving episodes data as uncompressed hdf5
    with h5py.File(os.path.join(config.save_path, f"data-{file_name}.hdf5"), "w", track_order=True) as df:
        for ep_id in tqdm(game_ids):
            data = load_game(dataset, game_id=ep_id)

            g = df.create_group(str(ep_id))
            g.create_dataset("tty_chars", data=data["tty_chars"], compression="gzip")
            g.create_dataset("tty_colors", data=data["tty_colors"], compression="gzip")
            g.create_dataset("tty_cursor", data=data["tty_cursor"], compression="gzip")
            g.create_dataset("timestamps", data=data["timestamps"], compression="gzip")
            g.create_dataset("dones", data=data["dones"], compression="gzip")

            # also save metadata as attrs
            for key, value in metadata[ep_id].items():
                if isinstance(value, (str, int, float, bool)):
                    g.attrs[key] = value

    # clearing and compressing at the end
    if config.compress_after:
        hdf5_path = os.path.join(config.save_path, f"data-{file_name}.hdf5")

        with zipfile.ZipFile(f"{hdf5_path}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(os.path.join(config.save_path, f"data-{file_name}.hdf5"))
        os.remove(hdf5_path)

    if nld.db.exists(dbfilename) and config.clean_db_after:
        os.remove(dbfilename)


if __name__ == "__main__":
    main()