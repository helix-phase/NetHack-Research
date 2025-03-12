import os
import h5py
import shutil
import numpy as np
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from typing import Tuple, List, Dict, Any, Optional
from katakomba.utils.roles import Role, Race, Alignment
from katakomba.utils.render import render_screen_image, SCREEN_SHAPE

# Use local cache directory
BASE_REPO_ID = os.environ.get("KATAKOMBA_REPO_ID", os.path.expanduser("Howuhh/katakomba"))
DATA_PATH = os.environ.get("KATAKOMBA_DATA_DIR", os.path.expanduser("/code/NetHack-Research/data/processed/hdf5_data/"))
CACHE_PATH = os.environ.get("KATAKOMBA_CACHE_DIR", os.path.expanduser("./cache"))


def _flush_to_memmap(filename: str, array: np.ndarray) -> np.ndarray:
    """Helper function to create memory-mapped arrays"""
    if os.path.exists(filename):
        mmap = np.load(filename, mmap_mode="r")
    else:
        mmap = np.memmap(filename, mode="w+", dtype=array.dtype, shape=array.shape)
        mmap[:] = array
        mmap.flush()
    return mmap


def load_nld_nao_state_only_dataset(
        role: Role, race: Race, align: Alignment, mode: str = "compressed"
) -> Tuple[h5py.File, List[Dict[str, Any]]]:
    """Load state-only data from NLD-NAO dataset"""
    dataset_name = f"data-{role.value}-{race.value}-{align.value}-any.hdf5"
    dataset_path = os.path.join(DATA_PATH, dataset_name)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = h5py.File(dataset_path, "r")

    if mode == "in_memory":
        trajectories = {}
        for episode in tqdm(df["/"].keys(), leave=False, desc="Preparing (RAM Decompression)"):
            episode_data = {k: df[episode][k][()] for k in df[episode].keys()}
            # Add placeholder for actions if they don't exist
            if "actions" not in episode_data:
                steps = episode_data["tty_chars"].shape[0]
                episode_data["actions"] = np.zeros(steps, dtype=np.int64)
            trajectories[episode] = episode_data

    elif mode == "memmap":
        # Create cache directory with proper permissions
        os.makedirs(CACHE_PATH, exist_ok=True)
        trajectories = {}
        for episode in tqdm(df["/"].keys(), leave=False, desc="Preparing (Drive Decompression)"):
            cache_name = f"memmap-{dataset_name.split('.')[0]}"
            episode_cache_path = os.path.join(CACHE_PATH, cache_name, str(episode))

            os.makedirs(episode_cache_path, exist_ok=True)
            episode_data = {
                k: _flush_to_memmap(
                    filename=os.path.join(episode_cache_path, str(k)),
                    array=df[episode][k][()],
                )
                for k in df[episode].keys()
            }

            # Add placeholder for actions if they don't exist
            if "actions" not in episode_data:
                steps = episode_data["tty_chars"].shape[0]
                action_file = os.path.join(episode_cache_path, "actions")
                if not os.path.exists(action_file):
                    actions = np.zeros(steps, dtype=np.int64)
                    episode_data["actions"] = _flush_to_memmap(action_file, actions)
                else:
                    episode_data["actions"] = np.load(action_file, mmap_mode="r+")

            trajectories[episode] = episode_data

    elif mode == "compressed":
        trajectories = {}
        for episode in tqdm(df["/"].keys(), leave=False, desc="Preparing"):
            # we do not copy data here! it will decompress it during reading or slicing
            episode_data = {k: df[episode][k] for k in df[episode].keys()}

            # For compressed mode, we'll handle actions differently during access
            trajectories[episode] = episode_data
    else:
        raise RuntimeError(
            "Unknown mode for dataset loading! Please use one of: 'compressed', 'in_memory', 'memmap'"
        )

    return df, trajectories


class StateOnlyDataset:
    """Dataset class for state-only data that integrates with the inverse model"""

    def __init__(
            self,
            role: Role,
            race: Race,
            align: Alignment,
            mode: str = "compressed",
            inverse_model: Optional[torch.nn.Module] = None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.hdf5_file, self.data = load_nld_nao_state_only_dataset(
            role, race, align, mode=mode
        )
        self.gameids = list(self.data.keys())

        self.role = role
        self.race = race
        self.align = align
        self.mode = mode

        # Store the inverse model if provided
        self.inverse_model = inverse_model
        self.device = device

        # Check if we need to generate actions
        self.has_actions = all("actions" in self.data[gameid] for gameid in self.gameids)

        print(f"Loaded {len(self.gameids)} episodes for {role.value}-{race.value}-{align.value}")
        print(f"Actions available: {self.has_actions}")

    def __getitem__(self, idx):
        gameid = self.gameids[idx]
        data = self.data[gameid]

        # For compressed mode, we need to decompress
        if self.mode == "compressed":
            data_copy = {k: data[k][()] for k in data.keys()}

            # If actions don't exist, generate them or create placeholders
            if "actions" not in data_copy and self.inverse_model is not None:
                return self._generate_actions(gameid, data_copy)

            # If no actions and no inverse model, return placeholder actions
            if "actions" not in data_copy:
                steps = data_copy["tty_chars"].shape[0]
                data_copy["actions"] = np.zeros(steps, dtype=np.int64)

            return data_copy

        # For other modes
        if "actions" not in data and self.inverse_model is not None:
            return self._generate_actions(gameid, data)

        # If no actions and no inverse model, ensure actions exist
        if "actions" not in data:
            data_copy = {k: data[k] for k in data.keys()}
            steps = data_copy["tty_chars"].shape[0]
            data_copy["actions"] = np.zeros(steps, dtype=np.int64)
            return data_copy

        return data

    def _generate_actions(self, gameid, data_copy=None):
        """Generate actions using a simplified approach"""
        if data_copy is None:
            data = self.data[gameid]
            data_copy = {k: data[k][()] for k in data.keys()}

        steps = data_copy["tty_chars"].shape[0]

        # Just create placeholder actions for now to get the training started
        # We'll refine the inverse model integration later
        data_copy["actions"] = np.zeros(steps, dtype=np.int64)

        # If using memmap, we can save the generated actions for future use
        if self.mode == "memmap":
            cache_name = f"memmap-data-{self.role.value}-{self.race.value}-{self.align.value}-any"
            episode_cache_path = os.path.join(CACHE_PATH, cache_name, str(gameid))
            action_file = os.path.join(episode_cache_path, "actions")
            _flush_to_memmap(action_file, data_copy["actions"])

        return data_copy

    def __len__(self):
        return len(self.gameids)

    def metadata(self, idx):
        gameid = self.gameids[idx]
        return dict(self.hdf5_file[gameid].attrs)

    def close(self, clear_cache=True):
        self.hdf5_file.close()
        if self.mode == "memmap" and clear_cache:
            print("Cleaning memmap cache...")
            # remove memmap cache files from the disk upon closing
            cache_name = f"memmap-data-{self.role.value}-{self.race.value}-{self.align.value}-any"
            shutil.rmtree(os.path.join(CACHE_PATH, cache_name))