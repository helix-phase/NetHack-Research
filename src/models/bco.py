import pyrallis
from dataclasses import dataclass, asdict
import random
import wandb
import os
import uuid
import torch
import torch.nn as nn

from gym.vector import AsyncVectorEnv
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
from tqdm.auto import tqdm, trange
from torch.distributions import Categorical
import numpy as np

from multiprocessing import set_start_method
from katakomba.env import NetHackChallenge, OfflineNetHackChallengeWrapper
from katakomba.nn.chaotic_dwarf import TopLineEncoder, BottomLinesEncoder, ScreenEncoder
from katakomba.utils.render import SCREEN_SHAPE, render_screen_image
from katakomba.utils.datasets import SequentialBuffer
from katakomba.utils.misc import Timeit
from typing import Optional, Tuple, List, Dict

# Import our custom adapter
from nao_dataset_adapter import extend_offline_wrapper

# Extend the wrapper to support human data
extend_offline_wrapper()

torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    character: str = "mon-hum-neu"
    data_mode: str = "compressed"
    data_dir: Optional[str] = None  # Custom directory for HDF5 files

    # Set to "human" to use NLD-NAO human data
    data_scale: str = "human"

    # Wandb logging
    project: str = "NetHack"
    group: str = "human_bc_inverse"
    name: str = "bc_inverse"
    version: int = 0

    # Model
    rnn_hidden_dim: int = 1024
    rnn_layers: int = 1
    use_prev_action: bool = True
    rnn_dropout: float = 0.0

    # Inverse model settings
    inverse_model_weight: float = 1.0
    use_difference_vector: bool = False
    inverse_model_warmup: int = 1000

    # Training
    update_steps: int = 25_000
    batch_size: int = 256
    seq_len: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = None
    checkpoints_path: Optional[str] = None
    eval_every: int = 5_000
    eval_episodes: int = 25
    eval_processes: int = 8
    render_processes: int = 8
    eval_seed: int = 50
    train_seed: int = 42

    def __post_init__(self):
        self.group = f"{self.group}-v{str(self.version)}"
        self.name = f"{self.name}-{self.character}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.group, self.name)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def filter_wd_params(model: nn.Module) -> Tuple[List[nn.parameter.Parameter], List[nn.parameter.Parameter]]:
    no_decay, decay = [], []
    for name, param in model.named_parameters():
        if hasattr(param, 'requires_grad') and not param.requires_grad:
            continue
        if 'weight' in name and 'norm' not in name and 'bn' not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    assert len(no_decay) + len(decay) == len(list(model.parameters()))
    return no_decay, decay


def dict_to_tensor(data: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in data.items()}


class InverseModel(nn.Module):
    def __init__(
            self,
            h_dim: int,
            action_space: int,
            use_difference_vector: bool = False
    ):
        super(InverseModel, self).__init__()
        self.h_dim = h_dim
        self.use_difference_vector = use_difference_vector
        if not use_difference_vector:
            self.h_dim *= 2
        self.action_space = action_space

        self.fwd_model = nn.Sequential(
            nn.Linear(self.h_dim, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, action_space),
        )

    def forward(self, obs):
        """
        Input:
            obs: encoded states of shape [T, B, h_dim]

        Output:
            action_logits: predicted actions of shape [T-1, B, action_space]
        """
        T, B, _ = obs.shape
        if self.use_difference_vector:
            x = obs[1:] - obs[:-1]  # [T-1, B, h_dim]
        else:
            x = torch.cat([obs[:-1], obs[1:]], dim=-1)  # [T-1, B, 2*h_dim]

        action_logits = self.fwd_model(x)  # [T-1, B, action_space]
        return action_logits


class Actor(nn.Module):
    def __init__(
            self,
            action_dim: int,
            rnn_hidden_dim: int = 512,
            rnn_layers: int = 1,
            rnn_dropout: float = 0.0,
            use_prev_action: bool = True,
            use_difference_vector: bool = False
    ):
        super().__init__()
        # Action dimensions and prev actions
        self.num_actions = action_dim
        self.use_prev_action = use_prev_action
        self.prev_actions_dim = self.num_actions if self.use_prev_action else 0

        # Encoders
        self.topline_encoder = TopLineEncoder()
        self.bottomline_encoder = torch.jit.script(BottomLinesEncoder())

        screen_shape = (SCREEN_SHAPE[1], SCREEN_SHAPE[2])
        self.screen_encoder = torch.jit.script(ScreenEncoder(screen_shape))

        self.h_dim = sum(
            [
                self.topline_encoder.hidden_dim,
                self.bottomline_encoder.hidden_dim,
                self.screen_encoder.hidden_dim,
                self.prev_actions_dim,
            ]
        )
        # Policy
        self.rnn = nn.LSTM(
            self.h_dim,
            rnn_hidden_dim,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            batch_first=True
        )
        self.head = nn.Linear(rnn_hidden_dim, self.num_actions)

        # Inverse model for action prediction
        self.inverse_model = InverseModel(
            h_dim=rnn_hidden_dim,
            action_space=self.num_actions,
            use_difference_vector=use_difference_vector
        )

    def forward(self, inputs, state=None):
        B, T, C, H, W = inputs["screen_image"].shape
        topline = inputs["tty_chars"][..., 0, :]
        bottom_line = inputs["tty_chars"][..., -2:, :]

        encoded_state = [
            self.topline_encoder(
                topline.float(memory_format=torch.contiguous_format).view(T * B, -1)
            ),
            self.bottomline_encoder(
                bottom_line.float(memory_format=torch.contiguous_format).view(T * B, -1)
            ),
            self.screen_encoder(
                inputs["screen_image"]
                .float(memory_format=torch.contiguous_format)
                .view(T * B, C, H, W)
            ),
        ]
        if self.use_prev_action:
            encoded_state.append(
                F.one_hot(inputs["prev_actions"], self.num_actions).view(T * B, -1)
            )

        encoded_state = torch.cat(encoded_state, dim=1)
        core_output, new_state = self.rnn(encoded_state.view(B, T, -1), state)
        policy_logits = self.head(core_output)

        # Get inverse model predictions
        core_output_t = core_output.transpose(0, 1)  # [T, B, h_dim]
        inverse_logits = self.inverse_model(core_output_t)  # [T-1, B, action_space]

        return policy_logits, inverse_logits, new_state, core_output

    @torch.no_grad()
    def vec_act(self, obs, state=None, device="cpu"):
        inputs = {
            "tty_chars": torch.tensor(obs["tty_chars"][:, None], device=device),
            "screen_image": torch.tensor(obs["screen_image"][:, None], device=device),
            "prev_actions": torch.tensor(obs["prev_actions"][:, None], dtype=torch.long, device=device)
        }
        policy_logits, _, new_state, _ = self(inputs, state)
        actions = torch.argmax(policy_logits.squeeze(1), dim=-1)
        return actions.cpu().numpy(), new_state


@torch.no_grad()
def vec_evaluate(
        vec_env: AsyncVectorEnv,
        actor: Actor,
        num_episodes: int,
        seed: str = 0,
        device: str = "cpu"
) -> Dict[str, np.ndarray]:
    actor.eval()
    # set seed for reproducibility (reseed=False by default)
    vec_env.seed(seed)
    n_envs = vec_env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_depths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    episode_count_targets = np.array([(num_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = vec_env.reset()
    observations["prev_actions"] = np.zeros(n_envs, dtype=float)

    rnn_states = None
    pbar = tqdm(total=num_episodes)
    while (episode_counts < episode_count_targets).any():
        observations["screen_image"] = render_screen_image(
            tty_chars=observations["tty_chars"][:, np.newaxis, ...],
            tty_colors=observations["tty_colors"][:, np.newaxis, ...],
            tty_cursor=observations["tty_cursor"][:, np.newaxis, ...],
        )
        observations["screen_image"] = np.squeeze(observations["screen_image"], 1)

        actions, rnn_states = actor.vec_act(observations, rnn_states, device=device)

        observations, rewards, dones, infos = vec_env.step(actions)
        observations["prev_actions"] = actions

        current_rewards += rewards
        current_lengths += 1

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                if dones[i]:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_depths.append(infos[i]["current_depth"])
                    episode_counts[i] += 1
                    pbar.update(1)

                    current_rewards[i] = 0
                    current_lengths[i] = 0

    pbar.close()
    result = {
        "reward_median": np.median(episode_rewards),
        "reward_mean": np.mean(episode_rewards),
        "reward_std": np.std(episode_rewards),
        "reward_min": np.min(episode_rewards),
        "reward_max": np.max(episode_rewards),
        "reward_raw": np.array(episode_rewards),
        # depth
        "depth_median": np.median(episode_depths),
        "depth_mean": np.mean(episode_depths),
        "depth_std": np.std(episode_depths),
        "depth_min": np.min(episode_depths),
        "depth_max": np.max(episode_depths),
        "depth_raw": np.array(episode_depths),
    }
    actor.train()
    return result


@pyrallis.wrap()
def train(config: TrainConfig):
    print(f"Device: {DEVICE}")
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
        save_code=True,
    )
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    set_seed(config.train_seed)

    def env_fn():
        env = NetHackChallenge(
            character=config.character,
            observation_keys=["tty_chars", "tty_colors", "tty_cursor"]
        )
        env = OfflineNetHackChallengeWrapper(env)
        return env

    tmp_env = env_fn()
    eval_env = AsyncVectorEnv(
        env_fns=[env_fn for _ in range(config.eval_processes)],
        copy=False
    )

    # Load dataset using the extended wrapper
    dataset = tmp_env.get_dataset(
        scale=config.data_scale,
        mode=config.data_mode,
        data_dir=config.data_dir
    )

    buffer = SequentialBuffer(
        dataset=dataset,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        seed=config.train_seed,
        add_next_step=True  # We need next steps for the inverse model
    )

    tp = ThreadPoolExecutor(max_workers=config.render_processes)

    actor = Actor(
        action_dim=eval_env.single_action_space.n,
        use_prev_action=config.use_prev_action,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_layers=config.rnn_layers,
        rnn_dropout=config.rnn_dropout,
        use_difference_vector=config.use_difference_vector
    ).to(DEVICE)

    no_decay_params, decay_params = filter_wd_params(actor)
    optim = torch.optim.AdamW([
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": decay_params, "weight_decay": config.weight_decay}
    ], lr=config.learning_rate)
    print("Number of parameters:", sum(p.numel() for p in actor.parameters()))

    scaler = torch.cuda.amp.GradScaler()

    rnn_state = None
    prev_actions = torch.zeros((config.batch_size, 1), dtype=torch.long, device=DEVICE)

    # Define cross-entropy loss function for the inverse model
    ce_loss = nn.CrossEntropyLoss()

    # Keep track of best inverse actions for updating the dataset
    best_inverse_actions = {}

    for step in trange(1, config.update_steps + 1, desc="Training"):
        with Timeit() as timer:
            batch = buffer.sample()
            screen_image = render_screen_image(
                tty_chars=batch["tty_chars"],
                tty_colors=batch["tty_colors"],
                tty_cursor=batch["tty_cursor"],
                threadpool=tp,
            )
            batch["screen_image"] = screen_image
            batch = dict_to_tensor(batch, device=DEVICE)

        wandb.log(
            {
                "times/batch_loading_cpu": timer.elapsed_time_cpu,
                "times/batch_loading_gpu": timer.elapsed_time_gpu,
            },
            step=step,
        )

        with Timeit() as timer