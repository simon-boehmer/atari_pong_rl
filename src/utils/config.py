# src/utils/config.py
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Config:
    """Configuration settings for the Atari Pong RL project.

    Attributes:
        ENV_ID (str): The environment ID (e.g., "PongNoFrameskip-v4").
        FRAME_STACK (int): Number of frames to stack for observation.
        N_ENVS (int): Number of parallel environments.
        TOTAL_TIMESTEPS (int): Total timesteps for training.
        BUFFER_SIZE (int): Replay buffer size.
        TRAIN_FREQ (int): Training frequency.
        LEARNING_STARTS (int): Steps before learning starts.
        BATCH_SIZE (int): Batch size for training.
        LEARNING_RATE (float): Learning rate for the model.
        GAMMA (float): Discount factor.
        TARGET_UPDATE_INTERVAL (int): Interval for target network updates.
        CHECKPOINT_FREQ (int): Frequency of checkpoint saving.
        EVAL_FREQ (int): Frequency of evaluation.
        EVAL_EPISODES (int): Number of episodes for evaluation.
        CHECKPOINT_DIR (Path): Directory for checkpoints.
        BEST_MODEL_DIR (Path): Directory for best models.
        LOG_DIR_TRAIN (Path): Directory for training logs.
        LOG_DIR_EVAL (Path): Directory for evaluation logs.
        MODEL_PATH (Path): Path to the final model.
    """

    ENV_ID: str = "PongNoFrameskip-v4"
    FRAME_STACK: int = 4
    N_ENVS: int = 4
    TOTAL_TIMESTEPS: int = int(2.5e6)
    BUFFER_SIZE: int = 200_000
    TRAIN_FREQ: int = 1
    LEARNING_STARTS: int = 50_000
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.00025
    GAMMA: float = 0.99
    TARGET_UPDATE_INTERVAL: int = 10_000
    CHECKPOINT_FREQ: int = 500_000
    EVAL_FREQ: int = 500_000
    EVAL_EPISODES: int = 10
    CHECKPOINT_DIR: Path = Path("outputs/checkpoints")
    BEST_MODEL_DIR: Path = Path("outputs/best_models")
    LOG_DIR_TRAIN: Path = Path("outputs/logs/train")
    LOG_DIR_EVAL: Path = Path("outputs/logs/eval")
    MODEL_PATH: Path = CHECKPOINT_DIR / "qrdqn_pong_final"

    def __post_init__(self):
        """Validate and create directories if they don't exist."""
        for dir_path in (
            self.CHECKPOINT_DIR,
            self.BEST_MODEL_DIR,
            self.LOG_DIR_TRAIN,
            self.LOG_DIR_EVAL,
        ):
            dir_path.mkdir(parents=True, exist_ok=True)


# Export default instance
config = Config()
