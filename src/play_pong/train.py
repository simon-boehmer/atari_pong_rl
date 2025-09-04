# src/play_pong/train.py
import os
import multiprocessing as mp
from pathlib import Path
import logging
from typing import List, Optional, Tuple  # Added Tuple import

import numpy as np
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import VecNormalize

from ..utils.config import config
from .env import make_env

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStoppingCallback(BaseCallback):
    """Callback to stop training early if performance plateaus.

    Args:
        patience (int): Number of epochs to wait for improvement.
        min_delta (float): Minimum change to qualify as improvement.
        verbose (int): Verbosity level.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.1, verbose: int = 1):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -np.inf
        self.num_bad_epochs = 0

    def _on_step(self) -> bool:
        """Check for early stopping based on evaluation mean reward.

        Returns:
            bool: True to continue, False to stop.
        """
        if (
            hasattr(self.model, "logger")
            and "eval/mean_reward" in self.model.logger.name_to_value
        ):
            mean_reward = self.model.logger.name_to_value["eval/mean_reward"]
            if mean_reward > self.best_mean_reward + self.min_delta:
                self.best_mean_reward = mean_reward
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
                if self.num_bad_epochs > self.patience:
                    logger.info(
                        f"Early stopping triggered at step {self.num_timesteps}. Best mean reward: {self.best_mean_reward}"
                    )
                    return False
        return True


class VecNormSaveCallback(BaseCallback):
    """Callback to save VecNormalize statistics periodically.

    Args:
        vecnorm_env (VecNormalize): The environment to save.
        save_path (Path): Path to save the normalization file.
        save_freq (int): Frequency of saves in steps.
        verbose (int): Verbosity level.
    """

    def __init__(
        self,
        vecnorm_env: VecNormalize,
        save_path: Path,
        save_freq: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.vecnorm_env = vecnorm_env
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        """Save VecNormalize stats if the frequency is met.

        Returns:
            bool: True to continue.
        """
        if self.n_calls % self.save_freq == 0:
            self.vecnorm_env.save(str(self.save_path))
            if self.verbose > 0:
                logger.info(
                    f"[VecNormalize] Saved at step {self.num_timesteps} to {self.save_path}"
                )
        return True


class ProgressCallback(BaseCallback):
    """Callback to display training progress.

    Args:
        total_timesteps (int): Total timesteps for training.
        print_freq (int): Frequency of progress updates.
        verbose (int): Verbosity level.
    """

    def __init__(self, total_timesteps: int, print_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        """Log progress percentage.

        Returns:
            bool: True to continue.
        """
        if self.num_timesteps % self.print_freq == 0:
            pct = 100 * self.num_timesteps / self.total_timesteps
            logger.info(
                f"[Progress] {self.num_timesteps}/{self.total_timesteps} ({pct:.1f}%)"
            )
        return True


def setup_environments() -> Tuple[VecNormalize, VecNormalize]:
    """Set up training and evaluation environments.

    Returns:
        Tuple of training and evaluation VecNormalize environments.
    """
    train_env = make_env()
    eval_env = make_env()
    return train_env, eval_env


def configure_model(train_env: VecNormalize) -> QRDQN:
    """Configure and initialize the QRDQN model.

    Args:
        train_env: The training environment.

    Returns:
        Configured QRDQN model.
    """
    return QRDQN(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=config.LEARNING_RATE,
        buffer_size=config.BUFFER_SIZE,
        train_freq=config.TRAIN_FREQ,
        learning_starts=config.LEARNING_STARTS,
        batch_size=config.BATCH_SIZE,
        gamma=config.GAMMA,
        target_update_interval=config.TARGET_UPDATE_INTERVAL,
        optimize_memory_usage=True,
        replay_buffer_kwargs={"handle_timeout_termination": False},
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        exploration_fraction=0.1,
        policy_kwargs={"n_quantiles": 50, "normalize_images": False},
        tensorboard_log=str(config.LOG_DIR_TRAIN),
        verbose=1,
    )


def setup_callbacks(
    train_env: VecNormalize, eval_env: VecNormalize
) -> List[BaseCallback]:
    """Set up training callbacks.

    Args:
        train_env: The training environment.
        eval_env: The evaluation environment.

    Returns:
        List of callback instances.
    """
    checkpoint_cb = CheckpointCallback(
        save_freq=25_000 // config.TRAIN_FREQ,
        save_path=str(config.CHECKPOINT_DIR),
        name_prefix="qrdqn_pong",
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(config.BEST_MODEL_DIR),
        log_path=str(config.LOG_DIR_EVAL),
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )
    progress_cb = ProgressCallback(
        total_timesteps=config.TOTAL_TIMESTEPS,
        print_freq=max(1, config.TOTAL_TIMESTEPS // 10),
    )
    vecnorm_cb = VecNormSaveCallback(
        train_env,
        config.CHECKPOINT_DIR / "vecnormalize.pkl",
        save_freq=25_000 // config.TRAIN_FREQ,
        verbose=1,
    )
    early_stop_cb = EarlyStoppingCallback(patience=10, min_delta=0.1, verbose=1)
    return [checkpoint_cb, eval_cb, progress_cb, vecnorm_cb, early_stop_cb]


def main() -> None:
    """Main function to train the QRDQN model on Atari Pong."""
    try:
        # Create directories
        for dir_path in (
            config.CHECKPOINT_DIR,
            config.BEST_MODEL_DIR,
            config.LOG_DIR_TRAIN,
            config.LOG_DIR_EVAL,
        ):
            dir_path.mkdir(parents=True, exist_ok=True)

        # Set multiprocessing method
        mp.set_start_method("fork", force=True)

        # Setup environments and model
        train_env, eval_env = setup_environments()
        model = configure_model(train_env)
        callbacks = setup_callbacks(train_env, eval_env)

        # Train the model
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=callbacks,
            reset_num_timesteps=False,
        )

        # Final save
        train_env.save(str(config.CHECKPOINT_DIR / "vecnormalize.pkl"))
        model.save(str(config.MODEL_PATH))
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if "train_env" in locals():
            train_env.close()
        if "eval_env" in locals():
            eval_env.close()


if __name__ == "__main__":
    main()
