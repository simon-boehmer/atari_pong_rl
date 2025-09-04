# src/play_pong/env.py
from typing import Tuple, Any

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from collections import deque
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import AtariPreprocessing

from ..utils.config import config


class FrameStack(ObservationWrapper):
    """Stack the last n frames (grayscale 84×84) into a (n, 84, 84) array.

    Args:
        env: The base environment.
        n_stack (int): Number of frames to stack.

    Attributes:
        n_stack (int): Number of frames stacked.
        frames (deque): Deque to store frames.
    """

    def __init__(self, env, n_stack: int):
        super().__init__(env)
        self.n_stack = n_stack
        h, w = env.observation_space.shape
        low = np.repeat(env.observation_space.low[np.newaxis, ...], n_stack, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], n_stack, axis=0)
        self.observation_space = Box(
            low=low, high=high, shape=(n_stack, h, w), dtype=env.observation_space.dtype
        )
        self.frames = deque(maxlen=n_stack)

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset the environment and stack initial frames.

        Returns:
            Tuple containing the stacked observation and info dict.
        """
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Perform a step in the environment and stack the new frame.

        Args:
            action: The action to take.

        Returns:
            Tuple containing the stacked observation, reward, terminated, truncated, and info.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = np.clip(reward, -1, 1)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Get the stacked observation.

        Returns:
            Stacked array of frames.
        """
        assert len(self.frames) == self.n_stack
        return np.stack(self.frames, axis=0)


def make_env() -> VecNormalize:
    """Build a VecEnv of N_ENVS Pong environments with preprocessing.

    Returns:
        A normalized VecEnv with Atari preprocessing and frame stacking.
    """

    def _init() -> gym.Env:
        base = gym.make(
            config.ENV_ID, frameskip=1, render_mode="rgb_array"
        )  # Added render_mode
        env = Monitor(base)
        env = AtariPreprocessing(
            env,
            screen_size=84,
            grayscale_obs=True,
            scale_obs=False,
            noop_max=30,
            frame_skip=4,
        )
        env = FrameStack(env, n_stack=config.FRAME_STACK)
        return env

    venv = DummyVecEnv([_init for _ in range(config.N_ENVS)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return venv
