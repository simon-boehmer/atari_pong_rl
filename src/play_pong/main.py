# src/play_pong/main.py
import logging
import sys
import time
from typing import Tuple, Any
from pathlib import Path

import ale_py
import gymnasium as gym
import pygame
import numpy as np
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import AtariPreprocessing

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
from ..utils.config import config
from .env import FrameStack, make_env  # Relative imports

# MODEL_PATH = config.CHECKPOINT_DIR / "qrdqn_pong_2200000_steps.zip"  # Interim checkpoint (commented out)
MODEL_PATH = config.BEST_MODEL_DIR / "best_model.zip"
SCALE = 4
WIN_W, WIN_H = 160 * SCALE, 210 * SCALE


def initialize_pygame() -> Tuple[pygame.Surface, pygame.time.Clock]:
    """Initialize Pygame and set up the display.

    Returns:
        Tuple of the screen surface and clock object.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("RL Agent as RIGHT Paddle vs AI (LEFT)")
    return screen, pygame.time.Clock()


def run_game_loop(
    env: VecNormalize, model: QRDQN, screen: pygame.Surface, clock: pygame.time.Clock
) -> None:
    """Run the main game loop with the RL agent.

    Args:
        env: The normalized vectorized environment.
        model: The trained QRDQN model.
        screen: Pygame surface for rendering.
        clock: Pygame clock for frame rate control.
    """
    obs = env.reset()
    done = False
    step = 0

    while not done:
        try:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            frame = env.envs[0].unwrapped.render()
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            surf = pygame.transform.scale(surf, (WIN_W, WIN_H))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)
            step += 1
            if step % 30 == 0:
                logger.info(f"Step {step}, Action: {action}, Reward: {rewards[0]}")
            done = dones[0]
        except Exception as e:
            logger.error(f"Error in game loop: {e}")
            break

    logger.info("Game over.")


def wait_for_window() -> None:
    """Wait for the Pygame window to appear."""
    screen = pygame.display.get_surface()
    if screen:
        screen.fill((0, 0, 0))
        pygame.display.flip()
        logger.info(">> Waiting for window to appear (1s)...")
        start = time.time()
        while time.time() - start < 1.0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()


def main() -> None:
    """Main function to run the Atari Pong RL game."""
    try:
        logger.info(f"Checking for model at: {MODEL_PATH}")  # Debug log
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Checkpoint {MODEL_PATH} not found.")
        env = make_env()
        model = QRDQN.load(str(MODEL_PATH), env=env)
        logger.info(f"Loaded model from {MODEL_PATH}")

        screen, clock = initialize_pygame()
        wait_for_window()
        run_game_loop(env, model, screen, clock)

        logger.info("Press any key or close the window to exit.")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
    except FileNotFoundError as e:
        logger.error(f"Model or checkpoint not found: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        if "env" in locals():
            env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
