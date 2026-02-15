# minigrid_kitchen.py
import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
# If the above fails, try:
# from gym_minigrid.wrappers import ImgObsWrapper

import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

import imageio
from pathlib import Path
import numpy as np
import random
import argparse
import os
from typing import List, Tuple

CURRENT_DIR = Path(__file__).parent
ENV_ID = "MiniGrid-Kitchen-v0"
SEED = 42

# --- MiniGrid imports for building custom env ---
# The exact import path may differ between installations (minigrid vs gym_minigrid).
# If these fail, change to: from gym_minigrid.minigrid import MiniGridEnv, Grid, Ball, Goal, Wall
try:
    from minigrid.minigrid import MiniGridEnv, Grid, Ball, Goal, Wall
except Exception:
    # fallback that often works in other installs
    from gym_minigrid.minigrid import MiniGridEnv, Grid, Ball, Goal, Wall


# -------------------------
# Custom MiniGrid Kitchen Env
# -------------------------
class KitchenMiniGridEnv(MiniGridEnv):
    """
    MiniGrid environment representing a tiny kitchen:
    - Dirty dishes are represented as red Ball objects.
    - Sink is a coordinate where agent can 'use' to rinse (we track rinsed status).
    - Dishwasher is a Goal tile: dropping a rinsed dish on the dishwasher counts as 'cleaned'.
    """

    def __init__(
        self,
        size: int = 5,
        num_dishes: Tuple[int, int] = (1, 3),
        max_steps: int = 100,
        seed: int = None,
        anti_loop_penalty: bool = False,
    ):
        self.kitchen_size = size
        self.num_dishes_range = num_dishes
        self.anti_loop_penalty = anti_loop_penalty

        # Observation / Agent view parameters:
        # We use fully observable (view size covers whole grid) for simplicity.
        super().__init__(
            width=size, height=size, max_steps=max_steps, see_through_walls=True
        )
        self.seed(seed)
        self.rinsed_ids = set()
        self.cleaned_count = 0
        self.dish_ids = []  # track object ids for dishes
        self._last_positions = []  # for anti-loop detection

    def _gen_grid(self, width, height):
        # create an empty grid
        self.grid = Grid(width, height)

        # Walls around
        self.grid.wall_rect(0, 0, width, height)

        # Place a couple of inner obstacles (counters)
        # Randomly choose some wall segments to place internal walls (as counters)
        # Make sure agent has a path.
        for i in range(1):  # keep kitchen simple
            # place a short wall
            wx = self._rand_int(1, width - 1)
            wy = self._rand_int(1, height - 1)
            if self.grid.get(wx, wy) is None:
                self.grid.set(wx, wy, Wall())

        # place dishwasher (Goal) at a random free cell near bottom-right
        dx = width - 2
        dy = height - 2
        self.put_obj(Goal(), dx, dy)
        self.dishwasher_pos = (dx, dy)

        # place a sink coordinate (we won't create a special object, just track coords)
        sx = 1
        sy = 1
        self.sink_pos = (sx, sy)

        # place dishes (Ball objects) randomly
        num_dishes = self._rand_int(self.num_dishes_range[0], self.num_dishes_range[1] + 1)
        self.dish_ids = []
        placed = 0
        while placed < num_dishes:
            x = self._rand_int(1, width - 1)
            y = self._rand_int(1, height - 1)
            if self.grid.get(x, y) is None and (x, y) != self.dishwasher_pos and (x, y) != self.sink_pos:
                ball = Ball(color="red")
                self.put_obj(ball, x, y)
                self.dish_ids.append(ball.cur_pos)
                placed += 1

        # place agent
        self.place_agent()
        # mission string (not used heavily)
        self.mission = "Clean the dishes: rinse at sink then drop them on dishwasher."

    def step(self, action):
        """
        Extend step to track rinsing/loading logic and custom rewards.
        Actions are the standard MiniGrid discrete actions.
        We'll add a special 'use' action: in MiniGrid the 'toggle' action (3) is typically used to interact.
        Reward logic:
          - +1 when a rinsed dish is dropped on dishwasher (i.e., loaded and rinsed)
          - small -0.01 per step
          - optional -0.5 if the agent repeatedly loops (anti-loop)
        """
        obs, reward, terminated, truncated, info = super().step(action)

        # base step penalty to encourage speed
        reward = reward - 0.01

        # record positions for loop detection
        pos = tuple(self.agent_pos)
        self._last_positions.append(pos)
        if len(self._last_positions) > 20:
            self._last_positions.pop(0)

        # If agent performed toggle (pickup / drop) we need to inspect inventory and cell
        # Note: MiniGrid sets reward=1 on reaching a Goal by default; we ignore that and implement custom.
        # We'll check if the agent is carrying an object and if it dropped (carrying becomes None).
        carried = getattr(self, "carrying", None)

        # Detect if the agent just dropped an object at dishwasher position:
        # If the agent is not carrying and the agent is at dishwasher pos, attempt to detect dish presence:
        ax, ay = self.agent_pos
        if (ax, ay) == self.dishwasher_pos:
            # scan cell for any Ball (dish)
            obj = self.grid.get(ax, ay)
            # Note: if Goal occupies the tile, get() returns goal; objects dropped may be in agent's cell pos
            # So as a heuristic, detect if agent previously was carrying and just dropped.
            # We'll check info dict for 'last_action' if available, else use a simple heuristic:
            if carried is None:
                # check if we have any object previously carried id in rinsed set:
                # The simplest approach: if rinsed_count > cleaned_count and the agent is at dishwasher, treat it as load event.
                # We'll scan neighbor cell for Ball objects (a robust impl would track object ids)
                # For simplicity: if a rinse flag exists in info, reward accordingly.
                pass

        # We'll compute rinsed status: if agent uses toggle (action == self.actions.toggle) at sink pos and is carrying ball -> mark rinsed
        if action == self.actions.toggle:
            # If agent is carrying a ball and is at sink, mark rinsed
            if pos == self.sink_pos and carried is not None and carried.type == "ball":
                # mark rinsed by attaching flag on object
                # MiniGrid objects don't have stable ids in this simple code, so we'll store a 'rinsed' flag on carrying object
                carried.rinsed = True
                # small reward for rinsing
                reward += 0.2

            # If agent toggled at dishwasher while carrying rinsed ball -> count as cleaned
            if pos == self.dishwasher_pos and carried is not None and getattr(carried, "rinsed", False):
                # agent loads and the environment will drop the object; give final reward
                reward += 1.0
                self.cleaned_count += 1
                # remove carried object
                self.carrying = None

        # Anti-loop penalty (simple heuristic): if agent repeats same small sequence of positions many times
        if self.anti_loop_penalty:
            if len(self._last_positions) >= 12:
                if len(set(self._last_positions[-6:])) <= 2:
                    reward -= 0.5  # discourage tight loops

        # termination: optional when all dishes cleaned
        # We approximate "all dishes cleaned" if cleaned_count >= initial number placed
        # Note: we do not keep exact initial count in this simple code; we approximate by checking grid for balls
        remaining_balls = 0
        for i in range(1, self.width - 1):
            for j in range(1, self.height - 1):
                o = self.grid.get(i, j)
                if o is not None and o.type == "ball":
                    remaining_balls += 1
        # Also check if agent carrying a ball
        if getattr(self, "carrying", None) is not None and self.carrying.type == "ball":
            remaining_balls += 1

        if remaining_balls == 0:
            terminated = True
            reward += 0.0  # no extra

        return obs, reward, terminated, truncated, info


# register env to gym
from gymnasium.envs.registration import register

try:
    register(
        id=ENV_ID,
        entry_point=lambda **kwargs: KitchenMiniGridEnv(**kwargs),
        max_episode_steps=100,
    )
except Exception:
    # If env was already registered, ignore
    pass


# -------------------------
# Feature extractor (same as your starter)
# -------------------------
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        # observation_space is image (C,H,W) after ImgObsWrapper, but Stable-Baselines expects channel-first images
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = observation_space.sample()[None]
            sample = th.as_tensor(sample).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)


# -------------------------
# Training / Eval / Recording
# -------------------------
def make_env(seed=SEED, anti_loop_penalty=False):
    def _init():
        env = gym.make(ENV_ID, size=5, num_dishes=(1, 3), max_steps=100, seed=seed, anti_loop_penalty=anti_loop_penalty)
        # ImgObsWrapper returns only the image observation (H,W,3) - we transpose below for SB3
        env = ImgObsWrapper(env)
        # SB3 expects (C,H,W) and VecTransposeImage will handle it in vectorized envs
        return env

    return _init


def train(policy="CnnPolicy", total_timesteps=200_000, seed=SEED, anti_loop=False, use_vec=True):
    """Train PPO model on the custom MiniGrid kitchen env"""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    if use_vec:
        vec_env = DummyVecEnv([make_env(seed=seed, anti_loop_penalty=anti_loop)])
        # transpose images to (C,H,W)
        vec_env = VecTransposeImage(vec_env)
        env = vec_env
    else:
        env = make_env(seed=seed)()

    model = PPO(
        policy,
        env,
        policy_kwargs=policy_kwargs if "Cnn" in policy else None,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        verbose=1,
        tensorboard_log=str(CURRENT_DIR / "logs"),
    )

    print("Starting training...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model_path = CURRENT_DIR / "model"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")
    return model


def evaluate(model, n_episodes=10, seed=SEED, anti_loop=False):
    """Evaluate model performance (returns mean, std)"""
    env = make_env(seed=seed, anti_loop_penalty=anti_loop)()
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes)
    print(f"\nEvaluation results ({n_episodes} episodes): Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    env.close()
    return mean_reward, std_reward


def record_video(model, filename="demo.mp4", seed=SEED, anti_loop=False, deterministic=True, max_steps=200):
    """Record a single evaluation rollout to a video file (mp4)"""
    env = make_env(seed=seed, anti_loop_penalty=anti_loop)()

    frames: List[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    # ImgObsWrapper env.render() returns an RGB array; obs returned is also the image
    frames.append(env.render(mode="rgb_array"))

    done = False
    steps = 0
    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render(mode="rgb_array"))
        done = terminated or truncated
        steps += 1

    output_path = CURRENT_DIR / filename
    imageio.mimsave(output_path, frames, fps=6)
    print(f"Video saved to {output_path}")
    env.close()


# -------------------------
# Simple CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["CnnPolicy", "MlpPolicy"], default="CnnPolicy")
    p.add_argument("--timesteps", type=int, default=200000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--eval_episodes", type=int, default=8)
    p.add_argument("--anti_loop", action="store_true", help="Enable anti-loop penalty (mitigation)")
    p.add_argument("--record", action="store_true", help="Record demo.mp4 after training")
    return p.parse_args()


def main():
    args = parse_args()
    print("Configuration:", args)

    model = train(policy=args.policy, total_timesteps=args.timesteps, seed=args.seed, anti_loop=args.anti_loop)

    evaluate(model, n_episodes=args.eval_episodes, seed=args.seed + 1, anti_loop=args.anti_loop)

    if args.record:
        record_video(model, filename="demo.mp4", seed=args.seed + 2, anti_loop=args.anti_loop)


if __name__ == "__main__":
    main()
