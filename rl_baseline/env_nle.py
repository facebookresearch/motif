import os
import gym
import numpy as np

import nle

from rl_baseline.obs_wrappers import RenderCharImagesWithNumpyWrapper, MessageWrapper, VectorFeaturesWrapper
from sample_factory.envs.env_registry import global_env_registry


class RootNLEWrapper(gym.Wrapper):
    """Some root-level additions to the NLE environment"""
    def __init__(self, env, env_id=-1):
        super().__init__(env)
        manual_spaces = {
            "tty_chars": gym.spaces.Box(0, 255, shape=(24, 80), dtype=np.uint8),
            "tty_colors": gym.spaces.Box(0, 31, shape=(24, 80), dtype=np.int8),
            "tty_cursor": gym.spaces.Box(0, 255, shape=(2,), dtype=np.uint8),
            "blstats": gym.spaces.Box(-2147483648, 2147483647, shape=(26,), dtype=np.int32),
            "message": gym.spaces.Box(0, 255, shape=(256,), dtype=np.uint8),
            "glyphs": gym.spaces.Box(0, 5976, (21, 79), dtype=np.int16),
        }
        self.env_id = np.array([env_id]).astype(np.int16)

        self.observation_space = gym.spaces.Dict(manual_spaces)

    def seed(self, *args):
        # Nethack does not allow seeding, so monkey-patch disable it here
        return

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs


def make_custom_env_func(full_env_name, cfg=None, env_config=None):
    root_env = cfg.root_env
    if env_config is not None and 'root_env' in env_config.keys():
        root_env = env_config['root_env']

    env = RootNLEWrapper(gym.make(
            root_env,
            save_ttyrec_every=cfg.save_ttyrec_every,
            savedir=cfg.save_dir,
            observation_keys=["tty_chars", "tty_colors", "tty_cursor", "blstats", "message", "glyphs"],
            ),
        )

    if full_env_name == "nle_fixed_eat_action":
        env = MessageWrapper(
                VectorFeaturesWrapper(
                    RenderCharImagesWithNumpyWrapper(
                        env, font_size=9, crop_size=12, rescale_font_size=(6, 6)
                    ),
                ),
            cfg=cfg,
            )
    else:
        raise ValueError(f"Env does not exist {full_env_name}")
    return env


global_env_registry().register_env(
    env_name_prefix='nle_',
    make_env_func=make_custom_env_func,
)
