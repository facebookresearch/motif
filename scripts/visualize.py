import os
import sys
import time
from collections import deque

import gym
import numpy as np
import scipy
import torch

# Needs to be imported to register models and envs
import rl_baseline.tasks_nle
import rl_baseline.encoders_nle
import rl_baseline.env_nle

from rlaif.reward_model import create_reward_model
from sample_factory.algorithms.appo.actor_worker import (
    transform_dict_observations)
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.algo_utils import ExperimentStatus
from sample_factory.algorithms.utils.arguments import (arg_parser, parse_args,
    load_from_checkpoint)
from sample_factory.algorithms.utils.multi_agent_wrapper import (
    MultiAgentWrapper)
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict, str2bool


def enjoy(cfg, max_num_frames=1e6, target_num_episodes=100):
    """
    This is a modified version of original appo.enjoy_appo.enjoy function.
    """

    cfg = load_from_checkpoint(cfg)

    cfg.env_frameskip = 1
    cfg.num_envs = 1

    env_config = AttrDict({'worker_index': 0, 'vector_index': 0})
    env = create_env(cfg.env, cfg=cfg, env_config=env_config)

    # sample-factory defaults to work with multiagent environments,
    # but we can wrap a single-agent env into one of these like this
    env = MultiAgentWrapper(env)

    # Create actor critic
    actor_critic = create_actor_critic(
        cfg, 
        env.observation_space, 
        env.action_space
    )
    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic.model_to_device(device)

    # Load actor critic
    checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(
                                                cfg, cfg.policy_index))
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])

    # Create and potentially load a reward model
    rew_checkpoints = LearnerWorker.get_checkpoints(os.path.join(cfg.reward_dir,
                                                             f'checkpoint_p0'))

    if len(rew_checkpoints) > 0:
        checkpoint_dict = LearnerWorker.load_checkpoint(
            rew_checkpoints, 
            device, 
            checkpoint_num=cfg.checkpoint_num
        )

        # Make sure the action space is correct. 
        # The reward function doe snot use action information, but it has
        # dependencies on the action space through sample-factory.
        bias_weight_name = 'action_parameterization.distribution_linear.bias'
        action_space = checkpoint_dict['model'][bias_weight_name].shape[0]
        action_space = gym.spaces.Discrete(action_space)

        reward_model = create_reward_model(
            cfg, 
            env.observation_space, 
            action_space
        )
        reward_model.model_to_device(device)
        reward_model.load_state_dict(checkpoint_dict['model'])

        mean = checkpoint_dict['reward_mean'].item()
        var = checkpoint_dict['reward_var'].item()
        log.info("Reward function loaded...\n")
    else:
        reward_model = create_reward_model(
            cfg, 
            env.observation_space, 
            env.action_space
        )
        reward_model.model_to_device(device)

        mean = 0
        var = 1
        log.info("No reward function loaded...\n")

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    episode_reward = np.zeros(env.num_agents)
    finished_episode = [False] * env.num_agents

    episode_msgs = {}
    episode_num_frames = 0
    num_frames = 0
    num_episodes = 0
    agent_i = 0 # Hard-coded to 0 since this is not a multiagent environment.

    obs = env.reset()
    rnn_states = torch.zeros(
        [env.num_agents, get_hidden_size(cfg)], 
        dtype=torch.float32, 
        device=device
    )

    while num_frames < max_num_frames and num_episodes < target_num_episodes:
        with torch.no_grad():
            obs_torch = AttrDict(transform_dict_observations(obs))

            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(device).float()

            policy_outputs = actor_critic(obs_torch, 
                                          rnn_states)

            # sample actions from the distribution by default
            actions = policy_outputs.actions
            actions = actions.cpu().numpy()
            rnn_states = policy_outputs.rnn_states

            obs, rew, done, infos = env.step(actions)

            # Copy observation as it gets modified in other places
            reward_obs = obs[0].copy()
            int_reward = reward_model(
                reward_obs, 
                add_dim=True
            ).rewards.cpu().numpy()[0][0]
            msg = bytes(obs[0]['message'])
            norm_int_reward = (int_reward - mean) / (var)**(1/2)

            if msg not in episode_msgs:
                episode_msgs[msg] = (1, int_reward, norm_int_reward)
            else:
                count, int_r, norm_int_r = episode_msgs[msg]
                episode_msgs[msg] = (count+1, int_r, norm_int_r)

            episode_reward += rew
            num_frames += 1
            episode_num_frames += 1

            if done[agent_i]:
                finished_episode[agent_i] = True
                episode_rewards[agent_i].append(episode_reward[agent_i])

                print("\n" * 29)
                print("===============Top messages=================")
                # Sort the top 100 messages from the current episode
                sorted_msgs = sorted(
                    episode_msgs.items(), 
                    key=lambda x: x[1][0], 
                    reverse=True
                )[:100]
                for msg, value in sorted_msgs:
                    print(f' {msg.decode()} {value[2]:.3f} ')
                print("===============Top messages=================")

                print(f"Episode finished at {num_frames} frames. " 
                        f"Return: {episode_reward[agent_i]:.3f}")

                episode_msgs = {}
                episode_num_frames = 0
                episode_reward[agent_i] = 0
                num_episodes += 1
                rnn_states[agent_i] = torch.zeros(
                    [get_hidden_size(cfg)], 
                    dtype=torch.float32, 
                    device=device
                )
                input("Press 'Enter' to continue...")

            if cfg.render:
                # Print timestep stats
                print(f"Timestep: {num_frames} Reward: {rew[0]:.3f} "
                      f"Return: {episode_reward[0]:.3f} "
                      f"Intrinsic reward: {int_reward:.3f} "
                      f"Norm Intrinsic reward: {norm_int_reward:.3f}")

                # Render environment
                env.render()

                # Go back 27 lines for smooth streaming.
                print("\033[%dA" % 27)

                # Pause rendering a certain amount
                time.sleep(cfg.sleep)

    env.close()

    return ExperimentStatus.SUCCESS, np.mean(episode_rewards)


def add_extra_params(parser):
    """
    Specify additional command line arguments for visualizing the RL agent.
    """
    p = parser
    p.add_argument("--sleep", default=0.0, type=float, 
                   help="Controls the speed at which rendering happens.")
    p.add_argument("--render", default=True, type=str2bool, 
                   help="To render or not the game.")


def parse_all_args(argv=None, evaluation=True):
    parser = arg_parser(argv, evaluation=evaluation)
    add_extra_params(parser)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg


def main():
    """Evaluation entry point."""
    cfg = parse_all_args()
    _ = enjoy(cfg)
    return


if __name__ == '__main__':
    sys.exit(main())
