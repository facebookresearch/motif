
import csv
import json
import os
import pickle
import sys
import tqdm
from collections import defaultdict, deque
from functools import partial

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from rlaif.dataset import flatten_pair_collate_fn, PairsDataset
from rlaif.reward_model import create_reward_model, RunningMeanStd
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.utils.arguments import (arg_parser, parse_args, 
    maybe_load_from_checkpoint)
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import AttrDict, cfg_file, log, str2bool
from utils.preprocessing import (DictAsAttributes, 
                                 GPT5BaselineTransform, 
                                 ToTensorDict)

# Needs to be imported to register models and envs
import rl_baseline.tasks_nle
import rl_baseline.encoders_nle
import rl_baseline.env_nle


def validate(
    reward_model, 
    loss_fn, 
    validation_loader, 
    pref_type_key, 
    device, 
    iteration, 
    val_met
):
    cur_val_loss = 0.
    validation_acc = 0.
    score_validation_acc = 0.

    for mb in tqdm.tqdm(validation_loader):
        labels = mb[pref_type_key].to(device).type(torch.float)
         # Label greater than 1 indicates no preference
        labels[torch.where(labels > 1)[0]] = 0.5

        with torch.no_grad():
            result = reward_model.forward_pairs(mb)
        # sequence length x BS x 2
        rewards = result.rewards
        rewards = rewards.mean(axis=0)

        soft_labels = torch.zeros(len(rewards), 2, device=device)
        soft_labels[:, 1] = labels
        soft_labels[:, 0] = 1. - labels
        predicted_log_probs = nn.functional.log_softmax(rewards, dim=1)

        val_loss = loss_fn(predicted_log_probs, soft_labels)
        cur_val_loss += val_loss.item()

        # Only measure accuracy on pairs where the annotator has a preference
        reward_argmax = np.argmax(rewards.detach().cpu().numpy(), axis=1)
        labels = labels.cpu().numpy()
        validation_acc += np.mean(reward_argmax[labels != 0.5] == 
                                  labels[labels != 0.5])
        score_labels = np.argmax(mb['score'], axis=1).cpu().numpy()
        score_validation_acc += np.mean(reward_argmax == score_labels)

    # Save and log validation metrics
    val_met.iter.append(iteration)
    val_met.score_validation_accs.append(
        score_validation_acc / len(validation_loader)
    )
    val_met.validation_accs.append(
        validation_acc / len(validation_loader)
    )
    val_met.total_val_loss.append(
        cur_val_loss / len(validation_loader)
    )

    log.info(
        f"Iteration {iteration} "
        f"Score Validation accuracy: {val_met.score_validation_accs[-1]:.3f}\n"
        f"Iteration {iteration} "
        f"Validation accuracy: {val_met.validation_accs[-1]:.3f}\n"
        f"Iteration {iteration} "
        f"Validation loss: {val_met.total_val_loss[-1]:.3f}"
    )

    return val_met


def train_reward(cfg):
    """
    This code will train the reward model, through binary cross entropy,
    to express the preferences over trajectories as in the dataset.
    """

    cfg = maybe_load_from_checkpoint(cfg)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)
    dummy_env = make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0}))

    # Save experiment configuration
    with open(cfg_file(cfg), 'w') as json_file:
        json.dump(cfg, json_file, indent=2)

    # Prepare keys for dataset
    pref_type_key = f'{cfg.annotator}_pref'
    data_keys = ['blstats', 'message', 'tty_chars', 'tty_colors', 'rewards']
    info_keys = ['score', 'turns', 'experience_level', 'character']

    # Don't collate these keys
    collate_ignore_keys = [
        pref_type_key, 'score', 'steps', 'turns', 'level_num',
        'experience_level', 'character', 'idx'
    ]

    # Create the dataset
    dataset = PairsDataset(
        cfg.dataset_dir,
        data_keys=data_keys,
        info_keys=info_keys,
        preference_keys=[cfg.annotator],
        transform=transforms.Compose([
            GPT5BaselineTransform(),
            ToTensorDict()
        ])
    )

    # Load the dataset loader
    train_dataset, validation_dataset = torch.utils.data.random_split(
       dataset=dataset, 
       lengths=[0.8, 0.2], 
       generator=torch.Generator().manual_seed(cfg.seed)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=partial(
            flatten_pair_collate_fn, 
            ignore_keys=collate_ignore_keys
        ),
        num_workers=cfg.num_workers,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=partial(
            flatten_pair_collate_fn, 
            ignore_keys=collate_ignore_keys
        ),
        num_workers=cfg.num_workers,
    )

    # Create the reward model

    # Infer the observation space
    dummy_batch = next(iter(train_loader))
    obs_space = {}
    for key, value in dummy_batch.items():
        if key not in collate_ignore_keys and 'pref' not in key:
            obs_space[key] = np.array(value)[0]

    reward_model = create_reward_model(
        cfg, AttrDict(obs_space), dummy_env.action_space
    )
    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    reward_model.model_to_device(device)

    optimizer = torch.optim.Adam(
        reward_model.parameters(), lr=cfg.reward_lr)

    loss_fn = lambda logprobs, target: \
        -(target * logprobs).sum() / logprobs.shape[0]

    # Define metrics
    train_metrics = {
        'epoch': [],
        'total_train_loss': [],
        'total_train_acc': [],
    }
    train_met = DictAsAttributes(train_metrics)
    val_metrics = {
        'iter': [],
        'total_val_loss': [],
        'score_validation_accs': [],
        'validation_accs': [],
    }
    val_met = DictAsAttributes(val_metrics)

    # Get all messages in training set. This is used for RMS calculation.
    all_msgs_input = get_all_messages(cfg, train_loader)

    # Training
    num_iter = 0
    for epoch in range(cfg.num_epochs):
        train_loss = 0.
        train_acc = 0.
        reward_rms = RunningMeanStd(device)

        for i, mb in enumerate(tqdm.tqdm(train_loader)):
            if num_iter % len(train_loader) == 0:
                full_reward_rms, all_msgs_rewards = get_rms(cfg, reward_model,
                                                            train_loader, 
                                                            all_msgs_input, 
                                                            device)

                log.info(f'\n Full Reward mean: {full_reward_rms.mean[0]:.3f} '
                         f'Full Reward variance: {full_reward_rms.var[0]:.3f}')

                val_met = validate(reward_model, loss_fn, validation_loader,
                                   pref_type_key, device, num_iter, val_met)

                save(cfg, num_iter, reward_model, optimizer, 
                     train_met._data_dict, val_met._data_dict, 
                     full_reward_rms, all_msgs_rewards)

            result = reward_model.forward_pairs(mb)
            rewards = result.rewards # sequence length x BS x 2
            reward_rms.update(rewards.reshape(-1, 1).detach())
            rewards = rewards.mean(axis=0)

            labels = mb[pref_type_key].to(device).type(torch.float) # BS
            labels[torch.where(labels > 1)[0]] = 0.5
            soft_labels = torch.zeros(len(rewards), 2, device=device)
            soft_labels[:, 1] = labels
            soft_labels[:, 0] = 1. - labels

            predicted_log_probs = nn.functional.log_softmax(rewards, dim=1)
            loss = loss_fn(predicted_log_probs, soft_labels)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()  # Perform back-propagation
            optimizer.step()  # Update the weights

            reward_argmax = np.argmax(rewards.detach().cpu().numpy(), axis=1)
            labels = labels.cpu().numpy()
            train_acc += np.mean(
                reward_argmax[labels != 0.5] == labels[labels != 0.5])
            cur_acc = train_acc / (i+1)

            num_iter += 1

        train_met.total_train_loss.append(train_loss / len(train_loader))
        train_met.total_train_acc.append(train_acc / len(train_loader))
        train_met.epoch.append(epoch)

        log.info(f"Epoch {epoch} "
                 f"Training accuracy: {train_met.total_train_acc[-1]} "
                 f"Train loss: {train_met.total_train_loss[-1]}")

    log.info(f'Saving final model...')

    full_reward_rms, all_msgs_rewards = get_rms(cfg, reward_model, 
                                                train_loader, all_msgs_input, 
                                                device)

    save(cfg, num_iter, reward_model, optimizer, train_met._data_dict, 
         val_met._data_dict, full_reward_rms, all_msgs_rewards)


def save(
    cfg, 
    num_iter, 
    reward_model, 
    optimizer, 
    train_met, 
    val_met, 
    full_reward_rms, 
    all_msgs_rewards
):
    log.info(f'Saving at iter {num_iter}...')
    exp_path = os.path.join(cfg.train_dir, cfg.experiment)

    # Saving train satistics
    with open(f'{exp_path}/train_metrics.csv', 'w', newline='') as csvfile:
        fieldnames = list(train_met.keys())
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        num_rows = len(train_met[fieldnames[0]])
        for i in range(num_rows):
            row_data = [train_met[key][i] for key in fieldnames]
            writer.writerow(row_data)

    with open(f'{exp_path}/val_metrics.csv', 'w', newline='') as csvfile:
        fieldnames = list(val_met.keys())
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        num_rows = len(val_met[fieldnames[0]])
        for i in range(num_rows):
            row_data = [val_met[key][i] for key in fieldnames]
            writer.writerow(row_data)

    # Saving checkpoint
    checkpoint = {
        'num_iter': num_iter,
        'model': reward_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'reward_mean': full_reward_rms.mean[0],
        'reward_var': full_reward_rms.var[0],
    }
    checkpoint_dir = LearnerWorker.checkpoint_dir(cfg, policy_id=0)
    tmp_filepath = os.path.join(checkpoint_dir, 'temp_checkpoint.pth')
    checkpoint_name = f'checkpoint_{num_iter}.pth'
    filepath = os.path.join(checkpoint_dir, checkpoint_name)

    log.info('Saving %s...', tmp_filepath)
    torch.save(checkpoint, tmp_filepath)
    os.rename(tmp_filepath, filepath)

    metrics_folder = f'{exp_path}/reward_metrics'
    os.makedirs(metrics_folder, exist_ok=True)

    # Save quantiles of the reward function .
    # This will be used in the RL training loop.
    all_msgs_rewards_norm = (all_msgs_rewards - full_reward_rms.mean
                             ) / torch.sqrt(full_reward_rms.var)
    quantiles = [i / 100 for i in range(5, 96, 5)]

    rew_norm_quantiles = []
    for quantile in quantiles: 
        rew_norm_quantiles.append(
            torch.quantile(all_msgs_rewards_norm, quantile).item()
        )
    rew_norm_quantiles = [f'{q:.2f}' for q in rew_norm_quantiles]
    csv_file = f'{metrics_folder}/train_norm_quantiles.csv'
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if num_iter == 0:
            writer.writerow(quantiles)
        writer.writerow(rew_norm_quantiles)


def get_all_messages(cfg, train_loader):
    file = f"{cfg.dataset_dir}/train_split_info/sorted_messages.pkl"

    if os.path.exists(file):
        # Check if the messages have already been sorted and saved
        with open(file, "rb") as f:
            sorted_info = pickle.load(f)
        all_msgs = []
        for info in sorted_info:
            all_msgs.append(info[1])
        all_msgs_input = {}
        all_msgs_input['message'] = torch.stack(all_msgs)
    else:
        # Create the sorted list of messages if it doesn't exist
        log.info('Sorting all training messages...')
        train_msgs = defaultdict(int)
        train_bytes = defaultdict(int)
        total_size = float(train_loader.batch_size * len(train_loader))

        for mb in tqdm.tqdm(train_loader):
            messages = mb['message'].reshape(-1, 256)
            for message in messages:
                msg = bytes(message)
                train_msgs[msg] += 1
                if msg not in train_bytes:
                    train_bytes[msg] = message
            assert len(train_msgs) == len(train_bytes)

        directory = f"{cfg.dataset_dir}/train_split_info/"
        os.makedirs(directory, exist_ok=True)

        # Useful for debugging
        sorted_items = sorted(train_msgs.items(), key=lambda x: x[1], 
                              reverse=True)
        with open(f"{directory}/messages_stats.txt", "w") as f:
            for key, value in sorted_items: 
                percentage = value / total_size * 100
                decoded_key = key.decode('utf-8')
                f.write(f"{value} ({percentage:.2f}%): {decoded_key}\n")

        all_info = []
        all_msgs = []
        for key, value in sorted_items:
            all_info.append((key.decode('utf-8'), train_bytes[key], value))
            all_msgs.append(train_bytes[key])
        with open(file, "wb") as f:
            pickle.dump(all_info, f)

        all_msgs_input = {}
        all_msgs_input['message'] = torch.stack(all_msgs)
    return all_msgs_input


def get_rms(cfg, reward_model, train_loader, all_msgs_input, device):
    reward_rms = RunningMeanStd(device)
    log.info("Calculating RMS of the reward function...")
    with torch.no_grad():
        for mb in tqdm.tqdm(train_loader):
            result = reward_model.forward_pairs(mb)
            rewards = result.rewards # sequence length x BS x 2
            reward_rms.update(rewards.reshape(-1, 1))
            all_msgs_rewards = reward_model(all_msgs_input, normalize=False)

    return reward_rms, all_msgs_rewards.rewards.flatten()


def add_extra_params(parser):
    """
    Specify any additional command line arguments.
    """
    p = parser
    p.add_argument("--reward_lr", default=1e-4, type=float, 
                   help="Reward model learning rate")
    p.add_argument("--num_epochs", default=5, type=int, 
                   help="The number of epochs to train the reward model.")
    p.add_argument("--annotator", default='score', type=str, 
                   help="The annotator used for the preferences")
    p.add_argument("--dataset_dir", default='motif_dataset', type=str, 
                   help="Directory from which we load the dataset.")


def parse_all_args(argv=None, evaluation=True):
    parser = arg_parser(argv, evaluation=evaluation)
    add_extra_params(parser)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg


def main():
    """Evaluation entry point."""
    cfg = parse_all_args()
    train_reward(cfg)
    return


if __name__ == '__main__':
    sys.exit(main())
