import torch
from torch import nn

from sample_factory.algorithms.appo.model_utils import create_encoder, create_core
from sample_factory.algorithms.appo.model import _ActorCriticSharedWeights
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import AttrDict


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, device, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape).to(device)
        self.var = torch.ones(shape).to(device)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RewardSharedWeights(_ActorCriticSharedWeights):
    def __init__(self, make_encoder, make_core, seq_len, action_space, cfg, timing):
        super().__init__(make_encoder, make_core, action_space, cfg, timing)
        self.seq_len = seq_len

        self.core_output_size = self.encoder.encoder_out_size
        self.reward_fn = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.core_output_size, 1),
        )

        self.device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')

        self.apply(self.initialize_weights)
        self.train()

    def add_mean_var(self, mean, var):
        self.mean = mean
        self.var = var

    def forward(self, mb, normalize=False, add_dim=False):
        for key, value in mb.items():
            if key in ['obs', 'message', 'glyphs', self.encoder.bl_stats_key]:
                if add_dim:
                    mb[key] = torch.tensor(value[None, ...]).to('cuda')
                else:
                    mb[key] = value.to('cuda')

        x = self.forward_head(mb, normalize=normalize)
        rewards = self.reward_fn(x)

        result = AttrDict(dict(rewards=rewards,))
        return result

    def forward_pairs(self, mb):
        for key, value in mb.items():
            if key in ['obs', 'message', 'glyphs', self.encoder.bl_stats_key]:
                mb[key] = value.to('cuda')

        batch_size = len(mb['message'].reshape(-1, 2, self.seq_len, mb['message'].shape[-1]))

        x = self.forward_head(mb)
        x = x.reshape(batch_size * 2, self.seq_len, -1) # Batch size x 2, sequence length, *
        x = x.transpose(0, 1) # sequence length, Batch size x 2, *
        rewards = self.reward_fn(x)
        rewards = rewards.reshape(self.seq_len, batch_size, 2) # sequence length, Batch size, 2

        result = AttrDict(dict(rewards=rewards,))
        return result


def create_reward_model(cfg, obs_space, action_space, seq_len=1, timing=None):
    if timing is None:
        timing = Timing()

    def make_encoder():
        return create_encoder(cfg, obs_space, timing, cfg.reward_encoder)

    def make_core(encoder):
        return create_core(cfg, encoder.get_encoder_out_size(), False)

    if cfg.actor_critic_share_weights:
        return RewardSharedWeights(make_encoder, make_core, seq_len, action_space, cfg, timing)
    else:
        raise NotImplementedError
