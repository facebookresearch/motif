import math

import nle
from nle import nethack
import torch
from torch import nn
import torch.nn.functional as F

from sample_factory.algorithms.appo.model_utils import (
    get_obs_shape, nonlinearity, create_standard_encoder, EncoderBase, register_custom_encoder
)
from sample_factory.utils.utils import log


NUM_CHARS = 256
PAD_CHAR = 0


def calc_conv_output_size(H, W, P, D, K, S, n_layers=2):
    for l in range(n_layers):
        H = math.floor((H + 2*P - D*(K-1) - 1)/S + 1)
        W = math.floor((W + 2*P - D*(K-1) - 1)/S + 1)
    return H * W


class Crop(nn.Module):
    """Helper class for NetHackNet below."""

    def __init__(self, height, width, height_target, width_target):
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        width_grid = _step_to_range(2 / (self.width - 1), self.width_target)[
            None, :
        ].expand(self.height_target, -1)
        height_grid = _step_to_range(2 / (self.height - 1), height_target)[
            :, None
        ].expand(-1, self.width_target)

        # "clone" necessary, https://github.com/pytorch/pytorch/issues/34880
        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def forward(self, inputs, coordinates):
        """Calculates centered crop around given x,y coordinates.
        Args:
           inputs [B x H x W]
           coordinates [B x 2] x,y coordinates
        Returns:
           [B x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        assert inputs.shape[1] == self.height
        assert inputs.shape[2] == self.width

        inputs = inputs[:, None, :, :].float()

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)
        
        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        # TODO: only cast to int if original tensor was int
        return (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
            .squeeze(1)
            .long()
        )


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def _step_to_range(delta, num_steps):
    """Range of `num_steps` integers with distance `delta` centered around zero."""
    return delta * torch.arange(-num_steps // 2, num_steps // 2)


class NLEMainEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        # Use standard CNN for the image observation in "obs"
        # See all arguments with "-h" to change this head to e.g. ResNet
        self.basic_encoder = create_standard_encoder(cfg, obs_space, timing)
        self.encoder_out_size = self.basic_encoder.encoder_out_size
        obs_shape = get_obs_shape(obs_space)
        self.bl_stats_key = 'vector_obs'
        # self.bl_stats_key = next((key for key in obs_space if "vector_obs" in key or "stats" in key), None)

        self.vector_obs_head = None
        self.message_head = None
    
        if self.bl_stats_key in obs_shape:
            self.vector_obs_head = nn.Sequential(
                nn.Linear(obs_shape[self.bl_stats_key][0], 128),
                nonlinearity(cfg),
                nn.Linear(128, 128),
                nonlinearity(cfg),
            )
            out_size = 128
            # Add vector_obs to NN output for more direct access by LSTM core
            self.encoder_out_size += out_size + obs_shape[self.bl_stats_key][0]
        if 'message' in obs_shape:
            # _Very_ poor for text understanding,
            # but it is simple and probably enough to overfit to specific sentences.
            self.message_head = nn.Sequential(
                nn.Linear(obs_shape.message[0], 128),
                nonlinearity(cfg),
                nn.Linear(128, 128),
                nonlinearity(cfg),
            )
            out_size = 128
            self.encoder_out_size += out_size

        log.debug('Policy head output size: %r', self.get_encoder_out_size())

    def forward(self, obs_dict):
        # This one handles the "obs" key which contains the main image
        x = self.basic_encoder(obs_dict)

        cats = [x]
        if self.vector_obs_head is not None:
            vector_obs = self.vector_obs_head(obs_dict[self.bl_stats_key].float())
            cats.append(vector_obs)
            cats.append(obs_dict[self.bl_stats_key])

        if self.message_head is not None:
            message = self.message_head(obs_dict['message'].float() / 255)
            cats.append(message)

        if len(cats) > 1:
            x = torch.cat(cats, dim=1)

        return x


class TorchBeastEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        self.bl_stats_key = next((key for key in obs_space if "vector_obs" in key or "stats" in key), None)
        assert self.bl_stats_key is not None

        self.k_dim = cfg.encoder_embedding_dim
        self.h_dim = cfg.encoder_hidden_dim
        self.crop_dim = cfg.encoder_crop_dim

        self.use_glyphs = cfg.use_glyphs
        self.use_crop = cfg.use_crop
        self.use_blstats = cfg.use_blstats

        if self.use_crop or self.use_glyphs:
            self.glyph_shape = obs_space["glyphs"].shape
            self.H = self.glyph_shape[0]
            self.W = self.glyph_shape[1]

            self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)

            K = self.k_dim  # number of input filters
            F = 3  # filter dimensions
            S = 1  # stride
            P = 1  # padding
            M = 16  # number of intermediate filters
            Y = 8  # number of output filters
            L = cfg.encoder_num_layers  # number of convnet layers

            in_channels = [K] + [M] * (L - 1)
            out_channels = [M] * (L - 1) + [Y]

            if self.use_crop:
                self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        # Define network
        out_dim = 0

        if self.use_blstats:
            self.blstats_size = obs_space[self.bl_stats_key].shape[0]
            self.embed_blstats = nn.Sequential(
                nn.Linear(self.blstats_size, self.k_dim),
                nn.ReLU(),
                nn.Linear(self.k_dim, self.k_dim),
                nn.ReLU(),
            )
            out_dim += self.k_dim

        # CNN over full glyph map
        if self.use_glyphs:
            conv_extract = [
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=(F, F),
                    stride=2,
                    padding=P,
                )
                for i in range(L)
            ]

            self.extract_representation = nn.Sequential(
                *interleave(conv_extract, [nn.Sequential(nn.ELU())] * len(conv_extract))
            )
            out_dim += calc_conv_output_size(self.H, self.W, P, 1, F, 2, n_layers=L) * Y

        # CNN crop model.
        if self.use_crop:
            conv_extract_crop = [
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=(F, F),
                    stride=S,
                    padding=P,
                )
                for i in range(L)
            ]

            self.extract_crop_representation = nn.Sequential(
                *interleave(conv_extract_crop, [nn.Sequential(nn.ELU())] * len(conv_extract_crop))
            )
            out_dim += self.crop_dim ** 2 * Y

        self.msg_model = cfg.encoder_msg_model
        if self.msg_model == 'lt_cnn_small':
            self.msg_hdim = 32
            self.msg_edim = 16
            self.char_lt = nn.Embedding(
                NUM_CHARS, self.msg_edim, padding_idx=PAD_CHAR
            )            
            self.conv1 = nn.Conv1d(
                self.msg_edim, self.msg_hdim, kernel_size=7
            )
            self.conv2_6_fc = nn.Sequential(
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv2
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=7),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # fc receives -- [ B x h_dim x 5 ]
                Flatten(),
                nn.Linear(3 * self.msg_hdim, self.msg_hdim),
                nn.ReLU(),
                nn.Linear(self.msg_hdim, self.msg_hdim),
                nn.ReLU(),
            )  # final output -- [ B x h_dim x 5 ]
            out_dim += self.msg_hdim
            
        elif self.msg_model == 'lt_cnn':
            self.msg_hdim = 64
            self.msg_edim = 32
            self.char_lt = nn.Embedding(
                NUM_CHARS, self.msg_edim, padding_idx=PAD_CHAR
            )            
            self.conv1 = nn.Conv1d(
                self.msg_edim, self.msg_hdim, kernel_size=7
            )
            # remaining convolutions, relus, pools, and a small FC network
            self.conv2_6_fc = nn.Sequential(
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv2
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=7),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv3
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                # conv4
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                # conv5
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                # conv6
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # fc receives -- [ B x h_dim x 5 ]
                Flatten(),
                nn.Linear(5 * self.msg_hdim, 2 * self.msg_hdim),
                nn.ReLU(),
                # nn.Linear(2 * self.msg_hdim, self.msg_hdim),
                # nn.ReLU(),
            )  # final output -- [ B x h_dim x 5 ]
            out_dim += 2 * self.msg_hdim

        log.debug(f'Encoder latent dimensionality: {out_dim}')

        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.h_dim // 2),
            nn.ReLU(),
            nn.Linear(self.h_dim // 2, self.h_dim)
        )

        self.encoder_out_size = self.h_dim
        log.debug('Policy head output size: %r', self.get_encoder_out_size())

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        try:
            return out.reshape(x.shape + (-1,))
        except Exception as e:
            raise ValueError("Invalid size") from e

    def forward(self, obs_dict):
        
        messages = obs_dict["message"].long()      
        batch_size = messages.shape[0]

        reps = []

        if self.use_blstats:
            blstats = obs_dict[self.bl_stats_key].float()
            blstats_emb = self.embed_blstats(blstats)
            reps.append(blstats_emb)

        if self.use_crop or self.use_glyphs:
            glyphs = obs_dict["glyphs"].long() 
            glyphs_emb = self._select(self.embed, glyphs)

        if self.use_crop:
            coordinates = blstats[:, :2] # -- [B x 2] x,y coordinates
            crop = self.crop(glyphs, coordinates)
            crop_emb = self._select(self.embed, crop)
            crop_emb = crop_emb.transpose(1, 3)  # -- TODO: slow?
            crop_rep = self.extract_crop_representation(crop_emb)
            crop_rep = crop_rep.view(batch_size, -1)
            reps.append(crop_rep)

        if self.use_glyphs:
            glyphs_emb = glyphs_emb.transpose(1, 3)  # -- TODO: slow?
            glyphs_rep = self.extract_representation(glyphs_emb)
            glyphs_rep = glyphs_rep.view(batch_size, -1)
            reps.append(glyphs_rep)

        if self.msg_model != 'none':
            if self.msg_model == "lt_cnn_small":
                messages = messages[:, :128] # most messages are much shorter than 256
            char_emb = self.char_lt(messages).transpose(1, 2)
            char_rep = self.conv2_6_fc(self.conv1(char_emb))
            reps.append(char_rep)

        # -- [batch size x K]
        reps = torch.cat(reps, dim=1)
        st = self.fc(reps)

        return st


register_custom_encoder('nle_rgbcrop_encoder', NLEMainEncoder)
register_custom_encoder('nle_torchbeast_encoder', TorchBeastEncoder)
