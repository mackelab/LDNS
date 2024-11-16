# %%

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from ldns.networks.s4 import FFTConv


class AutoEncoderBlock(nn.Module):
    def __init__(
        self,
        C,
        L,
        kernel="s4",
        bidirectional=True,
        kernel_params=None,
        num_lin_per_mlp=2,
        use_act2=False,
    ):
        super().__init__()
        self.C = C
        self.L = L
        self.bidirectional = bidirectional
        self.kernel_params = kernel_params
        self.time_mixer = self.get_time_mixer()
        self.post_tm_scale = nn.Conv1d(
            C, C, 1, bias=True, groups=C, padding="same"
        )  # channel-wise scale for post-act
        self.channel_mixer = self.get_channel_mixer(num_lin_per_mlp=num_lin_per_mlp)
        self.norm1 = nn.InstanceNorm1d(C, affine=False)  # make sure input is [B, C, L]!
        self.norm2 = nn.InstanceNorm1d(C, affine=False)  # we will use adaLN
        self.act1 = nn.GELU()
        self.act2 = nn.GELU() if use_act2 else nn.Identity()

        self.ada_ln = nn.Parameter(
            torch.zeros(1, C * 6, 1), requires_grad=True
        )  # 3 for each mixer, shift, scale, gate. gate remains unused for now

    @staticmethod
    def affine_op(x_, shift, scale):
        # x is [B, C, L], shift and scale are [B, C, 1]
        assert len(x_.shape) == len(shift.shape), f"{x_.shape} != {shift.shape}"
        return x_ * (1 + scale) + shift

    def get_time_mixer(self):
        time_mixer = FFTConv(
            self.C,
            bidirectional=self.bidirectional,
            activation=None,
        )

        return time_mixer

    def get_channel_mixer(self, num_lin_per_mlp=2):
        layers = [
            Rearrange("b c l -> b l c"),
            nn.Linear(self.C, self.C * 2, bias=False),  # required for zero-init block
        ]
        # extra linear layers prepended by GELU
        for _ in range(max(num_lin_per_mlp - 2, 0)):
            layers.extend(
                [
                    nn.GELU(),
                    nn.Linear(self.C * 2, self.C * 2, bias=False),
                ]
            )
        layers.extend(
            [
                nn.GELU(),
                nn.Linear(self.C * 2, self.C, bias=False),
            ]
        )
        # finally rearrange back to [B, C, L]
        layers.append(Rearrange("b l c -> b c l"))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = x  # x is residual stream
        y = self.norm1(y)
        ada_ln = repeat(self.ada_ln, "1 d c -> b d c", b=x.shape[0])
        shift_tm, scale_tm, gate_tm, shift_cm, scale_cm, gate_cm = ada_ln.chunk(
            6, dim=1
        )
        y = self.affine_op(y, shift_tm, scale_tm)
        y = self.time_mixer(y)
        y = y[0]  # get output not state for gconv and fftconv
        # y = x + gate_tm.unsqueeze(-1) * self.act1(y)
        y = x + self.post_tm_scale(self.act1(y))

        x = y  # x is again residual stream from last layer
        y = self.norm2(y)
        y = self.affine_op(y, shift_cm, scale_cm)
        # y = x + gate_cm.unsqueeze(-1) * self.act2(self.channel_mixer(y))
        y = x + self.act2(self.channel_mixer(y))
        return y


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class AutoEncoder(nn.Module):
    def __init__(
        self,
        C_in,
        C,
        C_latent,
        L,
        kernel="s4",
        bidirectional=True,
        kernel_params=None,
        in_groups=None,
        bottleneck_groups=None,
        num_blocks=4,
        num_blocks_decoder=None,
        num_lin_per_mlp=2,
        use_act_bottleneck=False,
    ):
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.C_latent = C_latent
        self.L = L
        self.bidirectional = bidirectional
        self.kernel_params = kernel_params
        if in_groups is None:
            if C % C_in == 0 and C > C_in:
                in_groups = C_in
            else:
                in_groups = 1

        if bottleneck_groups is None:
            if C % C_latent == 0 and C > C_latent:
                bottleneck_groups = C_latent
            else:
                bottleneck_groups = 1

        self.encoder_in = nn.Conv1d(
            C_in,
            C,
            1,
            # groups=in_groups,
        )  # in_groups matter for encoding count data
        self.encoder = nn.ModuleList(
            [
                AutoEncoderBlock(
                    C,
                    L,
                    kernel,
                    bidirectional,
                    kernel_params,
                    num_lin_per_mlp=num_lin_per_mlp,
                )
                for _ in range(num_blocks)
            ]
        )

        self.bottleneck = nn.Conv1d(C, C_latent, 1, groups=bottleneck_groups)
        self.act_bottleneck = nn.GELU() if use_act_bottleneck else nn.Identity()
        self.unbottleneck = nn.Conv1d(C_latent, C, 1, groups=bottleneck_groups)

        if num_blocks_decoder == 0:
            self.decoder = nn.ModuleList([nn.GELU()])  # jsut the activation

        else:
            self.decoder = nn.ModuleList(
                [
                    AutoEncoderBlock(
                        C,
                        L,
                        kernel,
                        bidirectional,
                        kernel_params,
                        num_lin_per_mlp=num_lin_per_mlp,
                    )
                    for _ in range(
                        (
                            num_blocks
                            if num_blocks_decoder is None
                            else num_blocks_decoder
                        )
                    )
                ]
            )

        # self.decoder_out = nn.Conv1d(C, C_in, 1, groups=in_groups)
        self.decoder_out = nn.Conv1d(C, C_in, 1)

    def encode(self, x):
        z = self.encoder_in(x)
        for block in self.encoder:
            z = block(z)
        z = self.bottleneck(z)
        return z

    def decode(self, z):
        xhat = self.act_bottleneck(z)
        xhat = self.unbottleneck(xhat)
        for block in self.decoder:
            xhat = block(xhat)
        xhat = self.decoder_out(xhat)
        return xhat

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z


class DenoiserBlock(nn.Module):
    def __init__(
        self, C, L, bidirectional=True, kernel_params=None, use_act2=False
    ):
        super().__init__()
        self.C = C
        self.L = L
        self.bidirectional = bidirectional
        self.kernel_params = kernel_params
        self.time_mixer = self.get_time_mixer()
        self.channel_mixer = self.get_channel_mixer()
        self.norm1 = nn.InstanceNorm1d(
            C, affine=False
        )  # affine=False because we will use adaLN
        self.norm2 = nn.InstanceNorm1d(C, affine=False)
        self.act1 = nn.GELU()
        self.act2 = nn.GELU() if use_act2 else nn.Identity()
        self.ada_ln = nn.Sequential(  # gets as input [B, C]
            nn.GELU(),
            nn.Linear(
                C // 4,
                C * 6,
                bias=True,
            ),
        )

        # zero-init all weights and biases of ada_ln linear layer
        self.ada_ln[-1].weight.data.zero_()
        self.ada_ln[-1].bias.data.zero_()

    def get_time_mixer(self):
        return FFTConv(
            self.C, bidirectional=self.bidirectional, activation=None
        )

    def get_channel_mixer(self):
        return nn.Sequential(
            Rearrange("b c l -> b l c"),
            nn.Linear(self.C, self.C * 2),
            nn.GELU(),
            nn.Linear(self.C * 2, self.C),
            Rearrange("b l c -> b c l"),
        )

    @staticmethod
    def affine_op(x, shift, scale):
        # x is [B, C, L], shift and scale are [B, C]
        return x * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

    def forward(self, x, t_cond):
        y = x  # x is residual stream
        y = self.norm1(y)
        shift_tm, scale_tm, gate_tm, shift_cm, scale_cm, gate_cm = self.ada_ln(
            t_cond
        ).chunk(6, dim=1)
        y = self.affine_op(y, shift_tm, scale_tm)
        y = self.time_mixer(y)
        y = y[0]  # get output not state for gconv and fftconv
        y = x + gate_tm.unsqueeze(-1) * self.act1(y)

        x = y  # x is again residual stream from last layer
        y = self.norm2(y)
        y = self.affine_op(y, shift_cm, scale_cm)
        y = x + gate_cm.unsqueeze(-1) * self.act2(self.channel_mixer(y))
        return y


class Denoiser(nn.Module):
    def __init__(
        self,
        C_in,
        C,
        L,
        kernel="s4",
        bidirectional=True,
        kernel_params=None,
        in_groups=None,
        num_blocks=6,
    ):
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.L = L
        self.bidirectional = bidirectional
        self.kernel_params = kernel_params
        if in_groups is None:  # grouped by default
            if C % C_in == 0 and C > C_in:
                in_groups = C_in
            else:
                in_groups = 1

        self.conv_in = nn.Conv1d(C_in, C, 1, groups=in_groups)
        self.blocks = nn.ModuleList(
            [
                DenoiserBlock(C, L, kernel, bidirectional, kernel_params)
                for _ in range(num_blocks)
            ]
        )
        self.conv_out = nn.Conv1d(C, C_in, 1, groups=in_groups)

        self.t_emb = TimestepEmbedder(C // 4)  # [B, C//4] to keep param count in check

    def forward(self, x, t):
        x = self.conv_in(x)
        t_emb = self.t_emb(t.to(x.device))
        # print(x.shape, t_emb.shape)
        for block in self.blocks:
            x = block(x, t_emb)
            # print(x.shape, t_emb.shape)
        x = self.conv_out(x)
        return x


class ConditionalDenoiser(nn.Module):
    def __init__(
        self,
        C_in,
        C,
        L,
        kernel="s4",
        bidirectional=True,
        kernel_params=None,
        in_groups=None,
        num_blocks=6,
        condition_dim=2,  # for now, just 2D condition
    ):
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.L = L
        self.bidirectional = bidirectional
        self.kernel_params = kernel_params
        if in_groups is None:  # grouped by default
            if C % C_in == 0 and C > C_in:
                in_groups = C_in
            else:
                in_groups = 1

        self.conv_in = nn.Conv1d(C_in, C, 1, groups=in_groups)
        self.blocks = nn.ModuleList(
            [
                DenoiserBlock(C, L, kernel, bidirectional, kernel_params)
                for _ in range(num_blocks)
            ]
        )
        self.conv_out = nn.Conv1d(C, C_in, 1, groups=in_groups)

        self.t_emb = TimestepEmbedder(C // 4)  # [B, C//4] to keep param count in check
        self.c_emb = nn.Sequential(
            nn.Linear(condition_dim, C // 4, bias=True),
            nn.SiLU(),
            nn.Linear(C // 4, C // 4, bias=True),
        )  # [B, C//4]

    def forward(self, x, t, c=None):
        x = self.conv_in(x)
        t_emb = self.t_emb(t.to(x.device))

        if c is not None:
            c_emb = self.c_emb(c.to(x.device))
            t_emb = t_emb + c_emb  # otherwise just use timestep embedding

        for block in self.blocks:
            x = block(x, t_emb)
        x = self.conv_out(x)
        return x

# %%

if __name__ == "__main__":

    import lovely_tensors
    from torchinfo import summary

    lovely_tensors.monkey_patch()



    # test AutoEncoderBlock
    block = AutoEncoderBlock(128, 500)
    x = torch.randn(2, 128, 500)
    print(summary(block, (2, 128, 500), device="cpu"))
    print(x - block(x))

    # test DenoiserBlock
    denoiser_block = DenoiserBlock(128, 500)
    x = torch.randn(2, 128, 500)
    t_cond = torch.randn(2, 128 // 4)  # [B, C//4] to keep param count in check
    print(summary(denoiser_block, [(2, 128, 500), (2, 128 // 4)], device="cpu"))
    print(x - denoiser_block(x, t_cond))

    # test AutoEncoder
    ae = AutoEncoder(128, 512, 16, 500, num_blocks=4, num_lin_per_mlp=1)
    x = torch.randn(2, 128, 500)
    print(summary(ae, (2, 128, 500), device="cpu", depth=4))

    # test Denoiser
    denoiser = Denoiser(8, 256, 500)
    x = torch.randn(2, 8, 500)
    t = torch.randn(2)
    print(summary(denoiser, [(2, 8, 500), (2,)], device="cpu"))

    # test ConditionalDenoiser
    conditional_denoiser = ConditionalDenoiser(8, 256, 500)
    x = torch.randn(2, 8, 500)
    t = torch.randn(2)
    c = torch.randn(2, 2)
    print(summary(conditional_denoiser, [(2, 8, 500), (2,), (2, 2)], device="cpu"))

# %%
