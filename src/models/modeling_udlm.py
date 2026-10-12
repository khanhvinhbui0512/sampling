"""QM9 diffusion transformer, with upstream checkpoint parameter names.

Adapted from MDM-VGB's mdlm_source/modeling_udlm.py.
Copyright (c) 2026 Kijung Jeon. SPDX-License-Identifier: MIT
See THIRD_PARTY_NOTICES.md. Unused CFG/JIT helpers are removed; the caller
controls autocast, and layer normalization always computes in float32.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput

from src.models.configuration_udlm import UDLMConfig


def modulate(x, shift, scale):
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Rotary(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("inv_freq", 1 / (10000 ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, q, k):
        positions = torch.arange(q.shape[1], device=q.device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inv_freq.float())
        angles = torch.cat((frequencies, frequencies), dim=-1)[None, :, None, :]
        cosine, sine = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
        return q * cosine + rotate_half(q) * sine, k * cosine + rotate_half(k) * sine


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x):
        # No learned bias, matching upstream's normalization parameterization.
        return F.layer_norm(x.float(), (self.dim,)) * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size),
                                 nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(-math.log(10000) *
                                torch.arange(half, device=t.device).float() / half)
        angles = t.float()[:, None] * frequencies[None, :]
        return self.mlp(torch.cat((angles.cos(), angles.sin()), dim=-1))


class DDiTBlock(nn.Module):
    def __init__(self, dim, cond_dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"),
                                 nn.Linear(4 * dim, dim))
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x, c, rotary):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(c).chunk(6, -1)
        batch, length, dim = x.shape
        qkv = self.attn_qkv(modulate(self.norm1(x), shift1, scale1))
        q, k, v = qkv.reshape(batch, length, 3, self.n_heads, dim // self.n_heads).unbind(2)
        q, k = rotary(q, k)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)
        attention = (weights @ v).transpose(1, 2).reshape(batch, length, dim)
        x = x + gate1[:, None, :] * F.dropout(self.attn_out(attention), self.dropout,
                                             training=self.training)
        update = self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x + gate2[:, None, :] * F.dropout(update, self.dropout, training=self.training)


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_size, dim))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, tokens):
        return self.embedding[tokens]


class DDitFinalLayer(nn.Module):
    def __init__(self, dim, vocab_size, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(dim)
        self.linear = nn.Linear(dim, vocab_size)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * dim)
        for layer in (self.linear, self.adaLN_modulation):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DITBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.time_conditioning = config.time_conditioning
        self.vocab_embed = EmbeddingLayer(config.vocab_size, config.hidden_dim)
        self.sigma_map = TimestepEmbedder(config.cond_dim)
        self.rotary_emb = Rotary(config.hidden_dim // config.n_heads)
        self.blocks = nn.ModuleList([
            DDiTBlock(config.hidden_dim, config.cond_dim, config.n_heads, config.dropout)
            for _ in range(config.n_blocks)
        ])
        self.output_layer = DDitFinalLayer(config.hidden_dim, config.vocab_size, config.cond_dim)

    def forward(self, tokens, sigma, output_hidden_states=False):
        x = self.vocab_embed(tokens)
        c = F.silu(self.sigma_map(sigma if self.time_conditioning else torch.zeros_like(sigma)))
        hidden = [x] if output_hidden_states else None
        for block in self.blocks:
            x = block(x, c, self.rotary_emb)
            if hidden is not None:
                hidden.append(x)
        return self.output_layer(x, c), tuple(hidden) if hidden is not None else None


class UDLM(PreTrainedModel):
    config_class = UDLMConfig
    base_model_prefix = "udlm"

    def __init__(self, config):
        super().__init__(config)
        self.backbone = DITBackbone(config)
        # Keep the upstream zero initialization of AdaLN and the output head.
        # Calling HF post_init() here would replace that initialization.

    def forward(self, input_ids, timesteps=None, cond=None, output_hidden_states=None,
                return_dict=None):
        if input_ids.ndim != 2 or input_ids.shape[1] > self.config.model_length:
            raise ValueError("Expected token IDs with shape [batch, length <= model_length]")
        if cond is not None:
            raise ValueError("QM9 uses an unconditional backbone")
        if timesteps is None:
            if self.config.time_conditioning:
                raise ValueError("Time-conditioned models require timesteps")
            timesteps = torch.zeros(input_ids.shape[0], device=input_ids.device)
        if timesteps.shape != (input_ids.shape[0],):
            raise ValueError("Expected one noise level per sequence")
        logits, hidden = self.backbone(input_ids, timesteps, bool(output_hidden_states))
        use_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if use_dict:
            return MaskedLMOutput(logits=logits, hidden_states=hidden)
        return (logits, hidden) if output_hidden_states else logits
