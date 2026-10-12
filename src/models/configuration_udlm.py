"""Local configuration for the QM9 MDLM backbone.

Adapted from MDM-VGB's mdlm_source/configuration_udlm.py.
Copyright (c) 2026 Kijung Jeon. SPDX-License-Identifier: MIT
See THIRD_PARTY_NOTICES.md for the source fingerprint and license.
"""

from transformers import PretrainedConfig


class UDLMConfig(PretrainedConfig):
    model_type = "udlm"

    def __init__(self, vocab_size=40, model_length=32, hidden_dim=768, cond_dim=128,
                 n_blocks=12, n_heads=12, dropout=0.1, time_conditioning=False,
                 cfg=False, cfg_num_classes=-1, **kwargs):
        # The 40 default follows upstream's architecture template. Training always
        # replaces it with the full prepared tokenizer vocabulary.
        kwargs.pop("auto_map", None)
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(tie_word_embeddings=False, **kwargs)
        if cfg or cfg_num_classes != -1:
            raise ValueError("This QM9 backbone is unconditional; CFG is unsupported")
        if any(not isinstance(value, int) or value <= 0 for value in
               (vocab_size, model_length, hidden_dim, cond_dim, n_blocks, n_heads)):
            raise ValueError("Architecture dimensions must be positive integers")
        if hidden_dim % n_heads or (hidden_dim // n_heads) % 2:
            raise ValueError("Rotary attention requires an even head dimension")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.vocab_size = vocab_size
        self.model_length = model_length
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout = dropout
        self.time_conditioning = time_conditioning
        self.cfg = False
        self.cfg_num_classes = -1
