# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kijung Jeon

"""Native absorbing MDLM transitions and optional pre-transition snapshots.

Schedule/posterior adapted from MDM-VGB's QM9 harness.py, MIT,
Copyright (c) 2026 Kijung Jeon. See THIRD_PARTY_NOTICES.md.
The explicit final denoise is a project addition to remove residual MASK.
"""

from dataclasses import asdict, dataclass

import torch

from src.budget import BudgetExceeded
from src.io import payload_hash
from src.models.backbone import loglinear_sigma
from src.sampling.completion import prediction_log_probs
from src.types import CostPhase


def absorbing_posterior(token_probabilities, tokens, mask_id, move_t, move_s):
    """At a masked site: reveal with probability 1-m_s/m_t; otherwise stay MASK."""
    if not 0 <= move_s <= move_t <= 1 or move_t <= 0:
        raise ValueError("Require 0 <= move_s <= move_t <= 1 and move_t > 0")
    probabilities = token_probabilities.double().clone() * ((move_t - move_s) / move_t)
    probabilities[..., mask_id] = move_s / move_t
    known = tokens != mask_id
    probabilities[known] = 0
    rows, positions = known.nonzero(as_tuple=True)
    probabilities[rows, positions, tokens[known]] = 1
    return probabilities / probabilities.sum(dim=-1, keepdim=True)


@dataclass(frozen=True)
class NativeBatch:
    states: tuple
    snapshots: dict[int, tuple]
    exhausted: bool = False
    error: str | None = None


class NativeMDLM:
    def __init__(self, backbone, *, steps=32, noise_eps=0.001, time_stop=0.00001,
                 final_denoise=True):
        if type(steps) is not int or steps <= 0:
            raise ValueError("steps must be a positive integer")
        if not 0 < noise_eps < 1 or not 0 < time_stop < 1:
            raise ValueError("noise_eps and time_stop must be in (0, 1)")
        if type(final_denoise) is not bool:
            raise ValueError("final_denoise must be bool")
        self.backbone = backbone
        self.rules = backbone.token_rules
        self.steps, self.noise_eps = steps, noise_eps
        self.time_stop, self.final_denoise = time_stop, final_denoise
        self.settings = {"steps": steps, "noise_eps": noise_eps, "time_stop": time_stop,
                         "final_denoise": final_denoise, "token_rules": asdict(self.rules),
                         "time_conditioning": backbone.time_conditioning,
                         "cache": "unchanged_batch_and_no_time_conditioning",
                         "categorical": "cpu_multinomial_float64", "stop_on_eos": False}
        self.policy_id = "mdlm-native-v1:" + payload_hash(self.settings)
        self.reference_id = "native-mdlm-rollout-v1:" + payload_hash({
            "checkpoint": backbone.checkpoint_id, "tokenizer": backbone.tokenizer_id,
            "policy": self.policy_id, "precision": getattr(backbone, "precision", "toy"),
        })

    def sample_batch(self, states, *, rng, budget, phase=CostPhase.SAMPLING,
                     reservation=None, snapshot_steps=()):
        current = tuple(states)
        if any(state.rules != self.rules for state in current):
            raise ValueError("State token rules differ from native MDLM")
        if any(type(step) is not int or not 0 <= step < self.steps for step in snapshot_steps):
            raise ValueError("Snapshot indices must be pre-transition steps in [0, steps)")
        snapshots = {}
        times = torch.linspace(1, self.time_stop, self.steps + 1, dtype=torch.float64)
        cache = None
        cache_allowed = self.backbone.time_conditioning is False

        def predictions(time):
            nonlocal cache
            if cache is None or not cache_allowed:
                sigma = loglinear_sigma(torch.full((len(current),), float(time)), self.noise_eps)
                logits = self.backbone.logits_at_noise_batch(
                    current, sigma, budget=budget, phase=phase, reservation=reservation,
                )
                if logits.shape != (len(current), self.rules.sequence_length, self.rules.vocab_size):
                    raise ValueError("Unexpected native backbone logits shape")
                cache = prediction_log_probs(logits, self.rules.mask_id).exp()
            return cache

        def draw(probabilities):
            tokens = torch.multinomial(probabilities.reshape(-1, self.rules.vocab_size), 1,
                                       generator=rng.torch_generator("cpu"))
            tokens = tokens.reshape(len(current), self.rules.sequence_length)
            return tuple(state.reveal({position: int(tokens[row, position])
                                       for position in state.masked_positions
                                       if int(tokens[row, position]) != self.rules.mask_id})
                         for row, state in enumerate(current))

        try:
            for step in range(self.steps):
                if step in snapshot_steps:
                    snapshots[step] = current
                if not current or all(state.is_complete for state in current):
                    continue  # Absorbing complete states need no model call or random draw.
                theta = predictions(times[step])
                tokens = torch.tensor([state.token_ids for state in current], dtype=torch.long)
                # Log-linear corruption has m(t) = (1-eps)*t exactly.
                move_t = (1 - self.noise_eps) * float(times[step])
                move_s = (1 - self.noise_eps) * float(times[step + 1])
                proposed = draw(absorbing_posterior(theta, tokens, self.rules.mask_id, move_t, move_s))
                if proposed != current:
                    cache = None
                current = proposed
            if self.final_denoise and any(not state.is_complete for state in current):
                # A transition to s=0 uses theta without MASK probability.
                current = draw(predictions(self.time_stop))
        except BudgetExceeded as exc:
            return NativeBatch(current, snapshots, True, str(exc))
        return NativeBatch(current, snapshots)
