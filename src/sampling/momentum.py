# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kijung Jeon

"""Momentum lift: paper Definition 4.3, with explicit upstream jump option."""

import math

from src.sampling.terminal import weighted_index
from src.sampling.vgb import VGBSampler


def momentum_choices(candidates, direction, *, chi=1., include_stay=True):
    """Return (event, candidate-or-None, weight) before ONE normalization.

    With stays, weights sum to F+B in either direction (Definition 4.3).
    Without stays, moves/switches are renormalized, as upstream directed_choices.
    """
    if direction not in ("down", "up") or not math.isfinite(chi) or not 0 <= chi <= 1:
        raise ValueError("Momentum requires down/up and chi in [0, 1]")
    if type(include_stay) is not bool:
        raise ValueError("include_stay must be bool")
    forward = sum(c.weight for c in candidates if c.direction == "forward")
    backward = sum(c.weight for c in candidates if c.direction == "backward")
    matched = chi * min(forward, backward)
    allowed = "forward" if direction == "down" else "backward"
    choices = [(c.direction, c, c.weight) for c in candidates if c.direction == allowed]
    switch = (backward if direction == "down" else forward) - matched
    choices.append(("switch", None, switch))
    if include_stay:
        choices.append(("stay", None, matched))
    return choices


class MomentumSampler(VGBSampler):
    method_id = "vgb_momentum"
    lifted = True

    def __init__(self, backbone, verifier, completion_sampler, *, chi=1., include_stay=True, **kwargs):
        momentum_choices([], "down", chi=chi, include_stay=include_stay)
        super().__init__(backbone, verifier, completion_sampler, **kwargs)
        self.chi, self.include_stay = chi, include_stay

    def _settings(self):
        return {"chi": self.chi, "include_stay": self.include_stay,
                "momentum_kernel": "paper_definition_4.3" if self.include_stay else "upstream_jump"}

    def _transition(self, candidates, direction, rng):
        choices = momentum_choices(candidates, direction, chi=self.chi, include_stay=self.include_stay)
        index = weighted_index([weight for _, _, weight in choices], rng)
        if index is None:
            return "zero_mass", None, direction
        event, candidate, _ = choices[index]
        if event == "switch":
            direction = "up" if direction == "down" else "down"
        return event, candidate.state if candidate else None, direction
