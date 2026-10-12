# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kijung Jeon

"""Forward-only VGR from paper Algorithm 2 and upstream vgr.py."""

from src.sampling.terminal import weighted_index
from src.sampling.vgb import VGBSampler


class VGRSampler(VGBSampler):
    method_id = "vgr"
    allow_backtracking = False

    def _transition(self, candidates, direction, rng):
        # Multiplicity already represents pi_ref; zeros revert to that same
        # empirical reference distribution, not a uniform law over unique tokens.
        weights = [candidate.weight for candidate in candidates]
        event = "forward"
        if sum(weights) == 0:
            weights = [candidate.proposal_mass for candidate in candidates]
            event = "zero_value_reference_forward"
        index = weighted_index(weights, rng)
        return event, candidates[index].state, direction
