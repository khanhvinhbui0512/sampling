# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kijung Jeon

"""Singleton QM9 shortlists adapted from MDM-VGB's vgb_candidates.py.

K draws represent the reference token probability empirically: count/K. Do
not multiply by that token's model probability again. Forward/backward depth
coefficients share one rescaling. Leaf weights follow paper Definition 4.2,
including its exception to the current/parent gate at terminal children.
"""

import math
from collections import Counter
from dataclasses import asdict, dataclass, replace

import torch

from src.budget import BudgetExceeded
from src.sampling.completion import prediction_log_probs
from src.sampling.terminal import remask
from src.types import CostPhase, EstimateStatus, MaskedState, require_count


@dataclass(frozen=True)
class CandidateSettings:
    L_f: int = 8
    L_b: int = 32
    K: int = 8
    lambda_value: float = 0.0
    temperature: float = 1.0
    forward_selection: str = "random"
    backward_selection: str = "low_conf"

    def __post_init__(self):
        for name in ("L_f", "L_b", "K"):
            require_count(getattr(self, name), name, positive=True)
        if not math.isfinite(self.lambda_value) or self.lambda_value < 0:
            raise ValueError("lambda must be finite and nonnegative")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if self.forward_selection != "random" or self.backward_selection not in ("random", "low_conf"):
            raise ValueError("QM9 uses random forward and random/low_conf backward positions")


@dataclass(frozen=True)
class Candidate:
    state: MaskedState
    direction: str
    position: int
    multiplicity: int
    proposal_mass: float
    weight: float = 0.0
    value: float | None = None


def depth_coefficients(n, k):
    """Rescale s[k,1] and s[k-1,1] by their COMMON maximum."""
    if not 0 <= k <= n or n < 1:
        raise ValueError("Require 0 <= k <= n, n >= 1")
    forward = 1 / math.comb(n - 1, k) if k < n else 0.0
    backward = 1 / math.comb(n - 1, k - 1) if k else 0.0
    scale = max(forward, backward)
    return forward / scale, backward / scale


def edge_weights(n, k, *, current, child=None, parent=None, proposal_mass=1.,
                 backward_scale=1., lambda_value=0., child_terminal=False, state_terminal=False):
    """Definition 4.2 weights; zeros stay zero, with 0**0 = 1 at lambda=0."""
    if not math.isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("Invalid lambda")
    for value in (current, child, parent):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("Values must be probabilities")
    if any(not math.isfinite(v) or v < 0 for v in (proposal_mass, backward_scale)):
        raise ValueError("Invalid shortlist mass")
    forward_coeff, backward_coeff = depth_coefficients(n, k)
    forward = 0.0 if child is None else forward_coeff * proposal_mass * child * (
        1.0 if child_terminal else current ** lambda_value)
    backward = 0.0 if parent is None else backward_coeff * backward_scale * current * (
        1.0 if state_terminal else parent ** lambda_value)
    return forward, backward


class ValueUnavailable(RuntimeError):
    """An estimator could not resolve its value; never treat that as zero."""


class ValueTable:
    """Full-state values frozen within a single run; no persistent trie/cache."""

    def __init__(self, verifier, rng, budget):
        self.verifier, self.rng, self.budget = verifier, rng, budget
        self.estimates = {}
        self.cache_hits = 0

    def query(self, states):
        missing = []
        seen = set()
        for state in states:
            if state.token_ids in self.estimates or state.token_ids in seen:
                self.cache_hits += 1
            else:
                missing.append(state)
                seen.add(state.token_ids)
        if missing:
            base, neural = self.verifier.required_work(missing)
            held = self.budget.reserve_query(label="guided-value-group", base_sequences=base,
                                             verifier_sequences=neural)
            if held is None:
                raise BudgetExceeded("Insufficient budget for the entire candidate value group")
            with held:
                estimates = self.verifier.values_for_states(missing, rng=self.rng,
                                                             budget=self.budget, reservation=held)
            if len(estimates) != len(missing):
                raise ValueError("Verifier returned a different number of estimates")
            for state, estimate in zip(missing, estimates):
                if estimate.target != self.verifier.target:
                    raise ValueError("Candidate query changed target")
                if estimate.value is None:
                    if estimate.status == EstimateStatus.BUDGET_EXHAUSTED:
                        raise BudgetExceeded(estimate.error or "Unavailable value query")
                    raise ValueUnavailable(estimate.error or "Verifier failed")
                self.estimates[state.token_ids] = estimate
        return [self.estimates[state.token_ids].value for state in states]


class CandidateBuilder:
    def __init__(self, backbone, settings=None):
        self.backbone = backbone
        self.settings = settings or CandidateSettings()

    def build(self, state, *, editable, values, rng, budget, backward=True):
        """One base forward supplies all position logits, including low_conf."""
        if state.is_complete:
            raise ValueError("Terminal rejection uses a pure backtrack, not candidate logits")
        if state.rules != self.backbone.token_rules:
            raise ValueError("Candidate state/backbone token rules differ")
        masks = list(state.masked_positions)
        if not set(masks).issubset(editable):
            raise ValueError("Originally observed context must remain fixed")
        known = [p for p in editable if p not in masks]
        n, k, settings = len(editable), len(known), self.settings
        logits = self.backbone.logits_batch([state], budget=budget, phase=CostPhase.SAMPLING)[0]
        probabilities = prediction_log_probs(logits, state.rules.mask_id, settings.temperature).exp()
        positions = sorted(rng.python.sample(masks, min(settings.L_f, len(masks))))
        forward = []
        for position in positions:
            draws = torch.multinomial(probabilities[position], settings.K, replacement=True,
                                      generator=rng.torch_generator()).tolist()
            for token, count in Counter(draws).items():
                child = state.reveal({position: token})
                mass = count / settings.K * len(masks) / len(positions)
                forward.append(Candidate(child, "forward", position, count, mass))
        parents = []
        if backward and known:
            count = min(settings.L_b, len(known))
            if settings.backward_selection == "low_conf":
                positions = sorted(known, key=lambda p: (float(probabilities[p, state.token_ids[p]]), p))[:count]
            else:
                positions = sorted(rng.python.sample(known, count))
            parents = [Candidate(remask(state, p), "backward", p, 1, len(known) / count)
                       for p in positions]
        # VGR's common current gate cancels; VGB needs V(z) for backward weights.
        need_current = backward and (bool(parents) or settings.lambda_value > 0)
        queries = ([state] if need_current else []) + [c.state for c in forward]
        if parents and settings.lambda_value > 0:
            queries += [p.state for p in parents]
        predictions = values.query(queries)
        offset = int(need_current)
        current = predictions[0] if need_current else 1.0
        children = predictions[offset:offset + len(forward)]
        parent_values = predictions[offset + len(forward):] if settings.lambda_value > 0 else [1.] * len(parents)
        result = []
        for candidate, value in zip(forward, children):
            weight, _ = edge_weights(n, k, current=current, child=value,
                                     proposal_mass=candidate.proposal_mass,
                                     lambda_value=settings.lambda_value if backward else 0.,
                                     child_terminal=candidate.state.is_complete)
            result.append(replace(candidate, weight=weight, value=value))
        for candidate, value in zip(parents, parent_values):
            _, weight = edge_weights(n, k, current=current, parent=value,
                                     backward_scale=candidate.proposal_mass,
                                     lambda_value=settings.lambda_value)
            result.append(replace(candidate, weight=weight,
                                  value=value if settings.lambda_value > 0 else None))
        return result

    def metadata(self):
        return asdict(self.settings)
