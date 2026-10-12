# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kijung Jeon

"""Shared finite-budget policies, adapted from MDM-VGB's terminal handling.

Acceptance is exact Pass@95, not upstream QM9's validity-only acceptance.
An unguided completion is reserved before guidance. No remask costs a forward.
These stopping/escape policies do not assert a stationary leaf distribution.
"""

import math
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from src.sampling.completion import cost_since
from src.types import CostPhase, MaskedState, SampleResult, SampleStatus


@dataclass(frozen=True)
class SamplingRun:
    result: SampleResult
    trace: tuple
    metadata: dict
    rng_state: dict
    direction: str | None = None


def completion_work(base_sampler, states):
    """Conservative sequence work, including native's optional final denoise."""
    if base_sampler.method_id == "base_rho":
        return sum(len(state.masked_positions) for state in states)
    generator = base_sampler.generator
    return sum(not state.is_complete for state in states) * (
        generator.steps + int(generator.final_denoise))


def remask(state, position):
    if position in state.masked_positions:
        raise ValueError("Cannot backtrack an already masked position")
    tokens = list(state.token_ids)
    tokens[position] = state.rules.mask_id
    return MaskedState(tuple(tokens), state.rules)


def backtrack(state, editable, rng):
    """Uniform singleton escape, preserving every originally observed token."""
    positions = [p for p in editable if p not in state.masked_positions]
    return remask(state, rng.python.choice(positions)) if positions else None


def weighted_index(weights, rng):
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Transition weights must be finite and nonnegative")
    if not weights or sum(weights) == 0:
        return None
    probabilities = torch.tensor(weights, dtype=torch.float64)
    return int(torch.multinomial(probabilities, 1, generator=rng.torch_generator()).item())


def finish_run(state, *, sample_id, target, reward, budget, before, trace,
               metadata, rng, direction=None, terminal=None, error=None,
               failure_status=SampleStatus.BUDGET_EXHAUSTED, estimation_rng=None):
    if state.is_complete:
        if terminal is None:
            with budget.oracle_check(phase=CostPhase.SAMPLING):
                terminal = reward.evaluate(state, require_binary=True)
        molecule = terminal.molecule
        status = SampleStatus.FAILED if failure_status == SampleStatus.FAILED else SampleStatus.COMPLETED
        result = SampleResult(
            sample_id, state, target, status, cost=cost_since(budget, before),
            smiles=molecule.smiles if molecule else None,
            canonical_smiles=molecule.canonical_smiles if molecule else None,
            valid=terminal.valid, score=terminal.score, reward=terminal.reward,
            error=error if status == SampleStatus.FAILED else None,
        )
    else:
        result = SampleResult(sample_id, state, target, failure_status,
                              cost=cost_since(budget, before), error=error or "Unfinished sample")
    metadata = {**metadata, "success": result.status == SampleStatus.COMPLETED and result.reward == 1,
                "stop_reason": error or "terminal", "seed": rng.seed}
    states = {"sampler": rng.state_dict()}
    if estimation_rng is not None:
        states["estimation"] = estimation_rng.state_dict()
    tqdm.write(f"{metadata['method']} {sample_id}: {result.status.value}, "
               f"Pass@95={result.reward}, base_calls={result.cost.base_calls}, "
               f"base_sequences={result.cost.base_sequences}")
    return SamplingRun(result, tuple(trace), metadata, states, direction)
