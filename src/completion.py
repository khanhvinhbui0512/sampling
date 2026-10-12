"""Full-vocabulary completion policies and probability traces for later CARS.

q_z is the explicitly defined left-to-right policy conditional on every revealed
coordinate and the molecular group. Its success probability is a policy value;
it is not asserted to equal the paper's ideal any-order value V*.

The reference policy uses the QM9 absorbing reverse posterior on a linear time
grid. Time/group conditioning and optional final denoising are adaptations.
No grammar filtering, top-k filtering, chemical reward, or trie is implemented
in this module; those belong to subsequent implementation steps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from src.config import config_digest
from src.mdlm import MDLMAdapter
from src.schemas import BaseModel, SmilesState, require_count


@dataclass(frozen=True, slots=True)
class Branch:
    before: SmilesState
    position: int
    token_id: int
    log_probability: float
    phase: str
    time: float | None = None
    next_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"before": self.before.to_dict(), "position": self.position, "token_id": self.token_id,
                "log_probability": self.log_probability, "phase": self.phase,
                "time": self.time, "next_time": self.next_time}


@dataclass(frozen=True, slots=True)
class CompletionResult:
    state: SmilesState
    branches: tuple[Branch, ...]
    log_probability: float
    probability_kind: str
    policy_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state.to_dict(), "branches": [branch.to_dict() for branch in self.branches],
                "log_probability": self.log_probability, "probability_kind": self.probability_kind,
                "policy_hash": self.policy_hash}


def normalize_eos_suffix(state: SmilesState) -> SmilesState:
    """PAD only unresolved coordinates outside a known EOS.

    A conflicting known suffix is retained so it can receive its actual invalid
    reward. Known coordinates (locked or unlocked) are never overwritten.
    """
    eos, special = state.eos_position, state.special_tokens
    if eos is None or any(token not in (special.pad_id, special.mask_id) for token in state.token_ids[eos + 1:]):
        return state
    return state.with_tokens({position: special.pad_id for position in state.masked_positions if position > eos})


def full_log_probabilities(model: BaseModel, states: Sequence[SmilesState]) -> torch.Tensor:
    """Only the absorbing MASK constraint is applied; retain the full vocabulary."""
    if not states or len({len(state.token_ids) for state in states}) != 1:
        raise ValueError("Completion requires a nonempty batch of equal-width states")
    if any(state.special_tokens != model.special_tokens for state in states):
        raise ValueError("Model/state special tokens differ")
    logits = model.logits_batch(states).float().clone()
    expected = (len(states), len(states[0].token_ids), model.vocab_size)
    if logits.shape != expected or torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError("Model returned invalid full-vocabulary logits")
    logits[..., model.special_tokens.mask_id] = -torch.inf
    for row, state in enumerate(states):
        for position in state.masked_positions:
            if not torch.isfinite(logits[row, position]).any():
                raise ValueError("Masked coordinate has no allowed clean token")
    return logits.log_softmax(-1)


class LeftToRightCompletion:
    """q_z branch probabilities include known tokens on both sides of the mask."""

    def __init__(self, model: BaseModel, *, seed: int = 0) -> None:
        require_count(seed, "completion seed")
        self.model = model
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.metadata = {"policy": "left_to_right-v1", "model_id": model.model_id,
                         "vocabulary": "all_non_MASK", "temperature": 1.0,
                         "known_tokens": "preserve_all", "eos_suffix": "deterministic_mask_to_PAD"}
        self.policy_hash = config_digest(self.metadata)

    def rng_state(self) -> torch.Tensor:
        return self.generator.get_state()

    def restore_rng_state(self, state: torch.Tensor) -> None:
        self.generator.set_state(state.cpu())

    @torch.inference_mode()
    def next_branch(self, state: SmilesState) -> tuple[SmilesState, int, torch.Tensor]:
        """Return normalized state, next coordinate, and every branch log probability.

        This is the entrypoint the future trie can use without resampling a
        prefix. Its result remains conditional on known suffix coordinates.
        """
        state = normalize_eos_suffix(state)
        if state.is_terminal:
            raise ValueError("Terminal states have no q_z branch")
        position = state.masked_positions[0]
        log_probs = full_log_probabilities(self.model, [state])[0, position]
        return state, position, log_probs

    @torch.inference_mode()
    def complete_batch(self, states: Sequence[SmilesState]) -> list[CompletionResult]:
        current = [normalize_eos_suffix(state) for state in states]
        traces: list[list[Branch]] = [[] for _ in current]
        while active := [index for index, state in enumerate(current) if not state.is_terminal]:
            batch = [current[index] for index in active]
            if len({len(state.token_ids) for state in batch}) != 1:
                raise ValueError("Completion batches must have equal-width states")
            log_probs = full_log_probabilities(self.model, batch)
            for row, index in enumerate(active):
                state = current[index]
                position = state.masked_positions[0]
                distribution = log_probs[row, position].exp().cpu()
                token = int(torch.multinomial(distribution, 1, generator=self.generator).item())
                probability = float(log_probs[row, position, token].item())
                traces[index].append(Branch(state, position, token, probability, "left_to_right"))
                current[index] = normalize_eos_suffix(state.with_tokens({position: token}))
        return [CompletionResult(state, tuple(trace), math.fsum(branch.log_probability for branch in trace),
                                 "q_z_path_and_leaf", self.policy_hash)
                for state, trace in zip(current, traces, strict=True)]

    def complete(self, state: SmilesState) -> CompletionResult:
        return self.complete_batch([state])[0]


class ReferenceCompletion:
    """Parallel absorbing posterior; trace probability is a trajectory probability.

    At masked coordinate i, the reverse transition from t to s has
    P(token=v)=p_theta(v|z,t)*(p(t)-p(s))/p(t) and
    P(MASK)=p(s)/p(t). Revealed coordinates are copied. Sampling uses the same
    pre-step state for every coordinate. Start time is 1 even for partial states;
    this defines the chosen completion kernel explicitly.
    """

    def __init__(self, model: MDLMAdapter, *, steps: int = 128, end_time: float = 1e-5,
                 final_denoise: bool = True, seed: int = 0) -> None:
        require_count(steps, "reference completion steps", positive=True)
        require_count(seed, "completion seed")
        if not 0 < end_time < 1 or type(final_denoise) is not bool:
            raise ValueError("Invalid final time or denoising flag")
        self.model, self.steps = model, steps
        self.end_time, self.final_denoise = end_time, final_denoise
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.metadata = {"policy": "absorbing_reverse-v1", "model_id": model.model_id,
                         "steps": steps, "start_time": 1.0, "end_time": end_time,
                         "grid": "linear", "final_denoise": final_denoise,
                         "vocabulary": "all_non_MASK", "temperature": 1.0,
                         "epsilon": model.model.specification["diffusion"]["epsilon"],
                         "known_tokens": "preserve_all", "eos_suffix": "deterministic_mask_to_PAD"}
        self.policy_hash = config_digest(self.metadata)

    def rng_state(self) -> torch.Tensor:
        return self.generator.get_state()

    def restore_rng_state(self, state: torch.Tensor) -> None:
        self.generator.set_state(state.cpu())

    @torch.inference_mode()
    def complete_batch(self, states: Sequence[SmilesState]) -> list[CompletionResult]:
        current = [normalize_eos_suffix(state) for state in states]
        traces: list[list[Branch]] = [[] for _ in current]
        grid = torch.linspace(1.0, self.end_time, self.steps + 1).tolist()
        transitions = [(float(t), float(s), "reverse") for t, s in zip(grid[:-1], grid[1:], strict=True)]
        if self.final_denoise:
            transitions.append((float(grid[-1]), 0.0, "final_denoise"))
        for time, next_time, phase in transitions:
            active = [index for index, state in enumerate(current) if not state.is_terminal]
            if not active:
                break
            batch = [current[index] for index in active]
            times = torch.full((len(batch),), time, device=self.model.device)
            clean = self.model.probabilities_batch(batch, times=times).cpu()
            # The epsilon multiplier cancels for this linear masking probability.
            retain = next_time / time
            posterior = clean * (1.0 - retain)
            posterior[..., self.model.special_tokens.mask_id] = retain
            for row, index in enumerate(active):
                before = current[index]
                positions = before.masked_positions
                distributions = posterior[row, list(positions)]
                if (not torch.isfinite(distributions).all() or torch.any(distributions < 0)
                        or torch.any(distributions.sum(-1) <= 0)):
                    raise ValueError("Reference posterior has no probability mass")
                distributions = distributions / distributions.sum(-1, keepdim=True)
                choices = torch.multinomial(distributions, 1, generator=self.generator).flatten()
                updates = {}
                for offset, position in enumerate(positions):
                    token = int(choices[offset].item())
                    log_probability = float(distributions[offset, token].double().log().item())
                    traces[index].append(Branch(before, position, token, log_probability, phase, time, next_time))
                    updates[position] = token
                current[index] = normalize_eos_suffix(before.with_tokens(updates))
        return [CompletionResult(state, tuple(trace), math.fsum(branch.log_probability for branch in trace),
                                 "reverse_trajectory_not_leaf_marginal", self.policy_hash)
                for state, trace in zip(current, traces, strict=True)]

    def complete(self, state: SmilesState) -> CompletionResult:
        return self.complete_batch([state])[0]
