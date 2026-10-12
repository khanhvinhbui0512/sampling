"""Independent completion Monte Carlo for explicitly named policy values.

Batching retains a fixed completion count; adapters record actual forward calls
and batch examples separately. The default batch size preserves scalar callers.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

from src.completion import LeftToRightCompletion, ReferenceCompletion
from src.reward import RewardChecker
from src.schemas import BaseModel, SmilesState, require_count
from src.tokenizer import SmilesTokenizer
from src.tracking import CostTracker


@dataclass(frozen=True, slots=True)
class ReferenceEstimate:
    value: float
    completions: int
    rewards: tuple[float, ...]
    standard_error: float | None
    confidence_interval: tuple[float, float]
    target: str
    policy: dict[str, Any]
    exact_terminal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ideal_value_claim": False,
                "interval_method": "exact terminal reward" if self.exact_terminal else "Wilson binomial (95%)"}


def estimate_completion_value(completion: LeftToRightCompletion | ReferenceCompletion, state: SmilesState,
                              tokenizer: SmilesTokenizer, checker: RewardChecker, *, m: int,
                              tracker: CostTracker | None = None, batch_size: int = 1) -> ReferenceEstimate:
    require_count(m, "reference completion count", positive=True)
    require_count(batch_size, "reference completion batch size", positive=True)
    targets = {"left_to_right-v1": "V_q", "absorbing_reverse-v1": "reference_rollout_return"}
    target = targets[completion.metadata["policy"]]
    if tracker is not None:
        tracker.require_budget()
    if state.is_terminal:
        value = checker.terminal_value(state, tokenizer)
        return ReferenceEstimate(value, 0, (), 0.0, (value, value), target, dict(completion.metadata), True)
    rewards = []
    for start in range(0, m, batch_size):
        if tracker is not None:
            tracker.require_budget()
        timing = nullcontext() if tracker is None else tracker.measure("reference_completion")
        with timing:
            size = min(batch_size, m-start)
            results = [completion.complete(state)] if size == 1 else completion.complete_batch([state] * size)
        if len(results) != size:
            raise ValueError("Reference completion returned the wrong batch size")
        for result in results:
            final = result.state
            if (not final.is_terminal or final.group != state.group or final.special_tokens != state.special_tokens
                    or len(final.token_ids) != len(state.token_ids) or final.locked != state.locked
                    or any(not masked and known != actual for known, actual, masked in zip(
                        state.token_ids, final.token_ids, state.mask, strict=True))):
                raise ValueError("Reference completion must be terminal and preserve its known context")
            timing = nullcontext() if tracker is None else tracker.measure("reference_reward")
            with timing:
                rewards.append(checker.terminal_value(final, tokenizer))
            if tracker is not None:
                tracker.event("reference_completion", target=target, state=final.to_dict(), reward=rewards[-1])
    value = math.fsum(rewards) / m
    standard_error = math.sqrt(math.fsum((reward - value)**2 for reward in rewards) / ((m - 1) * m)) if m > 1 else None
    z = 1.959963984540054
    denominator = 1 + z*z / m
    center = (value + z*z / (2*m)) / denominator
    radius = z * math.sqrt(value * (1-value) / m + z*z / (4*m*m)) / denominator
    return ReferenceEstimate(value, m, tuple(rewards), standard_error,
                             (max(0.0, center-radius), min(1.0, center+radius)), target, dict(completion.metadata))


def q_z_reference(model: BaseModel, tokenizer: SmilesTokenizer, checker: RewardChecker, state: SmilesState, *,
                  m: int, seed: int = 0, tracker: CostTracker | None = None, batch_size: int = 1) -> ReferenceEstimate:
    return estimate_completion_value(LeftToRightCompletion(model, seed=seed), state, tokenizer, checker,
                                     m=m, tracker=tracker, batch_size=batch_size)


def rollout_reference(model, tokenizer: SmilesTokenizer, checker: RewardChecker, state: SmilesState, *,
                      m: int, seed: int = 0, steps: int = 128, end_time: float = 1e-5,
                      tracker: CostTracker | None = None, batch_size: int = 1) -> ReferenceEstimate:
    completion = ReferenceCompletion(model, seed=seed, steps=steps, end_time=end_time, final_denoise=True)
    return estimate_completion_value(completion, state, tokenizer, checker, m=m, tracker=tracker, batch_size=batch_size)
