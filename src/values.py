"""Terminal-anchored neural and fixed-m CARS policy values.

CARS estimates V_q for its stated completion policy, not an ideal diffusion
conditional. Memoized noisy estimates are reused as such; cache hits are not
new independent Monte Carlo observations. Guidance floors live outside values.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import torch

from src.cars import CARS_VERSION, CARSSampler, CompletionPolicy, MDLMCompletionPolicy, _reward_identity
from src.completion import LeftToRightCompletion
from src.config import config_digest
from src.reward import RewardChecker
from src.schemas import BaseModel, SmilesState, ValueProvider, require_count
from src.tokenizer import SmilesTokenizer
from src.tracking import BudgetExceeded, CostTracker
from src.verifier import VerifierAdapter

VALUE_VERSION = "terminal-anchored-values-v1"


def checked_value(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("A raw value must be finite and in [0, 1]")
    return value


def guidance_value(value: float, state: SmilesState, floor: float) -> float:
    value, floor = checked_value(value), checked_value(floor)
    return value if state.is_terminal else max(value, floor)


def reward_context(checker: RewardChecker) -> dict[str, Any]:
    return {group: _reward_identity(checker, grammar) for group, grammar in checker.grammars.items()}


class AnchoredValueProvider:
    """Bounded LRU, keyed by complete state and immutable inference context."""

    def __init__(self, tokenizer: SmilesTokenizer, checker: RewardChecker, *, context: Mapping[str, Any],
                 target: str, cache_capacity: int = 128, tracker: CostTracker | None = None) -> None:
        require_count(cache_capacity, "value cache capacity")
        self.tokenizer, self.checker, self.tracker = tokenizer, checker, tracker
        self.target, self.cache_capacity = target, cache_capacity
        self.metadata = {"version": VALUE_VERSION, "target": target, "ideal_value_claim": False,
                         "tokenizer": tokenizer.fingerprint, "reward": reward_context(checker),
                         "cache_capacity": cache_capacity, **copy.deepcopy(dict(context))}
        self.context_hash = config_digest(self.metadata)
        self._cache: OrderedDict[str, float] = OrderedDict()

    def _key(self, state: SmilesState) -> str:
        if (state.special_tokens != self.tokenizer.special_tokens or len(state.token_ids) != self.tokenizer.max_length
                or any(token >= self.tokenizer.vocab_size for token in state.token_ids)):
            raise ValueError("Value state differs from its tokenizer/vocabulary/width")
        return config_digest({"context": self.context_hash, "state": state.to_dict()})

    def _lookup(self, state: SmilesState) -> tuple[str, float | None]:
        if self.tracker is not None:
            self.tracker.require_budget()
            self.tracker.increment("value_queries")
        key = self._key(state)
        if key in self._cache:
            self._cache.move_to_end(key)
            if self.tracker is not None:
                self.tracker.increment("cache_hits")
            return key, self._cache[key]
        if self.tracker is not None:
            self.tracker.increment("cache_misses")
        return key, None

    def _put(self, key: str, value: float) -> float:
        value = checked_value(value)
        if self.cache_capacity:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_capacity:
                self._cache.popitem(last=False)
        return value

    def _terminal(self, state: SmilesState) -> float:
        timing = nullcontext() if self.tracker is None else self.tracker.measure("terminal_value")
        with timing:
            result = self.checker.evaluate_state(state, self.tokenizer)
        if self.tracker is not None:
            self.tracker.event("terminal_value", state=state.to_dict(), reward=result.to_dict())
        return result.reward

    def value(self, state: SmilesState) -> float:
        return self.values([state])[0]

    def state_dict(self) -> dict[str, Any]:
        return {"version": VALUE_VERSION, "context_hash": self.context_hash, "cache": list(self._cache.items())}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload["version"] != VALUE_VERSION or payload["context_hash"] != self.context_hash:
            raise ValueError("Value provider context changed on resume")
        rows = payload["cache"]
        if len(rows) > self.cache_capacity or len({key for key, _ in rows}) != len(rows):
            raise ValueError("Invalid value cache snapshot")
        self._cache = OrderedDict((key, checked_value(value)) for key, value in rows)


class LearnedValueProvider(AnchoredValueProvider):
    def __init__(self, verifier: VerifierAdapter, tokenizer: SmilesTokenizer, checker: RewardChecker, *,
                 source_mdlm_sha256: str, suite: str, cache_capacity: int = 128,
                 tracker: CostTracker | None = None) -> None:
        source = verifier.target_metadata
        if (source["reward_version"] != checker.version or source["source_mdlm_sha256"] != source_mdlm_sha256
                or source["suite"] != suite
                or source["target_kind"] not in ("reference_rollout_return", "q_z_rollout_return")):
            raise ValueError("Learned value has incompatible reward, MDLM, suite or training target")
        super().__init__(tokenizer, checker, context={"verifier": verifier.model_id, "training_target": source,
                         "model": source_mdlm_sha256}, target=source["target_kind"],
                         cache_capacity=cache_capacity, tracker=tracker)
        self.verifier = verifier

    def values(self, states: Sequence[SmilesState]) -> list[float]:
        result: list[float | None] = [None] * len(states)
        missing: OrderedDict[str, tuple[SmilesState, list[int]]] = OrderedDict()
        for index, state in enumerate(states):
            key, value = self._lookup(state)
            if value is None and state.is_terminal:
                value = self._put(key, self._terminal(state))
            if value is not None:
                result[index] = value
            elif key in missing:
                missing[key][1].append(index)
            else:
                missing[key] = state, [index]
        if missing:
            predictions = self.verifier.predict_batch([state for state, _ in missing.values()])
            if len(predictions) != len(missing):
                raise ValueError("Verifier returned the wrong number of values")
            for (key, (_, indices)), value in zip(missing.items(), predictions, strict=True):
                value = self._put(key, value)
                if self.tracker is not None:
                    self.tracker.event("learned_value", query=key, value=value, target=self.target)
                for index in indices:
                    result[index] = value
        return [checked_value(value) for value in result]


class CARSValueProvider(AnchoredValueProvider):
    def __init__(self, tokenizer: SmilesTokenizer, checker: RewardChecker, *,
                 policy_factory: Callable[[SmilesState], CompletionPolicy], policy_context: Mapping[str, Any],
                 m: int = 16, max_nodes: int = 128, cache_capacity: int = 128, seed: int = 0,
                 tracker: CostTracker | None = None) -> None:
        require_count(m, "fixed CARS completion count", positive=True)
        require_count(max_nodes, "inner CARS trie capacity", positive=True)
        require_count(seed, "value seed")
        super().__init__(tokenizer, checker, context={"policy": dict(policy_context), "sampler": CARS_VERSION,
                         "m": m, "max_nodes": max_nodes, "seed": seed, "estimator": "mean(p_before * tau)"},
                         target="V_q", cache_capacity=cache_capacity, tracker=tracker)
        self.policy_factory, self.m, self.max_nodes = policy_factory, m, max_nodes
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.last_estimate: dict[str, Any] | None = None

    @classmethod
    def from_mdlm(cls, model: BaseModel, tokenizer: SmilesTokenizer, checker: RewardChecker, **kwargs):
        tracker = kwargs.get("tracker")
        completion = LeftToRightCompletion(model)
        return cls(tokenizer, checker,
                   policy_factory=lambda state: MDLMCompletionPolicy(completion, tokenizer, checker, state.group, tracker=tracker),
                   policy_context=completion.metadata, **kwargs)

    def values(self, states: Sequence[SmilesState]) -> list[float]:
        return [self._value(state) for state in states]

    def _value(self, state: SmilesState) -> float:
        # A hit/terminal/error must not leave another query's uncertainty trace
        # looking like a fresh estimate of this state.
        self.last_estimate = None
        key, cached = self._lookup(state)
        if cached is not None:
            return cached
        if state.is_terminal:
            return self._put(key, self._terminal(state))
        seed = int(torch.randint(0, 2**63 - 1, (1,), generator=self.generator))
        sampler = CARSSampler(self.policy_factory(state), state, seed=seed, max_nodes=self.max_nodes,
                              tracker=self.tracker, inner=True)
        contributions = []
        attempts = 0
        for index in range(self.m):
            if self.tracker is not None:
                self.tracker.require_budget()
            if sampler.trie.exhausted:
                # The remaining fixed-m terms are exactly zero by proof, not by
                # observing enough successes. No imaginary forwards are counted.
                contributions.extend([0.0] * (self.m - index))
                break
            attempt = sampler.attempt()
            attempts += 1
            if self.tracker is not None:
                self.tracker.event("cars_value_attempt", query=key, index=index,
                                   reward=None if attempt.reward_result is None else attempt.reward_result.reward,
                                   **attempt.to_dict())
            if attempt.weighted_reward is None:
                self.last_estimate = None
                raise BudgetExceeded(attempt.reason or "inner_CARS")
            contributions.append(attempt.weighted_reward)
        value = checked_value(math.fsum(contributions) / self.m)
        # Contributions are adaptive martingale samples, so an IID standard
        # error is inappropriate. Report a bounded-difference 95% interval.
        radius = math.sqrt(math.log(40.0) / (2 * self.m))
        self.last_estimate = {"query": key, "value": value, "m": self.m, "actual_attempts": attempts,
                              "contributions": contributions, "standard_error": None,
                              "confidence_interval": [max(0.0, value - radius), min(1.0, value + radius)],
                              "interval_method": "Hoeffding-Azuma bounded martingale (95%)", "target": self.target}
        if self.tracker is not None:
            self.tracker.event("cars_value", **self.last_estimate)
        return self._put(key, value)

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "rng": self.generator.get_state()}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        super().load_state_dict(payload)
        self.generator.set_state(payload["rng"].cpu())
        self.last_estimate = None


__all__ = ["CARSValueProvider", "LearnedValueProvider", "ValueProvider", "guidance_value"]
