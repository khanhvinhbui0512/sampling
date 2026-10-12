"""Exact finite distributions and small SMILES fixtures for later verification.

Toy probabilities/rewards are deliberately mathematical examples, not benchmark
data. These fixtures need no downloaded model, grammar asset, or resource/ file.
"""

from __future__ import annotations

import math
import os
import random
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import pytest
import torch

from src.artifacts import ArtifactStore
from src.config import load_config
from src.schemas import GROUPS, SmilesState, SpecialTokens, require_count


@dataclass(frozen=True)
class FiniteDistribution:
    weights: dict[tuple[int, ...], float]
    rewards: dict[tuple[int, ...], float]
    special_tokens: SpecialTokens
    group: str = GROUPS[0]

    def __post_init__(self) -> None:
        if not self.weights or set(self.weights) != set(self.rewards):
            raise ValueError("Every toy outcome needs a probability and reward")
        if any(not math.isfinite(p) or p <= 0 for p in self.weights.values()):
            raise ValueError("Toy outcome probabilities must be positive and finite")
        if not math.isclose(sum(self.weights.values()), 1.0, abs_tol=1e-12):
            raise ValueError("Toy probabilities must sum to one")
        if any(reward not in (0.0, 1.0) for reward in self.rewards.values()):
            raise ValueError("Toy rewards must be binary")
        if len({len(tokens) for tokens in self.weights}) != 1:
            raise ValueError("Toy outcomes must have a shared width")
        for tokens in self.weights:
            state = SmilesState(tokens, (False,) * len(tokens), self.group, self.special_tokens)
            if state.terminal_status != "complete":
                raise ValueError("Toy outcomes must contain EOS and a valid PAD suffix")

    @property
    def width(self) -> int:
        return len(next(iter(self.weights)))

    def conditioned(self, state: SmilesState) -> dict[tuple[int, ...], float]:
        if state.group != self.group or state.special_tokens != self.special_tokens:
            raise ValueError("Toy state has a different group/tokenizer")
        if len(state.token_ids) != self.width:
            raise ValueError("Toy state has a different width")
        matching = {
            tokens: probability for tokens, probability in self.weights.items()
            if all(masked or actual == known for actual, known, masked in zip(
                tokens, state.token_ids, state.mask, strict=True,
            ))
        }
        total = sum(matching.values())
        if total == 0:
            raise ValueError("Toy state has zero probability under the joint distribution")
        return {tokens: probability / total for tokens, probability in matching.items()}

    def exact_value(self, state: SmilesState) -> float:
        if state.terminal_status == "invalid":
            return 0.0
        outcomes = self.conditioned(state)
        return sum(probability * self.rewards[tokens] for tokens, probability in outcomes.items())

    def prefix_mass(self, prefix: Sequence[int]) -> float:
        """Unconditional mass of a prefix of the full token sequence."""
        prefix = tuple(prefix)
        for token in prefix:
            require_count(token, "prefix token")
        return sum(p for tokens, p in self.weights.items() if tokens[:len(prefix)] == prefix)

    def surviving_mass(self, rejected_prefixes: Sequence[Sequence[int]]) -> float:
        """Overlapping rejected cones are a union, so their mass is never counted twice."""
        prefixes = tuple(tuple(prefix) for prefix in rejected_prefixes)
        for prefix in prefixes:
            for token in prefix:
                require_count(token, "prefix token")
        return sum(p for tokens, p in self.weights.items() if not any(
            tokens[:len(prefix)] == prefix for prefix in prefixes
        ))

    def sample(self, state: SmilesState, rng: random.Random) -> tuple[int, ...]:
        outcomes = self.conditioned(state)
        tokens = list(outcomes)
        return rng.choices(tokens, weights=[outcomes[token] for token in tokens], k=1)[0]


class ToyModel:
    """Exact full-vocabulary conditionals of a consistent finite joint model."""

    model_id = "finite-joint-v1"

    def __init__(self, distribution: FiniteDistribution) -> None:
        self.distribution = distribution
        self.special_tokens = distribution.special_tokens
        self.vocab_size = 1 + max(token for tokens in distribution.weights for token in tokens)
        self.forward_calls = 0

    def logits_batch(self, states: Sequence[SmilesState]) -> torch.Tensor:
        if not states:
            return torch.empty((0, self.distribution.width, self.vocab_size), dtype=torch.float64)
        self.forward_calls += 1
        logits = torch.full(
            (len(states), self.distribution.width, self.vocab_size), -torch.inf, dtype=torch.float64,
        )
        for row, state in enumerate(states):
            outcomes = self.distribution.conditioned(state)
            for position in range(self.distribution.width):
                probabilities = {}
                for tokens, probability in outcomes.items():
                    token = tokens[position]
                    probabilities[token] = probabilities.get(token, 0.0) + probability
                for token, probability in probabilities.items():
                    logits[row, position, token] = math.log(probability)
        return logits


class ToyValueProvider:
    target = "finite_joint_conditional"

    def __init__(self, distribution: FiniteDistribution) -> None:
        self.distribution = distribution
        self.forward_calls = 0

    def values(self, states: Sequence[SmilesState]) -> list[float]:
        if states:
            self.forward_calls += 1
        return [self.distribution.exact_value(state) for state in states]

    def value(self, state: SmilesState) -> float:
        return self.values([state])[0]

    def state_dict(self) -> dict:
        return {"target": self.target, "forward_calls": self.forward_calls}

    def load_state_dict(self, payload: dict) -> None:
        if payload["target"] != self.target:
            raise ValueError("Toy value target differs")
        require_count(payload["forward_calls"], "toy value forward calls")
        self.forward_calls = payload["forward_calls"]


@pytest.fixture
def special_tokens() -> SpecialTokens:
    return SpecialTokens(mask_id=1, eos_id=2, pad_id=0)


@pytest.fixture
def toy_distribution(special_tokens: SpecialTokens) -> FiniteDistribution:
    # C=3, O=4. P(CC, CO, OC, OO)=(.1, .2, .3, .4); only CO succeeds.
    # The O-prefix is a failed cone of mass .7, and CC is a failed leaf of mass .1.
    weights = {(3, 3, 2, 0): 0.1, (3, 4, 2, 0): 0.2, (4, 3, 2, 0): 0.3, (4, 4, 2, 0): 0.4}
    return FiniteDistribution(weights, {tokens: float(tokens == (3, 4, 2, 0)) for tokens in weights}, special_tokens)


@pytest.fixture
def toy_initial_state(special_tokens: SpecialTokens) -> SmilesState:
    # EOS/PAD are fixed only in this finite example, not in primary generation.
    return SmilesState(
        (1, 1, 2, 0), (True, True, False, False), GROUPS[0], special_tokens,
        locked=(False, False, True, True),
    )


@pytest.fixture
def toy_terminal_states(toy_distribution: FiniteDistribution) -> list[SmilesState]:
    return [SmilesState(tokens, (False,) * len(tokens), GROUPS[0], toy_distribution.special_tokens)
            for tokens in toy_distribution.weights]


@pytest.fixture
def toy_model(toy_distribution: FiniteDistribution) -> ToyModel:
    return ToyModel(toy_distribution)


@pytest.fixture
def toy_value_provider(toy_distribution: FiniteDistribution) -> ToyValueProvider:
    return ToyValueProvider(toy_distribution)


@pytest.fixture
def seeded_rng() -> random.Random:
    return random.Random(0)


@pytest.fixture
def smiles_cases() -> list[dict[str, object]]:
    # Membership follows the planned benchmark SMARTS, including its broad N/CO rules.
    return [
        {"smiles": "C=CC(=O)OCC", "rdkit_valid": True, "groups": ("acrylates", "chain_extenders")},
        {"smiles": "CN=C=O", "rdkit_valid": True, "groups": ("isocyanates", "chain_extenders")},
        {"smiles": "NCCN", "rdkit_valid": True, "groups": ("chain_extenders",)},
        {"smiles": "CO", "rdkit_valid": True, "groups": ("chain_extenders",)},
        {"smiles": "c1ccccc1", "rdkit_valid": True, "groups": ()},
        {"smiles": "C1CC", "rdkit_valid": False, "groups": ()},
        {"smiles": "C(", "rdkit_valid": False, "groups": ()},
    ]


@pytest.fixture
def common_config() -> dict:
    return load_config()


@pytest.fixture
def artifact_store() -> Iterator[ArtifactStore]:
    # Keep fixture paths project-relative without pytest's pathlib-based tmp_path.
    os.makedirs(".pytest_cache", exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smiles-", dir=".pytest_cache") as directory:
        yield ArtifactStore(os.path.relpath(directory))
