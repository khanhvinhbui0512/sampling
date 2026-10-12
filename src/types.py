"""Shared, model-independent contracts. Token positions always include special tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


def require_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def require_count(value: int, name: str, *, positive: bool = False) -> None:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class TokenRules:
    """IDs must come from the selected tokenizer, never from a guessed vocabulary."""

    vocab_size: int
    sequence_length: int
    mask_id: int
    bos_id: int | None = None
    eos_id: int | None = None
    pad_id: int | None = None
    unk_id: int | None = None

    def __post_init__(self) -> None:
        require_count(self.vocab_size, "vocab_size", positive=True)
        require_count(self.sequence_length, "sequence_length", positive=True)
        self.validate_token(self.mask_id)
        for name in ("bos_id", "eos_id", "pad_id", "unk_id"):
            token = getattr(self, name)
            if token is not None:
                self.validate_token(token)
                if token == self.mask_id:
                    raise ValueError(f"{name} cannot equal mask_id")

    def validate_token(self, token: int) -> None:
        require_count(token, "token ID")
        if token >= self.vocab_size:
            raise ValueError(f"Token ID {token} is outside the tokenizer vocabulary")


@dataclass(frozen=True)
class MaskedState:
    """A fixed-length token state, including arbitrary, non-prefix masks."""

    token_ids: tuple[int, ...]
    rules: TokenRules

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        if len(self.token_ids) != self.rules.sequence_length:
            raise ValueError("State length differs from TokenRules.sequence_length")
        for token in self.token_ids:
            self.rules.validate_token(token)

    @property
    def masked_positions(self) -> tuple[int, ...]:
        return tuple(i for i, token in enumerate(self.token_ids) if token == self.rules.mask_id)

    @property
    def is_complete(self) -> bool:
        return self.rules.mask_id not in self.token_ids

    def reveal(self, assignments: Mapping[int, int]) -> MaskedState:
        tokens = list(self.token_ids)
        for position, token in assignments.items():
            require_count(position, "position")
            if position >= len(tokens):
                raise ValueError("Position is outside the state")
            self.rules.validate_token(token)
            if token == self.rules.mask_id:
                raise ValueError("A reveal cannot assign MASK")
            if tokens[position] != self.rules.mask_id and tokens[position] != token:
                raise ValueError("A completion cannot change an already known token")
            tokens[position] = token
        return MaskedState(tuple(tokens), self.rules)

    def accepts_completion(self, candidate: MaskedState) -> bool:
        return (
            self.rules == candidate.rules
            and candidate.is_complete
            and all(
                token == self.rules.mask_id or token == other
                for token, other in zip(self.token_ids, candidate.token_ids)
            )
        )


class TargetKind(str, Enum):
    RHO = "v_rho"
    NATIVE_ROLLOUT = "native_rollout"
    REFERENCE_CONDITIONAL = "reference_conditional_v_star"


@dataclass(frozen=True)
class ValueTarget:
    """Identity of a binary terminal-reward expectation, not a consistency proof.

    REFERENCE_CONDITIONAL requires a separately verified joint/conditional reference.
    Reusing a checkpoint alone does not make q_rho equal that conditional law.
    """

    target_id: str
    kind: TargetKind
    reference_id: str
    policy_id: str
    reward_id: str
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", TargetKind(self.kind))
        for name in ("target_id", "reference_id", "policy_id", "reward_id"):
            require_id(getattr(self, name), name)
        if self.calibration_id is not None:
            require_id(self.calibration_id, "calibration_id")


class CostPhase(str, Enum):
    SAMPLING = "sampling"
    DISCOVERY = "discovery"
    ESTIMATION = "estimation"
    IMPORTANCE = "importance"
    COMPLETION = "completion"
    VERIFICATION = "verification"
    CALIBRATION = "calibration"
    TRAINING = "training"


@dataclass(frozen=True)
class CostRecord:
    """Realized calls and work, including failed calls that actually started.

    base_calls is raw NFE. base_sequences counts the sum of forward batch sizes.
    Oracle checks are distinct from learned verifier forwards.
    """

    base_calls: int = 0
    base_sequences: int = 0
    verifier_calls: int = 0
    verifier_sequences: int = 0
    oracle_calls: int = 0
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "base_calls", "base_sequences", "verifier_calls", "verifier_sequences", "oracle_calls"
        ):
            require_count(getattr(self, name), name)
        if self.base_calls > self.base_sequences or self.verifier_calls > self.verifier_sequences:
            raise ValueError("Every forward must include at least one sequence")
        if ((self.base_calls == 0) != (self.base_sequences == 0)
                or (self.verifier_calls == 0) != (self.verifier_sequences == 0)):
            raise ValueError("Sequence work must come from an actual forward call")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")

    def __add__(self, other: CostRecord) -> CostRecord:
        if not isinstance(other, CostRecord):
            return NotImplemented
        return CostRecord(**{
            name: getattr(self, name) + getattr(other, name)
            for name in self.__dataclass_fields__
        })


class EstimateStatus(str, Enum):
    ESTIMATED = "estimated"
    EXACT = "exact"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"


@dataclass(frozen=True)
class ValueEstimate:
    """Unavailable estimates use value=None, never a fabricated reward of zero.

    surviving_mass is CARS's bound B_W, distinct from the value estimate. Draws
    count fresh pre-acceptance samples, with multiplicity, from a frozen proposal.
    """

    target: ValueTarget
    status: EstimateStatus
    value: float | None
    cost: CostRecord = field(default_factory=CostRecord)
    draws: int = 0
    standard_error: float | None = None
    interval: tuple[float, float] | None = None
    surviving_mass: float | None = None
    cache_hit: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", EstimateStatus(self.status))
        require_count(self.draws, "draws")
        available = self.status in (EstimateStatus.ESTIMATED, EstimateStatus.EXACT)
        if available != (self.value is not None):
            raise ValueError("Only estimated/exact results may contain a value")
        for name in ("value", "surviving_mass"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be a finite probability")
        if self.standard_error is not None:
            if not available or not math.isfinite(self.standard_error) or self.standard_error < 0:
                raise ValueError("standard_error requires an available estimate and must be >= 0")
        if self.interval is not None:
            low, high = self.interval
            if not available or not (0 <= low <= self.value <= high <= 1):
                raise ValueError("interval must contain the estimate within [0, 1]")
        if self.value is not None and self.surviving_mass is not None:
            if self.value > self.surviving_mass + 1e-12:
                raise ValueError("A CARS value cannot exceed its surviving-mass bound")
        if self.status == EstimateStatus.FAILED and not self.error:
            raise ValueError("A failed estimate must explain the failure")


class SampleStatus(str, Enum):
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"


@dataclass(frozen=True)
class SampleResult:
    sample_id: str
    state: MaskedState
    target: ValueTarget
    status: SampleStatus
    cost: CostRecord = field(default_factory=CostRecord)
    smiles: str | None = None
    canonical_smiles: str | None = None
    valid: bool | None = None
    score: float | None = None
    reward: int | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        require_id(self.sample_id, "sample_id")
        object.__setattr__(self, "status", SampleStatus(self.status))
        if self.status == SampleStatus.COMPLETED and not self.state.is_complete:
            raise ValueError("A completed sample must contain no MASK tokens")
        if not self.state.is_complete and any(
            value is not None
            for value in (self.smiles, self.canonical_smiles, self.valid, self.score, self.reward)
        ):
            raise ValueError("Do not decode or score an incomplete token state as a molecule")
        if self.score is not None and (not math.isfinite(self.score) or not 0 <= self.score <= 1):
            raise ValueError("QM9 score must be QED in [0, 1], or zero for invalid molecules")
        if self.valid is not None and type(self.valid) is not bool:
            raise ValueError("Molecular validity must be bool or None before checking")
        if self.score is not None and self.valid is None:
            raise ValueError("A molecular score requires a validity assessment")
        if self.reward is not None and (type(self.reward) is not int or self.reward not in (0, 1)):
            raise ValueError("Terminal reward must be binary or None before calibration")
        if self.reward is not None and self.target.calibration_id is None:
            raise ValueError("QM9 Pass@95 reward requires a calibration ID")
        if self.reward == 1 and self.valid is not True:
            raise ValueError("An invalid or unchecked molecule cannot be a success")
        if self.valid is False and self.score not in (None, 0):
            raise ValueError("An invalid molecule must have score zero")
        if self.canonical_smiles is not None and self.valid is not True:
            raise ValueError("Canonical molecular identity requires a valid molecule")
        if self.status == SampleStatus.FAILED and not self.error:
            raise ValueError("A failed sample must explain the failure")


@runtime_checkable
class Backbone(Protocol):
    """Adapters return [batch, position, vocabulary] logits without revealing tokens.

    Implementations instrument each real model forward with the supplied budget.
    Any permits NumPy/list fake backbones without importing Torch in this module.
    """

    reference_id: str
    checkpoint_id: str
    tokenizer_id: str
    token_rules: TokenRules

    def logits_batch(self, states: Sequence[MaskedState], *, budget: Any,
                     phase: CostPhase, reservation: Any = None) -> Any: ...


@runtime_checkable
class CompletionPolicy(Protocol):
    policy_id: str
    target: ValueTarget

    def complete(self, state: MaskedState, *, rng: Any, budget: Any,
                 phase: CostPhase, reservation: Any = None) -> SampleResult: ...


@runtime_checkable
class Sampler(Protocol):
    method_id: str
    target: ValueTarget

    def sample(self, initial_state: MaskedState, *, sample_id: str,
               rng: Any, budget: Any) -> SampleResult: ...
