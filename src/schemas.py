"""Shared contracts for the single SMILES benchmark.

These contracts are new project infrastructure. Terminal anchoring follows
MDM-VGB; root attempts and CARS prefix rejections remain separate observations.
No token IDs are assumed: the tokenizer supplies MASK, EOS, and PAD IDs.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor

SCHEMA_VERSION = 1
GROUPS = ("acrylates", "chain_extenders", "isocyanates")
SUITES = ("primary", "replication", "smoke")
METHODS = ("base", "cars", "mdm_vgb", "hybrid", "cars_ar")
RUN_STATUSES = (
    "running", "completed", "interrupted", "failed", "timeout", "budget_exhausted",
)
Suite = Literal["primary", "replication", "smoke"]
Method = Literal["base", "cars", "mdm_vgb", "hybrid", "cars_ar"]
RunStatus = Literal[
    "running", "completed", "interrupted", "failed", "timeout", "budget_exhausted",
]
AttemptKind = Literal["complete", "prefix_abort", "budget_exhausted", "error"]


def require_identifier(value: str, label: str = "identifier") -> str:
    """Accept one filename component, never a path or an empty identifier."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def require_sha256(value: str, label: str = "checksum") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def require_count(value: int, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'} integer")
    return value


def require_nonnegative(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return float(value)


@dataclass(frozen=True, slots=True)
class SpecialTokens:
    mask_id: int
    eos_id: int
    pad_id: int

    def __post_init__(self) -> None:
        ids = (self.mask_id, self.eos_id, self.pad_id)
        for token in ids:
            require_count(token, "special token ID")
        if len(set(ids)) != 3:
            raise ValueError("MASK, EOS, and PAD must have distinct IDs")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SpecialTokens:
        return cls(**dict(data))


@dataclass(frozen=True, slots=True)
class SmilesState:
    """Fixed-width state; True in mask means an unresolved coordinate.

    EOS is terminal only after all preceding coordinates are known. A known
    non-PAD token after EOS is an invalid terminal, rather than silently being
    discarded. Unresolved coordinates after a valid EOS are outside the molecule.
    Invalid complete states are representable so their actual reward can be zero.
    """

    token_ids: tuple[int, ...]
    mask: tuple[bool, ...]
    group: str
    special_tokens: SpecialTokens
    locked: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        object.__setattr__(self, "mask", tuple(self.mask))
        object.__setattr__(self, "locked", tuple(self.locked) or (False,) * len(self.token_ids))
        if self.group not in GROUPS:
            raise ValueError(f"Unknown molecular group: {self.group}")
        if not isinstance(self.special_tokens, SpecialTokens):
            raise TypeError("special_tokens must be SpecialTokens")
        if not self.token_ids or not (len(self.token_ids) == len(self.mask) == len(self.locked)):
            raise ValueError("token_ids, mask, and locked must have the same positive length")
        for token, masked, locked in zip(self.token_ids, self.mask, self.locked, strict=True):
            require_count(token, "token ID")
            if type(masked) is not bool or type(locked) is not bool:
                raise TypeError("mask and locked must contain booleans")
            if masked != (token == self.special_tokens.mask_id):
                raise ValueError("mask must exactly identify MASK token IDs")
            if masked and locked:
                raise ValueError("An unresolved coordinate cannot be locked")

    @classmethod
    def fully_masked(cls, length: int, group: str, special_tokens: SpecialTokens) -> SmilesState:
        require_count(length, "length", positive=True)
        return cls((special_tokens.mask_id,) * length, (True,) * length, group, special_tokens)

    @property
    def masked_positions(self) -> tuple[int, ...]:
        return tuple(i for i, masked in enumerate(self.mask) if masked)

    @property
    def eos_position(self) -> int | None:
        try:
            return self.token_ids.index(self.special_tokens.eos_id)
        except ValueError:
            return None

    @property
    def terminal_status(self) -> Literal["incomplete", "complete", "invalid"]:
        eos = self.eos_position
        if eos is None:
            return "incomplete" if any(self.mask) else "invalid"
        if any(self.mask[:eos]):
            return "incomplete"
        if self.special_tokens.pad_id in self.token_ids[:eos]:
            return "invalid"
        if any(
            token not in (self.special_tokens.pad_id, self.special_tokens.mask_id)
            for token in self.token_ids[eos + 1:]
        ):
            return "invalid"
        return "complete"

    @property
    def is_terminal(self) -> bool:
        return self.terminal_status != "incomplete"

    def molecular_token_ids(self) -> tuple[int, ...]:
        if self.terminal_status != "complete":
            raise ValueError(f"Cannot decode a {self.terminal_status} state as a complete SMILES")
        return self.token_ids[:self.eos_position]

    def with_tokens(self, updates: Mapping[int, int]) -> SmilesState:
        tokens = list(self.token_ids)
        for position, token in updates.items():
            require_count(position, "coordinate")
            require_count(token, "token ID")
            if position >= len(tokens):
                raise IndexError(f"Coordinate out of range: {position}")
            if self.locked[position] and token != tokens[position]:
                raise ValueError(f"Coordinate {position} is locked")
            tokens[position] = token
        return replace(
            self, token_ids=tuple(tokens),
            mask=tuple(token == self.special_tokens.mask_id for token in tokens),
        )

    def remask(self, positions: Sequence[int]) -> SmilesState:
        return self.with_tokens(dict.fromkeys(positions, self.special_tokens.mask_id))

    def cache_key(self) -> tuple[Any, ...]:
        """Providers additionally key caches by model/tokenizer/reward/policy hashes."""
        return (
            self.group, self.token_ids, self.mask, self.locked,
            self.special_tokens.mask_id, self.special_tokens.eos_id, self.special_tokens.pad_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SmilesState:
        values = dict(data)
        values["special_tokens"] = SpecialTokens.from_dict(values["special_tokens"])
        return cls(**values)


@runtime_checkable
class Tokenizer(Protocol):
    """The project's SMILES tokenizer; pretrained AR adapters retain their own tokenizer."""

    special_tokens: SpecialTokens
    vocab_size: int

    def encode_text(self, text: str, *, add_eos: bool = True) -> list[int]: ...
    def decode_state(self, state: SmilesState) -> str: ...


@runtime_checkable
class BaseModel(Protocol):
    """MDLM: full-vocabulary logits shaped [batch, state length, vocabulary]."""

    model_id: str
    vocab_size: int
    special_tokens: SpecialTokens

    def logits_batch(self, states: Sequence[SmilesState]) -> Tensor: ...


@runtime_checkable
class AutoregressiveModel(Protocol):
    """AR replication: prefixes include the prompt; logits have shape [batch, vocabulary]."""

    model_id: str
    vocab_size: int
    eos_token_ids: tuple[int, ...]

    def next_token_logits(self, prefixes: Sequence[Sequence[int]]) -> Tensor: ...


@runtime_checkable
class ValueProvider(Protocol):
    """Raw values in [0, 1]; guidance floors never redefine terminal reward."""

    target: str

    def value(self, state: SmilesState) -> float: ...
    def values(self, states: Sequence[SmilesState]) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class RewardResult:
    reward: float
    reward_version: str
    grammar_valid: bool
    rdkit_valid: bool
    class_member: bool
    canonical_smiles: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.reward, bool) or self.reward not in (0.0, 1.0):
            raise ValueError("Terminal reward must be binary, without an epsilon floor")
        require_identifier(self.reward_version, "reward version")
        if any(type(flag) is not bool for flag in (
            self.grammar_valid, self.rdkit_valid, self.class_member,
        )):
            raise TypeError("Checker results must be booleans")
        if self.class_member and not self.rdkit_valid:
            raise ValueError("Class membership requires an RDKit-valid molecule")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RewardResult:
        return cls(**dict(data))


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """One root-start attempt; leaf_visits may exceed one for VGB search."""

    run_id: str
    attempt_id: int
    method: Method
    group: str
    kind: AttemptKind
    leaf_visits: int = 0
    smiles: str | None = None
    state: SmilesState | None = None
    reward_result: RewardResult | None = None
    elapsed_seconds: float = 0.0
    counters: dict[str, int] = field(default_factory=dict)
    reason: str | None = None

    def __post_init__(self) -> None:
        require_identifier(self.run_id, "run ID")
        require_count(self.attempt_id, "attempt ID", positive=True)
        require_count(self.leaf_visits, "leaf visits")
        require_nonnegative(self.elapsed_seconds, "attempt time")
        if self.method not in METHODS or self.group not in GROUPS:
            raise ValueError("Unknown method or group")
        if self.kind not in ("complete", "prefix_abort", "budget_exhausted", "error"):
            raise ValueError(f"Unknown attempt kind: {self.kind}")
        if self.kind == "complete" and (self.smiles is None or self.reward_result is None):
            raise ValueError("A complete attempt must retain its output and checker results")
        if self.state is not None and self.state.group != self.group:
            raise ValueError("Attempt/state group mismatch")
        for name, count in self.counters.items():
            require_count(count, name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AttemptRecord:
        values = dict(data)
        if values.get("state") is not None:
            values["state"] = SmilesState.from_dict(values["state"])
        if values.get("reward_result") is not None:
            values["reward_result"] = RewardResult.from_dict(values["reward_result"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """A selected success, recorded separately from the attempt denominator."""

    run_id: str
    sample_id: int
    attempt_id: int
    group: str
    smiles: str
    identity: str
    reward_result: RewardResult
    identity_type: Literal["canonical_smiles", "raw_smiles"] = "canonical_smiles"

    def __post_init__(self) -> None:
        require_identifier(self.run_id, "run ID")
        require_count(self.sample_id, "sample ID", positive=True)
        require_count(self.attempt_id, "attempt ID", positive=True)
        if self.group not in GROUPS or not self.identity or not self.smiles:
            raise ValueError("A selected sample needs a group, output, and identity")
        if self.reward_result.reward != 1.0:
            raise ValueError("Selected samples must satisfy the suite's terminal reward")
        if self.identity_type not in ("canonical_smiles", "raw_smiles"):
            raise ValueError("Unknown sample identity type")
        if self.identity_type == "canonical_smiles" and (
            not self.reward_result.rdkit_valid
            or self.reward_result.canonical_smiles != self.identity
        ):
            raise ValueError("Molecular identity must match the checker canonical SMILES")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SampleRecord:
        values = dict(data)
        values["reward_result"] = RewardResult.from_dict(values["reward_result"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class TrackingSnapshot:
    counters: dict[str, int] = field(default_factory=dict)
    compute_seconds: dict[str, float] = field(default_factory=dict)
    gpu_seconds: dict[str, float] = field(default_factory=dict)
    flops: dict[str, float] = field(default_factory=dict)
    end_to_end_seconds: float = 0.0
    peak_gpu_memory_bytes: int = 0

    def __post_init__(self) -> None:
        for name, count in self.counters.items():
            require_count(count, name)
        for values in (self.compute_seconds, self.gpu_seconds, self.flops):
            for name, value in values.items():
                require_nonnegative(value, name)
        require_nonnegative(self.end_to_end_seconds, "end-to-end time")
        require_count(self.peak_gpu_memory_bytes, "peak GPU memory")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrackingSnapshot:
        return cls(**dict(data))


@dataclass(frozen=True, slots=True)
class RunIdentity:
    task_id: str
    run_id: str
    config_hash: str
    suite: Suite
    seed: int
    reward_version: str
    method: Method | None = None
    group: str | None = None
    dependency_hashes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_identifier(self.task_id, "task ID")
        require_identifier(self.run_id, "run ID")
        require_sha256(self.config_hash, "config hash")
        require_identifier(self.reward_version, "reward version")
        require_count(self.seed, "seed")
        if self.seed >= 2**32:
            raise ValueError("Seed must be smaller than 2**32")
        if self.suite not in SUITES or (self.method is not None and self.method not in METHODS):
            raise ValueError("Unknown suite or method")
        if self.group is not None and self.group not in GROUPS:
            raise ValueError("Unknown molecular group")
        for name, digest in self.dependency_hashes.items():
            require_identifier(name, "dependency name")
            require_sha256(digest, f"{name} hash")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunIdentity:
        return cls(**dict(data))


@dataclass(frozen=True, slots=True)
class RunRecord:
    identity: RunIdentity
    status: RunStatus
    created_at: str
    updated_at: str
    tracking: TrackingSnapshot = field(default_factory=TrackingSnapshot)
    artifacts: tuple[str, ...] = ()
    summary: str = ""
    error: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.status not in RUN_STATUSES:
            raise ValueError("Unsupported run schema or status")
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if not self.created_at or not self.updated_at:
            raise ValueError("Run timestamps are required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunRecord:
        values = dict(data)
        values["identity"] = RunIdentity.from_dict(values["identity"])
        values["tracking"] = TrackingSnapshot.from_dict(values["tracking"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    kind: str
    config_hash: str
    files: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_identifier(self.kind, "artifact kind")
        require_sha256(self.config_hash, "config hash")
        if self.schema_version != SCHEMA_VERSION or not self.files:
            raise ValueError("An artifact manifest needs a supported schema and at least one file")
        for name, digest in self.files.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Manifest file paths must be nonempty strings")
            require_sha256(digest, f"{name} hash")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArtifactManifest:
        return cls(**dict(data))


@dataclass(frozen=True, slots=True)
class TaskResult:
    summary: str
    status: RunStatus = "completed"
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in RUN_STATUSES or self.status == "running":
            raise ValueError("A returned task result must have a final status")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("A task result needs a short summary")
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
