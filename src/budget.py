"""Accounting at actual forward boundaries and reservations for nested queries.

Hard caps use sequence work (the sum of batch sizes). Raw NFE remains the number
of real base-model forward calls. A reservation holds a worst-case query's work
before it starts; completion is reserved separately. No backtrack/cache operation
increments NFE. One tracker belongs to one sequential worker, not to shared threads.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Iterator

from src.types import CostPhase, CostRecord, require_count, require_id


class BudgetExceeded(RuntimeError):
    """The requested forward cannot start within the available reservation/cap."""


@dataclass(frozen=True)
class MeasuredFlopRatio:
    """C_verif/C_base measured at these batch sizes and fixed model/input shapes.

    measurement_id identifies the persisted profiling result, including model,
    dtype, sequence length and profiling method. Parameter counts are not FLOPs.
    """

    verifier_to_base: float
    base_batch_size: int
    verifier_batch_size: int
    measurement_id: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.verifier_to_base) or self.verifier_to_base <= 0:
            raise ValueError("Measured FLOP ratio must be finite and positive")
        require_count(self.base_batch_size, "base_batch_size", positive=True)
        require_count(self.verifier_batch_size, "verifier_batch_size", positive=True)
        require_id(self.measurement_id, "measurement_id")


@dataclass(frozen=True)
class BudgetSnapshot:
    total: CostRecord
    by_phase: dict[str, CostRecord]
    base_sequence_limit: int | None
    verifier_sequence_limit: int | None
    reserved_base_sequences: int
    reserved_verifier_sequences: int
    adjusted_nfe: float | None
    flop_measurement_id: str | None
    base_batch_sizes: tuple[int, ...]
    verifier_batch_sizes: tuple[int, ...]


class Reservation:
    """Use as a context manager; unused work is released on every exit path."""

    def __init__(self, tracker: BudgetTracker, reservation_id: int, label: str,
                 base_sequences: int, verifier_sequences: int,
                 allowed_phases: frozenset[CostPhase]) -> None:
        self._tracker = tracker
        self._id = reservation_id
        self.label = label
        self._base_remaining = base_sequences
        self._verifier_remaining = verifier_sequences
        self._allowed_phases = allowed_phases
        self._active = True

    @property
    def base_remaining(self) -> int:
        return self._base_remaining

    @property
    def verifier_remaining(self) -> int:
        return self._verifier_remaining

    def close(self) -> None:
        if self._active:
            if self._tracker._reservations.get(self._id) is not self:
                raise ValueError("Reservation does not belong to this tracker")
            self._tracker._reservations.pop(self._id)
            self._active = False

    def __enter__(self) -> Reservation:
        if not self._active:
            raise ValueError("Reservation is already closed")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


class BudgetTracker:
    def __init__(self, *, base_sequence_limit: int | None,
                 verifier_sequence_limit: int | None = None,
                 flop_ratio: MeasuredFlopRatio | None = None) -> None:
        for name, value in (("base_sequence_limit", base_sequence_limit),
                            ("verifier_sequence_limit", verifier_sequence_limit)):
            if value is not None:
                require_count(value, name)
        self.base_sequence_limit = base_sequence_limit
        self.verifier_sequence_limit = verifier_sequence_limit
        self.flop_ratio = flop_ratio
        self._total = CostRecord()
        self._by_phase: dict[CostPhase, CostRecord] = {}
        self._reservations: dict[int, Reservation] = {}
        self._next_id = 0
        self._base_batch_sizes: set[int] = set()
        self._verifier_batch_sizes: set[int] = set()

    def _held(self, kind: str) -> int:
        return sum(getattr(item, f"_{kind}_remaining") for item in self._reservations.values())

    def _fits(self, kind: str, count: int) -> bool:
        limit = getattr(self, f"{kind}_sequence_limit")
        used = getattr(self._total, f"{kind}_sequences")
        return limit is None or used + self._held(kind) + count <= limit

    def _reserve(self, *, label: str, base_sequences: int, verifier_sequences: int,
                 allowed_phases: frozenset[CostPhase]) -> Reservation | None:
        require_id(label, "reservation label")
        require_count(base_sequences, "reserved base sequences")
        require_count(verifier_sequences, "reserved verifier sequences")
        if not self._fits("base", base_sequences) or not self._fits("verifier", verifier_sequences):
            return None
        reservation = Reservation(self, self._next_id, label, base_sequences,
                                  verifier_sequences, allowed_phases)
        self._reservations[self._next_id] = reservation
        self._next_id += 1
        return reservation

    def reserve_query(self, *, label: str, base_sequences: int,
                      verifier_sequences: int = 0) -> Reservation | None:
        """Reserve the whole fixed-m query, including discovery/IS/fallback work.

        The caller derives a conservative bound from the actual completion
        policy. None means the query must not begin; it is not a value of zero.
        Reserve terminal forward-only completion separately, before this query.
        """
        return self._reserve(label=label, base_sequences=base_sequences,
                             verifier_sequences=verifier_sequences,
                             allowed_phases=frozenset(CostPhase) - {CostPhase.COMPLETION})

    def reserve_completion(self, *, base_sequences: int,
                           verifier_sequences: int = 0) -> Reservation | None:
        return self._reserve(label="completion", base_sequences=base_sequences,
                             verifier_sequences=verifier_sequences,
                             allowed_phases=frozenset({CostPhase.COMPLETION}))

    def _charge(self, kind: str, batch_size: int, phase: CostPhase,
                reservation: Reservation | None) -> None:
        require_count(batch_size, "forward batch size", positive=True)
        if reservation is None:
            if not self._fits(kind, batch_size):
                raise BudgetExceeded(f"Insufficient unreserved {kind} sequence budget")
        else:
            if (reservation._tracker is not self or not reservation._active
                    or self._reservations.get(reservation._id) is not reservation):
                raise ValueError("Reservation is closed or belongs to another tracker")
            if phase not in reservation._allowed_phases:
                raise ValueError(f"Reservation {reservation.label} cannot fund phase {phase.value}")
            name = f"_{kind}_remaining"
            remaining = getattr(reservation, name)
            if remaining < batch_size:
                raise BudgetExceeded(f"Query exceeded its reserved {kind} sequence work")
            setattr(reservation, name, remaining - batch_size)
        self._add(CostRecord(**{f"{kind}_calls": 1, f"{kind}_sequences": batch_size}), phase)
        getattr(self, f"_{kind}_batch_sizes").add(batch_size)

    def _add(self, cost: CostRecord, phase: CostPhase) -> None:
        self._total = self._total + cost
        self._by_phase[phase] = self._by_phase.get(phase, CostRecord()) + cost

    @contextmanager
    def _forward(self, kind: str, *, batch_size: int, phase: CostPhase,
                 reservation: Reservation | None) -> Iterator[None]:
        phase = CostPhase(phase)
        self._charge(kind, batch_size, phase, reservation)
        started = time.perf_counter()
        try:
            # Adapter places exactly one real model(...) call inside this block.
            yield
        finally:
            self._add(CostRecord(elapsed_seconds=time.perf_counter() - started), phase)

    def base_forward(self, *, batch_size: int, phase: CostPhase,
                     reservation: Reservation | None = None) -> Any:
        """Wrap one base forward; charge before starting it, including failed calls."""
        return self._forward("base", batch_size=batch_size, phase=phase, reservation=reservation)

    def verifier_forward(self, *, batch_size: int, phase: CostPhase,
                         reservation: Reservation | None = None) -> Any:
        return self._forward("verifier", batch_size=batch_size, phase=phase, reservation=reservation)

    @contextmanager
    def oracle_check(self, *, phase: CostPhase) -> Iterator[None]:
        phase = CostPhase(phase)
        self._add(CostRecord(oracle_calls=1), phase)
        started = time.perf_counter()
        try:
            yield
        finally:
            self._add(CostRecord(elapsed_seconds=time.perf_counter() - started), phase)

    @property
    def raw_nfe(self) -> int:
        return self._total.base_calls

    def snapshot(self) -> BudgetSnapshot:
        """elapsed_seconds sums instrumented call times, not end-to-end wall time.

        GPU timing requires synchronization/profiling in the adapter. Adjusted
        NFE is unavailable if a verifier was used without a matching FLOP profile
        or if actual batch sizes differ from the measured forward shapes.
        """
        adjusted: float | None = None
        if self._total.verifier_calls == 0:
            adjusted = float(self.raw_nfe)
        elif self.flop_ratio is not None:
            ratio = self.flop_ratio
            if (self._base_batch_sizes.issubset({ratio.base_batch_size})
                    and self._verifier_batch_sizes.issubset({ratio.verifier_batch_size})):
                adjusted = self.raw_nfe + self._total.verifier_calls * ratio.verifier_to_base
        return BudgetSnapshot(
            total=replace(self._total),
            by_phase={phase.value: replace(cost) for phase, cost in self._by_phase.items()},
            base_sequence_limit=self.base_sequence_limit,
            verifier_sequence_limit=self.verifier_sequence_limit,
            reserved_base_sequences=self._held("base"),
            reserved_verifier_sequences=self._held("verifier"),
            adjusted_nfe=adjusted,
            flop_measurement_id=self.flop_ratio.measurement_id if self.flop_ratio else None,
            base_batch_sizes=tuple(sorted(self._base_batch_sizes)),
            verifier_batch_sizes=tuple(sorted(self._verifier_batch_sizes)),
        )
