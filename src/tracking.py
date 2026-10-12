"""Count sampler work and measure compute independently of elapsed run time.

Raw NFE counts actual MDLM/AR forward invocations, including hybrid completions.
Batch-example counts are also retained. Adjusted NFE is unavailable until every
forward has a FLOP measurement using the same reference unit. Worker elapsed time
is measured here; Modal wrappers can add measured queue/startup/RPC overhead.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Literal

from src.artifacts import utc_now
from src.schemas import (
    TrackingSnapshot,
    require_count,
    require_identifier,
    require_nonnegative,
)

ModelKind = Literal["mdlm", "ar", "verifier"]
COUNTERS = (
    "mdlm_forward_calls", "ar_forward_calls", "verifier_forward_calls",
    "mdlm_forward_examples", "ar_forward_examples", "verifier_forward_examples",
    "inner_mdlm_forward_calls", "inner_ar_forward_calls",
    "inner_mdlm_forward_examples", "inner_ar_forward_examples",
    "flop_profiled_mdlm_calls", "flop_profiled_ar_calls", "flop_profiled_verifier_calls",
    "value_queries", "cache_hits", "cache_misses", "cars_attempts", "cars_inner_attempts",
    "root_attempts", "prefix_aborts", "leaf_visits", "complete_candidates", "selected_samples",
    "inner_prefix_aborts", "inner_leaf_visits", "inner_complete_candidates",
    "markov_steps", "grammar_checks", "reward_checks",
)


class BudgetExceeded(RuntimeError):
    def __init__(self, budget: str) -> None:
        self.budget = budget
        super().__init__(f"Compute budget exhausted: {budget}")


class CostTracker:
    def __init__(
        self, *, run_id: str, event_sink: Callable[[dict[str, Any]], Any] | None = None,
        max_model_nfe: int | None = None, max_compute_seconds: float | None = None,
    ) -> None:
        self.run_id = require_identifier(run_id, "run ID")
        if max_model_nfe is not None:
            require_count(max_model_nfe, "max_model_nfe", positive=True)
        if max_compute_seconds is not None:
            require_nonnegative(max_compute_seconds, "max_compute_seconds")
            if max_compute_seconds == 0:
                raise ValueError("max_compute_seconds must be positive")
        self.max_model_nfe = max_model_nfe
        self.max_compute_seconds = max_compute_seconds
        self.event_sink = event_sink
        self.counters = dict.fromkeys(COUNTERS, 0)
        self.compute_seconds: dict[str, float] = {}
        self.gpu_seconds: dict[str, float] = {}
        self.flops: dict[str, float] = {}
        self.peak_gpu_memory_bytes = 0
        self._started = time.perf_counter()
        self._elapsed_before = 0.0
        self._frames: list[dict[str, Any]] = []

    @property
    def raw_nfe(self) -> int:
        return self.counters["mdlm_forward_calls"] + self.counters["ar_forward_calls"]

    @property
    def model_example_nfe(self) -> int:
        return self.counters["mdlm_forward_examples"] + self.counters["ar_forward_examples"]

    @property
    def tracked_compute_seconds(self) -> float:
        return sum(self.compute_seconds.values())

    @property
    def end_to_end_seconds(self) -> float:
        return self._elapsed_before + time.perf_counter() - self._started

    def add_external_wall_time(self, seconds: float) -> None:
        """Add measured overhead once; never estimate it from the compute timers."""
        self._elapsed_before += require_nonnegative(seconds, "external elapsed time")

    def event(self, name: str, **fields: Any) -> None:
        require_identifier(name, "event name")
        if {"timestamp", "run_id", "event"}.intersection(fields):
            raise ValueError("Event payload must not override its identity fields")
        if self.event_sink is not None:
            self.event_sink({"timestamp": utc_now(), "run_id": self.run_id, "event": name, **fields})

    def increment(self, name: str, count: int = 1) -> None:
        if name not in self.counters:
            raise ValueError(f"Unknown counter: {name}")
        require_count(count, name)
        self.counters[name] += count

    def require_budget(self, *, model_calls: int = 0) -> None:
        require_count(model_calls, "requested model calls")
        if self.max_model_nfe is not None and self.raw_nfe + model_calls > self.max_model_nfe:
            raise BudgetExceeded("model_nfe")
        if self.max_compute_seconds is not None and self.tracked_compute_seconds >= self.max_compute_seconds:
            raise BudgetExceeded("compute_seconds")

    def record_model_forward(
        self, kind: ModelKind, *, batch_size: int = 1, inner: bool = False,
        flops: float | None = None,
    ) -> None:
        """Call exactly once per actual forward; inner counters are subsets of totals."""
        if kind not in ("mdlm", "ar", "verifier"):
            raise ValueError(f"Unknown model kind: {kind}")
        require_count(batch_size, "batch size", positive=True)
        if type(inner) is not bool or (inner and kind == "verifier"):
            raise ValueError("inner denotes MDLM/AR CARS completion forwards")
        if flops is not None:
            require_nonnegative(flops, "forward FLOPs")
            if flops == 0:
                raise ValueError("Profiled forward FLOPs must be positive")
        self.increment(f"{kind}_forward_calls")
        self.increment(f"{kind}_forward_examples", batch_size)
        if inner:
            self.increment(f"inner_{kind}_forward_calls")
            self.increment(f"inner_{kind}_forward_examples", batch_size)
        if flops is not None:
            self.increment(f"flop_profiled_{kind}_calls")
            self.flops[kind] = self.flops.get(kind, 0.0) + float(flops)

    @contextmanager
    def model_call(
        self, kind: ModelKind, *, batch_size: int = 1, inner: bool = False,
        flops: float | None = None, gpu: bool = False, device: Any = None,
    ) -> Iterator[None]:
        """Wrap just the model invocation; do not also record_model_forward for it."""
        self.require_budget(model_calls=0 if kind == "verifier" else 1)
        with self.measure(kind, gpu=gpu, device=device):
            self.record_model_forward(kind, batch_size=batch_size, inner=inner, flops=flops)
            yield

    def record_root_attempt(self) -> int:
        self.increment("root_attempts")
        attempt_id = self.counters["root_attempts"]
        self.event("root_attempt", attempt_id=attempt_id)
        return attempt_id

    def record_cars_attempt(self, *, inner: bool = False) -> None:
        if type(inner) is not bool:
            raise TypeError("inner must be a boolean")
        self.increment("cars_attempts")
        if inner:
            self.increment("cars_inner_attempts")

    def record_prefix_abort(self, *, inner: bool = False) -> None:
        if type(inner) is not bool:
            raise TypeError("inner must be a boolean")
        self.increment("inner_prefix_aborts" if inner else "prefix_aborts")

    def record_leaf_visit(self, *, inner: bool = False) -> None:
        if type(inner) is not bool:
            raise TypeError("inner must be a boolean")
        self.increment("inner_leaf_visits" if inner else "leaf_visits")

    def record_complete_candidate(self, *, inner: bool = False) -> None:
        if type(inner) is not bool:
            raise TypeError("inner must be a boolean")
        self.increment("inner_complete_candidates" if inner else "complete_candidates")

    @contextmanager
    def measure(self, component: str, *, gpu: bool = False, device: Any = None) -> Iterator[None]:
        """CPU wall durations are exclusive; nested GPU measurements are disallowed.

        CUDA events measure the chosen stream and synchronize its end event.
        CPU duration includes waiting for GPU work. GPU durations are diagnostic
        measurements and are not added a second time to compute_seconds.
        """
        require_identifier(component, "compute component")
        if type(gpu) is not bool:
            raise TypeError("gpu must be a boolean")
        start_event = end_event = stream = None
        if gpu:
            if any(frame["gpu"] for frame in self._frames):
                raise ValueError("Nested GPU timers would double-count GPU work")
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("GPU timing requested without CUDA")
            with torch.cuda.device(device):
                stream = torch.cuda.current_stream()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(stream)
        frame = {"start": time.perf_counter(), "children": 0.0, "gpu": gpu}
        self._frames.append(frame)
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            try:
                if gpu:
                    end_event.record(stream)
                    end_event.synchronize()
                    seconds = start_event.elapsed_time(end_event) / 1000.0
                    self.gpu_seconds[component] = self.gpu_seconds.get(component, 0.0) + seconds
                    self.peak_gpu_memory_bytes = max(
                        self.peak_gpu_memory_bytes, torch.cuda.max_memory_allocated(device),
                    )
            finally:
                elapsed = time.perf_counter() - frame["start"]
                exclusive = max(0.0, elapsed - frame["children"])
                self._frames.pop()
                self.compute_seconds[component] = self.compute_seconds.get(component, 0.0) + exclusive
                if self._frames:
                    self._frames[-1]["children"] += elapsed
            self.event("compute", component=component, seconds=exclusive, succeeded=succeeded)

    def adjusted_nfe(self, *, reference_forward_flops: float) -> float | None:
        require_nonnegative(reference_forward_flops, "reference forward FLOPs")
        if reference_forward_flops == 0:
            raise ValueError("Reference forward FLOPs must be positive")
        for kind in ("mdlm", "ar", "verifier"):
            if self.counters[f"flop_profiled_{kind}_calls"] != self.counters[f"{kind}_forward_calls"]:
                return None
        return sum(self.flops.values()) / reference_forward_flops

    def snapshot(self) -> TrackingSnapshot:
        if self._frames:
            raise RuntimeError("Checkpoint tracking between measured operations")
        return TrackingSnapshot(
            counters=dict(self.counters), compute_seconds=dict(self.compute_seconds),
            gpu_seconds=dict(self.gpu_seconds), flops=dict(self.flops),
            end_to_end_seconds=self.end_to_end_seconds,
            peak_gpu_memory_bytes=self.peak_gpu_memory_bytes,
        )

    def restore(self, snapshot: TrackingSnapshot) -> None:
        if self._frames:
            raise RuntimeError("Cannot restore tracking while a timer is active")
        if set(snapshot.counters) != set(COUNTERS):
            raise ValueError("Tracking snapshot has incompatible counters")
        self.counters = dict(snapshot.counters)
        self.compute_seconds = dict(snapshot.compute_seconds)
        self.gpu_seconds = dict(snapshot.gpu_seconds)
        self.flops = dict(snapshot.flops)
        self.peak_gpu_memory_bytes = snapshot.peak_gpu_memory_bytes
        self._elapsed_before = snapshot.end_to_end_seconds
        self._started = time.perf_counter()

    def summary(self) -> dict[str, Any]:
        result = self.snapshot().to_dict()
        result["raw_nfe"] = self.raw_nfe
        result["model_example_nfe"] = self.model_example_nfe
        result["tracked_compute_seconds"] = self.tracked_compute_seconds
        return result
