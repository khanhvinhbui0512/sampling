"""Shared task lifecycle; used by local entrypoints and their Modal wrappers.

This file is a library, not a standalone workflow script. Callers pass their task
function and dependency declarations; no registry or future task is imported.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from tqdm.auto import tqdm

from src.artifacts import ArtifactStore, Dependency, utc_now
from src.config import Config, ConfigValidator, config_digest, load_config
from src.schemas import (
    GROUPS,
    AttemptRecord,
    Method,
    RunIdentity,
    RunRecord,
    SampleRecord,
    TaskResult,
    TrackingSnapshot,
    require_count,
    require_identifier,
)
from src.tracking import BudgetExceeded, CostTracker

TaskFunction = Callable[["TaskContext"], TaskResult]
DependencyResolver = Callable[[Config, ArtifactStore], Mapping[str, Dependency]]


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed global RNGs; private generators must be checkpointed by their owner."""
    import numpy as np
    import torch

    require_count(seed, "seed")
    if seed >= 2**32:
        raise ValueError("Seed must be smaller than 2**32")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False


def build_parser(
    description: str, *, default_config: str = "configs/common.yaml",
    supports_group: bool = False, supports_seed: bool = True,
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config, help="Project-relative task YAML.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-id", help="Run directory identifier; required with --resume.")
    parser.add_argument("--resume", action="store_true", help="Restore the committed snapshot of this run.")
    parser.add_argument("--smoke", action="store_true", help="Use the separate smoke suite.")
    parser.add_argument("--artifact-root", help="Project-relative artifact root.")
    if supports_group:
        parser.add_argument("--group", choices=GROUPS)
    if supports_seed:
        parser.add_argument("--seed", type=int)
    if add_arguments is not None:
        add_arguments(parser)
    return parser


@dataclass(slots=True)
class TaskContext:
    config: Config
    arguments: argparse.Namespace
    identity: RunIdentity
    store: ArtifactStore
    tracker: CostTracker
    record: RunRecord
    resumed: bool = False
    resume_state: Any = None
    training_state: Any = None
    _last_attempt_id: int = 0
    _sample_count: int = 0
    _finished: bool = False

    @property
    def run_directory(self) -> str:
        return self.store.run_directory(self.identity)

    def output_path(self, filename: str) -> str:
        """Return a logical path relative to the artifact root."""
        require_identifier(filename, "output filename")
        return os.path.join(self.run_directory, filename)

    def progress(self, iterable: Any = None, **kwargs: Any) -> Any:
        kwargs.setdefault("disable", not self.config["runtime"]["progress"])
        return tqdm(iterable, **kwargs)

    def log(self, text: str, *, level: str = "INFO") -> None:
        levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        if level not in levels:
            raise ValueError(f"Unknown log level: {level}")
        if levels[level] >= levels[self.config["runtime"]["log_level"]]:
            tqdm.write(f"{self.identity.task_id}: {text}")
        self.tracker.event("log", level=level, message=text)

    def write_attempt(self, attempt: AttemptRecord) -> None:
        if attempt.run_id != self.identity.run_id:
            raise ValueError("Attempt belongs to another run")
        if self.identity.group is not None and attempt.group != self.identity.group:
            raise ValueError("Attempt belongs to another group")
        if self.identity.method is not None and attempt.method != self.identity.method:
            raise ValueError("Attempt belongs to another method")
        if attempt.attempt_id != self._last_attempt_id + 1:
            raise ValueError("Root attempts must be written once, in their original order")
        if attempt.attempt_id > self.tracker.counters["root_attempts"]:
            raise ValueError("Register the root attempt before recording its result")
        if attempt.reward_result is not None and attempt.reward_result.reward_version != self.identity.reward_version:
            raise ValueError("Attempt uses a different reward version")
        self.store.append_jsonl(self.output_path("attempts.jsonl"), attempt.to_dict())
        self._last_attempt_id = attempt.attempt_id

    def write_sample(self, sample: SampleRecord) -> None:
        """Append a selector-approved sample and count it once; do not count it again."""
        if sample.run_id != self.identity.run_id or sample.sample_id != self._sample_count + 1:
            raise ValueError("Selected samples must belong to this run and have consecutive IDs")
        if self.identity.group is not None and sample.group != self.identity.group:
            raise ValueError("Sample belongs to another group")
        if sample.attempt_id > self._last_attempt_id:
            raise ValueError("Record the source attempt before selecting its sample")
        if sample.reward_result.reward_version != self.identity.reward_version:
            raise ValueError("Sample uses a different reward version")
        if self.identity.suite != "replication" and sample.identity_type != "canonical_smiles":
            raise ValueError("Primary/smoke samples must have a valid molecular identity")
        self.store.append_jsonl(self.output_path("samples.jsonl"), sample.to_dict())
        self._sample_count = sample.sample_id
        self.tracker.increment("selected_samples")

    def save_checkpoint(self, sampler_state: Any, *, training_state: Any = None) -> str:
        if self._finished:
            raise ValueError("Cannot checkpoint a finished task")
        return self.store.save_resume(
            self.identity, tracking=self.tracker.snapshot(),
            sampler_state={
                "task_state": sampler_state,
                "recording": {"last_attempt_id": self._last_attempt_id, "sample_count": self._sample_count},
            },
            training_state=training_state,
        )

    def finish(self, result: TaskResult, *, error: str | None = None) -> RunRecord:
        if self._finished:
            raise ValueError("Task has already finished")
        for artifact in result.artifacts:
            self.store.require_file(artifact)
        self.tracker.event("run_finished", status=result.status, summary=result.summary)
        self.record = replace(
            self.record, status=result.status, updated_at=utc_now(),
            tracking=self.tracker.snapshot(), artifacts=result.artifacts,
            summary=result.summary, error=error,
        )
        self.store.write_run(self.record)
        self._finished = True
        return self.record


def _require_repo_root() -> None:
    if not os.path.isfile("pyproject.toml") or not os.path.isfile("src/__init__.py"):
        raise RuntimeError("Run this task from the repository root after installing the project")


def create_context(
    task_id: str, arguments: argparse.Namespace, *, method: Method | Callable[[Config], Method] | None = None,
    execution: str = "local", validator: ConfigValidator | None = None,
    dependencies: DependencyResolver | None = None,
    argument_overrides: Callable[[argparse.Namespace], Sequence[str]] | None = None,
) -> TaskContext:
    started = time.perf_counter()
    _require_repo_root()
    require_identifier(task_id, "task ID")
    if arguments.resume and not arguments.run_id:
        raise ValueError("--resume requires the original --run-id")
    overrides = list(arguments.overrides)
    if argument_overrides is not None:
        overrides.extend(argument_overrides(arguments))
    for name in ("seed", "group"):
        value = getattr(arguments, name, None)
        if value is not None:
            overrides.append(f"{name}={value}")
    if arguments.smoke:
        overrides.append("suite=smoke")
    if arguments.artifact_root is not None:
        # YAML quoting keeps whitespace, colons, and other characters literal.
        overrides.append("paths.artifact_root=" + json.dumps(arguments.artifact_root))
    overrides.append(f"runtime.execution={execution}")
    config = load_config(arguments.config, overrides=overrides, validator=validator)
    paths = config["paths"]
    store = ArtifactStore(paths["artifact_root"], directories={
        key: paths[key] for key in ("data", "checkpoints", "outputs", "reports")
    })
    declared = dependencies(config, store) if dependencies is not None else {}
    hashes = store.check_dependencies(declared)
    run_id = arguments.run_id or f"{task_id}-{uuid.uuid4().hex[:12]}"
    identity = RunIdentity(
        task_id=task_id, run_id=run_id, config_hash=config_digest(config),
        suite=config["suite"], seed=config["seed"], reward_version=config["reward"]["version"],
        method=method(config) if callable(method) else method, group=config.get("group"),
        dependency_hashes=hashes,
    )
    preflight_seconds = time.perf_counter() - started
    tracker = CostTracker(
        run_id=run_id, max_model_nfe=config["protocol"]["max_model_nfe"],
        max_compute_seconds=config["protocol"]["max_compute_seconds"],
    )
    tracker.add_external_wall_time(preflight_seconds)
    seed_everything(config["seed"], deterministic=config["runtime"]["deterministic"])
    if arguments.resume:
        record = store.read_run(identity)
        payload = store.load_resume(identity)
        expected_logs = {
            os.path.join(store.run_directory(identity), name)
            for name in ("attempts.jsonl", "samples.jsonl", "events.jsonl")
        }
        if set(payload["log_offsets"]) != expected_logs:
            raise ValueError("Resume checkpoint lacks the complete set of run journal offsets")
        wrapped = payload["sampler_state"]
        if not isinstance(wrapped, dict) or set(wrapped) != {"task_state", "recording"}:
            raise ValueError("Resume snapshot lacks task/recording state")
        recording = wrapped["recording"]
        if not isinstance(recording, dict) or set(recording) != {"last_attempt_id", "sample_count"}:
            raise ValueError("Resume recording state is incomplete")
        require_count(recording["last_attempt_id"], "last attempt ID")
        require_count(recording["sample_count"], "sample count")
        snapshot = TrackingSnapshot.from_dict(payload["tracking"])
        if recording["sample_count"] != snapshot.counters.get("selected_samples"):
            raise ValueError("Resume sample count disagrees with tracking")
        if recording["last_attempt_id"] > snapshot.counters.get("root_attempts", 0):
            raise ValueError("Resume attempt count disagrees with tracking")
        resume_preflight_seconds = time.perf_counter() - started
        tracker.restore(snapshot)
        tracker.add_external_wall_time(resume_preflight_seconds)
        store.restore_resume(identity, payload)
        context = TaskContext(
            config, arguments, identity, store, tracker, record, resumed=True,
            resume_state=wrapped["task_state"], training_state=payload["training_state"],
            _last_attempt_id=recording["last_attempt_id"], _sample_count=recording["sample_count"],
        )
        context.record = replace(record, status="running", updated_at=utc_now(), error=None)
        store.write_run(context.record)
    else:
        record = store.create_run(identity, config)
        context = TaskContext(config, arguments, identity, store, tracker, record)
    tracker.event_sink = lambda row: store.append_jsonl(context.output_path("events.jsonl"), row)
    tracker.event("run_resumed" if arguments.resume else "run_started", task_id=task_id, execution=execution)
    if not arguments.resume:
        context.save_checkpoint(None)
    return context


def run_task(
    task_id: str, task: TaskFunction, *, description: str,
    argv: Sequence[str] | None = None, default_config: str = "configs/common.yaml",
    method: Method | Callable[[Config], Method] | None = None, execution: str = "local",
    supports_group: bool = False, supports_seed: bool = True,
    validator: ConfigValidator | None = None, dependencies: DependencyResolver | None = None,
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
    argument_overrides: Callable[[argparse.Namespace], Sequence[str]] | None = None,
) -> RunRecord:
    """Execute one supplied function and print a brief result through tqdm."""
    parser = build_parser(
        description, default_config=default_config, supports_group=supports_group,
        supports_seed=supports_seed, add_arguments=add_arguments,
    )
    arguments = parser.parse_args(argv)
    try:
        context = create_context(
            task_id, arguments, method=method, execution=execution, validator=validator,
            dependencies=dependencies, argument_overrides=argument_overrides,
        )
    except Exception as exc:
        tqdm.write(f"{task_id}: failed; {exc}")
        raise
    with context.progress(total=1, desc=task_id, unit="task") as progress:
        try:
            result = task(context)
            if not isinstance(result, TaskResult):
                raise TypeError("Task functions must return TaskResult")
            record = context.finish(result)
        except (BudgetExceeded, TimeoutError) as exc:
            status = "budget_exhausted" if isinstance(exc, BudgetExceeded) else "timeout"
            record = context.finish(TaskResult(str(exc), status=status), error=str(exc))
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            context.finish(TaskResult(str(exc) or type(exc).__name__, status=status), error=str(exc))
            tqdm.write(f"{task_id}: {status}; run={context.run_directory}")
            raise
        finally:
            progress.update(1)
    tqdm.write(f"{task_id}: {record.status}; {record.summary}; run={context.run_directory}")
    return record
