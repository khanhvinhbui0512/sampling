"""Artifact integrity, atomic writes, and resumable run state.

All public paths are relative to the repository or an ArtifactStore root. Each
worker owns one run directory. A resume pointer is committed only after the
checkpoint is durable; log entries after its byte offsets are uncommitted.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, BinaryIO

from src.config import dump_config, project_path
from src.schemas import (
    SCHEMA_VERSION,
    ArtifactManifest,
    RunIdentity,
    RunRecord,
    TrackingSnapshot,
    require_count,
    require_identifier,
    require_sha256,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str) -> str:
    path = project_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Artifact is missing or is not a file: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


@contextmanager
def _atomic_file(path: str) -> Iterator[BinaryIO]:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".write-", suffix=".tmp", dir=parent)
    temporary = os.path.relpath(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


@dataclass(frozen=True, slots=True)
class Dependency:
    path: str
    checksum: str | None = None
    manifest: bool = False
    kind: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        project_path(self.path)
        if self.checksum is not None:
            require_sha256(self.checksum)
        if self.kind is not None:
            require_identifier(self.kind, "dependency kind")
        if (self.kind is not None or self.metadata) and not self.manifest:
            raise ValueError("Kind/metadata checks require a dependency manifest")


def capture_rng_state() -> dict[str, Any]:
    """Private sampler generators belong in sampler_state, alongside trie/cache state."""
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    import numpy as np
    import torch

    if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Incomplete RNG snapshot")
    cuda_states = state["torch_cuda"]
    if cuda_states and (
        not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count()
    ):
        raise ValueError("CUDA RNG snapshot requires the same number of visible GPUs")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)


class ArtifactStore:
    """Map logical artifact paths to one relative root for local or Modal workers."""

    def __init__(self, root: str = ".", *, directories: Mapping[str, str] | None = None) -> None:
        self.root = project_path(root, allow_dot=True)
        self.directories = {"data": "data", "checkpoints": "checkpoints", "outputs": "outputs", "reports": "reports"}
        if directories is not None:
            if set(directories) != set(self.directories):
                raise ValueError("Artifact directories must define data/checkpoints/outputs/reports")
            self.directories.update(directories)
        for value in self.directories.values():
            project_path(value)

    def path(self, relative: str) -> str:
        relative = project_path(relative)
        result = project_path(os.path.join(self.root, relative))
        if os.path.commonpath((os.path.realpath(self.root), os.path.realpath(result))) != os.path.realpath(self.root):
            raise ValueError(f"Artifact escapes its root: {relative}")
        return result

    def directory(self, kind: str, *components: str) -> str:
        if kind not in self.directories:
            raise ValueError(f"Unknown artifact directory: {kind}")
        for component in components:
            require_identifier(component, "directory component")
        return os.path.join(self.directories[kind], *components)

    def require_file(self, relative: str, *, checksum: str | None = None) -> str:
        path = self.path(relative)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required artifact is missing: {path}")
        if checksum is not None:
            require_sha256(checksum)
            if sha256_file(path) != checksum:
                raise ValueError(f"Artifact checksum mismatch: {path}")
        return path

    def write_text(self, relative: str, text: str) -> str:
        path = self.path(relative)
        with _atomic_file(path) as handle:
            handle.write(text.encode("utf-8"))
        return path

    def write_json(self, relative: str, value: Any) -> str:
        encoded = _json_bytes(value)
        path = self.path(relative)
        with _atomic_file(path) as handle:
            handle.write(encoded)
        return path

    def read_json(self, relative: str) -> dict[str, Any]:
        path = self.require_file(relative)
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
        if not isinstance(result, dict):
            raise TypeError(f"Expected a JSON object in {path}")
        return result

    def write_jsonl(self, relative: str, rows: Iterable[Mapping[str, Any]]) -> str:
        path = self.path(relative)
        with _atomic_file(path) as handle:
            for row in rows:
                handle.write(_json_bytes(dict(row)))
        return path

    def append_jsonl(self, relative: str, row: Mapping[str, Any]) -> str:
        encoded = _json_bytes(dict(row))
        path = self.path(relative)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell():
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    raise ValueError(f"Uncommitted JSONL tail in {path}; resume from a checkpoint first")
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return path

    def read_jsonl(self, relative: str) -> Iterator[dict[str, Any]]:
        path = self.require_file(relative)
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL in {path}, line {number}") from exc
                if not isinstance(value, dict):
                    raise TypeError(f"Expected an object in {path}, line {number}")
                yield value

    def write_config(self, relative: str, config: Mapping[str, Any]) -> str:
        return self.write_text(relative, dump_config(config))

    def create_manifest(
        self, relative: str, *, kind: str, config_hash: str,
        files: Sequence[str], metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactManifest:
        names = [project_path(name) for name in files]
        if len(set(names)) != len(names) or project_path(relative) in names:
            raise ValueError("Manifest files must be distinct and exclude the manifest itself")
        manifest = ArtifactManifest(
            kind=kind, config_hash=config_hash,
            files={name: sha256_file(self.require_file(name)) for name in names},
            metadata=dict(metadata or {}),
        )
        self.write_json(relative, manifest.to_dict())
        return manifest

    def require_manifest(
        self, relative: str, *, kind: str | None = None,
        config_hash: str | None = None, metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactManifest:
        manifest = ArtifactManifest.from_dict(self.read_json(relative))
        if kind is not None and manifest.kind != kind:
            raise ValueError(f"Wrong artifact kind in {relative}: {manifest.kind}")
        if config_hash is not None and manifest.config_hash != config_hash:
            raise ValueError(f"Configuration mismatch in {relative}")
        for key, expected in (metadata or {}).items():
            if key not in manifest.metadata or manifest.metadata[key] != expected:
                raise ValueError(f"Dependency metadata mismatch in {relative}: {key}")
        for name, digest in manifest.files.items():
            self.require_file(name, checksum=digest)
        return manifest

    def check_dependencies(self, dependencies: Mapping[str, Dependency]) -> dict[str, str]:
        fingerprints = {}
        for name, dependency in dependencies.items():
            require_identifier(name, "dependency name")
            if not isinstance(dependency, Dependency):
                raise TypeError(f"Dependency {name} must be a Dependency record")
            path = self.require_file(dependency.path, checksum=dependency.checksum)
            if dependency.manifest:
                self.require_manifest(dependency.path, kind=dependency.kind, metadata=dependency.metadata)
            fingerprints[name] = sha256_file(path)
        return fingerprints

    def run_directory(self, identity: RunIdentity) -> str:
        return self.directory(
            "outputs", identity.suite, identity.method or identity.task_id,
            identity.group or "all", f"seed_{identity.seed}", identity.run_id,
        )

    def create_run(self, identity: RunIdentity, config: Mapping[str, Any]) -> RunRecord:
        relative = self.run_directory(identity)
        path = self.path(relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Exclusive creation prevents overwriting another run, including an empty one.
        os.mkdir(path)
        now = utc_now()
        record = RunRecord(identity, "running", now, now)
        self.write_config(os.path.join(relative, "resolved_config.yaml"), config)
        self.write_run(record)
        return record

    def read_run(self, identity: RunIdentity) -> RunRecord:
        relative = os.path.join(self.run_directory(identity), "manifest.json")
        record = RunRecord.from_dict(self.read_json(relative))
        if record.identity != identity:
            raise ValueError("Resume identity mismatch: config/dependencies/reward/group/seed changed")
        return record

    def write_run(self, record: RunRecord) -> str:
        relative = os.path.join(self.run_directory(record.identity), "manifest.json")
        return self.write_json(relative, record.to_dict())

    def save_resume(
        self, identity: RunIdentity, *, tracking: TrackingSnapshot,
        sampler_state: Any, training_state: Any = None,
        log_names: Sequence[str] = ("attempts.jsonl", "samples.jsonl", "events.jsonl"),
    ) -> str:
        """Save between operations; callers supply every private RNG/cache/sampler state."""
        import torch

        record = self.read_run(identity)
        if record.status == "completed":
            raise ValueError("Cannot checkpoint an already completed run")
        run = self.run_directory(identity)
        offsets = {}
        for name in log_names:
            require_identifier(name, "log filename")
            relative = os.path.join(run, name)
            path = self.path(relative)
            offsets[relative] = os.path.getsize(path) if os.path.isfile(path) else 0
        checkpoint = os.path.join(run, "resume", f"state-{uuid.uuid4().hex}.pt")
        payload = {
            "schema_version": SCHEMA_VERSION, "identity": identity.to_dict(),
            "tracking": tracking.to_dict(), "rng": capture_rng_state(),
            "sampler_state": sampler_state, "training_state": training_state,
            "log_offsets": offsets,
        }
        with _atomic_file(self.path(checkpoint)) as handle:
            torch.save(payload, handle)
        self.write_json(os.path.join(run, "resume.json"), {
            "schema_version": SCHEMA_VERSION, "checkpoint": checkpoint,
            "sha256": sha256_file(self.path(checkpoint)), "identity": identity.to_dict(),
        })
        return checkpoint

    def load_resume(self, identity: RunIdentity) -> dict[str, Any]:
        """Load project-created training snapshots, checking their committed checksum."""
        import torch

        record = self.read_run(identity)
        if record.status == "completed":
            raise ValueError("Run is completed; create a new run instead of resuming it")
        run = self.run_directory(identity)
        pointer = self.read_json(os.path.join(run, "resume.json"))
        if pointer.get("schema_version") != SCHEMA_VERSION or pointer.get("identity") != identity.to_dict():
            raise ValueError("Resume pointer schema/identity mismatch")
        checkpoint = project_path(pointer["checkpoint"])
        if not checkpoint.startswith(run + os.sep + "resume" + os.sep):
            raise ValueError("Resume checkpoint must belong to this run")
        path = self.require_file(checkpoint, checksum=pointer["sha256"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {"schema_version", "identity", "tracking", "rng", "sampler_state", "training_state", "log_offsets"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError("Incomplete resume checkpoint")
        if payload["schema_version"] != SCHEMA_VERSION or payload["identity"] != identity.to_dict():
            raise ValueError("Checkpoint identity mismatch")
        TrackingSnapshot.from_dict(payload["tracking"])
        self._check_log_offsets(run, payload["log_offsets"])
        return payload

    def _check_log_offsets(self, run: str, offsets: Mapping[str, int]) -> None:
        if not isinstance(offsets, Mapping):
            raise TypeError("Resume log offsets must be a mapping")
        for relative, offset in offsets.items():
            require_count(offset, "log offset")
            if project_path(relative) != relative or os.path.dirname(relative) != run or not relative.endswith(".jsonl"):
                raise ValueError("Resume logs must be JSONL files directly inside this run")
            path = self.path(relative)
            if not os.path.exists(path):
                if offset:
                    raise FileNotFoundError(f"Committed log is missing: {path}")
                continue
            if not os.path.isfile(path) or os.path.getsize(path) < offset:
                raise ValueError(f"Log is shorter than its committed offset: {path}")
            if offset:
                with open(path, "rb") as handle:
                    handle.seek(offset - 1)
                    if handle.read(1) != b"\n":
                        raise ValueError(f"Committed offset is not a JSONL boundary: {path}")

    def restore_resume(self, identity: RunIdentity, payload: Mapping[str, Any]) -> None:
        """Restore RNGs and discard only journal entries after the saved commit."""
        if payload.get("identity") != identity.to_dict():
            raise ValueError("Cannot restore a snapshot from another run")
        run = self.run_directory(identity)
        self._check_log_offsets(run, payload["log_offsets"])
        restore_rng_state(payload["rng"])
        for relative, offset in payload["log_offsets"].items():
            path = self.path(relative)
            if os.path.exists(path) and os.path.getsize(path) > offset:
                with open(path, "r+b") as handle:
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
