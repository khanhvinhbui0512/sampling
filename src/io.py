"""Atomic artifact writes, immutable attempts, hashes and completion manifests.

Only a completed manifest establishes a valid run. A crash can leave an
incomplete attempt or a lock file; retry with a new attempt ID. No helper
overwrites a data artifact. Paths are relative to the repo root.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping

import yaml
from tqdm.auto import tqdm

from src.config import project_path


def report_result(message: str) -> None:
    """Task entrypoints use this for a short result without disrupting tqdm bars."""
    tqdm.write(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Artifact JSON/YAML object keys must be strings")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Convert {type(value).__name__} to JSON-compatible values before saving")


def _json(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, allow_nan=False,
                      indent=2 if pretty else None, separators=None if pretty else (",", ":"))


def payload_hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(project_path(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _atomic_writer(path: str, *, binary: bool = False,
                   update_manifest: bool = False) -> Iterator[Any]:
    path = project_path(path)
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    # Cooperating workers cannot overwrite each other between existence check and rename.
    lock = path + ".lock"
    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    temporary: str | None = None
    try:
        os.close(lock_fd)
        if os.path.exists(path):
            if not update_manifest:
                raise FileExistsError(f"Artifact already exists: {path}")
            old = read_json(path)
            if old.get("status") != "running":
                raise FileExistsError(f"A finished manifest cannot be changed: {path}")
        fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=parent)
        # tempfile may return an absolute name; it remains internal and is never persisted.
        mode = "wb" if binary else "w"
        options = {} if binary else {"encoding": "utf-8", "newline": "\n"}
        with os.fdopen(fd, mode, **options) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)
        os.unlink(lock)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def read_json(path: str) -> Any:
    with open(project_path(path), encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_json_constant)


def write_json(path: str, value: Any) -> None:
    encoded = _json(value, pretty=True) + "\n"
    with _atomic_writer(path) as handle:
        handle.write(encoded)


def read_jsonl(path: str, *, progress: bool = True) -> Iterator[dict[str, Any]]:
    with open(project_path(path), encoding="utf-8") as handle:
        for line_number, line in enumerate(tqdm(handle, desc=f"Read {path}", unit="row",
                                                disable=not progress), start=1):
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
                if not isinstance(row, dict):
                    raise ValueError("Expected a JSON object")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            yield row


def write_jsonl(path: str, rows: Iterable[Mapping[str, Any]], *, progress: bool = True) -> int:
    count = 0
    with _atomic_writer(path) as handle:
        for row in tqdm(rows, desc=f"Write {path}", unit="row", disable=not progress):
            if not isinstance(row, Mapping):
                raise TypeError("Each JSONL row must be a mapping")
            handle.write(_json(row) + "\n")
            count += 1
    return count


def read_yaml(path: str) -> Any:
    """Read an artifact; experiment configs should use src.config.load_config."""
    with open(project_path(path), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: str, value: Any) -> None:
    plain = _plain(value)
    _json(plain)  # Validate finite, serializable values before starting an artifact write.
    with _atomic_writer(path) as handle:
        yaml.safe_dump(plain, handle, sort_keys=True, allow_unicode=True)


def write_checkpoint(path: str, payload: Mapping[str, Any], *,
                     metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Save tensor/plain-Python state plus a hash sidecar; no arbitrary object pickles.

    Callers must register both files in the attempt manifest. A sidecar/write
    failure leaves an unfinished attempt, never a completed checkpoint artifact.
    """
    import torch

    def validate(value: Any) -> None:
        if isinstance(value, torch.Tensor) or value is None or type(value) in (str, int, float, bool):
            return
        if type(value) in (dict, OrderedDict):
            for key, item in value.items():
                if type(key) not in (str, int):
                    raise TypeError("Checkpoint mapping keys must be strings or integers")
                validate(item)
            if isinstance(value, OrderedDict) and hasattr(value, "_metadata"):
                validate(value._metadata)
            return
        if type(value) in (list, tuple):
            for item in value:
                validate(item)
            return
        raise TypeError(f"Unsupported checkpoint object: {type(value).__name__}")

    path = project_path(path)
    if type(payload) not in (dict, OrderedDict):
        payload = dict(payload)
    validate(payload)
    _json(metadata)
    with _atomic_writer(path, binary=True) as handle:
        torch.save(payload, handle)
    sidecar = {"schema_version": 1, "path": path, "sha256": file_hash(path),
               "size_bytes": os.path.getsize(path), "metadata": _plain(metadata)}
    write_json(path + ".metadata.json", sidecar)
    return sidecar


def read_checkpoint(path: str, *, map_location: str = "cpu") -> dict[str, Any]:
    import torch

    path = project_path(path)
    sidecar = read_json(path + ".metadata.json")
    if sidecar.get("schema_version") != 1 or sidecar.get("path") != path:
        raise ValueError("Checkpoint metadata identity mismatch")
    if sidecar["sha256"] != file_hash(path):
        raise ValueError("Checkpoint hash mismatch")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a state dictionary")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Artifact IDs may contain only letters, digits, underscore, dot and hyphen")
    return value


def _attempt_path(directory: str, name: str) -> str:
    if not isinstance(name, str) or not name or os.path.isabs(name):
        raise ValueError("Artifact names must be relative to the attempt directory")
    normalized = os.path.normpath(name)
    if normalized == "." or normalized == ".." or normalized.startswith(".." + os.sep):
        raise ValueError("Artifact name escapes the attempt directory")
    path = project_path(os.path.join(directory, normalized))
    root = os.path.realpath(directory)
    if os.path.commonpath((root, os.path.realpath(path))) != root:
        raise ValueError("Artifact symlink escapes the attempt directory")
    return path


@dataclass
class RunArtifacts:
    """A single writer owns one attempt; workers must use distinct attempt IDs."""

    directory: str
    manifest: dict[str, Any]
    _artifacts: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    @classmethod
    def create(cls, base_dir: str, *, run_id: str, attempt_id: str,
               config: Any, target_id: str, reference_id: str) -> RunArtifacts:
        for value in (target_id, reference_id):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Run target/reference IDs must be explicit")
        directory = project_path(os.path.join(base_dir, _artifact_id(run_id), "attempts",
                                              _artifact_id(attempt_id)))
        config_digest = payload_hash(config)
        os.makedirs(directory, exist_ok=False)
        manifest = {
            "schema_version": 1, "run_id": run_id, "attempt_id": attempt_id,
            "status": "running", "created_at": _utc_now(), "config_sha256": config_digest,
            "target_id": target_id, "reference_id": reference_id, "artifacts": {},
        }
        attempt = cls(directory, manifest)
        write_json(attempt.path("manifest.json"), manifest)
        return attempt

    def path(self, name: str) -> str:
        return _attempt_path(self.directory, name)

    def register(self, name: str) -> dict[str, Any]:
        if self.manifest["status"] != "running":
            raise ValueError("Cannot modify a finished attempt")
        name = os.path.normpath(name)
        if name == "manifest.json" or name in self._artifacts:
            raise ValueError("Artifact is reserved or already registered")
        path = self.path(name)
        record = {"sha256": file_hash(path), "size_bytes": os.path.getsize(path)}
        artifacts = {**self._artifacts, name: record}
        manifest = {**self.manifest, "artifacts": artifacts}
        self._save_manifest(manifest)
        self._artifacts = artifacts
        self.manifest = manifest
        return dict(record)

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        with _atomic_writer(self.path("manifest.json"), update_manifest=True) as handle:
            handle.write(_json(manifest, pretty=True) + "\n")

    def complete(self, *, required_artifacts: Iterable[str]) -> dict[str, Any]:
        if self.manifest["status"] != "running":
            raise ValueError("Attempt has already finished")
        required = {os.path.relpath(self.path(name), self.directory) for name in required_artifacts}
        if not required or not required.issubset(self._artifacts):
            raise ValueError("All required artifacts must be written and registered first")
        for name, record in self._artifacts.items():
            if file_hash(self.path(name)) != record["sha256"]:
                raise ValueError(f"Artifact changed after registration: {name}")
        completed = {**self.manifest, "status": "completed", "finished_at": _utc_now()}
        self._save_manifest(completed)
        self.manifest = completed
        return dict(completed)

    def fail(self, error: str) -> None:
        if self.manifest["status"] != "running" or not isinstance(error, str) or not error.strip():
            raise ValueError("Only running attempts can be failed, with an error description")
        failed = {**self.manifest, "status": "failed", "finished_at": _utc_now(), "error": error}
        self._save_manifest(failed)
        self.manifest = failed


def read_completed_manifest(path: str) -> dict[str, Any]:
    """Consumers verify the marker and every recorded artifact before using a run."""
    path = project_path(path)
    manifest = read_json(path)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("status") != "completed"):
        raise ValueError(f"Artifact attempt is not complete: {path}")
    directory = os.path.dirname(path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Completed manifest has no artifact records")
    for name, record in artifacts.items():
        artifact = _attempt_path(directory, name)
        if not isinstance(record, dict) or not {"sha256", "size_bytes"}.issubset(record):
            raise ValueError(f"Missing artifact hash/size in manifest: {name}")
        if file_hash(artifact) != record["sha256"] or os.path.getsize(artifact) != record["size_bytes"]:
            raise ValueError(f"Artifact differs from its completed manifest: {artifact}")
    return manifest
