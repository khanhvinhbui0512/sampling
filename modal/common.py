"""Modal infrastructure shared by wrappers; load this file by its relative path.

Do not import ``modal.common``: ``modal`` is the installed SDK. Container image
destinations and Volume mountpoints are the only absolute paths used here, as
required by Modal. Task arguments and artifact paths remain project-relative.

API references: https://modal.com/docs/sdk/py/latest/Image and
https://modal.com/docs/guide/volumes . No image is built at helper import time.
"""

from __future__ import annotations

import json
import math
import os
import tomllib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import modal
from modal import Image, Volume

from src.artifacts import sha256_file
from src.config import project_path, read_yaml
from src.schemas import require_count, require_identifier
from src.tasks.common import build_parser

PACKAGE_DIRECTORIES = ("src", "configs", "modal", "tests", "assets")
PACKAGE_FILES = ("pyproject.toml", "requirements.lock")
IGNORED_DIRECTORIES = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache", ".git"})
_VOLUMES: dict[str, modal.Volume] = {}
_IMAGES: dict[tuple[str, str, str], modal.Image] = {}


def load_settings(path: str = "configs/modal.yaml") -> dict[str, Any]:
    if not os.path.isfile("pyproject.toml") or not os.path.isfile("src/__init__.py"):
        raise RuntimeError("Run Modal commands from the repository root")
    settings = read_yaml(path)
    if settings.get("schema_version") != 1 or settings.get("python_version") != "3.12":
        raise ValueError("Modal requires schema_version=1 and Python 3.12")
    require_identifier(settings.get("app_name"), "Modal app name")
    for name in ("repo_directory", "artifact_root", "dependency_file"):
        settings[name] = project_path(settings.get(name))
    if settings["dependency_file"] != "pyproject.toml":
        raise ValueError("Modal dependencies must come from pyproject.toml")
    if settings["artifact_root"].split(os.sep)[0] in (*PACKAGE_DIRECTORIES, *PACKAGE_FILES):
        raise ValueError("The Volume must not hide packaged source, tests, or configuration")
    for name in ("images", "volumes", "secrets", "profiles", "tasks"):
        if not isinstance(settings.get(name), dict) or (name != "secrets" and not settings[name]):
            raise TypeError(f"Modal {name} must be a mapping")
    for name, image in settings["images"].items():
        require_identifier(name, "image profile")
        packages = image.get("apt_packages") if isinstance(image, dict) else None
        if not isinstance(packages, list) or any(not isinstance(p, str) or not p for p in packages):
            raise TypeError(f"Image {name} requires an apt_packages list")
        if not isinstance(image.get("base_image"), str) or not image["base_image"]:
            raise ValueError(f"Image {name} requires an explicit registry tag")
    for name, descriptor in settings["volumes"].items():
        require_identifier(name, "Volume key")
        if not isinstance(descriptor, dict):
            raise TypeError(f"Volume {name} must be a mapping")
        require_identifier(descriptor.get("name"), "Volume name")
        if not isinstance(descriptor.get("create_if_missing"), bool):
            raise TypeError("Volume create_if_missing must be boolean")
    # One shared artifact tree prevents workers from writing to unmounted paths.
    if set(settings["volumes"]) != {"artifacts"}:
        raise ValueError("This project uses one shared artifacts Volume")
    for name, descriptor in settings["secrets"].items():
        require_identifier(name, "Secret key")
        if not isinstance(descriptor, dict):
            raise TypeError(f"Secret {name} must be a mapping")
        require_identifier(descriptor.get("name"), "Secret name")
        keys = descriptor.get("required_keys")
        if not isinstance(keys, list) or any(not isinstance(key, str) or not key for key in keys):
            raise TypeError("Secret required_keys must be a list of names")
    for name, profile in settings["profiles"].items():
        require_identifier(name, "resource profile")
        if not isinstance(profile, dict):
            raise TypeError(f"Profile {name} must be a mapping")
        cpu = profile.get("cpu")
        if isinstance(cpu, bool) or not isinstance(cpu, (int, float)) or not math.isfinite(cpu) or cpu <= 0:
            raise ValueError(f"Profile {name} needs a positive finite CPU allocation")
        require_count(profile.get("memory_mib"), "memory_mib", positive=True)
        require_count(profile.get("timeout_seconds"), "timeout_seconds", positive=True)
        if profile["timeout_seconds"] > 86400:
            raise ValueError("Modal function timeout cannot exceed 24 hours")
        if profile.get("gpu") is not None and (not isinstance(profile["gpu"], str) or not profile["gpu"]):
            raise TypeError("gpu must be null or a Modal GPU selector")
        if profile.get("image") not in settings["images"]:
            raise ValueError(f"Unknown image in profile {name}")
    task_ids = []
    for name, task in settings["tasks"].items():
        require_identifier(name, "task name")
        _validate_binding(settings, task)
        task_ids.append(require_identifier(task.get("id"), "task ID"))
    if len(task_ids) != 16 or set(task_ids) != {f"T{i:02d}" for i in range(1, 17)}:
        raise ValueError("Configure each of T01 through T16 exactly once")
    _validate_binding(settings, settings.get("coordinator"))
    if settings["profiles"][settings["coordinator"]["profile"]]["gpu"] is not None:
        raise ValueError("The coordinator must use a separate CPU profile")
    return settings


def _validate_binding(settings: Mapping[str, Any], binding: Any) -> None:
    if not isinstance(binding, dict) or binding.get("profile") not in settings["profiles"]:
        raise ValueError("Every worker/coordinator requires a known resource profile")
    if binding.get("volumes") != ["artifacts"]:
        raise ValueError("Every worker/coordinator must mount the shared artifacts Volume")
    names = binding.get("secrets")
    if not isinstance(names, list) or any(name not in settings["secrets"] for name in names):
        raise ValueError("Worker Secret references must name configured Secrets")
    if len(set(names)) != len(names):
        raise ValueError("Secret references must be distinct")


def project_files(*, require_lock: bool = False) -> tuple[str, ...]:
    """Enumerate allowed files; an optional existing lock can accompany source."""
    files = []
    for directory in PACKAGE_DIRECTORIES:
        if not os.path.isdir(directory):
            if directory == "assets":
                continue  # Benchmark assets are created in Step 4.
            raise FileNotFoundError(f"Packaged project directory is missing: {directory}")
        project_path(directory)
        for root, subdirectories, filenames in os.walk(directory, followlinks=False):
            for name in subdirectories:
                child = project_path(os.path.join(root, name))
                if os.path.islink(child):
                    raise ValueError(f"Do not package symlinked directories: {child}")
            subdirectories[:] = sorted(name for name in subdirectories if name not in IGNORED_DIRECTORIES)
            for name in sorted(filenames):
                path = project_path(os.path.join(root, name))
                if name.endswith((".pyc", ".pyo")) or name.startswith(".env"):
                    continue
                if os.path.islink(path):
                    raise ValueError(f"Do not package symlinked files: {path}")
                files.append(path)
    for path in PACKAGE_FILES:
        if not os.path.isfile(path):
            if path == "requirements.lock" and not require_lock:
                continue
            raise FileNotFoundError(f"Missing packaged project file: {path}")
        project_path(path)
        if os.path.islink(path):
            raise ValueError(f"Do not package symlinked files: {path}")
        files.append(path)
    return tuple(sorted(files))


def container_root(settings: Mapping[str, Any]) -> str:
    """Absolute image destination required by the Modal SDK, never a task path."""
    return "/" + settings["repo_directory"]


def dependency_packages(settings: Mapping[str, Any]) -> tuple[str, ...]:
    """Use project, Modal, development, and build requirements without duplication."""
    with open(project_path(settings["dependency_file"]), "rb") as handle:
        project = tomllib.load(handle)
    package_lists = [
        project["project"]["dependencies"],
        project["project"]["optional-dependencies"]["modal"],
        project["project"]["optional-dependencies"]["dev"],
        project["build-system"]["requires"],
    ]
    if any(not isinstance(items, list) or not items or any(not isinstance(item, str) or not item for item in items)
           for items in package_lists):
        raise ValueError("pyproject.toml must declare project, modal, dev, and build dependencies")
    return tuple(dict.fromkeys(package for items in package_lists for package in items))


def build_image(settings: Mapping[str, Any], profile: str) -> Image:
    if profile not in settings["images"]:
        raise ValueError(f"Unknown image profile: {profile}")
    files = project_files()
    key = (profile, sha256_file(settings["dependency_file"]), repr(settings))
    if key in _IMAGES:
        return _IMAGES[key]
    descriptor = settings["images"][profile]
    image = Image.from_registry(descriptor["base_image"], add_python=settings["python_version"])
    packages = descriptor["apt_packages"]
    if packages:
        image = image.apt_install(*packages)
    image = image.pip_install(*dependency_packages(settings))
    for path in files:
        image = image.add_local_file(path, os.path.join(container_root(settings), path), copy=True)
    image = image.workdir(container_root(settings)).run_commands(
        "pip install --no-deps --no-build-isolation -e .",
    )
    # Modal imports a script entrypoint by its basename ('verify'), not by the
    # SDK's 'modal' package. Resolve it from the explicitly packaged directory.
    image = image.env({"PYTHONPATH": "modal:."})
    _IMAGES[key] = image
    return image


def artifact_volume(settings: Mapping[str, Any]) -> Volume:
    descriptor = settings["volumes"]["artifacts"]
    name = descriptor["name"]
    if name not in _VOLUMES:
        _VOLUMES[name] = Volume.from_name(name, create_if_missing=descriptor["create_if_missing"])
    return _VOLUMES[name]


def worker_options(settings: Mapping[str, Any], task: str, *, coordinator: bool = False) -> dict[str, Any]:
    binding = settings["coordinator"] if coordinator else settings["tasks"][task]
    profile = settings["profiles"][binding["profile"]]
    secrets = [
        modal.Secret.from_name(
            settings["secrets"][name]["name"], required_keys=settings["secrets"][name]["required_keys"],
        ) for name in binding["secrets"]
    ]
    return {
        "image": build_image(settings, profile["image"]),
        "cpu": profile["cpu"], "memory": profile["memory_mib"],
        "gpu": profile["gpu"], "timeout": profile["timeout_seconds"],
        "volumes": {os.path.join(container_root(settings), settings["artifact_root"]): artifact_volume(settings)},
        "secrets": secrets, "max_containers": 1, "retries": 0,
        "include_source": False,
    }


def register_worker(app: modal.App, settings: Mapping[str, Any], task: str) -> Callable[..., Any]:
    """Return a decorator; the wrapper supplies the task implementation."""
    return app.function(**worker_options(settings, task))


def require_volume_path(settings: Mapping[str, Any], path: str) -> str:
    path = project_path(path)
    mount = settings["artifact_root"]
    if os.path.commonpath((mount, path)) != mount:
        raise ValueError(f"Remote artifact root must be {mount}/ or one of its subdirectories")
    return path


@contextmanager
def volume_session(settings: Mapping[str, Any], artifact_root: str) -> Iterator[None]:
    """Reload before reading; commit reports/checkpoints even on task failure."""
    if modal.is_local():
        raise RuntimeError("Volume sessions must execute inside a Modal worker")
    if os.path.realpath(os.getcwd()) != container_root(settings):
        raise RuntimeError("Modal worker must execute from its packaged repository root")
    require_volume_path(settings, artifact_root)
    volume = artifact_volume(settings)
    volume.reload()
    try:
        yield
    finally:
        # Task helpers use context managers, so files are closed before commit.
        volume.commit()


def require_packaged_config(path: str) -> str:
    path = project_path(path)
    if path.split(os.sep)[0] != "configs" or not os.path.isfile(path):
        raise FileNotFoundError(f"Modal task configuration must be a packaged configs/ file: {path}")
    return path


def shared_arguments(
    *, config: str, overrides: Sequence[str] = (), run_id: str = "", resume: bool = False,
    smoke: bool = False, artifact_root: str = "", seed: int | None = None, group: str = "",
) -> list[str]:
    """Forward business arguments to the existing local argparse engine."""
    argv = ["--config", require_packaged_config(config)]
    for override in overrides:
        if not isinstance(override, str) or "=" not in override:
            raise ValueError("Each --set override must be a KEY=VALUE string")
        argv.extend(("--set", override))
    for flag, value in (("--run-id", run_id), ("--artifact-root", artifact_root), ("--group", group)):
        if value:
            argv.extend((flag, value))
    if seed is not None:
        argv.extend(("--seed", str(seed)))
    if resume:
        argv.append("--resume")
    if smoke:
        argv.append("--smoke")
    return argv


def script_arguments(
    description: str, *, default_config: str, phases: Sequence[str] = (), argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Parse ``python modal/<task>.py`` with the same flags as its local task."""
    def add_phase(parser: Any) -> None:
        if phases:
            parser.add_argument("--phase", choices=phases, help="Verification phase.")

    parser = build_parser(
        description, default_config=default_config, supports_seed=False, add_arguments=add_phase,
    )
    arguments = parser.parse_args(argv)
    # LocalEntrypoint accepts a JSON list; the direct Python CLI accepts the
    # same repeated --set flags as src/tasks/<task>.py.
    values = {
        "config": arguments.config, "set": json.dumps(arguments.overrides),
        "run_id": arguments.run_id or "", "resume": arguments.resume,
        "smoke": arguments.smoke, "artifact_root": arguments.artifact_root or "",
    }
    if phases:
        values["phase"] = arguments.phase or ""
    return values
