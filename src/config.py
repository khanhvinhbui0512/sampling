"""Read one YAML file and keep runtime paths relative to the project root."""

import os

import yaml


class ConfigError(ValueError):
    """Invalid project configuration or path."""


def project_path(path: str) -> str:
    """Keep relative paths inside the repo or its workspace Volume, excluding resource/."""
    if not os.path.isfile("pyproject.toml") or not os.path.isdir("src"):
        raise ConfigError("Run this task from the repository root")
    if not isinstance(path, str) or not path.strip() or "\x00" in path:
        raise ConfigError("A project path must be a non-empty string")
    if os.path.isabs(path) or "\\" in path or ":" in path or path.startswith("~"):
        raise ConfigError(f"Use a project-relative path: {path!r}")
    normalized = os.path.normpath(path)
    if normalized == ".." or normalized.startswith(".." + os.sep):
        raise ConfigError(f"Path escapes the repository: {path!r}")
    root = os.path.realpath(".")
    actual = os.path.realpath(normalized)
    first = normalized.split(os.sep)[0]
    resource_root = os.path.realpath("resource")
    if first == "resource" or os.path.commonpath((resource_root, actual)) == resource_root:
        raise ConfigError("resource/ is reference-only and cannot be a runtime input/output")
    # Modal mounts workspace/ through a symlink outside the container's repo.
    # Descendants must still remain inside that Volume's resolved root.
    allowed_root = os.path.realpath("workspace") if first == "workspace" else root
    if os.path.commonpath((allowed_root, actual)) != allowed_root:
        raise ConfigError(f"Path escapes its project directory through a symlink: {path!r}")
    return normalized


def load_config(path: str = "configs/qm9.yaml") -> dict:
    """Load a plain dictionary, without inheritance, schemas, or CLI overrides."""
    with open(project_path(path), encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return config
