"""Resolve YAML configuration without importing tasks or model implementations."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import yaml

from src.schemas import GROUPS, SCHEMA_VERSION, SUITES, require_count, require_identifier

Config = dict[str, Any]
ConfigValidator = Callable[[Config], None]


def project_path(path: str, *, allow_dot: bool = False) -> str:
    """Validate a project-relative path, including its existing symlink targets."""
    if not isinstance(path, str) or not path or "\x00" in path or "\\" in path:
        raise ValueError(f"Invalid project-relative path: {path!r}")
    if os.path.isabs(path) or ".." in path.split("/"):
        raise ValueError(f"Path must stay relative to the project root: {path!r}")
    normalized = os.path.normpath(path)
    if normalized == "." and not allow_dot:
        raise ValueError("Expected a file or directory path, not the project root")
    if "resource" in normalized.split(os.sep):
        raise ValueError("Runtime paths must not use resource/")
    root = os.path.realpath(os.getcwd())
    resolved = os.path.realpath(normalized)
    if os.path.commonpath((root, resolved)) != root:
        raise ValueError(f"Path escapes the project through a symlink: {path!r}")
    if "resource" in os.path.relpath(resolved, root).split(os.sep):
        raise ValueError("Runtime paths must not resolve into resource/")
    return normalized


class _UniqueSafeLoader(yaml.SafeLoader):
    """Fail on duplicate keys instead of silently changing an experiment."""


def _construct_mapping(loader: _UniqueSafeLoader, node: yaml.MappingNode, deep: bool = False) -> Config:
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("Configuration mapping keys must be strings")
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def read_yaml(path: str) -> Config:
    path = project_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration is missing: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            values = yaml.load(handle, Loader=_UniqueSafeLoader)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"Cannot read configuration {path}: {exc}") from exc
    if not isinstance(values, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return values


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> Config:
    """Merge mappings recursively; replace lists and scalars, without mutation."""
    result = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_tree(path: str, common_path: str, stack: tuple[str, ...] = ()) -> Config:
    path = project_path(path)
    if path in stack:
        raise ValueError(f"Configuration defaults cycle: {' -> '.join((*stack, path))}")
    values = read_yaml(path)
    defaults = values.pop("defaults", [])
    if isinstance(defaults, str):
        defaults = [defaults]
    if not isinstance(defaults, list) or any(not isinstance(item, str) for item in defaults):
        raise TypeError(f"defaults in {path} must be a filename or list of filenames")
    merged: Config = {}
    for item in defaults:
        if os.path.isabs(item):
            raise ValueError("Configuration defaults must be project-relative")
        # Defaults are relative to their declaring YAML, as in the reference repo.
        dependency = project_path(os.path.normpath(os.path.join(os.path.dirname(path), item)))
        if dependency == path or dependency in stack:
            raise ValueError(f"Configuration defaults cycle at {dependency}")
        # Common is already loaded once, before the selected suite preset.
        if dependency != common_path:
            merged = deep_merge(merged, _read_tree(dependency, common_path, (*stack, path)))
    return deep_merge(merged, values)


def apply_overrides(config: Mapping[str, Any], overrides: Sequence[str] = ()) -> Config:
    """Override declared keys with YAML scalar/list types; reject misspelled keys."""
    result = copy.deepcopy(dict(config))
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value: {override!r}")
        key, raw_value = override.split("=", 1)
        parts = key.split(".")
        if not parts or any(not part.isidentifier() for part in parts):
            raise ValueError(f"Invalid dotted override key: {key!r}")
        try:
            value = yaml.load(raw_value, Loader=_UniqueSafeLoader)
        except (yaml.YAMLError, ValueError) as exc:
            raise ValueError(f"Invalid override {key}: {exc}") from exc
        target = result
        for part in parts[:-1]:
            if part not in target:
                raise ValueError(f"Unknown override key: {key}")
            if not isinstance(target[part], dict):
                raise ValueError(f"Override {key} traverses a non-mapping at {part}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown override key: {key}")
        target[parts[-1]] = value
    return result


def _mapping(config: Mapping[str, Any], key: str, required: set[str]) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"Missing {key} keys: {', '.join(missing)}")
    return value


def validate_config(config: Config) -> None:
    """Validate shared settings; task validators handle their own later sections."""
    if type(config.get("schema_version")) is not int or config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported configuration schema_version")
    project = _mapping(config, "project", {"name", "version", "task", "dataset"})
    if project["task"] != "molecular_generation" or project["dataset"] != "cars_smiles":
        raise ValueError("This project supports only the CARS SMILES benchmark")
    require_identifier(project["name"], "project name")
    require_identifier(project["version"], "project version")
    groups = config.get("groups")
    if not isinstance(groups, list) or tuple(groups) != GROUPS:
        raise ValueError(f"groups must be {list(GROUPS)} in this order")
    suite = config.get("suite")
    if suite not in SUITES:
        raise ValueError(f"Unknown suite: {suite!r}")
    require_count(config.get("seed"), "seed")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a nonempty list")
    for seed in [config["seed"], *seeds]:
        require_count(seed, "seed")
        if seed >= 2**32:
            raise ValueError("Seeds must be smaller than 2**32")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    if config.get("group") is not None and config["group"] not in GROUPS:
        raise ValueError("Unknown group selection")
    reward = _mapping(config, "reward", {"mode", "version"})
    expected_mode = "grammar_only" if suite == "replication" else "grammar_rdkit_smarts"
    if reward["mode"] != expected_mode:
        raise ValueError(f"Suite {suite} requires reward.mode={expected_mode}")
    require_identifier(reward["version"], "reward version")
    protocol = _mapping(config, "protocol", {
        "target_unique", "max_root_attempts", "exclude_prompt_examples", "uniqueness",
        "count_prefix_aborts", "report_incomplete_runs", "max_model_nfe", "max_compute_seconds",
    })
    for key in ("target_unique", "max_root_attempts"):
        require_count(protocol[key], f"protocol.{key}", positive=True)
    if protocol["target_unique"] > protocol["max_root_attempts"]:
        raise ValueError("target_unique cannot exceed max_root_attempts")
    for key in ("exclude_prompt_examples", "count_prefix_aborts", "report_incomplete_runs"):
        if protocol[key] is not True:
            raise ValueError(f"protocol.{key} must stay true for this benchmark")
    if protocol["uniqueness"] != "canonical_isomeric_smiles":
        raise ValueError("Molecular uniqueness must use canonical isomeric SMILES")
    if protocol["max_model_nfe"] is not None:
        require_count(protocol["max_model_nfe"], "max_model_nfe", positive=True)
    if protocol["max_compute_seconds"] is not None:
        value = protocol["max_compute_seconds"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float("inf"):
            raise ValueError("max_compute_seconds must be finite and positive or null")
    paths = _mapping(config, "paths", {
        "artifact_root", "benchmark", "data", "checkpoints", "outputs", "reports",
    })
    for key, value in paths.items():
        project_path(value, allow_dot=key == "artifact_root")
    runtime = _mapping(config, "runtime", {
        "execution", "modal_volume", "deterministic", "progress", "log_level",
    })
    if runtime["execution"] not in ("local", "modal"):
        raise ValueError("runtime.execution must be local or modal")
    require_identifier(runtime["modal_volume"], "Modal Volume")
    for key in ("deterministic", "progress"):
        if type(runtime[key]) is not bool:
            raise TypeError(f"runtime.{key} must be a boolean")
    if runtime["log_level"] not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError("Unknown runtime.log_level")
    try:
        json.dumps(config, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Configuration must contain finite JSON-compatible values: {exc}") from exc


def load_config(
    task_path: str | None = None,
    *,
    common_path: str = "configs/common.yaml",
    overrides: Sequence[str] = (),
    validator: ConfigValidator | None = None,
) -> Config:
    """Precedence: common < task/defaults < suite preset < explicit overrides."""
    common_path = project_path(common_path)
    common = _read_tree(common_path, common_path)
    task = {}
    if task_path is not None and project_path(task_path) != common_path:
        task = _read_tree(task_path, common_path)
    candidate = apply_overrides(deep_merge(common, task), overrides)
    suite = candidate.get("suite")
    presets = _mapping(candidate, "suites", set(SUITES))
    if suite not in SUITES or not isinstance(presets[suite], Mapping):
        raise ValueError(f"Invalid suite preset: {suite!r}")
    resolved = deep_merge(deep_merge(common, task), presets[suite])
    resolved = apply_overrides(resolved, overrides)
    validate_config(resolved)
    if validator is not None:
        validator(resolved)
    return resolved


def config_digest(config: Mapping[str, Any]) -> str:
    """Hash the complete resolved configuration, including its protocol and paths."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dump_config(config: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(config), sort_keys=True, allow_unicode=True)
