"""Verify source packaging and runtime behavior without reference repositories."""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tomllib

import pytest

from src.config import project_path


def _python_files(directory: str):
    for root, directories, files in os.walk(directory, followlinks=False):
        directories[:] = sorted(name for name in directories if name != "__pycache__")
        for name in sorted(files):
            if name.endswith(".py"):
                yield project_path(os.path.join(root, name))


def test_runtime_imports_do_not_use_reference_repositories_or_pathlib():
    violations = []
    for path in (*_python_files("src"), *_python_files("modal")):
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Call) and node.args:
                # Also cover literal dynamic imports and direct runtime reads.
                callee = ast.unparse(node.func)
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    value = argument.value
                    if callee.endswith(("import_module", "__import__")):
                        names = [value]
                    if callee in {"open", "os.open"} and "resource" in value.split("/"):
                        violations.append(f"{path}:{node.lineno}: runtime read of {value}")
                if callee in {"sys.path.insert", "sys.path.append", "sys.path.extend", "importlib.util.spec_from_file_location"}:
                    for argument in node.args:
                        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                            if "resource" in argument.value.split("/"):
                                violations.append(f"{path}:{node.lineno}: reference path in {callee}")
            for name in names:
                if name.split(".")[0] in {"resource", "pathlib"} or name == "modal.common":
                    violations.append(f"{path}:{node.lineno}: import {name}")
    assert not violations, "\n".join(violations)


def test_runtime_paths_reject_reference_files_and_escaping_symlinks():
    for path in ("resource/file.py", "assets/resource/data.json", "../outside.json"):
        with pytest.raises(ValueError):
            project_path(path)
    os.makedirs(".pytest_cache", exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="isolation-", dir=".pytest_cache") as directory:
        link = os.path.relpath(os.path.join(directory, "escape"))
        os.symlink(os.path.dirname(os.getcwd()), link)
        with pytest.raises(ValueError, match="symlink"):
            project_path(link)


def test_project_package_excludes_modal_and_reference_repositories():
    with open("pyproject.toml", "rb") as handle:
        settings = tomllib.load(handle)
    packages = settings["tool"]["setuptools"]["packages"]["find"]
    assert packages["where"] == ["."]
    assert set(packages["include"]) == {"src", "src.*"}
    assert {"resource", "resource.*", "modal", "modal.*", "tests", "tests.*"} <= set(packages["exclude"])
    assert packages["namespaces"] is False
    assert not os.path.exists("modal/__init__.py")


def test_modal_sdk_is_not_shadowed():
    try:
        importlib.metadata.version("modal")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Local Modal SDK is optional; install .[modal] for this check")
    spec = importlib.util.find_spec("modal")
    assert spec is not None and spec.origin is not None, "modal resolved to a namespace directory"
    sdk_path = os.path.realpath(spec.origin)
    wrapper_path = os.path.realpath("modal")
    assert os.path.commonpath((sdk_path, wrapper_path)) != wrapper_path


def test_modal_packaging_excludes_reference_repositories_and_generated_artifacts():
    try:
        importlib.metadata.version("modal")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Install .[modal] to check Modal packaging")
    spec = importlib.util.spec_from_file_location("smiles_modal_isolation", "modal/common.py")
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.load_settings()
    files = helper.project_files(require_lock=False)
    assert files and {"src/tasks/verify.py", "modal/common.py", "modal/verify.py"} <= set(files)
    assert all(path.split(os.sep)[0] in {"src", "configs", "modal", "tests", "assets", "pyproject.toml", "requirements.lock"} for path in files)
    assert not any("resource" in path.split(os.sep) for path in files)
    assert not any(path.split(os.sep)[0] in {"data", "checkpoints", "outputs", "reports"} for path in files)


def test_project_imports_and_artifacts_work_without_resource():
    """Import every implemented src module in a separate, reference-free root."""
    os.makedirs(".pytest_cache", exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="package-", dir=".pytest_cache") as directory:
        for name in ("src", "configs", "modal", "assets"):
            if os.path.isdir(name):
                shutil.copytree(name, os.path.join(directory, name), ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile("pyproject.toml", os.path.join(directory, "pyproject.toml"))
        probe = textwrap.dedent('''\
            import importlib
            import json
            import os
            import pkgutil
            import src
            from src.artifacts import ArtifactStore
            from src.config import load_config, project_path

            assert not os.path.exists("resource")
            imported = []
            for item in pkgutil.walk_packages(src.__path__, prefix="src."):
                importlib.import_module(item.name)
                imported.append(item.name)
            config = load_config()
            assert config["project"]["dataset"] == "cars_smiles"
            store = ArtifactStore("artifacts")
            store.write_json("checks/probe.json", {"groups": config["groups"]})
            assert store.read_json("checks/probe.json")["groups"] == config["groups"]
            try:
                project_path("resource/forbidden.json")
            except ValueError:
                pass
            else:
                raise AssertionError("Reference path was accepted")
            assert not any(name == "resource" or name.startswith("resource.") for name in imported)
            print(json.dumps({"imported": imported, "dataset": config["project"]["dataset"]}))
        ''')
        with open(os.path.join(directory, "isolation_probe.py"), "w", encoding="utf-8") as handle:
            handle.write(probe)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "isolation_probe.py"], cwd=directory, env=environment,
            capture_output=True, text=True, timeout=120, check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        assert {"src.config", "src.artifacts", "src.tracking", "src.tasks.verify"} <= set(payload["imported"])
        assert payload["dataset"] == "cars_smiles"
