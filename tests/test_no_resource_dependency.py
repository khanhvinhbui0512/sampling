"""Static boundaries plus a fresh-process smoke in a disposable reference-free copy."""

import ast
import os
import shutil
import subprocess
import sys
import tomllib

from src.config import load_config


def runtime_files():
    files = sorted(name for name in os.listdir(".") if name.endswith(".py"))
    for root in ("src", "modal"):
        for parent, directories, names in os.walk(root):
            directories[:] = sorted(name for name in directories if name != "__pycache__")
            files.extend(os.path.join(parent, name) for name in sorted(names) if name.endswith(".py"))
    return files


def resource_path(value):
    return isinstance(value, str) and "resource" in value.replace("\\", "/").split("/")


def test_runtime_imports_load_paths_and_modal_allowlist():
    loaders = {"from_pretrained", "load_from_disk", "load", "load_file", "open",
               "exec_module", "spec_from_file_location", "import_module", "__import__"}
    for path in runtime_files():
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] != "resource" for alias in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "resource", path
            if not isinstance(node, ast.Call):
                continue
            function = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            strings = [argument.value for argument in node.args if isinstance(argument, ast.Constant)]
            if function in loaders:
                assert not any(resource_path(value) or str(value).startswith("resource.") for value in strings), path
            if function in ("append", "insert", "extend") and isinstance(node.func, ast.Attribute):
                if ast.unparse(node.func.value) == "sys.path":
                    assert not any(resource_path(value) for value in strings), path
            if function in ("add_local_dir", "add_local_file"):
                assert node.args and isinstance(node.args[0], ast.Constant), path
                source = node.args[0].value
                assert isinstance(source, str) and not os.path.isabs(source) and not resource_path(source), path
                if function == "add_local_dir":
                    assert source in {"src", "configs", "tests", "modal"}, path
                else:
                    assert os.path.isfile(source) and ".." not in source.split("/"), path


def test_package_and_model_configuration_exclude_reference_code():
    with open("pyproject.toml", "rb") as handle:
        project = tomllib.load(handle)
    discovery = project["tool"]["setuptools"]["packages"]["find"]
    assert discovery["include"] == ["src", "src.*"] and discovery["namespaces"] is False
    assert project["tool"]["setuptools"]["include-package-data"] is False
    model = load_config("configs/base_training.yaml")["model"]
    assert not model.get("auto_map") and not resource_path(model.get("_name_or_path"))
    for name in os.listdir("configs"):
        if not name.endswith(".yaml"):
            continue
        def check(value):
            if isinstance(value, dict):
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
            else:
                assert not resource_path(value), name
        check(load_config(os.path.join("configs", name)))


def test_fresh_process_pipeline_and_wheel_without_resource(artifact_directory):
    isolated = os.path.join(artifact_directory, "isolated")
    os.makedirs(isolated)
    for directory in ("src", "configs", "tests", "modal"):
        shutil.copytree(directory, os.path.join(isolated, directory),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    files = {"pyproject.toml", "README.md", "THIRD_PARTY_NOTICES.md"}
    files.update(name for name in os.listdir(".") if name.endswith(".py"))
    for name in sorted(files):
        shutil.copy2(name, os.path.join(isolated, name))
    assert not os.path.exists(os.path.join(isolated, "resource"))
    environment = {**os.environ, "PYTHONPATH": "", "PYTHONNOUSERSITE": "1",
                   "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    result = subprocess.run([sys.executable, "tests/isolated_smoke.py"], cwd=isolated,
                            env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Isolated smoke passed" in result.stdout
