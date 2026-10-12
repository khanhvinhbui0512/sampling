"""Run the offline test suite and retain evidence, including failed attempts."""

import os
import platform
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout

import pytest
import torch
from tqdm.auto import tqdm

from src.config import load_config, project_path
from src.io import (RunArtifacts, _atomic_writer, file_hash, read_json,
                    report_result, write_json)


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, message):
        self.console.write(message)
        return self.log.write(message)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False


class TestProgress:
    """Keep setup/teardown failures as well as ordinary assertion failures."""

    def __init__(self):
        self.reports, self.collection_errors = [], []
        self.deselected = 0
        self.progress = None

    def pytest_configure(self, config):
        config.addinivalue_line("markers", "gpu: explicit opt-in CUDA smoke test")

    def pytest_deselected(self, items):
        self.deselected += len(items)

    def pytest_collection_finish(self, session):
        self.progress = tqdm(total=len(session.items), desc="Verify", unit="test")

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append({"nodeid": report.nodeid, "error": str(report.longrepr)})

    def pytest_runtest_logreport(self, report):
        self.reports.append({"nodeid": report.nodeid, "phase": report.when,
                             "outcome": report.outcome, "seconds": report.duration})

    def pytest_runtest_logfinish(self, nodeid, location):
        if self.progress is not None:
            self.progress.update(1)

    def counts(self):
        nodes = {}
        for report in self.reports:
            nodes.setdefault(report["nodeid"], []).append(report)
        counts = {"passed": 0, "failed": 0, "skipped": 0}
        for reports in nodes.values():
            if any(row["outcome"] == "failed" for row in reports):
                counts["failed"] += 1
            elif any(row["outcome"] == "skipped" for row in reports):
                counts["skipped"] += 1
            elif any(row["phase"] == "call" and row["outcome"] == "passed" for row in reports):
                counts["passed"] += 1
        return {**counts, "collection_errors": len(self.collection_errors), "deselected": self.deselected}


def test_selection(selection):
    """Accept test paths/node IDs, never arbitrary pytest switches or directories."""
    if isinstance(selection, str):
        selection = [item.strip() for item in selection.split(",") if item.strip()]
    if not isinstance(selection, (list, tuple)) or not selection:
        raise ValueError("Select tests or explicit tests/test_*.py node IDs")
    result = []
    for item in selection:
        if not isinstance(item, str):
            raise ValueError("Test selections must be strings")
        path, *nodes = item.split("::")
        path = project_path(path)
        if path != "tests" and not (path.startswith("tests/") and path.endswith(".py")):
            raise ValueError("Selections must be tests or Python test files under tests/")
        if not os.path.exists(path) or any(not node or node.startswith("-") for node in nodes):
            raise ValueError(f"Invalid test selection: {item}")
        result.append("::".join([path, *nodes]))
    return list(dict.fromkeys(result))


def verify(*, attempt_id, config_path="configs/verification.yaml", selection="",
           gpu_smoke=False, run_id="qm9-verify", execution_metadata=None):
    settings = load_config(config_path)
    selected = test_selection(selection or settings["selection"])
    if type(gpu_smoke) is not bool or type(settings["gpu_smoke"]) is not bool:
        raise ValueError("gpu_smoke must be a boolean")
    gpu_smoke = gpu_smoke or settings["gpu_smoke"]
    threads = settings["cpu_threads"]
    if type(threads) is not int or threads <= 0:
        raise ValueError("cpu_threads must be a positive integer")
    output_dir = project_path(settings["output_dir"])
    if output_dir != "workspace/results/raw" and not output_dir.startswith("workspace/results/raw/"):
        raise ValueError("Verification logs belong under workspace/results/raw/")
    paths = [config_path, "pyproject.toml"]
    paths.extend(sorted(name for name in os.listdir(".") if name.endswith(".py")))
    for root in ("src", "tests", "modal", "configs"):
        for parent, directories, files in os.walk(root):
            directories[:] = sorted(name for name in directories if name != "__pycache__")
            paths.extend(os.path.join(parent, name) for name in sorted(files)
                         if name.endswith((".py", ".yaml", ".json", ".lark", ".sh")))
    protocol = {"selection": selected, "gpu_smoke": gpu_smoke, "cpu_threads": threads,
                "execution": execution_metadata or {"backend": "local"},
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                "pytest": pytest.__version__},
                "source_sha256": {path: file_hash(path) for path in paths},
                "scope": "offline fixtures; no production QM9 checkpoint or benchmark gate"}
    attempt = RunArtifacts.create(output_dir, run_id=run_id, attempt_id=attempt_id,
                                  config=protocol, target_id="qm9-verification-v1",
                                  reference_id="offline-tests-and-independent-controls")
    attempt.manifest.update(task="verify_qm9")
    write_json(attempt.path("protocol.json"), protocol)
    attempt.register("protocol.json")
    plugin, started = TestProgress(), time.perf_counter()
    environment = {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "HF_HUB_OFFLINE": "1",
                   "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "QM9_GPU_SMOKE": "1" if gpu_smoke else "0"}
    previous = {name: os.environ.get(name) for name in environment}
    previous_threads = torch.get_num_threads()
    exit_code, error = 3, None
    try:
        os.environ.update(environment)
        torch.set_num_threads(threads)
        with _atomic_writer(attempt.path("tests.txt")) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                print(f"Selection: {selected}; GPU smoke: {gpu_smoke}", flush=True)
                arguments = ["-ra", "--color=no", "--capture=tee-sys", "-p", "no:cacheprovider"]
                if not gpu_smoke:
                    arguments.extend(["-m", "not gpu"])
                try:
                    exit_code = int(pytest.main([*arguments, *selected], plugins=[plugin]))
                except (Exception, KeyboardInterrupt) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    exit_code = 2 if isinstance(exc, KeyboardInterrupt) else 3
                    traceback.print_exc()
                finally:
                    if plugin.progress is not None:
                        plugin.progress.close()
    finally:
        torch.set_num_threads(previous_threads)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    summary = {"exit_code": exit_code, "counts": plugin.counts(), "error": error,
               "selection": selected, "gpu_smoke": gpu_smoke,
               "elapsed_seconds": time.perf_counter() - started,
               "reports": plugin.reports, "collection_errors": plugin.collection_errors,
               "scope": protocol["scope"]}
    write_json(attempt.path("summary.json"), summary)
    for name in ("tests.txt", "summary.json"):
        attempt.register(name)
    if exit_code:
        attempt.fail(error or f"pytest exited with code {exit_code}; see tests.txt")
    else:
        attempt.complete(required_artifacts=("protocol.json", "tests.txt", "summary.json"))
    result = {"manifest": attempt.path("manifest.json"), "exit_code": exit_code,
              "counts": summary["counts"]}
    report_result(f"Verification exit={exit_code}; {summary['counts']}; manifest={result['manifest']}")
    return result


def read_verification(manifest_path):
    """Read committed logs from another task, including an unsuccessful suite."""
    manifest_path = project_path(manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get("task") != "verify_qm9" or manifest.get("status") not in ("completed", "failed"):
        raise ValueError("Expected a finished verification attempt")
    for name in ("protocol.json", "tests.txt", "summary.json"):
        record = manifest["artifacts"][name]
        path = os.path.join(os.path.dirname(manifest_path), name)
        if file_hash(path) != record["sha256"] or os.path.getsize(path) != record["size_bytes"]:
            raise ValueError(f"Verification artifact changed: {name}")
    summary = read_json(os.path.join(os.path.dirname(manifest_path), "summary.json"))
    if (summary["exit_code"] == 0) != (manifest["status"] == "completed"):
        raise ValueError("Verification status and exit code disagree")
    return {"manifest": manifest_path, "exit_code": summary["exit_code"], "counts": summary["counts"]}
