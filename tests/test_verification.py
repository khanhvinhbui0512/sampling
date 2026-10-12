"""Runner failures must keep their logs and original pytest exit codes."""

import os
import tempfile
from types import SimpleNamespace

import pytest

from src.experiments.verification import read_verification, test_selection as select_tests, verify
from src.io import read_json, write_yaml


@pytest.mark.parametrize("exit_code", [0, 1, 2, 3, 4, 5])
def test_runner_persists_exit_codes_and_teardown_failures(monkeypatch, artifact_directory, exit_code):
    def fake_suite(arguments, plugins):
        assert arguments[-1] == "tests/test_budget.py"
        assert arguments[arguments.index("-m") + 1] == "not gpu"
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["QM9_GPU_SMOKE"] == "0"
        print("recorded test output")
        if exit_code in (0, 1):
            plugins[0].pytest_runtest_logreport(SimpleNamespace(nodeid="fake::case", when="call", outcome="passed", duration=.01))
        if exit_code == 1:
            plugins[0].pytest_runtest_logreport(SimpleNamespace(nodeid="fake::case", when="teardown", outcome="failed", duration=.01))
        return exit_code

    monkeypatch.setattr("src.experiments.verification.pytest.main", fake_suite)
    monkeypatch.setenv("HF_HUB_OFFLINE", "original")
    os.makedirs("workspace/results/raw", exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="step12-runner-", dir="workspace/results/raw") as directory:
        config = os.path.join(artifact_directory, "verification.yaml")
        write_yaml(config, {"output_dir": os.path.relpath(directory), "selection": ["tests/test_budget.py"],
                            "gpu_smoke": False, "cpu_threads": 1})
        result = verify(config_path=config, attempt_id="test")
        assert result["exit_code"] == exit_code and read_verification(result["manifest"]) == result
        manifest = read_json(result["manifest"])
        assert manifest["status"] == ("completed" if exit_code == 0 else "failed")
        with open(os.path.join(os.path.dirname(result["manifest"]), "tests.txt"), encoding="utf-8") as handle:
            assert "recorded test output" in handle.read()
        if exit_code == 1:
            assert result["counts"]["failed"] == 1 and result["counts"]["passed"] == 0
        assert os.environ["HF_HUB_OFFLINE"] == "original"
        # Readback checks actual hashes even for failed attempts.
        with open(os.path.join(os.path.dirname(result["manifest"]), "tests.txt"), "a", encoding="utf-8") as handle:
            handle.write("tampered")
        with pytest.raises(ValueError, match="changed"):
            read_verification(result["manifest"])


@pytest.mark.parametrize("selection", ["resource/example.py", "--help", "../tests/test_budget.py", "tests/missing.py", "tests/test_budget.py::"])
def test_selection_rejects_switches_outside_paths_and_missing_files(selection):
    with pytest.raises(ValueError):
        select_tests(selection)
