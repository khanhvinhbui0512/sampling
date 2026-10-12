"""Coordinator controls are checked with offline artifact-producing workers."""

import copy
import os
import shutil

import pytest

from src.config import load_config
from src.experiments import benchmark as orchestration
from src.io import (RunArtifacts, read_completed_manifest, read_json, read_jsonl,
                    write_json, write_yaml)


def finish_fixture(base, run_id, files, *, task, attempt_id="fixture"):
    attempt = RunArtifacts.create(base, run_id=run_id, attempt_id=attempt_id, config={},
                                  target_id="fixture", reference_id="fixture")
    attempt.manifest["task"] = task
    for name, payload in files.items():
        write_json(attempt.path(name), payload)
        attempt.register(name)
    attempt.complete(required_artifacts=files)
    return attempt.path("manifest.json")


@pytest.mark.parametrize("change", ["none", "subset", "stale", "failed"])
def test_verification_gate_requires_full_current_passing_evidence(artifact_directory, change):
    protocol = {"selection": ["tests"], "source_sha256": {"code.py": "current"}}
    summary = {"exit_code": 0, "counts": {"passed": 5, "failed": 0, "skipped": 0, "collection_errors": 0}}
    if change == "subset":
        protocol["selection"] = ["tests/test_budget.py"]
    elif change == "stale":
        protocol["source_sha256"]["code.py"] = "old"
    elif change == "failed":
        summary["exit_code"] = 1
    path = finish_fixture(artifact_directory, "verify", {"protocol.json": protocol, "summary.json": summary,
                                                       "tests.txt": {"fixture_log": True}}, task="verify_qm9")
    if change == "none":
        orchestration.require_verification(path, {"code.py": "current"})
    else:
        with pytest.raises(ValueError):
            orchestration.require_verification(path, {"code.py": "current"})


@pytest.mark.parametrize("ready,comparison,accepted", [(True, "same", True), (False, "same", False), (True, "different", False)])
def test_final_gate_requires_ready_matching_pilot(artifact_directory, ready, comparison, accepted):
    worker = finish_fixture(artifact_directory, "worker", {"fixture.json": {}}, task="fixture")
    path = finish_fixture(artifact_directory, "pilot", {
        "plan.json": {"phase": "pilot", "comparison_hash": comparison},
        "summary.json": {"pilot_ready": ready, "all_units_complete": True, "selected_manifests": [worker]}}, task="benchmark_qm9")
    if accepted:
        orchestration.require_pilot(path, "same")
    else:
        with pytest.raises(ValueError):
            orchestration.require_pilot(path, "same")


def test_matrix_retains_both_deliverables_and_distinct_laws(artifact_directory):
    settings = load_config("configs/benchmark.yaml")
    orchestration.validate_matrix(settings)
    missing = copy.deepcopy(settings)
    missing["matrix"] = [row for row in missing["matrix"] if row["method"] != "cars"]
    with pytest.raises(ValueError, match="both deliverables"):
        orchestration.validate_matrix(missing)
    changed = copy.deepcopy(settings)
    changed["matrix"][0]["value_law"] = "rho"
    with pytest.raises(ValueError):
        orchestration.validate_matrix(changed)
    unlocked = copy.deepcopy(settings)
    unlocked["locked"] = False
    path = os.path.join(artifact_directory, "unlocked.yaml")
    write_yaml(path, unlocked)
    with pytest.raises(ValueError, match="explicitly lock"):
        orchestration.lock_protocol(path, device="cpu", execution_metadata={"backend": "local"})


@pytest.fixture
def coordinator_case(monkeypatch, artifact_directory):
    settings = load_config("configs/benchmark.yaml")
    settings.update(locked=True, phase="pilot")
    settings["seeds"]["pilot"] = [7, 13]
    settings["outputs_per_seed"]["pilot"] = 3
    settings["matrix"] = [settings["matrix"][6]]  # One guided row plus a value job per seed.
    settings["inputs"] = {name: f"workspace/fixture/{name}.json" for name in settings["inputs"]}
    configs = {name: load_config(path) for name, path in settings["configs"].items()}
    lock = {"settings": settings, "sources": orchestration.python_sources(), "dependencies": {},
            "comparison_hash": "fixture-contract", "device": "cpu", "execution": {"backend": "local"}}
    monkeypatch.setattr(orchestration, "lock_protocol", lambda *args, **kwargs: (copy.deepcopy(lock), configs))
    calls, state = [], {"fail_once": True}
    benchmark_id = "step13-" + os.path.basename(artifact_directory)

    def produce(**args):
        is_generation = "method" in args
        seed = args["seed"] if is_generation else load_config(args["config_path"])["seed"]
        count = args.get("num_samples")
        calls.append((is_generation, seed, count, args["run_id"], args["attempt_id"]))
        attempt = RunArtifacts.create("workspace/results/raw" if is_generation else "workspace/results/value_checks",
                                      run_id=args["run_id"], attempt_id=args["attempt_id"], config={},
                                      target_id="fixture", reference_id="fixture")
        attempt.manifest["task"] = "fixture-worker"
        if is_generation and state["fail_once"]:
            state["fail_once"] = False
            total = {"base_calls": 7, "base_sequences": 7, "verifier_calls": 0,
                     "verifier_sequences": 0, "oracle_calls": 0, "elapsed_seconds": 0.}
            budget = {"total": total, "by_phase": {"sampling": total}, "adjusted_nfe": 7,
                      "flop_measurement_id": None, "base_batch_sizes": [1], "verifier_batch_sizes": []}
            write_json(attempt.path("traces/partial.json"), {"budget": budget, "wall_seconds": 0.})
            attempt.register("traces/partial.json")
            attempt.fail("injected worker failure after recorded work")
            raise RuntimeError("injected worker failure")
        write_json(attempt.path("fixture.json"), {"seed": seed, "count": count, "args": args})
        attempt.register("fixture.json")
        attempt.complete(required_artifacts=("fixture.json",))
        return {"manifest": attempt.path("manifest.json")}

    def inspect(unit, path, plan):
        read_completed_manifest(path)
        data = orchestration.registered_json(path, "fixture.json")
        assert data["seed"] == unit["seed"]
        assert data["count"] == unit["arguments"].get("num_samples")
        return {"unit_id": unit["unit_id"], "manifest": path, "status": "completed",
                "stats": {"pilot_ready": True}, "cost": None, "cost_is_lower_bound": False}

    monkeypatch.setattr(orchestration, "generate", produce)
    monkeypatch.setattr(orchestration, "check_values", produce)
    monkeypatch.setattr(orchestration, "inspect_unit", inspect)
    try:
        yield benchmark_id, lock, calls
    finally:
        for base in ("workspace/results/raw", "workspace/results/value_checks", "workspace/results/benchmarks"):
            if os.path.isdir(base):
                for name in os.listdir(base):
                    if name == benchmark_id or name.startswith(benchmark_id + "-"):
                        shutil.rmtree(os.path.join(base, name))


def test_resume_only_unfinished_units_and_keeps_failed_work(coordinator_case):
    benchmark_id, _, calls = coordinator_case
    first = orchestration.benchmark(attempt_id="first", benchmark_id=benchmark_id, device="cpu")
    assert first["exit_code"] == 1 and first["completed_units"] == 3 and len(calls) == 4
    resumed = orchestration.benchmark(attempt_id="resumed", benchmark_id=benchmark_id,
                                      resume_manifest=first["manifest"], device="cpu")
    assert resumed["exit_code"] == 0 and resumed["completed_units"] == 4 and len(calls) == 5
    assert calls[0][:4] == calls[-1][:4] and calls[0][4] != calls[-1][4]
    directory = os.path.dirname(resumed["manifest"])
    ledger = list(read_jsonl(os.path.join(directory, "attempts.jsonl"), progress=False))
    assert len(ledger) == 5 and len({row["manifest"] for row in ledger}) == 5
    failed = [row for row in ledger if row["status"] != "completed"]
    assert len(failed) == 1 and failed[0]["cost_is_lower_bound"]
    assert failed[0]["cost"]["total"]["base_calls"] == 7
    assert all(call[2] == 3 for call in calls if call[0])
    repeated = orchestration.benchmark(attempt_id="no-work", benchmark_id=benchmark_id,
                                       resume_manifest=resumed["manifest"], device="cpu")
    assert repeated["exit_code"] == 0 and len(calls) == 5


def test_resume_refuses_changed_counts_and_duplicate_completed_attempts(coordinator_case):
    benchmark_id, lock, calls = coordinator_case
    first = orchestration.benchmark(attempt_id="first", benchmark_id=benchmark_id, device="cpu")
    lock["settings"]["outputs_per_seed"]["pilot"] = 4
    with pytest.raises(ValueError, match="Resume requires"):
        orchestration.benchmark(attempt_id="changed", benchmark_id=benchmark_id,
                                resume_manifest=first["manifest"], device="cpu")
    assert len(calls) == 4
    lock["settings"]["outputs_per_seed"]["pilot"] = 3
    resumed = orchestration.benchmark(attempt_id="resumed", benchmark_id=benchmark_id,
                                      resume_manifest=first["manifest"], device="cpu")
    plan = orchestration.registered_json(resumed["manifest"], "plan.json")
    orchestration.run_unit({"plan_manifest": resumed["manifest"], "unit_id": plan["units"][0]["unit_id"],
                            "attempt_id": "duplicate"}, device="cpu")
    before = len(calls)
    ambiguous = orchestration.benchmark(attempt_id="ambiguous", benchmark_id=benchmark_id,
                                        resume_manifest=resumed["manifest"], device="cpu")
    assert ambiguous["exit_code"] == 1 and ambiguous["completed_units"] == 3 and len(calls) == before


def test_worker_refuses_changed_snapshot_and_backend(coordinator_case):
    benchmark_id, _, calls = coordinator_case
    result = orchestration.benchmark(attempt_id="first", benchmark_id=benchmark_id, device="cpu")
    plan = orchestration.registered_json(result["manifest"], "plan.json")
    job = {"plan_manifest": result["manifest"], "unit_id": plan["units"][0]["unit_id"],
           "attempt_id": "never-started"}
    before = len(calls)
    with pytest.raises(ValueError, match="device/backend"):
        orchestration.run_unit(job, device="cpu", execution_metadata={"backend": "modal"})
    snapshot = plan["units"][0]["arguments"]["config_path"]
    with open(snapshot, "rb") as handle:
        original = handle.read()
    try:
        with open(snapshot, "ab") as handle:
            handle.write(b"\n# injected post-lock change\n")
        with pytest.raises(ValueError, match="Locked benchmark input changed"):
            orchestration.run_unit(job, device="cpu")
    finally:
        with open(snapshot, "wb") as handle:
            handle.write(original)
    assert len(calls) == before
