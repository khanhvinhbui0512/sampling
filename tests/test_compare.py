"""Matched jobs, independent streams, incomplete runs and comparison resume.

These finite orchestration checks never load or download a trained model.
"""

from __future__ import annotations

import copy
import os
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.artifacts import Dependency, sha256_file
from src.config import config_digest, load_config
from src.schemas import GROUPS, RunIdentity
from src.tasks import compare as task
from src.tracking import CostTracker


def comparison_config(*overrides):
    return load_config("configs/comparison.yaml", overrides=["comparison.purpose=tuning", *overrides], validator=task.validate_comparison)


def bundle(config):
    return {"models": {"mdlm": "checkpoints/shared/manifest.json", "verifier": "checkpoints/verifier/manifest.json"},
        "variants": task.variants(config), "project_hashes": task.project_hashes(), "design": task.design(config),
        "shared_hashes": {name: str(index)*64 for index, name in enumerate(("mdlm", "verifier", "tokenizer", "vocabulary", "benchmark"), 1)}}


def context_for(config, store):
    prepared = bundle(config)
    identity = RunIdentity("T15", "matrix", config_digest(config), config["suite"], 0, config["reward"]["version"],
                           dependency_hashes=dict(prepared["shared_hashes"]))
    return SimpleNamespace(config=config, store=store, identity=identity), prepared


def test_sweep_deduplicates_shared_jobs_and_matches_kernel_caps(artifact_store):
    config = comparison_config("comparison.validation_sweep.N=[1,2]", "comparison.validation_sweep.m=[4,8]")
    context, prepared = context_for(config, artifact_store)
    jobs = task.build_jobs(context, prepared)
    cells = 3 * 3 * 4  # groups, seeds, allowances
    assert Counter(job["method"] for job in jobs) == {
        "base": cells, "cars": cells, "mdm_vgb": cells*2, "hybrid": cells*4}
    assert {job["identity"]["group"] for job in jobs} == set(GROUPS)
    assert {job["identity"]["seed"] for job in jobs} == {2**31, 2**31+1, 2**31+2}
    for job in jobs:
        assert job["identity"]["dependency_hashes"]["mdlm"] == prepared["shared_hashes"]["mdlm"]
        if job["method"] in ("cars", "base"):
            assert len(job["variants"]) == 4
    for variant in prepared["variants"]:
        cell = [job for job in jobs if variant["id"] in job["variants"]
                and job["logical_seed"] == 0 and job["identity"]["group"] == GROUPS[0] and job["budget"] == "nfe_16384"]
        kernels = [job["search"] for job in cell if job["method"] in ("mdm_vgb", "hybrid")]
        assert kernels[0] == kernels[1]
        for job in cell:
            assert 'protocol.max_model_nfe=16384' in job["argv"]
            assert 'protocol.max_compute_seconds=null' in job["argv"]


def test_final_requires_lock_and_smoke_has_separate_stream(artifact_store):
    with pytest.raises(ValueError, match="requires a reviewed"):
        load_config("configs/comparison.yaml", validator=task.validate_comparison)
    config = comparison_config("suite=smoke", "comparison.purpose=final")
    context, prepared = context_for(config, artifact_store)
    jobs = task.build_jobs(context, prepared)
    assert len(jobs) == 3*1*2*4
    assert all(job["identity"]["suite"] == "smoke" and job["identity"]["seed"] == 0 for job in jobs)
    assert all("outputs/smoke/" in job["directory"] for job in jobs)


@pytest.mark.parametrize("override", ["seeds=[0,1]", "comparison.methods=[cars,hybrid]",
    "comparison.budgets=[{name: unbounded, max_model_nfe: null, max_compute_seconds: null}]",
    "protocol.max_model_nfe=100"])
def test_invalid_matrices_fail_preflight(override):
    with pytest.raises(ValueError):
        comparison_config(override)


def test_replication_is_only_ar(artifact_store):
    config = comparison_config("suite=replication")
    context, prepared = context_for(config, artifact_store)
    prepared["models"] = {"mdlm": None, "verifier": None}
    prepared["shared_hashes"] = {"llama": "a"*64, "benchmark": "b"*64}
    context.identity = replace(context.identity, dependency_hashes=prepared["shared_hashes"])
    jobs = task.build_jobs(context, prepared)
    assert len(jobs) == 3*3*4
    assert {job["method"] for job in jobs} == {"cars_ar"}
    assert all(job["identity"]["reward_version"] == "grammar-only-v1" for job in jobs)


def test_changed_shared_checkpoint_is_rejected(artifact_store):
    context, prepared = context_for(comparison_config("suite=smoke"), artifact_store)
    prepared["shared_hashes"]["mdlm"] = "f"*64
    with pytest.raises(ValueError, match="different shared"):
        task.build_jobs(context, prepared)


def test_final_rejects_changed_checkpoint_contents_at_the_same_path(artifact_store, monkeypatch):
    config = load_config("configs/comparison.yaml", overrides=["comparison.lock_manifest=selection_manifest.json"],
                         validator=task.validate_comparison)
    paths = {name: f"checkpoints/fixture/{name}.json" for name in ("mdlm", "verifier", "tokenizer", "vocabulary", "benchmark")}
    for name, path in paths.items():
        artifact_store.write_json(path, {"fixture": name})
    expected = {name: sha256_file(artifact_store.require_file(path)) for name, path in paths.items()}
    expected["retrostar"] = None
    expected["mdlm"] = "0"*64  # The selected manifest previously had different contents.
    selection = {"design": task.design(config), "project_hashes": task.project_hashes(), "shared_hashes": expected,
                 "models": {"mdlm": paths["mdlm"], "verifier": paths["verifier"]}}
    monkeypatch.setattr(task, "checked_selection", lambda *args: selection)
    model_deps = {name: Dependency(path) for name, path in paths.items() if name != "verifier"}
    monkeypatch.setattr(task.generation, "mdlm_source", lambda *args: (paths["mdlm"], model_deps))
    monkeypatch.setattr(task.generation, "generation_dependencies", lambda *args: model_deps)
    monkeypatch.setattr(task.generation, "verifier_source", lambda *args, **kwargs: (paths["verifier"], {"verifier": Dependency(paths["verifier"])}))
    # The outer selection dependency itself is present; its checking is stubbed above.
    artifact_store.create_manifest("selection_manifest.json", kind="comparison_selection", config_hash=config_digest(config), files=[paths["mdlm"]])
    with pytest.raises(ValueError, match="contents changed"):
        task.prepare(config, artifact_store)


def test_timeout_with_fewer_samples_is_terminal(artifact_store):
    context, prepared = context_for(comparison_config("suite=smoke"), artifact_store)
    spec = task.build_jobs(context, prepared)[0]
    identity = RunIdentity.from_dict(spec["identity"])
    record = artifact_store.create_run(identity, {})
    artifact_store.write_run(replace(record, status="timeout"))
    report = os.path.join(spec["directory"], "generation.json")
    artifact_store.write_json(report, {"identity": spec["identity"], "status": "timeout", "selected_samples": 1,
                                      "target_attained": False, "stop_reason": "max_root_attempts"})
    artifact_store.create_manifest(spec["artifact"], kind="generation", config_hash=identity.config_hash, files=[report],
        metadata={"suite": identity.suite, "status": "timeout", "dependency_hashes": identity.dependency_hashes})
    state = task.observed(artifact_store, spec)
    assert state["done"] and state["attained"] == 1 and not state["target_attained"]


def test_resume_keeps_paid_inner_calls_without_changing_journal_ids():
    tracker = CostTracker(run_id="resume")
    tracker.increment("root_attempts", 1)
    tracker.increment("mdlm_forward_calls", 2)
    tracker.increment("inner_mdlm_forward_calls", 1)
    current = tracker.snapshot()
    old_counts = dict(current.counters)
    old_counts.update(root_attempts=2, mdlm_forward_calls=7, inner_mdlm_forward_calls=5)
    previous = replace(current, counters=old_counts, compute_seconds={"mdlm": 3.0})
    context = SimpleNamespace(resumed=True, tracker=tracker, record=SimpleNamespace(tracking=previous))
    task._recover_paid_work(context)
    assert tracker.raw_nfe == 7 and tracker.counters["inner_mdlm_forward_calls"] == 5
    assert tracker.counters["root_attempts"] == 1  # Replay preserves committed attempt IDs.
    assert tracker.tracked_compute_seconds == 3.0


def test_missing_dependencies_dispatch_nothing(artifact_store, monkeypatch):
    calls = []

    def missing(config, store):
        raise FileNotFoundError("shared checkpoint missing")

    monkeypatch.setattr(task, "prepare", missing)
    with pytest.raises(FileNotFoundError, match="checkpoint missing"):
        task.execute(["--smoke", "--artifact-root", artifact_store.root, "--run-id", "preflight"],
                     dispatch=lambda *args: calls.append(args))
    assert calls == []


def test_review_selection_rejects_final_metrics(monkeypatch):
    job = {"id": "job", "variants": ["N1_m4"], "directory": "outputs/validation", "identity": {"suite": "primary"}}
    plan = {"variants": [{"id": "N1_m4"}], "jobs": [job], "identity": {"suite": "primary"}}
    results = {"jobs": {"job": {"generation": {"done": True, "status": "timeout"},
                               "evaluate": {"done": True, "artifact": "metrics.json"}}}, "reports": {}}
    monkeypatch.setattr(task, "observed", lambda *args: {"done": True, "status": "timeout"})
    monkeypatch.setattr(task.report_task, "checked_evaluation", lambda *args, **kwargs: ({
        "source_run": job["directory"], "identity": job["identity"], "comparability": {"experiment": {"purpose": "final"}}}, "metrics.json"))
    with pytest.raises(ValueError, match="validation job"):
        task._selected_metrics(None, plan, results, "N1_m4")


def test_coordinator_resume_skips_stopped_generation_and_retries_missing_evaluation(artifact_store, monkeypatch):
    states, calls, fail_evaluation = {}, [], [True]

    def prepared(config, store):
        return bundle(config), {}

    monkeypatch.setattr(task, "prepare", prepared)
    original_jobs = task.build_jobs

    def one_job(context, ready):
        job = original_jobs(context, ready)[0]
        artifact_store.write_json(job["artifact"], {"fixture": "stopped generation"})
        states[job["directory"]] = {"status": "timeout", "done": True, "artifact": job["artifact"], "attained": 1}
        return [job]

    monkeypatch.setattr(task, "build_jobs", one_job)
    monkeypatch.setattr(task, "observed", lambda store, spec: copy.deepcopy(states.get(spec["directory"], {"status": "missing", "done": False})))

    def dispatch(stage, spec):
        calls.append(stage)
        assert stage != "generation"
        if stage == "evaluate" and fail_evaluation[0]:
            fail_evaluation[0] = False
            raise RuntimeError("injected evaluation interruption")
        artifact_store.write_json(spec["artifact"], {"fixture": stage})
        states[spec["directory"]] = {"status": "completed", "done": True, "artifact": spec["artifact"]}

    arguments = ["--smoke", "--artifact-root", artifact_store.root, "--run-id", "resume-matrix"]
    first = task.execute(arguments, dispatch=dispatch)
    assert first.status == "failed"
    second = task.execute([*arguments, "--resume"], dispatch=dispatch)
    assert second.status == "completed" and calls.count("evaluate") == 2 and "generation" not in calls
    manifest = artifact_store.require_manifest(os.path.join(artifact_store.run_directory(second.identity), "comparison_manifest.json"), kind="comparison")
    results = artifact_store.read_json(manifest.metadata["results"])
    assert next(iter(results["jobs"].values()))["generation"]["attained"] == 1
