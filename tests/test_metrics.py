"""Denominators, molecular identity, proxy semantics and checked reporting."""

from __future__ import annotations

import copy
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
from rdkit import DataStructs

from src.artifacts import sha256_file
from src.config import config_digest, load_config
from src.data import InvalidSmiles, canonicalize_smiles
from src.metrics import cost_metrics, fingerprint_diversity, generation_metrics
from src.reporting import paired_comparisons, quality_frontiers, seed_interval, summarize_generation, value_rows
from src.retrosynthesis import RetroStarProxy, load_retrostar
from src.schemas import AttemptRecord, RewardResult, RunIdentity, SampleRecord
from src.tasks import evaluate as evaluation_task
from src.tasks import report as report_task
from src.tasks.report import checked_evaluation
from src.tracking import CostTracker


def result(text):
    try:
        canonical = canonicalize_smiles(text)
    except InvalidSmiles:
        return RewardResult(0.0, "grammar-rdkit-smarts-v1", False, False, False)
    return RewardResult(1.0, "grammar-rdkit-smarts-v1", True, True, True, canonical)


def fixture_attempts():
    rows = [AttemptRecord("fixture", 1, "cars", "chain_extenders", "prefix_abort").to_dict()]
    for number, text in enumerate(("CO", "OC", "not_a_smiles", "N"), 2):
        rows.append(AttemptRecord("fixture", number, "cars", "chain_extenders", "complete",
            leaf_visits=1, smiles=text, reward_result=result(text)).to_dict())
    selected = SampleRecord("fixture", 1, 5, "chain_extenders", "N", "N", result("N")).to_dict()
    return rows, [selected]


def test_prefix_abort_duplicate_invalid_and_unattained_denominators():
    attempts, samples = fixture_attempts()
    tracker = CostTracker(run_id="fixture")
    tracker.increment("root_attempts", 5)
    metrics = generation_metrics(attempts, samples, [], tracking=tracker.summary(),
        protocol={"target_unique": 2}, prompts={"CO"},
        fingerprint={"radius": 2, "bits": 2048, "sensitivity_radii": [3]}, retro=RetroStarProxy(None))
    assert metrics["success_per_root"] == {"numerator": 3, "denominator": 5, "value": 0.6, "status": "available"}
    output = metrics["sets"]["outputs"]
    assert output["validity"]["value"] == 3/4
    assert output["membership"]["denominator"] == 4
    assert output["unique_fraction"]["value"] == 2/3  # OC and CO are one molecule.
    assert output["novelty"]["value"] == 1/3
    assert output["novel_unique_fraction"]["value"] == 1/2
    assert metrics["attained"] == 1 and metrics["missing_to_target"] == 1
    assert metrics["attempts_to_target"] is None
    assert metrics["tracked_seconds_to_target_upper_bound"] is None
    assert metrics["sets"]["selected"]["validity"]["value"] == 1
    assert metrics["sets"]["search_candidates"]["validity"]["value"] is None
    assert output["retro_score"]["value"] is None


def test_unrecorded_root_remains_in_denominator():
    attempts, samples = fixture_attempts()
    tracker = CostTracker(run_id="fixture")
    tracker.increment("root_attempts", 6)
    metrics = generation_metrics(attempts, samples, [], tracking=tracker.summary(),
        protocol={"target_unique": 2}, prompts=set(), fingerprint={}, retro=RetroStarProxy(None))
    assert metrics["success_per_root"]["value"] == 0.5
    assert metrics["unrecorded_root_attempts"] == 1 and metrics["success_is_lower_bound"]


def test_pairwise_tanimoto_known_bits_and_duplicate_weighting():
    left, right = DataStructs.ExplicitBitVect(8), DataStructs.ExplicitBitVect(8)
    left.SetBitsFromList([0, 1])
    right.SetBitsFromList([1, 2])
    assert fingerprint_diversity([left, right])["value"] == pytest.approx(2/3)
    assert fingerprint_diversity([left, right, left])["value"] == pytest.approx(4/9)
    assert fingerprint_diversity([left])["value"] is None


def test_inner_completions_and_gpu_time_are_not_added_twice():
    tracker = CostTracker(run_id="fixture")
    for name, count in {"mdlm_forward_calls": 10, "ar_forward_calls": 2,
                        "inner_mdlm_forward_calls": 7, "inner_ar_forward_calls": 1,
                        "verifier_forward_calls": 9, "inner_complete_candidates": 6}.items():
        tracker.increment(name, count)
    tracker.compute_seconds.update(model=2.0, selection=1.0)
    tracker.gpu_seconds["model"] = 99.0
    metrics = cost_metrics(tracker.summary(), reference_forward_flops=10)
    assert metrics["raw_nfe"] == 12 and metrics["inner_model_calls"] == 8
    assert metrics["outer_model_calls"] == 4 and metrics["verifier_forward_calls"] == 9
    assert metrics["tracked_compute_seconds"] == 3
    assert metrics["adjusted_nfe"] is None
    tracking = tracker.summary()
    tracking["counters"]["flop_profiled_mdlm_calls"] = 10
    tracking["counters"]["flop_profiled_ar_calls"] = 2
    tracking["counters"]["flop_profiled_verifier_calls"] = 9
    tracking["flops"] = {"mdlm": 100, "ar": 20, "verifier": 9}
    assert cost_metrics(tracking, reference_forward_flops=10)["adjusted_nfe"] == 12.9


def test_retro_strict_threshold_and_original_all_output_denominator():
    calls = []

    def predict(texts):
        calls.append(texts)
        return [{"CO": 0.4, "CC": 0.40001, "O": None}[text] for text in texts]

    proxy = RetroStarProxy(predict, batch_size=2)
    score = proxy.score(["CO", "CC", "O", None])
    assert score["numerator"] == 1 and score["denominator"] == 4
    assert score["value"] == 0.25 and score["valid_only_value"] == 1/3
    assert score["no_reaction_outputs"] == 1
    assert proxy.score(["CC", "CC"])["value"] == 1
    assert calls == [["CO", "CC"], ["O"]]
    assert RetroStarProxy(None).score([None])["value"] is None
    assert proxy.score([None])["value"] == 0


def test_retro_missing_model_is_unavailable_without_download(artifact_store):
    config = load_config("configs/evaluation.yaml", overrides=["evaluation.source_run=outputs/missing"])
    proxy = load_retrostar(artifact_store, config["models"]["retrostar"], config["evaluation"]["retrosynthesis"])
    score = proxy.score(["CO"])
    assert score["status"] == "unavailable" and score["value"] is None
    assert score["reason"] == "missing_T04_manifest"


def test_retro_malformed_inference_is_not_scored_as_zero():
    assert RetroStarProxy(lambda texts: []).score(["CO"])["status"] == "unavailable"
    assert RetroStarProxy(lambda texts: [float("nan")]).score(["CO"])["value"] is None


def test_value_report_reads_actual_T13_group_mask_and_calibration_contract():
    from src.tasks.evaluate_value import summarize

    config = load_config("configs/value_eval.yaml")
    evaluation = summarize([], [], config["value_eval"],
        {"learned_q_aligned": False, "learned_rollout_policy_matches": False})
    payload = {"identity": {"suite": "primary", "seed": 0, "run_id": "fixture", "config_hash": "a"*64,
                           "dependency_hashes": {"mdlm": "b"*64}},
               "status": "completed", "purpose": "final", "evaluation": evaluation}
    rows = value_rows(payload, "outputs/fixture/metrics.json")
    assert {row["subset"] for row in rows} == {"overall", "terminal", "nonterminal", "group", "mask_fraction"}
    assert any(row["subset_label"] == "0.0:0.25" for row in rows)
    assert all(row["mse_to_mc_reference"] is None and row["paired_queries"] == 0 for row in rows)
    assert all("reliability_bins_json" in row for row in rows)


def report_row(method, seed, value, *, checkpoint="same", suite="primary", cost=10, budget="cap"):
    return {"suite": suite, "comparison_key": checkpoint, "group": "acrylates", "method": method,
            "variant_key": method, "budget_key": budget, "seed": seed, "attained": value,
            "raw_nfe": cost, "tracked_compute_seconds": cost/10, "target_attained": False,
            "source_status": "timeout", "validity": None}


BOOTSTRAP = {"repetitions": 200, "confidence": 0.95, "seed": 0}


def test_paired_seed_differences_keep_timeouts_and_separate_backbones_suites():
    rows = [report_row("cars", 0, 2), report_row("cars", 1, 3), report_row("hybrid", 0, 4),
            report_row("hybrid", 1, 6), report_row("hybrid", 2, 9),
            report_row("mdm_vgb", 0, 100, checkpoint="different"), report_row("mdm_vgb", 0, 100, suite="smoke")]
    comparisons = paired_comparisons(rows, ["attained"], BOOTSTRAP)
    assert len(comparisons) == 1
    assert comparisons[0]["mean"] == 2.5 and comparisons[0]["n"] == 2
    assert comparisons[0]["right_only_seeds"] == 1
    aggregate = summarize_generation(rows, ["attained", "validity"], BOOTSTRAP)
    cars = next(row for row in aggregate if row["method"] == "cars" and row["metric"] == "attained")
    assert cars["noncompleted_runs"] == 2 and cars["unattained_runs"] == 2
    assert next(row for row in aggregate if row["metric"] == "validity")["n"] == 0
    assert seed_interval([1], **BOOTSTRAP)["ci_low"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_generation([rows[0], rows[0]], ["attained"], BOOTSTRAP)


def test_frontier_uses_only_observed_points_within_backbone():
    rows = [report_row("cars", 0, 5, cost=10, budget="a"), report_row("cars", 0, 4, cost=20, budget="b"),
            report_row("cars", 0, 100, checkpoint="another", cost=1, budget="c")]
    points = quality_frontiers(summarize_generation(rows, ["attained", "raw_nfe", "tracked_compute_seconds"], BOOTSTRAP))
    raw = [point for point in points if point["cost_metric"] == "raw_nfe"]
    assert len(raw) == 3
    assert next(point for point in raw if point["budget_key"] == "a")["pareto"]
    assert not next(point for point in raw if point["budget_key"] == "b")["pareto"]
    assert all(point["observed_only"] for point in points)


def test_journal_validation_does_not_rewrite_inputs(artifact_store):
    attempts, samples = fixture_attempts()
    identity = RunIdentity("T09", "fixture", "a"*64, "primary", 0, "grammar-rdkit-smarts-v1", "cars", "chain_extenders")
    directory = artifact_store.run_directory(identity)
    paths = [os.path.join(directory, filename) for filename in ("attempts.jsonl", "samples.jsonl")]
    for path, rows in zip(paths, (attempts, samples), strict=True):
        artifact_store.write_jsonl(path, rows)
    hashes = [sha256_file(artifact_store.require_file(path)) for path in paths]
    checker = SimpleNamespace(evaluate_smiles=lambda text, group: result(text))
    assert evaluation_task._checked_journals(artifact_store, directory, identity, checker) == (attempts, samples)
    assert [sha256_file(artifact_store.require_file(path)) for path in paths] == hashes
    bad = copy.deepcopy(samples)
    bad[0]["attempt_id"] = 2
    artifact_store.write_jsonl(paths[1], bad)
    with pytest.raises(ValueError, match="recorded successful"):
        evaluation_task._checked_journals(artifact_store, directory, identity, checker)


def test_missing_generation_produces_explicit_metrics_artifact(artifact_store):
    config = load_config("configs/evaluation.yaml", overrides=["evaluation.source_run=outputs/missing"])
    identity = RunIdentity("T14", "fixture", config_digest(config), "primary", 0, config["reward"]["version"])
    directory = artifact_store.run_directory(identity)
    artifact_store.write_config(os.path.join(directory, "resolved_config.yaml"), config)
    context = SimpleNamespace(store=artifact_store, config=config, identity=identity,
        output_path=lambda filename: os.path.join(directory, filename), log=lambda *args, **kwargs: None)
    completed = evaluation_task.evaluate(context)
    assert completed.status == "failed"
    payload = artifact_store.read_json(os.path.join(directory, "metrics.json"))
    assert payload["status"] == "missing" and "FileNotFoundError" in payload["reason"]
    artifact_store.require_manifest(os.path.join(directory, "metrics_manifest.json"), kind="molecular_evaluation")


def test_report_rejects_tampered_metrics_and_configuration(artifact_store):
    config = load_config("configs/common.yaml")
    identity = RunIdentity("T14", "fixture", config_digest(config), "primary", 0, config["reward"]["version"])
    record = artifact_store.create_run(identity, config)
    artifact_store.write_run(replace(record, status="failed"))
    directory = artifact_store.run_directory(identity)
    metrics_path, manifest_path = (os.path.join(directory, name) for name in ("metrics.json", "metrics_manifest.json"))
    payload = {"version": "smiles-metrics-v1", "status": "missing", "source_run": "outputs/missing",
               "evaluation_identity": identity.to_dict(), "reason": "missing source"}
    artifact_store.write_json(metrics_path, payload)
    artifact_store.create_manifest(manifest_path, kind="molecular_evaluation", config_hash=identity.config_hash,
        files=[metrics_path], metadata={"suite": "primary", "status": "missing", "source_run": "outputs/missing", "metrics_path": metrics_path})
    assert checked_evaluation(artifact_store, manifest_path, kind="molecular_evaluation", suite="primary")[0] == payload
    artifact_store.write_json(metrics_path, {**payload, "reason": "tampered"})
    with pytest.raises(ValueError, match="checksum"):
        checked_evaluation(artifact_store, manifest_path, kind="molecular_evaluation", suite="primary")
    artifact_store.write_json(metrics_path, payload)
    artifact_store.write_config(os.path.join(directory, "resolved_config.yaml"), {**config, "seed": 1})
    with pytest.raises(ValueError, match="configuration"):
        checked_evaluation(artifact_store, manifest_path, kind="molecular_evaluation", suite="primary")


def test_report_exports_audit_and_empty_plots_for_missing_runs(artifact_store):
    config = load_config("configs/report.yaml", overrides=["report.expected_generation_runs=[outputs/missing]"],
                         validator=report_task.validate_report)
    identity = RunIdentity("T16", "fixture", config_digest(config), "primary", 0, config["reward"]["version"])
    context = SimpleNamespace(store=artifact_store, config=config, identity=identity, resume_state=None,
        save_checkpoint=lambda state: None, progress=lambda rows, **kwargs: rows, log=lambda *args, **kwargs: None)
    completed = report_task.report(context)
    assert completed.status == "failed"
    manifest = artifact_store.require_manifest("reports/primary/fixture/manifest.json", kind="report")
    assert manifest.metadata["generation_runs"] == 0
    provenance = artifact_store.read_json("reports/primary/fixture/sources.json")
    assert provenance["audits"][0]["status"] == "missing"
    for filename in ("quality_cost.png", "value_accuracy.png"):
        with open(artifact_store.require_file("reports/primary/fixture/"+filename), "rb") as handle:
            assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
