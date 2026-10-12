"""T13 policy targets, finite-MC uncertainty, cost caps and shard resume."""

from __future__ import annotations

import os
import sqlite3
import time
from types import SimpleNamespace

import pytest
import torch

import src.tasks.evaluate_value as task
from src.config import load_config
from src.data import molecule_id
from src.reference import q_z_reference
from src.schemas import GROUPS
from src.tracking import BudgetExceeded, CostTracker
from test_values import CountedModel, OracleChecker, tiny_tokenizer  # noqa: F401


def evaluation_config(*updates):
    return load_config("configs/value_eval.yaml", overrides=[
        "generation.model.checkpoint_manifest=checkpoints/mock/mdlm.json",
        "value.learned.checkpoint_manifest=checkpoints/mock/verifier.json", *updates], validator=task.validate_value_eval)


@pytest.mark.parametrize("override", ["value_eval.split=validation", "value_eval.purpose=tuning",
                                      "value_eval.cars.cache_capacity=1", "value_eval.exclude_verifier_states=false",
                                      "value_eval.reference.batch_size=0",
                                      "value_eval.provider_budgets=[{name: reference, max_model_nfe: 10, max_compute_seconds: 1}]"])
def test_evaluation_rejects_leakage_or_unmeasured_settings(override):
    with pytest.raises((ValueError, TypeError)):
        evaluation_config(override)


def test_tuning_and_final_state_ids_have_a_seed_independent_partition(tiny_tokenizer, toy_initial_state):
    options = evaluation_config()["value_eval"]
    identity = task.state_identity(toy_initial_state, tiny_tokenizer)
    chosen = task.state_purpose(identity, options)
    assert chosen in ("tuning", "final")
    # Purpose, corpus split, and run RNG seed cannot move an identical input.
    for purpose in ("tuning", "final"):
        for seed in (0, 999):
            config = evaluation_config("seed=" + str(seed), "value_eval.purpose=" + purpose,
                                       "value_eval.split=" + ("validation" if purpose == "tuning" else "test"))
            assert task.state_purpose(identity, config["value_eval"]) == chosen
    assert task.state_identity(toy_initial_state.with_tokens({0: 3}), tiny_tokenizer) != identity


def test_state_builder_excludes_training_identities_and_exact_snapshots(artifact_store, tiny_tokenizer, monkeypatch):
    config = evaluation_config("value_eval.molecules_per_group=2", "value_eval.mask_fractions=[0.0,1.0]")
    index_path = "data/fixture/partitions.sqlite"
    os.makedirs(os.path.dirname(artifact_store.path(index_path)), exist_ok=True)
    connection = sqlite3.connect(artifact_store.path(index_path))
    connection.execute("CREATE TABLE identities (identity_key TEXT PRIMARY KEY)")
    connection.execute("INSERT INTO identities VALUES (?)", ("molecule:" + molecule_id("CO"),))
    connection.commit()
    connection.close()
    verifier = SimpleNamespace(metadata={"rollout_manifest": "data/fixture/rollouts.json"})
    rollouts = SimpleNamespace(metadata={"partition": {"path": index_path}})
    monkeypatch.setattr(artifact_store, "require_manifest", lambda path, **kwargs: verifier if path.endswith("verifier.json") else rollouts)
    monkeypatch.setattr(task, "verifier_source", lambda *args, **kwargs: ("checkpoints/mock/verifier.json", {}))
    monkeypatch.setattr(task, "state_purpose", lambda identity, options: "final")
    tokens = [[3, 3, 2, 0], [3, 4, 2, 0], [4, 4, 2, 0], [3, 3, 4, 2]]

    class Sources:
        shards = [{"path": "data/fixture/source.pt"}]

        def __init__(self, *args, **kwargs):
            pass

        def load_shard(self, number):
            texts = ["CC", "CO", "OO", "CCO"]
            return {"token_ids": torch.tensor(tokens), "canonical_smiles": texts,
                "molecule_ids": [molecule_id(text) for text in texts], "augmentation_indices": torch.zeros(4),
                "group_mask": torch.ones((4, 3), dtype=torch.bool)}

    class Snapshots(Sources):
        def load_shard(self, number):
            return {"state_ids": torch.full((1, 4), tiny_tokenizer.special_tokens.mask_id), "group_ids": torch.tensor([0])}

    monkeypatch.setattr(task, "TokenShards", Sources)
    monkeypatch.setattr(task, "SnapshotShards", Snapshots)
    context = SimpleNamespace(store=artifact_store, config=config,
        identity=SimpleNamespace(dependency_hashes={"mdlm": "a"*64}), progress=lambda iterable, **kwargs: iterable)
    pool = task.build_states(context, tiny_tokenizer)
    assert pool["selection_counts"]["excluded_verifier_identities"] == 1
    assert pool["selection_counts"]["excluded_verifier_states"] == 1
    assert all(origin["molecule_id"] != molecule_id("CO") for row in pool["states"] for origin in row["origins"])
    assert not any(row["group"] == GROUPS[0] and row["mask_fraction"] == 1. for row in pool["states"])
    assert any(row["group"] == GROUPS[1] and row["mask_fraction"] == 1. for row in pool["states"])


def test_batched_q_reference_counts_calls_and_examples_separately(toy_model, tiny_tokenizer, toy_initial_state):
    tracker = CostTracker(run_id="batched-reference")
    result = q_z_reference(CountedModel(toy_model, tracker), tiny_tokenizer, OracleChecker(tracker=tracker),
                           toy_initial_state, m=7, batch_size=3, seed=15, tracker=tracker)
    assert result.completions == len(result.rewards) == 7
    assert tracker.raw_nfe == 6  # Three batches, two unresolved coordinates.
    assert tracker.model_example_nfe == 14
    assert result.standard_error is not None and result.target == "V_q"


def test_query_cost_commits_once_and_global_caps_include_inner_calls():
    parent = CostTracker(run_id="value-cost", max_model_nfe=2)
    query = task.QueryTracker(parent, {"max_model_nfe": 10, "max_compute_seconds": 60}, job_id="query")
    with query.model_call("mdlm"):
        pass
    with query.model_call("verifier"):
        pass
    with query.model_call("mdlm", inner=True):
        pass
    with pytest.raises(BudgetExceeded):
        with query.model_call("mdlm", inner=True):
            pytest.fail("A forward was entered after the run's global cap")
    assert query.global_exhausted
    query.commit_cost()
    assert parent.raw_nfe == 2
    assert parent.counters["inner_mdlm_forward_calls"] == 1
    assert parent.counters["verifier_forward_calls"] == 1
    with pytest.raises(ValueError, match="already"):
        query.commit_cost()


def test_query_wall_deadline_includes_cpu_sampling_overhead():
    query = task.QueryTracker(CostTracker(run_id="deadline"),
                              {"max_model_nfe": 100, "max_compute_seconds": 1}, job_id="query")
    query._started = time.perf_counter()-2
    with pytest.raises(BudgetExceeded, match="query_compute_seconds"):
        query.require_budget()
    assert query.raw_nfe == 0 and not query.global_exhausted


def metric_row(state_id, kind, value, *, status="completed", repeat=0):
    return {"state_id": state_id, "kind": kind, "value": value, "status": status, "budget": "primary", "repeat": repeat,
            "group": GROUPS[0], "terminal": False, "mask_fraction": .5,
            "estimate": {"standard_error": .1, "confidence_interval": [max(0., (value or 0.)-.2), min(1., (value or 0.)+.2)]},
            "cost": CostTracker(run_id="fixture").summary()}


def test_accuracy_retains_reference_noise_and_missing_query_denominators():
    options = evaluation_config()["value_eval"]
    predictions = [metric_row("one", "cars", .4), metric_row("two", "cars", .8),
                   metric_row("three", "cars", None, status="budget_exhausted")]
    references = {key: metric_row(key, "reference_q", value) for key, value in (("one", .2), ("two", .6), ("three", .5))}
    report = task.accuracy_block(predictions, references, options)
    assert report["attempted_queries"] == 3 and report["paired_queries"] == 2
    assert report["mse_to_mc_reference"] == pytest.approx(.04)
    assert report["mae_to_mc_reference"] == pytest.approx(.2)
    assert report["mean_reference_variance"] == pytest.approx(.01)
    assert report["mse_minus_reference_variance"] == pytest.approx(.03)
    assert report["imprecise_reference_queries"] == 2
    assert report["cost_including_failed_queries"]["budget_exhausted"] == 1
    assert report["ece_to_mc_reference"] == pytest.approx(.2)


def test_summary_labels_cross_policy_predictions_in_the_q_comparison():
    options = evaluation_config()["value_eval"]
    rows = [metric_row("one", "reference_q", .2), metric_row("one", "reference_rollout", .8),
            metric_row("one", "learned", .7), metric_row("one", "cars", .2)]
    report = task.summarize(rows, [{"state_id": "one"}], options,
                            {"learned_q_aligned": False, "learned_rollout_policy_matches": True})
    q_learned = next(row for row in report["results"] if row["provider"] == "learned" and row["reference_policy"] == "reference_q")
    assert q_learned["interpretation"] == "cross_policy_diagnostic"
    assert not report["paired_q_comparisons"][0]["learned_training_target_aligned"]
    assert not report["paired_q_comparisons"][0]["equal_actual_compute_claim"]
    assert not report["ideal_value_claim"] and not report["calibration_fitted"]


def test_merge_rejects_reordered_or_changed_shards(artifact_store):
    jobs = [{"job_id": "first", "state_id": "one", "kind": "cars", "budget": "primary", "repeat": 0},
            {"job_id": "second", "state_id": "two", "kind": "cars", "budget": "primary", "repeat": 0}]
    rows = [{**metric_row(job["state_id"], "cars", .3), **job} for job in jobs]
    path = "data/value_eval/fixture/shard_00000.parquet"
    task._write_results(artifact_store, path, rows)
    shard = {"path": path, "sha256": task.sha256_file(artifact_store.require_file(path)), "rows": 2}
    assert task.merge_shards(artifact_store, [shard], jobs) == rows
    with pytest.raises(ValueError, match="reordered"):
        task.merge_shards(artifact_store, [shard], list(reversed(jobs)))
    with pytest.raises(ValueError, match="checksum"):
        task.merge_shards(artifact_store, [{**shard, "sha256": "0"*64}], jobs)


@pytest.mark.parametrize("interrupt_inside_query", [False, True])
def test_resume_reuses_fixed_states_and_does_not_duplicate_cost_or_shards(artifact_store, tiny_tokenizer,
                                                                       toy_initial_state, monkeypatch, interrupt_inside_query):
    monkeypatch.setattr(task, "evaluation_dependencies", lambda config, store: {})
    monkeypatch.setattr(task, "state_purpose", lambda identity, options: "final")
    actual = {"device": "cpu", "learned_q_aligned": True, "learned_rollout_policy_matches": False}
    monkeypatch.setattr(task, "build_runtime", lambda context: {"tokenizer": tiny_tokenizer, "actual": actual})
    built = []
    interrupted_queries = []

    def pool(context, tokenizer):
        built.append(context.identity.run_id)
        identity = task.state_identity(toy_initial_state, tokenizer)
        return {"version": task.VALUE_EVAL_VERSION, "purpose": "final", "split": "test", "tokenizer": tokenizer.fingerprint,
            "dependency_hashes": {}, "selection_counts": {}, "molecules_per_group": {group: 1 for group in GROUPS},
            "states_per_group": {GROUPS[0]: 1}, "states": [{"state_id": identity, "state": toy_initial_state.to_dict(),
                "group": GROUPS[0], "mask_fraction": .5, "terminal": False}]}

    def query(context, runtime, item, job):
        tracker = task.QueryTracker(context.tracker, {"max_model_nfe": 10, "max_compute_seconds": 60}, job_id=job["job_id"])
        with tracker.model_call("verifier" if job["kind"] == "learned" else "mdlm", inner=job["kind"] == "cars"):
            pass
        row = {**metric_row(item["state_id"], job["kind"], .4), **job,
               "global_exhausted": False, "reason": None, "cost": tracker.summary()}
        tracker.commit_cost()
        if interrupt_inside_query and not interrupted_queries:
            interrupted_queries.append(job["job_id"])
            raise KeyboardInterrupt
        return row

    monkeypatch.setattr(task, "build_states", pool)
    monkeypatch.setattr(task, "measure_query", query)
    args = ["--smoke", "--checkpoint", "checkpoints/mock/mdlm.json", "--verifier-checkpoint", "checkpoints/mock/verifier.json",
            "--artifact-root", artifact_store.root, "--set", "value_eval.shard_rows=2"]
    commits = []

    def interrupt():
        commits.append(True)
        if len(commits) == 3:  # Two queries; one committed parquet shard.
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        task.execute([*args, "--run-id", "resumed"], on_checkpoint=None if interrupt_inside_query else interrupt)
    resumed = task.execute([*args, "--run-id", "resumed", "--resume"])
    reference = task.execute([*args, "--run-id", "reference"])
    assert resumed.status == reference.status == "completed"
    assert built == ["resumed", "reference"]
    if interrupt_inside_query:
        assert resumed.tracking.counters["mdlm_forward_calls"] == reference.tracking.counters["mdlm_forward_calls"]+1
    else:
        assert resumed.tracking.counters == reference.tracking.counters
    left, right = [artifact_store.read_json(artifact_store.run_directory(record.identity)+"/metrics.json") for record in (resumed, reference)]
    left_rows, right_rows = [task.read_parquet(artifact_store, report["values_path"]) for report in (left, right)]
    assert [(row["job_id"], row["value"]) for row in left_rows] == [(row["job_id"], row["value"]) for row in right_rows]
    assert len({row["job_id"] for row in left_rows}) == left["expected_queries"] == 4
    if interrupt_inside_query:
        assert left["recovery_costs"][0]["raw_nfe"] == 1
    manifest_path = "data/value_eval/smoke/final/seed_0/resumed/manifest.json"
    artifact_store.require_manifest(manifest_path, kind="value_evaluation")
