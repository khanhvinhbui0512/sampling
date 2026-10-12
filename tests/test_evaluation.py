"""Hand-checkable Step 11 regressions. Written for review; NOT executed."""

import copy
import math
import os
import tempfile
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from src.config import load_config
from src.evaluation.aggregate import (aggregate_records, evaluate, grouping, select_manifests,
                                      unusable_attempt)
from src.evaluation.metrics import (QUALITY_METRICS, cache_metrics, cost_metrics, diversity,
                                    generation_metrics, rollout_error_interval, value_metrics)
from src.io import (RunArtifacts, file_hash, read_completed_manifest, read_json,
                    write_json, write_jsonl, write_yaml)
from src.types import CostRecord, ValueEstimate, ValueTarget


TARGET = asdict(ValueTarget("target", "v_rho", "reference", "policy", "pass95:calibration", "calibration"))
DIVERSITY = {"radius": 2, "bits": 2048, "include_chirality": False, "max_molecules": 1000, "seed": 17}
UNCERTAINTY = {"confidence": 0.95, "bootstrap_replicates": 100, "seed": 23}


@pytest.fixture
def evaluation_config(artifact_directory):
    os.makedirs("workspace/results/summary", exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="step11-test-", dir="workspace/results/summary") as directory:
        config = load_config("configs/evaluation.yaml")
        config["output_dir"] = os.path.relpath(directory)
        config["uncertainty"] = UNCERTAINTY
        path = os.path.join(artifact_directory, "evaluation.yaml")
        write_yaml(path, config)
        yield path


def ledger(*, base_calls=0, base_sequences=0, verifier_calls=0, verifier_sequences=0):
    total = asdict(CostRecord(base_calls=base_calls, base_sequences=base_sequences,
                             verifier_calls=verifier_calls, verifier_sequences=verifier_sequences))
    return {"total": total, "by_phase": {"estimation": total}, "adjusted_nfe": None,
            "flop_measurement_id": None, "base_batch_sizes": [base_sequences // base_calls] if base_calls else [],
            "verifier_batch_sizes": [verifier_sequences // verifier_calls] if verifier_calls else []}


def estimate(value, *, cache_hit=False):
    return asdict(ValueEstimate(target=ValueTarget(**TARGET), status="estimated" if value is not None else "budget_exhausted",
                                value=value, draws=4 if value is not None else 0,
                                interval=(0, 1) if value is not None else None, cache_hit=cache_hit))


def test_requested_denominator_includes_invalid_failed_and_unfinished(token_rules):
    # Six requests: three valid (C,C,N), one invalid, one unfinished, one failed.
    class FakeReward:
        def __init__(self):
            self.token_rules = token_rules

        def evaluate(self, state, **kwargs):
            identity, score = {0: ("C", .8), 1: ("N", .2), 2: (None, 0)}[state.token_ids[0]]
            molecule = SimpleNamespace(canonical_smiles=identity, smiles=identity or "bad")
            return SimpleNamespace(valid=identity is not None, score=score, reward=int(score >= .5), molecule=molecule)

    samples = []
    for index, token in enumerate((0, 0, 1, 2, token_rules.mask_id, token_rules.mask_id)):
        complete = index < 4
        state = {"token_ids": [token] + [0] * (token_rules.sequence_length-1), "rules": asdict(token_rules)}
        identity, score = {0: ("C", .8), 1: ("N", .2), 2: (None, 0)}.get(token, (None, None))
        samples.append({"sample_id": str(index), "state": state, "target": TARGET,
                        "status": "completed" if complete else ("failed" if index == 5 else "budget_exhausted"),
                        "cost": asdict(CostRecord()), "smiles": (identity or "bad") if complete else None,
                        "canonical_smiles": identity, "valid": identity is not None if complete else None,
                        "score": score, "reward": int(score >= .5) if complete else None,
                        "error": None if complete else "stopped"})
    result = generation_metrics(samples, requested=6, reward=FakeReward(), target=TARGET,
                                train_smiles={"C"}, diversity_settings=DIVERSITY)
    assert result["pass95"] == pytest.approx(2/6)
    assert result["validity"] == .5
    assert result["qed_valid_mean"] == pytest.approx(.6)
    assert result["qed_requested_mean"] == pytest.approx(.3)
    assert result["unique_pass95"] == pytest.approx(1/6)
    assert result["uniqueness"] == pytest.approx(2/3)
    assert result["novelty"] == pytest.approx(1/3)
    assert result["novelty_of_unique"] == .5
    with pytest.raises(ValueError, match="every requested"):
        generation_metrics(samples[:-1], requested=6, reward=FakeReward(), target=TARGET,
                           train_smiles=set(), diversity_settings=DIVERSITY)


def test_diversity_keeps_duplicate_outputs_and_locks_subset():
    identical = diversity(["C", "C"], DIVERSITY)
    assert identical["value"] == 0
    assert identical["pairs"] == 1
    assert diversity(["C"], DIVERSITY)["value"] is None
    settings = {**DIVERSITY, "max_molecules": 2}
    first = diversity(["C", "N", "O", "CC"], settings)
    second = diversity(["C", "N", "O", "CC"], settings)
    assert first == second
    assert first["selected_outputs"] == 2 and first["pairs"] == 1


def test_cost_counts_batched_work_once_and_keeps_neural_nfe_unavailable():
    # 2 batch-4 base forwards and one batch-2 neural forward: raw NFE=2, sequence work=8.
    row = {"budget": ledger(base_calls=2, base_sequences=8, verifier_calls=1, verifier_sequences=2), "wall_seconds": 3}
    result = cost_metrics([row])
    assert result["total"]["base_calls"] == 2
    assert result["total"]["base_sequences"] == 8
    assert result["adjusted_nfe"] is None
    assert result["wall_seconds"] == 3
    row["budget"]["adjusted_nfe"] = 2.5
    with pytest.raises(ValueError, match="measured FLOP"):
        cost_metrics([row])
    row["budget"]["flop_measurement_id"] = "measured-batch4-batch2"
    assert cost_metrics([row])["adjusted_nfe"] == 2.5
    row["budget"]["by_phase"]["estimation"] = asdict(CostRecord())
    with pytest.raises(ValueError, match="phases"):
        cost_metrics([row])


def test_cache_origin_is_retained_once_without_charging_it_to_hits():
    origin = {"snapshot_id": "tree", "origin_run_id": "old-run", "origin_draws": 10,
              "origin_cost": asdict(CostRecord(base_calls=3, base_sequences=12))}
    group = {"cost": asdict(CostRecord()), "wall_seconds": .1,
             "items": [{"estimate": estimate(.5, cache_hit=True), "cache_provenance": origin}]}
    result = cache_metrics([group, group])
    assert result["queries"] == 2 and result["hits"] == 2
    assert result["current_query_cost"]["base_sequences"] == 0
    assert result["unique_snapshot_origin_cost"]["base_sequences"] == 12
    assert len(result["origins"]) == 1


def test_value_error_retains_unavailable_states_and_noisy_reference():
    rows = []
    for index, (value, ref) in enumerate(((.5, .4), (None, .8), (.3, None))):
        reference = estimate(ref)
        reference["interval"] = [.2, .6] if ref == .4 else reference["interval"]
        rows.append({"state_id": str(index), "stratum": "hole", "origin": {"draw_id": index},
                     "reference": reference, "reference_queries": [],
                     "estimators": {"cars": {"estimate": estimate(value), "queries": [], "budget": ledger()}}})
    result = value_metrics(rows, "cars")
    assert result["requested_states"] == 3 and result["paired_states"] == 1
    assert result["availability"] == pytest.approx(2/3)
    assert result["mae"] == pytest.approx(.1) and result["rmse"] == pytest.approx(.1)
    assert len(result["states"]) == 3
    assert result["states"][0]["absolute_error_reference_interval_min"] == 0
    assert result["states"][0]["absolute_error_reference_interval_max"] == pytest.approx(.3)
    assert result["full_panel_mae"] is None


def test_common_pair_errors_do_not_compare_different_refusal_subsets():
    rows = []
    for index, (mc, cars) in enumerate(((.5, .5), (.25, None))):
        rows.append({"state_id": str(index), "reference": estimate(.25), "reference_queries": [],
                     "estimators": {name: {"estimate": estimate(value), "queries": [], "budget": ledger()}
                                    for name, value in (("mc", mc), ("cars", cars))}})
    mc, cars = value_metrics(rows, "mc"), value_metrics(rows, "cars")
    assert mc["mae"] == .125 and cars["mae"] == .25
    assert mc["common_pair_mae"] == cars["common_pair_mae"] == .25
    assert mc["common_pair_state_ids"] == cars["common_pair_state_ids"] == ["0"]


def test_rollout_clusters_keep_repeated_masks_together():
    # Every mask from a rollout has the same error. Cluster bootstrap with two
    # rollouts can only yield MAE 0, .5 or 1; an IID mask bootstrap would differ.
    details = [{"cluster": cluster, "difference": error} for cluster, error in (("a", 0), ("b", 1)) for _ in range(20)]
    interval = rollout_error_interval(details, UNCERTAINTY)
    assert interval["clusters"] == 2
    assert interval["mae"] == [0, 1]
    assert rollout_error_interval(details[:20], UNCERTAINTY) is None
    assert rollout_error_interval([{"cluster": None, "difference": 0}], UNCERTAINTY) is None


def generation_record(seed, value):
    metrics = {key: value for key in QUALITY_METRICS}
    metrics.update(base_calls_per_output=4, base_sequences_per_output=8, verifier_sequences_per_output=0, wall_seconds_per_output=2,
                   adjusted_nfe_per_output=None, base_sequences_per_success=8/value if value else None)
    return {"config_id": "config", "family_id": "family", "kind": "generation", "label": "base",
            "seed": seed, "manifest": f"seed-{seed}", "metrics": metrics}


def test_equal_seed_statistics_and_duplicate_attempts_are_not_pseudoreplicates():
    one = aggregate_records([generation_record(1, .2)], UNCERTAINTY)[0]
    assert one["metrics"]["pass95"]["seed_interval"] is None
    two = aggregate_records([generation_record(1, .2), generation_record(2, .6)], UNCERTAINTY)[0]
    assert two["metrics"]["pass95"]["mean"] == pytest.approx(.4)
    assert two["metrics"]["pass95"]["std"] == pytest.approx(math.sqrt(.08))
    duplicate = aggregate_records([generation_record(1, .2), generation_record(1, .6)], UNCERTAINTY)[0]
    assert duplicate["duplicate_seed_attempts"]
    assert duplicate["metrics"]["pass95"]["mean"] is None


def test_family_separates_calibration_and_config_separates_native_rho_cache():
    protocol = {"identities": {"checkpoint_id": "model", "calibration": {"threshold": .8}},
                "source_sha256": {}, "seed": 1, "method": "vgb", "verifier": "cars",
                "value_law": "native-conditional", "cache_mode": "cold"}
    family, config, _ = grouping(protocol, {}, kind="generation")
    other_seed = {**protocol, "seed": 2}
    assert grouping(other_seed, {}, kind="generation")[:2] == (family, config)
    for change in ({"value_law": "rho"}, {"cache_mode": "warm"}):
        changed = grouping({**protocol, **change}, {}, kind="generation")
        assert changed[0] == family and changed[1] != config
    changed = copy.deepcopy(protocol)
    changed["identities"]["calibration"]["threshold"] = .9
    assert grouping(changed, {}, kind="generation")[0] != family


def test_missing_and_corrupt_attempts_are_audited_without_model_loading(artifact_directory, evaluation_config, monkeypatch):
    # No generation or backbone is called. A report must still explain missing input.
    missing = os.path.join(artifact_directory, "missing.json")
    monkeypatch.setattr("src.evaluation.aggregate.load_inputs", lambda protocol: pytest.fail("Must not load QM9 for missing input"))
    result = evaluate(attempt_id="missing-input", manifests=(missing,), config_path=evaluation_config)
    manifest = read_completed_manifest(result["manifest"])
    assert not result["selection_complete"]
    report = read_json(os.path.join(os.path.dirname(result["manifest"]), "metrics.json"))
    assert not report["groups"] and report["audit"][0]["status"] == "unusable"
    assert "metrics.csv" in manifest["artifacts"]
    corrupt = os.path.join(artifact_directory, "corrupt.json")
    with open(corrupt, "x", encoding="utf-8") as handle:
        handle.write("not json")
    assert "partial_audit_error" in unusable_attempt(corrupt, ValueError("corrupt"))
    with pytest.raises(ValueError, match="once"):
        select_manifests((missing, missing), ())


def test_failed_attempt_keeps_verified_cost_lower_bound(artifact_directory):
    attempt = RunArtifacts.create(artifact_directory, run_id="failed-generation", attempt_id="one",
                                  config={}, target_id="target", reference_id="reference")
    write_json(attempt.path("traces/sample-0.json"), {"budget": ledger(base_calls=2, base_sequences=8), "wall_seconds": 3})
    attempt.register("traces/sample-0.json")
    attempt.fail("interrupted before samples.jsonl completed")
    before = file_hash(attempt.path("manifest.json"))
    audit = unusable_attempt(attempt.path("manifest.json"), ValueError("not completed"))
    assert audit["source_status"] == "failed"
    assert audit["verified_trace_outputs"] == 1
    assert audit["partial_cost_lower_bound"]["total"]["base_sequences"] == 8
    assert file_hash(attempt.path("manifest.json")) == before


def test_run_id_selection_includes_attempts_missing_their_manifest():
    os.makedirs("workspace/results/raw", exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="step11-run-", dir="workspace/results/raw") as directory:
        attempt_directory = os.path.join(directory, "attempts", "crashed")
        os.makedirs(attempt_directory)
        paths, missing = select_manifests((), (os.path.basename(directory),))
        assert paths == [os.path.join(os.path.relpath(attempt_directory), "manifest.json")]
        assert not missing


def test_toy_artifacts_evaluate_against_hand_values_and_write_value_plot(artifact_directory, evaluation_config):
    # Exact values 1/4 and 3/4, predictions 1/2 and 1/2: MAE=RMSE=1/4.
    protocol = {"mode": "toy", "target": TARGET, "value_law": "rho", "fixture_sha256": "fixture",
                "settings": {"seed": 1}, "reference": "exact_finite_enumeration", "source_sha256": {}}
    source = RunArtifacts.create(artifact_directory, run_id="toy-source", attempt_id="one",
                                 config=protocol, target_id="target", reference_id="reference")
    source.manifest.update(task="check_toy_values", target=TARGET)
    rows = [{"name": str(i), "tokens": [0], "exact_value": exact,
             "estimators": {"mc": {"estimate": estimate(.5), "queries": [], "budget": ledger()}}}
            for i, exact in enumerate((.25, .75))]
    write_json(source.path("protocol.json"), protocol)
    write_json(source.path("execution.json"), {"states": 2, "enumeration_budget": ledger(), "device": "cpu"})
    write_jsonl(source.path("values.jsonl"), rows, progress=False)
    for name in ("protocol.json", "execution.json", "values.jsonl"):
        source.register(name)
    source.complete(required_artifacts=("protocol.json", "execution.json", "values.jsonl"))
    result = evaluate(attempt_id="toy-report", manifests=(source.path("manifest.json"),), config_path=evaluation_config)
    assert result["selection_complete"]
    report = read_json(os.path.join(os.path.dirname(result["manifest"]), "metrics.json"))
    metrics = report["groups"][0]["metrics"]
    assert metrics["mae"]["mean"] == .25 and metrics["rmse"]["mean"] == .25
    assert metrics["bias"]["mean"] == 0
    assert "figures/value_error_cost.png" in result["artifacts"]
    retry = RunArtifacts.create(artifact_directory, run_id="toy-source", attempt_id="retry",
                                config=protocol, target_id="target", reference_id="reference")
    retry.manifest.update(task="check_toy_values", target=TARGET)
    for name, payload in (("protocol.json", protocol), ("execution.json", {"states": 2, "enumeration_budget": ledger(), "device": "cpu"})):
        write_json(retry.path(name), payload)
        retry.register(name)
    write_jsonl(retry.path("values.jsonl"), rows, progress=False)
    retry.register("values.jsonl")
    retry.complete(required_artifacts=("protocol.json", "execution.json", "values.jsonl"))
    duplicate = evaluate(attempt_id="duplicate-report", manifests=(source.path("manifest.json"), retry.path("manifest.json")), config_path=evaluation_config)
    assert not duplicate["selection_complete"]
    duplicate_report = read_json(os.path.join(os.path.dirname(duplicate["manifest"]), "metrics.json"))
    assert any(item["status"] == "duplicate_seed_attempts" for item in duplicate_report["audit"])
