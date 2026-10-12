"""Step 10 factory/panel/cache regressions, written for review and NOT run."""

import os
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from src.budget import BudgetTracker
from src.config import load_config
from src.experiments.run import (expected_target, generate, make_sampler, make_verifier, save_cache, warm_cache)
from src.experiments.toy import check_toy_values
from src.experiments.value_checks import check_values, load_panel, masked_variants, prepare_value_panel
from src.io import RunArtifacts, read_completed_manifest, read_json, read_jsonl, write_yaml
from src.randomness import RNGStream
from src.sampling.completion import RhoCompletion
from src.types import MaskedState, TargetKind


@pytest.fixture
def context(native_case, artifact_directory):
    native, reward = native_case
    native.backbone.device, native.backbone.precision = torch.device("cpu"), "toy"
    inference = load_config("configs/mdm_vgb.yaml")
    inference["native_settings"] = {key: getattr(native.policy, key)
                                    for key in ("steps", "noise_eps", "time_stop", "final_denoise")}
    cars = load_config("configs/cars.yaml")
    cars["native"].update(discovery_attempts=0, numerator_draws=8, denominator_draws=8,
                          mc_control_draws=16, min_denominator_hits=1, max_nodes=100,
                          max_cache_entries=1, persistent_cache=True,
                          directory=os.path.join(artifact_directory, "cache"))
    return SimpleNamespace(native=native, rho=RhoCompletion(native.backbone), reward=reward,
                           backbone=native.backbone, config=inference, cars=cars,
                           identities={"native_reference": native.protocol}, inputs={})


def test_factory_targets_and_guided_kernel_are_explicit(context):
    targets = [expected_target(context, law) for law in ("native-conditional", "rho", "upstream-native")]
    assert len({target.target_id for target in targets}) == 3
    assert targets[0].kind == TargetKind.REFERENCE_CONDITIONAL and targets[1].kind == TargetKind.RHO
    adapter = make_verifier(context, "mc", "native-conditional", cache_scope="factory")
    for method in ("vgr", "vgb", "vgb_momentum"):
        sampler = make_sampler(context, method, verifier=adapter)
        assert sampler.verifier is adapter and sampler.target == targets[0]
        assert sampler.completion_sampler.generator.backbone is context.backbone
    with pytest.raises(ValueError, match="no matched"):
        make_verifier(context, "mc", "upstream-native", cache_scope="wrong-target")
    context.cars["native"]["numerator_draws"] = None
    with pytest.raises(ValueError, match="Set numerator_draws"):
        make_verifier(context, "cars", "native-conditional", cache_scope="unprofiled")


def test_baselines_need_no_neural_checkpoint_and_reject_changed_native_law(context):
    assert make_sampler(context, "base_native").reference_id == context.native.reference_id
    assert make_sampler(context, "base_rho").target.kind == TargetKind.RHO
    assert make_sampler(context, "bon").base_sampler.generator.backbone is context.backbone
    context.config["native_settings"]["steps"] = 3
    with pytest.raises(ValueError, match="calibrated normalized BASE"):
        make_sampler(context, "base_native")


def test_cars_warm_snapshot_preserves_value_uncertainty_and_origin_cost(context, artifact_directory):
    cold = make_verifier(context, "cars", "native-conditional", cache_scope="cold")
    root = context.native.root
    budget = BudgetTracker(base_sequence_limit=None)
    value = cold.values_for_states([root], rng=RNGStream(2), budget=budget)[0]
    assert value.value is not None and value.cost.base_sequences > 0
    attempt = RunArtifacts.create(artifact_directory, run_id="cache-source", attempt_id="cold",
                                  config={}, target_id=value.target.target_id, reference_id=value.target.reference_id)
    save_cache(attempt, cold, {root.token_ids}, {})
    attempt.complete(required_artifacts=("cache_index.jsonl",))
    warm = make_verifier(context, "cars", "native-conditional", cache_scope="warm")
    origins = warm_cache(warm, attempt.path("manifest.json"))
    assert root.token_ids in origins
    zero = BudgetTracker(base_sequence_limit=0)
    reused = warm.values_for_states([root], rng=RNGStream(3), budget=zero)[0]
    assert reused.cache_hit and reused.value == value.value and reused.interval == value.interval
    assert reused.cost.base_sequences == 0 and zero.raw_nfe == 0
    assert warm.records[0]["items"][0]["cache_provenance"]["origin_cost"]["base_sequences"] > 0
    context.cars["native"]["denominator_draws"] = 16
    other = make_verifier(context, "cars", "native-conditional", cache_scope="incompatible")
    with pytest.raises(ValueError, match="metadata/full context"):
        warm_cache(other, attempt.path("manifest.json"))


def test_panel_masks_include_future_observations(context):
    terminal = MaskedState((1, 2), context.native.rules)
    variants = list(masked_variants(terminal))
    assert ("prefix", MaskedState((1, 0), terminal.rules)) in variants
    assert ("hole", MaskedState((0, 2), terminal.rules)) in variants
    assert all(state.accepts_completion(terminal) for _, state in variants)


def test_generation_keeps_every_requested_budget_failure(context, artifact_directory, monkeypatch):
    monkeypatch.setattr("src.experiments.run.load_context", lambda *args, **kwargs: context)

    class LocalAttempts:
        @staticmethod
        def create(base_dir, **kwargs):
            return RunArtifacts.create(os.path.join(artifact_directory, "generation"), **kwargs)

    monkeypatch.setattr("src.experiments.run.RunArtifacts", LocalAttempts)
    context.config["budget"]["base_sequence_limit"] = 0
    result = generate("unused", "unused", "unused", method="base_native", attempt_id="budget-zero",
                      num_samples=2, device="cpu")
    manifest = read_completed_manifest(result["manifest"])
    directory = os.path.dirname(result["manifest"])
    rows = list(read_jsonl(os.path.join(directory, "samples.jsonl"), progress=False))
    assert len(rows) == manifest["requested_outputs"] == result["outputs"] == 2
    assert all(row["status"] == "budget_exhausted" and row["reward"] is None for row in rows)
    assert read_json(os.path.join(directory, "summary.json"))["successes"] == 0
    for row in rows:
        trace = read_json(os.path.join(directory, row["trace_artifact"]))
        assert trace["result"]["sample_id"] == row["sample_id"]
        assert trace["budget"]["total"]["base_sequences"] == 0


def test_locked_panel_checks_keep_refused_values_and_detect_corruption(context, artifact_directory, monkeypatch):
    monkeypatch.setattr("src.experiments.value_checks.load_context", lambda *args, **kwargs: context)
    settings = load_config("configs/value_checks.yaml")
    settings.update(output_dir=os.path.join(artifact_directory, "panels"),
                    check_output_dir=os.path.join(artifact_directory, "checks"))
    settings["panel"].update(max_per_stratum=2, terminal_draws=4, snapshot_steps=[0, 1])
    settings["reference"].update(draws=8, min_denominator_hits=1)
    config = os.path.join(artifact_directory, "panel.yaml")
    write_yaml(config, settings)
    panel = prepare_value_panel("unused", "unused", "unused", attempt_id="panel", config_path=config, device="cpu")
    manifest, states = load_panel(panel["manifest"], context, "native-conditional")
    assert {"root", "prefix", "hole"}.issubset({row["stratum"] for row in states})
    with pytest.raises(ValueError, match="identity, target"):
        load_panel(panel["manifest"], context, "rho")
    settings["budget"]["base_sequence_limit"] = 0
    check_config = os.path.join(artifact_directory, "check.yaml")
    write_yaml(check_config, settings)
    result = check_values("unused", "unused", "unused", panel_manifest=panel["manifest"], attempt_id="check",
                          estimators=("mc",), config_path=check_config, device="cpu")
    read_completed_manifest(result["manifest"])
    rows = list(read_jsonl(os.path.join(os.path.dirname(result["manifest"]), "values.jsonl"), progress=False))
    assert len(rows) == len(states)
    assert all(row["estimators"]["mc"]["estimate"]["value"] is None for row in rows if 0 in row["tokens"])
    assert all(row["reference"]["status"] == "exact" for row in rows if 0 not in row["tokens"])
    protocol = read_json(os.path.join(os.path.dirname(panel["manifest"]), "protocol.json"))
    assert protocol["locked_before_compared_estimators"] and manifest["target"] == asdict(expected_target(context, "native-conditional"))
    with open(os.path.join(os.path.dirname(panel["manifest"]), "states.jsonl"), "a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="differs"):
        load_panel(panel["manifest"], context, "native-conditional")


@pytest.mark.parametrize("law", ["native-conditional", "rho"])
def test_finite_check_artifacts_keep_exact_and_diagnostic_targets_separate(law, artifact_directory):
    settings = load_config("configs/value_checks.yaml")
    settings["check_output_dir"] = os.path.join(artifact_directory, "toy-checks")
    settings["toy"].update(discovery_attempts=2, numerator_draws=32, denominator_draws=64,
                           mc_draws=64, min_denominator_hits=1)
    config = os.path.join(artifact_directory, "toy.yaml")
    write_yaml(config, settings)
    result = check_toy_values(attempt_id=law, value_law=law, config_path=config)
    manifest = read_completed_manifest(result["manifest"])
    rows = list(read_jsonl(os.path.join(os.path.dirname(result["manifest"]), "values.jsonl"), progress=False))
    assert len(rows) == 6 and all(row["exact_denominator"] > 0 for row in rows)
    assert all(row["estimators"][name]["estimate"]["target"] == manifest["target"]
               for row in rows for name in ("mc", "cars"))
    if law == "rho":
        future = next(row for row in rows if row["name"] == "future_observation")
        assert future["exact_value"] == pytest.approx(.5)
        assert future["pi_rho_conditional_diagnostic_not_v_rho"] == pytest.approx(.7)
