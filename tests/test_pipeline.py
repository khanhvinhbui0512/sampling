"""A real, tiny, offline model crosses the same task boundaries as QM9."""

import copy
import json
import os
import tempfile

import pytest
import torch

from src.budget import BudgetTracker
from src.config import load_config
from src.evaluation.aggregate import evaluate
from src.experiments.run import expected_target, generate, load_context, make_verifier
from src.io import (read_checkpoint, read_completed_manifest, read_json,
                    read_jsonl, write_yaml)
from src.models.backbone import QM9Backbone
from src.qm9.calibration import calibrate_qm9
from src.randomness import RNGStream
from src.training.base import train_base
from src.types import CostPhase, MaskedState, TokenRules


def assert_equal_state(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal_state(a, b)
    else:
        assert left == right


@pytest.fixture
def tiny_training(offline_assets, artifact_directory):
    config = load_config("configs/base_training.yaml")
    config["model"].update(hidden_dim=32, cond_dim=16, n_blocks=2, n_heads=4, dropout=0.)
    config["training"].update(batch_size=2, train_steps=2, warmup_steps=1,
                              precision="fp32", save_every=1, log_every=1)
    config["output_dir"] = os.path.join(artifact_directory, "models")
    path = os.path.join(artifact_directory, "base.yaml")
    write_yaml(path, config)
    result = train_base(offline_assets.manifest_path, config_path=path,
                        run_id="tiny", attempt_id="full", device="cpu")
    return offline_assets, path, result


def test_base_resume_checkpoint_batch_order_and_actual_forward_hooks(tiny_training):
    assets, path, full = tiny_training
    directory = os.path.dirname(os.path.dirname(full["checkpoint_manifest"]))
    first = os.path.join(directory, "step-0000001", "manifest.json")
    resumed = train_base(assets.manifest_path, config_path=path, run_id="tiny",
                         attempt_id="resumed", device="cpu", resume_manifest=first)
    before = QM9Backbone.from_checkpoint(full["checkpoint_manifest"], device="cpu")
    after = QM9Backbone.from_checkpoint(resumed["checkpoint_manifest"], device="cpu")
    assert_equal_state(before.model.state_dict(), after.model.state_dict())
    full_state = read_checkpoint(os.path.join(os.path.dirname(full["checkpoint_manifest"]), "training_state.pt"))
    resumed_state = read_checkpoint(os.path.join(os.path.dirname(resumed["checkpoint_manifest"]), "training_state.pt"))
    for key in ("step", "optimizer", "scheduler", "rng", "cumulative_base_calls"):
        assert_equal_state(full_state[key], resumed_state[key])
    assert full["attempt_cost"]["total"]["base_calls"] == 2
    assert resumed["attempt_cost"]["total"]["base_calls"] == 1
    before.model.eval()
    rules = assets.token_rules
    states = [MaskedState((rules.mask_id,) * rules.sequence_length, rules),
              MaskedState((5,) + (rules.mask_id,) * (rules.sequence_length - 1), rules)]
    calls = []
    hook = before.model.register_forward_pre_hook(lambda module, args, kwargs: calls.append(len(kwargs["input_ids"])),
                                                  with_kwargs=True)
    budget = BudgetTracker(base_sequence_limit=4)
    try:
        logits = before.logits_batch(states, budget=budget, phase=CostPhase.VERIFICATION)
        reverse = before.logits_batch(states[::-1], budget=budget, phase=CostPhase.VERIFICATION)
    finally:
        hook.remove()
    torch.testing.assert_close(logits, reverse.flip(0))
    assert calls == [2, 2]
    assert budget.raw_nfe == len(calls) and budget.snapshot().total.base_sequences == sum(calls)
    invalid = copy.deepcopy(states[0].rules.__dict__)
    invalid["vocab_size"] += 1
    with pytest.raises(ValueError):
        before.logits_batch([MaskedState(states[0].token_ids, TokenRules(**invalid))],
                             budget=budget, phase=CostPhase.VERIFICATION)
    assert budget.raw_nfe == 2


def test_calibration_guided_cars_generation_and_evaluation_artifact_flow(tiny_training, artifact_directory):
    assets, _, training = tiny_training
    reference = load_config("configs/reference_checks.yaml")
    reference["qm9"]["native"].update(steps=2, time_stop=.2)
    reference["qm9"]["calibration"].update(sample_count=4, batch_size=2, bootstrap_replicates=10,
                                              output_dir=os.path.join(artifact_directory, "calibration"))
    reference_path = os.path.join(artifact_directory, "reference.yaml")
    write_yaml(reference_path, reference)
    calibrated = calibrate_qm9(assets.manifest_path, training["checkpoint_manifest"],
                              config_path=reference_path, attempt_id="tiny", device="cpu")
    calibration_rows = list(read_jsonl(os.path.join(os.path.dirname(calibrated["manifest"]), "samples.jsonl"), progress=False))
    assert len(calibration_rows) == 4
    assert read_completed_manifest(calibrated["manifest"])["task"] == "calibrate_qm9"
    inference = load_config("configs/mdm_vgb.yaml")
    inference["native_settings"] = reference["qm9"]["native"]
    inference["candidates"].update(L_f=1, L_b=1, K=1)
    inference["walk"]["step_limit"] = 1
    inference["budget"]["base_sequence_limit"] = 1000
    inference_path = os.path.join(artifact_directory, "inference.yaml")
    write_yaml(inference_path, inference)
    cars = load_config("configs/cars.yaml")
    cars["native"].update(discovery_attempts=1, numerator_draws=4, denominator_draws=4,
                           mc_control_draws=4, min_denominator_hits=1, max_nodes=500, max_cache_entries=2)
    cars_path = os.path.join(artifact_directory, "cars.yaml")
    write_yaml(cars_path, cars)
    inputs = (assets.manifest_path, training["checkpoint_manifest"], calibrated["manifest"])
    settings = dict(config_path=inference_path, cars_config=cars_path, reference_config=reference_path, device="cpu")
    context = load_context(*inputs, **settings)
    mc = make_verifier(context, "mc", "native-conditional", cache_scope="tiny-mc")
    cars_verifier = make_verifier(context, "cars", "native-conditional", cache_scope="tiny-cars")
    assert mc.target == cars_verifier.target == expected_target(context, "native-conditional")
    assert mc.target != expected_target(context, "rho")
    # Ordered queries include exact terminals and duplicate states; terminal values
    # come from reward rather than from a stochastic value approximation.
    rules = assets.token_rules
    root = MaskedState((rules.mask_id,) * rules.sequence_length, rules)
    leaf = MaskedState((rules.bos_id, 5, rules.eos_id, rules.pad_id, rules.pad_id, rules.pad_id), rules)
    exact = context.reward.evaluate(leaf, require_binary=True).reward
    for verifier in (mc, cars_verifier):
        calls = []
        hook = context.backbone.model.register_forward_pre_hook(
            lambda module, args, kwargs: calls.append(len(kwargs["input_ids"])), with_kwargs=True)
        budget = BudgetTracker(base_sequence_limit=1000)
        try:
            values = verifier.values_for_states([leaf, root, leaf], rng=RNGStream(11), budget=budget)
        finally:
            hook.remove()
        assert len(values) == 3 and values[0].value == values[2].value == exact
        assert all(value.target == mc.target for value in values)
        assert budget.raw_nfe == len(calls)
        assert budget.snapshot().total.base_sequences == sum(calls)
        assert sum(value.cost.base_calls for value in values) == len(calls)
    os.makedirs("workspace/results/raw", exist_ok=True)
    os.makedirs("workspace/results/summary", exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="step12-", dir="workspace/results/raw") as raw:
        with tempfile.TemporaryDirectory(prefix="step12-", dir="workspace/results/summary") as summary:
            results = [generate(*inputs, method=method, verifier=verifier, attempt_id=verifier,
                                run_id=os.path.basename(raw), num_samples=2, seed=7, **settings)
                       for method, verifier in (("base_native", "none"), ("vgb", "mc"), ("vgb", "cars"))]
            for index, result in enumerate(results):
                manifest = read_completed_manifest(result["manifest"])
                assert manifest["requested_outputs"] == 2
                expected = "native_rollout" if index == 0 else "reference_conditional_v_star"
                assert manifest["target"]["kind"] == expected
            evaluation = load_config("configs/evaluation.yaml")
            evaluation["output_dir"] = os.path.relpath(summary)
            evaluation["uncertainty"]["bootstrap_replicates"] = 10
            evaluation_path = os.path.join(artifact_directory, "evaluation.yaml")
            write_yaml(evaluation_path, evaluation)
            report = evaluate(manifests=[item["manifest"] for item in results], attempt_id="tiny",
                              comparison_id="pipeline", config_path=evaluation_path)
            assert report["selection_complete"] and report["records"] == 3
            read_completed_manifest(report["manifest"])
            metrics = read_json(os.path.join(os.path.dirname(report["manifest"]), "metrics.json"))
            assert len(metrics["groups"]) == 3
            # Metadata has no self-hash: tampering with its target must still fail
            # validation, while preserving the rest of the registered artifacts.
            damaged = results[-1]["manifest"]
            manifest = read_json(damaged)
            manifest["target"]["kind"] = "v_rho"
            with open(damaged, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            rejected = evaluate(manifests=[damaged], attempt_id="rejected",
                                comparison_id="pipeline", config_path=evaluation_path)
            assert not rejected["selection_complete"] and rejected["records"] == 0


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("QM9_GPU_SMOKE") != "1", reason="GPU smoke requires an explicit request")
def test_explicit_cuda_smoke(tiny_model, tokenizer, token_rules):
    assert torch.cuda.is_available(), "GPU smoke was requested but CUDA is unavailable"
    backbone = QM9Backbone(tiny_model, tokenizer, token_rules, tokenizer_sha256="a" * 64,
                           device="cuda", precision="fp32")
    root = MaskedState((token_rules.mask_id,) * token_rules.sequence_length, token_rules)
    budget = BudgetTracker(base_sequence_limit=1)
    logits = backbone.logits_batch([root], budget=budget, phase=CostPhase.VERIFICATION)
    assert next(backbone.model.parameters()).is_cuda and torch.isfinite(logits).all()
    assert budget.raw_nfe == 1
