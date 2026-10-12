"""Offline checks of the Step 4 probability and QM9 calibration protocols."""

import json
import math
import os
from dataclasses import asdict

import datasets
import numpy as np
import pytest
import torch

from src.budget import BudgetTracker
from src.config import load_config
from src.io import file_hash, read_json, read_jsonl, write_json, write_yaml
from src.models.backbone import QM9Backbone
from src.qm9.calibration import calibrate_qm9, scores_from_terminals, threshold_statistics
from src.qm9.chemistry import assess_smiles, rdkit_version
from src.qm9.dataset import load_prepared_qm9
from src.qm9.reward import BinaryCalibration, CalibrationRequired, QM9Reward, load_calibration
from src.randomness import RNGStream
from src.sampling.base import BaseBatch, BaseSampler
from src.sampling.completion import RhoCompletion
from src.sampling.native import NativeMDLM, absorbing_posterior
from src.sampling.reference import RhoJointReference
from src.training.base import build_scheduler, save_training_checkpoint
from src.types import CostPhase, CostRecord, MaskedState, SampleStatus


def budget():
    return BudgetTracker(base_sequence_limit=None)


def test_joint_and_completion_are_different_for_future_observations(toy_joint, toy_backbone):
    policy = RhoCompletion(toy_backbone)
    reference = RhoJointReference(policy)
    rules = toy_backbone.token_rules
    leaves = [MaskedState(leaf["tokens"], rules) for leaf in toy_joint["leaves"]]
    tracker = budget()
    pi = reference.log_prob_batch(leaves, budget=tracker).exp()
    assert tracker.raw_nfe == 3  # Three actual batched prefix forwards, not 8*3.
    assert torch.allclose(pi, torch.tensor([leaf["probability"] for leaf in toy_joint["leaves"]], dtype=torch.float64))
    assert float(pi.sum()) == pytest.approx(1)
    z = MaskedState((0, 0, 2), rules)
    indices = [i for i, leaf in enumerate(leaves) if z.accepts_completion(leaf)]
    compatible = [leaves[i] for i in indices]
    log_q = policy.log_prob_batch(compatible, [z] * len(compatible), budget=tracker, phase=CostPhase.IMPORTANCE)
    q = log_q.exp()
    rewards = torch.tensor([toy_joint["leaves"][i]["reward"] for i in indices])
    assert float(q.sum()) == pytest.approx(1)
    v_rho = float((q * rewards).sum())
    conditional_pi = pi[indices] / pi[indices].sum()
    v_star = float((conditional_pi * rewards).sum())
    assert v_rho == pytest.approx(toy_joint["expected_future_observation"]["v_rho"])
    assert v_star == pytest.approx(toy_joint["expected_future_observation"]["v_star_pi_rho"])
    weights = (pi[indices].log() - log_q).exp()
    assert float((q * weights).sum()) == pytest.approx(float(pi[indices].sum()))
    assert float((q * weights * rewards).sum() / (q * weights).sum()) == pytest.approx(v_star)
    assert reference.log_weight(compatible[0], z, budget=tracker) == pytest.approx(float(weights[0].log()))


def test_prefix_conditionals_and_terminal_likelihoods(toy_joint, toy_backbone):
    policy, rules = RhoCompletion(toy_backbone), toy_backbone.token_rules
    reference = RhoJointReference(policy)
    prefix = MaskedState((1, 0, 0), rules)
    leaves = [MaskedState(leaf["tokens"], rules) for leaf in toy_joint["leaves"] if leaf["tokens"][0] == 1]
    tracker = budget()
    joint = reference.log_prob_batch(leaves, budget=tracker)
    conditional = policy.log_prob_batch(leaves, [prefix] * len(leaves), budget=tracker, phase=CostPhase.COMPLETION)
    assert torch.allclose(conditional, joint - math.log(0.5))
    before = tracker.raw_nfe
    assert policy.log_prob(leaves[0], leaves[0], budget=tracker, phase=CostPhase.COMPLETION) == 0
    assert policy.log_prob(MaskedState((2, 1, 1), rules), prefix,
                           budget=tracker, phase=CostPhase.COMPLETION) == -math.inf
    assert tracker.raw_nfe == before


def test_rho_replay_matches_sampling_and_budget_preserves_partial_state(toy_backbone):
    policy, rules = RhoCompletion(toy_backbone), toy_backbone.token_rules
    z = MaskedState((0, 0, 2), rules)
    sampled = policy.complete_batch([z], rng=RNGStream(9), budget=budget(), phase=CostPhase.ESTIMATION)
    assert z.accepts_completion(sampled.states[0])
    scored = policy.log_prob(sampled.states[0], z, budget=budget(), phase=CostPhase.ESTIMATION)
    assert sampled.log_probabilities[0] == pytest.approx(scored)
    short = policy.complete(z, rng=RNGStream(9), budget=BudgetTracker(base_sequence_limit=1),
                             phase=CostPhase.COMPLETION)
    assert short.status == SampleStatus.BUDGET_EXHAUSTED
    assert len(short.state.masked_positions) == 1 and short.state.token_ids[2] == 2
    assert short.cost.base_calls == 1 and short.score is None


def test_absorbing_posterior_conserves_mass_and_known_tokens():
    theta = torch.tensor([[[0.0, 0.25, 0.75], [0.0, 0.25, 0.75]]])
    tokens = torch.tensor([[0, 2]])
    posterior = absorbing_posterior(theta, tokens, 0, 0.8, 0.2)
    assert torch.allclose(posterior.sum(-1), torch.ones(1, 2, dtype=torch.float64))
    assert torch.allclose(posterior[0, 0], torch.tensor([0.25, 0.1875, 0.5625], dtype=torch.float64))
    assert torch.equal(posterior[0, 1], torch.tensor([0., 0., 1.], dtype=torch.float64))


def test_native_cache_snapshots_and_explicit_final_denoise(toy_backbone, monkeypatch):
    def keep_mask_until_denoise(probabilities, num_samples, generator):
        return torch.where(probabilities[:, :1] > 0, 0, probabilities.argmax(-1, keepdim=True))

    monkeypatch.setattr(torch, "multinomial", keep_mask_until_denoise)
    root = MaskedState((0, 0, 2), toy_backbone.token_rules)
    sampler = NativeMDLM(toy_backbone, steps=10)
    tracker = budget()
    result = sampler.sample_batch([root], rng=RNGStream(1), budget=tracker, snapshot_steps=(0, 5))
    assert root.accepts_completion(result.states[0])
    assert result.snapshots == {0: (root,), 5: (root,)}
    assert tracker.raw_nfe == 1  # Reuse only while the entire state batch is unchanged.
    no_denoise = NativeMDLM(toy_backbone, steps=2, final_denoise=False)
    assert not no_denoise.sample_batch([root], rng=RNGStream(1), budget=budget()).states[0].is_complete
    toy_backbone.time_conditioning = True
    tracker = budget()
    sampler.sample_batch([root], rng=RNGStream(1), budget=tracker)
    assert tracker.raw_nfe == 11  # Time-conditioned models may not reuse that cache.
    assert float(toy_backbone.noise_calls[-11][0]) == pytest.approx(-math.log(0.001), rel=1e-4)


def test_unfinished_base_never_reaches_reward(fake_backbone_factory, token_rules, tokenizer,
                                             chemistry_policy, monkeypatch):
    probabilities = [0.] * token_rules.vocab_size
    probabilities[5] = 1.
    backbone = fake_backbone_factory(probabilities)
    reward = QM9Reward(tokenizer=tokenizer, token_rules=token_rules,
                       chemistry_policy=chemistry_policy, tokenizer_sha256="a" * 64)
    backbone.tokenizer_id = "a" * 64
    sampler = BaseSampler(backbone, reward, native_settings={"steps": 2, "final_denoise": False})
    monkeypatch.setattr(torch, "multinomial", lambda probabilities, num_samples, generator:
                        torch.full((len(probabilities), 1), token_rules.mask_id, dtype=torch.long))
    tracker = budget()
    state = MaskedState((1, 3, 3, 2, 0, 0), token_rules)
    result = sampler.sample(state, sample_id="unfinished", rng=RNGStream(2), budget=tracker)
    assert result.status == SampleStatus.FAILED and result.score is None and result.valid is None
    assert tracker.snapshot().total.oracle_calls == 0


def test_quantile_keeps_invalid_zeros_and_bootstrap_does_not_change_threshold():
    scores = [0.] * 19 + [0.8]
    first = threshold_statistics(scores, bootstrap_rng=np.random.default_rng(1), replicates=20)
    second = threshold_statistics(scores, bootstrap_rng=np.random.default_rng(2), replicates=20)
    assert first["threshold"] == pytest.approx(0.04)
    assert first["threshold"] == second["threshold"]
    assert first["sample_count"] == 20 and first["uncertainty"]["standard_error"] >= 0
    with pytest.raises(ValueError):
        scores_from_terminals([None])
    for values in ([], [float("nan")], [-0.1], [1.1]):
        with pytest.raises(ValueError):
            threshold_statistics(values, bootstrap_rng=np.random.default_rng(1), replicates=20)


def test_prepared_split_and_canonical_train_identity(offline_assets):
    loaded = load_prepared_qm9(offline_assets.manifest_path)
    assert read_json(loaded.splits_path)["train_row_ids"] == [0, 1]
    train = list(read_jsonl(loaded.train_canonical_path, progress=False))
    assert train == [{"canonical_smiles": "C"}]
    assert assess_smiles(" C ", policy=loaded.chemistry_policy).canonical_smiles == train[0]["canonical_smiles"]
    raw = datasets.load_from_disk(loaded.raw_directory)
    assert raw.select([0, 1])["canonical_smiles"] == ["C", "C"]  # Training keeps both rows.


def test_overlapping_split_rejected_even_with_updated_file_hashes(offline_assets):
    splits = read_json(offline_assets.splits_path)
    splits["test_row_ids"] = [1]
    with open(offline_assets.splits_path, "w", encoding="utf-8") as handle:
        json.dump(splits, handle)
    manifest = read_json(offline_assets.manifest_path)
    digest = file_hash(offline_assets.splits_path)
    manifest["artifacts"]["splits.json"] = {
        "sha256": digest, "size_bytes": os.path.getsize(offline_assets.splits_path),
    }
    manifest["preparation"]["splits_sha256"] = digest
    bad_manifest = os.path.join(os.path.dirname(offline_assets.manifest_path), "bad_manifest.json")
    write_json(bad_manifest, manifest)
    with pytest.raises(ValueError, match="disjoint partition"):
        load_prepared_qm9(bad_manifest)


def test_calibration_uses_generated_draws_and_locks_identity(offline_assets, tiny_model, tokenizer,
                                                           artifact_directory, monkeypatch):
    backbone = QM9Backbone(tiny_model, tokenizer, offline_assets.token_rules,
                           tokenizer_sha256=offline_assets.tokenizer_sha256)
    settings = {"lr": 3e-4, "min_lr": 3e-6, "warmup_steps": 1, "train_steps": 5}
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=settings["lr"])
    scheduler = build_scheduler(optimizer, settings)
    checkpoint = save_training_checkpoint(
        os.path.join(artifact_directory, "model"), backbone, optimizer, scheduler, step=1,
        config={"training": settings}, environment={"device": "cpu"}, cumulative_base_calls=0,
        identities={"assets_manifest_sha256": file_hash(offline_assets.manifest_path)},
    )
    config = load_config("configs/reference_checks.yaml")
    config["qm9"]["native"]["steps"] = 2
    config["qm9"]["calibration"].update(sample_count=4, batch_size=2, bootstrap_replicates=10,
                                         output_dir=os.path.join(artifact_directory, "calibrations"))
    config_path = os.path.join(artifact_directory, "calibration.yaml")
    write_yaml(config_path, config)

    def forbid_dataset_loading(*args, **kwargs):
        raise AssertionError("Calibration must not read train/test rows to compute its threshold")

    monkeypatch.setattr(datasets, "load_from_disk", forbid_dataset_loading)
    first = calibrate_qm9(offline_assets.manifest_path, checkpoint, config_path=config_path,
                          calibration_id="offline", attempt_id="one", device="cpu")
    torch.manual_seed(999)
    second = calibrate_qm9(offline_assets.manifest_path, checkpoint, config_path=config_path,
                           calibration_id="offline", attempt_id="two", device="cpu")
    assert first["calibration_id"] == second["calibration_id"]
    calibration = load_calibration(first["manifest"])
    rows = list(read_jsonl(os.path.join(os.path.dirname(first["manifest"]), "samples.jsonl"), progress=False))
    assert len(rows) == calibration.sample_count == 4
    assert calibration.threshold == pytest.approx(np.quantile([row["score"] for row in rows], .95))
    assert all(row["purpose"] == "calibration" and row["reward"] is None for row in rows)
    assert first["cost"]["total"]["oracle_calls"] == 4
    with pytest.raises(FileExistsError):
        calibrate_qm9(offline_assets.manifest_path, checkpoint, config_path=config_path,
                       calibration_id="offline", attempt_id="one", device="cpu")
    monkeypatch.setattr(BaseSampler, "generate_batch", lambda self, states, **kwargs:
                        BaseBatch(tuple(states), (None,) * len(states), CostRecord(), False, None))
    with pytest.raises(ValueError, match="complete draws"):
        calibrate_qm9(offline_assets.manifest_path, checkpoint, config_path=config_path,
                       calibration_id="offline", attempt_id="unfinished", device="cpu")
    failure_dir = os.path.join(config["qm9"]["calibration"]["output_dir"], "offline", "attempts", "unfinished")
    assert read_json(os.path.join(failure_dir, "manifest.json"))["status"] == "failed"
    assert not os.path.exists(os.path.join(failure_dir, "calibration.json"))


def test_zero_threshold_still_requires_valid_molecule(tokenizer, token_rules, chemistry_policy):
    calibration = BinaryCalibration(
        "zero-threshold", "base-reference", "base-policy", "b" * 64, "a" * 64, 0., .95, 20, 1,
        "linear", asdict(chemistry_policy), rdkit_version(), "independent_base_calibration",
        "qed_or_zero", "strict_special_tokens_v1",
    )
    reward = QM9Reward(tokenizer=tokenizer, token_rules=token_rules, chemistry_policy=chemistry_policy,
                       tokenizer_sha256="a" * 64, calibration=calibration,
                       calibration_reference_id="base-reference", calibration_policy_id="base-policy",
                       checkpoint_sha256="b" * 64)
    assert reward.evaluate(MaskedState((1, 5, 2, 0, 0, 0), token_rules), require_binary=True).reward == 1
    assert reward.evaluate(MaskedState((1, 4, 2, 0, 0, 0), token_rules), require_binary=True).reward == 0
    with pytest.raises(ValueError):
        QM9Reward(tokenizer=tokenizer, token_rules=token_rules, chemistry_policy=chemistry_policy,
                  tokenizer_sha256="a" * 64, calibration=calibration,
                  calibration_reference_id="wrong", calibration_policy_id="base-policy", checkpoint_sha256="b" * 64)
    uncalibrated = QM9Reward(tokenizer=tokenizer, token_rules=token_rules,
                             chemistry_policy=chemistry_policy, tokenizer_sha256="a" * 64)
    with pytest.raises(CalibrationRequired):
        uncalibrated.evaluate(MaskedState((1, 5, 2, 0, 0, 0), token_rules), require_binary=True)
