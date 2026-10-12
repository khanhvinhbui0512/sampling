"""Offline Step 5 checks. Written for review; no downloads or large models."""

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from src.budget import BudgetTracker
from src.config import load_config
from src.io import file_hash, read_json, write_yaml
from src.models.backbone import QM9Backbone
from src.qm9.calibration import calibrate_qm9
from src.qm9.reward import QM9Reward, load_calibration
from src.qm9.rollouts import collect_rollouts, load_rollout_shards, snapshot_indices, validation_rollout_ids
from src.randomness import RNGStream
from src.sampling.completion import RhoCompletion
from src.training.base import build_scheduler, save_training_checkpoint
from src.training.verifier import train_verifier
from src.types import CostPhase, EstimateStatus, MaskedState, ValueTarget
from src.verifiers.interface import value_target
from src.verifiers.monte_carlo import MonteCarloVerifier, bernoulli_uncertainty
from src.verifiers.neural import NeuralVerifier, TokenStateVerifier


@pytest.fixture
def toy_reward(toy_joint, toy_backbone):
    rewards = {tuple(leaf["tokens"]): leaf["reward"] for leaf in toy_joint["leaves"]}

    class ToyReward:
        token_rules = toy_backbone.token_rules
        tokenizer_sha256 = toy_backbone.tokenizer_id
        calibration = SimpleNamespace(calibration_id="toy-binary", checkpoint_sha256=toy_backbone.checkpoint_id)

        def evaluate(self, state, *, require_binary):
            assert require_binary and state.is_complete
            return SimpleNamespace(reward=rewards[state.token_ids])

    return ToyReward()


def test_mc_keeps_multiplicity_terminal_order_and_actual_cost(toy_backbone, toy_reward):
    policy = RhoCompletion(toy_backbone)
    verifier = MonteCarloVerifier(policy, toy_reward, draws=8, draw_batch_size=4)
    leaf = MaskedState((1, 1, 1), policy.rules)
    state = MaskedState((0, 0, 2), policy.rules)
    tracker = BudgetTracker(base_sequence_limit=None)
    results = verifier.values_for_states([state, leaf, state], rng=RNGStream(8), budget=tracker)
    assert results[1].status == EstimateStatus.EXACT and results[1].value == 1
    assert results[1].draws == 0
    assert results[0].draws == results[2].draws == 8
    assert results[0].target == results[2].target == value_target(policy, toy_reward)
    assert tracker.raw_nfe == 8  # Two positions * two draw batches * two state queries.
    assert tracker.snapshot().total.base_sequences == 32
    assert tracker.snapshot().total.oracle_calls == 17
    assert sum(result.cost.base_calls for result in results) == tracker.raw_nfe
    for result in (results[0], results[2]):
        assert result.interval[0] <= result.value <= result.interval[1]


def test_mc_reserves_entire_fixed_query_and_preserves_completion_budget(toy_backbone, toy_reward):
    policy = RhoCompletion(toy_backbone)
    verifier = MonteCarloVerifier(policy, toy_reward, draws=8)
    state = MaskedState((0, 0, 2), policy.rules)
    tracker = BudgetTracker(base_sequence_limit=18)
    with tracker.reserve_completion(base_sequences=3):
        result = verifier.values_for_states([state], rng=RNGStream(1), budget=tracker)[0]
        assert result.status == EstimateStatus.BUDGET_EXHAUSTED and result.value is None
        assert result.draws == 0 and tracker.raw_nfe == 0
    assert tracker.snapshot().reserved_base_sequences == 0
    tracker = BudgetTracker(base_sequence_limit=16)
    result = verifier.values_for_states([state], rng=RNGStream(1), budget=tracker)[0]
    assert result.draws == 8 and tracker.snapshot().total.base_sequences == 16
    assert tracker.snapshot().reserved_base_sequences == 0


def test_wilson_interval_is_nontrivial_with_zero_or_all_successes():
    se, interval = bernoulli_uncertainty(0, 16)
    assert se == 0 and interval[0] == 0 and interval[1] > 0
    _, interval = bernoulli_uncertainty(16, 16)
    assert interval[0] < 1 and interval[1] == 1


def test_neural_batch_order_terminal_anchoring_and_budget(toy_backbone, toy_reward):
    model = TokenStateVerifier(3, 3, hidden_dim=8, num_layers=1, num_heads=2, dropout=0.)
    policy = RhoCompletion(toy_backbone)
    adapter = NeuralVerifier(model, toy_reward, value_target(policy, toy_reward), batch_size=2)
    states = [MaskedState(tokens, policy.rules) for tokens in ((0, 0, 2), (1, 1, 1), (1, 0, 2), (0, 0, 2))]
    with torch.no_grad():
        expected = model(torch.tensor([state.token_ids for state in (states[0], states[2], states[3])]))
    tracker = BudgetTracker(base_sequence_limit=0, verifier_sequence_limit=3)
    results = adapter.values_for_states(states, rng=RNGStream(1), budget=tracker)
    assert [results[i].value for i in (0, 2, 3)] == pytest.approx(expected.tolist())
    assert results[1].status == EstimateStatus.EXACT and results[1].value == 1
    assert tracker.snapshot().total.verifier_calls == 2
    assert sum(result.cost.verifier_calls for result in results) == 2
    assert tracker.snapshot().total.oracle_calls == 1 and tracker.raw_nfe == 0
    short = adapter.values_for_states(states, rng=RNGStream(1),
                                      budget=BudgetTracker(base_sequence_limit=0, verifier_sequence_limit=0))
    assert short[0].value is None and short[1].value == 1


def test_rho_snapshots_do_not_change_sampling_law(toy_backbone):
    policy = RhoCompletion(toy_backbone)
    root = MaskedState((0, 0, 0), policy.rules)
    kwargs = dict(budget=BudgetTracker(base_sequence_limit=None), phase=CostPhase.TRAINING)
    first = policy.complete_batch([root], rng=RNGStream(99), snapshot_steps=(0, 1, 2), **kwargs)
    second = policy.complete_batch([root], rng=RNGStream(99), **kwargs)
    assert first.states == second.states and first.log_probabilities == second.log_probabilities
    assert first.snapshots[0] == (root,)
    assert all(state.accepts_completion(first.states[0]) for states in first.snapshots.values() for state in states)


def test_snapshot_selection_and_group_partition():
    assert snapshot_indices(128, 5) == (21, 42, 64, 85, 106)
    assert snapshot_indices(32, 5) == (5, 10, 16, 21, 26)
    first = validation_rollout_ids(100, .05, 42)
    assert first == validation_rollout_ids(100, .05, 42) and len(first) == 5
    with pytest.raises(ValueError):
        validation_rollout_ids(1, .05, 42)


@pytest.fixture
def offline_labels(offline_assets, tiny_model, tokenizer, artifact_directory):
    backbone = QM9Backbone(tiny_model, tokenizer, offline_assets.token_rules,
                           tokenizer_sha256=offline_assets.tokenizer_sha256)
    settings = {"lr": 3e-4, "min_lr": 3e-6, "warmup_steps": 1, "train_steps": 5}
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=settings["lr"])
    checkpoint = save_training_checkpoint(
        os.path.join(artifact_directory, "base"), backbone, optimizer, build_scheduler(optimizer, settings),
        step=1, config={"training": settings}, environment={"device": "cpu"},
        cumulative_base_calls=0, identities={"assets_manifest_sha256": file_hash(offline_assets.manifest_path)},
    )
    config = load_config("configs/reference_checks.yaml")
    config["qm9"]["native"]["steps"] = 2
    config["qm9"]["calibration"].update(sample_count=4, batch_size=2, bootstrap_replicates=10,
                                         output_dir=os.path.join(artifact_directory, "calibration"))
    calibration_config = os.path.join(artifact_directory, "calibration.yaml")
    write_yaml(calibration_config, config)
    calibration = calibrate_qm9(offline_assets.manifest_path, checkpoint, config_path=calibration_config,
                                attempt_id="calibration", device="cpu")
    config = load_config("configs/rollouts.yaml")
    config.update(num_rollouts=6, batch_size=2, snapshots_per_rollout=2,
                  output_dir=os.path.join(artifact_directory, "rollouts"))
    config["native"]["steps"] = 4
    rollout_config = os.path.join(artifact_directory, "rollouts.yaml")
    write_yaml(rollout_config, config)
    return offline_assets, checkpoint, calibration["manifest"], rollout_config


def test_rollout_shards_keep_groups_and_reject_overlap_or_mixed_targets(offline_labels):
    assets, checkpoint, calibration, config = offline_labels
    manifests = []
    for index in range(2):
        result = collect_rollouts(assets.manifest_path, checkpoint, calibration, policy="rho",
                                  config_path=config, attempt_id=f"shard-{index}",
                                  shard_index=index, num_shards=2, device="cpu")
        manifests.append(result["manifest"])
    protocol, rows = load_rollout_shards(manifests, policy="rho")
    assert len(rows["labels"]) == 12 and protocol["target"]["kind"] == "v_rho"
    train = set(rows["rollout_ids"][~rows["is_validation"]].tolist())
    val = set(rows["rollout_ids"][rows["is_validation"]].tolist())
    assert not train & val and len(train | val) == 6
    with pytest.raises(ValueError, match="Missing rollout shards"):
        load_rollout_shards([manifests[0]], policy="rho")
    with pytest.raises(ValueError, match="overlap"):
        load_rollout_shards([manifests[0], manifests[0]], policy="rho")
    with pytest.raises(ValueError, match="policy"):
        load_rollout_shards(manifests, policy="native")
    native = collect_rollouts(assets.manifest_path, checkpoint, calibration, policy="native",
                              config_path=config, attempt_id="native", device="cpu")
    native_protocol, _ = load_rollout_shards([native["manifest"]], policy="native")
    assert native_protocol["target"]["target_id"] != protocol["target"]["target_id"]


def test_verifier_checkpoint_resume_and_target_binding(offline_labels, artifact_directory):
    assets, checkpoint, calibration_manifest, config = offline_labels
    rollout = collect_rollouts(assets.manifest_path, checkpoint, calibration_manifest, policy="rho",
                               config_path=config, attempt_id="training-labels", device="cpu")
    training = load_config("configs/verifier_training.yaml")
    training["model"].update(hidden_dim=16, num_layers=1, num_heads=2)
    training["training"].update(epochs=2, batch_size=8, save_every_epochs=1)
    training["output_dir"] = os.path.join(artifact_directory, "verifier")
    config_path = os.path.join(artifact_directory, "verifier.yaml")
    write_yaml(config_path, training)
    first = train_verifier([rollout["manifest"]], policy="rho", attempt_id="full",
                           config_path=config_path, device="cpu")
    middle = os.path.join(os.path.dirname(first["manifest"]), "checkpoints", "epoch-0001", "manifest.json")
    resumed = train_verifier([rollout["manifest"]], policy="rho", attempt_id="resumed",
                             config_path=config_path, resume_manifest=middle, device="cpu")
    first_weights = load_file(os.path.join(os.path.dirname(first["checkpoint_manifest"]), "model.safetensors"))
    resumed_weights = load_file(os.path.join(os.path.dirname(resumed["checkpoint_manifest"]), "model.safetensors"))
    assert all(torch.equal(first_weights[key], resumed_weights[key]) for key in first_weights)
    calibration = load_calibration(calibration_manifest)
    reward = QM9Reward.from_assets(assets.manifest_path, calibration_manifest=calibration_manifest,
                                   calibration_reference_id=calibration.reference_id,
                                   calibration_policy_id=calibration.policy_id,
                                   checkpoint_sha256=calibration.checkpoint_sha256)
    target = ValueTarget(**first["target"])
    kwargs = dict(reward=reward, expected_target=target, backbone_checkpoint_id=calibration.checkpoint_sha256,
                  tokenizer_id=assets.tokenizer_sha256, device="cpu")
    adapter = NeuralVerifier.from_checkpoint(first["checkpoint_manifest"], **kwargs)
    assert adapter.target == target
    kwargs["expected_target"] = replace(target, target_id="different-target")
    with pytest.raises(ValueError, match="mismatch"):
        NeuralVerifier.from_checkpoint(first["checkpoint_manifest"], **kwargs)
    assert read_json(first["manifest"])["status"] == "completed"


def test_independent_masks_population_regression_is_native_conditional(native_case, enumerate_native):
    import math
    from itertools import product
    from src.qm9.rollouts import apply_mask_pattern, independent_mask_patterns

    reference, reward = native_case
    paths = enumerate_native(reference)
    # Integrate the ACTUAL chosen mask distribution, not just a protocol name.
    joint = {}
    for _, terminal, p in paths:
        for pattern in product((False, True), repeat=reference.rules.sequence_length):
            k = sum(pattern)
            p_mask = 1 / (reference.rules.sequence_length + 1) / math.comb(reference.rules.sequence_length, k)
            state = apply_mask_pattern(terminal, pattern)
            n, d = joint.get(state, (0., 0.))
            joint[state] = (n + p * p_mask * reward.evaluate(terminal, require_binary=True).reward,
                            d + p * p_mask)
    for query, (n, d) in joint.items():
        direct_d = sum(p for _, leaf, p in paths if query.accepts_completion(leaf))
        direct_n = sum(p * reward.evaluate(leaf, require_binary=True).reward
                       for _, leaf, p in paths if query.accepts_completion(leaf))
        assert n / d == pytest.approx(direct_n / direct_d)
    # The API chooses masks from length + RNG alone, before inspecting Y.
    a = independent_mask_patterns(reference.rules, count=100, rng=RNGStream(7))
    b = independent_mask_patterns(reference.rules, count=100, rng=RNGStream(7))
    assert a == b and set(a) == set(product((False, True), repeat=2))


def test_native_mc_counts_unpruned_failures_and_uncertainty(native_case, enumerate_native):
    from src.verifiers.interface import native_conditional_target
    from src.verifiers.monte_carlo import NativeConditionalMonteCarloVerifier

    reference, reward = native_case
    query = MaskedState((0, 2), reference.rules)
    verifier = NativeConditionalMonteCarloVerifier(reference, reward, draws=512,
                                                  min_denominator_hits=5, progress=False)
    tracker = BudgetTracker(base_sequence_limit=None)
    result = verifier.values_for_states([query], rng=RNGStream(19), budget=tracker)[0]
    assert result.target == native_conditional_target(reference, reward) and result.draws == 512
    assert result.cost == tracker.snapshot().total
    paths = enumerate_native(reference)
    d = sum(p for _, leaf, p in paths if query.accepts_completion(leaf))
    n = sum(p for _, leaf, p in paths if query.accepts_completion(leaf) and reward.evaluate(leaf, require_binary=True).reward)
    assert result.value is not None and result.interval[0] <= n / d <= result.interval[1]
    assert verifier.summaries[0].denominator > verifier.summaries[0].numerator
    refused = verifier.values_for_states([query], rng=RNGStream(19), budget=BudgetTracker(base_sequence_limit=1))[0]
    assert refused.value is None and refused.draws == 0


def test_native_mc_zero_support_and_leaf_anchoring(native_case_factory):
    from src.verifiers.monte_carlo import NativeConditionalMonteCarloVerifier

    reference, reward = native_case_factory(zero_second_two=True)
    verifier = NativeConditionalMonteCarloVerifier(reference, reward, draws=16,
                                                  min_denominator_hits=1, progress=False)
    results = verifier.values_for_states([MaskedState((0, 2), reference.rules), MaskedState((1, 1), reference.rules)],
                                         rng=RNGStream(2), budget=BudgetTracker(base_sequence_limit=None))
    assert results[0].value is None and "denominator" in results[0].error
    assert results[1].status == EstimateStatus.EXACT and results[1].value == 1.


def test_native_conditional_collection_group_split_training_and_checkpoint(offline_labels, artifact_directory):
    from src.io import read_checkpoint
    from src.types import TargetKind

    assets, checkpoint, calibration, _ = offline_labels
    settings = load_config("configs/native_conditional_rollouts.yaml")
    settings.update(num_rollouts=6, batch_size=2, snapshots_per_rollout=3,
                    output_dir=os.path.join(artifact_directory, "conditional-rollouts"))
    settings["native"]["steps"] = 2  # Match the offline BASE calibration exactly.
    config_path = os.path.join(artifact_directory, "conditional.yaml")
    write_yaml(config_path, settings)
    manifests = [collect_rollouts(assets.manifest_path, checkpoint, calibration, policy="native-conditional",
                                  config_path=config_path, attempt_id=f"conditional-{i}",
                                  shard_index=i, num_shards=2, device="cpu")["manifest"] for i in range(2)]
    protocol, data = load_rollout_shards(manifests, policy="native-conditional")
    assert protocol["target"]["kind"] == TargetKind.REFERENCE_CONDITIONAL and protocol["snapshot_steps"] == []
    assert len(data["labels"]) == 18
    assert not set(data["rollout_ids"][data["is_validation"]].tolist()) & set(data["rollout_ids"][~data["is_validation"]].tolist())
    stored = read_checkpoint(os.path.join(os.path.dirname(manifests[0]), "rollouts.pt"))
    assert torch.all(stored["snapshot_steps"] == -1) and stored["mask_patterns"].dtype == torch.bool
    with pytest.raises(ValueError, match="policy"):
        load_rollout_shards(manifests, policy="native")
    with pytest.raises(ValueError, match="overlap"):
        load_rollout_shards([manifests[0], manifests[0]], policy="native-conditional")
    training = load_config("configs/verifier_training.yaml")
    training["model"].update(hidden_dim=16, num_layers=1, num_heads=2)
    training["training"].update(epochs=1, batch_size=8, save_every_epochs=1)
    training["output_dir"] = os.path.join(artifact_directory, "conditional-verifier")
    train_config = os.path.join(artifact_directory, "conditional-train.yaml")
    write_yaml(train_config, training)
    result = train_verifier(manifests, policy="native-conditional", config_path=train_config,
                            attempt_id="conditional-training", device="cpu")
    locked = load_calibration(calibration)
    reward = QM9Reward.from_assets(assets.manifest_path, calibration_manifest=calibration,
                                   calibration_reference_id=locked.reference_id, calibration_policy_id=locked.policy_id,
                                   checkpoint_sha256=locked.checkpoint_sha256)
    target = ValueTarget(**protocol["target"])
    kwargs = dict(reward=reward, expected_target=target, backbone_checkpoint_id=locked.checkpoint_sha256,
                  tokenizer_id=assets.tokenizer_sha256, device="cpu")
    assert NeuralVerifier.from_checkpoint(result["checkpoint_manifest"], **kwargs).target == target
    kwargs["expected_target"] = replace(target, kind=TargetKind.NATIVE_ROLLOUT)
    with pytest.raises(ValueError, match="mismatch"):
        NeuralVerifier.from_checkpoint(result["checkpoint_manifest"], **kwargs)
    settings["native"]["steps"] = 4
    write_yaml(config_path, settings)
    with pytest.raises(ValueError, match="locked calibration"):
        collect_rollouts(assets.manifest_path, checkpoint, calibration, policy="native-conditional",
                          config_path=config_path, attempt_id="wrong-reference", device="cpu")
