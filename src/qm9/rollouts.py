# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kijung Jeon

"""Independent BASE trajectories, grouped snapshots, and immutable rollout shards.

Native snapshots follow upstream QM9 harness/task selection (MIT, copyright
2026 Kijung Jeon). Group splitting and rho labels are project additions.
"""

import math
import os
import platform
from types import SimpleNamespace
from dataclasses import asdict
from importlib.metadata import version

import torch
from tqdm.auto import tqdm

from src.budget import BudgetTracker
from src.config import load_config, project_path
from src.io import (RunArtifacts, file_hash, payload_hash, read_checkpoint,
                    read_completed_manifest, read_json, write_checkpoint, write_json, write_yaml)
from src.models.backbone import QM9Backbone
from src.qm9.dataset import load_prepared_qm9
from src.qm9.reward import QM9Reward, load_calibration
from src.randomness import RNGPool
from src.sampling.completion import RhoCompletion
from src.sampling.native import NativeMDLM
from src.sampling.native_reference import NativeReference
from src.types import CostPhase, MaskedState, TargetKind, TokenRules, ValueTarget, require_count
from src.verifiers.interface import native_conditional_target, value_target


MASK_DISTRIBUTION = "uniform_mask_count_then_uniform_subset_v1"
CONDITIONAL_LABELS = "native_root_terminal_independent_masks_v1"


def independent_mask_patterns(rules, *, count, rng):
    """No terminal or reward input: every mask subset has positive probability."""
    require_count(count, "masks per terminal", positive=True)
    patterns = []
    for _ in range(count):
        size = rng.python.randrange(rules.sequence_length + 1)
        positions = set(rng.python.sample(range(rules.sequence_length), size))
        patterns.append(tuple(i in positions for i in range(rules.sequence_length)))
    return tuple(patterns)


def masks_for_rollout(settings, rules, rollout_id):
    pool = RNGPool(settings["seed"], "qm9-native-conditional-masks-v1")
    return independent_mask_patterns(rules, count=settings["snapshots_per_rollout"],
                                     rng=pool.stream(f"terminal-{rollout_id}"))


def apply_mask_pattern(terminal, pattern):
    if not terminal.is_complete or len(pattern) != terminal.rules.sequence_length:
        raise ValueError("Posthoc masking requires a complete terminal and a full mask pattern")
    return MaskedState(tuple(terminal.rules.mask_id if masked else token
                             for masked, token in zip(pattern, terminal.token_ids)), terminal.rules)


def snapshot_indices(steps, count):
    """Upstream pre-transition interior grid; retain any rounding multiplicity."""
    require_count(steps, "rollout steps", positive=True)
    require_count(count, "snapshots per rollout", positive=True)
    if count == 1:
        return (steps // 2,)
    return tuple(round((steps - 1) * i / (count + 1)) for i in range(1, count + 1))


def validation_rollout_ids(count, fraction, seed):
    require_count(count, "rollout count", positive=True)
    if count < 2 or not 0 < fraction < 1:
        raise ValueError("At least two rollout groups and a validation fraction in (0, 1) are required")
    n_validation = min(count - 1, max(1, math.ceil(count * fraction)))
    stream = RNGPool(seed, "qm9-verifier-group-split-v1").stream("split")
    return set(int(index) for index in stream.numpy.permutation(count)[:n_validation])


def collect_rollouts(assets_manifest, checkpoint_manifest, calibration_manifest, *, policy,
                     attempt_id, config_path="configs/rollouts.yaml", run_id="qm9-rollouts",
                     shard_index=0, num_shards=1, device="cuda", execution_metadata=None):
    if policy not in ("native", "rho", "native-conditional"):
        raise ValueError("Rollout policy must be native, rho or native-conditional")
    config_path = project_path(config_path)
    config = load_config(config_path)
    for name in ("num_rollouts", "batch_size", "snapshots_per_rollout"):
        require_count(config[name], name, positive=True)
    require_count(config["seed"], "rollout seed")
    require_count(num_shards, "num_shards", positive=True)
    require_count(shard_index, "shard_index")
    if not shard_index < num_shards <= config["num_rollouts"]:
        raise ValueError("Require shard_index < num_shards <= num_rollouts")
    validation = validation_rollout_ids(config["num_rollouts"], config["validation_fraction"], config["split_seed"])
    assets = load_prepared_qm9(assets_manifest)
    source = read_completed_manifest(checkpoint_manifest)
    if source.get("identities", {}).get("assets_manifest_sha256") != file_hash(assets_manifest):
        raise ValueError("Backbone was trained with a different assets manifest")
    backbone = QM9Backbone.from_checkpoint(checkpoint_manifest, device=device, precision="bf16")
    calibration = load_calibration(calibration_manifest)
    reward = QM9Reward.from_assets(
        assets_manifest, calibration_manifest=calibration_manifest,
        calibration_reference_id=calibration.reference_id, calibration_policy_id=calibration.policy_id,
        checkpoint_sha256=backbone.checkpoint_id,
    )
    if backbone.token_rules != assets.token_rules or backbone.tokenizer_id != assets.tokenizer_sha256:
        raise ValueError("Rollout backbone differs from prepared tokenizer")
    backbone.model.requires_grad_(False)
    conditional = policy == "native-conditional"
    if conditional and (config.get("label_protocol") != CONDITIONAL_LABELS
                        or config.get("mask_distribution") != MASK_DISTRIBUTION):
        raise ValueError("Native-conditional collection needs the independent-mask config")
    generator = (NativeMDLM(backbone, **config["native"]) if policy != "rho"
                 else RhoCompletion(backbone, **config["rho"]))
    steps = generator.steps if policy != "rho" else assets.token_rules.sequence_length
    snapshots = () if conditional else snapshot_indices(steps, config["snapshots_per_rollout"])
    # Native regression learns E[label | sampled state], marginalizing snapshot
    # time. It is not a fresh 128-step completion from every observed state.
    label_protocol = ({"law": "native_trajectory_terminal_conditional",
                       "snapshot_steps": snapshots, "time_input": False} if policy == "native" else None)
    if conditional:
        reference = NativeReference(generator)
        if (calibration.reference_id != reference.reference_id or calibration.policy_id != reference.policy_id):
            raise ValueError("Native control must use the locked calibration BASE reference/settings")
        target = native_conditional_target(reference, reward)
        label_protocol = {"law": CONDITIONAL_LABELS, "mask_distribution": MASK_DISTRIBUTION,
                          "masks_per_terminal": config["snapshots_per_rollout"], "time_input": False}
    else:
        target = value_target(generator, reward, kind=(TargetKind.NATIVE_ROLLOUT if policy == "native"
                                                      else TargetKind.RHO), label_protocol=label_protocol)
    protocol = {
        "policy": policy, "target": asdict(target), "label_protocol": label_protocol,
        "backbone_checkpoint_sha256": backbone.checkpoint_id, "tokenizer_sha256": backbone.tokenizer_id,
        "token_rules": asdict(assets.token_rules), "calibration_id": calibration.calibration_id,
        "assets_manifest_sha256": file_hash(assets_manifest),
        "calibration_manifest_sha256": file_hash(calibration_manifest),
        "settings": config, "num_shards": num_shards, "snapshot_steps": snapshots,
        "label": "binary_pass95", "sampling_precision": backbone.precision,
        "rho_training_support": "root_trajectory_prefix_states" if policy == "rho" else None,
    }
    if conditional:
        protocol["native_reference"] = reference.protocol
    collection_id = "qm9-rollouts:" + payload_hash(protocol)
    attempt = RunArtifacts.create(config["output_dir"], run_id=run_id + "-" + policy,
                                  attempt_id=attempt_id, config=config, target_id=target.target_id,
                                  reference_id=target.reference_id)
    attempt.manifest.update(task="collect_qm9_rollouts", collection_id=collection_id,
                            target=asdict(target), policy=policy, shard_index=shard_index,
                            num_shards=num_shards)
    budget = BudgetTracker(base_sequence_limit=None)
    pool = RNGPool(config["seed"], collection_id)
    rng = pool.stream(f"shard-{shard_index:05d}")
    rollout_ids = list(range(shard_index, config["num_rollouts"], num_shards))
    root = MaskedState((assets.token_rules.mask_id,) * assets.token_rules.sequence_length, assets.token_rules)
    state_rows, labels, groups, snapshot_steps, is_validation = [], [], [], [], []
    terminal_ids, terminal_rewards, terminal_valid, terminal_qeds, mask_rows = [], [], [], [], []
    try:
        write_yaml(attempt.path("config.yaml"), config)
        write_json(attempt.path("protocol.json"), protocol)
        write_json(attempt.path("environment.json"), {
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ("torch", "transformers", "numpy", "rdkit", "safetensors")},
            "device": str(backbone.device), "precision": backbone.precision,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(backbone.device) if backbone.device.type == "cuda" else None,
        })
        write_json(attempt.path("execution.json"), {
            "backend": "local", **(execution_metadata or {}), "shard_index": shard_index,
            "num_shards": num_shards, "checkpoint_manifest": checkpoint_manifest,
            "calibration_manifest": calibration_manifest, "assets_manifest": assets_manifest,
            "source_sha256": {path: file_hash(path) for path in (
                "src/qm9/rollouts.py", "src/sampling/native.py", "src/sampling/native_reference.py",
                "src/sampling/completion.py",
                "src/models/backbone.py", "src/qm9/reward.py", "src/verifiers/interface.py",
                "src/randomness.py")},
        })
        for name in ("config.yaml", "protocol.json", "environment.json", "execution.json"):
            attempt.register(name)
        with tqdm(total=len(rollout_ids), desc=f"Collect {policy} rollouts", unit="rollout") as progress:
            for start in range(0, len(rollout_ids), config["batch_size"]):
                ids = rollout_ids[start:start + config["batch_size"]]
                # A separate seed namespace, generated BEFORE seeing Y or tau(Y).
                masks = {i: masks_for_rollout(config, assets.token_rules, i) for i in ids} if conditional else {}
                kwargs = dict(rng=rng, budget=budget, phase=CostPhase.TRAINING, snapshot_steps=snapshots)
                batch = (generator.sample_batch([root] * len(ids), **kwargs) if policy != "rho"
                         else generator.complete_batch([root] * len(ids), **kwargs))
                if batch.exhausted or len(batch.states) != len(ids) or any(not s.is_complete for s in batch.states):
                    raise ValueError("Rollout shard contains unfinished/missing draws; no filtering or retry")
                for row, (rollout_id, leaf) in enumerate(zip(ids, batch.states)):
                    with budget.oracle_check(phase=CostPhase.TRAINING):
                        terminal = reward.evaluate(leaf, require_binary=True)
                    terminal_ids.append(leaf.token_ids)
                    terminal_rewards.append(terminal.reward)
                    terminal_valid.append(terminal.valid)
                    terminal_qeds.append(terminal.score)
                    rows = ([(apply_mask_pattern(leaf, pattern), -1, pattern) for pattern in masks[rollout_id]]
                            if conditional else [(batch.snapshots[step][row], step, None) for step in snapshots])
                    for state, step, pattern in rows:
                        if not state.accepts_completion(leaf):
                            raise ValueError("Snapshot changed an observed token before termination")
                        state_rows.append(state.token_ids)
                        labels.append(terminal.reward)
                        groups.append(rollout_id)
                        snapshot_steps.append(step)
                        is_validation.append(rollout_id in validation)
                        if conditional:
                            mask_rows.append(pattern)
                progress.update(len(ids))
        payload = {
            "state_ids": torch.tensor(state_rows, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "rollout_ids": torch.tensor(groups, dtype=torch.long),
            "snapshot_steps": torch.tensor(snapshot_steps, dtype=torch.long),
            "is_validation": torch.tensor(is_validation, dtype=torch.bool),
            "terminal_rollout_ids": torch.tensor(rollout_ids, dtype=torch.long),
            "terminal_ids": torch.tensor(terminal_ids, dtype=torch.long),
            "terminal_rewards": torch.tensor(terminal_rewards, dtype=torch.float32),
            "terminal_valid": torch.tensor(terminal_valid, dtype=torch.bool),
            "terminal_qeds": torch.tensor(terminal_qeds, dtype=torch.float32),
        }
        if conditional:
            payload["mask_patterns"] = torch.tensor(mask_rows, dtype=torch.bool)
        write_checkpoint(attempt.path("rollouts.pt"), payload,
                         metadata={"collection_id": collection_id, "target": asdict(target)})
        write_json(attempt.path("rng.json"), pool.state_dict())
        summary = {"manifest": attempt.path("manifest.json"), "collection_id": collection_id,
                   "target": asdict(target), "rollouts": len(rollout_ids), "snapshots": len(labels),
                   "validation_rollouts": sum(i in validation for i in rollout_ids),
                   "cost": asdict(budget.snapshot())}
        write_json(attempt.path("summary.json"), summary)
        required = ["rollouts.pt", "rollouts.pt.metadata.json", "rng.json", "summary.json"]
        for name in required:
            attempt.register(name)
        attempt.complete(required_artifacts=required)
        return summary
    except Exception as exc:
        if attempt.manifest["status"] == "running":
            attempt.fail(f"{type(exc).__name__}: {exc}")
        raise


def load_rollout_shards(manifest_paths, *, policy):
    """Reject mixed targets, overlapping groups, and row-level split leakage."""
    if not manifest_paths:
        raise ValueError("Provide at least one completed rollout shard")
    protocols, shards, seen = [], [], set()
    for path in manifest_paths:
        manifest = read_completed_manifest(path)
        if manifest.get("task") != "collect_qm9_rollouts" or manifest.get("policy") != policy:
            raise ValueError("Rollout shard task/policy mismatch")
        if not {"protocol.json", "rollouts.pt", "rollouts.pt.metadata.json"}.issubset(manifest["artifacts"]):
            raise ValueError("Rollout shard is missing required artifacts")
        directory = os.path.dirname(project_path(path))
        protocol = read_json(os.path.join(directory, "protocol.json"))
        target = ValueTarget(**protocol["target"])
        rules = TokenRules(**protocol["token_rules"])
        if policy == "native-conditional":
            declared = protocol["native_reference"]
            identity_backbone = SimpleNamespace(token_rules=rules,
                checkpoint_id=protocol["backbone_checkpoint_sha256"], tokenizer_id=protocol["tokenizer_sha256"],
                precision=protocol["sampling_precision"],
                time_conditioning=declared["settings"]["time_conditioning"])
            reference = NativeReference(NativeMDLM(identity_backbone, **protocol["settings"]["native"]))
            labels = {"law": CONDITIONAL_LABELS, "mask_distribution": MASK_DISTRIBUTION,
                      "masks_per_terminal": protocol["settings"]["snapshots_per_rollout"], "time_input": False}
            identity = {"kind": TargetKind.REFERENCE_CONDITIONAL, "reference": target.reference_id,
                        "policy": target.policy_id, "reward": target.reward_id,
                        "label_protocol": {"law": "native_root_terminal_given_observations_v1"}}
            if (protocol["label_protocol"] != labels or protocol["snapshot_steps"] != []
                    or protocol["settings"].get("label_protocol") != CONDITIONAL_LABELS
                    or protocol["settings"].get("mask_distribution") != MASK_DISTRIBUTION
                    or target.kind != TargetKind.REFERENCE_CONDITIONAL
                    or declared != reference.protocol or target.reference_id != reference.reference_id
                    or target.policy_id != reference.policy_id
                    or target.calibration_id != protocol["calibration_id"]
                    or target.reward_id != "pass95:" + protocol["calibration_id"]
                    or target.target_id != "value:" + payload_hash(identity)):
                raise ValueError("Malformed native conditional target/independent-mask protocol")
        if manifest.get("target") != asdict(target) or manifest["target_id"] != target.target_id:
            raise ValueError("Rollout target differs from its manifest")
        if manifest["num_shards"] != protocol["num_shards"] or protocol["policy"] != policy:
            raise ValueError("Shard partition/policy differs from its protocol")
        if manifest.get("collection_id") != "qm9-rollouts:" + payload_hash(protocol):
            raise ValueError("Rollout collection identity differs")
        if protocols and protocol != protocols[0]:
            raise ValueError("Do not mix rollout collections, targets, or split protocols")
        data = read_checkpoint(os.path.join(directory, "rollouts.pt"))
        count = len(data["labels"])
        if (data["state_ids"].shape != (count, rules.sequence_length)
                or data["state_ids"].dtype != torch.long
                or data["rollout_ids"].dtype != torch.long
                or data["snapshot_steps"].dtype != torch.long
                or data["is_validation"].dtype != torch.bool
                or any(data[name].shape != (count,) for name in
                       ("labels", "rollout_ids", "snapshot_steps", "is_validation"))
                or not torch.all((data["labels"] == 0) | (data["labels"] == 1))):
            raise ValueError("Malformed rollout snapshot tensors")
        if torch.any(data["state_ids"] < 0) or torch.any(data["state_ids"] >= rules.vocab_size):
            raise ValueError("Snapshot token outside prepared vocabulary")
        train_groups = set(data["rollout_ids"][~data["is_validation"]].tolist())
        val_groups = set(data["rollout_ids"][data["is_validation"]].tolist())
        groups = train_groups | val_groups
        if train_groups & val_groups or groups & seen:
            raise ValueError("Rollout groups overlap across split/shards")
        expected = set(range(manifest["shard_index"], protocol["settings"]["num_rollouts"], manifest["num_shards"]))
        if groups != expected or set(data["terminal_rollout_ids"].tolist()) != expected:
            raise ValueError("Shard rollout IDs differ from its declared partition")
        k = protocol["settings"]["snapshots_per_rollout"]
        _, group_counts = torch.unique(data["rollout_ids"], return_counts=True)
        if count != len(expected) * k or not torch.all(group_counts == k):
            raise ValueError("Shard snapshot count differs from its protocol")
        terminal_count = len(expected)
        if (data["terminal_ids"].shape != (terminal_count, rules.sequence_length)
                or data["terminal_ids"].dtype != torch.long
                or any(data[name].shape != (terminal_count,) for name in
                       ("terminal_rollout_ids", "terminal_rewards", "terminal_valid", "terminal_qeds"))):
            raise ValueError("Malformed rollout terminal tensors")
        expected_validation = validation_rollout_ids(protocol["settings"]["num_rollouts"],
                                                     protocol["settings"]["validation_fraction"],
                                                     protocol["settings"]["split_seed"])
        if val_groups != groups & expected_validation:
            raise ValueError("Shard split membership differs from the grouped split")
        terminal_lookup = {int(group): index for index, group in enumerate(data["terminal_rollout_ids"])}
        if policy == "native-conditional":
            if ("mask_patterns" not in data or data["mask_patterns"].shape != (count, rules.sequence_length)
                    or data["mask_patterns"].dtype != torch.bool or not torch.all(data["snapshot_steps"] == -1)):
                raise ValueError("Independent masks cannot be loaded as diffusion snapshots")
            for group in groups:
                rows = data["rollout_ids"] == group
                patterns = masks_for_rollout(protocol["settings"], rules, group)
                if not torch.equal(data["mask_patterns"][rows], torch.tensor(patterns, dtype=torch.bool)):
                    raise ValueError("Stored masks differ from the terminal-independent mask RNG")
        for row in range(count):
            terminal_index = terminal_lookup[int(data["rollout_ids"][row])]
            leaf = MaskedState(tuple(data["terminal_ids"][terminal_index].tolist()), rules)
            state = MaskedState(tuple(data["state_ids"][row].tolist()), rules)
            if policy == "native-conditional" and state != apply_mask_pattern(leaf, data["mask_patterns"][row]):
                raise ValueError("Posthoc masked state differs from its independent pattern")
            if not state.accepts_completion(leaf) or data["labels"][row] != data["terminal_rewards"][terminal_index]:
                raise ValueError("Snapshot/terminal label mismatch")
        seen.update(groups)
        protocols.append(protocol)
        shards.append(data)
    if seen != set(range(protocols[0]["settings"]["num_rollouts"])):
        raise ValueError("Missing rollout shards: training requires the complete declared collection")
    fields = ("state_ids", "labels", "rollout_ids", "is_validation")
    return protocols[0], {name: torch.cat([data[name] for data in shards]) for name in fields}
