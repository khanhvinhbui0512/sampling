"""T06: generate labeled snapshots and publish partitions without molecule leakage.

QM9's absorbing rollouts/interior snapshots are the default. The optional q_z
policy is recorded separately. Labels are terminal tau, with no reward-based
filtering. A rollout and every occurrence of its terminal molecule/string share
one identity partition, even across class conditions. T13 test states are not
used here. Runtime inputs are project artifacts and packaged benchmark assets.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from tqdm.auto import tqdm

from src.artifacts import ArtifactStore, Dependency, sha256_file
from src.completion import CompletionResult, LeftToRightCompletion, ReferenceCompletion
from src.config import Config, config_digest, project_path
from src.data import IDENTITY_VERSION, molecule_id
from src.mdlm import MDLMAdapter
from src.reward import RewardChecker
from src.schemas import GROUPS, ArtifactManifest, SmilesState, TaskResult, require_count
from src.tasks.common import TaskContext, run_task
from src.tasks.prepare_corpus import benchmark_dependency, checker_versions, commit_file
from src.tasks.train_mdlm import _device_precision
from src.tokenizer import SmilesTokenizer

ROLLOUT_VERSION = "smiles-rollouts-v1"


def model_overrides(*, training_report: str = "", checkpoint: str = "") -> list[str]:
    import json

    if training_report and checkpoint:
        raise ValueError("Choose --mdlm-report or --checkpoint, not both")
    if training_report:
        return ["rollouts.model.training_report=" + json.dumps(project_path(training_report)),
                "rollouts.model.checkpoint_manifest=null"]
    if checkpoint:
        return ["rollouts.model.checkpoint_manifest=" + json.dumps(project_path(checkpoint)),
                "rollouts.model.training_report=null"]
    return []


def validate_rollouts(config: Config) -> None:
    options = config["rollouts"]
    if options["version"] != ROLLOUT_VERSION or config["suite"] == "replication" or config.get("group") is not None:
        raise ValueError("T06 builds all three groups in primary or smoke")
    source = options["model"]
    references = [source["training_report"], source["checkpoint_manifest"]]
    if sum(value is not None for value in references) != 1:
        raise ValueError("T06 requires one --mdlm-report or --checkpoint (or its YAML setting)")
    for value in references:
        if value is not None:
            project_path(value)
    if source["device"] not in ("auto", "cpu", "cuda") or source["dtype"] not in ("bfloat16", "float32"):
        raise ValueError("Unsupported rollout device or dtype")
    if options["policy"] not in ("absorbing_reverse", "left_to_right"):
        raise ValueError("Unsupported rollout policy")
    for name in ("snapshots_per_rollout", "batch_size", "shard_rows"):
        require_count(options[name], "rollouts." + name, positive=True)
    if options["num_rollouts_per_group"] is not None:
        require_count(options["num_rollouts_per_group"], "num_rollouts_per_group", positive=True)
    reverse = options["reverse"]
    require_count(reverse["steps"], "reverse steps", positive=True)
    if (reverse["steps"] < options["snapshots_per_rollout"] + 2
            or not 0 < reverse["end_time"] < 1 or reverse["final_denoise"] is not True):
        raise ValueError("Reverse rollouts need distinct interior snapshots and final denoising")
    if (options["snapshot_selection"] != "evenly_spaced_interior_transitions"
            or options["shard_rows"] % options["snapshots_per_rollout"]):
        raise ValueError("Use interior snapshots and a shard_rows multiple of snapshots_per_rollout")
    partition = options["partition"]
    require_count(partition["seed"], "partition seed")
    if (not 0 < partition["validation_fraction"] < 1
            or partition["algorithm"] != "seeded_identity_rank"
            or partition["identity"] != "canonical_molecule_or_exact_invalid_token_string"
            or partition["keep_rollout_together"] is not True
            or partition["keep_molecule_together_across_groups"] is not True):
        raise ValueError("Verifier partitions must keep whole rollouts and repeated identities together")


def resolve_source(config: Config, store: ArtifactStore) -> tuple[str, dict[str, Dependency]]:
    source = config["rollouts"]["model"]
    dependencies = {"benchmark": benchmark_dependency(config, store)}
    if source["training_report"] is not None:
        report_path = source["training_report"]
        report = store.read_json(report_path)
        if report["suite"] != config["suite"] or report.get("test_used_for_selection") is not False:
            raise ValueError("T05 report has a different suite or checkpoint selection provenance")
        path = project_path(report["best_manifest"])
        dependencies["mdlm_report"] = Dependency(report_path)
    else:
        path = project_path(source["checkpoint_manifest"])
    tokenizer_path = config["data"]["tokenizer"]["metadata_path"]
    tokenizer = SmilesTokenizer.load(store, tokenizer_path)
    metadata = store.read_json(tokenizer_path)
    checkpoint = store.require_manifest(path, kind="mdlm_checkpoint", metadata={
        "suite": config["suite"], "groups": list(GROUPS), "tokenizer_fingerprint": tokenizer.fingerprint,
    })
    expected_inputs = checkpoint.metadata["dependency_hashes"]
    if (sha256_file(store.require_file(tokenizer_path)) != expected_inputs["tokenizer"]
            or metadata["provenance"]["corpus_sha256"] != checkpoint.metadata["corpus_sha256"]):
        raise ValueError("Rollout tokenizer/corpus differs from the trained MDLM")
    train_path = store.directory("data", "tokenized", "train", "manifest.json")
    dependencies.update({
        "mdlm": Dependency(path, manifest=True, kind="mdlm_checkpoint"),
        "tokenizer": Dependency(tokenizer_path, checksum=expected_inputs["tokenizer"]),
        "vocabulary": Dependency(metadata["vocabulary_path"], checksum=expected_inputs["vocabulary"]),
        "train_counts": Dependency(train_path, checksum=expected_inputs["train"]),
    })
    train = ArtifactManifest.from_dict(store.read_json(train_path))
    if (train.kind != "tokenized" or train.metadata["split"] != "train"
            or train.metadata["tokenizer_fingerprint"] != tokenizer.fingerprint
            or train.metadata["group_order"] != list(GROUPS)):
        raise ValueError("Rollout counts must come from this MDLM's train manifest")
    return path, dependencies


def rollout_dependencies(config: Config, store: ArtifactStore) -> Mapping[str, Dependency]:
    return resolve_source(config, store)[1]


def _interior_indices(transitions: int, count: int) -> list[int]:
    # Match QM9's rounded linspace(0, steps-1, k+2)[1:-1] for reverse rollouts.
    # Short q_z trajectories may repeat a snapshot, retaining exactly k labels.
    if transitions <= 1:
        return [0] * count
    return [round((transitions - 1) * index / (count + 1)) for index in range(1, count + 1)]


def _snapshots(result: CompletionResult, options: dict, root: SmilesState):
    count = options["snapshots_per_rollout"]
    if options["policy"] == "left_to_right":
        wanted = _interior_indices(len(result.branches), count)
        states = [result.branches[index].before if result.branches else root for index in wanted]
        return states, wanted, [-1.0] * count
    # Branches for one reverse transition share the same immutable pre-step state.
    observed = {}
    for branch in result.branches:
        if branch.phase == "reverse":
            observed.setdefault(branch.time, branch.before)
    steps = options["reverse"]["steps"]
    grid = torch.linspace(1.0, options["reverse"]["end_time"], steps + 1).tolist()
    wanted = _interior_indices(steps, count)
    return [observed.get(float(grid[index]), result.state) for index in wanted], wanted, [float(grid[index]) for index in wanted]


def terminal_identity(state: SmilesState, tokenizer: SmilesTokenizer, canonical: str | None) -> tuple[str, str, str]:
    if canonical is not None:
        return "molecule:" + molecule_id(canonical), "canonical_molecule", canonical
    # No token stripping, EOS truncation, or class-specific salt for invalid outputs.
    exact = "".join(tokenizer.vocabulary[token] for token in state.token_ids)
    digest = hashlib.sha256(("invalid-token-string-v1\0" + tokenizer.fingerprint + "\0" + exact).encode("utf-8")).hexdigest()
    return "invalid:" + digest, "invalid_token_string", exact


def _write_tensor(store: ArtifactStore, path: str, payload: dict) -> str:
    physical = store.path(path)
    os.makedirs(os.path.dirname(physical), exist_ok=True)
    torch.save(payload, physical + ".part")
    commit_file(physical + ".part", physical)
    return sha256_file(store.require_file(path))


def _raw_payload(context: TaskContext, policy, tokenizer, checker, group, start, size):
    options = context.config["rollouts"]
    roots = [SmilesState.fully_masked(tokenizer.max_length, group, tokenizer.special_tokens) for _ in range(size)]
    results = policy.complete_batch(roots)
    snapshots, snapshot_steps, snapshot_times = [], [], []
    rewards, identities, identity_types, identity_texts, diagnostics = [], [], [], [], []
    for root, result in zip(roots, results, strict=True):
        if not result.state.is_terminal:
            raise ValueError("Configured rollout did not produce a terminal state")
        selected, selected_steps, selected_times = _snapshots(result, options, root)
        with context.tracker.measure("rollout_reward"):
            checked = checker.evaluate_state(result.state, tokenizer)
        identity, identity_type, text = terminal_identity(result.state, tokenizer, checked.canonical_smiles)
        snapshots.append([state.token_ids for state in selected])
        snapshot_steps.append(selected_steps)
        snapshot_times.append(selected_times)
        rewards.append(checked.reward)
        identities.append(identity)
        identity_types.append(identity_type)
        identity_texts.append(text)
        diagnostics.append(checked.to_dict())
    ids = torch.tensor(snapshots, dtype=torch.long)
    return {
        "version": ROLLOUT_VERSION, "tokenizer_fingerprint": tokenizer.fingerprint,
        "state_ids": ids, "mask": ids == tokenizer.special_tokens.mask_id,
        "locked": torch.zeros_like(ids, dtype=torch.bool),
        "final_ids": torch.tensor([result.state.token_ids for result in results], dtype=torch.long),
        "group_ids": torch.full((size,), GROUPS.index(group), dtype=torch.long),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "snapshot_steps": torch.tensor(snapshot_steps, dtype=torch.int32),
        "snapshot_times": torch.tensor(snapshot_times, dtype=torch.float32),
        "rollout_ids": [f"{group}-{number:010d}" for number in range(start, start + size)],
        "identity_keys": identities, "identity_types": identity_types, "identity_texts": identity_texts,
        "diagnostics": diagnostics, "policy_hash": policy.policy_hash,
        "source_mdlm_sha256": context.identity.dependency_hashes["mdlm"],
        "reward_version": checker.version,
    }


def _partition(context: TaskContext, directory: str, raw: list[dict]) -> tuple[str, dict]:
    path = os.path.join(directory, "partitions.sqlite")
    temporary = context.store.path(path + ".part")
    if os.path.exists(temporary):
        os.remove(temporary)  # Only this run's uncommitted partition database.
    connection = sqlite3.connect(temporary)
    options = context.config["rollouts"]["partition"]
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE identities (identity_key TEXT PRIMARY KEY, identity_type TEXT NOT NULL, identity_text TEXT NOT NULL, sort_key TEXT NOT NULL, partition TEXT)")
        for shard in context.progress(raw, desc="Index rollout identities", unit="shard"):
            payload = torch.load(context.store.require_file(shard["path"], checksum=shard["sha256"]), map_location="cpu", weights_only=True)
            entries = [(key, kind, text, hashlib.sha256(f"{options['seed']}\0{key}".encode("utf-8")).hexdigest())
                       for key, kind, text in zip(payload["identity_keys"], payload["identity_types"], payload["identity_texts"], strict=True)]
            connection.executemany("INSERT OR IGNORE INTO identities VALUES (?, ?, ?, ?, NULL)", entries)
            connection.commit()
        identities = connection.execute("SELECT COUNT(*) FROM identities").fetchone()[0]
        if identities < 2:
            raise ValueError("At least two distinct terminal identities are needed for leakage-free train/validation; increase rollout diversity/count")
        validation_count = max(1, min(identities - 1, math.floor(identities * options["validation_fraction"])))
        # Rank globally over identity keys, never independently within each class.
        connection.execute("UPDATE identities SET partition='train'")
        connection.execute("UPDATE identities SET partition='validation' WHERE identity_key IN (SELECT identity_key FROM identities ORDER BY sort_key, identity_key LIMIT ?)", (validation_count,))
        connection.commit()
        counts = {split: connection.execute("SELECT COUNT(*) FROM identities WHERE partition=?", (split,)).fetchone()[0]
                  for split in ("train", "validation")}
        types = dict(connection.execute("SELECT identity_type, COUNT(*) FROM identities GROUP BY identity_type"))
    finally:
        connection.close()
    commit_file(temporary, context.store.path(path))
    return path, {"algorithm": options["algorithm"], "seed": options["seed"],
                  "validation_fraction": options["validation_fraction"], "identities": counts,
                  "identity_types": types, "total_identities": identities,
                  "identity_version": IDENTITY_VERSION, "path": path}


def _publish(context: TaskContext, directory: str, state: dict, tokenizer, policy) -> tuple[str, dict]:
    connection = sqlite3.connect(context.store.require_file(state["partition_path"], checksum=state["partition_sha256"]))
    connection.execute("PRAGMA query_only=ON")
    buffers = {split: [] for split in ("train", "validation")}
    counts = {split: {"rows": 0, "rollouts": 0, "positive_rows": 0, "group_rows": dict.fromkeys(GROUPS, 0), "shards": []}
              for split in buffers}
    files = [state["partition_path"], *[entry["path"] for entry in state["raw_shards"]]]
    options = context.config["rollouts"]
    k = options["snapshots_per_rollout"]

    def flush(split):
        incoming = buffers[split]
        if not incoming:
            return
        ids = torch.stack([item["states"] for item in incoming]).reshape(-1, tokenizer.max_length)
        final = torch.stack([item["final"] for item in incoming]).repeat_interleave(k, 0)
        groups = torch.tensor([item["group"] for item in incoming], dtype=torch.long).repeat_interleave(k)
        rewards = torch.tensor([item["reward"] for item in incoming], dtype=torch.float32).repeat_interleave(k)
        payload = {
            "version": ROLLOUT_VERSION, "split": split, "tokenizer_fingerprint": tokenizer.fingerprint,
            "special_tokens": tokenizer.special_tokens.to_dict(), "groups": list(GROUPS),
            "source_mdlm_sha256": context.identity.dependency_hashes["mdlm"], "policy_hash": policy.policy_hash,
            "reward_version": context.config["reward"]["version"], "state_ids": ids, "final_ids": final,
            "mask": ids == tokenizer.special_tokens.mask_id, "locked": torch.zeros_like(ids, dtype=torch.bool),
            "group_ids": groups, "rewards": rewards,
            "snapshot_indices": torch.arange(k, dtype=torch.int32).repeat(len(incoming)),
            "snapshot_steps": torch.stack([item["steps"] for item in incoming]).flatten(),
            "snapshot_times": torch.stack([item["times"] for item in incoming]).flatten(),
            "rollout_ids": [item["rollout"] for item in incoming for _ in range(k)],
            "identity_keys": [item["identity"] for item in incoming for _ in range(k)],
        }
        path = os.path.join(directory, split, f"shard_{len(counts[split]['shards']):05d}.pt")
        digest = _write_tensor(context.store, path, payload)
        counts[split]["shards"].append({"path": path, "sha256": digest, "rows": len(ids)})
        counts[split]["rows"] += len(ids)
        counts[split]["rollouts"] += len(incoming)
        counts[split]["positive_rows"] += int(rewards.sum())
        for index, group in enumerate(GROUPS):
            counts[split]["group_rows"][group] += int((groups == index).sum())
        files.append(path)
        incoming.clear()

    try:
        for raw in context.progress(state["raw_shards"], desc="Write snapshot partitions", unit="shard"):
            payload = torch.load(context.store.require_file(raw["path"], checksum=raw["sha256"]), map_location="cpu", weights_only=True)
            for row, identity in enumerate(payload["identity_keys"]):
                match = connection.execute("SELECT partition FROM identities WHERE identity_key=?", (identity,)).fetchone()
                if match is None or match[0] not in buffers:
                    raise ValueError("Rollout identity is missing its locked partition")
                split = match[0]
                buffers[split].append({"states": payload["state_ids"][row].clone(), "final": payload["final_ids"][row].clone(),
                    "group": int(payload["group_ids"][row]), "reward": float(payload["rewards"][row]),
                    "steps": payload["snapshot_steps"][row].clone(), "times": payload["snapshot_times"][row].clone(),
                    "rollout": payload["rollout_ids"][row], "identity": identity})
                if len(buffers[split]) * k == options["shard_rows"]:
                    flush(split)
        for split in buffers:
            flush(split)
    finally:
        connection.close()
    target = "reference_rollout_return" if options["policy"] == "absorbing_reverse" else "q_z_rollout_return"
    metadata = {
        "version": ROLLOUT_VERSION, "suite": context.identity.suite, "seed": context.identity.seed,
        "groups": list(GROUPS), "special_tokens": tokenizer.special_tokens.to_dict(),
        "tokenizer_fingerprint": tokenizer.fingerprint, "vocab_size": tokenizer.vocab_size, "max_length": tokenizer.max_length,
        "reward_version": context.config["reward"]["version"], "reward_mode": context.config["reward"]["mode"],
        "source_model_manifest": state["source_manifest"], "source_mdlm_sha256": context.identity.dependency_hashes["mdlm"],
        "dependency_hashes": dict(context.identity.dependency_hashes), "policy": dict(policy.metadata), "policy_hash": policy.policy_hash,
        "target_kind": target, "target_definition": "Terminal tau of the same trajectory; regress its conditional return given a snapshot and group.",
        "ideal_value_claim": False, "snapshot_selection": options["snapshot_selection"], "snapshots_per_rollout": k,
        "snapshot_step_meaning": "pre_reverse_transition" if options["policy"] == "absorbing_reverse" else "pre_reveal_transition",
        "snapshot_time_meaning": "scheduled_time" if options["policy"] == "absorbing_reverse" else "-1 means state-derived time",
        "terminal_snapshots": "After early completion, use the unchanged terminal state for later scheduled snapshots.",
        "raw_shards": state["raw_shards"], "partition": state["partition"], "splits": counts,
        "counts_per_group": state["targets"], "checker_versions": checker_versions(), "runtime": state["runtime"],
    }
    path = os.path.join(directory, "manifest.json")
    context.store.create_manifest(path, kind="rollout_dataset", config_hash=context.identity.config_hash, files=files, metadata=metadata)
    return path, metadata


def build_rollouts(context: TaskContext, *, on_checkpoint: Callable[[], None] | None = None) -> TaskResult:
    options, store = context.config["rollouts"], context.store
    source, _ = resolve_source(context.config, store)
    tokenizer = SmilesTokenizer.load(store, context.config["data"]["tokenizer"]["metadata_path"])
    train = ArtifactManifest.from_dict(store.read_json(store.directory("data", "tokenized", "train", "manifest.json")))
    targets = {group: options["num_rollouts_per_group"] or train.metadata["group_rows"]["kept"][group] for group in GROUPS}
    if any(type(count) is not int or count <= 0 for count in targets.values()):
        raise ValueError("Every group needs a positive rollout count from retained train rows")
    device, precision = _device_precision(options["model"])
    adapter = MDLMAdapter.from_checkpoint(store, source, tokenizer, device=str(device),
                                         dtype=precision or torch.float32, tracker=context.tracker)
    if options["policy"] == "absorbing_reverse":
        policy = ReferenceCompletion(adapter, seed=context.identity.seed, **options["reverse"])
    else:
        policy = LeftToRightCompletion(adapter, seed=context.identity.seed)
    checker = RewardChecker.from_config(context.config, tracker=context.tracker)
    directory = store.directory("data", "rollouts", context.identity.suite, f"seed_{context.identity.seed}", context.identity.run_id)
    actual = {"device": device.type, "precision": str(precision or torch.float32), "torch_version": str(torch.__version__)}
    state = context.resume_state or {"version": ROLLOUT_VERSION, "stage": "generation", "source_manifest": source,
        "targets": targets, "completed": dict.fromkeys(GROUPS, 0), "raw_shards": [], "policy_hash": policy.policy_hash,
        "runtime": actual, "positive_rollouts": dict.fromkeys(GROUPS, 0)}
    if (state["version"] != ROLLOUT_VERSION or state["source_manifest"] != source or state["targets"] != targets
            or state["policy_hash"] != policy.policy_hash or state["runtime"] != actual):
        raise ValueError("Resume rollout source, policy, counts, or runtime changed")
    if context.training_state is not None:
        policy.restore_rng_state(context.training_state["policy_rng"])
    elif state["raw_shards"] or any(state["completed"].values()):
        raise ValueError("Resume is missing the rollout policy RNG state")

    def checkpoint():
        context.save_checkpoint(state, training_state={"policy_rng": policy.rng_state()})
        if on_checkpoint is not None:
            on_checkpoint()

    for entry in state["raw_shards"]:
        store.require_file(entry["path"], checksum=entry["sha256"])
    if state["stage"] == "generation":
        checkpoint()
        for group in GROUPS:
            with context.progress(total=targets[group], initial=state["completed"][group], desc=f"Rollouts {group}", unit="rollout") as progress:
                while state["completed"][group] < targets[group]:
                    start = state["completed"][group]
                    size = min(options["batch_size"], targets[group] - start)
                    payload = _raw_payload(context, policy, tokenizer, checker, group, start, size)
                    path = os.path.join(directory, "raw", f"batch_{len(state['raw_shards']):06d}.pt")
                    digest = _write_tensor(store, path, payload)
                    state["raw_shards"].append({"path": path, "sha256": digest, "rollouts": size, "group": group})
                    state["completed"][group] += size
                    state["positive_rollouts"][group] += int(payload["rewards"].sum())
                    context.tracker.event("rollout_batch", group=group, completed=state["completed"][group], positive=int(payload["rewards"].sum()))
                    checkpoint()
                    progress.update(size)
            context.log(f"{group}: rollouts={state['completed'][group]}; positive={state['positive_rollouts'][group]}")
        state["partition_path"], state["partition"] = _partition(context, directory, state["raw_shards"])
        state["partition_sha256"] = sha256_file(store.require_file(state["partition_path"]))
        state["stage"] = "partitioned"
        checkpoint()
    if state["stage"] == "complete":
        store.require_file(state["manifest"], checksum=state["manifest_sha256"])
        manifest = store.require_manifest(state["manifest"], kind="rollout_dataset", config_hash=context.identity.config_hash)
        path, metadata = state["manifest"], manifest.metadata
    elif state["stage"] == "partitioned":
        path, metadata = _publish(context, directory, state, tokenizer, policy)
        state.update({"stage": "complete", "manifest": path,
                      "manifest_sha256": sha256_file(store.require_file(path))})
        checkpoint()
    else:
        raise ValueError("Unknown rollout resume phase")
    report = context.output_path("rollouts.json")
    store.write_json(report, {"manifest": path, "target_kind": metadata["target_kind"],
        "counts_per_group": targets, "positive_rollouts": state["positive_rollouts"],
        "partition": metadata["partition"], "splits": metadata["splits"], "test_used": False})
    return TaskResult(f"rollouts={sum(targets.values())}; snapshots={sum(part['rows'] for part in metadata['splits'].values())}; manifest={path}", artifacts=(path, report))


def _arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mdlm-report", help="Project-relative T05 training.json; use its best-validation checkpoint.")
    group.add_argument("--checkpoint", help="Project-relative MDLM checkpoint manifest.")


def execute(argv: Sequence[str] | None = None, *, execution: str = "local", on_checkpoint: Callable[[], None] | None = None):
    return run_task("T06", lambda context: build_rollouts(context, on_checkpoint=on_checkpoint), argv=argv, execution=execution,
        description="Build grouped SMILES rollout snapshots.", default_config="configs/rollouts.yaml",
        validator=validate_rollouts, dependencies=rollout_dependencies, add_arguments=_arguments,
        argument_overrides=lambda arguments: model_overrides(training_report=arguments.mdlm_report or "", checkpoint=arguments.checkpoint or ""))


if __name__ == "__main__":
    try:
        record = execute()
        if record.status != "completed":
            raise SystemExit(1)
    except KeyboardInterrupt:
        tqdm.write("T06: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"T06: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
