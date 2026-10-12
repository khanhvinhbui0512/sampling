"""T13: independent masked states, policy-specific MC and measured value queries.

State identities are partitioned before evaluation, independently of run seeds.
Both providers receive the same allowances, not a claim of equal actual cost.
Finite MC references retain uncertainty and never become an ideal V* label.
Each query is atomic for resume; completed shards are hash-checked before merge.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from tqdm.auto import tqdm

from src.artifacts import ArtifactStore, Dependency, _atomic_file, sha256_file
from src.completion import LeftToRightCompletion, ReferenceCompletion, normalize_eos_suffix
from src.config import Config, config_digest
from src.data import IDENTITY_VERSION, molecule_id, read_parquet, write_parquet
from src.mdlm import MDLMAdapter
from src.reference import q_z_reference, rollout_reference
from src.reward import RewardChecker
from src.schemas import GROUPS, SmilesState, TaskResult, require_count, require_identifier
from src.tasks.build_tokenizer import SHARD_VERSION
from src.tasks.common import TaskContext, run_task
from src.tasks.generation import add_arguments, mdlm_source, source_overrides, verifier_source
from src.tasks.prepare_benchmark import validate_data
from src.tasks.prepare_corpus import benchmark_dependency, checker_versions
from src.tasks.train_mdlm import TokenShards, _device_precision
from src.tasks.train_verifier import SnapshotShards
from src.tokenizer import SmilesTokenizer
from src.tracking import BudgetExceeded, CostTracker
from src.values import CARSValueProvider, LearnedValueProvider, reward_context
from src.verifier import VerifierAdapter

VALUE_EVAL_VERSION = "smiles-value-eval-v1"


def _validate_budget(settings: Mapping[str, Any]) -> None:
    require_count(settings["max_model_nfe"], "query model NFE cap", positive=True)
    seconds = settings["max_compute_seconds"]
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Every query needs a finite positive compute-time allowance")


def validate_value_eval(config: Config) -> None:
    validate_data(config)
    options = config["value_eval"]
    if (options["version"] != VALUE_EVAL_VERSION or config["suite"] == "replication" or config.get("group") is not None
            or (options["purpose"], options["split"]) not in (("tuning", "validation"), ("final", "test"))
            or options["masking"] != "uniform_fixed_count_over_full_width"
            or options["exclude_verifier_identities"] is not True or options["exclude_verifier_states"] is not True
            or type(options["require_q_trained_verifier"]) is not bool):
        raise ValueError("T13 requires independent all-group validation/tuning or test/final states")
    # The source flags deliberately match T08/T10/T11.
    for source in (config["generation"]["model"], config["value"]["learned"]):
        if sum(source[name] is not None for name in ("training_report", "checkpoint_manifest")) != 1:
            raise ValueError("Supply one MDLM source and one verifier source")
    model = config["generation"]["model"]
    if model["device"] not in ("auto", "cpu", "cuda") or model["dtype"] not in ("bfloat16", "float32"):
        raise ValueError("Unsupported evaluation device/dtype")
    require_count(config["value"]["learned"]["batch_size"], "verifier inference batch size", positive=True)
    for name in ("molecules_per_group", "repetitions", "calibration_bins", "shard_rows", "checkpoint_every", "log_every"):
        require_count(options[name], "value_eval." + name, positive=True)
    require_count(options["state_partition_seed"], "state partition seed")
    if not 0 < options["validation_state_fraction"] < 1:
        raise ValueError("State identities need a nonempty tuning/final partition")
    fractions, edges = options["mask_fractions"], options["mask_fraction_edges"]
    if (not isinstance(fractions, list) or not fractions or len(set(fractions)) != len(fractions)
            or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1 for value in fractions)
            or len(edges) < 2 or edges[0] != 0 or edges[-1] != 1
            or any(not a < b for a, b in zip(edges[:-1], edges[1:], strict=True))):
        raise ValueError("Invalid mask fractions or reporting bins")
    cars, reference = options["cars"], options["reference"]
    for name in ("m", "trie_max_nodes"):
        require_count(cars[name], "CARS " + name, positive=True)
    require_count(cars["cache_capacity"], "CARS cache capacity")
    if cars["cache_capacity"] != 0:
        raise ValueError("Measured CARS queries must be fresh, with caching disabled")
    budgets = options["provider_budgets"]
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("Configure at least one common provider allowance")
    for budget in budgets:
        require_identifier(budget["name"], "provider budget name")
        if budget["name"] == "reference":
            raise ValueError("reference is reserved for the independent MC allowance")
        _validate_budget(budget)
    if len({budget["name"] for budget in budgets}) != len(budgets):
        raise ValueError("Provider budget names must be unique")
    for name in ("m", "batch_size"):
        require_count(reference[name], "reference " + name, positive=True)
    _validate_budget(reference)
    reverse = reference["rollout"]
    require_count(reverse["steps"], "reference reverse steps", positive=True)
    if (not 0 < reverse["end_time"] < 1 or reverse["final_denoise"] is not True
            or not 0 < reference["maximum_ci_half_width"] < 1):
        raise ValueError("Invalid reference stopping/uncertainty settings")


def evaluation_dependencies(config: Config, store: ArtifactStore) -> dict[str, Dependency]:
    """Follow the selected verifier back to its immutable T06 leakage index."""
    checker_versions()
    mdlm_path, dependencies = mdlm_source(config, store)
    model_hash = sha256_file(store.require_file(mdlm_path))
    verifier_path, verifier_dependencies = verifier_source(config, store, mdlm_sha256=model_hash)
    dependencies.update(verifier_dependencies)
    verifier = store.require_manifest(verifier_path, kind="verifier_checkpoint")
    rollout_path = verifier.metadata["rollout_manifest"]
    rollout = store.require_manifest(rollout_path, kind="rollout_dataset", metadata={
        "suite": config["suite"], "source_mdlm_sha256": model_hash,
        "reward_version": config["reward"]["version"], "reward_mode": config["reward"]["mode"],
        "policy_hash": verifier.metadata["policy_hash"], "target_kind": verifier.metadata["target_kind"],
    })
    if sha256_file(store.require_file(rollout_path)) != verifier.metadata["dependency_hashes"]["rollouts"]:
        raise ValueError("Verifier's training rollout manifest changed")
    benchmark = benchmark_dependency(config, store)
    if (rollout.metadata["checker_versions"] != checker_versions()
            or rollout.metadata["dependency_hashes"]["benchmark"] != sha256_file(store.require_file(benchmark.path))):
        raise ValueError("Evaluation reward/checker differs from verifier training")
    tokenizer = SmilesTokenizer.load(store, config["data"]["tokenizer"]["metadata_path"])
    partition = rollout.metadata["partition"]["path"]
    if partition not in rollout.files:
        raise ValueError("Rollout manifest must cover the identity leakage index")
    corpus_hash = store.read_json(config["data"]["tokenizer"]["metadata_path"])["provenance"]["corpus_sha256"]
    split = config["value_eval"]["split"]
    dependencies.update({
        "benchmark": benchmark,
        "verifier_rollouts": Dependency(rollout_path, manifest=True, kind="rollout_dataset"),
        "verifier_identities": Dependency(partition, checksum=rollout.files[partition]),
        "corpus": Dependency(store.directory("data", "corpus", "manifest.json"), checksum=corpus_hash,
                             manifest=True, kind="corpus"),
        "state_source": Dependency(store.directory("data", "tokenized", split, "manifest.json"),
            manifest=True, kind="tokenized", metadata={"version": SHARD_VERSION, "split": split,
                "tokenizer_fingerprint": tokenizer.fingerprint, "corpus_sha256": corpus_hash,
                "max_length": tokenizer.max_length, "vocab_size": tokenizer.vocab_size, "group_order": list(GROUPS)}),
    })
    return dependencies


def _seed(seed: int, *parts: Any) -> int:
    return int(config_digest({"seed": seed, "parts": parts})[:16], 16) % (2**63-1)


def state_identity(state: SmilesState, tokenizer: SmilesTokenizer) -> str:
    # The verifier sees token IDs and group, never source molecule IDs or locks.
    return config_digest({"tokens": state.token_ids, "group": state.group, "tokenizer": tokenizer.fingerprint})


def state_purpose(state_id: str, options: Mapping[str, Any]) -> str:
    rank = _seed(options["state_partition_seed"], "state-purpose", state_id) / (2**63-1)
    return "tuning" if rank < options["validation_state_fraction"] else "final"


def _save_tensor(store: ArtifactStore, path: str, payload: Mapping[str, Any]) -> str:
    with _atomic_file(store.path(path)) as handle:
        torch.save(dict(payload), handle)
    return sha256_file(store.require_file(path))


def build_states(context: TaskContext, tokenizer: SmilesTokenizer) -> dict[str, Any]:
    """Bounded seeded selection, followed by exact snapshot-overlap exclusion."""
    store, config, options = context.store, context.config, context.config["value_eval"]
    verifier_path = verifier_source(config, store, mdlm_sha256=context.identity.dependency_hashes["mdlm"])[0]
    verifier = store.require_manifest(verifier_path, kind="verifier_checkpoint")
    rollouts = store.require_manifest(verifier.metadata["rollout_manifest"], kind="rollout_dataset")
    index_path = store.require_file(rollouts.metadata["partition"]["path"])
    connection = sqlite3.connect("file:" + index_path + "?mode=ro", uri=True)
    source = TokenShards(store, options["split"], tokenizer, cache_size=1)
    chosen: dict[str, dict[str, tuple]] = {group: {} for group in GROUPS}
    counts: Counter[str] = Counter()
    try:
        for number in context.progress(range(len(source.shards)), desc="Select independent molecules", unit="shard"):
            payload = source.load_shard(number)
            for row, identity in enumerate(payload["molecule_ids"]):
                if int(payload["augmentation_indices"][row]) != 0:
                    continue
                counts["canonical_source_rows"] += 1
                if identity != molecule_id(payload["canonical_smiles"][row]):
                    raise ValueError("State-source molecule identity differs from its canonical SMILES")
                if connection.execute("SELECT 1 FROM identities WHERE identity_key=?", ("molecule:" + identity,)).fetchone():
                    counts["excluded_verifier_identities"] += 1
                    continue
                for group_index, group in enumerate(GROUPS):
                    if not payload["group_mask"][row, group_index]:
                        continue
                    rank = _seed(config["seed"], "molecule-selection", group, identity)
                    selected = chosen[group]
                    selected[identity] = (rank, payload["token_ids"][row].tolist(), source.shards[number]["path"], row)
                    if len(selected) > options["molecules_per_group"]:
                        worst = max(selected, key=lambda key: (selected[key][0], key))
                        del selected[worst]
    finally:
        connection.close()
    candidates = {}
    for group, selected in chosen.items():
        for identity, (_, tokens, source_path, source_row) in sorted(selected.items()):
            clean = SmilesState(tuple(tokens), (False,) * len(tokens), group, tokenizer.special_tokens)
            for fraction in options["mask_fractions"]:
                generator = torch.Generator(device="cpu").manual_seed(_seed(config["seed"], "mask", identity, group, fraction))
                positions = torch.randperm(len(tokens), generator=generator)[:round(fraction*len(tokens))].tolist()
                state = clean.remask(positions)
                key = state_identity(state, tokenizer)
                if state_purpose(key, options) != options["purpose"]:
                    counts["excluded_other_purpose"] += 1
                    continue
                origin = {"molecule_id": identity, "shard": source_path, "row": source_row, "requested_mask_fraction": fraction}
                if key in candidates:
                    candidates[key]["origins"].append(origin)
                    counts["duplicate_states"] += 1
                else:
                    candidates[key] = {"state_id": key, "state": state.to_dict(), "origins": [origin],
                        "mask_fraction": sum(state.mask)/len(tokens),
                        "effective_mask_fraction": sum(normalize_eos_suffix(state).mask)/len(tokens),
                        "terminal": state.is_terminal, "group": group}
    # Even a shared fully masked root is excluded if T07 saw that exact input.
    overlaps = set()
    for split in ("train", "validation"):
        snapshots = SnapshotShards(store, rollouts, split, tokenizer, cache_size=1)
        for number in context.progress(range(len(snapshots.shards)), desc="Exclude verifier " + split, unit="shard"):
            payload = snapshots.load_shard(number)
            for tokens, group_id in zip(payload["state_ids"].tolist(), payload["group_ids"].tolist(), strict=True):
                state = SmilesState(tuple(tokens), tuple(token == tokenizer.special_tokens.mask_id for token in tokens),
                                    GROUPS[group_id], tokenizer.special_tokens)
                key = state_identity(state, tokenizer)
                if key in candidates:
                    overlaps.add(key)
    rows = [candidates[key] for key in sorted(candidates.keys()-overlaps)]
    counts["excluded_verifier_states"] = len(overlaps)
    return {"version": VALUE_EVAL_VERSION, "purpose": options["purpose"], "split": options["split"],
            "seed": config["seed"], "tokenizer": tokenizer.fingerprint,
            "dependency_hashes": dict(context.identity.dependency_hashes), "states": rows,
            "selection_counts": dict(counts), "molecules_per_group": {group: len(value) for group, value in chosen.items()},
            "states_per_group": dict(Counter(row["group"] for row in rows)),
            "masking": options["masking"], "identity_version": IDENTITY_VERSION,
            "state_partition_seed": options["state_partition_seed"], "validation_state_fraction": options["validation_state_fraction"],
            "used_for_training": False, "used_for_selection": options["purpose"] == "tuning"}


class QueryTracker(CostTracker):
    """Own actual calls once; fold measured work into the run after the query."""

    def __init__(self, parent: CostTracker, budget: Mapping[str, Any], *, job_id: str) -> None:
        self.parent, self.global_exhausted = parent, False
        self._committed = False
        super().__init__(run_id=parent.run_id, max_model_nfe=budget["max_model_nfe"],
                         max_compute_seconds=budget["max_compute_seconds"],
                         event_sink=lambda event: parent.event("value_eval_trace", job_id=job_id,
                             trace={key: value for key, value in event.items() if key not in ("timestamp", "run_id")}))

    def require_budget(self, *, model_calls: int = 0) -> None:
        try:
            self.parent.require_budget(model_calls=self.raw_nfe+model_calls)
            if (self.parent.max_compute_seconds is not None
                    and self.parent.tracked_compute_seconds+self.end_to_end_seconds >= self.parent.max_compute_seconds):
                raise BudgetExceeded("global_compute_seconds")
        except BudgetExceeded:
            self.global_exhausted = True
            raise
        super().require_budget(model_calls=model_calls)
        if self.end_to_end_seconds >= self.max_compute_seconds:
            raise BudgetExceeded("query_compute_seconds")

    def commit_cost(self) -> None:
        if self._committed:
            raise ValueError("Query cost was already committed")
        snapshot = self.snapshot()
        for name, count in snapshot.counters.items():
            self.parent.increment(name, count)
        for name in ("compute_seconds", "gpu_seconds", "flops"):
            target = getattr(self.parent, name)
            for component, value in getattr(snapshot, name).items():
                target[component] = target.get(component, 0.0)+value
        self.parent.peak_gpu_memory_bytes = max(self.parent.peak_gpu_memory_bytes, snapshot.peak_gpu_memory_bytes)
        self._committed = True


def build_runtime(context: TaskContext) -> dict[str, Any]:
    config, store = context.config, context.store
    tokenizer = SmilesTokenizer.load(store, config["data"]["tokenizer"]["metadata_path"])
    device, precision = _device_precision(config["generation"]["model"])
    dtype = precision or torch.float32
    model = MDLMAdapter.from_checkpoint(store, mdlm_source(config, store)[0], tokenizer, device=str(device), dtype=dtype)
    inner = MDLMAdapter(model.model, model_id=model.model_id, inner=True)
    model_hash = context.identity.dependency_hashes["mdlm"]
    verifier_path = verifier_source(config, store, mdlm_sha256=model_hash)[0]
    verifier = VerifierAdapter.from_checkpoint(store, verifier_path, tokenizer, device=str(device), dtype=dtype,
        batch_size=config["value"]["learned"]["batch_size"], expected_reward_version=config["reward"]["version"],
        expected_source_mdlm=model_hash)
    checker = RewardChecker.from_config(config)
    LearnedValueProvider(verifier, tokenizer, checker, source_mdlm_sha256=model_hash,
                         suite=config["suite"], cache_capacity=0)
    reverse = config["value_eval"]["reference"]["rollout"]
    q_policy = LeftToRightCompletion(model)
    rollout_policy = ReferenceCompletion(model, steps=reverse["steps"], end_time=reverse["end_time"], final_denoise=True)
    source = verifier.target_metadata
    native_target = "V_q" if source["target_kind"] == "q_z_rollout_return" else source["target_kind"]
    q_aligned = native_target == "V_q" and source["policy_hash"] == q_policy.policy_hash
    if config["value_eval"]["require_q_trained_verifier"] and not q_aligned:
        raise ValueError("This evaluation requires a verifier trained on the exact current q_z policy")
    actual = {"device": str(device), "dtype": str(dtype), "torch": str(torch.__version__),
        "mdlm": model.model_id, "verifier": verifier.model_id, "reward": reward_context(checker),
        "q_policy": q_policy.metadata, "q_policy_hash": q_policy.policy_hash,
        "rollout_policy": rollout_policy.metadata, "rollout_policy_hash": rollout_policy.policy_hash,
        "learned_training_target": dict(source), "learned_q_aligned": q_aligned,
        "learned_rollout_policy_matches": native_target == "reference_rollout_return" and source["policy_hash"] == rollout_policy.policy_hash,
        "rollout_reference_note": "Partial-state rollouts restart at time 1. Matching the training policy hash does not prove equality to its snapshot-regression conditional target."}
    return {"tokenizer": tokenizer, "model": model, "inner": inner, "verifier": verifier, "checker": checker, "actual": actual}


def evaluation_jobs(states: Sequence[Mapping[str, Any]], options: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for item in states:
        descriptors = [{"kind": kind, "budget": "reference", "repeat": 0} for kind in ("reference_q", "reference_rollout")]
        descriptors.extend({"kind": kind, "budget": budget["name"], "repeat": repeat}
            for budget in options["provider_budgets"] for repeat in range(options["repetitions"]) for kind in ("learned", "cars"))
        for descriptor in descriptors:
            job = {"state_id": item["state_id"], **descriptor}
            jobs.append({**job, "job_id": config_digest(job)})
    return jobs


def measure_query(context: TaskContext, runtime: Mapping[str, Any], item: Mapping[str, Any],
                  job: Mapping[str, Any]) -> dict[str, Any]:
    options = context.config["value_eval"]
    budget = options["reference"] if job["budget"] == "reference" else next(
        value for value in options["provider_budgets"] if value["name"] == job["budget"])
    tracker = QueryTracker(context.tracker, budget, job_id=job["job_id"])
    model, inner, verifier, checker, tokenizer = [runtime[name] for name in ("model", "inner", "verifier", "checker", "tokenizer")]
    for adapter in (model, inner, verifier, checker):
        adapter.tracker = tracker
    state = SmilesState.from_dict(item["state"])
    seed = _seed(context.identity.seed, "query", job["job_id"])
    row = {**job, "group": state.group, "seed": seed, "mask_fraction": item["mask_fraction"],
           "terminal": item["terminal"], "status": "completed", "value": None, "estimate": None,
           "target": None, "policy_hash": None, "reason": None,
           "allowance": {key: budget[key] for key in ("max_model_nfe", "max_compute_seconds")}}
    try:
        tracker.require_budget()
        with tracker.measure("value_query"):
            if job["kind"] in ("reference_q", "reference_rollout"):
                settings = options["reference"]
                common = {"m": settings["m"], "seed": seed, "tracker": tracker, "batch_size": settings["batch_size"]}
                if job["kind"] == "reference_q":
                    estimate = q_z_reference(model, tokenizer, checker, state, **common)
                    row["policy_hash"] = runtime["actual"]["q_policy_hash"]
                else:
                    estimate = rollout_reference(model, tokenizer, checker, state, steps=settings["rollout"]["steps"],
                                                 end_time=settings["rollout"]["end_time"], **common)
                    row["policy_hash"] = runtime["actual"]["rollout_policy_hash"]
                row.update(value=estimate.value, target=estimate.target, estimate=estimate.to_dict())
            elif job["kind"] == "learned":
                provider = LearnedValueProvider(verifier, tokenizer, checker, source_mdlm_sha256=context.identity.dependency_hashes["mdlm"],
                                                suite=context.identity.suite, cache_capacity=0, tracker=tracker)
                row.update(value=provider.value(state), target=provider.target, provider=provider.metadata,
                           policy_hash=verifier.target_metadata["policy_hash"])
            else:
                provider = CARSValueProvider.from_mdlm(inner, tokenizer, checker, m=options["cars"]["m"],
                    max_nodes=options["cars"]["trie_max_nodes"], cache_capacity=0, seed=seed, tracker=tracker)
                row.update(value=provider.value(state), target=provider.target, estimate=provider.last_estimate,
                           provider=provider.metadata, policy_hash=runtime["actual"]["q_policy_hash"])
        # Time includes CPU sampling/grammar work, not just the model invocation.
        tracker.require_budget()
    except BudgetExceeded as exc:
        # Never publish a partial fixed-m estimate or silently substitute zero.
        row.update(status="budget_exhausted", value=None, estimate=None, reason=str(exc))
    finally:
        row["cost"] = tracker.summary()
        row["global_exhausted"] = tracker.global_exhausted
        tracker.commit_cost()
        for adapter in (model, inner, verifier, checker):
            adapter.tracker = None
    context.tracker.event("value_eval_query", **row)
    return row


def _cost_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"queries": len(rows), "completed": sum(row["status"] == "completed" for row in rows),
        "budget_exhausted": sum(row["status"] == "budget_exhausted" for row in rows),
        "raw_nfe": sum(row["cost"]["raw_nfe"] for row in rows),
        "model_example_nfe": sum(row["cost"]["model_example_nfe"] for row in rows),
        "verifier_calls": sum(row["cost"]["counters"]["verifier_forward_calls"] for row in rows),
        "inner_mdlm_calls": sum(row["cost"]["counters"]["inner_mdlm_forward_calls"] for row in rows),
        "compute_seconds": math.fsum(row["cost"]["tracked_compute_seconds"] for row in rows),
        "elapsed_seconds": math.fsum(row["cost"]["end_to_end_seconds"] for row in rows), "adjusted_nfe": None}


def accuracy_block(predictions: Sequence[Mapping[str, Any]], references: Mapping[str, Mapping[str, Any]],
                   options: Mapping[str, Any]) -> dict[str, Any]:
    pairs = [(row, references[row["state_id"]]) for row in predictions
             if row["status"] == "completed" and row["value"] is not None
             and row["state_id"] in references and references[row["state_id"]]["status"] == "completed"]
    errors = [row["value"]-ref["value"] for row, ref in pairs]
    precise = [(row, ref) for row, ref in pairs if (ref["estimate"]["confidence_interval"][1]
               - ref["estimate"]["confidence_interval"][0])/2 <= options["reference"]["maximum_ci_half_width"]]
    variances = [ref["estimate"]["standard_error"]**2 for _, ref in pairs if ref["estimate"]["standard_error"] is not None]
    mse = math.fsum(error*error for error in errors)/len(errors) if errors else None
    variance = math.fsum(variances)/len(variances) if variances and len(variances) == len(pairs) else None
    reliability, ece = [], 0.0
    bins = options["calibration_bins"]
    for index in range(bins):
        members = [(row, ref) for row, ref in pairs if min(int(row["value"]*bins), bins-1) == index]
        predicted = math.fsum(row["value"] for row, _ in members)/len(members) if members else None
        observed = math.fsum(ref["value"] for _, ref in members)/len(members) if members else None
        if members:
            ece += len(members)*abs(predicted-observed)
        reliability.append({"lower": index/bins, "upper": (index+1)/bins, "queries": len(members),
            "mean_prediction": predicted, "mean_mc_reference": observed,
            "mean_reference_ci": [math.fsum(ref["estimate"]["confidence_interval"][side] for _, ref in members)/len(members)
                                  for side in (0, 1)] if members else None})
    return {"attempted_queries": len(predictions), "paired_queries": len(pairs),
        "distinct_states": len({row["state_id"] for row, _ in pairs}), "unpaired_queries": len(predictions)-len(pairs),
        "mse_to_mc_reference": mse, "mae_to_mc_reference": math.fsum(map(abs, errors))/len(errors) if errors else None,
        "mean_reference_variance": variance,
        "mse_minus_reference_variance": mse-variance if mse is not None and variance is not None else None,
        "precise_reference_queries": len(precise), "imprecise_reference_queries": len(pairs)-len(precise),
        "mse_on_precise_references": math.fsum((row["value"]-ref["value"])**2 for row, ref in precise)/len(precise) if precise else None,
        "ece_to_mc_reference": ece/len(pairs) if pairs else None, "reliability_bins": reliability,
        "cost_including_failed_queries": _cost_report(predictions),
        "uncertainty_note": "Finite MC labels; repeated queries share a reference and are not independent states. Noise-adjusted MSE may be negative."}


def summarize(rows: Sequence[Mapping[str, Any]], states: Sequence[Mapping[str, Any]],
              options: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, Any]:
    references = {kind: {row["state_id"]: row for row in rows if row["kind"] == kind}
                  for kind in ("reference_q", "reference_rollout")}
    results, paired = [], []
    for budget in options["provider_budgets"]:
        for kind in ("learned", "cars"):
            predictions = [row for row in rows if row["kind"] == kind and row["budget"] == budget["name"]]
            for reference_kind, reference in references.items():
                aligned = reference_kind == "reference_q" and (kind == "cars" or actual["learned_q_aligned"])
                policy_matches = aligned or (kind == "learned" and reference_kind == "reference_rollout"
                                             and actual["learned_rollout_policy_matches"])
                results.append({"provider": kind, "budget": budget["name"], "reference_policy": reference_kind,
                    "training_or_estimator_target_aligned": aligned,
                    "training_policy_matches_reference": policy_matches,
                    "interpretation": "policy_value_error" if aligned else (
                        "same_rollout_policy_diagnostic" if policy_matches else "cross_policy_diagnostic"),
                    "overall": accuracy_block(predictions, reference, options),
                    "nonterminal": accuracy_block([row for row in predictions if not row["terminal"]], reference, options),
                    "terminal": accuracy_block([row for row in predictions if row["terminal"]], reference, options),
                    "groups": {group: accuracy_block([row for row in predictions if row["group"] == group], reference, options) for group in GROUPS},
                    "mask_fraction": [{"lower": lower, "upper": upper, **accuracy_block(
                        [row for row in predictions if lower <= row["mask_fraction"] and
                         (row["mask_fraction"] <= upper if upper == 1 else row["mask_fraction"] < upper)], reference, options)}
                         for lower, upper in zip(options["mask_fraction_edges"][:-1], options["mask_fraction_edges"][1:], strict=True)]})
        learned = {(row["state_id"], row["repeat"]): row for row in rows if row["kind"] == "learned" and row["budget"] == budget["name"]}
        cars = {(row["state_id"], row["repeat"]): row for row in rows if row["kind"] == "cars" and row["budget"] == budget["name"]}
        triples = [(left, cars[key], references["reference_q"][key[0]]) for key, left in learned.items()
            if key in cars and key[0] in references["reference_q"] and all(row["status"] == "completed"
                for row in (left, cars[key], references["reference_q"][key[0]]))]
        paired.append({"budget": dict(budget), "paired_queries": len(triples),
            "distinct_states": len({left["state_id"] for left, _, _ in triples}), "reference_policy": "q_z",
            "learned_training_target_aligned": actual["learned_q_aligned"],
            "cars_minus_learned_mse": math.fsum((right["value"]-ref["value"])**2-(left["value"]-ref["value"])**2
                for left, right, ref in triples)/len(triples) if triples else None,
            "cars_minus_learned_mae": math.fsum(abs(right["value"]-ref["value"])-abs(left["value"]-ref["value"])
                for left, right, ref in triples)/len(triples) if triples else None,
            "same_allowance": True, "equal_actual_compute_claim": False})
    return {"states": len(states), "queries": len(rows), "results": results, "paired_q_comparisons": paired,
        "reference_costs": {kind: _cost_report(list(reference.values())) for kind, reference in references.items()},
        "ideal_value_claim": False, "calibration_fitted": False,
        "comparison_note": "Both providers are scored against the same q_z reference under identical caps. Native rollout error is separate; actual time, verifier calls and MDLM calls remain distinct."}


def _write_results(store: ArtifactStore, path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = ["job_id", "state_id", "group", "kind", "budget", "repeat", "status", "value", "details_json"]
    flat = [{**{name: row[name] for name in columns[:-1]},
             "details_json": json.dumps(row, sort_keys=True, allow_nan=False)} for row in rows]
    write_parquet(store, path, flat, columns=columns)


def merge_shards(store: ArtifactStore, shards: Sequence[Mapping[str, Any]], jobs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for shard in shards:
        store.require_file(shard["path"], checksum=shard["sha256"])
        incoming = [json.loads(row["details_json"]) for row in read_parquet(store, shard["path"])]
        if len(incoming) != shard["rows"]:
            raise ValueError("Value-evaluation shard row count differs")
        rows.extend(incoming)
    if [row["job_id"] for row in rows] != [job["job_id"] for job in jobs]:
        raise ValueError("Value-evaluation shards have missing, duplicate or reordered jobs")
    for row, job in zip(rows, jobs, strict=True):
        if any(row[name] != value for name, value in job.items()):
            raise ValueError("Value-evaluation shard belongs to another state/provider/budget")
    return rows


def recover_uncommitted_cost(context: TaskContext) -> dict[str, Any] | None:
    """A graceful interruption saved actual work in the run record, after its cursor.

    Retain that work when the query replays. Hard termination without a final
    run record cannot provide these counters and is explicitly not reconstructed.
    """
    if not context.resumed:
        return None
    previous, current = context.record.tracking, context.tracker.snapshot()
    counts = {name: previous.counters.get(name, 0)-current.counters[name] for name in current.counters}
    if any(count < 0 for count in counts.values()):
        return None  # A stale run record predates the committed checkpoint.
    timing = {name: {component: max(0., value-getattr(current, name).get(component, 0.))
                    for component, value in getattr(previous, name).items()}
              for name in ("compute_seconds", "gpu_seconds", "flops")}
    if not any(counts.values()) and not any(value for components in timing.values() for value in components.values()):
        return None
    for name, count in counts.items():
        context.tracker.increment(name, count)
    for name, components in timing.items():
        destination = getattr(context.tracker, name)
        for component, value in components.items():
            destination[component] = destination.get(component, 0.)+value
    context.tracker.peak_gpu_memory_bytes = max(context.tracker.peak_gpu_memory_bytes, previous.peak_gpu_memory_bytes)
    return {"counters": counts, **timing, "raw_nfe": counts["mdlm_forward_calls"]+counts["ar_forward_calls"],
            "source": "Durable run record minus the committed query checkpoint",
            "end_to_end_time_reconstructed": False}


def evaluate_value(context: TaskContext, *, on_checkpoint: Callable[[], None] | None = None) -> TaskResult:
    config, store, tracker = context.config, context.store, context.tracker
    options = config["value_eval"]
    directory = store.directory("data", "value_eval", context.identity.suite, options["purpose"],
                                f"seed_{context.identity.seed}", context.identity.run_id)
    states_path = os.path.join(directory, "states.pt")
    recovered = recover_uncommitted_cost(context)
    with tracker.measure("value_eval_setup"):
        runtime = build_runtime(context)
        state = context.resume_state
        if state is None:
            pool = build_states(context, runtime["tokenizer"])
            digest = _save_tensor(store, states_path, pool)
            state = {"version": VALUE_EVAL_VERSION, "runtime": runtime["actual"], "states_path": states_path,
                "states_sha256": digest, "next_job": 0, "pending": [], "shards": [], "stage": "evaluation",
                "status": None, "stop_reason": None, "recovery_costs": []}
        else:
            if (state["version"] != VALUE_EVAL_VERSION or state["runtime"] != runtime["actual"]
                    or state["states_path"] != states_path or state["stage"] not in ("evaluation", "finished")):
                raise ValueError("Value-evaluation runtime/state context changed on resume")
            pool = torch.load(store.require_file(states_path, checksum=state["states_sha256"]), map_location="cpu", weights_only=True)
        if (pool["version"] != VALUE_EVAL_VERSION or pool["purpose"] != options["purpose"] or pool["split"] != options["split"]
                or pool["tokenizer"] != runtime["tokenizer"].fingerprint
                or pool["dependency_hashes"] != dict(context.identity.dependency_hashes)):
            raise ValueError("Saved states have incompatible split/tokenizer/dependencies")
    states = {item["state_id"]: item for item in pool["states"]}
    if len(states) != len(pool["states"]):
        raise ValueError("Saved evaluation states must have unique IDs")
    for key, item in states.items():
        if (state_identity(SmilesState.from_dict(item["state"]), runtime["tokenizer"]) != key
                or state_purpose(key, options) != options["purpose"]):
            raise ValueError("Saved evaluation state ID/purpose differs")
    jobs = evaluation_jobs(pool["states"], options)
    if recovered is not None:
        state["recovery_costs"].append(recovered)
    require_count(state["next_job"], "saved evaluation cursor")
    committed = sum(shard["rows"] for shard in state["shards"])
    if state["next_job"] > len(jobs) or committed+len(state["pending"]) != state["next_job"]:
        raise ValueError("Value-evaluation cursor/shards are inconsistent")
    merge_shards(store, state["shards"], jobs[:committed])
    if [row["job_id"] for row in state["pending"]] != [job["job_id"] for job in jobs[committed:state["next_job"]]]:
        raise ValueError("Pending query results differ from the saved cursor")

    def checkpoint() -> None:
        context.save_checkpoint(state)
        if on_checkpoint is not None:
            on_checkpoint()

    def flush() -> None:
        if state["pending"]:
            path = os.path.join(directory, "shards", f"shard_{len(state['shards']):05d}.parquet")
            _write_results(store, path, state["pending"])
            state["shards"].append({"path": path, "sha256": sha256_file(store.require_file(path)), "rows": len(state["pending"])})
            state["pending"] = []

    checkpoint()
    with context.progress(total=len(jobs), initial=state["next_job"], desc="Value evaluation", unit="query") as progress:
        while state["stage"] == "evaluation" and state["next_job"] < len(jobs):
            try:
                tracker.require_budget()
                if tracker.max_model_nfe is not None and tracker.raw_nfe >= tracker.max_model_nfe:
                    raise BudgetExceeded("global_model_nfe")
            except BudgetExceeded as exc:
                state.update(status="budget_exhausted", stop_reason=str(exc))
                break
            job = jobs[state["next_job"]]
            row = measure_query(context, runtime, states[job["state_id"]], job)
            state["pending"].append(row)
            state["next_job"] += 1
            progress.update(1)
            if len(state["pending"]) >= options["shard_rows"]:
                flush()
            if row["global_exhausted"]:
                state.update(status="budget_exhausted", stop_reason=row["reason"])
                break
            if state["next_job"] % options["log_every"] == 0:
                context.log(f"queries={state['next_job']}/{len(jobs)}; NFE={tracker.raw_nfe}; last={row['status']}")
            if state["next_job"] % options["checkpoint_every"] == 0:
                checkpoint()
    flush()
    if state["status"] is None:
        state.update(status="completed" if states else "failed",
                     stop_reason="evaluation_finished" if states else "no_independent_states")
    state["stage"] = "finished"
    checkpoint()
    with tracker.measure("value_eval_export"):
        rows = merge_shards(store, state["shards"], jobs[:state["next_job"]])
        values_path = os.path.join(directory, "reference_values.parquet")
        _write_results(store, values_path, rows)
        report_path = context.output_path("metrics.json")
        report = {"version": VALUE_EVAL_VERSION, "status": state["status"], "stop_reason": state["stop_reason"],
            "identity": context.identity.to_dict(), "purpose": options["purpose"], "source_split": options["split"],
            "used_for_selection": options["purpose"] == "tuning", "test_used_for_selection": False,
            "states_path": states_path, "states_sha256": state["states_sha256"], "values_path": values_path,
            "expected_queries": len(jobs), "completed_query_jobs": state["next_job"], "missing_query_jobs": len(jobs)-state["next_job"],
            "query_status_counts": dict(Counter(row["status"] for row in rows)), "selection_counts": pool["selection_counts"],
            "requested_molecules_per_group": options["molecules_per_group"], "molecules_per_group": pool["molecules_per_group"],
            "states_per_group": {group: pool["states_per_group"].get(group, 0) for group in GROUPS},
            "state_coverage": [{"group": group, "requested_mask_fraction": fraction,
                "distinct_states": sum(item["group"] == group and any(origin["requested_mask_fraction"] == fraction
                    for origin in item.get("origins", [])) for item in pool["states"])}
                for group in GROUPS for fraction in options["mask_fractions"]],
            "runtime": runtime["actual"], "checker_versions": checker_versions(),
            "evaluation": summarize(rows, pool["states"], options, runtime["actual"]),
            "uncertainty": {"reference_interval": "Wilson binomial 95%; pointwise, not simultaneous",
                "reference_precision_threshold": options["reference"]["maximum_ci_half_width"],
                "cars_interval": "Hoeffding-Azuma bounded martingale 95%; no IID standard error",
                "reference_is_exact": "Only terminal anchors", "adaptive_mc_stopping": False},
            "resume_unit": "One complete query; interrupted queries replay from their deterministic independent seed",
            "recovery_costs": state["recovery_costs"],
            "recovery_note": "Gracefully recorded uncommitted work is retained in total cost, separately from successful query costs. Hard-kill work and interruption end-to-end time cannot be reconstructed.",
            "cost_protocol": {"matched_provider_allowances": options["provider_budgets"], "equal_actual_compute_claim": False,
                "cache_capacity": 0, "setup_cost_is_separate": True,
                "raw_nfe_excludes_verifier": True, "verifier_calls_and_time_are_reported": True,
                "reference_cost_is_separate": True, "adjusted_nfe_available": False}}
    report["tracking"] = tracker.summary()
    store.write_json(report_path, report)
    manifest_path = os.path.join(directory, "manifest.json")
    store.create_manifest(manifest_path, kind="value_evaluation", config_hash=context.identity.config_hash,
        files=[states_path, values_path, report_path, *[shard["path"] for shard in state["shards"]]],
        metadata={"version": VALUE_EVAL_VERSION, "suite": context.identity.suite, "purpose": options["purpose"],
            "split": options["split"], "status": state["status"], "dependency_hashes": dict(context.identity.dependency_hashes),
            "state_count": len(states), "evaluated_jobs": state["next_job"], "expected_jobs": len(jobs),
            "used_for_selection": options["purpose"] == "tuning", "ideal_value_claim": False})
    if on_checkpoint is not None:
        on_checkpoint()
    return TaskResult(f"states={len(states)}; queries={state['next_job']}/{len(jobs)}; NFE={tracker.raw_nfe}; report={report_path}",
                      status=state["status"], artifacts=(states_path, values_path, report_path, manifest_path))


def _arguments(parser: argparse.ArgumentParser) -> None:
    add_arguments(parser, "mdm_vgb")


def execute(argv: Sequence[str] | None = None, *, execution="local", on_checkpoint=None):
    return run_task("T13", lambda context: evaluate_value(context, on_checkpoint=on_checkpoint), argv=argv,
        execution=execution, description="Evaluate learned and CARS policy values on independent states.",
        default_config="configs/value_eval.yaml", validator=validate_value_eval, dependencies=evaluation_dependencies,
        add_arguments=_arguments, argument_overrides=lambda args: source_overrides(mdlm_report=args.mdlm_report or "",
            checkpoint=args.checkpoint or "", verifier_report=args.verifier_report or "", verifier_checkpoint=args.verifier_checkpoint or ""))


if __name__ == "__main__":
    try:
        record = execute()
        if record.status != "completed":
            raise SystemExit(1)
    except KeyboardInterrupt:
        tqdm.write("T13: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"T13: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
