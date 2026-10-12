"""T09: CARS generation, attempt journaling and external unique-output selection.

Run with ``python src/tasks/generate_cars.py`` from the repository root.
Primary/smoke require a shared MDLM checkpoint and tokenizer. Replication loads
only the pinned offline Llama snapshot, without needing a trained SMILES MDLM.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence

import torch
from tqdm.auto import tqdm

from src.artifacts import ArtifactStore, Dependency, _atomic_file, sha256_file
from src.cars import CARS_VERSION, ARCompletionPolicy, CARSSampler, MDLMCompletionPolicy
from src.completion import LeftToRightCompletion
from src.config import Config, project_path
from src.data import IDENTITY_VERSION, InvalidSmiles, canonicalize_smiles
from src.mdlm import MDLMAdapter
from src.reward import RewardChecker
from src.schemas import GROUPS, AttemptRecord, RewardResult, SampleRecord, SmilesState, TaskResult, require_count
from src.tasks.common import TaskContext, run_task
from src.tasks.prepare_benchmark import read_benchmark_assets, validate_data
from src.tasks.prepare_corpus import benchmark_dependency, checker_versions
from src.tasks.train_mdlm import _device_precision
from src.tokenizer import SmilesTokenizer
from src.tracking import BudgetExceeded


def model_overrides(*, training_report: str = "", checkpoint: str = "") -> list[str]:
    if training_report and checkpoint:
        raise ValueError("Choose --mdlm-report or --checkpoint")
    if training_report:
        return ["cars.model.training_report=" + json.dumps(project_path(training_report)),
                "cars.model.checkpoint_manifest=null"]
    if checkpoint:
        return ["cars.model.checkpoint_manifest=" + json.dumps(project_path(checkpoint)),
                "cars.model.training_report=null"]
    return []


def validate_cars(config: Config) -> None:
    validate_data(config)
    if config.get("group") not in GROUPS:
        raise ValueError("T09 requires --group (one chemical condition per run)")
    options = config["cars"]
    if (options["version"] != CARS_VERSION or options["learn_level"] != 3
            or options["constrain_first"] is not True or options["learning"] != "after_attempt"
            or options["vocabulary"] != "full"
            or (options["temperature"], options["top_p"], options["top_k"]) != (1.0, 1.0, 0)):
        raise ValueError("CARS requires level 3, constrained root, frozen attempts and full vocabulary at temperature 1")
    for name in ("checkpoint_every", "log_every", "max_new_tokens"):
        require_count(options[name], "cars." + name, positive=True)
    trie = options["trie"]
    require_count(trie["max_nodes"], "cars.trie.max_nodes", positive=True)
    if trie["capacity_policy"] != "freeze_new_nodes" or trie["persist"] is not True:
        raise ValueError("CARS must persist its trie and freeze learning at new nodes on capacity")
    source = options["model"]
    for name in ("training_report", "checkpoint_manifest"):
        if source[name] is not None:
            project_path(source[name])
    if source["device"] not in ("auto", "cpu", "cuda") or source["dtype"] not in ("bfloat16", "float32"):
        raise ValueError("Unsupported MDLM generation device/dtype")
    if config["suite"] == "replication":
        from src.autoregressive import validate_models

        validate_models(config)
        llama = config["models"]["llama"]
        if (options["backend"] != "ar" or options["policy"] != "bounded_ar"
                or options["max_new_tokens"] != 256
                or any(source[name] is not None for name in ("training_report", "checkpoint_manifest"))
                or llama["model_id"] != "meta-llama/Llama-3.1-8B-Instruct"
                or llama["revision"] != "0e9e39f249a16976918f6564b8830bc894c89659"):
            raise ValueError("Replication requires pinned Llama, original prompt and the 256-token AR policy")
    elif (options["backend"] != "mdlm" or options["policy"] != "left_to_right"
          or sum(source[name] is not None for name in ("training_report", "checkpoint_manifest")) != 1):
        raise ValueError("Primary/smoke CARS requires q_z and one --mdlm-report or --checkpoint")


def _mdlm_source(config: Config, store: ArtifactStore) -> tuple[str, dict[str, Dependency]]:
    source = config["cars"]["model"]
    dependencies: dict[str, Dependency] = {}
    if source["training_report"] is not None:
        report = store.read_json(source["training_report"])
        if report["suite"] != config["suite"] or report.get("test_used_for_selection") is not False:
            raise ValueError("T05 report suite or checkpoint selection provenance differs")
        path = project_path(report["best_manifest"])
        dependencies["mdlm_report"] = Dependency(source["training_report"])
    else:
        path = project_path(source["checkpoint_manifest"])
    token_path = config["data"]["tokenizer"]["metadata_path"]
    tokenizer = SmilesTokenizer.load(store, token_path)
    metadata = store.read_json(token_path)
    checkpoint = store.require_manifest(path, kind="mdlm_checkpoint", metadata={
        "suite": config["suite"], "groups": list(GROUPS), "tokenizer_fingerprint": tokenizer.fingerprint,
    })
    hashes = checkpoint.metadata["dependency_hashes"]
    if (sha256_file(store.require_file(token_path)) != hashes["tokenizer"]
            or metadata["vocabulary_path"] != config["data"]["tokenizer"]["vocabulary_path"]
            or metadata.get("group_order") != list(GROUPS)
            or metadata["provenance"]["training_split"] != "train"
            or metadata["provenance"]["corpus_sha256"] != checkpoint.metadata["corpus_sha256"]):
        raise ValueError("CARS tokenizer provenance differs from the trained shared MDLM")
    dependencies.update({"mdlm": Dependency(path, manifest=True, kind="mdlm_checkpoint"),
                         "tokenizer": Dependency(token_path, checksum=hashes["tokenizer"]),
                         "vocabulary": Dependency(metadata["vocabulary_path"], checksum=hashes["vocabulary"])})
    return path, dependencies


def generation_dependencies(config: Config, store: ArtifactStore) -> Mapping[str, Dependency]:
    checker_versions()
    dependencies = {"benchmark": benchmark_dependency(config, store)}
    if config["cars"]["backend"] == "ar":
        from src.autoregressive import require_model_snapshot

        options = config["models"]["llama"]
        require_model_snapshot(store, options)
        dependencies["llama"] = Dependency(options["manifest"], manifest=True, kind="pretrained_model")
    else:
        dependencies.update(_mdlm_source(config, store)[1])
    return dependencies


class UniqueSelector:
    """Uniqueness and prompt exclusion do not change tau or the rejection trie.

    Valid outputs use molecular identity in both suites. Grammar-only successes
    without an RDKit identity use their exact raw text, explicitly labeled.
    """

    def __init__(self, examples: Mapping[str, Sequence[str]], *, replication: bool = False,
                 selected: Sequence[str] = ()) -> None:
        self.replication = replication
        self.prompt_raw = {text for rows in examples.values() for text in rows}
        self.prompt_molecules = set()
        for text in self.prompt_raw:
            try:
                self.prompt_molecules.add(canonicalize_smiles(text))
            except InvalidSmiles:
                pass
        self.selected = set(selected)
        if len(self.selected) != len(selected):
            raise ValueError("Duplicate identities in selector snapshot")
        self.last_reason = ""

    def consider(self, text: str, result: RewardResult) -> tuple[str, str] | None:
        if result.reward != 1.0:
            self.last_reason = "reward_zero"
            return None
        if result.canonical_smiles is not None:
            identity, kind = result.canonical_smiles, "canonical_smiles"
            excluded = identity in self.prompt_molecules
        elif self.replication:
            identity, kind = text, "raw_smiles"
            excluded = text in self.prompt_raw
        else:
            raise ValueError("Primary success must have a molecular identity")
        key = kind + ":" + identity
        if excluded or key in self.selected:
            self.last_reason = "prompt_example" if excluded else "duplicate"
            return None
        self.selected.add(key)
        self.last_reason = "selected"
        return identity, kind


def _policy(context: TaskContext, checker: RewardChecker):
    config, store, group = context.config, context.store, context.identity.group
    if config["cars"]["backend"] == "ar":
        from src.autoregressive import LlamaAdapter

        adapter = LlamaAdapter(store, config["models"]["llama"], tracker=context.tracker)
        with open(os.path.join(config["paths"]["benchmark"], group + ".prompt.txt"), encoding="utf-8") as handle:
            prompt = handle.read()
        policy = ARCompletionPolicy(adapter, prompt, checker, group,
                                    max_new_tokens=config["cars"]["max_new_tokens"], tracker=context.tracker)
        actual = {"device": str(adapter.device), "dtype": str(adapter.model.dtype), "torch": str(torch.__version__)}
        return policy, (), actual
    tokenizer = SmilesTokenizer.load(store, config["data"]["tokenizer"]["metadata_path"])
    device, precision = _device_precision(config["cars"]["model"])
    adapter = MDLMAdapter.from_checkpoint(store, _mdlm_source(config, store)[0], tokenizer,
                                         device=str(device), dtype=precision or torch.float32, tracker=context.tracker)
    policy = MDLMCompletionPolicy(LeftToRightCompletion(adapter), tokenizer, checker, group, tracker=context.tracker)
    root = SmilesState.fully_masked(tokenizer.max_length, group, tokenizer.special_tokens)
    return policy, root, {"device": str(device), "dtype": str(precision or torch.float32), "torch": str(torch.__version__)}


def generate_cars(context: TaskContext, *, on_checkpoint: Callable[[], None] | None = None) -> TaskResult:
    options, tracker, store = context.config["cars"], context.tracker, context.store
    with tracker.measure("setup"):
        _, examples = read_benchmark_assets(context.config)
        checker = RewardChecker.from_config(context.config, tracker=tracker)
        policy, root, actual = _policy(context, checker)
        sampler = CARSSampler(policy, root, seed=context.identity.seed, max_nodes=options["trie"]["max_nodes"],
                              constrain_first=options["constrain_first"], tracker=tracker)
    state = context.resume_state or {"version": CARS_VERSION, "runtime": actual, "selected": [],
                                    "policy_context": sampler.trie.context_hash, "stage": "generation",
                                    "status": None, "stop_reason": None}
    if (state["version"] != CARS_VERSION or state["runtime"] != actual
            or state["policy_context"] != sampler.trie.context_hash):
        raise ValueError("CARS resume runtime/model/root/reward/policy context changed")
    if context.resume_state is not None:
        sampler.load_state_dict(context.training_state["sampler"])
    selector = UniqueSelector(examples, replication=context.identity.suite == "replication", selected=state["selected"])
    if len(selector.selected) != tracker.counters["selected_samples"]:
        raise ValueError("CARS resume selector disagrees with its recorded sample count")
    for name in ("attempts.jsonl", "samples.jsonl"):
        if not os.path.isfile(store.path(context.output_path(name))):
            store.write_jsonl(context.output_path(name), [])

    def checkpoint() -> None:
        state["selected"] = sorted(selector.selected)
        context.save_checkpoint(state, training_state={"sampler": sampler.state_dict()})
        if on_checkpoint is not None:
            on_checkpoint()

    protocol = context.config["protocol"]
    target, cap = protocol["target_unique"], protocol["max_root_attempts"]
    checkpoint()
    with context.progress(total=target, initial=len(selector.selected), desc=f"CARS {context.identity.group}", unit="molecule") as progress:
        while state["stage"] == "generation" and len(selector.selected) < target and tracker.counters["root_attempts"] < cap:
            try:
                tracker.require_budget()
            except BudgetExceeded as exc:
                state.update(status="budget_exhausted", stop_reason=str(exc))
                break
            before = dict(tracker.counters)
            attempt_id = tracker.record_root_attempt()
            started = time.perf_counter()
            try:
                attempt = sampler.attempt()
            except Exception as exc:
                context.write_attempt(AttemptRecord(context.identity.run_id, attempt_id, context.identity.method,
                    context.identity.group, "error", elapsed_seconds=time.perf_counter() - started,
                    reason=f"{type(exc).__name__}: {exc}"))
                raise
            context.write_attempt(AttemptRecord(
                context.identity.run_id, attempt_id, context.identity.method, context.identity.group, attempt.kind,
                leaf_visits=int(attempt.kind == "complete"), smiles=attempt.smiles,
                state=attempt.state if isinstance(attempt.state, SmilesState) else None,
                reward_result=attempt.reward_result, elapsed_seconds=time.perf_counter() - started,
                counters={key: value - before[key] for key, value in tracker.counters.items()}, reason=attempt.reason,
            ))
            selection = None
            selector.last_reason = attempt.kind
            if attempt.kind == "complete":
                selection = selector.consider(attempt.smiles, attempt.reward_result)
            if selection is not None:
                identity, kind = selection
                context.write_sample(SampleRecord(context.identity.run_id, len(selector.selected), attempt_id,
                    context.identity.group, attempt.smiles, identity, attempt.reward_result, identity_type=kind))
                progress.update(1)
            tracker.event("cars_attempt", attempt_id=attempt_id, selection=selector.last_reason, **attempt.to_dict())
            if attempt_id % options["log_every"] == 0:
                context.log(f"attempts={attempt_id}; unique={len(selector.selected)}/{target}; NFE={tracker.raw_nfe}; log remaining mass={sampler.trie.log_mass:.6g}")
            if attempt.kind == "budget_exhausted" or sampler.trie.exhausted:
                state.update(status="budget_exhausted", stop_reason=attempt.reason if attempt.kind == "budget_exhausted" else "remaining_mass_zero")
                break
            if attempt_id % options["checkpoint_every"] == 0:
                checkpoint()
    if len(selector.selected) >= target:
        state.update(status="completed", stop_reason="target_unique_reached")
    elif state["status"] is None:
        state.update(status="timeout", stop_reason="max_root_attempts")
    state["stage"] = "finished"
    checkpoint()
    sampler_path = context.output_path("cars_state.pt")
    with _atomic_file(store.path(sampler_path)) as handle:
        torch.save(sampler.state_dict(), handle)
    report_path = context.output_path("generation.json")
    store.write_json(report_path, {
        "version": CARS_VERSION, "status": state["status"], "stop_reason": state["stop_reason"],
        "identity": context.identity.to_dict(), "runtime": actual, "checker_versions": checker_versions(),
        "protocol": protocol, "identity_version": IDENTITY_VERSION,
        "selected_samples": len(selector.selected), "target_attained": len(selector.selected) >= target,
        "policy": policy.metadata, "policy_context_hash": sampler.trie.context_hash,
        "value_target": "V_q" if options["backend"] == "mdlm" else "bounded_AR_policy",
        "ideal_value_claim": False, "learning": "frozen_draw_then_update",
        "log_remaining_mass": sampler.trie.log_mass if math.isfinite(sampler.trie.log_mass) else None,
        "remaining_mass": sampler.trie.mass, "trie_nodes": len(sampler.trie.nodes),
        "trie_capacity": sampler.trie.max_nodes, "sampler_state": sampler_path,
        "attempts": context.output_path("attempts.jsonl"), "samples": context.output_path("samples.jsonl"),
        "attempt_probability_traces": context.output_path("events.jsonl"),
        "log_zero_representation": "null log with zero ordinary mass; budget-aborted contributions are null",
        "tracking": tracker.summary(), "test_used_for_selection": False,
    })
    manifest_path = context.output_path("generation_manifest.json")
    store.create_manifest(manifest_path, kind="generation", config_hash=context.identity.config_hash,
        files=[report_path, sampler_path, context.output_path("attempts.jsonl"), context.output_path("samples.jsonl")],
        metadata={"version": CARS_VERSION, "status": state["status"], "suite": context.identity.suite,
                  "method": context.identity.method, "group": context.identity.group,
                  "dependency_hashes": context.identity.dependency_hashes, "policy_context": sampler.trie.context_hash})
    if on_checkpoint is not None:
        on_checkpoint()
    return TaskResult(f"unique={len(selector.selected)}/{target}; attempts={tracker.counters['root_attempts']}; NFE={tracker.raw_nfe}; report={report_path}",
                      status=state["status"], artifacts=(report_path, manifest_path, sampler_path))


def _arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mdlm-report", help="T05 training.json selecting its best-validation checkpoint.")
    group.add_argument("--checkpoint", help="Project-relative shared MDLM checkpoint manifest.")


def execute(argv: Sequence[str] | None = None, *, execution: str = "local", on_checkpoint: Callable[[], None] | None = None):
    return run_task("T09", lambda context: generate_cars(context, on_checkpoint=on_checkpoint), argv=argv,
        execution=execution, description="Generate SMILES with CARS.", default_config="configs/cars.yaml", supports_group=True,
        method=lambda config: "cars_ar" if config["cars"]["backend"] == "ar" else "cars",
        validator=validate_cars, dependencies=generation_dependencies, add_arguments=_arguments,
        argument_overrides=lambda args: model_overrides(training_report=args.mdlm_report or "", checkpoint=args.checkpoint or ""))


if __name__ == "__main__":
    try:
        record = execute()
        if record.status != "completed":
            raise SystemExit(1)
    except KeyboardInterrupt:
        tqdm.write("T09: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"T09: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
