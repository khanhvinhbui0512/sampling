"""Shared T08/T10/T11 model provenance, selector, journals and resume handling."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence

import torch

from src.artifacts import ArtifactStore, Dependency, _atomic_file, sha256_file
from src.completion import ReferenceCompletion
from src.config import Config, config_digest, project_path
from src.data import IDENTITY_VERSION
from src.hybrid import build_hybrid
from src.mdlm import MDLMAdapter
from src.mdm_vgb import GenerationAttempt, MDMVGBSampler, output_state, validate_search
from src.reward import RewardChecker
from src.schemas import GROUPS, AttemptRecord, SampleRecord, SmilesState, TaskResult, require_count
from src.tasks.common import TaskContext, run_task
from src.tasks.generate_cars import UniqueSelector, _mdlm_source
from src.tasks.prepare_benchmark import read_benchmark_assets, validate_data
from src.tasks.prepare_corpus import benchmark_dependency, checker_versions
from src.tasks.train_mdlm import _device_precision
from src.tokenizer import SmilesTokenizer
from src.tracking import BudgetExceeded
from src.values import LearnedValueProvider, reward_context
from src.verifier import VerifierAdapter

GENERATION_VERSION = "shared-mdlm-generation-v1"
TASKS = {"base": ("T08", "configs/base.yaml"), "mdm_vgb": ("T10", "configs/mdm_vgb.yaml"),
         "hybrid": ("T11", "configs/hybrid.yaml")}


def source_overrides(*, mdlm_report: str = "", checkpoint: str = "", verifier_report: str = "",
                     verifier_checkpoint: str = "") -> list[str]:
    overrides = []
    for section, report, manifest in (("generation.model", mdlm_report, checkpoint),
                                      ("value.learned", verifier_report, verifier_checkpoint)):
        if report and manifest:
            raise ValueError("Choose a training report or explicit checkpoint for each model")
        if report:
            overrides.extend([section + ".training_report=" + json.dumps(project_path(report)),
                              section + ".checkpoint_manifest=null"])
        elif manifest:
            overrides.extend([section + ".checkpoint_manifest=" + json.dumps(project_path(manifest)),
                              section + ".training_report=null"])
    return overrides


def validate_generation(config: Config, method: str) -> None:
    validate_data(config)
    options = config["generation"]
    if (config["suite"] == "replication" or config.get("group") not in GROUPS
            or options["version"] != GENERATION_VERSION or options["method"] != method):
        raise ValueError("Generation requires one group in primary/smoke and the task's matching method")
    source = options["model"]
    if sum(source[name] is not None for name in ("training_report", "checkpoint_manifest")) != 1:
        raise ValueError("Supply one --mdlm-report or --checkpoint")
    for name in ("training_report", "checkpoint_manifest"):
        if source[name] is not None:
            project_path(source[name])
    if source["device"] not in ("auto", "cpu", "cuda") or source["dtype"] not in ("bfloat16", "float32"):
        raise ValueError("Unsupported generation device/dtype")
    for name in ("checkpoint_every", "checkpoint_every_steps", "log_every"):
        require_count(options[name], "generation." + name, positive=True)
    baseline = options["baseline"]
    require_count(baseline["steps"], "baseline steps", positive=True)
    if (baseline["policy"] != "absorbing_reverse" or not 0 < baseline["end_time"] < 1
            or baseline["final_denoise"] is not True or baseline["temperature"] != 1.0
            or baseline["vocabulary"] != "full" or baseline["known_tokens"] != "preserve_all"):
        raise ValueError("Baseline requires the full-vocabulary reference sampler and exact EOS/PAD contract")
    if method == "base":
        if any(config.get("value", {}).get("learned", {}).get(name) is not None
               for name in ("training_report", "checkpoint_manifest")):
            raise ValueError("Baseline does not accept verifier inputs")
        return
    validate_search(config["search"])
    values = config["value"]
    require_count(values["cache_capacity"], "value cache capacity")
    learned = values["learned"]
    if method == "mdm_vgb":
        if values["kind"] != "learned" or sum(learned[name] is not None for name in ("training_report", "checkpoint_manifest")) != 1:
            raise ValueError("MDM-VGB requires one --verifier-report or --verifier-checkpoint")
        require_count(learned["batch_size"], "verifier inference batch size", positive=True)
        for name in ("training_report", "checkpoint_manifest"):
            if learned[name] is not None:
                project_path(learned[name])
    elif (values["kind"] != "cars"
          or any(learned[name] is not None for name in ("training_report", "checkpoint_manifest"))):
        raise ValueError("Hybrid uses CARS values and does not accept a learned-verifier checkpoint")
    else:
        require_count(values["m"], "fixed CARS m", positive=True)
        require_count(values["trie_max_nodes"], "inner CARS trie capacity", positive=True)


def mdlm_source(config: Config, store: ArtifactStore) -> tuple[str, dict[str, Dependency]]:
    # Reuse T09's checked shared-MDLM/tokenizer provenance without changing its
    # public configuration or loading corpus shards just to generate samples.
    return _mdlm_source({**config, "cars": {"model": config["generation"]["model"]}}, store)


def verifier_source(config: Config, store: ArtifactStore, *, mdlm_sha256: str) -> tuple[str, dict[str, Dependency]]:
    source = config["value"]["learned"]
    dependencies = {}
    if source["training_report"] is not None:
        report = store.read_json(source["training_report"])
        if report["suite"] != config["suite"] or report.get("test_used_for_selection") is not False:
            raise ValueError("Verifier report has incompatible suite or selection provenance")
        path = project_path(report["best_manifest"])
        dependencies["verifier_report"] = Dependency(source["training_report"])
    else:
        path = project_path(source["checkpoint_manifest"])
    tokenizer = SmilesTokenizer.load(store, config["data"]["tokenizer"]["metadata_path"])
    manifest = store.require_manifest(path, kind="verifier_checkpoint", metadata={
        "suite": config["suite"], "groups": list(GROUPS), "reward_version": config["reward"]["version"],
        "tokenizer_fingerprint": tokenizer.fingerprint, "source_mdlm_sha256": mdlm_sha256,
        "test_used_for_selection": False,
    })
    specification = store.read_json(manifest.metadata["specification_path"])
    if specification["reward_mode"] != config["reward"]["mode"] or specification["suite"] != config["suite"]:
        raise ValueError("Verifier training reward/suite differs from generation")
    dependencies["verifier"] = Dependency(path, manifest=True, kind="verifier_checkpoint")
    return path, dependencies


def generation_dependencies(config: Config, store: ArtifactStore) -> Mapping[str, Dependency]:
    checker_versions()
    mdlm_path, dependencies = mdlm_source(config, store)
    dependencies["benchmark"] = benchmark_dependency(config, store)
    if config["generation"]["method"] == "mdm_vgb":
        dependencies.update(verifier_source(config, store, mdlm_sha256=sha256_file(store.require_file(mdlm_path)))[1])
    return dependencies


class BaselineSampler:
    """One root is an atomic reference rollout; resume checkpoints separate roots."""

    def __init__(self, completion: ReferenceCompletion, tokenizer: SmilesTokenizer, checker: RewardChecker, tracker) -> None:
        self.completion, self.tokenizer, self.checker, self.tracker = completion, tokenizer, checker, tracker
        self.metadata = {**completion.metadata, "reward": reward_context(checker), "tokenizer": tokenizer.fingerprint,
                         "kind": "unguided_baseline"}
        self.context_hash = config_digest(self.metadata)
        self.root = None

    def start(self, root: SmilesState) -> None:
        self.root = root

    def attempt(self, *, on_step=None) -> GenerationAttempt:
        try:
            self.tracker.require_budget()
            result = self.completion.complete(self.root)
            self.tracker.require_budget()
            with self.tracker.measure("reward"):
                text, reward = output_state(result.state, self.tokenizer, self.checker)
            self.tracker.record_leaf_visit()
            self.tracker.record_complete_candidate()
            self.tracker.event("baseline_completion", state=result.state.to_dict(),
                               log_trajectory_probability=result.log_probability, policy=self.completion.metadata)
            return GenerationAttempt("complete", result.state, 1, text, reward)
        except BudgetExceeded as exc:
            return GenerationAttempt("budget_exhausted", self.root, reason=str(exc))

    def state_dict(self) -> dict:
        return {"context_hash": self.context_hash, "rng": self.completion.rng_state(),
                "root": None if self.root is None else self.root.to_dict()}

    def load_state_dict(self, payload: Mapping) -> None:
        if payload["context_hash"] != self.context_hash:
            raise ValueError("Baseline policy changed on resume")
        self.completion.restore_rng_state(payload["rng"])
        self.root = None if payload["root"] is None else SmilesState.from_dict(payload["root"])


def build_sampler(context: TaskContext, checker: RewardChecker):
    config, store, tracker = context.config, context.store, context.tracker
    options = config["generation"]
    tokenizer = SmilesTokenizer.load(store, config["data"]["tokenizer"]["metadata_path"])
    device, precision = _device_precision(options["model"])
    dtype = precision or torch.float32
    model = MDLMAdapter.from_checkpoint(store, mdlm_source(config, store)[0], tokenizer,
                                        device=str(device), dtype=dtype, tracker=tracker)
    method = context.identity.method
    if method == "base":
        settings = options["baseline"]
        completion = ReferenceCompletion(model, steps=settings["steps"], end_time=settings["end_time"],
                                         final_denoise=settings["final_denoise"], seed=context.identity.seed)
        sampler = BaselineSampler(completion, tokenizer, checker, tracker)
    elif method == "hybrid":
        # Share weights; a second lightweight adapter owns inner-forward tags.
        inner = MDLMAdapter(model.model, model_id=model.model_id, tracker=tracker, inner=True)
        sampler = build_hybrid(model, inner, tokenizer, checker, config["search"], config["value"],
                               seed=context.identity.seed, tracker=tracker)
    else:
        path = verifier_source(config, store, mdlm_sha256=context.identity.dependency_hashes["mdlm"])[0]
        verifier = VerifierAdapter.from_checkpoint(store, path, tokenizer, device=str(device), dtype=dtype, tracker=tracker,
            batch_size=config["value"]["learned"]["batch_size"], expected_reward_version=checker.version,
            expected_source_mdlm=context.identity.dependency_hashes["mdlm"])
        provider = LearnedValueProvider(verifier, tokenizer, checker, source_mdlm_sha256=context.identity.dependency_hashes["mdlm"],
            suite=context.identity.suite, cache_capacity=config["value"]["cache_capacity"], tracker=tracker)
        sampler = MDMVGBSampler(model, provider, tokenizer, checker, config["search"], seed=context.identity.seed, tracker=tracker)
    root = SmilesState.fully_masked(tokenizer.max_length, context.identity.group, tokenizer.special_tokens)
    actual = {"device": str(device), "dtype": str(dtype), "torch": str(torch.__version__)}
    return sampler, root, actual


def generate(context: TaskContext, *, on_checkpoint: Callable[[], None] | None = None) -> TaskResult:
    tracker, store, options = context.tracker, context.store, context.config["generation"]
    with tracker.measure("setup"):
        _, examples = read_benchmark_assets(context.config)
        checker = RewardChecker.from_config(context.config, tracker=tracker)
        sampler, root, actual = build_sampler(context, checker)
        selector = UniqueSelector(examples)
    state = context.resume_state or {"version": GENERATION_VERSION, "runtime": actual, "context_hash": sampler.context_hash,
        "selected": [], "stage": "generation", "active": None, "status": None, "stop_reason": None}
    if state["version"] != GENERATION_VERSION or state["runtime"] != actual or state["context_hash"] != sampler.context_hash:
        raise ValueError("Generation runtime/model/policy context changed on resume")
    if context.resume_state is not None:
        sampler.load_state_dict(context.training_state["sampler"])
        selector = UniqueSelector(examples, selected=state["selected"])
    if len(selector.selected) != tracker.counters["selected_samples"]:
        raise ValueError("Selector and resume sample count disagree")
    for name in ("attempts.jsonl", "samples.jsonl"):
        if not os.path.isfile(store.path(context.output_path(name))):
            store.write_jsonl(context.output_path(name), [])
    clock_started = time.perf_counter()
    elapsed_before = 0.0 if state["active"] is None else state["active"]["elapsed"]

    def checkpoint() -> None:
        if state["active"] is not None:
            state["active"]["elapsed"] = elapsed_before + time.perf_counter()-clock_started
        state["selected"] = sorted(selector.selected)
        context.save_checkpoint(state, training_state={"sampler": sampler.state_dict()})
        if on_checkpoint is not None:
            on_checkpoint()

    def step_checkpoint() -> None:
        if tracker.counters["markov_steps"] % options["checkpoint_every_steps"] == 0:
            checkpoint()

    target, cap = (context.config["protocol"][name] for name in ("target_unique", "max_root_attempts"))
    checkpoint()
    with context.progress(total=target, initial=len(selector.selected), desc=f"{context.identity.method} {root.group}", unit="molecule") as progress:
        while state["stage"] == "generation" and len(selector.selected) < target:
            if state["active"] is None:
                if tracker.counters["root_attempts"] >= cap:
                    state.update(status="timeout", stop_reason="max_root_attempts")
                    break
                try:
                    tracker.require_budget()
                except BudgetExceeded as exc:
                    state.update(status="budget_exhausted", stop_reason=str(exc))
                    break
                before = dict(tracker.counters)
                attempt_id = tracker.record_root_attempt()
                state["active"] = {"id": attempt_id, "before": before, "elapsed": 0.0}
                sampler.start(root)
                clock_started, elapsed_before = time.perf_counter(), 0.0
            active = state["active"]
            try:
                result = sampler.attempt(on_step=step_checkpoint)
            except Exception as exc:
                context.write_attempt(AttemptRecord(context.identity.run_id, active["id"], context.identity.method, root.group,
                    "error", elapsed_seconds=elapsed_before+time.perf_counter()-clock_started,
                    counters={key: value-active["before"][key] for key, value in tracker.counters.items()},
                    reason=f"{type(exc).__name__}: {exc}"))
                raise
            context.write_attempt(AttemptRecord(context.identity.run_id, active["id"], context.identity.method, root.group,
                result.kind, leaf_visits=result.leaf_visits, smiles=result.smiles, state=result.state, reward_result=result.reward_result,
                elapsed_seconds=elapsed_before+time.perf_counter()-clock_started,
                counters={key: value-active["before"][key] for key, value in tracker.counters.items()}, reason=result.reason))
            selection = selector.consider(result.smiles, result.reward_result) if result.kind == "complete" else None
            if selection is not None:
                identity, identity_type = selection
                context.write_sample(SampleRecord(context.identity.run_id, len(selector.selected), active["id"], root.group,
                    result.smiles, identity, result.reward_result, identity_type=identity_type))
                progress.update(1)
            tracker.event("generation_attempt", attempt_id=active["id"], kind=result.kind,
                          selection=selector.last_reason if result.kind == "complete" else result.kind,
                          stats=result.stats, reason=result.reason)
            if active["id"] % options["log_every"] == 0:
                context.log(f"attempts={active['id']}; unique={len(selector.selected)}/{target}; NFE={tracker.raw_nfe}")
            state["active"] = None
            if result.kind == "budget_exhausted" and result.reason != "search_step_budget":
                state.update(status="budget_exhausted", stop_reason=result.reason)
                break
            if active["id"] % options["checkpoint_every"] == 0:
                checkpoint()
    if len(selector.selected) >= target:
        state.update(status="completed", stop_reason="target_unique_reached")
    elif state["status"] is None:
        state.update(status="timeout", stop_reason="max_root_attempts")
    state["stage"] = "finished"
    checkpoint()
    snapshot = context.output_path("sampler_state.pt")
    with _atomic_file(store.path(snapshot)) as handle:
        torch.save(sampler.state_dict(), handle)
    report = context.output_path("generation.json")
    store.write_json(report, {"version": GENERATION_VERSION, "status": state["status"], "stop_reason": state["stop_reason"],
        "identity": context.identity.to_dict(), "runtime": actual, "checker_versions": checker_versions(),
        "protocol": context.config["protocol"], "selected_samples": len(selector.selected), "target_attained": len(selector.selected) >= target,
        "identity_version": IDENTITY_VERSION,
        "sampler": sampler.metadata, "ideal_value_claim": False, "test_used_for_selection": False,
        "attempts": context.output_path("attempts.jsonl"), "samples": context.output_path("samples.jsonl"),
        "candidate_and_value_traces": context.output_path("events.jsonl"), "sampler_state": snapshot, "tracking": tracker.summary()})
    manifest = context.output_path("generation_manifest.json")
    store.create_manifest(manifest, kind="generation", config_hash=context.identity.config_hash,
        files=[report, snapshot, context.output_path("attempts.jsonl"), context.output_path("samples.jsonl")],
        metadata={"version": GENERATION_VERSION, "status": state["status"], "suite": context.identity.suite,
                  "method": context.identity.method, "group": root.group, "dependency_hashes": context.identity.dependency_hashes,
                  "policy_context": sampler.context_hash})
    if on_checkpoint is not None:
        on_checkpoint()
    return TaskResult(f"unique={len(selector.selected)}/{target}; attempts={tracker.counters['root_attempts']}; NFE={tracker.raw_nfe}; report={report}",
                      status=state["status"], artifacts=(report, manifest, snapshot))


def add_arguments(parser: argparse.ArgumentParser, method: str) -> None:
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--mdlm-report", help="T05 training.json selecting the shared checkpoint.")
    model.add_argument("--checkpoint", help="Project-relative shared MDLM checkpoint manifest.")
    if method == "mdm_vgb":
        verifier = parser.add_mutually_exclusive_group()
        verifier.add_argument("--verifier-report", help="T07 training.json selecting the learned verifier.")
        verifier.add_argument("--verifier-checkpoint", help="Project-relative learned-verifier checkpoint manifest.")


def execute_generation(method: str, argv: Sequence[str] | None = None, *, execution="local", on_checkpoint=None):
    task_id, config_path = TASKS[method]
    return run_task(task_id, lambda context: generate(context, on_checkpoint=on_checkpoint), argv=argv,
        execution=execution, description="Generate SMILES with " + method, default_config=config_path, supports_group=True,
        method=method, validator=lambda config: validate_generation(config, method), dependencies=generation_dependencies,
        add_arguments=lambda parser: add_arguments(parser, method),
        argument_overrides=lambda args: source_overrides(mdlm_report=args.mdlm_report or "", checkpoint=args.checkpoint or "",
            verifier_report=getattr(args, "verifier_report", None) or "", verifier_checkpoint=getattr(args, "verifier_checkpoint", None) or ""))
