"""T15: preflight, lock and resume a matched comparison without training models.

Local and Modal coordinators share the plan and dispatch contract. Generation
validation uses independent random trials, not a held-out set of prompts.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from tqdm.auto import tqdm

from src.artifacts import ArtifactStore, Dependency, sha256_file
from src.config import Config, config_digest, load_config, project_path
from src.schemas import GROUPS, RunIdentity, RunRecord, TaskResult, require_count, require_identifier
from src.tasks import generate_cars, generation
from src.tasks import evaluate as evaluation_task
from src.tasks import report as report_task
from src.tasks.common import TaskContext, run_task

VERSION = "smiles-comparison-v1"
TERMINAL_GENERATION = {"completed", "timeout", "budget_exhausted"}
VALIDATION_SEED_OFFSET = 2**31
Dispatch = Callable[[str, dict[str, Any]], Any]


def validate_comparison(config: Config) -> None:
    options = config["comparison"]
    if options["version"] != VERSION or options["action"] not in ("run", "lock"):
        raise ValueError("Unsupported comparison version/action")
    if options["purpose"] not in ("tuning", "final"):
        raise ValueError("comparison.purpose must be tuning or final")
    expected = ["cars_ar"] if config["suite"] == "replication" else ["cars", "mdm_vgb", "hybrid"]
    if options["methods"] != expected or type(options["include_baseline"]) is not bool:
        raise ValueError("Keep the three primary methods together and AR replication separate")
    if config["suite"] == "replication" and options["include_baseline"]:
        raise ValueError("AR replication cannot include the MDLM baseline")
    if config["group"] is not None or config["groups"] != list(GROUPS):
        raise ValueError("Comparison covers all three groups; do not pass a single group")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Comparison seeds must be a nonempty distinct list")
    for seed in seeds:
        require_count(seed, "comparison seed")
        if seed >= VALIDATION_SEED_OFFSET:
            raise ValueError("Logical seeds must be below 2**31 to reserve validation streams")
    if config["suite"] != "smoke" and len(seeds) < 3:
        raise ValueError("Primary and replication comparisons require at least three seeds")
    protocol = config["protocol"]
    if any(protocol[key] is not None for key in ("max_model_nfe", "max_compute_seconds")):
        raise ValueError("Set comparison.budgets; coordinator-wide compute caps are not job allowances")
    if config["suite"] != "smoke" and (protocol["target_unique"], protocol["max_root_attempts"]) != (100, 1000):
        raise ValueError("Scientific comparisons retain the 100-unique / 1,000-root protocol")
    budgets = options["budgets"]
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("Provide comparison compute budgets")
    names, allowances = set(), set()
    for budget in budgets:
        if set(budget) != {"name", "max_model_nfe", "max_compute_seconds"}:
            raise ValueError("Each budget needs a name and both explicit cap fields")
        require_identifier(budget["name"], "budget name")
        nfe, seconds = budget["max_model_nfe"], budget["max_compute_seconds"]
        if nfe is not None:
            require_count(nfe, "budget NFE", positive=True)
        if seconds is not None and (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("Time allowances must be finite and positive")
        if nfe is None and seconds is None:
            raise ValueError("Each job needs a finite NFE or compute-time allowance")
        allowance = (nfe, seconds)
        if budget["name"] in names or allowance in allowances:
            raise ValueError("Budget names and allowances must be distinct")
        names.add(budget["name"])
        allowances.add(allowance)
    if set(options["parameters"]) != {"search", "hybrid"} or set(options["parameters"]["hybrid"]) != {"m"}:
        raise ValueError("Parameters accept shared search settings and hybrid.m")
    require_count(options["parameters"]["search"]["N"], "search.N", positive=True)
    require_count(options["parameters"]["hybrid"]["m"], "hybrid.m", positive=True)
    for name in ("N", "m"):
        grid = options["validation_sweep"][name]
        if not isinstance(grid, list) or not grid or len(set(grid)) != len(grid):
            raise ValueError("Validation grids must be nonempty distinct lists")
        for value in grid:
            require_count(value, "validation " + name, positive=True)
    if set(options["task_configs"]) != {"cars", "cars_ar", "base", "mdm_vgb", "hybrid", "evaluate", "report"}:
        raise ValueError("Declare all generation/evaluation/report task configurations")
    for path in options["task_configs"].values():
        project_path(path)
    if config["suite"] == "replication" and any(options[section][key] is not None
            for section in ("model", "verifier") for key in ("training_report", "checkpoint_manifest")):
        raise ValueError("AR replication cannot accept MDLM/verifier sources")
    for key in ("lock_manifest", "lock_from"):
        if options[key] is not None:
            project_path(options[key])
    if options["action"] == "lock":
        if not options["lock_from"] or not options["selected_variant"] or options["lock_manifest"]:
            raise ValueError("Lock requires --lock-from and an explicitly reviewed --select-variant")
        require_identifier(options["selected_variant"], "selected variant")
    elif options["lock_from"] is not None or options["selected_variant"] is not None:
        raise ValueError("Selection arguments belong to the lock action")
    elif options["purpose"] == "tuning" and options["lock_manifest"] is not None:
        raise ValueError("Validation sweep cannot consume a final selection")
    elif options["purpose"] == "final" and config["suite"] != "smoke" and not options["lock_manifest"]:
        raise ValueError("Final comparison requires a reviewed validation --lock-manifest")
    if type(options["retrosynthesis"]) is not bool:
        raise ValueError("comparison.retrosynthesis must be boolean")
    if not isinstance(options["value_manifests"], list) or len(set(options["value_manifests"])) != len(options["value_manifests"]):
        raise ValueError("Value inputs must be distinct manifest paths")
    for path in options["value_manifests"]:
        project_path(path)
    if config["suite"] == "replication" and options["value_manifests"]:
        raise ValueError("MDLM value evaluations cannot enter AR replication")


def project_hashes() -> dict[str, str]:
    """Freeze runtime source and configuration; never inspect reference folders."""
    paths = ["pyproject.toml"]
    for root in ("src", "configs", "modal"):
        project_path(root)
        if not os.path.isdir(root):
            raise FileNotFoundError("Missing runtime project directory: " + root)
        for directory, children, files in os.walk(root):
            children[:] = sorted(name for name in children if not name.startswith(".") and name != "__pycache__")
            paths.extend(os.path.join(directory, name) for name in sorted(files) if name.endswith((".py", ".yaml")))
    return {project_path(path): sha256_file(path) for path in sorted(paths)}


def design(config: Config) -> dict:
    options = config["comparison"]
    return {"suite": config["suite"], "groups": config["groups"], "seeds": config["seeds"],
        "reward": config["reward"], "protocol": config["protocol"], "data": config["data"], "models": config["models"],
        "paths": {key: value for key, value in config["paths"].items() if key != "artifact_root"},
        "methods": options["methods"], "include_baseline": options["include_baseline"], "budgets": options["budgets"],
        "search": {key: value for key, value in options["parameters"]["search"].items() if key != "N"},
        "model_runtime": {key: options["model"][key] for key in ("device", "dtype")},
        "retrosynthesis": options["retrosynthesis"], "task_configs": options["task_configs"],
        "deterministic": config["runtime"]["deterministic"], "validation_seed_offset": VALIDATION_SEED_OFFSET}


def variants(config: Config, selection: Mapping | None = None) -> list[dict]:
    options = config["comparison"]
    if config["suite"] == "replication":
        return [{"id": "ar", "N": None, "m": None}]
    if selection is not None:
        return [dict(selection["variant"])]
    if options["purpose"] == "tuning":
        return [{"id": f"N{n}_m{m}", "N": n, "m": m}
                for n in options["validation_sweep"]["N"] for m in options["validation_sweep"]["m"]]
    n, m = options["parameters"]["search"]["N"], options["parameters"]["hybrid"]["m"]
    return [{"id": f"N{n}_m{m}", "N": n, "m": m}]


def _overrides(values: Mapping) -> list[str]:
    return [key + "=" + json.dumps(value, allow_nan=False) for key, value in values.items()]


def shared_overrides(config: Config, task_path: str) -> dict:
    # T16 declares neither data nor models; T08 declares data but not models.
    # Keep the existing strict override parser rather than adding undeclared keys.
    declared = load_config(task_path)
    return {key: copy.deepcopy(config[key]) for key in ("suite", "seeds", "reward", "protocol", "paths", "runtime", "data", "models")
            if key in declared}


def child_config(config: Config, task: str, overrides: Mapping, validator) -> Config:
    path = config["comparison"]["task_configs"][task]
    values = {**shared_overrides(config, path), **overrides}
    return load_config(path, overrides=_overrides(values), validator=validator)


def _comparison_input(store: ArtifactStore, path: str, suite: str) -> tuple[dict, dict]:
    manifest = store.require_manifest(path, kind="comparison", metadata={"suite": suite, "purpose": "tuning"})
    plan_path, results_path = manifest.metadata["plan"], manifest.metadata["results"]
    if plan_path not in manifest.files or results_path not in manifest.files:
        raise ValueError("Comparison manifest must cover its plan and results")
    plan, results = store.read_json(plan_path), store.read_json(results_path)
    if plan["version"] != VERSION or plan["purpose"] != "tuning" or results["plan_hash"] != config_digest(plan):
        raise ValueError("Invalid validation comparison provenance")
    record = RunRecord.from_dict(store.read_json(os.path.join(os.path.dirname(path), "manifest.json")))
    if (record.identity.to_dict() != plan["identity"] or record.identity.config_hash != manifest.config_hash
            or record.identity.task_id != "T15" or record.identity.suite != suite):
        raise ValueError("Comparison run identity differs from its plan")
    if record.status != "completed" or manifest.metadata["status"] != "completed":
        raise ValueError("Review a completed validation matrix before locking")
    return plan, results


def checked_selection(store: ArtifactStore, path: str, suite: str) -> dict:
    manifest = store.require_manifest(path, kind="comparison_selection", metadata={"suite": suite})
    selection_path = manifest.metadata["selection"]
    if selection_path not in manifest.files:
        raise ValueError("Selection manifest must cover its selection document")
    selection = store.read_json(selection_path)
    record = RunRecord.from_dict(store.read_json(os.path.join(os.path.dirname(selection_path), "manifest.json")))
    if record.status != "completed" or record.identity.task_id != "T15" or record.identity.config_hash != manifest.config_hash:
        raise ValueError("Selection must belong to a completed T15 lock run")
    plan, results = _comparison_input(store, selection["validation_manifest"], suite)
    if (selection["version"] != VERSION or selection["test_used_for_selection"] is not False
            or selection["validation_manifest"] not in manifest.files
            or selection["design"] != plan["design"] or selection["project_hashes"] != plan["project_hashes"]
            or selection["models"] != plan["models"] or selection["shared_hashes"] != plan["shared_hashes"]
            or selection["variant"] not in plan["variants"]):
        raise ValueError("Selection document differs from reviewed validation inputs")
    _selected_metrics(store, plan, results, selection["variant"]["id"])
    return selection


def prepare(config: Config, store: ArtifactStore) -> tuple[dict, dict[str, Dependency]]:
    """Resolve every prerequisite before dispatching any work; no download calls."""
    options, suite = config["comparison"], config["suite"]
    if options["action"] == "lock":
        plan, results = _comparison_input(store, options["lock_from"], suite)
        if plan["project_hashes"] != project_hashes():
            raise ValueError("Project changed since validation; review a new validation run")
        _selected_metrics(store, plan, results, options["selected_variant"])
        return {"source_plan": plan, "source_results": results}, {
            "validation": Dependency(options["lock_from"], manifest=True, kind="comparison")}
    selection, dependencies = None, {}
    if options["lock_manifest"]:
        selection = checked_selection(store, options["lock_manifest"], suite)
        if selection["design"] != design(config) or selection["project_hashes"] != project_hashes():
            raise ValueError("Final settings/source differ from the reviewed validation selection")
        dependencies["selection"] = Dependency(options["lock_manifest"], manifest=True, kind="comparison_selection")
    common = {"group": GROUPS[0], "seed": config["seeds"][0]}
    sources = {"mdlm": None, "verifier": None}
    if suite == "replication":
        probe = child_config(config, "cars_ar", common, generate_cars.validate_cars)
        dependencies.update(generate_cars.generation_dependencies(probe, store))
    else:
        source = {key: options["model"][key] for key in ("training_report", "checkpoint_manifest", "device", "dtype")}
        if selection is not None and not any(source[key] for key in ("training_report", "checkpoint_manifest")):
            source["checkpoint_manifest"] = selection["models"]["mdlm"]
        probe = child_config(config, "base", {**common, "generation.model": source},
                             lambda cfg: generation.validate_generation(cfg, "base"))
        sources["mdlm"], model_deps = generation.mdlm_source(probe, store)
        dependencies.update(model_deps)
        dependencies.update(generation.generation_dependencies(probe, store))
        verifier = dict(options["verifier"])
        if selection is not None and not any(verifier.values()):
            verifier["checkpoint_manifest"] = selection["models"]["verifier"]
        guided = child_config(config, "mdm_vgb", {**common,
            "generation.model": source, "value.learned.training_report": verifier["training_report"],
            "value.learned.checkpoint_manifest": verifier["checkpoint_manifest"]},
            lambda cfg: generation.validate_generation(cfg, "mdm_vgb"))
        sources["verifier"], verifier_deps = generation.verifier_source(guided, store,
            mdlm_sha256=sha256_file(store.require_file(sources["mdlm"])))
        dependencies.update(verifier_deps)
    if selection is not None and sources != selection["models"]:
        raise ValueError("Final checkpoints differ from the locked shared checkpoints")
    retro_path = config["models"]["retrostar"]["manifest"]
    if options["retrosynthesis"] and os.path.isfile(store.path(retro_path)):
        from src.autoregressive import require_model_snapshot

        require_model_snapshot(store, config["models"]["retrostar"])
        dependencies["retrostar"] = Dependency(retro_path, manifest=True, kind="pretrained_model")
    hashes = store.check_dependencies(dependencies)
    shared_hashes = {key: hashes[key] for key in ("mdlm", "verifier", "tokenizer", "vocabulary", "benchmark", "llama") if key in hashes}
    shared_hashes["retrostar"] = hashes.get("retrostar")  # Lock absence too; never replace an unavailable score with zero.
    if selection is not None and shared_hashes != selection["shared_hashes"]:
        raise ValueError("Locked checkpoint/tokenizer/benchmark contents changed since validation")
    for index, path in enumerate(options["value_manifests"]):
        payload, _ = report_task.checked_evaluation(store, path, kind="value_evaluation", suite=suite)
        if payload["purpose"] != options["purpose"] or payload["source_split"] != ("validation" if options["purpose"] == "tuning" else "test"):
            raise ValueError("Value evaluation purpose/split differs from the comparison")
        value_hashes = payload["identity"]["dependency_hashes"]
        if any(value_hashes.get(key) != hashes.get(key) for key in ("mdlm", "verifier", "tokenizer")):
            raise ValueError("Value evaluation uses different shared checkpoints/tokenizer")
        dependencies[f"value_{index}"] = Dependency(path, manifest=True, kind="value_evaluation")
    prepared = {"models": sources, "selection": selection, "variants": variants(config, selection),
                "project_hashes": project_hashes(), "design": design(config), "shared_hashes": shared_hashes}
    # Evaluate/report configuration errors must also fail before generation.
    child_config(config, "evaluate", {**common,
        "evaluation.source_run": "outputs/preflight", "evaluation.retrosynthesis.enabled": options["retrosynthesis"]},
        evaluation_task.validate_evaluation)
    child_config(config, "report", {**common, "report.expected_generation_runs": ["outputs/preflight"]}, report_task.validate_report)
    return prepared, dependencies


def make_spec(config: Config, store: ArtifactStore, *, stage: str, run_id: str,
              overrides: Mapping, method: str | None = None, dependency_hashes: Mapping | None = None) -> dict:
    task = method if stage == "generation" else stage
    task_path = config["comparison"]["task_configs"][task]
    merged = {**shared_overrides(config, task_path), **overrides}
    arguments = ["--config", task_path, "--run-id", run_id]
    for item in _overrides(merged):
        arguments.extend(["--set", item])
    validator = (generate_cars.validate_cars if method in ("cars", "cars_ar")
                 else (lambda cfg: generation.validate_generation(cfg, method)) if stage == "generation"
                 else evaluation_task.validate_evaluation if stage == "evaluate" else report_task.validate_report)
    resolved = load_config(task_path, overrides=_overrides(merged), validator=validator)
    if dependency_hashes is None:
        dependencies = ((generate_cars.generation_dependencies if method in ("cars", "cars_ar") else generation.generation_dependencies)
                        (resolved, store) if stage == "generation" else {})
        dependency_hashes = store.check_dependencies(dependencies)
    task_id = ("T09" if method in ("cars", "cars_ar") else generation.TASKS[method][0]) if stage == "generation" else ("T14" if stage == "evaluate" else "T16")
    identity = RunIdentity(task_id, run_id, config_digest(resolved), config["suite"], resolved["seed"],
        config["reward"]["version"], method=method, group=resolved.get("group"), dependency_hashes=dict(dependency_hashes))
    directory = store.run_directory(identity)
    artifact = (os.path.join(directory, "generation_manifest.json") if stage == "generation"
                else os.path.join(directory, "metrics_manifest.json") if stage == "evaluate"
                else store.directory("reports", config["suite"], run_id, "manifest.json"))
    return {"stage": stage, "method": method, "argv": arguments, "identity": identity.to_dict(),
            "directory": directory, "artifact": artifact, "search": resolved.get("search")}


def build_jobs(context: TaskContext, prepared: Mapping) -> list[dict]:
    config, options, store = context.config, context.config["comparison"], context.store
    methods = options["methods"] + (["base"] if options["include_baseline"] else [])
    jobs, by_key, kernels = [], {}, {}
    selection_hash = context.identity.dependency_hashes.get("selection")
    experiment = {"purpose": options["purpose"], "split": "validation" if options["purpose"] == "tuning" else "test",
                  "selection_sha256": selection_hash, "validation_seed_offset": VALIDATION_SEED_OFFSET}
    for variant in prepared["variants"]:
        for group in GROUPS:
            for logical_seed in config["seeds"]:
                seed = logical_seed + (VALIDATION_SEED_OFFSET if options["purpose"] == "tuning" else 0)
                for budget in options["budgets"]:
                    for method in methods:
                        values = {"seed": seed, "group": group,
                            "runtime": {**config["runtime"], "comparison": experiment},
                            "protocol.max_model_nfe": budget["max_model_nfe"],
                            "protocol.max_compute_seconds": budget["max_compute_seconds"]}
                        if method != "cars_ar":
                            section = "cars.model" if method == "cars" else "generation.model"
                            values[section] = {**options["model"], "training_report": None,
                                               "checkpoint_manifest": prepared["models"]["mdlm"]}
                        if method in ("mdm_vgb", "hybrid"):
                            for key, value in {**options["parameters"]["search"], "N": variant["N"]}.items():
                                values["search." + key] = value
                            values["value.learned.training_report"] = None
                            values["value.learned.checkpoint_manifest"] = prepared["models"]["verifier"] if method == "mdm_vgb" else None
                            if method == "hybrid":
                                values["value.m"] = variant["m"]
                        key = config_digest({"method": method, "values": values})
                        if key in by_key:
                            by_key[key]["variants"].append(variant["id"])
                            continue
                        run_id = context.identity.run_id + "-" + method + "-" + key[:12]
                        keys = ("llama", "benchmark") if method == "cars_ar" else ("mdlm", "tokenizer", "vocabulary", "benchmark")
                        if method == "mdm_vgb":
                            keys = (*keys, "verifier")
                        pinned_hashes = {name: prepared["shared_hashes"][name] for name in keys}
                        spec = make_spec(config, store, stage="generation", run_id=run_id, overrides=values,
                                         method=method, dependency_hashes=pinned_hashes)
                        if method in ("mdm_vgb", "hybrid"):
                            n = variant["N"]
                            if n in kernels and kernels[n] != spec["search"]:
                                raise ValueError("Learned and hybrid search kernels differ; use shared comparison.parameters.search overrides")
                            kernels[n] = spec["search"]
                        spec.update(id=key, variants=[variant["id"]], logical_seed=logical_seed, budget=budget["name"])
                        by_key[key] = spec
                        jobs.append(spec)
    # All primary jobs must resolve to the same physical model/reward/tokenizer.
    expected = {key: context.identity.dependency_hashes[key] for key in ("mdlm", "tokenizer", "vocabulary", "benchmark")
                if key in context.identity.dependency_hashes}
    for job in jobs:
        identity = job["identity"]
        if any(identity["dependency_hashes"].get(key) != value for key, value in expected.items()):
            raise ValueError("A comparison job resolves to different shared model/tokenizer/benchmark dependencies")
        if identity["reward_version"] != context.identity.reward_version:
            raise ValueError("A comparison job uses a different reward")
    return jobs


def _recover_paid_work(context: TaskContext) -> None:
    """Charge durable interrupted work again only in the budget, not sample IDs."""
    if not context.resumed:
        return
    previous, current = context.record.tracking, context.tracker.snapshot()
    names = [name for name in current.counters if "forward_" in name or name.startswith("flop_profiled_")]
    deltas = {name: previous.counters.get(name, 0) - current.counters[name] for name in names}
    if any(value < 0 for value in deltas.values()):
        return  # A hard kill can leave the RunRecord older than the checkpoint.
    for name, count in deltas.items():
        context.tracker.increment(name, count)
    for name in ("compute_seconds", "gpu_seconds", "flops"):
        old, new = getattr(previous, name), getattr(current, name)
        for key, value in old.items():
            delta = max(0.0, value - new.get(key, 0.0))
            if delta:
                getattr(context.tracker, name)[key] = getattr(context.tracker, name).get(key, 0.0) + delta
    context.tracker.peak_gpu_memory_bytes = max(previous.peak_gpu_memory_bytes, current.peak_gpu_memory_bytes)
    context.tracker.event("comparison_resume_paid_work", counters=deltas,
        limitation="Uncommitted hard-kill work absent from durable tracking cannot be reconstructed")


def execute_spec(stage: str, spec: dict, *, execution: str = "local", on_checkpoint: Callable[[], None] | None = None):
    """Worker entrypoint; isolated child run identities are fixed by the plan."""
    if stage != spec["stage"]:
        raise ValueError("Dispatch stage differs from its job")
    argv = spec["argv"] + (["--resume"] if spec.get("resume") else [])
    if stage == "evaluate":
        return evaluation_task.execute(argv, execution=execution)
    if stage == "report":
        return report_task.execute(argv, execution=execution)
    method = spec["method"]
    cars = method in ("cars", "cars_ar")

    def task(context):
        if context.identity.to_dict() != spec["identity"]:
            raise ValueError("Worker configuration/dependencies differ from the planned identity")
        _recover_paid_work(context)
        return (generate_cars.generate_cars if cars else generation.generate)(context, on_checkpoint=on_checkpoint)

    return run_task(spec["identity"]["task_id"], task, argv=argv, description="Execute a pinned comparison generation job.",
        default_config="configs/cars.yaml" if cars else generation.TASKS[method][1], execution=execution, method=method,
        supports_group=True, validator=generate_cars.validate_cars if cars else lambda cfg: generation.validate_generation(cfg, method),
        dependencies=generate_cars.generation_dependencies if cars else generation.generation_dependencies)


def observed(store: ArtifactStore, spec: Mapping) -> dict:
    """Adopt finished jobs even when the coordinator died before saving results."""
    identity = RunIdentity.from_dict(spec["identity"])
    directory = spec["directory"]
    if not os.path.isfile(store.path(os.path.join(directory, "manifest.json"))):
        return {"status": "missing", "done": False, "directory": directory}
    record = store.read_run(identity)
    result = {"status": record.status, "done": False, "directory": directory,
              "artifact": spec["artifact"], "tracking": record.tracking.to_dict(), "error": record.error,
              "attained": record.tracking.counters.get("selected_samples", 0),
              "count_provenance": "durable_run_counter; journals validated separately by T14"}
    artifact_exists = os.path.isfile(store.path(spec["artifact"]))
    if spec["stage"] == "generation" and record.status in TERMINAL_GENERATION and artifact_exists:
        store.require_manifest(spec["artifact"], kind="generation", config_hash=identity.config_hash,
            metadata={"suite": identity.suite, "status": record.status, "dependency_hashes": identity.dependency_hashes})
        payload = store.read_json(os.path.join(directory, "generation.json"))
        if payload["identity"] != identity.to_dict() or payload["status"] != record.status:
            raise ValueError("Generation identity/status differs from the comparison plan")
        result.update(done=True, attained=payload["selected_samples"], target_attained=payload["target_attained"],
                      stop_reason=payload["stop_reason"], count_provenance="checked_generation_report")
    elif spec["stage"] == "evaluate" and artifact_exists and record.status == "completed":
        payload, _ = report_task.checked_evaluation(store, spec["artifact"], kind="molecular_evaluation", suite=identity.suite)
        result["done"] = payload["status"] in ("available", "partial")
    elif spec["stage"] == "report" and artifact_exists and record.status in ("completed", "failed"):
        store.require_manifest(spec["artifact"], kind="report", config_hash=identity.config_hash, metadata={"suite": identity.suite})
        result["done"] = record.status == "completed"
        result["published"] = True
    return result


def source_fingerprint(store: ArtifactStore, directory: str) -> str:
    names = ("manifest.json", "resolved_config.yaml", "generation_manifest.json", "attempts.jsonl", "samples.jsonl", "events.jsonl")
    return config_digest({name: sha256_file(store.path(os.path.join(directory, name)))
        for name in names if os.path.isfile(store.path(os.path.join(directory, name)))})


def _selected_metrics(store: ArtifactStore, plan: Mapping, results: Mapping, variant_id: str) -> list[str]:
    if variant_id not in {variant["id"] for variant in plan["variants"]}:
        raise ValueError("Selected variant does not exist in the validation matrix")
    paths = []
    for job in plan["jobs"]:
        if variant_id not in job["variants"]:
            continue
        result = results["jobs"].get(job["id"], {})
        if not result.get("generation", {}).get("done") or not result.get("evaluate", {}).get("done"):
            raise ValueError("Selected validation cell contains unfinished/invalid jobs")
        current = observed(store, job)
        if not current["done"] or current["status"] != result["generation"]["status"]:
            raise ValueError("Selected generation changed after validation")
        path = result["evaluate"]["artifact"]
        payload, _ = report_task.checked_evaluation(store, path, kind="molecular_evaluation", suite=plan["identity"]["suite"])
        if (payload["source_run"] != job["directory"] or payload["identity"] != job["identity"]
                or payload["comparability"]["experiment"]["purpose"] != "tuning"):
            raise ValueError("Selected metric does not belong to its validation job")
        paths.append(path)
    if not paths or not results["reports"].get(variant_id, {}).get("done"):
        raise ValueError("Selected validation report is missing or incomplete")
    store.require_manifest(results["reports"][variant_id]["artifact"], kind="report")
    return paths


def lock(context: TaskContext, prepared: Mapping) -> TaskResult:
    options, store = context.config["comparison"], context.store
    plan, results = prepared["source_plan"], prepared["source_results"]
    selected = next(variant for variant in plan["variants"] if variant["id"] == options["selected_variant"])
    metrics = _selected_metrics(store, plan, results, selected["id"])
    path, manifest_path = context.output_path("selection.json"), context.output_path("selection_manifest.json")
    store.write_json(path, {"version": VERSION, "variant": selected, "models": plan["models"],
        "shared_hashes": plan["shared_hashes"],
        "design": plan["design"], "project_hashes": plan["project_hashes"], "validation_manifest": options["lock_from"],
        "validation_metrics": metrics, "selection": "explicit_user_review", "test_used_for_selection": False,
        "ideal_value_claim": False})
    store.create_manifest(manifest_path, kind="comparison_selection", config_hash=context.identity.config_hash,
        files=[path, options["lock_from"]], metadata={"version": VERSION, "suite": context.identity.suite,
            "selection": path, "variant": selected["id"], "test_used_for_selection": False})
    context.log(f"locked {selected['id']}; selection={manifest_path}")
    return TaskResult(f"locked={selected['id']}; manifest={manifest_path}", artifacts=(path, manifest_path))


def compare(context: TaskContext, prepared: Mapping, *, dispatch: Dispatch | None = None,
            on_checkpoint: Callable[[], None] | None = None, reload_artifacts: Callable[[], None] | None = None) -> TaskResult:
    if context.config["comparison"]["action"] == "lock":
        return lock(context, prepared)
    config, store, options = context.config, context.store, context.config["comparison"]
    plan_path, results_path = context.output_path("job_plan.json"), context.output_path("job_results.json")
    jobs = build_jobs(context, prepared)
    plan = {"version": VERSION, "identity": context.identity.to_dict(), "purpose": options["purpose"],
        "design": prepared["design"], "project_hashes": prepared["project_hashes"], "models": prepared["models"],
        "shared_hashes": prepared["shared_hashes"],
        "variants": prepared["variants"], "jobs": jobs, "ideal_value_claim": False,
        "validation_definition": "Independent generation trials on the same benchmark prompts; not held-out molecules"}
    if context.resumed and os.path.isfile(store.path(plan_path)):
        if store.read_json(plan_path) != plan:
            raise ValueError("Comparison plan changed; resume with identical source/configuration/dependencies")
        results = store.read_json(results_path) if os.path.isfile(store.path(results_path)) else {
            "plan_hash": config_digest(plan), "jobs": {}, "reports": {}}
        if results["plan_hash"] != config_digest(plan):
            raise ValueError("Comparison results refer to another plan")
    else:
        if context.resumed and context.resume_state is not None:
            raise ValueError("Committed comparison checkpoint is missing its job plan")
        store.write_json(plan_path, plan)
        results = {"plan_hash": config_digest(plan), "jobs": {}, "reports": {}}

    def checkpoint():
        store.write_json(results_path, results)
        context.save_checkpoint({"version": VERSION, "plan_hash": results["plan_hash"]})
        if on_checkpoint is not None:
            on_checkpoint()

    def run(spec):
        state = observed(store, spec)
        if state["done"] or (spec["stage"] == "report" and state.get("published")):
            return state
        # A completed but corrupt/missing artifact cannot safely be overwritten.
        if state["status"] == "completed":
            raise ValueError("Completed child has missing artifacts: " + spec["directory"])
        retry = state["status"] != "missing"
        if retry and not context.resumed:
            raise ValueError("Existing unfinished child requires coordinator --resume")
        if project_hashes() != prepared["project_hashes"]:
            raise ValueError("Runtime source/configuration changed during the comparison")
        retro_path = config["models"]["retrostar"]["manifest"]
        actual_retro = sha256_file(store.require_file(retro_path)) if options["retrosynthesis"] and os.path.isfile(store.path(retro_path)) else None
        if actual_retro != prepared["shared_hashes"].get("retrostar"):
            raise ValueError("RetroStar availability/manifest changed after comparison preflight")
        request = {**spec, "resume": retry}
        checkpoint()  # Publish coordinator state before a Modal worker reloads.
        error = None
        try:
            if dispatch is None:
                execute_spec(spec["stage"], request, execution=config["runtime"]["execution"], on_checkpoint=on_checkpoint)
            else:
                dispatch(spec["stage"], request)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            context.log(f"{spec['stage']} {spec['identity']['run_id']}: {error}", level="ERROR")
        finally:
            if reload_artifacts is not None:
                reload_artifacts()  # Workers close/commit files before returning.
        state = observed(store, spec)
        if error:
            state["dispatch_error"] = error
        return state

    checkpoint()
    for job in context.progress(jobs, desc="Comparison generation/evaluation", unit="job"):
        entry = results["jobs"].setdefault(job["id"], {})
        entry["generation"] = run(job)
        checkpoint()
        # Failed/stopped generation still receives an audited partial evaluation.
        if entry["generation"]["status"] != "missing":
            fingerprint = source_fingerprint(store, job["directory"])
            spec = make_spec(config, store, stage="evaluate", run_id=job["identity"]["run_id"] + "-e-" + fingerprint[:12],
                overrides={"group": None, "seed": job["identity"]["seed"], "evaluation.source_run": job["directory"],
                           "evaluation.retrosynthesis.enabled": options["retrosynthesis"]})
            entry["evaluate"] = run(spec)
        else:
            entry["evaluate"] = {"status": "missing", "done": False}
        checkpoint()
    for variant in context.progress(prepared["variants"], desc="Comparison reports", unit="variant"):
        selected_jobs = [job for job in jobs if variant["id"] in job["variants"]]
        manifests = [results["jobs"][job["id"]]["evaluate"]["artifact"] for job in selected_jobs
                     if results["jobs"][job["id"]]["evaluate"].get("done")]
        fingerprint = config_digest({path: sha256_file(store.require_file(path)) for path in manifests})
        spec = make_spec(config, store, stage="report", run_id=context.identity.run_id + "-" + variant["id"] + "-r-" + fingerprint[:12],
            overrides={"group": None, "seed": config["seed"], "report.evaluation_manifests": manifests,
                "report.value_manifests": options["value_manifests"],
                "report.expected_generation_runs": [job["directory"] for job in selected_jobs]})
        results["reports"][variant["id"]] = run(spec)
        checkpoint()
    complete = all(entry["generation"].get("done") and entry["evaluate"].get("done") for entry in results["jobs"].values())
    complete = complete and all(entry.get("done") for entry in results["reports"].values())
    status = "completed" if complete else "failed"
    manifests = {entry[stage]["artifact"] for entry in results["jobs"].values() for stage in ("generation", "evaluate")
                 if entry[stage].get("done")}
    manifests.update(entry["artifact"] for entry in results["reports"].values() if entry.get("published") or entry.get("done"))
    manifest_path = context.output_path("comparison_manifest.json")
    store.create_manifest(manifest_path, kind="comparison", config_hash=context.identity.config_hash,
        files=[plan_path, results_path, context.output_path("resolved_config.yaml"), *sorted(manifests)],
        metadata={"version": VERSION, "suite": context.identity.suite, "purpose": options["purpose"], "status": status,
                  "plan": plan_path, "results": results_path, "ideal_value_claim": False})
    stopped = sum(entry["generation"]["status"] in {"timeout", "budget_exhausted"} for entry in results["jobs"].values())
    return TaskResult(f"jobs={len(jobs)}; stopped={stopped}; reports={len(results['reports'])}; manifest={manifest_path}",
                      status=status, artifacts=(plan_path, results_path, manifest_path))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--mdlm-report")
    model.add_argument("--checkpoint")
    verifier = parser.add_mutually_exclusive_group()
    verifier.add_argument("--verifier-report")
    verifier.add_argument("--verifier-checkpoint")
    parser.add_argument("--purpose", choices=("tuning", "final"))
    parser.add_argument("--lock-from", help="Completed validation comparison_manifest.json.")
    parser.add_argument("--select-variant", help="Variant ID chosen after reviewing validation reports.")
    parser.add_argument("--lock-manifest", help="Reviewed selection_manifest.json for final runs.")


def argument_overrides(arguments: argparse.Namespace) -> list[str]:
    values = {}
    for section, report, manifest in (("model", arguments.mdlm_report, arguments.checkpoint),
                                      ("verifier", arguments.verifier_report, arguments.verifier_checkpoint)):
        if report or manifest:
            values[f"comparison.{section}.training_report"] = project_path(report) if report else None
            values[f"comparison.{section}.checkpoint_manifest"] = project_path(manifest) if manifest else None
    for name in ("purpose", "lock_from", "lock_manifest"):
        if getattr(arguments, name):
            values["comparison." + name] = getattr(arguments, name)
    if arguments.select_variant:
        values["comparison.selected_variant"] = arguments.select_variant
    if arguments.lock_from or arguments.select_variant:
        values["comparison.action"] = "lock"
    return _overrides(values)


def execute(argv: Sequence[str] | None = None, *, execution: str = "local", dispatch: Dispatch | None = None,
            on_checkpoint: Callable[[], None] | None = None, reload_artifacts: Callable[[], None] | None = None):
    prepared = {}

    def dependencies(config, store):
        bundle, declared = prepare(config, store)
        prepared.update(bundle)
        return declared

    return run_task("T15", lambda context: compare(context, prepared, dispatch=dispatch, on_checkpoint=on_checkpoint,
        reload_artifacts=reload_artifacts), argv=argv, description="Run or lock a matched molecular comparison.",
        default_config="configs/comparison.yaml", execution=execution, supports_seed=False,
        validator=validate_comparison, dependencies=dependencies, add_arguments=add_arguments, argument_overrides=argument_overrides)


if __name__ == "__main__":
    try:
        result = execute()
        if result.status != "completed":
            raise SystemExit(1)
    except KeyboardInterrupt:
        tqdm.write("T15: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"T15: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
