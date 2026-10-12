"""T14: validate immutable generation journals and publish molecular metrics."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence

from tqdm.auto import tqdm

from src.artifacts import sha256_file
from src.config import Config, config_digest, project_path, read_yaml, validate_config
from src.data import InvalidSmiles, canonicalize_smiles
from src.grammar import GrammarError
from src.metrics import generation_metrics
from src.retrosynthesis import load_retrostar
from src.reward import RewardChecker
from src.schemas import AttemptRecord, RunRecord, SampleRecord, SmilesState, TaskResult
from src.tasks.common import TaskContext, run_task
from src.tasks.prepare_benchmark import read_benchmark_assets
from src.tasks.prepare_corpus import checker_versions
from src.tokenizer import SmilesTokenizer

METRICS_VERSION = "smiles-metrics-v1"


def validate_evaluation(config: Config) -> None:
    options = config["evaluation"]
    if options["version"] != METRICS_VERSION:
        raise ValueError("Unsupported molecular metrics version")
    project_path(options["source_run"])
    if options["sample_sets"] != ["outputs", "search_candidates", "selected"]:
        raise ValueError("Retain outputs, full search candidates and selected samples separately")
    fp, retro = options["fingerprint"], options["retrosynthesis"]
    if fp != {"radius": 2, "bits": 2048, "sensitivity_radii": [3]}:
        raise ValueError("Primary Morgan radius 2/2048 bits and radius-3 sensitivity are fixed")
    if options["novelty_reference"] != "all_prompt_molecular_identities":
        raise ValueError("Novelty must use all benchmark prompt molecule identities")
    if (type(retro["enabled"]) is not bool or retro["threshold"] != 0.4
            or retro["num_results"] != 1 or retro["expansion_topk"] != 50
            or retro["device"] not in ("auto", "cpu", "cuda")
            or type(retro["batch_size"]) is not int or retro["batch_size"] <= 0):
        raise ValueError("Unsupported original RetroStar proxy settings")
    unit = options["reference_forward_flops"]
    if unit is not None and (isinstance(unit, bool) or not isinstance(unit, (int, float)) or not 0 < unit < float("inf")):
        raise ValueError("reference_forward_flops must be positive and finite")


def _checked_journals(store, directory, identity, checker, tokenizer=None):
    attempts = [AttemptRecord.from_dict(row).to_dict() for row in store.read_jsonl(os.path.join(directory, "attempts.jsonl"))]
    samples = [SampleRecord.from_dict(row).to_dict() for row in store.read_jsonl(os.path.join(directory, "samples.jsonl"))]
    for index, row in enumerate(attempts, 1):
        if (row["attempt_id"] != index or row["run_id"] != identity.run_id
                or row["method"] != identity.method or row["group"] != identity.group):
            raise ValueError("Attempt journal identity/order mismatch")
        if row["kind"] == "complete":
            state = SmilesState.from_dict(row["state"]) if row["state"] is not None else None
            if state is not None and tokenizer is not None:
                result = checker.evaluate_state(state, tokenizer).to_dict()
                if state.terminal_status == "complete" and tokenizer.decode_state(state) != row["smiles"]:
                    raise ValueError("Attempt text differs from its token state")
            else:
                result = checker.evaluate_smiles(row["smiles"], identity.group).to_dict()
            # Error strings can change independently of checker semantics.
            if any(result[key] != row["reward_result"][key] for key in result if key != "error"):
                raise ValueError("Recorded terminal reward/identity differs from the current pinned checker")
    identities = set()
    previous_attempt = 0
    for index, row in enumerate(samples, 1):
        if (row["sample_id"] != index or row["run_id"] != identity.run_id or row["group"] != identity.group
                or row["attempt_id"] > len(attempts) or row["identity"] in identities
                or row["attempt_id"] <= previous_attempt):
            raise ValueError("Selected journal identity/order/uniqueness mismatch")
        attempt = attempts[row["attempt_id"]-1]
        if (attempt["kind"] != "complete" or row["smiles"] != attempt["smiles"]
                or row["reward_result"] != attempt["reward_result"]):
            raise ValueError("Selected sample is not its recorded successful root output")
        identities.add(row["identity"])
        previous_attempt = row["attempt_id"]
    return attempts, samples


def evaluate_source(context: TaskContext) -> tuple[dict, list[str]]:
    store, options = context.store, context.config["evaluation"]
    directory = options["source_run"]
    record_path = os.path.join(directory, "manifest.json")
    record = RunRecord.from_dict(store.read_json(record_path))
    identity = record.identity
    expected_methods = {"T08": ("base",), "T09": ("cars", "cars_ar"), "T10": ("mdm_vgb",), "T11": ("hybrid",)}
    if (directory != store.run_directory(identity) or identity.task_id not in ("T08", "T09", "T10", "T11")
            or identity.method not in expected_methods.get(identity.task_id, ())
            or identity.group is None or identity.suite != context.identity.suite):
        raise ValueError("Expected a generation directory in the same evaluation suite")
    if record.status == "running":
        raise ValueError("Generation is still running; evaluate after the worker has stopped")
    config_path = os.path.join(directory, "resolved_config.yaml")
    source_config = read_yaml(store.require_file(config_path))
    validate_config(source_config)
    if (config_digest(source_config) != identity.config_hash
            or source_config["reward"]["version"] != identity.reward_version
            or source_config["suite"] != identity.suite or source_config["group"] != identity.group
            or source_config["seed"] != identity.seed):
        raise ValueError("Generation RunRecord/config/reward metadata mismatch")
    if identity.task_id == "T09":
        from src.tasks.generate_cars import generation_dependencies
    else:
        from src.tasks.generation import generation_dependencies
    dependencies = generation_dependencies(source_config, store)
    if store.check_dependencies(dependencies) != identity.dependency_hashes:
        raise ValueError("Generation model/tokenizer/benchmark dependencies have changed")
    checker = RewardChecker(mode=source_config["reward"]["mode"], version=identity.reward_version,
                            asset_directory=source_config["paths"]["benchmark"])
    inputs = [record_path, config_path, *[dep.path for dep in dependencies.values()],
              *[os.path.join(directory, name) for name in ("attempts.jsonl", "samples.jsonl", "events.jsonl")]]
    report_path, manifest_path = (os.path.join(directory, name) for name in ("generation.json", "generation_manifest.json"))
    generation = None
    if os.path.isfile(store.path(report_path)):
        generation = store.read_json(report_path)
        store.require_manifest(manifest_path, kind="generation", config_hash=identity.config_hash,
                               metadata={"dependency_hashes": identity.dependency_hashes, "status": record.status})
        if (generation["identity"] != identity.to_dict() or generation["status"] != record.status
                or generation["protocol"] != source_config["protocol"]
                or generation["checker_versions"] != checker_versions()
                or generation.get("ideal_value_claim") is not False or generation.get("test_used_for_selection") is not False):
            raise ValueError("Generation report protocol/checker/model/selection provenance mismatch")
        for key, filename in (("attempts", "attempts.jsonl"), ("samples", "samples.jsonl")):
            if generation[key] != os.path.join(directory, filename):
                raise ValueError("Generation report references another run's journals")
        inputs.extend([report_path, manifest_path])
    elif record.status == "completed":
        raise FileNotFoundError("Completed generation has no generation.json")
    hashes = {path: sha256_file(store.require_file(path)) for path in inputs}
    tokenizer = (SmilesTokenizer.load(store, source_config["data"]["tokenizer"]["metadata_path"])
                 if "mdlm" in identity.dependency_hashes else None)
    attempts, samples = _checked_journals(store, directory, identity, checker, tokenizer)
    if record.tracking.counters.get("selected_samples", 0) != len(samples):
        raise ValueError("Selected journal and final counters disagree")
    if generation is not None and (generation["selected_samples"] != len(samples)
            or generation["target_attained"] != (len(samples) >= source_config["protocol"]["target_unique"])):
        raise ValueError("Generation attained count differs from its journals")
    _, examples = read_benchmark_assets(source_config)
    prompts = set()
    for texts in examples.values():
        for text in texts:
            try:
                prompts.add(canonicalize_smiles(text))
            except InvalidSmiles:
                pass
    prompt_raw = {text for texts in examples.values() for text in texts}
    if any(row["identity"] in prompts or row["smiles"] in prompt_raw for row in samples):
        raise ValueError("Selected samples contain prompt exemplars")
    candidates, full_candidates, noncomplete = [], [], 0
    events = store.read_jsonl(os.path.join(directory, "events.jsonl"))
    for index, event in enumerate(context.progress(events, desc="Candidate journals", unit="event")):
        if event["run_id"] != identity.run_id:
            raise ValueError("Event journal belongs to another run")
        if event["event"] == "search_terminal_candidate":
            state = SmilesState.from_dict(event["state"])
            if state.group != identity.group or tokenizer is None:
                raise ValueError("Candidate state/group/tokenizer mismatch")
            result = checker.evaluate_state(state, tokenizer)
            if result.reward != event["reward"]:
                raise ValueError("Candidate terminal reward mismatch")
            if state.terminal_status != "complete":
                noncomplete += 1
                full_candidates.append({"event_index": index, "state": state.to_dict(),
                    "terminal_status": state.terminal_status, "smiles": None, "reward_result": result.to_dict()})
                continue
            text = tokenizer.decode_state(state)
            candidate = {"event_index": index, "state": state.to_dict(), "terminal_status": state.terminal_status,
                         "smiles": text, "reward_result": result.to_dict()}
            full_candidates.append(candidate)
            candidates.append(candidate)
    candidates_path = context.output_path("full_candidates.jsonl")
    store.write_jsonl(candidates_path, full_candidates)
    retro = load_retrostar(store, context.config["models"]["retrostar"], options["retrosynthesis"])
    if retro.provenance:
        retro_path = retro.provenance["manifest"]
        hashes[retro_path] = retro.provenance["manifest_sha256"]
    with context.tracker.measure("molecular_metrics"):
        metrics = generation_metrics(attempts, samples, candidates, tracking=record.tracking.to_dict(),
            protocol=source_config["protocol"], prompts=prompts, fingerprint=options["fingerprint"], retro=retro,
            reference_forward_flops=options["reference_forward_flops"])
    for path, digest in hashes.items():
        store.require_file(path, checksum=digest)
    if context.resume_state and context.resume_state != {"input_hashes": hashes}:
        raise ValueError("Evaluation inputs changed since the saved checkpoint")
    context.save_checkpoint({"input_hashes": hashes})
    # Exclude source paths and seed from comparison keys, preserve scientific settings.
    comparability = {"suite": identity.suite, "backbone": identity.dependency_hashes.get("mdlm", identity.dependency_hashes.get("llama")),
        "experiment": source_config["runtime"].get("comparison"),
        "tokenizer": identity.dependency_hashes.get("tokenizer"), "benchmark": identity.dependency_hashes["benchmark"],
        "reward": source_config["reward"], "checkers": checker_versions(),
        "generation_runtime": (generation or {}).get("runtime"),
        "evaluation": {key: options[key] for key in ("version", "fingerprint", "novelty_reference")},
        "retro_source": context.config["models"]["retrostar"], "retro_threshold": 0.4,
        "retrosynthesis_enabled": options["retrosynthesis"]["enabled"],
        "retro_manifest_sha256": retro.provenance.get("manifest_sha256"),
        "protocol": {key: value for key, value in source_config["protocol"].items()
                     if key not in ("max_model_nfe", "max_compute_seconds")}}
    budget = {key: source_config["protocol"][key] for key in ("max_model_nfe", "max_compute_seconds")}
    variant = {"search": source_config["search"]} if "search" in source_config else {}
    if "value" in source_config:
        variant["value"] = {key: value for key, value in source_config["value"].items() if key != "learned"}
        if source_config["value"]["kind"] == "learned":
            variant["verifier"] = identity.dependency_hashes["verifier"]
            variant["verifier_batch_size"] = source_config["value"]["learned"]["batch_size"]
    if "cars" in source_config:
        variant["cars"] = {key: value for key, value in source_config["cars"].items()
                           if key not in ("model", "checkpoint_every", "log_every")}
    if "generation" in source_config:
        variant["baseline"] = source_config["generation"].get("baseline")
    payload = {"version": METRICS_VERSION, "status": "available" if generation is not None else "partial",
        "source_run": directory, "source_status": record.status, "stop_reason": (generation or {}).get("stop_reason", record.error),
        "identity": identity.to_dict(), "evaluation_identity": context.identity.to_dict(),
        "input_hashes": hashes, "source_config": config_path, "source_dependencies": {key: dep.path for key, dep in dependencies.items()},
        "comparability": comparability, "comparison_key": config_digest(comparability),
        "budget": budget, "budget_key": config_digest(budget), "variant": variant, "variant_key": config_digest(variant),
        "candidate_noncomplete_terminal_visits": noncomplete, "metrics": metrics,
        "full_candidates_path": candidates_path,
        "limitations": ["V_q is a policy value; no established equality to ideal V*",
                        "Original class SMARTS overlap; chain-extender membership is broad",
                        "Selected-set validity is conditional; use all returned outputs for validity",
                        "Failed/interrupted runs may have unrecorded work; hard-kill costs are unavailable"],
        "ideal_value_claim": False}
    return payload, list(hashes)


def evaluate(context: TaskContext) -> TaskResult:
    try:
        payload, inputs = evaluate_source(context)
    except (OSError, ValueError, TypeError, KeyError, GrammarError) as exc:
        payload = {"version": METRICS_VERSION, "status": "missing" if isinstance(exc, FileNotFoundError) else "invalid",
                   "source_run": context.config["evaluation"]["source_run"],
                   "evaluation_identity": context.identity.to_dict(), "reason": f"{type(exc).__name__}: {exc}"}
        try:
            stopped = RunRecord.from_dict(context.store.read_json(os.path.join(payload["source_run"], "manifest.json")))
            if stopped.identity.task_id in ("T08", "T09", "T10", "T11"):
                payload.update(identity=stopped.identity.to_dict(), source_status=stopped.status,
                    reported_tracking_attained=stopped.tracking.counters.get("selected_samples", 0),
                    reported_tracking_note="Final counter only; source journals could not be fully validated")
        except (OSError, ValueError, TypeError, KeyError):
            pass
        inputs = []
        context.log(payload["reason"], level="ERROR")
    report = context.output_path("metrics.json")
    context.store.write_json(report, payload)
    files = [report, context.output_path("resolved_config.yaml"), *inputs]
    if payload["status"] in ("available", "partial"):
        files.append(payload["full_candidates_path"])
        source_copy = os.path.join(payload["source_run"], "evaluations", context.identity.run_id, "metrics.json")
        context.store.write_json(source_copy, payload)
        files.append(source_copy)
    manifest = context.output_path("metrics_manifest.json")
    context.store.create_manifest(manifest, kind="molecular_evaluation", config_hash=context.identity.config_hash,
        files=list(dict.fromkeys(files)), metadata={"version": METRICS_VERSION, "status": payload["status"],
            "source_run": payload["source_run"], "metrics_path": report, "suite": context.identity.suite})
    good = payload["status"] in ("available", "partial")
    attained = payload.get("metrics", {}).get("attained", "unavailable")
    return TaskResult(f"metrics={payload['status']}; attained={attained}; report={report}",
                      status="completed" if good else "failed", artifacts=(report, manifest))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generation-run", help="Project-relative generation run directory, inside the artifact root.")


def argument_overrides(arguments: argparse.Namespace) -> list[str]:
    return ["evaluation.source_run="+json.dumps(arguments.generation_run)] if arguments.generation_run else []


def execute(argv: Sequence[str] | None = None, *, execution: str = "local"):
    return run_task("T14", evaluate, description="Evaluate completed or stopped generation runs.", argv=argv,
        default_config="configs/evaluation.yaml", execution=execution, validator=validate_evaluation,
        add_arguments=add_arguments, argument_overrides=argument_overrides)


if __name__ == "__main__":
    try:
        if execute().status != "completed":
            raise SystemExit(1)
    except KeyboardInterrupt:
        tqdm.write("T14: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"T14: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
