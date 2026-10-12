"""T16: publish reports from explicitly selected, checksum-checked evaluations."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Sequence

from tqdm.auto import tqdm

from src.artifacts import sha256_file
from src.config import Config, config_digest, project_path, read_yaml
from src.reporting import (
    generation_row,
    paired_comparisons,
    quality_frontiers,
    summarize_generation,
    value_rows,
    write_csv,
    write_plots,
)
from src.schemas import RunRecord, TaskResult, require_identifier
from src.tasks.common import TaskContext, run_task

REPORT_VERSION = "smiles-report-v1"


def validate_report(config: Config) -> None:
    options = config["report"]
    if options["version"] != REPORT_VERSION:
        raise ValueError("Unsupported report version")
    for key in ("evaluation_manifests", "value_manifests", "expected_generation_runs"):
        paths = options[key]
        if not isinstance(paths, list) or len(set(paths)) != len(paths):
            raise ValueError(f"report.{key} must list distinct project-relative paths")
        for path in paths:
            project_path(path)
    if not options["evaluation_manifests"] and not options["value_manifests"] and not options["expected_generation_runs"]:
        raise ValueError("Select evaluation manifests or explicitly expected generation runs")
    bootstrap = options["bootstrap"]
    if (type(bootstrap["repetitions"]) is not int or bootstrap["repetitions"] < 100
            or not 0 < bootstrap["confidence"] < 1 or type(bootstrap["seed"]) is not int or bootstrap["seed"] < 0):
        raise ValueError("Invalid seed-bootstrap settings")
    expected_quality = ["success_per_root", "attained", "validity", "membership", "unique_fraction", "novelty", "diversity", "retro_score"]
    expected_cost = ["raw_nfe", "verifier_forward_calls", "tracked_compute_seconds", "end_to_end_seconds", "inner_model_calls"]
    if options["quality_metrics"] != expected_quality or options["cost_metrics"] != expected_cost:
        raise ValueError("Keep molecular quality and cost metric groups explicit")
    required_files = {"summary", "aggregates", "paired", "frontiers", "value_accuracy", "missing", "quality_plot", "value_plot", "markdown"}
    if set(options["files"]) != required_files or len(set(options["files"].values())) != len(required_files):
        raise ValueError("Report output filenames must be complete and distinct")
    for key, filename in options["files"].items():
        require_identifier(filename, "report filename")
        suffix = ".png" if key.endswith("plot") else (".md" if key == "markdown" else ".csv")
        if not filename.endswith(suffix):
            raise ValueError(f"Wrong report filename extension: {key}")


def checked_evaluation(store, path: str, *, kind: str, suite: str) -> tuple[dict, str]:
    manifest = store.require_manifest(path, kind=kind, metadata={"suite": suite})
    candidates = [filename for filename in manifest.files if os.path.basename(filename) == "metrics.json"]
    if kind == "molecular_evaluation":
        metrics_path = manifest.metadata["metrics_path"]
        if metrics_path not in manifest.files:
            raise ValueError("Evaluation manifest does not cover its metrics")
    elif len(candidates) == 1:
        metrics_path = candidates[0]
    else:
        raise ValueError("Value manifest must cover exactly one metrics.json")
    payload = store.read_json(metrics_path)
    identity = payload.get("evaluation_identity", payload.get("identity"))
    run = RunRecord.from_dict(store.read_json(os.path.join(os.path.dirname(metrics_path), "manifest.json")))
    if (run.identity.to_dict() != identity or run.identity.suite != suite
            or run.identity.config_hash != manifest.config_hash or run.status == "running"):
        raise ValueError("Evaluation run/manifest identity or status mismatch")
    config = read_yaml(store.require_file(os.path.join(os.path.dirname(metrics_path), "resolved_config.yaml")))
    if config_digest(config) != run.identity.config_hash:
        raise ValueError("Evaluation resolved configuration hash mismatch")
    if payload["status"] != manifest.metadata["status"]:
        raise ValueError("Evaluation payload and manifest statuses disagree")
    if kind == "molecular_evaluation":
        if payload["version"] != "smiles-metrics-v1" or payload["source_run"] != manifest.metadata["source_run"]:
            raise ValueError("Unsupported or mismatched molecular metrics")
        if payload["status"] in ("available", "partial"):
            if payload["identity"]["suite"] != suite or payload["ideal_value_claim"] is not False:
                raise ValueError("Molecular suite/value provenance mismatch")
            for name in ("comparability", "budget", "variant"):
                hash_key = "comparison_key" if name == "comparability" else name+"_key"
                if config_digest(payload[name]) != payload[hash_key]:
                    raise ValueError("Molecular comparison metadata hash mismatch")
            for source, digest in payload["input_hashes"].items():
                if manifest.files.get(source) != digest:
                    raise ValueError("Molecular metric input is not covered by its manifest")
    elif (payload["version"] != "smiles-value-eval-v1" or payload["identity"]["task_id"] != "T13"
          or payload["purpose"] != manifest.metadata["purpose"] or payload["test_used_for_selection"] is not False
          or payload["evaluation"]["ideal_value_claim"] is not False):
        raise ValueError("Unsupported value metrics or selection/target provenance")
    return payload, metrics_path


def report(context: TaskContext) -> TaskResult:
    options, store = context.config["report"], context.store
    rows, values, audits, sources, hashes = [], [], [], [], {}
    jobs = [(path, "molecular_evaluation") for path in options["evaluation_manifests"]]
    jobs.extend((path, "value_evaluation") for path in options["value_manifests"])
    seen_sources = set()
    for path, kind in context.progress(jobs, desc="Checked evaluations", unit="run"):
        try:
            payload, metrics_path = checked_evaluation(store, path, kind=kind, suite=context.identity.suite)
            hashes[path] = sha256_file(store.require_file(path))
            if kind == "molecular_evaluation":
                source = payload["source_run"]
                if source in seen_sources:
                    raise ValueError("Select only one molecular evaluation per generation run")
                seen_sources.add(source)
                if payload["status"] in ("available", "partial"):
                    rows.append(generation_row(payload, metrics_path))
                    sources.append({"source_run": source, "identity": payload["identity"],
                        "dependencies": payload["source_dependencies"], "comparability": payload["comparability"],
                        "variant": payload["variant"], "metrics_manifest": path})
                    if payload["source_status"] != "completed":
                        audits.append({"path": source, "status": payload["source_status"],
                            "reason": payload["stop_reason"], "attained": payload["metrics"]["attained"]})
                else:
                    audits.append({"path": source, "status": payload["status"], "reason": payload["reason"],
                        "source_status": payload.get("source_status"), "reported_tracking_attained": payload.get("reported_tracking_attained"),
                        "tracking_note": payload.get("reported_tracking_note")})
            else:
                values.extend(value_rows(payload, metrics_path))
                sources.append({"source_run": metrics_path, "identity": payload["identity"],
                                "dependencies": payload["identity"]["dependency_hashes"], "metrics_manifest": path})
                if payload["status"] != "completed":
                    audits.append({"path": metrics_path, "status": payload["status"], "reason": payload["stop_reason"]})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            audits.append({"path": path, "status": "missing" if isinstance(exc, FileNotFoundError) else "invalid",
                           "reason": f"{type(exc).__name__}: {exc}"})
            context.log(f"{path}: {exc}", level="WARNING")
    for path in options["expected_generation_runs"]:
        if path not in seen_sources:
            audits.append({"path": path, "status": "missing", "reason": "No checked molecular evaluation supplied"})
    def observation_key(row):
        return tuple(row[key] for key in ("comparison_key", "group", "method", "variant_key", "budget_key", "seed"))

    observations = Counter(observation_key(row) for row in rows)
    usable = []
    for row in rows:
        if observations[observation_key(row)] == 1:
            usable.append(row)
        else:
            audits.append({"path": row["source_run"], "status": "invalid", "attained": row["attained"],
                           "reason": "Repeated method/variant/budget/seed; retained in summary, excluded from inference"})
    metrics = options["quality_metrics"]+options["cost_metrics"]
    aggregates = summarize_generation(usable, metrics, options["bootstrap"])
    paired = paired_comparisons(usable, metrics, options["bootstrap"])
    frontiers = quality_frontiers(aggregates)
    for aggregate in aggregates:
        observed = {row["seed"] for row in usable if all(row[key] == aggregate[key]
                    for key in ("suite", "comparison_key", "group", "method", "variant_key", "budget_key"))}
        aggregate["expected_seeds"] = len(context.config["seeds"])
        aggregate["missing_expected_seeds"] = ",".join(str(seed) for seed in context.config["seeds"] if seed not in observed)
    for path, digest in hashes.items():
        store.require_file(path, checksum=digest)
    if context.resume_state and context.resume_state != {"input_hashes": hashes}:
        raise ValueError("Report inputs changed since the saved checkpoint")
    context.save_checkpoint({"input_hashes": hashes})
    directory = store.directory("reports", context.identity.suite, context.identity.run_id)
    files = options["files"]
    os.makedirs(store.path(directory), exist_ok=True)
    for key, data in (("summary", rows), ("aggregates", aggregates), ("paired", paired), ("frontiers", frontiers),
                      ("value_accuracy", values), ("missing", audits)):
        write_csv(store, os.path.join(directory, files[key]), data)
    write_plots(store, directory, files, frontiers, values)
    lines = [f"# SMILES report: {context.identity.suite}", "",
        "Molecular quality uses all completed root outputs. Selected diversity is supplementary and conditional on selection.", "",
        "Seed means and paired differences use percentile bootstrap intervals. One seed has no interval; small seed counts limit inference.",
        "Missing metrics are blank. Timeouts and budget stops retain attained samples; unattained costs are not extrapolated.",
        "Comparisons share checkpoint, tokenizer, benchmark, reward, protocol and requested caps. Actual compute can differ.",
        "Hardware model is not recorded in generation logs; timing comparisons require the same execution environment.",
        "Backbones, suites, groups and algorithm variants remain separate. Frontiers contain observed points only.", "",
        "RetroStar is a one-step top-1 probability > 0.4 proxy over all outputs, not proof of a multi-step synthesis route.",
        "Morgan diversity uses radius 2/2048 bits; radius 3 is a sensitivity. Novelty uses canonical identity against all prompts.",
        "Original SMARTS groups overlap and chain-extender membership is broad. QED/Lipinski are excluded from primary metrics.",
        "V_q and reverse-rollout diagnostics are policy-dependent; neither establishes ideal V*. Finite MC error is separate from molecular quality and cost.",
        "Raw NFE includes inner completions and excludes verifier calls. Verifier calls are separate; adjusted NFE needs complete measured FLOPs.",
        "Recorded compute and worker elapsed time exclude unavailable Modal queue/startup/RPC timing. Hard-kill work may be missing.", "",
        "| Group | Seed | Method | Status | Attained / target | Success / roots | NFE | Checkpoint | Config |",
        "|---|---:|---|---|---:|---:|---:|---|---|"]
    for row in rows:
        lines.append(f"| {row['group']} | {row['seed']} | {row['method']} | {row['source_status']} | {row['attained']}/{row['target']} | {row['success_numerator']}/{row['success_denominator']} | {row['raw_nfe']} | {row['checkpoint'][:12]} | {row['config_hash'][:12]} |")
    lines.extend(["", "Files: "+", ".join(f"[{filename}]({filename})" for key, filename in files.items() if key != "markdown"),
                  "", "Input provenance and full settings: [sources.json](sources.json); checksums: [manifest.json](manifest.json).", "",
                  f"Audit entries: {len(audits)}. Missing/invalid inputs and failed/stopped runs are retained in {files['missing']}.", ""])
    store.write_text(os.path.join(directory, files["markdown"]), "\n".join(lines))
    output_paths = [os.path.join(directory, filename) for filename in files.values()]
    provenance_path = os.path.join(directory, "sources.json")
    store.write_json(provenance_path, {"version": REPORT_VERSION, "identity": context.identity.to_dict(),
        "sources": sources, "input_hashes": hashes, "audits": audits, "settings": options})
    output_paths.append(provenance_path)
    manifest_path = os.path.join(directory, "manifest.json")
    store.create_manifest(manifest_path, kind="report", config_hash=context.identity.config_hash,
        files=output_paths, metadata={"version": REPORT_VERSION, "suite": context.identity.suite,
            "sources_path": provenance_path, "input_hashes": hashes, "generation_runs": len(rows), "value_rows": len(values)})
    incomplete = any(row["status"] in ("missing", "invalid") for row in audits)
    return TaskResult(f"generation_runs={len(rows)}; value_rows={len(values)}; audits={len(audits)}; report={directory}/{files['markdown']}",
                      status="failed" if incomplete else "completed", artifacts=tuple([manifest_path, *output_paths]))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evaluation-manifest", action="append", default=[], help="T14 metrics_manifest.json; repeat for each run.")
    parser.add_argument("--value-manifest", action="append", default=[], help="T13 value-evaluation manifest; repeat for each run.")


def argument_overrides(arguments: argparse.Namespace) -> list[str]:
    result = []
    for key in ("evaluation_manifests", "value_manifests"):
        value = getattr(arguments, key[:-1])
        if value:
            result.append("report."+key+"="+json.dumps(value))
    return result


def execute(argv: Sequence[str] | None = None, *, execution: str = "local"):
    return run_task("T16", report, description="Summarize checked molecular and value evaluations.", argv=argv,
        default_config="configs/report.yaml", execution=execution, validator=validate_report,
        add_arguments=add_arguments, argument_overrides=argument_overrides)


if __name__ == "__main__":
    try:
        if execute().status != "completed":
            raise SystemExit(1)
    except KeyboardInterrupt:
        tqdm.write("T16: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"T16: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
