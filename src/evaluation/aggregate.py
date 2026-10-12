"""Evaluate explicit immutable attempts on CPU, without loading a backbone."""

import copy
import csv
import glob
import os
import platform
import statistics
from dataclasses import asdict
from importlib.metadata import version

from tqdm.auto import tqdm

from src.config import load_config, project_path
from src.io import (RunArtifacts, _atomic_writer, file_hash, payload_hash,
                    read_completed_manifest, read_json, read_jsonl, report_result,
                    write_json, write_jsonl)
from src.evaluation.metrics import (QUALITY_METRICS, bootstrap_interval, cache_metrics,
                                    cost_metrics, finite_number, generation_metrics, mean,
                                    rollout_error_interval, value_metrics)
from src.qm9.dataset import load_prepared_qm9
from src.qm9.reward import QM9Reward, load_calibration
from src.types import CostRecord, ValueEstimate, ValueTarget, require_count


def artifact(manifest_path, manifest, name):
    if name not in manifest["artifacts"]:
        raise ValueError(f"Missing registered artifact: {name}")
    return project_path(os.path.join(os.path.dirname(manifest_path), name))


def load_inputs(protocol):
    identities, inputs = protocol["identities"], protocol["inputs"]
    for name in ("assets_manifest", "checkpoint_manifest", "calibration_manifest"):
        if file_hash(inputs[name]) != identities[name + "_sha256"]:
            raise ValueError(f"Input provenance changed: {name}")
    assets = load_prepared_qm9(inputs["assets_manifest"])
    calibration = load_calibration(inputs["calibration_manifest"])
    if (asdict(calibration) != identities["calibration"] or asdict(assets.token_rules) != identities["rules"]
            or assets.tokenizer_sha256 != identities["tokenizer_id"]
            or calibration.checkpoint_sha256 != identities["checkpoint_id"]):
        raise ValueError("Saved QM9 identities differ from verified assets/calibration")
    reward = QM9Reward.from_assets(
        inputs["assets_manifest"], calibration_manifest=inputs["calibration_manifest"],
        calibration_reference_id=calibration.reference_id, calibration_policy_id=calibration.policy_id,
        checkpoint_sha256=identities["checkpoint_id"])
    train = {row["canonical_smiles"] for row in read_jsonl(assets.train_canonical_path, progress=False)}
    return reward, train


def grouping(protocol, execution, *, kind, estimator=None, legacy_manifest=None, cohort=None):
    """A family permits side-by-side plots; only an identical config pools seeds."""
    family = {key: protocol.get(key) for key in ("identities", "source_sha256")}
    family.update(kind=kind, precision=execution.get("precision"), device=execution.get("device"),
                  backend=execution.get("backend"), python=execution.get("python"), torch=execution.get("torch"),
                  hardware=execution.get("hardware"))
    if kind != "generation":
        family.update(target=protocol["target"], panel_id=protocol.get("panel_id"),
                      comparison_estimators=cohort,
                      panel_manifest_sha256=protocol.get("panel_manifest_sha256"),
                      fixture_sha256=protocol.get("fixture_sha256"))
    descriptor = copy.deepcopy(protocol)
    descriptor.pop("seed", None)
    descriptor.pop("inputs", None)
    if "settings" in descriptor:
        descriptor["settings"].pop("seed", None)
    if estimator:
        descriptor["estimators"] = [estimator]
        if estimator != "neural":
            descriptor["verifier_manifest_sha256"] = None
        if estimator != "cars":
            descriptor["cache_mode"] = "cold"
            descriptor["warm_source_sha256"] = None
    # Older MC protocols omitted YAML defaults; never pool unverifiable settings.
    if (protocol.get("verifier") == "mc" or estimator == "mc") and "mc_settings" not in protocol and kind != "values_toy":
        descriptor["legacy_unpooled_manifest"] = legacy_manifest
    descriptor["execution_context"] = family
    return payload_hash(family), payload_hash(descriptor), descriptor


def validate_estimates(rows, target, estimators, *, toy):
    for row in rows:
        if set(row["estimators"]) != set(estimators):
            raise ValueError("Value row estimator list differs from protocol")
        entries = [entry["estimate"] for entry in row["estimators"].values()]
        if not toy:
            entries.append(row["reference"])
        for raw in entries:
            if raw["target"] != target:
                raise ValueError("Value estimate has a different target")
            ValueEstimate(**{**raw, "target": ValueTarget(**raw["target"]), "cost": CostRecord(**raw["cost"])})


def validate_queries(groups, target):
    for group in groups:
        CostRecord(**group["cost"])
        finite_number(group["wall_seconds"], "query wall seconds")
        for item in group["items"]:
            raw = item["estimate"]
            if raw["target"] != target:
                raise ValueError("Query estimate target differs from protocol")
            ValueEstimate(**{**raw, "target": ValueTarget(**raw["target"]), "cost": CostRecord(**raw["cost"])})


def read_run(path, settings, contexts):
    manifest = read_completed_manifest(path)
    protocol = read_json(artifact(path, manifest, "protocol.json"))
    execution = read_json(artifact(path, manifest, "execution.json"))
    if not isinstance(protocol, dict) or not isinstance(execution, dict):
        raise ValueError("Protocol and execution must be JSON objects")
    if payload_hash(protocol) != manifest["config_sha256"]:
        raise ValueError("Protocol does not match manifest config hash")
    target = ValueTarget(**protocol["target"])
    if (target.target_id != manifest["target_id"] or target.reference_id != manifest["reference_id"]
            or manifest.get("target") != protocol["target"]):
        raise ValueError("Manifest and protocol targets differ")
    task = manifest["task"]
    toy = task == "check_toy_values"
    if task not in ("generate_qm9", "check_qm9_values", "check_toy_values"):
        raise ValueError(f"Unsupported evaluator input task: {task}")
    if toy and (protocol.get("mode") != "toy" or protocol.get("reference") != "exact_finite_enumeration"):
        raise ValueError("Toy inputs require the explicit finite-enumeration protocol")
    if task == "check_qm9_values" and protocol.get("reference") != "independent_fixed_count_mc_not_exact":
        raise ValueError("QM9 value checks require their explicit uncertain MC reference")
    reward, train = None, None
    if not toy:
        context_key = payload_hash(protocol["identities"])
        # Recheck file identity even when tokenizer/novelty data are already cached.
        for name, source in protocol["inputs"].items():
            if file_hash(source) != protocol["identities"][name + "_sha256"]:
                raise ValueError(f"Changed input: {name}")
        if context_key not in contexts:
            contexts[context_key] = load_inputs(protocol)
        reward, train = contexts[context_key]
        if target.calibration_id != reward.calibration.calibration_id:
            raise ValueError("Target uses another Pass@95 calibration")
    common = {"manifest": path, "manifest_sha256": file_hash(path), "run_id": manifest["run_id"],
              "attempt_id": manifest["attempt_id"], "target": protocol["target"],
              "protocol": protocol, "execution": execution}
    if task == "generate_qm9":
        requested = protocol["requested_outputs"]
        if requested != manifest["requested_outputs"]:
            raise ValueError("Requested output counts differ")
        samples = list(read_jsonl(artifact(path, manifest, "samples.jsonl"), progress=False))
        costs = list(read_jsonl(artifact(path, manifest, "costs.jsonl"), progress=False))
        by_id = {row["sample_id"]: row for row in costs}
        if len(costs) != requested or len(by_id) != requested or set(by_id) != {row["sample_id"] for row in samples}:
            raise ValueError("Cost ledger must cover every output exactly once")
        queries, fallbacks = [], 0
        for sample in samples:
            trace = read_json(artifact(path, manifest, sample["trace_artifact"]))
            cost = by_id[sample["sample_id"]]
            if (trace["budget"] != cost["budget"] or trace["wall_seconds"] != cost["wall_seconds"]
                    or sample["cost"] != cost["budget"]["total"]
                    or any(trace["result"][key] != sample[key] for key in trace["result"])):
                raise ValueError("Sample, trace and cost ledger disagree")
            queries.extend(read_json(artifact(path, manifest, sample["queries_artifact"])))
            fallbacks += int(bool(trace["metadata"].get("fallback_used", False)))
        metrics = generation_metrics(samples, requested=requested, reward=reward, target=protocol["target"],
                                     train_smiles=train, diversity_settings=settings["diversity"])
        cost, cache = cost_metrics(costs), cache_metrics(queries)
        validate_queries(queries, protocol["target"])
        metrics.update({"base_calls_per_output": cost["total"]["base_calls"] / requested,
                        "base_sequences_per_output": cost["total"]["base_sequences"] / requested,
                        "verifier_sequences_per_output": cost["total"]["verifier_sequences"] / requested,
                        "wall_seconds_per_output": cost["wall_seconds"] / requested,
                        "adjusted_nfe_per_output": cost["adjusted_nfe"] / requested if cost["adjusted_nfe"] is not None else None,
                        "base_sequences_per_success": cost["total"]["base_sequences"] / metrics["passing_outputs"] if metrics["passing_outputs"] else None,
                        "fallback_outputs": fallbacks})
        summary = read_json(artifact(path, manifest, "summary.json"))
        if summary["successes"] != metrics["passing_outputs"] or summary["requested_outputs"] != requested:
            raise ValueError("Saved summary disagrees with recomputed generation metrics")
        family, config_id, descriptor = grouping(protocol, execution, kind="generation", legacy_manifest=path)
        return [{**common, "kind": "generation", "family_id": family, "config_id": config_id,
                 "config": descriptor, "label": protocol["method"] + "/" + protocol["verifier"] + "/" + protocol["value_law"] + "/" + protocol["cache_mode"],
                 "seed": protocol["seed"], "metrics": metrics, "cost": cost, "cache": cache,
                 "setup_seconds": finite_number(execution["setup_seconds"], "setup seconds")}]
    rows = list(read_jsonl(artifact(path, manifest, "values.jsonl"), progress=False))
    if not rows or len(rows) != execution["states"]:
        raise ValueError("Value file does not contain all requested states")
    names = sorted(rows[0]["estimators"]) if toy else protocol["estimators"]
    validate_estimates(rows, protocol["target"], names, toy=toy)
    if not toy:
        ids = [row["state_id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Repeated panel state IDs")
        panel_path = protocol["panel_manifest"]
        panel = read_completed_manifest(panel_path)
        if file_hash(panel_path) != protocol["panel_manifest_sha256"] or panel.get("panel_id") != protocol["panel_id"]:
            raise ValueError("Locked panel identity differs from value-check protocol")
        panel_protocol = read_json(artifact(panel_path, panel, "protocol.json"))
        panel_rows = list(read_jsonl(artifact(panel_path, panel, "states.jsonl"), progress=False))
        if (panel_protocol["identities"] != protocol["identities"] or panel_protocol["target"] != protocol["target"]
                or panel.get("task") != "prepare_qm9_value_panel"
                or panel["panel_id"] != "qm9-value-panel:" + payload_hash(panel_protocol)
                or panel_protocol.get("locked_before_compared_estimators") is not True
                or payload_hash(panel_rows) != panel_protocol["states_sha256"]
                or [{key: row[key] for key in ("state_id", "stratum", "tokens", "origin")} for row in rows] != panel_rows):
            raise ValueError("Value rows differ from the original locked panel/order")
        for row in rows:
            validate_queries(row["reference_queries"], protocol["target"])
        reference_cost = cost_metrics([{"budget": row["reference_budget"], "wall_seconds": sum(group["wall_seconds"] for group in row["reference_queries"])} for row in rows])
    else:
        reference_cost = {"enumeration_budget": execution["enumeration_budget"]}
    records = []
    for name in names:
        for row in rows:
            validate_queries(row["estimators"][name]["queries"], protocol["target"])
        result = value_metrics(rows, name, toy=toy)
        result["rollout_cluster_interval"] = rollout_error_interval(result["states"], settings["uncertainty"])
        common_ids = set(result["common_pair_state_ids"])
        common_details = [{**item, "difference": item["difference"] if item["state_id"] in common_ids else None}
                          for item in result["states"]]
        result["common_pair_cluster_interval"] = rollout_error_interval(common_details, settings["uncertainty"])
        result["strata"] = {}
        for stratum in sorted({row.get("stratum", "toy") for row in rows}):
            subset = value_metrics([row for row in rows if row.get("stratum", "toy") == stratum], name, toy=toy)
            result["strata"][stratum] = {key: subset[key] for key in ("requested_states", "paired_states", "availability", "mae", "rmse", "bias", "common_pair_states", "common_pair_mae", "common_pair_rmse")}
        count, cost = result["requested_states"], result["cost"]
        result.update(base_calls_per_query=cost["total"]["base_calls"] / count,
                      base_sequences_per_query=cost["total"]["base_sequences"] / count,
                      verifier_sequences_per_query=cost["total"]["verifier_sequences"] / count,
                      adjusted_nfe_per_query=cost["adjusted_nfe"] / count if cost["adjusted_nfe"] is not None else None,
                      wall_seconds_per_query=cost["wall_seconds"] / count)
        kind = "values_toy" if toy else "values_qm9"
        family, config_id, descriptor = grouping(protocol, execution, kind=kind, estimator=name, legacy_manifest=path, cohort=sorted(names))
        records.append({**common, "kind": kind, "family_id": family, "config_id": config_id,
                        "config": descriptor, "label": name + "/" + target.kind.value + "/" + protocol.get("cache_mode", "cold"),
                        "seed": protocol["settings"]["seed"], "estimator": name, "metrics": result,
                        "cost": cost, "cache": result["cache"], "reference_cost": reference_cost,
                        "setup_seconds": execution.get("setup_seconds")})
    return records


def aggregate_records(records, uncertainty):
    grouped = {}
    for record in records:
        require_count(record["seed"], "run seed")
        grouped.setdefault(record["config_id"], []).append(record)
    results = []
    for config_id, items in sorted(grouped.items()):
        seeds = [item["seed"] for item in items]
        duplicate = len(set(seeds)) != len(seeds)
        if items[0]["kind"] == "generation":
            keys = QUALITY_METRICS + ("base_calls_per_output", "base_sequences_per_output",
                    "verifier_sequences_per_output", "wall_seconds_per_output",
                    "adjusted_nfe_per_output", "base_sequences_per_success")
        else:
            keys = ("availability", "mae", "rmse", "bias", "full_panel_mae", "common_pair_states",
                    "common_pair_mae", "common_pair_rmse", "base_calls_per_query", "base_sequences_per_query",
                    "verifier_sequences_per_query", "adjusted_nfe_per_query", "wall_seconds_per_query")
        summaries = {}
        for key in keys:
            values = [item["metrics"][key] for item in items]
            defined = [value for value in values if value is not None]
            # Do not hide undefined rates (e.g. no valid molecules) or duplicate attempts.
            summaries[key] = {"mean": mean(defined) if len(defined) == len(items) and not duplicate else None,
                              "defined_seeds": len(defined),
                              "std": statistics.stdev(defined) if len(defined) == len(items) and len(defined) > 1 and not duplicate else None,
                              "seed_interval": bootstrap_interval(values, uncertainty) if not duplicate else None}
        results.append({"config_id": config_id, "family_id": items[0]["family_id"], "kind": items[0]["kind"],
                        "label": items[0]["label"], "seeds": seeds, "runs": len(items),
                        "manifests": [item["manifest"] for item in items],
                        "duplicate_seed_attempts": duplicate, "metrics": summaries,
                        "uncertainty": "equal-seed mean; percentile bootstrap over independent seeds only; no CI with one seed"})
    return results


def select_manifests(manifests, run_ids):
    paths, missing = [project_path(path) for path in manifests], []
    for run_id in run_ids:
        if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in run_id):
            raise ValueError("Use simple run IDs (letters, digits, underscore, hyphen)")
        found = []
        for base in ("workspace/results/raw", "workspace/results/value_checks"):
            directories = sorted(glob.glob(os.path.join(base, run_id, "attempts", "*")))
            found.extend(os.path.join(directory, "manifest.json") for directory in directories if os.path.isdir(directory))
        if found:
            paths.extend(found)
        else:
            missing.append({"run_id": run_id, "status": "missing_run", "error": "No attempt manifests found"})
    if len(set(paths)) != len(paths):
        raise ValueError("Select each manifest once; do not overlap --manifests and --run-ids")
    if not paths and not missing:
        raise ValueError("Select manifests or run IDs explicitly; evaluator never selects latest/best attempts")
    return paths, missing


def unusable_attempt(path, error):
    """Audit verified partial costs as a lower bound, without imputing outcomes."""
    entry = {"manifest": path, "status": "unusable", "error": f"{type(error).__name__}: {error}"}
    try:
        raw = read_json(path)
        if not isinstance(raw, dict):
            raise ValueError("Manifest must be a JSON object")
        entry.update(manifest_sha256=file_hash(path), source_status=raw.get("status"),
                     requested_outputs=raw.get("requested_outputs"), source_error=raw.get("error"))
        rows, failures = [], []
        for name, saved in raw.get("artifacts", {}).items():
            if not name.startswith("traces/") or not name.endswith(".json"):
                continue
            filename = project_path(os.path.join(os.path.dirname(path), name))
            if os.path.commonpath((os.path.realpath(os.path.dirname(path)), os.path.realpath(filename))) != os.path.realpath(os.path.dirname(path)):
                raise ValueError("Partial trace escapes attempt")
            if file_hash(filename) != saved["sha256"] or os.path.getsize(filename) != saved["size_bytes"]:
                failures.append(name)
                continue
            trace = read_json(filename)
            rows.append({"budget": trace["budget"], "wall_seconds": trace["wall_seconds"]})
        entry.update(verified_trace_outputs=len(rows), unverified_trace_artifacts=failures,
                     partial_cost_lower_bound=cost_metrics(rows),
                     partial_cost_note="Registered generation traces only; unfinished/unregistered work and value-check costs may be unknown")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        entry["partial_audit_error"] = f"{type(exc).__name__}: {exc}"
    return entry


def report_text(groups, audit, figures, complete):
    lines = ["# QM9 artifact evaluation", "", f"Selection complete and unambiguous: **{complete}**.", "",
             "Failed/running/missing/corrupt attempts remain in audit.jsonl. Their unknown results and costs are not imputed. Group statistics describe usable completed attempts only; incomplete selection is not a benchmark comparison.", "",
             "Pass@95, validity, unique Pass@95 and zero-filled QED use all requested output slots, including failed/unfinished/invalid outputs. Mean valid QED, uniqueness and novelty use completed valid outputs; empty populations are null. Novelty counts unique canonical molecules absent from prepared TRAIN, divided by valid outputs (MDM-VGB released-code convention).", "",
             "Morgan diversity follows CARS paper Appendix H.4 (radius 2, 2048 bits by default); released CARS code uses radius 3. Seeded valid-output subsampling is a project addition and retains duplicates. Fingerprint settings and indices are saved per run.", "",
             "Families share model/tokenizer/calibration, source and execution context. Methods/value laws have separate config IDs. Generation may compare native and rho adaptations side by side; value checks require the same target and locked panel. Toy results are separate from QM9. Equal-seed means have seed-bootstrap intervals only with at least two distinct seeds. Duplicate attempts for a seed suppress group statistics until explicitly resolved.", "",
             "Value errors condition on available estimate AND reference; availability and every refused state are retained. Individual available-pair errors and common-pair errors are separate. Value plots use states available for ALL estimators in the selected cohort, with common counts shown; full-panel MAE is null if any pair is missing. Neither conditional error ranks methods on refused states. QM9 uses noisy independent MC, not exact V*. Per-state reference intervals and N/D/ESS query diagnostics remain in runs.jsonl. Rollout-cluster intervals keep masks/snapshots together and do not eliminate MC reference uncertainty; no IID state CI is claimed.", "",
             "Raw NFE is actual base calls; base/verifier sequences and phase work are separate. Nested MC/CARS work is already in ledgers. Measured FLOP provenance is required for neural adjusted NFE. Setup, sampling/query wall and historical cache origin work are separate; historical origins are not an extra charge. Value-check setup and reference evaluation are shared per input manifest, not charged once per estimator record. No FLOPs are inferred from parameter counts.", "",
             "This evaluator is a project implementation. It does not establish paper-table reproduction, native correctness/feasibility, or that verification/benchmark gates have passed.", "",
             "| Config | Family | Method / target / cache | Seeds | Pass@95 or common-pair MAE |", "| --- | --- | --- | --- | --- |"]
    for group in groups:
        metric = "pass95" if group["kind"] == "generation" else "common_pair_mae"
        value = group["metrics"][metric]
        lines.append(f"| {group['config_id'][:12]} | {group['family_id'][:12]} | {group['label']} | {group['seeds']} | {value['mean']} (CI {value['seed_interval']}) |")
    lines.extend(["", f"Audit entries: {len(audit)}. See metrics.json/metrics.csv and runs.jsonl for denominators, uncertainty, costs and provenance.", "", "Figures:"])
    lines.extend([f"- {item['path']}" for item in figures["written"]] or ["- No applicable figure."])
    lines.extend([f"- Skipped {item['plot']}: {item['reason']}" for item in figures["skipped"]])
    return "\n".join(lines) + "\n"


def evaluate(*, attempt_id, manifests=(), run_ids=(), comparison_id="qm9-evaluation",
             config_path="configs/evaluation.yaml", execution_metadata=None):
    settings = load_config(config_path)
    confidence = settings["uncertainty"]["confidence"]
    if isinstance(confidence, bool) or not 0 < confidence < 1:
        raise ValueError("Confidence must lie in (0, 1)")
    require_count(settings["uncertainty"]["bootstrap_replicates"], "bootstrap replicates", positive=True)
    require_count(settings["uncertainty"]["seed"], "bootstrap seed")
    paths, audit = select_manifests(manifests, run_ids)
    output_dir = project_path(settings["output_dir"])
    if not output_dir.startswith("workspace/results/summary/") and output_dir != "workspace/results/summary":
        raise ValueError("Evaluation reports belong under workspace/results/summary/")
    source_paths = ["evaluate.py", "src/io.py", "src/config.py", "src/types.py",
                    "src/qm9/chemistry.py", "src/qm9/reward.py", "src/qm9/dataset.py"]
    source_paths.extend(os.path.join("src/evaluation", name) for name in sorted(os.listdir("src/evaluation")) if name.endswith(".py"))
    source = {path: file_hash(path) for path in source_paths}
    execution = {"backend": "local", **(execution_metadata or {}), "python": platform.python_version(),
                 "packages": {name: version(name) for name in ("numpy", "rdkit", "matplotlib", "transformers", "PyYAML")}}
    protocol = {"settings": settings, "manifests": paths, "run_ids": list(run_ids),
                "evaluator_source_sha256": source, "execution": execution}
    attempt = RunArtifacts.create(settings["output_dir"], run_id=comparison_id, attempt_id=attempt_id,
                                  config=protocol, target_id="qm9-artifact-evaluation-v1", reference_id="selected-immutable-attempts")
    attempt.manifest.update(task="evaluate_qm9_artifacts")
    try:
        write_json(attempt.path("protocol.json"), protocol)
        attempt.register("protocol.json")
        records, contexts = [], {}
        for path in tqdm(paths, desc="Evaluate saved attempts", unit="attempt"):
            try:
                records.extend(read_run(path, settings, contexts))
                audit.append({"manifest": path, "manifest_sha256": file_hash(path), "status": "usable_completed"})
            except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
                audit.append(unusable_attempt(path, exc))
        groups = aggregate_records(records, settings["uncertainty"])
        for group in groups:
            if group["duplicate_seed_attempts"]:
                audit.append({"status": "duplicate_seed_attempts", "config_id": group["config_id"],
                              "seeds": group["seeds"], "manifests": group["manifests"],
                              "error": "Select one explicit attempt per config/seed; retries are not independent seeds"})
        complete = all(item["status"] == "usable_completed" for item in audit) and not any(group["duplicate_seed_attempts"] for group in groups)
        from src.evaluation.plots import create_plots
        figures = create_plots(attempt, groups, records)
        report = {"selection_complete": complete, "groups": groups, "audit": audit,
                  "figures": figures, "uncertainty_settings": settings["uncertainty"]}
        write_json(attempt.path("metrics.json"), report)
        write_jsonl(attempt.path("runs.jsonl"), records, progress=False)
        write_jsonl(attempt.path("audit.jsonl"), audit, progress=False)
        with _atomic_writer(attempt.path("metrics.csv")) as handle:
            writer = csv.writer(handle)
            writer.writerow(["family_id", "config_id", "kind", "label", "seeds", "metric", "mean", "std", "ci_low", "ci_high", "defined_seeds", "duplicate_seed_attempts"])
            for group in groups:
                for key, value in group["metrics"].items():
                    interval = value["seed_interval"] or [None, None]
                    writer.writerow([group["family_id"], group["config_id"], group["kind"], group["label"], str(group["seeds"]), key, value["mean"], value["std"], *interval, value["defined_seeds"], group["duplicate_seed_attempts"]])
        with _atomic_writer(attempt.path("report.md")) as handle:
            handle.write(report_text(groups, audit, figures, complete))
        names = ("metrics.json", "metrics.csv", "runs.jsonl", "audit.jsonl", "report.md")
        for name in names:
            attempt.register(name)
        attempt.complete(required_artifacts=("protocol.json", *names))
    except Exception as exc:
        attempt.fail(f"{type(exc).__name__}: {exc}")
        raise
    report_result(f"Evaluated {len(records)} method records from {len(paths)} attempts; complete={complete}; manifest={attempt.path('manifest.json')}")
    return {"manifest": attempt.path("manifest.json"), "records": len(records), "selection_complete": complete,
            "artifacts": {"manifest.json": {"sha256": file_hash(attempt.path("manifest.json"))}, **attempt.manifest["artifacts"]}}
