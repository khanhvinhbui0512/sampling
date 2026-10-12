"""A fixed QM9 matrix, immutable worker attempts and explicit unit-level resume.

The coordinator reads metadata on CPU. It never trains, calibrates, makes a
panel, changes a draw count, chooses a best config or retries inside an attempt.
"""

import copy
import os
import re
import uuid

from tqdm.auto import tqdm

from src.config import load_config, project_path
from src.evaluation.aggregate import unusable_attempt
from src.evaluation.metrics import cost_metrics
from src.experiments.run import generate, source_hashes
from src.experiments.value_checks import check_values
from src.io import (RunArtifacts, file_hash, payload_hash, read_completed_manifest,
                    read_json, read_jsonl, report_result, write_json, write_jsonl, write_yaml)
from src.qm9.dataset import load_prepared_qm9
from src.qm9.reward import load_calibration
from src.types import require_count


def python_sources():
    paths = sorted(name for name in os.listdir(".") if name.endswith(".py"))
    paths.append("pyproject.toml")
    for root in ("src", "modal", "tests"):
        for parent, directories, files in os.walk(root):
            directories[:] = sorted(name for name in directories if name != "__pycache__")
            paths.extend(os.path.join(parent, name) for name in sorted(files)
                         if name.endswith((".py", ".json", ".lark")))
    return {path: file_hash(path) for path in paths}


def registered_json(manifest_path, name):
    """Read one hash-checked artifact even after coordinator failure/interruption."""
    manifest = read_json(manifest_path)
    path = project_path(os.path.join(os.path.dirname(manifest_path), name))
    parent = os.path.realpath(os.path.dirname(manifest_path))
    if os.path.commonpath((parent, os.path.realpath(path))) != parent:
        raise ValueError("Registered artifact escapes its attempt")
    record = manifest["artifacts"][name]
    if file_hash(path) != record["sha256"] or os.path.getsize(path) != record["size_bytes"]:
        raise ValueError(f"Registered artifact changed: {name}")
    return read_json(path)


def positive(settings, name, *, zero=False):
    value = settings.get(name)
    require_count(value, name, positive=not zero)
    return value


def require_verification(path, sources):
    verified = read_completed_manifest(path)
    protocol = registered_json(path, "protocol.json")
    summary = registered_json(path, "summary.json")
    if (verified.get("task") != "verify_qm9" or summary["exit_code"] != 0
            or "tests.txt" not in verified["artifacts"]
            or protocol["selection"] != ["tests"] or summary["counts"]["passed"] <= 0
            or any(summary["counts"][name] for name in ("failed", "skipped", "collection_errors"))
            or summary["counts"].get("deselected", 0) > (0 if protocol.get("gpu_smoke") else 1)
            or any(protocol["source_sha256"].get(name) != digest for name, digest in sources.items())):
        raise ValueError("A full passing verification of current code/fixtures is required")


def require_pilot(path, comparison_hash):
    pilot = read_completed_manifest(path)
    plan = registered_json(path, "plan.json")
    summary = registered_json(path, "summary.json")
    if (pilot.get("task") != "benchmark_qm9" or plan["phase"] != "pilot"
            or plan["comparison_hash"] != comparison_hash or summary["pilot_ready"] is not True
            or summary.get("all_units_complete") is not True or not summary.get("selected_manifests")):
        raise ValueError("Final matrix needs a ready pilot with the same scientific contract")
    for manifest_path in summary["selected_manifests"]:
        read_completed_manifest(manifest_path)


def validate_matrix(settings):
    rows = settings["matrix"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Declare a nonempty QM9 matrix")
    ids, signatures = set(), set()
    for row in rows:
        if set(row) != {"id", "group", "method", "verifier", "value_law"}:
            raise ValueError("Matrix rows need id/group/method/verifier/value_law only")
        if not isinstance(row["id"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", row["id"]):
            raise ValueError("Matrix IDs use lowercase letters, digits and hyphens")
        signature = (row["method"], row["verifier"], row["value_law"])
        if row["id"] in ids or signature in signatures:
            raise ValueError("Duplicate matrix ID or method/verifier/law")
        ids.add(row["id"])
        signatures.add(signature)
        if row["method"] not in ("base_native", "base_rho", "bon", "rs", "ars", "cars", "vgr", "vgb", "vgb_momentum"):
            raise ValueError("Unsupported QM9 matrix method")
        guided = row["method"] in ("vgr", "vgb", "vgb_momentum")
        if guided != (row["verifier"] != "none") or row["verifier"] not in ("none", "neural", "mc", "cars"):
            raise ValueError("Only guided rows select neural/mc/cars")
        if row["value_law"] not in ("native-conditional", "upstream-native", "rho"):
            raise ValueError("Unknown value law")
        if row["group"] not in ("standalone", "integration", "diagnostic"):
            raise ValueError("Unknown comparison group")
        if row["value_law"] == "upstream-native" and (not guided or row["verifier"] != "neural"):
            raise ValueError("Upstream snapshot values require the separate neural control")
        if guided and row["value_law"] == "rho" and row["verifier"] == "neural":
            raise ValueError("This matrix supplies native and upstream neural checkpoints only")
        if row["method"] in ("base_rho", "rs", "ars", "cars") and row["value_law"] != "rho":
            raise ValueError("Standalone rejection/CARS and BASE-rho have the rho law")
        if row["method"] in ("base_native", "bon") and row["value_law"] != "native-conditional":
            raise ValueError("This matrix uses native BASE/BoN; label rho diagnostics separately")
        if row["group"] == "integration" and (not guided or row["value_law"] != "native-conditional"):
            raise ValueError("Integration rows must compare native-conditional verifiers")
        if row["value_law"] == "rho" and row["method"] != "cars" and row["group"] != "diagnostic":
            raise ValueError("Keep rho controls in the diagnostic group")
    required = {("cars", "none", "rho"), ("vgb", "neural", "upstream-native"),
                ("vgb", "neural", "native-conditional"), ("vgb", "mc", "native-conditional"),
                ("vgb", "cars", "native-conditional")}
    if not required.issubset(signatures) or settings["value_estimators"] != ["mc", "neural", "cars"]:
        raise ValueError("Retain both deliverables and their matched MC/neural/CARS value check")
    for row in rows:
        if (row["method"], row["verifier"], row["value_law"]) in required:
            group = "integration" if row["value_law"] == "native-conditional" else "standalone"
            if row["group"] != group:
                raise ValueError("Required rows must retain their standalone/integration group")


def lock_protocol(config_path, *, device, execution_metadata):
    """Validate actual inputs and gates without constructing a backbone model."""
    settings = load_config(config_path)
    if settings.get("task") != "qm9" or settings.get("locked") is not True:
        raise ValueError("Review and explicitly lock configs/benchmark.yaml before running")
    phase = settings["phase"]
    if phase not in ("pilot", "final") or settings["cache_mode"] != "cold":
        raise ValueError("Use pilot/final and cold method/seed boundaries")
    seeds = settings["seeds"][phase]
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Set distinct seeds explicitly for the selected phase")
    for seed in seeds:
        require_count(seed, "benchmark seed")
    require_count(settings["outputs_per_seed"][phase], "outputs per seed", positive=True)
    validate_matrix(settings)
    inputs = {name: project_path(path) for name, path in settings["inputs"].items()}
    required_inputs = {"assets_manifest", "checkpoint_manifest", "calibration_manifest",
                       "upstream_verifier_manifest", "native_verifier_manifest", "panel_manifest"}
    if set(inputs) != required_inputs or any(not path.startswith("workspace/") for path in inputs.values()):
        raise ValueError("Declare all six immutable workspace input manifests")
    settings["inputs"] = inputs
    manifests = {name: read_completed_manifest(path) for name, path in inputs.items()}
    assets = load_prepared_qm9(inputs["assets_manifest"])
    backbone = manifests["checkpoint_manifest"]
    calibration = load_calibration(inputs["calibration_manifest"])
    if (backbone.get("task") != "train_qm9_base_checkpoint"
            or backbone["identities"]["assets_manifest_sha256"] != file_hash(inputs["assets_manifest"])
            or backbone["checkpoint_sha256"] != calibration.checkpoint_sha256
            or assets.tokenizer_sha256 != calibration.tokenizer_sha256):
        raise ValueError("Shared assets/backbone/calibration differ")
    configs = {name: load_config(path) for name, path in settings["configs"].items()}
    if set(configs) != {"inference", "cars", "reference", "values", "evaluation"}:
        raise ValueError("Declare inference/cars/reference/values/evaluation configs")
    inference, cars, reference = configs["inference"], configs["cars"], configs["reference"]
    if reference["qm9"]["native_reference"]["gate_passed"] is not True or cars["native"]["enumeration_verified"] is not True:
        raise ValueError("Complete the declared native-reference verification before this matrix")
    if inference["native_settings"] != reference["qm9"]["native"]:
        raise ValueError("One native schedule must be shared by calibration, completion and values")
    if reference["qm9"]["native"]["final_denoise"] is not True:
        raise ValueError("The declared normalized native reference requires final denoise")
    calibration_protocol = registered_json(inputs["calibration_manifest"], "protocol.json")
    if any(calibration_protocol["native"][key] != value for key, value in inference["native_settings"].items()):
        raise ValueError("Calibration was produced under another native schedule")
    n = assets.token_rules.sequence_length
    positive(inference["bon"], "N")
    if inference["walk"]["step_limit"] != n * inference["bon"]["N"]:
        raise ValueError("Keep the declared paper-style n*N walk cap; total work is a separate cap")
    cap = positive(inference["budget"], "base_sequence_limit")
    if cap <= reference["qm9"]["native"]["steps"] + 1:
        raise ValueError("Reserve native completion and leave positive work for guidance")
    if inference["budget"]["verifier_sequence_limit"] is not None:
        positive(inference["budget"], "verifier_sequence_limit")
    for name in ("discovery_attempts", "numerator_draws", "denominator_draws", "mc_control_draws",
                 "min_denominator_hits", "max_nodes", "max_cache_entries"):
        positive(cars["native"], name, zero=name == "discovery_attempts")
    positive(cars["generation"], "max_attempts")
    positive(cars["memory"], "max_nodes")
    # Optional rho guided diagnostics also need their explicit fixed query counts.
    if any(row["value_law"] == "rho" and row["verifier"] == "cars" for row in settings["matrix"]):
        positive(cars["discovery"], "attempts", zero=True)
        positive(cars["estimation"], "draws")
        positive(cars["cache"], "max_entries")
    for name in ("draws", "min_denominator_hits"):
        positive(configs["values"]["reference"], name)
    positive(configs["values"]["budget"], "base_sequence_limit")
    if (cars["native"]["min_denominator_hits"] > min(cars["native"]["denominator_draws"], cars["native"]["mc_control_draws"])
            or configs["values"]["reference"]["min_denominator_hits"] > configs["values"]["reference"]["draws"]):
        raise ValueError("Minimum denominator hits cannot exceed fixed denominator draw counts")
    if configs["values"]["check_output_dir"] != "workspace/results/value_checks":
        raise ValueError("QM9 benchmark value attempts belong under workspace/results/value_checks")
    if phase == "final" and not all(flag is True for flag in (cars["profiled"], cars["ready_for_benchmark"],
                                      cars["native"]["profiled"], cars["native"]["ready_for_benchmark"])):
        raise ValueError("Final runs require the explicit profiled/ready CARS flags")
    # Inspect neural metadata, not the neural weights on a coordinator GPU.
    for name, kind in (("upstream_verifier_manifest", "native_rollout"),
                       ("native_verifier_manifest", "reference_conditional_v_star")):
        metadata = registered_json(inputs[name], "model.json")
        if (manifests[name].get("task") != "train_qm9_verifier_checkpoint"
                or metadata["target"]["kind"] != kind
                or metadata["target"]["calibration_id"] != calibration.calibration_id
                or metadata["backbone_checkpoint_sha256"] != backbone["checkpoint_sha256"]
                or metadata["tokenizer_sha256"] != assets.tokenizer_sha256):
            raise ValueError(f"Wrong neural label target/provenance: {name}")
    panel = manifests["panel_manifest"]
    panel_protocol = registered_json(inputs["panel_manifest"], "protocol.json")
    if (panel.get("task") != "prepare_qm9_value_panel"
            or panel["target"] != registered_json(inputs["native_verifier_manifest"], "model.json")["target"]
            or panel_protocol.get("locked_before_compared_estimators") is not True):
        raise ValueError("Lock one native-conditional panel with the same neural/MC/CARS target")
    sources = python_sources()
    verification_path = settings["gates"]["verification_manifest"]
    require_verification(verification_path, sources)
    input_hashes = {path: file_hash(path) for path in inputs.values()}
    scientific = copy.deepcopy(configs)
    # Readiness flags are bookkeeping; every numerical setting remains pinned.
    for section in (scientific["cars"], scientific["cars"]["native"]):
        section.pop("profiled", None)
        section.pop("ready_for_benchmark", None)
    comparison = {"inputs": input_hashes, "configs": scientific, "matrix": settings["matrix"],
                  "cache_mode": "cold", "value_estimators": settings["value_estimators"],
                  "rollouts": load_config("configs/rollouts.yaml"), "sources": sources,
                  "execution": execution_metadata, "device": device}
    comparison_hash = payload_hash(comparison)
    if phase == "final":
        pilot_path = settings["gates"]["pilot_manifest"]
        require_pilot(pilot_path, comparison_hash)
    dependencies = {**input_hashes, config_path: file_hash(config_path),
                    **{path: file_hash(path) for path in settings["configs"].values()},
                    "configs/rollouts.yaml": file_hash("configs/rollouts.yaml"),
                    verification_path: file_hash(verification_path)}
    if phase == "final":
        dependencies[pilot_path] = file_hash(pilot_path)
    lock = {"settings": settings, "sources": sources, "dependencies": dependencies,
            "comparison_hash": comparison_hash, "device": device, "execution": execution_metadata}
    return lock, configs


def make_units(settings, *, benchmark_id, lock_hash, directory):
    """One generation row/seed plus one shared-panel value-check job/seed."""
    inputs, units = settings["inputs"], []
    shared = {name: inputs[name] for name in ("assets_manifest", "checkpoint_manifest", "calibration_manifest")}
    for seed in settings["seeds"][settings["phase"]]:
        for row in settings["matrix"]:
            unit_id = f"{row['id']}-seed{seed}"
            neural = (inputs["upstream_verifier_manifest"] if row["value_law"] == "upstream-native"
                      else inputs["native_verifier_manifest"]) if row["verifier"] == "neural" else None
            rho = row["value_law"] == "rho"
            arguments = {**shared, "method": row["method"], "verifier": row["verifier"],
                         "value_law": row["value_law"], "verifier_manifest": neural,
                         "num_samples": settings["outputs_per_seed"][settings["phase"]], "seed": seed,
                         "config_path": os.path.join(directory, "inference-rho.yaml" if rho else "inference-native.yaml"),
                         "cars_config": os.path.join(directory, "cars.yaml"),
                         "reference_config": os.path.join(directory, "reference.yaml"), "cache_mode": "cold"}
            units.append({"unit_id": unit_id, "task": "generate", "group": row["group"], "seed": seed,
                          "arguments": arguments, "output_dir": "workspace/results/raw"})
        unit_id = f"native-values-seed{seed}"
        units.append({"unit_id": unit_id, "task": "check_values", "group": "integration", "seed": seed,
                      "arguments": {**shared, "panel_manifest": inputs["panel_manifest"],
                                    "estimators": settings["value_estimators"], "value_law": "native-conditional",
                                    "verifier_manifest": inputs["native_verifier_manifest"],
                                    "config_path": os.path.join(directory, f"values-seed{seed}.yaml"),
                                    "inference_config": os.path.join(directory, "inference-native.yaml"),
                                    "cars_config": os.path.join(directory, "cars.yaml"),
                                    "reference_config": os.path.join(directory, "reference.yaml"), "cache_mode": "cold"},
                      "output_dir": "workspace/results/value_checks"})
    for unit in units:
        unit["arguments"]["run_id"] = f"{benchmark_id}-{lock_hash[:12]}-{unit['unit_id']}"
    return units


def run_unit(job, *, device="cuda", execution_metadata=None):
    """Worker entrypoint shared by local execution and Modal GPU functions."""
    plan = registered_json(job["plan_manifest"], "plan.json")
    unit = next(item for item in plan["units"] if item["unit_id"] == job["unit_id"])
    if plan["lock"]["sources"] != python_sources():
        raise ValueError("Coordinator and worker source/image contents differ")
    for path, digest in {**plan["lock"]["dependencies"], **plan["snapshot_hashes"]}.items():
        if file_hash(path) != digest:
            raise ValueError(f"Locked benchmark input changed: {path}")
    execution = execution_metadata or {"backend": "local"}
    if device != plan["lock"]["device"] or execution != plan["lock"]["execution"]:
        raise ValueError("Worker device/backend differs from the locked benchmark")
    arguments = {**unit["arguments"], "attempt_id": job["attempt_id"], "device": device,
                 "execution_metadata": execution}
    return (generate if unit["task"] == "generate" else check_values)(**arguments)


def inspect_unit(unit, manifest_path, plan):
    """Check a completed unit against its locked job, then read work once."""
    manifest = read_completed_manifest(manifest_path)
    protocol = registered_json(manifest_path, "protocol.json")
    args = unit["arguments"]
    if (manifest["run_id"] != args["run_id"] or protocol["source_sha256"] != source_hashes()
            or payload_hash(protocol) != manifest["config_sha256"]):
        raise ValueError("Unit run/source differs from the locked matrix")
    execution = registered_json(manifest_path, "execution.json")
    if execution["backend"] != plan["lock"]["execution"]["backend"]:
        raise ValueError("Completed unit backend differs from the locked benchmark")
    for name in ("assets_manifest", "checkpoint_manifest", "calibration_manifest"):
        path = args[name]
        if (protocol["inputs"][name] != path or
                protocol["identities"][name + "_sha256"] != plan["lock"]["dependencies"][path]):
            raise ValueError("Unit input differs from the locked matrix")
    inference_path = args["config_path"] if unit["task"] == "generate" else args["inference_config"]
    if (protocol["inference"] != load_config(inference_path)
            or protocol["cars"] != load_config(args["cars_config"])
            or protocol["cache_mode"] != "cold"
            or protocol["mc_settings"] != load_config("configs/rollouts.yaml")["mc"]
            or protocol["target"] != manifest["target"]
            or protocol["verifier_manifest_sha256"] != (file_hash(args["verifier_manifest"]) if args.get("verifier_manifest") else None)):
        raise ValueError("Unit settings/target/cache differ from the locked matrix")
    directory = os.path.dirname(manifest_path)
    if unit["task"] == "generate":
        law = args["value_law"] if args["verifier"] != "none" else (
            "native-base" if args["method"] in ("base_native", "bon") else "rho")
        if (manifest["task"] != "generate_qm9" or protocol["method"] != args["method"]
                or protocol["verifier"] != args["verifier"] or protocol["value_law"] != law
                or protocol["seed"] != unit["seed"] or protocol["requested_outputs"] != args["num_samples"]
                or manifest["requested_outputs"] != args["num_samples"]):
            raise ValueError("Generation row/seed/output count differs")
        rows = list(read_jsonl(os.path.join(directory, "samples.jsonl"), progress=False))
        costs = list(read_jsonl(os.path.join(directory, "costs.jsonl"), progress=False))
        ids = [f"sample-{i:06d}" for i in range(args["num_samples"])]
        if [row["sample_id"] for row in rows] != ids or [row["sample_id"] for row in costs] != ids:
            raise ValueError("A completed unit must retain every requested output in order")
        guided_outputs = 0
        for row, cost in zip(rows, costs):
            if row["target"] != protocol["target"]:
                raise ValueError("Generation output changed target")
            trace = registered_json(manifest_path, row["trace_artifact"])
            if (trace["budget"] != cost["budget"] or trace["wall_seconds"] != cost["wall_seconds"]
                    or row["cost"] != cost["budget"]["total"]
                    or any(trace["result"][key] != row[key] for key in trace["result"])):
                raise ValueError("Generation output, trace and cost ledger disagree")
            queries = registered_json(manifest_path, row["queries_artifact"])
            mask = protocol["identities"]["rules"]["mask_id"]
            guided_outputs += int(any(item["estimate"]["value"] is not None and mask in item["tokens"]
                                      for group in queries for item in group["items"]))
        completed = sum(row["status"] == "completed" for row in rows)
        ready = completed > 0 and (args["verifier"] == "none" or guided_outputs > 0)
        stats = {"requested_outputs": len(rows), "completed_outputs": completed,
                 "outputs_with_available_nonterminal_guidance": guided_outputs,
                 "pilot_ready": ready}
    else:
        if (manifest["task"] != "check_qm9_values" or protocol["estimators"] != args["estimators"]
                or protocol["target"]["kind"] != "reference_conditional_v_star"
                or protocol["settings"] != load_config(args["config_path"])
                or protocol["settings"]["seed"] != unit["seed"]
                or protocol["panel_manifest_sha256"] != file_hash(args["panel_manifest"])):
            raise ValueError("Value-check estimators/seed/panel differ")
        rows = list(read_jsonl(os.path.join(directory, "values.jsonl"), progress=False))
        panel_rows = list(read_jsonl(os.path.join(os.path.dirname(args["panel_manifest"]), "states.jsonl"), progress=False))
        if len(rows) != len(panel_rows) or any(any(row.get(key) != value for key, value in panel.items())
                                             for row, panel in zip(rows, panel_rows)):
            raise ValueError("Value rows must retain the locked panel and its order")
        costs, common = [], 0
        mask = protocol["identities"]["rules"]["mask_id"]
        for row in rows:
            if set(row["estimators"]) != set(args["estimators"]):
                raise ValueError("Value row omitted an estimator")
            estimates = [row["reference"], *(item["estimate"] for item in row["estimators"].values())]
            if any(item["target"] != protocol["target"] for item in estimates):
                raise ValueError("Value row changed target")
            common += int(mask in row["tokens"] and all(item["value"] is not None for item in estimates))
            costs.append({"budget": row["reference_budget"],
                          "wall_seconds": sum(group["wall_seconds"] for group in row["reference_queries"])})
            costs.extend({"budget": item["budget"],
                          "wall_seconds": sum(group["wall_seconds"] for group in item["queries"])}
                         for item in row["estimators"].values())
        stats = {"requested_states": len(rows), "common_nonterminal_pairs": common, "pilot_ready": common > 0}
    return {"unit_id": unit["unit_id"], "manifest": manifest_path, "manifest_sha256": file_hash(manifest_path), "status": "completed",
            "stats": stats, "cost": cost_metrics(costs), "cost_is_lower_bound": False,
            "includes_value_reference_work": unit["task"] == "check_values"}


def dispatch_history(resume_manifest, lock_hash):
    """An explicit resume chain retains intents even if a worker created no manifest."""
    history, seen = [], set()
    while resume_manifest:
        if resume_manifest in seen:
            raise ValueError("Cyclic benchmark resume chain")
        seen.add(resume_manifest)
        manifest = read_json(resume_manifest)
        plan = registered_json(resume_manifest, "plan.json")
        if manifest.get("task") != "benchmark_qm9" or plan["lock_hash"] != lock_hash:
            raise ValueError("Resume chain belongs to another locked matrix")
        for name in sorted(manifest["artifacts"]):
            if name.startswith("dispatch/"):
                history.append(registered_json(resume_manifest, name))
        resume_manifest = manifest.get("resume_manifest")
    return history


def audit_attempts(unit, dispatches, plan):
    paths = {item["manifest"] for item in dispatches if item["unit_id"] == unit["unit_id"]}
    root = project_path(os.path.join(unit["output_dir"], unit["arguments"]["run_id"], "attempts"))
    if os.path.isdir(root):
        paths.update(os.path.join(root, name, "manifest.json") for name in sorted(os.listdir(root))
                     if os.path.isdir(os.path.join(root, name)))
    records = []
    for path in sorted(paths):
        try:
            records.append(inspect_unit(unit, path, plan))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            audit = unusable_attempt(path, exc)
            records.append({"unit_id": unit["unit_id"], **audit, "cost_is_lower_bound": True,
                            "cost": audit.get("partial_cost_lower_bound"),
                            "includes_value_reference_work": unit["task"] == "check_values"})
    return records


def benchmark(*, attempt_id, config_path="configs/benchmark.yaml", benchmark_id="qm9-benchmark",
              resume_manifest=None, device="cuda", worker=None, execution_metadata=None):
    execution = execution_metadata or {"backend": "local"}
    lock, configs = lock_protocol(config_path, device=device, execution_metadata=execution)
    lock_hash = payload_hash(lock)
    previous_plan = registered_json(resume_manifest, "plan.json") if resume_manifest else None
    if previous_plan and (previous_plan["lock_hash"] != lock_hash
                          or read_json(resume_manifest)["run_id"] != benchmark_id):
        raise ValueError("Resume requires the same benchmark ID, phase, inputs, settings and source")
    dispatches = dispatch_history(resume_manifest, lock_hash)
    attempt = RunArtifacts.create("workspace/results/benchmarks", run_id=benchmark_id, attempt_id=attempt_id,
                                  config=lock, target_id="qm9-benchmark:" + lock_hash,
                                  reference_id=lock["comparison_hash"])
    attempt.manifest.update(task="benchmark_qm9", resume_manifest=resume_manifest)
    try:
        if previous_plan:
            plan = previous_plan
        else:
            snapshots = {}
            for name in ("cars", "reference", "evaluation"):
                snapshots[name + ".yaml"] = configs[name]
            for law in ("native", "rho"):
                inference = copy.deepcopy(configs["inference"])
                inference["base_method"] = "base_native" if law == "native" else "base_rho"
                snapshots[f"inference-{law}.yaml"] = inference
            for seed in lock["settings"]["seeds"][lock["settings"]["phase"]]:
                values = copy.deepcopy(configs["values"])
                values["seed"] = seed
                snapshots[f"values-seed{seed}.yaml"] = values
            for name, value in snapshots.items():
                write_yaml(attempt.path(name), value)
                attempt.register(name)
            plan = {"phase": lock["settings"]["phase"], "lock_hash": lock_hash, "lock": lock,
                    "comparison_hash": lock["comparison_hash"],
                    "snapshot_hashes": {attempt.path(name): file_hash(attempt.path(name)) for name in snapshots},
                    "units": make_units(lock["settings"], benchmark_id=benchmark_id,
                                        lock_hash=lock_hash, directory=attempt.directory)}
        for path, digest in plan["snapshot_hashes"].items():
            if file_hash(path) != digest:
                raise ValueError("A benchmark config snapshot changed")
        write_json(attempt.path("plan.json"), plan)
        attempt.register("plan.json")
        # New runs never silently reuse data. Resume always names a prior attempt.
        if not resume_manifest and any(audit_attempts(unit, [], plan) for unit in plan["units"]):
            raise FileExistsError("Unit attempts already exist; provide an explicit resume manifest")
        selected, unit_results, ledger = [], [], []
        for unit in tqdm(plan["units"], desc="QM9 benchmark", unit="unit"):
            records = audit_attempts(unit, dispatches, plan)
            completed = [row for row in records if row["status"] == "completed"]
            error = None
            if len(completed) > 1:
                error = "Multiple completed attempts for one row/seed; no automatic best/latest selection"
            elif not completed:
                unit_attempt = uuid.uuid4().hex
                manifest_path = os.path.join(unit["output_dir"], unit["arguments"]["run_id"], "attempts", unit_attempt, "manifest.json")
                dispatch = {"unit_id": unit["unit_id"], "attempt_id": unit_attempt, "manifest": manifest_path}
                name = f"dispatch/{unit['unit_id']}-{unit_attempt}.json"
                write_json(attempt.path(name), dispatch)
                attempt.register(name)
                dispatches.append(dispatch)
                job = {"plan_manifest": attempt.path("manifest.json"), "unit_id": unit["unit_id"], "attempt_id": unit_attempt}
                try:
                    result = (worker(job) if worker else run_unit(job, device=device, execution_metadata=execution))
                    if result["manifest"] != manifest_path:
                        raise ValueError("Worker returned another unit/attempt manifest")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                write_json(attempt.path(f"returns/{unit['unit_id']}-{unit_attempt}.json"), {"manifest": manifest_path, "error": error})
                attempt.register(f"returns/{unit['unit_id']}-{unit_attempt}.json")
                records = audit_attempts(unit, dispatches, plan)
                completed = [row for row in records if row["status"] == "completed"]
            ledger.extend(records)
            if len(completed) == 1:
                selected.append(completed[0]["manifest"])
                unit_results.append({"unit_id": unit["unit_id"], "group": unit["group"], "seed": unit["seed"],
                                     "selected_manifest": completed[0]["manifest"], "stats": completed[0]["stats"],
                                     "attempt_count": len(records), "error": error})
            else:
                unit_results.append({"unit_id": unit["unit_id"], "group": unit["group"], "seed": unit["seed"],
                                     "selected_manifest": None, "attempt_count": len(records),
                                     "error": error or "No verified completed attempt; resume with a new coordinator attempt"})
        complete = len(selected) == len(plan["units"])
        pilot_ready = complete and all(row["stats"]["pilot_ready"] for row in unit_results)
        summary = {"phase": plan["phase"], "all_units_complete": complete, "pilot_ready": pilot_ready,
                   "units": unit_results, "selected_manifests": selected,
                   "worker_attempts": len(ledger), "retry_or_unusable_attempts": sum(row["status"] != "completed" for row in ledger),
                   "cost_note": "attempts.jsonl retains each attempt exactly once; incomplete work is a lower bound. Value-job costs include independent MC reference work. Selected output metrics alone exclude failed retry work.",
                   "pilot_note": "Readiness means at least one completed output per generation row, available nonterminal guidance in each guided row, and a common nonterminal value pair per seed. It is not a value-accuracy or paper-reproduction claim."}
        write_jsonl(attempt.path("attempts.jsonl"), ledger, progress=False)
        write_json(attempt.path("selected_manifests.json"), selected)
        write_json(attempt.path("summary.json"), summary)
        for name in ("attempts.jsonl", "selected_manifests.json", "summary.json"):
            attempt.register(name)
        if complete:
            attempt.complete(required_artifacts=("plan.json", "attempts.jsonl", "selected_manifests.json", "summary.json"))
        else:
            attempt.fail("Matrix has unfinished/ambiguous units; inspect summary and resume explicitly")
        exit_code = int(not complete or (plan["phase"] == "pilot" and not pilot_ready))
        report_result(f"Benchmark {plan['phase']}: {len(selected)}/{len(plan['units'])} units; pilot_ready={pilot_ready}; manifest={attempt.path('manifest.json')}")
        return {"manifest": attempt.path("manifest.json"), "completed_units": len(selected),
                "total_units": len(plan["units"]), "pilot_ready": pilot_ready, "exit_code": exit_code}
    except Exception as exc:
        if attempt.manifest["status"] == "running":
            attempt.fail(f"{type(exc).__name__}: {exc}")
        raise
