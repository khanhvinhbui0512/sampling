"""Freeze a reward-independent state panel, then compare values under ONE law."""

import os
import platform
import time
from dataclasses import asdict

import torch
from tqdm.auto import tqdm

from src.budget import BudgetExceeded, BudgetTracker
from src.config import load_config, project_path
from src.experiments.run import (RecordedVerifier, configured_count, expected_target, load_context,
                                 make_verifier, record, save_cache, source_hashes, synchronize, warm_cache)
from src.io import (RunArtifacts, file_hash, payload_hash, read_completed_manifest,
                    read_json, read_jsonl, report_result, write_json, write_jsonl)
from src.randomness import RNGPool
from src.types import CostPhase, MaskedState
from src.verifiers.monte_carlo import MonteCarloVerifier, NativeConditionalMonteCarloVerifier


def masked_variants(terminal):
    """Fixed masks chosen without reward; include future observations and holes."""
    rules, length = terminal.rules, len(terminal.token_ids)
    yield "root", MaskedState((rules.mask_id,) * length, rules)
    counts = sorted({max(1, length // 4), max(1, length // 2), max(1, 3 * length // 4)})
    spread = list(range(1, length, 2)) + list(range(0, length, 2))
    for count in counts:
        if count >= length:
            continue
        for stratum, positions in (("prefix", range(count)), ("hole", spread[:count])):
            known = set(positions)
            yield stratum, MaskedState(tuple(token if i in known else rules.mask_id
                                            for i, token in enumerate(terminal.token_ids)), rules)
    yield "terminal", terminal


def prepare_value_panel(assets_manifest, checkpoint_manifest, calibration_manifest, *, attempt_id,
                        source_manifests=(), value_law="native-conditional", run_id="qm9-value-panel",
                        config_path="configs/value_checks.yaml", inference_config="configs/mdm_vgb.yaml",
                        cars_config="configs/cars.yaml", reference_config="configs/reference_checks.yaml",
                        device="cuda", execution_metadata=None):
    settings = load_config(config_path)
    cap = configured_count(settings["panel"], "max_per_stratum")
    strata = settings["panel"]["strata"]
    allowed = {"root", "prefix", "hole", "native_snapshot", "rho_snapshot", "walk_snapshot", "terminal"}
    if len(set(strata)) != len(strata) or not set(strata).issubset(allowed) or not {"root", "prefix", "hole"}.issubset(strata):
        raise ValueError("A panel requires unique known strata including root, prefix and hole")
    if value_law == "upstream-native":
        raise ValueError("Prepare native-conditional or rho panels; upstream snapshots remain a separate generation diagnostic")
    context = load_context(assets_manifest, checkpoint_manifest, calibration_manifest, device=device,
                           config_path=inference_config, cars_config=cars_config, reference_config=reference_config)
    target = expected_target(context, value_law)
    buckets, seen, sources = {name: [] for name in strata}, set(), []
    budget = BudgetTracker(**settings["budget"])
    started = time.perf_counter()

    def add(stratum, state, origin):
        key = (stratum, state.token_ids)
        if stratum in buckets and len(buckets[stratum]) < cap and key not in seen:
            seen.add(key)
            buckets[stratum].append({"stratum": stratum, "tokens": list(state.token_ids), "origin": origin})

    def terminal_rows(terminal, origin):
        for stratum, state in masked_variants(terminal):
            add(stratum, state, {**origin, "masking": "fixed_posthoc_pattern_not_diffusion_snapshot"})

    # Source order and per-stratum first-occurrence selection are fixed before
    # any compared estimator is called. No validity/reward filter is applied.
    for source_path in tqdm(source_manifests, desc="Panel source runs", unit="run"):
        source_path = project_path(source_path)
        manifest = read_completed_manifest(source_path)
        directory = os.path.dirname(source_path)
        if manifest.get("task") != "generate_qm9" or "protocol.json" not in manifest["artifacts"]:
            raise ValueError("Panel source must be a completed generation run")
        protocol = read_json(os.path.join(directory, "protocol.json"))
        if (protocol["identities"] != context.identities or protocol["verifier"] == "cars"
                or protocol["method"] not in ("base_native", "base_rho", "vgr", "vgb", "vgb_momentum")):
            raise ValueError("Use matching BASE/baseline VGB runs; no CARS/BoN/reward-selected source")
        if (value_law == "rho") != (protocol["value_law"] == "rho"):
            raise ValueError("Source run and panel value law differ")
        sources.append({"manifest": source_path, "sha256": file_hash(source_path)})
        for row in read_jsonl(os.path.join(directory, "samples.jsonl"), progress=False):
            origin = {"source_manifest": source_path, "sample_id": row["sample_id"]}
            if row["state"]["rules"] != asdict(context.backbone.token_rules):
                raise ValueError("Source sample token rules differ")
            state = MaskedState(tuple(row["state"]["token_ids"]), context.backbone.token_rules)
            if state.is_complete:
                terminal_rows(state, origin)
            name = row["trace_artifact"]
            if name not in manifest["artifacts"]:
                raise ValueError("Unregistered source trajectory")
            trace = read_json(os.path.join(directory, name))["trace"]
            for step, event in enumerate(trace):
                for key in ("tokens", "next_tokens"):
                    if key in event:
                        add("walk_snapshot", MaskedState(tuple(event[key]), context.backbone.token_rules),
                            {**origin, "trace_step": step, "trace_field": key})
    if not source_manifests:
        count = configured_count(settings["panel"], "terminal_draws")
        pool = RNGPool(settings["seed"], "qm9-independent-value-panel-v1")
        steps = settings["panel"]["snapshot_steps"]
        maximum = context.native.policy.steps if value_law == "native-conditional" else context.rho.rules.sequence_length
        if any(type(step) is not int or not 0 <= step < maximum for step in steps):
            raise ValueError("Panel snapshot indices must fit the selected reference")
        for index in tqdm(range(count), desc="Independent panel trajectories", unit="draw"):
            origin = {"draw_id": index, "source": "independent_root_trajectory"}
            rng = pool.stream(f"draw-{index:06d}")
            unit = context.native.max_base_sequences if value_law == "native-conditional" else maximum
            held = budget.reserve_query(label="panel-root-draw", base_sequences=unit)
            if held is None:
                raise BudgetExceeded("Entire fixed panel preparation draw does not fit budget")
            with held:
                if value_law == "native-conditional":
                    draw = context.native.draw(rng=rng, budget=budget, reservation=held, phase=CostPhase.SAMPLING)
                    cursor = context.native.cursor()
                    for token in draw.tokens:
                        if cursor.position == 0 and cursor.step in steps:
                            add("native_snapshot", cursor.pre, {**origin, "native_step": cursor.step})
                        cursor.advance(token)
                    terminal_rows(draw.terminal, origin)
                else:
                    draw = context.rho.complete_batch([context.native.root], rng=rng, budget=budget,
                                                       reservation=held, phase=CostPhase.SAMPLING, snapshot_steps=steps)
                    if draw.exhausted or not draw.states[0].is_complete:
                        raise BudgetExceeded(draw.error or "Incomplete panel root draw")
                    for step, states in draw.snapshots.items():
                        add("rho_snapshot", states[0], {**origin, "rho_step": step})
                    terminal_rows(draw.states[0], origin)
    if any(not buckets[name] for name in ("root", "prefix", "hole")):
        raise ValueError("Source runs did not produce root/prefix/hole strata; no partial panel is locked")
    rows = [{"state_id": f"state-{i:06d}", **row} for i, row in enumerate(
        row for name in strata for row in buckets[name])]
    protocol = {"target": asdict(target), "value_law": value_law, "identities": context.identities,
                "inputs": context.inputs, "settings": settings, "sources": sources,
                "states_sha256": payload_hash(rows), "selection": "first_unique_state_per_stratum_in_fixed_source_order",
                "source_sha256": source_hashes(), "locked_before_compared_estimators": True}
    panel_id = "qm9-value-panel:" + payload_hash(protocol)
    attempt = RunArtifacts.create(settings["output_dir"], run_id=run_id, attempt_id=attempt_id,
                                  config=protocol, target_id=target.target_id, reference_id=target.reference_id)
    attempt.manifest.update(task="prepare_qm9_value_panel", panel_id=panel_id, target=asdict(target))
    try:
        synchronize(context.backbone.device)
        write_json(attempt.path("protocol.json"), protocol)
        write_jsonl(attempt.path("states.jsonl"), rows)
        write_json(attempt.path("preparation.json"), {"backend": "local", **(execution_metadata or {}),
                   "budget": record(budget.snapshot()), "wall_seconds": time.perf_counter() - started,
                   "strata_counts": {name: len(items) for name, items in buckets.items()},
                   "source_generation_cost_included": False})
        for name in ("protocol.json", "states.jsonl", "preparation.json"):
            attempt.register(name)
        attempt.complete(required_artifacts=("protocol.json", "states.jsonl", "preparation.json"))
    except Exception as exc:
        attempt.fail(f"{type(exc).__name__}: {exc}")
        raise
    report_result(f"Locked {len(rows)} {value_law} states; manifest={attempt.path('manifest.json')}")
    return {"manifest": attempt.path("manifest.json"), "states": len(rows)}


def load_panel(manifest_path, context, value_law):
    manifest = read_completed_manifest(project_path(manifest_path))
    if manifest.get("task") != "prepare_qm9_value_panel":
        raise ValueError("Expected a locked QM9 value panel")
    directory = os.path.dirname(manifest_path)
    protocol = read_json(os.path.join(directory, "protocol.json"))
    target = expected_target(context, value_law)
    rows = list(read_jsonl(os.path.join(directory, "states.jsonl"), progress=False))
    if (protocol["value_law"] != value_law or protocol["identities"] != context.identities
            or protocol["target"] != asdict(target) or manifest["target"] != asdict(target)
            or protocol["states_sha256"] != payload_hash(rows)
            or manifest["panel_id"] != "qm9-value-panel:" + payload_hash(protocol)
            or protocol.get("locked_before_compared_estimators") is not True
            or len({row["state_id"] for row in rows}) != len(rows) or not rows):
        raise ValueError("Panel identity, target, state order or lock differs")
    for row in rows:
        MaskedState(tuple(row["tokens"]), context.backbone.token_rules)
    return manifest, rows


def check_values(assets_manifest, checkpoint_manifest, calibration_manifest, *, panel_manifest,
                 attempt_id, estimators=("mc",), value_law="native-conditional", verifier_manifest=None,
                 config_path="configs/value_checks.yaml", inference_config="configs/mdm_vgb.yaml",
                 cars_config="configs/cars.yaml", reference_config="configs/reference_checks.yaml",
                 run_id="qm9-check-values", device="cuda", cache_mode="cold", warm_cache_manifest=None,
                 execution_metadata=None):
    setup_started = time.perf_counter()
    if not estimators or len(set(estimators)) != len(estimators) or not set(estimators).issubset({"mc", "neural", "cars"}):
        raise ValueError("Choose a nonempty unique estimator list from mc,neural,cars")
    if value_law not in ("native-conditional", "rho"):
        raise ValueError("Value comparisons require the native-conditional or rho law")
    if ("neural" in estimators) != bool(verifier_manifest):
        raise ValueError("Supply a neural checkpoint exactly when neural is selected")
    if (cache_mode not in ("cold", "warm") or (cache_mode == "warm") != bool(warm_cache_manifest)
            or cache_mode == "warm" and "cars" not in estimators):
        raise ValueError("Warm mode needs CARS and a completed cache source manifest")
    settings = load_config(config_path)
    context = load_context(assets_manifest, checkpoint_manifest, calibration_manifest, device=device,
                           config_path=inference_config, cars_config=cars_config, reference_config=reference_config)
    panel, rows = load_panel(panel_manifest, context, value_law)
    target = expected_target(context, value_law)
    ref = settings["reference"]
    draws = configured_count(ref, "draws")
    reference = (NativeConditionalMonteCarloVerifier(
        context.native, context.reward, draws=draws,
        min_denominator_hits=configured_count(ref, "min_denominator_hits"), confidence=ref["confidence"])
        if value_law == "native-conditional" else
        MonteCarloVerifier(context.rho, context.reward, draws=draws, confidence=ref["confidence"]))
    reference = RecordedVerifier(reference, context.backbone.device)
    adapters = {name: make_verifier(context, name, value_law, cache_scope=run_id + ":" + attempt_id,
                                    verifier_manifest=verifier_manifest if name == "neural" else None)
                for name in estimators}
    origins = warm_cache(adapters["cars"], warm_cache_manifest) if cache_mode == "warm" else {}
    setup_seconds = time.perf_counter() - setup_started
    protocol = {"target": asdict(target), "identities": context.identities, "inputs": context.inputs,
                "panel_id": panel["panel_id"], "panel_manifest": project_path(panel_manifest),
                "panel_manifest_sha256": file_hash(panel_manifest),
                "estimators": list(estimators), "settings": settings, "cars": context.cars,
                "inference": context.config, "mc_settings": load_config("configs/rollouts.yaml")["mc"],
                "reference": "independent_fixed_count_mc_not_exact", "cache_mode": cache_mode,
                "warm_source_sha256": file_hash(warm_cache_manifest) if warm_cache_manifest else None,
                "verifier_manifest_sha256": file_hash(verifier_manifest) if verifier_manifest else None,
                "source_sha256": source_hashes()}
    attempt = RunArtifacts.create(settings["check_output_dir"], run_id=run_id, attempt_id=attempt_id,
                                  config=protocol, target_id=target.target_id, reference_id=target.reference_id)
    attempt.manifest.update(task="check_qm9_values", panel_id=panel["panel_id"], target=asdict(target))
    pool, cached_states, available = RNGPool(settings["seed"], "qm9-value-check-v1"), set(), 0
    try:
        write_json(attempt.path("protocol.json"), protocol)
        attempt.register("protocol.json")

        def results():
            nonlocal available
            for row in tqdm(rows, desc="Check fixed value panel", unit="state"):
                state = MaskedState(tuple(row["tokens"]), context.backbone.token_rules)
                reference.scope = row["state_id"]
                ref_budget = BudgetTracker(**settings["budget"])
                ref_value = reference.values_for_states([state], rng=pool.stream(row["state_id"] + ":reference"),
                                                        budget=ref_budget)[0]
                candidates = {}
                for name, adapter in adapters.items():
                    adapter.scope = row["state_id"]
                    budget = BudgetTracker(**settings["budget"])
                    estimate = adapter.values_for_states([state], rng=pool.stream(row["state_id"] + ":" + name),
                                                          budget=budget)[0]
                    if estimate.target != target:
                        raise ValueError("Compared estimator changed target")
                    available += int(estimate.value is not None)
                    if name == "cars" and estimate.value is not None and not state.is_complete:
                        cached_states.add(state.token_ids)
                    candidates[name] = {"estimate": record(estimate), "queries": adapter.records,
                                        "budget": record(budget.snapshot()),
                                        "difference_from_mc_reference": estimate.value - ref_value.value
                                        if estimate.value is not None and ref_value.value is not None else None}
                    adapter.records = []
                yield {**row, "reference": record(ref_value), "reference_queries": reference.records,
                       "reference_budget": record(ref_budget.snapshot()), "estimators": candidates}
                reference.records = []

        write_jsonl(attempt.path("values.jsonl"), results(), progress=False)
        attempt.register("values.jsonl")
        save_cache(attempt, adapters.get("cars"), cached_states, origins)
        write_json(attempt.path("execution.json"), {"backend": "local", **(execution_metadata or {}),
                   "device": str(context.backbone.device), "precision": context.backbone.precision,
                   "python": platform.python_version(), "torch": torch.__version__,
                   "hardware": torch.cuda.get_device_name(context.backbone.device) if context.backbone.device.type == "cuda" else (platform.processor() or platform.machine()),
                   "states": len(rows), "available_estimates": available, "requested_estimates": len(rows) * len(estimators),
                   "setup_seconds": setup_seconds, "setup_includes_warm_cache_loading": True,
                   "ground_truth": "uncertain MC reference; terminal rewards alone are exact",
                   "cache_stats": adapters["cars"].inner.cache.stats if "cars" in adapters else None})
        attempt.register("execution.json")
        attempt.complete(required_artifacts=("protocol.json", "values.jsonl", "cache_index.jsonl", "execution.json"))
    except Exception as exc:
        attempt.fail(f"{type(exc).__name__}: {exc}")
        raise
    report_result(f"Checked {len(rows)} states; available={available}/{len(rows) * len(estimators)}; manifest={attempt.path('manifest.json')}")
    return {"manifest": attempt.path("manifest.json"), "states": len(rows), "available": available}
