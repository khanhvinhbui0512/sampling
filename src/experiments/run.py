"""QM9 run assembly and immutable artifacts; no benchmark settings are inferred."""

import math
import os
import platform
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from types import SimpleNamespace

import torch
from tqdm.auto import tqdm

from src.budget import BudgetTracker
from src.cars.cache import CARSCache, NativeCARSCache
from src.cars.estimator import CARSValueEstimator
from src.cars.native import NativeCARSValueEstimator
from src.cars.oracle import QM9Oracle
from src.config import load_config, project_path
from src.io import (RunArtifacts, file_hash, read_completed_manifest,
                    read_json, read_jsonl, report_result, write_json, write_jsonl)
from src.models.backbone import QM9Backbone
from src.qm9.dataset import load_prepared_qm9
from src.qm9.reward import QM9Reward, load_calibration
from src.qm9.rollouts import snapshot_indices
from src.randomness import RNGPool
from src.sampling.base import BaseSampler
from src.sampling.bon import BoNSampler
from src.sampling.candidates import CandidateSettings
from src.sampling.completion import RhoCompletion, cost_since
from src.sampling.momentum import MomentumSampler
from src.sampling.native import NativeMDLM
from src.sampling.native_reference import NativeReference
from src.sampling.rejection import RejectionSampler
from src.sampling.vgb import VGBSampler
from src.sampling.vgr import VGRSampler
from src.types import MaskedState, TargetKind, require_count
from src.verifiers.cars import CARSVerifier, NativeCARSVerifier
from src.verifiers.interface import Verifier, native_conditional_target, value_target
from src.verifiers.monte_carlo import MonteCarloVerifier, NativeConditionalMonteCarloVerifier
from src.verifiers.neural import NeuralVerifier


METHODS = ("base_native", "base_rho", "bon", "rs", "ars", "cars", "vgr", "vgb", "vgb_momentum")
VALUE_LAWS = ("native-conditional", "rho", "upstream-native")


def record(value):
    """Strict JSON, with tagged non-finite LOG diagnostics instead of JSON Infinity."""
    if is_dataclass(value):
        return record(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [record(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def configured_count(settings, key, *, positive=True):
    value = settings.get(key)
    if value is None:
        raise ValueError(f"Set {key} explicitly after profiling; no benchmark default is supplied")
    require_count(value, key, positive=positive)
    return value


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_context(assets_manifest, checkpoint_manifest, calibration_manifest, *, device="cuda",
                 config_path="configs/mdm_vgb.yaml", cars_config="configs/cars.yaml",
                 reference_config="configs/reference_checks.yaml"):
    paths = [project_path(p) for p in (assets_manifest, checkpoint_manifest, calibration_manifest)]
    assets = load_prepared_qm9(paths[0])
    checkpoint = read_completed_manifest(paths[1])
    if checkpoint.get("identities", {}).get("assets_manifest_sha256") != file_hash(paths[0]):
        raise ValueError("Backbone and prepared QM9 assets differ")
    backbone = QM9Backbone.from_checkpoint(paths[1], device=device, precision="bf16")
    if backbone.tokenizer_id != assets.tokenizer_sha256 or backbone.token_rules != assets.token_rules:
        raise ValueError("Backbone tokenizer/token rules differ from prepared QM9")
    backbone.model.eval().requires_grad_(False)
    calibration = load_calibration(paths[2])
    reward = QM9Reward.from_assets(paths[0], calibration_manifest=paths[2],
                                  calibration_reference_id=calibration.reference_id,
                                  calibration_policy_id=calibration.policy_id,
                                  checkpoint_sha256=backbone.checkpoint_id)
    reference_settings = load_config(reference_config)["qm9"]["native"]
    native = NativeReference(NativeMDLM(backbone, **reference_settings))
    # Also for rho comparisons: retain the ONE locked native BASE threshold.
    native_conditional_target(native, reward)
    config, cars = load_config(config_path), load_config(cars_config)
    if config.get("task") != "qm9" or cars.get("task") != "qm9":
        raise ValueError("This factory supports the QM9 experiment only")
    identities = {
        "assets_manifest_sha256": file_hash(paths[0]), "checkpoint_manifest_sha256": file_hash(paths[1]),
        "calibration_manifest_sha256": file_hash(paths[2]), "checkpoint_id": backbone.checkpoint_id,
        "tokenizer_id": backbone.tokenizer_id, "rules": asdict(backbone.token_rules),
        "calibration": asdict(calibration), "native_reference": native.protocol,
    }
    return SimpleNamespace(backbone=backbone, reward=reward, native=native,
                           rho=RhoCompletion(backbone, temperature=config["candidates"]["temperature"]),
                           config=config, cars=cars, identities=identities,
                           inputs=dict(zip(("assets_manifest", "checkpoint_manifest", "calibration_manifest"), paths)))


def expected_target(context, value_law):
    if value_law == "native-conditional":
        return native_conditional_target(context.native, context.reward)
    if value_law == "rho":
        return value_target(context.rho, context.reward)
    if value_law == "upstream-native":
        settings = load_config("configs/rollouts.yaml")
        policy = NativeMDLM(context.backbone, **settings["native"])
        labels = {"law": "native_trajectory_terminal_conditional",
                  "snapshot_steps": snapshot_indices(policy.steps, settings["snapshots_per_rollout"]),
                  "time_input": False}
        return value_target(policy, context.reward, kind=TargetKind.NATIVE_ROLLOUT, label_protocol=labels)
    raise ValueError("Unknown value law")


class RecordedVerifier(Verifier):
    """Keep ordered queries, refused values, real work and synchronized group wall time."""

    def __init__(self, inner, device):
        super().__init__(inner.target, inner.reward)
        self.inner, self.device, self.records = inner, device, []
        self.scope = ""

    def required_work(self, states):
        return self.inner.required_work(states)

    def values_for_states(self, states, *, rng, budget, reservation=None):
        before = budget.snapshot().total
        synchronize(self.device)
        started = time.perf_counter()
        estimates = self.inner.values_for_states(states, rng=rng, budget=budget, reservation=reservation)
        synchronize(self.device)
        group = {"scope": self.scope, "items": [
            {"tokens": list(state.token_ids), "estimate": record(estimate)}
            for state, estimate in zip(states, estimates)],
            "wall_seconds": time.perf_counter() - started, "cost": asdict(cost_since(budget, before))}
        if isinstance(self.inner, NativeConditionalMonteCarloVerifier):
            summaries = iter(self.inner.summaries)
            for item, estimate in zip(group["items"], estimates):
                if estimate.draws == self.inner.draws and estimate.status.value in ("estimated", "failed"):
                    item["conditional_summary"] = record(next(summaries))
            self.inner.summaries.clear()
        if isinstance(self.inner, (CARSVerifier, NativeCARSVerifier)):
            cold = iter(self.inner.runs)
            for state, item, estimate in zip(states, group["items"], estimates):
                if not state.is_complete and not estimate.cache_hit:
                    run = next(cold)
                    item["cars"] = record({"metadata": run.metadata, "discovery": run.discovery_trace,
                                           "estimation": run.estimation_trace, "rng": run.rng_state})
                if not state.is_complete and estimate.value is not None:
                    item["cache_provenance"] = self.inner.cache.details(state)
            self.inner.runs.clear()
        self.records.append(group)
        return estimates


def make_verifier(context, name, value_law, *, cache_scope, verifier_manifest=None):
    target = expected_target(context, value_law)
    if name == "neural":
        if not verifier_manifest:
            raise ValueError("Neural guidance requires --verifier-manifest for the selected value law")
        inner = NeuralVerifier.from_checkpoint(
            project_path(verifier_manifest), reward=context.reward, expected_target=target,
            backbone_checkpoint_id=context.backbone.checkpoint_id, tokenizer_id=context.backbone.tokenizer_id,
            device=context.backbone.device)
    elif name == "mc":
        if value_law == "native-conditional":
            settings = context.cars["native"]
            inner = NativeConditionalMonteCarloVerifier(
                context.native, context.reward, draws=configured_count(settings, "mc_control_draws"),
                min_denominator_hits=configured_count(settings, "min_denominator_hits"),
                confidence=settings["confidence"])
        elif value_law == "rho":
            inner = MonteCarloVerifier(context.rho, context.reward, **load_config("configs/rollouts.yaml")["mc"])
        else:
            raise ValueError("Upstream snapshot labels have no matched root-conditional MC estimator")
    elif name == "cars":
        settings = context.cars
        if value_law == "native-conditional":
            settings = settings["native"]
            estimator = NativeCARSValueEstimator(
                context.native, context.reward,
                discovery_attempts=configured_count(settings, "discovery_attempts", positive=False),
                numerator_draws=configured_count(settings, "numerator_draws"),
                denominator_draws=configured_count(settings, "denominator_draws"),
                max_nodes=configured_count(settings, "max_nodes"),
                min_denominator_hits=configured_count(settings, "min_denominator_hits"),
                discovery_max_base_sequences=settings["discovery_max_base_sequences"],
                mode=settings["mode"], confidence=settings["confidence"])
            cache = NativeCARSCache(
                estimator.cache_metadata, reference=context.native, reward=context.reward,
                max_entries=configured_count(settings, "max_cache_entries"), run_id=cache_scope,
                directory=settings["directory"] if settings["persistent_cache"] else None)
            inner = NativeCARSVerifier(estimator, cache=cache)
        elif value_law == "rho":
            oracle = QM9Oracle(context.reward, mode=settings["oracle"]["branch_mode"])
            estimator = CARSValueEstimator(
                context.rho, context.reward, oracle, mode="cars",
                discovery_attempts=configured_count(settings["discovery"], "attempts", positive=False),
                draws=configured_count(settings["estimation"], "draws"),
                max_nodes=configured_count(settings["memory"], "max_nodes"),
                discovery_max_base_sequences=settings["discovery"]["max_base_sequences"],
                confidence=settings["estimation"]["confidence"])
            cache = CARSCache(estimator.cache_metadata, max_entries=configured_count(settings["cache"], "max_entries"),
                              run_id=cache_scope, directory=settings["cache"]["directory"]
                              if settings["cache"]["persistent_cache"] else None)
            inner = CARSVerifier(estimator, cache=cache)
        else:
            raise ValueError("CARS-native V* cannot replace an upstream snapshot target by relabeling")
    else:
        raise ValueError("Verifier must be neural, mc or cars")
    if inner.target != target:
        raise ValueError("Assembled verifier differs from the requested target")
    return RecordedVerifier(inner, context.backbone.device)


def make_sampler(context, method, *, verifier=None):
    if method not in METHODS:
        raise ValueError("Unknown QM9 method")
    config = context.config
    if method in ("rs", "ars", "cars"):
        settings = context.cars
        return RejectionSampler(
            context.rho, context.reward, method=method,
            max_attempts=configured_count(settings["generation"], "max_attempts"),
            max_nodes=configured_count(settings["memory"], "max_nodes") if method != "rs" else None)
    base_method = method if method in ("base_native", "base_rho") else config["base_method"]
    base = BaseSampler(context.backbone, context.reward, method=base_method,
                       native_settings=config["native_settings"], temperature=context.rho.temperature)
    if base_method == "base_native" and base.reference_id != context.native.reference_id:
        raise ValueError("Generation native settings differ from the calibrated normalized BASE")
    if method in ("base_native", "base_rho"):
        return base
    if method == "bon":
        return BoNSampler(base, **config["bon"])
    if verifier is None:
        raise ValueError("Guided generation needs a verifier")
    rho = verifier.target.kind == TargetKind.RHO
    if (base_method == "base_rho") != rho:
        raise ValueError("Set base_method to base_rho only for a separately labelled rho comparison")
    kwargs = dict(candidates=CandidateSettings(**config["candidates"]), **config["walk"])
    if method == "vgr":
        return VGRSampler(context.backbone, verifier, base, **kwargs)
    if method == "vgb":
        return VGBSampler(context.backbone, verifier, base, **kwargs)
    return MomentumSampler(context.backbone, verifier, base, **config["momentum"], **kwargs)


def warm_cache(verifier, source_manifest):
    if not isinstance(verifier.inner, (CARSVerifier, NativeCARSVerifier)) or verifier.inner.cache.directory is None:
        raise ValueError("Warm cache requires CARS with persistent_cache=true")
    source = read_completed_manifest(project_path(source_manifest))
    if "cache_index.jsonl" not in source["artifacts"]:
        raise ValueError("Warm source has no registered CARS snapshots")
    directory, origins = os.path.dirname(source_manifest), {}
    for row in read_jsonl(os.path.join(directory, "cache_index.jsonl")):
        if row["snapshot"] not in source["artifacts"]:
            raise ValueError("Warm snapshot is outside the completed source manifest")
        state = MaskedState(tuple(row["tokens"]), verifier.rules)
        path = project_path(os.path.join(directory, row["snapshot"]))
        if not verifier.inner.cache.load(state, path=path):
            raise ValueError("Missing warm snapshot")
        origins[state.token_ids] = path
    if not origins:
        raise ValueError("Warm source contains no available cached queries")
    return origins


def save_cache(attempt, verifier, states, origins):
    index = []
    if verifier is not None and isinstance(verifier.inner, (CARSVerifier, NativeCARSVerifier)):
        cache = verifier.inner.cache
        if cache.directory is not None:
            for tokens in sorted(states):
                state = MaskedState(tokens, verifier.rules)
                path = cache.path(state)
                if not os.path.isfile(path):
                    path = origins[tokens]
                name = "cache/" + cache.key(state) + ".json"
                write_json(attempt.path(name), read_json(path))
                attempt.register(name)
                index.append({"tokens": list(tokens), "snapshot": name,
                              "provenance": cache.details(state)})
    write_jsonl(attempt.path("cache_index.jsonl"), index, progress=False)
    attempt.register("cache_index.jsonl")


def source_hashes():
    files = []
    for directory, _, names in os.walk("src"):
        files.extend(os.path.join(directory, name) for name in names if name.endswith(".py"))
    return {path: file_hash(path) for path in sorted(files)}


def generate(assets_manifest, checkpoint_manifest, calibration_manifest, *, method, attempt_id,
             num_samples=1, seed=1, verifier="none", value_law="native-conditional",
             verifier_manifest=None, config_path="configs/mdm_vgb.yaml", cars_config="configs/cars.yaml",
             reference_config="configs/reference_checks.yaml", run_id="qm9-generate", device="cuda",
             cache_mode="cold", warm_cache_manifest=None, execution_metadata=None):
    require_count(num_samples, "requested outputs", positive=True)
    require_count(seed, "generation seed")
    if method not in METHODS or value_law not in VALUE_LAWS or cache_mode not in ("cold", "warm"):
        raise ValueError("Invalid generation method/value law/cache mode")
    guided = method in ("vgr", "vgb", "vgb_momentum")
    if guided != (verifier != "none") or (verifier != "neural" and verifier_manifest):
        raise ValueError("Only guided methods select a verifier; only neural uses its checkpoint")
    if (cache_mode == "warm") != bool(warm_cache_manifest) or (cache_mode == "warm" and verifier != "cars"):
        raise ValueError("Warm CARS needs exactly one completed --warm-cache-manifest")
    setup_started = time.perf_counter()
    context = load_context(assets_manifest, checkpoint_manifest, calibration_manifest, device=device,
                           config_path=config_path, cars_config=cars_config, reference_config=reference_config)
    adapter = make_verifier(context, verifier, value_law, cache_scope=run_id + ":" + attempt_id,
                            verifier_manifest=verifier_manifest) if guided else None
    sampler = make_sampler(context, method, verifier=adapter)
    actual_law = value_law if guided else ("rho" if sampler.target.kind == TargetKind.RHO else "native-base")
    protocol = {"method": method, "verifier": verifier, "value_law": actual_law,
                "target": asdict(sampler.target), "identities": context.identities, "inputs": context.inputs,
                "inference": context.config, "cars": context.cars,
                "mc_settings": load_config("configs/rollouts.yaml")["mc"],
                "seed": seed, "requested_outputs": num_samples,
                "cache_mode": cache_mode, "warm_cache_manifest": warm_cache_manifest,
                "warm_source_sha256": file_hash(warm_cache_manifest) if warm_cache_manifest else None,
                "verifier_manifest_sha256": file_hash(verifier_manifest) if verifier_manifest else None,
                "source_sha256": source_hashes(), "claim": "project_variant_not_verified_paper_reproduction",
                "native_gate_passed": load_config(reference_config)["qm9"]["native_reference"]["gate_passed"]}
    attempt = RunArtifacts.create("workspace/results/raw", run_id=run_id, attempt_id=attempt_id,
                                  config=protocol, target_id=sampler.target.target_id,
                                  reference_id=sampler.target.reference_id)
    attempt.manifest.update(task="generate_qm9", method=method, value_law=actual_law,
                            target=asdict(sampler.target), requested_outputs=num_samples)
    costs, cached_states, successes = [], set(), 0
    try:
        origins = warm_cache(adapter, warm_cache_manifest) if cache_mode == "warm" else {}
        write_json(attempt.path("protocol.json"), protocol)
        attempt.register("protocol.json")
        write_json(attempt.path("execution.json"), {"backend": "local", **(execution_metadata or {}),
                   "python": platform.python_version(), "torch": torch.__version__, "device": str(context.backbone.device),
                   "hardware": torch.cuda.get_device_name(context.backbone.device) if context.backbone.device.type == "cuda" else (platform.processor() or platform.machine()),
                   "precision": context.backbone.precision, "setup_seconds": time.perf_counter() - setup_started})
        attempt.register("execution.json")
        pool = RNGPool(seed, "qm9-generation-v1")
        root = context.native.root

        def rows():
            nonlocal successes
            for index in tqdm(range(num_samples), desc="QM9 generation", unit="output"):
                sample_id = f"sample-{index:06d}"
                rng, budget = pool.stream(sample_id), BudgetTracker(
                    base_sequence_limit=context.config["budget"]["base_sequence_limit"],
                    verifier_sequence_limit=context.config["budget"]["verifier_sequence_limit"])
                if adapter:
                    adapter.scope = sample_id
                synchronize(context.backbone.device)
                started = time.perf_counter()
                if isinstance(sampler, BaseSampler):
                    result = sampler.sample(root, sample_id=sample_id, rng=rng, budget=budget)
                    trace, metadata, rng_state = [], {"initial_tokens": root.token_ids}, rng.state_dict()
                else:
                    run = sampler.run(root, sample_id=sample_id, rng=rng, budget=budget)
                    result, trace, metadata, rng_state = run.result, run.trace, run.metadata, run.rng_state
                synchronize(context.backbone.device)
                wall = time.perf_counter() - started
                queries = adapter.records if adapter else []
                for group in queries:
                    cached_states.update(tuple(item["tokens"]) for item in group["items"]
                                         if item["estimate"]["value"] is not None
                                         and context.backbone.token_rules.mask_id in item["tokens"])
                trace_name, query_name = f"traces/{sample_id}.json", f"value_queries/{sample_id}.json"
                write_json(attempt.path(trace_name), record({"result": result, "trace": trace, "metadata": metadata,
                           "rng": rng_state, "budget": budget.snapshot(), "wall_seconds": wall}))
                write_json(attempt.path(query_name), record(queries))
                attempt.register(trace_name)
                attempt.register(query_name)
                if adapter:
                    adapter.records = []
                costs.append({"sample_id": sample_id, "wall_seconds": wall, "budget": record(budget.snapshot())})
                successes += int(result.status.value == "completed" and result.reward == 1)
                yield {**record(result), "trace_artifact": trace_name, "queries_artifact": query_name}

        write_jsonl(attempt.path("samples.jsonl"), rows(), progress=False)
        attempt.register("samples.jsonl")
        write_jsonl(attempt.path("costs.jsonl"), costs, progress=False)
        attempt.register("costs.jsonl")
        save_cache(attempt, adapter, cached_states, origins)
        write_json(attempt.path("summary.json"), {"requested_outputs": num_samples, "successes": successes,
                   "wall_seconds": sum(row["wall_seconds"] for row in costs),
                   "cache_stats": adapter.inner.cache.stats if verifier == "cars" else None,
                   "timing": "synchronized sampling including queries/trie; excludes setup and artifact writes"})
        attempt.register("summary.json")
        attempt.complete(required_artifacts=("protocol.json", "execution.json", "samples.jsonl", "costs.jsonl",
                                             "cache_index.jsonl", "summary.json"))
    except Exception as exc:
        attempt.fail(f"{type(exc).__name__}: {exc}")
        raise
    report_result(f"Generated {num_samples} outputs; Pass@95={successes}/{num_samples}; manifest={attempt.path('manifest.json')}")
    return {"manifest": attempt.path("manifest.json"), "outputs": num_samples, "successes": successes}
