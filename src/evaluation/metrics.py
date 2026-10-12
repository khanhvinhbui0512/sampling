"""Artifact metrics. Generation outputs and estimator draws have different roles."""

import math
import random
from collections import Counter
from dataclasses import asdict

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from tqdm.auto import tqdm

from src.types import CostRecord, MaskedState, SampleResult, TokenRules, ValueTarget, require_count


QUALITY_METRICS = ("pass95", "validity", "qed_valid_mean", "qed_requested_mean",
                   "unique_pass95", "uniqueness", "novelty", "diversity")


def mean(values):
    return sum(values) / len(values) if values else None


def finite_number(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def diversity(smiles, settings):
    """Uniform subsample before fingerprinting; repeated valid outputs retain weight."""
    require_count(settings["radius"], "Morgan radius", positive=True)
    require_count(settings["bits"], "Morgan bits", positive=True)
    require_count(settings["max_molecules"], "diversity cap", positive=True)
    require_count(settings["seed"], "diversity seed")
    if type(settings["include_chirality"]) is not bool or settings["max_molecules"] < 2:
        raise ValueError("Diversity needs a boolean chirality setting and cap >= 2")
    indices = list(range(len(smiles)))
    if len(indices) > settings["max_molecules"]:
        indices = sorted(random.Random(settings["seed"]).sample(indices, settings["max_molecules"]))
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=settings["radius"], fpSize=settings["bits"],
        includeChirality=settings["include_chirality"])
    fingerprints = []
    for index in tqdm(indices, desc="Morgan fingerprints", unit="molecule"):
        molecule = Chem.MolFromSmiles(smiles[index])
        if molecule is None:
            raise ValueError("A validated canonical SMILES failed fingerprint parsing")
        fingerprints.append(generator.GetFingerprint(molecule))
    pairs = len(indices) * (len(indices) - 1) // 2
    distance = sum(i - sum(DataStructs.BulkTanimotoSimilarity(fp, fingerprints[:i]))
                   for i, fp in enumerate(fingerprints))
    return {"value": distance / pairs if pairs else None, "valid_outputs": len(smiles),
            "selected_outputs": len(indices), "pairs": pairs, "selected_indices": indices,
            "settings": settings, "population": "completed valid outputs, including duplicates",
            "selection": "uniform without replacement, fixed seed, before fingerprinting"}


def generation_metrics(samples, *, requested, reward, target, train_smiles, diversity_settings):
    require_count(requested, "requested outputs", positive=True)
    if len(samples) != requested or len({row["sample_id"] for row in samples}) != requested:
        raise ValueError("Generation must contain one unique row for every requested output")
    valid, passing, qeds, statuses = [], [], [], Counter()
    for row in tqdm(samples, desc="Recompute terminal metrics", unit="output"):
        state = MaskedState(tuple(row["state"]["token_ids"]), TokenRules(**row["state"]["rules"]))
        if state.rules != reward.token_rules or row["target"] != target:
            raise ValueError("Sample tokenizer/target differs from its protocol")
        SampleResult(**{key: row[key] for key in ("sample_id", "status", "smiles", "canonical_smiles",
                                                  "valid", "score", "reward", "error")},
                     state=state, target=ValueTarget(**target), cost=CostRecord(**row["cost"]))
        statuses[row["status"]] += 1
        if row["status"] != "completed":
            continue
        terminal = reward.evaluate(state, require_binary=True)
        canonical = terminal.molecule.canonical_smiles if terminal.molecule else None
        if (row["valid"] != terminal.valid or row["reward"] != terminal.reward
                or row["canonical_smiles"] != canonical
                or row["smiles"] != (terminal.molecule.smiles if terminal.molecule else None)
                or row["score"] is None
                or not math.isclose(row["score"], terminal.score, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("Saved terminal assessment differs from locked tokenizer/RDKit/calibration")
        if row["status"] == "completed" and terminal.valid:
            valid.append(canonical)
            qeds.append(terminal.score)
            if terminal.reward == 1:
                passing.append(canonical)
    unique = set(valid)
    novel = unique - train_smiles
    div = diversity(valid, diversity_settings)
    return {"requested_outputs": requested, "statuses": dict(statuses), "valid_outputs": len(valid),
            "passing_outputs": len(passing), "unique_valid": len(unique), "unique_novel": len(novel),
            "pass95": len(passing) / requested, "validity": len(valid) / requested,
            "qed_valid_mean": mean(qeds), "qed_requested_mean": sum(qeds) / requested,
            "unique_pass95": len(set(passing)) / requested,
            "uniqueness": len(unique) / len(valid) if valid else None,
            # Same denominator as released MDM-VGB novel_ratio_of_valid.
            "novelty": len(novel) / len(valid) if valid else None,
            "novelty_of_unique": len(novel) / len(unique) if unique else None,
            "diversity": div["value"], "diversity_details": div,
            "denominators": {"pass95_validity_unique_pass95_qed_requested": requested,
                             "qed_valid_uniqueness_novelty": len(valid), "diversity_pairs": div["pairs"]}}


def cost_metrics(rows):
    """Sum ledger work once. Query costs are diagnostic subsets, never added again."""
    total, phases, adjusted, measurements = CostRecord(), {}, [], set()
    wall, batch_sizes = 0.0, {"base": set(), "verifier": set()}
    for row in rows:
        budget = row["budget"]
        cost = CostRecord(**budget["total"])
        total += cost
        wall += finite_number(row["wall_seconds"], "wall seconds")
        phase_total = CostRecord()
        for phase, raw in budget["by_phase"].items():
            item = CostRecord(**raw)
            phases[phase] = phases.get(phase, CostRecord()) + item
            phase_total += item
        for key, value in asdict(cost).items():
            if not math.isclose(value, getattr(phase_total, key), rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError("Budget phases do not sum to total")
        for prefix in ("base", "verifier"):
            sizes = budget[prefix + "_batch_sizes"]
            calls, sequences = getattr(cost, prefix + "_calls"), getattr(cost, prefix + "_sequences")
            # BudgetSnapshot stores DISTINCT observed sizes, not one item per call.
            if (any(type(size) is not int or size < 1 for size in sizes) or sizes != sorted(set(sizes))
                    or bool(sizes) != bool(calls) or len(sizes) > calls):
                raise ValueError("Recorded batch sizes disagree with call/sequence work")
            if sizes:
                remaining = calls - len(sizes)
                lower, upper = sum(sizes) + remaining*min(sizes), sum(sizes) + remaining*max(sizes)
                if not lower <= sequences <= upper:
                    raise ValueError("Sequence work cannot come from the recorded distinct batch sizes")
            batch_sizes[prefix].update(sizes)
        value = budget["adjusted_nfe"]
        if value is None and cost.verifier_calls == 0:
            value = cost.base_calls
        if value is not None:
            finite_number(value, "adjusted NFE")
            if value < cost.base_calls:
                raise ValueError("Adjusted NFE cannot be below raw NFE")
            if cost.verifier_calls and not budget["flop_measurement_id"]:
                raise ValueError("Neural adjusted NFE requires measured FLOP provenance")
        adjusted.append(value)
        if budget["flop_measurement_id"]:
            measurements.add(budget["flop_measurement_id"])
    return {"total": asdict(total), "by_phase": {key: asdict(value) for key, value in phases.items()},
            "wall_seconds": wall, "adjusted_nfe": sum(adjusted) if all(x is not None for x in adjusted) and len(measurements) <= 1 else None,
            "flop_measurement_ids": sorted(measurements),
            "distinct_batch_sizes": {key: sorted(value) for key, value in batch_sizes.items()},
            "timing": "sampling/query wall excludes setup and artifact I/O; call elapsed is not wall time"}


def cache_metrics(groups):
    """Current work and unique historical snapshot work are separate accounting views."""
    counts, hits, current, origins = 0, 0, CostRecord(), {}
    wall = 0.0
    for group in groups:
        current += CostRecord(**group["cost"])
        wall += finite_number(group["wall_seconds"], "query wall seconds")
        for item in group["items"]:
            counts += 1
            hits += int(item["estimate"]["cache_hit"])
            provenance = item.get("cache_provenance")
            if provenance:
                key = provenance["origin_run_id"] + ":" + provenance["snapshot_id"]
                origin = {name: provenance[name] for name in ("origin_run_id", "origin_draws", "origin_cost")}
                if key in origins and origins[key] != origin:
                    raise ValueError("Cache snapshot has conflicting historical provenance")
                origins[key] = origin
    historical = CostRecord()
    for item in origins.values():
        historical += CostRecord(**item["origin_cost"])
    return {"queries": counts, "hits": hits, "hit_rate": hits / counts if counts else None,
            "current_query_cost": asdict(current), "query_wall_seconds": wall,
            "unique_snapshot_origin_cost": asdict(historical), "origins": origins,
            "accounting": "origin work is historical, may already be in cold-run ledger; never add to current total"}


def cluster_key(row):
    origin = row.get("origin", {})
    if "sample_id" in origin and "source_manifest" in origin:
        return origin["source_manifest"] + ":" + origin["sample_id"]
    if "draw_id" in origin:
        return "independent-draw:" + str(origin["draw_id"])
    return None


def value_metrics(rows, estimator, *, toy=False):
    """Keep all requested states; error summaries condition explicitly on availability."""
    details, costs, groups = [], [], []
    for index, row in enumerate(rows):
        candidate = row["estimators"][estimator]
        estimate = candidate["estimate"]
        value = estimate["value"]
        reference = row["exact_value"] if toy else row["reference"]["value"]
        interval = [reference, reference] if toy and reference is not None else (None if toy else row["reference"]["interval"])
        for number in (value, reference):
            if number is not None and not 0 <= finite_number(number, "value") <= 1:
                raise ValueError("Available values must be within [0, 1]")
        if interval is not None:
            if len(interval) != 2 or not 0 <= interval[0] <= interval[1] <= 1:
                raise ValueError("Invalid reference interval")
        difference = value - reference if value is not None and reference is not None else None
        details.append({"state_id": row.get("state_id", row.get("name", str(index))),
                        "stratum": row.get("stratum", "toy"), "cluster": cluster_key(row),
                        "value": value, "reference": reference, "reference_interval": interval,
                        "status": estimate["status"], "draws": estimate["draws"],
                        "standard_error": estimate["standard_error"], "interval": estimate["interval"],
                        "cache_hit": estimate["cache_hit"], "error": estimate["error"],
                        "reference_estimate": None if toy else row["reference"],
                        "reference_queries": row.get("reference_queries", []),
                        "conditional_diagnostics": [item.get("conditional_summary")
                            or item.get("cars", {}).get("metadata", {}).get("summary")
                            or item.get("cache_provenance", {}).get("conditional_summary")
                            for group in candidate["queries"] for item in group["items"]],
                        "difference": difference,
                        "absolute_error_reference_interval_min": max(interval[0] - value, value - interval[1], 0)
                        if value is not None and interval is not None else None,
                        "absolute_error_reference_interval_max": max(abs(value - interval[0]), abs(value - interval[1]))
                        if value is not None and interval is not None else None})
        groups.extend(candidate["queries"])
        costs.append({"budget": candidate["budget"], "wall_seconds": sum(group["wall_seconds"] for group in candidate["queries"])})
    errors = [item["difference"] for item in details if item["difference"] is not None]
    # Compare estimators on the same available states; never reward refusal with
    # a fabricated zero error. Individual available-pair errors remain separate.
    common = [item["difference"] for item, row in zip(details, rows)
              if item["difference"] is not None
              and all(entry["estimate"]["value"] is not None for entry in row["estimators"].values())]
    return {"requested_states": len(rows), "available_estimates": sum(item["value"] is not None for item in details),
            "available_references": sum(item["reference"] is not None for item in details),
            "paired_states": len(errors), "availability": sum(item["value"] is not None for item in details) / len(rows) if rows else None,
            "mae": mean([abs(x) for x in errors]), "rmse": math.sqrt(mean([x*x for x in errors])) if errors else None,
            "bias": mean(errors), "cost": cost_metrics(costs), "cache": cache_metrics(groups), "states": details,
            "common_pair_states": len(common), "common_pair_mae": mean([abs(x) for x in common]),
            "common_pair_rmse": math.sqrt(mean([x*x for x in common])) if common else None,
            "common_pair_state_ids": [item["state_id"] for item, row in zip(details, rows)
                if item["difference"] is not None
                and all(entry["estimate"]["value"] is not None for entry in row["estimators"].values())],
            "full_panel_mae": mean([abs(x) for x in errors]) if len(errors) == len(rows) else None,
            "reference_kind": "exact finite enumeration" if toy else "independent noisy MC; point error is not error against exact V*"}


def bootstrap_interval(values, settings):
    """Independent seed-level means only; one seed cannot produce a seed CI."""
    if len(values) < 2 or any(value is None for value in values):
        return None
    rng = np.random.default_rng(settings["seed"])
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True)))
             for _ in range(settings["bootstrap_replicates"])]
    tail = (1 - settings["confidence"]) / 2
    return np.quantile(means, [tail, 1-tail]).tolist()


def rollout_error_interval(details, settings):
    """Resample rollout origins, keeping masks/snapshots together. No toy CI."""
    if any(item["cluster"] is None for item in details):
        return None
    clusters = {}
    for item in details:
        clusters.setdefault(item["cluster"], []).append(item["difference"])
    if len(clusters) < 2:
        return None
    values = list(clusters.values())
    rng = np.random.default_rng(settings["seed"])
    draws = []
    for _ in range(settings["bootstrap_replicates"]):
        errors = [error for index in rng.integers(len(values), size=len(values))
                  for error in values[index] if error is not None]
        if errors:
            draws.append([mean([abs(x) for x in errors]), math.sqrt(mean([x*x for x in errors]))])
    if len(draws) != settings["bootstrap_replicates"]:
        return None
    tail = (1-settings["confidence"])/2
    intervals = np.quantile(draws, [tail, 1-tail], axis=0)
    return {"mae": intervals[:, 0].tolist(), "rmse": intervals[:, 1].tolist(),
            "clusters": len(clusters), "scope": "rollout-cluster bootstrap conditional on retained panel and realized estimates; not estimator confidence, does not remove MC reference noise"}
