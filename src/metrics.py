"""Explicit generation denominators and costs; selected successes are conditional."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from src.data import normalize_smiles
from src.retrosynthesis import RetroStarProxy


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {"value": numerator/denominator if denominator else None, "numerator": numerator,
            "denominator": denominator, "status": "available" if denominator else "unavailable"}


def fingerprint_diversity(fingerprints: Sequence[Any]) -> dict[str, Any]:
    pairs = len(fingerprints)*(len(fingerprints)-1)//2
    distance = math.fsum(1-similarity for index, fp in enumerate(fingerprints)
                         for similarity in DataStructs.BulkTanimotoSimilarity(fp, fingerprints[:index]))
    return {"value": distance/pairs if pairs else None, "pair_denominator": pairs,
            "valid_molecules": len(fingerprints), "status": "available" if pairs else "unavailable"}


def molecule_metrics(rows: Sequence[Mapping[str, Any]], prompts: set[str], *, radius: int = 2,
                     bits: int = 2048, sensitivity_radii: Sequence[int] = (3,),
                     retro: RetroStarProxy | None = None) -> dict[str, Any]:
    results = [row["reward_result"] for row in rows]
    if any(result["rdkit_valid"] and not result["canonical_smiles"] for result in results):
        raise ValueError("A valid molecule must retain its canonical identity")
    canonical = [result["canonical_smiles"] if result["rdkit_valid"] else None for result in results]
    valid = [text for text in canonical if text is not None]
    identities = set(valid)
    molecules = [normalize_smiles(text)[0] for text in valid]
    diversities = {}
    for current in (radius, *sensitivity_radii):
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=current, fpSize=bits)
        diversities[str(current)] = fingerprint_diversity([generator.GetFingerprint(mol) for mol in molecules])
    return {"count": len(rows), "valid_molecules": len(valid), "unique_molecules": len(identities),
            "grammar_validity": ratio(sum(result["grammar_valid"] for result in results), len(rows)),
            "validity": ratio(len(valid), len(rows)),
            "membership": ratio(sum(result["class_member"] for result in results), len(rows)),
            "membership_on_valid": ratio(sum(result["class_member"] for result in results), len(valid)),
            "reward_success": ratio(sum(result["reward"] == 1 for result in results), len(rows)),
            "unique_fraction": ratio(len(identities), len(valid)),
            "novelty": ratio(sum(text not in prompts for text in valid), len(valid)),
            "novel_unique_fraction": ratio(len(identities-prompts), len(identities)),
            "novel_unique_molecules": len(identities-prompts),
            "diversity": diversities[str(radius)], "diversity_by_radius": diversities,
            "fingerprint": {"radius": radius, "bits": bits, "use_chirality": False},
            "retro_score": (retro or RetroStarProxy(None)).score(canonical)}


def cost_metrics(tracking: Mapping[str, Any], *, reference_forward_flops: float | None = None) -> dict[str, Any]:
    counters = dict(tracking["counters"])
    raw = sum(counters.get(name+"_forward_calls", 0) for name in ("mdlm", "ar"))
    inner = sum(counters.get("inner_"+name+"_forward_calls", 0) for name in ("mdlm", "ar"))
    if inner > raw:
        raise ValueError("Inner calls must be a subset of total model calls")
    fully_profiled = all(counters.get("flop_profiled_"+name+"_calls", 0) == counters.get(name+"_forward_calls", 0)
                         for name in ("mdlm", "ar", "verifier"))
    adjusted = (sum(tracking.get("flops", {}).values())/reference_forward_flops
                if reference_forward_flops and fully_profiled else None)
    return {**counters, "raw_nfe": raw, "inner_model_calls": inner, "outer_model_calls": raw-inner,
            "model_example_nfe": sum(counters.get(name+"_forward_examples", 0) for name in ("mdlm", "ar")),
            "tracked_compute_seconds": sum(tracking.get("compute_seconds", {}).values()),
            "compute_seconds": tracking.get("compute_seconds", {}), "gpu_seconds": tracking.get("gpu_seconds", {}),
            "end_to_end_seconds": tracking.get("end_to_end_seconds"),
            "peak_gpu_memory_bytes": tracking.get("peak_gpu_memory_bytes"),
            "adjusted_nfe": adjusted, "adjusted_nfe_status": "available" if adjusted is not None else "unavailable",
            "adjusted_nfe_reason": None if adjusted is not None else "Requires measured FLOPs for every call and a reference forward unit",
            "modal_queue_startup_rpc_seconds": None, "raw_nfe_includes_inner": True,
            "inner_cost_is_subset_not_added_twice": True, "raw_nfe_excludes_verifier": True}


def generation_metrics(attempts: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]],
                       candidates: Sequence[Mapping[str, Any]], *, tracking: Mapping[str, Any],
                       protocol: Mapping[str, Any], prompts: set[str], fingerprint: dict,
                       retro: RetroStarProxy, reference_forward_flops: float | None = None) -> dict[str, Any]:
    outputs = [row for row in attempts if row["kind"] == "complete"]
    roots = tracking["counters"].get("root_attempts", 0)
    if roots < len(attempts):
        raise ValueError("Root count is smaller than recorded attempts")
    attained, target = len(samples), protocol["target_unique"]
    costs = cost_metrics(tracking, reference_forward_flops=reference_forward_flops)
    return {"success_per_root": ratio(sum(row["reward_result"]["reward"] == 1 for row in outputs), roots),
            "selected_per_root": ratio(attained, roots), "root_attempts": roots,
            "recorded_attempts": len(attempts), "unrecorded_root_attempts": roots-len(attempts),
            "success_is_lower_bound": roots > len(attempts),
            "prefix_aborts": sum(row["kind"] == "prefix_abort" for row in attempts),
            "attained": attained, "target": target, "target_attained": attained >= target,
            "missing_to_target": max(0, target-attained),
            "attempts_to_target": samples[target-1]["attempt_id"] if attained >= target else None,
            "tracked_seconds_to_target_upper_bound": costs["tracked_compute_seconds"] if attained >= target else None,
            "time_note": "Final recorded generation compute bounds time to target; no extrapolation for unattained targets",
            "sets": {name: molecule_metrics(rows, prompts, **fingerprint, retro=retro)
                     for name, rows in (("outputs", outputs), ("search_candidates", candidates), ("selected", samples))},
            "set_notes": {"outputs": "One returned complete output per root; failures and duplicates retained",
                          "search_candidates": "Every logged complete shortlist visit, including duplicate/memoized candidates; excludes inner completions",
                          "selected": "Conditional on reward and unique/prompt selection; validity is not efficiency evidence"},
            "cost": costs}
