"""Seed-level summaries, paired uncertainty and observed quality/cost frontiers."""

from __future__ import annotations

import csv
import io
import itertools
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Sequence

from src.artifacts import _atomic_file


def seed_interval(values: Sequence[float], *, repetitions: int, confidence: float, seed: int) -> dict:
    if not values:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    mean = math.fsum(values)/len(values)
    if len(values) == 1:
        return {"n": 1, "mean": mean, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    means = sorted(math.fsum(rng.choices(values, k=len(values)))/len(values) for _ in range(repetitions))
    tail = (1-confidence)/2
    return {"n": len(values), "mean": mean,
            "ci_low": means[int(tail*(repetitions-1))], "ci_high": means[int((1-tail)*(repetitions-1))]}


def generation_row(payload: dict, path: str) -> dict:
    identity, metrics = payload["identity"], payload["metrics"]
    outputs = metrics["sets"]["outputs"]
    row = {"suite": identity["suite"], "group": identity["group"], "seed": identity["seed"],
           "method": identity["method"], "run_id": identity["run_id"], "source_run": payload["source_run"],
           "source_status": payload["source_status"], "metrics_status": payload["status"], "metrics_path": path,
           "checkpoint": payload["comparability"]["backbone"], "config_hash": identity["config_hash"],
           "comparison_key": payload["comparison_key"], "budget_key": payload["budget_key"],
           "variant_key": payload["variant_key"], **payload["budget"],
           "target": metrics["target"], "target_attained": metrics["target_attained"],
           "attained": metrics["attained"], "missing_to_target": metrics["missing_to_target"],
           "root_attempts": metrics["root_attempts"], "recorded_attempts": metrics["recorded_attempts"],
           "unrecorded_root_attempts": metrics["unrecorded_root_attempts"],
           "complete_outputs": outputs["count"], "valid_molecules": outputs["valid_molecules"],
           "unique_molecules": outputs["unique_molecules"], "prefix_aborts": metrics["prefix_aborts"],
           "success_per_root": metrics["success_per_root"]["value"],
           "success_numerator": metrics["success_per_root"]["numerator"],
           "success_denominator": metrics["success_per_root"]["denominator"],
           "attempts_to_target": metrics["attempts_to_target"],
           "tracked_seconds_to_target_upper_bound": metrics["tracked_seconds_to_target_upper_bound"]}
    for name in ("validity", "membership", "unique_fraction", "novelty", "diversity", "retro_score"):
        block = outputs[name]
        row[name] = block["value"]
        row[name+"_denominator"] = block.get("denominator", block.get("pair_denominator"))
        row[name+"_status"] = block["status"]
    row["selected_diversity"] = metrics["sets"]["selected"]["diversity"]["value"]
    row["radius3_diversity"] = outputs["diversity_by_radius"]["3"]["value"]
    for name, value in metrics["cost"].items():
        if not isinstance(value, dict):
            row[name] = value
    return row


def summarize_generation(rows: Sequence[dict], metrics: Sequence[str], bootstrap: dict) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in ("suite", "comparison_key", "group", "method", "variant_key", "budget_key"))
        buckets[key].append(row)
    summary = []
    for key, members in sorted(buckets.items()):
        seeds = [row["seed"] for row in members]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate method/group/budget/variant/seed: choose one run explicitly")
        context = dict(zip(("suite", "comparison_key", "group", "method", "variant_key", "budget_key"), key, strict=True))
        for name in metrics:
            values = [row[name] for row in members if isinstance(row.get(name), (int, float)) and not isinstance(row[name], bool)]
            summary.append({**context, "metric": name, **seed_interval(values, **bootstrap),
                            "runs": len(members), "unavailable_runs": len(members)-len(values),
                            "unattained_runs": sum(not row["target_attained"] for row in members),
                            "noncompleted_runs": sum(row["source_status"] != "completed" for row in members)})
    return summary


def paired_comparisons(rows: Sequence[dict], metrics: Sequence[str], bootstrap: dict) -> list[dict]:
    buckets = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = tuple(row[name] for name in ("suite", "comparison_key", "group", "budget_key"))
        arm = (row["method"], row["variant_key"])
        if row["seed"] in buckets[key][arm]:
            raise ValueError("Duplicate paired seed")
        buckets[key][arm][row["seed"]] = row
    comparisons = []
    for key, arms in sorted(buckets.items()):
        for left, right in itertools.combinations(sorted(arms), 2):
            if left[0] == right[0]:
                continue
            shared = sorted(arms[left].keys() & arms[right].keys())
            for metric in metrics:
                pairs = [(arms[left][seed].get(metric), arms[right][seed].get(metric)) for seed in shared]
                differences = [b-a for a, b in pairs if a is not None and b is not None]
                comparisons.append({**dict(zip(("suite", "comparison_key", "group", "budget_key"), key, strict=True)),
                    "left": left[0], "left_variant": left[1], "right": right[0], "right_variant": right[1],
                    "metric": metric, "direction": "right_minus_left", **seed_interval(differences, **bootstrap),
                    "matched_seeds": len(shared), "left_only_seeds": len(arms[left].keys()-arms[right].keys()),
                    "right_only_seeds": len(arms[right].keys()-arms[left].keys()),
                    "unavailable_pairs": len(shared)-len(differences), "equal_actual_compute_claim": False})
    return comparisons


def quality_frontiers(aggregates: Sequence[dict]) -> list[dict]:
    points = defaultdict(dict)
    for row in aggregates:
        key = tuple(row[name] for name in ("suite", "comparison_key", "group", "method", "variant_key", "budget_key"))
        points[key][row["metric"]] = row
    buckets = defaultdict(list)
    for key, metrics in points.items():
        quality = metrics.get("attained", {})
        for cost in ("raw_nfe", "tracked_compute_seconds"):
            expense = metrics.get(cost, {})
            if quality.get("mean") is None or expense.get("mean") is None:
                continue
            point = {**dict(zip(("suite", "comparison_key", "group", "method", "variant_key", "budget_key"), key, strict=True)),
                     "cost_metric": cost, "cost": expense["mean"], "quality": quality["mean"],
                     "quality_ci_low": quality["ci_low"], "quality_ci_high": quality["ci_high"],
                     "seeds": quality["n"], "unattained_runs": quality["unattained_runs"]}
            # Different caps may form one frontier; backbone/protocol/group cannot.
            # Compare observed variants within a method without averaging them.
            buckets[(key[0], key[1], key[2], key[3], cost)].append(point)
    output = []
    for points in buckets.values():
        for point in points:
            dominated = any(other["cost"] <= point["cost"] and other["quality"] >= point["quality"]
                            and (other["cost"] < point["cost"] or other["quality"] > point["quality"]) for other in points)
            output.append({**point, "pareto": not dominated, "observed_only": True})
    return output


def value_rows(payload: dict, path: str) -> list[dict]:
    output = []
    for result in payload["evaluation"]["results"]:
        slices = [("overall", "all", result["overall"]), ("terminal", "all", result["terminal"]),
                  ("nonterminal", "all", result["nonterminal"])]
        slices.extend(("group", group, block) for group, block in result["groups"].items())
        slices.extend(("mask_fraction", f"{block['lower']}:{block['upper']}", block) for block in result["mask_fraction"])
        for subset, label, block in slices:
            row = {"suite": payload["identity"]["suite"], "seed": payload["identity"]["seed"],
                   "run_id": payload["identity"]["run_id"], "metrics_path": path, "status": payload["status"],
                   "config_hash": payload["identity"]["config_hash"], "purpose": payload["purpose"],
                   "checkpoint": payload["identity"]["dependency_hashes"].get("mdlm"),
                   "provider": result["provider"], "budget": result["budget"],
                   "reference_policy": result["reference_policy"], "interpretation": result["interpretation"],
                   "training_or_estimator_target_aligned": result["training_or_estimator_target_aligned"],
                   "subset": subset, "subset_label": label}
            row.update({name: value for name, value in block.items() if not isinstance(value, (dict, list))})
            row["reliability_bins_json"] = json.dumps(block["reliability_bins"], sort_keys=True, allow_nan=False)
            row.update({"cost_"+name: value for name, value in block["cost_including_failed_queries"].items()
                        if not isinstance(value, (dict, list))})
            output.append(row)
    return output


def write_csv(store, path: str, rows: Sequence[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    store.write_text(path, buffer.getvalue())


def write_plots(store, directory: str, files: dict, frontiers: Sequence[dict], values: Sequence[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, cost in zip(axes, ("raw_nfe", "tracked_compute_seconds"), strict=True):
        buckets = defaultdict(list)
        for point in frontiers:
            if point["cost_metric"] == cost:
                key = (point["suite"], point["comparison_key"], point["group"], point["method"], point["variant_key"])
                buckets[key].append(point)
        for key, points in sorted(buckets.items()):
            label = f"{key[0]}/{key[2]}/{key[3]} {key[1][:6]}/{key[4][:6]}"
            axis.scatter([point["cost"] for point in points], [point["quality"] for point in points], label=label)
        axis.set(xlabel=cost, ylabel="Mean attained unique successes across seeds")
        if buckets:
            axis.legend(fontsize=6)
        else:
            axis.text(0.5, 0.5, "No checked generation metrics", ha="center", transform=axis.transAxes)
    with _atomic_file(store.path(os.path.join(directory, files["quality_plot"]))) as handle:
        figure.savefig(handle, format="png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    plotted = [row for row in values if row["subset"] == "nonterminal" and row["mse_to_mc_reference"] is not None]
    for index, row in enumerate(plotted):
        label = f"{row['suite']}/{row['purpose']}/{row['provider']}/{row['reference_policy']} seed={row['seed']} {row['checkpoint'][:6]} {row['run_id']}"
        axis.scatter(index, row["mse_to_mc_reference"], label=label)
    axis.set(xlabel="Separate checked evaluations (no pooling)", ylabel="Nonterminal MSE to finite MC reference")
    if plotted:
        axis.legend(fontsize=6)
    else:
        axis.text(0.5, 0.5, "No checked value metrics", ha="center", transform=axis.transAxes)
    with _atomic_file(store.path(os.path.join(directory, files["value_plot"]))) as handle:
        figure.savefig(handle, format="png", dpi=160)
    plt.close(figure)
