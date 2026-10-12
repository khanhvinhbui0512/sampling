"""Export static figures from evaluator tables; no model or sampler imports."""

import os

from src.io import _atomic_writer


def create_plots(attempt, groups, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written, skipped = [], []

    def save(figure, name, description):
        figure.tight_layout()
        filename = "figures/" + name + ".png"
        os.makedirs(os.path.dirname(attempt.path(filename)), exist_ok=True)
        with _atomic_writer(attempt.path(filename), binary=True) as handle:
            figure.savefig(handle, format="png", dpi=180)
        plt.close(figure)
        attempt.register(filename)
        written.append({"path": filename, "description": description})

    def point(axis, x, y, x_interval, y_interval, label):
        artist = axis.scatter([x], [y], label=label)
        color = artist.get_facecolor()[0]
        if x_interval:
            axis.plot(x_interval, [y, y], color=color)
        if y_interval:
            axis.plot([x, x], y_interval, color=color)

    def frontier(kind, y_key, x_keys, filename):
        eligible = [group for group in groups if group["kind"] in kind and not group["duplicate_seed_attempts"]
                    and group["metrics"][y_key]["mean"] is not None]
        families = sorted({group["family_id"] for group in eligible})
        if not families:
            skipped.append({"plot": filename, "reason": "No unambiguous usable groups with defined quality/error"})
            return
        figure, axes = plt.subplots(len(families), len(x_keys), figsize=(7 * len(x_keys), 5 * len(families)), squeeze=False)
        for row, family in enumerate(families):
            for column, (x_key, x_label) in enumerate(x_keys):
                axis = axes[row, column]
                count = 0
                for group in eligible:
                    if group["family_id"] != family:
                        continue
                    x, y = group["metrics"][x_key], group["metrics"][y_key]
                    if x["mean"] is None:
                        continue
                    y_interval = y["seed_interval"]
                    suffix = "seed CI" if y_interval else "no seed CI"
                    if len(group["manifests"]) == 1 and group["kind"] == "values_qm9":
                        record = next(item for item in records if item["config_id"] == group["config_id"])
                        cluster = record["metrics"]["common_pair_cluster_interval"]
                        if cluster:
                            y_interval = cluster["rmse"]
                            suffix = "rollout-cluster CI; noisy MC reference"
                    label = group["label"] + " [" + group["config_id"][:10] + "] " + suffix
                    if group["kind"] != "generation":
                        label += f"; available={group['metrics']['availability']['mean']:.3f}"
                        label += f"; common pairs={group['metrics']['common_pair_states']['mean']:g}"
                    point(axis, x["mean"], y["mean"], x["seed_interval"], y_interval, label)
                    count += 1
                axis.set_xlabel(x_label)
                axis.set_ylabel("Pass@95 (fraction of requested outputs)" if y_key == "pass95" else "RMSE vs stated reference (common available states in cohort)")
                family_kind = next(group["kind"] for group in eligible if group["family_id"] == family)
                axis.set_title(f"Family {family[:12]} ({family_kind})")
                axis.grid(alpha=0.2)
                if count:
                    axis.legend(fontsize=6)
        save(figure, filename, "Separate axes per compatible family; bars are explicitly labelled uncertainty, no fitted frontier")

    frontier(("generation",), "pass95", (("base_sequences_per_output", "Base sequence forwards / requested output"),
                                         ("wall_seconds_per_output", "Sampling wall seconds / requested output (setup excluded)")), "quality_cost")
    frontier(("values_qm9", "values_toy"), "common_pair_rmse", (("base_sequences_per_query", "Base sequence forwards / requested state query"),
                                                     ("wall_seconds_per_query", "Query wall seconds / requested state (setup excluded)")), "value_error_cost")
    cached = [record for record in records if record["cache"]["queries"] > 0
              and (record.get("estimator") == "cars" or record["protocol"].get("verifier") == "cars")]
    if cached:
        figure, axes = plt.subplots(1, 2, figsize=(15, max(4, len(cached) * 0.6)))
        labels, current, historical, wall = [], [], [], []
        for record in cached:
            cache = record["cache"]
            queries = cache["queries"]
            labels.append(record["label"] + " [" + record["config_id"][:8] + "] family=" + record["family_id"][:8] + " seed=" + str(record["seed"]) + "; " + record["attempt_id"][:8])
            current.append(cache["current_query_cost"]["base_sequences"] / queries if queries else 0)
            historical.append(cache["unique_snapshot_origin_cost"]["base_sequences"] / queries if queries else 0)
            wall.append(cache["query_wall_seconds"] / queries if queries else 0)
        positions = list(range(len(cached)))
        axes[0].barh([x-0.18 for x in positions], current, height=0.35, label="Actual current query work")
        axes[0].barh([x+0.18 for x in positions], historical, height=0.35, label="Unique snapshot origin work / queries (historical; overlaps cold work)")
        axes[0].set_xlabel("Base sequence forwards / recorded state query; bars are not additive")
        axes[0].legend(fontsize=7)
        axes[1].barh(positions, wall)
        axes[1].set_xlabel("Current query wall seconds / recorded state query")
        for axis in axes:
            axis.set_yticks(positions, labels, fontsize=6)
            axis.grid(axis="x", alpha=0.2)
        save(figure, "cache_cost", "Per-attempt cache accounting, no seed uncertainty; setup includes warm snapshot loading and is reported separately")
    else:
        skipped.append({"plot": "cache_cost", "reason": "No CARS-guided generation or CARS value-check records"})
    return {"written": written, "skipped": skipped}
