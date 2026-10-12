"""Lock a QED percentile from independent, unfiltered BASE-native draws."""

import platform
from dataclasses import asdict
from importlib.metadata import version

import numpy as np
from tqdm.auto import tqdm

from src.budget import BudgetTracker
from src.config import load_config, project_path
from src.io import (
    RunArtifacts,
    file_hash,
    payload_hash,
    read_completed_manifest,
    write_json,
    write_jsonl,
    write_yaml,
)
from src.models.backbone import QM9Backbone
from src.qm9.chemistry import rdkit_version
from src.qm9.dataset import load_prepared_qm9
from src.qm9.reward import BinaryCalibration, QM9Reward, TerminalStatus
from src.randomness import RNGPool
from src.sampling.base import BaseSampler
from src.types import CostPhase, MaskedState, require_count


def scores_from_terminals(terminals):
    scores = []
    for terminal in terminals:
        if terminal is None or terminal.status == TerminalStatus.INCOMPLETE or terminal.score is None:
            raise ValueError("Calibration requires complete draws; do not drop or retry unfinished samples")
        scores.append(float(terminal.score))  # Includes invalid SMILES/tokens with R=0.
    return scores


def threshold_statistics(scores, *, bootstrap_rng, quantile=0.95, method="linear",
                         replicates=1000, confidence=0.95):
    values = np.asarray(scores, dtype=np.float64)
    if (values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)):
        raise ValueError("Calibration needs nonempty finite QED-or-zero scores")
    if quantile != 0.95 or method != "linear":
        raise ValueError("The QM9 calibration protocol fixes quantile=0.95 and method=linear")
    if type(replicates) is not int or replicates < 2 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap replicates or confidence")
    threshold = float(np.quantile(values, quantile, method=method))
    bootstrapped = np.empty(replicates, dtype=np.float64)
    for i in tqdm(range(replicates), desc="Calibration uncertainty", unit="bootstrap"):
        resampled = bootstrap_rng.choice(values, size=len(values), replace=True)
        bootstrapped[i] = np.quantile(resampled, quantile, method=method)
    tail = (1 - confidence) / 2
    interval = np.quantile(bootstrapped, [tail, 1 - tail], method=method)
    return {"threshold": threshold, "quantile": quantile, "quantile_method": method,
            "sample_count": len(values), "uncertainty": {
                "method": "nonparametric_percentile_bootstrap", "replicates": replicates,
                "confidence": confidence, "interval": [float(value) for value in interval],
                "standard_error": float(bootstrapped.std(ddof=1)),
                "interpretation": "diagnostic; finite-sample coverage is not guaranteed",
            }}


def calibrate_qm9(assets_manifest, checkpoint_manifest, *, attempt_id,
                  config_path="configs/reference_checks.yaml", calibration_id="qm9-base-pass95",
                  device="cuda", execution_metadata=None):
    assets_manifest, checkpoint_manifest = project_path(assets_manifest), project_path(checkpoint_manifest)
    config = load_config(config_path)
    qm9 = config["qm9"]
    settings = qm9["calibration"]
    if settings["sampler"] != "base_native" or settings["zero_threshold_policy"] != "require_valid_molecule":
        raise ValueError("QM9 shared calibration uses BASE-native and a valid-molecule success gate")
    for key in ("sample_count", "batch_size"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"Calibration {key} must be a positive integer")
    require_count(settings["seed"], "calibration seed")
    if settings["quantile"] != 0.95 or settings["quantile_method"] != "linear":
        raise ValueError("QM9 calibration requires quantile=0.95 and method=linear")
    if (type(settings["bootstrap_replicates"]) is not int or settings["bootstrap_replicates"] < 2
            or not 0 < settings["confidence"] < 1):
        raise ValueError("Invalid calibration uncertainty settings")
    assets = load_prepared_qm9(assets_manifest)
    checkpoint = read_completed_manifest(checkpoint_manifest)
    if checkpoint.get("identities", {}).get("assets_manifest_sha256") != file_hash(assets_manifest):
        raise ValueError("Checkpoint was trained with a different QM9 assets manifest")
    backbone = QM9Backbone.from_checkpoint(checkpoint_manifest, device=device, precision="bf16")
    if backbone.token_rules != assets.token_rules or backbone.tokenizer_id != assets.tokenizer_sha256:
        raise ValueError("Checkpoint tokenizer differs from the prepared QM9 tokenizer")
    backbone.model.requires_grad_(False)
    reward = QM9Reward(tokenizer=backbone.tokenizer, token_rules=assets.token_rules,
                       chemistry_policy=assets.chemistry_policy, tokenizer_sha256=assets.tokenizer_sha256,
                       terminal_policy=assets.terminal_policy)
    sampler = BaseSampler(backbone, reward, native_settings=qm9["native"])
    attempt = RunArtifacts.create(settings["output_dir"], run_id=calibration_id, attempt_id=attempt_id,
                                  config=config, target_id="qm9-pass95-calibration",
                                  reference_id=sampler.reference_id)
    attempt.manifest["task"] = "calibrate_qm9"
    pool = RNGPool(settings["seed"], "qm9-base-calibration-v1")
    sample_rng, bootstrap_rng = pool.stream("samples"), pool.stream("threshold-bootstrap")
    budget = BudgetTracker(base_sequence_limit=None)
    scores, valid = [], []
    try:
        write_yaml(attempt.path("config.yaml"), config)
        source_files = ("src/sampling/native.py", "src/sampling/completion.py", "src/sampling/base.py",
                        "src/qm9/calibration.py", "src/qm9/reward.py", "src/qm9/chemistry.py",
                        "src/models/backbone.py", "src/randomness.py", "src/budget.py", "src/io.py")
        write_json(attempt.path("environment.json"), {
            "python": platform.python_version(), "rdkit_version": rdkit_version(),
            "packages": {name: version(name) for name in ("torch", "numpy", "transformers", "rdkit")},
            "source_sha256": {name: file_hash(name) for name in source_files},
            "config_file_sha256": file_hash(config_path), "precision": backbone.precision,
        })
        protocol = {
            "purpose": "threshold_calibration_only", "no_dataset_rows_used": True,
            "sampler": sampler.method_id, "policy_id": sampler.policy_id,
            "reference_id": sampler.reference_id, "native": sampler.generator.settings,
            "assets_manifest": assets_manifest, "assets_manifest_sha256": file_hash(assets_manifest),
            "checkpoint_manifest": checkpoint_manifest, "checkpoint_sha256": backbone.checkpoint_id,
            "tokenizer_sha256": backbone.tokenizer_id, "seed": settings["seed"],
            "sample_count": settings["sample_count"], "batch_size": settings["batch_size"],
            "rng_namespace": pool.run_id, "sample_stream_seed": sample_rng.seed,
            "bootstrap_stream_seed": bootstrap_rng.seed,
            "binary_definition": "complete_valid_and_qed_ge_threshold",
            "zero_threshold_policy": settings["zero_threshold_policy"],
            "filter_or_deduplicate": False, "execution": execution_metadata or {"backend": "local"},
        }
        write_json(attempt.path("sampling_protocol.json"), protocol)
        for name in ("config.yaml", "environment.json", "sampling_protocol.json"):
            attempt.register(name)
        root = MaskedState((assets.token_rules.mask_id,) * assets.token_rules.sequence_length,
                           assets.token_rules)

        def rows():
            with tqdm(total=settings["sample_count"], desc="BASE calibration", unit="molecule") as progress:
                for start in range(0, settings["sample_count"], settings["batch_size"]):
                    count = min(settings["batch_size"], settings["sample_count"] - start)
                    batch = sampler.generate_batch([root] * count, rng=sample_rng, budget=budget,
                                                     phase=CostPhase.CALIBRATION)
                    if len(batch.states) != count or len(batch.terminals) != count:
                        raise ValueError("BASE returned a different number of calibration draws")
                    batch_scores = scores_from_terminals(batch.terminals)
                    scores.extend(batch_scores)
                    for offset, (state, terminal) in enumerate(zip(batch.states, batch.terminals)):
                        valid.append(bool(terminal.valid))
                        molecule = terminal.molecule
                        yield {"sample_id": f"calibration-{start + offset:07d}", "purpose": "calibration",
                               "tokens": state.token_ids, "status": terminal.status,
                               "valid": terminal.valid, "score": terminal.score, "reward": None,
                               "canonical_smiles": molecule.canonical_smiles if molecule else None,
                               "smiles": molecule.smiles if molecule else None}
                    progress.update(count)

        write_jsonl(attempt.path("samples.jsonl"), rows(), progress=False)
        if len(scores) != settings["sample_count"]:
            raise ValueError("Calibration sample count differs from the locked configuration")
        statistics = threshold_statistics(scores, bootstrap_rng=bootstrap_rng.numpy,
                                           quantile=settings["quantile"], method=settings["quantile_method"],
                                           replicates=settings["bootstrap_replicates"], confidence=settings["confidence"])
        locked_id = "qm9-pass95:" + payload_hash({
            "protocol": protocol, "scores_sha256": file_hash(attempt.path("samples.jsonl")),
            "quantile": settings["quantile"], "method": settings["quantile_method"],
            "threshold": statistics["threshold"], "chemistry": asdict(assets.chemistry_policy),
        })
        calibration = BinaryCalibration(
            calibration_id=locked_id, reference_id=sampler.reference_id, policy_id=sampler.policy_id,
            checkpoint_sha256=backbone.checkpoint_id, tokenizer_sha256=backbone.tokenizer_id,
            threshold=statistics["threshold"], quantile=settings["quantile"], sample_count=len(scores),
            seed=settings["seed"], quantile_method=settings["quantile_method"],
            chemistry_policy=asdict(assets.chemistry_policy), rdkit_version=rdkit_version(),
            threshold_source="independent_base_calibration", score_name="qed_or_zero",
            terminal_policy=assets.terminal_policy,
        )
        write_json(attempt.path("calibration.json"), calibration)
        write_json(attempt.path("uncertainty.json"), statistics["uncertainty"])
        values = np.asarray(scores)
        summary = {"manifest": attempt.path("manifest.json"), "calibration_id": locked_id,
                   **statistics, "valid_count": sum(valid), "invalid_count": len(scores) - sum(valid),
                   "unfinished_count": 0, "zero_threshold_policy": settings["zero_threshold_policy"],
                   "calibration_pass_fraction": float(np.mean(np.asarray(valid) & (values >= calibration.threshold))),
                   "cost": asdict(budget.snapshot())}
        write_json(attempt.path("summary.json"), summary)
        write_json(attempt.path("rng.json"), pool.state_dict())
        required = ["samples.jsonl", "calibration.json", "uncertainty.json", "summary.json", "rng.json"]
        for name in required:
            attempt.register(name)
        attempt.manifest["calibration_id"] = locked_id
        attempt.manifest["policy_id"] = sampler.policy_id
        attempt.complete(required_artifacts=required)
        return summary
    except Exception as exc:
        if attempt.manifest["status"] == "running":
            try:
                attempt.fail(f"{type(exc).__name__}: {exc}")
            except Exception as manifest_error:
                exc.add_note(f"Could not save calibration failure manifest: {manifest_error}")
        raise
