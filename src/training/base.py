"""Single-GPU QM9 MDLM training and immutable, resumable checkpoints.

The objective, batch sampling and learning-rate schedule are adapted from
MDM-VGB's QM9 train_base_model.py (MIT, copyright 2026 Kijung Jeon).
See THIRD_PARTY_NOTICES.md. Checkpoint/resume handling is added here.
"""

import json
import math
import os
import platform
import random
from dataclasses import asdict
from importlib.metadata import version

import datasets
import numpy as np
import torch
from tqdm.auto import tqdm

from src.budget import BudgetTracker
from src.config import load_config, project_path
from src.io import (
    RunArtifacts,
    file_hash,
    payload_hash,
    read_checkpoint,
    read_completed_manifest,
    read_json,
    report_result,
    write_checkpoint,
    write_json,
    write_yaml,
)
from src.models.backbone import QM9Backbone, loglinear_sigma, subs_log_probs
from src.qm9.dataset import load_prepared_qm9
from src.randomness import capture_training_rng, restore_training_rng
from src.types import CostPhase


def build_scheduler(optimizer, settings):
    lr, min_lr = settings["lr"], settings["min_lr"]
    warmup, steps = settings["warmup_steps"], settings["train_steps"]

    def multiplier(step):
        if warmup > 0 and step < warmup:
            return max(1e-8, step / warmup)
        progress = min(1.0, max(0.0, (step - warmup) / max(1, steps - warmup)))
        return (min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))) / lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def mdlm_loss(backbone, clean_tokens, *, budget, time_min=0.001, noise_eps=0.001):
    """Upstream's masked SUBS loss averaged over all 32 positions, including PAD."""
    t = torch.rand(clean_tokens.shape[0], device=clean_tokens.device).clamp_min(time_min)
    sigma = loglinear_sigma(t, noise_eps)
    corrupted = torch.rand_like(clean_tokens.float()) < (1 - torch.exp(-sigma[:, None]))
    tokens = torch.where(corrupted, backbone.token_rules.mask_id, clean_tokens)
    logits = backbone.forward_tokens(tokens, sigma, budget=budget, phase=CostPhase.TRAINING)
    log_probs = subs_log_probs(logits, tokens, backbone.token_rules.mask_id)
    token_loss = -log_probs.gather(-1, clean_tokens[:, :, None]).squeeze(-1)
    return (token_loss / t[:, None]).mean()


def training_environment(backbone):
    source_files = (
        "src/models/configuration_udlm.py", "src/models/modeling_udlm.py",
        "src/models/backbone.py", "src/training/base.py", "src/randomness.py",
        "src/config.py", "src/types.py", "src/budget.py", "src/io.py", "src/qm9/dataset.py",
    )
    return {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in
                     ("torch", "transformers", "tokenizers", "datasets", "numpy", "safetensors")},
        "cuda": torch.version.cuda,
        "device": str(backbone.device), "precision": backbone.precision,
        "gpu": torch.cuda.get_device_name(backbone.device) if backbone.device.type == "cuda" else None,
        "torch_threads": torch.get_num_threads(),
        "source_sha256": {name: file_hash(name) for name in source_files},
    }


def save_training_checkpoint(directory, backbone, optimizer, scheduler, *, step,
                             config, identities, environment, cumulative_base_calls):
    """Write the completed marker last; partial directories cannot be loaded."""
    directory = project_path(directory)
    os.makedirs(directory, exist_ok=False)
    rng = capture_training_rng(include_cuda=backbone.device.type == "cuda")
    backbone.model.save_pretrained(directory, safe_serialization=True, max_shard_size="5GB")
    backbone.tokenizer.save_pretrained(os.path.join(directory, "tokenizer"))
    if not os.path.isfile(os.path.join(directory, "model.safetensors")):
        raise ValueError("QM9 checkpoints must have a single model.safetensors file")
    state = {
        "step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "rng": rng,
        "config": config, "identities": identities, "environment": environment,
        "cumulative_base_calls": cumulative_base_calls,
    }
    write_checkpoint(os.path.join(directory, "training_state.pt"), state,
                     metadata={"step": step, **identities})
    records = {}
    for parent, directories, files in os.walk(directory):
        directories.sort()
        for filename in sorted(files):
            path = os.path.join(parent, filename)
            records[os.path.relpath(path, directory)] = {
                "sha256": file_hash(path), "size_bytes": os.path.getsize(path),
            }
    manifest = {
        "schema_version": 1, "status": "completed", "task": "train_qm9_base_checkpoint",
        "step": step, "artifacts": records, "identities": identities,
        "config_sha256": payload_hash(config), "token_rules": asdict(backbone.token_rules),
        "tokenizer_sha256": backbone.tokenizer_id,
        "vocabulary_sha256": backbone.vocabulary_sha256,
        "checkpoint_sha256": records["model.safetensors"]["sha256"],
        "parameter_count": sum(parameter.numel() for parameter in backbone.model.parameters()),
    }
    manifest_path = os.path.join(directory, "manifest.json")
    write_json(manifest_path, manifest)
    restore_training_rng(rng)  # Serialization must not advance the training streams.
    return manifest_path


def _validate_settings(config):
    if config["task"] != "qm9" or config["objective"] != "mdlm":
        raise ValueError("This trainer implements only QM9 MDLM")
    training = config["training"]
    if training["smiles_column"] != "canonical_smiles":
        raise ValueError("Upstream QM9 training uses canonical_smiles")
    if config["model"]["time_conditioning"] is not False:
        raise ValueError("The upstream QM9 MDLM config uses time_conditioning=False")
    for name in ("batch_size", "train_steps", "save_every", "log_every"):
        if type(training[name]) is not int or training[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(training["seed"]) is not int or training["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not 0 <= training["warmup_steps"] < training["train_steps"]:
        raise ValueError("warmup_steps must be smaller than train_steps")
    if not 0 < training["min_lr"] <= training["lr"]:
        raise ValueError("Require 0 < min_lr <= lr")
    if not 0 < training["time_min"] < 1 or not 0 < training["noise_eps"] < 1:
        raise ValueError("time_min and noise_eps must be in (0, 1)")
    if training["clip_grad_norm"] <= 0 or training["weight_decay"] < 0:
        raise ValueError("Invalid gradient clipping or weight decay")
    project_path(config["output_dir"])


def train_base(assets_manifest, *, config_path="configs/base_training.yaml",
               run_id="mdlm-qm9", attempt_id, device="cuda", resume_manifest=None,
               on_checkpoint=None, execution_metadata=None):
    config_path = project_path(config_path)
    assets_manifest = project_path(assets_manifest)
    config = load_config(config_path)
    _validate_settings(config)
    settings = config["training"]
    assets = load_prepared_qm9(assets_manifest)
    identities = {
        "assets_manifest_sha256": file_hash(assets_manifest),
        "splits_sha256": file_hash(assets.splits_path),
        "tokenizer_sha256": assets.tokenizer_sha256,
        "config_file_sha256": file_hash(config_path),
    }
    attempt = RunArtifacts.create(config["output_dir"], run_id=run_id, attempt_id=attempt_id,
                                  config=config, target_id="qm9-mdlm-training",
                                  reference_id="qm9-assets:" + identities["assets_manifest_sha256"])
    attempt.manifest["task"] = "train_qm9_base"
    try:
        write_yaml(attempt.path("config.yaml"), config)
        write_json(attempt.path("execution.json"), {
            "assets_manifest": assets_manifest, "resume_manifest": resume_manifest,
            "backend": "local", **(execution_metadata or {}),
        })
        # Direct seed values match upstream; do not derive a different seed stream.
        seed = settings["seed"]
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if resume_manifest:
            resume_manifest = project_path(resume_manifest)
            saved = read_completed_manifest(resume_manifest)
            if saved.get("identities") != identities or saved.get("config_sha256") != payload_hash(config):
                raise ValueError("Resume requires the same dataset, tokenizer and training config")
            backbone = QM9Backbone.from_checkpoint(resume_manifest, device=device,
                                                    precision=settings["precision"])
        else:
            backbone = QM9Backbone.from_assets(assets, config["model"], device=device,
                                               precision=settings["precision"])
        environment = training_environment(backbone)
        write_json(attempt.path("environment.json"), environment)
        for name in ("config.yaml", "execution.json", "environment.json"):
            attempt.register(name)
        optimizer = torch.optim.AdamW(backbone.model.parameters(), lr=settings["lr"],
                                      weight_decay=settings["weight_decay"],
                                      betas=(0.9, 0.999), eps=1e-8)
        scheduler = build_scheduler(optimizer, settings)
        start, previous_calls = 0, 0
        raw = datasets.load_from_disk(assets.raw_directory)
        if settings["smiles_column"] not in raw.column_names:
            raise ValueError("Prepared dataset has no canonical_smiles column")
        train_ids = read_json(assets.splits_path)["train_row_ids"]
        smiles = list(raw.select(train_ids)[settings["smiles_column"]])
        if len(smiles) < settings["batch_size"] or not all(isinstance(text, str) for text in smiles):
            raise ValueError("Insufficient or malformed QM9 training rows")
        if resume_manifest:
            state_path = os.path.join(os.path.dirname(resume_manifest), "training_state.pt")
            state = read_checkpoint(state_path)
            if (state["identities"] != identities or state["config"] != config
                    or state["environment"] != environment or state["step"] != saved["step"]):
                raise ValueError("Resume state/config/software/device differs from the checkpoint")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start, previous_calls = state["step"], state["cumulative_base_calls"]
            if not 0 < start < settings["train_steps"]:
                raise ValueError("Checkpoint already finished or has an invalid training step")
            # Restore last: model construction and setup must not consume resumed RNG.
            restore_training_rng(state["rng"])
        budget = BudgetTracker(base_sequence_limit=None)
        backbone.model.train()
        report_result(f"QM9 training attempt: {attempt.directory}; start={start}; "
                      f"precision={backbone.precision}")
        last_checkpoint = None
        with tqdm(range(start + 1, settings["train_steps"] + 1), initial=start,
                  total=settings["train_steps"], desc="Train QM9 MDLM", unit="step") as progress:
            for step in progress:
                batch = random.sample(smiles, settings["batch_size"])
                tokens = backbone.tokenizer(batch, return_tensors="pt", padding="max_length",
                                            truncation=True, max_length=assets.token_rules.sequence_length,
                                            add_special_tokens=True)["input_ids"].to(backbone.device)
                if (torch.any(tokens == assets.token_rules.mask_id)
                        or torch.any(tokens == assets.token_rules.unk_id)):
                    raise ValueError("QM9 training tokenization produced MASK/UNK")
                optimizer.zero_grad(set_to_none=True)
                loss = mdlm_loss(backbone, tokens, budget=budget,
                                 time_min=settings["time_min"], noise_eps=settings["noise_eps"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(backbone.model.parameters(),
                                                      settings["clip_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                record = {"step": step, "loss": float(loss.detach()), "grad_norm": float(norm),
                          "next_lr": optimizer.param_groups[0]["lr"],
                          "cumulative_base_calls": previous_calls + budget.raw_nfe}
                # Close the log before checkpoint commits on the Modal Volume.
                with open(attempt.path("metrics.jsonl"), "a", encoding="utf-8") as metrics:
                    metrics.write(json.dumps(record, allow_nan=False) + "\n")
                if step == 1 or step % settings["log_every"] == 0:
                    progress.set_postfix(loss=f"{record['loss']:.4f}", lr=record["next_lr"])
                if step % settings["save_every"] == 0 or step == settings["train_steps"]:
                    rng = capture_training_rng(include_cuda=backbone.device.type == "cuda")
                    try:
                        last_checkpoint = save_training_checkpoint(
                            attempt.path(f"checkpoints/step-{step:07d}"), backbone, optimizer, scheduler,
                            step=step, config=config, identities=identities, environment=environment,
                            cumulative_base_calls=previous_calls + budget.raw_nfe,
                        )
                        # Each completed checkpoint survives a later failed attempt.
                        if on_checkpoint is not None:
                            on_checkpoint()
                        report_result(f"Saved QM9 checkpoint: step={step}; manifest={last_checkpoint}")
                    finally:
                        # Cloud persistence/retries must not change the next batch/noise.
                        restore_training_rng(rng)
        final = read_completed_manifest(last_checkpoint)
        summary = {"manifest": attempt.path("manifest.json"), "checkpoint_manifest": last_checkpoint,
                   "checkpoint_sha256": final["checkpoint_sha256"], "step": settings["train_steps"],
                   "parameter_count": final["parameter_count"], "precision": backbone.precision,
                   "attempt_cost": asdict(budget.snapshot()),
                   "cumulative_base_calls": previous_calls + budget.raw_nfe}
        write_json(attempt.path("summary.json"), summary)
        required = []
        for parent, directories, files in os.walk(attempt.directory):
            directories.sort()
            for filename in sorted(files):
                relative = os.path.relpath(os.path.join(parent, filename), attempt.directory)
                if relative == "manifest.json" or relative in attempt.manifest["artifacts"]:
                    continue
                attempt.register(relative)
                required.append(relative)
        attempt.manifest["reference_id"] = "qm9-mdlm:" + final["checkpoint_sha256"]
        attempt.complete(required_artifacts=required)
        return summary
    except Exception as exc:
        if attempt.manifest["status"] == "running":
            try:
                attempt.fail(f"{type(exc).__name__}: {exc}")
            except Exception as manifest_error:
                exc.add_note(f"Could not save training failure manifest: {manifest_error}")
        raise
