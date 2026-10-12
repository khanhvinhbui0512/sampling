# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kijung Jeon

"""Single-GPU QM9 verifier MSE training, grouped validation and epoch resume.

Model/MSE/AdamW/clipping/epoch shuffling follow upstream qm9/train_verifier.py.
Validation, immutable checkpoints and complete RNG/optimizer resume are added.
"""

import os
import platform
import random
from dataclasses import asdict
from importlib.metadata import version

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

from src.budget import BudgetTracker
from src.config import load_config, project_path
from src.io import (RunArtifacts, file_hash, payload_hash, read_checkpoint,
                    read_completed_manifest, report_result, write_checkpoint,
                    write_json, write_yaml)
from src.qm9.rollouts import load_rollout_shards
from src.randomness import capture_training_rng, restore_training_rng
from src.types import CostPhase, CostRecord, TokenRules, ValueTarget, require_count
from src.verifiers.neural import TokenStateVerifier


def verifier_environment(device):
    return {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "numpy", "safetensors")},
        "device": str(device), "cuda": torch.version.cuda, "precision": "fp32",
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_threads": torch.get_num_threads(),
        "source_sha256": {name: file_hash(name) for name in (
            "src/verifiers/neural.py", "src/verifiers/interface.py", "src/training/verifier.py",
            "src/qm9/rollouts.py", "src/randomness.py", "src/budget.py", "src/io.py", "src/types.py")},
    }


def save_verifier_checkpoint(directory, model, optimizer, *, epoch, config, protocol,
                             identities, environment, history, cumulative_cost):
    directory = project_path(directory)
    os.makedirs(directory, exist_ok=False)
    device = next(model.parameters()).device
    rng = capture_training_rng(include_cuda=device.type == "cuda")
    metadata = {"model": model.settings, "target": protocol["target"],
                "token_rules": protocol["token_rules"], "output_activation": "sigmoid",
                "backbone_checkpoint_sha256": protocol["backbone_checkpoint_sha256"],
                "tokenizer_sha256": protocol["tokenizer_sha256"], "label_protocol": protocol["label_protocol"]}
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()},
              os.path.join(directory, "model.safetensors"))
    write_json(os.path.join(directory, "model.json"), metadata)
    write_checkpoint(os.path.join(directory, "training_state.pt"), {
        "epoch": epoch, "optimizer": optimizer.state_dict(), "rng": rng, "history": history,
        "config": config, "protocol": protocol, "identities": identities,
        "environment": environment, "cumulative_cost": asdict(cumulative_cost),
    }, metadata={"epoch": epoch, "target_id": protocol["target"]["target_id"]})
    names = ("model.safetensors", "model.json", "training_state.pt", "training_state.pt.metadata.json")
    records = {name: {"sha256": file_hash(os.path.join(directory, name)),
                      "size_bytes": os.path.getsize(os.path.join(directory, name))} for name in names}
    manifest_path = os.path.join(directory, "manifest.json")
    write_json(manifest_path, {
        "schema_version": 1, "status": "completed", "task": "train_qm9_verifier_checkpoint",
        "epoch": epoch, "artifacts": records, "target": protocol["target"],
        "checkpoint_sha256": records["model.safetensors"]["sha256"],
        "identities": identities, "config_sha256": payload_hash(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    })
    restore_training_rng(rng)
    return manifest_path


def verifier_forward(model, tokens, *, budget, device):
    with budget.verifier_forward(batch_size=len(tokens), phase=CostPhase.TRAINING):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values = model(tokens)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return values


@torch.no_grad()
def validation_mse(model, tokens, labels, *, batch_size, device, budget):
    model.eval()
    total = 0.
    for start in tqdm(range(0, len(tokens), batch_size), desc="Validate verifier", leave=False):
        x, y = tokens[start:start + batch_size].to(device), labels[start:start + batch_size].to(device)
        pred = verifier_forward(model, x, budget=budget, device=device)
        loss = torch.nn.functional.mse_loss(pred, y, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite verifier validation loss")
        total += float(loss)
    return total / len(tokens)


def train_verifier(rollout_manifests, *, policy, attempt_id, config_path="configs/verifier_training.yaml",
                   run_id="qm9-verifier", device="cuda", resume_manifest=None,
                   on_checkpoint=None, execution_metadata=None):
    if policy not in ("native", "rho", "native-conditional"):
        raise ValueError("Verifier policy must be native, rho or native-conditional")
    config_path = project_path(config_path)
    config = load_config(config_path)
    settings = config["training"]
    for key in ("epochs", "batch_size", "save_every_epochs"):
        require_count(settings[key], key, positive=True)
    require_count(settings["seed"], "verifier training seed")
    if not 0 < settings["lr"] or settings["weight_decay"] < 0 or settings["clip_grad_norm"] <= 0:
        raise ValueError("Invalid verifier optimization settings")
    expected_labels = {"rho": "q_rho_terminal_pass95", "native": "native_trajectory_terminal_pass95",
                       "native-conditional": "native_root_terminal_independent_mask_pass95"}
    if config["targets"][policy] != expected_labels[policy]:
        raise ValueError("Verifier configuration label target differs")
    rollout_manifests = sorted(project_path(path) for path in rollout_manifests)
    protocol, data = load_rollout_shards(rollout_manifests, policy=policy)
    target = ValueTarget(**protocol["target"])
    rules = TokenRules(**protocol["token_rules"])
    validation = data["is_validation"]
    if not validation.any() or validation.all():
        raise ValueError("Provide shards containing both train and validation rollout groups")
    train_x, train_y = data["state_ids"][~validation], data["labels"][~validation]
    val_x, val_y = data["state_ids"][validation], data["labels"][validation]
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("QM9 verifier supports CPU or CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    identities = {"rollout_manifest_sha256": {path: file_hash(path) for path in rollout_manifests},
                  "collection_id": "qm9-rollouts:" + payload_hash(protocol),
                  "target_id": target.target_id, "config_file_sha256": file_hash(config_path)}
    attempt = RunArtifacts.create(config["output_dir"], run_id=run_id + "-" + policy,
                                  attempt_id=attempt_id, config=config,
                                  target_id=target.target_id, reference_id=target.reference_id)
    attempt.manifest.update(task="train_qm9_verifier", target=asdict(target), policy=policy)
    try:
        write_yaml(attempt.path("config.yaml"), config)
        write_json(attempt.path("execution.json"), {"backend": "local", **(execution_metadata or {}),
                   "rollout_manifests": rollout_manifests, "resume_manifest": resume_manifest})
        random.seed(settings["seed"])
        np.random.seed(settings["seed"] % (2**32 - 1))
        torch.manual_seed(settings["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(settings["seed"])
        model = TokenStateVerifier(rules.vocab_size, rules.sequence_length, **config["model"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
        environment = verifier_environment(device)
        write_json(attempt.path("environment.json"), environment)
        write_json(attempt.path("label_protocol.json"), protocol)
        for name in ("config.yaml", "execution.json", "environment.json", "label_protocol.json"):
            attempt.register(name)
        start, history = 0, []
        previous_cost = CostRecord()
        if resume_manifest:
            saved = read_completed_manifest(resume_manifest)
            if saved.get("task") != "train_qm9_verifier_checkpoint" or saved.get("target") != protocol["target"]:
                raise ValueError("Resume verifier target differs")
            required = {"model.json", "model.safetensors", "training_state.pt", "training_state.pt.metadata.json"}
            if (not required.issubset(saved["artifacts"])
                    or saved.get("checkpoint_sha256") != saved["artifacts"]["model.safetensors"]["sha256"]):
                raise ValueError("Resume verifier artifacts/weight identity differ")
            directory = os.path.dirname(project_path(resume_manifest))
            state = read_checkpoint(os.path.join(directory, "training_state.pt"))
            if (state["config"] != config or state["protocol"] != protocol or state["identities"] != identities
                    or state["environment"] != environment or state["epoch"] != saved["epoch"]):
                raise ValueError("Resume requires unchanged target, data, settings, software and device")
            model.load_state_dict(load_file(os.path.join(directory, "model.safetensors")), strict=True)
            optimizer.load_state_dict(state["optimizer"])
            start, history = state["epoch"], state["history"]
            previous_cost = CostRecord(**state["cumulative_cost"])
            if not 0 < start < settings["epochs"]:
                raise ValueError("Resume requires an intermediate completed epoch checkpoint")
            restore_training_rng(state["rng"])  # Last setup operation that can affect draws.
        budget = BudgetTracker(base_sequence_limit=None)
        last_checkpoint = None
        report_result(f"QM9 {policy} verifier: {len(train_y)} train / {len(val_y)} validation snapshots")
        for epoch in tqdm(range(start + 1, settings["epochs"] + 1), initial=start,
                          total=settings["epochs"], desc=f"Train {policy} verifier", unit="epoch"):
            permutation = torch.randperm(len(train_y))  # CPU shuffle, as upstream.
            model.train()
            total_loss = 0.
            for offset in tqdm(range(0, len(train_y), settings["batch_size"]), desc="MSE batches", leave=False):
                indices = permutation[offset:offset + settings["batch_size"]]
                x, y = train_x[indices].to(device), train_y[indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(verifier_forward(model, x, budget=budget, device=device), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite verifier training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
            val_mse = validation_mse(model, val_x, val_y, batch_size=settings["batch_size"], device=device, budget=budget)
            row = {"epoch": epoch, "train_mse": total_loss / len(train_y), "validation_mse": val_mse}
            history.append(row)
            report_result(f"Verifier epoch {epoch}: train MSE={row['train_mse']:.6f}, validation MSE={val_mse:.6f}")
            if epoch % settings["save_every_epochs"] == 0 or epoch == settings["epochs"]:
                last_checkpoint = save_verifier_checkpoint(
                    attempt.path(f"checkpoints/epoch-{epoch:04d}"), model, optimizer, epoch=epoch,
                    config=config, protocol=protocol, identities=identities, environment=environment,
                    history=history, cumulative_cost=previous_cost + budget.snapshot().total,
                )
                if on_checkpoint is not None:
                    on_checkpoint()
        summary = {"manifest": attempt.path("manifest.json"), "checkpoint_manifest": last_checkpoint,
                   "epoch": settings["epochs"], "target": asdict(target),
                   "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                   "train_snapshots": len(train_y), "validation_snapshots": len(val_y),
                   "cumulative_cost": asdict(previous_cost + budget.snapshot().total)}
        write_json(attempt.path("history.json"), history)
        write_json(attempt.path("summary.json"), summary)
        required = ["history.json", "summary.json"]
        for directory, directories, files in os.walk(attempt.path("checkpoints")):
            directories.sort()
            for name in sorted(files):
                required.append(os.path.relpath(os.path.join(directory, name), attempt.directory))
        for name in required:
            attempt.register(name)
        attempt.complete(required_artifacts=required)
        return summary
    except Exception as exc:
        if attempt.manifest["status"] == "running":
            attempt.fail(f"{type(exc).__name__}: {exc}")
        raise
