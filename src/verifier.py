"""Small state/group Transformer regressing binary rollout returns.

The token/position embeddings, full-width mean pooling and sigmoid MSE model
follow the QM9 reference architecture. This independent implementation adds
class conditioning and validates the project tokenizer/checkpoint provenance.
Predictions are neural estimates; exact terminal anchoring belongs to the value
provider in Step 9. Native rollout labels and q_z labels have distinct targets.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from src.artifacts import ArtifactStore, sha256_file
from src.config import Config, config_digest, project_path
from src.schemas import GROUPS, ArtifactManifest, SmilesState, SpecialTokens, require_count
from src.tasks.prepare_corpus import commit_file
from src.tokenizer import SmilesTokenizer
from src.tracking import CostTracker

VERIFIER_VERSION = "smiles-process-verifier-v1"


def data_overrides(manifest: str = "") -> list[str]:
    import json

    return ["verifier.data.manifest=" + json.dumps(project_path(manifest))] if manifest else []


def validate_verifier(config: Config) -> None:
    options = config["verifier"]
    if options["version"] != VERIFIER_VERSION or config["suite"] == "replication" or config.get("group") is not None:
        raise ValueError("T07 trains all groups in primary or smoke")
    if options["data"]["manifest"] is None:
        raise ValueError("T07 requires --rollouts with a published T06 manifest")
    project_path(options["data"]["manifest"])
    architecture = options["architecture"]
    for name in ("layers", "hidden_size", "heads", "feedforward_ratio", "max_length"):
        require_count(architecture[name], "verifier.architecture." + name, positive=True)
    if (architecture["hidden_size"] % architecture["heads"] or not 0 <= architecture["dropout"] < 1
            or architecture["max_length"] != config["data"]["tokenizer"]["max_length"]
            or architecture["output_activation"] != "sigmoid" or architecture["pooling"] != "fixed_width_mean"
            or architecture["group_conditioning"] is not True):
        raise ValueError("Verifier requires matching width, valid heads/dropout, group conditioning, and mean/sigmoid output")
    training = options["training"]
    for name in ("epochs", "global_batch_size", "micro_batch_size", "cache_shards", "log_every_updates", "save_every_updates"):
        require_count(training[name], "verifier.training." + name, positive=True)
    if training["global_batch_size"] % training["micro_batch_size"]:
        raise ValueError("Verifier global batch must be a multiple of the micro batch")
    for name in ("learning_rate", "gradient_clip", "adam_epsilon"):
        if not math.isfinite(training[name]) or training[name] <= 0:
            raise ValueError(f"Verifier {name} must be finite and positive")
    if (not math.isfinite(training["weight_decay"]) or training["weight_decay"] < 0
            or len(training["betas"]) != 2 or any(not 0 <= beta < 1 for beta in training["betas"])
            or training["scheduler"] != "constant" or training["sampler"] != "shuffled_shards_and_rows"
            or training["device"] not in ("auto", "cpu", "cuda") or training["dtype"] not in ("bfloat16", "float32")):
        raise ValueError("Unsupported verifier optimizer, sampler, device or precision")
    if training["early_stopping_patience"] is not None:
        require_count(training["early_stopping_patience"], "early_stopping_patience", positive=True)
    validation = options["validation"]
    if validation["split"] != "validation" or validation["selection"] != "minimum_mse":
        raise ValueError("Use only the locked validation partition for checkpoint selection")
    for name in ("batch_size", "calibration_bins"):
        require_count(validation[name], "verifier.validation." + name, positive=True)
    edges = validation["mask_fraction_edges"]
    if (not isinstance(edges, list) or len(edges) < 2 or edges[0] != 0 or edges[-1] != 1
            or any(isinstance(edge, bool) or not isinstance(edge, (int, float)) or not math.isfinite(edge) for edge in edges)
            or any(first >= second for first, second in zip(edges[:-1], edges[1:], strict=True))):
        raise ValueError("Calibration depth edges must strictly increase from 0 to 1")


def verifier_specification(config: Config, tokenizer: SmilesTokenizer, dataset: ArtifactManifest) -> dict[str, Any]:
    source = dataset.metadata
    return {"version": VERIFIER_VERSION, "architecture": dict(config["verifier"]["architecture"]),
            "vocab_size": tokenizer.vocab_size, "special_tokens": tokenizer.special_tokens.to_dict(),
            "groups": list(GROUPS), "tokenizer_fingerprint": tokenizer.fingerprint,
            "reward_version": source["reward_version"], "reward_mode": source["reward_mode"],
            "source_mdlm_sha256": source["source_mdlm_sha256"], "policy_hash": source["policy_hash"],
            "target_kind": source["target_kind"], "target_definition": source["target_definition"],
            "suite": config["suite"]}


class StateVerifier(nn.Module):
    def __init__(self, specification: Mapping[str, Any]) -> None:
        super().__init__()
        self.specification = dict(specification)
        architecture = specification["architecture"]
        self.vocab_size, self.max_length = specification["vocab_size"], architecture["max_length"]
        self.special_tokens = SpecialTokens.from_dict(specification["special_tokens"])
        if (specification["version"] != VERIFIER_VERSION or tuple(specification["groups"]) != GROUPS
                or max(self.special_tokens.to_dict().values()) >= self.vocab_size
                or architecture["output_activation"] != "sigmoid"):
            raise ValueError("Invalid verifier specification")
        width = architecture["hidden_size"]
        self.token_embedding = nn.Embedding(self.vocab_size, width)
        self.position_embedding = nn.Embedding(self.max_length, width)
        self.group_embedding = nn.Embedding(len(GROUPS), width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=architecture["heads"], dim_feedforward=width * architecture["feedforward_ratio"],
            dropout=architecture["dropout"], activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=architecture["layers"], enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, ids: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or ids.shape[1] != self.max_length or groups.shape != (ids.shape[0],):
            raise ValueError("Verifier expects a full-width state and one group per row")
        if torch.any(ids < 0) or torch.any(ids >= self.vocab_size) or torch.any(groups < 0) or torch.any(groups >= len(GROUPS)):
            raise ValueError("Verifier token/group ID out of range")
        positions = torch.arange(self.max_length, device=ids.device)
        hidden = self.token_embedding(ids) + self.position_embedding(positions)[None] + self.group_embedding(groups)[:, None]
        # PAD and MASK are learned state tokens, as in QM9's fixed-width verifier.
        score = self.head(self.encoder(hidden).mean(1)).squeeze(-1)
        return score.float().sigmoid()


def verifier_loss(predictions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    if (predictions.ndim != 1 or predictions.shape != rewards.shape or predictions.numel() == 0
            or not torch.isfinite(predictions).all() or not torch.isfinite(rewards).all()
            or torch.any((rewards != 0) & (rewards != 1)) or torch.any(predictions < 0) or torch.any(predictions > 1)):
        raise ValueError("Verifier MSE requires finite sigmoid predictions and binary tau labels")
    return F.mse_loss(predictions.float(), rewards.float())


def save_verifier(store: ArtifactStore, directory: str, model: StateVerifier, *, config_hash: str,
                  metadata: Mapping[str, Any], calibration: Mapping[str, Any]) -> str:
    directory = project_path(directory)
    weights, specification, metrics = [os.path.join(directory, filename) for filename in (
        "model.safetensors", "config.json", "calibration.json",
    )]
    target = store.path(weights)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    save_file({key: tensor.detach().cpu().contiguous() for key, tensor in model.state_dict().items()}, target + ".part")
    commit_file(target + ".part", target)
    store.write_json(specification, model.specification)
    store.write_json(metrics, dict(calibration))
    details = {**dict(metadata), "weights_path": weights, "specification_path": specification,
               "specification_hash": config_digest(model.specification), "calibration_path": metrics,
               "tokenizer_fingerprint": model.specification["tokenizer_fingerprint"], "groups": list(GROUPS),
               "reward_version": model.specification["reward_version"], "source_mdlm_sha256": model.specification["source_mdlm_sha256"],
               "policy_hash": model.specification["policy_hash"], "target_kind": model.specification["target_kind"]}
    path = os.path.join(directory, "manifest.json")
    store.create_manifest(path, kind="verifier_checkpoint", config_hash=config_hash,
                          files=[weights, specification, metrics], metadata=details)
    return path


class VerifierAdapter:
    """Batched neural predictions with one owner of verifier-forward accounting.

    This is a model adapter, not a terminal-anchored ValueProvider. Step 9 must
    check target provenance and use exact tau on terminal states.
    """

    def __init__(self, model: StateVerifier, *, model_id: str, tracker: CostTracker | None = None,
                 batch_size: int = 64, autocast_dtype: torch.dtype | None = None) -> None:
        require_count(batch_size, "verifier inference batch_size", positive=True)
        self.model, self.model_id, self.tracker = model.eval(), model_id, tracker
        self.batch_size, self.autocast_dtype = batch_size, autocast_dtype
        self.device = next(model.parameters()).device
        self.target_metadata = {key: model.specification[key] for key in (
            "target_kind", "target_definition", "reward_version", "source_mdlm_sha256", "policy_hash", "suite",
        )}

    @classmethod
    def from_checkpoint(cls, store: ArtifactStore, path: str, tokenizer: SmilesTokenizer, *, device: str = "cpu",
                        tracker=None, batch_size: int = 64, dtype=torch.float32,
                        expected_reward_version: str | None = None, expected_source_mdlm: str | None = None):
        expected = {"tokenizer_fingerprint": tokenizer.fingerprint, "groups": list(GROUPS)}
        if expected_reward_version is not None:
            expected["reward_version"] = expected_reward_version
        if expected_source_mdlm is not None:
            expected["source_mdlm_sha256"] = expected_source_mdlm
        manifest = store.require_manifest(path, kind="verifier_checkpoint", metadata=expected)
        details = manifest.metadata
        if any(details[key] not in manifest.files for key in ("weights_path", "specification_path", "calibration_path")):
            raise ValueError("Verifier manifest does not cover all checkpoint files")
        specification = store.read_json(details["specification_path"])
        if (config_digest(specification) != details["specification_hash"]
                or specification["tokenizer_fingerprint"] != tokenizer.fingerprint
                or specification["special_tokens"] != tokenizer.special_tokens.to_dict()
                or specification["vocab_size"] != tokenizer.vocab_size
                or specification["architecture"]["max_length"] != tokenizer.max_length
                or any(specification[key] != details[key] for key in ("reward_version", "source_mdlm_sha256", "policy_hash", "target_kind"))):
            raise ValueError("Verifier checkpoint specification/provenance differs")
        with torch.random.fork_rng(devices=[]):
            model = StateVerifier(specification)
        weights = store.require_file(details["weights_path"])
        model.load_state_dict(load_file(weights, device="cpu"), strict=True)
        model.to(device=device, dtype=dtype)
        identity = {"weights_sha256": sha256_file(weights), "specification_hash": config_digest(specification), "dtype": str(dtype)}
        return cls(model, model_id="verifier:" + config_digest(identity), tracker=tracker, batch_size=batch_size)

    @torch.inference_mode()
    def predict_batch(self, states: Sequence[SmilesState]) -> list[float]:
        values = []
        for start in range(0, len(states), self.batch_size):
            batch = states[start:start + self.batch_size]
            if any(state.special_tokens != self.model.special_tokens or len(state.token_ids) != self.model.max_length for state in batch):
                raise ValueError("Verifier state/tokenizer width differs")
            ids = torch.tensor([state.token_ids for state in batch], dtype=torch.long, device=self.device)
            groups = torch.tensor([GROUPS.index(state.group) for state in batch], dtype=torch.long, device=self.device)
            timing = nullcontext() if self.tracker is None else self.tracker.model_call(
                "verifier", batch_size=len(batch), gpu=self.device.type == "cuda", device=self.device,
            )
            precision = nullcontext() if self.autocast_dtype is None else torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)
            with timing, precision:
                predicted = self.model(ids, groups)
            if not torch.isfinite(predicted).all():
                raise ValueError("Verifier predictions are not finite")
            values.extend(predicted.cpu().tolist())
        return values

    def predict(self, state: SmilesState) -> float:
        return self.predict_batch([state])[0]
