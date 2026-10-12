"""Group-conditioned absorbing MDLM, written independently for this project.

The bidirectional rotary/AdaLN backbone and log-linear masked objective follow
the QM9 reference's architecture ideas. ChEMBL vocabulary/length, class labels,
enabled time conditioning, and explicit EOS handling are new. PAD remains a
learned clean token in the fixed-width diffusion sequence, as in QM9.
No QM9 checkpoint compatibility or original parameter count is implied.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.artifacts import ArtifactStore, sha256_file
from src.config import Config, config_digest
from src.schemas import GROUPS, SmilesState, SpecialTokens, require_count
from src.tokenizer import SmilesTokenizer
from src.tracking import CostTracker

MODEL_VERSION = "smiles-group-mdlm-v1"


def validate_mdlm(config: Config) -> None:
    options = config["mdlm"]
    if options["version"] != MODEL_VERSION or options["initialization"] != "scratch":
        raise ValueError("Unsupported MDLM version or pretrained initialization")
    architecture = options["architecture"]
    for name in ("layers", "hidden_size", "heads", "conditioning_size", "max_length", "feedforward_ratio"):
        require_count(architecture[name], f"mdlm.architecture.{name}", positive=True)
    if (architecture["hidden_size"] % architecture["heads"]
            or (architecture["hidden_size"] // architecture["heads"]) % 2):
        raise ValueError("Hidden size must divide into even rotary head dimensions")
    if (architecture["position_encoding"] != "rotary"
            or architecture["time_conditioning"] is not True
            or architecture["group_conditioning"] is not True):
        raise ValueError("This MDLM requires rotary positions and time/group conditioning")
    if not 0 <= architecture["dropout"] < 1:
        raise ValueError("MDLM dropout must be in [0,1)")
    if architecture["max_length"] != config["data"]["tokenizer"]["max_length"]:
        raise ValueError("Model/tokenizer lengths differ; rebuild T03 after a train/validation length change")
    diffusion = options["diffusion"]
    if (diffusion["schedule"] != "loglinear" or diffusion["mask_pad"] is not True
            or diffusion["objective"] != "masked_elbo_per_fixed_width_token"
            or not 0 < diffusion["epsilon"] < 1 or not 0 < diffusion["minimum_time"] < 1):
        raise ValueError("Unsupported diffusion schedule, PAD policy, or objective")
    training = options["training"]
    for key in ("steps", "global_batch_size", "micro_batch_size", "cache_shards", "log_every", "validate_every", "save_every"):
        require_count(training[key], f"mdlm.training.{key}", positive=True)
    require_count(training["warmup_steps"], "warmup_steps")
    if training["global_batch_size"] % training["micro_batch_size"]:
        raise ValueError("Global batch must be a multiple of the micro batch")
    if training["warmup_steps"] >= training["steps"]:
        raise ValueError("Warmup must end before the final training step")
    for key in ("learning_rate", "minimum_learning_rate", "gradient_clip", "adam_epsilon"):
        if not math.isfinite(training[key]) or training[key] <= 0:
            raise ValueError(f"Training {key} must be finite and positive")
    if (training["minimum_learning_rate"] > training["learning_rate"]
            or not math.isfinite(training["weight_decay"]) or training["weight_decay"] < 0
            or training["optimizer"] != "adamw" or training["scheduler"] != "warmup_cosine"
            or training["row_sampling"] != "uniform_with_replacement"
            or training["group_sampling"] != "uniform_matching_group"
            or training["device"] not in ("auto", "cpu", "cuda")
            or training["dtype"] not in ("bfloat16", "float32")):
        raise ValueError("Unsupported training settings")
    if len(training["betas"]) != 2 or any(not 0 <= beta < 1 for beta in training["betas"]):
        raise ValueError("Adam betas must have two values in [0,1)")
    validation = options["validation"]
    if validation["split"] != "validation" or validation["selection"] != "minimum_masked_elbo":
        raise ValueError("Only validation can select a checkpoint")
    require_count(validation["batch_size"], "validation batch_size", positive=True)
    require_count(validation["seed"], "validation seed")
    if validation["max_rows"] is not None:
        require_count(validation["max_rows"], "validation max_rows", positive=True)
    if config["suite"] == "replication" or config.get("group") is not None:
        raise ValueError("T05 trains all three groups in primary or smoke, not the AR replication suite")
    completion = options["completion"]
    require_count(completion["steps"], "completion steps", positive=True)
    if (completion["trie_policy"] != "left_to_right"
            or completion["reference_policy"] != "absorbing_reverse"
            or type(completion["final_denoise"]) is not bool
            or not 0 < completion["end_time"] < 1
            or (completion["temperature"], completion["top_p"], completion["top_k"]) != (1.0, 1.0, 0)):
        raise ValueError("Completion requires full-vocabulary policies at temperature 1")


def masking_probability(times: torch.Tensor, epsilon: float) -> torch.Tensor:
    return (1.0 - epsilon) * times


def diffusion_sigma(times: torch.Tensor, epsilon: float) -> torch.Tensor:
    return -torch.log1p(-masking_probability(times.float(), epsilon))


def corrupt_tokens(ids: torch.Tensor, attention: torch.Tensor, special: SpecialTokens,
                   times: torch.Tensor, epsilon: float, *, generator=None):
    draws = torch.rand(ids.shape, device=ids.device, generator=generator)
    masked = (draws < masking_probability(times, epsilon)[:, None]) & attention
    return ids.masked_fill(masked, special.mask_id), masked


def masked_objective(logits: torch.Tensor, targets: torch.Tensor, masked: torch.Tensor,
                     attention: torch.Tensor, times: torch.Tensor, special: SpecialTokens) -> torch.Tensor:
    """Mean per-sequence ELBO estimator over the full fixed-width sequence.

    For p(t)=(1-epsilon)t, p'(t)/p(t)=1/t. Only newly MASKed
    positions contribute, including EOS and PAD. A zero-mask example contributes
    zero without resampling t or forcing a mask (which would change the estimator).
    """
    if logits.shape[:2] != targets.shape or masked.shape != targets.shape or attention.shape != targets.shape:
        raise ValueError("Objective tensor shapes differ")
    if times.shape != (targets.shape[0],) or torch.any(times <= 0) or torch.any(times > 1):
        raise ValueError("Objective requires times in (0,1]")
    if not torch.all(attention):
        raise ValueError("The fixed-width objective includes every position, including PAD")
    if torch.any(targets == special.mask_id):
        raise ValueError("Clean targets contain MASK")
    allowed = logits.float().clone()
    allowed[..., special.mask_id] = -torch.inf
    nll = -allowed.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
    selected = torch.where(masked, nll, torch.zeros_like(nll))
    return (selected.sum(-1) / attention.sum(-1) / times.float()).mean()


def _rotate(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor[..., 0::2], tensor[..., 1::2]
    return torch.stack((-second, first), dim=-1).flatten(-2)


class RotaryAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.head_width, self.dropout = heads, width // heads, dropout
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        frequencies = 1.0 / (10000 ** (torch.arange(0, self.head_width, 2).float() / self.head_width))
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, hidden: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, length, 3, self.heads, self.head_width)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        angles = torch.outer(torch.arange(length, device=hidden.device).float(), self.frequencies.float())
        cos = angles.cos().repeat_interleave(2, -1).to(query.dtype)[None, None]
        sin = angles.sin().repeat_interleave(2, -1).to(query.dtype)[None, None]
        query = query * cos + _rotate(query) * sin
        key = key * cos + _rotate(key) * sin
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0, is_causal=False,
        )
        return self.output(output.transpose(1, 2).reshape(batch, length, width))


class ConditionedBlock(nn.Module):
    def __init__(self, architecture: Mapping[str, Any]) -> None:
        super().__init__()
        width = architecture["hidden_size"]
        self.attention = RotaryAttention(width, architecture["heads"], architecture["dropout"])
        self.norm_attention = nn.LayerNorm(width, elementwise_affine=False)
        self.norm_ff = nn.LayerNorm(width, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(width, width * architecture["feedforward_ratio"]), nn.GELU(approximate="tanh"),
            nn.Linear(width * architecture["feedforward_ratio"], width),
        )
        self.modulation = nn.Linear(architecture["conditioning_size"], 6 * width)
        self.dropout = nn.Dropout(architecture["dropout"])
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.modulation(F.silu(condition)).chunk(6, -1)
        normalized = self.norm_attention(hidden) * (1 + scale_a[:, None]) + shift_a[:, None]
        hidden = hidden + gate_a[:, None] * self.dropout(self.attention(normalized, attention))
        normalized = self.norm_ff(hidden) * (1 + scale_f[:, None]) + shift_f[:, None]
        return hidden + gate_f[:, None] * self.dropout(self.ff(normalized))


class GroupMDLM(nn.Module):
    def __init__(self, specification: Mapping[str, Any]) -> None:
        super().__init__()
        self.specification = dict(specification)
        architecture = specification["architecture"]
        self.vocab_size = specification["vocab_size"]
        self.max_length = architecture["max_length"]
        self.special_tokens = SpecialTokens.from_dict(specification["special_tokens"])
        if (specification["version"] != MODEL_VERSION or tuple(specification["groups"]) != GROUPS
                or max(self.special_tokens.to_dict().values()) >= self.vocab_size):
            raise ValueError("Invalid MDLM specification")
        width, condition = architecture["hidden_size"], architecture["conditioning_size"]
        self.embedding = nn.Embedding(self.vocab_size, width)
        self.groups = nn.Embedding(len(GROUPS), condition)
        self.time_embedding = nn.Sequential(nn.Linear(256, condition), nn.SiLU(), nn.Linear(condition, condition))
        frequencies = torch.exp(-math.log(10000) * torch.arange(128).float() / 127)
        self.register_buffer("time_frequencies", frequencies, persistent=False)
        self.blocks = nn.ModuleList([ConditionedBlock(architecture) for _ in range(architecture["layers"])])
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.final_modulation = nn.Linear(condition, 2 * width)
        self.output = nn.Linear(width, self.vocab_size)
        nn.init.zeros_(self.final_modulation.weight)
        nn.init.zeros_(self.final_modulation.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, ids: torch.Tensor, times: torch.Tensor, groups: torch.Tensor,
                attention: torch.Tensor) -> torch.Tensor:
        if (ids.ndim != 2 or ids.shape[1] > self.max_length or attention.shape != ids.shape
                or times.shape != (ids.shape[0],) or groups.shape != times.shape
                or attention.dtype != torch.bool or torch.any(attention.sum(-1) == 0)):
            raise ValueError("Invalid MDLM batch dimensions or attention mask")
        if (torch.any(ids < 0) or torch.any(ids >= self.vocab_size) or torch.any(groups < 0)
                or torch.any(groups >= len(GROUPS)) or torch.any(times < 0) or torch.any(times > 1)):
            raise ValueError("Invalid token, group, or diffusion time")
        sigma = diffusion_sigma(times, self.specification["diffusion"]["epsilon"])
        angles = sigma[:, None] * self.time_frequencies[None]
        time_features = torch.cat((angles.cos(), angles.sin()), -1)
        condition = self.time_embedding(time_features.to(self.embedding.weight.dtype)) + self.groups(groups)
        hidden = self.embedding(ids)
        for block in self.blocks:
            hidden = block(hidden, condition, attention)
        shift, scale = self.final_modulation(F.silu(condition)).chunk(2, -1)
        hidden = self.final_norm(hidden) * (1 + scale[:, None]) + shift[:, None]
        return self.output(hidden)


def model_specification(config: Config, tokenizer: SmilesTokenizer) -> dict[str, Any]:
    if tokenizer.max_length != config["mdlm"]["architecture"]["max_length"]:
        raise ValueError("Tokenizer metadata does not match the configured MDLM width")
    return {
        "version": MODEL_VERSION, "architecture": dict(config["mdlm"]["architecture"]),
        "diffusion": dict(config["mdlm"]["diffusion"]), "groups": list(GROUPS),
        "vocab_size": tokenizer.vocab_size, "special_tokens": tokenizer.special_tokens.to_dict(),
        "tokenizer_fingerprint": tokenizer.fingerprint,
    }


class MDLMAdapter:
    """Inference only; this adapter owns forward timing/NFE accounting.

    State-derived time is used for q_z; the reference sampler supplies scheduled
    times explicitly. Callers must not count the same forward again.
    """

    def __init__(self, model: GroupMDLM, *, model_id: str, tracker: CostTracker | None = None,
                 inner: bool = False, autocast_dtype: torch.dtype | None = None) -> None:
        self.model = model.eval()
        self.model_id, self.tracker, self.inner = model_id, tracker, inner
        self.vocab_size, self.special_tokens = model.vocab_size, model.special_tokens
        self.device = next(model.parameters()).device
        self.autocast_dtype = autocast_dtype

    @classmethod
    def from_checkpoint(cls, store: ArtifactStore, manifest_path: str, tokenizer: SmilesTokenizer,
                        *, device: str = "cpu", tracker=None, inner: bool = False, dtype=torch.float32):
        from safetensors.torch import load_file

        manifest = store.require_manifest(manifest_path, kind="mdlm_checkpoint", metadata={
            "tokenizer_fingerprint": tokenizer.fingerprint, "groups": list(GROUPS),
        })
        specification = store.read_json(manifest.metadata["specification_path"])
        if (config_digest(specification) != manifest.metadata["specification_hash"]
                or specification["tokenizer_fingerprint"] != tokenizer.fingerprint
                or specification["special_tokens"] != tokenizer.special_tokens.to_dict()
                or specification["vocab_size"] != tokenizer.vocab_size
                or specification["architecture"]["max_length"] != tokenizer.max_length):
            raise ValueError("MDLM checkpoint specification/tokenizer mismatch")
        weights = store.require_file(manifest.metadata["weights_path"])
        if manifest.metadata["weights_path"] not in manifest.files or manifest.metadata["specification_path"] not in manifest.files:
            raise ValueError("MDLM manifest does not cover its weights and specification")
        # Loading a trained model must not advance the caller's sampling RNG.
        with torch.random.fork_rng(devices=[]):
            model = GroupMDLM(specification)
        model.load_state_dict(load_file(weights, device="cpu"), strict=True)
        model.to(device=device, dtype=dtype)
        identity = {"weights_sha256": sha256_file(weights), "specification_hash": config_digest(specification),
                    "dtype": str(dtype)}
        return cls(model, model_id="mdlm:" + config_digest(identity), tracker=tracker, inner=inner)

    @torch.inference_mode()
    def logits_batch(self, states: Sequence[SmilesState], *, times: torch.Tensor | None = None) -> torch.Tensor:
        if not states or len({len(state.token_ids) for state in states}) != 1:
            raise ValueError("MDLM inference requires a nonempty batch of equal-width states")
        for state in states:
            if state.special_tokens != self.special_tokens:
                raise ValueError("State special tokens differ from the model")
        ids = torch.tensor([state.token_ids for state in states], device=self.device, dtype=torch.long)
        # PAD carries length information in the learned fixed-width sequence.
        attention = torch.ones_like(ids, dtype=torch.bool)
        groups = torch.tensor([GROUPS.index(state.group) for state in states], device=self.device)
        if times is None:
            fractions = ((ids == self.special_tokens.mask_id) & attention).sum(-1) / attention.sum(-1).clamp_min(1)
            minimum = self.model.specification["diffusion"]["minimum_time"]
            epsilon = self.model.specification["diffusion"]["epsilon"]
            times = (fractions / (1 - epsilon)).clamp(min=minimum, max=1.0)
        else:
            times = times.to(self.device, dtype=torch.float32)
        timing = nullcontext() if self.tracker is None else self.tracker.model_call(
            "mdlm", batch_size=len(states), inner=self.inner,
            gpu=self.device.type == "cuda", device=self.device,
        )
        precision = nullcontext() if self.autocast_dtype is None else torch.autocast(
            device_type=self.device.type, dtype=self.autocast_dtype,
        )
        with timing, precision:
            logits = self.model(ids, times, groups, attention).float()
        if not torch.isfinite(logits).all():
            raise ValueError("MDLM produced nonfinite logits")
        # Absorbing parameterization: MASK cannot be a predicted clean token;
        # observed coordinates are delta distributions. Keep every other token.
        logits[..., self.special_tokens.mask_id] = -torch.inf
        known = ids != self.special_tokens.mask_id
        if known.any():
            logits[known] = -torch.inf
            rows, positions = known.nonzero(as_tuple=True)
            logits[rows, positions, ids[rows, positions]] = 0.0
        return logits

    def probabilities_batch(self, states: Sequence[SmilesState], *, times=None) -> torch.Tensor:
        return self.logits_batch(states, times=times).softmax(-1)
