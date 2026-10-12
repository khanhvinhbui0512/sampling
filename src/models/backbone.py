"""Load the local QM9 model and count every actual batched forward.

The SUBS parameterization follows MDM-VGB's QM9 harness (MIT, copyright
2026 Kijung Jeon); see THIRD_PARTY_NOTICES.md. No sampling is implemented here.
"""

import os

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.config import project_path
from src.io import payload_hash, read_completed_manifest, read_json
from src.models.configuration_udlm import UDLMConfig
from src.models.modeling_udlm import UDLM
from src.types import TokenRules


def loglinear_sigma(t, eps=0.001):
    return -torch.log1p(-(1 - eps) * t)


def subs_log_probs(logits, tokens, mask_id):
    """Exclude MASK predictions and copy every known position, as upstream does."""
    scores = logits.float().clone()
    scores[..., mask_id] = -1e6
    known = tokens != mask_id
    scores[known] = -1e6
    rows, positions = known.nonzero(as_tuple=True)
    scores[rows, positions, tokens[known]] = 0
    return scores.log_softmax(dim=-1)


class QM9Backbone:
    def __init__(self, model, tokenizer, token_rules, *, tokenizer_sha256,
                 checkpoint_id="untrained", device="cpu", precision="fp32"):
        self.model = model
        self.tokenizer = tokenizer
        self.token_rules = token_rules
        self.time_conditioning = model.config.time_conditioning
        self.tokenizer_id = tokenizer_sha256
        self.checkpoint_id = checkpoint_id
        # This identifies the backbone, not a completion policy or a joint pi_rho.
        self.reference_id = "qm9-mdlm:" + checkpoint_id
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("QM9 supports CPU or CUDA")
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not support bf16")
        self.precision = precision if self.device.type == "cuda" else "fp32"
        vocabulary = tokenizer.get_vocab()
        if (len(tokenizer) != token_rules.vocab_size or len(vocabulary) != len(tokenizer)
                or set(vocabulary.values()) != set(range(len(tokenizer)))):
            raise ValueError("Tokenizer must match the full contiguous vocabulary")
        if (model.config.vocab_size != token_rules.vocab_size
                or model.config.model_length != token_rules.sequence_length):
            raise ValueError("Model vocabulary/length differs from token rules")
        for role in ("mask", "bos", "eos", "pad", "unk"):
            if getattr(tokenizer, role + "_token_id", None) != getattr(token_rules, role + "_id"):
                raise ValueError(f"Tokenizer {role} ID differs from token rules")
        for role in ("mask", "bos", "eos", "pad"):
            if getattr(model.config, role + "_token_id", None) != getattr(token_rules, role + "_id"):
                raise ValueError(f"Model config {role} ID differs from token rules")
        self.vocabulary_sha256 = payload_hash(vocabulary)
        tokenizer.padding_side = tokenizer.truncation_side = "right"
        self.model.to(self.device)

    @classmethod
    def from_assets(cls, assets, model_settings, *, device="cpu", precision="fp32"):
        tokenizer = AutoTokenizer.from_pretrained(assets.tokenizer_directory,
                                                  local_files_only=True, trust_remote_code=False)
        vocabulary_path = os.path.join(os.path.dirname(assets.manifest_path), "vocabulary.json")
        if tokenizer.get_vocab() != read_json(vocabulary_path):
            raise ValueError("Loaded tokenizer differs from the prepared vocabulary")
        rules = assets.token_rules
        config = UDLMConfig(vocab_size=rules.vocab_size, model_length=rules.sequence_length,
                            mask_token_id=rules.mask_id, bos_token_id=rules.bos_id,
                            eos_token_id=rules.eos_id, pad_token_id=rules.pad_id, **model_settings)
        return cls(UDLM(config), tokenizer, rules, tokenizer_sha256=assets.tokenizer_sha256,
                   device=device, precision=precision)

    @classmethod
    def from_checkpoint(cls, manifest_path, *, device="cpu", precision="fp32"):
        manifest = read_completed_manifest(manifest_path)
        if manifest.get("task") != "train_qm9_base_checkpoint":
            raise ValueError("Expected a completed QM9 backbone checkpoint manifest")
        required = {"config.json", "model.safetensors", "training_state.pt",
                    "training_state.pt.metadata.json"}
        if not required.issubset(manifest["artifacts"]):
            raise ValueError("Checkpoint is missing model or resume artifacts")
        if not any(name.startswith("tokenizer/") for name in manifest["artifacts"]):
            raise ValueError("Checkpoint has no saved tokenizer")
        directory = os.path.dirname(project_path(manifest_path))
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(directory, "tokenizer"),
                                                  local_files_only=True, trust_remote_code=False)
        config = UDLMConfig(**read_json(os.path.join(directory, "config.json")))
        model = UDLM(config)
        weights = load_file(os.path.join(directory, "model.safetensors"), device="cpu")
        model.load_state_dict(weights, strict=True)
        identity = manifest["artifacts"]["model.safetensors"]["sha256"]
        if manifest["checkpoint_sha256"] != identity:
            raise ValueError("Checkpoint identity differs from its weights")
        backbone = cls(model, tokenizer, TokenRules(**manifest["token_rules"]),
                       tokenizer_sha256=manifest["tokenizer_sha256"], checkpoint_id=identity,
                       device=device, precision=precision)
        if backbone.vocabulary_sha256 != manifest["vocabulary_sha256"]:
            raise ValueError("Checkpoint tokenizer vocabulary changed")
        backbone.model.eval()
        return backbone

    def forward_tokens(self, tokens, sigma, *, budget, phase, reservation=None):
        """Training and inference share this single instrumented model call."""
        if (tokens.ndim != 2 or tokens.shape[0] == 0
                or tokens.shape[1] != self.token_rules.sequence_length):
            raise ValueError("Expected a nonempty [batch, sequence_length] token tensor")
        if tokens.dtype != torch.long or tokens.device != self.device:
            raise ValueError("Token tensor must contain int64 IDs on the model device")
        if sigma.shape != (tokens.shape[0],) or sigma.device != tokens.device:
            raise ValueError("Expected one noise level on the model device per sequence")
        if torch.any(tokens < 0) or torch.any(tokens >= self.token_rules.vocab_size):
            raise ValueError("Token tensor contains IDs outside the vocabulary")
        if not torch.isfinite(sigma).all() or torch.any(sigma < 0):
            raise ValueError("Noise levels must be finite and nonnegative")
        with budget.base_forward(batch_size=tokens.shape[0], phase=phase, reservation=reservation):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                enabled=self.precision == "bf16"):
                output = self.model(input_ids=tokens, timesteps=sigma, return_dict=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        if output.logits.shape != (*tokens.shape, self.token_rules.vocab_size):
            raise ValueError("Backbone returned an unexpected logits shape")
        return output.logits

    @torch.no_grad()
    def logits_batch(self, states, *, budget, phase, reservation=None):
        ratio = torch.tensor([len(state.masked_positions) / self.token_rules.sequence_length
                              for state in states], dtype=torch.float32)
        sigma = -torch.log1p(-ratio.clamp(min=1e-5, max=0.999))
        return self.logits_at_noise_batch(states, sigma, budget=budget, phase=phase,
                                          reservation=reservation)

    @torch.no_grad()
    def logits_at_noise_batch(self, states, sigma, *, budget, phase, reservation=None):
        """Native diffusion passes schedule noise explicitly, rather than mask ratio."""
        if not states:
            return torch.empty((0, self.token_rules.sequence_length, self.token_rules.vocab_size))
        if any(state.rules != self.token_rules for state in states):
            raise ValueError("State token rules differ from the backbone")
        tokens = torch.tensor([state.token_ids for state in states], dtype=torch.long, device=self.device)
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=self.device)
        self.model.eval()
        return self.forward_tokens(tokens, sigma, budget=budget, phase=phase,
                                   reservation=reservation).detach().float().cpu()

    @torch.no_grad()
    def log_probs_batch(self, states, *, budget, phase, reservation=None):
        logits = self.logits_batch(states, budget=budget, phase=phase, reservation=reservation)
        tokens = torch.tensor([state.token_ids for state in states], dtype=torch.long)
        if not states:
            return logits
        return subs_log_probs(logits, tokens, self.token_rules.mask_id)
