# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kijung Jeon

"""QM9 token Transformer adapted from MDM-VGB's qm9/verifier.py.

Architecture and parameter names are retained. Strict target checks, safe local
checkpoints, terminal anchoring and actual-forward accounting are added here.
"""

import os
from dataclasses import asdict

import torch
from safetensors.torch import load_file
from torch import nn

from src.budget import BudgetExceeded
from src.config import project_path
from src.io import read_completed_manifest, read_json
from src.sampling.completion import cost_since
from src.types import CostPhase, CostRecord, EstimateStatus, TokenRules, ValueEstimate, ValueTarget, require_count
from src.verifiers.interface import Verifier


class TokenStateVerifier(nn.Module):
    def __init__(self, vocab_size, max_length, *, hidden_dim=256, num_layers=4,
                 num_heads=4, dropout=0.1):
        super().__init__()
        for name, value in (("vocab_size", vocab_size), ("max_length", max_length),
                            ("hidden_dim", hidden_dim), ("num_layers", num_layers), ("num_heads", num_heads)):
            require_count(value, name, positive=True)
        if hidden_dim % num_heads or not 0 <= dropout < 1:
            raise ValueError("Invalid verifier attention dimensions/dropout")
        self.max_length = max_length
        self.settings = dict(hidden_dim=hidden_dim, num_layers=num_layers,
                             num_heads=num_heads, dropout=dropout)
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                                  nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, input_ids):
        if input_ids.ndim != 2 or input_ids.shape[1] != self.max_length or input_ids.dtype != torch.long:
            raise ValueError("Verifier expects fixed-length int64 token states")
        positions = torch.arange(self.max_length, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)[None, :, :]
        x = self.encoder(x)  # No PAD mask: all positions participate, as upstream.
        return torch.sigmoid(self.head(x.mean(dim=1)).squeeze(-1))


class NeuralVerifier(Verifier):
    def __init__(self, model, reward, target, *, batch_size=8192):
        super().__init__(target, reward)
        require_count(batch_size, "verifier inference batch size", positive=True)
        if model.max_length != self.rules.sequence_length or model.token_emb.num_embeddings != self.rules.vocab_size:
            raise ValueError("Neural verifier length/vocabulary differs from reward")
        self.model, self.batch_size = model.eval(), batch_size
        self.model.requires_grad_(False)
        self.device = next(model.parameters()).device

    def required_work(self, states):
        return 0, sum(not state.is_complete for state in states)

    @classmethod
    def from_checkpoint(cls, manifest_path, *, reward, expected_target,
                        backbone_checkpoint_id, tokenizer_id, device="cuda", batch_size=8192):
        manifest = read_completed_manifest(manifest_path)
        if manifest.get("task") != "train_qm9_verifier_checkpoint":
            raise ValueError("Expected a completed QM9 verifier checkpoint")
        required = {"model.json", "model.safetensors", "training_state.pt", "training_state.pt.metadata.json"}
        if not required.issubset(manifest["artifacts"]):
            raise ValueError("Verifier checkpoint is missing required artifacts")
        directory = os.path.dirname(project_path(manifest_path))
        metadata = read_json(os.path.join(directory, "model.json"))
        if metadata["output_activation"] != "sigmoid":
            raise ValueError("QM9 Pass@95 verifier requires sigmoid output")
        target = ValueTarget(**metadata["target"])
        if (target != expected_target or manifest.get("target") != asdict(target)
                or metadata["backbone_checkpoint_sha256"] != backbone_checkpoint_id
                or metadata["tokenizer_sha256"] != tokenizer_id
                or tokenizer_id != reward.tokenizer_sha256
                or TokenRules(**metadata["token_rules"]) != reward.token_rules
                or reward.calibration is None
                or reward.calibration.checkpoint_sha256 != backbone_checkpoint_id):
            raise ValueError("Verifier checkpoint target/backbone/tokenizer/calibration mismatch")
        if manifest.get("checkpoint_sha256") != manifest["artifacts"]["model.safetensors"]["sha256"]:
            raise ValueError("Verifier weight identity differs from its manifest")
        model = TokenStateVerifier(reward.token_rules.vocab_size, reward.token_rules.sequence_length,
                                   **metadata["model"])
        model.load_state_dict(load_file(os.path.join(directory, "model.safetensors")), strict=True)
        result = cls(model.to(device), reward, target, batch_size=batch_size)
        result.checkpoint_id = manifest["checkpoint_sha256"]
        return result

    @torch.no_grad()
    def _estimate_batch(self, states, *, rng, budget, reservation):
        del rng
        results = []
        self.model.eval()
        for start in range(0, len(states), self.batch_size):
            chunk = states[start:start + self.batch_size]
            before = budget.snapshot().total
            tokens = torch.tensor([state.token_ids for state in chunk], dtype=torch.long, device=self.device)
            try:
                with budget.verifier_forward(batch_size=len(chunk), phase=CostPhase.VERIFICATION,
                                             reservation=reservation):
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    values = self.model(tokens)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
            except BudgetExceeded as exc:
                results.extend(ValueEstimate(self.target, EstimateStatus.BUDGET_EXHAUSTED, None,
                                              error=str(exc)) for _ in states[start:])
                break
            if values.shape != (len(chunk),) or not torch.isfinite(values).all():
                raise FloatingPointError("Invalid neural verifier predictions")
            cost = cost_since(budget, before)
            for index, value in enumerate(values.cpu().tolist()):
                results.append(ValueEstimate(self.target, EstimateStatus.ESTIMATED, float(value),
                                              cost=cost if index == 0 else CostRecord()))
        return results
