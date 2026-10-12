"""Token safety, tiny model behavior, accounting and checkpoint/resume contracts."""

import os
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.budget import BudgetExceeded, BudgetTracker
from src.io import read_checkpoint, read_json
from src.models.backbone import QM9Backbone, subs_log_probs
from src.models.configuration_udlm import UDLMConfig
from src.models.modeling_udlm import UDLM
from src.qm9.reward import IncompleteStateError, QM9Reward, TerminalStatus
from src.randomness import restore_training_rng
from src.training.base import build_scheduler, mdlm_loss, save_training_checkpoint
from src.types import CostPhase, MaskedState


def test_arbitrary_masks_preserve_every_known_token(token_rules):
    state = MaskedState((1, 3, 6, 3, 2, 0), token_rules)
    assert state.masked_positions == (1, 3)
    partial = state.reveal({3: 7})
    assert partial.token_ids == (1, 3, 6, 7, 2, 0)
    completed = partial.reveal({1: 5})
    assert completed.is_complete and state.accepts_completion(completed)
    assert not state.accepts_completion(partial)
    with pytest.raises(ValueError):
        state.reveal({2: 7})
    with pytest.raises(ValueError):
        state.reveal({1: token_rules.mask_id})


def test_bad_lengths_ids_and_mask_aliases_fail(token_rules):
    for ids in ((1, 5, 2), (1, 5, 10, 2, 0, 0), (1, -1, 2, 0, 0, 0)):
        with pytest.raises(ValueError):
            MaskedState(ids, token_rules)
    with pytest.raises(ValueError):
        replace(token_rules, eos_id=token_rules.mask_id)


def test_decoding_cannot_hide_masks_or_unknowns(tokenizer, token_rules, chemistry_policy):
    reward = QM9Reward(tokenizer=tokenizer, token_rules=token_rules,
                       chemistry_policy=chemistry_policy, tokenizer_sha256="a" * 64)
    incomplete = MaskedState((1, 5, 2, 0, 3, 0), token_rules)
    # A permissive decoder would turn even this incomplete sequence into "C".
    assert tokenizer.decode(incomplete.token_ids, skip_special_tokens=True) == "C"
    result = reward.evaluate(incomplete)
    assert result.status == TerminalStatus.INCOMPLETE
    assert result.score is None and result.reward is None
    with pytest.raises(IncompleteStateError):
        reward.evaluate(incomplete, require_binary=True)
    for ids in ((1, 5, 4, 2, 0, 0), (1, 5, 2, 7, 0, 0),
                (1, 5, 0, 2, 0, 0), (1, 5, 1, 2, 0, 0)):
        invalid = reward.evaluate(MaskedState(ids, token_rules))
        assert invalid.status == TerminalStatus.INVALID_TOKENS
        assert invalid.score == 0 and invalid.reward is None
    valid = reward.evaluate(MaskedState((1, 5, 2, 0, 0, 0), token_rules))
    assert valid.status == TerminalStatus.SCORED and valid.valid
    assert valid.score > 0 and valid.reward is None  # No fabricated calibration.


def test_tiny_model_forward_backward_and_checkpoint_names(tiny_model, token_rules):
    tokens = torch.full((2, 6), token_rules.mask_id, dtype=torch.long)
    output = tiny_model(tokens, timesteps=torch.tensor([0.1, 0.8]), return_dict=True,
                        output_hidden_states=True)
    assert output.logits.shape == (2, 6, token_rules.vocab_size)
    assert len(output.hidden_states) == 3
    loss = torch.nn.functional.cross_entropy(output.logits.flatten(0, 1),
                                             torch.full((12,), 5, dtype=torch.long))
    loss.backward()
    head = tiny_model.backbone.output_layer.linear
    assert torch.isfinite(loss) and head.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in tiny_model.parameters() if p.grad is not None)
    keys = tiny_model.state_dict()
    for name in ("backbone.vocab_embed.embedding", "backbone.sigma_map.mlp.0.weight",
                 "backbone.rotary_emb.inv_freq", "backbone.blocks.0.attn_qkv.weight",
                 "backbone.blocks.0.adaLN_modulation.weight", "backbone.output_layer.linear.weight"):
        assert name in keys


def test_timestep_conditioning_is_explicit(tiny_model, token_rules):
    config = UDLMConfig(vocab_size=token_rules.vocab_size, model_length=6, hidden_dim=32,
                        cond_dim=16, n_blocks=2, n_heads=4, dropout=0, time_conditioning=True)
    conditioned = UDLM(config).eval()
    # The fresh zero output head hides conditioning, so make that path observable.
    torch.nn.init.normal_(conditioned.backbone.output_layer.linear.weight, std=0.1)
    torch.nn.init.normal_(conditioned.backbone.output_layer.adaLN_modulation.weight, std=0.1)
    tokens = torch.full((1, 6), token_rules.mask_id, dtype=torch.long)
    with pytest.raises(ValueError):
        conditioned(tokens)
    first = conditioned(tokens, torch.tensor([0.0]), return_dict=True).logits
    second = conditioned(tokens, torch.tensor([4.0]), return_dict=True).logits
    assert not torch.allclose(first, second)
    torch.nn.init.normal_(tiny_model.backbone.output_layer.linear.weight, std=0.1)
    torch.nn.init.normal_(tiny_model.backbone.output_layer.adaLN_modulation.weight, std=0.1)
    tiny_model.eval()
    assert torch.equal(tiny_model(tokens, torch.tensor([0.0]), return_dict=True).logits,
                       tiny_model(tokens, torch.tensor([4.0]), return_dict=True).logits)


def test_subs_excludes_mask_and_copies_known_positions(token_rules):
    tokens = torch.tensor([[1, 3, 6, 3, 2, 0]])
    logits = torch.randn(1, 6, token_rules.vocab_size)
    probabilities = subs_log_probs(logits, tokens, token_rules.mask_id).exp()
    assert torch.allclose(probabilities.sum(-1), torch.ones(1, 6))
    assert torch.equal(probabilities[..., token_rules.mask_id], torch.zeros(1, 6))
    for position in (0, 2, 4, 5):
        assert probabilities[0, position, tokens[0, position]] == 1
    for position in (1, 3):
        assert torch.all(probabilities[0, position, torch.arange(token_rules.vocab_size) != 3] > 0)


def test_adapter_batches_and_charges_actual_calls(tiny_backbone, token_rules):
    states = [MaskedState((1, 3, 6, 3, 2, 0), token_rules),
              MaskedState((1, 5, 3, 0, 2, 0), token_rules)]
    actual_calls = []
    hook = tiny_backbone.model.register_forward_hook(lambda *args: actual_calls.append(1))
    try:
        budget = BudgetTracker(base_sequence_limit=2)
        result = tiny_backbone.logits_batch(states, budget=budget, phase=CostPhase.COMPLETION)
        assert result.shape == (2, 6, token_rules.vocab_size)
        assert budget.raw_nfe == len(actual_calls) == 1
        assert budget.snapshot().total.base_sequences == 2
        empty = tiny_backbone.logits_batch([], budget=budget, phase=CostPhase.COMPLETION)
        assert empty.shape == (0, 6, token_rules.vocab_size) and budget.raw_nfe == 1
        with pytest.raises(BudgetExceeded):
            tiny_backbone.logits_batch(states, budget=budget, phase=CostPhase.COMPLETION)
        assert len(actual_calls) == 1
    finally:
        hook.remove()


def test_vocab_mismatch_and_state_rule_mismatch_fail(tiny_model, tokenizer, token_rules,
                                                    tiny_backbone):
    with pytest.raises(ValueError):
        QM9Backbone(tiny_model, tokenizer, replace(token_rules, vocab_size=11),
                    tokenizer_sha256="a" * 64)
    other_state = MaskedState((1, 3, 6, 3, 2, 0), replace(token_rules, mask_id=8))
    budget = BudgetTracker(base_sequence_limit=None)
    with pytest.raises(ValueError):
        tiny_backbone.logits_batch([other_state], budget=budget, phase=CostPhase.COMPLETION)
    assert budget.raw_nfe == 0


def test_fake_backbone_uses_supplied_toy_distribution(fake_backbone_factory, token_rules):
    probabilities = [0.0] * token_rules.vocab_size
    probabilities[5], probabilities[6] = 0.25, 0.75
    backbone = fake_backbone_factory(probabilities)
    state = MaskedState((1, 3, 6, 3, 2, 0), token_rules)
    result = backbone.logits_batch([state], budget=BudgetTracker(base_sequence_limit=None),
                                   phase=CostPhase.ESTIMATION)
    assert torch.allclose(result.exp()[0, 1], torch.tensor(probabilities))


def test_checkpoint_resume_reproduces_next_update(tiny_backbone, artifact_directory, monkeypatch):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    settings = {"lr": 3e-4, "min_lr": 3e-6, "warmup_steps": 2, "train_steps": 10}
    optimizer = torch.optim.AdamW(tiny_backbone.model.parameters(), lr=settings["lr"], weight_decay=0)
    scheduler = build_scheduler(optimizer, settings)
    clean = torch.tensor([[1, 5, 6, 7, 2, 0], [1, 6, 5, 7, 2, 0]])

    def update(backbone, opt, schedule):
        backbone.model.train()
        opt.zero_grad(set_to_none=True)
        loss = mdlm_loss(backbone, clean, budget=BudgetTracker(base_sequence_limit=None))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone.model.parameters(), 1.0)
        opt.step()
        schedule.step()
        return loss.detach()

    update(tiny_backbone, optimizer, scheduler)
    original_save = tiny_backbone.model.save_pretrained

    def save_with_rng_side_effects(*args, **kwargs):
        random.random()
        np.random.rand()
        torch.rand(1)
        return original_save(*args, **kwargs)

    monkeypatch.setattr(tiny_backbone.model, "save_pretrained", save_with_rng_side_effects)
    manifest = save_training_checkpoint(
        os.path.join(artifact_directory, "checkpoint"), tiny_backbone, optimizer, scheduler,
        step=1, config={"training": settings}, identities={"dataset": "offline-fixture"},
        environment={"device": "cpu"}, cumulative_base_calls=1,
    )
    saved_config = read_json(os.path.join(os.path.dirname(manifest), "config.json"))
    assert "auto_map" not in saved_config
    expected_python, expected_numpy = random.sample(range(100), 5), np.random.rand(3)
    expected_loss = update(tiny_backbone, optimizer, scheduler)
    expected_weights = {name: tensor.clone() for name, tensor in tiny_backbone.model.state_dict().items()}
    expected_lr = optimizer.param_groups[0]["lr"]

    restored = QM9Backbone.from_checkpoint(manifest)
    state = read_checkpoint(os.path.join(os.path.dirname(manifest), "training_state.pt"))
    resumed_optimizer = torch.optim.AdamW(restored.model.parameters(), lr=settings["lr"], weight_decay=0)
    resumed_scheduler = build_scheduler(resumed_optimizer, settings)
    resumed_optimizer.load_state_dict(state["optimizer"])
    resumed_scheduler.load_state_dict(state["scheduler"])
    restore_training_rng(state["rng"])
    assert random.sample(range(100), 5) == expected_python
    assert np.array_equal(np.random.rand(3), expected_numpy)
    assert torch.equal(update(restored, resumed_optimizer, resumed_scheduler), expected_loss)
    assert resumed_optimizer.param_groups[0]["lr"] == expected_lr
    for name, tensor in restored.model.state_dict().items():
        assert torch.equal(tensor, expected_weights[name])
    # A missing/partial marker and modified files must never load as checkpoints.
    with pytest.raises(FileNotFoundError):
        QM9Backbone.from_checkpoint(os.path.join(artifact_directory, "partial", "manifest.json"))
    with open(os.path.join(os.path.dirname(manifest), "config.json"), "a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError):
        QM9Backbone.from_checkpoint(manifest)
