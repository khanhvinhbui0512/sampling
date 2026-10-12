"""Independent finite-oracle checks; no downloaded checkpoints or corpus.

The mathematical reward below is deliberately not benchmark chemistry. The
native grammar still checks token prefixes in tests of the production adapter.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from src.cars import Feedback
from src.completion import CompletionResult, normalize_eos_suffix
from src.reference import estimate_completion_value, q_z_reference, rollout_reference
from src.reward import RewardChecker
from src.schemas import GROUPS, RewardResult, SmilesState
from src.tokenizer import SmilesTokenizer
from src.tracking import BudgetExceeded, CostTracker
from src.values import AnchoredValueProvider, CARSValueProvider, LearnedValueProvider, guidance_value


@pytest.fixture
def tiny_tokenizer():
    return SmilesTokenizer(("<pad>", "<mask>", "<eos>", "C", "O"), max_length=4)


class OracleChecker(RewardChecker):
    """Finite fixture: exactly CO has reward 1, all other leaves have reward 0."""

    def evaluate_state(self, state, tokenizer):
        self._count("reward_checks")
        good = state.terminal_status == "complete" and state.token_ids == (3, 4, 2, 0)
        return RewardResult(float(good), self.version, state.terminal_status == "complete", True, good,
                            canonical_smiles="CO" if good else None, error=None if good else "finite_oracle_zero")


@pytest.fixture
def oracle_checker():
    return OracleChecker()


class FiniteTree:
    metadata = {"policy": "finite-joint-left-to-right", "failed_cone": "first token O"}

    def __init__(self, model, tokenizer, checker):
        self.model, self.tokenizer, self.checker = model, tokenizer, checker

    def context(self, root):
        return {"model": self.model.model_id, "tokenizer": self.tokenizer.fingerprint, "state": root.to_dict(),
                "group": root.group, "reward": "finite-CO-oracle-v1", "policy": self.metadata}

    def normalize(self, state):
        return normalize_eos_suffix(state)

    def begin(self):
        pass

    def terminal(self, state):
        return state.is_terminal

    def feedback(self, state):
        if state.token_ids[0] == 4:
            return Feedback(True, reason="finite oracle proves first-O cone has reward zero")
        return Feedback(forbidden=torch.tensor([4])) if state.mask[0] else Feedback()

    def next_logs(self, state):
        return self.model.logits_batch([state])[0, state.masked_positions[0]].log_softmax(-1)

    def advance(self, state, token):
        return self.normalize(state.with_tokens({state.masked_positions[0]: token}))

    def output(self, state):
        return self.tokenizer.decode_state(state), self.checker.evaluate_state(state, self.tokenizer)


def finite_provider(model, tokenizer, checker, **kwargs):
    return CARSValueProvider(tokenizer, checker, policy_factory=lambda state: FiniteTree(model, tokenizer, checker),
                             policy_context={"model": model.model_id, **FiniteTree.metadata}, **kwargs)


def force_draws(monkeypatch, tokens):
    draws = iter(tokens)

    def sample(weights, count, *, generator, **kwargs):
        assert weights.ndim == 1 and count == 1
        token = next(draws)
        assert weights[token] > 0
        return torch.tensor([token])

    monkeypatch.setattr(torch, "multinomial", sample)


@pytest.mark.parametrize("tokens,contributions", [([3, 3, 3, 4], [0., .2]),
                                                  ([3, 4, 3, 3], [.3, 0.]),
                                                  ([3, 4, 3, 4], [.3, .3])])
def test_fixed_m_uses_pre_draw_mass_and_keeps_failures(toy_model, tiny_tokenizer, oracle_checker,
                                                      toy_initial_state, monkeypatch, tokens, contributions):
    provider = finite_provider(toy_model, tiny_tokenizer, oracle_checker, m=2)
    force_draws(monkeypatch, tokens)
    value = provider.value(toy_initial_state)
    assert provider.last_estimate["contributions"] == pytest.approx(contributions)
    assert provider.last_estimate["m"] == provider.last_estimate["actual_attempts"] == 2
    assert value == pytest.approx(math.fsum(contributions)/2)


def test_exact_expectation_over_all_adaptive_two_attempt_histories(toy_distribution):
    # Initial C-only cone has mass .3. CC occurs with probability 1/3;
    # after rejecting CC, only CO remains, with unconditional mass .2.
    probabilities = (1/3, (2/3)*(1/3), (2/3)*(2/3))
    estimates = ((0.+.2)/2, (.3+0.)/2, (.3+.3)/2)
    assert math.fsum(probabilities) == pytest.approx(1.)
    assert math.fsum(p*v for p, v in zip(probabilities, estimates, strict=True)) == pytest.approx(.2)
    assert toy_distribution.exact_value(SmilesState((1, 1, 2, 0), (True, True, False, False),
                                                   GROUPS[0], toy_distribution.special_tokens)) == pytest.approx(.2)


def test_success_does_not_stop_fixed_m_estimator(toy_model, tiny_tokenizer, oracle_checker, toy_initial_state, monkeypatch):
    provider = finite_provider(toy_model, tiny_tokenizer, oracle_checker, m=3)
    force_draws(monkeypatch, [3, 4] * 3)
    assert provider.value(toy_initial_state) == pytest.approx(.3)
    assert provider.last_estimate["contributions"] == pytest.approx([.3]*3)
    assert provider.last_estimate["standard_error"] is None


def test_proven_exhaustion_keeps_m_terms_without_imaginary_attempts(toy_model, tiny_tokenizer,
                                                                  oracle_checker, toy_initial_state):
    tracker = CostTracker(run_id="zero-cone")
    provider = finite_provider(toy_model, tiny_tokenizer, oracle_checker, m=8, tracker=tracker)
    assert provider.value(toy_initial_state.with_tokens({0: 4})) == 0.
    assert provider.last_estimate["contributions"] == [0.]*8
    assert provider.last_estimate["actual_attempts"] == tracker.counters["cars_inner_attempts"] == 1
    assert toy_model.forward_calls == tracker.raw_nfe == 0


class FakeVerifier:
    model_id = "finite-neural-stub"
    target_metadata = {"target_kind": "reference_rollout_return", "target_definition": "fixture",
                       "reward_version": "grammar-rdkit-smarts-v1", "source_mdlm_sha256": "a"*64,
                       "policy_hash": "b"*64, "suite": "smoke"}

    def __init__(self):
        self.calls = 0

    def predict_batch(self, states):
        self.calls += 1
        assert all(not state.is_terminal for state in states)
        return [.77] * len(states)


def learned_provider(tokenizer, checker, **kwargs):
    verifier = FakeVerifier()
    provider = LearnedValueProvider(verifier, tokenizer, checker, source_mdlm_sha256="a"*64, suite="smoke", **kwargs)
    return provider, verifier


def test_terminal_values_are_exact_and_never_floored(toy_model, tiny_tokenizer, oracle_checker, toy_terminal_states):
    learned, verifier = learned_provider(tiny_tokenizer, oracle_checker)
    cars = finite_provider(toy_model, tiny_tokenizer, oracle_checker, m=16)
    for provider in (learned, cars):
        assert provider.values(toy_terminal_states) == [0., 1., 0., 0.]
        assert guidance_value(provider.value(toy_terminal_states[0]), toy_terminal_states[0], .1) == 0.
    assert verifier.calls == 0 and toy_model.forward_calls == 0


def test_learned_batches_unique_nonterminal_queries_and_bounds_cache(tiny_tokenizer, oracle_checker, toy_initial_state):
    provider, verifier = learned_provider(tiny_tokenizer, oracle_checker, cache_capacity=1)
    assert provider.values([toy_initial_state, toy_initial_state]) == [.77, .77]
    assert verifier.calls == 1
    provider.value(toy_initial_state.with_tokens({0: 3}))
    assert len(provider.state_dict()["cache"]) == 1
    provider.value(toy_initial_state)
    assert verifier.calls == 3


@pytest.mark.parametrize("field", ["source_mdlm_sha256", "reward_version", "suite", "target_kind"])
def test_learned_provider_rejects_incompatible_training_provenance(tiny_tokenizer, oracle_checker, field):
    verifier = FakeVerifier()
    verifier.target_metadata = {**verifier.target_metadata, field: "different"}
    with pytest.raises(ValueError, match="incompatible"):
        LearnedValueProvider(verifier, tiny_tokenizer, oracle_checker, source_mdlm_sha256="a"*64, suite="smoke")


def test_value_cache_context_binds_model_reward_policy_and_full_state(tiny_tokenizer, oracle_checker, toy_initial_state):
    first, _ = learned_provider(tiny_tokenizer, oracle_checker)
    first.value(toy_initial_state)
    changed = dict(first.metadata)
    changed["training_target"] = {"different": True}
    other = AnchoredValueProvider(tiny_tokenizer, oracle_checker, context=changed, target="different")
    with pytest.raises(ValueError, match="context"):
        other.load_state_dict(first.state_dict())
    assert first._key(toy_initial_state) != first._key(toy_initial_state.with_tokens({1: 4}))
    assert guidance_value(0., toy_initial_state, .1) == .1


def test_cars_provider_resume_replays_rng_and_memoized_estimate(toy_model, tiny_tokenizer, oracle_checker, toy_initial_state):
    provider = finite_provider(toy_model, tiny_tokenizer, oracle_checker, m=4, cache_capacity=1, seed=42)
    value = provider.value(toy_initial_state)
    payload = provider.state_dict()
    restored = finite_provider(toy_model, tiny_tokenizer, oracle_checker, m=4, cache_capacity=1, seed=42)
    restored.load_state_dict(payload)
    assert restored.value(toy_initial_state) == value
    partial = toy_initial_state.with_tokens({0: 3})
    assert provider.value(partial) == restored.value(partial)
    assert provider.value(toy_initial_state) == restored.value(toy_initial_state)


class CountedModel:
    def __init__(self, model, tracker, *, inner=False):
        self.base, self.tracker, self.inner = model, tracker, inner
        self.model_id, self.vocab_size, self.special_tokens = model.model_id, model.vocab_size, model.special_tokens

    def logits_batch(self, states):
        with self.tracker.model_call("mdlm", batch_size=len(states), inner=self.inner):
            return self.base.logits_batch(states)


def test_inner_budget_cannot_return_partial_or_floored_estimate(toy_model, tiny_tokenizer, oracle_checker,
                                                               toy_initial_state, monkeypatch):
    tracker = CostTracker(run_id="inner-budget", max_model_nfe=1)
    model = CountedModel(toy_model, tracker, inner=True)
    provider = finite_provider(model, tiny_tokenizer, oracle_checker, m=16, tracker=tracker)
    force_draws(monkeypatch, [3])
    with pytest.raises(BudgetExceeded):
        provider.value(toy_initial_state)
    assert not provider.state_dict()["cache"] and provider.last_estimate is None
    assert tracker.raw_nfe == tracker.counters["inner_mdlm_forward_calls"] == 1


def test_known_suffix_completions_and_reference_uncertainty(toy_model, tiny_tokenizer, oracle_checker, toy_initial_state):
    partial = toy_initial_state.with_tokens({1: 4})
    result = q_z_reference(toy_model, tiny_tokenizer, oracle_checker, partial, m=8, seed=123)
    assert result.completions == 8 and len(result.rewards) == 8 and result.target == "V_q"
    assert 0 <= result.confidence_interval[0] <= result.value <= result.confidence_interval[1] <= 1
    terminal = partial.with_tokens({0: 3})
    before = toy_model.forward_calls
    exact = q_z_reference(toy_model, tiny_tokenizer, oracle_checker, terminal, m=8)
    assert exact.exact_terminal and exact.value == 1. and exact.completions == 0
    assert toy_model.forward_calls == before


def test_reference_rollout_is_labeled_separately_from_q_z(tiny_tokenizer, oracle_checker, toy_initial_state):
    class ScriptedPolicy:
        def __init__(self, policy, tokens):
            self.metadata = {"policy": policy}
            self.final = toy_initial_state.with_tokens(tokens)

        def complete(self, state):
            return CompletionResult(self.final, (), 0., "fixture", "fixture")

    q = estimate_completion_value(ScriptedPolicy("left_to_right-v1", {0: 3, 1: 3}), toy_initial_state,
                                 tiny_tokenizer, oracle_checker, m=4)
    rollout = estimate_completion_value(ScriptedPolicy("absorbing_reverse-v1", {0: 3, 1: 4}), toy_initial_state,
                                       tiny_tokenizer, oracle_checker, m=4)
    assert q.target == "V_q" and q.value == 0.
    assert rollout.target == "reference_rollout_return" and rollout.value == 1.
    assert not q.to_dict()["ideal_value_claim"] and not rollout.to_dict()["ideal_value_claim"]


def test_scheduled_reference_preserves_known_suffix(toy_model, tiny_tokenizer, oracle_checker, toy_initial_state):
    class ScheduledModel:
        device = torch.device("cpu")
        model = SimpleNamespace(specification={"diffusion": {"epsilon": 1e-5}})
        model_id, vocab_size, special_tokens = toy_model.model_id, toy_model.vocab_size, toy_model.special_tokens

        def probabilities_batch(self, states, *, times):
            return toy_model.logits_batch(states).softmax(-1)

    state = toy_initial_state.with_tokens({1: 4})
    estimate = rollout_reference(ScheduledModel(), tiny_tokenizer, oracle_checker, state, m=4, steps=2, seed=8)
    assert estimate.target == "reference_rollout_return" and estimate.completions == 4
