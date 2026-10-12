"""Independent network-call records test nested accounting and hard caps."""

from types import SimpleNamespace

import pytest
import torch

from src.budget import BudgetTracker, MeasuredFlopRatio
from src.cars.cache import CARSCache
from src.cars.estimator import CARSValueEstimator
from src.cars.oracle import FiniteLanguageOracle
from src.randomness import RNGStream
from src.sampling.base import BaseSampler
from src.sampling.candidates import CandidateSettings
from src.sampling.completion import RhoCompletion
from src.sampling.reference import RhoJointReference
from src.sampling.terminal import backtrack
from src.sampling.vgb import VGBSampler
from src.types import CostPhase, EstimateStatus, MaskedState, SampleStatus, TokenRules
from src.verifiers.cars import CARSVerifier
from src.verifiers.importance import rho_importance_diagnostic
from src.verifiers.interface import value_target
from src.verifiers.monte_carlo import MonteCarloVerifier
from src.verifiers.neural import NeuralVerifier, TokenStateVerifier


@pytest.fixture
def counted_case():
    rules = TokenRules(3, 2, mask_id=0)

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls, self.phase, self.fail = [], None, False

        def forward(self, tokens):
            self.calls.append((self.phase.value, len(tokens)))
            if self.fail:
                raise RuntimeError("injected network failure")
            return torch.tensor([0., .7, .3]).log().expand(len(tokens), 2, 3).clone()

    class Backbone:
        checkpoint_id, tokenizer_id = "counted-weights", "counted-tokenizer"
        reference_id, time_conditioning = "counted-model", False
        token_rules = rules

        def __init__(self):
            self.model = Network()

        def logits_batch(self, states, *, budget, phase, reservation=None):
            tokens = torch.tensor([state.token_ids for state in states])
            with budget.base_forward(batch_size=len(states), phase=phase, reservation=reservation):
                self.model.phase = phase
                return self.model(tokens)

    backbone = Backbone()

    class Reward:
        token_rules, tokenizer_sha256 = rules, backbone.tokenizer_id
        calibration = SimpleNamespace(calibration_id="counted-pass95", threshold=.5,
                                      checkpoint_sha256=backbone.checkpoint_id)

        def evaluate(self, state, *, require_binary=True):
            assert state.is_complete and require_binary
            return SimpleNamespace(reward=1, valid=True, score=1., molecule=None)

    reward, policy = Reward(), RhoCompletion(backbone)
    root = MaskedState((0, 0), rules)
    oracle = FiniteLanguageOracle(rules, [(1, 1), (1, 2), (2, 1), (2, 2)])
    return backbone, reward, policy, root, oracle


def assert_actual_calls(backbone, budget):
    calls, snapshot = backbone.model.calls, budget.snapshot()
    assert snapshot.total.base_calls == len(calls)
    assert snapshot.total.base_sequences == sum(size for _, size in calls)
    assert snapshot.base_batch_sizes == tuple(sorted({size for _, size in calls}))
    for phase in {phase for phase, _ in calls}:
        actual = [size for label, size in calls if label == phase]
        assert snapshot.by_phase[phase].base_calls == len(actual)
        assert snapshot.by_phase[phase].base_sequences == sum(actual)
    assert sum(record.base_calls for record in snapshot.by_phase.values()) == len(calls)
    assert snapshot.reserved_base_sequences == 0


def test_mc_batching_counts_calls_and_sequences_separately(counted_case):
    backbone, reward, policy, root, _ = counted_case
    budget = BudgetTracker(base_sequence_limit=10)
    verifier = MonteCarloVerifier(policy, reward, draws=5, draw_batch_size=2)
    estimate = verifier.values_for_states([root], rng=RNGStream(1), budget=budget)[0]
    assert estimate.value == 1. and estimate.draws == 5
    # Two reveal sites for each of batches 2, 2, 1: six calls, ten sequences.
    assert [size for _, size in backbone.model.calls] == [2, 2, 2, 2, 1, 1]
    assert estimate.cost.base_calls == 6 and estimate.cost.base_sequences == 10
    assert_actual_calls(backbone, budget)


@pytest.mark.parametrize("name", ["mc", "cars", "cars-no-discovery"])
def test_vgb_nested_queries_and_reserved_completion_are_all_counted(counted_case, name):
    backbone, reward, policy, root, oracle = counted_case
    if name == "mc":
        verifier = MonteCarloVerifier(policy, reward, draws=3, draw_batch_size=2)
    else:
        estimator = CARSValueEstimator(policy, reward, oracle, mode="cars",
                                       discovery_attempts=2 if name == "cars" else 0, draws=3, max_nodes=100)
        cache = CARSCache(estimator.cache_metadata, max_entries=2, run_id="accounting")
        verifier = CARSVerifier(estimator, cache=cache, progress=False)
    sampler = VGBSampler(backbone, verifier, BaseSampler(backbone, reward, method="base_rho"),
                         candidates=CandidateSettings(L_f=1, L_b=1, K=1), step_limit=1)
    budget = BudgetTracker(base_sequence_limit=100)
    run = sampler.run(root, sample_id="nested", rng=RNGStream(2), budget=budget)
    assert run.result.status == SampleStatus.COMPLETED and run.metadata["fallback_used"]
    assert run.result.cost.base_calls == len(backbone.model.calls)
    phases = {phase for phase, _ in backbone.model.calls}
    assert {"sampling", "completion"}.issubset(phases)
    if name != "cars":
        assert "estimation" in phases
    if name == "cars":
        assert "discovery" in phases
        # Frozen proposals can reuse raw logits already obtained in discovery;
        # cache reads incur no forward and must not be charged as one.
    assert_actual_calls(backbone, budget)


@pytest.mark.parametrize("name", ["mc", "cars"])
def test_query_refusal_preserves_rng_and_completion_reserve(counted_case, name):
    backbone, reward, policy, root, oracle = counted_case
    if name == "mc":
        verifier = MonteCarloVerifier(policy, reward, draws=3, draw_batch_size=2)
    else:
        estimator = CARSValueEstimator(policy, reward, oracle, mode="cars",
                                       discovery_attempts=1, draws=3, max_nodes=100)
        verifier = CARSVerifier(estimator, cache=CARSCache(estimator.cache_metadata,
                                max_entries=1, run_id="refusal"), progress=False)
    budget, rng = BudgetTracker(base_sequence_limit=7), RNGStream(7)
    before = rng.state_dict()
    with budget.reserve_completion(base_sequences=2) as held:
        estimate = verifier.values_for_states([root], rng=rng, budget=budget)[0]
        assert estimate.status == EstimateStatus.BUDGET_EXHAUSTED and estimate.value is None
        assert backbone.model.calls == [] and rng.state_dict() == before
        batch = policy.complete_batch([root], rng=rng, budget=budget,
                                      phase=CostPhase.COMPLETION, reservation=held)
        assert batch.states[0].is_complete
    assert_actual_calls(backbone, budget)


def test_neural_batch_counter_matches_actual_module_and_keeps_query_order(counted_case):
    backbone, reward, policy, root, _ = counted_case
    model = TokenStateVerifier(3, 2, hidden_dim=8, num_layers=1, num_heads=2, dropout=0.)
    verifier = NeuralVerifier(model, reward, value_target(policy, reward))
    inputs = []
    hook = model.register_forward_pre_hook(lambda module, args: inputs.append(args[0].clone()))
    budget = BudgetTracker(base_sequence_limit=0, verifier_sequence_limit=2)
    leaf, child = MaskedState((1, 2), root.rules), MaskedState((1, 0), root.rules)
    try:
        values = verifier.values_for_states([root, leaf, child], rng=RNGStream(5), budget=budget)
    finally:
        hook.remove()
    assert len(inputs) == 1 and torch.equal(inputs[0], torch.tensor([[0, 0], [1, 0]]))
    assert values[1].status == EstimateStatus.EXACT and values[1].value == 1.
    assert budget.snapshot().total.verifier_calls == len(inputs)
    assert budget.snapshot().total.verifier_sequences == sum(len(tokens) for tokens in inputs)
    assert sum(value.cost.verifier_calls for value in values) == len(inputs)
    assert budget.raw_nfe == 0 and backbone.model.calls == []
    assert budget.snapshot().adjusted_nfe is None


def test_importance_replay_costs_real_calls_and_backtrack_costs_none(counted_case):
    backbone, reward, policy, root, _ = counted_case
    budget = BudgetTracker(base_sequence_limit=20)
    leaf = policy.complete_batch([root], rng=RNGStream(3), budget=budget,
                                 phase=CostPhase.SAMPLING).states[0]
    diagnostic = rho_importance_diagnostic(RhoJointReference(policy), reward, root, [leaf],
                                           budget=budget, support_verified=True,
                                           log_weight_bound=0., min_ess=1)
    assert diagnostic.summary.numerator == diagnostic.summary.denominator == 1.
    assert diagnostic.estimate.value is None  # One draw cannot resolve this confidence bound.
    # Score q from root and pi from root independently: four extra model calls.
    assert [phase for phase, _ in backbone.model.calls].count("importance") == 4
    before = budget.snapshot().total
    parent = backtrack(leaf, root.masked_positions, RNGStream(4))
    assert len(parent.masked_positions) == 1 and budget.snapshot().total == before
    assert_actual_calls(backbone, budget)


def test_failed_forward_is_charged_and_unused_reservation_released(counted_case):
    backbone, _, _, root, _ = counted_case
    budget = BudgetTracker(base_sequence_limit=2)
    backbone.model.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        with budget.reserve_query(label="failure", base_sequences=2) as held:
            backbone.logits_batch([root], budget=budget, phase=CostPhase.ESTIMATION, reservation=held)
    assert len(backbone.model.calls) == 1
    assert_actual_calls(backbone, budget)
    with budget.reserve_completion(base_sequences=1) as held:
        with pytest.raises(ValueError, match="cannot fund phase"):
            backbone.logits_batch([root], budget=budget, phase=CostPhase.DISCOVERY, reservation=held)
    assert len(backbone.model.calls) == 1


def test_adjusted_nfe_requires_matching_measured_batch_sizes():
    ratio = MeasuredFlopRatio(.25, base_batch_size=2, verifier_batch_size=1, measurement_id="measured-toy")
    budget = BudgetTracker(base_sequence_limit=None, flop_ratio=ratio)
    with budget.base_forward(batch_size=2, phase=CostPhase.SAMPLING):
        pass
    with budget.verifier_forward(batch_size=1, phase=CostPhase.VERIFICATION):
        pass
    assert budget.snapshot().adjusted_nfe == 1.25
    with budget.base_forward(batch_size=1, phase=CostPhase.SAMPLING):
        pass
    assert budget.snapshot().adjusted_nfe is None
