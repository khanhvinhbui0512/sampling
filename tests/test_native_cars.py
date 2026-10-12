"""Native trajectory mass/pruning/estimation, independent of rho failures."""

import copy
import math

import pytest

from src.budget import BudgetTracker
from src.cars.native import NativeCARSValueEstimator, NativeTrajectoryTrie, native_proposal_draw
from src.randomness import RNGStream
from src.types import EstimateStatus, MaskedState


def fill_path(trie, reference, path, budget):
    for depth in range(len(path)):
        prefix = path[:depth]
        if trie.raw_log_probs(prefix) is None:
            trie.cache_distribution(prefix, reference.cursor(prefix).next_log_probs(
                budget=budget, phase="discovery"))


def test_enumerated_pruning_mass_and_frozen_proposal(native_case, enumerate_native):
    reference, reward = native_case
    query = MaskedState((0, 2), reference.rules)
    trie = NativeTrajectoryTrie(reference, reward, query, max_nodes=5000)
    budget, paths = BudgetTracker(base_sequence_limit=None), enumerate_native(reference)
    n = sum(p for _, leaf, p in paths if query.accepts_completion(leaf) and reward.evaluate(leaf, require_binary=True).reward)
    d = sum(p for _, leaf, p in paths if query.accepts_completion(leaf))
    assert 0 < n < d  # Failed compatible leaves are part of D.
    for path, leaf, _ in paths:
        fill_path(trie, reference, path, budget)
        removed = trie.learn_failure(path, reference, reward, budget=budget)
        if query.accepts_completion(leaf) and reward.evaluate(leaf, require_binary=True).reward:
            assert not removed and trie.log_mass(path) == 0.
    assert math.exp(trie.log_mass()) == pytest.approx(n)
    frozen = trie.freeze()
    proposal_total = sum(p / n for path, _, p in paths if frozen.log_mass(path) != -math.inf)
    assert proposal_total == pytest.approx(1.)
    for seed in range(12):
        draw, _ = native_proposal_draw(reference, frozen, rng=RNGStream(seed), budget=budget)
        assert query.accepts_completion(draw.terminal)
        assert reward.evaluate(draw.terminal, require_binary=True).reward == 1
        assert draw.log_proposal == pytest.approx(draw.log_probability - math.log(n))
    with pytest.raises(ValueError, match="Frozen"):
        frozen.learn_failure((), reference, reward, budget=budget)
    loaded = NativeTrajectoryTrie.from_dict(frozen.to_dict(), reference=reference, reward=reward, state=query)
    assert loaded.snapshot_id == frozen.snapshot_id and loaded.log_mass() == frozen.log_mass()
    with pytest.raises(ValueError, match="identity/schema"):
        NativeTrajectoryTrie.from_dict(frozen.to_dict(), reference=reference, reward=reward, state=reference.root)


def test_mass_zero_is_not_positive_underflow(native_case):
    reference, reward = native_case
    query = MaskedState((1, 0), reference.rules)
    trie = NativeTrajectoryTrie(reference, reward, query, max_nodes=10)
    # Arithmetic unit test: prune the dominant incompatible edge, retain e^-800.
    trie.cache_distribution((), (-math.inf, -800., 0.))
    trie.learn_failure((2,), reference, reward, budget=BudgetTracker(base_sequence_limit=None))
    assert trie.log_mass() == -800. and trie.mass_underflow and not trie.support_exhausted
    trie.cache_distribution((1,), (-math.inf, 0., -math.inf))
    trie.learn_failure((1, 1), reference, reward, budget=BudgetTracker(base_sequence_limit=None))
    assert not trie.support_exhausted  # A successful compatible trajectory cannot be removed.
    zero = NativeTrajectoryTrie(reference, reward, query, max_nodes=10)
    zero.cache_distribution((), (-math.inf, -math.inf, 0.))
    zero.learn_failure((2,), reference, reward, budget=BudgetTracker(base_sequence_limit=None))
    assert zero.log_mass() == -math.inf and zero.support_exhausted and not zero.mass_underflow


def test_learning_rejects_reference_and_calibration_drift(native_case_factory):
    reference, reward = native_case_factory()
    trie = NativeTrajectoryTrie(reference, reward, reference.root, max_nodes=10)
    budget = BudgetTracker(base_sequence_limit=None)
    snapshot, before = trie.to_dict(), budget.snapshot().total
    other_reference, other_reward = native_case_factory(steps=3)
    with pytest.raises(ValueError, match="reference context"):
        trie.learn_failure((), other_reference, other_reward, budget=budget)
    changed_reward = copy.copy(reward)
    changed_reward.calibration = copy.copy(reward.calibration)
    changed_reward.calibration.threshold = .9  # Same calibration ID is insufficient.
    with pytest.raises(ValueError, match="calibration context"):
        trie.learn_failure((), reference, changed_reward, budget=budget)
    with pytest.raises(ValueError, match="identity/schema"):
        NativeTrajectoryTrie.from_dict(trie.freeze().to_dict(), reference=reference,
                                       reward=changed_reward, state=reference.root)
    assert trie.to_dict() == snapshot and budget.snapshot().total == before


def test_proposal_rejects_edge_underflow_even_with_ordinary_root_mass(native_case):
    reference, reward = native_case
    trie = NativeTrajectoryTrie(reference, reward, reference.root, max_nodes=10)
    # Arithmetic fixture: half the root mass remains, but one retained child
    # cone has relative mass exp(-800). This is not a model likelihood fixture.
    trie.cache_distribution((), (-math.inf, math.log(.5), math.log(.5)))
    trie.cache_distribution((2,), (-math.inf, 0., -800.))
    budget = BudgetTracker(base_sequence_limit=None)
    trie.learn_failure((2, 1), reference, reward, budget=budget)
    assert math.exp(trie.log_mass()) == pytest.approx(.5) and not trie.mass_underflow
    rng = RNGStream(3)
    before = rng.state_dict()
    with pytest.raises(FloatingPointError, match="edge underflowed"):
        native_proposal_draw(reference, trie.freeze(), rng=rng, budget=budget)
    assert rng.state_dict() == before


def test_numerical_failure_is_unavailable_and_releases_budget(native_case, monkeypatch):
    reference, reward = native_case
    estimator = NativeCARSValueEstimator(reference, reward, discovery_attempts=0, numerator_draws=4,
                                        denominator_draws=4, max_nodes=10, min_denominator_hits=1)

    def failed_draw(*args, **kwargs):
        raise FloatingPointError("Positive native proposal edge underflowed; support would be lost")

    monkeypatch.setattr("src.cars.native.native_proposal_draw", failed_draw)
    budget = BudgetTracker(base_sequence_limit=estimator.required_work(reference.root))
    run = estimator.estimate(reference.root, rng=RNGStream(3), budget=budget, progress=False)
    assert run.estimate.status == EstimateStatus.FAILED and run.estimate.value is None
    assert "underflowed" in run.estimate.error
    assert budget.snapshot().reserved_base_sequences == 0
    assert set(run.rng_state) == {"discovery", "numerator", "denominator"}


def test_fixed_split_denominator_and_budget(native_case, enumerate_native):
    reference, reward = native_case
    estimator = NativeCARSValueEstimator(reference, reward, discovery_attempts=8, numerator_draws=256,
                                        denominator_draws=512, max_nodes=5000, min_denominator_hits=5)
    query = MaskedState((0, 2), reference.rules)
    budget = BudgetTracker(base_sequence_limit=None)
    run = estimator.estimate(query, rng=RNGStream(7), budget=budget, progress=False)
    assert len([row for row in run.estimation_trace if row["stage"] == "denominator"]) == 512
    assert len([row for row in run.estimation_trace if row["stage"] == "numerator"]) == 256
    assert set(run.rng_state) == {"discovery", "numerator", "denominator"}
    assert run.rng_state["numerator"]["seed"] != run.rng_state["denominator"]["seed"]
    assert run.metadata["denominator"] == "unpruned_native_root"
    assert run.estimate.surviving_mass is None and run.estimate.draws == 768
    assert run.estimate.cost == budget.snapshot().total
    paths = enumerate_native(reference)
    d = sum(p for _, leaf, p in paths if query.accepts_completion(leaf))
    n = sum(p for _, leaf, p in paths if query.accepts_completion(leaf) and reward.evaluate(leaf, require_binary=True).reward)
    assert run.estimate.value is not None and run.estimate.interval[0] <= n / d <= run.estimate.interval[1]
    short = BudgetTracker(base_sequence_limit=estimator.required_work(query) - 1)
    rng = RNGStream(7)
    before = rng.state_dict()
    refused = estimator.estimate(query, rng=rng, budget=short, progress=False)
    assert refused.estimate.status == EstimateStatus.BUDGET_EXHAUSTED
    assert short.raw_nfe == 0 and rng.state_dict() == before


def test_unreachable_observation_and_capacity_never_invent_zero(native_case_factory):
    reference, reward = native_case_factory(zero_second_two=True)
    query = MaskedState((0, 2), reference.rules)
    estimator = NativeCARSValueEstimator(reference, reward, discovery_attempts=4, numerator_draws=8,
                                        denominator_draws=16, max_nodes=1, min_denominator_hits=1)
    run = estimator.estimate(query, rng=RNGStream(9), budget=BudgetTracker(base_sequence_limit=None), progress=False)
    assert run.estimate.value is None and run.summary.denominator == 0.
    assert "denominator" in run.estimate.error
    assert run.metadata["capacity_reached"] and len(run.trie.nodes) == 1


def test_long_native_path_mass_uses_no_recursive_stack(native_case_factory):
    reference, reward = native_case_factory(steps=600)
    trie = NativeTrajectoryTrie(reference, reward, reference.root, max_nodes=1500)
    # Arithmetic stress case with a supported all-MASK timing prefix. This
    # checks the mass engine, not a claimed model likelihood for these rows.
    path = ()
    for _ in range(1200):
        trie.cache_distribution(path, (0., -math.inf, -math.inf))
        path += (0,)
    assert trie.log_mass() == 0. and not trie.support_exhausted
