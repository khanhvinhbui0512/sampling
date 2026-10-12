"""Fixed-work ratios/support; written for review, not executed."""

import math

import pytest

from src.budget import BudgetTracker
from src.randomness import RNGStream
from src.sampling.completion import RhoCompletion
from src.sampling.reference import RhoJointReference
from src.types import MaskedState
from src.verifiers.importance import (conditional_ratio, importance_summary,
                                      native_importance_diagnostic, rho_importance_diagnostic)
from src.verifiers.interface import native_conditional_target, value_target


def test_ratio_retains_failed_denominator_and_refuses_unresolved_values():
    result = conditional_ratio(250, 1000, 500, 1000, min_denominator_hits=10)
    assert result.value == .5 and result.denominator == .5 and result.ess == 500
    assert result.interval[0] <= .5 <= result.interval[1]
    assert conditional_ratio(0, 1000, 0, 1000, min_denominator_hits=1).value is None
    assert conditional_ratio(0, 1000, 2, 1000, min_denominator_hits=1).value is None
    assert conditional_ratio(0, 1000, 500, 1000, min_denominator_hits=501).value is None
    assert conditional_ratio(900, 1000, 500, 1000, min_denominator_hits=1).value is None
    zero_n = conditional_ratio(0, 1000, 500, 1000, log_mass=-math.inf, min_denominator_hits=1)
    assert zero_n.value == 0. and zero_n.interval == (0., 0.)
    underflow = conditional_ratio(1, 1000, 500, 1000, log_mass=-800., min_denominator_hits=1)
    assert underflow.value is None and "underflow" in underflow.error


def test_importance_weights_ess_no_clipping_and_failed_leaves():
    # Equal proposal multiplicities, unnormalized joint masses (3,1,1,3).
    weights = [math.log(w) for w in (3, 1, 1, 3)] * 100
    events = [True, True, True, True] * 100
    rewards = [1, 0, 0, 1] * 100
    result = importance_summary(weights, events, rewards, log_weight_bound=math.log(3), min_ess=10)
    assert result.value == pytest.approx(.75)
    assert result.ess == pytest.approx(320.)
    assert result.interval[0] <= .75 <= result.interval[1]
    assert importance_summary(weights, events, rewards, log_weight_bound=math.log(3), min_ess=321).value is None
    with pytest.raises(ValueError, match="violated bound"):
        importance_summary(weights, events, rewards, log_weight_bound=0., min_ess=1)
    with pytest.raises(FloatingPointError, match="underflow"):
        importance_summary([-800.], [True], [1], log_weight_bound=0., min_ess=1)


def test_rho_joint_correction_differs_from_rho_and_native(toy_backbone, cars_reward_factory, native_case):
    policy, reward = RhoCompletion(toy_backbone), cars_reward_factory()
    reference, query = RhoJointReference(policy), MaskedState((0, 0, 2), policy.rules)
    # q_rho given future 2 has masses .4,.1,.1,.4 over 112,122,212,222.
    leaves = [MaskedState(tokens, policy.rules) for tokens, count in
              [((1, 1, 2), 400), ((1, 2, 2), 100), ((2, 1, 2), 100), ((2, 2, 2), 400)] for _ in range(count)]
    budget = BudgetTracker(base_sequence_limit=None)
    run = rho_importance_diagnostic(reference, reward, query, leaves, budget=budget,
                                    support_verified=True, log_weight_bound=0., min_ess=10)
    assert run.estimate.value == pytest.approx(.7)
    assert run.estimate.value != .5  # q_rho target.
    assert run.estimate.target != value_target(policy, reward)
    assert run.estimate.target != native_conditional_target(*native_case)
    assert run.estimate.cost == budget.snapshot().total and budget.raw_nfe > 0
    with pytest.raises(ValueError, match="support"):
        rho_importance_diagnostic(reference, reward, query, leaves[:1], budget=budget,
                                  support_verified=False, log_weight_bound=0., min_ess=1)


def test_ess_survives_underflow_of_squared_weights():
    # exp(-500) is representable, but its square is not. Equal compatible
    # weights still have ESS=count; uncertainty must independently reject D.
    result = importance_summary([-500.] * 64, [True] * 64, [1] * 64,
                                log_weight_bound=0., min_ess=1)
    assert result.denominator > 0.
    assert result.ess == pytest.approx(64.)
    assert result.value is None and "denominator" in result.error
    empty = importance_summary([-500.] * 64, [False] * 64, [1] * 64,
                               log_weight_bound=0., min_ess=1)
    assert empty.ess == 0. and empty.value is None


def test_native_path_identity_and_scoring_cost(native_case):
    reference, reward = native_case
    budget = BudgetTracker(base_sequence_limit=None)
    paths = [reference.draw(rng=RNGStream(seed), budget=budget) for seed in range(128)]
    before = budget.snapshot().total.base_sequences
    run = native_importance_diagnostic(reference, reward, reference.root, paths, budget=budget,
                                       support_verified=True, log_weight_bound=0., min_ess=1)
    assert run.estimate.target == native_conditional_target(reference, reward)
    assert run.log_weights == pytest.approx([0.] * len(paths))
    assert run.estimate.value == sum(reward.evaluate(path.terminal, require_binary=True).reward for path in paths) / len(paths)
    assert run.estimate.cost.base_sequences == budget.snapshot().total.base_sequences - before
