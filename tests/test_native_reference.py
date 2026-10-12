"""Step 9 native path enumeration; written but not executed."""

import math
from collections import defaultdict
from itertools import product

import pytest
import torch

from src.budget import BudgetTracker
from src.models.backbone import loglinear_sigma
from src.randomness import RNGStream
from src.sampling.completion import prediction_log_probs
from src.sampling.native import NativeMDLM, absorbing_posterior
from src.sampling.native_reference import NativeReference
from src.types import CostPhase, MaskedState


def transition_enumeration(reference, start):
    """Independent whole-parallel-step recursion, without the path cursor."""
    law = {start: 1.}
    policy = reference.policy
    for step, time in enumerate(reference.times):
        next_law = defaultdict(float)
        for state, mass in law.items():
            if state.is_complete:
                next_law[state] += mass
                continue
            sigma = loglinear_sigma(torch.full((1,), time), policy.noise_eps)
            logits = reference.backbone.logits_at_noise_batch(
                [state], sigma, budget=BudgetTracker(base_sequence_limit=None), phase=CostPhase.ESTIMATION)
            theta = prediction_log_probs(logits, state.rules.mask_id).exp()
            posterior = absorbing_posterior(theta, torch.tensor([state.token_ids]), state.rules.mask_id,
                                           (1 - policy.noise_eps) * time,
                                           0. if step == policy.steps else
                                           (1 - policy.noise_eps) * reference.times[step + 1])[0]
            for tokens in product(range(state.rules.vocab_size), repeat=state.rules.sequence_length):
                probability = math.prod(float(posterior[i, t]) for i, t in enumerate(tokens))
                if probability:
                    next_law[MaskedState(tokens, state.rules)] += mass * probability
        law = next_law
    return law


@pytest.mark.parametrize("time_conditioning", [False, True])
def test_parallel_joint_paths_all_conditionals_and_log_scoring(native_case_factory, enumerate_native,
                                                               time_conditioning):
    reference, reward = native_case_factory(time_conditioning=time_conditioning)
    paths, joint = enumerate_native(reference), defaultdict(float)
    for path, terminal, probability in paths:
        joint[terminal] += probability
        log_p = reference.log_path_probability(path, budget=BudgetTracker(base_sequence_limit=None))
        assert math.exp(log_p) == pytest.approx(probability, abs=1e-12)
    assert sum(joint.values()) == pytest.approx(1.)
    assert dict(joint) == pytest.approx(dict(transition_enumeration(reference, reference.root)))
    assert len(paths) > len(joint)  # Multiple MASK/reveal timings reach one terminal.
    assert any(0 in path[:2] for path, _, _ in paths)
    for tokens in product(range(3), repeat=2):
        query = MaskedState(tokens, reference.rules)
        d = sum(p for leaf, p in joint.items() if query.accepts_completion(leaf))
        n = sum(p * reward.evaluate(leaf, require_binary=True).reward
                for leaf, p in joint.items() if query.accepts_completion(leaf))
        direct_d = sum(p for _, leaf, p in paths if query.accepts_completion(leaf))
        assert direct_d == pytest.approx(d)
        assert 0 <= n <= d + 1e-12
        if d:
            assert 0 <= n / d <= 1 + 1e-12


def test_logits_stay_at_pre_step_and_endpoint_is_not_dropped(native_case):
    reference, _ = native_case
    cursor, tracker = reference.cursor(), BudgetTracker(base_sequence_limit=None)
    first = cursor.next_log_probs(budget=tracker, phase=CostPhase.ESTIMATION)
    cursor.advance(1)
    second = cursor.next_log_probs(budget=tracker, phase=CostPhase.ESTIMATION)
    assert tracker.raw_nfe == 1 and reference.backbone.inputs == [(0, 0)]
    assert math.exp(first[1]) / (1 - math.exp(first[0])) == pytest.approx(.7)
    assert math.exp(second[1]) / (1 - math.exp(second[0])) == pytest.approx(.3)
    # The partial within-step observation (1,0) must NOT change .3 to .8.
    cursor.advance(0)
    assert cursor.pre.token_ids == (1, 0)
    loop_path = (0, 0) * reference.policy.steps + (1, 2)
    assert reference.cursor(loop_path).done
    assert math.isfinite(reference.log_path_probability(loop_path, budget=tracker))
    with pytest.raises(ValueError, match="full terminal"):
        reference.log_path_probability(loop_path[:-1], budget=tracker)
    with pytest.raises(ValueError, match="final denoise"):
        NativeReference(NativeMDLM(reference.backbone, final_denoise=False))


def test_root_conditional_differs_from_fresh_native_completion(native_case, enumerate_native):
    reference, reward = native_case
    query = MaskedState((0, 2), reference.rules)
    paths = enumerate_native(reference)
    compatible = [(leaf, p) for _, leaf, p in paths if query.accepts_completion(leaf)]
    conditional = sum(p * reward.evaluate(leaf, require_binary=True).reward for leaf, p in compatible) / sum(p for _, p in compatible)
    fresh_law = transition_enumeration(reference, query)
    fresh = sum(p * reward.evaluate(leaf, require_binary=True).reward for leaf, p in fresh_law.items())
    assert conditional != pytest.approx(fresh, abs=1e-3)


def test_zero_support_and_seeded_draws(native_case_factory, enumerate_native):
    reference, _ = native_case_factory(zero_second_two=True)
    impossible = MaskedState((0, 2), reference.rules)
    assert sum(p for _, leaf, p in enumerate_native(reference) if impossible.accepts_completion(leaf)) == 0.
    assert reference.log_path_probability((1, 2), budget=BudgetTracker(base_sequence_limit=None)) == -math.inf
    kwargs = dict(budget=BudgetTracker(base_sequence_limit=None))
    assert reference.draw(rng=RNGStream(7), **kwargs) == reference.draw(rng=RNGStream(7), **kwargs)


def test_native_reference_rejects_mutated_schedule(native_case):
    reference, _ = native_case
    reference.policy.steps += 1
    with pytest.raises(ValueError, match="context changed"):
        reference.cursor()
