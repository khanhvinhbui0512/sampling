"""Enumeration is the primary check; seeded sampling is supplementary only."""

import math
from collections import Counter
from itertools import product

import pytest

from src.budget import BudgetTracker
from src.cars.oracle import FiniteLanguageOracle
from src.cars.proposal import FrozenProposal
from src.cars.trie import FailureTrie
from src.randomness import RNGStream
from src.sampling.completion import RhoCompletion
from src.types import CostPhase, MaskedState, TokenRules


def prepare(policy, joint, initial, *, prune=True):
    oracle = FiniteLanguageOracle(policy.rules,
                                 [leaf["tokens"] for leaf in joint["leaves"] if leaf["reward"]])
    trie = FailureTrie(initial, policy_id=policy.policy_id, reference_id=policy.reference_id,
                       oracle_id=oracle.oracle_id)
    discovery = BudgetTracker(base_sequence_limit=None)
    for depth in range(len(trie.positions)):
        for path in product((1, 2), repeat=depth):
            raw = policy.next_log_probs([trie.state_at(path)], budget=discovery,
                                        phase=CostPhase.DISCOVERY)[0]
            trie.cache_distribution(path, raw)
    if prune:
        # Leave some failing leaves unpruned: remaining mass must not be confused
        # with success mass. Certificates are exact for this finite language.
        for leaf in joint["leaves"]:
            state = MaskedState(leaf["tokens"], policy.rules)
            if initial.accepts_completion(state) and not leaf["reward"] and leaf["tokens"][0] == 1:
                path = trie.path_for(state)
                trie.record_failure(path, oracle.check(state, budget=discovery))
    return trie, oracle, discovery


def exact_q(leaf, initial, joint):
    """Independent table calculation for the prefix-only toy backbone.

    It ignores future observations at each reveal, exactly as the fixture's
    specified model does; it does NOT use pi_rho conditioned on those observations.
    """
    current, probability = list(initial.token_ids), 1.
    for position in initial.masked_positions:
        prefix = current[:position]
        compatible = [item for item in joint["leaves"] if item["tokens"][:position] == prefix]
        total = sum(item["probability"] for item in compatible)
        probability *= sum(item["probability"] for item in compatible
                           if item["tokens"][position] == leaf.token_ids[position]) / total
        current[position] = leaf.token_ids[position]
    return probability


@pytest.mark.parametrize("tokens", [(0, 0, 0), (1, 0, 0), (0, 0, 2), (0, 2, 0), (1, 0, 2)])
@pytest.mark.parametrize("prune", [False, True])
def test_exact_proposal_law_and_normalization_on_prefixes_and_holes(toy_backbone, toy_joint, tokens, prune):
    policy = RhoCompletion(toy_backbone)
    initial = MaskedState(tokens, policy.rules)
    trie, _, discovery = prepare(policy, toy_joint, initial, prune=prune)
    proposal = FrozenProposal(policy, trie.freeze())
    leaves = [MaskedState(item["tokens"], policy.rules) for item in toy_joint["leaves"]
              if initial.accepts_completion(MaskedState(item["tokens"], policy.rules))]
    q = {leaf.token_ids: exact_q(leaf, initial, toy_joint) for leaf in leaves}
    survivors = [leaf for leaf in leaves if trie.log_mass(trie.path_for(leaf)) != -math.inf]
    expected_b = sum(q[leaf.token_ids] for leaf in survivors)
    assert sum(q.values()) == pytest.approx(1.)
    assert trie.root_mass == pytest.approx(expected_b)
    budget = BudgetTracker(base_sequence_limit=0)
    actual = {}
    for leaf in leaves:
        actual[leaf.token_ids] = math.exp(proposal.log_prob(leaf, budget=budget))
        expected = q[leaf.token_ids] / expected_b if leaf in survivors else 0.
        assert actual[leaf.token_ids] == pytest.approx(expected)
    assert sum(actual.values()) == pytest.approx(1.)
    assert budget.raw_nfe == 0  # Cached discovery work is charged to discovery only.
    assert discovery.raw_nfe == 2 ** len(trie.positions) - 1
    before = proposal.trie.snapshot_id
    draw = proposal.draw(rng=RNGStream(2), budget=budget, progress=False)
    assert draw.status == "drawn" and initial.accepts_completion(draw.state)
    assert draw.log_q == pytest.approx(math.log(q[draw.state.token_ids]))
    assert draw.log_proposal == pytest.approx(draw.log_q - draw.log_root_mass)
    assert proposal.trie.snapshot_id == before


def test_unknown_suffix_uses_fresh_counted_forwards_without_learning(toy_backbone, toy_joint):
    policy = RhoCompletion(toy_backbone)
    initial = MaskedState((0, 0, 0), policy.rules)
    oracle = FiniteLanguageOracle(policy.rules, [])
    trie = FailureTrie(initial, policy_id=policy.policy_id, reference_id=policy.reference_id,
                       oracle_id=oracle.oracle_id)
    discovery = BudgetTracker(base_sequence_limit=1)
    trie.cache_distribution((), policy.next_log_probs([initial], budget=discovery,
                                                     phase=CostPhase.DISCOVERY)[0])
    proposal = FrozenProposal(policy, trie.freeze())
    before = proposal.trie.to_dict()
    budget = BudgetTracker(base_sequence_limit=4)
    rng = RNGStream(9)
    for _ in range(2):
        draw = proposal.draw(rng=rng, budget=budget, progress=False)
        assert draw.status == "drawn" and draw.cost.base_calls == 2
        assert draw.log_q == pytest.approx(math.log(exact_q(draw.state, initial, toy_joint)))
    assert budget.raw_nfe == 4
    assert budget.snapshot().by_phase[CostPhase.ESTIMATION.value].base_sequences == 4
    assert proposal.trie.to_dict() == before and proposal.trie.node_count == 1


def test_budget_failure_is_unavailable_draw_not_a_zero_reward(toy_backbone):
    policy = RhoCompletion(toy_backbone)
    initial = MaskedState((0, 0, 2), policy.rules)
    trie = FailureTrie(initial, policy_id=policy.policy_id, reference_id=policy.reference_id,
                       oracle_id="toy-oracle")
    proposal = FrozenProposal(policy, trie.freeze())
    budget = BudgetTracker(base_sequence_limit=1)
    draw = proposal.draw(rng=RNGStream(3), budget=budget, progress=False)
    assert draw.status == "budget_exhausted" and not draw.state.is_complete
    assert draw.state.token_ids[2] == 2 and draw.cost.base_calls == 1
    assert draw.log_q is None and draw.log_proposal is None and draw.error
    assert proposal.trie.node_count == 1


def test_proposal_respects_whole_query_and_completion_reservations(toy_backbone):
    policy = RhoCompletion(toy_backbone)
    initial = MaskedState((0, 0, 0), policy.rules)
    trie = FailureTrie(initial, policy_id=policy.policy_id, reference_id=policy.reference_id,
                       oracle_id="toy-oracle")
    proposal = FrozenProposal(policy, trie.freeze())
    budget = BudgetTracker(base_sequence_limit=5)
    with budget.reserve_completion(base_sequences=2):
        held = budget.reserve_query(label="frozen-draw", base_sequences=3)
        assert held is not None
        with held:
            draw = proposal.draw(rng=RNGStream(1), budget=budget, reservation=held, progress=False)
        assert draw.status == "drawn" and draw.cost.base_sequences == 3
        assert budget.snapshot().reserved_base_sequences == 2
    assert budget.snapshot().reserved_base_sequences == 0


def test_freeze_and_reference_identity_are_required(toy_backbone, toy_joint):
    policy = RhoCompletion(toy_backbone)
    trie, oracle, budget = prepare(policy, toy_joint, MaskedState((0, 0, 0), policy.rules))
    with pytest.raises(ValueError, match="Freeze"):
        FrozenProposal(policy, trie)
    with pytest.raises(ValueError, match="same q_rho"):
        FrozenProposal(RhoCompletion(toy_backbone, temperature=.5), trie.freeze())
    proposal = FrozenProposal(policy, trie.freeze())
    before = proposal.trie.snapshot_id
    # The caller's next discovery update cannot change the proposal snapshot.
    trie.record_failure((2, 1, 1), oracle.check(trie.state_at((2, 1, 1)), budget=budget))
    assert trie.root_mass == pytest.approx(.8)
    assert proposal.trie.root_mass == pytest.approx(.85) and proposal.trie.snapshot_id == before


def test_tiny_mass_still_draws_and_only_structural_zero_exhausts_support():
    class Backbone:
        token_rules = TokenRules(3, 1, mask_id=0)
        checkpoint_id, tokenizer_id, time_conditioning = "tiny-model", "tiny-tokens", False

        def logits_batch(self, *args, **kwargs):
            raise AssertionError("The raw distribution is already cached")

    policy = RhoCompletion(Backbone())
    initial = MaskedState((0,), policy.rules)
    oracle = FiniteLanguageOracle(policy.rules, [])
    trie = FailureTrie(initial, policy_id=policy.policy_id, reference_id=policy.reference_id,
                       oracle_id=oracle.oracle_id)
    trie.cache_distribution((), (-math.inf, 0., -1000.))
    budget = BudgetTracker(base_sequence_limit=0)
    trie.record_failure((1,), oracle.check(trie.state_at((1,)), budget=budget))
    draw = FrozenProposal(policy, trie.freeze()).draw(rng=RNGStream(1), budget=budget, progress=False)
    assert draw.status == "drawn" and draw.state.token_ids == (2,)
    assert draw.mass_underflow and draw.log_root_mass == draw.log_q == -1000.
    assert draw.log_proposal == 0. and draw.cost.base_calls == 0
    trie.record_failure((2,), oracle.check(trie.state_at((2,)), budget=budget))
    proposal = FrozenProposal(policy, trie.freeze())
    exhausted = proposal.draw(rng=RNGStream(1), budget=budget, progress=False)
    assert exhausted.status == "support_exhausted" and exhausted.log_q is None
    assert not exhausted.mass_underflow and exhausted.cost.base_calls == 0
    with pytest.raises(ValueError, match="undefined"):
        proposal.log_prob(MaskedState((2,), policy.rules), budget=budget)


def test_accepted_law_matches_q_conditioned_on_reward_one(toy_backbone, toy_joint):
    policy = RhoCompletion(toy_backbone)
    initial = MaskedState((0, 0, 0), policy.rules)
    trie, _, _ = prepare(policy, toy_joint, initial)
    proposal = FrozenProposal(policy, trie.freeze())
    budget = BudgetTracker(base_sequence_limit=0)
    success = [item for item in toy_joint["leaves"] if item["reward"]]
    z = sum(item["probability"] for item in success)
    masses = {tuple(item["tokens"]): math.exp(proposal.log_prob(
        MaskedState(item["tokens"], policy.rules), budget=budget)) for item in success}
    acceptance = sum(masses.values())
    for item in success:
        assert masses[tuple(item["tokens"])] / acceptance == pytest.approx(item["probability"] / z)
    assert acceptance == pytest.approx(z / trie.root_mass)
    # Supplementary, deterministically seeded distribution check with a broad
    # six-standard-error tolerance; no estimator claim follows from this check.
    counts = Counter()
    for seed in range(16):
        rng = RNGStream(seed)
        for _ in range(64):
            draw = proposal.draw(rng=rng, budget=budget, progress=False)
            if draw.state.token_ids in masses:
                counts[draw.state.token_ids] += 1
    total = sum(counts.values())
    assert total > 100
    for item in success:
        p = item["probability"] / z
        tolerance = 6 * math.sqrt(p * (1 - p) / total) + 1 / total
        assert abs(counts[tuple(item["tokens"])] / total - p) <= tolerance
    assert budget.raw_nfe == 0
