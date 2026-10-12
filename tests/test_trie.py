"""Exact finite-support checks for Step 7; written, not executed."""

import json
import math
from dataclasses import replace
from itertools import product

import pytest

from src.budget import BudgetTracker
from src.cars.oracle import FiniteLanguageOracle
from src.cars.trie import FailureTrie, TrieCapacityError
from src.sampling.completion import RhoCompletion
from src.types import CostPhase, MaskedState, TokenRules


def cache_paths(trie, policy, budget):
    for depth in range(len(trie.positions)):
        for path in product((1, 2), repeat=depth):
            raw = policy.next_log_probs([trie.state_at(path)], budget=budget,
                                        phase=CostPhase.DISCOVERY)[0]
            trie.cache_distribution(path, raw)


@pytest.fixture
def problem(toy_backbone, toy_joint):
    policy = RhoCompletion(toy_backbone)
    root = MaskedState((0, 0, 0), policy.rules)
    oracle = FiniteLanguageOracle(policy.rules,
                                 [leaf["tokens"] for leaf in toy_joint["leaves"] if leaf["reward"]])
    trie = FailureTrie(root, policy_id=policy.policy_id, reference_id=policy.reference_id,
                       oracle_id=oracle.oracle_id)
    budget = BudgetTracker(base_sequence_limit=None)
    cache_paths(trie, policy, budget)
    return trie, policy, oracle, budget


def fail(trie, oracle, budget, path):
    certificate = oracle.check(trie.state_at(path), budget=budget)
    return trie.record_failure(path, certificate)


def test_leaf_pruning_matches_enumeration_and_is_monotone(problem, toy_joint):
    trie, _, oracle, budget = problem
    removed, previous = set(), 1.
    for leaf in toy_joint["leaves"]:
        if leaf["reward"]:
            continue
        path = tuple(leaf["tokens"])
        assert fail(trie, oracle, budget, path)
        removed.add(path)
        expected = sum(item["probability"] for item in toy_joint["leaves"]
                       if tuple(item["tokens"]) not in removed)
        assert trie.root_mass == pytest.approx(expected)
        assert trie.root_mass <= previous + 1e-14
        previous = trie.root_mass
    assert trie.root_mass == pytest.approx(.7)
    assert not trie.support_exhausted


def test_overlapping_cones_and_duplicates_are_counted_once(problem):
    _, policy, _, budget = problem
    oracle = FiniteLanguageOracle(policy.rules, [(2, 2, 2)])
    trie = FailureTrie(MaskedState((0, 0, 0), policy.rules), policy_id=policy.policy_id,
                       reference_id=policy.reference_id, oracle_id=oracle.oracle_id)
    cache_paths(trie, policy, budget)
    assert fail(trie, oracle, budget, (1, 1, 2))
    assert trie.root_mass == pytest.approx(.9)
    nodes = trie.node_count
    assert not fail(trie, oracle, budget, (1, 1, 2))
    assert trie.node_count == nodes and trie.root_mass == pytest.approx(.9)
    assert fail(trie, oracle, budget, (1, 1))
    assert trie.root_mass == pytest.approx(.6)
    assert trie.node_count < nodes
    assert not fail(trie, oracle, budget, (1, 1, 1))
    assert fail(trie, oracle, budget, (1,))
    assert trie.root_mass == pytest.approx(.5)
    assert trie.log_mass((1, 2, 2)) == -math.inf
    assert not fail(trie, oracle, budget, (1, 2))
    assert trie.root_mass == pytest.approx(.5)


def test_unknown_children_have_mass_one(problem):
    _, policy, _, budget = problem
    oracle = FiniteLanguageOracle(policy.rules, [(2, 2, 2)])
    trie = FailureTrie(MaskedState((0, 0, 0), policy.rules), policy_id=policy.policy_id,
                       reference_id=policy.reference_id, oracle_id=oracle.oracle_id)
    for path in ((), (1,)):
        trie.cache_distribution(path, policy.next_log_probs([trie.state_at(path)], budget=budget,
                                                           phase=CostPhase.DISCOVERY)[0])
    fail(trie, oracle, budget, (1, 1))
    assert trie.log_mass((2,)) == 0.
    assert trie.log_mass((1, 2)) == 0.
    assert trie.root_mass == pytest.approx(.6)


def test_pruning_requires_certificates_and_all_ancestor_distributions(problem):
    full, policy, oracle, budget = problem
    trie = FailureTrie(full.initial_state, policy_id=policy.policy_id,
                       reference_id=policy.reference_id, oracle_id=oracle.oracle_id)
    certificate = oracle.check(trie.state_at((1, 1, 2)), budget=budget)
    before = trie.to_dict()
    with pytest.raises(ValueError, match="ancestor"):
        trie.record_failure((1, 1, 2), certificate)
    assert trie.to_dict() == before
    trie.cache_distribution((), full.raw_log_probs())
    before = trie.to_dict()
    with pytest.raises(ValueError, match="ancestor"):
        trie.record_failure((1, 1, 2), certificate)
    assert trie.to_dict() == before
    unknown = oracle.check(full.state_at((1,)), budget=budget)
    assert not unknown.certified_failure
    for bad in (unknown, replace(certificate, oracle_id="other-oracle"),
                replace(certificate, state=full.state_at((1, 2, 2)))):
        with pytest.raises(ValueError, match="certified-failure"):
            full.record_failure((1, 1, 2), bad)


def test_raw_probabilities_cannot_drift_and_capacity_is_atomic(problem):
    full, policy, oracle, _ = problem
    with pytest.raises(ValueError, match="cannot change"):
        full.cache_distribution((), (-math.inf, math.log(.4), math.log(.6)))
    with pytest.raises(ValueError, match="sum to one"):
        full.cache_distribution((), (-math.inf, -1., -1.))
    limited = FailureTrie(full.initial_state, policy_id=policy.policy_id,
                          reference_id=policy.reference_id, oracle_id=oracle.oracle_id, max_nodes=1)
    limited.cache_distribution((), full.raw_log_probs())
    before = limited.to_dict()
    with pytest.raises(TrieCapacityError):
        limited.cache_distribution((1,), full.raw_log_probs((1,)))
    assert limited.to_dict() == before and limited.node_count == 1


def test_log_mass_underflow_is_not_support_exhaustion():
    rules = TokenRules(vocab_size=3, sequence_length=1, mask_id=0)
    oracle = FiniteLanguageOracle(rules, [])
    trie = FailureTrie(MaskedState((0,), rules), policy_id="rho", reference_id="reference",
                       oracle_id=oracle.oracle_id)
    trie.cache_distribution((), (-math.inf, 0., -1000.))
    budget = BudgetTracker(base_sequence_limit=0)
    fail(trie, oracle, budget, (1,))
    assert trie.log_mass() == -1000.
    assert trie.root_mass == 0. and trie.mass_underflow and not trie.support_exhausted
    restored = FailureTrie.from_dict(json.loads(json.dumps(trie.to_dict(), allow_nan=False)),
                                     expected_identity=trie.identity)
    assert restored.log_mass() == -1000. and restored.mass_underflow
    fail(restored, oracle, budget, (2,))
    assert restored.support_exhausted and not restored.mass_underflow
    assert restored.log_mass() == -math.inf


def test_freeze_round_trip_is_independent_and_checks_full_context(problem):
    trie, _, oracle, budget = problem
    fail(trie, oracle, budget, (1, 1, 2))
    frozen = trie.freeze()
    snapshot_id = frozen.snapshot_id
    payload = json.loads(json.dumps(frozen.to_dict(), allow_nan=False))
    restored = FailureTrie.from_dict(payload, expected_identity=trie.identity)
    assert restored.is_frozen and restored.snapshot_id == snapshot_id
    assert restored.root_mass == pytest.approx(.9)
    fail(trie, oracle, budget, (1, 2, 2))
    assert trie.root_mass == pytest.approx(.85)
    assert frozen.root_mass == pytest.approx(.9) and frozen.snapshot_id == snapshot_id
    with pytest.raises(ValueError, match="Frozen"):
        frozen.cache_distribution((), trie.raw_log_probs())
    with pytest.raises(ValueError, match="Frozen"):
        fail(frozen, oracle, budget, (1, 2, 2))
    changed_context = {**trie.identity, "initial_tokens": [0, 0, 2]}
    with pytest.raises(ValueError, match="context"):
        FailureTrie.from_dict(payload, expected_identity=changed_context)
    payload["root"]["raw"] = [None, -1., -1.]
    with pytest.raises(ValueError, match="sum to one"):
        FailureTrie.from_dict(payload)


def test_mask_holes_keep_observations_and_assignment_order(toy_backbone):
    policy = RhoCompletion(toy_backbone)
    initial = MaskedState((0, 2, 0), policy.rules)
    trie = FailureTrie(initial, policy_id=policy.policy_id,
                       reference_id=policy.reference_id, oracle_id="toy-oracle")
    assert trie.state_at((1,)).token_ids == (1, 2, 0)
    assert trie.path_for(trie.state_at((1, 2))) == (1, 2)
    with pytest.raises(ValueError, match="left-to-right"):
        trie.path_for(MaskedState((0, 2, 1), policy.rules))
    with pytest.raises(ValueError, match="original observations"):
        trie.path_for(MaskedState((1, 1, 2), policy.rules))
