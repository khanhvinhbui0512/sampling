"""Exact-context cache checks, including disk snapshots and run value freezing."""

import copy
import math
import os
from dataclasses import replace

import pytest

from src.budget import BudgetTracker
from src.cars.cache import CARSCache
from src.cars.estimator import CARSValueEstimator
from src.cars.oracle import FiniteLanguageOracle
from src.cars.trie import FailureTrie
from src.io import payload_hash, read_json, write_json
from src.randomness import RNGStream
from src.sampling.completion import RhoCompletion
from src.types import EstimateStatus, MaskedState
from src.verifiers.cars import CARSVerifier


@pytest.fixture
def cache_inputs(toy_backbone, cars_reward_factory):
    reward = cars_reward_factory()
    policy = RhoCompletion(toy_backbone)
    oracle = FiniteLanguageOracle(policy.rules, reward.successful_tokens)
    estimator = CARSValueEstimator(policy, reward, oracle, discovery_attempts=2, draws=8)
    state = MaskedState((0, 0, 2), policy.rules)
    run = estimator.estimate(state, rng=RNGStream(8), budget=BudgetTracker(base_sequence_limit=None),
                              progress=False)
    assert run.estimate.status == EstimateStatus.ESTIMATED
    return estimator, state, run


def test_eviction_keeps_completed_values_and_snapshots_independent(cache_inputs):
    estimator, first, run = cache_inputs
    second = MaskedState((1, 0, 0), first.rules)
    other = estimator.estimate(second, rng=RNGStream(9), budget=BudgetTracker(base_sequence_limit=None),
                                progress=False)
    cache = CARSCache(estimator.cache_metadata, max_entries=1, run_id="one-run")
    cache.put(first, run.estimate, run.trie)
    snapshot = cache.snapshot(first)
    cache.put(second, other.estimate, other.trie)
    assert cache.snapshot(first) is None and cache.stats["trie_evictions"] == 1
    cached = cache.get(first)
    assert cached.value == run.estimate.value and cached.interval == run.estimate.interval
    assert cached.cache_hit and cached.cost.base_calls == 0 and cached.draws == 0
    assert snapshot.to_dict() == run.trie.to_dict() and snapshot.is_frozen
    # Eviction must not re-estimate or alter a value when VGB revisits the state.
    verifier = CARSVerifier(estimator, cache=cache, progress=False)
    budget = BudgetTracker(base_sequence_limit=0)
    assert verifier.required_work([first, second]) == (0, 0)
    results = verifier.values_for_states([first, second, first], rng=RNGStream(100), budget=budget)
    assert [row.value for row in results] == [run.estimate.value, other.estimate.value, run.estimate.value]
    assert budget.raw_nfe == 0 and verifier.runs == []


def test_run_value_cannot_be_replaced_by_a_new_estimate(cache_inputs):
    estimator, state, run = cache_inputs
    cache = CARSCache(estimator.cache_metadata, max_entries=1, run_id="fixed")
    cache.put(state, run.estimate, run.trie)
    changed_value = 0. if run.estimate.value else run.estimate.surviving_mass / 2
    changed = replace(run.estimate, value=changed_value, interval=(0., run.estimate.surviving_mass))
    with pytest.raises(ValueError, match="frozen"):
        cache.put(state, changed, run.trie)
    assert cache.get(state).value == run.estimate.value


def test_a_new_run_without_persistence_pays_for_recomputation(cache_inputs):
    estimator, state, run = cache_inputs
    first = CARSCache(estimator.cache_metadata, max_entries=1, run_id="first")
    first.put(state, run.estimate, run.trie)
    fresh = CARSCache(estimator.cache_metadata, max_entries=1, run_id="fresh")
    verifier = CARSVerifier(estimator, cache=fresh, progress=False)
    assert verifier.required_work([state]) == (estimator.required_work(state), 0)
    budget = BudgetTracker(base_sequence_limit=estimator.required_work(state))
    result = verifier.values_for_states([state], rng=RNGStream(3), budget=budget)[0]
    assert not result.cache_hit and result.cost.base_calls == budget.raw_nfe > 0
    assert first.get(state).value == run.estimate.value  # Separate run value remains pinned.


def test_json_round_trip_preserves_raw_probabilities_mass_and_origin_cost(cache_inputs, artifact_directory):
    estimator, state, run = cache_inputs
    directory = os.path.join(artifact_directory, "cars-cache")
    first = CARSCache(estimator.cache_metadata, max_entries=1, run_id="origin", directory=directory)
    first.put(state, run.estimate, run.trie)
    path = first.path(state)
    payload = read_json(path)
    assert payload["estimate"]["cost"]["base_calls"] == run.estimate.cost.base_calls
    assert payload["estimate"]["draws"] == 8
    second = CARSCache(estimator.cache_metadata, max_entries=1, run_id="resumed", directory=directory)
    assert second.load(state, path=path)
    loaded = second.get(state)
    assert loaded.value == run.estimate.value and loaded.interval == run.estimate.interval
    assert loaded.cost.base_calls == loaded.cost.base_sequences == loaded.draws == 0
    assert second.snapshot(state).to_dict() == run.trie.to_dict()
    assert second.details(state)["origin_draws"] == 8
    assert second.details(state)["origin_cost"]["base_calls"] == run.estimate.cost.base_calls
    assert second.details(state)["origin_run_id"] == "origin"
    assert second.stats["disk_reads"] == 1 and second.stats["computed_values"] == 0


def test_same_run_can_autoload_and_evicted_trie_can_reload(cache_inputs, artifact_directory):
    estimator, state, run = cache_inputs
    directory = os.path.join(artifact_directory, "reload")
    cache = CARSCache(estimator.cache_metadata, max_entries=1, run_id="saved", directory=directory)
    cache.put(state, run.estimate, run.trie)
    other_state = MaskedState((1, 0, 0), state.rules)
    other = estimator.estimate(other_state, rng=RNGStream(3), budget=BudgetTracker(base_sequence_limit=None),
                                progress=False)
    cache.put(other_state, other.estimate, other.trie)
    assert cache.stats["trie_evictions"] == 1
    assert cache.snapshot(state).snapshot_id == run.trie.snapshot_id
    assert cache.stats["disk_reads"] == 1 and cache.stats["trie_evictions"] == 2
    resumed = CARSCache(estimator.cache_metadata, max_entries=1, run_id="saved", directory=directory)
    assert resumed.get(state).cache_hit and resumed.stats["disk_reads"] == 1


@pytest.mark.parametrize("field", ["checkpoint", "tokenizer", "threshold", "rho", "oracle", "draws", "source"])
def test_no_reuse_across_full_metadata_changes(cache_inputs, artifact_directory, field):
    estimator, state, run = cache_inputs
    directory = os.path.join(artifact_directory, "metadata")
    original = CARSCache(estimator.cache_metadata, max_entries=1, run_id="original", directory=directory)
    original.put(state, run.estimate, run.trie)
    changed = copy.deepcopy(estimator.cache_metadata)
    if field == "checkpoint":
        changed["checkpoint_id"] = "different-weights"
    elif field == "tokenizer":
        changed["tokenizer_id"] = "different-tokenizer"
    elif field == "threshold":
        changed["calibration"]["threshold"] = .8  # Even if someone reused the same calibration ID.
    elif field == "rho":
        changed["policy_id"] = "different-reveal-policy"
    elif field == "oracle":
        changed["oracle_id"] = "different-failure-oracle"
    elif field == "draws":
        changed["estimator"]["draws"] += 1
    else:
        changed["source_hashes"]["src/cars/estimator.py"] = "different-code"
    other = CARSCache(changed, max_entries=1, run_id="other", directory=directory)
    assert other.key(state) != original.key(state)
    assert other.get(state) is None
    with pytest.raises(ValueError, match="metadata/full context"):
        other.load(state, path=original.path(state))


def test_future_observations_are_part_of_the_key(cache_inputs, artifact_directory):
    estimator, state, run = cache_inputs
    cache = CARSCache(estimator.cache_metadata, max_entries=1, run_id="holes",
                      directory=os.path.join(artifact_directory, "holes"))
    cache.put(state, run.estimate, run.trie)
    other_state = MaskedState((0, 0, 1), state.rules)
    assert cache.key(state) != cache.key(other_state) and cache.get(other_state) is None
    with pytest.raises(ValueError, match="metadata/full context"):
        cache.load(other_state, path=cache.path(state))


def test_cache_does_not_store_unavailable_values_or_mutable_tries(cache_inputs):
    estimator, state, run = cache_inputs
    cache = CARSCache(estimator.cache_metadata, max_entries=1, run_id="available-only")
    unavailable = estimator.estimate(state, rng=RNGStream(0), budget=BudgetTracker(base_sequence_limit=0),
                                      progress=False)
    assert unavailable.estimate.value is None
    with pytest.raises(ValueError, match="available rho value"):
        cache.put(state, unavailable.estimate, unavailable.trie)
    payload = run.trie.to_dict()
    payload["frozen"] = False
    with pytest.raises(ValueError, match="matching frozen trie"):
        cache.put(state, run.estimate, FailureTrie.from_dict(payload))


def test_snapshot_corruption_is_rejected(cache_inputs, artifact_directory):
    estimator, state, run = cache_inputs
    cache = CARSCache(estimator.cache_metadata, max_entries=1, run_id="integrity",
                      directory=os.path.join(artifact_directory, "integrity"))
    cache.put(state, run.estimate, run.trie)
    payload = read_json(cache.path(state))
    payload["estimate"]["value"] = .123
    corrupted = os.path.join(artifact_directory, "corrupted.json")
    write_json(corrupted, payload)
    with pytest.raises(ValueError, match="checksum"):
        cache.load(state, path=corrupted)
    payload = read_json(cache.path(state))
    payload.pop("checksum")
    payload["trie"]["root"]["raw"][1] = .2  # Not a normalized log-probability vector.
    payload["checksum"] = payload_hash(payload)
    malformed = os.path.join(artifact_directory, "malformed.json")
    write_json(malformed, payload)
    with pytest.raises(ValueError, match="log probabilities"):
        cache.load(state, path=malformed)


def test_certified_zero_snapshot_uses_json_null_for_negative_infinity(
        toy_backbone, cars_reward_factory, artifact_directory):
    reward = cars_reward_factory([])
    policy = RhoCompletion(toy_backbone)
    oracle = FiniteLanguageOracle(policy.rules, [])
    estimator = CARSValueEstimator(policy, reward, oracle, discovery_attempts=1, draws=4, mode="cars")
    state = MaskedState((0, 0, 0), policy.rules)
    run = estimator.estimate(state, rng=RNGStream(9), budget=BudgetTracker(base_sequence_limit=None),
                              progress=False)
    cache = CARSCache(estimator.cache_metadata, max_entries=1, run_id="zero",
                      directory=os.path.join(artifact_directory, "zero"))
    cache.put(state, run.estimate, run.trie)
    assert read_json(cache.path(state))["details"]["log_surviving_mass"] is None
    assert cache.load(state) and cache.snapshot(state).log_mass() == -math.inf
    assert cache.get(state).status == EstimateStatus.EXACT and cache.get(state).value == 0


def test_calibration_drift_is_rejected_before_reusing_a_cached_value(cache_inputs):
    estimator, state, run = cache_inputs
    cache = CARSCache(estimator.cache_metadata, max_entries=1, run_id="calibration")
    cache.put(state, run.estimate, run.trie)
    verifier = CARSVerifier(estimator, cache=cache, progress=False)
    estimator.reward.calibration.threshold = .9
    with pytest.raises(ValueError, match="calibration changed"):
        verifier.values_for_states([state], rng=RNGStream(0), budget=BudgetTracker(base_sequence_limit=0))
