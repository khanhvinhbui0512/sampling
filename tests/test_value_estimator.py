"""Step 8 checks written for review. Enumeration supplies primary evidence."""

import math
from itertools import product
from types import SimpleNamespace

import pytest

from src.budget import BudgetExceeded, BudgetTracker
from src.cars.cache import CARSCache
from src.cars.estimator import CARSValueEstimator, adaptive_draw, scaled_bernoulli_summary
from src.cars.oracle import FiniteLanguageOracle
from src.cars.trie import FailureTrie
from src.randomness import RNGStream
from src.sampling.candidates import ValueTable
from src.sampling.completion import RhoCompletion
from src.sampling.rejection import RejectionSampler
from src.types import CostPhase, EstimateStatus, MaskedState, SampleStatus, TokenRules
from src.verifiers.cars import CARSVerifier


def test_native_neural_mc_cars_share_target_and_value_table(native_case, enumerate_native):
    import torch
    from src.cars.cache import NativeCARSCache
    from src.cars.native import NativeCARSValueEstimator
    from src.verifiers.cars import NativeCARSVerifier
    from src.verifiers.interface import native_conditional_target, value_target
    from src.verifiers.monte_carlo import NativeConditionalMonteCarloVerifier
    from src.verifiers.neural import NeuralVerifier

    reference, reward = native_case
    paths = enumerate_native(reference)

    class ExactToyNetwork(torch.nn.Module):
        """Population conditional control; trained QM9 error remains to measure."""
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.))

        def forward(self, tokens):
            values = []
            for row in tokens:
                query = MaskedState(tuple(row.tolist()), reference.rules)
                d = sum(p for _, leaf, p in paths if query.accepts_completion(leaf))
                n = sum(p for _, leaf, p in paths if query.accepts_completion(leaf) and reward.evaluate(leaf, require_binary=True).reward)
                values.append(n / d)
            return torch.tensor(values, device=tokens.device)

    target = native_conditional_target(reference, reward)
    # NeuralVerifier checks the model's vocabulary/length as well as target.
    network = ExactToyNetwork()
    network.max_length = reference.rules.sequence_length
    network.token_emb = torch.nn.Embedding(reference.rules.vocab_size, 1)
    neural = NeuralVerifier(network, reward, target)
    mc = NativeConditionalMonteCarloVerifier(reference, reward, draws=256, min_denominator_hits=1, progress=False)
    estimator = NativeCARSValueEstimator(reference, reward, discovery_attempts=4, numerator_draws=256,
                                        denominator_draws=256, max_nodes=5000, min_denominator_hits=1)
    cache = NativeCARSCache(estimator.cache_metadata, reference=reference, reward=reward, max_entries=1, run_id="native-toy")
    cars = NativeCARSVerifier(estimator, cache=cache, progress=False)
    assert neural.target == mc.target == cars.target == target
    assert target != value_target(RhoCompletion(reference.backbone), reward)
    states = [reference.root, MaskedState((1, 1), reference.rules), reference.root]
    for verifier in (neural, mc, cars):
        tracker = BudgetTracker(base_sequence_limit=None)
        results = verifier.values_for_states(states, rng=RNGStream(21), budget=tracker)
        assert len(results) == 3 and results[1].value == 1. and results[1].status == EstimateStatus.EXACT
        assert all(result.target == target for result in results)
        assert sum(result.cost.base_sequences for result in results) == tracker.snapshot().total.base_sequences
    assert cars.runs[0].estimate.interval[0] <= neural.values_for_states([reference.root], rng=RNGStream(2),
                budget=BudgetTracker(base_sequence_limit=None))[0].value <= cars.runs[0].estimate.interval[1]
    tracker = BudgetTracker(base_sequence_limit=0)
    table = ValueTable(cars, RNGStream(9), tracker)
    assert table.query(states) == [cache.get(reference.root).value, 1., cache.get(reference.root).value]
    assert tracker.raw_nfe == 0  # Pinning survives repeated candidate value queries.
    with pytest.raises(ValueError, match="native trajectory"):
        NativeCARSVerifier(estimator, cache=CARSCache({**estimator.cache_metadata, "namespace": "cars-rho",
                           "target": {**estimator.cache_metadata["target"], "kind": "v_rho"}}, max_entries=1, run_id="wrong"))


def test_native_cache_roundtrip_namespace_and_future_observations(native_case, artifact_directory):
    from src.cars.cache import NativeCARSCache
    from src.cars.native import NativeCARSValueEstimator, NativeTrajectoryTrie

    reference, reward = native_case
    estimator = NativeCARSValueEstimator(reference, reward, discovery_attempts=2, numerator_draws=64,
                                        denominator_draws=64, max_nodes=5000, min_denominator_hits=1)
    run = estimator.estimate(reference.root, rng=RNGStream(2), budget=BudgetTracker(base_sequence_limit=None), progress=False)
    kwargs = dict(reference=reference, reward=reward, max_entries=1, run_id="native-cache", directory=artifact_directory)
    cache = NativeCARSCache(estimator.cache_metadata, **kwargs)
    cache.put(reference.root, run.estimate, run.trie, summary=run.summary)
    restored = NativeCARSCache(estimator.cache_metadata, **kwargs)
    hit = restored.get(reference.root)
    assert hit.cache_hit and hit.draws == 0 and hit.cost.base_calls == 0
    assert hit.value == run.estimate.value and restored.snapshot(reference.root).snapshot_id == run.trie.snapshot_id
    assert cache.key(MaskedState((0, 1), reference.rules)) != cache.key(MaskedState((0, 2), reference.rules))
    with pytest.raises(ValueError, match="namespace/schema"):
        CARSCache(estimator.cache_metadata, max_entries=1, run_id="rho")
    payload = {"schema_version": 1, "namespace": "cars-rho", "max_nodes": 10}
    with pytest.raises(ValueError, match="identity/schema"):
        NativeTrajectoryTrie.from_dict(payload, reference=reference, reward=reward, state=reference.root)


def make_estimator(backbone, reward, *, discovery=2, draws=16, mode="ars", max_nodes=None, cap=None):
    policy = RhoCompletion(backbone)
    oracle = FiniteLanguageOracle(policy.rules, reward.successful_tokens)
    return CARSValueEstimator(policy, reward, oracle, discovery_attempts=discovery, draws=draws,
                              mode=mode, max_nodes=max_nodes, discovery_max_base_sequences=cap)


def exact_q(leaf, initial, joint):
    """Independent q_rho calculation for the fixture's prefix-only model."""
    current, probability = list(initial.token_ids), 1.
    for position in initial.masked_positions:
        rows = [row for row in joint["leaves"] if row["tokens"][:position] == current[:position]]
        probability *= sum(row["probability"] for row in rows
                           if row["tokens"][position] == leaf.token_ids[position]) / sum(
                               row["probability"] for row in rows)
        current[position] = leaf.token_ids[position]
    return probability


def enumerated_trie(estimator, state, joint, prune_count):
    policy, oracle = estimator.policy, estimator.oracle
    trie = FailureTrie(state, policy_id=policy.policy_id, reference_id=policy.reference_id,
                       oracle_id=oracle.oracle_id)
    budget = BudgetTracker(base_sequence_limit=None)
    for depth in range(len(trie.positions)):
        for path in product((1, 2), repeat=depth):
            raw = policy.next_log_probs([trie.state_at(path)], budget=budget, phase=CostPhase.DISCOVERY)[0]
            trie.cache_distribution(path, raw)
    failed = [MaskedState(row["tokens"], policy.rules) for row in joint["leaves"]
              if not row["reward"] and state.accepts_completion(MaskedState(row["tokens"], policy.rules))]
    for leaf in failed[:prune_count]:
        trie.record_failure(trie.path_for(leaf), oracle.check(leaf, budget=budget))
    return trie.freeze()


@pytest.mark.parametrize("tokens", [(0, 0, 0), (1, 0, 0), (0, 0, 2)])
@pytest.mark.parametrize("prune_count", [0, 1, 99])
def test_exact_expectation_and_conditional_interval_coverage(
        toy_backbone, toy_joint, cars_reward_factory, tokens, prune_count):
    estimator = make_estimator(toy_backbone, cars_reward_factory(), discovery=0, draws=16)
    state = MaskedState(tokens, estimator.policy.rules)
    trie = enumerated_trie(estimator, state, toy_joint, prune_count)
    exact_value = sum(exact_q(MaskedState(row["tokens"], state.rules), state, toy_joint)
                      for row in toy_joint["leaves"] if row["reward"]
                      and state.accepts_completion(MaskedState(row["tokens"], state.rules)))
    p, m, b = exact_value / trie.root_mass, estimator.draws, trie.root_mass
    # Enumerate every possible success count rather than claiming coverage from
    # a convenient random sample. Hoeffding is conservative for each fixed W.
    p = min(1., p)  # Round-off in the independent enumeration, not weight clipping.
    expected, coverage = 0., 0.
    for successes in range(m + 1):
        probability = math.comb(m, successes) * p ** successes * (1 - p) ** (m - successes)
        value, _, interval = scaled_bernoulli_summary(successes, m, trie.log_mass(), .95)
        expected += probability * value
        coverage += probability * int(interval[0] - 1e-12 <= exact_value <= interval[1] + 1e-12)
    assert expected == pytest.approx(exact_value)
    assert coverage >= .95 - 1e-12
    # Supplementary seeded exercise of the actual proposal/estimation path.
    values = []
    for seed in range(8):
        budget = BudgetTracker(base_sequence_limit=None)
        run = estimator.estimate(state, initial_trie=trie, rng=RNGStream(seed), budget=budget, progress=False)
        estimate = run.estimate
        assert estimate.status == EstimateStatus.ESTIMATED and estimate.draws == m
        assert len(run.estimation_trace) == m and run.discovery_trace == ()
        assert estimate.value == pytest.approx(b * sum(row["reward"] for row in run.estimation_trace) / m)
        assert all(row["log_root_mass"] == trie.log_mass() for row in run.estimation_trace)
        assert run.trie.snapshot_id == trie.snapshot_id and trie.is_frozen
        assert budget.raw_nfe == 0  # All probabilities were charged to enumeration setup.
        values.append(estimate.value)
    tolerance = 6 * b * math.sqrt(p * (1 - p) / (8 * m)) + 1e-12
    assert abs(sum(values) / len(values) - exact_value) <= tolerance


@pytest.mark.parametrize("seed", [0, 1, 8, 29])
def test_discovery_freeze_fixed_count_and_actual_cost(toy_backbone, cars_reward_factory, seed):
    estimator = make_estimator(toy_backbone, cars_reward_factory(), discovery=5, draws=24)
    state = MaskedState((0, 0, 2), estimator.policy.rules)
    budget = BudgetTracker(base_sequence_limit=estimator.required_work(state))
    run = estimator.estimate(state, rng=RNGStream(seed), budget=budget, progress=False)
    assert run.estimate.draws == len(run.estimation_trace) == 24
    assert len(run.discovery_trace) == 5 and run.trie.is_frozen
    assert run.estimate.cost == budget.snapshot().total
    assert sum(row["cost"]["base_calls"] for row in run.discovery_trace + run.estimation_trace) == budget.raw_nfe
    assert len({tuple(row["tokens"]) for row in run.estimation_trace}) < 24  # Duplicates retained.
    assert budget.snapshot().by_phase["estimation"].oracle_calls == 24
    assert budget.snapshot().reserved_base_sequences == 0
    assert run.rng_state["discovery"]["seed"] != run.rng_state["estimation"]["seed"]
    assert run.estimate.target.kind.value == "v_rho"


@pytest.mark.parametrize("success", [False, True])
def test_no_pruning_zero_and_all_successes(toy_backbone, toy_joint, cars_reward_factory, success):
    reward = cars_reward_factory([row["tokens"] for row in toy_joint["leaves"]] if success else [])
    estimator = make_estimator(toy_backbone, reward, discovery=0, draws=12)
    state = MaskedState((0, 0, 0), estimator.policy.rules)
    budget = BudgetTracker(base_sequence_limit=36)
    run = estimator.estimate(state, rng=RNGStream(8), budget=budget, progress=False)
    estimate = run.estimate
    assert estimate.status == EstimateStatus.ESTIMATED and estimate.draws == 12
    assert estimate.value == float(success) and estimate.surviving_mass == 1.
    assert budget.raw_nfe == 36 and estimate.cost.oracle_calls == 12
    if success:
        assert estimate.interval[0] < 1 and estimate.interval[1] == 1
    else:
        assert estimate.interval[0] == 0 and estimate.interval[1] > 0


def test_certified_zero_needs_no_estimation_or_budget(toy_backbone, cars_reward_factory):
    estimator = make_estimator(toy_backbone, cars_reward_factory([]), discovery=3, mode="cars")
    state = MaskedState((0, 0, 0), estimator.policy.rules)
    first = estimator.estimate(state, rng=RNGStream(3), budget=BudgetTracker(base_sequence_limit=None),
                               progress=False)
    assert first.estimate.status == EstimateStatus.EXACT and first.estimate.value == 0
    assert first.estimate.draws == 0 and first.estimation_trace == ()
    assert first.trie.support_exhausted and len(first.discovery_trace) == 1
    budget = BudgetTracker(base_sequence_limit=0)
    second = estimator.estimate(state, initial_trie=first.trie, rng=RNGStream(4), budget=budget, progress=False)
    assert second.estimate.status == EstimateStatus.EXACT and second.estimate.interval == (0., 0.)
    assert second.estimate.cost.base_calls == 0 and second.estimate.surviving_mass == 0


def test_entire_query_is_reserved_before_rng_discovery_or_forwards(toy_backbone, cars_reward_factory):
    estimator = make_estimator(toy_backbone, cars_reward_factory(), discovery=4, draws=7)
    state = MaskedState((0, 0, 2), estimator.policy.rules)
    rng = RNGStream(10)
    rng_before = rng.state_dict()
    budget = BudgetTracker(base_sequence_limit=estimator.required_work(state) - 1)
    run = estimator.estimate(state, rng=rng, budget=budget, progress=False)
    assert run.estimate.status == EstimateStatus.BUDGET_EXHAUSTED and run.estimate.value is None
    assert run.discovery_trace == run.estimation_trace == ()
    assert budget.raw_nfe == budget.snapshot().total.oracle_calls == 0
    assert rng.state_dict() == rng_before and budget.snapshot().reserved_base_sequences == 0


def test_interrupted_fixed_query_is_unavailable_with_real_work_preserved(
        toy_backbone, cars_reward_factory, monkeypatch):
    estimator = make_estimator(toy_backbone, cars_reward_factory(), discovery=0, draws=4)
    state = MaskedState((0, 0, 0), estimator.policy.rules)
    original = estimator.policy.next_log_probs
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        if calls == 3:
            raise BudgetExceeded("Injected interruption after one complete draw")
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(estimator.policy, "next_log_probs", interrupted)
    budget = BudgetTracker(base_sequence_limit=12)
    run = estimator.estimate(state, rng=RNGStream(9), budget=budget, progress=False)
    assert run.estimate.status == EstimateStatus.BUDGET_EXHAUSTED and run.estimate.value is None
    assert run.estimate.draws == 1 and len(run.estimation_trace) == 2
    assert run.estimation_trace[-1]["reward"] is None
    assert run.estimate.cost.base_calls == budget.raw_nfe == 3
    assert run.estimate.cost.oracle_calls == 1 and budget.snapshot().reserved_base_sequences == 0


def test_completion_and_external_group_reservations_are_preserved(toy_backbone, cars_reward_factory):
    estimator = make_estimator(toy_backbone, cars_reward_factory(), discovery=1, draws=3)
    state = MaskedState((0, 0, 2), estimator.policy.rules)
    required = estimator.required_work(state)
    budget = BudgetTracker(base_sequence_limit=required + 2)
    with budget.reserve_completion(base_sequences=2):
        with budget.reserve_query(label="group", base_sequences=required) as held:
            run = estimator.estimate(state, rng=RNGStream(3), budget=budget, reservation=held, progress=False)
            assert run.estimate.value is not None
            # The estimator must not close the caller's group reservation.
            assert budget.snapshot().reserved_base_sequences == 2 + held.base_remaining
        assert budget.snapshot().reserved_base_sequences == 2
    assert budget.snapshot().reserved_base_sequences == 0


def test_memory_and_discovery_work_limits_freeze_without_reducing_m(toy_backbone, cars_reward_factory):
    state = MaskedState((0, 0, 0), toy_backbone.token_rules)
    estimator = make_estimator(toy_backbone, cars_reward_factory(), discovery=20, draws=7, max_nodes=1)
    run = estimator.estimate(state, rng=RNGStream(9), budget=BudgetTracker(base_sequence_limit=None),
                              progress=False)
    assert run.metadata["discovery_stop"] == "memory_limit_freeze" and run.trie.node_count == 1
    assert len(run.discovery_trace) == 1 and run.estimate.draws == 7
    capped = make_estimator(toy_backbone, cars_reward_factory(), discovery=20, draws=7, cap=2)
    assert capped.required_work(state) == 2 + 3 * 7
    other = capped.estimate(state, rng=RNGStream(9), budget=BudgetTracker(base_sequence_limit=23),
                            progress=False)
    assert other.metadata["discovery_stop"] == "work_limit" and other.discovery_trace == ()
    assert other.estimate.draws == 7


def test_estimation_rng_does_not_advance_with_discovery(toy_backbone, toy_joint, cars_reward_factory):
    reward = cars_reward_factory([row["tokens"] for row in toy_joint["leaves"]])
    state = MaskedState((0, 0, 0), toy_backbone.token_rules)
    runs = [make_estimator(toy_backbone, reward, discovery=count, draws=12).estimate(
        state, rng=RNGStream(8), budget=BudgetTracker(base_sequence_limit=None), progress=False)
        for count in (0, 6)]
    assert [row["tokens"] for row in runs[0].estimation_trace] == [
        row["tokens"] for row in runs[1].estimation_trace]


def test_underflow_returns_unavailable_instead_of_certified_zero():
    class Backbone:
        token_rules = TokenRules(3, 1, mask_id=0)
        checkpoint_id, tokenizer_id, time_conditioning = "tiny-model", "tiny-tokenizer", False

        def logits_batch(self, *args, **kwargs):
            raise AssertionError("Tiny probabilities are already cached")

    backbone = Backbone()
    reward = SimpleNamespace(token_rules=backbone.token_rules, tokenizer_sha256=backbone.tokenizer_id,
                             calibration=SimpleNamespace(calibration_id="tiny-pass95",
                                                         checkpoint_sha256=backbone.checkpoint_id))
    oracle = FiniteLanguageOracle(backbone.token_rules, [(2,)])
    estimator = CARSValueEstimator(RhoCompletion(backbone), reward, oracle, discovery_attempts=0, draws=4)
    state = MaskedState((0,), backbone.token_rules)
    trie = FailureTrie(state, policy_id=estimator.policy.policy_id, reference_id=estimator.policy.reference_id,
                       oracle_id=oracle.oracle_id)
    trie.cache_distribution((), (-math.inf, 0., -1000.))
    trie.record_failure((1,), oracle.check(trie.state_at((1,)), budget=BudgetTracker(base_sequence_limit=0)))
    run = estimator.estimate(state, initial_trie=trie.freeze(), rng=RNGStream(1),
                              budget=BudgetTracker(base_sequence_limit=None), progress=False)
    assert run.estimate.status == EstimateStatus.FAILED and run.estimate.value is None
    assert "underflow" in run.estimate.error and not run.trie.support_exhausted


def test_adapter_order_duplicates_leaf_anchoring_and_value_table(toy_backbone, cars_reward_factory):
    estimator = make_estimator(toy_backbone, cars_reward_factory(), discovery=1, draws=5)
    cache = CARSCache(estimator.cache_metadata, max_entries=1, run_id="adapter")
    verifier = CARSVerifier(estimator, cache=cache, progress=False)
    hole, leaf, prefix = [MaskedState(tokens, toy_backbone.token_rules)
                          for tokens in ((0, 0, 2), (1, 1, 1), (1, 0, 0))]
    work = estimator.required_work(hole) + estimator.required_work(prefix)
    assert verifier.required_work([hole, leaf, hole, prefix]) == (work, 0)
    budget = BudgetTracker(base_sequence_limit=work)
    with budget.reserve_query(label="candidate-group", base_sequences=work) as held:
        results = verifier.values_for_states([hole, leaf, hole, prefix], rng=RNGStream(8),
                                              budget=budget, reservation=held)
    assert results[1].status == EstimateStatus.EXACT and results[1].value == 1
    assert results[0].value == results[2].value and results[2].cache_hit and results[2].draws == 0
    assert len(verifier.runs) == 2 and verifier.required_work([hole, leaf, prefix]) == (0, 0)
    assert sum(result.cost.base_calls for result in results) == budget.raw_nfe
    assert sum(result.cost.oracle_calls for result in results) == budget.snapshot().total.oracle_calls
    table = ValueTable(verifier, RNGStream(90), budget)
    before = budget.raw_nfe
    assert table.query([prefix, hole, leaf, hole]) == [results[3].value, results[0].value, 1., results[0].value]
    assert budget.raw_nfe == before


@pytest.mark.parametrize("method", ["rs", "ars", "cars"])
def test_rejection_methods_keep_observations_and_return_only_success(
        toy_backbone, toy_joint, cars_reward_factory, method):
    reward = cars_reward_factory([row["tokens"] for row in toy_joint["leaves"]])
    policy = RhoCompletion(toy_backbone)
    oracle = FiniteLanguageOracle(policy.rules, reward.successful_tokens)
    sampler = RejectionSampler(policy, reward, method=method, max_attempts=3, oracle=oracle)
    state = MaskedState((0, 0, 2), policy.rules)
    budget = BudgetTracker(base_sequence_limit=6)
    run = sampler.run(state, sample_id="accepted", rng=RNGStream(8), budget=budget, progress=False)
    assert run.result.status == SampleStatus.COMPLETED and run.result.reward == 1
    assert state.accepts_completion(run.result.state) and run.metadata["attempts"] == 1
    assert run.result.cost == budget.snapshot().total
    assert run.trace[0]["log_proposal"] == pytest.approx(run.trace[0]["log_q"])


def test_cars_checks_sibling_cones_while_ars_only_removes_the_sampled_leaf(
        toy_backbone, toy_joint, cars_reward_factory):
    reward = cars_reward_factory([row["tokens"] for row in toy_joint["leaves"] if row["tokens"][0] == 1])
    policy = RhoCompletion(toy_backbone)
    oracle = FiniteLanguageOracle(policy.rules, reward.successful_tokens)
    state = MaskedState((0, 0, 0), policy.rules)
    tries = {}
    for mode in ("ars", "cars"):
        trie = FailureTrie(state, policy_id=policy.policy_id, reference_id=policy.reference_id,
                           oracle_id=oracle.oracle_id)
        adaptive_draw(policy, trie, oracle, mode=mode, rng=RNGStream(7),
                      budget=BudgetTracker(base_sequence_limit=None))
        tries[mode] = trie
    assert tries["cars"].log_mass((2,)) == -math.inf
    assert tries["ars"].log_mass((2,)) != -math.inf


def test_rejection_retains_trie_between_calls_and_budget_failure_is_not_success(
        toy_backbone, cars_reward_factory):
    reward = cars_reward_factory([])
    policy = RhoCompletion(toy_backbone)
    oracle = FiniteLanguageOracle(policy.rules, [])
    sampler = RejectionSampler(policy, reward, method="ars", max_attempts=1, oracle=oracle)
    state = MaskedState((0, 0, 0), policy.rules)
    first = sampler.run(state, sample_id="one", rng=RNGStream(8),
                        budget=BudgetTracker(base_sequence_limit=3), progress=False)
    snapshot = sampler.snapshot(state)
    second = sampler.run(state, sample_id="two", rng=RNGStream(8),
                         budget=BudgetTracker(base_sequence_limit=3), progress=False)
    assert first.result.status == second.result.status == SampleStatus.FAILED
    assert first.result.state != second.result.state  # The previous failed leaf has zero proposal mass.
    assert sampler.snapshot(state).root_mass < snapshot.root_mass
    assert second.trace[0]["log_root_mass"] == pytest.approx(math.log(snapshot.root_mass))
    assert second.trace[0]["log_proposal"] == pytest.approx(
        second.trace[0]["log_q"] - second.trace[0]["log_root_mass"])
    assert snapshot.is_frozen and snapshot.root_mass > 0
    blocked = sampler.run(state, sample_id="blocked", rng=RNGStream(9),
                          budget=BudgetTracker(base_sequence_limit=2), progress=False)
    assert blocked.result.status == SampleStatus.BUDGET_EXHAUSTED and blocked.result.reward is None
    assert blocked.metadata["attempts"] == 0 and blocked.result.cost.base_calls == 0
