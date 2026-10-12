"""Step 6 toy checks, written for review and intentionally not executed.

These check local probabilities, finite-budget policies and actual work.
They do not establish reference consistency or a stationary leaf law.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.budget import BudgetExceeded, BudgetTracker
from src.config import load_config
from src.randomness import RNGStream
from src.sampling.base import BaseSampler
from src.sampling.bon import BoNSampler
from src.sampling.candidates import Candidate, CandidateBuilder, CandidateSettings, ValueTable, depth_coefficients, edge_weights
from src.sampling.completion import RhoCompletion, cost_since
from src.sampling.momentum import MomentumSampler, momentum_choices
from src.sampling.terminal import backtrack, weighted_index
from src.sampling.vgb import VGBSampler
from src.sampling.vgr import VGRSampler
from src.types import CostPhase, CostRecord, EstimateStatus, MaskedState, SampleStatus, ValueEstimate
from src.verifiers.interface import Verifier, value_target
from src.verifiers.monte_carlo import MonteCarloVerifier
from src.verifiers.neural import NeuralVerifier, TokenStateVerifier


class ToyReward:
    """Chemistry-free exact binary reward, with validity separate from success."""

    def __init__(self, backbone, succeeds):
        self.token_rules = backbone.token_rules
        self.tokenizer_sha256 = backbone.tokenizer_id
        self.calibration = SimpleNamespace(calibration_id="step6-pass95",
                                           checkpoint_sha256=backbone.checkpoint_id)
        self.succeeds = succeeds

    def evaluate(self, state, *, require_binary=True):
        assert require_binary and state.is_complete
        reward = int(self.succeeds(state))
        return SimpleNamespace(valid=True, score=0.8 if reward else 0.2,
                               reward=reward, molecule=None)


class ToyVerifier(Verifier):
    def __init__(self, backbone, reward, value=0.5):
        super().__init__(value_target(RhoCompletion(backbone), reward), reward)
        self.value, self.queried = value, []

    def required_work(self, states):
        return 0, sum(not state.is_complete for state in states)

    def _estimate_batch(self, states, *, rng, budget, reservation):
        del rng
        self.queried.extend(states)
        before = budget.snapshot().total
        with budget.verifier_forward(batch_size=len(states), phase=CostPhase.VERIFICATION,
                                     reservation=reservation):
            values = [self.value(state) if callable(self.value) else self.value for state in states]
        cost = cost_since(budget, before)
        return [ValueEstimate(self.target, EstimateStatus.ESTIMATED, value,
                              cost=cost if i == 0 else CostRecord()) for i, value in enumerate(values)]


@pytest.fixture
def problem(fake_backbone_factory, token_rules):
    probabilities = [0.] * token_rules.vocab_size
    probabilities[5] = 1.  # Deterministic C makes multiplicity checks exact.
    backbone = fake_backbone_factory(probabilities)
    reward = ToyReward(backbone, lambda state: True)
    base = BaseSampler(backbone, reward, method="base_rho")
    verifier = ToyVerifier(backbone, reward)
    root = MaskedState((token_rules.mask_id,) * token_rules.sequence_length, token_rules)
    return SimpleNamespace(backbone=backbone, reward=reward, base=base, verifier=verifier, root=root)


def test_depth_coefficients_preserve_relative_mass():
    # n=6, k=2: s_f=1/C(5,2)=.1; s_b=1/C(5,1)=.2.
    assert depth_coefficients(6, 2) == (0.5, 1.0)
    assert depth_coefficients(6, 0) == (1.0, 0.0)
    assert depth_coefficients(6, 6) == (0.0, 1.0)


def test_paper_edge_weights_and_terminal_gate_exception():
    forward, backward = edge_weights(6, 2, current=.4, child=.8, parent=.3,
                                     proposal_mass=.25, lambda_value=2.)
    assert forward == pytest.approx(.5 * .25 * .8 * .4**2)
    assert backward == pytest.approx(.4 * .3**2)
    # A successful leaf remains reachable from a zero-valued parent (Def. 4.2).
    forward, _ = edge_weights(6, 5, current=0., child=1., proposal_mass=.25,
                              lambda_value=2., child_terminal=True)
    assert forward == pytest.approx(.25)
    _, backward = edge_weights(6, 6, current=1., parent=0., lambda_value=2., state_terminal=True)
    assert backward == 1.


def test_lambda_zero_and_exact_zero_support():
    fwd, bwd = edge_weights(6, 2, current=0., child=.8, parent=0., lambda_value=0.)
    assert fwd == pytest.approx(.4) and bwd == 0.
    assert edge_weights(6, 2, current=0., child=.8, parent=0., lambda_value=1.) == (0., 0.)
    assert edge_weights(6, 2, current=.5, child=0., parent=0., lambda_value=1.) == (0., 0.)
    assert weighted_index([0., 0.], RNGStream(1)) is None


def test_sampled_duplicates_are_counted_once_with_multiplicity(problem):
    tracker = BudgetTracker(base_sequence_limit=None)
    table = ValueTable(problem.verifier, RNGStream(2), tracker)
    builder = CandidateBuilder(problem.backbone, CandidateSettings(L_f=2, K=8))
    edges = builder.build(problem.root, editable=problem.root.masked_positions,
                          values=table, rng=RNGStream(1), budget=tracker)
    assert len(edges) == 2
    assert all(edge.multiplicity == 8 for edge in edges)
    assert all(edge.proposal_mass == 3. for edge in edges)  # (8/8) * 6/2
    assert all(edge.weight == 1.5 for edge in edges)
    assert len(problem.verifier.queried) == 2  # No needless root query at lambda=0.
    assert tracker.snapshot().total.base_calls == 1


def test_empirical_probability_is_not_multiplied_twice(problem, fake_backbone_factory):
    probabilities = [0.] * problem.root.rules.vocab_size
    probabilities[5], probabilities[6] = .25, .75
    backbone = fake_backbone_factory(probabilities)
    verifier = ToyVerifier(backbone, problem.reward, value=1.)
    tracker = BudgetTracker(base_sequence_limit=None)
    builder = CandidateBuilder(backbone, CandidateSettings(L_f=1, K=64))
    edges = builder.build(problem.root, editable=problem.root.masked_positions,
                          values=ValueTable(verifier, RNGStream(7), tracker),
                          rng=RNGStream(3), budget=tracker)
    assert sum(c.multiplicity for c in edges) == 64
    for candidate in edges:
        assert candidate.weight == pytest.approx(6 * candidate.multiplicity / 64)
    assert sum(c.weight for c in edges) == pytest.approx(6.)


def test_leaf_anchoring_ignores_neural_prediction(problem):
    problem.reward.succeeds = lambda state: False
    verifier = ToyVerifier(problem.backbone, problem.reward, value=1.)
    leaf = MaskedState((5,) * problem.root.rules.sequence_length, problem.root.rules)
    state = MaskedState((problem.root.rules.mask_id, *leaf.token_ids[1:]), leaf.rules)
    tracker = BudgetTracker(base_sequence_limit=None)
    edges = CandidateBuilder(problem.backbone).build(
        state, editable=(0,), values=ValueTable(verifier, RNGStream(4), tracker),
        rng=RNGStream(3), budget=tracker)
    assert edges[0].state.is_complete and edges[0].value == edges[0].weight == 0.
    assert verifier.queried == []
    assert tracker.snapshot().total.verifier_calls == 0
    assert tracker.snapshot().total.oracle_calls == 1


def test_lambda_zero_skips_parent_queries_and_uses_common_scale(problem):
    state = problem.root.reveal({0: 5, 1: 5})
    tracker = BudgetTracker(base_sequence_limit=None)
    edges = CandidateBuilder(problem.backbone, CandidateSettings(L_f=2, L_b=1)).build(
        state, editable=problem.root.masked_positions,
        values=ValueTable(problem.verifier, RNGStream(5), tracker), rng=RNGStream(6), budget=tracker)
    forward = [c for c in edges if c.direction == "forward"]
    backward = [c for c in edges if c.direction == "backward"]
    assert len(forward) == 2 and len(backward) == 1
    assert all(c.weight == .5 for c in forward)  # .5 depth * (4/2) * .5 value
    assert backward[0].weight == 1.  # 1 depth * (2/1) * .5 current
    assert backward[0].value is None
    assert problem.verifier.queried == [state, *(c.state for c in forward)]


def test_positive_lambda_queries_parents_and_applies_their_gate(problem):
    state = problem.root.reveal({0: 5, 1: 5})
    tracker = BudgetTracker(base_sequence_limit=None)
    edges = CandidateBuilder(problem.backbone, CandidateSettings(L_f=1, L_b=1, lambda_value=2.)).build(
        state, editable=problem.root.masked_positions,
        values=ValueTable(problem.verifier, RNGStream(5), tracker), rng=RNGStream(6), budget=tracker)
    child, parent = edges
    assert problem.verifier.queried == [state, child.state, parent.state]
    assert child.weight == pytest.approx(.5 * 4 * .5**2 * .5)
    assert parent.weight == pytest.approx(2 * .5 * .5**2)


def test_low_conf_uses_one_raw_logit_forward_and_locks_context(problem, fake_backbone_factory):
    probabilities = [0.] * problem.root.rules.vocab_size
    probabilities[5], probabilities[6] = .75, .25
    backbone = fake_backbone_factory(probabilities)
    verifier = ToyVerifier(backbone, problem.reward)
    state = problem.root.reveal({0: 5, 1: 6, 2: 6})
    tracker = BudgetTracker(base_sequence_limit=None)
    edges = CandidateBuilder(backbone, CandidateSettings(L_b=1)).build(
        state, editable=(0, 1, 3, 4, 5), values=ValueTable(verifier, RNGStream(4), tracker),
        rng=RNGStream(3), budget=tracker)
    parents = [c for c in edges if c.direction == "backward"]
    assert len(parents) == 1 and parents[0].position == 1
    assert all(c.state.token_ids[2] == 6 for c in edges)  # Initially observed context.
    assert tracker.snapshot().total.base_calls == 1


def test_pure_backtrack_preserves_context_and_does_not_charge_nfe(problem):
    leaf = MaskedState((5,) * problem.root.rules.sequence_length, problem.root.rules)
    tracker = BudgetTracker(base_sequence_limit=0)
    parent = backtrack(leaf, (2,), RNGStream(1))
    assert parent.masked_positions == (2,)
    assert parent.token_ids[0] == 5
    assert tracker.raw_nfe == tracker.snapshot().total.base_sequences == 0
    assert backtrack(problem.root, problem.root.masked_positions, RNGStream(1)) is None


def test_values_freeze_by_full_state_and_duplicate_queries_are_free(problem):
    tracker = BudgetTracker(base_sequence_limit=None)
    table = ValueTable(problem.verifier, RNGStream(3), tracker)
    a, b = problem.root.reveal({0: 5}), problem.root.reveal({1: 5})
    assert table.query([a, a]) == [.5, .5]
    problem.verifier.value = .9
    assert table.query([a, b]) == [.5, .9]
    assert problem.verifier.queried == [a, b]
    assert tracker.snapshot().total.verifier_sequences == 2


def test_whole_mc_candidate_group_is_reserved_before_any_draw(problem):
    verifier = MonteCarloVerifier(RhoCompletion(problem.backbone), problem.reward, draws=2)
    states = [problem.root.reveal({0: 5}), problem.root.reveal({1: 5})]
    # Each query costs 10 sequences; 15 allows one but not the whole group.
    tracker = BudgetTracker(base_sequence_limit=15)
    table = ValueTable(verifier, RNGStream(1), tracker)
    with pytest.raises(BudgetExceeded, match="entire candidate"):
        table.query(states)
    assert tracker.raw_nfe == 0 and table.estimates == {}
    assert tracker.snapshot().reserved_base_sequences == 0


def test_unavailable_value_is_not_converted_to_zero(problem):
    class Unavailable(ToyVerifier):
        def _estimate_batch(self, states, **kwargs):
            return [ValueEstimate(self.target, EstimateStatus.BUDGET_EXHAUSTED, None,
                                  error="unavailable") for _ in states]

    table = ValueTable(Unavailable(problem.backbone, problem.reward), RNGStream(1),
                       BudgetTracker(base_sequence_limit=None))
    with pytest.raises(BudgetExceeded, match="unavailable"):
        table.query([problem.root])
    assert table.estimates == {}


@pytest.mark.parametrize("sampler_class", [VGRSampler, VGBSampler, MomentumSampler])
def test_reserved_completion_at_exact_hard_cap(problem, sampler_class):
    # All six sequence units are held for fallback, leaving none for guidance.
    tracker = BudgetTracker(base_sequence_limit=6)
    sampler = sampler_class(problem.backbone, problem.verifier, problem.base)
    run = sampler.run(problem.root, sample_id="reserved", rng=RNGStream(9), budget=tracker)
    assert run.result.status == SampleStatus.COMPLETED and run.result.reward == 1
    assert run.metadata["fallback_used"]
    assert tracker.snapshot().total.base_sequences == 6
    assert tracker.snapshot().total.verifier_calls == 0
    assert tracker.snapshot().by_phase["completion"].base_sequences == 6
    assert tracker.snapshot().reserved_base_sequences == 0


def test_unresolved_estimate_finishes_failed_sample_and_releases_completion(problem):
    class Unresolved(ToyVerifier):
        def _estimate_batch(self, states, **kwargs):
            return [ValueEstimate(self.target, EstimateStatus.FAILED, None,
                                  error="Unresolved native denominator") for _ in states]

    verifier = Unresolved(problem.backbone, problem.reward)
    tracker = BudgetTracker(base_sequence_limit=None)
    run = VGBSampler(problem.backbone, verifier, problem.base).run(
        problem.root, sample_id="unresolved", rng=RNGStream(4), budget=tracker)
    assert run.result.status == SampleStatus.FAILED and run.result.reward is None
    assert "Unresolved native denominator" in run.result.error
    assert run.trace[-1]["event"] == "value_unavailable" and not run.metadata["fallback_used"]
    assert tracker.snapshot().reserved_base_sequences == tracker.snapshot().reserved_verifier_sequences == 0


def test_unfinished_when_completion_cannot_be_reserved(problem):
    tracker = BudgetTracker(base_sequence_limit=5)
    run = VGBSampler(problem.backbone, problem.verifier, problem.base).run(
        problem.root, sample_id="too-small", rng=RNGStream(1), budget=tracker)
    assert run.result.status == SampleStatus.BUDGET_EXHAUSTED
    assert run.result.reward is None and run.result.smiles is None and not run.metadata["success"]
    assert tracker.raw_nfe == 0


def test_vgr_zero_values_fall_back_to_empirical_base_candidates(problem):
    verifier = ToyVerifier(problem.backbone, problem.reward, value=0.)
    run = VGRSampler(problem.backbone, verifier, problem.base).run(
        problem.root, sample_id="vgr-zero", rng=RNGStream(4),
        budget=BudgetTracker(base_sequence_limit=None))
    assert run.result.reward == 1
    assert run.trace[0]["event"] == "zero_value_reference_forward"
    assert not any("backtrack" in step["event"] for step in run.trace)
    assert not run.metadata["fallback_used"]


def test_vgb_zero_root_uses_unguided_fallback_without_epsilon(problem):
    verifier = ToyVerifier(problem.backbone, problem.reward, value=0.)
    tracker = BudgetTracker(base_sequence_limit=None)
    run = VGBSampler(problem.backbone, verifier, problem.base).run(
        problem.root, sample_id="zero-root", rng=RNGStream(4), budget=tracker)
    assert run.trace[0]["event"] == "zero_mass_no_parent"
    assert all(edge["weight"] == 0 for edge in run.trace[0]["edges"])
    assert run.metadata["fallback_used"] and tracker.snapshot().total.base_sequences == 7


def test_vgb_zero_interior_backtrack_is_an_explicit_escape(problem):
    class ForcedFirstReveal(VGBSampler):
        def _transition(self, candidates, direction, rng):
            # Enter a zero-valued interior state to isolate its escape policy.
            if all(c.direction == "forward" for c in candidates):
                return "forced_forward", candidates[0].state, direction
            return super()._transition(candidates, direction, rng)

    verifier = ToyVerifier(problem.backbone, problem.reward, value=0.)
    tracker = BudgetTracker(base_sequence_limit=None)
    run = ForcedFirstReveal(problem.backbone, verifier, problem.base, step_limit=2).run(
        problem.root, sample_id="zero-interior", rng=RNGStream(4),
        budget=tracker)
    assert run.trace[1]["event"] == "zero_mass_backtrack"
    assert run.trace[1]["next_tokens"] == problem.root.token_ids
    assert tracker.snapshot().total.base_sequences == 8  # Two proposals + six completion calls.
    assert run.metadata["fallback_used"]


def test_valid_low_qed_leaf_is_backtracked_without_extra_base_forward(problem):
    problem.reward.succeeds = lambda state: False
    state = problem.root.reveal({p: 5 for p in range(1, 6)})
    verifier = ToyVerifier(problem.backbone, problem.reward, value=0.)

    class ForcedLeaf(VGBSampler):
        # Isolate the terminal policy from edge weights, which correctly assign
        # zero to this failed leaf and would not normally select it.
        def _transition(self, candidates, direction, rng):
            return "forward", next(c.state for c in candidates if c.direction == "forward"), direction

    tracker = BudgetTracker(base_sequence_limit=None)
    run = ForcedLeaf(problem.backbone, verifier, problem.base, step_limit=2).run(
        state, sample_id="valid-failure", rng=RNGStream(4), budget=tracker)
    assert run.trace[1]["event"] == "failed_leaf_backtrack"
    assert run.trace[1]["valid"] is True and run.trace[1]["reward"] == 0
    assert tracker.snapshot().total.base_calls == 2  # One candidate + one completion, no remask call.
    assert run.result.valid is True and run.result.reward == 0 and not run.metadata["success"]
    assert all(run.result.state.token_ids[p] == 5 for p in range(1, 6))


@pytest.mark.parametrize("sampler_class", [VGBSampler, MomentumSampler])
def test_mc_and_neural_share_guided_sampler_interface(problem, sampler_class):
    mc = MonteCarloVerifier(RhoCompletion(problem.backbone), problem.reward, draws=2)
    neural = NeuralVerifier(TokenStateVerifier(problem.root.rules.vocab_size, 6, hidden_dim=8,
                                               num_layers=1, num_heads=2, dropout=0.),
                            problem.reward, mc.target)
    for verifier in (mc, neural):
        tracker = BudgetTracker(base_sequence_limit=None)
        run = sampler_class(problem.backbone, verifier, problem.base, step_limit=2).run(
            problem.root, sample_id="adapter", rng=RNGStream(1), budget=tracker)
        assert run.result.status == SampleStatus.COMPLETED and run.result.reward == 1
        assert run.result.target == verifier.target
        assert tracker.snapshot().reserved_base_sequences == tracker.snapshot().reserved_verifier_sequences == 0


def test_bon_accounts_all_N_and_uses_first_binary_tie(problem):
    tracker = BudgetTracker(base_sequence_limit=18)
    run = BoNSampler(problem.base, N=3).run(problem.root, sample_id="bon", rng=RNGStream(5), budget=tracker)
    assert run.metadata["completed_candidates"] == 3 and run.metadata["selected_index"] == 0
    assert len(run.trace) == 3  # Duplicates are not deduplicated.
    assert run.result.cost.base_sequences == 18 and run.result.cost.base_calls == 6
    assert run.result.cost.oracle_calls == 3  # Winner is not re-evaluated.
    assert tracker.snapshot().reserved_base_sequences == 0


def test_bon_insufficient_budget_does_not_start_partial_group(problem):
    tracker = BudgetTracker(base_sequence_limit=17)
    run = BoNSampler(problem.base, N=3).run(problem.root, sample_id="bon-short", rng=RNGStream(1), budget=tracker)
    assert run.result.status == SampleStatus.BUDGET_EXHAUSTED
    assert run.metadata["completed_candidates"] == 0 and tracker.raw_nfe == 0


def test_configured_native_bon_budget_allows_all_N(problem):
    config = load_config("configs/mdm_vgb.yaml")
    base = BaseSampler(problem.backbone, problem.reward, method=config["base_method"],
                       native_settings=config["native_settings"])
    tracker = BudgetTracker(base_sequence_limit=config["budget"]["base_sequence_limit"])
    run = BoNSampler(base, **config["bon"]).run(problem.root, sample_id="configured-bon", rng=RNGStream(1),
                                               budget=tracker)
    assert run.result.status == SampleStatus.COMPLETED
    assert run.metadata["completed_candidates"] == config["bon"]["N"]
    assert tracker.snapshot().total.base_sequences <= config["budget"]["base_sequence_limit"]
    assert tracker.snapshot().reserved_base_sequences == 0


def test_bon_unsuccessful_N_still_has_completed_output(problem):
    problem.reward.succeeds = lambda state: False
    run = BoNSampler(problem.base, N=3).run(problem.root, sample_id="bon-fail", rng=RNGStream(1),
                                          budget=BudgetTracker(base_sequence_limit=None))
    assert run.result.status == SampleStatus.COMPLETED
    assert run.result.reward == 0 and not run.metadata["success"]
    assert run.metadata["selected_index"] == 0


def test_bon_qed_ranking_is_explicit_project_option(problem):
    class RankedBase:
        method_id, generator, reward = problem.base.method_id, problem.base.generator, problem.reward
        target, policy_id = problem.base.target, problem.base.policy_id

        def generate_batch(self, states, **kwargs):
            batch = problem.base.generate_batch(states, **kwargs)
            terminals = tuple(SimpleNamespace(valid=True, score=score, reward=1, molecule=None)
                              for score in (.7, .9, .8))
            return replace(batch, terminals=terminals)

    binary = BoNSampler(RankedBase(), N=3).run(problem.root, sample_id="binary", rng=RNGStream(2),
                                             budget=BudgetTracker(base_sequence_limit=None))
    qed = BoNSampler(RankedBase(), N=3, ranking="qed").run(problem.root, sample_id="qed", rng=RNGStream(2),
                                                         budget=BudgetTracker(base_sequence_limit=None))
    assert binary.metadata["selected_index"] == 0 and qed.metadata["selected_index"] == 1


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("chi", [0., .5, 1.])
def test_momentum_paper_mass_and_upstream_omitted_stays(problem, direction, chi):
    edges = [Candidate(problem.root, "forward", 0, 1, 1., weight=2.),
             Candidate(problem.root, "backward", 1, 1, 1., weight=3.)]
    paper = momentum_choices(edges, direction, chi=chi, include_stay=True)
    upstream = momentum_choices(edges, direction, chi=chi, include_stay=False)
    assert sum(w for _, _, w in paper) == pytest.approx(5.)
    assert sum(w for _, _, w in upstream) == pytest.approx(5. - 2 * chi)
    assert next(w for event, _, w in paper if event == "stay") == 2 * chi
    expected_switch = (3. if direction == "down" else 2.) - 2 * chi
    assert next(w for event, _, w in paper if event == "switch") == expected_switch


def test_momentum_equal_mass_chi_one_and_direction_rng_record(problem):
    edges = [Candidate(problem.root, "forward", 0, 1, 1., weight=1.),
             Candidate(problem.root, "backward", 1, 1, 1., weight=1.)]
    choices = momentum_choices(edges, "down", chi=1., include_stay=True)
    assert [(event, weight) for event, _, weight in choices] == [("forward", 1.), ("switch", 0.), ("stay", 1.)]
    run = MomentumSampler(problem.backbone, problem.verifier, problem.base, step_limit=2).run(
        problem.root, sample_id="momentum-state", rng=RNGStream(8), budget=BudgetTracker(base_sequence_limit=None))
    assert run.direction in ("down", "up") and run.metadata["seed"] == 8
    assert RNGStream.from_state_dict(run.rng_state["sampler"]).state_dict() == run.rng_state["sampler"]
    assert "estimation" in run.rng_state and run.metadata["include_stay"] is True


def test_one_sided_momentum_switches_or_moves_with_no_artificial_support(problem):
    edge = Candidate(problem.root, "forward", 0, 1, 1., weight=2.)
    up = momentum_choices([edge], "up", chi=1.)
    down = momentum_choices([edge], "down", chi=1.)
    assert [(event, weight) for event, _, weight in up] == [("switch", 2.), ("stay", 0.)]
    assert [(event, weight) for event, _, weight in down] == [("forward", 2.), ("switch", 0.), ("stay", 0.)]


def test_cooperative_timeout_records_failure_and_releases_reservation(problem, monkeypatch):
    readings = iter((0., 2.))
    monkeypatch.setattr("src.sampling.vgb.time.monotonic", lambda: next(readings))
    tracker = BudgetTracker(base_sequence_limit=6)
    run = VGBSampler(problem.backbone, problem.verifier, problem.base, time_limit_seconds=1.).run(
        problem.root, sample_id="timeout", rng=RNGStream(1), budget=tracker)
    assert run.result.status == SampleStatus.FAILED and run.result.error == "time_limit"
    assert run.result.reward is None and not run.metadata["success"]
    assert tracker.raw_nfe == 0 and tracker.snapshot().reserved_base_sequences == 0


def test_completion_finishing_after_deadline_is_still_timeout_failure(problem, monkeypatch):
    # Check before fallback, let its single unguided operation finish, then
    # classify the over-deadline result as failure without interrupting it.
    readings = iter((0., .5, 2.))
    monkeypatch.setattr("src.sampling.vgb.time.monotonic", lambda: next(readings))
    tracker = BudgetTracker(base_sequence_limit=6)
    run = VGBSampler(problem.backbone, problem.verifier, problem.base, step_limit=0,
                     time_limit_seconds=1.).run(problem.root, sample_id="completion-timeout",
                                               rng=RNGStream(1), budget=tracker)
    assert run.result.state.is_complete and run.result.reward == 1
    assert run.result.status == SampleStatus.FAILED and not run.metadata["success"]
    assert run.result.error == "time_limit" and run.result.cost.base_sequences == 6
    assert tracker.snapshot().reserved_base_sequences == 0


def test_step_zero_native_fallback_preserves_original_context(problem):
    base = BaseSampler(problem.backbone, problem.reward, method="base_native",
                       native_settings={"steps": 2})
    state = problem.root.reveal({0: 6, 3: 6})
    tracker = BudgetTracker(base_sequence_limit=3)  # steps + optional final denoise.
    run = VGBSampler(problem.backbone, problem.verifier, base, step_limit=0).run(
        state, sample_id="native-fallback", rng=RNGStream(3), budget=tracker)
    assert run.result.status == SampleStatus.COMPLETED
    assert run.result.state.token_ids[0] == run.result.state.token_ids[3] == 6
    assert run.metadata["fallback_used"] and run.metadata["steps"] == 0
    assert tracker.snapshot().total.base_sequences <= 3
    assert tracker.snapshot().total.verifier_calls == 0
    assert tracker.snapshot().reserved_base_sequences == 0
