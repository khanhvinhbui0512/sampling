"""Shared-kernel, cost, EOS/PAD and journal integration using finite fixtures."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from src.completion import ReferenceCompletion
from src.config import load_config
from src.hybrid import build_hybrid
from src.mdm_vgb import MDMVGBSampler, Proposal, log_depth_coefficient
from src.schemas import GROUPS
from src.tasks.generation import BaselineSampler, validate_generation
from src.tracking import CostTracker
from src.values import AnchoredValueProvider
from test_values import CountedModel, OracleChecker, tiny_tokenizer  # noqa: F401


class ExactProvider(AnchoredValueProvider):
    def __init__(self, distribution, tokenizer, checker, *, tracker=None):
        super().__init__(tokenizer, checker, context={"model": "finite-joint-v1"},
                         target="finite_joint_conditional", tracker=tracker)
        self.distribution = distribution

    def values(self, states):
        result = []
        for state in states:
            key, value = self._lookup(state)
            if value is None:
                value = self._terminal(state) if state.is_terminal else self.distribution.exact_value(state)
                value = self._put(key, value)
            result.append(value)
        return result


def search_options(**updates):
    options = load_config("configs/mdm_vgb.yaml")["search"]
    options.update(updates)
    return options


def make_sampler(toy_model, distribution, tokenizer, *, tracker=None, seed=9, **settings):
    checker = OracleChecker(tracker=tracker)
    model = toy_model if tracker is None else CountedModel(toy_model, tracker)
    provider = ExactProvider(distribution, tokenizer, checker, tracker=tracker)
    return MDMVGBSampler(model, provider, tokenizer, checker, search_options(**settings), seed=seed, tracker=tracker)


def test_full_enumeration_weights_match_independent_balanced_formula(toy_model, toy_distribution,
                                                                    tiny_tokenizer, toy_initial_state):
    sampler = make_sampler(toy_model, toy_distribution, tiny_tokenizer, K="all")
    state = toy_initial_state.with_tokens({0: 3})
    forward, backward = sampler.proposals(state)
    weights = {proposal.state.token_ids: math.exp(proposal.log_weight) for proposal in forward}
    assert weights[(3, 3, 2, 0)] == 0.  # terminal zero must remain zero despite floor
    assert weights[(3, 4, 2, 0)] == pytest.approx(2/3)
    assert len(backward) == 1 and backward[0].positions == (0,)
    assert math.exp(backward[0].log_weight) == pytest.approx(2/3)
    for n in range(2, 8):
        for k in range(1, n):
            assert (n-k) * math.exp(log_depth_coefficient(k, n, 1)) == pytest.approx(
                k * math.exp(log_depth_coefficient(k-1, n, 1)))


def test_known_input_and_locked_suffix_survive_search_and_fallback(toy_model, toy_distribution,
                                                                 tiny_tokenizer, toy_initial_state):
    root = toy_initial_state.with_tokens({1: 4})
    sampler = make_sampler(toy_model, toy_distribution, tiny_tokenizer, N=1, fallback_forward=True)
    sampler.start(root)
    result = sampler.attempt()
    assert result.state.token_ids[1:] == root.token_ids[1:]
    assert result.state.locked[1:] == (True, True, True)
    assert result.state.is_terminal and result.reward_result is not None


def test_learned_and_hybrid_use_identical_kernel_for_equal_provider_values(toy_model, toy_distribution,
                                                                         tiny_tokenizer, toy_initial_state, monkeypatch):
    left_tracker, right_tracker = CostTracker(run_id="left"), CostTracker(run_id="right")
    left = make_sampler(toy_model, toy_distribution, tiny_tokenizer, tracker=left_tracker, seed=5)
    checker = OracleChecker(tracker=right_tracker)
    provider = ExactProvider(toy_distribution, tiny_tokenizer, checker, tracker=right_tracker)
    monkeypatch.setattr("src.hybrid.CARSValueProvider.from_mdlm", lambda *args, **kwargs: provider)
    outer = CountedModel(toy_model, right_tracker)
    inner = CountedModel(toy_model, right_tracker, inner=True)
    right = build_hybrid(outer, inner, tiny_tokenizer, checker, search_options(),
                         {"m": 16, "trie_max_nodes": 32, "cache_capacity": 128}, seed=5, tracker=right_tracker)
    assert type(left) is type(right) is MDMVGBSampler
    left.start(toy_initial_state)
    right.start(toy_initial_state)
    for _ in range(40):
        lresult, rresult = left.step(), right.step()
        assert left.current == right.current and left.stats == right.stats
        if lresult is not None:
            assert lresult == rresult
            break
    assert left_tracker.counters == right_tracker.counters
    assert left_tracker.raw_nfe == left_tracker.counters["markov_steps"] + left.stats["fallback_nfe"]


def test_search_step_budget_and_fallback_nfe_are_separate(toy_model, toy_distribution,
                                                         tiny_tokenizer, toy_initial_state, monkeypatch):
    tracker = CostTracker(run_id="fallback")
    sampler = make_sampler(toy_model, toy_distribution, tiny_tokenizer, tracker=tracker, N=1, fallback_forward=True)
    sampler.start(toy_initial_state)
    # A shortlist with no positive transitions consumes search steps as stays.
    monkeypatch.setattr(sampler, "proposals", lambda state: ([], []))
    result = sampler.attempt()
    assert sampler.steps == sampler.limit == 2
    assert tracker.counters["markov_steps"] == 2
    assert result.stats["fallback_forward"] == result.stats["fallback_nfe"] == 2
    assert tracker.raw_nfe == 2 and tracker.counters["complete_candidates"] == 1


def test_global_budget_stops_without_an_extra_value_query(toy_model, toy_distribution,
                                                        tiny_tokenizer, toy_initial_state):
    tracker = CostTracker(run_id="search-budget", max_model_nfe=1)
    sampler = make_sampler(toy_model, toy_distribution, tiny_tokenizer, tracker=tracker)
    sampler.start(toy_initial_state)
    result = sampler.attempt()
    assert result.kind == "budget_exhausted" and tracker.raw_nfe == 1
    assert tracker.counters["complete_candidates"] == 0
    assert tracker.counters["markov_steps"] == 1


def test_nN_exhaustion_leaves_an_incomplete_attempt_without_unpaid_completion(toy_model, toy_distribution,
                                                                           tiny_tokenizer, toy_initial_state, monkeypatch):
    tracker = CostTracker(run_id="nN-only")
    sampler = make_sampler(toy_model, toy_distribution, tiny_tokenizer, tracker=tracker, N=1, fallback_forward=False)
    sampler.start(toy_initial_state)
    monkeypatch.setattr(sampler, "proposals", lambda state: ([], []))
    result = sampler.attempt()
    assert result.kind == "budget_exhausted" and result.reason == "search_step_budget"
    assert not result.state.is_terminal and result.reward_result is None
    assert tracker.counters["markov_steps"] == 2 and tracker.raw_nfe == 0


def test_kernel_snapshot_replays_partial_search(toy_model, toy_distribution, tiny_tokenizer, toy_initial_state):
    sampler = make_sampler(toy_model, toy_distribution, tiny_tokenizer, seed=123)
    sampler.start(toy_initial_state)
    sampler.step()
    restored = make_sampler(toy_model, toy_distribution, tiny_tokenizer, seed=999)
    restored.load_state_dict(sampler.state_dict())
    assert sampler.attempt() == restored.attempt()


def test_momentum_is_explicit_and_switches_direction(toy_model, toy_distribution, tiny_tokenizer, toy_initial_state, monkeypatch):
    sampler = make_sampler(toy_model, toy_distribution, tiny_tokenizer, momentum=True, chi=1.0)
    sampler.start(toy_initial_state)
    assert sampler.direction == "down"
    with pytest.raises(ValueError):
        make_sampler(toy_model, toy_distribution, tiny_tokenizer, momentum=False, chi=1.0)
    forward = [Proposal("forward", sampler.current, (0,), 0., .5)]
    backward = [Proposal("backward", sampler.current, (1,), math.log(3.), .5)]
    monkeypatch.setattr(sampler, "proposals", lambda state: (forward, backward))

    def choose_switch(weights, count, *, generator):
        assert (weights/weights.sum()).tolist() == pytest.approx([1/3, 2/3])
        return torch.tensor([1])

    monkeypatch.setattr(torch, "multinomial", choose_switch)
    sampler.step()
    assert sampler.direction == "up" and sampler.stats["switches"] == 1


def test_baseline_preserves_context_replays_rng_and_counts_real_forwards(toy_model, tiny_tokenizer, toy_initial_state):
    class ScheduledModel(CountedModel):
        device = torch.device("cpu")
        model = SimpleNamespace(specification={"diffusion": {"epsilon": 1e-5}})

        def probabilities_batch(self, states, *, times):
            return self.logits_batch(states).softmax(-1)

    def baseline(tracker, seed):
        model = ScheduledModel(toy_model, tracker)
        completion = ReferenceCompletion(model, steps=3, seed=seed)
        return BaselineSampler(completion, tiny_tokenizer, OracleChecker(tracker=tracker), tracker)

    tracker, replay_tracker = CostTracker(run_id="base"), CostTracker(run_id="base-replay")
    sampler = baseline(tracker, 3)
    sampler.start(toy_initial_state)
    payload = sampler.state_dict()
    result = sampler.attempt()
    replay = baseline(replay_tracker, 999)
    replay.load_state_dict(payload)
    assert replay.attempt() == result
    assert result.kind == "complete" and result.state.token_ids[2:] == (2, 0)
    assert 1 <= tracker.raw_nfe <= 4 and tracker.counters["inner_mdlm_forward_calls"] == 0
    assert tracker.counters["leaf_visits"] == tracker.counters["complete_candidates"] == 1
    assert tracker.counters == replay_tracker.counters


def test_configs_share_search_settings_and_reject_verifier_inputs_for_hybrid():
    args = ["group=acrylates", "generation.model.checkpoint_manifest=checkpoints/mock/manifest.json"]
    learned = load_config("configs/mdm_vgb.yaml", overrides=[*args,
        "value.learned.checkpoint_manifest=checkpoints/mock/verifier/manifest.json"],
        validator=lambda cfg: validate_generation(cfg, "mdm_vgb"))
    hybrid = load_config("configs/hybrid.yaml", overrides=args, validator=lambda cfg: validate_generation(cfg, "hybrid"))
    assert learned["search"] == hybrid["search"]
    assert hybrid["value"]["m"] == 16 and learned["search"]["N"] == 16
    with pytest.raises(ValueError):
        load_config("configs/hybrid.yaml", overrides=[*args, "value.learned.checkpoint_manifest=checkpoints/wrong/manifest.json"],
                    validator=lambda cfg: validate_generation(cfg, "hybrid"))


@pytest.mark.parametrize("method", ["mdm_vgb", "hybrid"])
def test_generation_resumes_inside_attempt_without_duplicate_journals(toy_model, toy_distribution, tiny_tokenizer,
                                                                     toy_initial_state, artifact_store, monkeypatch, method):
    import src.tasks.generation as task

    monkeypatch.setattr(task, "generation_dependencies", lambda config, store: {})
    monkeypatch.setattr(task, "read_benchmark_assets", lambda config: ({}, {group: [] for group in GROUPS}))

    def builder(context, checker):
        sampler = make_sampler(toy_model, toy_distribution, tiny_tokenizer, tracker=context.tracker,
                               seed=context.identity.seed, **context.config["search"])
        return sampler, toy_initial_state, {"device": "cpu"}

    monkeypatch.setattr(task, "build_sampler", builder)
    args = ["--group", GROUPS[0], "--smoke", "--checkpoint", "checkpoints/mock/manifest.json",
            "--artifact-root", artifact_store.root, "--set", "protocol.target_unique=1",
            "--set", "protocol.max_root_attempts=100"]
    if method == "mdm_vgb":
        args.extend(["--verifier-checkpoint", "checkpoints/mock/verifier/manifest.json"])
    commits = []

    def interrupt():
        commits.append(True)
        if len(commits) == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        task.execute_generation(method, [*args, "--run-id", "resumed"], on_checkpoint=interrupt)
    resumed = task.execute_generation(method, [*args, "--run-id", "resumed", "--resume"])
    reference = task.execute_generation(method, [*args, "--run-id", "reference"])
    assert resumed.status == reference.status == "completed"
    left_dir, right_dir = (artifact_store.run_directory(record.identity) for record in (resumed, reference))
    left = list(artifact_store.read_jsonl(left_dir + "/attempts.jsonl"))
    right = list(artifact_store.read_jsonl(right_dir + "/attempts.jsonl"))
    assert [(row["kind"], row["smiles"]) for row in left] == [(row["kind"], row["smiles"]) for row in right]
    assert [row["attempt_id"] for row in left] == list(range(1, len(left)+1))
    assert resumed.tracking.counters == reference.tracking.counters
    artifact_store.require_manifest(left_dir + "/generation_manifest.json", kind="generation")
    assert len(list(artifact_store.read_jsonl(left_dir + "/samples.jsonl"))) == 1
