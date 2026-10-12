"""Finite exact distributions and CARS integration checks; no model downloads.

These tests are source for later verification, not experimental results.
The table oracle is independently enumerated; every excluded cone has reward 0.
"""

from __future__ import annotations

import io
import math
from collections import Counter

import pytest
import torch

from src.cars import ARCompletionPolicy, CARSSampler, Feedback, MDLMCompletionPolicy
from src.completion import LeftToRightCompletion
from src.config import load_config
from src.grammar import GrammarError
from src.reward import RewardChecker
from src.schemas import GROUPS, RewardResult, SmilesState
from src.tasks.generate_cars import UniqueSelector, validate_cars
from src.tokenizer import SmilesTokenizer
from src.tracking import CostTracker
from src.trie import EmptyTrie, RejectionTrie

WEIGHTS = {(0, 0): .05, (0, 1): .15, (1, 0): .25, (1, 1): .15, (2, 0): .10, (2, 1): .30}
GOOD = {(0, 1), (1, 0)}


class TablePolicy:
    metadata = {"policy": "finite-table-v1"}

    def __init__(self, *, weights=None, good=None, feedback=None, tracker=None):
        self.weights = dict(WEIGHTS if weights is None else weights)
        self.good = set(GOOD if good is None else good)
        self.width = len(next(iter(self.weights)))
        self.vocab_size = 1 + max(token for leaf in self.weights for token in leaf)
        self.custom_feedback, self.tracker = feedback, tracker
        self.calls = 0

    def context(self, root):
        return {"model": "table", "tokenizer": "integer-tokens", "state": list(root),
                "group": GROUPS[0], "reward": "two-good-leaves", "policy": self.metadata}

    def normalize(self, state):
        return tuple(state)

    def begin(self):
        pass

    def terminal(self, state):
        return len(state) == self.width

    def feedback(self, state):
        if self.custom_feedback is not None:
            return self.custom_feedback(state)
        if self.width == 2 and state and state[0] == 2:
            return Feedback(True, reason="proven failed cone")
        return Feedback(forbidden=torch.tensor([2], dtype=torch.long)) if not state and self.vocab_size == 3 else Feedback()

    def next_logs(self, state):
        def calculate():
            self.calls += 1
            matching = {leaf: p for leaf, p in self.weights.items() if leaf[:len(state)] == state}
            total = math.fsum(matching.values())
            logs = torch.full((self.vocab_size,), -torch.inf, dtype=torch.float64)
            for token in range(self.vocab_size):
                mass = math.fsum(p for leaf, p in matching.items() if leaf[len(state)] == token)
                if mass:
                    logs[token] = math.log(mass / total)
            return logs

        if self.tracker is None:
            return calculate()
        with self.tracker.model_call("mdlm"):
            return calculate()

    def advance(self, state, token):
        return (*state, token)

    def output(self, state):
        text = "leaf-" + "-".join(map(str, state))
        success = state in self.good
        return text, RewardResult(float(success), "grammar-rdkit-smarts-v1", True, True, success,
                                  canonical_smiles=text, error=None if success else "class_mismatch")


def expanded_table(policy=None, *, max_nodes=100):
    policy = policy or TablePolicy()
    trie = RejectionTrie(policy.context(()), max_nodes=max_nodes)
    prefixes = {leaf[:depth] for leaf in policy.weights for depth in range(policy.width)}
    for prefix in sorted(prefixes, key=lambda item: (len(item), item)):
        trie.expand(prefix, policy.next_logs(prefix))
    return policy, trie


def enumerated_proposal(policy, trie):
    probabilities = {}
    for leaf in policy.weights:
        log_probability = 0.0
        for depth, token in enumerate(leaf):
            prefix = leaf[:depth]
            try:
                logs = trie.conditional_logs(prefix, policy.next_logs(prefix))
            except EmptyTrie:
                log_probability = -math.inf
                break
            log_probability += float(logs[token])
        probabilities[leaf] = math.exp(log_probability)
    return probabilities


def force_tokens(monkeypatch, tokens, observe=None):
    choices = iter(tokens)

    def draw(weights, count, *, generator):
        assert count == 1
        token = next(choices)
        assert weights[token] > 0, "Test tried to force a zero-probability branch"
        if observe is not None:
            observe()
        return torch.tensor([token], dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", draw)


def test_trie_mass_overlap_duplicates_and_exact_exhaustion():
    _, trie = expanded_table()
    assert trie.mass == 1.0
    assert trie.reject_prefix((2, 0))
    assert trie.mass == pytest.approx(.9)
    assert not trie.reject_prefix((2, 0))
    assert trie.reject_prefix((2,))
    assert trie.mass == pytest.approx(.6)
    assert not trie.reject_prefix((2, 1))
    assert (2,) not in trie.nodes
    assert trie.reject_prefix((0, 0))
    assert trie.mass == pytest.approx(.55)
    assert trie.reject_prefix((1, 1))
    assert trie.mass == pytest.approx(.4)
    trie.reject_prefix((0,))
    trie.reject_prefix((1,))
    assert trie.log_mass == -math.inf and trie.exhausted
    with pytest.raises(EmptyTrie):
        trie.conditional_logs(())


@pytest.mark.parametrize("excluded", [[], [(2,)], [(2,), (0, 0)], [(2,), (0, 0), (1, 1)]])
def test_exact_reweighted_law_and_accepted_distribution(excluded):
    policy, trie = expanded_table()
    for prefix in excluded:
        assert not any(leaf[:len(prefix)] == prefix for leaf in policy.good)
        trie.reject_prefix(prefix)
    expected_mass = math.fsum(p for leaf, p in policy.weights.items()
                              if not any(leaf[:len(prefix)] == prefix for prefix in excluded))
    assert trie.mass == pytest.approx(expected_mass)
    proposal = enumerated_proposal(policy, trie)
    assert math.fsum(proposal.values()) == pytest.approx(1.0)
    for leaf, probability in proposal.items():
        expected = 0 if any(leaf[:len(prefix)] == prefix for prefix in excluded) else policy.weights[leaf] / expected_mass
        assert probability == pytest.approx(expected)
    good_mass = math.fsum(proposal[leaf] for leaf in policy.good)
    for leaf in policy.good:
        assert proposal[leaf] / good_mass == pytest.approx(policy.weights[leaf] / .4)
    # Exact finite expectation of p_t*tau, retaining every zero return.
    assert math.fsum(expected_mass * probability * (leaf in policy.good)
                     for leaf, probability in proposal.items()) == pytest.approx(.4)


def test_tiny_nonzero_mass_does_not_become_exhaustion():
    trie = RejectionTrie(TablePolicy().context(()))
    trie.expand((), torch.tensor([0., -1000.], dtype=torch.float64))
    trie.reject_tokens((), [0])
    assert trie.log_mass == -1000.0
    assert trie.mass == 0.0 and not trie.exhausted  # ordinary-space underflow only
    assert trie.conditional_logs(()).tolist() == [-math.inf, 0.0]


def test_capacity_freezes_new_learning_without_truncating_base_support():
    policy, trie = expanded_table(max_nodes=1)
    assert len(trie.nodes) == 1
    assert not trie.reject_prefix((0, 0))
    trie.reject_prefix((2,))
    proposal = enumerated_proposal(policy, trie)
    assert proposal[(0, 0)] == pytest.approx(.05 / .6)
    assert proposal[(0, 1)] == pytest.approx(.15 / .6)
    assert proposal[(1, 0)] == pytest.approx(.25 / .6)


@pytest.mark.parametrize("field", ["model", "tokenizer", "state", "group", "reward", "policy"])
def test_trie_resume_binds_every_context_component(field):
    _, trie = expanded_table()
    context = dict(trie.context)
    context[field] = {"changed": True}
    with pytest.raises(ValueError, match="context"):
        RejectionTrie.from_state_dict(trie.state_dict(), context=context)


def test_snapshot_mass_is_verified_instead_of_trusted():
    _, trie = expanded_table()
    payload = trie.state_dict()
    payload["nodes"][0]["log_mass"] = -3.
    with pytest.raises(ValueError, match="mass"):
        RejectionTrie.from_state_dict(payload, context=trie.context)


def test_snapshot_preserves_cached_floats_and_exact_survival_one():
    trie = RejectionTrie(TablePolicy().context(()))
    # Sum differs from one by roundoff but is a valid stored distribution.
    logs = torch.tensor([math.log(.2), math.log(.8)], dtype=torch.float64) + 2e-16
    trie.expand((), logs)
    payload = trie.state_dict()
    payload["nodes"][0]["log_probs"] = logs.clone()
    restored = RejectionTrie.from_state_dict(payload, context=trie.context)
    assert torch.equal(restored.cached_logs(()), logs)
    assert restored.log_mass == 0.0
    payload["nodes"][0]["log_probs"] += .01
    with pytest.raises(ValueError, match="normalized"):
        RejectionTrie.from_state_dict(payload, context=trie.context)


def test_batch_learning_matches_union_of_failed_cones():
    policy, trie = expanded_table()
    trie.learn([((), torch.tensor([2])), ((0,), torch.tensor([0]))], failed_prefix=(1, 1))
    assert trie.mass == pytest.approx(.4)
    proposal = enumerated_proposal(policy, trie)
    assert proposal[(0, 1)] == pytest.approx(.375)
    assert proposal[(1, 0)] == pytest.approx(.625)


def test_mutating_caller_context_or_snapshot_cannot_rekey_live_trie():
    context = TablePolicy().context(())
    trie = RejectionTrie(context)
    original = trie.context_hash
    context["state"].append(0)
    payload = trie.state_dict()
    payload["context"]["state"].append(1)
    assert trie.context["state"] == [] and trie.context_hash == original


def test_attempt_freezes_trie_and_learns_neighbors_after_success(monkeypatch):
    def feedback(state):
        forbidden = [2] if not state else ([0] if state == (0,) else [])
        return Feedback(forbidden=torch.tensor(forbidden, dtype=torch.long))

    sampler = CARSSampler(TablePolicy(feedback=feedback), ())
    observed = []
    force_tokens(monkeypatch, [0, 1], lambda: observed.append(sampler.trie.mass))
    result = sampler.attempt()
    assert observed == pytest.approx([.6, .6])
    assert result.reward_result.reward == 1.
    assert result.mass_before == pytest.approx(.6)
    assert result.weighted_reward == pytest.approx(.6)
    assert math.exp(result.log_proposal_probability) == pytest.approx(.15 / .6)
    assert math.exp(result.log_base_probability) == pytest.approx(.15)
    assert sampler.trie.mass == pytest.approx(.55)  # .05 learned after success


def test_chemical_failure_rejects_only_its_leaf(monkeypatch):
    sampler = CARSSampler(TablePolicy(), ())
    force_tokens(monkeypatch, [0, 0, 0, 1])
    failed = sampler.attempt()
    assert failed.kind == "complete" and failed.reward_result.reward == 0.
    assert failed.mass_before == pytest.approx(.6)
    assert failed.weighted_reward == 0.
    assert sampler.trie.mass == pytest.approx(.55)
    success = sampler.attempt()
    assert success.reward_result.reward == 1. and success.state == (0, 1)
    assert success.mass_before == pytest.approx(.55)


def test_prefix_abort_keeps_frozen_mass_and_zero_contribution(monkeypatch):
    sampler = CARSSampler(TablePolicy(), (), constrain_first=False)
    force_tokens(monkeypatch, [2])
    result = sampler.attempt()
    assert result.kind == "prefix_abort" and result.state == (2,)
    assert result.mass_before == 1. and result.weighted_reward == 0.
    assert sampler.trie.mass == pytest.approx(.6)


def test_all_failed_root_and_nonfinite_base_distributions():
    policy = TablePolicy(feedback=lambda state: Feedback(forbidden=torch.arange(3)))
    sampler = CARSSampler(policy, ())
    result = sampler.attempt()
    assert result.kind == "prefix_abort" and result.reason == "remaining_mass_zero"
    assert result.mass_before == 0. and result.weighted_reward == 0.
    assert sampler.trie.exhausted
    for logs in (torch.tensor([math.nan, 0.]), torch.tensor([math.inf]), torch.tensor([-math.inf])):
        trie = RejectionTrie(policy.context(()))
        with pytest.raises(ValueError):
            trie.expand((), logs)


def test_full_vocabulary_keeps_tokens_outside_an_eight_token_shortlist(monkeypatch):
    weights = {(token,): 1. / 12 for token in range(12)}
    sampler = CARSSampler(TablePolicy(weights=weights, good=set(weights)), ())
    force_tokens(monkeypatch, [11])
    result = sampler.attempt()
    assert result.state == (11,) and result.reward_result.reward == 1.
    assert math.exp(result.log_proposal_probability) == pytest.approx(1. / 12)


def test_adaptive_sampler_accepted_frequencies():
    sampler = CARSSampler(TablePolicy(), (), seed=1729)
    outcomes = Counter()
    for _ in range(2000):
        result = sampler.attempt()
        if result.reward_result is not None and result.reward_result.reward:
            outcomes[result.state] += 1
    assert set(outcomes) == GOOD
    assert outcomes[(0, 1)] / sum(outcomes.values()) == pytest.approx(.375, abs=.035)


def test_rng_and_trie_roundtrip_replay_identical_choices():
    sampler = CARSSampler(TablePolicy(), (), seed=7)
    for _ in range(8):
        sampler.attempt()
    buffer = io.BytesIO()
    torch.save(sampler.state_dict(), buffer)
    buffer.seek(0)
    restored = CARSSampler(TablePolicy(), (), seed=999)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    for prefix, node in sampler.trie.nodes.items():
        assert torch.equal(node.log_probs, restored.trie.cached_logs(prefix))
        assert node.log_mass == restored.trie.nodes[prefix].log_mass
    for _ in range(12):
        left, right = sampler.attempt(), restored.attempt()
        assert left.state == right.state and left.kind == right.kind
        assert left.log_mass_before == right.log_mass_before
        assert left.log_proposal_probability == right.log_proposal_probability


def test_budget_abort_is_not_a_zero_reward_mc_sample(monkeypatch):
    tracker = CostTracker(run_id="budget", max_model_nfe=1)
    sampler = CARSSampler(TablePolicy(tracker=tracker), (), tracker=tracker)
    force_tokens(monkeypatch, [0])
    result = sampler.attempt()
    assert result.kind == "budget_exhausted" and result.weighted_reward is None
    assert tracker.raw_nfe == 1 and tracker.counters["cars_attempts"] == 1
    assert tracker.counters["complete_candidates"] == 0
    assert tracker.counters["prefix_aborts"] == 0


def test_cache_reuses_probabilities_without_extra_model_forwards(monkeypatch):
    tracker = CostTracker(run_id="cache")
    policy = TablePolicy(tracker=tracker)
    sampler = CARSSampler(policy, (), tracker=tracker)
    force_tokens(monkeypatch, [0, 1, 0, 1])
    sampler.attempt()
    calls = tracker.raw_nfe
    sampler.attempt()
    assert calls == 2 and tracker.raw_nfe == 2 and policy.calls == 2


@pytest.fixture
def smiles_checker():
    return RewardChecker(mode="grammar_only")


def test_mdlm_policy_preserves_known_suffix_and_locked_coordinates(toy_model, special_tokens, smiles_checker, monkeypatch):
    tokenizer = SmilesTokenizer(("<pad>", "<mask>", "<eos>", "C", "O"), max_length=4)
    root = SmilesState((1, 4, 2, 0), (True, False, False, False), GROUPS[0], special_tokens,
                       locked=(False, True, True, True))
    policy = MDLMCompletionPolicy(LeftToRightCompletion(toy_model), tokenizer, smiles_checker, GROUPS[0])
    policy.begin()
    assert special_tokens.eos_id in policy.feedback(root).forbidden
    sampler = CARSSampler(policy, root)
    force_tokens(monkeypatch, [3])
    result = sampler.attempt()
    assert result.state.token_ids == (3, 4, 2, 0)
    assert result.state.locked == root.locked and result.smiles == "CO"
    assert result.reward_result.reward == 1.
    assert math.exp(result.log_base_probability) == pytest.approx(1. / 3, rel=1e-6)


def test_mdlm_eos_pad_conflict_is_preserved_and_exactly_rejected(toy_model, special_tokens, smiles_checker):
    tokenizer = SmilesTokenizer(("<pad>", "<mask>", "<eos>", "C", "O"), max_length=4)
    # EOS before a known non-PAD suffix is invalid, even if later coordinates mask.
    root = SmilesState((3, 2, 4, 1), (False, False, False, True), GROUPS[0], special_tokens)
    policy = MDLMCompletionPolicy(LeftToRightCompletion(toy_model), tokenizer, smiles_checker, GROUPS[0])
    result = CARSSampler(policy, root).attempt()
    assert result.kind == "complete" and result.reward_result.reward == 0.
    assert result.state.token_ids == root.token_ids and toy_model.forward_calls == 0


def test_unknown_parser_feedback_never_blocks_ordinary_tokens(toy_model, toy_initial_state, smiles_checker, monkeypatch):
    from src.grammar import ContinuationCheck, PrefixCheck

    class UnknownCursor:
        def advance(self, values):
            return PrefixCheck("unknown", error="parser limit")

        def continuations(self):
            return ContinuationCheck(PrefixCheck("unknown", error="parser limit"), None)

    tokenizer = SmilesTokenizer(("<pad>", "<mask>", "<eos>", "C", "O"), max_length=4)
    policy = MDLMCompletionPolicy(LeftToRightCompletion(toy_model), tokenizer, smiles_checker, GROUPS[0])
    monkeypatch.setattr(policy.grammar, "token_cursor", lambda tokenizer: UnknownCursor())
    policy.begin()
    feedback = policy.feedback(toy_initial_state)
    assert not feedback.invalid and not {3, 4} & set(feedback.forbidden.tolist())

    def unavailable(text):
        raise GrammarError("parser limit")

    monkeypatch.setattr(policy.grammar, "accepts", unavailable)
    with pytest.raises(GrammarError):
        CARSSampler(policy, toy_initial_state).attempt()


def test_native_grammar_feedback_rejects_invalid_prefix_tokens(toy_model, smiles_checker):
    tokenizer = SmilesTokenizer(("<pad>", "<mask>", "<eos>", "C", "O"), max_length=4)
    policy = MDLMCompletionPolicy(LeftToRightCompletion(toy_model), tokenizer, smiles_checker, GROUPS[0])
    policy.begin()
    root = SmilesState.fully_masked(4, GROUPS[0], tokenizer.special_tokens)
    feedback = policy.feedback(root)
    assert not feedback.invalid
    assert {0, 1, 2} <= set(feedback.forbidden.tolist())
    assert 3 not in feedback.forbidden and 4 not in feedback.forbidden


@pytest.fixture
def ar_model():
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.BPE(
        {"C": 0, "CC": 1, "(": 2, ")": 3, "<eos>": 4, "<stop>": 5, "<unk>": 6},
        [("C", "C")], unk_token="<unk>",
    ))
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", unk_token="<unk>",
                                       additional_special_tokens=["<stop>"])

    class Model:
        model_id = "tiny-native-ar"
        vocab_size = 7
        eos_token_ids = (4, 5)
        metadata = {"tokenizer_revision": "toy", "dtype": "float64"}

        def __init__(self):
            self.tokenizer = tokenizer

        def encode_prompt(self, text):
            return (0,)

        def next_token_logits(self, prefixes):
            assert prefixes and prefixes[0][0] == 0
            return torch.zeros((len(prefixes), self.vocab_size), dtype=torch.float64)

    return Model()


def test_ar_native_multichar_token_and_pinned_tokenizer_eos(ar_model, smiles_checker, monkeypatch):
    policy = ARCompletionPolicy(ar_model, "prompt", smiles_checker, GROUPS[0], max_new_tokens=3)
    force_tokens(monkeypatch, [1, 4])
    result = CARSSampler(policy, ()).attempt()
    assert result.smiles == "CC" and result.reward_result.reward == 1.
    assert result.state == (1, 4) and policy.metadata["eos_token_ids"] == [4]
    policy.begin()
    feedback = policy.feedback((1,))
    assert 5 in feedback.forbidden  # other model stop tokens keep native grammar semantics


def test_ar_cap_accepts_complete_text_and_rejects_incomplete_leaf(ar_model, smiles_checker, monkeypatch):
    good = ARCompletionPolicy(ar_model, "prompt", smiles_checker, GROUPS[0], max_new_tokens=1)
    force_tokens(monkeypatch, [1])
    result = CARSSampler(good, ()).attempt()
    assert result.reward_result.reward == 1. and result.state == (1,)
    bad = ARCompletionPolicy(ar_model, "prompt", smiles_checker, GROUPS[0], max_new_tokens=2)
    force_tokens(monkeypatch, [0, 2])
    sampler = CARSSampler(bad, ())
    result = sampler.attempt()
    assert result.kind == "complete" and result.smiles == "C(" and result.reward_result.reward == 0.
    assert torch.isfinite(sampler.trie.conditional_logs(())[0])


def test_selector_uses_molecule_identity_and_raw_grammar_only_fallback():
    examples = {GROUPS[0]: ["CO"], GROUPS[1]: [], GROUPS[2]: []}
    selector = UniqueSelector(examples)
    checker = RewardChecker(mode="grammar_only")
    prompted = checker.evaluate_smiles("OC", GROUPS[0])
    assert selector.consider("OC", prompted) is None and selector.last_reason == "prompt_example"
    good = checker.evaluate_smiles("CCO", GROUPS[0])
    assert selector.consider("CCO", good) == ("CCO", "canonical_smiles")
    duplicate = checker.evaluate_smiles("OCC", GROUPS[0])
    assert selector.consider("OCC", duplicate) is None and selector.last_reason == "duplicate"
    selector = UniqueSelector(examples, replication=True)
    invalid = checker.evaluate_smiles("C1CC", GROUPS[0])
    assert invalid.reward == 1. and not invalid.rdkit_valid
    assert selector.consider("C1CC", invalid) == ("C1CC", "raw_smiles")


def test_cars_configs_keep_primary_and_replication_separate():
    primary = load_config("configs/cars.yaml", overrides=["group=acrylates", "cars.model.checkpoint_manifest=checkpoints/model/manifest.json"], validator=validate_cars)
    replication = load_config("configs/replication.yaml", overrides=["group=isocyanates"], validator=validate_cars)
    assert primary["reward"]["mode"] == "grammar_rdkit_smarts"
    assert replication["reward"]["mode"] == "grammar_only" and replication["cars"]["backend"] == "ar"
    assert replication["protocol"]["max_root_attempts"] == 1000
    assert replication["cars"]["max_new_tokens"] == 256
    for override in ("cars.top_k=8", "cars.learn_level=2", "cars.learning=online", "cars.vocabulary=shortlist"):
        with pytest.raises(ValueError):
            load_config("configs/cars.yaml", overrides=["group=acrylates", "cars.model.checkpoint_manifest=checkpoints/model/manifest.json", override], validator=validate_cars)


def _mock_task_policy(monkeypatch):
    import src.tasks.generate_cars as task

    monkeypatch.setattr(task, "generation_dependencies", lambda config, store: {})
    monkeypatch.setattr(task, "read_benchmark_assets", lambda config: ({}, {group: [] for group in GROUPS}))
    monkeypatch.setattr(task, "checker_versions", lambda: {"rdkit": "2025.3.6", "llguidance": "0.7.30"})
    monkeypatch.setattr(task, "_policy", lambda context, checker: (TablePolicy(tracker=context.tracker), (), {"device": "cpu"}))
    return task


def test_t09_journal_resume_and_costs_match_uninterrupted_run(monkeypatch, artifact_store):
    task = _mock_task_policy(monkeypatch)
    base = ["--group", GROUPS[0], "--smoke", "--checkpoint", "checkpoints/mock/manifest.json",
            "--artifact-root", artifact_store.root, "--set", "protocol.max_root_attempts=100"]
    commits = []

    def interrupt_after_attempt():
        commits.append(True)
        if len(commits) == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        task.execute([*base, "--run-id", "resumed"], on_checkpoint=interrupt_after_attempt)
    resumed = task.execute([*base, "--run-id", "resumed", "--resume"])
    reference = task.execute([*base, "--run-id", "reference"])
    assert resumed.status == reference.status == "completed"
    left_dir, right_dir = (artifact_store.run_directory(record.identity) for record in (resumed, reference))
    left = list(artifact_store.read_jsonl(left_dir + "/attempts.jsonl"))
    right = list(artifact_store.read_jsonl(right_dir + "/attempts.jsonl"))
    assert [(row["kind"], row["smiles"]) for row in left] == [(row["kind"], row["smiles"]) for row in right]
    assert [row["attempt_id"] for row in left] == list(range(1, len(left) + 1))
    assert resumed.tracking.counters == reference.tracking.counters
    artifact_store.require_manifest(left_dir + "/generation_manifest.json", kind="generation")
    assert len(list(artifact_store.read_jsonl(left_dir + "/samples.jsonl"))) == 2


def test_t09_budget_abort_is_journaled_and_reported(monkeypatch, artifact_store):
    task = _mock_task_policy(monkeypatch)
    record = task.execute(["--group", GROUPS[0], "--smoke", "--checkpoint", "checkpoints/mock/manifest.json",
                           "--artifact-root", artifact_store.root, "--run-id", "budget-task",
                           "--set", "protocol.max_model_nfe=1"])
    directory = artifact_store.run_directory(record.identity)
    rows = list(artifact_store.read_jsonl(directory + "/attempts.jsonl"))
    assert record.status == "budget_exhausted" and len(rows) == 1
    assert rows[0]["kind"] == "budget_exhausted" and record.tracking.counters["mdlm_forward_calls"] == 1
    assert rows[0]["counters"]["root_attempts"] == 1
    report = artifact_store.read_json(directory + "/generation.json")
    assert not report["target_attained"] and report["selected_samples"] == 0
    events = list(artifact_store.read_jsonl(directory + "/events.jsonl"))
    attempt = next(row for row in events if row["event"] == "cars_attempt")
    assert attempt["weighted_reward"] is None
