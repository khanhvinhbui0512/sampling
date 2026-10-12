"""Soundness against a finite grammar and the exact QM9 terminal decoder."""

from dataclasses import asdict, replace
from itertools import product

import pytest
from lark import Lark, UnexpectedInput

from src.budget import BudgetTracker
from src.cars.oracle import FiniteLanguageOracle, QM9Oracle
from src.qm9.chemistry import assess_smiles, rdkit_version
from src.qm9.reward import BinaryCalibration, QM9Reward
from src.types import MaskedState, TokenRules


def grammar_language(horizon):
    with open("tests/fixtures/toy_grammar.lark", encoding="utf-8") as handle:
        parser = Lark(handle.read(), parser="lalr")
    language = []
    for length in range(horizon):
        for letters in product("ab", repeat=length):
            try:
                parser.parse("".join(letters))
            except UnexpectedInput:
                continue
            tokens = tuple(1 if char == "a" else 2 for char in letters) + (3,)
            language.append(tokens + (4,) * (horizon - len(tokens)))
    return language


@pytest.mark.parametrize("horizon", [3, 4])
def test_finite_grammar_never_prunes_any_compatible_success(horizon):
    rules = TokenRules(vocab_size=5, sequence_length=horizon, mask_id=0, eos_id=3, pad_id=4)
    language = grammar_language(horizon)
    oracle = FiniteLanguageOracle(rules, language)
    budget = BudgetTracker(base_sequence_limit=0)
    for tokens in product(range(rules.vocab_size), repeat=horizon):
        state = MaskedState(tokens, rules)
        compatible = [leaf for leaf in language
                      if all(token == 0 or token == other for token, other in zip(tokens, leaf))]
        assert oracle.check(state, budget=budget).certified_failure == (len(compatible) == 0)
    assert budget.raw_nfe == 0
    assert budget.snapshot().total.oracle_calls == rules.vocab_size ** horizon


def test_length_cap_and_future_observations_are_part_of_the_language():
    for horizon, impossible in ((3, True), (4, False)):
        rules = TokenRules(5, horizon, mask_id=0, eos_id=3, pad_id=4)
        oracle = FiniteLanguageOracle(rules, grammar_language(horizon))
        state = MaskedState((1, 2) + (0,) * (horizon - 2), rules)
        assert oracle.check(state, budget=BudgetTracker(base_sequence_limit=0)).certified_failure == impossible
    rules = TokenRules(5, 4, mask_id=0, eos_id=3, pad_id=4)
    oracle = FiniteLanguageOracle(rules, grammar_language(4))
    budget = BudgetTracker(base_sequence_limit=0)
    assert not oracle.check(MaskedState((0, 2, 2, 3), rules), budget=budget).certified_failure
    # Ignoring the known final PAD would incorrectly keep this branch viable.
    assert oracle.check(MaskedState((0, 2, 2, 4), rules), budget=budget).certified_failure
    # The known EOS after a hole still permits ba; retain its actual position.
    assert not oracle.check(MaskedState((2, 0, 3, 4), rules), budget=budget).certified_failure


@pytest.fixture
def qm9_reward(tokenizer, token_rules, chemistry_policy):
    calibration = BinaryCalibration(
        calibration_id="oracle-test-pass95", reference_id="toy-base", policy_id="toy-policy",
        checkpoint_sha256="b" * 64, tokenizer_sha256="a" * 64, threshold=0., quantile=.95,
        sample_count=10, seed=1, quantile_method="linear", chemistry_policy=asdict(chemistry_policy),
        rdkit_version=rdkit_version(), threshold_source="independent_base_calibration",
        score_name="qed_or_zero", terminal_policy="strict_special_tokens_v1")
    return QM9Reward(tokenizer=tokenizer, token_rules=token_rules, chemistry_policy=chemistry_policy,
                     tokenizer_sha256="a" * 64, calibration=calibration,
                     calibration_reference_id="toy-base", calibration_policy_id="toy-policy",
                     checkpoint_sha256="b" * 64)


def test_leaf_only_checks_exact_reward_and_leaves_partial_unknown(qm9_reward, token_rules):
    oracle = QM9Oracle(qm9_reward)
    budget = BudgetTracker(base_sequence_limit=0)
    successful = MaskedState((1, 5, 2, 0, 0, 0), token_rules)
    assert qm9_reward.evaluate(successful, require_binary=True).reward == 1
    assert not oracle.check(successful, budget=budget).certified_failure
    failed = MaskedState((1, 4, 2, 0, 0, 0), token_rules)
    assert oracle.check(failed, budget=budget).certified_failure
    # Even obvious partial UNK failure stays UNKNOWN in leaf-only ARS mode.
    partial = MaskedState((1, 4, 3, 3, 3, 3), token_rules)
    assert not oracle.check(partial, budget=budget).certified_failure
    assert budget.raw_nfe == 0 and budget.snapshot().total.oracle_calls == 3


@pytest.mark.parametrize("tokens", [
    (1, 4, 3, 3, 3, 3),  # Known UNK cannot be replaced by filling masks.
    (3, 5, 1, 3, 3, 3),  # Known misplaced BOS.
    (1, 2, 3, 5, 3, 3),  # EOS followed by a known non-PAD token.
    (1, 0, 3, 3, 2, 3),  # PAD followed by a known EOS.
])
def test_token_oracle_certifies_only_irreparable_known_token_errors(qm9_reward, token_rules, tokens):
    oracle = QM9Oracle(qm9_reward, mode="token_constraints")
    state = MaskedState(tokens, token_rules)
    budget = BudgetTracker(base_sequence_limit=0)
    assert oracle.check(state, budget=budget).certified_failure
    # Exhaust the small remaining token support; each terminal must really fail.
    for fill in product((0, 1, 2, 5), repeat=len(state.masked_positions)):
        leaf = state.reveal(dict(zip(state.masked_positions, fill)))
        assert qm9_reward.evaluate(leaf, require_binary=True).reward == 0


def test_no_successful_qm9_leaf_or_partial_assignment_is_pruned(qm9_reward, token_rules):
    oracle = QM9Oracle(qm9_reward, mode="token_constraints")
    budget = BudgetTracker(base_sequence_limit=0)
    for content in (5, 6, 7):  # C, N, O, each a valid one-atom molecule.
        tokens = (1, content, 2, 0, 0, 0)
        assert qm9_reward.evaluate(MaskedState(tokens, token_rules), require_binary=True).reward == 1
        for keep in product((False, True), repeat=token_rules.sequence_length):
            partial = MaskedState(tuple(token if observed else 3 for token, observed in zip(tokens, keep)),
                                  token_rules)
            assert not oracle.check(partial, budget=budget).certified_failure


def test_partial_rdkit_failure_does_not_certify_a_failure_cone(qm9_reward, token_rules, monkeypatch):
    assert not assess_smiles("C(", policy=qm9_reward.chemistry_policy).valid
    partial = MaskedState((1, 5, 8, 3, 3, 3), token_rules)

    def forbidden(*args, **kwargs):
        raise AssertionError("A partial state must not be passed to terminal chemistry scoring")

    monkeypatch.setattr(qm9_reward, "evaluate", forbidden)
    for mode in ("leaf_only", "token_constraints"):
        oracle = QM9Oracle(qm9_reward, mode=mode)
        assert not oracle.check(partial, budget=BudgetTracker(base_sequence_limit=0)).certified_failure


def test_oracle_requires_calibration_and_binds_mode_and_threshold(qm9_reward):
    original = QM9Oracle(qm9_reward)
    assert original.oracle_id != QM9Oracle(qm9_reward, mode="token_constraints").oracle_id
    qm9_reward.calibration = replace(qm9_reward.calibration, threshold=.5)
    assert original.oracle_id != QM9Oracle(qm9_reward).oracle_id
    with pytest.raises(ValueError, match="calibration changed"):
        original.check(MaskedState((1, 5, 2, 0, 0, 0), qm9_reward.token_rules),
                       budget=BudgetTracker(base_sequence_limit=0))
    qm9_reward.calibration = None
    with pytest.raises(ValueError, match="calibrated"):
        QM9Oracle(qm9_reward)
