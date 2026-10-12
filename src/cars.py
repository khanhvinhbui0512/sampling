"""CARS on fixed-order MDLM q_z and the separate pinned AR replication policy.

Written from CARS Algorithm 1, Eq. 1–3 and the pinned reference's learn_level=3 /
constrain_first behavior. Newly discovered non-root masks are learned AFTER an
attempt, including successful attempts, rather than applied as greedy decoding.
The frozen proposal is q(avoid W), and its pre-attempt mass is returned even for
failures. The MDLM adaptation makes no claim about an ideal diffusion conditional.
"""

from __future__ import annotations

import importlib.metadata
import math
import os
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from src.artifacts import sha256_file
from src.completion import LeftToRightCompletion, normalize_eos_suffix
from src.config import config_digest
from src.grammar import GrammarCursor, SmilesGrammar
from src.reward import RewardChecker
from src.schemas import RewardResult, SmilesState, require_count
from src.tokenizer import SmilesTokenizer, TokenizationError
from src.tracking import BudgetExceeded, CostTracker
from src.trie import EmptyTrie, RejectionTrie, normalized_logs

CARS_VERSION = "cars-frozen-attempt-v1"


@dataclass(frozen=True, slots=True)
class Feedback:
    invalid: bool = False
    forbidden: torch.Tensor | None = None
    reason: str | None = None


class CompletionPolicy(Protocol):
    """Finite, deterministic-order completion tree; logits retain full support."""

    metadata: dict[str, Any]

    def context(self, root: Any) -> dict[str, Any]: ...
    def normalize(self, state: Any) -> Any: ...
    def begin(self) -> None: ...
    def terminal(self, state: Any) -> bool: ...
    def feedback(self, state: Any) -> Feedback: ...
    def next_logs(self, state: Any) -> torch.Tensor: ...
    def advance(self, state: Any, token: int) -> Any: ...
    def output(self, state: Any) -> tuple[str, RewardResult]: ...


def _reward_identity(checker: RewardChecker, grammar: SmilesGrammar) -> dict[str, Any]:
    return {"mode": checker.mode, "version": checker.version,
            "grammar": config_digest({"source": grammar.source}),
            "membership": sha256_file(os.path.join(checker.asset_directory, "smarts.json")),
            "checker_versions": {name: importlib.metadata.version(name) for name in ("rdkit", "llguidance")}}


def _forbidden(cursor: GrammarCursor, vocab_size: int, eos_ids: Sequence[int], *,
               eos_possible: bool = True) -> torch.Tensor | None:
    decision = cursor.continuations()
    if decision.prefix.status == "unknown":
        return None
    allowed = torch.zeros(vocab_size, dtype=torch.bool)
    if decision.allowed_token_ids is not None:
        allowed[list(decision.allowed_token_ids)] = True
    # Terminal tokens are handled as grammar completion, never fed as EOS bytes.
    allowed[list(eos_ids)] = decision.prefix.accepting and eos_possible
    return (~allowed).nonzero().flatten()


class MDLMCompletionPolicy:
    """q_z uses every known coordinate as model context, including known suffixes."""

    def __init__(self, completion: LeftToRightCompletion, tokenizer: SmilesTokenizer,
                 checker: RewardChecker, group: str, *, tracker: CostTracker | None = None) -> None:
        if completion.model.special_tokens != tokenizer.special_tokens or completion.model.vocab_size != tokenizer.vocab_size:
            raise ValueError("CARS model/tokenizer mismatch")
        self.completion, self.tokenizer, self.checker, self.group = completion, tokenizer, checker, group
        self.grammar = checker.grammars[group]
        self.tracker = tracker
        self.metadata = dict(completion.metadata)
        self.metadata["tokenizer"] = tokenizer.fingerprint
        self._reward_context = _reward_identity(checker, self.grammar)
        self._cursor: GrammarCursor | None = None
        self._fed = 0

    def context(self, root: SmilesState) -> dict[str, Any]:
        if (root.group != self.group or root.special_tokens != self.tokenizer.special_tokens
                or len(root.token_ids) > self.tokenizer.max_length
                or any(token >= self.tokenizer.vocab_size for token in root.token_ids)):
            raise ValueError("CARS root differs from its group, vocabulary, or maximum width")
        return {"model": self.completion.model.model_id, "tokenizer": self.tokenizer.fingerprint,
                "state": root.to_dict(), "group": self.group, "reward": self._reward_context,
                "policy": self.metadata}

    def normalize(self, state: SmilesState) -> SmilesState:
        return normalize_eos_suffix(state)

    def begin(self) -> None:
        self._cursor = self.grammar.token_cursor(self.tokenizer)
        self._fed = 0

    def terminal(self, state: SmilesState) -> bool:
        return state.is_terminal

    def feedback(self, state: SmilesState) -> Feedback:
        if self.tracker is not None:
            self.tracker.increment("grammar_checks")
        position = state.masked_positions[0]
        # Feed contiguous known tokens, without assuming arbitrary known suffixes
        # are adjacent across unresolved positions. EOS/PAD are handled explicitly.
        known = state.token_ids[self._fed:position]
        if self.tokenizer.special_tokens.pad_id in known:
            return Feedback(True, reason="PAD before molecular EOS")
        result = self._cursor.advance(known)
        self._fed = position
        if result.prunable:
            return Feedback(True, reason=result.error or "grammar-impossible prefix")
        special = self.tokenizer.special_tokens
        eos_possible = all(token in (special.mask_id, special.pad_id) for token in state.token_ids[position + 1:])
        forbidden = _forbidden(self._cursor, self.tokenizer.vocab_size, [special.eos_id], eos_possible=eos_possible)
        # Even when parser feedback is UNKNOWN, PAD at this unresolved molecular
        # coordinate is provably invalid under the explicit EOS/PAD state contract.
        structural = torch.tensor([special.mask_id, special.pad_id] + ([] if eos_possible else [special.eos_id]), dtype=torch.long)
        forbidden = structural if forbidden is None else torch.unique(torch.cat((forbidden, structural)))
        return Feedback(forbidden=forbidden)

    def next_logs(self, state: SmilesState) -> torch.Tensor:
        _, _, logs = self.completion.next_branch(state)
        return logs

    def advance(self, state: SmilesState, token: int) -> SmilesState:
        return self.normalize(state.with_tokens({state.masked_positions[0]: token}))

    def output(self, state: SmilesState) -> tuple[str, RewardResult]:
        result = self.checker.evaluate_state(state, self.tokenizer)
        try:
            text = self.tokenizer.decode_state(state)
        except (TokenizationError, ValueError):
            # Retain an exact diagnostic string; never repair an invalid leaf.
            text = "".join(self.tokenizer.vocabulary[token] for token in state.token_ids)
        return text, result


class ARCompletionPolicy:
    """Finite AR tree: EOS stops; a grammar-accepting text at the cap may succeed.

    The cap rule follows the pinned source's generation_ended check. No EOS is
    forced and no extra token beyond max_new_tokens is generated. The prompt is
    model context only; the native-token grammar sees generated tokens only.
    """

    def __init__(self, model: Any, prompt: str, checker: RewardChecker, group: str, *,
                 max_new_tokens: int = 256, tracker: CostTracker | None = None) -> None:
        require_count(max_new_tokens, "AR max_new_tokens", positive=True)
        if checker.mode != "grammar_only":
            raise ValueError("AR replication uses grammar-only reward")
        self.model, self.checker, self.group, self.tracker = model, checker, group, tracker
        self.prompt_ids = tuple(model.encode_prompt(prompt))
        self.max_new_tokens = max_new_tokens
        self.grammar = checker.grammars[group]
        # The pinned CARS source explicitly passes tokenizer.eos_token_id to
        # generate, rather than using the model's potentially broader stop list.
        eos_id = model.tokenizer.eos_token_id
        if type(eos_id) is not int or not 0 <= eos_id < model.vocab_size:
            raise ValueError("AR replication requires the pinned tokenizer's EOS ID")
        self.eos_ids = (eos_id,)
        self.metadata = {"policy": "bounded_ar-v1", **model.metadata,
                         "max_new_tokens": max_new_tokens, "cap_acceptance": "complete_grammar_text",
                         "prompt_ids": list(self.prompt_ids), "vocabulary": "all_model_tokens",
                         "model_eos_token_ids": list(model.eos_token_ids), "eos_token_ids": list(self.eos_ids),
                         "stop_rule": "pinned_tokenizer_EOS"}
        self._reward_context = _reward_identity(checker, self.grammar)
        self._cursor: GrammarCursor | None = None
        self._fed = 0

    def context(self, root: tuple[int, ...]) -> dict[str, Any]:
        if root:
            raise ValueError("AR replication starts at an empty generated prefix")
        return {"model": self.model.model_id,
                "tokenizer": {"revision": self.model.metadata["tokenizer_revision"], "vocab_size": self.model.vocab_size},
                "state": list(root), "group": self.group, "reward": self._reward_context,
                "policy": self.metadata}

    def normalize(self, state: Sequence[int]) -> tuple[int, ...]:
        return tuple(state)

    def begin(self) -> None:
        self._cursor = self.grammar.token_cursor(self.model.tokenizer, n_vocab=self.model.vocab_size,
                                                eos_token=self.eos_ids[0])
        self._fed = 0

    def terminal(self, state: tuple[int, ...]) -> bool:
        return len(state) >= self.max_new_tokens or bool(state and state[-1] in self.eos_ids)

    def feedback(self, state: tuple[int, ...]) -> Feedback:
        if self.tracker is not None:
            self.tracker.increment("grammar_checks")
        result = self._cursor.advance(state[self._fed:])
        self._fed = len(state)
        if result.prunable:
            return Feedback(True, reason=result.error or "grammar-impossible prefix")
        return Feedback(forbidden=_forbidden(self._cursor, self.model.vocab_size, self.eos_ids))

    def next_logs(self, state: tuple[int, ...]) -> torch.Tensor:
        return self.model.next_token_logits([(*self.prompt_ids, *state)])[0].double().log_softmax(-1)

    def advance(self, state: tuple[int, ...], token: int) -> tuple[int, ...]:
        return (*state, token)

    def output(self, state: tuple[int, ...]) -> tuple[str, RewardResult]:
        tokens = state[:-1] if state and state[-1] in self.eos_ids else state
        text = self.model.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return text, self.checker.evaluate_smiles(text, self.group)


@dataclass(frozen=True, slots=True)
class CARSAttempt:
    kind: str
    state: Any
    token_ids: tuple[int, ...]
    log_mass_before: float | None
    log_mass_after: float
    log_base_probability: float
    log_proposal_probability: float
    smiles: str | None = None
    reward_result: RewardResult | None = None
    reason: str | None = None

    @property
    def mass_before(self) -> float | None:
        return None if self.log_mass_before is None else math.exp(self.log_mass_before)

    @property
    def weighted_reward(self) -> float | None:
        # An unfinished compute-budget attempt has no Monte Carlo contribution.
        if self.kind == "budget_exhausted" or self.log_mass_before is None:
            return None
        reward = 0.0 if self.reward_result is None else self.reward_result.reward
        return self.mass_before * reward

    def to_dict(self) -> dict[str, Any]:
        # JSON forbids infinities. null log mass/probability denotes exactly zero;
        # mass_before distinguishes zero mass from an uninitialized budget abort.
        def log_value(value):
            return value if value is not None and math.isfinite(value) else None

        return {"kind": self.kind, "token_ids": list(self.token_ids), "mass_before": self.mass_before,
                "log_mass_before": log_value(self.log_mass_before), "log_mass_after": log_value(self.log_mass_after),
                "log_base_probability": log_value(self.log_base_probability),
                "log_proposal_probability": log_value(self.log_proposal_probability),
                "weighted_reward": self.weighted_reward, "reason": self.reason}


class CARSSampler:
    def __init__(self, policy: CompletionPolicy, root: Any, *, seed: int = 0, max_nodes: int = 4096,
                 constrain_first: bool = True, tracker: CostTracker | None = None, inner: bool = False) -> None:
        require_count(seed, "CARS seed")
        if type(constrain_first) is not bool or type(inner) is not bool:
            raise TypeError("CARS flags must be boolean")
        self.policy, self.root = policy, policy.normalize(root)
        self.trie = RejectionTrie(policy.context(self.root), max_nodes=max_nodes)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.constrain_first, self.initialized = constrain_first, False
        self.tracker, self.inner = tracker, inner

    def _raw(self, prefix: tuple[int, ...], state: Any) -> torch.Tensor:
        with self._measure("cars_policy"):
            logs = self.trie.cached_logs(prefix)
            if logs is not None:
                if self.tracker is not None:
                    self.tracker.increment("cache_hits")
                return logs
            if self.tracker is not None:
                self.tracker.increment("cache_misses")
            logs = normalized_logs(self.policy.next_logs(state))
            self.trie.expand(prefix, logs)
            return logs

    def _measure(self, name: str):
        return nullcontext() if self.tracker is None else self.tracker.measure(name)

    @torch.inference_mode()
    def attempt(self) -> CARSAttempt:
        state, prefix = self.root, ()
        before = None
        raw_log = proposal_log = 0.0
        pending: list[tuple[tuple[int, ...], torch.Tensor]] = []
        failed: tuple[int, ...] | None = None
        kind, reason, text, reward = "prefix_abort", None, None, None
        if self.tracker is not None:
            self.tracker.record_cars_attempt(inner=self.inner)
        with self._measure("grammar"):
            self.policy.begin()
        try:
            if self.tracker is not None:
                self.tracker.require_budget()
            if not self.initialized:
                if self.constrain_first and not self.policy.terminal(state):
                    with self._measure("grammar"):
                        first = self.policy.feedback(state)
                    if first.invalid:
                        self.trie.reject_prefix(())
                    else:
                        self._raw((), state)
                        if first.forbidden is not None:
                            with self._measure("trie"):
                                self.trie.reject_tokens((), first.forbidden)
                self.initialized = True
            before = self.trie.log_mass
            if self.trie.exhausted:
                reason = "remaining_mass_zero"
            else:
                while True:
                    if self.tracker is not None:
                        self.tracker.require_budget()
                    if self.policy.terminal(state):
                        if self.tracker is not None:
                            self.tracker.record_leaf_visit(inner=self.inner)
                            self.tracker.record_complete_candidate(inner=self.inner)
                        with self._measure("reward"):
                            text, reward = self.policy.output(state)
                        kind = "complete"
                        if reward.reward == 0.0:
                            failed = prefix
                            reason = reward.error or "terminal_reward_zero"
                        break
                    with self._measure("grammar"):
                        feedback = self.policy.feedback(state)
                    if feedback.invalid:
                        failed, reason = prefix, feedback.reason
                        break
                    logs = self._raw(prefix, state)
                    if feedback.forbidden is not None:
                        pending.append((prefix, feedback.forbidden))
                    # Trie expansion above caches only base probabilities. All
                    # reject_* calls, except initial root preparation, are below.
                    with self._measure("trie"):
                        conditional = self.trie.conditional_logs(prefix, logs)
                        token = int(torch.multinomial(conditional.exp(), 1, generator=self.generator))
                    raw_log += float(logs[token])
                    proposal_log += float(conditional[token])
                    prefix = (*prefix, token)
                    state = self.policy.advance(state, token)
        except BudgetExceeded as exc:
            kind, reason = "budget_exhausted", str(exc)
        # Unknown parser decisions never certify a failed cone. Infrastructure
        # errors propagate; they are not terminal chemical failures.
        with self._measure("trie"):
            self.trie.learn(pending, failed_prefix=failed)
        if kind == "prefix_abort" and self.tracker is not None:
            self.tracker.record_prefix_abort(inner=self.inner)
        return CARSAttempt(kind, state, prefix, before, self.trie.log_mass, raw_log, proposal_log,
                           text, reward, reason)

    def state_dict(self) -> dict[str, Any]:
        return {"version": CARS_VERSION, "constrain_first": self.constrain_first,
                "initialized": self.initialized, "trie": self.trie.state_dict(),
                "rng": self.generator.get_state()}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if (payload["version"] != CARS_VERSION or payload["constrain_first"] != self.constrain_first
                or type(payload["initialized"]) is not bool):
            raise ValueError("Incompatible CARS sampler snapshot")
        trie = RejectionTrie.from_state_dict(payload["trie"], context=self.trie.context)
        if trie.max_nodes != self.trie.max_nodes:
            raise ValueError("CARS trie capacity changed on resume")
        self.generator.set_state(payload["rng"].cpu())
        self.trie, self.initialized = trie, payload["initialized"]


__all__ = ["ARCompletionPolicy", "CARSAttempt", "CARSSampler", "CompletionPolicy", "EmptyTrie",
           "Feedback", "MDLMCompletionPolicy"]
