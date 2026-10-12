"""Original CARS grammars with conservative prefix and token feedback.

llguidance 0.7.30 exposes parser errors and limits as matcher states. Resource
limits, unexpected errors, and API failures yield UNKNOWN, never a failed cone.
Complete-string evaluation raises on UNKNOWN rather than changing terminal tau.
Both SMILES tokens and fast Hugging Face AR tokens use the same cursor interface;
AR continuations use native token bytes, never decoded-token concatenation.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from src.config import project_path
from src.schemas import GROUPS, require_count

PrefixStatus = Literal["viable", "invalid", "unknown"]
_DEAD_ENDS = {"NoExtension", "NoExtensionBias"}
_ORDINARY_STOPS = {"NotStopped", "EndOfSentence", *_DEAD_ENDS}


class GrammarError(RuntimeError):
    """Parser failure prevents a definite grammar decision."""


@dataclass(frozen=True, slots=True)
class PrefixCheck:
    status: PrefixStatus
    accepting: bool = False
    error: str | None = None

    @property
    def prunable(self) -> bool:
        return self.status == "invalid"


@dataclass(frozen=True, slots=True)
class ContinuationCheck:
    prefix: PrefixCheck
    allowed_token_ids: tuple[int, ...] | None

    @property
    def prunable(self) -> bool:
        return self.prefix.prunable


class _ByteTokenizer:
    """Full byte vocabulary makes text checking independent of trained vocabularies."""

    tokens = tuple(bytes([value]) for value in range(256)) + (b"<eos>",)
    eos_token_id = 256
    bos_token_id = None
    special_token_ids = (256,)

    def __call__(self, text: str | bytes) -> list[int]:
        return list(text.encode("utf-8") if isinstance(text, str) else text)


class GrammarCursor:
    """Mutable cursor for one generated molecular prefix, excluding the AR prompt."""

    def __init__(self, matcher: Any, tokenizer: Any, *, blocked_token_ids: Sequence[int] = ()) -> None:
        self._matcher = matcher
        self._tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.eos_token_id = tokenizer.eos_token
        self._blocked = frozenset(blocked_token_ids)
        if any(type(token) is not int or token < 0 or token >= self.vocab_size for token in self._blocked):
            raise ValueError("Blocked token outside grammar tokenizer vocabulary")
        self._decision = self._inspect()

    def _inspect(self) -> PrefixCheck:
        try:
            reason = self._matcher.stop_reason()
            error = self._matcher.get_error() if self._matcher.is_error() else None
            if reason not in _ORDINARY_STOPS:
                return PrefixCheck("unknown", error=error or f"parser stopped: {reason}")
            if self._matcher.is_error():
                return PrefixCheck("invalid" if reason in _DEAD_ENDS else "unknown", error=error)
            accepting = bool(self._matcher.is_accepting())
            if reason in _DEAD_ENDS and not accepting:
                return PrefixCheck("invalid", error=f"parser proved no extension: {reason}")
            return PrefixCheck("viable", accepting=accepting)
        except Exception as exc:
            return PrefixCheck("unknown", error=f"{type(exc).__name__}: {exc}")

    @property
    def result(self) -> PrefixCheck:
        return self._decision

    def copy(self) -> GrammarCursor:
        clone = GrammarCursor(self._matcher.deep_copy(), self._tokenizer, blocked_token_ids=tuple(self._blocked))
        clone._decision = self._decision
        return clone

    def advance(self, token_ids: Sequence[int]) -> PrefixCheck:
        values = list(token_ids)
        for token_id in values:
            require_count(token_id, "grammar token ID")
            if token_id >= self.vocab_size:
                raise ValueError("Token ID outside grammar tokenizer vocabulary")
        if self._decision.status != "viable" or not values:
            return self._decision
        if any(token_id in self._blocked for token_id in values):
            self._decision = PrefixCheck("invalid", error="MASK/PAD cannot occur in a molecular prefix")
            return self._decision
        try:
            consumed = self._matcher.try_consume_tokens(values)
            result = self._inspect()
            if result.status == "unknown":
                self._decision = result
            elif type(consumed) is not int or not 0 <= consumed <= len(values):
                self._decision = PrefixCheck("unknown", error="Invalid parser consumption count")
            elif consumed != len(values):
                self._decision = PrefixCheck("invalid", error=f"Grammar rejects token at prefix offset {consumed}")
            else:
                self._decision = result
        except Exception as exc:
            self._decision = PrefixCheck("unknown", error=f"{type(exc).__name__}: {exc}")
        return self._decision

    def continuations(self) -> ContinuationCheck:
        """UNKNOWN has no mask; callers must keep candidates or propagate the error."""
        if self._decision.status == "unknown":
            return ContinuationCheck(self._decision, None)
        if self._decision.status == "invalid":
            return ContinuationCheck(self._decision, ())
        try:
            bias = self._matcher.compute_logit_bias()
            result = self._inspect()
            if result.status != "viable":
                self._decision = result
                return ContinuationCheck(result, () if result.prunable else None)
            if len(bias) != self.vocab_size or any(value not in (0, 200) for value in bias):
                self._decision = PrefixCheck("unknown", error="Invalid llguidance token mask")
                return ContinuationCheck(self._decision, None)
            allowed = tuple(index for index, value in enumerate(bias) if value == 200 and index not in self._blocked)
            self._decision = result
            # An empty native-token mask is not proof that the text language is
            # empty: a finite vocabulary may be unable to express an extension.
            return ContinuationCheck(result, allowed)
        except Exception as exc:
            self._decision = PrefixCheck("unknown", error=f"{type(exc).__name__}: {exc}")
            return ContinuationCheck(self._decision, None)


class SmilesGrammar:
    """Compile once; independent cursors are deep copies of a clean matcher."""

    def __init__(self, source: str, *, limits: Any = None) -> None:
        import llguidance

        if not isinstance(source, str) or not source:
            raise ValueError("Grammar source must be nonempty")
        self.source = source
        self.limits = limits
        self.compiled = llguidance.grammar_from("lark", source)
        self._byte_tokenizer = llguidance.LLTokenizer(llguidance.TokenizerWrapper(_ByteTokenizer()), slices=[])
        self._templates: dict[int, tuple[Any, Any]] = {}
        self._adapters: dict[int, tuple[Any, Any, tuple[int, ...]]] = {}
        is_error, messages = llguidance.LLMatcher.validate_grammar_with_warnings(
            self.compiled, self._byte_tokenizer, limits=limits,
        )
        if is_error:
            raise GrammarError("Cannot compile SMILES grammar: " + "; ".join(messages))
        self.warnings = tuple(messages)
        self._template(self._byte_tokenizer)

    @classmethod
    def from_file(cls, path: str, *, limits: Any = None) -> SmilesGrammar:
        with open(project_path(path), encoding="utf-8") as handle:
            return cls(handle.read(), limits=limits)

    @classmethod
    def for_group(cls, group: str, *, asset_directory: str = "assets/benchmark", limits: Any = None) -> SmilesGrammar:
        if group not in GROUPS:
            raise ValueError(f"Unknown molecular group: {group}")
        return cls.from_file(os.path.join(project_path(asset_directory), group + ".lark"), limits=limits)

    def _template(self, tokenizer: Any) -> Any:
        import llguidance

        key = id(tokenizer)
        if key not in self._templates:
            matcher = llguidance.LLMatcher(tokenizer, self.compiled, log_level=0, limits=self.limits)
            result = GrammarCursor(matcher, tokenizer).result
            if result.status != "viable":
                raise GrammarError(f"Cannot initialize grammar matcher: {result.error}")
            self._templates[key] = (tokenizer, matcher)
        return self._templates[key][1]

    def native_cursor(self, tokenizer: Any, *, blocked_token_ids: Sequence[int] = ()) -> GrammarCursor:
        """Accept an LLTokenizer; the caller retains responsibility for token IDs."""
        return GrammarCursor(self._template(tokenizer).deep_copy(), tokenizer, blocked_token_ids=blocked_token_ids)

    def token_cursor(self, tokenizer: Any, *, n_vocab: int | None = None, eos_token: int | None = None) -> GrammarCursor:
        """SMILES adapter or native fast HF AR adapter, cached per tokenizer object."""
        import llguidance
        from src.tokenizer import SmilesTokenizer

        key = id(tokenizer)
        cached = self._adapters.get(key)
        if cached is None:
            if isinstance(tokenizer, SmilesTokenizer):
                if n_vocab is not None and n_vocab != tokenizer.vocab_size:
                    raise ValueError("SMILES grammar vocabulary must match the model")
                if eos_token is not None and eos_token != tokenizer.eos_token_id:
                    raise ValueError("SMILES grammar EOS must match the model")
                native = llguidance.LLTokenizer(llguidance.TokenizerWrapper(tokenizer), slices=[])
                blocked = (tokenizer.special_tokens.mask_id, tokenizer.special_tokens.pad_id)
            else:
                from llguidance.hf import from_tokenizer

                native = from_tokenizer(tokenizer, n_vocab=n_vocab, eos_token=eos_token, slices=[])
                blocked = ()
            self._adapters[key] = (tokenizer, native, blocked)
        else:
            _, native, blocked = cached
            if n_vocab is not None and n_vocab != native.vocab_size:
                raise ValueError("Cached AR grammar adapter has a different vocabulary size")
            if eos_token is not None and eos_token != native.eos_token:
                raise ValueError("Cached grammar adapter has a different EOS token")
        return self.native_cursor(native, blocked_token_ids=blocked)

    def check_prefix(self, prefix: str) -> PrefixCheck:
        if not isinstance(prefix, str):
            raise TypeError("Grammar prefix must be a string")
        return self.native_cursor(self._byte_tokenizer).advance(list(prefix.encode("utf-8")))

    def accepts(self, smiles: str) -> bool:
        result = self.check_prefix(smiles)
        if result.status == "unknown":
            raise GrammarError(result.error or "Unknown grammar decision")
        return result.status == "viable" and result.accepting
