"""Lossless SMILES tokenization with explicit MASK/EOS/PAD and no UNK token.

Bracket atoms, Br/Cl, percent ring labels, and dative arrows are single tokens.
Tokenization is lexical, not a chemical validity check. Vocabulary is built from
training strings only; unsupported lexemes and overlength rows are errors.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from src.artifacts import ArtifactStore, sha256_file
from src.schemas import SmilesState, SpecialTokens, require_count, require_sha256

TOKENIZER_VERSION = "smiles-lexical-v1"
SPECIAL_NAMES = ("<pad>", "<mask>", "<eos>")
TOKEN_PATTERN = r"\[[^\[\]\s]+\]|Br|Cl|%\([0-9]+\)|%[0-9]{2}|->|<-|@@?|[A-Za-z]|[0-9]|[().=#$:/\\*+~\-]"
_TOKEN = re.compile(TOKEN_PATTERN, re.ASCII)


class TokenizationError(ValueError):
    pass


class UnknownToken(TokenizationError):
    pass


def tokenize_smiles(text: str) -> list[str]:
    """Consume the entire string; never discard unmatched characters or specials."""
    if not isinstance(text, str) or not text or not text.isascii():
        raise TokenizationError("SMILES must be a nonempty ASCII string")
    tokens = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            raise TokenizationError(f"Unsupported SMILES character at offset {position}: {text[position:position + 8]!r}")
        tokens.append(match.group())
        position = match.end()
    return tokens


def detokenize_smiles(tokens: Sequence[str]) -> str:
    if not tokens or any(not isinstance(token, str) or token in SPECIAL_NAMES for token in tokens):
        raise TokenizationError("Only nonempty molecular token sequences can be joined")
    if any(tokenize_smiles(token) != [token] for token in tokens):
        raise TokenizationError("Sequence contains an unsupported lexical token")
    # Generated lexemes may merge when retokenized (e.g. @ + @). Their exact
    # concatenation remains the molecule text; chemistry is checked by reward.
    return "".join(tokens)


class SmilesTokenizer:
    """No padding or special-token stripping is implicit in encode/decode."""

    bos_token_id = None

    def __init__(self, vocabulary: Sequence[str], *, max_length: int = 256) -> None:
        require_count(max_length, "max_length", positive=True)
        if max_length < 2:
            raise ValueError("max_length must leave room for one molecule token and EOS")
        self.vocabulary = tuple(vocabulary)
        if any(not isinstance(token, str) for token in self.vocabulary) or len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("Vocabulary must contain distinct strings")
        if any(name not in self.vocabulary for name in SPECIAL_NAMES):
            raise ValueError("Vocabulary must contain PAD, MASK, and EOS")
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}
        self.special_tokens = SpecialTokens(
            mask_id=self.token_to_id["<mask>"], eos_id=self.token_to_id["<eos>"], pad_id=self.token_to_id["<pad>"],
        )
        for token in self.vocabulary:
            if token not in SPECIAL_NAMES and tokenize_smiles(token) != [token]:
                raise ValueError(f"Vocabulary entry is not one lexical token: {token!r}")
        if len(self.vocabulary) == len(SPECIAL_NAMES):
            raise ValueError("Vocabulary needs at least one molecular token")
        self.max_length = max_length
        self.vocab_size = len(self.vocabulary)
        self.eos_token_id = self.special_tokens.eos_id
        self.special_token_ids = (
            self.special_tokens.pad_id, self.special_tokens.mask_id, self.special_tokens.eos_id,
        )
        # The llguidance adapter can retain multi-character tokens exactly.
        self.tokens = tuple(token.encode("ascii") for token in self.vocabulary)

    @classmethod
    def build(cls, training_smiles: Iterable[str], *, max_length: int = 256) -> SmilesTokenizer:
        vocabulary = set()
        for text in training_smiles:
            vocabulary.update(tokenize_smiles(text))
        return cls((*SPECIAL_NAMES, *sorted(vocabulary)), max_length=max_length)

    @property
    def fingerprint(self) -> str:
        payload = {"version": TOKENIZER_VERSION, "tokens": self.vocabulary, "max_length": self.max_length}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def encode_text(self, text: str, *, add_eos: bool = True) -> list[int]:
        if type(add_eos) is not bool:
            raise TypeError("add_eos must be boolean")
        lexemes = tokenize_smiles(text)
        unknown = sorted(set(lexemes) - self.token_to_id.keys())
        if unknown:
            raise UnknownToken(f"Tokens absent from training vocabulary: {unknown}")
        ids = [self.token_to_id[token] for token in lexemes]
        if add_eos:
            ids.append(self.special_tokens.eos_id)
        if len(ids) > self.max_length:
            raise TokenizationError(f"Length {len(ids)} exceeds max_length={self.max_length}; truncation is forbidden")
        return ids

    def __call__(self, text: str | bytes) -> list[int]:
        if isinstance(text, bytes):
            text = text.decode("ascii")
        return self.encode_text(text, add_eos=False)

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        lexemes = []
        for token_id in token_ids:
            require_count(token_id, "token ID")
            if token_id >= self.vocab_size:
                raise UnknownToken(f"Token ID outside vocabulary: {token_id}")
            if token_id in self.special_token_ids:
                raise TokenizationError("Special tokens must be handled through SmilesState")
            lexemes.append(self.vocabulary[token_id])
        return detokenize_smiles(lexemes)

    def decode_state(self, state: SmilesState) -> str:
        if state.special_tokens != self.special_tokens:
            raise TokenizationError("State and tokenizer have different special token IDs")
        # Validate even the suffix: an out-of-vocabulary token is never hidden.
        if any(token_id >= self.vocab_size for token_id in state.token_ids):
            raise UnknownToken("State contains an out-of-vocabulary token ID")
        return self.decode_tokens(state.molecular_token_ids())

    def encode_state(self, text: str, group: str, *, width: int | None = None) -> SmilesState:
        width = self.max_length if width is None else width
        require_count(width, "state width", positive=True)
        if width > self.max_length:
            raise TokenizationError("State width exceeds tokenizer max_length")
        ids = self.encode_text(text)
        if len(ids) > width:
            raise TokenizationError("SMILES plus EOS does not fit the state width")
        ids.extend([self.special_tokens.pad_id] * (width - len(ids)))
        return SmilesState(tuple(ids), (False,) * width, group, self.special_tokens)

    def save(
        self, store: ArtifactStore, *, vocabulary_path: str = "data/tokenizer/vocab.json",
        metadata_path: str = "data/tokenizer/metadata.json", provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if vocabulary_path == metadata_path:
            raise ValueError("Vocabulary and metadata need distinct paths")
        store.write_json(vocabulary_path, {"version": TOKENIZER_VERSION, "tokens": list(self.vocabulary)})
        metadata = {
            "version": TOKENIZER_VERSION, "pattern": TOKEN_PATTERN, "special_tokens": self.special_tokens.to_dict(),
            "max_length": self.max_length, "vocab_size": self.vocab_size, "fingerprint": self.fingerprint,
            "vocabulary_path": vocabulary_path, "vocabulary_sha256": sha256_file(store.require_file(vocabulary_path)),
            "training_only_vocabulary": True, "unknown_policy": "reject", "overlength_policy": "reject",
            "provenance": dict(provenance or {}),
        }
        store.write_json(metadata_path, metadata)
        return metadata

    @classmethod
    def load(cls, store: ArtifactStore, metadata_path: str = "data/tokenizer/metadata.json") -> SmilesTokenizer:
        metadata = store.read_json(metadata_path)
        if metadata.get("version") != TOKENIZER_VERSION or metadata.get("pattern") != TOKEN_PATTERN:
            raise ValueError("Unsupported tokenizer version or lexical rules")
        require_sha256(metadata["vocabulary_sha256"])
        store.require_file(metadata["vocabulary_path"], checksum=metadata["vocabulary_sha256"])
        vocabulary = store.read_json(metadata["vocabulary_path"])
        if vocabulary.get("version") != TOKENIZER_VERSION:
            raise ValueError("Vocabulary has a different version")
        tokenizer = cls(vocabulary["tokens"], max_length=metadata["max_length"])
        if (
            tokenizer.special_tokens.to_dict() != metadata["special_tokens"]
            or tokenizer.fingerprint != metadata["fingerprint"] or tokenizer.vocab_size != metadata["vocab_size"]
            or metadata.get("training_only_vocabulary") is not True
            or metadata.get("unknown_policy") != "reject" or metadata.get("overlength_policy") != "reject"
        ):
            raise ValueError("Tokenizer metadata disagrees with its vocabulary or policy")
        return tokenizer
