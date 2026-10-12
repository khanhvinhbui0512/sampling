"""One terminal checker for corpus labels, verifier targets, values, and sampling.

Primary tau is grammar AND RDKit AND the original CARS SMARTS membership.
Replication tau is grammar only; chemical diagnostics are still reported.
Neither novelty nor uniqueness changes tau. Incomplete states have no terminal
value, and infrastructure failures are raised instead of labeled failures.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from src.config import project_path
from src.data import InvalidSmiles, normalize_smiles
from src.grammar import SmilesGrammar
from src.schemas import GROUPS, RewardResult, SmilesState, Tokenizer
from src.tokenizer import TokenizationError

REWARD_VERSIONS = {
    "grammar_rdkit_smarts": "grammar-rdkit-smarts-v1",
    "grammar_only": "grammar-only-v1",
}


class RewardChecker:
    def __init__(
        self, *, mode: str = "grammar_rdkit_smarts", version: str | None = None,
        asset_directory: str = "assets/benchmark", tracker: Any = None,
        grammars: Mapping[str, SmilesGrammar] | None = None,
    ) -> None:
        from rdkit import Chem

        if mode not in REWARD_VERSIONS:
            raise ValueError(f"Unknown reward mode: {mode}")
        self.mode = mode
        self.version = version or REWARD_VERSIONS[mode]
        if self.version != REWARD_VERSIONS[mode]:
            raise ValueError("Reward version does not match its definition")
        self.asset_directory = project_path(asset_directory)
        self.tracker = tracker
        with open(os.path.join(self.asset_directory, "smarts.json"), encoding="utf-8") as handle:
            definitions = json.load(handle)
        if (
            definitions.get("schema_version") != 1
            or definitions.get("matcher") != "rdkit_has_substruct_match_any"
            or definitions.get("use_chirality") is not False
            or set(definitions.get("groups", {})) != set(GROUPS)
        ):
            raise ValueError("Unsupported membership definition")
        self.patterns = {}
        for group in GROUPS:
            patterns = definitions["groups"][group]
            if not isinstance(patterns, list) or not patterns or any(not isinstance(pattern, str) for pattern in patterns):
                raise ValueError(f"Invalid SMARTS list for {group}")
            compiled = tuple(Chem.MolFromSmarts(pattern) for pattern in patterns)
            if any(pattern is None for pattern in compiled):
                raise ValueError(f"Cannot compile SMARTS for {group}")
            self.patterns[group] = compiled
        if grammars is not None and set(grammars) != set(GROUPS):
            raise ValueError("Supply a grammar for each group")
        self.grammars = dict(grammars) if grammars is not None else {
            group: SmilesGrammar.for_group(group, asset_directory=self.asset_directory) for group in GROUPS
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, tracker: Any = None) -> RewardChecker:
        return cls(
            mode=config["reward"]["mode"], version=config["reward"]["version"],
            asset_directory=config["paths"]["benchmark"], tracker=tracker,
        )

    def _count(self, counter: str) -> None:
        if self.tracker is not None:
            self.tracker.increment(counter)

    def classify_molecule(self, molecule: Any) -> tuple[str, ...]:
        """Original SMARTS labels; grammar and corpus filters are separate checks."""
        if molecule is None:
            raise ValueError("Classification requires an RDKit molecule")
        return tuple(group for group in GROUPS if any(
            molecule.HasSubstructMatch(pattern, useChirality=False) for pattern in self.patterns[group]
        ))

    def classify_smiles(self, smiles: str) -> tuple[str, ...]:
        try:
            molecule, _ = normalize_smiles(smiles)
        except InvalidSmiles:
            return ()
        return self.classify_molecule(molecule)

    def evaluate_smiles(self, smiles: str, group: str) -> RewardResult:
        if group not in GROUPS:
            raise ValueError(f"Unknown molecular group: {group}")
        self._count("reward_checks")
        self._count("grammar_checks")
        # GrammarError must propagate: a parser limit is not evidence of tau=0.
        grammar_valid = self.grammars[group].accepts(smiles)
        canonical = None
        class_member = False
        errors = [] if grammar_valid else ["grammar_invalid"]
        try:
            molecule, canonical = normalize_smiles(smiles)
            rdkit_valid = True
            class_member = group in self.classify_molecule(molecule)
            if not class_member:
                errors.append("class_mismatch")
        except InvalidSmiles as exc:
            rdkit_valid = False
            errors.append(str(exc))
        accepted = grammar_valid and (self.mode == "grammar_only" or (rdkit_valid and class_member))
        return RewardResult(
            float(accepted), self.version, grammar_valid, rdkit_valid, class_member,
            canonical_smiles=canonical, error=";".join(errors) or None,
        )

    def evaluate_state(self, state: SmilesState, tokenizer: Tokenizer) -> RewardResult:
        if state.special_tokens != tokenizer.special_tokens:
            raise ValueError("State uses a different tokenizer")
        if state.terminal_status != "complete":
            self._count("reward_checks")
            return RewardResult(0.0, self.version, False, False, False, error=state.terminal_status + "_state")
        try:
            smiles = tokenizer.decode_state(state)
        except TokenizationError as exc:
            self._count("reward_checks")
            return RewardResult(0.0, self.version, False, False, False, error=f"invalid_tokens:{exc}")
        return self.evaluate_smiles(smiles, state.group)

    def label(self, smiles: str, group: str) -> float:
        """Verifier/rollout label uses exactly the same terminal tau as acceptance."""
        return self.evaluate_smiles(smiles, group).reward

    def terminal_value(self, state: SmilesState, tokenizer: Tokenizer) -> float:
        if not state.is_terminal:
            raise ValueError("An incomplete state needs a value estimate, not terminal tau")
        return self.evaluate_state(state, tokenizer).reward

    def terminal_accept(self, state: SmilesState, tokenizer: Tokenizer) -> bool:
        return self.evaluate_state(state, tokenizer).reward == 1.0
