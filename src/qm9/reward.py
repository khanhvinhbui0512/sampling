"""QED scoring and calibrated Pass@95 with token-level terminal checks.

Calibration JSON schema is defined by BinaryCalibration. The calibration task
writes it as calibration.json in a completed calibrate_qm9 attempt.
This module never computes a dataset percentile or supplies a default threshold.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.io import read_completed_manifest, read_json
from src.qm9.chemistry import ChemistryPolicy, MoleculeAssessment, assess_smiles, rdkit_version
from src.qm9.dataset import load_prepared_qm9
from src.types import MaskedState, TokenRules, require_count, require_id


class CalibrationRequired(ValueError):
    """A binary reward was requested before independent calibration was available."""


class IncompleteStateError(ValueError):
    """An incomplete state cannot have an exact terminal reward."""


@dataclass(frozen=True)
class BinaryCalibration:
    """policy_id/reference_id describe the independent BASE calibration law.

    The resulting threshold is shared by all comparison methods, including
    q_rho completions; it is not recalibrated for each guided sampling policy.
    """

    calibration_id: str
    reference_id: str
    policy_id: str
    checkpoint_sha256: str
    tokenizer_sha256: str
    threshold: float
    quantile: float
    sample_count: int
    seed: int
    quantile_method: str
    chemistry_policy: dict[str, Any]
    rdkit_version: str
    threshold_source: str
    score_name: str
    terminal_policy: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.threshold_source != "independent_base_calibration":
            raise ValueError("Expected independent BASE calibration schema version 1")
        for name in ("calibration_id", "reference_id", "policy_id", "quantile_method", "rdkit_version"):
            require_id(getattr(self, name), name)
        for name in ("checkpoint_sha256", "tokenizer_sha256"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"Calibration {name} must be a SHA-256 digest")
        if isinstance(self.threshold, bool) or not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("Calibration threshold must be finite and within [0, 1]")
        if self.quantile != 0.95:
            raise ValueError("This QM9 protocol uses Pass@95")
        if self.score_name != "qed_or_zero" or self.terminal_policy != "strict_special_tokens_v1":
            raise ValueError("Calibration score or terminal token policy differs from the QM9 protocol")
        require_count(self.sample_count, "calibration sample_count", positive=True)
        require_count(self.seed, "calibration seed")
        ChemistryPolicy(**self.chemistry_policy)


def load_calibration(manifest_path: str) -> BinaryCalibration:
    manifest = read_completed_manifest(manifest_path)
    if manifest.get("task") != "calibrate_qm9" or "calibration.json" not in manifest["artifacts"]:
        raise ValueError("Expected a completed QM9 calibration manifest")
    payload = read_json(os.path.join(os.path.dirname(manifest_path), "calibration.json"))
    calibration = BinaryCalibration(**payload)
    if calibration.reference_id != manifest["reference_id"]:
        raise ValueError("Calibration reference differs from its manifest")
    return calibration


class TerminalStatus(str, Enum):
    INCOMPLETE = "incomplete"
    INVALID_TOKENS = "invalid_tokens"
    INVALID_SMILES = "invalid_smiles"
    SCORED = "scored"


@dataclass(frozen=True)
class TerminalReward:
    status: TerminalStatus
    valid: bool | None
    score: float | None
    reward: int | None
    calibration_id: str | None
    binary_status: str
    molecule: MoleculeAssessment | None = None
    error: str | None = None


def _terminal_text(state: MaskedState, tokenizer: Any) -> tuple[str | None, str | None]:
    """No text replacement: inspect all tokens before interpreting EOS/PAD."""
    rules = state.rules
    if not state.is_complete:
        return None, "remaining_mask"
    ids = list(state.token_ids)
    if rules.unk_id is not None and rules.unk_id in ids:
        return None, "unknown_token"
    if ids and rules.bos_id is not None and ids[0] == rules.bos_id:
        ids = ids[1:]
    content: list[int] = []
    for index, token in enumerate(ids):
        if rules.eos_id is not None and token == rules.eos_id:
            if any(item != rules.pad_id for item in ids[index + 1:]):
                return None, "non_padding_after_eos"
            break
        if rules.pad_id is not None and token == rules.pad_id:
            if any(item != rules.pad_id for item in ids[index + 1:]):
                return None, "non_padding_after_pad"
            break
        if rules.bos_id is not None and token == rules.bos_id:
            return None, "misplaced_bos"
        if token in tokenizer.all_special_ids:
            return None, "unsupported_special_token"
        content.append(token)
    return tokenizer.decode(content, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False), None


class QM9Reward:
    """Score complete states using the same tokenizer/chemistry as prepared assets.

    Providing calibration requires the expected BASE calibration identities. No model
    weights are loaded here. Omitting calibration gives continuous scores with
    reward=None and binary_status=unavailable_calibration.
    """

    def __init__(self, *, tokenizer: Any, token_rules: TokenRules,
                 chemistry_policy: ChemistryPolicy, tokenizer_sha256: str,
                 terminal_policy: str = "strict_special_tokens_v1",
                 calibration: BinaryCalibration | None = None,
                 calibration_reference_id: str | None = None,
                 calibration_policy_id: str | None = None,
                 checkpoint_sha256: str | None = None) -> None:
        if terminal_policy != "strict_special_tokens_v1":
            raise ValueError("Unsupported QM9 terminal token policy")
        if len(tokenizer) != token_rules.vocab_size:
            raise ValueError("Reward tokenizer vocabulary differs from token rules")
        for role in ("mask", "bos", "eos", "pad", "unk"):
            if getattr(tokenizer, f"{role}_token_id", None) != getattr(token_rules, f"{role}_id"):
                raise ValueError(f"Reward tokenizer {role} ID differs from token rules")
        if not isinstance(tokenizer_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", tokenizer_sha256):
            raise ValueError("Reward requires the full prepared-tokenizer SHA-256 identity")
        if calibration is not None:
            expected = (calibration_reference_id, calibration_policy_id,
                        checkpoint_sha256, tokenizer_sha256)
            actual = (calibration.reference_id, calibration.policy_id,
                      calibration.checkpoint_sha256, calibration.tokenizer_sha256)
            if any(value is None for value in expected) or expected != actual:
                raise ValueError("Calibration BASE reference/policy/checkpoint/tokenizer mismatch")
            if (ChemistryPolicy(**calibration.chemistry_policy) != chemistry_policy
                    or calibration.rdkit_version != rdkit_version()
                    or calibration.terminal_policy != terminal_policy):
                raise ValueError("Calibration chemistry policy or RDKit version mismatch")
        self.tokenizer = tokenizer
        self.token_rules = token_rules
        self.chemistry_policy = chemistry_policy
        self.tokenizer_sha256 = tokenizer_sha256
        self.calibration = calibration

    @classmethod
    def from_assets(cls, manifest_path: str, *, calibration_manifest: str | None = None,
                    calibration_reference_id: str | None = None,
                    calibration_policy_id: str | None = None,
                    checkpoint_sha256: str | None = None) -> QM9Reward:
        """Load verified local tokenizer assets; fetching or executing Hub code is disabled."""
        from transformers import AutoTokenizer

        assets = load_prepared_qm9(manifest_path)
        tokenizer = AutoTokenizer.from_pretrained(assets.tokenizer_directory,
                                                  local_files_only=True, trust_remote_code=False)
        vocabulary = read_json(os.path.join(os.path.dirname(assets.manifest_path), "vocabulary.json"))
        if tokenizer.get_vocab() != vocabulary:
            raise ValueError("Loaded tokenizer vocabulary differs from the prepared vocabulary")
        calibration = load_calibration(calibration_manifest) if calibration_manifest is not None else None
        return cls(tokenizer=tokenizer, token_rules=assets.token_rules,
                   chemistry_policy=assets.chemistry_policy, tokenizer_sha256=assets.tokenizer_sha256,
                   terminal_policy=assets.terminal_policy, calibration=calibration,
                   calibration_reference_id=calibration_reference_id,
                   calibration_policy_id=calibration_policy_id, checkpoint_sha256=checkpoint_sha256)

    def evaluate(self, state: MaskedState, *, require_binary: bool = False) -> TerminalReward:
        if state.rules != self.token_rules:
            raise ValueError("State token rules differ from reward token rules")
        calibration_id = self.calibration.calibration_id if self.calibration is not None else None
        if not state.is_complete:
            if require_binary:
                raise IncompleteStateError("Cannot assign an exact reward to remaining MASK tokens")
            return TerminalReward(TerminalStatus.INCOMPLETE, None, None, None, calibration_id,
                                  "unavailable_incomplete", error="remaining_mask")
        if require_binary and self.calibration is None:
            raise CalibrationRequired("Pass@95 requires an independent BASE calibration manifest")
        binary_status = "available" if self.calibration is not None else "unavailable_calibration"
        text, error = _terminal_text(state, self.tokenizer)
        if error is not None:
            return TerminalReward(TerminalStatus.INVALID_TOKENS, False, 0.0,
                                  0 if self.calibration is not None else None,
                                  calibration_id, binary_status, error=error)
        molecule = assess_smiles(text, policy=self.chemistry_policy)
        reward = None
        if self.calibration is not None:
            reward = int(molecule.valid and molecule.qed >= self.calibration.threshold)
        status = TerminalStatus.SCORED if molecule.valid else TerminalStatus.INVALID_SMILES
        return TerminalReward(status, molecule.valid, molecule.qed, reward, calibration_id,
                              binary_status, molecule, molecule.error)
