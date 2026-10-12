"""Sound failure certificates; UNKNOWN always leaves a branch available.

The CARS paper allows incomplete prefix oracles. QM9 uses exact leaf rewards
by default. The optional token checks below prove failure only for this project's
strict decoder; they are not a SMILES grammar or a bound on partial-molecule QED.
"""

from dataclasses import asdict, dataclass

from src.io import payload_hash
from src.types import CostPhase, MaskedState, require_id


@dataclass(frozen=True)
class OracleResult:
    state: MaskedState
    oracle_id: str
    certified_failure: bool
    reason: str

    def __post_init__(self):
        require_id(self.oracle_id, "oracle_id")
        require_id(self.reason, "oracle reason")
        if type(self.certified_failure) is not bool:
            raise ValueError("certified_failure must be bool")


class QM9Oracle:
    def __init__(self, reward, *, mode="leaf_only"):
        if mode not in ("leaf_only", "token_constraints"):
            raise ValueError("QM9 oracle mode must be leaf_only or token_constraints")
        if reward.calibration is None:
            raise ValueError("The QM9 failure oracle requires calibrated binary reward")
        self.reward, self.rules, self.mode = reward, reward.token_rules, mode
        self._calibration = reward.calibration
        # The partial-token proofs assume QM9's distinct special-token roles.
        special = [getattr(self.rules, name) for name in
                   ("mask_id", "bos_id", "eos_id", "pad_id", "unk_id")]
        special = [token for token in special if token is not None]
        if len(special) != len(set(special)):
            raise ValueError("QM9 oracle requires distinct special-token IDs")
        self.oracle_id = "qm9-oracle-v1:" + payload_hash({
            "mode": mode, "rules": asdict(self.rules),
            "calibration": asdict(reward.calibration),
            "tokenizer": reward.tokenizer_sha256,
            "terminal_policy": "strict_special_tokens_v1",
        })

    def _known_token_failure(self, state):
        tokens, rules = state.token_ids, self.rules
        if rules.unk_id is not None and rules.unk_id in tokens:
            return "known_unk"
        if rules.bos_id is not None and rules.bos_id in tokens[1:]:
            return "known_bos_outside_first_position"
        allowed = {rules.mask_id, rules.bos_id, rules.eos_id, rules.pad_id, rules.unk_id}
        unsupported = set(self.reward.tokenizer.all_special_ids) - allowed
        if any(token in unsupported for token in tokens):
            return "known_unsupported_special_token"
        terminators = {token for token in (rules.eos_id, rules.pad_id) if token is not None}
        for position, token in enumerate(tokens):
            if token in terminators and any(
                later not in (rules.mask_id, rules.pad_id) for later in tokens[position + 1:]
            ):
                return "known_non_padding_after_terminator"
        return None

    def check(self, state, *, budget, phase=CostPhase.DISCOVERY):
        if state.rules != self.rules:
            raise ValueError("Oracle/state token rules differ")
        if self.reward.calibration != self._calibration:
            raise ValueError("Oracle calibration changed; create a new oracle and trie context")
        with budget.oracle_check(phase=phase):
            if state.is_complete:
                terminal = self.reward.evaluate(state, require_binary=True)
                return OracleResult(state, self.oracle_id, terminal.reward == 0,
                                    "terminal_reward_zero" if terminal.reward == 0
                                    else "successful_terminal")
            reason = self._known_token_failure(state) if self.mode == "token_constraints" else None
            return OracleResult(state, self.oracle_id, reason is not None,
                                reason or "unknown_partial_completion")


class FiniteLanguageOracle:
    """Exact oracle for an explicitly enumerated finite toy language only.

    Keep every observed position, including observations after masks. The test
    grammar is expanded into fixed-horizon token sequences before construction.
    This class is not a QM9 grammar or a scalable molecular oracle.
    """

    def __init__(self, rules, accepted_sequences):
        self.rules = rules
        tokens = sorted(set(tuple(sequence) for sequence in accepted_sequences))
        self._accepted = tuple(MaskedState(sequence, rules) for sequence in tokens)
        if any(not state.is_complete for state in self._accepted):
            raise ValueError("Finite-language members must be complete")
        self.oracle_id = "finite-language-v1:" + payload_hash({
            "rules": asdict(rules), "accepted": tokens,
        })

    def check(self, state, *, budget, phase=CostPhase.DISCOVERY):
        if state.rules != self.rules:
            raise ValueError("Oracle/state token rules differ")
        with budget.oracle_check(phase=phase):
            possible = any(state.accepts_completion(leaf) for leaf in self._accepted)
            return OracleResult(state, self.oracle_id, not possible,
                                "compatible_member_exists" if possible
                                else "no_compatible_member_at_this_horizon")
