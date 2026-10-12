"""One RDKit policy for dataset identity and generated-molecule evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rdkit import Chem, rdBase
from rdkit.Chem import QED

from src.types import require_count


@dataclass(frozen=True)
class ChemistryPolicy:
    canonical: bool
    isomeric_smiles: bool
    sanitize: bool
    parse_names: bool
    allow_cxsmiles: bool
    heavy_atom_limit: int

    def __post_init__(self) -> None:
        require_count(self.heavy_atom_limit, "heavy_atom_limit", positive=True)
        for name in ("canonical", "isomeric_smiles", "sanitize", "parse_names", "allow_cxsmiles"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")
        if not self.canonical or not self.sanitize:
            raise ValueError("QM9 identity and QED require canonical, sanitized molecules")


@dataclass(frozen=True)
class MoleculeAssessment:
    raw_smiles: str
    smiles: str
    valid: bool
    canonical_smiles: str | None
    qed: float
    heavy_atoms: int | None
    within_heavy_atom_limit: bool | None
    error: str | None


def rdkit_version() -> str:
    return rdBase.rdkitVersion


def assess_smiles(text: str, *, policy: ChemistryPolicy) -> MoleculeAssessment:
    """Return zero score for invalid text, without changing global RDKit logging.

    The heavy-atom flag is a diagnostic, not a full test of QM9 membership and
    not an extra reward constraint. QED/calculation failures on parsed molecules
    propagate; they must not silently become invalid samples or zero rewards.
    """
    if not isinstance(text, str):
        raise TypeError("SMILES must be text")
    smiles = text.strip()

    def invalid(reason: str) -> MoleculeAssessment:
        return MoleculeAssessment(text, smiles, False, None, 0.0, None, None, reason)

    if not smiles:
        return invalid("empty_smiles")
    parameters = Chem.SmilesParserParams()
    parameters.sanitize = policy.sanitize
    parameters.parseName = policy.parse_names
    parameters.allowCXSMILES = policy.allow_cxsmiles
    try:
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(smiles, parameters)
    except (ValueError, RuntimeError) as exc:
        return invalid(f"smiles_parse_error: {exc}")
    if molecule is None or molecule.GetNumAtoms() == 0:
        return invalid("invalid_smiles")
    canonical = Chem.MolToSmiles(molecule, canonical=policy.canonical,
                                isomericSmiles=policy.isomeric_smiles)
    score = float(QED.qed(molecule))
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("RDKit returned QED outside [0, 1]")
    heavy_atoms = molecule.GetNumHeavyAtoms()
    return MoleculeAssessment(text, smiles, True, canonical, score, heavy_atoms,
                              heavy_atoms <= policy.heavy_atom_limit, None)


def molecular_identity(text: str, *, policy: ChemistryPolicy) -> str | None:
    """Canonical isomeric SMILES; invalid molecules have no identity."""
    return assess_smiles(text, policy=policy).canonical_smiles
