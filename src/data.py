"""Molecule identity, leakage-free splits, augmentation, and relative-path I/O.

These corpus utilities are new experiment settings. Identity is RDKit canonical
isomeric SMILES with atom maps removed. Charges, isotopes, stereochemistry,
tautomers, and disconnected fragments are preserved; filtering is separate.
The caller supplies membership classification, avoiding a reward import cycle.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.artifacts import ArtifactStore
from src.config import project_path
from src.schemas import GROUPS, require_count

IDENTITY_VERSION = "rdkit-canonical-isomeric-no-atom-maps-v1"
SPLITS = ("train", "validation", "test")


class InvalidSmiles(ValueError):
    """Input cannot represent one sanitized SMILES without hidden cleanup."""


def parse_smiles(smiles: str) -> Any:
    """Reject names, CXSMILES, whitespace, and empty strings instead of stripping."""
    from rdkit import Chem, rdBase

    if not isinstance(smiles, str) or not smiles or not smiles.isascii():
        raise InvalidSmiles("empty_or_non_ascii_smiles")
    if any(character.isspace() or ord(character) < 32 for character in smiles):
        raise InvalidSmiles("whitespace_in_smiles")
    parameters = Chem.SmilesParserParams()
    parameters.sanitize = True
    parameters.removeHs = True
    parameters.parseName = False
    parameters.allowCXSMILES = False
    # Preserve errors as structured diagnostics, without flooding tqdm output.
    with rdBase.BlockLogs():
        try:
            molecule = Chem.MolFromSmiles(smiles, parameters)
        except (ValueError, RuntimeError) as exc:
            raise InvalidSmiles("rdkit_parse_or_sanitize_failed") from exc
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise InvalidSmiles("rdkit_parse_or_sanitize_failed")
    return molecule


def normalize_smiles(smiles: str) -> tuple[Any, str]:
    """Return a sanitized graph and its identity, preserving chemical structure."""
    from rdkit import Chem

    molecule = parse_smiles(smiles)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    if not canonical:
        raise InvalidSmiles("empty_canonical_smiles")
    return molecule, canonical


def canonicalize_smiles(smiles: str) -> str:
    return normalize_smiles(smiles)[1]


def molecule_id(canonical_smiles: str) -> str:
    if not isinstance(canonical_smiles, str) or not canonical_smiles:
        raise ValueError("A molecule ID requires nonempty canonical SMILES")
    return hashlib.sha256((IDENTITY_VERSION + "\0" + canonical_smiles).encode("ascii")).hexdigest()


def read_smiles(path: str) -> list[str]:
    """Read every row; a blank or malformed row is an error, never silently lost."""
    path = project_path(path)
    with open(path, encoding="utf-8", newline="") as handle:
        lines = handle.read().splitlines()
    if not lines or any(not line or line != line.strip() for line in lines):
        raise ValueError(f"Empty file, blank row, or surrounding whitespace in {path}")
    return lines


@dataclass(frozen=True, slots=True)
class MoleculeRecord:
    canonical_smiles: str
    groups: tuple[str, ...]
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.canonical_smiles:
            raise ValueError("MoleculeRecord needs canonical SMILES")
        if any(group not in GROUPS for group in self.groups):
            raise ValueError("Unknown molecular group")
        object.__setattr__(self, "groups", tuple(group for group in GROUPS if group in self.groups))
        if any(not isinstance(value, str) or not value for value in self.source_ids):
            raise ValueError("source_ids must contain nonempty strings")
        object.__setattr__(self, "source_ids", tuple(sorted(set(self.source_ids))))

    @property
    def molecule_id(self) -> str:
        return molecule_id(self.canonical_smiles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_smiles": self.canonical_smiles, "molecule_id": self.molecule_id,
            "groups": list(self.groups), "source_ids": list(self.source_ids),
        }


def prepare_molecules(
    rows: Iterable[Mapping[str, Any]], classify: Callable[[Any], Sequence[str]], *,
    exclude_smiles: Iterable[str] = (), require_membership: bool = True,
    allow_disconnected: bool = False, allow_dummy_atoms: bool = False,
    max_heavy_atoms: int | None = None,
) -> tuple[list[MoleculeRecord], dict[str, int]]:
    """Canonicalize, exclude prompt identities across all labels, and merge duplicates.

    Invalid exclusion examples have no molecular identity; the benchmark report
    records them separately. Valid exclusions are applied before splitting.
    """
    from rdkit import Chem

    if any(type(flag) is not bool for flag in (require_membership, allow_disconnected, allow_dummy_atoms)):
        raise TypeError("Corpus filters must be booleans")
    if max_heavy_atoms is not None:
        require_count(max_heavy_atoms, "max_heavy_atoms", positive=True)
    excluded = set()
    for smiles in exclude_smiles:
        try:
            excluded.add(canonicalize_smiles(smiles))
        except InvalidSmiles:
            continue
    merged: dict[str, MoleculeRecord] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        counts["input_rows"] += 1
        try:
            molecule, canonical = normalize_smiles(row["smiles"])
        except InvalidSmiles as exc:
            counts[str(exc)] += 1
            continue
        if canonical in excluded:
            counts["prompt_identity"] += 1
            continue
        if not allow_disconnected and len(Chem.GetMolFrags(molecule)) > 1:
            counts["disconnected"] += 1
            continue
        if not allow_dummy_atoms and any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
            counts["dummy_atom"] += 1
            continue
        if max_heavy_atoms is not None and molecule.GetNumHeavyAtoms() > max_heavy_atoms:
            counts["too_many_heavy_atoms"] += 1
            continue
        groups = tuple(classify(molecule))
        if any(group not in GROUPS for group in groups):
            raise ValueError("Classifier returned an unknown group")
        if require_membership and not groups:
            counts["no_membership"] += 1
            continue
        source = row.get("source_id")
        sources = () if source is None else (str(source),)
        old = merged.get(canonical)
        if old is not None:
            counts["duplicate_rows"] += 1
            groups = (*old.groups, *groups)
            sources = (*old.source_ids, *sources)
        merged[canonical] = MoleculeRecord(canonical, groups, sources)
        counts["retained_rows"] += 1
    result = sorted(merged.values(), key=lambda record: record.molecule_id)
    counts["unique_molecules"] = len(result)
    return result, dict(counts)


def split_molecules(
    records: Sequence[MoleculeRecord], *, seed: int = 0,
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
) -> dict[str, list[MoleculeRecord]]:
    """Split unique graphs once; all labels and later SMILES variants inherit it.

    Sort by seeded SHA-256, independent of input order. Train and validation counts
    are floored; test receives the remainder. This split is not label-stratified.
    """
    require_count(seed, "split seed")
    if len(fractions) != 3 or any(
        isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0
        for value in fractions
    ) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-12):
        raise ValueError("Three positive finite split fractions must sum to one")
    if len({record.molecule_id for record in records}) != len(records):
        raise ValueError("Deduplicate molecules before splitting")
    ordered = sorted(records, key=lambda record: (
        hashlib.sha256(f"{seed}\0{record.molecule_id}".encode("ascii")).digest(), record.molecule_id,
    ))
    train_end = math.floor(len(ordered) * fractions[0])
    validation_end = train_end + math.floor(len(ordered) * fractions[1])
    return dict(zip(SPLITS, (ordered[:train_end], ordered[train_end:validation_end], ordered[validation_end:]), strict=True))


def augment_molecule(
    record: MoleculeRecord, *, split: str, seed: int = 0, random_variants: int = 2,
    max_attempts_per_variant: int = 20,
) -> list[dict[str, Any]]:
    """Keep canonical SMILES, then enumerate unique train-only atom permutations.

    A private RNG and RenumberAtoms avoid RDKit's global random generator.
    Symmetric molecules may yield fewer variants than requested. Every result is
    checked against the original graph identity; no truncation or relabeling occurs.
    """
    from rdkit import Chem

    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    require_count(seed, "augmentation seed")
    require_count(random_variants, "random_variants")
    require_count(max_attempts_per_variant, "max_attempts_per_variant", positive=True)
    molecule, canonical = normalize_smiles(record.canonical_smiles)
    if canonical != record.canonical_smiles:
        raise ValueError("Augmentation requires a canonical MoleculeRecord")
    variants = [canonical]
    seen = {canonical}
    if split == "train" and random_variants:
        rng = random.Random(hashlib.sha256(f"{seed}\0{record.molecule_id}".encode("ascii")).digest())
        for _ in range(random_variants * max_attempts_per_variant):
            order = list(range(molecule.GetNumAtoms()))
            rng.shuffle(order)
            candidate = Chem.MolToSmiles(Chem.RenumberAtoms(molecule, order), canonical=False, isomericSmiles=True)
            if canonicalize_smiles(candidate) != canonical:
                raise RuntimeError("Augmentation changed molecular identity")
            if candidate not in seen:
                seen.add(candidate)
                variants.append(candidate)
            if len(variants) == random_variants + 1:
                break
    return [{**record.to_dict(), "smiles": smiles, "split": split, "augmentation_index": index}
            for index, smiles in enumerate(variants)]


def assert_disjoint_splits(rows_by_split: Mapping[str, Iterable[Mapping[str, Any]]]) -> None:
    """Check graph identity, including augmented strings and repeated label rows."""
    if set(rows_by_split) != set(SPLITS):
        raise ValueError("Expected train, validation, test")
    owner: dict[str, str] = {}
    for split in SPLITS:
        for row in rows_by_split[split]:
            canonical = canonicalize_smiles(row.get("smiles", row["canonical_smiles"]))
            if canonical != row["canonical_smiles"] or molecule_id(canonical) != row["molecule_id"]:
                raise ValueError("Row identity does not match its SMILES")
            if row.get("split", split) != split:
                raise ValueError("Row has the wrong split label")
            identity = row["molecule_id"]
            if identity in owner and owner[identity] != split:
                raise ValueError(f"Molecule leaks between {owner[identity]} and {split}: {identity}")
            owner[identity] = split


def write_parquet(
    store: ArtifactStore, relative: str, rows: Iterable[Mapping[str, Any]], *,
    columns: Sequence[str] | None = None,
) -> str:
    """Commit one complete parquet file atomically inside the artifact root."""
    import pandas as pd

    path = store.path(relative)
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".parquet-", suffix=".tmp", dir=parent)
    temporary = os.path.relpath(temporary)
    os.close(descriptor)
    try:
        pd.DataFrame.from_records(rows, columns=columns).to_parquet(temporary, index=False, engine="pyarrow")
        with open(temporary, "r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def read_parquet(store: ArtifactStore, relative: str) -> list[dict[str, Any]]:
    import pandas as pd

    return pd.read_parquet(store.require_file(relative), engine="pyarrow").to_dict("records")
