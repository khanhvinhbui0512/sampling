"""Prepare pinned QM9 assets and retain upstream row order and split semantics."""

from __future__ import annotations

import os
import platform
import re
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterator

from tqdm.auto import tqdm

from src.config import ConfigError, load_config, project_path
from src.io import RunArtifacts, file_hash, payload_hash, read_completed_manifest, read_json
from src.io import write_json, write_jsonl, write_yaml
from src.qm9.chemistry import ChemistryPolicy, assess_smiles, rdkit_version
from src.types import TokenRules


def load_qm9_config(path: str = "configs/qm9.yaml") -> dict:
    """Read the single QM9 configuration used locally and on Modal."""
    if not project_path(path).startswith("configs" + os.sep):
        raise ConfigError("Keep the QM9 configuration inside configs/")
    config = load_config(path)
    if config["task"] != "qm9":
        raise ConfigError("This project uses the QM9 task only")
    for directory in (config["dataset"]["cache_dir"], config["assets"]["output_dir"]):
        if not project_path(directory).startswith("workspace" + os.sep):
            raise ConfigError("Keep QM9 data and cache under workspace/ for Modal persistence")
    return config


def _hub_revision(api: Any, *, repo_id: str, revision: str, repo_type: str) -> dict[str, Any]:
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
    if not isinstance(info.sha, str) or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
        raise ValueError(f"Hub did not return an immutable commit SHA for {repo_id}")
    card = info.card_data.to_dict() if info.card_data is not None else {}
    prefix = "datasets/" if repo_type == "dataset" else ""
    return {"repo_id": repo_id, "requested_revision": revision, "commit_sha": info.sha,
            "url": f"https://huggingface.co/{prefix}{repo_id}/tree/{info.sha}",
            "reported_license": card.get("license")}


def token_rules_from_tokenizer(tokenizer: Any, settings: dict[str, Any]) -> tuple[TokenRules, dict[str, Any]]:
    vocabulary = tokenizer.get_vocab()
    if (not vocabulary or any(type(token) is not int for token in vocabulary.values())
            or set(vocabulary.values()) != set(range(len(tokenizer)))):
        raise ValueError("Tokenizer must expose a contiguous full vocabulary with integer IDs")
    roles = {role: getattr(tokenizer, f"{role}_token_id", None)
             for role in ("mask", "bos", "eos", "pad", "unk")}
    for role in settings["required_special_tokens"]:
        if roles[role] is None:
            raise ValueError(f"Tokenizer has no required {role.upper()} token")
    rules = TokenRules(len(tokenizer), settings["sequence_length"],
                       **{f"{role}_id": token for role, token in roles.items()})
    if tokenizer.vocab_size > rules.vocab_size:
        raise ValueError("Tokenizer base vocabulary exceeds its full vocabulary")
    if len({roles[role] for role in ("mask", "pad", "unk")}) != 3:
        raise ValueError("QM9 requires distinct MASK, PAD and UNK IDs")
    return rules, {"rules": asdict(rules), "base_vocab_size": tokenizer.vocab_size,
                   "terminal_policy": settings["terminal_policy"],
                   "special_tokens": {role: getattr(tokenizer, f"{role}_token", None)
                                      for role in roles}}


def _register_directory(attempt: RunArtifacts, name: str) -> list[str]:
    registered: list[str] = []
    for directory, directories, files in os.walk(attempt.path(name)):
        directories[:] = sorted(item for item in directories if item not in (".cache", "__pycache__"))
        for filename in sorted(files):
            relative = os.path.relpath(os.path.join(directory, filename), attempt.directory)
            attempt.register(relative)
            registered.append(relative)
    if not registered:
        raise ValueError(f"Prepared asset directory is empty: {name}")
    return registered


def _environment(config_path: str) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "PyYAML", "torch", "transformers", "tokenizers", "datasets",
                 "huggingface-hub", "rdkit", "tqdm", "modal"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    files = ("pyproject.toml", "src/config.py", "src/types.py", "src/io.py",
             "src/qm9/chemistry.py", "src/qm9/dataset.py", "src/qm9/reward.py",
             "src/qm9/tokenizer.py",
             "prepare_assets.py", "modal/prepare_assets.py", config_path)
    return {"python": platform.python_version(), "packages": packages,
            "rdkit_version": rdkit_version(),
            "source_sha256": {path: file_hash(path) for path in dict.fromkeys(files)}}


@dataclass(frozen=True)
class PreparedQM9:
    manifest_path: str
    raw_directory: str
    tokenizer_directory: str
    splits_path: str
    train_canonical_path: str
    token_rules: TokenRules
    chemistry_policy: ChemistryPolicy
    terminal_policy: str
    tokenizer_sha256: str
    train_rows: int
    test_rows: int

    def summary(self) -> dict[str, Any]:
        return {"manifest": self.manifest_path, "train_rows": self.train_rows,
                "test_rows": self.test_rows, "vocab_size": self.token_rules.vocab_size,
                "base_weights_created": False}


def load_prepared_qm9(manifest_path: str) -> PreparedQM9:
    manifest = read_completed_manifest(manifest_path)
    if manifest.get("task") != "prepare_qm9_assets":
        raise ValueError("Manifest is not a QM9 asset preparation result")
    required = {"config.yaml", "execution.json", "revisions.json", "splits.json", "environment.json",
                "token_rules.json", "vocabulary.json", "molecules.jsonl", "train_canonical.jsonl", "audit.json"}
    if not required.issubset(manifest["artifacts"]):
        raise ValueError("QM9 manifest is missing required preparation artifacts")
    directory = os.path.dirname(project_path(manifest_path))
    environment = read_json(os.path.join(directory, "environment.json"))
    if environment["rdkit_version"] != rdkit_version():
        raise ValueError("RDKit version differs from the saved canonicalization environment")
    metadata = read_json(os.path.join(directory, "token_rules.json"))
    splits = read_json(os.path.join(directory, "splits.json"))
    train_ids, test_ids = splits["train_row_ids"], splits["test_row_ids"]
    all_ids = train_ids + test_ids
    if (len(all_ids) != splits["source_rows"]
            or any(type(item) is not int for item in all_ids)
            or len(set(all_ids)) != splits["source_rows"]
            or set(all_ids) != set(range(splits["source_rows"]))):
        raise ValueError("QM9 split IDs are not a disjoint partition of source rows")
    preparation = manifest["preparation"]
    if (len(train_ids) != preparation["train_rows"] or len(test_ids) != preparation["test_rows"]
            or manifest["artifacts"]["splits.json"]["sha256"] != preparation["splits_sha256"]):
        raise ValueError("QM9 split identity differs from preparation metadata")
    tokenizer_files = [name for name in manifest["artifacts"] if name.startswith("tokenizer/")]
    if not tokenizer_files or not any(name.startswith("raw/") for name in manifest["artifacts"]):
        raise ValueError("QM9 manifest has no raw dataset or tokenizer files")
    tokenizer_hash = payload_hash({name: manifest["artifacts"][name]["sha256"]
                                   for name in [*tokenizer_files, "token_rules.json", "vocabulary.json"]})
    if tokenizer_hash != preparation["tokenizer_sha256"]:
        raise ValueError("Tokenizer identity differs from preparation metadata")
    if metadata["terminal_policy"] != "strict_special_tokens_v1":
        raise ValueError("Unsupported prepared terminal token policy")
    return PreparedQM9(
        project_path(manifest_path), os.path.join(directory, "raw"),
        os.path.join(directory, "tokenizer"), os.path.join(directory, "splits.json"),
        os.path.join(directory, "train_canonical.jsonl"), TokenRules(**metadata["rules"]),
        ChemistryPolicy(**preparation["chemistry_policy"]), metadata["terminal_policy"],
        tokenizer_hash, len(train_ids), len(test_ids),
    )


def prepare_qm9_assets(config_path: str = "configs/qm9.yaml", *, asset_id: str, attempt_id: str,
                       execution_metadata: dict[str, Any] | None = None) -> PreparedQM9:
    """Download only data/tokenizer; no pretrained model weights or generation.

    Split IDs refer to the original dataset order. Invalid molecules remain in
    the raw dataset and audits; no filtering/dedup changes the training split.
    Canonical train identities contain valid molecules only, for later novelty.
    """
    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from src.qm9.tokenizer import prepare_qm9_tokenizer, reference_qm9_ids

    settings = load_qm9_config(config_path)
    data_cfg, tok_cfg = settings["dataset"], settings["tokenizer"]
    policy = ChemistryPolicy(**settings["chemistry"])
    attempt = RunArtifacts.create(
        settings["assets"]["output_dir"], run_id=asset_id, attempt_id=attempt_id,
        config=settings, target_id="qm9-assets",
        reference_id="qm9-assets-request:" + payload_hash(settings)[:16],
    )
    attempt.manifest["task"] = "prepare_qm9_assets"
    try:
        write_yaml(attempt.path("config.yaml"), settings)
        attempt.register("config.yaml")
        write_json(attempt.path("execution.json"), execution_metadata or {"backend": "library"})
        attempt.register("execution.json")
        api = HfApi()
        revisions = {
            "dataset": _hub_revision(api, repo_id=data_cfg["repo_id"],
                                     revision=data_cfg["revision"], repo_type="dataset"),
            "tokenizer": _hub_revision(api, repo_id=tok_cfg["repo_id"],
                                       revision=tok_cfg["revision"], repo_type="model"),
        }
        write_json(attempt.path("revisions.json"), revisions)
        attempt.register("revisions.json")
        raw = load_dataset(data_cfg["repo_id"], revision=revisions["dataset"]["commit_sha"],
                           split=data_cfg["split"], cache_dir=data_cfg["cache_dir"])
        expected = data_cfg["expected_train_rows"] + data_cfg["expected_test_rows"]
        if len(raw) != expected or data_cfg["smiles_column"] not in raw.column_names:
            raise ValueError(f"QM9 source mismatch: rows={len(raw)}, expected={expected}, "
                             f"required column={data_cfg['smiles_column']!r}")
        row_ids = Dataset.from_dict({"source_row_id": list(range(len(raw)))})
        partition = row_ids.train_test_split(test_size=data_cfg["held_out_fraction"],
                                             seed=data_cfg["split_seed"], shuffle=True)
        train_ids = list(partition["train"]["source_row_id"])
        test_ids = list(partition["test"]["source_row_id"])
        if (len(train_ids) != data_cfg["expected_train_rows"]
                or len(test_ids) != data_cfg["expected_test_rows"]):
            raise ValueError("Actual split sizes differ from the configured paper protocol")
        raw.save_to_disk(attempt.path("raw"))
        required = ["config.yaml", "execution.json", "revisions.json",
                    *_register_directory(attempt, "raw")]
        splits = {"source_rows": len(raw), "held_out_fraction": data_cfg["held_out_fraction"],
                  "seed": data_cfg["split_seed"], "train_row_ids": train_ids,
                  "test_row_ids": test_ids, "dataset_fingerprint": raw._fingerprint}
        write_json(attempt.path("splits.json"), splits)
        snapshot_download(
            repo_id=tok_cfg["repo_id"], repo_type="model",
            revision=revisions["tokenizer"]["commit_sha"], local_dir=attempt.path("tokenizer"),
            allow_patterns=["*.json", "*.txt", "*.model", "*.vocab", "*.merges", "README*", "LICENSE*"],
        )
        tokenizer = prepare_qm9_tokenizer(attempt.path("tokenizer"))
        tokenizer.padding_side, tokenizer.truncation_side = tok_cfg["padding_side"], tok_cfg["truncation_side"]
        rules, metadata = token_rules_from_tokenizer(tokenizer, tok_cfg)
        vocabulary = tokenizer.get_vocab()
        write_json(attempt.path("vocabulary.json"), vocabulary)
        write_json(attempt.path("token_rules.json"), metadata)
        train_set = set(train_ids)
        canonical_train: set[str] = set()
        canonical_test: set[str] = set()
        audit = {"rows": 0, "invalid_smiles": 0, "overlength_rows": 0,
                 "max_untruncated_tokens": 0, "outside_heavy_atom_limit": 0,
                 "unknown_tokens": 0, "raw_rows_filtered": 0,
                 "tokenizer_backend": "fast_wordlevel_smiles_regex_v1",
                 "tokenization_mismatches": 0}

        def audit_rows() -> Iterator[dict[str, Any]]:
            for row_id, row in enumerate(tqdm(raw, total=len(raw), desc="Audit QM9", unit="molecule")):
                text = row[data_cfg["smiles_column"]]
                molecule = assess_smiles(text, policy=policy)
                ids = tokenizer.encode(text, add_special_tokens=tok_cfg["add_special_tokens"],
                                       truncation=False)
                if ids != reference_qm9_ids(text, vocabulary,
                                            add_special_tokens=tok_cfg["add_special_tokens"]):
                    raise ValueError(f"Fast tokenizer differs from the reference rule at source row {row_id}")
                for token in ids:
                    rules.validate_token(token)
                if rules.mask_id in ids or rules.unk_id in ids:
                    raise ValueError(f"Tokenizer produced MASK/UNK for source row {row_id}")
                overlength = len(ids) > rules.sequence_length
                split = "train" if row_id in train_set else "test"
                audit["rows"] += 1
                audit["invalid_smiles"] += int(not molecule.valid)
                audit["overlength_rows"] += int(overlength)
                audit["max_untruncated_tokens"] = max(audit["max_untruncated_tokens"], len(ids))
                audit["outside_heavy_atom_limit"] += int(molecule.within_heavy_atom_limit is False)
                if molecule.valid:
                    identities = canonical_train if split == "train" else canonical_test
                    identities.add(molecule.canonical_smiles)
                yield {"source_row_id": row_id, "split": split, **asdict(molecule),
                       "untruncated_tokens": len(ids), "would_truncate": overlength}

        write_jsonl(attempt.path("molecules.jsonl"), audit_rows(), progress=False)
        write_jsonl(attempt.path("train_canonical.jsonl"),
                    ({"canonical_smiles": identity} for identity in sorted(canonical_train)))
        audit.update({"train_unique_valid": len(canonical_train),
                      "test_unique_valid": len(canonical_test),
                      "canonical_overlap_train_test": len(canonical_train & canonical_test),
                      "overlength_policy": tok_cfg["overlength_policy"],
                      "chemistry_policy": asdict(policy), "rdkit_version": rdkit_version()})
        write_json(attempt.path("audit.json"), audit)
        write_json(attempt.path("environment.json"), _environment(config_path))
        names = ["splits.json", "vocabulary.json", "token_rules.json", "molecules.jsonl",
                 "train_canonical.jsonl", "audit.json", "environment.json"]
        records = {name: attempt.register(name) for name in names}
        tokenizer_files = _register_directory(attempt, "tokenizer")
        required.extend([*names, *tokenizer_files])
        tokenizer_hash = payload_hash({name: file_hash(attempt.path(name))
                                       for name in [*tokenizer_files, "token_rules.json", "vocabulary.json"]})
        attempt.manifest["reference_id"] = "qm9-assets:" + payload_hash(revisions)[:16]
        attempt.manifest["preparation"] = {
            "revisions": revisions, "train_rows": len(train_ids), "test_rows": len(test_ids),
            "tokenizer_sha256": tokenizer_hash, "splits_sha256": records["splits.json"]["sha256"],
            "chemistry_policy": asdict(policy), "base_weights_created": False,
        }
        attempt.complete(required_artifacts=required)
    except Exception as exc:
        if attempt.manifest["status"] == "running":
            try:
                attempt.fail(f"{type(exc).__name__}: {exc}")
            except Exception as manifest_error:
                exc.add_note(f"Could not save the failure manifest: {manifest_error}")
        raise
    return load_prepared_qm9(attempt.path("manifest.json"))
