"""T03: build a train-only SMILES vocabulary and bounded CPU tensor shards.

Run from the repository root with ``python src/tasks/build_tokenizer.py``.
Every corpus row is audited; unknown tokens and overlength rows are reported,
never replaced or truncated. Completed vocabulary/splits resume by checksum.
Shards are staged privately and tokenizer metadata is published last.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import closing
from itertools import islice
from typing import Any

from tqdm.auto import tqdm

from src.artifacts import ArtifactStore, Dependency, sha256_file
from src.config import Config, config_digest, project_path
from src.data import IDENTITY_VERSION, SPLITS, molecule_id
from src.schemas import GROUPS, ArtifactManifest, RunRecord, TaskResult, require_count
from src.tasks.common import TaskContext, run_task
from src.tasks.prepare_corpus import (
    CORPUS_VERSION,
    IO_ROWS,
    benchmark_dependency,
    checker_versions,
    commit_file,
    corpus_lock,
    parquet_rows,
    preparation_fingerprint,
    validate_preparation,
)
from src.tokenizer import TOKEN_PATTERN, TOKENIZER_VERSION, SmilesTokenizer, tokenize_smiles

SHARD_VERSION = "smiles-token-shards-v1"
EXAMPLE_LIMIT = 20


def _dependencies(config: Config, store: ArtifactStore) -> dict[str, Dependency]:
    benchmark = benchmark_dependency(config, store)
    benchmark_hash = sha256_file(store.require_file(benchmark.path))
    return {
        "benchmark": benchmark,
        "corpus": Dependency(
            store.directory("data", "corpus", "manifest.json"), manifest=True, kind="corpus",
            metadata={
                "version": CORPUS_VERSION, "identity_version": IDENTITY_VERSION,
                "preparation_fingerprint": preparation_fingerprint(config, benchmark_hash, checker_versions()),
            },
        ),
    }


def _fingerprint(config: Config, corpus_hash: str) -> str:
    return config_digest({
        "version": SHARD_VERSION, "tokenizer_version": TOKENIZER_VERSION, "pattern": TOKEN_PATTERN,
        "corpus_sha256": corpus_hash, "tokenizer": config["data"]["tokenizer"], "group_order": list(GROUPS),
    })


def _manifest_path(store: ArtifactStore, split: str) -> str:
    return store.directory("data", "tokenized", split, "manifest.json")


def _stage_path(context: TaskContext, split: str, filename: str) -> str:
    return os.path.join(context.output_path("tokenized-stage"), split, filename)


def _stage_shard(context: TaskContext, split: str, final_path: str) -> str:
    expected_parent = context.store.directory("data", "tokenized", split)
    filename = os.path.basename(final_path)
    if os.path.dirname(final_path) != expected_parent or not filename.startswith("shard_") or not filename.endswith(".pt"):
        raise ValueError(f"Unexpected token shard path: {final_path}")
    return _stage_path(context, split, filename)


def _check_sources(context: TaskContext) -> None:
    paths = {
        "benchmark": benchmark_dependency(context.config, context.store).path,
        "corpus": context.store.directory("data", "corpus", "manifest.json"),
    }
    for name, path in paths.items():
        if sha256_file(context.store.require_file(path)) != context.identity.dependency_hashes[name]:
            raise ValueError(f"{name} changed after task preflight; run T02 then T03 sequentially")


def _length_summary(histogram: Counter[int]) -> dict[str, Any]:
    total = sum(histogram.values())
    if not total:
        return {"min": None, "max": None, "mean": None, "quantiles": {}, "histogram": {}}
    ordered = sorted(histogram.items())
    quantiles = {}
    for name, fraction in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99)):
        rank = math.ceil(total * fraction)
        cumulative = 0
        for length, count in ordered:
            cumulative += count
            if cumulative >= rank:
                quantiles[name] = length
                break
    return {
        "min": ordered[0][0], "max": ordered[-1][0],
        "mean": sum(length * count for length, count in ordered) / total,
        "quantiles": quantiles, "histogram": {str(length): count for length, count in ordered},
    }


class _SplitAudit:
    """Validate graph order without keeping the full corpus in memory."""

    def __init__(self, context: TaskContext, split: str, tokenizer: SmilesTokenizer) -> None:
        self.context = context
        self.split = split
        self.tokenizer = tokenizer
        self.counts: Counter[str] = Counter()
        self.groups = {reason: Counter() for reason in ("input", "kept", "unknown", "overlength")}
        self.lengths: Counter[int] = Counter()
        self.unknown_tokens: Counter[str] = Counter()
        self.unknown_token_rows: Counter[str] = Counter()
        self.examples: dict[str, list[dict[str, Any]]] = {"unknown": [], "overlength": []}
        self.previous: dict[str, Any] | None = None
        self.previous_key: tuple[str, str] | None = None
        self.first_key: tuple[str, str] | None = None
        self.last_kept_id: str | None = None

    def observe(self, row: dict[str, Any]) -> list[int] | None:
        identifier, canonical, labels = row["molecule_id"], row["canonical_smiles"], row["groups"]
        index = row["augmentation_index"]
        require_count(index, "augmentation_index")
        if not isinstance(labels, list):
            raise ValueError(f"Corpus groups must be a list: {identifier}")
        expected_labels = [group for group in GROUPS if group in labels]
        if (
            row["split"] != self.split or identifier != molecule_id(canonical)
            or labels != expected_labels
            or (self.context.config["data"]["filters"]["require_membership"] and not labels)
            or index > self.context.config["data"]["augmentation"]["random_variants"]
            or (self.split != "train" and index != 0)
            or (index == 0 and row["smiles"] != canonical)
        ):
            raise ValueError(f"Malformed or inconsistent {self.split} corpus row: {identifier}")
        same_graph = self.previous is not None and identifier == self.previous["molecule_id"]
        if same_graph:
            if (
                canonical != self.previous["canonical_smiles"] or labels != self.previous["groups"]
                or index != self.previous["augmentation_index"] + 1
            ):
                raise ValueError("Augmentation rows must keep graph labels and consecutive indices")
        else:
            seed = self.context.config["data"]["split"]["seed"]
            key = (hashlib.sha256(f"{seed}\0{identifier}".encode("ascii")).hexdigest(), identifier)
            if index != 0 or (self.previous_key is not None and key <= self.previous_key):
                raise ValueError("Corpus graphs must be unique and in the declared seeded SHA-256 order")
            self.first_key = key if self.first_key is None else self.first_key
            self.previous_key = key
            self.counts["input_molecules"] += 1
        self.previous = row
        lexemes = tokenize_smiles(row["smiles"])  # T02 guarantees supported lexical input.
        length = len(lexemes) + 1  # EOS is part of every measured sequence length.
        self.lengths[length] += 1
        self.counts["input_rows"] += 1
        self.groups["input"].update(labels)
        unknown = [token for token in lexemes if token not in self.tokenizer.token_to_id]
        too_long = length > self.tokenizer.max_length
        self.counts["length_over_limit_rows"] += int(too_long)
        # Match SmilesTokenizer.encode_text: unknown takes precedence over length.
        if unknown:
            reason = "unknown"
            self.unknown_tokens.update(unknown)
            self.unknown_token_rows.update(set(unknown))
            self.counts["unknown_and_overlength_rows"] += int(too_long)
        elif too_long:
            reason = "overlength"
        else:
            self.counts["rows"] += 1
            self.groups["kept"].update(labels)
            if identifier != self.last_kept_id:
                self.counts["kept_molecules"] += 1
                self.last_kept_id = identifier
            return self.tokenizer.encode_text(row["smiles"])
        self.counts[reason + "_rows"] += 1
        self.groups[reason].update(labels)
        if len(self.examples[reason]) < EXAMPLE_LIMIT:
            self.examples[reason].append({
                "molecule_id": identifier, "augmentation_index": index, "smiles": row["smiles"],
                "length_with_eos": length, "unknown_tokens": sorted(set(unknown)),
            })
        return None

    def report(self, expected: Mapping[str, Any]) -> dict[str, Any]:
        names = (
            "input_rows", "input_molecules", "rows", "kept_molecules", "unknown_rows", "overlength_rows",
            "length_over_limit_rows", "unknown_and_overlength_rows",
        )
        counts = {name: self.counts[name] for name in names}
        if counts["input_rows"] != expected["rows"] or counts["input_molecules"] != expected["molecules"]:
            raise ValueError("Corpus row/graph counts disagree with its manifest")
        group_counts = {reason: {group: counter[group] for group in GROUPS} for reason, counter in self.groups.items()}
        if group_counts["input"] != expected["group_rows"]:
            raise ValueError("Corpus membership counts disagree with its manifest")
        if counts["input_rows"] != counts["rows"] + counts["unknown_rows"] + counts["overlength_rows"]:
            raise ValueError("Tokenization accounting does not cover every input row")
        total = counts["input_rows"]
        return {
            **counts, "group_rows": group_counts,
            "fractions": {name: counts[name] / total if total else None for name in (
                "rows", "unknown_rows", "overlength_rows", "length_over_limit_rows",
            )},
            "group_fractions": {
                group: {reason: counter[group] / self.groups["input"][group] if self.groups["input"][group] else None
                        for reason, counter in self.groups.items() if reason != "input"}
                for group in GROUPS
            },
            "lengths_with_eos": _length_summary(self.lengths),
            "unknown_token_occurrences": dict(sorted(self.unknown_tokens.items())),
            "unknown_token_rows": dict(sorted(self.unknown_token_rows.items())),
            "rejection_examples": self.examples,
            "first_graph_key": list(self.first_key) if self.first_key is not None else None,
            "last_graph_key": list(self.previous_key) if self.previous_key is not None else None,
        }


def _write_shard(
    context: TaskContext, split: str, tokenizer: SmilesTokenizer, corpus_hash: str,
    number: int, rows: list[dict[str, Any]], encoded: list[list[int]],
) -> tuple[str, str, dict[str, Any]]:
    import torch

    count, width = len(rows), tokenizer.max_length
    ids = torch.full((count, width), tokenizer.special_tokens.pad_id, dtype=torch.int64, device="cpu")
    attention = torch.zeros((count, width), dtype=torch.bool, device="cpu")
    lengths = torch.tensor([len(tokens) for tokens in encoded], dtype=torch.int32, device="cpu")
    for index, tokens in enumerate(encoded):
        ids[index, :len(tokens)] = torch.tensor(tokens, dtype=torch.int64, device="cpu")
        attention[index, :len(tokens)] = True
    group_mask = torch.tensor([[group in row["groups"] for group in GROUPS] for row in rows], dtype=torch.bool, device="cpu")
    group_mask = group_mask.reshape(count, len(GROUPS))
    payload = {
        "version": SHARD_VERSION, "split": split, "tokenizer_fingerprint": tokenizer.fingerprint,
        "corpus_sha256": corpus_hash, "group_order": list(GROUPS), "special_tokens": tokenizer.special_tokens.to_dict(),
        "token_ids": ids, "attention_mask": attention, "lengths": lengths, "group_mask": group_mask,
        "molecule_ids": [row["molecule_id"] for row in rows],
        "row_ids": [f"{row['molecule_id']}-{row['augmentation_index']}" for row in rows],
        "augmentation_indices": torch.tensor([row["augmentation_index"] for row in rows], dtype=torch.int32, device="cpu"),
        "canonical_smiles": [row["canonical_smiles"] for row in rows], "smiles": [row["smiles"] for row in rows],
    }
    final_path = context.store.directory("data", "tokenized", split, f"shard_{number:05d}.pt")
    staged = _stage_shard(context, split, final_path)
    target = context.store.path(staged)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    torch.save(payload, context.store.path(staged + ".part"))
    commit_file(context.store.path(staged + ".part"), target)
    digest = sha256_file(context.store.require_file(staged))
    return final_path, digest, {"path": final_path, "rows": count, "sha256": digest}


def _build_split(
    context: TaskContext, split: str, tokenizer: SmilesTokenizer,
    corpus: ArtifactManifest, corpus_hash: str, fingerprint: str, state: dict[str, Any],
) -> tuple[str, ArtifactManifest]:
    source = context.store.directory("data", "corpus", split + ".parquet")
    expected = corpus.metadata["splits"][split]
    audit = _SplitAudit(context, split, tokenizer)
    files: dict[str, str] = {}
    shard_list = []
    rows: list[dict[str, Any]] = []
    encoded: list[list[int]] = []
    shard_rows = context.config["data"]["tokenizer"]["shard_rows"]

    def flush() -> None:
        with context.tracker.measure("shard_write"):
            path, digest, detail = _write_shard(context, split, tokenizer, corpus_hash, len(files), rows, encoded)
        files[path] = digest
        shard_list.append(detail)
        rows.clear()
        encoded.clear()

    with closing(parquet_rows(context.store, source)) as incoming:
        with context.progress(total=expected["rows"], desc=f"Tokenize {split}", unit="row") as progress:
            while True:
                with context.tracker.measure("tokenize"):
                    chunk = list(islice(incoming, IO_ROWS))
                    if not chunk:
                        break
                    for row in chunk:
                        tokens = audit.observe(row)
                        if tokens is not None:
                            rows.append(row)
                            encoded.append(tokens)
                            if len(rows) == shard_rows:
                                flush()
                progress.update(len(chunk))
                # Retain the cost of completed batches. An unfinished split
                # restarts from its source; only complete splits are reused.
                context.save_checkpoint(state)
    if rows or not files:
        flush()  # An empty split has an explicit empty tensor shard, never a fake example.
    metadata = {
        **audit.report(expected), "version": SHARD_VERSION, "split": split,
        "tokenizer_fingerprint": tokenizer.fingerprint, "tokenization_fingerprint": fingerprint,
        "corpus_sha256": corpus_hash, "source_sha256": corpus.files[source], "source_path": source,
        "max_length": tokenizer.max_length, "vocab_size": tokenizer.vocab_size, "group_order": list(GROUPS),
        "shard_rows": shard_rows, "shards": shard_list,
        "group_label_definition": "Multi-label original SMARTS membership; not terminal tau labels.",
        "row_unit": "One retained SMILES row; overlapping labels do not duplicate examples.",
        "rejection_order": "Unknown first, then length; length_over_limit_rows counts long rows including unknowns.",
        "tensor_schema": {
            "token_ids": "int64 [rows,max_length]; molecule tokens, EOS, right PAD",
            "attention_mask": "bool [rows,max_length]; true through EOS, false for PAD",
            "lengths": "int32 [rows]; includes EOS",
            "group_mask": "bool [rows,3]; columns in group_order",
            "augmentation_indices": "int32 [rows]; zero is canonical",
            "strings": "molecule_ids, row_ids, canonical_smiles, smiles: lists aligned with tensor rows",
        },
    }
    manifest = ArtifactManifest(kind="tokenized", config_hash=context.identity.config_hash, files=files, metadata=metadata)
    staged = _stage_path(context, split, "manifest.json")
    context.store.write_json(staged, manifest.to_dict())
    context.log(
        f"{split}: kept={metadata['rows']}/{metadata['input_rows']}; "
        f"unknown={metadata['unknown_rows']}; overlength={metadata['overlength_rows']}; shards={len(files)}",
    )
    if not metadata["rows"] or any(not metadata["group_rows"]["kept"][group] for group in GROUPS):
        context.log(f"{split}: empty retained rows or missing group coverage; inspect the length report", level="WARNING")
    return staged, manifest


def _resume_split(
    context: TaskContext, split: str, saved: Mapping[str, Any], tokenizer: SmilesTokenizer,
    corpus_hash: str, fingerprint: str,
) -> ArtifactManifest:
    staged = _stage_path(context, split, "manifest.json")
    context.store.require_file(staged, checksum=saved["manifest_sha256"])
    manifest = ArtifactManifest.from_dict(context.store.read_json(staged))
    if (
        manifest.kind != "tokenized" or manifest.config_hash != context.identity.config_hash
        or manifest.metadata.get("split") != split or manifest.metadata.get("corpus_sha256") != corpus_hash
        or manifest.metadata.get("tokenizer_fingerprint") != tokenizer.fingerprint
        or manifest.metadata.get("tokenization_fingerprint") != fingerprint
    ):
        raise ValueError("Resumed token shards disagree with the current corpus/tokenizer")
    for path, digest in manifest.files.items():
        private = _stage_shard(context, split, path)
        location = private if os.path.isfile(context.store.path(private)) else path
        context.store.require_file(location, checksum=digest)
    context.log(f"resumed checksum-verified {split} shards")
    return manifest


def _vocabulary(
    context: TaskContext, corpus: ArtifactManifest, corpus_hash: str, state: dict[str, Any],
) -> tuple[SmilesTokenizer, dict[str, Any]]:
    options = context.config["data"]["tokenizer"]
    vocabulary_path = os.path.join(context.output_path("tokenizer-stage"), "vocab.json")
    metadata_path = os.path.join(context.output_path("tokenizer-stage"), "metadata.json")
    if state.get("vocabulary_metadata_sha256"):
        context.store.require_file(metadata_path, checksum=state["vocabulary_metadata_sha256"])
        tokenizer = SmilesTokenizer.load(context.store, metadata_path)
        metadata = context.store.read_json(metadata_path)
        context.log("resumed train-only vocabulary")
    else:
        train_path = context.store.directory("data", "corpus", "train.parquet")
        with context.tracker.measure("vocabulary"):
            with closing(parquet_rows(context.store, train_path, columns=["smiles"])) as incoming:
                with context.progress(incoming, total=corpus.metadata["splits"]["train"]["rows"], desc="Train vocabulary", unit="row") as progress:
                    tokenizer = SmilesTokenizer.build((row["smiles"] for row in progress), max_length=options["max_length"])
            metadata = tokenizer.save(
                context.store, vocabulary_path=vocabulary_path, metadata_path=metadata_path,
                provenance={
                    "corpus_sha256": corpus_hash, "training_split_sha256": corpus.files[train_path],
                    "training_split": "train", "training_rows": corpus.metadata["splits"]["train"]["rows"],
                    "vocabulary_includes_overlength_train_rows": True,
                },
            )
        state["vocabulary_metadata_sha256"] = sha256_file(context.store.require_file(metadata_path))
        state["stage"] = "splits"
        context.save_checkpoint(state)
    train_path = context.store.directory("data", "corpus", "train.parquet")
    if (
        tokenizer.max_length != options["max_length"] or metadata["provenance"]["corpus_sha256"] != corpus_hash
        or metadata["provenance"]["training_split_sha256"] != corpus.files[train_path]
    ):
        raise ValueError("Resumed vocabulary has different settings or training data")
    return tokenizer, metadata


def _reuse(context: TaskContext, corpus: ArtifactManifest, corpus_hash: str, fingerprint: str) -> TaskResult | None:
    options = context.config["data"]["tokenizer"]
    path = options["metadata_path"]
    if not os.path.isfile(context.store.path(path)):
        return None
    metadata = context.store.read_json(path)
    if metadata.get("tokenization_fingerprint") != fingerprint:
        return None
    tokenizer = SmilesTokenizer.load(context.store, path)
    train_path = context.store.directory("data", "corpus", "train.parquet")
    if (
        tokenizer.max_length != options["max_length"] or metadata["provenance"]["corpus_sha256"] != corpus_hash
        or metadata["provenance"]["training_split_sha256"] != corpus.files[train_path]
        or project_path(metadata["vocabulary_path"]) != project_path(options["vocabulary_path"])
        or project_path(metadata["length_report_path"]) != project_path(options["length_report_path"])
    ):
        raise ValueError("Published tokenizer has inconsistent provenance")
    context.store.require_file(options["length_report_path"], checksum=metadata["length_report_sha256"])
    manifests = {}
    for split in SPLITS:
        manifest_path = _manifest_path(context.store, split)
        if metadata["tokenized_manifests"][split]["path"] != manifest_path:
            raise ValueError("Published tokenizer points to an unexpected shard manifest")
        context.store.require_file(manifest_path, checksum=metadata["tokenized_manifests"][split]["sha256"])
        manifests[split] = context.store.require_manifest(manifest_path, kind="tokenized", metadata={
            "split": split, "corpus_sha256": corpus_hash, "tokenizer_fingerprint": tokenizer.fingerprint,
            "tokenization_fingerprint": fingerprint,
        })
    context.log("using verified tokenizer and shards with matching training source and settings")
    return _result(context, tokenizer, manifests)


def _result(context: TaskContext, tokenizer: SmilesTokenizer, manifests: Mapping[str, ArtifactManifest]) -> TaskResult:
    options = context.config["data"]["tokenizer"]
    artifacts = [options[name] for name in ("vocabulary_path", "metadata_path", "length_report_path")]
    for split in SPLITS:
        artifacts.extend(manifests[split].files)
        artifacts.append(_manifest_path(context.store, split))
    sizes = "/".join(str(manifests[split].metadata["rows"]) for split in SPLITS)
    return TaskResult(
        f"vocab={tokenizer.vocab_size}; kept train/validation/test={sizes}; report={options['length_report_path']}",
        artifacts=tuple(artifacts),
    )


def build_tokenizer(context: TaskContext) -> TaskResult:
    options = context.config["data"]["tokenizer"]
    corpus_path = context.store.directory("data", "corpus", "manifest.json")
    corpus_hash = context.identity.dependency_hashes["corpus"]
    fingerprint = _fingerprint(context.config, corpus_hash)
    with corpus_lock(context.store, execution=context.config["runtime"]["execution"]):
        _check_sources(context)
        corpus = context.store.require_manifest(corpus_path, kind="corpus")
        for split in SPLITS:
            source = context.store.directory("data", "corpus", split + ".parquet")
            if source not in corpus.files or corpus.files[source] != corpus.metadata["splits"][split]["sha256"]:
                raise ValueError(f"Corpus manifest does not bind its {split} partition")
        # Tokenizer JSON files must never overwrite corpus inputs or shards.
        forbidden = {project_path(path) for path in (
            corpus_path, *corpus.files, benchmark_dependency(context.config, context.store).path,
        )}
        shard_parent = context.store.directory("data", "tokenized")
        output_paths = [project_path(options[name]) for name in ("vocabulary_path", "metadata_path", "length_report_path")]
        if len(set(output_paths)) != len(output_paths):
            raise ValueError("Tokenizer output paths must refer to three distinct files")
        for name in ("vocabulary_path", "metadata_path", "length_report_path"):
            path = project_path(options[name])
            if path in forbidden or path == shard_parent or path.startswith(shard_parent + os.sep):
                raise ValueError(f"Tokenizer {name} collides with corpus or shard artifacts")
        reused = _reuse(context, corpus, corpus_hash, fingerprint)
        if reused is not None:
            return reused
        state = dict(context.resume_state or {"stage": "vocabulary", "completed_splits": {}})
        tokenizer, metadata = _vocabulary(context, corpus, corpus_hash, state)
        manifests = {}
        for split in SPLITS:
            saved = state["completed_splits"].get(split)
            if saved is not None:
                manifests[split] = _resume_split(context, split, saved, tokenizer, corpus_hash, fingerprint)
            else:
                staged, manifests[split] = _build_split(context, split, tokenizer, corpus, corpus_hash, fingerprint, state)
                state["completed_splits"][split] = {"manifest_sha256": sha256_file(context.store.require_file(staged))}
                context.save_checkpoint(state)
        for left, right in zip(SPLITS, SPLITS[1:]):
            end = manifests[left].metadata["last_graph_key"]
            start = manifests[right].metadata["first_graph_key"]
            if end is not None and start is not None and end >= start:
                raise ValueError("Corpus graph ordering overlaps split boundaries")
        _check_sources(context)
        state["stage"] = "publish"
        context.save_checkpoint(state)
        # Invalidate the previous completion marker before changing any outputs.
        if os.path.exists(context.store.path(options["metadata_path"])):
            os.remove(context.store.path(options["metadata_path"]))
        vocabulary = context.store.read_json(metadata["vocabulary_path"])
        context.store.write_json(options["vocabulary_path"], vocabulary)
        manifest_refs = {}
        with context.tracker.measure("publish"):
            with context.progress(total=sum(len(manifest.files) for manifest in manifests.values()), desc="Publish shards", unit="shard") as progress:
                for split in SPLITS:
                    manifest = manifests[split]
                    for path, digest in manifest.files.items():
                        private = _stage_shard(context, split, path)
                        if os.path.isfile(context.store.path(private)):
                            context.store.require_file(private, checksum=digest)
                            target = context.store.path(path)
                            os.makedirs(os.path.dirname(target), exist_ok=True)
                            commit_file(context.store.path(private), target)
                        else:
                            context.store.require_file(path, checksum=digest)
                        progress.update(1)
                    path = _manifest_path(context.store, split)
                    context.store.write_json(path, manifest.to_dict())
                    manifest_refs[split] = {"path": path, "sha256": sha256_file(context.store.require_file(path))}
        context.store.write_json(options["length_report_path"], {
            "version": SHARD_VERSION, "corpus_sha256": corpus_hash, "tokenizer_fingerprint": tokenizer.fingerprint,
            "tokenization_fingerprint": fingerprint, "max_length": tokenizer.max_length,
            "length_definition": "Lexical SMILES token count plus EOS, before any exclusions.",
            "rejection_order": "Unknown before overlength; length_over_limit_rows also includes unknown rows.",
            "test_policy": "Test is audited with frozen train vocabulary and configured length; it never changes either.",
            "splits": {split: manifests[split].metadata for split in SPLITS},
        })
        metadata.update({
            "vocabulary_path": options["vocabulary_path"],
            "vocabulary_sha256": sha256_file(context.store.require_file(options["vocabulary_path"])),
            "tokenization_fingerprint": fingerprint, "shard_version": SHARD_VERSION,
            "length_report_path": options["length_report_path"],
            "length_report_sha256": sha256_file(context.store.require_file(options["length_report_path"])),
            "tokenized_manifests": manifest_refs, "group_order": list(GROUPS),
            "coverage": {split: manifests[split].metadata["fractions"] for split in SPLITS},
            "paper_relation": "New shared MDLM tokenizer; not CARS's Llama tokenizer or MDM-VGB's QM9 tokenizer.",
        })
        context.store.write_json(options["metadata_path"], metadata)
        state.update({"stage": "complete", "metadata_sha256": sha256_file(context.store.require_file(options["metadata_path"]))})
        context.save_checkpoint(state)
        return _result(context, tokenizer, manifests)


def execute(argv: Sequence[str] | None = None, *, execution: str = "local") -> RunRecord:
    return run_task(
        "T03", build_tokenizer, description="Build the train-only tokenizer and SMILES tensor shards.",
        argv=argv, default_config="configs/data.yaml", execution=execution, supports_seed=False,
        validator=validate_preparation, dependencies=_dependencies,
    )


if __name__ == "__main__":
    try:
        record = execute()
    except KeyboardInterrupt:
        tqdm.write("T03: interrupted")
        raise SystemExit(130) from None
    except Exception:
        raise SystemExit(1) from None
    raise SystemExit(0 if record.status == "completed" else 1)
