"""T02: prepare the pinned ChEMBL corpus for the shared SMILES MDLM.

This is new training data, distinct from CARS exemplars and MDM-VGB's QM9.
Run from the repository root with ``python src/tasks/prepare_corpus.py``.
SQLite deduplicates graphs on disk; Parquet is written in bounded batches.
Completed stages/splits can resume; an unfinished import or split restarts.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.metadata
import math
import os
import sqlite3
import urllib.request
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from itertools import islice
from typing import Any

from tqdm.auto import tqdm

from src.artifacts import ArtifactStore, Dependency, sha256_file
from src.config import Config, config_digest
from src.data import IDENTITY_VERSION, SPLITS, MoleculeRecord, augment_molecule, prepare_molecules
from src.reward import RewardChecker
from src.schemas import GROUPS, RunRecord, TaskResult
from src.tasks.common import TaskContext, run_task
from src.tasks.prepare_benchmark import read_benchmark_assets, validate_data
from src.tokenizer import TOKEN_PATTERN, TOKENIZER_VERSION, TokenizationError, tokenize_smiles

CORPUS_VERSION = "chembl-smiles-corpus-v1"
IO_ROWS = 10000


def validate_preparation(config: Config) -> None:
    validate_data(config)
    if config.get("group") is not None:
        raise ValueError("Data preparation processes all three groups; leave group=null")


def checker_versions() -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in ("rdkit", "llguidance")}
    if versions != {"rdkit": "2025.3.6", "llguidance": "0.7.30"}:
        raise RuntimeError(f"Data preparation requires the project-pinned checker versions: {versions}")
    return versions


def benchmark_dependency(config: Config, store: ArtifactStore) -> Dependency:
    read_benchmark_assets(config)
    return Dependency(
        store.directory("data", "benchmark", config["data"]["benchmark"]["manifest_file"]),
        manifest=True, kind="benchmark", metadata={
            "source_sha256": config["data"]["benchmark"]["source_sha256"], "identity_version": IDENTITY_VERSION,
        },
    )


def preparation_fingerprint(config: Config, benchmark_sha256: str, versions: Mapping[str, str]) -> str:
    data = config["data"]
    return config_digest({
        "version": CORPUS_VERSION, "benchmark_sha256": benchmark_sha256, "library_versions": dict(versions),
        "data": {key: data[key] for key in ("corpus", "normalization", "filters", "split", "augmentation")},
        "groups": list(GROUPS), "tokenizer_version": TOKENIZER_VERSION, "lexical_pattern": TOKEN_PATTERN,
    })


@contextmanager
def corpus_lock(store: ArtifactStore, *, execution: str = "local") -> Iterator[None]:
    """Serialize local T02/T03 processes; Modal tasks run in workflow order.

    A POSIX lock on a container's Volume snapshot does not coordinate separate
    Modal containers. The remote entrypoints must finish T02 before starting T03.
    """
    if execution == "modal":
        yield
        return
    if execution != "local":
        raise ValueError(f"Unknown execution environment: {execution}")
    import fcntl

    path = store.path(os.path.join(store.directories["data"], ".corpus.lock"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another T02/T03 job is using this artifact root") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def commit_file(temporary: str, target: str) -> None:
    """Both paths are validated by their ArtifactStore before calling this helper."""
    with open(temporary, "r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    descriptor = os.open(os.path.dirname(target) or ".", os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parquet_rows(store: ArtifactStore, relative: str, *, columns: Sequence[str] | None = None) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    with pq.ParquetFile(store.require_file(relative)) as parquet:
        for batch in parquet.iter_batches(batch_size=IO_ROWS, columns=columns):
            yield from batch.to_pylist()


def _dependencies(config: Config, store: ArtifactStore) -> dict[str, Dependency]:
    return {"benchmark": benchmark_dependency(config, store)}


def _download(context: TaskContext) -> str:
    source = context.config["data"]["corpus"]
    relative = source["archive"]
    target = context.store.path(relative)
    if os.path.isfile(target):
        with context.tracker.measure("source_integrity"):
            context.store.require_file(relative, checksum=source["sha256"])
        context.log("using checksum-verified ChEMBL 36 archive")
        return relative
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temporary = context.store.path(context.output_path("chembl-download.part"))
    digest = hashlib.sha256()
    request = urllib.request.Request(source["url"], headers={"User-Agent": "smiles-cars-mdm-vgb/0.1.0", "Accept-Encoding": "identity"})
    with context.tracker.measure("download"):
        with urllib.request.urlopen(request, timeout=60) as response, open(temporary, "wb") as handle:
            length = response.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None
            with context.progress(total=total, desc="Download ChEMBL 36", unit="B", unit_scale=True) as progress:
                while block := response.read(1024 * 1024):
                    handle.write(block)
                    digest.update(block)
                    progress.update(len(block))
        if digest.hexdigest() != source["sha256"]:
            raise ValueError("Downloaded ChEMBL archive fails its pinned SHA-256; it was not published")
        commit_file(temporary, target)
    context.log(f"downloaded and verified ChEMBL 36: {relative}")
    return relative


def _source_rows(context: TaskContext, archive: str) -> Iterator[dict[str, str]]:
    source = context.config["data"]["corpus"]
    # Large molecules can exceed csv's default field limit; this is restored.
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(16 * 1024 * 1024)
    try:
        with gzip.open(context.store.require_file(archive), "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            headers = reader.fieldnames or []
            required = {source["id_column"], source["smiles_column"]}
            if len(set(headers)) != len(headers) or not required <= set(headers):
                raise ValueError("ChEMBL TSV header differs from its declared representation columns")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed ChEMBL TSV record at line {reader.line_num}")
                identifier = row[source["id_column"]]
                if not identifier or identifier != identifier.strip():
                    raise ValueError(f"Invalid ChEMBL identifier at line {reader.line_num}")
                yield {"source_id": identifier, "smiles": row[source["smiles_column"]]}
    finally:
        csv.field_size_limit(previous_limit)


def _import_molecules(context: TaskContext, archive: str, database: str, state: dict[str, Any]) -> dict[str, int]:
    path = context.store.path(database)
    # Incomplete imports are private scratch work, never corpus artifacts.
    if os.path.exists(path):
        for suffix in ("", "-journal", "-wal", "-shm"):
            scratch = context.store.path(database + suffix)
            if os.path.exists(scratch):
                os.remove(scratch)
        context.log("restarting unfinished molecule import")
    counts: Counter[str] = Counter()
    _, examples = read_benchmark_assets(context.config)
    excluded = [smiles for group in GROUPS for smiles in examples[group]]
    checker = RewardChecker(asset_directory=context.config["paths"]["benchmark"])
    filters = context.config["data"]["filters"]
    split_seed = context.config["data"]["split"]["seed"]
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript("""
            CREATE TABLE molecules (
                molecule_id TEXT PRIMARY KEY, canonical_smiles TEXT UNIQUE NOT NULL,
                sort_key BLOB NOT NULL, group_mask INTEGER NOT NULL
            );
            CREATE INDEX molecule_order ON molecules(sort_key, molecule_id);
            CREATE TABLE sources (
                source_id TEXT PRIMARY KEY, molecule_id TEXT NOT NULL REFERENCES molecules(molecule_id)
            );
            CREATE INDEX sources_by_molecule ON sources(molecule_id);
        """)
        with (
            closing(_source_rows(context, archive)) as incoming,
            context.progress(desc="Normalize and deduplicate", unit="row") as progress,
        ):
            while True:
                with context.tracker.measure("preprocess"):
                    chunk = list(islice(incoming, IO_ROWS))
                    if not chunk:
                        break
                    counts["source_rows"] += len(chunk)
                    supported = []
                    for row in chunk:
                        try:
                            tokenize_smiles(row["smiles"])
                        except TokenizationError:
                            counts["unsupported_smiles_rows"] += 1
                        else:
                            supported.append(row)
                    records, batch_counts = prepare_molecules(
                        supported, checker.classify_molecule, exclude_smiles=excluded, **filters,
                    )
                    counts.update({key: value for key, value in batch_counts.items() if key != "unique_molecules"})
                    for record in records:
                        tokenize_smiles(record.canonical_smiles)  # No hidden cleanup of normalized strings.
                        group_mask = sum(1 << index for index, group in enumerate(GROUPS) if group in record.groups)
                        sort_key = hashlib.sha256(f"{split_seed}\0{record.molecule_id}".encode("ascii")).digest()
                        cursor = connection.execute(
                            "INSERT INTO molecules VALUES (?, ?, ?, ?) ON CONFLICT(molecule_id) DO NOTHING",
                            (record.molecule_id, record.canonical_smiles, sort_key, group_mask),
                        )
                        if cursor.rowcount == 0:
                            previous = connection.execute(
                                "SELECT canonical_smiles, group_mask FROM molecules WHERE molecule_id=?", (record.molecule_id,),
                            ).fetchone()
                            if previous != (record.canonical_smiles, group_mask):
                                raise ValueError("Conflicting molecular identity or membership across source rows")
                            counts["duplicate_rows"] += 1
                        for identifier in record.source_ids:
                            inserted = connection.execute(
                                "INSERT INTO sources VALUES (?, ?) ON CONFLICT(source_id) DO NOTHING", (identifier, record.molecule_id),
                            )
                            if inserted.rowcount == 0 and connection.execute(
                                "SELECT molecule_id FROM sources WHERE source_id=?", (identifier,),
                            ).fetchone()[0] != record.molecule_id:
                                raise ValueError(f"ChEMBL ID has conflicting molecular identities: {identifier}")
                    connection.commit()
                progress.update(len(chunk))
                # Commit cost/log state. The unfinished import itself restarts
                # on resume; previously incurred preprocessing cost is retained.
                context.save_checkpoint(state)
        counts["unique_molecules"] = connection.execute("SELECT COUNT(*) FROM molecules").fetchone()[0]
        if not counts["unique_molecules"]:
            raise ValueError("No molecules survived the declared corpus filters")
    finally:
        connection.close()
    return dict(counts)


def _corpus_schema():
    import pyarrow as pa

    return pa.schema([
        ("canonical_smiles", pa.string()), ("molecule_id", pa.string()), ("groups", pa.list_(pa.string())),
        ("source_ids", pa.list_(pa.string())), ("smiles", pa.string()), ("split", pa.string()), ("augmentation_index", pa.int32()),
    ])


def _write_split(context: TaskContext, database: str, split: str, offset: int, size: int, relative: str) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    temporary = context.store.path(context.output_path(split + ".parquet.part"))
    target = context.store.path(relative)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    options = context.config["data"]["augmentation"]
    groups: Counter[str] = Counter()
    row_groups: Counter[str] = Counter()
    overlaps: Counter[str] = Counter()
    schema = _corpus_schema()
    buffer = []
    rows = 0
    visited = 0
    connection = sqlite3.connect(context.store.require_file(database))
    try:
        cursor = connection.execute(
            "SELECT molecule_id, canonical_smiles, group_mask FROM molecules ORDER BY sort_key, molecule_id LIMIT ? OFFSET ?",
            (size, offset),
        )
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            with context.progress(total=size, desc=f"Write {split}", unit="molecule") as progress:
                while True:
                    chunk = cursor.fetchmany(512)
                    if not chunk:
                        break
                    with context.tracker.measure("augmentation"):
                        for identity, canonical, group_mask in chunk:
                            labels = tuple(group for index, group in enumerate(GROUPS) if group_mask & (1 << index))
                            sources = tuple(row[0] for row in connection.execute(
                                "SELECT source_id FROM sources WHERE molecule_id=? ORDER BY source_id", (identity,),
                            ))
                            record = MoleculeRecord(canonical, labels, sources)
                            if record.molecule_id != identity:
                                raise ValueError("Scratch database contains an inconsistent molecule ID")
                            variants = augment_molecule(
                                record, split=split, seed=options["seed"], random_variants=options["random_variants"],
                                max_attempts_per_variant=options["max_attempts_per_variant"],
                            )
                            for row in variants:
                                tokenize_smiles(row["smiles"])
                            groups.update(labels)
                            row_groups.update({group: len(variants) for group in labels})
                            overlaps["&".join(labels) or "unclassified"] += 1
                            buffer.extend(variants)
                            rows += len(variants)
                            visited += 1
                            if len(buffer) >= IO_ROWS:
                                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                                buffer.clear()
                    progress.update(len(chunk))
                if buffer:
                    with context.tracker.measure("parquet_write"):
                        writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
        if visited != size:
            raise ValueError("Molecule partition contains fewer records than its split definition")
        commit_file(temporary, target)
    finally:
        connection.close()
    return {
        "molecules": visited, "rows": rows, "augmentation_rows": rows - visited,
        "group_molecules": {group: groups[group] for group in GROUPS},
        "group_rows": {group: row_groups[group] for group in GROUPS}, "membership_combinations": dict(overlaps),
        "sha256": sha256_file(context.store.require_file(relative)),
    }


def _summary(metadata: Mapping[str, Any], manifest_path: str) -> str:
    sizes = "/".join(str(metadata["splits"][split]["molecules"]) for split in SPLITS)
    return f"{metadata['filter_counts']['unique_molecules']} molecules; train/validation/test={sizes}; manifest={manifest_path}"


def prepare_corpus(context: TaskContext) -> TaskResult:
    versions = checker_versions()
    benchmark_hash = context.identity.dependency_hashes["benchmark"]
    fingerprint = preparation_fingerprint(context.config, benchmark_hash, versions)
    manifest_path = context.store.directory("data", "corpus", "manifest.json")
    paths = {split: context.store.directory("data", "corpus", split + ".parquet") for split in SPLITS}
    archive = context.config["data"]["corpus"]["archive"]
    if archive in {manifest_path, *paths.values()}:
        raise ValueError("Source archive and corpus output paths must be distinct")
    artifacts = (archive, *paths.values(), manifest_path)
    with corpus_lock(context.store, execution=context.config["runtime"]["execution"]):
        # Recheck the dependency after acquiring the lock, before using its hash.
        dependency = benchmark_dependency(context.config, context.store)
        context.store.require_manifest(dependency.path, kind="benchmark", metadata=dependency.metadata)
        if sha256_file(context.store.require_file(dependency.path)) != benchmark_hash:
            raise ValueError("Benchmark changed after task preflight")
        if os.path.isfile(context.store.path(manifest_path)):
            existing = context.store.require_manifest(manifest_path, kind="corpus")
            if existing.metadata.get("preparation_fingerprint") == fingerprint:
                context.log("using verified corpus with matching source and preprocessing")
                return TaskResult(_summary(existing.metadata, manifest_path), artifacts=artifacts)
        state = dict(context.resume_state or {})
        _download(context)
        database = context.output_path("corpus.sqlite")
        if state.get("database_sha256"):
            context.store.require_file(database, checksum=state["database_sha256"])
            counts = state["filter_counts"]
            context.log("resuming after completed molecule import")
        else:
            state = {"stage": "import", "completed_splits": {}}
            context.save_checkpoint(state)
            counts = _import_molecules(context, archive, database, state)
            state.update({
                "stage": "splits", "database_sha256": sha256_file(context.store.require_file(database)), "filter_counts": counts,
            })
            context.save_checkpoint(state)
        fractions = context.config["data"]["split"]["fractions"]
        train_size = math.floor(counts["unique_molecules"] * fractions[0])
        validation_size = math.floor(counts["unique_molecules"] * fractions[1])
        sizes = (train_size, validation_size, counts["unique_molecules"] - train_size - validation_size)
        if min(sizes) < 1:
            raise ValueError("The corpus is too small for nonempty train/validation/test partitions")
        # Remove the previous aggregate manifest before replacing any split.
        # Readers sharing this lock cannot observe a partially updated corpus.
        if os.path.exists(context.store.path(manifest_path)):
            os.remove(context.store.path(manifest_path))
        offset = 0
        for split, size in zip(SPLITS, sizes, strict=True):
            previous = state["completed_splits"].get(split)
            if previous is not None:
                context.store.require_file(paths[split], checksum=previous["sha256"])
                if previous["molecules"] != size:
                    raise ValueError("Resumed split has a different molecule count")
                context.log(f"resumed committed {split} partition")
            else:
                state["completed_splits"][split] = _write_split(context, database, split, offset, size, paths[split])
                context.save_checkpoint(state)
            offset += size
        metadata = {
            "version": CORPUS_VERSION, "preparation_fingerprint": fingerprint, "identity_version": IDENTITY_VERSION,
            "benchmark_sha256": benchmark_hash, "benchmark_source_sha256": context.config["data"]["benchmark"]["source_sha256"],
            "source": context.config["data"]["corpus"], "library_versions": versions,
            "normalization": context.config["data"]["normalization"], "filters": context.config["data"]["filters"],
            "split": context.config["data"]["split"], "augmentation": context.config["data"]["augmentation"],
            "filter_counts": counts, "splits": state["completed_splits"], "group_order": list(GROUPS),
            "group_label_definition": "Original SMARTS membership, allowing overlap; these are not terminal tau labels.",
            "count_units": "source_rows includes lexical rejections; input_rows counts rows passed to chemical filtering; split molecules precede augmentation.",
            "lexical_filter": "Reject unsupported input strings; never remove characters or truncate SMILES.",
            "attribution": "ChEMBL data is from https://www.ebi.ac.uk/chembl; the version of ChEMBL is chembl_36.",
            "license_url": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/LICENSE",
            "paper_relation": "New ChEMBL training corpus; CARS uses prompt exemplars and MDM-VGB uses QM9.",
        }
        context.store.create_manifest(
            manifest_path, kind="corpus", config_hash=context.identity.config_hash, files=[archive, *paths.values()], metadata=metadata,
        )
        summary = _summary(metadata, manifest_path)
        state.update({"stage": "complete", "manifest_sha256": sha256_file(context.store.require_file(manifest_path))})
        context.save_checkpoint(state)
        for split in SPLITS:
            detail = state["completed_splits"][split]
            context.log(f"{split}: {detail['molecules']} molecules, {detail['rows']} rows; groups={detail['group_molecules']}")
        return TaskResult(summary, artifacts=artifacts)


def execute(argv: Sequence[str] | None = None, *, execution: str = "local") -> RunRecord:
    return run_task(
        "T02", prepare_corpus, description="Prepare the pinned ChEMBL SMILES training corpus.",
        argv=argv, default_config="configs/data.yaml", execution=execution, supports_seed=False,
        validator=validate_preparation, dependencies=_dependencies,
    )


if __name__ == "__main__":
    try:
        record = execute()
    except KeyboardInterrupt:
        tqdm.write("T02: interrupted")
        raise SystemExit(130) from None
    except Exception:
        raise SystemExit(1) from None
    raise SystemExit(0 if record.status == "completed" else 1)
