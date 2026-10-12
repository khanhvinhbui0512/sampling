"""T01: verify pinned assets and preserve all 54 examples with checker diagnostics.

Run from the repository root with ``python src/tasks/prepare_benchmark.py``.
Only packaged project assets are read; no corpus, model, or reference folder is
needed. The Modal wrapper calls execute with the same arguments and engine.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
from collections.abc import Sequence
from typing import Any

from tqdm.auto import tqdm

from src.artifacts import ArtifactStore, sha256_file
from src.config import Config, project_path
from src.data import IDENTITY_VERSION, molecule_id, read_smiles, write_parquet
from src.reward import REWARD_VERSIONS, RewardChecker
from src.schemas import GROUPS, RunRecord, TaskResult, require_count, require_identifier, require_sha256
from src.tasks.common import TaskContext, run_task
from src.tokenizer import SPECIAL_NAMES, TOKENIZER_VERSION

EXPECTED_COUNTS = {"acrylates": 32, "chain_extenders": 11, "isocyanates": 11}
BENCHMARK_COMMIT = "f46607f983d89e4ca4f40527cd231ab04adc6ed7"


def validate_data(config: Config) -> None:
    """Validate declared policies; unsupported switches must not silently do nothing."""
    data = config.get("data")
    sections = {"benchmark", "corpus", "normalization", "filters", "split", "augmentation", "tokenizer"}
    if not isinstance(data, dict) or set(data) != sections or any(not isinstance(data[name], dict) for name in sections):
        raise ValueError("data must define benchmark, corpus, normalization, filters, split, augmentation, tokenizer")
    benchmark = data["benchmark"]
    if benchmark.get("commit") != BENCHMARK_COMMIT or benchmark.get("expected_counts") != EXPECTED_COUNTS:
        raise ValueError("Benchmark must retain the pinned CARS revision and 32/11/11 examples")
    if benchmark.get("preserve_original_rows") is not True or benchmark.get("mismatch_policy") != "report":
        raise ValueError("Keep all original examples and report mismatches")
    require_sha256(benchmark["source_sha256"], "benchmark provenance checksum")
    for name in ("source_file", "examples_file", "manifest_file"):
        require_identifier(benchmark[name], name)
    if benchmark["examples_file"] == benchmark["manifest_file"]:
        raise ValueError("Benchmark examples and manifest need separate files")
    normalization = data["normalization"]
    expected_normalization = {
        "identity_version": IDENTITY_VERSION, "sanitize": True, "canonical": True, "isomeric": True,
        "remove_atom_maps": True, "remove_ordinary_explicit_hydrogens": True, "neutralize": False,
        "canonicalize_tautomers": False, "select_largest_fragment": False, "strip_invalid_tokens": False,
        "allow_smiles_names": False, "allow_cxsmiles": False,
    }
    if normalization != expected_normalization:
        raise ValueError("Unsupported normalization; changing molecule identity needs a new implementation/version")
    corpus = data["corpus"]
    if (
        corpus.get("name") != "chembl" or type(corpus.get("version")) is not int or corpus["version"] != 36
        or corpus.get("format") != "gzip_tsv" or corpus.get("license") != "CC-BY-SA-3.0"
        or corpus.get("id_column") != "chembl_id" or corpus.get("smiles_column") != "canonical_smiles"
        or corpus.get("exclude_all_prompt_identities") is not True
        or corpus.get("classifier") != "original_smarts" or corpus.get("multilabel") is not True
    ):
        raise ValueError("Corpus must use ChEMBL 36, original SMARTS, multilabel graphs, and global prompt exclusion")
    base = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/"
    if corpus.get("url") != base + "chembl_36_chemreps.txt.gz" or corpus.get("checksums_url") != base + "checksums.txt":
        raise ValueError("Use the pinned EBI ChEMBL 36 source URLs")
    if corpus.get("sha256") != "f9c2fa730ac5000da9b0c0d4033c99bfd8fe2b99950f19df1d842c6e73dd7cc4":
        raise ValueError("ChEMBL archive checksum differs from the official release")
    project_path(corpus["archive"])
    filters = data["filters"]
    for name in ("require_membership", "allow_disconnected", "allow_dummy_atoms"):
        if type(filters.get(name)) is not bool:
            raise TypeError(f"filters.{name} must be boolean")
    if filters.get("max_heavy_atoms") is not None:
        require_count(filters["max_heavy_atoms"], "max_heavy_atoms", positive=True)
    split = data["split"]
    require_count(split["seed"], "split seed")
    fractions = split.get("fractions")
    if not isinstance(fractions, list) or len(fractions) != 3 or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in fractions
    ) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-12):
        raise ValueError("data.split.fractions must have three positive fractions summing to one")
    if (
        split.get("unit") != "molecule" or split.get("algorithm") != "seeded_sha256_sort"
        or split.get("rounding") != "floor_train_validation_test_remainder" or split.get("stratified") is not False
    ):
        raise ValueError("Split unique graphs by seeded SHA-256 before augmentation")
    augmentation = data["augmentation"]
    if (
        augmentation.get("algorithm") != "atom_permutation" or augmentation.get("train_only") is not True
        or augmentation.get("keep_canonical") is not True or augmentation.get("verify_identity") is not True
    ):
        raise ValueError("Use identity-checked atom permutations only after assigning train graphs")
    require_count(augmentation["seed"], "augmentation seed")
    require_count(augmentation["random_variants"], "random_variants")
    require_count(augmentation["max_attempts_per_variant"], "max_attempts_per_variant", positive=True)
    tokenizer = data["tokenizer"]
    if (
        tokenizer.get("version") != TOKENIZER_VERSION or tokenizer.get("vocabulary_from") != "train"
        or tokenizer.get("special_tokens") != list(SPECIAL_NAMES) or tokenizer.get("include_eos_in_length") is not True
        or tokenizer.get("unknown_policy") != "reject_and_report" or tokenizer.get("overlength_policy") != "reject_and_report"
    ):
        raise ValueError("Tokenizer must use train vocabulary, explicit PAD/MASK/EOS, and report rejected rows")
    require_count(tokenizer["max_length"], "max_length", positive=True)
    if tokenizer["max_length"] < 2:
        raise ValueError("Tokenizer max_length must include a molecule token and EOS")
    require_count(tokenizer["shard_rows"], "shard_rows", positive=True)
    paths = [tokenizer[name] for name in ("vocabulary_path", "metadata_path", "length_report_path")]
    for path in paths:
        project_path(path)
    if len(set(paths)) != 3:
        raise ValueError("Vocabulary, tokenizer metadata, and length report need distinct paths")


def read_benchmark_assets(config: Config) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Verify provenance and bytes before parsing chemistry or writing artifacts."""
    options = config["data"]["benchmark"]
    directory = project_path(config["paths"]["benchmark"])
    path = project_path(os.path.join(directory, options["source_file"]))
    if sha256_file(path) != options["source_sha256"]:
        raise ValueError(f"Benchmark provenance checksum mismatch: {path}")
    with open(path, encoding="utf-8") as handle:
        source = json.load(handle)
    if source.get("schema_version") != 1 or source.get("commit") != BENCHMARK_COMMIT or set(source.get("groups", {})) != set(GROUPS):
        raise ValueError("Unsupported benchmark provenance or group definitions")
    expected_files = {group + suffix for group in GROUPS for suffix in (".prompt.txt", ".examples.smi", ".lark")} | {"smarts.json"}
    if set(source.get("files", {})) != expected_files:
        raise ValueError("Provenance must cover all ten prompt/example/grammar/SMARTS assets")
    for filename, checksum in source["files"].items():
        require_identifier(filename, "asset filename")
        require_sha256(checksum)
        if sha256_file(os.path.join(directory, filename)) != checksum:
            raise ValueError(f"Benchmark asset checksum mismatch: {filename}")
    examples = {}
    for group in GROUPS:
        definition = source["groups"][group]
        examples[group] = read_smiles(os.path.join(directory, group + ".examples.smi"))
        if definition["count"] != EXPECTED_COUNTS[group] or len(examples[group]) != EXPECTED_COUNTS[group]:
            raise ValueError(f"Incorrect example count for {group}")
        with open(os.path.join(directory, group + ".prompt.txt"), encoding="utf-8") as handle:
            prompt_lines = handle.read().splitlines()
        extracted = [(number, line[len("Molecule: "):]) for number, line in enumerate(prompt_lines, 1)
                     if line.startswith("Molecule: ") and line != "Molecule: "]
        if (
            [text for _, text in extracted] != examples[group]
            or [number for number, _ in extracted] != definition["example_source_lines"]
            or not prompt_lines or prompt_lines[-1] != "Molecule: "
        ):
            raise ValueError(f"Examples do not reproduce the original prompt for {group}")
    return source, examples


def _preflight(config: Config, store: ArtifactStore) -> dict:
    # Packaged assets are relative to the project, not the remote artifact root.
    # Their pinned source digest is part of config_hash and commits to all ten
    # file digests, so it also binds resume to the exact benchmark input bytes.
    read_benchmark_assets(config)
    if config.get("group") is not None:
        raise ValueError("T01 prepares all three groups; leave group=null")
    return {}


def prepare_benchmark(context: TaskContext) -> TaskResult:
    source, examples = read_benchmark_assets(context.config)
    versions = {name: importlib.metadata.version(name) for name in ("rdkit", "llguidance")}
    if versions != {"rdkit": "2025.3.6", "llguidance": "0.7.30"}:
        raise RuntimeError(f"Benchmark checks require the project-pinned RDKit/llguidance versions: {versions}")
    options = context.config["data"]["benchmark"]
    examples_path = context.store.directory("data", "benchmark", options["examples_file"])
    manifest_path = context.store.directory("data", "benchmark", options["manifest_file"])
    resumed = context.resume_state
    if isinstance(resumed, dict) and resumed.get("stage") == "written":
        context.store.require_file(manifest_path, checksum=resumed["manifest_sha256"])
        context.store.require_manifest(manifest_path, kind="benchmark", config_hash=context.identity.config_hash)
        context.log("resumed committed benchmark artifacts")
        return TaskResult(resumed["summary"], artifacts=(examples_path, manifest_path))
    rows = []
    diagnostics = {}
    # Benchmark diagnostics include both rewards, independent of the run suite.
    with context.tracker.measure("benchmark"):
        checker = RewardChecker(asset_directory=context.config["paths"]["benchmark"], tracker=context.tracker)
        for group, grammar in checker.grammars.items():
            for warning in grammar.warnings:
                context.log(f"{group} grammar: {warning}", level="WARNING")
        with context.progress(total=sum(EXPECTED_COUNTS.values()), desc="Check benchmark", unit="example") as progress:
            for group in GROUPS:
                failures = {name: [] for name in ("grammar", "rdkit", "membership", "joint")}
                for index, smiles in enumerate(examples[group], 1):
                    result = checker.evaluate_smiles(smiles, group)
                    source_line = source["groups"][group]["example_source_lines"][index - 1]
                    flags = {"grammar": result.grammar_valid, "rdkit": result.rdkit_valid,
                             "membership": result.class_member, "joint": result.reward == 1.0}
                    for name, passed in flags.items():
                        if not passed:
                            failures[name].append(index)
                    rows.append({
                        "group": group, "example_index": index, "smiles": smiles,
                        "canonical_smiles": result.canonical_smiles,
                        "molecule_id": molecule_id(result.canonical_smiles) if result.canonical_smiles is not None else None,
                        "groups": list(checker.classify_smiles(smiles)),
                        "grammar_valid": result.grammar_valid, "rdkit_valid": result.rdkit_valid,
                        "class_member": result.class_member, "joint_reward": result.reward,
                        "grammar_only_reward": float(result.grammar_valid), "error": result.error,
                        "source_file": source["groups"][group]["prompt_source"], "source_line": source_line,
                        "source_commit": source["commit"],
                    })
                    progress.update(1)
                canonical = {row["canonical_smiles"] for row in rows if row["group"] == group and row["rdkit_valid"]}
                diagnostics[group] = {
                    "examples": len(examples[group]), "unique_valid_molecules": len(canonical),
                    "failure_counts": {name: len(indices) for name, indices in failures.items()},
                    "failure_example_indices": failures,
                }
                context.log(
                    f"{group}: {len(examples[group])} examples; "
                    f"grammar/RDKit/SMARTS failures={len(failures['grammar'])}/{len(failures['rdkit'])}/{len(failures['membership'])}",
                )
        # No filtering or deduplication: all source prompt rows survive here.
        write_parquet(context.store, examples_path, rows)
        context.store.create_manifest(
            manifest_path, kind="benchmark", config_hash=context.identity.config_hash, files=[examples_path],
            metadata={
                "source_sha256": options["source_sha256"], "repository": source["repository"], "commit": source["commit"],
                "asset_sha256": source["files"], "identity_version": IDENTITY_VERSION,
                "reward_versions": REWARD_VERSIONS, "library_versions": versions,
                "grammar_warnings": {group: list(grammar.warnings) for group, grammar in checker.grammars.items()},
                "expected_counts": EXPECTED_COUNTS, "rows": len(rows), "groups": diagnostics,
                "preserve_original_rows": True, "mismatch_policy": "report",
                "prompt_exclusion": "all_rdkit_valid_prompt_identities_across_all_groups",
                "paper_relation": "Original assets/SMARTS; joint reward and canonical identity are new experiment settings.",
            },
        )
    total_failures = sum(group["failure_counts"]["joint"] for group in diagnostics.values())
    summary = f"54 examples (32/11/11); {total_failures} joint-check mismatches; manifest={manifest_path}"
    context.save_checkpoint({"stage": "written", "manifest_sha256": sha256_file(context.store.require_file(manifest_path)), "summary": summary})
    return TaskResult(summary, artifacts=(examples_path, manifest_path))


def execute(argv: Sequence[str] | None = None, *, execution: str = "local") -> RunRecord:
    return run_task(
        "T01", prepare_benchmark, description="Prepare the original CARS SMILES benchmark and report checker mismatches.",
        argv=argv, default_config="configs/data.yaml", execution=execution,
        supports_seed=False, validator=validate_data, dependencies=_preflight,
    )


if __name__ == "__main__":
    try:
        record = execute()
    except KeyboardInterrupt:
        tqdm.write("T01: interrupted")
        raise SystemExit(130) from None
    except Exception:
        # The shared task lifecycle already printed the short failure result.
        raise SystemExit(1) from None
    raise SystemExit(0 if record.status == "completed" else 1)
