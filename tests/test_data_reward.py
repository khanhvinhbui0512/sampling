"""Benchmark fixtures and phase-gated checks of real prepared artifacts.

Run through T12 locally or on Modal. Corpus checks never pass using toy rows;
they require the actual T02/T03 artifacts when the selected phase is corpus.
"""

from __future__ import annotations

import json
import os
from collections import Counter

import pytest

from src.artifacts import ArtifactStore, sha256_file
from src.config import load_config, read_yaml
from src.data import (
    IDENTITY_VERSION, SPLITS, InvalidSmiles, MoleculeRecord, assert_disjoint_splits,
    augment_molecule, canonicalize_smiles, molecule_id, prepare_molecules,
    read_parquet, read_smiles, split_molecules, write_parquet,
)
from src.grammar import GrammarCursor, GrammarError
from src.reward import RewardChecker
from src.schemas import GROUPS, SmilesState
from src.tasks.prepare_benchmark import read_benchmark_assets, validate_data
from src.tokenizer import (
    SPECIAL_NAMES, SmilesTokenizer, TokenizationError, UnknownToken,
    detokenize_smiles, tokenize_smiles,
)


@pytest.fixture(scope="module")
def checker():
    return RewardChecker()


@pytest.fixture(scope="module")
def grammars(checker):
    return checker.grammars


@pytest.fixture
def tokenizer():
    return SmilesTokenizer.build(
        ["C=CC(=O)OCC", "CN=C=O", "NCCN", "CO", "CC", "ClCBr", "C1CC1", "[NH4+]"], max_length=64,
    )


def _phase_store():
    path = os.environ.get("SMILES_VERIFICATION_CONFIG")
    if not path:
        raise AssertionError("Artifact checks require the T12 resolved configuration")
    config = read_yaml(path)
    return ArtifactStore(
        os.environ["SMILES_ARTIFACT_ROOT"],
        directories={name: config["paths"][name] for name in ("data", "checkpoints", "outputs", "reports")},
    )


def test_original_assets_and_prompt_extraction():
    with open("assets/benchmark/source.json", encoding="utf-8") as handle:
        source = json.load(handle)
    assert source["commit"] == "f46607f983d89e4ca4f40527cd231ab04adc6ed7"
    assert {group: source["groups"][group]["count"] for group in GROUPS} == {
        "acrylates": 32, "chain_extenders": 11, "isocyanates": 11,
    }
    expected_files = {group + suffix for group in GROUPS for suffix in (".prompt.txt", ".examples.smi", ".lark")} | {"smarts.json"}
    assert set(source["files"]) == expected_files
    for name, checksum in source["files"].items():
        assert sha256_file("assets/benchmark/" + name) == checksum
    for group in GROUPS:
        with open(f"assets/benchmark/{group}.prompt.txt", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        examples = read_smiles(f"assets/benchmark/{group}.examples.smi")
        extracted = [(number, line[len("Molecule: "):]) for number, line in enumerate(lines, 1)
                     if line.startswith("Molecule: ") and line != "Molecule: "]
        assert [text for _, text in extracted] == examples
        assert [number for number, _ in extracted] == source["groups"][group]["example_source_lines"]
        assert len(examples) == source["groups"][group]["count"]
        assert lines[-1] == "Molecule: "


def test_data_configuration_pins_assets_and_rejects_unsupported_policy():
    config = load_config("configs/data.yaml", validator=validate_data)
    source, examples = read_benchmark_assets(config)
    assert source["commit"] == config["data"]["benchmark"]["commit"]
    assert sum(map(len, examples.values())) == 54
    with pytest.raises(ValueError, match="normalization"):
        load_config("configs/data.yaml", overrides=["data.normalization.neutralize=true"], validator=validate_data)
    with pytest.raises(ValueError, match="Tokenizer"):
        load_config("configs/data.yaml", overrides=["data.tokenizer.max_length=1"], validator=validate_data)


def test_canonical_identity_retains_stereochemistry_and_removes_atom_maps():
    assert canonicalize_smiles("CCO") == canonicalize_smiles("OCC")
    assert canonicalize_smiles("[CH3:7][OH:9]") == canonicalize_smiles("CO")
    assert canonicalize_smiles("F[C@H](Cl)Br") != canonicalize_smiles("F[C@@H](Cl)Br")
    assert canonicalize_smiles("[13CH4]") != canonicalize_smiles("C")


@pytest.mark.parametrize("text", ["", " CO", "CO name", "CO\n", "C1CC", "C(", "<mask>CO", "CO |atomProp:0.x.y|"])
def test_invalid_input_is_not_silently_cleaned(text):
    with pytest.raises(InvalidSmiles):
        canonicalize_smiles(text)


def test_multilabel_deduplication_and_global_prompt_exclusion(checker):
    records, stats = prepare_molecules([
        {"smiles": "C=CC(=O)OCC", "source_id": "a"},
        {"smiles": "CCOC(=O)C=C", "source_id": "b"},
        {"smiles": "NCCN", "source_id": "c"},
        {"smiles": "CO", "source_id": "d"},
    ], checker.classify_molecule, exclude_smiles=["OC"])
    assert len(records) == 2 and stats["duplicate_rows"] == 1 and stats["prompt_identity"] == 1
    acrylate = next(record for record in records if "acrylates" in record.groups)
    assert acrylate.groups == ("acrylates", "chain_extenders")
    assert acrylate.source_ids == ("a", "b")


def test_split_and_augmentation_preserve_graph_partition(checker):
    records, _ = prepare_molecules([{"smiles": "C" * length + "O"} for length in range(1, 21)], checker.classify_molecule)
    split = split_molecules(records, seed=17)
    assert [len(split[name]) for name in SPLITS] == [16, 2, 2]
    assert split == split_molecules(list(reversed(records)), seed=17)
    expanded = {name: [row for record in split[name] for row in augment_molecule(record, split=name, seed=8)]
                for name in SPLITS}
    assert_disjoint_splits(expanded)
    assert all(row["augmentation_index"] == 0 for name in ("validation", "test") for row in expanded[name])
    record = MoleculeRecord(canonicalize_smiles("F[C@H](Cl)Br"), ("chain_extenders",))
    assert augment_molecule(record, split="train", seed=3) == augment_molecule(record, split="train", seed=3)
    assert all(canonicalize_smiles(row["smiles"]) == record.canonical_smiles for row in augment_molecule(record, split="train"))
    leaked = {name: list(rows) for name, rows in expanded.items()}
    leaked["test"].append({**expanded["train"][0], "split": "test"})
    with pytest.raises(ValueError, match="leaks"):
        assert_disjoint_splits(leaked)
    with pytest.raises(ValueError, match="Deduplicate"):
        split_molecules([records[0], records[0]])


def test_parquet_roundtrip(artifact_store):
    rows = [{"smiles": "CO", "groups": ["chain_extenders"], "molecule_id": molecule_id("CO")}]
    write_parquet(artifact_store, "data/fixture.parquet", rows)
    restored = read_parquet(artifact_store, "data/fixture.parquet")
    assert restored[0]["smiles"] == "CO"
    assert list(restored[0]["groups"]) == ["chain_extenders"]


@pytest.mark.parametrize("text", ["ClCBr", "N[C@@H](C)C(=O)O", "[NH4+]", "C%12CCCCC%12", "C%(100)CCCCC%(100)", "[NH3]->[Cu+2]", "C/C=C\\C"])
def test_lossless_lexical_tokenization(text):
    assert detokenize_smiles(tokenize_smiles(text)) == text
    assert SmilesTokenizer.build([text]).decode_tokens(SmilesTokenizer.build([text]).encode_text(text, add_eos=False)) == text


def test_unknown_characters_unknown_vocabulary_and_lengths(tokenizer):
    for text in ("CO ", "C?<mask>", "[]", "C%1", "<eos>"):
        with pytest.raises(TokenizationError):
            tokenize_smiles(text)
    with pytest.raises(UnknownToken):
        tokenizer.encode_text("[Xe]")
    with pytest.raises(TokenizationError, match="truncation"):
        tokenizer.encode_text("C" * tokenizer.max_length)
    with pytest.raises(TokenizationError):
        tokenizer.decode_tokens([tokenizer.special_tokens.mask_id])
    assert "[Xe]" not in tokenizer.vocabulary and "<unk>" not in tokenizer.vocabulary


def test_tokenizer_save_load_checksums(artifact_store, tokenizer):
    tokenizer.save(artifact_store)
    restored = SmilesTokenizer.load(artifact_store)
    assert restored.fingerprint == tokenizer.fingerprint
    assert restored.decode_state(restored.encode_state("CO", "chain_extenders")) == "CO"
    artifact_store.write_json("data/tokenizer/vocab.json", {"version": "smiles-lexical-v1", "tokens": [*SPECIAL_NAMES, "C"]})
    with pytest.raises(ValueError, match="checksum"):
        SmilesTokenizer.load(artifact_store)


@pytest.mark.parametrize("group", GROUPS)
def test_grammar_prefix_and_complete_string_distinction(grammars, group):
    grammar = grammars[group]
    assert grammar.check_prefix("").status == "viable"
    assert grammar.check_prefix("C(").status == "viable"
    assert grammar.check_prefix(")").prunable
    assert not grammar.accepts("C(")
    assert grammar.accepts("CC")
    assert grammar.accepts("C1CC")  # Grammar does not pair ring digits.


def test_smiles_native_token_continuations(grammars, tokenizer):
    cursor = grammars["chain_extenders"].token_cursor(tokenizer)
    cursor.advance(tokenizer.encode_text("CO", add_eos=False))
    assert cursor.result.accepting
    allowed = cursor.continuations().allowed_token_ids
    assert allowed is not None and tokenizer.eos_token_id in allowed
    assert not set(tokenizer.special_token_ids[:2]) & set(allowed)
    assert cursor.copy().advance([tokenizer.token_to_id[")"]]).prunable
    assert cursor.result.status == "viable"


def test_fast_ar_adapter_uses_native_multi_character_tokens(grammars):
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.BPE(
        {"C": 0, "CC": 1, "(": 2, ")": 3, "<eos>": 4, "<unk>": 5}, [("C", "C")], unk_token="<unk>",
    ))
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", unk_token="<unk>")
    cursor = grammars["chain_extenders"].token_cursor(tokenizer)
    assert cursor.advance([tokenizer.convert_tokens_to_ids("CC")]).accepting
    assert tokenizer.eos_token_id in cursor.continuations().allowed_token_ids
    assert cursor.copy().advance([tokenizer.convert_tokens_to_ids(")")]).prunable


def test_parser_limits_never_certify_failed_cones():
    class Tokenizer:
        vocab_size = 2
        eos_token = 1

    class LimitedMatcher:
        failed = False

        def stop_reason(self):
            return "ParserTooComplex" if self.failed else "NotStopped"

        def is_error(self):
            return self.failed

        def get_error(self):
            return "parser budget exhausted" if self.failed else ""

        def is_accepting(self):
            return False

        def try_consume_tokens(self, tokens):
            self.failed = True
            return 0

    cursor = GrammarCursor(LimitedMatcher(), Tokenizer())
    assert cursor.advance([0]).status == "unknown"
    assert not cursor.result.prunable
    assert cursor.continuations().allowed_token_ids is None


def test_primary_reward_and_original_membership(checker, smiles_cases):
    for case in smiles_cases:
        assert set(checker.classify_smiles(case["smiles"])) == set(case["groups"])
        for group in GROUPS:
            result = checker.evaluate_smiles(case["smiles"], group)
            assert result.rdkit_valid is case["rdkit_valid"]
            assert result.class_member is (group in case["groups"])
            assert result.reward == float(result.grammar_valid and result.rdkit_valid and result.class_member)
    assert checker.label("CO", "chain_extenders") == 1.0  # Not required to be difunctional.
    assert checker.evaluate_smiles("CC", "chain_extenders").grammar_valid
    assert checker.label("CC", "chain_extenders") == 0.0
    assert checker.evaluate_smiles("C1CC", "chain_extenders").grammar_valid
    assert not checker.evaluate_smiles("C1CC", "chain_extenders").rdkit_valid


def test_replication_keeps_grammar_only_acceptance(checker):
    replication = RewardChecker(mode="grammar_only", grammars=checker.grammars)
    for text in ("CC", "C1CC"):
        assert replication.label(text, "chain_extenders") == 1.0
        assert checker.label(text, "chain_extenders") == 0.0


def test_eos_pad_mask_and_terminal_anchoring(checker, tokenizer):
    terminal = tokenizer.encode_state("CO", "chain_extenders", width=8)
    assert checker.terminal_value(terminal, tokenizer) == 1.0
    assert checker.terminal_accept(terminal, tokenizer)
    assert checker.label(tokenizer.decode_state(terminal), terminal.group) == 1.0
    masked_before_eos = terminal.remask([0])
    assert not checker.terminal_accept(masked_before_eos, tokenizer)
    with pytest.raises(ValueError, match="incomplete"):
        checker.terminal_value(masked_before_eos, tokenizer)
    masked_after_eos = terminal.remask([7])
    assert checker.terminal_accept(masked_after_eos, tokenizer)
    for state in (
        terminal.with_tokens({0: tokenizer.special_tokens.pad_id}),
        terminal.with_tokens({7: tokenizer.token_to_id["C"]}),
        terminal.with_tokens({7: tokenizer.special_tokens.eos_id}),
        terminal.with_tokens({0: tokenizer.vocab_size}),
    ):
        assert not checker.terminal_accept(state, tokenizer)
        assert checker.terminal_value(state, tokenizer) == 0.0
    no_eos = SmilesState((tokenizer.token_to_id["C"],), (False,), "chain_extenders", tokenizer.special_tokens)
    assert not checker.terminal_accept(no_eos, tokenizer)


def test_unknown_grammar_decision_is_not_a_negative_label(checker):
    class UnknownGrammar:
        def accepts(self, text):
            raise GrammarError("parser limit")

    limited = RewardChecker(grammars={group: UnknownGrammar() for group in GROUPS})
    with pytest.raises(GrammarError, match="parser limit"):
        limited.label("CO", "chain_extenders")


def test_real_benchmark_artifacts_preserve_all_examples(checker):
    # Decide inside the test: T12 all invokes pytest repeatedly in one process,
    # where import-time skip conditions could retain an earlier phase's value.
    if os.environ.get("SMILES_VERIFICATION_PHASE") not in {"benchmark", "corpus"}:
        pytest.skip("Requires T01 artifacts through T12")
    store = _phase_store()
    manifest = store.require_manifest(store.directory("data", "benchmark", "manifest.json"), kind="benchmark")
    assert manifest.metadata["identity_version"] == IDENTITY_VERSION
    assert manifest.metadata["source_sha256"] == sha256_file("assets/benchmark/source.json")
    rows = read_parquet(store, store.directory("data", "benchmark", "examples.parquet"))
    assert len(rows) == 54
    assert Counter(row["group"] for row in rows) == {"acrylates": 32, "chain_extenders": 11, "isocyanates": 11}
    for group in GROUPS:
        original = read_smiles(f"assets/benchmark/{group}.examples.smi")
        group_rows = sorted((row for row in rows if row["group"] == group), key=lambda row: row["example_index"])
        assert [row["smiles"] for row in group_rows] == original
        for row in group_rows:
            result = checker.evaluate_smiles(row["smiles"], group)
            assert row["grammar_valid"] == result.grammar_valid
            assert row["rdkit_valid"] == result.rdkit_valid
            assert row["class_member"] == result.class_member
            assert row["joint_reward"] == float(result.grammar_valid and result.rdkit_valid and result.class_member)
        reported = manifest.metadata["groups"][group]["failure_example_indices"]
        for name, flag in (("grammar", "grammar_valid"), ("rdkit", "rdkit_valid"), ("membership", "class_member"), ("joint", "joint_reward")):
            indices = [row["example_index"] for row in group_rows if not row[flag]]
            assert reported[name] == indices
            assert manifest.metadata["groups"][group]["failure_counts"][name] == len(indices)


def test_real_corpus_and_tokenizer_have_no_leakage_or_test_vocabulary(checker):
    if os.environ.get("SMILES_VERIFICATION_PHASE") != "corpus":
        pytest.skip("Requires actual T02/T03 artifacts through T12")
    import pyarrow.parquet as pq

    store = _phase_store()
    corpus_path = store.directory("data", "corpus", "manifest.json")
    corpus = store.require_manifest(corpus_path, kind="corpus")
    metadata_path = store.directory("data", "tokenizer", "metadata.json")
    tokenizer = SmilesTokenizer.load(store, metadata_path)
    metadata = store.read_json(metadata_path)
    train_path = store.directory("data", "corpus", "train.parquet")
    assert metadata["provenance"]["training_split_sha256"] == sha256_file(store.require_file(train_path))
    assert metadata["provenance"]["corpus_sha256"] == sha256_file(store.require_file(corpus_path))
    assert corpus.metadata["identity_version"] == IDENTITY_VERSION
    excluded = set()
    for group in GROUPS:
        for smiles in read_smiles(f"assets/benchmark/{group}.examples.smi"):
            try:
                excluded.add(canonicalize_smiles(smiles))
            except InvalidSmiles:
                continue  # T01 preserves/reports invalid examples, which have no graph identity.
    vocabulary = set(SPECIAL_NAMES)
    owners = {}
    for split in SPLITS:
        path = store.directory("data", "corpus", split + ".parquet")
        assert path in corpus.files, "Every corpus split must be covered by its manifest"
        counts = Counter()
        for batch in pq.ParquetFile(store.require_file(path)).iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                canonical = canonicalize_smiles(row["smiles"])
                assert canonical == row["canonical_smiles"] and canonical not in excluded
                assert row["molecule_id"] == molecule_id(canonical)
                assert row["split"] == split
                assert owners.setdefault(row["molecule_id"], split) == split
                assert set(row["groups"]) == set(checker.classify_smiles(row["smiles"]))
                if split == "train":
                    vocabulary.update(tokenize_smiles(row["smiles"]))
                counts["input_rows"] += 1
                try:
                    tokenizer.encode_text(row["smiles"])
                except UnknownToken:
                    counts["unknown_rows"] += 1
                except TokenizationError:
                    counts["overlength_rows"] += 1
                else:
                    counts["rows"] += 1
        token_manifest = store.require_manifest(store.directory("data", "tokenized", split, "manifest.json"), kind="tokenized")
        assert token_manifest.metadata["split"] == split
        assert token_manifest.metadata["tokenizer_fingerprint"] == tokenizer.fingerprint
        assert token_manifest.metadata["corpus_sha256"] == sha256_file(store.require_file(corpus_path))
        for name in ("input_rows", "unknown_rows", "overlength_rows", "rows"):
            assert token_manifest.metadata[name] == counts[name]
    assert set(tokenizer.vocabulary) == vocabulary
