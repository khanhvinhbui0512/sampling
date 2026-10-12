"""Offline fixtures; no downloaded QM9 assets or large checkpoints."""

import os
import tempfile

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from src.models.backbone import QM9Backbone
from src.models.configuration_udlm import UDLMConfig
from src.models.modeling_udlm import UDLM
from src.qm9.chemistry import ChemistryPolicy
from src.types import TokenRules


@pytest.fixture
def tokenizer():
    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<mask>": 3, "<unk>": 4,
                  "C": 5, "N": 6, "O": 7, "(": 8, ")": 9}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>", special_tokens=[("<bos>", 1), ("<eos>", 2)],
    )
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                  bos_token="<bos>", eos_token="<eos>", mask_token="<mask>",
                                  unk_token="<unk>", model_max_length=6)


@pytest.fixture
def token_rules(tokenizer):
    return TokenRules(len(tokenizer), 6, mask_id=tokenizer.mask_token_id,
                      bos_id=tokenizer.bos_token_id, eos_id=tokenizer.eos_token_id,
                      pad_id=tokenizer.pad_token_id, unk_id=tokenizer.unk_token_id)


@pytest.fixture
def tiny_model(token_rules):
    return UDLM(UDLMConfig(vocab_size=token_rules.vocab_size, model_length=6, hidden_dim=32,
                           cond_dim=16, n_blocks=2, n_heads=4, dropout=0.0,
                           time_conditioning=False, mask_token_id=token_rules.mask_id,
                           bos_token_id=token_rules.bos_id, eos_token_id=token_rules.eos_id,
                           pad_token_id=token_rules.pad_id))


@pytest.fixture
def tiny_backbone(tiny_model, tokenizer, token_rules):
    return QM9Backbone(tiny_model, tokenizer, token_rules, tokenizer_sha256="a" * 64)


@pytest.fixture
def chemistry_policy():
    return ChemistryPolicy(canonical=True, isomeric_smiles=True, sanitize=True,
                           parse_names=False, allow_cxsmiles=False, heavy_atom_limit=9)


@pytest.fixture
def artifact_directory():
    # Keep temporary artifact paths inside the repo, matching runtime path rules.
    os.makedirs("workspace", exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="step3-test-", dir="workspace") as directory:
        yield os.path.relpath(directory)


@pytest.fixture
def fake_backbone_factory(token_rules):
    def make(probabilities):
        distribution = torch.tensor(probabilities, dtype=torch.float32)
        if (distribution.shape != (token_rules.vocab_size,) or not torch.isfinite(distribution).all()
                or torch.any(distribution < 0) or not torch.isclose(distribution.sum(), torch.tensor(1.0))):
            raise ValueError("Supply a normalized probability for every toy token")

        class FakeBackbone:
            reference_id = "toy-backbone"
            checkpoint_id = "toy-weights"
            tokenizer_id = "toy-tokenizer"
            time_conditioning = False

            def __init__(self):
                self.token_rules = token_rules

            def logits_batch(self, states, *, budget, phase, reservation=None):
                if any(state.rules != token_rules for state in states):
                    raise ValueError("Toy state token rules mismatch")
                if not states:
                    return torch.empty((0, token_rules.sequence_length, token_rules.vocab_size))
                with budget.base_forward(batch_size=len(states), phase=phase, reservation=reservation):
                    return distribution.log().expand(len(states), token_rules.sequence_length, -1).clone()

            def logits_at_noise_batch(self, states, sigma, *, budget, phase, reservation=None):
                return self.logits_batch(states, budget=budget, phase=phase, reservation=reservation)

        return FakeBackbone()

    return make


@pytest.fixture
def toy_joint():
    from src.io import read_json

    return read_json("tests/fixtures/toy_joint.json")


@pytest.fixture
def native_case_factory():
    """Two-site native model whose future observations change its predictions."""
    from types import SimpleNamespace
    from src.sampling.native import NativeMDLM
    from src.sampling.native_reference import NativeReference

    def make(*, zero_second_two=False, time_conditioning=False, steps=2):
        rules = TokenRules(3, 2, mask_id=0)

        class Backbone:
            checkpoint_id, tokenizer_id = "native-toy-weights", "native-toy-tokenizer"

            def __init__(self):
                self.token_rules, self.time_conditioning = rules, time_conditioning
                self.inputs = []

            def logits_at_noise_batch(self, states, sigma, *, budget, phase, reservation=None):
                with budget.base_forward(batch_size=len(states), phase=phase, reservation=reservation):
                    self.inputs.extend(state.token_ids for state in states)
                    probabilities = []
                    for state in states:
                        a, b = state.token_ids
                        p0 = .9 if b == 1 else .2 if b == 2 else .7
                        p1 = .8 if a == 1 else .1 if a == 2 else .3
                        if zero_second_two:
                            p1 = 1.
                        if time_conditioning:
                            p0 = .2 + .5 / (1 + float(sigma[len(probabilities)]))
                        probabilities.append([[0., p0, 1 - p0], [0., p1, 1 - p1]])
                    return torch.tensor(probabilities, dtype=torch.float64).log()

        class Reward:
            token_rules, tokenizer_sha256 = rules, Backbone.tokenizer_id
            calibration = SimpleNamespace(calibration_id="native-toy-pass95", threshold=.5,
                                          checkpoint_sha256=Backbone.checkpoint_id)

            def evaluate(self, state, *, require_binary):
                assert state.is_complete and require_binary and state.rules == rules
                reward = int(state.token_ids in ((1, 1), (2, 2)))
                return SimpleNamespace(reward=reward, valid=True, score=float(reward))

        return NativeReference(NativeMDLM(Backbone(), steps=steps, time_stop=.2)), Reward()

    return make


@pytest.fixture
def native_case(native_case_factory):
    return native_case_factory()


@pytest.fixture
def enumerate_native():
    """Enumerate full step/coordinate paths, retaining distinct reveal timings."""
    import math
    from src.budget import BudgetTracker
    from src.types import CostPhase

    def enumerate_paths(reference):
        paths = []
        budget = BudgetTracker(base_sequence_limit=None)

        def visit(path, probability):
            cursor = reference.cursor(path)
            if cursor.done:
                paths.append((path, cursor.observations, probability))
                return
            raw = cursor.next_log_probs(budget=budget, phase=CostPhase.ESTIMATION)
            for token, log_p in enumerate(raw):
                if math.isfinite(log_p):
                    visit(path + (token,), probability * math.exp(log_p))

        visit((), 1.)
        return paths

    return enumerate_paths


@pytest.fixture
def cars_reward_factory(toy_joint, toy_backbone):
    """Calibrated finite binary reward; no molecular assets or network needed."""
    from types import SimpleNamespace

    def make(successful_tokens=None):
        successful = set(tuple(row["tokens"]) for row in toy_joint["leaves"] if row["reward"])
        if successful_tokens is not None:
            successful = set(tuple(tokens) for tokens in successful_tokens)

        class ToyReward:
            token_rules = toy_backbone.token_rules
            tokenizer_sha256 = toy_backbone.tokenizer_id
            calibration = SimpleNamespace(calibration_id="cars-toy-pass95", threshold=.5,
                                          checkpoint_sha256=toy_backbone.checkpoint_id)
            successful_tokens = successful

            def evaluate(self, state, *, require_binary):
                assert require_binary and state.is_complete and state.rules == self.token_rules
                reward = int(state.token_ids in self.successful_tokens)
                return SimpleNamespace(reward=reward, valid=True, score=float(reward), molecule=None)

        return ToyReward()

    return make


@pytest.fixture
def toy_backbone(toy_joint):
    """Match the joint on prefixes, but deliberately ignore future observations."""
    class ToyBackbone:
        reference_id = "toy-prefix-backbone"
        checkpoint_id = "toy-prefix-weights"
        tokenizer_id = "toy-prefix-tokenizer"
        time_conditioning = False

        def __init__(self):
            self.token_rules = TokenRules(**toy_joint["token_rules"])
            self.noise_calls = []

        def logits_batch(self, states, *, budget, phase, reservation=None):
            with budget.base_forward(batch_size=len(states), phase=phase, reservation=reservation):
                probabilities = torch.zeros(len(states), 3, 3, dtype=torch.float64)
                for row, state in enumerate(states):
                    prefix = state.token_ids[:state.masked_positions[0]] if not state.is_complete else state.token_ids
                    leaves = [leaf for leaf in toy_joint["leaves"]
                              if tuple(leaf["tokens"][:len(prefix)]) == prefix]
                    total = sum(leaf["probability"] for leaf in leaves)
                    for leaf in leaves:
                        for position, token in enumerate(leaf["tokens"]):
                            probabilities[row, position, token] += leaf["probability"] / total
                return probabilities.log()

        def logits_at_noise_batch(self, states, sigma, *, budget, phase, reservation=None):
            self.noise_calls.append(torch.as_tensor(sigma).clone())
            return self.logits_batch(states, budget=budget, phase=phase, reservation=reservation)

    return ToyBackbone()


@pytest.fixture
def offline_assets(artifact_directory, tokenizer, token_rules, chemistry_policy):
    """A three-row prepared artifact with real local tokenizer/Arrow files."""
    from dataclasses import asdict

    import datasets

    from src.io import RunArtifacts, file_hash, payload_hash, write_json, write_jsonl, write_yaml
    from src.qm9.chemistry import rdkit_version
    from src.qm9.dataset import load_prepared_qm9, token_rules_from_tokenizer

    attempt = RunArtifacts.create(artifact_directory, run_id="assets", attempt_id="offline",
                                  config={}, target_id="offline-assets", reference_id="offline-qm9")
    attempt.manifest["task"] = "prepare_qm9_assets"
    tokenizer.save_pretrained(attempt.path("tokenizer"))
    raw = datasets.Dataset.from_dict({"smiles": ["C", "C ", "O"],
                                      "canonical_smiles": ["C", "C", "O"]})
    raw.save_to_disk(attempt.path("raw"))
    _, token_metadata = token_rules_from_tokenizer(tokenizer, {
        "sequence_length": token_rules.sequence_length, "required_special_tokens": ["mask", "pad", "unk"],
        "terminal_policy": "strict_special_tokens_v1",
    })
    write_yaml(attempt.path("config.yaml"), {"offline_fixture": True})
    write_json(attempt.path("execution.json"), {"backend": "offline-fixture"})
    write_json(attempt.path("revisions.json"), {"offline_fixture": True})
    write_json(attempt.path("environment.json"), {"rdkit_version": rdkit_version()})
    write_json(attempt.path("token_rules.json"), token_metadata)
    write_json(attempt.path("vocabulary.json"), tokenizer.get_vocab())
    write_json(attempt.path("splits.json"), {
        "source_rows": 3, "train_row_ids": [0, 1], "test_row_ids": [2], "seed": 42,
    })
    write_jsonl(attempt.path("molecules.jsonl"), [{"source_row_id": i} for i in range(3)], progress=False)
    write_jsonl(attempt.path("train_canonical.jsonl"), [{"canonical_smiles": "C"}], progress=False)
    write_json(attempt.path("audit.json"), {"offline_fixture": True})
    names = []
    for directory, directories, files in os.walk(attempt.directory):
        directories.sort()
        names.extend(os.path.relpath(os.path.join(directory, name), attempt.directory)
                     for name in sorted(files) if os.path.join(directory, name) != attempt.path("manifest.json"))
    records = {name: attempt.register(name) for name in names}
    token_files = [name for name in names if name.startswith("tokenizer/")]
    token_hash = payload_hash({name: file_hash(attempt.path(name))
                              for name in [*token_files, "token_rules.json", "vocabulary.json"]})
    attempt.manifest["preparation"] = {
        "train_rows": 2, "test_rows": 1, "tokenizer_sha256": token_hash,
        "splits_sha256": records["splits.json"]["sha256"], "chemistry_policy": asdict(chemistry_policy),
    }
    attempt.complete(required_artifacts=names)
    return load_prepared_qm9(attempt.path("manifest.json"))
