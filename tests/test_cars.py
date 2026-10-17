import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.algorithms.cars import _TrieNode, _update_trie, sample_cars
from src.algorithms.config import apply_budget_overrides, cars_config, value_guidance_config


class ToyHarness:
    mask_id = 0
    vocab_size = 5

    def __init__(self):
        self.evaluations = 0

    def decode_state(self, state):
        return "".join("_ABXY"[token] for token in state)

    def logits_batch(self, prompts, states):
        self.evaluations += len(states)
        scores = torch.full((len(states), len(states[0]), 5), -math.inf)
        for row, state in enumerate(states):
            for pos, token in enumerate(state):
                if token != self.mask_id:
                    continue
                if pos == 0:
                    scores[row, pos, 1:3] = torch.tensor([0.8, 0.2]).log()
                else:
                    valid = 0.25 if state[0] == 1 else 0.75
                    scores[row, pos, 3:5] = torch.tensor([valid, 1 - valid]).log()
        return scores


class ToyTask:
    @staticmethod
    def default_length(example):
        return len(example.get("initial_state", [0, 0]))

    @staticmethod
    def make_prompt(example):
        return ""

    @staticmethod
    def initial_state(example, harness):
        return list(example.get("initial_state", [0, 0]))

    @staticmethod
    def reward(example, output):
        return float(output in {"AX", "BX"})

    @staticmethod
    def cars_context_key(example, harness):
        return "shared"


class CarsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_probability_mass_propagates_to_all_ancestors(self):
        root = _TrieNode()
        root.log_probs = torch.tensor([0.8, 0.2], dtype=torch.float64).log()
        root.log_theta = torch.zeros(2, dtype=torch.float64)
        root.allowed = torch.ones(2, dtype=torch.bool)
        child = _TrieNode()
        child.log_probs = torch.tensor([0.25, 0.75], dtype=torch.float64).log()
        child.log_theta = torch.zeros(2, dtype=torch.float64)
        child.allowed = torch.tensor([True, False])
        _update_trie([(root, 0), (child, 1)], rejected=True)
        self.assertAlmostEqual(math.exp(root.log_mass()), 0.4)
        self.assertAlmostEqual(math.exp(float(root.log_theta[0])), 0.25)
        self.assertEqual(float(child.log_theta[1]), -math.inf)

    def test_samples_follow_conditioned_distribution_and_reuse_trie(self):
        harness = ToyHarness()
        samples = sample_cars(harness, ToyTask(), [{} for _ in range(2000)],
                              config={"batch_size": 64, "max_steps_multiplier": 100})
        self.assertTrue(all(sample["stats"]["accepted"] for sample in samples))
        # P(AX | valid) = .8*.25 / (.8*.25 + .2*.75) = 4/7.
        fraction = sum(sample["output"] == "AX" for sample in samples) / len(samples)
        self.assertAlmostEqual(fraction, 4 / 7, delta=0.04)
        self.assertEqual(harness.evaluations, 3)
        self.assertEqual(sum(sample["stats"]["nfe"] for sample in samples), 3)
        self.assertGreater(sum(sample["stats"]["rejected"] for sample in samples), 0)

    def test_missing_context_hook_isolates_examples(self):
        task = SimpleNamespace(default_length=ToyTask.default_length,
                               make_prompt=ToyTask.make_prompt,
                               reward=lambda example, output: float(output == example["answer"]))
        harness = ToyHarness()
        samples = sample_cars(harness, task, [{"answer": "AX"}, {"answer": "BX"}],
                              config={"max_steps_multiplier": 100})
        self.assertTrue(all(sample["stats"]["accepted"] for sample in samples))
        self.assertEqual([sample["output"] for sample in samples], ["AX", "BX"])
        self.assertGreaterEqual(harness.evaluations, 4)

    def test_fresh_tries_preserve_distribution_from_first_draw(self):
        task = SimpleNamespace(default_length=ToyTask.default_length,
                               make_prompt=ToyTask.make_prompt, reward=ToyTask.reward)
        samples = sample_cars(ToyHarness(), task, [{} for _ in range(1200)],
                              config={"batch_size": 64, "max_steps_multiplier": 100})
        self.assertTrue(all(sample["stats"]["accepted"] for sample in samples))
        fraction = sum(sample["output"] == "AX" for sample in samples) / len(samples)
        self.assertAlmostEqual(fraction, 4 / 7, delta=0.05)

    def test_prefix_oracle_learns_unsampled_invalid_siblings(self):
        task = ToyTask()
        task.cars_token_mask = lambda example, state, harness, pos: [False] * 5
        harness = ToyHarness()
        samples = sample_cars(harness, task, [{}, {}], config={"batch_size": 1})
        self.assertTrue(all(sample["stats"]["exhausted"] for sample in samples))
        self.assertTrue(all(not sample["stats"]["accepted"] for sample in samples))
        self.assertEqual(harness.evaluations, 1)
        self.assertEqual(samples[1]["stats"]["attempts"], 0)

    def test_terminal_sibling_pruning_exhausts_impossible_constraint(self):
        task = ToyTask()
        task.reward = lambda example, output: 0.0
        # A validity hook must not override the configured reward constraint.
        task.terminal_accept = lambda example, output: True
        samples = sample_cars(ToyHarness(), task, [{}], config={"max_steps_multiplier": 100})
        self.assertFalse(samples[0]["stats"]["accepted"])
        self.assertTrue(samples[0]["stats"]["exhausted"])
        self.assertEqual(samples[0]["stats"]["rejected"], 2)

    def test_budget_exhaustion_does_not_force_or_accept_partial_state(self):
        sample = sample_cars(ToyHarness(), ToyTask(), [{}], config={"max_steps": 1})[0]
        self.assertEqual(sample["stats"]["steps"], 1)
        self.assertEqual(sample["stats"]["nfe"], 1)
        self.assertTrue(sample["stats"]["budget_exhausted"])
        self.assertFalse(sample["stats"]["accepted"])
        self.assertEqual(sample["stats"]["forced"], 0)
        self.assertEqual(sample["output"], "")
        self.assertIn("_", sample["last_attempt_output"])

    def test_initial_tokens_and_different_lengths_are_preserved(self):
        examples = [{"initial_state": [2, 0]}, {"initial_state": [1, 3]},
                    {"initial_state": [0]}]
        task = ToyTask()
        task.reward = lambda example, output: float("_" not in output)
        samples = sample_cars(ToyHarness(), task, examples)
        self.assertEqual([len(sample["state"]) for sample in samples], [2, 2, 1])
        self.assertEqual(samples[0]["state"][0], 2)
        self.assertEqual(samples[1]["state"], [1, 3])
        self.assertEqual(samples[1]["stats"]["nfe"], 0)
        self.assertTrue(all(sample["stats"]["accepted"] for sample in samples))
        self.assertEqual(sample_cars(ToyHarness(), task, []), [])

    def test_eos_is_checked_before_remaining_masks(self):
        harness = ToyHarness()
        harness.eos_id = 1
        task = ToyTask()
        task.reward = lambda example, output: float(output.startswith("A"))
        samples = sample_cars(harness, task, [{}], config={"stop_on_eos": True})
        self.assertTrue(samples[0]["stats"]["accepted"])
        self.assertEqual(samples[0]["state"], [1, 0])

    def test_config_and_cli_budget_overrides(self):
        config = {"generation": {"batch_size": 3, "temperature": 0.5},
                  "algorithms": {"value_guidance": {"K": 2}, "CARS": {"reward_threshold": 0.7}}}
        apply_budget_overrides(config, "CARS", n=7)
        cfg = cars_config(config)
        self.assertEqual(cfg["max_steps_multiplier"], 7)
        self.assertEqual(cfg["batch_size"], 3)
        self.assertEqual(cfg["temperature"], 0.5)
        self.assertEqual(cfg["reward_threshold"], 0.7)
        self.assertEqual(value_guidance_config(config, "VGR")["K"], 2)
        with self.assertRaises(ValueError):
            apply_budget_overrides(config, "CARS", n=2, max_steps_multiplier=3)
        with self.assertRaises(ValueError):
            cars_config({"algorithms": {"CARS": {"unknown": 1}}})
        with self.assertRaises(TypeError):
            cars_config({"algorithms": {"CARS": 1}})
        for bad in ({"temperature": 0}, {"temperature": float("nan")},
                    {"max_steps": 0}, {"max_steps_multiplier": 0}, {"batch_size": 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sample_cars(ToyHarness(), ToyTask(), [{}], config=bad)

    def test_qm9_inference_builds_rows_without_verifier(self):
        from src.qm9 import task, utils

        with patch.dict(sys.modules, {"utils": utils}):
            from src.qm9.inference import run_cars

        class SmilesHarness(ToyHarness):
            def decode_state(self, state):
                return "".join("_COCO"[token] for token in state)

        with patch.object(task, "REWARD_THRESHOLD", 0.0):
            rows = run_cars(task, SmilesHarness(), [{"length": 2}],
                            {"algorithms": {"CARS": {"batch_size": 1}}}, rank=1)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["stats"]["accepted"])
        self.assertEqual(rows[0]["reward"], 1.0)
        self.assertTrue(rows[0]["valid"])
        self.assertTrue(rows[0]["pass95"])
        self.assertEqual(len(rows[0]["state_ids"]), 2)
        with patch.object(task, "REWARD_THRESHOLD", 0.0):
            failures = run_cars(task, SmilesHarness(), [{"length": 2}],
                                {"algorithms": {"CARS": {"max_steps": 1}}}, rank=1)
        self.assertFalse(failures[0]["stats"]["accepted"])
        self.assertFalse(failures[0]["valid"])
        self.assertFalse(failures[0]["pass95"])
        self.assertEqual(failures[0]["reward"], 0.0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
