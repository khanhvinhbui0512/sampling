"""Small finite CPU diagnostic; no QM9 weights, network downloads or test imports."""

import math
from collections import defaultdict
from dataclasses import asdict
from itertools import product
from types import SimpleNamespace

import torch
from tqdm.auto import tqdm

from src.budget import BudgetTracker
from src.cars.cache import CARSCache, NativeCARSCache
from src.cars.estimator import CARSValueEstimator
from src.cars.native import NativeCARSValueEstimator
from src.cars.oracle import FiniteLanguageOracle
from src.config import load_config
from src.experiments.run import RecordedVerifier, record, source_hashes
from src.io import RunArtifacts, file_hash, read_json, report_result, write_json, write_jsonl
from src.models.backbone import loglinear_sigma
from src.randomness import RNGPool
from src.sampling.completion import RhoCompletion, prediction_log_probs
from src.sampling.native import NativeMDLM, absorbing_posterior
from src.sampling.native_reference import NativeReference
from src.types import CostPhase, MaskedState, TokenRules
from src.verifiers.cars import CARSVerifier, NativeCARSVerifier
from src.verifiers.interface import native_conditional_target, value_target
from src.verifiers.monte_carlo import MonteCarloVerifier, NativeConditionalMonteCarloVerifier


class FiniteBackbone:
    """Prefix-only joint marginals deliberately ignore observations after MASK."""

    checkpoint_id = "finite-prefix-toy-weights-v1"
    tokenizer_id = "finite-prefix-toy-tokenizer-v1"
    precision = "toy"
    time_conditioning = False
    device = torch.device("cpu")

    def __init__(self, fixture):
        self.token_rules = TokenRules(**fixture["token_rules"])
        self.leaves = fixture["leaves"]

    def logits_batch(self, states, *, budget, phase, reservation=None):
        rules = self.token_rules
        with budget.base_forward(batch_size=len(states), phase=phase, reservation=reservation):
            probabilities = torch.zeros(len(states), rules.sequence_length, rules.vocab_size, dtype=torch.float64)
            for row, state in enumerate(states):
                length = state.masked_positions[0] if state.masked_positions else rules.sequence_length
                leaves = [leaf for leaf in self.leaves if tuple(leaf["tokens"][:length]) == state.token_ids[:length]]
                total = sum(leaf["probability"] for leaf in leaves)
                if total <= 0:
                    raise ValueError("Unsupported toy prefix")
                for leaf in leaves:
                    for position, token in enumerate(leaf["tokens"]):
                        probabilities[row, position, token] += leaf["probability"] / total
            return probabilities.log()

    def logits_at_noise_batch(self, states, sigma, **kwargs):
        return self.logits_batch(states, **kwargs)


def native_joint(reference, budget):
    """Enumerate entire parallel steps independently of NativeCursor/trie code."""
    policy, rules = reference.policy, reference.rules
    joint = {reference.root.token_ids: 1.}
    for step in range(policy.steps + 1):
        next_joint = defaultdict(float)
        for tokens, mass in joint.items():
            state = MaskedState(tokens, rules)
            if state.is_complete:
                next_joint[tokens] += mass
                continue
            t = reference.times[step]
            logits = reference.backbone.logits_at_noise_batch(
                [state], loglinear_sigma(torch.full((1,), t), policy.noise_eps), budget=budget, phase=CostPhase.ESTIMATION)
            theta = prediction_log_probs(logits, rules.mask_id).exp()
            move_t = (1 - policy.noise_eps) * t
            move_s = 0. if step == policy.steps else (1 - policy.noise_eps) * reference.times[step + 1]
            probabilities = absorbing_posterior(theta, torch.tensor([tokens]), rules.mask_id, move_t, move_s)[0]
            choices = [torch.nonzero(row > 0, as_tuple=False).flatten().tolist() for row in probabilities]
            for outcome in product(*choices):
                p = math.prod(float(probabilities[i, token]) for i, token in enumerate(outcome))
                next_joint[outcome] += mass * p
        joint = dict(next_joint)
    if not math.isclose(sum(joint.values()), 1., abs_tol=1e-10) or any(rules.mask_id in tokens for tokens in joint):
        raise AssertionError("Toy native joint is not a normalized terminal law")
    return joint


def rho_completions(policy, query, budget):
    joint = {}

    def visit(state, mass):
        if state.is_complete:
            joint[state.token_ids] = mass
            return
        raw = policy.next_log_probs([state], budget=budget, phase=CostPhase.ESTIMATION)[0]
        for token, log_p in enumerate(raw.tolist()):
            if math.isfinite(log_p):
                visit(state.reveal({state.masked_positions[0]: token}), mass * math.exp(log_p))

    visit(query, 1.)
    return joint


def check_toy_values(*, attempt_id, value_law="native-conditional", config_path="configs/value_checks.yaml",
                     run_id="toy-check-values", execution_metadata=None):
    if value_law not in ("native-conditional", "rho"):
        raise ValueError("Toy checks support native-conditional or rho")
    config = load_config(config_path)
    settings, fixture = config["toy"], read_json(config["toy"]["fixture"])
    backbone = FiniteBackbone(fixture)
    policy = NativeMDLM(backbone, **{key: settings[key] for key in ("steps", "noise_eps", "time_stop", "final_denoise")})
    reference, rho = NativeReference(policy), RhoCompletion(backbone)
    successful = {tuple(leaf["tokens"]) for leaf in fixture["leaves"] if leaf["reward"]}

    class Reward:
        token_rules, tokenizer_sha256 = backbone.token_rules, backbone.tokenizer_id
        calibration = SimpleNamespace(calibration_id="finite-toy-pass95-v1", threshold=.5,
                                      checkpoint_sha256=backbone.checkpoint_id,
                                      reference_id=reference.reference_id, policy_id=reference.policy_id)

        def evaluate(self, state, *, require_binary=True):
            if not state.is_complete or state.rules != self.token_rules:
                raise ValueError("Toy rewards require a complete compatible state")
            return SimpleNamespace(reward=int(state.token_ids in successful))

    reward = Reward()
    enumeration_budget = BudgetTracker(base_sequence_limit=None)
    joint = native_joint(reference, enumeration_budget) if value_law == "native-conditional" else None
    if value_law == "native-conditional":
        target = native_conditional_target(reference, reward)
        mc = NativeConditionalMonteCarloVerifier(reference, reward, draws=settings["mc_draws"],
                                                min_denominator_hits=settings["min_denominator_hits"], progress=False)
        estimator = NativeCARSValueEstimator(
            reference, reward, discovery_attempts=settings["discovery_attempts"],
            numerator_draws=settings["numerator_draws"], denominator_draws=settings["denominator_draws"],
            max_nodes=settings["max_nodes"], min_denominator_hits=settings["min_denominator_hits"],
            confidence=settings["confidence"])
        cache = NativeCARSCache(estimator.cache_metadata, reference=reference, reward=reward,
                                max_entries=1, run_id=run_id + ":" + attempt_id)
        cars = NativeCARSVerifier(estimator, cache=cache, progress=False)
    else:
        target = value_target(rho, reward)
        mc = MonteCarloVerifier(rho, reward, draws=settings["mc_draws"])
        oracle = FiniteLanguageOracle(backbone.token_rules, successful)
        estimator = CARSValueEstimator(rho, reward, oracle, discovery_attempts=settings["discovery_attempts"],
                                       draws=settings["numerator_draws"], mode="cars", max_nodes=settings["max_nodes"],
                                       confidence=settings["confidence"])
        cache = CARSCache(estimator.cache_metadata, max_entries=1, run_id=run_id + ":" + attempt_id)
        cars = CARSVerifier(estimator, cache=cache, progress=False)
    adapters = {name: RecordedVerifier(inner, "cpu") for name, inner in (("mc", mc), ("cars", cars))}
    protocol = {"mode": "toy", "value_law": value_law, "target": asdict(target), "settings": config,
                "fixture_sha256": file_hash(settings["fixture"]), "source_sha256": source_hashes(),
                "qm9_gate_passed": False, "reference": "exact_finite_enumeration"}
    attempt = RunArtifacts.create(config["check_output_dir"], run_id=run_id, attempt_id=attempt_id,
                                  config=protocol, target_id=target.target_id, reference_id=target.reference_id)
    attempt.manifest.update(task="check_toy_values", target=asdict(target))
    pool = RNGPool(config["seed"], "finite-value-check-v1")
    try:
        write_json(attempt.path("protocol.json"), protocol)
        attempt.register("protocol.json")

        def rows():
            for query_row in tqdm(fixture["states"], desc="Finite enumeration value check", unit="state"):
                query = MaskedState(tuple(query_row["tokens"]), backbone.token_rules)
                law = joint if joint is not None else rho_completions(rho, query, enumeration_budget)
                d = sum(p for tokens, p in law.items() if query.accepts_completion(MaskedState(tokens, query.rules)))
                n = sum(p for tokens, p in law.items() if tokens in successful and query.accepts_completion(MaskedState(tokens, query.rules)))
                exact = n / d if d else None
                results = {}
                for name, adapter in adapters.items():
                    adapter.scope = query_row["name"]
                    budget = BudgetTracker(base_sequence_limit=None)
                    value = adapter.values_for_states([query], rng=pool.stream(query_row["name"] + ":" + name), budget=budget)[0]
                    results[name] = {"estimate": record(value), "queries": adapter.records,
                                     "budget": record(budget.snapshot()), "difference_from_exact":
                                     value.value - exact if exact is not None and value.value is not None else None}
                    adapter.records = []
                diagnostic = None
                if value_law == "rho":
                    compatible = [leaf for leaf in fixture["leaves"] if query.accepts_completion(MaskedState(tuple(leaf["tokens"]), query.rules))]
                    diagnostic = sum(leaf["probability"] * leaf["reward"] for leaf in compatible) / sum(leaf["probability"] for leaf in compatible)
                yield {**query_row, "exact_value": exact, "exact_numerator": n, "exact_denominator": d,
                       "pi_rho_conditional_diagnostic_not_v_rho": diagnostic, "estimators": results}

        write_jsonl(attempt.path("values.jsonl"), rows(), progress=False)
        attempt.register("values.jsonl")
        write_json(attempt.path("execution.json"), {"backend": "local", **(execution_metadata or {}),
                   "device": "cpu", "enumeration_budget": record(enumeration_budget.snapshot()),
                   "states": len(fixture["states"]), "qm9_feasibility_established": False})
        attempt.register("execution.json")
        attempt.complete(required_artifacts=("protocol.json", "values.jsonl", "execution.json"))
    except Exception as exc:
        attempt.fail(f"{type(exc).__name__}: {exc}")
        raise
    report_result(f"Checked {len(fixture['states'])} toy states against exact {value_law} enumeration; manifest={attempt.path('manifest.json')}")
    return {"manifest": attempt.path("manifest.json"), "states": len(fixture["states"])}
