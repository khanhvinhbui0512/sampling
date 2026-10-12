"""Separate fixed-count rho completion and native-root conditional MC controls."""

import copy
import math
from statistics import NormalDist

from tqdm.auto import tqdm

from src.sampling.completion import RhoCompletion, cost_since
from src.types import CostPhase, EstimateStatus, ValueEstimate, require_count
from src.budget import BudgetExceeded
from src.verifiers.importance import conditional_ratio
from src.verifiers.interface import Verifier, native_conditional_target, value_target


def bernoulli_uncertainty(successes, draws, confidence=0.95):
    """Plug-in standard error and Wilson interval, including zero/all successes."""
    require_count(draws, "MC draws", positive=True)
    require_count(successes, "MC successes")
    if successes > draws or not 0 < confidence < 1:
        raise ValueError("Invalid Bernoulli summary")
    p = successes / draws
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    denominator = 1 + z * z / draws
    center = (p + z * z / (2 * draws)) / denominator
    radius = z * math.sqrt(p * (1 - p) / draws + z * z / (4 * draws * draws)) / denominator
    return math.sqrt(p * (1 - p) / draws), (max(0., min(p, center - radius)),
                                           min(1., max(p, center + radius)))


class MonteCarloVerifier(Verifier):
    def __init__(self, policy, reward, *, draws=128, draw_batch_size=256, confidence=0.95):
        if not isinstance(policy, RhoCompletion):
            raise ValueError("MC-rho requires the fixed left-to-right RhoCompletion policy")
        require_count(draws, "MC draws", positive=True)
        require_count(draw_batch_size, "MC draw batch size", positive=True)
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        if (policy.rules != reward.token_rules
                or policy.backbone.tokenizer_id != reward.tokenizer_sha256
                or reward.calibration is None
                or policy.backbone.checkpoint_id != reward.calibration.checkpoint_sha256):
            raise ValueError("MC policy and calibrated QM9 reward differ")
        super().__init__(value_target(policy, reward), reward)
        self.policy, self.draws = policy, draws
        self.draw_batch_size, self.confidence = draw_batch_size, confidence

    def required_work(self, states):
        return sum(len(state.masked_positions) for state in states) * self.draws, 0

    def _estimate_batch(self, states, *, rng, budget, reservation):
        return [self._estimate(state, rng=rng, budget=budget, reservation=reservation)
                for state in states]

    def _estimate(self, state, *, rng, budget, reservation):
        before = budget.snapshot().total
        required = len(state.masked_positions) * self.draws
        owned = reservation is None
        held = reservation or budget.reserve_query(label="MC-rho", base_sequences=required)
        if held is None or held.base_remaining < required:
            return ValueEstimate(self.target, EstimateStatus.BUDGET_EXHAUSTED, None,
                                 error="Insufficient budget for the entire fixed-count MC query")
        completed, successes = 0, 0
        try:
            for start in range(0, self.draws, self.draw_batch_size):
                count = min(self.draw_batch_size, self.draws - start)
                batch = self.policy.complete_batch([state] * count, rng=rng, budget=budget,
                                                   phase=CostPhase.ESTIMATION, reservation=held)
                if batch.exhausted or any(not leaf.is_complete for leaf in batch.states):
                    return ValueEstimate(self.target, EstimateStatus.BUDGET_EXHAUSTED, None,
                                         cost=cost_since(budget, before), draws=completed,
                                         error=batch.error or "Incomplete MC completion batch")
                if len(batch.states) != count:
                    raise ValueError("MC completion count differs from the fixed protocol")
                for leaf in batch.states:  # Keep every draw, including duplicates and failures.
                    with budget.oracle_check(phase=CostPhase.ESTIMATION):
                        terminal = self.reward.evaluate(leaf, require_binary=True)
                    successes += terminal.reward
                completed += count
            standard_error, interval = bernoulli_uncertainty(successes, completed, self.confidence)
            return ValueEstimate(self.target, EstimateStatus.ESTIMATED, successes / completed,
                                 cost=cost_since(budget, before), draws=completed,
                                 standard_error=standard_error, interval=interval)
        finally:
            if owned:
                held.close()


class NativeConditionalMonteCarloVerifier(Verifier):
    """Fixed root draws, paired N/D counts; never a fresh completion from z.

The serial path implementation exposes costs plainly. Batching/feasibility must
be profiled before benchmarking; no performance equivalence is claimed.
"""

    def __init__(self, reference, reward, *, draws, min_denominator_hits, confidence=.95, progress=True):
        require_count(draws, "native MC draws", positive=True)
        require_count(min_denominator_hits, "minimum compatible hits", positive=True)
        if min_denominator_hits > draws or not 0 < confidence < 1:
            raise ValueError("Invalid native MC denominator/uncertainty protocol")
        super().__init__(native_conditional_target(reference, reward), reward)
        self.reference, self.draws = reference, draws
        self._calibration = copy.deepcopy(reward.calibration)
        self.min_denominator_hits, self.confidence = min_denominator_hits, confidence
        self.progress, self.summaries = progress, []

    def required_work(self, states):
        self._check_context()
        return sum(not state.is_complete for state in states) * self.draws * self.reference.max_base_sequences, 0

    def _check_context(self):
        if (self.reward.calibration != self._calibration
                or native_conditional_target(self.reference, self.reward) != self.target):
            raise ValueError("Native MC reference/reward context changed")

    def values_for_states(self, states, *, rng, budget, reservation=None):
        self._check_context()
        return super().values_for_states(states, rng=rng, budget=budget, reservation=reservation)

    def _estimate_batch(self, states, *, rng, budget, reservation):
        return [self._estimate(state, rng=rng, budget=budget, reservation=reservation) for state in states]

    def _estimate(self, state, *, rng, budget, reservation):
        before = budget.snapshot().total
        required = self.draws * self.reference.max_base_sequences
        owned = reservation is None
        held = reservation or budget.reserve_query(label="MC-native-conditional", base_sequences=required)
        if held is None or held.base_remaining < required:
            if owned and held is not None:
                held.close()
            return ValueEstimate(self.target, EstimateStatus.BUDGET_EXHAUSTED, None,
                                 error="Entire fixed-count native MC query does not fit budget")
        hits, successes, completed = 0, 0, 0
        try:
            for _ in tqdm(range(self.draws), desc="Native conditional MC", disable=not self.progress):
                draw = self.reference.draw(rng=rng, budget=budget, phase=CostPhase.ESTIMATION, reservation=held)
                event = state.accepts_completion(draw.terminal)
                with budget.oracle_check(phase=CostPhase.ESTIMATION):
                    reward = self.reward.evaluate(draw.terminal, require_binary=True).reward
                hits += int(event)  # Reward-zero leaves still count here.
                successes += int(event) * reward
                completed += 1
            summary = conditional_ratio(successes, self.draws, hits, self.draws,
                                         min_denominator_hits=self.min_denominator_hits,
                                         confidence=self.confidence, denominator_is_one=state == self.reference.root)
            self.summaries.append(summary)
            return ValueEstimate(self.target, EstimateStatus.ESTIMATED if summary.value is not None
                                 else EstimateStatus.FAILED, summary.value, cost=cost_since(budget, before),
                                 draws=completed, interval=summary.interval, error=summary.error)
        except BudgetExceeded as exc:
            return ValueEstimate(self.target, EstimateStatus.BUDGET_EXHAUSTED, None,
                                 cost=cost_since(budget, before), draws=completed, error=str(exc))
        finally:
            if owned:
                held.close()
