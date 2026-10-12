"""Conditional ratios with explicit denominator and support diagnostics.

Finite-sample ratios are generally biased. Bounds use fixed-work Hoeffding
intervals and a union bound; an unresolved denominator is unavailable, not zero.
For importance sampling the caller must establish global proposal support and
a finite log weight bound BEFORE use. An observed maximum is not such a bound.
"""

import math
from dataclasses import dataclass

from src.sampling.completion import cost_since
from src.types import CostPhase, EstimateStatus, TargetKind, ValueEstimate, require_count
from src.verifiers.interface import native_conditional_target, value_target


@dataclass(frozen=True)
class ConditionalSummary:
    value: float | None
    interval: tuple | None
    numerator: float
    denominator: float
    denominator_interval: tuple
    ess: float
    error: str | None = None


def _finish_ratio(n, d, n_interval, d_interval, ess, minimum):
    if ess < minimum or d <= 0. or d_interval[0] <= 0.:
        return ConditionalSummary(None, None, n, d, d_interval, ess,
                                  "Unresolved denominator or insufficient compatible ESS/hits")
    value = n / d
    if not 0 <= value <= 1:
        return ConditionalSummary(None, None, n, d, d_interval, ess,
                                  "Finite-sample ratio outside [0, 1]; no silent clipping")
    # Intersect the confidence set with the known parameter range, not weights.
    interval = (max(0., min(value, n_interval[0] / d_interval[1])),
                min(1., max(value, n_interval[1] / d_interval[0])))
    return ConditionalSummary(value, interval, n, d, d_interval, ess)


def conditional_ratio(successes, numerator_draws, compatible, denominator_draws, *,
                      log_mass=0., confidence=.95, min_denominator_hits,
                      denominator_is_one=False):
    """Frozen-CARS numerator B*mean(tau*event); unpruned root denominator.

Works also with paired root MC counts. The bound needs no independence between
the two means. CARS discovery and estimation must still be split independently.
"""
    for name, count in (("numerator draws", numerator_draws), ("denominator draws", denominator_draws),
                        ("minimum denominator hits", min_denominator_hits)):
        require_count(count, name, positive=True)
    require_count(successes, "successes")
    require_count(compatible, "compatible draws")
    if (successes > numerator_draws or compatible > denominator_draws
            or not 0 < confidence < 1 or math.isnan(log_mass) or log_mass > 0):
        raise ValueError("Invalid conditional count/mass protocol")
    mass = math.exp(log_mass)
    if math.isfinite(log_mass) and mass == 0.:
        return ConditionalSummary(None, None, 0., compatible / denominator_draws, (0., 1.),
                                  float(compatible), "Positive surviving mass underflowed")
    n, d = mass * successes / numerator_draws, compatible / denominator_draws
    if successes and mass and n == 0.:
        return ConditionalSummary(None, None, n, d, (0., 1.), float(compatible),
                                  "Positive numerator underflowed")
    radius_n = math.sqrt(math.log(4 / (1 - confidence)) / (2 * numerator_draws))
    radius_d = math.sqrt(math.log(4 / (1 - confidence)) / (2 * denominator_draws))
    n_interval = (mass * max(0., successes / numerator_draws - radius_n),
                  mass * min(1., successes / numerator_draws + radius_n))
    if denominator_is_one:
        if compatible != denominator_draws:
            raise ValueError("Known root event D=1 contradicts denominator draws")
        d_interval = (1., 1.)
    else:
        d_interval = (max(0., d - radius_d), min(1., d + radius_d))
    return _finish_ratio(n, d, n_interval, d_interval, float(compatible), min_denominator_hits)


def importance_summary(log_weights, compatible, rewards, *, log_weight_bound,
                       min_ess, confidence=.95):
    """Self-normalized joint ratio; bounded, un-clipped weights and ESS on C(z)."""
    count = len(log_weights)
    require_count(count, "importance draws", positive=True)
    if (len(compatible) != count or len(rewards) != count or not 0 < confidence < 1
            or not math.isfinite(log_weight_bound) or not math.isfinite(min_ess) or min_ess <= 0):
        raise ValueError("Invalid importance protocol or missing global weight bound")
    scaled = []
    for log_w, event, reward in zip(log_weights, compatible, rewards):
        if (math.isnan(log_w) or log_w == math.inf or log_w > log_weight_bound + 1e-12
                or type(event) is not bool or reward not in (0, 1)):
            raise ValueError("Unsupported/non-finite weight, invalid event, or violated bound")
        weight = math.exp(log_w - log_weight_bound)
        if math.isfinite(log_w) and weight == 0.:
            raise FloatingPointError("Importance weight underflow; no silent removal")
        scaled.append(weight if event else 0.)
    d = sum(scaled) / count
    n = sum(w * r for w, r in zip(scaled, rewards)) / count
    # ESS is invariant to a common scale. Normalize again for this diagnostic:
    # positive weights can survive exp() while their squares underflow to zero.
    maximum = max(scaled)
    if maximum:
        relative = [w / maximum for w in scaled]
        ess = math.fsum(relative) ** 2 / math.fsum(w * w for w in relative)
    else:
        ess = 0.
    radius = math.sqrt(math.log(4 / (1 - confidence)) / (2 * count))
    return _finish_ratio(n, d, (max(0., n - radius), min(1., n + radius)),
                         (max(0., d - radius), min(1., d + radius)), ess, min_ess)


@dataclass(frozen=True)
class ImportanceRun:
    estimate: ValueEstimate
    summary: ConditionalSummary
    log_weights: tuple


def _importance_result(target, summary, weights, budget, before):
    estimate = ValueEstimate(target, EstimateStatus.ESTIMATED if summary.value is not None
                             else EstimateStatus.FAILED, summary.value,
                             cost=cost_since(budget, before), draws=len(weights),
                             interval=summary.interval, error=summary.error)
    return ImportanceRun(estimate, summary, tuple(weights))


def rho_importance_diagnostic(reference, reward, given, completions, *, budget,
                              support_verified, log_weight_bound, min_ess, confidence=.95,
                              reservation=None):
    """pi_rho/q_rho diagnostic, never labeled native V*. Scoring is charged."""
    if support_verified is not True:
        raise ValueError("A verified global proposal support argument is required")
    from src.sampling.completion import RhoCompletion
    from src.sampling.reference import RhoJointReference

    if (not isinstance(reference, RhoJointReference) or not isinstance(reference.policy, RhoCompletion)
            or given.rules != reference.policy.rules or reward.token_rules != reference.policy.rules
            or reference.policy.backbone.tokenizer_id != reward.tokenizer_sha256
            or reward.calibration is None
            or reference.policy.backbone.checkpoint_id != reward.calibration.checkpoint_sha256):
        raise ValueError("Rho joint diagnostic policy/reward/context differs")
    target = value_target(reference.policy, reward, kind=TargetKind.REFERENCE_CONDITIONAL,
                          label_protocol={"law": "rho_root_terminal_given_observations_v1"})
    before, weights, rewards = budget.snapshot().total, [], []
    for leaf in completions:
        if not given.accepts_completion(leaf):
            raise ValueError("Draw is outside q_rho's completion support")
        weights.append(reference.log_weight(leaf, given, budget=budget,
                                             phase=CostPhase.IMPORTANCE, reservation=reservation))
        with budget.oracle_check(phase=CostPhase.IMPORTANCE):
            rewards.append(reward.evaluate(leaf, require_binary=True).reward)
    summary = importance_summary(weights, [True] * len(weights), rewards,
                                 log_weight_bound=log_weight_bound, min_ess=min_ess,
                                 confidence=confidence)
    return _importance_result(target, summary, weights, budget, before)


def native_importance_diagnostic(reference, reward, given, paths, *, budget,
                                 support_verified, log_weight_bound, min_ess, confidence=.95,
                                 reservation=None):
    """Full path P_native/Q weights. Q must cover all compatible native paths.

Not for a reward-pruned Q denominator: that proposal omits failed compatible
paths and violates support. Native CARS instead uses separate unpruned D draws.
"""
    if support_verified is not True:
        raise ValueError("A verified global proposal support argument is required")
    target = native_conditional_target(reference, reward)
    if given.rules != reference.rules:
        raise ValueError("Native importance query rules differ")
    before, weights, events, rewards = budget.snapshot().total, [], [], []
    for path in paths:
        if not math.isfinite(path.log_proposal):
            raise ValueError("A sampled path needs finite positive proposal probability")
        log_p = reference.log_path_probability(path.tokens, budget=budget,
                                               phase=CostPhase.IMPORTANCE, reservation=reservation)
        cursor = reference.cursor(path.tokens)
        if not cursor.done or cursor.observations != path.terminal:
            raise ValueError("Path and terminal differ")
        weights.append(log_p - path.log_proposal)
        events.append(given.accepts_completion(path.terminal))
        with budget.oracle_check(phase=CostPhase.IMPORTANCE):
            rewards.append(reward.evaluate(path.terminal, require_binary=True).reward)
    summary = importance_summary(weights, events, rewards, log_weight_bound=log_weight_bound,
                                 min_ess=min_ess, confidence=confidence)
    return _importance_result(target, summary, weights, budget, before)
