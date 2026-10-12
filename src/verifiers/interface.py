"""Ordered value queries with exact terminal anchoring and explicit targets."""

from src.io import payload_hash
from src.sampling.completion import cost_since
from src.types import CostPhase, EstimateStatus, TargetKind, ValueEstimate, ValueTarget


def value_target(policy, reward, *, kind=TargetKind.RHO, label_protocol=None):
    if reward.calibration is None:
        raise ValueError("Value estimation requires a locked Pass@95 calibration")
    calibration = reward.calibration
    identity = {"kind": kind, "reference": policy.reference_id, "policy": policy.policy_id,
                "reward": "pass95:" + calibration.calibration_id,
                "label_protocol": label_protocol}
    return ValueTarget("value:" + payload_hash(identity), kind, policy.reference_id,
                       policy.policy_id, identity["reward"], calibration.calibration_id)


def native_conditional_target(reference, reward):
    """One target for neural, root MC and CARS, independent of their estimators."""
    from src.sampling.native_reference import NativeReference

    if not isinstance(reference, NativeReference):
        raise ValueError("Native V* requires the explicit native root reference")
    reference.check_context()
    policy = reference.policy
    if (reference.rules != reward.token_rules or reward.calibration is None
            or policy.backbone.tokenizer_id != reward.tokenizer_sha256
            or policy.backbone.checkpoint_id != reward.calibration.checkpoint_sha256):
        raise ValueError("Native reference and calibrated reward differ")
    for name, expected in (("reference_id", reference.reference_id), ("policy_id", reference.policy_id)):
        if hasattr(reward.calibration, name) and getattr(reward.calibration, name) != expected:
            raise ValueError("Native conditional target differs from the locked calibration BASE reference")
    return value_target(reference, reward, kind=TargetKind.REFERENCE_CONDITIONAL,
                        label_protocol={"law": "native_root_terminal_given_observations_v1"})


class Verifier:
    """Return one ValueEstimate per input, including duplicate states, in order.

    Sum result costs for query cost. A shared neural forward is charged to the
    first result in that chunk; it is never multiplied by the number of states.
    The BudgetTracker remains the authoritative cost ledger.
    """

    def __init__(self, target, reward):
        if reward.calibration is None or target.calibration_id != reward.calibration.calibration_id:
            raise ValueError("Verifier and terminal reward calibration differ")
        if target.reward_id != "pass95:" + reward.calibration.calibration_id:
            raise ValueError("Verifier requires the binary Pass@95 target")
        self.target, self.reward = target, reward
        self.rules = reward.token_rules

    def required_work(self, states):
        """Upper bound (base sequences, verifier sequences) for one whole group.

        Guided samplers reserve this group before any query starts. Future
        adapters must supply a bound; unavailable values must not become zero.
        Complete leaves need no model work.
        """
        raise NotImplementedError

    def values_for_states(self, states, *, rng, budget, reservation=None):
        if any(state.rules != self.rules for state in states):
            raise ValueError("Verifier state token rules differ")
        results, pending = [None] * len(states), []
        for index, state in enumerate(states):
            if not state.is_complete:
                pending.append(index)
                continue
            before = budget.snapshot().total
            with budget.oracle_check(phase=CostPhase.VERIFICATION):
                terminal = self.reward.evaluate(state, require_binary=True)
            value = float(terminal.reward)
            results[index] = ValueEstimate(self.target, EstimateStatus.EXACT, value,
                                           cost=cost_since(budget, before), standard_error=0.,
                                           interval=(value, value))
        if pending:
            estimates = self._estimate_batch([states[i] for i in pending], rng=rng,
                                             budget=budget, reservation=reservation)
            if len(estimates) != len(pending):
                raise ValueError("Verifier returned a different number of estimates")
            for index, estimate in zip(pending, estimates):
                if estimate.target != self.target:
                    raise ValueError("Verifier changed the value target")
                results[index] = estimate
        return results

    def _estimate_batch(self, states, *, rng, budget, reservation):
        raise NotImplementedError
