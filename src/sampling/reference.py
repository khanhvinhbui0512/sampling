"""A tractable joint pi_rho defined by completion from the fully masked root.

Joint scoring must start at that root. Scoring from an arbitrary observed state
instead computes q_rho, which can differ from the conditional of this joint.
"""

import math

from src.types import CostPhase, MaskedState


class RhoJointReference:
    def __init__(self, completion_policy):
        self.policy = completion_policy
        self.reference_id = completion_policy.reference_id
        rules = completion_policy.rules
        self.root = MaskedState((rules.mask_id,) * rules.sequence_length, rules)

    def log_prob_batch(self, completions, *, budget, phase=CostPhase.IMPORTANCE, reservation=None):
        return self.policy.log_prob_batch(completions, [self.root] * len(completions),
                                           budget=budget, phase=phase, reservation=reservation)

    def log_prob(self, completion, *, budget, phase=CostPhase.IMPORTANCE, reservation=None):
        return float(self.log_prob_batch([completion], budget=budget, phase=phase,
                                         reservation=reservation)[0])

    def log_weight(self, completion, given, *, budget, phase=CostPhase.IMPORTANCE,
                   reservation=None):
        log_q = self.policy.log_prob(completion, given, budget=budget, phase=phase,
                                      reservation=reservation)
        if not math.isfinite(log_q):
            raise ValueError("Importance scoring requires a compatible positive-proposal leaf")
        return self.log_prob(completion, budget=budget, phase=phase,
                             reservation=reservation) - log_q
