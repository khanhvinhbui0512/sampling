"""The project's q_rho: reveal the leftmost remaining MASK, keeping all context.

This is a new completion policy for comparing MC/CARS. Its law for a state with
future observations is not assumed to be a conditional of any native MDLM joint.
"""

import math
from dataclasses import asdict, dataclass, field

import torch

from src.budget import BudgetExceeded
from src.io import payload_hash
from src.types import CostRecord, SampleResult, SampleStatus, TargetKind, ValueTarget


def cost_since(budget, before):
    after = budget.snapshot().total
    return CostRecord(**{name: getattr(after, name) - getattr(before, name)
                         for name in CostRecord.__dataclass_fields__})


def prediction_log_probs(logits, mask_id, temperature=1.0):
    """Normalize only non-MASK tokens; -inf is allowed for exact toy zeros."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scores = torch.as_tensor(logits).detach().double().cpu().clone() / temperature
    scores[..., mask_id] = -torch.inf
    if torch.isnan(scores).any() or torch.isposinf(scores).any():
        raise FloatingPointError("Invalid token logits")
    if not torch.isfinite(scores).any(dim=-1).all():
        raise ValueError("Every predicted position needs non-MASK support")
    return scores.log_softmax(dim=-1)


@dataclass(frozen=True)
class CompletionBatch:
    states: tuple
    log_probabilities: tuple[float, ...]
    exhausted: bool = False
    error: str | None = None
    snapshots: dict[int, tuple] = field(default_factory=dict)


class RhoCompletion:
    def __init__(self, backbone, *, temperature=1.0, target=None):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.backbone = backbone
        self.rules = backbone.token_rules
        self.temperature = temperature
        self.policy_id = "rho-ltr-v1:" + payload_hash({
            "order": "left_to_right", "temperature": temperature,
            "support": "all_non_mask_tokens", "stop_on_eos": False,
            "token_rules": asdict(self.rules), "categorical": "cpu_multinomial_float64",
        })
        self.reference_id = "pi-rho-v1:" + payload_hash({
            "checkpoint": backbone.checkpoint_id, "tokenizer": backbone.tokenizer_id,
            "policy": self.policy_id, "precision": getattr(backbone, "precision", "toy"),
            "time_conditioning": backbone.time_conditioning,
        })
        self.target = target or ValueTarget("unscored:" + self.reference_id, TargetKind.RHO,
                                           self.reference_id, self.policy_id, "unscored")
        if (self.target.kind != TargetKind.RHO or self.target.reference_id != self.reference_id
                or self.target.policy_id != self.policy_id):
            raise ValueError("Completion target differs from q_rho's reference/policy")

    def next_log_probs(self, states, *, budget, phase, reservation=None):
        if any(state.rules != self.rules or state.is_complete for state in states):
            raise ValueError("Expected incomplete states with the backbone's token rules")
        if not states:
            return torch.empty((0, self.rules.vocab_size), dtype=torch.float64)
        logits = torch.as_tensor(self.backbone.logits_batch(
            states, budget=budget, phase=phase, reservation=reservation,
        ))
        if logits.shape != (len(states), self.rules.sequence_length, self.rules.vocab_size):
            raise ValueError("Unexpected backbone logits shape")
        positions = [state.masked_positions[0] for state in states]
        return prediction_log_probs(logits[torch.arange(len(states)), positions],
                                    self.rules.mask_id, self.temperature)

    def complete_batch(self, states, *, rng, budget, phase, reservation=None, snapshot_steps=()):
        current = list(states)
        if any(state.rules != self.rules for state in current):
            raise ValueError("State token rules differ from q_rho")
        log_probabilities = [0.0] * len(current)
        steps = max((len(state.masked_positions) for state in current), default=0)
        if any(type(step) is not int or not 0 <= step < steps for step in snapshot_steps):
            raise ValueError("Snapshot indices must be pre-reveal steps in [0, initial MASK count)")
        snapshots = {}
        for step in range(steps):
            if step in snapshot_steps:
                snapshots[step] = tuple(current)
            active = [i for i, state in enumerate(current) if not state.is_complete]
            if not active:
                continue
            try:
                log_probs = self.next_log_probs([current[i] for i in active], budget=budget,
                                                phase=phase, reservation=reservation)
            except BudgetExceeded as exc:
                return CompletionBatch(tuple(current), tuple(log_probabilities), True, str(exc), snapshots)
            choices = torch.multinomial(log_probs.exp(), 1, generator=rng.torch_generator("cpu"))[:, 0]
            for row, index in enumerate(active):
                position = current[index].masked_positions[0]
                token = int(choices[row])
                current[index] = current[index].reveal({position: token})
                log_probabilities[index] += float(log_probs[row, token])
        return CompletionBatch(tuple(current), tuple(log_probabilities), snapshots=snapshots)

    def complete(self, state, *, rng, budget, phase, reservation=None):
        before = budget.snapshot().total
        batch = self.complete_batch([state], rng=rng, budget=budget, phase=phase,
                                     reservation=reservation)
        status = SampleStatus.COMPLETED if batch.states[0].is_complete else SampleStatus.BUDGET_EXHAUSTED
        return SampleResult("completion", batch.states[0], self.target, status,
                            cost=cost_since(budget, before), error=batch.error)

    def log_prob_batch(self, completions, given, *, budget, phase, reservation=None):
        """Score q_rho(y|z) by replaying that exact z, including known future tokens."""
        if len(completions) != len(given):
            raise ValueError("Pair each completion with one conditioning state")
        current = list(given)
        totals = torch.zeros(len(current), dtype=torch.float64)
        for i, (state, leaf) in enumerate(zip(current, completions)):
            if state.rules != self.rules or leaf.rules != self.rules:
                raise ValueError("Token rules differ from q_rho")
            if not state.accepts_completion(leaf):
                totals[i] = -torch.inf
        while True:
            active = [i for i, state in enumerate(current)
                      if not state.is_complete and torch.isfinite(totals[i])]
            if not active:
                return totals
            probabilities = self.next_log_probs([current[i] for i in active], budget=budget,
                                                 phase=phase, reservation=reservation)
            for row, index in enumerate(active):
                position = current[index].masked_positions[0]
                token = completions[index].token_ids[position]
                totals[index] += probabilities[row, token]
                current[index] = current[index].reveal({position: token})

    def log_prob(self, completion, given, *, budget, phase, reservation=None):
        return float(self.log_prob_batch([completion], [given], budget=budget, phase=phase,
                                         reservation=reservation)[0])
