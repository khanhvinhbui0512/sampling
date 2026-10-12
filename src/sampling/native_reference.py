"""Trajectory law of the selected NativeMDLM root reference.

Coordinates serialize a parallel transition: every coordinate uses logits from
the SAME pre-step state. MASK self loops and final denoising are actual edges.
This exposes path likelihoods, not intractable terminal marginal likelihoods.
The baseline decoder is unchanged. Only its explicit-final-denoise protocol is
supported here; the upstream residual-MASK endpoint has a separate identity.
"""

import copy
import math
from dataclasses import dataclass

import torch

from src.models.backbone import loglinear_sigma
from src.sampling.completion import prediction_log_probs
from src.sampling.native import NativeMDLM, absorbing_posterior
from src.types import CostPhase, MaskedState


@dataclass(frozen=True)
class NativePath:
    tokens: tuple
    terminal: MaskedState
    log_probability: float
    log_proposal: float


class NativeReference:
    def __init__(self, policy):
        if not isinstance(policy, NativeMDLM) or not policy.final_denoise:
            raise ValueError("Native reference requires NativeMDLM with explicit final denoise")
        self.policy, self.backbone, self.rules = policy, policy.backbone, policy.rules
        self.policy_id, self.reference_id = policy.policy_id, policy.reference_id
        self.root = MaskedState((self.rules.mask_id,) * self.rules.sequence_length, self.rules)
        self.times = tuple(float(t) for t in torch.linspace(
            1, policy.time_stop, policy.steps + 1, dtype=torch.float64))
        self.max_base_sequences = policy.steps + 1
        self._settings = copy.deepcopy(policy.settings)
        self._backbone_identity = (self.backbone.checkpoint_id, self.backbone.tokenizer_id,
                                   getattr(self.backbone, "precision", "toy"))

    def check_context(self):
        if (self.policy.backbone is not self.backbone or self.policy.settings != self._settings
                or any(getattr(self.policy, name) != self._settings[name]
                       for name in ("steps", "noise_eps", "time_stop", "final_denoise"))
                or self.backbone.time_conditioning != self._settings["time_conditioning"]
                or self.backbone.token_rules != self.rules or self.policy.rules != self.rules
                or self.root.rules != self.rules
                or self.root.token_ids != (self.rules.mask_id,) * self.rules.sequence_length
                or self.policy.policy_id != self.policy_id or self.policy.reference_id != self.reference_id
                or (self.backbone.checkpoint_id, self.backbone.tokenizer_id,
                    getattr(self.backbone, "precision", "toy")) != self._backbone_identity):
            raise ValueError("Native root reference context changed")

    @property
    def protocol(self):
        self.check_context()
        return {"root": list(self.root.token_ids), "settings": copy.deepcopy(self._settings),
                "reference_id": self.reference_id, "policy_id": self.policy_id}

    def cursor(self, path=()):
        self.check_context()
        cursor = NativeCursor(self)
        for token in path:
            cursor.advance(token)
        return cursor

    def draw(self, *, rng, budget, phase=CostPhase.ESTIMATION, reservation=None):
        cursor, tokens, log_p = self.cursor(), [], 0.
        while not cursor.done:
            raw = cursor.next_log_probs(budget=budget, phase=phase, reservation=reservation)
            token = int(torch.multinomial(torch.tensor(raw, dtype=torch.float64).exp(), 1,
                                          generator=rng.torch_generator("cpu"))[0])
            log_p += raw[token]
            tokens.append(token)
            cursor.advance(token)
        return NativePath(tuple(tokens), cursor.observations, log_p, log_p)

    def log_path_probability(self, path, *, budget, phase=CostPhase.IMPORTANCE, reservation=None):
        cursor, log_p = self.cursor(), 0.
        for token in path:
            if cursor.done:
                raise ValueError("Trailing tokens after an absorbing terminal")
            raw = cursor.next_log_probs(budget=budget, phase=phase, reservation=reservation)
            self.rules.validate_token(token)
            if raw[token] == -math.inf:
                return -math.inf
            log_p += raw[token]
            cursor.advance(token)
        if not cursor.done:
            raise ValueError("A path likelihood requires the full terminal trajectory")
        return log_p


class NativeCursor:
    """Small per-path cursor; no logits are queried between coordinate updates."""

    def __init__(self, reference):
        self.reference = reference
        self.pre = reference.root
        self.observations = self.pre
        self.step, self.position = 0, 0
        self.theta, self.probabilities = None, None

    @property
    def done(self):
        return self.position == 0 and self.pre.is_complete

    def next_log_probs(self, *, budget, phase, reservation=None):
        if self.done:
            raise ValueError("No transition after a complete trajectory")
        policy, rules = self.reference.policy, self.reference.rules
        if self.probabilities is None:
            if self.theta is None or policy.backbone.time_conditioning:
                t = self.reference.times[min(self.step, policy.steps)]
                sigma = loglinear_sigma(torch.full((1,), t), policy.noise_eps)
                logits = policy.backbone.logits_at_noise_batch(
                    [self.pre], sigma, budget=budget, phase=phase, reservation=reservation)
                if logits.shape != (1, rules.sequence_length, rules.vocab_size):
                    raise ValueError("Unexpected native reference logits shape")
                self.theta = prediction_log_probs(logits, rules.mask_id).exp()
            tokens = torch.tensor([self.pre.token_ids], dtype=torch.long)
            move_t = (1 - policy.noise_eps) * self.reference.times[self.step]
            move_s = (0. if self.step == policy.steps else
                      (1 - policy.noise_eps) * self.reference.times[self.step + 1])
            self.probabilities = absorbing_posterior(self.theta, tokens, rules.mask_id, move_t, move_s)
        return tuple(float(x) for x in self.probabilities[0, self.position].log())

    def advance(self, token):
        rules = self.reference.rules
        rules.validate_token(token)
        if self.done or self.step > self.reference.policy.steps:
            raise ValueError("Trajectory continues after its endpoint")
        old = self.pre.token_ids[self.position]
        if old != rules.mask_id and token != old:
            raise ValueError("Native trajectories cannot change a known token")
        if self.step == self.reference.policy.steps and token == rules.mask_id:
            raise ValueError("Final denoise cannot retain MASK")
        if old == rules.mask_id and token != rules.mask_id:
            self.observations = self.observations.reveal({self.position: token})
        self.position += 1
        if self.position == rules.sequence_length:
            if self.observations != self.pre:
                self.theta = None
            self.pre = self.observations
            self.step += 1
            self.position, self.probabilities = 0, None
            if self.step > self.reference.policy.steps and not self.done:
                raise ValueError("Native endpoint contains residual MASK")
