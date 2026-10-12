"""One MDM-VGB transition kernel for learned and CARS value providers.

Depth coefficients and sampled, multiplicity-weighted shortlists follow the
pinned QM9 implementation. Terminal-zero candidates have zero weight, including
when a guidance floor is enabled. EOS/PAD normalization, locked input context,
finite search budgets and explicit fallback accounting are SMILES adaptations.
No stationary-law or ideal-value claim is made for this practical shortlist.
"""

from __future__ import annotations

import copy
import itertools
import math
from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch

from src.completion import full_log_probabilities, normalize_eos_suffix
from src.config import config_digest
from src.reward import RewardChecker
from src.schemas import BaseModel, RewardResult, SmilesState, ValueProvider, require_count
from src.tokenizer import SmilesTokenizer, TokenizationError
from src.tracking import BudgetExceeded, CostTracker
from src.values import checked_value, guidance_value, reward_context

SEARCH_VERSION = "mdm-vgb-shortlist-v1"


def validate_search(options: Mapping[str, Any]) -> None:
    for name in ("L_f", "L_b", "B", "N"):
        require_count(options[name], "search." + name, positive=True)
    if options["K"] != "all":
        require_count(options["K"], "search.K", positive=True)
    if (options["version"] != SEARCH_VERSION or options["temperature"] != 1.0
            or options["forward_selection"] not in ("random", "high_conf", "top_margin")
            or options["backward_selection"] not in ("random", "low_conf")
            or options["shortlist"] != "sample_with_replacement"
            or options["L_f"] < options["B"] or options["L_b"] < options["B"]
            or not math.isfinite(options["lambda"]) or options["lambda"] < 0
            or not 0 < options["value_floor"] <= 1
            or type(options["momentum"]) is not bool or type(options["fallback_forward"]) is not bool
            or (options["momentum"] and options["chi"] != 1.0)
            or (not options["momentum"] and options["chi"] is not None)):
        raise ValueError("Unsupported MDM-VGB search/shortlist/floor/momentum settings")


def log_depth_coefficient(k: int, n: int, r: int) -> float:
    if r <= 0 or k < 0 or k > n-r:
        return -math.inf
    return -(math.lgamma(n-r+1) - math.lgamma(k+1) - math.lgamma(n-r-k+1))


@dataclass(frozen=True, slots=True)
class Proposal:
    kind: str
    state: SmilesState
    positions: tuple[int, ...]
    log_weight: float
    value: float


@dataclass(frozen=True, slots=True)
class GenerationAttempt:
    kind: str
    state: SmilesState
    leaf_visits: int = 0
    smiles: str | None = None
    reward_result: RewardResult | None = None
    reason: str | None = None
    stats: dict[str, int] | None = None


def output_state(state: SmilesState, tokenizer: SmilesTokenizer, checker: RewardChecker) -> tuple[str, RewardResult]:
    reward = checker.evaluate_state(state, tokenizer)
    try:
        text = tokenizer.decode_state(state)
    except (TokenizationError, ValueError):
        text = "".join(tokenizer.vocabulary[token] for token in state.token_ids)
    return text, reward


class MDMVGBSampler:
    def __init__(self, model: BaseModel, provider: ValueProvider, tokenizer: SmilesTokenizer,
                 checker: RewardChecker, options: Mapping[str, Any], *, seed: int = 0,
                 tracker: CostTracker | None = None) -> None:
        validate_search(options)
        require_count(seed, "search seed")
        self.model, self.provider, self.tokenizer, self.checker = model, provider, tokenizer, checker
        self.options, self.tracker = copy.deepcopy(dict(options)), tracker
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        if model.vocab_size != tokenizer.vocab_size or model.special_tokens != tokenizer.special_tokens:
            raise ValueError("Search MDLM and tokenizer differ")
        self.metadata = {"version": SEARCH_VERSION, "model": model.model_id, "tokenizer": tokenizer.fingerprint,
                         "reward": reward_context(checker), "search": self.options,
                         "value_target": provider.target, "provider": getattr(provider, "context_hash", provider.target),
                         "value_provider": copy.deepcopy(getattr(provider, "metadata", {"target": provider.target})),
                         "step_budget": "editable_width * N", "N_is_NFE": False,
                         "confidence": "raw_denoiser_on_current_state", "input_observations": "preserve_all",
                         "eos_suffix": "deterministic_mask_to_PAD", "fallback_policy": "full_vocabulary_left_to_right",
                         "ideal_value_claim": False}
        self.context_hash = config_digest(self.metadata)
        self.current: SmilesState | None = None
        self.phase = "idle"
        self.direction = "down"
        self.steps = 0
        self.limit = 0
        self.stats: dict[str, int] = {}
        self.result: GenerationAttempt | None = None

    def _measure(self, name: str):
        return nullcontext() if self.tracker is None else self.tracker.measure(name)

    def start(self, root: SmilesState) -> None:
        if (root.special_tokens != self.tokenizer.special_tokens or len(root.token_ids) != self.tokenizer.max_length
                or any(token >= self.tokenizer.vocab_size for token in root.token_ids)):
            raise ValueError("Search root differs from tokenizer/width")
        root = normalize_eos_suffix(root)
        # Input observations are conditioning context. Only newly generated
        # coordinates may be remasked by this root-start sampler.
        self.current = replace(root, locked=tuple(locked or not masked for locked, masked in zip(root.locked, root.mask, strict=True)))
        self.steps, self.limit = 0, sum(not locked for locked in self.current.locked) * self.options["N"]
        self.phase, self.direction, self.result = "search", "down", None
        self.stats = {"forward": 0, "backward": 0, "switches": 0, "stays": 0,
                      "fallback_forward": 0, "fallback_nfe": 0}

    def _blocks(self, positions: list[int], scores: torch.Tensor, limit: int, mode: str) -> list[tuple[int, ...]]:
        if not positions:
            return []
        if mode == "random":
            order = torch.randperm(len(positions), generator=self.generator).tolist()
            selected = [positions[index] for index in order[:limit]]
        else:
            selected = sorted(positions, key=lambda pos: (-float(scores[pos]), pos))[:limit]
        r = min(self.options["B"], len(positions))
        return [tuple(selected[start:start+r]) for start in range(0, len(selected)-r+1, r)]

    @torch.inference_mode()
    def proposals(self, state: SmilesState) -> tuple[list[Proposal], list[Proposal]]:
        if state.is_terminal:
            raise ValueError("Terminal reward is handled before constructing search proposals")
        if self.tracker is not None:
            self.tracker.require_budget()
        with self._measure("search_proposals"):
            logits_fn = getattr(self.model, "proposal_logits_batch", self.model.logits_batch)
            logits = logits_fn([state])[0].detach().to(device="cpu", dtype=torch.float64).clone()
            if (logits.shape != (len(state.token_ids), self.model.vocab_size)
                    or torch.isnan(logits).any() or torch.isposinf(logits).any()):
                raise ValueError("Invalid proposal logits")
            logits[:, state.special_tokens.mask_id] = -torch.inf
            if not torch.isfinite(logits).any(-1).all():
                raise ValueError("Search coordinate has no clean-token probability")
            logs = logits.log_softmax(-1)
            masked = [pos for pos in state.masked_positions if not state.locked[pos]]
            revealed = [pos for pos, token in enumerate(state.token_ids)
                        if token != state.special_tokens.mask_id and not state.locked[pos]]
            n, k = len(masked) + len(revealed), len(revealed)
            confidence = logs.exp().max(-1).values
            if self.options["forward_selection"] == "top_margin":
                top = logs.exp().topk(min(2, self.model.vocab_size), -1).values
                confidence = top[:, 0] - (top[:, 1] if top.shape[1] > 1 else 0)
            backward_scores = -logs[torch.arange(len(state.token_ids)), torch.tensor(state.token_ids)].exp()
            fblocks = self._blocks(masked, confidence, self.options["L_f"], self.options["forward_selection"])
            bblocks = self._blocks(revealed, backward_scores, self.options["L_b"], self.options["backward_selection"])
            raw = checked_value(self.provider.value(state))
            current_value = guidance_value(raw, state, self.options["value_floor"])
            fwd_meta = []
            for block in fblocks:
                r = len(block)
                scale = log_depth_coefficient(k, n, r) + math.log(math.comb(len(masked), r) / len(fblocks))
                if self.options["K"] == "all":
                    clean = [token for token in range(self.model.vocab_size) if token != state.special_tokens.mask_id]
                    if len(clean)**r > 100000:
                        raise ValueError("Full block enumeration exceeds 100000 assignments; choose a finite K")
                    assignments = [(tokens, math.fsum(float(logs[pos, token]) for pos, token in zip(block, tokens, strict=True)))
                                   for tokens in itertools.product(clean, repeat=r)]
                else:
                    draws = [torch.multinomial(logs[pos].exp(), self.options["K"], replacement=True,
                                               generator=self.generator).tolist() for pos in block]
                    counts = Counter(zip(*draws, strict=True))
                    # Sampling already supplies the base-policy factor. Applying
                    # another q factor would square it and change the kernel.
                    assignments = [(tokens, math.log(count / self.options["K"])) for tokens, count in counts.items()]
                for tokens, log_q in assignments:
                    if not math.isfinite(log_q):
                        continue
                    child = normalize_eos_suffix(state.with_tokens(dict(zip(block, tokens, strict=True))))
                    fwd_meta.append((child, block, scale + log_q))
            parents = [state.remask(block) for block in bblocks]
            children_values = self.provider.values([child for child, _, _ in fwd_meta])
            parent_values = self.provider.values(parents) if self.options["lambda"] > 0 else [1.0] * len(parents)
            forward = []
            for (child, block, weight), value in zip(fwd_meta, children_values, strict=True):
                value = guidance_value(value, child, self.options["value_floor"])
                gate = 0.0 if child.is_terminal else self.options["lambda"] * math.log(current_value)
                log_weight = -math.inf if value == 0.0 else weight + math.log(value) + gate
                forward.append(Proposal("forward", child, block, log_weight, value))
                if child.is_terminal and self.tracker is not None:
                    # Retain every complete shortlist candidate, including
                    # terminal-zero and memoized candidates, before selection.
                    self.tracker.event("search_terminal_candidate", state=child.to_dict(), reward=value,
                                       positions=list(block), log_weight=log_weight if math.isfinite(log_weight) else None)
            backward = []
            for parent, block, value in zip(parents, bblocks, parent_values, strict=True):
                r = len(block)
                weight = log_depth_coefficient(k-r, n, r) + math.log(math.comb(len(revealed), r) / len(bblocks))
                parent_value = guidance_value(value, parent, self.options["value_floor"])
                weight += math.log(current_value) + self.options["lambda"] * math.log(parent_value)
                backward.append(Proposal("backward", parent, block, weight, parent_value))
        return forward, backward

    def _choose(self, forward: list[Proposal], backward: list[Proposal]) -> Proposal | None:
        proposals = forward + backward
        finite = [item.log_weight for item in proposals if math.isfinite(item.log_weight)]
        if not finite:
            return None
        offset = max(finite)
        weights = [math.exp(item.log_weight-offset) for item in proposals]
        if self.options["momentum"]:
            fmass, bmass = math.fsum(weights[:len(forward)]), math.fsum(weights[len(forward):])
            if self.direction == "down":
                proposals, weights = forward, weights[:len(forward)]
                switch = max(bmass-fmass, 0.0)
            else:
                proposals, weights = backward, weights[len(forward):]
                switch = max(fmass-bmass, 0.0)
            proposals = [*proposals, Proposal("switch", self.current, (), 0.0, 0.0)]
            weights = [*weights, switch]
        if not math.fsum(weights):
            return None
        index = int(torch.multinomial(torch.tensor(weights, dtype=torch.float64), 1, generator=self.generator))
        return proposals[index]

    def _complete(self) -> GenerationAttempt:
        with self._measure("reward"):
            text, reward = output_state(self.current, self.tokenizer, self.checker)
        if self.tracker is not None:
            self.tracker.record_leaf_visit()
            self.tracker.record_complete_candidate()
        self.phase = "done"
        self.result = GenerationAttempt("complete", self.current, 1, text, reward, stats=dict(self.stats))
        return self.result

    @torch.inference_mode()
    def step(self) -> GenerationAttempt | None:
        if self.phase == "idle":
            raise ValueError("Call start(root) before search")
        if self.phase == "done":
            return self.result
        if self.tracker is not None:
            self.tracker.require_budget()
        if self.current.is_terminal:
            return self._complete()
        if self.steps >= self.limit:
            if not self.options["fallback_forward"]:
                self.phase = "done"
                self.result = GenerationAttempt("budget_exhausted", self.current, reason="search_step_budget", stats=dict(self.stats))
                return self.result
            self.phase = "fallback"
        if self.phase == "fallback":
            before = 0 if self.tracker is None else self.tracker.raw_nfe
            with self._measure("fallback_forward"):
                state = self.current
                pos = state.masked_positions[0]
                logs = full_log_probabilities(self.model, [state])[0, pos].double().cpu()
                token = int(torch.multinomial(logs.exp(), 1, generator=self.generator))
                self.current = normalize_eos_suffix(state.with_tokens({pos: token}))
            self.stats["fallback_forward"] += 1
            self.stats["fallback_nfe"] += 0 if self.tracker is None else self.tracker.raw_nfe-before
        else:
            forward, backward = self.proposals(self.current)
            with self._measure("search_transition"):
                choice = self._choose(forward, backward)
                if choice is None:
                    self.stats["stays"] += 1
                    if self.options["momentum"]:
                        self.direction = "up" if self.direction == "down" else "down"
                elif choice.kind == "switch":
                    self.direction = "up" if self.direction == "down" else "down"
                    self.stats["switches"] += 1
                else:
                    self.current = choice.state
                    self.stats[choice.kind] += 1
            self.steps += 1
            if self.tracker is not None:
                self.tracker.increment("markov_steps")
        if self.tracker is not None:
            self.tracker.event("search_step", step=self.steps, phase=self.phase, direction=self.direction,
                               state=self.current.to_dict(), stats=dict(self.stats))
        return None

    def attempt(self, *, on_step: Callable[[], None] | None = None) -> GenerationAttempt:
        try:
            while self.phase != "done":
                result = self.step()
                if on_step is not None:
                    on_step()
                if result is not None:
                    return result
            return self.result
        except BudgetExceeded as exc:
            return GenerationAttempt("budget_exhausted", self.current, reason=str(exc), stats=dict(self.stats))

    def state_dict(self) -> dict[str, Any]:
        return {"context_hash": self.context_hash, "rng": self.generator.get_state(), "phase": self.phase,
                "direction": self.direction, "steps": self.steps, "limit": self.limit, "stats": dict(self.stats),
                "state": None if self.current is None else self.current.to_dict(),
                "result": None if self.result is None else asdict(self.result),
                "provider": self.provider.state_dict()}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload["context_hash"] != self.context_hash or payload["phase"] not in ("idle", "search", "fallback", "done"):
            raise ValueError("Search snapshot context/phase differs")
        require_count(payload["steps"], "saved search steps")
        require_count(payload["limit"], "saved search limit")
        if (payload["direction"] not in ("up", "down") or payload["steps"] > payload["limit"]
                or (payload["phase"] == "idle") != (payload["state"] is None)
                or (payload["phase"] == "done") != (payload["result"] is not None)):
            raise ValueError("Inconsistent search snapshot state/direction/budget")
        for name, count in payload["stats"].items():
            require_count(count, "saved " + name)
        self.provider.load_state_dict(payload["provider"])
        self.generator.set_state(payload["rng"].cpu())
        self.current = None if payload["state"] is None else SmilesState.from_dict(payload["state"])
        if self.current is not None and (len(self.current.token_ids) != self.tokenizer.max_length
                                        or self.current.special_tokens != self.tokenizer.special_tokens
                                        or any(token >= self.tokenizer.vocab_size for token in self.current.token_ids)):
            raise ValueError("Search snapshot tokenizer/width differs")
        self.phase, self.direction = payload["phase"], payload["direction"]
        self.steps, self.limit, self.stats = payload["steps"], payload["limit"], dict(payload["stats"])
        self.result = None
        if payload["result"] is not None:
            data = dict(payload["result"])
            data["state"] = SmilesState.from_dict(data["state"])
            if data["reward_result"] is not None:
                data["reward_result"] = RewardResult.from_dict(data["reward_result"])
            self.result = GenerationAttempt(**data)


__all__ = ["GenerationAttempt", "MDMVGBSampler", "Proposal", "log_depth_coefficient", "validate_search"]
