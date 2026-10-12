# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kijung Jeon

"""Finite-budget singleton VGB; shared walk bookkeeping for VGR and Momentum.

Adapted from MDM-VGB vgb.py/vgb_terminal.py, using paper Definition 4.2
weights. Success leaves stop immediately; zero/failed leaves use explicitly
labelled escapes. At a step/work cap, the selected BASE completes unguided.
This wrapper is a practical sampler, not the paper's stationary chain.
"""

import math
import time

from tqdm.auto import tqdm

from src.budget import BudgetExceeded
from src.randomness import RNGStream, derive_seed
from src.sampling.candidates import CandidateBuilder, CandidateSettings, ValueTable, ValueUnavailable
from src.sampling.terminal import backtrack, completion_work, finish_run, weighted_index
from src.types import CostPhase, SampleStatus, require_count


class VGBSampler:
    method_id = "vgb"
    allow_backtracking = True
    lifted = False

    def __init__(self, backbone, verifier, completion_sampler, *, candidates=None,
                 step_limit=512, time_limit_seconds=None):
        require_count(step_limit, "step_limit")
        if time_limit_seconds is not None and (not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0):
            raise ValueError("time_limit_seconds must be positive or None")
        reward = completion_sampler.reward
        if (verifier.reward is not reward or verifier.rules != backbone.token_rules
                or completion_sampler.generator.backbone is not backbone):
            raise ValueError("Walk, verifier and completion must share one backbone/reward")
        if (reward.calibration is None or reward.calibration.checkpoint_sha256 != backbone.checkpoint_id
                or reward.tokenizer_sha256 != backbone.tokenizer_id):
            raise ValueError("Guidance requires this backbone's calibrated reward/tokenizer")
        self.backbone, self.verifier, self.reward = backbone, verifier, reward
        self.completion_sampler, self.target = completion_sampler, verifier.target
        self.builder = CandidateBuilder(backbone, candidates or CandidateSettings())
        self.step_limit, self.time_limit_seconds = step_limit, time_limit_seconds

    def _settings(self):
        return {}

    def _transition(self, candidates, direction, rng):
        index = weighted_index([candidate.weight for candidate in candidates], rng)
        if index is None:
            return "zero_mass", None, direction
        selected = candidates[index]
        return selected.direction, selected.state, direction

    def sample(self, initial_state, *, sample_id, rng, budget):
        return self.run(initial_state, sample_id=sample_id, rng=rng, budget=budget).result

    def run(self, initial_state, *, sample_id, rng, budget):
        if initial_state.rules != self.backbone.token_rules:
            raise ValueError("Walk state/backbone token rules differ")
        before, state, trace = budget.snapshot().total, initial_state, []
        steps = 0
        editable = tuple(initial_state.masked_positions)
        direction = "down" if self.lifted else None
        estimation_rng = RNGStream(derive_seed(rng.seed, "guided-values", sample_id))
        values = ValueTable(self.verifier, estimation_rng, budget)
        metadata = {"method": self.method_id, "candidates": self.builder.metadata(),
                    "step_limit": self.step_limit, "time_limit_seconds": self.time_limit_seconds,
                    "base_method": self.completion_sampler.method_id,
                    "completion_policy_id": self.completion_sampler.policy_id,
                    "completion_reference_id": self.completion_sampler.reference_id,
                    "value_target": self.target.target_id, "leaf_accept": "pass95",
                    "value_freeze": "per_run_full_state", "stationarity_claim": False,
                    "initial_tokens": state.token_ids, "initial_rng_state": rng.state_dict(),
                    "fallback_used": False, **self._settings()}
        terminal_cache = {}

        def terminal_for(leaf):
            if leaf.token_ids not in terminal_cache:
                with budget.oracle_check(phase=CostPhase.SAMPLING):
                    terminal_cache[leaf.token_ids] = self.reward.evaluate(leaf, require_binary=True)
            return terminal_cache[leaf.token_ids]

        def finish(reason=None, terminal=None, status=SampleStatus.BUDGET_EXHAUSTED):
            metadata["value_queries"] = len(values.estimates)
            metadata["value_cache_hits"] = values.cache_hits
            metadata["steps"] = steps
            # Full-state records retain target/uncertainty/cost for later writers.
            metadata["value_records"] = tuple((tokens, estimate) for tokens, estimate in values.estimates.items())
            return finish_run(state, sample_id=sample_id, target=self.target, reward=self.reward,
                              budget=budget, before=before, trace=trace, metadata=metadata,
                              rng=rng, direction=direction, terminal=terminal, error=reason,
                              failure_status=status, estimation_rng=estimation_rng)

        if state.is_complete:
            return finish(terminal=terminal_for(state))
        # Reserve for the INITIAL number of editable positions: a later
        # backtrack can increase the remaining masks again.
        held = budget.reserve_completion(base_sequences=completion_work(self.completion_sampler, [state]))
        if held is None:
            return finish("Insufficient budget to reserve unguided completion")
        started, reason = time.monotonic(), "step_limit"

        def timed_out():
            return self.time_limit_seconds is not None and time.monotonic() - started >= self.time_limit_seconds

        with held, tqdm(total=self.step_limit, desc=self.method_id, leave=False) as progress:
            for step in range(self.step_limit):
                if timed_out():
                    return finish("time_limit", status=SampleStatus.FAILED)
                record = {"step": step, "tokens": state.token_ids, "direction": direction}
                if state.is_complete:
                    terminal = terminal_for(state)
                    if terminal.reward == 1 or not self.allow_backtracking:
                        return finish(terminal=terminal)
                    parent = backtrack(state, editable, rng)
                    if parent is None:
                        return finish("failed_leaf_no_editable_position", terminal=terminal)
                    state = parent
                    direction = "up" if self.lifted else None
                    record.update(event="failed_leaf_backtrack", next_tokens=state.token_ids,
                                  valid=terminal.valid, qed=terminal.score, reward=terminal.reward)
                else:
                    if self.lifted and all(state.token_ids[p] == state.rules.mask_id for p in editable):
                        direction = "down"
                    record["direction"] = direction
                    try:
                        candidates = self.builder.build(state, editable=editable, values=values,
                                                        rng=rng, budget=budget,
                                                        backward=self.allow_backtracking)
                    except BudgetExceeded as exc:
                        reason = str(exc)
                        trace.append({**record, "event": "value_or_base_budget"})
                        break
                    except ValueUnavailable as exc:
                        trace.append({**record, "event": "value_unavailable"})
                        return finish(str(exc), status=SampleStatus.FAILED)
                    record["edges"] = tuple({"tokens": c.state.token_ids, "direction": c.direction,
                                               "position": c.position, "multiplicity": c.multiplicity,
                                               "proposal_mass": c.proposal_mass, "value": c.value,
                                               "weight": c.weight} for c in candidates)
                    event, next_state, direction = self._transition(candidates, direction, rng)
                    if event == "zero_mass":
                        next_state = backtrack(state, editable, rng) if self.allow_backtracking else None
                        if next_state is None:
                            reason = "zero_mass_no_parent"
                            trace.append({**record, "event": reason})
                            break
                        event = "zero_mass_backtrack"
                        direction = "up" if self.lifted else None
                    if next_state is not None:
                        state = next_state
                    record.update(event=event, next_tokens=state.token_ids, next_direction=direction)
                trace.append(record)
                steps += 1
                progress.update(1)
            if timed_out():
                return finish("time_limit", status=SampleStatus.FAILED)
            if state.is_complete:
                return finish(terminal=terminal_for(state))
            # No verifier query or shortlisted action is used in this completion.
            metadata["fallback_used"] = True
            batch = self.completion_sampler.generate_batch([state], rng=rng, budget=budget,
                                                            phase=CostPhase.COMPLETION, reservation=held)
            state = batch.states[0]
            trace.append({"event": "reserved_unguided_completion", "reason": reason,
                          "next_tokens": state.token_ids})
            if timed_out():
                return finish("time_limit", terminal=batch.terminals[0], status=SampleStatus.FAILED)
            return finish(batch.error or reason, terminal=batch.terminals[0])
