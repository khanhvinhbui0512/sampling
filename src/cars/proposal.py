"""Draw from a FROZEN CARS proposal, without updating its failure set.

For B_W > 0, Q_W(y|z) = q_rho(y|z) * 1[y avoids W] / B_W. B_W is
remaining mass, not a success probability. Discovery/acceptance/value estimation
are separate Step 8 operations. This module never retries until success.
"""

import math
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from src.budget import BudgetExceeded
from src.sampling.completion import cost_since
from src.types import CostPhase, CostRecord, MaskedState


@dataclass(frozen=True)
class ProposalDraw:
    state: MaskedState
    status: str  # drawn, support_exhausted, budget_exhausted
    log_q: float | None
    log_proposal: float | None
    log_root_mass: float
    cost: CostRecord
    error: str | None = None

    @property
    def root_mass(self):
        return math.exp(self.log_root_mass)

    @property
    def mass_underflow(self):
        return math.isfinite(self.log_root_mass) and self.root_mass == 0.


class FrozenProposal:
    def __init__(self, completion, trie):
        if not trie.is_frozen:
            raise ValueError("Freeze the discovery trie before proposal evaluation")
        if (completion.rules != trie.initial_state.rules or completion.policy_id != trie.policy_id
                or completion.reference_id != trie.reference_id):
            raise ValueError("Proposal and trie must use the same q_rho context")
        # Own the snapshot, so replacing attributes on the caller's trie cannot
        # alter a proposal halfway through estimation. No mutation API is used.
        self.completion, self.trie = completion, trie.freeze()

    def next_log_probs(self, path, *, budget, phase=CostPhase.ESTIMATION, reservation=None):
        path = tuple(path)
        state = self.trie.state_at(path)
        if state.is_complete or self.trie.log_mass(path) == -math.inf:
            raise ValueError("No proposal transition from a terminal or exhausted prefix")
        raw = self.trie.raw_log_probs(path)
        if raw is None:
            raw = self.completion.next_log_probs(
                [state], budget=budget, phase=phase, reservation=reservation)[0]
        raw = torch.as_tensor(raw, dtype=torch.float64).cpu()
        scores = raw + torch.tensor(self.trie.child_log_masses(path), dtype=torch.float64)
        normalizer = torch.logsumexp(scores, dim=0)
        if not torch.isfinite(normalizer):
            raise FloatingPointError("Positive surviving mass has no finite proposal transition")
        if not math.isclose(float(normalizer), self.trie.log_mass(path), rel_tol=0., abs_tol=1e-9):
            raise FloatingPointError("Cached surviving mass and proposal normalization differ")
        return raw, scores - normalizer

    def log_prob(self, leaf, *, budget, phase=CostPhase.ESTIMATION, reservation=None):
        """Score Q_W directly for enumeration checks; all fresh forwards count."""
        if self.trie.support_exhausted:
            raise ValueError("Q_W is undefined when B_W=0")
        if not self.trie.initial_state.accepts_completion(leaf):
            return -math.inf
        path, total = (), 0.
        for position in self.trie.positions:
            token = leaf.token_ids[position]
            if self.trie.log_mass(path + (token,)) == -math.inf:
                return -math.inf
            _, proposal = self.next_log_probs(path, budget=budget, phase=phase, reservation=reservation)
            total += float(proposal[token])
            if total == -math.inf:
                return total
            path += (token,)
        return total

    def draw(self, *, rng, budget, phase=CostPhase.ESTIMATION, reservation=None, progress=True):
        before = budget.snapshot().total
        state, path, log_q, log_proposal = self.trie.initial_state, (), 0., 0.
        log_b = self.trie.log_mass()

        def finish(status, error=None):
            result = ProposalDraw(state, status, log_q if status == "drawn" else None,
                                  log_proposal if status == "drawn" else None, log_b,
                                  cost_since(budget, before), error)
            if progress:
                print(f"CARS proposal: {status}, base_calls={result.cost.base_calls}, "
                      f"log_B={log_b:.6g}")
            return result

        if self.trie.support_exhausted:
            return finish("support_exhausted", "All q_rho support has been certified unsuccessful")
        with tqdm(total=len(self.trie.positions), desc="CARS proposal", leave=False,
                  disable=not progress) as bar:
            for position in self.trie.positions:
                try:
                    raw, proposal = self.next_log_probs(path, budget=budget, phase=phase,
                                                         reservation=reservation)
                except BudgetExceeded as exc:
                    return finish("budget_exhausted", str(exc))
                # Normalize in log space before exponentiating: even log_B=-1000
                # can produce an ordinary categorical distribution. Float64 CPU
                # categorical draws follow the project's q_rho convention.
                token = int(torch.multinomial(proposal.exp(), 1,
                                              generator=rng.torch_generator("cpu"))[0])
                log_q += float(raw[token])
                log_proposal += float(proposal[token])
                state = state.reveal({position: token})
                path += (token,)
                bar.update(1)
        if not math.isclose(log_proposal, log_q - log_b, rel_tol=0., abs_tol=1e-8):
            raise FloatingPointError("Proposal failed the q_rho/B_W telescoping check")
        return finish("drawn")
