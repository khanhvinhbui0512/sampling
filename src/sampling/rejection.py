"""RS, leaf-only ARS and oracle CARS on the SAME fixed q_rho policy.

Tries survive attempts and calls in the same context. CARS learns only after a
complete pre-acceptance draw, so adaptation cannot change that draw's law.
Accepted outputs are generation results, never inputs to the value estimator.
"""

from dataclasses import asdict

from tqdm.auto import tqdm

from src.cars.estimator import adaptive_draw, calibration_metadata, check_rho_reward
from src.cars.oracle import QM9Oracle
from src.cars.proposal import FrozenProposal
from src.cars.trie import FailureTrie
from src.sampling.completion import cost_since
from src.sampling.terminal import SamplingRun
from src.types import CostPhase, SampleResult, SampleStatus, require_count
from src.verifiers.interface import value_target


class RejectionSampler:
    def __init__(self, policy, reward, *, method, max_attempts, oracle=None, max_nodes=None):
        check_rho_reward(policy, reward)
        require_count(max_attempts, "maximum rejection attempts", positive=True)
        if max_nodes is not None:
            require_count(max_nodes, "max_nodes", positive=True)
        if method not in ("rs", "ars", "cars"):
            raise ValueError("Choose rs, ars or cars")
        if oracle is None and method != "rs":
            oracle = QM9Oracle(reward, mode="leaf_only" if method == "ars" else "token_constraints")
        if oracle is not None and (oracle.rules != policy.rules
                                  or (hasattr(oracle, "reward") and oracle.reward is not reward)):
            raise ValueError("Rejection oracle differs from the completion/reward context")
        if method == "cars" and getattr(oracle, "mode", None) == "leaf_only":
            raise ValueError("leaf_only must be labelled ars")
        self.policy, self.reward, self.oracle = policy, reward, oracle
        self.method, self.method_id = method, method + "_rho"
        self.max_attempts, self.max_nodes = max_attempts, max_nodes
        self.target = value_target(policy, reward)
        self._calibration = calibration_metadata(reward)
        self._tries, self._capacity_frozen = {}, set()

    def _trie(self, state):
        key = state.token_ids
        if key not in self._tries:
            self._tries[key] = FailureTrie(
                state, policy_id=self.policy.policy_id, reference_id=self.policy.reference_id,
                oracle_id=self.oracle.oracle_id, max_nodes=self.max_nodes)
        return self._tries[key]

    def snapshot(self, state):
        """A caller may persist this snapshot; generation keeps its mutable copy."""
        if state.rules != self.policy.rules:
            raise ValueError("Snapshot token rules differ")
        trie = self._tries.get(state.token_ids)
        return None if trie is None else trie.freeze()

    def run(self, initial_state, *, sample_id, rng, budget, progress=True):
        check_rho_reward(self.policy, self.reward)
        if (initial_state.rules != self.policy.rules
                or calibration_metadata(self.reward) != self._calibration):
            raise ValueError("Rejection context/calibration changed")
        before = budget.snapshot().total
        state, terminal, trace = initial_state, None, []
        trie = self._trie(initial_state) if self.method != "rs" and not state.is_complete else None

        def finish(status, reason):
            molecule = terminal.molecule if terminal is not None else None
            result = SampleResult(
                sample_id, state, self.target, status, cost=cost_since(budget, before),
                smiles=molecule.smiles if molecule else None,
                canonical_smiles=molecule.canonical_smiles if molecule else None,
                valid=terminal.valid if terminal is not None else None,
                score=terminal.score if terminal is not None else None,
                reward=terminal.reward if terminal is not None else None,
                error=reason if status != SampleStatus.COMPLETED else None)
            metadata = {"method": self.method_id, "attempts": len(trace),
                        "success": status == SampleStatus.COMPLETED and result.reward == 1,
                        "stop_reason": reason, "max_attempts": self.max_attempts,
                        "capacity_frozen": initial_state.token_ids in self._capacity_frozen,
                        "log_surviving_mass": (None if trie is not None and trie.support_exhausted
                                               else trie.log_mass() if trie is not None else 0.),
                        "support_exhausted": trie.support_exhausted if trie is not None else False,
                        "snapshot_id": trie.freeze().snapshot_id if trie is not None else None,
                        "target": asdict(self.target)}
            if progress:
                tqdm.write(f"{self.method_id} {sample_id}: {status.value}, attempts={len(trace)}, "
                           f"Pass@95={result.reward}, base_calls={result.cost.base_calls}")
            return SamplingRun(result, tuple(trace), metadata, {"sampler": rng.state_dict()})

        if state.is_complete:
            with budget.oracle_check(phase=CostPhase.SAMPLING):
                terminal = self.reward.evaluate(state, require_binary=True)
            return finish(SampleStatus.COMPLETED if terminal.reward else SampleStatus.FAILED,
                          "successful_terminal" if terminal.reward else "terminal_reward_zero")
        for _ in tqdm(range(self.max_attempts), desc=self.method_id, leave=False, disable=not progress):
            if trie is not None and trie.support_exhausted:
                return finish(SampleStatus.FAILED, "certified_support_exhausted")
            # Conservative whole-attempt admission. Unused held work is released;
            # cache lookups and remapping never count as model forwards.
            held = budget.reserve_query(label=self.method_id,
                                        base_sequences=len(initial_state.masked_positions))
            if held is None:
                return finish(SampleStatus.BUDGET_EXHAUSTED, "insufficient_whole_attempt_budget")
            draw_before = budget.snapshot().total
            with held:
                if self.method == "rs":
                    batch = self.policy.complete_batch([initial_state], rng=rng, budget=budget,
                                                       phase=CostPhase.SAMPLING, reservation=held)
                    state = batch.states[0]
                    record = {"status": "drawn" if state.is_complete else "budget_exhausted",
                              "log_q": batch.log_probabilities[0],
                              "log_proposal": batch.log_probabilities[0], "log_root_mass": 0.,
                              "pruned_cones": 0, "capacity_reached": False}
                    error = batch.error
                elif initial_state.token_ids in self._capacity_frozen:
                    draw = FrozenProposal(self.policy, trie.freeze()).draw(
                        rng=rng, budget=budget, phase=CostPhase.SAMPLING, reservation=held, progress=False)
                    state, record, error = draw.state, asdict(draw), draw.error
                    record.update(pruned_cones=0, capacity_reached=True)
                else:
                    draw = adaptive_draw(self.policy, trie, self.oracle, mode=self.method,
                                         rng=rng, budget=budget, phase=CostPhase.SAMPLING, reservation=held)
                    state, record, error = draw.state, asdict(draw), draw.error
                    if draw.capacity_reached:
                        self._capacity_frozen.add(initial_state.token_ids)
                terminal = None
                if record["status"] == "drawn":
                    with budget.oracle_check(phase=CostPhase.SAMPLING):
                        terminal = self.reward.evaluate(state, require_binary=True)
                    if terminal.reward not in (0, 1):
                        raise ValueError("Rejection sampling requires binary reward")
                record.update(tokens=list(state.token_ids), reward=terminal.reward if terminal else None,
                              cost=asdict(cost_since(budget, draw_before)))
                trace.append(record)
            if record["status"] != "drawn":
                status = (SampleStatus.BUDGET_EXHAUSTED if record["status"] == "budget_exhausted"
                          else SampleStatus.FAILED)
                return finish(status, error or record["status"])
            if terminal.reward == 1:
                return finish(SampleStatus.COMPLETED, "accepted")
        return finish(SampleStatus.FAILED, "attempt_limit")

    def sample(self, initial_state, *, sample_id, rng, budget):
        return self.run(initial_state, sample_id=sample_id, rng=rng, budget=budget).result
