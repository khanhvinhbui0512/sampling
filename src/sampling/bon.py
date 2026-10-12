"""Best-of-N: paper Algorithm 1, with explicit binary/QED ranking and ties."""

from tqdm.auto import tqdm

from src.sampling.terminal import completion_work, finish_run
from src.types import require_count


class BoNSampler:
    method_id = "bon"

    def __init__(self, base_sampler, *, N=16, ranking="binary_reward", tie="first"):
        require_count(N, "N", positive=True)
        if ranking not in ("binary_reward", "qed") or tie != "first":
            raise ValueError("BoN uses binary_reward/qed ranking and first-index ties")
        if base_sampler.reward.calibration is None:
            raise ValueError("BoN comparison requires a locked Pass@95 calibration")
        self.base_sampler, self.N, self.ranking = base_sampler, N, ranking
        self.reward, self.target = base_sampler.reward, base_sampler.target

    def sample(self, initial_state, *, sample_id, rng, budget):
        return self.run(initial_state, sample_id=sample_id, rng=rng, budget=budget).result

    def run(self, initial_state, *, sample_id, rng, budget):
        if initial_state.rules != self.reward.token_rules:
            raise ValueError("BoN state/reward token rules differ")
        before, trace = budget.snapshot().total, []
        metadata = {"method": self.method_id, "N": self.N, "ranking": self.ranking,
                    "tie": "first", "base_method": self.base_sampler.method_id,
                    "completion_policy_id": self.base_sampler.policy_id,
                    "selected_index": None, "completed_candidates": 0,
                    "initial_rng_state": rng.state_dict()}
        states = [initial_state] * self.N
        held = budget.reserve_query(label="BoN-all-N", base_sequences=completion_work(self.base_sampler, states))
        if held is None:
            return finish_run(initial_state, sample_id=sample_id, target=self.target,
                              reward=self.reward, budget=budget, before=before, trace=trace,
                              metadata=metadata, rng=rng, error="Insufficient budget for all N candidates")
        with held, tqdm(total=self.N, desc="BoN completions", leave=False) as progress:
            batch = self.base_sampler.generate_batch(states, rng=rng, budget=budget, reservation=held)
            progress.update(sum(state.is_complete for state in batch.states))
        for index, (state, terminal) in enumerate(zip(batch.states, batch.terminals)):
            trace.append({"index": index, "tokens": state.token_ids,
                          "valid": terminal.valid if terminal else None,
                          "qed": terminal.score if terminal else None,
                          "reward": terminal.reward if terminal else None})
        metadata["completed_candidates"] = sum(state.is_complete for state in batch.states)
        if (batch.exhausted or len(batch.states) != self.N
                or any(terminal is None for terminal in batch.terminals)):
            # An incomplete group is never silently reported as best-of-N.
            return finish_run(initial_state, sample_id=sample_id, target=self.target,
                              reward=self.reward, budget=budget, before=before, trace=trace,
                              metadata=metadata, rng=rng, error=batch.error or "Incomplete BoN group")
        key = (lambda i: batch.terminals[i].reward) if self.ranking == "binary_reward" else (
            lambda i: batch.terminals[i].score)
        winner = max(range(self.N), key=key)  # Python max retains the first tied index.
        metadata["selected_index"] = winner
        return finish_run(batch.states[winner], sample_id=sample_id, target=self.target,
                          reward=self.reward, budget=budget, before=before, trace=trace,
                          metadata=metadata, rng=rng, terminal=batch.terminals[winner])
