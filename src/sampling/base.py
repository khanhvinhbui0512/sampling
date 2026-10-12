"""BASE-native and BASE-rho share scoring and realized-cost accounting."""

from dataclasses import dataclass

from src.io import payload_hash
from src.sampling.completion import RhoCompletion, cost_since
from src.sampling.native import NativeMDLM
from src.types import CostPhase, CostRecord, SampleResult, SampleStatus, TargetKind, ValueTarget


@dataclass(frozen=True)
class BaseBatch:
    states: tuple
    terminals: tuple
    cost: CostRecord  # A batched forward is counted once, not once per output row.
    exhausted: bool
    error: str | None


class BaseSampler:
    def __init__(self, backbone, reward, *, method="base_native", native_settings=None,
                 temperature=1.0):
        if reward.token_rules != backbone.token_rules or reward.tokenizer_sha256 != backbone.tokenizer_id:
            raise ValueError("BASE and reward must use the same tokenizer/token rules")
        calibration = reward.calibration
        if calibration is not None and calibration.checkpoint_sha256 != backbone.checkpoint_id:
            raise ValueError("Calibration was generated with a different backbone")
        self.reward, self.method_id = reward, method
        if method == "base_native":
            self.generator = NativeMDLM(backbone, **(native_settings or {}))
            kind = TargetKind.NATIVE_ROLLOUT
        elif method == "base_rho":
            self.generator = RhoCompletion(backbone, temperature=temperature)
            kind = TargetKind.RHO
        else:
            raise ValueError("BASE method must be base_native or base_rho")
        self.policy_id = self.generator.policy_id
        self.reference_id = self.generator.reference_id
        reward_id = "qed_or_zero" if calibration is None else "pass95:" + calibration.calibration_id
        identity = {"reference": self.reference_id, "policy": self.policy_id, "reward": reward_id}
        self.target = ValueTarget("base-target:" + payload_hash(identity), kind,
                                  self.reference_id, self.policy_id, reward_id,
                                  calibration.calibration_id if calibration else None)

    def generate_batch(self, states, *, rng, budget, phase=CostPhase.SAMPLING,
                       reservation=None):
        before = budget.snapshot().total
        if self.method_id == "base_native":
            batch = self.generator.sample_batch(states, rng=rng, budget=budget, phase=phase,
                                                  reservation=reservation)
        else:
            batch = self.generator.complete_batch(states, rng=rng, budget=budget, phase=phase,
                                                    reservation=reservation)
        terminals = []
        for state in batch.states:
            if not state.is_complete:
                terminals.append(None)  # Never decode/score unfinished states as molecules.
                continue
            with budget.oracle_check(phase=phase):
                terminals.append(self.reward.evaluate(state))
        return BaseBatch(batch.states, tuple(terminals), cost_since(budget, before),
                         batch.exhausted, batch.error)

    def sample(self, initial_state, *, sample_id, rng, budget):
        batch = self.generate_batch([initial_state], rng=rng, budget=budget)
        state, terminal = batch.states[0], batch.terminals[0]
        if state.is_complete:
            molecule = terminal.molecule
            return SampleResult(
                sample_id, state, self.target, SampleStatus.COMPLETED, cost=batch.cost,
                smiles=molecule.smiles if molecule else None,
                canonical_smiles=molecule.canonical_smiles if molecule else None,
                valid=terminal.valid, score=terminal.score, reward=terminal.reward,
            )
        status = SampleStatus.BUDGET_EXHAUSTED if batch.exhausted else SampleStatus.FAILED
        return SampleResult(sample_id, state, self.target, status, cost=batch.cost,
                            error=batch.error or "Native sampler stopped with remaining MASK")
