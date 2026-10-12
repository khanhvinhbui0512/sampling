"""CARS-rho discovery followed by independent, fixed-count value estimation.

The trie/proposal law follows CARS Section 3. B_W * mean(binary reward),
sample splitting and the Hoeffding interval are project additions for V_rho.
They do not estimate native-MDM V* without the separate reference work in Step 9.
"""

import math
from dataclasses import asdict, dataclass, is_dataclass

import torch
from tqdm.auto import tqdm

from src.budget import BudgetExceeded
from src.cars.proposal import FrozenProposal
from src.cars.trie import FailureTrie, TrieCapacityError
from src.io import file_hash, payload_hash
from src.randomness import RNGStream, derive_seed
from src.sampling.completion import RhoCompletion, cost_since
from src.types import CostPhase, CostRecord, EstimateStatus, MaskedState, ValueEstimate, require_count
from src.verifiers.interface import value_target


def calibration_metadata(reward):
    calibration = reward.calibration
    return asdict(calibration) if is_dataclass(calibration) else dict(vars(calibration))


def check_rho_reward(policy, reward):
    if not isinstance(policy, RhoCompletion):
        raise ValueError("CARS-rho requires RhoCompletion, not a native diffusion sampler")
    if (policy.rules != reward.token_rules or policy.backbone.tokenizer_id != reward.tokenizer_sha256
            or reward.calibration is None
            or policy.backbone.checkpoint_id != reward.calibration.checkpoint_sha256):
        raise ValueError("CARS policy and calibrated reward differ")


@dataclass(frozen=True)
class AdaptiveDraw:
    state: MaskedState
    status: str
    log_q: float | None
    log_proposal: float | None
    log_root_mass: float
    pruned_cones: int
    capacity_reached: bool
    cost: CostRecord
    error: str | None = None


def adaptive_draw(policy, trie, oracle, *, mode, rng, budget,
                  phase=CostPhase.DISCOVERY, reservation=None):
    """Draw once under the OLD W, then learn failures for the NEXT draw.

    Caching raw probabilities during the draw does not change surviving mass.
    ARS removes only the sampled failing leaf. CARS also checks all supported
    one-token continuations of visited prefixes, including successful paths.
    Capacity freezes learning; it never removes an unproved branch.
    """
    if mode not in ("ars", "cars") or trie.is_frozen:
        raise ValueError("Adaptive discovery needs a mutable trie and ars/cars mode")
    if (trie.policy_id != policy.policy_id or trie.reference_id != policy.reference_id
            or trie.oracle_id != oracle.oracle_id or trie.initial_state.rules != policy.rules
            or oracle.rules != policy.rules):
        raise ValueError("Discovery policy, oracle and trie context differ")
    before = budget.snapshot().total
    state, path, visited = trie.initial_state, (), []
    log_q, log_proposal, log_b = 0., 0., trie.log_mass()
    pruned, capacity = 0, False

    def finish(status, error=None):
        return AdaptiveDraw(state, status, log_q if status == "drawn" else None,
                            log_proposal if status == "drawn" else None, log_b,
                            pruned, capacity, cost_since(budget, before), error)

    if trie.support_exhausted:
        return finish("support_exhausted")
    for position in trie.positions:
        raw = trie.raw_log_probs(path)
        if raw is None:
            try:
                raw = tuple(float(x) for x in policy.next_log_probs(
                    [state], budget=budget, phase=phase, reservation=reservation)[0])
            except BudgetExceeded as exc:
                return finish("budget_exhausted", str(exc))
            if not capacity:
                try:
                    trie.cache_distribution(path, raw)
                except TrieCapacityError:
                    capacity = True
        raw = torch.tensor(raw, dtype=torch.float64)
        scores = raw + torch.tensor(trie.child_log_masses(path), dtype=torch.float64)
        normalizer = float(torch.logsumexp(scores, dim=0))
        if not math.isfinite(normalizer) or not math.isclose(
                normalizer, trie.log_mass(path), rel_tol=0., abs_tol=1e-9):
            raise FloatingPointError("Discovery proposal and surviving mass differ")
        visited.append((path, tuple(float(x) for x in raw)))
        token = int(torch.multinomial((scores - normalizer).exp(), 1,
                                      generator=rng.torch_generator("cpu"))[0])
        log_q += float(raw[token])
        log_proposal += float(scores[token]) - normalizer
        state = state.reveal({position: token})
        path += (token,)
    if not math.isclose(log_proposal, log_q - log_b, rel_tol=0., abs_tol=1e-8):
        raise FloatingPointError("Discovery failed the q_rho/B_W identity")

    def prune(candidate_path):
        nonlocal pruned, capacity
        if trie.log_mass(candidate_path) == -math.inf:
            return
        certificate = oracle.check(trie.state_at(candidate_path), budget=budget, phase=phase)
        if certificate.certified_failure:
            try:
                pruned += int(trie.record_failure(candidate_path, certificate))
            except TrieCapacityError:
                capacity = True

    if not capacity:
        if mode == "ars":
            prune(path)
        else:
            prune(())
            for prefix, raw in visited:
                if capacity or trie.support_exhausted:
                    break
                if trie.log_mass(prefix) == -math.inf:
                    continue
                for token, log_probability in enumerate(raw):
                    if math.isfinite(log_probability):
                        prune(prefix + (token,))
                    if capacity:
                        break
    return finish("drawn")


def scaled_bernoulli_summary(successes, draws, log_mass, confidence):
    """Conditional Hoeffding interval, scaled by B; no asymptotic coverage claim."""
    require_count(draws, "estimation draws", positive=True)
    require_count(successes, "estimation successes")
    if successes > draws or not 0 < confidence < 1 or not math.isfinite(log_mass) or log_mass > 0:
        raise ValueError("Invalid fixed-count CARS summary")
    mass = math.exp(log_mass)
    p = successes / draws
    value = mass * p
    if mass == 0. or (successes > 0 and value == 0.):
        raise FloatingPointError("Positive CARS mass/value underflowed; this is not certified zero")
    radius = math.sqrt(math.log(2 / (1 - confidence)) / (2 * draws))
    interval = (mass * max(0., p - radius), mass * min(1., p + radius))
    if successes == 0 and interval[1] == 0.:
        raise FloatingPointError("Positive uncertainty bound underflowed")
    return value, mass * math.sqrt(p * (1 - p) / draws), interval


@dataclass(frozen=True)
class EstimationRun:
    estimate: ValueEstimate
    trie: FailureTrie | None
    discovery_trace: tuple
    estimation_trace: tuple
    metadata: dict
    rng_state: dict


class CARSValueEstimator:
    def __init__(self, policy, reward, oracle, *, discovery_attempts, draws,
                 mode="ars", max_nodes=None, discovery_max_base_sequences=None, confidence=0.95):
        check_rho_reward(policy, reward)
        require_count(discovery_attempts, "discovery attempts")
        require_count(draws, "fixed estimation draws", positive=True)
        if max_nodes is not None:
            require_count(max_nodes, "max_nodes", positive=True)
        if discovery_max_base_sequences is not None:
            require_count(discovery_max_base_sequences, "discovery work cap")
        if mode not in ("ars", "cars") or not 0 < confidence < 1:
            raise ValueError("Choose ars/cars discovery and confidence in (0, 1)")
        if (oracle.rules != policy.rules or (hasattr(oracle, "reward") and oracle.reward is not reward)
                or (mode == "cars" and getattr(oracle, "mode", None) == "leaf_only")):
            raise ValueError("Oracle differs from reward, or leaf_only must be labelled ars")
        self.policy, self.reward, self.oracle = policy, reward, oracle
        self.target = value_target(policy, reward)
        self.discovery_attempts, self.draws, self.mode = discovery_attempts, draws, mode
        self.max_nodes, self.discovery_max_base_sequences = max_nodes, discovery_max_base_sequences
        self.confidence = confidence
        self._calibration = calibration_metadata(reward)
        self.cache_metadata = {
            "schema_version": 1, "namespace": "cars-rho", "target": asdict(self.target),
            "policy_id": policy.policy_id, "reference_id": policy.reference_id,
            "checkpoint_id": policy.backbone.checkpoint_id, "tokenizer_id": policy.backbone.tokenizer_id,
            "temperature": policy.temperature,
            "token_rules": asdict(policy.rules), "oracle_id": oracle.oracle_id,
            "calibration": self._calibration,
            "estimator": {"version": "cars-rho-v1", "discovery_attempts": discovery_attempts,
                          "draws": draws, "mode": mode, "max_nodes": max_nodes,
                          "discovery_max_base_sequences": discovery_max_base_sequences,
                          "confidence": confidence, "interval": "conditional_hoeffding"},
            "source_hashes": {name: file_hash(name) for name in (
                "src/cars/estimator.py", "src/cars/trie.py", "src/cars/oracle.py",
                "src/cars/proposal.py", "src/sampling/completion.py", "src/qm9/reward.py",
                "src/cars/cache.py", "src/verifiers/cars.py", "src/verifiers/interface.py",
                "src/randomness.py")},
        }
        self.estimator_id = "cars-rho:" + payload_hash(self.cache_metadata)

    def check_context(self, state):
        check_rho_reward(self.policy, self.reward)
        if state.rules != self.policy.rules or calibration_metadata(self.reward) != self._calibration:
            raise ValueError("Estimator context/calibration changed; create a new estimator")
        if (self.policy.policy_id != self.cache_metadata["policy_id"]
                or self.policy.reference_id != self.cache_metadata["reference_id"]
                or self.policy.backbone.checkpoint_id != self.cache_metadata["checkpoint_id"]
                or self.policy.backbone.tokenizer_id != self.cache_metadata["tokenizer_id"]
                or self.policy.temperature != self.cache_metadata["temperature"]
                or self.oracle.oracle_id != self.cache_metadata["oracle_id"]):
            raise ValueError("Estimator model/policy/oracle changed; create a new estimator")

    def required_work(self, state):
        self.check_context(state)
        count = len(state.masked_positions)
        discovery = count * self.discovery_attempts
        if self.discovery_max_base_sequences is not None:
            discovery = min(discovery, self.discovery_max_base_sequences)
        return discovery + count * self.draws

    def estimate(self, state, *, rng, budget, reservation=None, initial_trie=None, progress=True):
        self.check_context(state)
        before = budget.snapshot()
        discovery_trace, estimation_trace, rng_states = [], [], {}
        trie = None
        metadata = {"estimator_id": self.estimator_id, "target": asdict(self.target),
                    "discovery_stop": "not_started", "fixed_draws": self.draws,
                    "successes": 0, "log_surviving_mass": None}

        def finish(status, value=None, *, draws=0, standard_error=None, interval=None,
                   surviving_mass=None, error=None):
            estimate = ValueEstimate(self.target, status, value, cost=cost_since(budget, before.total),
                                     draws=draws, standard_error=standard_error, interval=interval,
                                     surviving_mass=surviving_mass, error=error)
            after = budget.snapshot()
            metadata["phase_costs"] = {
                phase: {name: getattr(cost, name) - getattr(before.by_phase.get(phase, type(cost)()), name)
                        for name in cost.__dataclass_fields__}
                for phase, cost in after.by_phase.items()}
            frozen = trie.freeze() if trie is not None else None
            if frozen is not None:
                metadata["log_surviving_mass"] = None if frozen.support_exhausted else frozen.log_mass()
                metadata["support_exhausted"] = frozen.support_exhausted
                metadata["snapshot_id"] = frozen.snapshot_id
            if progress:
                tqdm.write(f"CARS-rho value: {status.value}, value={value}, draws={draws}, "
                           f"base_calls={estimate.cost.base_calls}")
            return EstimationRun(estimate, frozen, tuple(discovery_trace), tuple(estimation_trace),
                                 metadata, rng_states)

        if state.is_complete:
            with budget.oracle_check(phase=CostPhase.VERIFICATION):
                value = float(self.reward.evaluate(state, require_binary=True).reward)
            return finish(EstimateStatus.EXACT, value, standard_error=0., interval=(value, value))
        trie = FailureTrie(state, policy_id=self.policy.policy_id, reference_id=self.policy.reference_id,
                           oracle_id=self.oracle.oracle_id, max_nodes=self.max_nodes)
        if initial_trie is not None:
            payload = initial_trie.to_dict()
            payload["frozen"] = False
            if payload["max_nodes"] != self.max_nodes:
                raise ValueError("Warm trie uses a different memory protocol")
            trie = FailureTrie.from_dict(payload, expected_identity=trie.identity)
        if trie.support_exhausted:
            metadata["discovery_stop"] = "certified_support_exhausted"
            return finish(EstimateStatus.EXACT, 0., standard_error=0., interval=(0., 0.), surviving_mass=0.)
        required = self.required_work(state)
        owned = reservation is None
        held = reservation or budget.reserve_query(label="CARS-rho", base_sequences=required)
        if held is None or held.base_remaining < required:
            if owned and held is not None:
                held.close()
            return finish(EstimateStatus.BUDGET_EXHAUSTED,
                          error="Insufficient budget for discovery AND all fixed estimation draws")
        query_seed = rng.python.getrandbits(63)
        discovery_rng = RNGStream(derive_seed(query_seed, "cars-discovery"))
        estimation_rng = RNGStream(derive_seed(query_seed, "cars-estimation"))
        completed, successes = 0, 0
        try:
            discovery_start = budget.snapshot().total.base_sequences
            metadata["discovery_stop"] = "attempt_limit"
            for _ in tqdm(range(self.discovery_attempts), desc="CARS discovery", leave=False,
                          disable=not progress):
                used = budget.snapshot().total.base_sequences - discovery_start
                if (self.discovery_max_base_sequences is not None
                        and used + len(state.masked_positions) > self.discovery_max_base_sequences):
                    metadata["discovery_stop"] = "work_limit"
                    break
                draw = adaptive_draw(self.policy, trie, self.oracle, mode=self.mode,
                                     rng=discovery_rng, budget=budget, reservation=held)
                discovery_trace.append(asdict(draw))
                if draw.status == "budget_exhausted":
                    return finish(EstimateStatus.BUDGET_EXHAUSTED, error=draw.error)
                if trie.support_exhausted:
                    metadata["discovery_stop"] = "certified_support_exhausted"
                    return finish(EstimateStatus.EXACT, 0., standard_error=0., interval=(0., 0.),
                                  surviving_mass=0.)
                if draw.capacity_reached:
                    metadata["discovery_stop"] = "memory_limit_freeze"
                    break
            proposal = FrozenProposal(self.policy, trie.freeze())
            if proposal.trie.mass_underflow:
                raise FloatingPointError("Positive surviving mass underflowed; not certified zero")
            snapshot_id = proposal.trie.snapshot_id
            for _ in tqdm(range(self.draws), desc="CARS estimation", leave=False, disable=not progress):
                draw_before = budget.snapshot().total
                draw = proposal.draw(rng=estimation_rng, budget=budget, reservation=held, progress=False)
                if draw.status != "drawn":
                    estimation_trace.append({"tokens": list(draw.state.token_ids), "reward": None,
                                             "status": draw.status, "error": draw.error,
                                             "cost": asdict(cost_since(budget, draw_before))})
                    return finish(EstimateStatus.BUDGET_EXHAUSTED, draws=completed,
                                  error=draw.error or "Incomplete fixed-count estimation")
                with budget.oracle_check(phase=CostPhase.ESTIMATION):
                    reward = self.reward.evaluate(draw.state, require_binary=True).reward
                if reward not in (0, 1):
                    raise ValueError("CARS estimation requires binary terminal reward")
                successes += int(reward)
                completed += 1
                metadata["successes"] = successes
                estimation_trace.append({"tokens": list(draw.state.token_ids), "reward": int(reward),
                                         "status": "drawn",
                                         "log_q": draw.log_q, "log_proposal": draw.log_proposal,
                                         "log_root_mass": draw.log_root_mass,
                                         "cost": asdict(cost_since(budget, draw_before))})
            if proposal.trie.snapshot_id != snapshot_id or trie.freeze().snapshot_id != snapshot_id:
                raise ValueError("Estimation changed the frozen discovery trie")
            metadata["successes"] = successes
            value, standard_error, interval = scaled_bernoulli_summary(
                successes, completed, trie.log_mass(), self.confidence)
            return finish(EstimateStatus.ESTIMATED, value, draws=completed, standard_error=standard_error,
                          interval=interval, surviving_mass=trie.root_mass)
        except (BudgetExceeded, FloatingPointError) as exc:
            status = (EstimateStatus.BUDGET_EXHAUSTED if isinstance(exc, BudgetExceeded)
                      else EstimateStatus.FAILED)
            return finish(status, draws=completed, error=str(exc))
        finally:
            rng_states.update(discovery=discovery_rng.state_dict(), estimation=estimation_rng.state_dict())
            if owned:
                held.close()
