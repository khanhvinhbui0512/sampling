"""CARS on native trajectories for N=P(C(z),tau=1), with unpruned D=P(C(z)).

New integration, not an algorithm supplied by either paper. The trie stores
step/coordinate paths, including MASK loops, separately from the rho trie.
Discovery may remove incompatible observations or exact reward-zero terminals.
Freeze W, draw a fixed independent numerator sample under P/B_W, then estimate
D from independent UNPRUNED native root draws, including reward-zero leaves.
"""

import copy
import math
from dataclasses import asdict, dataclass
from itertools import chain

import torch
from tqdm.auto import tqdm

from src.budget import BudgetExceeded
from src.cars.estimator import calibration_metadata
from src.cars.trie import TrieCapacityError
from src.io import file_hash, payload_hash
from src.randomness import RNGStream, derive_seed
from src.sampling.completion import cost_since
from src.sampling.native_reference import NativePath
from src.types import CostPhase, EstimateStatus, MaskedState, ValueEstimate, require_count
from src.verifiers.importance import conditional_ratio
from src.verifiers.interface import native_conditional_target


def _logsum(values):
    maximum = max(values)
    return (-math.inf if maximum == -math.inf else
            maximum + math.log(sum(math.exp(x - maximum) for x in values)))


class NativeTrajectoryTrie:
    schema_version = 1
    namespace = "cars-native-trajectory"

    def __init__(self, reference, reward, state, *, max_nodes):
        native_conditional_target(reference, reward)
        if state.rules != reference.rules:
            raise ValueError("Native trie query rules differ")
        require_count(max_nodes, "native trie max_nodes", positive=True)
        self.initial_state, self.max_nodes = state, max_nodes
        self.identity = {"namespace": self.namespace, "initial_tokens": list(state.token_ids),
                         "rules": asdict(state.rules), "reference_id": reference.reference_id,
                         "policy_id": reference.policy_id, "schedule": copy.deepcopy(reference.policy.settings),
                         "calibration": copy.deepcopy(calibration_metadata(reward)),
                         "oracle_id": "compatibility-and-pass95:" + reward.calibration.calibration_id}
        self.is_frozen = False
        self.nodes = {(): {"raw": None, "failure": None}}
        self._masses = None

    def check_context(self, reference, reward=None):
        reference.check_context()
        if (reference.reference_id != self.identity["reference_id"]
                or reference.policy_id != self.identity["policy_id"]
                or asdict(reference.rules) != self.identity["rules"]
                or reference.policy.settings != self.identity["schedule"]):
            raise ValueError("Native trie reference context differs")
        if reward is not None:
            native_conditional_target(reference, reward)
            if calibration_metadata(reward) != self.identity["calibration"]:
                raise ValueError("Native trie calibration context differs")

    def _ensure(self, path):
        if self.is_frozen:
            raise ValueError("Frozen native proposal cannot learn")
        missing, prefix = [], tuple(path)
        while prefix not in self.nodes:
            missing.append(prefix)
            prefix = prefix[:-1]
        if len(self.nodes) + len(missing) > self.max_nodes:
            raise TrieCapacityError("Native trajectory trie capacity reached")
        for prefix in reversed(missing):
            self.nodes[prefix] = {"raw": None, "failure": None}
        if missing:
            self._masses = None

    def raw_log_probs(self, path):
        return self.nodes.get(tuple(path), {}).get("raw")

    def cache_distribution(self, path, raw):
        path, raw = tuple(path), tuple(raw)
        if (len(raw) != self.initial_state.rules.vocab_size
                or any(math.isnan(x) or x == math.inf for x in raw)
                or not math.isclose(_logsum(raw), 0., abs_tol=1e-10)):
            raise ValueError("Native raw distribution must be normalized")
        self._ensure(path)
        old = self.nodes[path]["raw"]
        if old is not None and old != raw:
            raise ValueError("Native raw probabilities cannot change")
        self.nodes[path]["raw"] = raw
        self._masses = None

    def _local_log_mass(self, path):
        # Bottom-up avoids Python recursion limits: 33 steps x 32 coordinates
        # can already exceed 1,000 edges in a single QM9 trajectory.
        if self._masses is None:
            masses = {}
            for prefix in sorted(self.nodes, key=len, reverse=True):
                node = self.nodes[prefix]
                if node["failure"] is not None:
                    masses[prefix] = -math.inf
                elif node["raw"] is None:
                    masses[prefix] = 0.
                else:
                    value = _logsum([p + masses.get(prefix + (token,), 0.)
                                     for token, p in enumerate(node["raw"])])
                    if value > 1e-10:
                        raise FloatingPointError("Native surviving probability exceeds one")
                    masses[prefix] = min(0., value)  # Roundoff only, since B <= 1.
            self._masses = masses
        return self._masses.get(tuple(path), 0.)

    def log_mass(self, path=()):
        path = tuple(path)
        if any(self.nodes.get(path[:i], {}).get("failure") is not None for i in range(len(path) + 1)):
            return -math.inf
        return self._local_log_mass(path)  # Unexplored cones retain mass one.

    @property
    def support_exhausted(self):
        return self.log_mass() == -math.inf

    @property
    def mass_underflow(self):
        log_b = self.log_mass()
        return math.isfinite(log_b) and math.exp(log_b) == 0.

    def learn_failure(self, path, reference, reward, *, budget):
        """Exact conservative oracle: compatibility cone or binary terminal fail."""
        path = tuple(path)
        if self.is_frozen:
            raise ValueError("Frozen native proposal cannot learn")
        self.check_context(reference, reward)
        if self.log_mass(path) == -math.inf:
            return False
        cursor = reference.cursor(path)
        with budget.oracle_check(phase=CostPhase.DISCOVERY):
            mask = reference.rules.mask_id
            conflict = any(a != mask and b != mask and a != b
                           for a, b in zip(self.initial_state.token_ids, cursor.observations.token_ids))
            if conflict:
                proof = {"reason": "incompatible_observation"}
            elif cursor.done and reward.evaluate(cursor.observations, require_binary=True).reward == 0:
                proof = {"reason": "terminal_reward_zero", "reward": 0}
            else:
                return False
        self._ensure(path)
        # Every prefix distribution is needed to propagate the ORIGINAL mass.
        if any(self.raw_log_probs(path[:i]) is None for i in range(len(path))):
            raise ValueError("A pruned native path needs all raw ancestor distributions")
        self.nodes[path]["failure"] = proof
        self._masses = None
        return True

    def freeze(self):
        result = copy.copy(self)
        result.identity, result.nodes = copy.deepcopy(self.identity), copy.deepcopy(self.nodes)
        result._masses = None
        result.is_frozen = True
        return result

    @property
    def snapshot_id(self):
        return payload_hash(self.to_dict())

    def to_dict(self):
        # None encodes -inf for strict JSON; it is never confused with a tiny mass.
        return {"schema_version": self.schema_version, "namespace": self.namespace,
                "identity": self.identity, "max_nodes": self.max_nodes, "frozen": self.is_frozen,
                "nodes": [{"path": list(path), "raw": None if node["raw"] is None else
                           [None if x == -math.inf else x for x in node["raw"]],
                           "failure": node["failure"]} for path, node in sorted(self.nodes.items())]}

    @classmethod
    def from_dict(cls, payload, *, reference, reward, state):
        if payload.get("schema_version") != cls.schema_version or payload.get("namespace") != cls.namespace:
            raise ValueError("Native trajectory snapshot identity/schema differs")
        trie = cls(reference, reward, state, max_nodes=payload["max_nodes"])
        if (payload.get("schema_version") != cls.schema_version or payload.get("namespace") != cls.namespace
                or payload.get("identity") != trie.identity or payload.get("frozen") is not True):
            raise ValueError("Native trajectory snapshot identity/schema differs")
        rows = payload["nodes"]
        if len(rows) > trie.max_nodes or len({tuple(row["path"]) for row in rows}) != len(rows):
            raise ValueError("Malformed native snapshot node count")
        for row in sorted(rows, key=lambda item: len(item["path"])):
            path = tuple(row["path"])
            cursor = reference.cursor(path)
            if path and (path[:-1] not in trie.nodes or trie.raw_log_probs(path[:-1]) is None
                         or trie.raw_log_probs(path[:-1])[path[-1]] == -math.inf):
                raise ValueError("Native snapshot has unsupported/disconnected path")
            trie._ensure(path)
            if row["raw"] is not None:
                if cursor.done:
                    raise ValueError("Terminal native node cannot have a next distribution")
                trie.cache_distribution(path, [-math.inf if x is None else x for x in row["raw"]])
            proof = row["failure"]
            if proof is not None:
                conflict = any(a != reference.rules.mask_id and b != reference.rules.mask_id and a != b
                               for a, b in zip(state.token_ids, cursor.observations.token_ids))
                if not ((proof == {"reason": "incompatible_observation"} and conflict)
                        or (proof == {"reason": "terminal_reward_zero", "reward": 0} and cursor.done)):
                    raise ValueError("Malformed native failure certificate")
                trie.nodes[path]["failure"] = proof
                trie._masses = None
        if not rows or tuple(rows[0]["path"]) != ():
            raise ValueError("Native snapshot is missing its root")
        return trie.freeze()


def native_proposal_draw(reference, trie, *, rng, budget, reservation=None, discovery=False,
                         mode="cars", reward=None):
    """Sample under old W, then (discovery only) update for the NEXT draw."""
    if discovery == trie.is_frozen or mode not in ("ars", "cars"):
        raise ValueError("Discovery needs mutable trie; estimation needs frozen trie")
    if discovery and reward is None:
        raise ValueError("Native discovery requires its calibrated reward")
    trie.check_context(reference, reward if discovery else None)
    log_b = trie.log_mass()
    if log_b == -math.inf:
        return None, False
    cursor, path, visited = reference.cursor(), (), []
    log_p, log_q, capacity = 0., 0., False
    phase = CostPhase.DISCOVERY if discovery else CostPhase.ESTIMATION
    while not cursor.done:
        raw = trie.raw_log_probs(path)
        if raw is None:
            raw = cursor.next_log_probs(budget=budget, phase=phase, reservation=reservation)
            if discovery and not capacity:
                try:
                    trie.cache_distribution(path, raw)
                except TrieCapacityError:
                    capacity = True
        # The current draw prefix is supported. Local child masses therefore
        # suffice; no repeated scan of its (possibly thousand-edge) ancestors.
        scores = [p + trie._local_log_mass(path + (token,)) for token, p in enumerate(raw)]
        normalizer = _logsum(scores)
        if not math.isfinite(normalizer) or not math.isclose(normalizer, trie._local_log_mass(path), abs_tol=1e-9):
            raise FloatingPointError("Native proposal surviving mass differs")
        log_probabilities = torch.tensor(scores, dtype=torch.float64).sub(normalizer)
        probabilities = log_probabilities.exp()
        if bool((torch.isfinite(log_probabilities) & (probabilities == 0.)).any()):
            raise FloatingPointError("Positive native proposal edge underflowed; support would be lost")
        token = int(torch.multinomial(probabilities, 1,
                                      generator=rng.torch_generator("cpu"))[0])
        visited.append((path, raw))
        log_p, log_q = log_p + raw[token], log_q + scores[token] - normalizer
        path += (token,)
        cursor.advance(token)
    if not math.isclose(log_q, log_p - log_b, abs_tol=1e-8):
        raise FloatingPointError("Native trajectory P/B_W identity failed")
    if discovery and not capacity:
        # Stream siblings: materializing every full prefix before the node cap
        # is checked can use substantial memory for thousand-edge QM9 paths.
        candidates = ((path,) if mode == "ars" else chain(((),),
                      (prefix + (token,) for prefix, raw in visited for token, p in enumerate(raw)
                       if math.isfinite(p))))
        for candidate in candidates:
            try:
                trie.learn_failure(candidate, reference, reward, budget=budget)
            except TrieCapacityError:
                capacity = True
                break
    return NativePath(path, cursor.observations, log_p, log_q), capacity


@dataclass(frozen=True)
class NativeEstimationRun:
    estimate: ValueEstimate
    trie: NativeTrajectoryTrie
    discovery_trace: list
    estimation_trace: list
    metadata: dict
    rng_state: dict
    summary: object = None


class NativeCARSValueEstimator:
    def __init__(self, reference, reward, *, discovery_attempts, numerator_draws, denominator_draws,
                 max_nodes, min_denominator_hits, mode="cars", confidence=.95,
                 discovery_max_base_sequences=None):
        self.target = native_conditional_target(reference, reward)
        for name, value in (("discovery attempts", discovery_attempts), ("numerator draws", numerator_draws),
                            ("denominator draws", denominator_draws), ("max nodes", max_nodes),
                            ("minimum denominator hits", min_denominator_hits)):
            require_count(value, name, positive=name != "discovery attempts")
        if discovery_max_base_sequences is not None:
            require_count(discovery_max_base_sequences, "discovery base cap")
        if mode not in ("ars", "cars") or not 0 < confidence < 1 or min_denominator_hits > denominator_draws:
            raise ValueError("Invalid native CARS protocol")
        self.reference, self.reward = reference, reward
        self.discovery_attempts, self.numerator_draws = discovery_attempts, numerator_draws
        self.denominator_draws, self.max_nodes = denominator_draws, max_nodes
        self.min_denominator_hits, self.mode, self.confidence = min_denominator_hits, mode, confidence
        self.discovery_max_base_sequences = discovery_max_base_sequences
        self.calibration = calibration_metadata(reward)
        self.cache_metadata = {
            "schema_version": 1, "namespace": "cars-native-trajectory", "target": asdict(self.target),
            "token_rules": asdict(reference.rules), "reference_id": reference.reference_id,
            "policy_id": reference.policy_id, "schedule": copy.deepcopy(reference.policy.settings),
            "native_reference": reference.protocol,
            "checkpoint_id": reference.backbone.checkpoint_id, "tokenizer_id": reference.backbone.tokenizer_id,
            "calibration": self.calibration,
            "estimator": {"discovery_attempts": discovery_attempts, "numerator_draws": numerator_draws,
                          "denominator_draws": denominator_draws, "max_nodes": max_nodes,
                          "min_denominator_hits": min_denominator_hits, "mode": mode, "confidence": confidence,
                          "discovery_max_base_sequences": discovery_max_base_sequences,
                          "interval": "hoeffding_joint_ratio", "denominator": "unpruned_native_root"},
            "source_sha256": {name: file_hash(name) for name in (
                "src/cars/native.py", "src/sampling/native_reference.py", "src/sampling/native.py",
                "src/verifiers/importance.py", "src/verifiers/interface.py", "src/qm9/reward.py",
                "src/models/backbone.py", "src/sampling/completion.py", "src/cars/cache.py",
                "src/verifiers/cars.py", "src/qm9/chemistry.py", "src/randomness.py",
                "src/budget.py", "src/types.py")}}

    def check_context(self, state):
        if (state.rules != self.reference.rules or calibration_metadata(self.reward) != self.calibration
                or native_conditional_target(self.reference, self.reward) != self.target
                or self.reference.policy.settings != self.cache_metadata["schedule"]
                or self.reference.backbone.checkpoint_id != self.cache_metadata["checkpoint_id"]
                or self.reference.backbone.tokenizer_id != self.cache_metadata["tokenizer_id"]):
            raise ValueError("Native estimator context changed")

    def required_work(self, state):
        self.check_context(state)
        if state.is_complete:
            return 0
        unit = self.reference.max_base_sequences
        discovery = self.discovery_attempts * unit
        if self.discovery_max_base_sequences is not None:
            discovery = min(discovery, self.discovery_max_base_sequences)
        return discovery + (self.numerator_draws + self.denominator_draws) * unit

    def estimate(self, state, *, rng, budget, reservation=None, progress=True):
        self.check_context(state)
        before = budget.snapshot().total
        trie = NativeTrajectoryTrie(self.reference, self.reward, state, max_nodes=self.max_nodes)
        discovery, estimation, rng_states = [], [], {}
        details = {"context": self.cache_metadata, "denominator": "unpruned_native_root",
                   "ratio_finite_sample_bias": True, "capacity_reached": False}

        def finish(status, value=None, error=None, summary=None):
            frozen = trie.freeze()
            details["log_numerator_mass"] = None if frozen.support_exhausted else frozen.log_mass()
            estimate = ValueEstimate(self.target, status, value, cost=cost_since(budget, before),
                                     draws=len(estimation), interval=summary.interval if summary else None,
                                     error=error)
            return NativeEstimationRun(estimate, frozen, discovery, estimation, details, rng_states, summary)

        if state.is_complete:
            with budget.oracle_check(phase=CostPhase.VERIFICATION):
                value = float(self.reward.evaluate(state, require_binary=True).reward)
            return finish(EstimateStatus.EXACT, value)
        required = self.required_work(state)
        owned = reservation is None
        held = reservation or budget.reserve_query(label="CARS-native-conditional", base_sequences=required)
        if held is None or held.base_remaining < required:
            if owned and held is not None:
                held.close()
            return finish(EstimateStatus.BUDGET_EXHAUSTED, error="Entire fixed N/D query does not fit budget")
        seed = rng.python.getrandbits(63)
        streams = {name: RNGStream(derive_seed(seed, name)) for name in ("discovery", "numerator", "denominator")}
        try:
            discovery_before = budget.snapshot().total.base_sequences
            for _ in tqdm(range(self.discovery_attempts), desc="Native CARS discovery", disable=not progress):
                if trie.support_exhausted:
                    break
                spent = budget.snapshot().total.base_sequences - discovery_before
                if (self.discovery_max_base_sequences is not None
                        and spent + self.reference.max_base_sequences > self.discovery_max_base_sequences):
                    break  # Freeze learning; never shorten the fixed estimation stages.
                draw, capacity = native_proposal_draw(self.reference, trie, rng=streams["discovery"],
                                                      budget=budget, reservation=held, discovery=True,
                                                      mode=self.mode, reward=self.reward)
                if draw is not None:
                    discovery.append(asdict(draw))
                if capacity:
                    details["capacity_reached"] = True
                    break
            trie = trie.freeze()
            if trie.mass_underflow:
                return finish(EstimateStatus.FAILED, error="Positive native surviving mass underflowed")
            successes, compatible = 0, 0
            if not trie.support_exhausted:
                for _ in tqdm(range(self.numerator_draws), desc="Native CARS numerator", disable=not progress):
                    draw, _ = native_proposal_draw(self.reference, trie, rng=streams["numerator"],
                                                   budget=budget, reservation=held)
                    with budget.oracle_check(phase=CostPhase.ESTIMATION):
                        success = state.accepts_completion(draw.terminal) and bool(
                            self.reward.evaluate(draw.terminal, require_binary=True).reward)
                    successes += int(success)
                    estimation.append({"stage": "numerator", **asdict(draw), "event_reward": int(success)})
            for _ in tqdm(range(self.denominator_draws), desc="Native unpruned denominator", disable=not progress):
                draw = self.reference.draw(rng=streams["denominator"], budget=budget,
                                           phase=CostPhase.ESTIMATION, reservation=held)
                event = state.accepts_completion(draw.terminal)
                compatible += int(event)  # Includes every failure and duplicate; no reward filtering.
                estimation.append({"stage": "denominator", **asdict(draw), "compatible": event})
            summary = conditional_ratio(successes, self.numerator_draws, compatible, self.denominator_draws,
                                         log_mass=trie.log_mass(), confidence=self.confidence,
                                         min_denominator_hits=self.min_denominator_hits,
                                         denominator_is_one=state == self.reference.root)
            details["summary"] = asdict(summary)
            return finish(EstimateStatus.ESTIMATED if summary.value is not None else EstimateStatus.FAILED,
                          summary.value, summary.error, summary)
        except BudgetExceeded as exc:
            return finish(EstimateStatus.BUDGET_EXHAUSTED, error=str(exc))
        except FloatingPointError as exc:
            return finish(EstimateStatus.FAILED, error=str(exc))
        finally:
            rng_states.update({name: stream.state_dict() for name, stream in streams.items()})
            if owned:
                held.close()
