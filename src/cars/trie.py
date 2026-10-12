"""Independent implementation of CARS Section 3, equations (1)-(3).

Each edge assigns one ORIGINAL masked position, in left-to-right order. The
root retains all original observations, so a path is never a concatenation of
known tokens across holes. Raw probabilities belong to this exact q_rho context.
"""

import math
from dataclasses import asdict, dataclass, field

from src.cars.oracle import OracleResult
from src.io import payload_hash
from src.types import MaskedState, TokenRules, require_count, require_id


def logsumexp(values):
    values = tuple(values)
    maximum = max(values, default=-math.inf)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def _checked_log_probs(values, rules):
    values = tuple(float(value) for value in values)
    if len(values) != rules.vocab_size or values[rules.mask_id] != -math.inf:
        raise ValueError("Supply one raw log probability per token, with MASK=-inf")
    if any(math.isnan(value) or value == math.inf or value > 1e-10 for value in values):
        raise ValueError("Invalid raw log probabilities")
    if not math.isclose(logsumexp(values), 0., rel_tol=0., abs_tol=1e-10):
        raise ValueError("Raw token probabilities must sum to one")
    return values


@dataclass
class _Node:
    raw: tuple | None = None
    children: dict = field(default_factory=dict)
    failure_reason: str | None = None
    log_mass: float = 0.0


class TrieCapacityError(RuntimeError):
    """Discovery must stop or freeze explicitly; existing failure cones stay intact."""


class FailureTrie:
    def __init__(self, initial_state, *, policy_id, reference_id, oracle_id, max_nodes=None):
        for name, value in (("policy_id", policy_id), ("reference_id", reference_id),
                            ("oracle_id", oracle_id)):
            require_id(value, name)
        if max_nodes is not None:
            require_count(max_nodes, "max_nodes", positive=True)
        self.initial_state = initial_state
        self.positions = initial_state.masked_positions
        self.policy_id, self.reference_id, self.oracle_id = policy_id, reference_id, oracle_id
        self.max_nodes = max_nodes
        self._root, self._node_count, self._frozen = _Node(), 1, False

    @property
    def identity(self):
        return {"initial_tokens": list(self.initial_state.token_ids),
                "rules": asdict(self.initial_state.rules), "policy_id": self.policy_id,
                "reference_id": self.reference_id, "oracle_id": self.oracle_id}

    @property
    def is_frozen(self):
        return self._frozen

    @property
    def node_count(self):
        return self._node_count

    @property
    def root_mass(self):
        return math.exp(self._root.log_mass)

    @property
    def support_exhausted(self):
        return self._root.log_mass == -math.inf

    @property
    def mass_underflow(self):
        return not self.support_exhausted and self.root_mass == 0.

    def state_at(self, path):
        path = tuple(path)
        if len(path) > len(self.positions):
            raise ValueError("Trie path is longer than the initial MASK count")
        return self.initial_state.reveal(dict(zip(self.positions, path)))

    def path_for(self, state):
        if state.rules != self.initial_state.rules:
            raise ValueError("Trie/state token rules differ")
        if any(token != self.initial_state.rules.mask_id and token != other
               for token, other in zip(self.initial_state.token_ids, state.token_ids)):
            raise ValueError("Trie paths cannot change original observations")
        path, saw_mask = [], False
        for position in self.positions:
            token = state.token_ids[position]
            if token == state.rules.mask_id:
                saw_mask = True
            elif saw_mask:
                raise ValueError("Assignments must follow the original left-to-right MASK order")
            else:
                path.append(token)
        return tuple(path)

    def _find(self, path):
        self.state_at(path)  # Validate even paths absent from the stored trie.
        node = self._root
        for token in path:
            if node.failure_reason is not None:
                return node  # The whole cone, including absent descendants, has mass zero.
            node = node.children.get(token)
            if node is None:
                return None
        return node

    def log_mass(self, path=()):
        node = self._find(tuple(path))
        return 0. if node is None else node.log_mass

    def raw_log_probs(self, path=()):
        node = self._find(tuple(path))
        return None if node is None else node.raw

    def child_log_masses(self, path=()):
        node = self._find(tuple(path))
        if node is not None and node.failure_reason is not None:
            return (-math.inf,) * self.initial_state.rules.vocab_size
        return tuple(0. if node is None or token not in node.children
                     else node.children[token].log_mass
                     for token in range(self.initial_state.rules.vocab_size))

    def _ensure_path(self, path):
        if self._frozen:
            raise ValueError("Frozen trie cannot be updated")
        self.state_at(path)
        node, ancestors = self._root, [self._root]
        for index, token in enumerate(path):
            if node.failure_reason is not None:
                raise ValueError("Cannot expand an already pruned cone")
            if node.raw is None:
                raise ValueError("Record raw probabilities at every ancestor before adding a path")
            if token not in node.children:
                if index != len(path) - 1:
                    raise ValueError("Record raw probabilities at every ancestor before adding a path")
                if self.max_nodes is not None and self._node_count + 1 > self.max_nodes:
                    raise TrieCapacityError("Trie node limit reached; no update was applied")
                child = _Node()
                node.children[token] = child
                ancestors.append(child)
                self._node_count += 1
                return ancestors
            node = node.children[token]
            ancestors.append(node)
        return ancestors

    def cache_distribution(self, path, log_probs):
        path = tuple(path)
        if len(path) >= len(self.positions):
            raise ValueError("A terminal path has no next-token distribution")
        raw = _checked_log_probs(log_probs, self.initial_state.rules)
        node = self._ensure_path(path)[-1]
        if node.failure_reason is not None:
            raise ValueError("Cannot cache probabilities inside a pruned cone")
        if node.raw is not None and node.raw != raw:
            raise ValueError("Raw probabilities cannot change within one trie context")
        node.raw = raw

    def record_failure(self, path, certificate):
        path = tuple(path)
        if self._frozen:
            raise ValueError("Frozen trie cannot be updated")
        if (not isinstance(certificate, OracleResult) or not certificate.certified_failure
                or certificate.oracle_id != self.oracle_id or certificate.state != self.state_at(path)):
            raise ValueError("Pruning requires a matching certified-failure oracle result")
        existing = self._find(path)
        if existing is not None and existing.failure_reason is not None:
            return False  # Duplicate or descendant of an existing failure cone.
        ancestors = self._ensure_path(path)
        node = ancestors[-1]

        def count(current):
            return 1 + sum(count(child) for child in current.children.values())

        self._node_count -= count(node) - 1
        node.raw, node.children, node.failure_reason, node.log_mass = None, {}, certificate.reason, -math.inf
        for parent in reversed(ancestors[:-1]):
            # Sum SURVIVING terms, never subtract almost-one failure mass.
            parent.log_mass = min(0., logsumexp(
                raw + (parent.children[token].log_mass if token in parent.children else 0.)
                for token, raw in enumerate(parent.raw)
            ))
        return True

    def to_dict(self):
        def encode(node):
            return {"raw": None if node.raw is None else
                    [None if value == -math.inf else value for value in node.raw],
                    "failure_reason": node.failure_reason,
                    "children": {str(token): encode(child)
                                 for token, child in sorted(node.children.items())}}

        return {"schema_version": 1, "identity": self.identity, "frozen": self._frozen,
                "max_nodes": self.max_nodes, "root": encode(self._root)}

    @classmethod
    def from_dict(cls, payload, *, expected_identity=None):
        if payload.get("schema_version") != 1 or type(payload.get("frozen")) is not bool:
            raise ValueError("Unsupported trie snapshot schema")
        identity = payload["identity"]
        if expected_identity is not None and identity != expected_identity:
            raise ValueError("Trie snapshot context differs")
        rules = TokenRules(**identity["rules"])
        trie = cls(MaskedState(tuple(identity["initial_tokens"]), rules),
                   policy_id=identity["policy_id"], reference_id=identity["reference_id"],
                   oracle_id=identity["oracle_id"], max_nodes=payload["max_nodes"])
        trie._node_count = 0

        def decode(item, depth):
            trie._node_count += 1
            if depth > len(trie.positions):
                raise ValueError("Snapshot exceeds the completion horizon")
            reason = item["failure_reason"]
            if reason is not None:
                require_id(reason, "failure_reason")
                if item["raw"] is not None or item["children"]:
                    raise ValueError("Pruned snapshot nodes must have no distribution or children")
                return _Node(failure_reason=reason, log_mass=-math.inf)
            raw = item["raw"]
            if raw is not None:
                if depth == len(trie.positions):
                    raise ValueError("Terminal snapshot nodes cannot have a distribution")
                raw = _checked_log_probs(
                    (-math.inf if value is None else value for value in raw), rules)
            children = {}
            for key, child in item["children"].items():
                token = int(key)
                rules.validate_token(token)
                if str(token) != key or token == rules.mask_id or raw is None:
                    raise ValueError("Snapshot child requires a valid edge and ancestor distribution")
                children[token] = decode(child, depth + 1)
            mass = 0. if not children else min(0., logsumexp(
                value + (children[token].log_mass if token in children else 0.)
                for token, value in enumerate(raw)))
            return _Node(raw, children, log_mass=mass)

        trie._root = decode(payload["root"], 0)
        if trie.max_nodes is not None and trie.node_count > trie.max_nodes:
            raise TrieCapacityError("Snapshot exceeds its declared node limit")
        trie._frozen = payload["frozen"]
        return trie

    def freeze(self):
        """Return an independent snapshot; later discovery updates cannot affect it."""
        payload = self.to_dict()
        payload["frozen"] = True
        return self.from_dict(payload, expected_identity=self.identity)

    @property
    def snapshot_id(self):
        return payload_hash(self.to_dict())
