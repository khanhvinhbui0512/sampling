"""CARS remaining-mass trie, written independently from Algorithm 1 / Eq. 1–3.

Each expanded node stores the full base-policy distribution. Its surviving mass
is a logsumexp of base branch probabilities times child survival probabilities;
there is no subtraction of nearly equal probabilities. Missing children have
survival one. Only the caller's proven zero-reward cones may be rejected.

At capacity, learning at new nodes stops; existing exclusions remain in force.
Uncached nodes still sample the full base distribution. This bounds memory
without introducing a new rejection rule or changing the constrained target.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from src.config import config_digest
from src.schemas import require_count

TRIE_VERSION = "cars-log-mass-trie-v1"
CONTEXT_FIELDS = {"model", "tokenizer", "state", "group", "reward", "policy"}


class EmptyTrie(RuntimeError):
    """The remaining distribution has exactly zero mass."""


def normalized_logs(values: torch.Tensor) -> torch.Tensor:
    logs = values.detach().to(device="cpu", dtype=torch.float64).clone()
    if logs.ndim != 1 or not logs.numel() or torch.isnan(logs).any() or torch.isposinf(logs).any():
        raise ValueError("Expected a full, nonempty vector of finite/-inf log probabilities")
    total = torch.logsumexp(logs, 0)
    if not torch.isfinite(total):
        raise ValueError("Base policy has no probability mass")
    return logs - total


@dataclass(slots=True)
class _Node:
    log_probs: torch.Tensor | None = None
    blocked: torch.Tensor | None = None
    children: set[int] = field(default_factory=set)
    log_mass: float = 0.0


class RejectionTrie:
    """One trie belongs to one immutable model/tokenizer/root/reward/policy context.

    Prefixes contain stochastic token choices, excluding prompts and known MDLM
    coordinates. Their probabilities are conditional on the entire root state.
    Expansion caches probabilities but does not learn exclusions. The sampler
    defers reject_* calls until the frozen attempt has finished.
    """

    def __init__(self, context: Mapping[str, Any], *, max_nodes: int = 4096) -> None:
        if set(context) != CONTEXT_FIELDS:
            raise ValueError("Trie context must include model/tokenizer/state/group/reward/policy")
        require_count(max_nodes, "trie max_nodes", positive=True)
        # JSON round-trip semantics are enforced by config_digest.
        self.context = copy.deepcopy(dict(context))
        self.context_hash = config_digest(self.context)
        self.max_nodes = max_nodes
        self.nodes: dict[tuple[int, ...], _Node] = {(): _Node()}
        self.root_failed = False

    @staticmethod
    def _prefix(prefix: Sequence[int]) -> tuple[int, ...]:
        result = tuple(prefix)
        for token in result:
            require_count(token, "trie token ID")
        return result

    @property
    def log_mass(self) -> float:
        return -math.inf if self.root_failed else self.nodes[()].log_mass

    @property
    def mass(self) -> float:
        return math.exp(self.log_mass)

    @property
    def exhausted(self) -> bool:
        # A tiny nonzero mass may underflow in ordinary probability space.
        return self.log_mass == -math.inf

    def cached_logs(self, prefix: Sequence[int]) -> torch.Tensor | None:
        node = self.nodes.get(self._prefix(prefix))
        return None if node is None or node.log_probs is None else node.log_probs.clone()

    def expand(self, prefix: Sequence[int], log_probs: torch.Tensor) -> bool:
        prefix = self._prefix(prefix)
        if self.root_failed:
            raise EmptyTrie("Cannot expand an exhausted trie")
        node = self.nodes.get(prefix)
        if node is not None and node.log_probs is not None:
            return True
        if prefix:
            parent = self.nodes.get(prefix[:-1])
            if parent is None or parent.log_probs is None:
                return False
            token = prefix[-1]
            if token >= parent.log_probs.numel() or parent.blocked[token]:
                raise ValueError("Cannot expand an excluded or out-of-vocabulary branch")
            if node is None and len(self.nodes) >= self.max_nodes:
                return False
        logs = normalized_logs(log_probs)
        if node is None:
            node = _Node()
            self.nodes[prefix] = node
            self.nodes[prefix[:-1]].children.add(prefix[-1])
        node.log_probs = logs
        node.blocked = torch.zeros(logs.numel(), dtype=torch.bool)
        # An expanded node with no exclusions still has exact survival one.
        return True

    def _weights(self, prefix: tuple[int, ...], node: _Node) -> torch.Tensor:
        if node.log_probs is None:
            raise ValueError("Trie node has not been expanded")
        weights = node.log_probs.clone()
        weights[node.blocked] = -torch.inf
        for token in node.children:
            weights[token] += self.nodes[(*prefix, token)].log_mass
        return weights

    def conditional_logs(self, prefix: Sequence[int], raw: torch.Tensor | None = None) -> torch.Tensor:
        prefix = self._prefix(prefix)
        if self.root_failed:
            raise EmptyTrie("No surviving completion")
        # Also reject requests under an excluded ancestor.
        for depth, token in enumerate(prefix):
            ancestor = self.nodes.get(prefix[:depth])
            if ancestor is not None and ancestor.blocked is not None:
                if token >= ancestor.blocked.numel() or ancestor.blocked[token]:
                    raise EmptyTrie("Prefix belongs to a rejected cone")
        node = self.nodes.get(prefix)
        if node is None or node.log_probs is None:
            if raw is None:
                raise ValueError("Uncached prefix needs full base-policy probabilities")
            return normalized_logs(raw)
        weights = self._weights(prefix, node)
        total = torch.logsumexp(weights, 0)
        if not torch.isfinite(total):
            raise EmptyTrie("No surviving branch")
        return weights - total

    def _propagate(self, prefix: tuple[int, ...]) -> None:
        while True:
            node = self.nodes[prefix]
            node.log_mass = min(0.0, float(torch.logsumexp(self._weights(prefix, node), 0)))
            if not prefix:
                break
            prefix = prefix[:-1]

    def _mark_tokens(self, prefix: Sequence[int], token_ids: Sequence[int] | torch.Tensor) -> bool:
        prefix = self._prefix(prefix)
        node = self.nodes.get(prefix)
        if self.root_failed or node is None or node.log_probs is None:
            return False
        if isinstance(token_ids, torch.Tensor):
            if token_ids.ndim != 1 or token_ids.dtype != torch.long:
                raise TypeError("Rejected IDs must be a one-dimensional long tensor")
            ids = token_ids.detach().cpu()
        else:
            ids = torch.tensor(self._prefix(token_ids), dtype=torch.long)
        if not ids.numel():
            return False
        if torch.any(ids < 0) or torch.any(ids >= node.log_probs.numel()):
            raise ValueError("Rejected token outside vocabulary")
        if node.blocked[ids].all():
            return False
        node.blocked[ids] = True
        for token in tuple(node.children):
            if node.blocked[token]:
                cone = (*prefix, token)
                for descendant in tuple(self.nodes):
                    if descendant[:len(cone)] == cone:
                        del self.nodes[descendant]
                node.children.remove(token)
        return True

    def reject_tokens(self, prefix: Sequence[int], token_ids: Sequence[int] | torch.Tensor) -> bool:
        """Reject whole next-token cones; duplicate/overlapping updates are idempotent."""
        prefix = self._prefix(prefix)
        changed = self._mark_tokens(prefix, token_ids)
        if changed:
            self._propagate(prefix)
        return changed

    def learn(self, rejections: Sequence[tuple[Sequence[int], torch.Tensor]], *,
              failed_prefix: Sequence[int] | None = None) -> None:
        """Commit one attempt's evidence, recomputing each affected node once."""
        dirty: set[tuple[int, ...]] = set()
        for prefix, tokens in rejections:
            prefix = self._prefix(prefix)
            if self._mark_tokens(prefix, tokens):
                dirty.update(prefix[:depth] for depth in range(len(prefix) + 1))
        if failed_prefix is not None:
            failed_prefix = self._prefix(failed_prefix)
            if not failed_prefix:
                self.reject_prefix(())
                return
            parent = failed_prefix[:-1]
            if self._mark_tokens(parent, [failed_prefix[-1]]):
                dirty.update(parent[:depth] for depth in range(len(parent) + 1))
        for prefix in sorted(dirty, key=len, reverse=True):
            # A later, shorter failed cone can subsume earlier learned nodes.
            node = self.nodes.get(prefix)
            if node is not None and node.log_probs is not None:
                node.log_mass = min(0.0, float(torch.logsumexp(self._weights(prefix, node), 0)))

    def reject_prefix(self, prefix: Sequence[int]) -> bool:
        prefix = self._prefix(prefix)
        if not prefix:
            changed = not self.root_failed
            self.root_failed = True
            self.nodes = {(): _Node(log_mass=-math.inf)}
            return changed
        return self.reject_tokens(prefix[:-1], [prefix[-1]])

    def state_dict(self) -> dict[str, Any]:
        return {"version": TRIE_VERSION, "context": copy.deepcopy(self.context), "context_hash": self.context_hash,
                "max_nodes": self.max_nodes, "root_failed": self.root_failed,
                "nodes": [{"prefix": prefix, "log_probs": None if node.log_probs is None else node.log_probs.clone(),
                           "blocked": None if node.blocked is None else node.blocked.clone(),
                           "log_mass": node.log_mass}
                          for prefix, node in sorted(self.nodes.items(), key=lambda item: (len(item[0]), item[0]))]}

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any], *, context: Mapping[str, Any]) -> RejectionTrie:
        if (payload["version"] != TRIE_VERSION or payload["context_hash"] != config_digest(context)
                or config_digest(payload["context"]) != payload["context_hash"]):
            raise ValueError("Trie snapshot has a different version or context")
        trie = cls(context, max_nodes=payload["max_nodes"])
        rows = payload["nodes"]
        if not rows or len(rows) > trie.max_nodes or tuple(rows[0]["prefix"]) != ():
            raise ValueError("Invalid trie snapshot size/root")
        if type(payload["root_failed"]) is not bool:
            raise ValueError("Invalid failed-root flag")
        if payload["root_failed"]:
            if len(rows) != 1 or rows[0]["log_mass"] != -math.inf:
                raise ValueError("Invalid exhausted trie snapshot")
            trie.reject_prefix(())
            return trie
        seen = set()
        for row in rows:
            prefix = trie._prefix(row["prefix"])
            if prefix in seen or (prefix and prefix[:-1] not in seen):
                raise ValueError("Duplicate trie node or missing parent")
            seen.add(prefix)
            logs, blocked = row["log_probs"], row["blocked"]
            if logs is None:
                if prefix or blocked is not None or len(rows) != 1 or row["log_mass"] != 0.0:
                    raise ValueError("Only the initial root may be unexpanded")
                continue
            if not trie.expand(prefix, logs):
                raise ValueError("Trie snapshot exceeds capacity")
            node = trie.nodes[prefix]
            # Validate the cached distribution without renormalizing its saved
            # values: even a second logsumexp subtraction can change replay.
            saved_logs = logs.detach().to(device="cpu", dtype=torch.float64).clone()
            if not math.isclose(float(torch.logsumexp(saved_logs, 0)), 0.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("Snapshot base probabilities are not normalized")
            node.log_probs = saved_logs
            if (not isinstance(blocked, torch.Tensor) or blocked.dtype != torch.bool
                    or blocked.shape != node.log_probs.shape):
                raise ValueError("Invalid rejected-token mask")
            node.blocked = blocked.cpu().clone()
        for row in sorted(rows, key=lambda item: len(item["prefix"]), reverse=True):
            prefix = tuple(row["prefix"])
            node = trie.nodes[prefix]
            if node.log_probs is not None:
                if any(node.blocked[token] for token in node.children):
                    raise ValueError("Snapshot retains descendants of a rejected cone")
                actual = min(0.0, float(torch.logsumexp(trie._weights(prefix, node), 0)))
                stored = row["log_mass"]
                if (not isinstance(stored, (float, int)) or math.isnan(stored) or stored > 0.0
                        or (actual != stored and not math.isclose(actual, stored, rel_tol=1e-12, abs_tol=1e-12))):
                    raise ValueError("Trie snapshot mass disagrees with its branches")
                # Expansion uses exact survival one until an exclusion changes
                # the node. Preserve that and every saved mass bit on resume.
                node.log_mass = stored
        return trie
