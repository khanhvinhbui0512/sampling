"""Exact-context rho cache: pinned run values, bounded LRU trie snapshots.

Evicting a trie never changes a completed value in the same run. Disk snapshots
are optional, immutable JSON records; their historical NFE is not charged again.
Only successful, complete value queries can enter this cache.
"""

import json
import math
import os
from collections import OrderedDict
from dataclasses import asdict, replace

from src.cars.trie import FailureTrie
from src.config import project_path
from src.io import payload_hash, read_json, write_json
from src.types import (
    CostRecord, EstimateStatus, TargetKind, ValueEstimate, ValueTarget, require_count, require_id,
)


def _estimate_payload(estimate):
    return asdict(estimate)


def _read_estimate(payload):
    payload = dict(payload)
    payload["target"] = ValueTarget(**payload["target"])
    payload["cost"] = CostRecord(**payload["cost"])
    if payload["interval"] is not None:
        payload["interval"] = tuple(payload["interval"])
    return ValueEstimate(**payload)


def _fixed_value(estimate):
    return replace(estimate, cost=CostRecord(), cache_hit=False)


class CARSCache:
    NAMESPACE = "cars-rho"
    TARGET_KIND = TargetKind.RHO

    def __init__(self, metadata, *, max_entries, run_id, directory=None):
        require_count(max_entries, "resident trie entries", positive=True)
        require_id(run_id, "cache run_id")
        self.metadata = json.loads(json.dumps(metadata, allow_nan=False))
        if (self.metadata.get("schema_version") != 1
                or self.metadata.get("namespace") != self.NAMESPACE):
            raise ValueError("Cache namespace/schema differs")
        self.target = ValueTarget(**self.metadata["target"])
        if self.target.kind != self.TARGET_KIND:
            raise ValueError("Cache target kind differs")
        self.max_entries, self.run_id = max_entries, run_id
        self.directory = project_path(directory) if directory is not None else None
        self.cache_id = payload_hash(self.metadata)
        self._values, self._details, self._tries = {}, {}, OrderedDict()
        self.stats = {"value_hits": 0, "value_misses": 0, "trie_evictions": 0,
                      "trie_misses": 0, "disk_reads": 0, "disk_writes": 0, "computed_values": 0}

    def key(self, state):
        if asdict(state.rules) != self.metadata["token_rules"]:
            raise ValueError("Cache state token rules differ")
        return payload_hash({"metadata": self.metadata, "tokens": list(state.token_ids)})

    def path(self, state):
        if self.directory is None:
            raise ValueError("Disk caching is disabled")
        return project_path(os.path.join(self.directory, self.cache_id,
                                         payload_hash(self.run_id), self.key(state) + ".json"))

    def _remember_trie(self, key, trie):
        self._tries[key] = trie.freeze()
        self._tries.move_to_end(key)
        while len(self._tries) > self.max_entries:
            self._tries.popitem(last=False)
            self.stats["trie_evictions"] += 1

    def _expected_identity(self, state):
        return {"initial_tokens": list(state.token_ids), "rules": asdict(state.rules),
                "policy_id": self.metadata["policy_id"],
                "reference_id": self.metadata["reference_id"],
                "oracle_id": self.metadata["oracle_id"]}

    def _validate(self, state, estimate, trie):
        key = self.key(state)
        if (state.is_complete or estimate.value is None or estimate.target != self.target
                or not trie.is_frozen or trie.identity != self._expected_identity(state)):
            raise ValueError("Cache requires an available rho value and its matching frozen trie")
        if estimate.surviving_mass != trie.root_mass:
            raise ValueError("Value and trie surviving mass differ")
        if trie.max_nodes != self.metadata["estimator"]["max_nodes"]:
            raise ValueError("Trie memory protocol differs from cache metadata")
        if (estimate.status == EstimateStatus.ESTIMATED
                and estimate.draws != self.metadata["estimator"]["draws"]):
            raise ValueError("Cache requires the complete fixed-count protocol")
        if estimate.status == EstimateStatus.EXACT and not (
                trie.support_exhausted and estimate.value == 0. and estimate.draws == 0):
            raise ValueError("An exact partial-state value requires certified support exhaustion")
        if trie.mass_underflow:
            raise ValueError("Underflow is not a certified cache value")
        if key in self._values and _fixed_value(self._values[key]) != _fixed_value(estimate):
            raise ValueError("A completed value is frozen for the entire run")
        if key in self._details and self._details[key]["snapshot_id"] != trie.snapshot_id:
            raise ValueError("A completed value cannot acquire a different discovery snapshot")
        return key

    def _record(self, state, estimate, trie, details):
        record = {"schema_version": 1, "namespace": self.NAMESPACE, "metadata": self.metadata,
                  "key": self.key(state), "initial_tokens": list(state.token_ids),
                  "trie": trie.to_dict(), "estimate": _estimate_payload(estimate), "details": details}
        return {**record, "checksum": payload_hash(record)}

    def put(self, state, estimate, trie):
        key = self._validate(state, estimate, trie)
        if key in self._values:
            return  # Retain the first completed value and its original cost/provenance.
        details = self._provenance(estimate, trie)
        self._validate_details(details, estimate, trie)
        if self.directory is not None:
            path = self.path(state)
            record = self._record(state, estimate, trie, details)
            if os.path.exists(path):
                if read_json(path) != json.loads(json.dumps(record)):
                    raise FileExistsError("Existing cache record differs; use a fresh run_id")
            else:
                write_json(path, record)
                self.stats["disk_writes"] += 1
        self._values[key], self._details[key] = estimate, details
        self._remember_trie(key, trie)
        self.stats["computed_values"] += 1

    def _provenance(self, estimate, trie):
        return {"snapshot_id": trie.snapshot_id, "origin_run_id": self.run_id,
                "origin_draws": estimate.draws, "origin_cost": asdict(estimate.cost),
                "log_surviving_mass": None if trie.support_exhausted else trie.log_mass()}

    def _validate_details(self, details, estimate, trie):
        pass  # The native subclass also validates numerator/denominator diagnostics.

    def load(self, state, *, path=None):
        path = project_path(path) if path is not None else self.path(state)
        if not os.path.isfile(path):
            return False
        payload = read_json(path)
        checksum = payload.pop("checksum", None)
        if checksum != payload_hash(payload):
            raise ValueError("Cache snapshot checksum differs")
        if (payload.get("schema_version") != 1 or payload.get("namespace") != self.NAMESPACE
                or payload["metadata"] != self.metadata or payload["key"] != self.key(state)
                or payload["initial_tokens"] != list(state.token_ids)):
            raise ValueError("Cache snapshot metadata/full context differs")
        trie = self._read_trie(payload["trie"], state)
        estimate = _read_estimate(payload["estimate"])
        key = self._validate(state, estimate, trie)
        details = payload["details"]
        log_b = -math.inf if details["log_surviving_mass"] is None else details["log_surviving_mass"]
        if (details["snapshot_id"] != trie.snapshot_id or log_b != trie.log_mass()
                or details["origin_draws"] != estimate.draws
                or details["origin_cost"] != asdict(estimate.cost)):
            raise ValueError("Cache provenance and trie/value differ")
        self._validate_details(details, estimate, trie)
        self._values.setdefault(key, estimate)
        self._details.setdefault(key, details)
        self._remember_trie(key, trie)
        self.stats["disk_reads"] += 1
        return True

    def contains(self, state):
        key = self.key(state)
        if key not in self._values and self.directory is not None:
            self.load(state)
        return key in self._values

    def _read_trie(self, payload, state):
        return FailureTrie.from_dict(payload, expected_identity=self._expected_identity(state))

    def get(self, state):
        if not self.contains(state):
            self.stats["value_misses"] += 1
            return None
        key = self.key(state)
        self.stats["value_hits"] += 1
        if key in self._tries:
            self._tries.move_to_end(key)
        # No new estimation draws or model work; interval retains origin uncertainty.
        return replace(self._values[key], cost=CostRecord(), draws=0, cache_hit=True)

    def snapshot(self, state):
        key = self.key(state)
        if key not in self._tries and self.directory is not None:
            self.load(state)
        if key not in self._tries:
            self.stats["trie_misses"] += 1
            return None
        self._tries.move_to_end(key)
        return self._tries[key].freeze()

    def details(self, state):
        return dict(self._details[self.key(state)])


class NativeCARSCache(CARSCache):
    """Same pinned-value/LRU storage, separate native path schema and context.

The stored log_surviving_mass is NUMERATOR proposal mass B_W, not a bound on
the conditional ratio N/D. Values keep surviving_mass=None for this reason.
"""

    NAMESPACE = "cars-native-trajectory"
    TARGET_KIND = TargetKind.REFERENCE_CONDITIONAL

    def __init__(self, metadata, *, reference, reward, max_entries, run_id, directory=None):
        from src.cars.estimator import calibration_metadata
        from src.verifiers.interface import native_conditional_target

        super().__init__(metadata, max_entries=max_entries, run_id=run_id, directory=directory)
        self.reference, self.reward = reference, reward
        self._pending_summary = None
        if (self.target != native_conditional_target(reference, reward)
                or self.metadata["schedule"] != reference.policy.settings
                or self.metadata["native_reference"] != reference.protocol
                or self.metadata["calibration"] != calibration_metadata(reward)):
            raise ValueError("Native cache reference/target differs")

    def key(self, state):
        from src.cars.estimator import calibration_metadata

        if (self.metadata["native_reference"] != self.reference.protocol
                or self.metadata["calibration"] != calibration_metadata(self.reward)):
            raise ValueError("Native cache reference/calibration changed")
        return super().key(state)

    def put(self, state, estimate, trie, *, summary):
        self._pending_summary = asdict(summary)
        try:
            super().put(state, estimate, trie)
        finally:
            self._pending_summary = None

    def _provenance(self, estimate, trie):
        return {**super()._provenance(estimate, trie), "conditional_summary": self._pending_summary}

    def _validate_details(self, details, estimate, trie):
        summary = details.get("conditional_summary") or {}
        interval = summary.get("interval")
        if (summary.get("value") != estimate.value or summary.get("error") is not None
                or interval is None or tuple(interval) != estimate.interval
                or summary.get("denominator", 0) <= 0 or summary.get("denominator_interval", [0])[0] <= 0
                or summary.get("ess", 0) < self.metadata["estimator"]["min_denominator_hits"]):
            raise ValueError("Native cache numerator/denominator uncertainty differs")

    def _expected_identity(self, state):
        from src.cars.native import NativeTrajectoryTrie

        return NativeTrajectoryTrie(self.reference, self.reward, state,
                                    max_nodes=self.metadata["estimator"]["max_nodes"]).identity

    def _read_trie(self, payload, state):
        from src.cars.native import NativeTrajectoryTrie

        return NativeTrajectoryTrie.from_dict(payload, reference=self.reference, reward=self.reward, state=state)

    def _validate(self, state, estimate, trie):
        from src.cars.native import NativeTrajectoryTrie

        key = self.key(state)
        settings = self.metadata["estimator"]
        if not isinstance(trie, NativeTrajectoryTrie):
            raise ValueError("Native cache cannot store a rho trie")
        expected_draws = settings["denominator_draws"] + (0 if trie.support_exhausted else settings["numerator_draws"])
        if (not isinstance(trie, NativeTrajectoryTrie) or not trie.is_frozen or state.is_complete
                or trie.identity != self._expected_identity(state) or trie.max_nodes != settings["max_nodes"]
                or trie.mass_underflow or estimate.target != self.target or estimate.value is None
                or estimate.status != EstimateStatus.ESTIMATED or estimate.draws != expected_draws
                or estimate.interval is None or estimate.surviving_mass is not None):
            raise ValueError("Native cache requires a resolved complete N/D estimate and frozen native trie")
        if key in self._values and _fixed_value(self._values[key]) != _fixed_value(estimate):
            raise ValueError("A completed native value is frozen for the entire run")
        if key in self._details and self._details[key]["snapshot_id"] != trie.snapshot_id:
            raise ValueError("A completed native value cannot change its discovery snapshot")
        return key
