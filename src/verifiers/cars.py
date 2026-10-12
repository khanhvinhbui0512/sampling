"""Separate CARS-rho and native conditional values through the VGB interface.

Both replace a source of values without changing the outer kernel or decoder.
Their targets and trajectory cache schemas remain distinct.
"""

from src.cars.cache import CARSCache, NativeCARSCache
from src.cars.estimator import CARSValueEstimator
from src.cars.native import NativeCARSValueEstimator
from src.verifiers.interface import Verifier


class CARSVerifier(Verifier):
    def __init__(self, estimator, *, cache, progress=True):
        if not isinstance(estimator, CARSValueEstimator) or not isinstance(cache, CARSCache):
            raise ValueError("CARSVerifier requires the rho estimator and exact-context cache")
        if cache.metadata != estimator.cache_metadata:
            raise ValueError("CARS cache uses a different estimator/context")
        super().__init__(estimator.target, estimator.reward)
        self.estimator, self.cache, self.progress = estimator, cache, progress
        self.runs = []  # Cold-query traces; cache hits retain provenance in cache.details().

    def required_work(self, states):
        missing = set()
        work = 0
        for state in states:
            self.estimator.check_context(state)
            if state.is_complete or state.token_ids in missing or self.cache.contains(state):
                continue
            missing.add(state.token_ids)
            work += self.estimator.required_work(state)
        return work, 0

    def values_for_states(self, states, *, rng, budget, reservation=None):
        for state in states:
            self.estimator.check_context(state)
        return super().values_for_states(states, rng=rng, budget=budget, reservation=reservation)

    def _estimate_batch(self, states, *, rng, budget, reservation):
        estimates = []
        for state in states:
            cached = self.cache.get(state)
            if cached is not None:
                estimates.append(cached)
                continue
            run = self.estimator.estimate(state, rng=rng, budget=budget, reservation=reservation,
                                          progress=self.progress)
            self.runs.append(run)
            if run.estimate.value is not None:
                self._store_run(state, run)
            estimates.append(run.estimate)
        return estimates

    def _store_run(self, state, run):
        self.cache.put(state, run.estimate, run.trie)


class NativeCARSVerifier(CARSVerifier):
    """Native conditional N/D adapter; the existing outer VGB kernel is unchanged."""

    def __init__(self, estimator, *, cache, progress=True):
        if not isinstance(estimator, NativeCARSValueEstimator) or not isinstance(cache, NativeCARSCache):
            raise ValueError("Native CARS requires a native trajectory estimator and cache")
        if cache.metadata != estimator.cache_metadata:
            raise ValueError("Native cache estimator/context differs")
        Verifier.__init__(self, estimator.target, estimator.reward)
        self.estimator, self.cache, self.progress = estimator, cache, progress
        self.runs = []

    def _store_run(self, state, run):
        self.cache.put(state, run.estimate, run.trie, summary=run.summary)
