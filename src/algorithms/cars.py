"""CARS (arXiv:2510.01902), adapted to sequential masked-token reveals.

The proposal reveals the leftmost mask at each step. Its distribution is the
product of those model conditionals, rather than the harness's diffusion
schedule. Trie updates are deferred until every draw in a batch has finished:
applying newly discovered masks during a draw would bias accepted samples.
"""

import math

import torch
from tqdm.auto import tqdm

from .algorithm_utils import (
    eos_token_ids,
    initial_states_and_prompts,
    logits,
    masked_positions,
    sampler_outputs,
    terminal_state,
)


class _TrieNode:
    def __init__(self):
        self.children = {}
        self.log_probs = None
        self.log_theta = None
        self.allowed = None

    def log_mass(self):
        if self.log_probs is None:
            return 0.0
        return float(torch.logsumexp(self.log_probs + self.log_theta, dim=0))


def _accept(task, harness, example, state, threshold):
    if hasattr(task, "reward_state"):
        reward = task.reward_state(example, state, harness)
    elif hasattr(task, "reward"):
        reward = task.reward(example, harness.decode_state(state))
    else:
        return bool(task.terminal_accept(example, harness.decode_state(state)))
    return reward is not None and math.isfinite(float(reward)) and float(reward) >= threshold


def _initialize_node(node, scores, task, harness, example, state, pos, config):
    scores = scores.detach().to(device="cpu", dtype=torch.float64).clone()
    scores[int(harness.mask_id)] = -math.inf
    if torch.isnan(scores).any() or torch.isposinf(scores).any() or not torch.isfinite(scores).any():
        raise ValueError("CARS requires finite model probability mass on a non-mask token")
    node.log_probs = torch.log_softmax(scores / config["temperature"], dim=0)
    node.log_theta = torch.zeros_like(node.log_probs)
    node.allowed = torch.ones_like(node.log_probs, dtype=torch.bool)
    # This optional oracle must only exclude tokens whose entire continuation
    # cone is invalid. It may conservatively allow tokens that later fail.
    if hasattr(task, "cars_token_mask"):
        allowed = task.cars_token_mask(example, list(state), harness, pos)
        if allowed is not None:
            allowed = torch.as_tensor(allowed, dtype=torch.bool, device="cpu")
            if allowed.shape != node.allowed.shape:
                raise ValueError("cars_token_mask must return one boolean per vocabulary token")
            node.allowed &= allowed
    node.allowed[int(harness.mask_id)] = False
    # With no partial-state oracle, exact terminal checks still let CARS learn
    # all invalid siblings at the final reveal, rather than just a rejected leaf.
    last_mask = len(masked_positions(state, harness.mask_id)) == 1
    if config.get("prune_terminal_tokens", True):
        tokens = ()
        if last_mask:
            tokens = range(len(scores))
        elif config["stop_on_eos"]:
            tokens = eos_token_ids(harness)
        for token in tokens:
            if not node.allowed[token] or not torch.isfinite(node.log_probs[token]):
                continue
            child = list(state)
            child[pos] = token
            node.allowed[token] = _accept(task, harness, example, child, config["reward_threshold"])


def _update_trie(path, rejected):
    child_mass = None
    for node, token in reversed(path):
        node.log_theta[~node.allowed] = -math.inf
        if child_mass is None and rejected:
            node.log_theta[token] = -math.inf
        elif child_mass is not None:
            node.log_theta[token] = min(float(node.log_theta[token]), child_mass)
        child_mass = node.log_mass()


def _sample_batch(harness, task, examples, states, prompts, roots, config):
    stats = [
        {"forward": 0, "backward": 0, "forced": 0, "nfe": 0, "steps": 0,
         "attempts": 0, "rejected": 0, "accepted": False, "cache_hits": 0,
         "budget_exhausted": False, "exhausted": False}
        for _ in examples
    ]
    budgets = [
        int(config["max_steps"]) if config.get("max_steps") is not None else
        max(1, len(masked_positions(state, harness.mask_id))) * config["max_steps_multiplier"]
        for state in states
    ]
    initial = [list(state) for state in states]
    finished = [False] * len(states)
    for idx, state in enumerate(states):
        if terminal_state(harness, state, config):
            stats[idx]["accepted"] = _accept(task, harness, examples[idx], state, config["reward_threshold"])
            stats[idx]["exhausted"] = not stats[idx]["accepted"]
            finished[idx] = True

    while not all(finished):
        paths = [[] for _ in states]
        rejected = [False] * len(states)
        nodes = list(roots)
        active = []
        for idx in range(len(states)):
            if finished[idx]:
                continue
            if roots[idx].log_mass() == -math.inf:
                stats[idx]["exhausted"] = True
                finished[idx] = True
                continue
            states[idx] = list(initial[idx])
            stats[idx]["attempts"] += 1
            active.append(idx)

        # Keep the proposal trie fixed throughout these complete attempts,
        # including when several examples explicitly share a sampling context.
        while active:
            positions = {idx: masked_positions(states[idx], harness.mask_id)[0] for idx in active}
            need_logits = {}
            for idx in active:
                if nodes[idx].log_probs is None:
                    need_logits.setdefault(nodes[idx], idx)
            # Harnesses may require rectangular state tensors.
            by_length = {}
            for idx in need_logits.values():
                by_length.setdefault(len(states[idx]), []).append(idx)
            evaluated = set()
            for indices in by_length.values():
                scores = logits(harness, [prompts[idx] for idx in indices], [states[idx] for idx in indices])
                for row, idx in enumerate(indices):
                    _initialize_node(nodes[idx], scores[row, positions[idx]], task, harness,
                                     examples[idx], states[idx], positions[idx], config)
                    stats[idx]["nfe"] += 1
                    evaluated.add(idx)
                del scores

            next_active = []
            for idx in active:
                node = nodes[idx]
                if idx not in evaluated:
                    stats[idx]["cache_hits"] += 1
                weights = node.log_probs + node.log_theta
                if not torch.isfinite(weights).any():
                    # A zero-mass cone is never sampled. This is possible when
                    # an existing trie was exhausted by a preceding batch.
                    stats[idx]["exhausted"] = True
                    finished[idx] = True
                    continue
                token = int(torch.multinomial(torch.softmax(weights, dim=0), 1))
                states[idx][positions[idx]] = token
                paths[idx].append((node, token))
                stats[idx]["forward"] += 1
                stats[idx]["steps"] += 1
                if not node.allowed[token]:
                    rejected[idx] = True
                elif terminal_state(harness, states[idx], config):
                    stats[idx]["accepted"] = _accept(task, harness, examples[idx], states[idx], config["reward_threshold"])
                    rejected[idx] = not stats[idx]["accepted"]
                    finished[idx] = stats[idx]["accepted"]
                elif stats[idx]["steps"] < budgets[idx]:
                    nodes[idx] = node.children.setdefault(token, _TrieNode())
                    next_active.append(idx)
                if stats[idx]["steps"] >= budgets[idx] and not stats[idx]["accepted"]:
                    stats[idx]["budget_exhausted"] = True
                    finished[idx] = True
            active = next_active

        for idx, path in enumerate(paths):
            _update_trie(path, rejected[idx])
            stats[idx]["rejected"] += int(rejected[idx])
    outputs = sampler_outputs(harness, states, stats)
    for output in outputs:
        if not output["stats"]["accepted"]:
            output["last_attempt_output"] = output["output"]
            output["output"] = ""
    return outputs


@torch.no_grad()
def sample_cars(harness, task, examples, verifier=None, config=None):
    """Return one bounded CARS result per example; inspect ``stats.accepted``.

    Exact acceptance uses ``task.reward_state`` or ``task.reward`` >=
    ``reward_threshold`` (default 1), falling back to ``terminal_accept``.
    A task may provide ``cars_token_mask(example, state, harness, position)``
    for sound prefix pruning, and ``cars_context_key(example, harness)`` to
    share a trie between examples with identical constraints. Otherwise tries
    are isolated. A learned verifier is unnecessary and is never consulted.
    """
    del verifier
    config = {"temperature": 1.0, "reward_threshold": 1.0,
              "max_steps_multiplier": 16, "stop_on_eos": False, **(config or {})}
    config["temperature"] = float(config["temperature"])
    config["reward_threshold"] = float(config["reward_threshold"])
    config["max_steps_multiplier"] = int(config["max_steps_multiplier"])
    if not math.isfinite(config["temperature"]) or config["temperature"] <= 0:
        raise ValueError("CARS temperature must be finite and positive")
    if not math.isfinite(config["reward_threshold"]) or config["reward_threshold"] <= 0:
        raise ValueError("CARS reward_threshold must be finite and positive")
    if config["max_steps_multiplier"] <= 0 or (config.get("max_steps") is not None and int(config["max_steps"]) <= 0):
        raise ValueError("CARS step budgets must be positive")
    batch_size = int(config.get("batch_size", 64))
    if batch_size <= 0:
        raise ValueError("CARS batch_size must be positive")
    roots = {}
    outputs = []
    starts = range(0, len(examples), batch_size)
    progress = tqdm(starts, desc=str(config.get("progress_desc", "CARS inference"))) if config.get("progress") else starts
    for start in progress:
        batch = examples[start:start + batch_size]
        states, prompts = initial_states_and_prompts(task, harness, batch, config)
        batch_roots = []
        for idx, (example, state, prompt) in enumerate(zip(batch, states, prompts)):
            context = task.cars_context_key(example, harness) if hasattr(task, "cars_context_key") else start + idx
            key = (context, prompt, tuple(state))
            batch_roots.append(roots.setdefault(key, _TrieNode()))
        outputs.extend(_sample_batch(harness, task, batch, states, prompts, batch_roots, config))
        if not hasattr(task, "cars_context_key"):
            roots.clear()
    return outputs
