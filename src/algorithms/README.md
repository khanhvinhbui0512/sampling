# CARS

Run QM9 inference with:

```bash
python src/qm9/inference.py --config config/qm9/qm9_inference.yaml --algorithm CARS
```

`--N 16` or `--max-steps-multiplier 16` sets the token-reveal budget to 16
times the initial number of masks. Configure CARS independently of value
guidance under `algorithms.CARS`:

```yaml
algorithms:
  CARS:
    max_steps_multiplier: 16
    reward_threshold: 1.0
    stop_on_eos: false
    prune_terminal_tokens: true
```

Other options are `max_steps` (an absolute reveal budget per output),
`batch_size` and `temperature` (defaulting to `generation` settings).
Temperature must be positive. CARS does not load a learned verifier.

The implementation follows [Constrained Adaptive Rejection Sampling,
Algorithm 1](https://arxiv.org/html/2510.01902v2#S3): it caches model
conditionals and invalid continuations in a trie, propagates surviving
probability mass to ancestors, and samples with weights `P(token) *
surviving_mass(child)`. Newly discovered invalid continuations are applied
after complete attempts, including successful attempts. Applying those masks
immediately would change the proposal within a draw and distort its distribution.

For masked diffusion models, the proposal reveals one mask at a time from
left to right. Conditioning refers to this sequential proposal; it differs
from the diffusion schedule used by `Base` generation. Acceptance requires
an exact task reward of at least `reward_threshold`. With QM9's default
`quantile_indicator` reward, this means a valid molecule meeting the configured
QED threshold. This is a hard acceptance event, not continuous reward weighting.

QM9 shares tries between examples with the same prompt and initial state,
since their reward function is identical. Full terminal vocabulary checks
learn invalid sibling completions at the final reveal. There is no assumption
that an RDKit failure on an incomplete SMILES proves its prefix invalid.
Set `prune_terminal_tokens: false` to skip terminal sibling enumeration and
learn only observed rejected completions. Other tasks can supply:

- `cars_token_mask(example, state, harness, position)`: a boolean vocabulary
  mask that only rules out tokens with no acceptable continuation. Returning
  `None` skips prefix pruning at that state.
- `cars_context_key(example, harness)`: a hashable identifier for identical
  constraints across examples. Without this hook, tries are isolated.

Every requested example returns a row. `stats.accepted` distinguishes successful
draws from failures; `budget_exhausted` and `exhausted` report the reason.
Unfinished states are never force-completed or counted as accepted. A finite
budget can censor a draw; failures emit an empty output and retain diagnostics
in `state_ids` and `last_attempt_output`. Only uncensored sampling has the paper's exactness
guarantee. `steps` and `forward` count reveals, `nfe` counts actual model
evaluations attributable to that row, and `cache_hits` counts reused conditionals.
Shared evaluations are charged once, so a later row can have zero NFE.

Run the focused tests with:

```bash
python -m unittest discover -s tests -v
```
