# SMILES project: conversation summary and implementation status

Snapshot: **12 October 2026**, updated after the Step 12 implementation and the user's request to publish a handoff for another AI agent. Source code has been written through **Step 12** of [implementation.md](implementation.md). **Written does not mean executed, tested, or scientifically validated.** No experimental results are available in the inspected workspace.

The latest active request is to update this file with the project status and our discussion, then push the accumulated changes to GitHub. The preceding Step 13 request was interrupted before edits: section 13 of the plan lists generated artifacts, rather than a new executable task. Keep that distinction when continuing.

## 1. Project objective and references

Compare three methods on **one Molecular Generation (SMILES) benchmark**:

1. CARS.
2. MDM-VGB with a trained process verifier.
3. MDM-VGB with a CARS value estimate replacing the learned verifier.

The additional research question is whether CARS can provide a useful approximation to the value needed by MDM-VGB. The benchmark contains three chemical conditions: **acrylates, chain extenders, and isocyanates**. These are groups within one benchmark.

The requested project workflow is **PLAN → IMPLEMENT → VERIFY**. Planning is recorded in [PLAN.md](PLAN.md); file organization and implementation order are recorded in [implementation.md](implementation.md). [README.md](README.md) documents the implemented local and Modal workflows.

Materials considered in the planning and implementation:

| Reference | Role |
|---|---|
| [CARS paper, 2510.01902v2](https://arxiv.org/abs/2510.01902v2) | Molecular benchmark in Appendix H; CARS rejection sampling and experimental settings. |
| [MDM-VGB paper, 2606.28301v1](https://arxiv.org/abs/2606.28301v1) | Value guidance, Section 5, and experimental settings in Appendix F. |
| Local `resource/CARS-code/`, `resource/MDM-VGB/`, and reference PDFs | Referenced in the plan but absent from the inspected workspace; runtime code must not depend on them. Pinned online papers/source were consulted during implementation. |
| [Pinned CARS benchmark repository](https://github.com/pparys/cars/tree/f46607f983d89e4ca4f40527cd231ab04adc6ed7/datasets/smiles) | Full prompts and grammars, beyond the abbreviated examples shown in the paper. |

Source revisions, copied assets, checksums, and license information are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [assets/benchmark/source.json](assets/benchmark/source.json). The reference PDFs and repositories are excluded from Git.

## 2. Decisions and working rules from the chat

- Discussion was requested in Vietnamese; README documentation was explicitly requested in English. This summary is in English for the GitHub repository.
- Implement one requested step at a time, then stop for review. Recent implementation requests explicitly prohibited running code.
- Use `src/` directly, with local workflow scripts in `src/tasks/`; there is no `src/smiles_project/` nesting.
- Each implemented workflow task has a matching Python entrypoint in `modal/`, calling the same task logic.
- Run scripts from the repository root using `python script.py`; do not use `python -m`.
- Use project-relative paths and no `pathlib` in runtime code.
- Never import runtime code, models, or data from `resource/`. Copy selected assets with provenance or write independent project code.
- Scripts should show progress/logs through `tqdm` and print a short result.
- Local setup uses **Miniconda and pip**. Modal also uses pip. **uv is not used.**
- Python is **3.12** in the current package and Modal settings. PyTorch **2.8.0**, llguidance **0.7.30**, and RDKit **2025.3.6** are pinned in [pyproject.toml](pyproject.toml).
- README sections follow numbered workflow order; update only sections belonging to the requested step. The README has no separate status section.
- The ordered plan lists files to implement. Dependency-lock generation/setup actions were removed as standalone implementation tasks; `requirements.lock` is optional.
- The actual filename is lowercase `implementation.md`, although user prompts call it `Implementation.md`. If a required file is missing or unreadable, tell the user first; read papers and code before deciding details. Ask at most two clarification questions, then proceed with the available evidence.
- The latest user explicitly requested that this summary include both status and the decisions/discussion needed by another AI agent. It is a handoff, not a verbatim transcript or an experimental report.

During setup, `pip install -e ".[dev]"` failed with **`OSError: [Errno 28] No space left on device`** while installing build dependencies. The README now uses a project temporary directory for pip. This changes the temporary-file location; sufficient free disk space is still required. A successful reinstall has not been established by the available conversation evidence.

## 3. Research design and comparison protocol

### 3.1. What does the task do?

Generate SMILES belonging to the requested chemical group. The primary binary terminal reward is:

$$
\tau_c(y)=\mathbf{1}\{y\in G_c\}\,
\mathbf{1}\{\mathrm{RDKitValid}(y)\}\,
\mathbf{1}\{\mathrm{SMARTS}_c(y)\}.
$$

The implemented primary collection protocol targets **100 unique successful molecules per group and seed**, excluding prompt molecule identities, within **1,000 root attempts**. Default comparison seeds are **0, 1, and 2**. Primary, grammar-only replication, and smoke suites are separate. Timeouts and budget stops retain their actual counts; reaching fewer than 100 is not grounds for discarding or extrapolating a run.

The original grammar permits group motifs without requiring them to appear. The original chain-extender SMARTS matcher is also broad and does not guarantee two reactive functional groups. Preserve these benchmark definitions and report their limitations; stricter chemistry checks would be secondary analyses.

**Relation to the papers:** the benchmark groups and original assets follow CARS. Requiring grammar, RDKit validity, and class membership jointly is a new primary reward. The MDM-VGB QM9 experiment uses a QED-based target, which is not the target here.

### 3.2. How are the datasets prepared?

There are three distinct data roles:

| Data | Preparation |
|---|---|
| Benchmark examples | Preserve the original **32 acrylate, 11 chain extender, and 11 isocyanate** examples, prompts, grammars, and SMARTS. Report checker mismatches without silently removing examples. |
| MDLM training corpus | Download checksum-pinned **ChEMBL 36** chemical representations; sanitize/canonicalize molecules, filter by original SMARTS membership, exclude prompt molecule identities, and split unique molecules **80/10/10** before augmentation. |
| Verifier rollout dataset | Generate independent completions from fully MASKed roots; attach each final binary reward to its intermediate snapshots and partition by terminal identity. |

ChEMBL labels can overlap across groups. Training augmentation retains canonical SMILES and requests two random variants; validation/test are not augmented. The tokenizer vocabulary comes only from train rows, uses explicit PAD/MASK/EOS tokens, and initially permits **256 tokens including EOS**. Unknown and overlength rows are rejected and reported.

Verifier train/validation use a separate seeded terminal-identity partition, reserving about **10% of distinct identities** for validation with at least one identity in each partition. All snapshots from a rollout stay together. The same canonical molecule stays in one partition across groups; invalid outputs use an exact token-string identity.

**Relation to the papers:** benchmark assets follow CARS. ChEMBL training, augmentation, the shared tokenizer, and verifier identity partitions are new settings; the MDM-VGB molecular reference uses QM9. The 54 benchmark examples are not used as the entire training corpus.

### 3.3. Which models are used?

The primary comparison will share **one group-conditioned SMILES MDLM checkpoint** across all three methods. It is trained from scratch, with group/time conditioning and explicit fixed-width EOS/PAD handling.

| Component | Current configuration |
|---|---|
| Shared MDLM | 12 layers, hidden 768, 12 heads, dropout 0.1, length 256. |
| MDLM optimization | 25,000 updates, AdamW, LR 3e-4, 1,000 warmup updates, cosine decay, global batch 256, micro batch 32. |
| Learned verifier | 4 layers, hidden 256, 4 heads, dropout 0.1, group conditioning, mean pooling, sigmoid output. |
| Optional CARS AR replication | Pinned `meta-llama/Llama-3.1-8B-Instruct` model/tokenizer; separate from the shared-MDLM comparison. |
| Retrosynthesis metric dependency | Pinned RetroStar reaction classifier/templates and the T14 one-step proxy adapter. Missing models produce an unavailable metric. |

The MDLM selects checkpoints by fixed-seed validation masked-ELBO loss. Llama requires Hugging Face access/license acceptance. Its download and RetroStar's download are not prerequisites for training the primary MDLM.

**Relation to the papers:** MDLM/verifier dimensions and optimization starting points follow MDM-VGB's QM9 configuration; the actual parameter counts are measured at runtime. ChEMBL vocabulary, length, and conditioning are adaptations. Llama is for the separate CARS replication arm.

### 3.4. How is CARS used?

For the primary comparison, CARS operates on a fixed **left-to-right completion policy `q_z`** derived from the shared MDLM. It preserves all revealed coordinates, including known suffix tokens. Temperature is **1**, with the full clean-token vocabulary: no top-k/top-p truncation beyond excluding absorbing MASK as a clean token.

The trie stores rejected prefixes/leaves and their probability mass. A prefix may be pruned only when it is proved unable to yield reward 1. A terminal RDKit/SMARTS failure rejects that leaf unless a stronger prefix impossibility has been established. Unique-output selection and prompt exclusion remain outside the reward/trie failure definition.

The primary CARS configuration uses the equivalent of **`learn_level=3`** and **`constrain_first`**. Each attempt freezes trie mass, samples, then learns proven failures. The configured node cap freezes learning at new nodes while retaining existing exclusions. Trie, sampler RNG, selection and probability traces are persisted. The separate AR replication uses original prompts, temperature 1, top-p 1, top-k 0, BF16, and **256 new tokens**; its acceptance is grammar-only. Grammar-only successes without a valid molecular identity use explicitly labeled exact-text identities.

**Relation to the papers:** rejection-trie logic follows CARS. Adapting it to MDLM's fixed-order `q_z`, combined reward, bounded trie and resume is new. The probability guarantees require verification for this policy; they are not automatically guarantees for an ideal diffusion conditional. Step 8 source and finite-distribution checks exist, but the checks have not been run.

### 3.5. How is MDM-VGB used?

Step 7 writes the learned-value training pipeline. Default rollouts use **128 absorbing reverse steps**, **five interior snapshots**, batch **32**, and an explicit final denoising step. Each snapshot inherits its trajectory's terminal reward; generation applies no chemical filtering. The default rollout count per group matches retained train rows carrying that group.

The verifier trains with **MSE**, LR **1e-4**, weight decay **0**, global batch **2,048**, micro batch **64**, and **200 epochs** by default. It selects minimum validation MSE and records Brier score, MAE, ECE, and reliability bins overall, by group, and by MASK fraction. Calibration is measured; no calibration transform is fitted.

The MDM-VGB search starts with `L_f=8`, `L_b=32`, `K=8`, block size `B=1`, `lambda=0`, temperature 1, random forward selection, low-confidence backward selection, and `N=16`. Validation explores `N` in `{1,2,4,8,16}`. The primary kernel is ordinary VGB; momentum with `chi=1` is optional and requires explicit configuration. Shortlists sample with replacement and retain multiplicity. The `nN` search-step budget is distinct from actual model forward counts. Terminal values use exact reward; the positive guidance floor applies only to incomplete states.

The baseline uses the 128-step absorbing reverse sampler. Search stops at its `nN` limit by default, matching the reference QM9 setting; an explicitly enabled forward fallback is charged separately. Low-confidence remasking reads raw denoiser scores before absorbing copy. Known input observations and EOS/PAD contracts remain fixed. Baseline resumes between root rollouts; guided search resumes between steps.

**Relation to the papers:** rollout depth/snapshot count and verifier architecture/optimization follow the QM9 reference. Class reward, conditioning, length, EOS/PAD handling, final denoising, identity partitions, and validation/calibration reporting are adaptations. The full-graph stationary-law claim is not asserted for the practical shortlist kernel. Step 9 kernel/provider/generation code and integration checks exist; execution is pending.

### 3.6. How is CARS integrated into MDM-VGB?

Keep one MDM-VGB transition kernel and substitute its value provider. For each partially masked state/group, the CARS provider uses `q_z` completions and the remaining probability mass before each attempt:

$$
\widehat V_q(z,c)=\frac{1}{m}\sum_{t=1}^{m}p_t\,\tau_c(Y_t),
\qquad p_t=q_z(\text{avoid }W_t).
$$

Here `W_t` contains only proven zero-reward branches. Sample using the trie/mass frozen for that attempt, evaluate the outcome, then update the trie. Failed attempts contribute zero and remain in the estimator. An accepted-only success rate would estimate the wrong quantity.

Start with fixed **`m=16`**, exploring `{4,8,16,32}` on validation. Cache keys include state, group, model, tokenizer, reward, and policy context. Every inner completion and model forward counts toward the hybrid's compute cost. Terminal states use exact reward rather than a neural/Monte Carlo estimate. An interrupted fixed-m query does not publish a partial estimate. CARS uncertainty uses a bounded martingale interval rather than an IID standard error; independent MC references use Wilson intervals.

**Relation to the papers:** replacing the learned verifier with this CARS estimator is the new experiment. Its estimator targets **`V_q`**, not an established ideal **`V*`**. Native reverse-rollout labels target the same trajectory's scheduled return; the optional left-to-right rollout dataset targets `q_z` return. These targets are recorded separately. Whether they approximate the ideal value must be evaluated. Value providers and the hybrid are written; no experiment has established their accuracy or performance.

### 3.7. Which metrics will compare the methods?

Generation metrics include success per root attempt, attempts/time to target only when actually attained, attained sample count/timeouts, RDKit validity and original SMARTS membership on all completed root outputs, uniqueness, novelty relative to prompt examples, and molecular diversity. Root outputs, full search-candidate visits and selected samples are reported separately; selected-set validity is conditional on selection.

Diversity uses Morgan fingerprints with **radius 2 and 2,048 bits**, and mean pairwise Tanimoto distance. Radius 3 is implemented as sensitivity because the original evaluation code differs from the paper's radius-2 description. Novelty uses canonical molecule identity. Retrosynthesis is a **one-step proxy** based on top-1 reaction probability **greater than 0.4**, not a claim that a full synthesis route exists. Its main denominator is all completed outputs; the valid-only denominator is also reported. QED and Lipinski are excluded from primary metrics.

Compute reporting distinguishes root attempts, prefix aborts, completed candidates, leaf visits, raw MDLM/AR forward counts, verifier calls, value queries, cache hits, inner hybrid completions, compute time, recorded worker elapsed time, and memory. Inner forwards are included once in raw NFE; verifier calls stay separate. Adjusted NFE requires complete measured FLOPs. Modal queue/startup/RPC timing and unrecorded hard-kill work are not reconstructed from worker logs. Hardware model is not recorded by generation logs, so timing comparisons require consistent execution resources.

T13 directly measures MSE/MAE/calibration and Monte Carlo uncertainty separately under `q_z` and reverse-rollout reference policies. Defaults select up to 16 source molecules per group, mask fractions 0/0.25/0.5/0.75/1, fixed m=16 with three repetitions, and 512 independent reference completions in batches of 32. Both providers receive matched caps. State IDs are partitioned between validation/tuning and test/final; verifier training/validation identities and exact states are excluded. Missing state coverage and failed/budget-aborted queries remain explicit. Reverse-trained verifier predictions against `q_z` are cross-policy diagnostics, not proof of target equivalence.

T14 checks generation configuration, dependencies, checker versions and journals before writing checked metrics; invalid inputs produce explicit failed artifacts. T16 produces per-group/per-seed CSV, Markdown and PNG reports, seed-bootstrap intervals, paired comparisons at matching caps and observed quality–cost frontiers. Duplicate seed/arm observations are audited rather than treated as independent evidence. One seed has no bootstrap interval. Suites, backbones, purposes and variants remain separate.

**Relation to the papers:** molecular validity, membership, diversity, and the retrosynthesis proxy follow the CARS evaluation with explicitly documented differences. Paired three-method comparison, value accuracy, and complete inner-completion cost accounting are additions. T13/T14/T16 source and tests are written; no metrics or reports have been generated.

### 3.8. How are validation and final comparisons separated?

T15 covers CARS, learned MDM-VGB and hybrid, plus the supplemental baseline, all three groups and at least three seeds. Default job allowances are 16,384 or 65,536 raw model NFEs, or 60 or 300 tracked compute seconds. These allowances and the Cartesian N/m sweep are new project experiment settings, not paper hyperparameters. Identical baseline/CARS jobs are reused across sweep cells; learned jobs are reused across m values. Each cell receives its own report.

Validation uses logical seed + **2³¹**; final uses the listed seed. These are independent generation trials on the same prompt benchmark, **not held-out molecule splits**. T13 separately uses validation/test state partitions. Molecular comparison keys record purpose in `runtime.comparison`, and also distinguish RetroStar availability/source.

After reviewing completed validation reports, the user explicitly chooses a variant, such as `N16_m16`, with `--lock-from` and `--select-variant`. This writes a checked selection manifest. Non-smoke final runs require `--lock-manifest`, which supplies the chosen N/m and checkpoints and verifies source/configuration hashes, shared checkpoint/tokenizer/benchmark contents, protocol, budgets and RetroStar availability. T15 never trains, downloads, silently picks a checkpoint, or automatically selects the best test result.

The coordinator adopts completed/timeout/budget-exhausted generation jobs without rerunning them to fill the target. Failed or interrupted work can resume with identical inputs. Evaluations/reports use separate run directories and input-derived revisions so resumed generation does not silently reuse stale metrics. Modal uses a CPU coordinator, method-specific GPU generation workers, a separate evaluation worker and a CPU reporting worker; downstream reads follow worker commits and Volume reloads.

## 4. Code written through Step 12

The status below means **source exists for review**. It does not mean the code has passed execution or tests.

| Implementation step | Files/components present | Status |
|---|---|---|
| 1. Package and file management | `pyproject.toml`, `.gitignore`, `THIRD_PARTY_NOTICES.md`, package initializers. | Written. |
| 2. Shared contracts | `schemas.py`, `config.py`, `artifacts.py`, `tracking.py`, common task lifecycle, `configs/common.yaml`, toy fixtures. | Written. |
| 3. Modal and verification | `modal/common.py`, resource profiles, local/remote `verify.py`, verification configuration, resource-isolation tests. | Written; finite test phases and real-artifact prerequisites are configured. |
| 4. Benchmark and chemistry | Original benchmark assets, `data.py`, `tokenizer.py`, `grammar.py`, `reward.py`, T01 preparation, chemistry tests. | Written. |
| 5. Corpus and tokenizer tasks | T02 `prepare_corpus.py` and T03 `build_tokenizer.py`, with matching Modal scripts. | Written; no corpus/tokenizer artifacts present. |
| 6. Models and MDLM | Model settings, `autoregressive.py`, `mdlm.py`, `completion.py`, T04 downloads and T05 training, with Modal scripts. | Written; no downloaded/trained checkpoints present. |
| 7. Rollouts and verifier | Two configurations, T06 rollout builder, `verifier.py`, T07 trainer, and two Modal scripts. | Written; execution/review pending. |
| 8. CARS | `src/trie.py`, `src/cars.py`, CARS/replication configs, T09 local/Modal scripts, `tests/test_cars.py`, README section 8. | Written; not executed. |
| 9. Values and guided search | `src/values.py`, `src/reference.py`, `src/mdm_vgb.py`, `src/hybrid.py`, shared generation task code, baseline/VGB/hybrid configs, T08/T10/T11 local/Modal scripts, value/integration tests, README section 9. | Written; not executed. |
| 10. Direct value evaluation | `configs/value_eval.yaml`, T13 local/Modal scripts, `tests/test_value_eval.py`, README section 10. | Written; not executed. |
| 11. Molecular metrics and reports | `src/metrics.py`, `src/retrosynthesis.py`, `src/reporting.py`, evaluation/report configs, T14/T16 local/Modal scripts, `tests/test_metrics.py`, README section 11. | Written; not executed. |
| 12. Comparison orchestration | `configs/comparison.yaml`, T15 local/Modal scripts, `tests/test_compare.py`, verification integration, T14 comparison metadata, README section 12. | Written; not executed. |
| 13. Generated artifact inventory | Plan section 13 describes files emitted by the existing tasks. | Latest implementation request interrupted before edits or artifact generation. |

Current local/Modal workflow pairs:

| Task | Local entrypoint | Modal entrypoint |
|---|---|---|
| T12 verification | `src/tasks/verify.py` | `modal/verify.py` |
| T01 benchmark | `src/tasks/prepare_benchmark.py` | `modal/prepare_benchmark.py` |
| T02 corpus | `src/tasks/prepare_corpus.py` | `modal/prepare_corpus.py` |
| T03 tokenizer | `src/tasks/build_tokenizer.py` | `modal/build_tokenizer.py` |
| T04 optional downloads | `src/tasks/fetch_models.py` | `modal/fetch_models.py` |
| T05 shared MDLM | `src/tasks/train_mdlm.py` | `modal/train_mdlm.py` |
| T06 rollouts | `src/tasks/build_rollouts.py` | `modal/build_rollouts.py` |
| T07 learned verifier | `src/tasks/train_verifier.py` | `modal/train_verifier.py` |
| T08 baseline | `src/tasks/generate_base.py` | `modal/generate_base.py` |
| T09 CARS / separate AR replication | `src/tasks/generate_cars.py` | `modal/generate_cars.py` |
| T10 learned MDM-VGB | `src/tasks/generate_mdm_vgb.py` | `modal/generate_mdm_vgb.py` |
| T11 CARS-value hybrid | `src/tasks/generate_hybrid.py` | `modal/generate_hybrid.py` |
| T13 direct value evaluation | `src/tasks/evaluate_value.py` | `modal/evaluate_value.py` |
| T14 molecular evaluation | `src/tasks/evaluate.py` | `modal/evaluate.py` |
| T15 comparison / selection lock | `src/tasks/compare.py` | `modal/compare.py` |
| T16 report | `src/tasks/report.py` | `modal/report.py` |

Tests include `tests/conftest.py`, `test_resource_isolation.py`, `test_data_reward.py`, `test_cars.py`, `test_values.py`, `test_integration.py`, `test_value_eval.py`, `test_metrics.py`, and `test_compare.py`. T12 selects phases `environment`, `benchmark`, `corpus`, `cars`, `values`, `integration`, `metrics`, or `all`. The integration phase includes comparison tests. Finite fixtures avoid downloading models just to check algorithms and orchestration. Presence of these tests supplies no passing-test evidence.

Supporting changes include raw denoiser proposal logits in `src/mdlm.py`, provider state serialization in `src/schemas.py` and toy fixtures, verification phase wiring, and the revision-pinned RetroStar Git dependency in `pyproject.toml`. Modal installs Git for that dependency. Provenance and adaptations through Step 12 are appended in `THIRD_PARTY_NOTICES.md`.

## 5. Runtime and artifact organization

The package is `src`, version **0.1.0**. Shared modules live directly in `src/`; algorithms are called by `src/tasks/`, while `modal/` contains remote infrastructure and entrypoints without becoming a package that shadows the Modal SDK.

Modal uses **`nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`**, Python **3.12**, and dependencies from `pyproject.toml`. Training/generation profiles request **A100-40GB**; evaluation requests **A10**. The T15 coordinator and T16 reporting use CPU profiles. All stages share the configured **`smiles-artifacts`** Volume. The gated-model Secret is **`smiles-model-access`** containing `HF_TOKEN`, used by T04 downloads; comparison workers consume existing artifacts offline.

Local artifact root defaults to the repository root. Modal's repository directory is `/workspace`, with the Volume mounted at `/workspace/artifacts`; task paths remain relative to that artifact root. Local files and remote Volume contents are separate.

Important actual output patterns in the written implementation:

| Artifact | Logical path |
|---|---|
| Benchmark | `data/benchmark/examples.parquet`, `data/benchmark/manifest.json` |
| Corpus | `data/corpus/{train,validation,test}.parquet`, `data/corpus/manifest.json` |
| Tokenizer | `data/tokenizer/{vocab,metadata,lengths}.json` |
| Tokenized shards | `data/tokenized/<split>/shard_<id>.pt` and `manifest.json` |
| Rollouts | `data/rollouts/<suite>/seed_<seed>/<run-id>/`, containing raw batches, train/validation shards, `partitions.sqlite`, and `manifest.json` |
| Model exports | `checkpoints/{mdlm,verifier}/<suite>/seed_<seed>/<run-id>/step_<update>/`, with safetensors weights and metadata manifests |
| Non-generation task logs | `outputs/<suite>/<task-id>/all/seed_<seed>/<run-id>/` |
| Generation logs | `outputs/<suite>/<method>/<group>/seed_<seed>/<run-id>/`, with attempts/samples/events, resolved config, run record, generation report, checked generation manifest and sampler snapshot |
| Direct value evaluation | `data/value_eval/<suite>/<purpose>/seed_<seed>/<run-id>/`, with `states.pt`, checked shards, `reference_values.parquet` and a manifest; metrics are under the T13 output run |
| Molecular evaluation | T14 output run: `metrics.json`, `metrics_manifest.json`, `full_candidates.jsonl`; a metrics copy is under the generation run's `evaluations/<T14-run-id>/` |
| Comparison | T15 output run: `job_plan.json`, `job_results.json`, `comparison_manifest.json`; lock runs write `selection.json` and `selection_manifest.json` |
| Published reports | `reports/<suite>/<T16-run-id>/`: summary/aggregate/paired/frontier/value-accuracy/missing CSV, `quality_cost.png`, `value_accuracy.png`, `report.md`, provenance and manifest |

Run directories also hold the framework's resume snapshots/journal. T05/T07 resume includes optimizer/model/RNG state; T07 additionally preserves its epoch/shard cursor. T06 preserves sampling RNG, completed batches, and partition/publication progress. T09 resumes from saved attempt boundaries, T08 from saved root-rollout boundaries, T10/T11 from saved search steps, and T13 from saved complete queries. Compatibility checks bind resumes to their configuration and input artifacts. Durable interrupted model cost is retained by T13/T15 recovery logic; costs absent from durable hard-kill tracking remain unavailable. Modal commits resume checkpoints to the Volume.

The run-level `manifest.json` is a `RunRecord` with identity/status/tracking; artifact manifests such as `generation_manifest.json` or `metrics_manifest.json` carry file checksums. Do not confuse these contracts. Paths in plan section 13 are sometimes schematic; use the actual task functions/configs above as the authority for emitted filenames and run nesting. The optional `requirements.lock` is not a required generated task.

These paths describe intended runtime outputs. **None of `data/`, `checkpoints/`, `outputs/`, or `reports/` exists in the inspected local workspace.** Remote Volume contents were not inspected.

## 6. Conversation progress, verification and continuation

The implementation requests followed the instruction **do not run code**. The assistant did not execute project scripts, tests, compilation/lint checks, training, model downloads, or Modal jobs during Steps 8–12 or this summary update. Review used source/paper reading and Git whitespace/diff checks. No passing verification report, trained model, generated molecular dataset, or comparison metric is available here. Do not describe static review as successful runtime validation.

Recent conversation sequence:

1. The user asked the agent to read the project, especially this summary, for context.
2. The user requested Step 8 only, then asked for another review before moving on. Implementation and static review were carried out within the no-execution constraint.
3. The user successively requested Steps 9, 10 and 11, each implemented separately with matching Modal scripts, tests written for later use, short README additions, and a stop for review.
4. Step 12 added the matrix/coordinator. Discussion made the validation/final distinction explicit: independent generation seed streams on the same prompts, an explicit reviewed selection lock, shared checkpoint contents, reuse of identical sweep jobs, and retention of unfinished targets. T14 metadata prevents pooling purposes and reaction-model sources. Local and Modal call the same task functions.
5. The user requested Step 13. The agent read the plan and workspace, finding that section 13 is a generated-artifact inventory; the turn was interrupted before any Step 13 implementation/documentation edit.
6. The user then requested an updated summary and a GitHub push, and clarified that status plus our discussion should be recorded so another AI agent can continue. This handoff and the accumulated Step 8–12 changes are the publication scope.

For the next agent:

- Start with this file, then read `implementation.md`, `PLAN.md`, the relevant code/configs, README sections 8–12 and provenance. Source exists through Step 12; do not recreate those steps or assume they passed tests.
- Keep the pending Step 13 request visible, but do not manufacture generated datasets, checkpoints, metrics, reports or passing status under the no-execution rule. Its table describes outputs of existing tasks. Determine the concrete remaining artifact/documentation work from code before editing.
- Preserve the user's one-step scope and stop-for-review workflow. Publishing source does not authorize running experiments, tests or Modal jobs. Obtain an explicit execution request before starting them.
- The first runtime checks remain environment/dependency installation, the finite CARS/value/integration/metrics checks, and real benchmark/corpus/tokenizer preparation. Full `all` verification has artifact prerequisites and is not a substitute for the ordered workflow.
- After execution is authorized, use the separate smoke pipeline before large training and comparison. T15 requires existing T05/T07 sources; default final comparisons require a reviewed validation selection. T13 is a separate task and is not launched by T15.
- Check the current Git state before new edits. Runtime artifacts/caches/credentials are ignored; generated files in local and remote stores are separate. No remote Volume contents have been inspected.

The scientific claim that CARS approximates ideal `V*`, and any claim that one method outperforms another, remain open. They require algorithm checks, independent value references, and matched-budget experiments.

After review and explicit authorization to execute, verify the environment first, prepare T01 → T02 → T03, then run the separate smoke model through T05 → T06 → T07 before large training. The README supplies matching local/Modal commands, config overrides, gated-model access, artifact paths, selection locking and resume instructions. Optional T04 downloads support AR replication and retrosynthesis evaluation.

## 7. GitHub state and publication scope

Repository remote: [khanhvinhbui0512/sampling](https://github.com/khanhvinhbui0512/sampling).

At the start of this publication task:

- `origin` is `https://github.com/khanhvinhbui0512/sampling.git`.
- The current branch is **`smiles`**, tracking **`origin/smiles`**.
- Local HEAD is **`f3cfb25`**, message **`summary_10122026`**; the earlier source commit is `86db22e` (`Add SMILES project`).
- A live GitHub `ls-remote` check confirmed `origin/smiles` at `f3cfb2566695646a390861f7f10aebeefa05a022`, matching local HEAD before this update. No remote divergence was observed.
- Steps 8–12 were uncommitted local changes. Publication includes their source/configs/tests/README/provenance changes together with this updated summary, so the handoff describes code available in the same repository snapshot.

The user explicitly authorized updating the summary and pushing to GitHub. The target is the existing **`smiles`** branch on `origin`; no force push, branch replacement or history rewrite is intended. The final commit hash and confirmed push result belong in the completion message, since a document cannot embed the hash of the commit that contains it.

`.gitignore` excludes `resource/`, environments/caches/builds, credentials in `.env` files, and generated `data/`, `checkpoints/`, `outputs/`, and `reports/`. Source code, configurations, small benchmark assets, tests, and documentation remain in version control.

This document records the planning decisions, implementation scope, conversation progress and observed repository state. It is not an experimental results report. No scientific claim or passing-test result should be inferred from the source publication.
