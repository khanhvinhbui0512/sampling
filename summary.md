# SMILES project: conversation summary and implementation status

Snapshot: **12 October 2026**. Source code has been written through **Step 7** of [implementation.md](implementation.md). The three-method experiment is still under development; no experimental results are available in the inspected workspace.

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
| Local `resource/CARS-code/` and `resource/MDM-VGB/` | Reference implementations; runtime code must not depend on these directories. |
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

During setup, `pip install -e ".[dev]"` failed with **`OSError: [Errno 28] No space left on device`** while installing build dependencies. The README now uses a project temporary directory for pip. This changes the temporary-file location; sufficient free disk space is still required. A successful reinstall has not been established by the available conversation evidence.

## 3. Research design: the seven questions

### 3.1. What does the task do?

Generate SMILES belonging to the requested chemical group. The primary binary terminal reward is:

$$
\tau_c(y)=\mathbf{1}\{y\in G_c\}\,
\mathbf{1}\{\mathrm{RDKitValid}(y)\}\,
\mathbf{1}\{\mathrm{SMARTS}_c(y)\}.
$$

The planned primary collection protocol targets **100 unique successful molecules per group and seed**, excluding prompt molecule identities, within **1,000 root attempts**. Default comparison seeds are **0, 1, and 2**. Primary, grammar-only replication, and smoke suites are separate.

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
| Retrosynthesis metric dependency | Pinned RetroStar reaction classifier and templates; metric computation belongs to a later step. |

The MDLM selects checkpoints by fixed-seed validation masked-ELBO loss. Llama requires Hugging Face access/license acceptance. Its download and RetroStar's download are not prerequisites for training the primary MDLM.

**Relation to the papers:** MDLM/verifier dimensions and optimization starting points follow MDM-VGB's QM9 configuration; the actual parameter counts are measured at runtime. ChEMBL vocabulary, length, and conditioning are adaptations. Llama is for the separate CARS replication arm.

### 3.4. How will CARS be used?

For the primary comparison, CARS will operate on a fixed **left-to-right completion policy `q_z`** derived from the shared MDLM. It must preserve all revealed coordinates, including known suffix tokens. Temperature is **1**, with the full clean-token vocabulary: no top-k/top-p truncation beyond excluding absorbing MASK as a clean token.

The planned trie stores rejected prefixes/leaves and their probability mass. A prefix may be pruned only when it is proved unable to yield reward 1. A terminal RDKit/SMARTS failure rejects that leaf unless a stronger prefix impossibility has been established. Unique-output selection and prompt exclusion remain outside the reward/trie failure definition.

The planned primary CARS configuration uses the equivalent of **`learn_level=3`** and **`constrain_first`**. The separate AR replication uses original prompts, temperature 1, top-p 1, top-k 0, BF16, and **256 new tokens**; its acceptance is grammar-only.

**Relation to the papers:** rejection-trie logic follows CARS. Adapting it to MDLM's fixed-order `q_z` is new. The probability guarantees require verification for this policy; they are not automatically guarantees for an ideal diffusion conditional. **The trie and CARS sampler have not yet been implemented.**

### 3.5. How will MDM-VGB be used?

Step 7 writes the learned-value training pipeline. Default rollouts use **128 absorbing reverse steps**, **five interior snapshots**, batch **32**, and an explicit final denoising step. Each snapshot inherits its trajectory's terminal reward; generation applies no chemical filtering. The default rollout count per group matches retained train rows carrying that group.

The verifier trains with **MSE**, LR **1e-4**, weight decay **0**, global batch **2,048**, micro batch **64**, and **200 epochs** by default. It selects minimum validation MSE and records Brier score, MAE, ECE, and reliability bins overall, by group, and by MASK fraction. Calibration is measured; no calibration transform is fitted.

The planned MDM-VGB search starts with `L_f=8`, `L_b=32`, `K=8`, block size `B=1`, `lambda=0`, temperature 1, random forward selection, low-confidence backward selection, and `N=16`. Validation will explore `N` in `{1,2,4,8,16}`. The primary kernel is ordinary VGB; momentum with `chi=1` is an optional setting. The `nN` search-step budget must be distinguished from model forward counts. Terminal values must use exact reward.

**Relation to the papers:** rollout depth/snapshot count and verifier architecture/optimization follow the QM9 reference. Class reward, conditioning, length, final denoising, identity partitions, and validation/calibration reporting are adaptations. **The MDM-VGB unmask/remask search kernel is still planned.**

### 3.6. How will CARS be integrated into MDM-VGB?

Keep one MDM-VGB transition kernel and substitute its value provider. For each partially masked state/group, the CARS provider will use `q_z` completions and the remaining probability mass before each attempt:

$$
\widehat V_q(z,c)=\frac{1}{m}\sum_{t=1}^{m}p_t\,\tau_c(Y_t),
\qquad p_t=q_z(\text{avoid }W_t).
$$

Here `W_t` contains only proven zero-reward branches. Sample using the trie/mass frozen for that attempt, evaluate the outcome, then update the trie. Failed attempts contribute zero and remain in the estimator. An accepted-only success rate would estimate the wrong quantity.

Start with fixed **`m=16`**, exploring `{4,8,16,32}` on validation. Cache keys must include state, group, model, tokenizer, reward, and policy context. Every inner completion and model forward counts toward the hybrid's compute cost. Terminal states use exact reward rather than a neural/Monte Carlo estimate.

**Relation to the papers:** replacing the learned verifier with this CARS estimator is the new experiment. Its estimator targets **`V_q`**, not an established ideal **`V*`**. Native reverse-rollout labels target the same trajectory's scheduled return; the optional left-to-right rollout dataset targets `q_z` return. These targets are recorded separately. Whether they approximate the ideal value must be evaluated. **Value providers and the hybrid have not yet been implemented.**

### 3.7. Which metrics will compare the methods?

Planned generation metrics include success per root attempt, attempts/time to 100 unique successes, attained sample count/timeouts, RDKit validity and original SMARTS membership on all completed candidates, uniqueness, novelty relative to prompt examples, and molecular diversity.

Diversity uses Morgan fingerprints with **radius 2 and 2,048 bits**, and mean pairwise Tanimoto distance. Radius 3 is a possible sensitivity check because the original evaluation code differs from the paper's radius-2 description. Retrosynthesis is a **one-step proxy** based on top-1 reaction probability **greater than 0.4**, not a claim that a full synthesis route exists. QED and Lipinski are not primary metrics for this benchmark.

Compute reporting will distinguish root attempts, prefix aborts, completed candidates, leaf visits, raw MDLM/AR forward counts, verifier calls, value queries, cache hits, inner hybrid completions, compute time, Modal end-to-end time, and memory. Compare quality at matched compute budgets with per-group/per-seed summaries and uncertainty.

Direct value evaluation will measure MSE/MAE/calibration and Monte Carlo uncertainty separately under `q_z` and the reference policy. Step 7 already contains validation-calibration reporting code, but no report has been generated.

**Relation to the papers:** molecular validity, membership, diversity, and the retrosynthesis proxy follow the CARS evaluation with explicitly documented differences. Paired three-method comparison, value accuracy, and complete inner-completion cost accounting are additions. **Final evaluation/reporting tasks remain planned.**

## 4. Code written through Step 7

The status below means **source exists for review**. It does not mean the code has passed execution or tests.

| Implementation step | Files/components present | Status |
|---|---|---|
| 1. Package and file management | `pyproject.toml`, `.gitignore`, `THIRD_PARTY_NOTICES.md`, package initializers. | Written. |
| 2. Shared contracts | `schemas.py`, `config.py`, `artifacts.py`, `tracking.py`, common task lifecycle, `configs/common.yaml`, toy fixtures. | Written. |
| 3. Modal and verification | `modal/common.py`, resource profiles, local/remote `verify.py`, verification configuration, resource-isolation tests. | Written; future verification phases still require later tests/artifacts. |
| 4. Benchmark and chemistry | Original benchmark assets, `data.py`, `tokenizer.py`, `grammar.py`, `reward.py`, T01 preparation, chemistry tests. | Written. |
| 5. Corpus and tokenizer tasks | T02 `prepare_corpus.py` and T03 `build_tokenizer.py`, with matching Modal scripts. | Written; no corpus/tokenizer artifacts present. |
| 6. Models and MDLM | Model settings, `autoregressive.py`, `mdlm.py`, `completion.py`, T04 downloads and T05 training, with Modal scripts. | Written; no downloaded/trained checkpoints present. |
| 7. Rollouts and verifier | Two configurations, T06 rollout builder, `verifier.py`, T07 trainer, and two Modal scripts. | Written; execution/review pending. |

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

Existing tests are `tests/conftest.py`, `tests/test_resource_isolation.py`, and `tests/test_data_reward.py`. Their presence supplies no passing-test evidence.

## 5. Runtime and artifact organization

The package is `src`, version **0.1.0**. Shared modules live directly in `src/`; algorithms are called by `src/tasks/`, while `modal/` contains remote infrastructure and entrypoints without becoming a package that shadows the Modal SDK.

Modal uses **`nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`**, Python **3.12**, and dependencies from `pyproject.toml`. Training/generation profiles request **A100-40GB**; the later evaluation profile requests **A10**. All stages share the configured **`smiles-artifacts`** Volume. The optional gated-model Secret is **`smiles-model-access`** containing `HF_TOKEN`.

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
| Task logs/reports | `outputs/<suite>/<task-id>/all/seed_<seed>/<run-id>/` for the currently implemented tasks |

Run directories also hold the framework's resume snapshots/journal. T05/T07 resume includes optimizer/model/RNG state; T07 additionally preserves its epoch/shard cursor. T06 preserves sampling RNG, completed batches, and partition/publication progress. Compatibility checks bind resumes to their configuration and input artifacts. Modal commits resume checkpoints to the Volume.

These paths describe intended runtime outputs. **None of `data/`, `checkpoints/`, `outputs/`, or `reports/` exists in the inspected local workspace.** Remote Volume contents were not inspected.

## 6. Verification and remaining implementation

The recent implementation work followed the instruction **do not run code**. Scripts, tests, training, model downloads, and Modal jobs were not executed by the assistant during Step 7 or this documentation task. No passing verification report, trained model, generated molecular dataset, or comparison metric is available here.

Remaining implementation order from the plan:

1. **Step 8 — CARS:** `src/trie.py`, `src/cars.py`, CARS/replication configurations, exact toy-distribution tests, and local/Modal T09 generation.
2. **Step 9 — Value providers and search:** `src/values.py`, `src/reference.py`, value tests, `src/mdm_vgb.py`, `src/hybrid.py`, baseline/VGB/hybrid configurations, local/Modal T08/T10/T11 generation, and integration tests.
3. **Step 10 — Direct value evaluation:** value-evaluation configuration and local/Modal T13.
4. **Step 11 — Metrics and reports:** molecular metrics, RetroStar proxy, metric tests, evaluation/report configurations, reporting helpers, and local/Modal T14/T16.
5. **Step 12 — Comparison orchestration:** comparison matrix/configuration, local/Modal T15 coordinator, and the remaining README workflow sections.

The scientific claim that CARS approximates ideal `V*`, and any claim that one method outperforms another, remain open. They require algorithm checks, independent value references, and matched-budget experiments.

After review and authorization to execute, verify the environment first, prepare T01 → T02 → T03, then run the separate smoke model through T05 → T06 → T07 before large training. The README supplies matching local/Modal commands. Optional T04 downloads support AR replication and retrosynthesis evaluation.

## 7. GitHub state and publishing this summary

Repository remote: [khanhvinhbui0512/sampling](https://github.com/khanhvinhbui0512/sampling).

At inspection:

- `origin` is `https://github.com/khanhvinhbui0512/sampling.git`.
- The current branch is **`smiles`**, tracking **`origin/smiles`**.
- Source HEAD is **`86db22e`**, message **`Add SMILES project`**.
- Local HEAD and the locally stored `origin/smiles` reference match. The live GitHub server was not queried for this snapshot.
- The working tree was clean before adding this document. `summary.md` is the new file to commit.

The chat supplied GitHub CLI browser-login and Git credential-helper commands. Successful authentication was not independently checked. The branch was created locally, and commit/push commands were provided; the observed commit and upstream reference are recorded above.

`.gitignore` excludes `resource/`, environments/caches/builds, credentials in `.env` files, and generated `data/`, `checkpoints/`, `outputs/`, and `reports/`. Source code, configurations, small benchmark assets, tests, and documentation remain in version control.

Commit and push this document from the repository root:

```bash
git add summary.md
git commit -m "Document SMILES project summary and status"
git push origin smiles
```

This document records the planning decisions, implementation scope, and observed repository state. It is not an experimental results report.
