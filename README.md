# CARS and MDM-VGB on QM9

The project has two required experiments: compare MDM-VGB and CARS adapted to QM9, then use CARS in MDM-VGB's value-verifier slot while keeping the backbone and outer kernel fixed. The main value target must use a declared native reference; `V_rho` and `pi_rho` experiments are intermediate checks. This is a new QM9 integration of CARS, whose original molecular experiment used different tasks/models. See [PLAN.md](PLAN.md) and [PAPER_CHECK.md](PAPER_CHECK.md) for the required scope and unfinished native-reference work.

## 1. Setup

Use Miniconda with Python 3.11. Run all project commands and tasks from the repository root.

Create and activate a dedicated Miniconda environment:

```bash
conda create --name cars-vgb-qm9 python=3.11 pip
conda activate cars-vgb-qm9
```

Install the project with the Modal SDK, analysis tools, and development dependencies:

```bash
pip install -e ".[modal,analysis,dev]"
```

The package uses the `src` namespace; do not add the `src/` directory itself to `PYTHONPATH`. Runtime paths are relative to the repository root. The `resource/` directory is for reference only.

Dependencies do not yet have a verified lock file. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for sources and licenses.

Follow [implementation.md](implementation.md) for the implementation order. Each task will have a local command using `python run.py --task <task>` and equivalent Modal files/scripts described in [modal/README.md](modal/README.md). These entrypoints are planned for later steps.

## 2. Prepare QM9 assets

Use `configs/qm9.yaml` for the QM9 dataset/tokenizer, 32-token sequences, and the upstream 5% held-out split with seed 42. Preparation checks the paper's 127,190 train and 6,695 test rows. Revision selectors are resolved to immutable Hub commit SHAs before downloading.

Prepare assets locally in the activated Miniconda environment:

```bash
python prepare_assets.py --config configs/qm9.yaml --asset-id qm9-v1
```

Authenticate with Modal once:

```bash
modal setup
```

Run the equivalent preparation task on Modal:

```bash
bash modal/run_prepare_assets.sh --config configs/qm9.yaml --asset-id qm9-v1
```

Or launch the same Modal task directly with Python:

```bash
python modal/prepare_assets.py --config configs/qm9.yaml --asset-id qm9-v1
```

Artifacts are saved under `workspace/data/qm9/<asset-id>/attempts/<attempt-id>/`. Each invocation creates a new attempt ID unless `--attempt-id` is supplied; existing attempts are never overwritten. The completed `manifest.json` identifies the raw dataset, tokenizer, split IDs, canonical train identities, audits, and hashes.

Modal settings are written directly in `modal/prepare_assets.py`: the required CUDA image, Python 3.11, 2 CPUs, 8,192 MiB memory, and a 3,600-second timeout. One Volume persists `workspace/` at `/root/workspace`; local runs use the same relative paths. `configs/qm9.yaml` is a plain settings file with no inheritance or override framework.

Preparation saves a standard fast tokenizer with the upstream SMILES regex and the IDs in the published `vocab.json`, checking encoded IDs against that rule for every audited row. The published vocabulary already includes special tokens; the old custom class adds an extra offset, which this project fixes. CLS/SEP are also exposed as BOS/EOS. Progress comes from the processing code; Modal wrappers only print the final result.

Verified on Modal on 2026-10-12: 127,190 train rows, 6,695 test rows, 35 tokens, and all 133,885 rows audited. The [completed run](https://modal.com/apps/reinforcement/main/ap-0EZnIR6PoV3xkA88Djqc7Q) saved `workspace/data/qm9/qm9-v1/attempts/3d6c18b413394019953f839048b57df2/manifest.json` on Volume `nam-cars-vgb-qm9-artifacts`. Asset preparation is the only pipeline task executed.

Preparation creates no base-model weights. Binary Pass@95 remains unavailable until independent BASE calibration. See [PAPER_CHECK.md](PAPER_CHECK.md) for matching settings, deliberate differences from upstream decoding, and items not yet verified at runtime.

## 3. Train the QM9 backbone

Use the completed assets manifest printed by Step 2. This task trains the local MDLM architecture from scratch; it does not download pretrained weights. `configs/base_training.yaml` follows upstream's batch 256, seed 1, 25,000 steps, and AdamW with warmup/cosine decay. These defaults come from the code; the paper does not specify them all.

Activate `cars-vgb-qm9`, then train locally with CUDA using the completed Step 2 manifest:

```bash
python train_base.py --assets-manifest workspace/data/qm9/qm9-v1/attempts/3d6c18b413394019953f839048b57df2/manifest.json
```

Run the equivalent task on Modal, using assets already prepared on its Volume:

```bash
bash modal/run_train_base.sh --assets-manifest workspace/data/qm9/qm9-v1/attempts/3d6c18b413394019953f839048b57df2/manifest.json
```

Or launch the Modal task with Python:

```bash
python modal/train_base.py --assets-manifest workspace/data/qm9/qm9-v1/attempts/3d6c18b413394019953f839048b57df2/manifest.json
```

Resume an intermediate checkpoint locally (replace `TRAIN_ATTEMPT`):

```bash
python train_base.py --assets-manifest workspace/data/qm9/qm9-v1/attempts/3d6c18b413394019953f839048b57df2/manifest.json --resume-manifest workspace/model_data/qm9/mdlm-qm9/attempts/TRAIN_ATTEMPT/checkpoints/step-0005000/manifest.json
```

Resume that task on Modal:

```bash
bash modal/run_train_base.sh --assets-manifest workspace/data/qm9/qm9-v1/attempts/3d6c18b413394019953f839048b57df2/manifest.json --resume-manifest workspace/model_data/qm9/mdlm-qm9/attempts/TRAIN_ATTEMPT/checkpoints/step-0005000/manifest.json
```

Each invocation creates a new attempt under `workspace/model_data/qm9/mdlm-qm9/attempts/`. Checkpoints every 5,000 steps include model config/weights, tokenizer, optimizer, scheduler, RNG, and hashes. Resume requires unchanged settings, assets, core source, software, and device environment; it continues the original schedule. A completed 25,000-step checkpoint is for inference. Local CPU runs use float32 with `--device cpu`.

Modal resources are hardcoded in `modal/train_base.py`: one RTX PRO 6000, 4 CPUs, 32,768 MiB memory, and a 24-hour timeout. Each checkpoint is committed to the shared Volume; local artifacts are not uploaded automatically. The returned `checkpoint_manifest` is the input for later sampling tasks.

Offline tests for token safety, tiny forward/backward, accounting, and checkpoint/resume are written in `tests/`. No code, tests, downloads, or Modal jobs have been run during this implementation. See [PAPER_CHECK.md](PAPER_CHECK.md) for architecture adaptations and precision differences from upstream.

## 4. BASE sampling and Pass@95 calibration

Use the completed assets manifest from Step 2 and checkpoint manifest from Step 3. `configs/reference_checks.yaml` separates native MDLM from the project's left-to-right completion `q_rho` and root joint `pi_rho`. The latter two support later CARS/value checks; they are not claimed to equal native diffusion conditionals.

Generate independent BASE calibration locally (replace both attempt placeholders):

```bash
python calibrate.py --assets-manifest workspace/data/qm9/qm9-v1/attempts/ASSET_ATTEMPT/manifest.json --checkpoint-manifest workspace/model_data/qm9/mdlm-qm9/attempts/TRAIN_ATTEMPT/checkpoints/step-0025000/manifest.json
```

Run the equivalent calibration on Modal with artifacts already on its Volume:

```bash
bash modal/run_calibrate.sh --assets-manifest workspace/data/qm9/qm9-v1/attempts/ASSET_ATTEMPT/manifest.json --checkpoint-manifest workspace/model_data/qm9/mdlm-qm9/attempts/TRAIN_ATTEMPT/checkpoints/step-0025000/manifest.json
```

Or launch the Modal task directly with Python:

```bash
python modal/calibrate.py --assets-manifest workspace/data/qm9/qm9-v1/attempts/ASSET_ATTEMPT/manifest.json --checkpoint-manifest workspace/model_data/qm9/mdlm-qm9/attempts/TRAIN_ATTEMPT/checkpoints/step-0025000/manifest.json
```

Calibration uses 10,000 fresh BASE-native draws, batch 256, seed 2026, and the 95th percentile with linear interpolation. Size, seed, interpolation, and 1,000 bootstrap replicates are project choices; the paper specifies an independent BASE calibration set. All invalid outputs contribute zero. Draws are kept with multiplicity; dataset rows and evaluation samples are not used. Unfinished draws fail the attempt.

Native sampling follows the upstream 32-step inference setting and log-linear absorbing posterior. The explicit final denoise removes residual MASK at `t=1e-5`, using at most one additional forward; reuse occurs only when the state batch is unchanged and the model has no time conditioning. This denoise is an addition to upstream's finite-endpoint sampler. BASE-native and BASE-rho preserve all known tokens and report actual batched forwards separately from oracle checks.

Artifacts are written under `workspace/data/qm9/calibration/qm9-base-pass95/attempts/<attempt-id>/`: raw samples, locked threshold/ID, sampling protocol, bootstrap uncertainty, RNG, costs, and hashes. Use the returned completed manifest for later reward/verifier tasks. Success requires a complete valid molecule even when the threshold is zero; the numeric upstream threshold is not reused.

Modal resources are hardcoded in `modal/calibrate.py`: one RTX PRO 6000, 4 CPUs, 32,768 MiB memory, and a 6-hour timeout. The shared Volume is committed on success or failure. Local runs can use `--device cpu` with float32. Step 4 code and offline tests have been written but have not been executed; see [PAPER_CHECK.md](PAPER_CHECK.md) for differences from the sources.

## 5. Verifier rollouts, Monte Carlo and neural training

Use completed manifests from Steps 2–4. Choose `rho` for comparison with MC/CARS, or `native` for the diffusion trajectory baseline. Keep their rollout data and verifier checkpoints separate. No training is needed for MC-rho.

Set the three input paths from earlier task outputs:

```bash
ASSETS=workspace/data/qm9/qm9-v1/attempts/ASSET_ATTEMPT/manifest.json
BASE=workspace/model_data/qm9/mdlm-qm9/attempts/TRAIN_ATTEMPT/checkpoints/step-0025000/manifest.json
CALIBRATION=workspace/data/qm9/calibration/qm9-base-pass95/attempts/CALIBRATION_ATTEMPT/manifest.json
```

Collect rho rollout labels locally:

```bash
python collect_rollouts.py --assets-manifest "$ASSETS" --checkpoint-manifest "$BASE" --calibration-manifest "$CALIBRATION" --policy rho
```

Collect the equivalent labels on Modal, with inputs already on its Volume:

```bash
bash modal/run_collect_rollouts.sh --assets-manifest "$ASSETS" --checkpoint-manifest "$BASE" --calibration-manifest "$CALIBRATION" --policy rho
```

`configs/rollouts.yaml` uses 127,190 independent rollouts, batch 256 and five interior snapshots, matching upstream's all-train-example rollout count. The QM9 generator is unconditional; no train/test molecules supply labels. Native uses 128 diffusion steps; rho reveals 32 positions from left to right. All snapshots receive their rollout's exact binary Pass@95 label, including invalid outputs labelled zero. The 5% validation split is by complete rollout group. Seed 1, split seed 42 and grouped validation are project choices.

Use `--shard-index i --num-shards n` to collect disjoint shards with the same config and shard count. Each invocation writes a completed manifest under `workspace/data/qm9/rollouts/qm9-rollouts-<policy>/attempts/`. Give training all shard manifests as one comma-separated argument, without spaces. Missing shards, mixed collections/targets, duplicate groups and split leakage are rejected; missing train or validation groups cause an error.

Train the rho neural verifier locally using the returned rollout manifest:

```bash
ROLLOUTS=workspace/data/qm9/rollouts/qm9-rollouts-rho/attempts/ROLLOUT_ATTEMPT/manifest.json
python train_verifier.py --rollout-manifests "$ROLLOUTS" --policy rho
```

Train the equivalent verifier on Modal:

```bash
bash modal/run_train_verifier.sh --rollout-manifests "$ROLLOUTS" --policy rho
```

Repeat collection and training with `--policy native` and its own manifests for neural-native. Its labels describe the 128-step trajectory distribution, marginalizing snapshot time; they are not automatically values for a fresh 32-step rollout. Rho training currently covers prefix snapshots; performance on arbitrary masks remains unverified. Neither learned baseline establishes reference consistency or accuracy for `V*` by itself.

`configs/verifier_training.yaml` follows upstream: token/position embeddings, Transformer hidden 256, four layers, four heads, dropout 0.1, mean pooling, sigmoid, MSE, AdamW lr 1e-4, weight decay 0, batch 2,048, seed 1, 200 epochs and gradient clipping 1. Vocabulary/length come from prepared assets. Grouped validation, single-GPU execution and checkpoints every ten epochs are project additions; training uses float32 and keeps the frozen backbone unchanged.

Resume an intermediate epoch locally with the same data, config and environment:

```bash
RESUME=workspace/model_data/qm9/verifier/qm9-verifier-rho/attempts/VERIFIER_ATTEMPT/checkpoints/epoch-0010/manifest.json
python train_verifier.py --rollout-manifests "$ROLLOUTS" --policy rho --resume-manifest "$RESUME"
```

Resume the equivalent task on Modal:

```bash
bash modal/run_train_verifier.sh --rollout-manifests "$ROLLOUTS" --policy rho --resume-manifest "$RESUME"
```

The returned checkpoint manifest binds weights, vocabulary, backbone, target, threshold, data hashes, optimizer, epoch and RNG. Loading requires an explicit matching target. Final-epoch weights are used; validation does not select a different checkpoint. Both Modal tasks hardcode one RTX PRO 6000, 4 CPUs, 32,768 MiB and 24 hours. Training commits each checkpoint; both tasks reload inputs and commit outputs/failures. Local tasks also accept `--device cpu`; direct `python modal/collect_rollouts.py` and `python modal/train_verifier.py` accept the same task arguments.

`src/verifiers/monte_carlo.py` provides fixed-count MC-rho through `values_for_states`, defaulting to 128 draws in batches of 256, with a plug-in standard error and 95% Wilson interval. These MC settings are project choices; Wilson coverage is approximate. Duplicate draws count separately. Insufficient budget returns `value=None` before a partial query starts. Both MC and neural adapters return exact terminal rewards without a learned-model forward and preserve input order. Actual forwards, sequence work and oracle checks are recorded separately. Step 5 code and offline tests are written but have not been run.

## 6. BoN, VGR, VGB and Momentum baselines

`configs/mdm_vgb.yaml` records the upstream QM9 defaults: singleton moves, L_f=8, L_b=32, K=8, N=16, lambda=0 and chi=1. K means draws with replacement, preserving multiplicity. The walk limit is 32 × 16 = 512 steps. The separate hard cap of 528 base sequences is a project choice: 16 × (32 steps + final denoise) funds a complete native BoN group. It includes nested queries and completion; large MC candidate groups can trigger fallback without reducing draw counts.

`BoNSampler` samples all N completions with the selected `BaseSampler`, ranks by binary reward and chooses the first tied index. Optional `ranking="qed"` is a project ablation. All candidates, including failures and duplicates, count toward cost. Insufficient work for all N prevents the group from starting. Completed outputs with reward zero remain unsuccessful.

`VGRSampler`, `VGBSampler` and `MomentumSampler` share backbone, verifier, completion `BaseSampler` and `CandidateSettings`. Both MC-rho and neural adapters are supported. Their files are under `src/sampling/`. `sample(...)` returns `SampleResult`; `run(...)` adds trace, settings, frozen full-state value records, RNG states and momentum direction for later persistence. These records support auditing, without mid-walk resume.

The candidate builder follows Definition 4.2: shared depth rescaling, terminal-gate exception, exact leaf rewards and no epsilon floor. Lambda zero skips parent-value queries. One raw-logit forward serves both position selections. VGR disables backtracking and uses empirical reference candidates when all values vanish, following Algorithm 2.

Success requires valid SMILES **and Pass@95**; upstream QM9 accepts validity alone. Failed leaves and zero-mass VGB states use labelled backtrack/completion escapes. Pure remasks cost no model forward, and original observations stay fixed. Guidance reserves unguided BASE completion first, then each whole value-query group. Unavailable values trigger fallback. At the step/work cap, native or rho BASE completes without guidance; upstream continues with shortlisted value guidance. Cooperative timeouts and unfinished runs count as failures.

Momentum defaults to `include_stay=true` from Definition 4.3; false selects upstream's renormalized move/switch variant. Shortlisting, early success and finite stopping are practical policies, without a stationary-distribution or exact V* claim. Native trajectory and rho completion values remain distinct. See [PAPER_CHECK.md](PAPER_CHECK.md).

Step 6 adds libraries and `tests/test_vgb.py`; code/tests have **not been run**. The Modal generation task belongs to Step 10; the unified local `run.py` belongs to Step 14. No task script is introduced here. Samplers emit tqdm progress and short results; Modal wrappers add no extra progress bar.

## 7. CARS trie, oracle and frozen proposal

`src/cars/trie.py` stores raw next-token probabilities and certified failure cones, updating surviving mass with log-sum-exp. Unknown branches retain mass one; duplicate/overlapping cones are counted once. `freeze()` creates an independent snapshot. JSON serialization binds the complete initial masked context, token rules, completion policy, reference and oracle; it recomputes mass on loading. Numerical mass underflow is distinct from certified support exhaustion.

`src/cars/oracle.py` defaults to `QM9Oracle(..., mode="leaf_only")`: only complete outputs with exact calibrated reward zero can be removed. Optional `token_constraints` can certify irreparable UNK, misplaced special tokens and known non-PAD tokens after EOS/PAD. These proofs apply to this project's strict decoder. Partial RDKit parse failure and partial QED are never pruning certificates. The finite grammar oracle is for toy verification only.

`src/cars/proposal.py` draws once from a frozen trie with `Q_W = q_rho × 1[avoids W] / B_W`, returning raw/proposal log probabilities and counted costs. It does not adapt, accept/reject, or estimate a value. Remaining mass `B_W` is not the success probability. This follows CARS's reweighting law; using QM9 masked contexts and the project's `q_rho` is a new integration, not the original molecular experiment or a native-MDM `V*` guarantee.

`configs/cars.yaml` records the contract; discovery work, fixed estimation count and memory cap remain unset until profiling. It is explicitly **not ready for benchmark execution**. The 528-sequence cap in the earlier baseline config cannot fund even one cold MC query with 128 draws and 31 masks. Before comparison, lock a feasible cost protocol and use the same target, kernel, checkpoint, decoding and reward across verifier variants; native results belong to a separately labelled comparison. See [PAPER_CHECK.md](PAPER_CHECK.md) for the required checks.

Step 7 adds the toy grammar and trie/oracle/proposal tests; none have been run. Lark is a development dependency for the toy grammar, with no QM9 runtime parser dependency. The estimator, discovery driver and persistent cache belong to Step 8. No task script is added here; local/Modal generation entrypoints remain scheduled for Steps 14/10. Proposal draws provide tqdm and a short result; an outer task may disable per-draw progress.

## 8. CARS value estimation, cache and adaptive sampling

`src/cars/estimator.py` discovers failure cones, freezes the trie, then uses an independent RNG for exactly `m` fresh pre-acceptance draws. It returns `V_hat = B_W × mean(reward)`, a scaled Hoeffding interval, separate discovery/estimation traces and actual costs. Failures and duplicates count. The whole query is reserved before discovery; an incomplete query is unavailable. Certified support exhaustion gives exact zero; numerical underflow is an error.

`src/cars/cache.py` binds the full masked state, checkpoint, tokenizer, calibration, rho, oracle, estimator settings and source hashes. Completed values stay fixed within a run. LRU eviction bounds resident tries; compact values remain in memory. Optional immutable JSON snapshots preserve raw probabilities and origin costs. Cache hits charge no new model work or estimation draws; disk I/O is not part of NFE.

`src/verifiers/cars.py` exposes ordered values through the existing verifier interface, with exact terminal rewards and duplicate reuse. Its target is **V_rho**. Native sampling and the outer VGB kernel are unchanged; native `V*` work belongs to Step 9.

`src/sampling/rejection.py` implements RS, leaf-only ARS and oracle CARS using the same `q_rho`. ARS/CARS keep their trie across attempts and calls in the same context. CARS checks supported sibling continuations after each complete draw. A node limit freezes learning; attempt/budget limits return explicit failures. Accepted generation outputs are separate from value-estimation draws.

Step 8 adds `tests/test_value_estimator.py` and `tests/test_cache.py`, plus a shared finite-reward fixture. Code and tests have **not been run**. Counts and memory limits in `configs/cars.yaml` remain unset pending profiling. This step adds libraries, with tqdm and short results in their processing loops; local/Modal task entrypoints remain scheduled for Steps 14/10. No new task script is introduced.

## 9. Native conditional value and matched controls

`NativeReference` follows the locked ALLMASK native root law: 32 log-linear steps and counted final denoise, using the same start-of-step logits for every parallel coordinate. It retains MASK self loops and exposes path probabilities. This is the project's normalized baseline; the upstream 128-step snapshot target stays separate. The calibrated BASE reference and threshold must match; this step never recalibrates reward.

`NativeCARSValueEstimator` estimates `V*(z) = N(z)/D(z)`: discovery prunes incompatible observations or exact terminal failures for the numerator, then freezes the trajectory trie before independent fixed-count estimation. The denominator uses independent unpruned root draws, including failures and duplicates. `NativeConditionalMonteCarloVerifier` uses paired root counts. Both return unavailable for unresolved denominators, insufficient compatible hits or numerical underflow. The Hoeffding ratio interval is conservative; the finite-sample ratio can be biased. `src/verifiers/importance.py` keeps the pi_rho diagnostic separate and requires verified support plus a global weight bound for altered proposals; scoring work is counted.

The neural control draws native-root terminals and chooses posthoc mask patterns independently of tokens/reward. It keeps the existing architecture/MSE and terminal-group split. Each mask subset has positive support. Its target ID matches MC/CARS native values; upstream trajectory labels and rho checkpoints cannot be substituted. Native cache snapshots use their own schema/context and retain numerator/denominator uncertainty and origin costs. `NativeCARSVerifier` uses the existing ordered value interface and exact terminal reward anchoring without changing the outer VGB kernel. The conditional target requires `D(z)>0`.

After profiling, set `num_rollouts` and `snapshots_per_rollout` in `configs/native_conditional_rollouts.yaml`, then collect the new control locally using the manifest variables from section 5:

```bash
python collect_rollouts.py --assets-manifest "$ASSETS" --checkpoint-manifest "$BASE" --calibration-manifest "$CALIBRATION" --policy native-conditional --config configs/native_conditional_rollouts.yaml
```

Collect the equivalent control on Modal:

```bash
bash modal/run_collect_rollouts.sh --assets-manifest "$ASSETS" --checkpoint-manifest "$BASE" --calibration-manifest "$CALIBRATION" --policy native-conditional --config configs/native_conditional_rollouts.yaml
```

Train locally using the returned rollout manifest, or all shard manifests separated by commas:

```bash
CONDITIONAL_ROLLOUTS=workspace/data/qm9/rollouts/qm9-rollouts-native-conditional/attempts/ROLLOUT_ATTEMPT/manifest.json
python train_verifier.py --rollout-manifests "$CONDITIONAL_ROLLOUTS" --policy native-conditional
```

Train the equivalent control on Modal:

```bash
bash modal/run_train_verifier.sh --rollout-manifests "$CONDITIONAL_ROLLOUTS" --policy native-conditional
```

Static review added stable ESS calculation, a guard against losing positive proposal edges to underflow, full calibration checks for trie learning, and streamed sibling checks to reduce temporary memory. Regression tests are written but **not run**. Profiling must measure end-to-end query wall time, including trie work; `CostRecord.elapsed_seconds` sums instrumented model/oracle times only.

Step 9 code and enumeration/integration tests are written but **not run**. Native MC/CARS counts, hit thresholds, memory and work caps in `configs/cars.yaml` remain `null` pending QM9 profiling. Rare compatible events may make this estimator unavailable or too costly. Check the numerical agreement of serial path scoring and batched native sampling at the chosen precision. The native bridge and independent-mask control are new project designs, not experiments already implemented in either paper. Enumeration, uncertainty and QM9 feasibility gates remain pending. Generation/value-panel entrypoints belong to Step 10; no benchmark or stationary-law guarantee is claimed.

## 10. Generation and fixed value panels

Step 10 adds three local tasks and their Modal equivalents. The examples request one output to inspect the wiring; this is a project example, not a paper benchmark. Use the `ASSETS`, `BASE` and `CALIBRATION` manifests from the earlier sections.

Generate a native BASE output locally:

```bash
python generate.py --assets-manifest "$ASSETS" --checkpoint-manifest "$BASE" --calibration-manifest "$CALIBRATION" --method base_native --num-samples 1 --seed 1
```

Generate the equivalent output on Modal:

```bash
bash modal/run_generate.sh --assets-manifest "$ASSETS" --checkpoint-manifest "$BASE" --calibration-manifest "$CALIBRATION" --method base_native --num-samples 1 --seed 1
```

Supported methods are `base_native`, `base_rho`, `bon`, `rs`, `ars`, `cars`, `vgr`, `vgb` and `vgb_momentum`. Standalone RS/ARS/CARS use `q_rho` and are labelled rho adaptations. Guided methods choose `--verifier neural|mc|cars` and `--value-law native-conditional|rho|upstream-native`; the last law accepts only its neural snapshot checkpoint. Rho guided comparisons require `base_method: base_rho` in the inference config.

Set profiled count/cap fields in `configs/cars.yaml` before CARS/native-MC runs. Increase the total work cap in `configs/mdm_vgb.yaml` if needed: 528 sequence-forwards funds BoN but can refuse an entire nested value group. Neural guidance alone needs `--verifier-manifest`; native guidance requires the independent-mask control checkpoint from section 9.

Generate with CARS estimating native V* locally:

```bash
python generate.py --assets-manifest "$ASSETS" --checkpoint-manifest "$BASE" --calibration-manifest "$CALIBRATION" --method vgb --verifier cars --value-law native-conditional --num-samples 1 --seed 1
```

Generate the equivalent CARS-guided output on Modal:

```bash
bash modal/run_generate.sh --assets-manifest "$ASSETS" --checkpoint-manifest "$BASE" --calibration-manifest "$CALIBRATION" --method vgb --verifier cars --value-law native-conditional --num-samples 1 --seed 1
```

Generate standalone CARS-QM9 locally after setting its attempt/node caps:

```bash
python generate.py --assets-manifest "$ASSETS" --checkpoint-manifest "$BASE" --calibration-manifest "$CALIBRATION" --method cars --value-law rho --num-samples 1 --seed 1
```

Generate the equivalent standalone CARS run on Modal:

```bash
bash modal/run_generate.sh --assets-manifest "$ASSETS" --checkpoint-manifest "$BASE" --calibration-manifest "$CALIBRATION" --method cars --value-law rho --num-samples 1 --seed 1
```

Set `panel.max_per_stratum` in `configs/value_checks.yaml` and lock a panel from completed BASE/baseline VGB manifests. Selection uses source order and fixed masks, without filtering on validity/reward; CARS and BoN sources are rejected. Omitting sources uses independent root trajectories and also requires `panel.terminal_draws`. Posthoc masks and actual trajectory snapshots have distinct provenance.

Prepare the panel locally using the returned BASE generation manifest:

```bash
BASE_RUN=workspace/results/raw/qm9-generate/attempts/GENERATION_ATTEMPT/manifest.json
python prepare_value_panel.py --assets-manifest "$ASSETS" --checkpoint-manifest "$BASE" --calibration-manifest "$CALIBRATION" --source-manifests "$BASE_RUN" --value-law native-conditional
```

Prepare the equivalent panel on Modal:

```bash
bash modal/run_prepare_value_panel.sh --assets-manifest "$ASSETS" --checkpoint-manifest "$BASE" --calibration-manifest "$CALIBRATION" --source-manifests "$BASE_RUN" --value-law native-conditional
```

Check finite toy values against exact enumeration locally on CPU:

```bash
python check_values.py --mode toy --value-law native-conditional --run-id toy-native-check
```

Run the equivalent toy check on a Modal CPU container:

```bash
bash modal/run_check_values.sh --mode toy --value-law native-conditional --run-id toy-native-check
```

Set the independent MC reference counts in `configs/value_checks.yaml` before QM9 checks. QM9 reference values have uncertainty; they are not exact ground truth. `--value-law rho` creates/checks a separate rho panel. Toy rho checks also show the distinct `pi_rho` conditional diagnostic.

Compare MC/neural/CARS locally on the locked native panel:

```bash
PANEL=workspace/data/qm9/value_panels/qm9-value-panel/attempts/PANEL_ATTEMPT/manifest.json
CONDITIONAL_VERIFIER=workspace/model_data/qm9/verifier/qm9-verifier-native-conditional/attempts/VERIFIER_ATTEMPT/checkpoints/epoch-0200/manifest.json
python check_values.py --mode qm9 --assets-manifest "$ASSETS" --checkpoint-manifest "$BASE" --calibration-manifest "$CALIBRATION" --panel-manifest "$PANEL" --estimators mc,neural,cars --verifier-manifest "$CONDITIONAL_VERIFIER"
```

Run the equivalent comparison on Modal:

```bash
bash modal/run_check_values.sh --mode qm9 --assets-manifest "$ASSETS" --checkpoint-manifest "$BASE" --calibration-manifest "$CALIBRATION" --panel-manifest "$PANEL" --estimators mc,neural,cars --verifier-manifest "$CONDITIONAL_VERIFIER"
```

Generation saves every requested output, per-output traces/value queries, RNG, budget by phase, synchronized wall time and a completed manifest under `workspace/results/raw/`. Unresolved estimates produce explicit failed samples; budget refusal can use the declared reserved completion. Value checks retain unavailable estimates and reference costs under `workspace/results/value_checks/`. Setup time includes warm-cache loading; historical cache costs remain provenance and are not charged again.

Cold mode starts with a fresh cache scope. To reuse values, enable `persistent_cache` in the appropriate CARS block and add `--cache-mode warm --warm-cache-manifest SOURCE_MANIFEST` to generation/check commands. The source must contain registered snapshots with the exact state/model/reward/estimator context. Tagged `{"nonfinite": "-inf"}` values encode log diagnostics in strict JSON.

Source/config/tests for Step 10 are written but **not run**. The runner preserves the project's documented baseline differences; it does not reproduce either paper's tables. Step 9 correctness/feasibility gates remain pending. No benchmark matrix, evaluation report or Step 11 work is implemented here.

## 11. Evaluate saved artifacts

The evaluator reads completed generation/value-check attempts, the locked panel, tokenizer, calibration and canonical training identities. It does not load a backbone or generate samples. Reports are new immutable attempts under `workspace/results/summary/`; raw artifacts stay unchanged.

Install the plotting dependencies in the existing Miniconda environment:

```bash
pip install -e ".[analysis]"
```

Evaluate explicitly selected generation and value-check attempts locally (replace the attempt placeholders):

```bash
GEN_RUN=workspace/results/raw/qm9-generate/attempts/GENERATION_ATTEMPT/manifest.json
VALUE_RUN=workspace/results/value_checks/qm9-check-values/attempts/VALUE_ATTEMPT/manifest.json
python evaluate.py --manifests "$GEN_RUN,$VALUE_RUN" --comparison-id qm9-comparison
```

Run the equivalent evaluation on Modal CPU and download selected reports:

```bash
bash modal/run_evaluate.sh --manifests "$GEN_RUN,$VALUE_RUN" --comparison-id qm9-comparison --download "report.md,metrics.json,metrics.csv,audit.jsonl"
```

Select all attempts for known run IDs locally:

```bash
python evaluate.py --run-ids qm9-generate,qm9-check-values --comparison-id qm9-all-attempts
```

Select the same run IDs on Modal:

```bash
bash modal/run_evaluate.sh --run-ids qm9-generate,qm9-check-values --comparison-id qm9-all-attempts
```

Pass@95, validity, unique Pass@95 and zero-filled QED use all requested outputs, including failed/unfinished/invalid outputs. Mean valid QED, uniqueness and novelty use completed valid outputs. Novelty follows the released MDM-VGB convention: unique canonical molecules outside TRAIN divided by valid outputs; `novelty_of_unique` is also saved. Empty conditional metrics are `null`.

`configs/evaluation.yaml` locks Morgan radius 2 / 2,048 bits from the CARS paper; released CARS code uses radius 3. The project caps diversity at a seeded uniform subset of 1,000 valid outputs, retaining duplicates and recording selected indices. This is an evaluator cost cap, not a benchmark output count. Estimator draws are never deduplicated or used as generated molecules.

Reports contain `metrics.json`, `metrics.csv`, `runs.jsonl`, `audit.jsonl`, `report.md` and applicable `figures/{quality_cost,value_error_cost,cache_cost}.png`. Config/model/calibration/value-target/cache families stay separate. Generation uncertainty uses independent seeds; one seed has no seed CI. Value reports retain unavailable states, reference intervals and N/D/ESS diagnostics; rollout-cluster intervals keep related masks together. Individual available-pair errors and common-pair errors are separate; plots use the states available for every estimator in the selected cohort, showing their count and availability. Full-panel MAE is `null` when any pair is missing. QM9 errors use a noisy independent MC reference; only toy enumeration is exact.

Costs include nested work once, with raw NFE, sequence forwards, phases, sampling/query wall, setup and historical cache origin work reported separately. Neural adjusted NFE remains `null` without measured FLOP provenance. Missing/failed/corrupt attempts stay in the audit, including verified partial generation costs as a lower bound. Incomplete selections and duplicate attempts for the same config/seed save a report and return a nonzero exit code; choose explicit attempts after reviewing the audit.

Modal downloads only named report artifacts plus their manifest to `workspace/downloads/evaluation/<comparison>/<attempt>/`, without overwriting files. Add existing figure paths to `--download` or change `--download-dir`. A selected download is a partial report bundle; the manifest lists all remote artifacts. New value-check artifacts record the panel path and MC/execution settings; older artifacts missing the panel path are audited as unusable, and older MC protocols are not pooled across attempts.

Step 11 source and hand-checkable regression tests are written but **not run**. No evaluation report or numeric validation is claimed. The evaluator verification gate, Step 9–10 gates and Step 12 remain pending.

## 12. Verify the offline pipeline

`configs/verification.yaml` selects all tests on CPU with two Torch threads. Fixtures use small local models and datasets; no full QM9 download or production checkpoint is required. Tests cover probabilities, rewards, nested costs, checkpoint/resume, calibration, guided sampling, evaluation and independence from `resource/`. Wheel verification requires the updated development dependencies from section 1.

Run the CPU suite locally:

```bash
python verify.py --config configs/verification.yaml
```

Run the equivalent suite on Modal:

```bash
bash modal/run_verify.sh --config configs/verification.yaml
```

Select test files or node IDs with the same comma-separated argument locally or on Modal:

```bash
python verify.py --selection tests/test_budget.py,tests/test_pipeline.py
bash modal/run_verify.sh --selection tests/test_budget.py,tests/test_pipeline.py
```

Request only the optional CUDA smoke explicitly; Modal then uses RTX-PRO-6000:1:

```bash
python verify.py --gpu-smoke --selection tests/test_pipeline.py::test_explicit_cuda_smoke
bash modal/run_verify.sh --gpu-smoke --selection tests/test_pipeline.py::test_explicit_cuda_smoke
```

Each attempt saves `protocol.json`, `tests.txt`, `summary.json` and a manifest under `workspace/results/raw/qm9-verify/attempts/<attempt-id>/`. Failures retain logs and return pytest's nonzero exit code. Modal commits the logs, then checks their hashes from a separate CPU function after reload. Progress bars belong to Python processing, not Modal wrappers. Selection and skipped/deselected tests remain visible in the summary; a subset run does not establish the full verification gate.

Step 12 is implemented but **not run**. No test result, Modal readback or passed gate is claimed; Step 13 is not implemented.

## 13. Run the controlled benchmark

`configs/benchmark.yaml` compares standalone CARS-QM9 with the upstream-label VGB control, then compares VGB with native-conditional neural/MC/CARS verifiers. BASE/BoN are native controls; BASE-rho/RS/ARS remain diagnostics. Each seed also checks all three native estimators on the same locked panel. The coordinator uses the existing generation/value runners.

Before a pilot, fill the six input manifests, pilot seeds/output count, fixed CARS/MC counts and finite generation/value budgets; set the native verification flags and supply a completed full Step 12 verification manifest for the current source. Review the matrix, inference/reference/evaluation settings, then set `locked: true`. Counts remain unset by default; neither 128 nor 3x1,000 is asserted as a paper setting. The total work cap must fund nested estimator queries as well as reserved completion.

Choose local or Modal execution for the pilot after setting `phase: pilot`:

```bash
python benchmark.py --config configs/benchmark.yaml --benchmark-id qm9-pilot
bash modal/run_benchmark.sh --config configs/benchmark.yaml --benchmark-id qm9-pilot
```

Resume unfinished units with the same config, source and benchmark ID (choose the matching backend and replace `PILOT_ATTEMPT`):

```bash
python benchmark.py --config configs/benchmark.yaml --benchmark-id qm9-pilot --resume-manifest workspace/results/benchmarks/qm9-pilot/attempts/PILOT_ATTEMPT/manifest.json
bash modal/run_benchmark.sh --config configs/benchmark.yaml --benchmark-id qm9-pilot --resume-manifest workspace/results/benchmarks/qm9-pilot/attempts/PILOT_ATTEMPT/manifest.json
```

Resume skips verified completed units and creates new attempts for unfinished units. It restarts a unit from its declared seed, retaining earlier attempts and known partial costs. Duplicate completed attempts are reported as ambiguous. Use one coordinator per benchmark ID. Every worker starts cold; online trie reuse within its attempt is preserved.

For the final run, set `phase: final`, declare final seeds/output count, set the reviewed CARS `profiled`/`ready_for_benchmark` flags and point `gates.pilot_manifest` to a completed ready pilot. Numerical settings, inputs, source and backend must match that pilot; changing them requires a new pilot. Readiness checks completion, available nonterminal guidance and common value pairs; it does not establish value accuracy.

Choose local or Modal execution for the final matrix:

```bash
python benchmark.py --config configs/benchmark.yaml --benchmark-id qm9-final
bash modal/run_benchmark.sh --config configs/benchmark.yaml --benchmark-id qm9-final
```

Coordinator artifacts live under `workspace/results/benchmarks/<benchmark-id>/attempts/<attempt-id>/`: `plan.json`, config snapshots, dispatch/return records, `attempts.jsonl`, `selected_manifests.json` and `summary.json`. Worker artifacts remain under `workspace/results/raw/` and `workspace/results/value_checks/`. Modal uses a CPU coordinator and one RTX-PRO-6000 GPU worker at a time, with hardcoded resources and the shared Volume.

Pass explicit final worker paths from `selected_manifests.json` to the Step 11 evaluator, using `--config` with the `evaluation.yaml` snapshot named in `plan.json`; keep pilot results separate. Retain `attempts.jsonl` for end-to-end cost claims because selected outputs exclude failed retry work. In-flight/unregistered work can remain unknown; recorded partial work is a lower bound. Value-job costs include independent reference MC work and are separate from generation costs. Incomplete units or an unready pilot return a nonzero exit code.

Step 13 source, tests and Modal wrappers are written but **not run**. The default config remains unlocked with missing counts/manifests; no pilot, benchmark result or passed gate is claimed. Step 14 remains unimplemented.

## 14. Use the shared entrypoint and reproduce a run

`run.py` selects one existing local runner and forwards its arguments using the active Miniconda interpreter. Task processing provides tqdm progress and its artifact/result summary; the dispatcher prints the exit code. Importing this dispatcher loads only the standard library. Help does not start a task, create artifacts or contact Modal.

List supported tasks or inspect a task's arguments:

```bash
python run.py --help
python run.py --task generate --help
```

Each selector uses the existing runner and Modal pair; task-specific commands and required manifests remain in sections 2–13:

| Task selector | Local runner | Modal Python file | Modal shell script |
| --- | --- | --- | --- |
| `prepare_assets` | `prepare_assets.py` | `modal/prepare_assets.py` | `modal/run_prepare_assets.sh` |
| `train_base` | `train_base.py` | `modal/train_base.py` | `modal/run_train_base.sh` |
| `calibrate` | `calibrate.py` | `modal/calibrate.py` | `modal/run_calibrate.sh` |
| `collect_rollouts` | `collect_rollouts.py` | `modal/collect_rollouts.py` | `modal/run_collect_rollouts.sh` |
| `train_verifier` | `train_verifier.py` | `modal/train_verifier.py` | `modal/run_train_verifier.sh` |
| `generate` | `generate.py` | `modal/generate.py` | `modal/run_generate.sh` |
| `prepare_value_panel` | `prepare_value_panel.py` | `modal/prepare_value_panel.py` | `modal/run_prepare_value_panel.sh` |
| `check_values` | `check_values.py` | `modal/check_values.py` | `modal/run_check_values.sh` |
| `verify` | `verify.py` | `modal/verify.py` | `modal/run_verify.sh` |
| `benchmark` | `benchmark.py` | `modal/benchmark.py` | `modal/run_benchmark.sh` |
| `evaluate` | `evaluate.py` | `modal/evaluate.py` | `modal/run_evaluate.sh` |

Choose local or Modal execution for the finite native-conditional toy check:

```bash
python run.py --task check_values --mode toy --value-law native-conditional
bash modal/run_check_values.sh --mode toy --value-law native-conditional
```

Choose local or Modal execution for full offline verification:

```bash
python run.py --task verify --config configs/verification.yaml
bash modal/run_verify.sh --config configs/verification.yaml
```

The backbone is the local UDLM-QM9 architecture trained with the MDLM objective in Step 3. Asset preparation downloads data/tokenizer only. Choose one completed checkpoint manifest and keep its hash, tokenizer and independent BASE calibration fixed across methods. The paper's 92.4M model description does not establish that a local checkpoint finished training; the documented 5,000/10,000-step checkpoints and interrupted training attempt in [PAPER_CHECK.md](PAPER_CHECK.md) do not establish 25,000-step completion.

For execution after review, follow setup/offline verification → assets → backbone → calibration → neural controls → baseline sources and locked panel → value checks and profiling → pilot → final benchmark → evaluation. Neural controls use separate `native` snapshot and `native-conditional` independent-mask labels. Individual CARS/MC runs need no neural training; the full benchmark includes both learned controls and therefore requires their checkpoints. Rho diagnostics retain their own target.

Artifacts use project-relative `workspace/` paths. Modal stores that tree on Volume `nam-cars-vgb-qm9-artifacts` at `/root/workspace`; local files and the remote Volume are separate stores. Select explicit completed manifests, retain failed attempts and preserve the benchmark's evaluation snapshot/retry ledger. Resume uses the task-specific manifest flag and a new attempt ID. Existing Modal-only options such as report downloads belong to the Modal wrapper.

The existing Modal calibration shell submits a detached job. Its exit code reports submission; inspect the remote job and wait for its completed calibration manifest before starting downstream tasks.

Review finding before the first benchmark: Step 13 currently requests calibration artifact `protocol.json` in `src/experiments/benchmark.py`, while Step 4 registers `sampling_protocol.json`. That existing reader/writer mismatch must be resolved before the benchmark can pass preflight. See the Step 14 checklist in [implementation.md](implementation.md).

Step 14 adds the dispatcher and reconciles documentation/source packaging. No code, help command, test, image build or Modal job was run for this step. Verification, native feasibility, profiling and benchmark gates remain pending; `configs/benchmark.yaml` remains unlocked. Earlier sections retain their step-specific implementation notes.

For project history, recorded execution evidence, open issues and instructions for the next agent, read [HANDOFF.md](HANDOFF.md).
