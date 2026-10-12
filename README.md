# SMILES sampling comparison

This project uses one Molecular Generation (SMILES) benchmark, with acrylates, chain extenders, and isocyanates, to compare three methods: **MDM-VGB with a trained value model**, **CARS**, and **MDM-VGB with a CARS value estimate**. The [ordered implementation plan](implementation.md) gives the file creation order and mathematical details.

**Relation to the papers:** the benchmark and rejection trie follow the CARS paper; MDM-VGB is adapted from the QM9 setup in the MDM-VGB paper. The shared SMILES MDLM, fixed-order CARS completion adapter, combined grammar/RDKit/class reward, and CARS-value hybrid are new experiment settings. CARS values estimate success under the completion policy; their relationship to the ideal value is part of the evaluation.

## 1. Setup

Create a Miniconda environment from the repository root:

```bash
conda create -n smiles python=3.12 pip -y
conda activate smiles
python --version
```

Use a project temporary directory for pip, then install the project and its development tools:

```bash
mkdir -p .cache/pip-tmp
export TMPDIR="$PWD/.cache/pip-tmp"
pip install -e ".[dev]"
```

Install Modal and connect your account before running remote tasks:

```bash
pip install -e ".[modal]"
modal setup
```

## 2. Shared workflow foundation

Shared helpers define SMILES states, configuration, artifact integrity, resume, and cost tracking. The primary protocol targets 100 unique successful molecules per group/seed within 1,000 root attempts. `configs/common.yaml` separates primary, replication, and smoke settings; `tests/conftest.py` provides exact toy distributions. These are libraries; local and matching Modal entrypoints are implemented in later steps.

## 3. Verify the environment

Check the installed environment and resource isolation from the repository root:

```bash
python src/tasks/verify.py --phase environment
```

Modal images use `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` with Python 3.12 and install dependencies from `pyproject.toml` using pip. No dependency lock is required to launch a task.

Run the same verification on Modal from the repository root:

```bash
python modal/verify.py --phase environment
```

Both entrypoints show progress and logs through `tqdm`, then print a short result with the report path. Each run saves `verification.json` and `verification.txt`. Modal reports persist in `smiles-artifacts`; local files are separate.

[Modal settings](configs/modal.yaml) select worker resources. Other [verification phases](configs/verification.yaml) require tests and artifacts from later steps; missing inputs fail. Both local and remote scripts accept repeated `--set KEY=VALUE` overrides. Remote scripts use `app.run()` and show Modal output and worker logs.

## 4. Prepare and check the SMILES benchmark

Verify the original assets and prepare all 32 acrylate, 11 chain extender, and 11 isocyanate examples:

```bash
python src/tasks/prepare_benchmark.py --config configs/data.yaml
```

Run the same preparation on Modal:

```bash
python modal/prepare_benchmark.py --config configs/data.yaml
```

T01 keeps every original example and reports grammar, RDKit, and SMARTS mismatches separately. It writes `data/benchmark/examples.parquet` and `data/benchmark/manifest.json` under the selected artifact root, with `tqdm` progress and a short result. Remote artifacts persist in the shared Volume.

Check benchmark artifacts, molecule identity, tokenizer, grammar, and reward using the matching local or remote verification:

```bash
python src/tasks/verify.py --phase benchmark
python modal/verify.py --phase benchmark
```

Prompts, grammars, and SMARTS follow the pinned CARS source; the primary joint reward and canonical identity are new settings. [Data settings](configs/data.yaml) declare ChEMBL 36, molecule splits, and train-only augmentation for the next step.

## 5. Prepare the training corpus and tokenizer

After T01, run corpus preparation followed by tokenization from the repository root:

```bash
python src/tasks/prepare_corpus.py --config configs/data.yaml
python src/tasks/build_tokenizer.py --config configs/data.yaml
```

Run the same tasks sequentially on Modal, after preparing T01 in the same Volume:

```bash
python modal/prepare_corpus.py --config configs/data.yaml
python modal/build_tokenizer.py --config configs/data.yaml
```

T02 downloads the checksum-pinned ChEMBL 36 archive, removes benchmark molecule identities, and splits unique graphs 80/10/10 before train-only augmentation. Original SMARTS labels can overlap. It writes `data/corpus/{train,validation,test}.parquet` and `data/corpus/manifest.json`, with filter and group counts.

T03 builds vocabulary from all train rows, adds EOS and right PAD, and writes CPU tensor shards under `data/tokenized/<split>/`. Unknown and overlength rows are rejected and reported for every split in `data/tokenizer/lengths.json`; vocabulary and metadata are separate JSON files. Test rows never change vocabulary or the configured length.

Both tasks show `tqdm` progress, print a short result, and support `--run-id` with `--resume`. Unfinished import or split stages restart; completed stages are checked and reused.

Check the resulting corpus and tokenizer locally or on Modal:

```bash
python src/tasks/verify.py --phase corpus
python modal/verify.py --phase corpus
```

This ChEMBL corpus and shared MDLM tokenizer are new experiment settings. CARS uses pretrained Llama and prompt examples; the MDM-VGB molecular experiment uses QM9.

## 6. Fetch pretrained models and train the MDLM

After accepting the Llama license on Hugging Face, provide `HF_TOKEN` in your environment and fetch the pinned Llama and RetroStar artifacts:

```bash
python src/tasks/fetch_models.py --config configs/models.yaml
```

Create the Modal Secret `smiles-model-access` with `HF_TOKEN`, then fetch the same artifacts remotely:

```bash
python modal/fetch_models.py --config configs/models.yaml
```

T04 verifies exact revisions and checksums, then writes manifests under `checkpoints/pretrained/`. Select models with `--set 'models.download=[retrostar]'` or `[llama]`. Llama is for AR replication; RetroStar's classifier and templates are for the one-step retrosynthesis metric. The primary MDLM starts from scratch and does not need T04.

After T03, train the shared group-conditioned MDLM locally:

```bash
python src/tasks/train_mdlm.py --config configs/mdlm.yaml --seed 0 --run-id mdlm-seed0
```

Train the same model on Modal after T03 has completed in its Volume:

```bash
python modal/train_mdlm.py --config configs/mdlm.yaml --seed 0 --run-id mdlm-seed0
```

T05 trains on train rows, selects checkpoints by fixed-seed validation loss, and saves weights plus optimizer, scheduler, and RNG state. Test shards are not loaded. Checkpoints live under `checkpoints/mdlm/<suite>/seed_<seed>/<run-id>/`; `training.json` records the best and final manifests. Modal commits every resume checkpoint to the shared Volume. Both tasks show `tqdm` progress and print a short result.

Use a separate small model for a smoke run, locally or remotely:

```bash
python src/tasks/train_mdlm.py --smoke --run-id mdlm-smoke0
python modal/train_mdlm.py --smoke --run-id mdlm-smoke0
```

Resume an unfinished run with the same configuration, seed, suite, and artifact location:

```bash
python src/tasks/train_mdlm.py --seed 0 --run-id mdlm-seed0 --resume
python modal/train_mdlm.py --seed 0 --run-id mdlm-seed0 --resume
```

The 12-layer, 768-hidden, 12-head baseline, optimization settings, and learning PAD in fixed-width sequences follow MDM-VGB's QM9 configuration. ChEMBL vocabulary, length 256, group/time conditioning, and explicit EOS handling are adaptations. `src/completion.py` defines full-vocabulary left-to-right q_z and 128-step absorbing reverse completion with a final denoising step; revealed tokens are preserved. q_z estimates a completion-policy value, with no claim that it equals V*. Model and completion libraries will be used by later workflow tasks.

## 7. Build rollouts and train the verifier

After T05, generate labeled snapshots locally using its best checkpoint:

```bash
python src/tasks/build_rollouts.py --mdlm-report outputs/primary/T05/all/seed_0/mdlm-seed0/training.json --run-id rollouts-seed0
```

Generate the same rollouts on Modal after T05 has completed in its Volume:

```bash
python modal/build_rollouts.py --mdlm-report outputs/primary/T05/all/seed_0/mdlm-seed0/training.json --run-id rollouts-seed0
```

T06 starts fully MASKed states for each group, uses 128 reverse steps and five interior snapshots, and labels each snapshot with its trajectory's terminal grammar/RDKit/SMARTS reward. The default count matches retained train rows per group. [Rollout settings](configs/rollouts.yaml) control counts and batches; generation applies no chemical filtering. Raw trajectories, snapshot shards, identity partitions, and a manifest are saved under `data/rollouts/<suite>/seed_<seed>/<run-id>/`.

Train the verifier on the published rollouts locally:

```bash
python src/tasks/train_verifier.py --rollouts data/rollouts/primary/seed_0/rollouts-seed0/manifest.json --run-id verifier-seed0
```

Train the same verifier on Modal using rollouts in its Volume:

```bash
python modal/train_verifier.py --rollouts data/rollouts/primary/seed_0/rollouts-seed0/manifest.json --run-id verifier-seed0
```

T07 uses a group-conditioned Transformer with sigmoid output and MSE. [Verifier settings](configs/verifier.yaml) start with four layers, hidden 256, four heads, dropout 0.1, LR 1e-4, global batch 2,048, and 200 epochs. It selects the lowest validation MSE and reports Brier score, MAE, ECE, and reliability bins overall, by group, and by MASK fraction. Calibration statistics describe snapshot predictions; no calibration transform is fitted.

All snapshots from one rollout and every occurrence of the same canonical molecule share one partition across groups. Invalid outputs use their exact token string. The seeded identity split reserves about 10% of distinct identities for validation, with at least one identity in each partition; test states are never used for selection. Checkpoints live under `checkpoints/verifier/<suite>/seed_<seed>/<run-id>/`; `training.json` records best/final manifests and their calibration reports.

Resume unfinished local or Modal runs by repeating their command with `--resume` and the same inputs, configuration, seed, suite, and run ID. T06 saves its sampling RNG and completed batches; T07 saves model, optimizer, epoch cursor, and RNG. Both print short results with `tqdm`; Modal commits each resume checkpoint. Use `--smoke` with a smoke MDLM report for T06 and the resulting smoke rollout manifest for T07. Local and remote artifacts remain separate.

The 128 steps, five snapshots, verifier dimensions, sigmoid/MSE, LR, and global batch follow the MDM-VGB QM9 reference. Class reward/conditioning, batch-32 rollout generation, identity partitions, validation selection, and calibration are adaptations. Native labels estimate the same trajectory's scheduled return; `--set rollouts.policy=left_to_right` creates a separately recorded q_z return dataset. These targets differ, and neither is asserted to equal ideal V*.

## 8. Generate SMILES with CARS

After T01 and T05, generate one group with the shared MDLM's best checkpoint, locally or on Modal:

```bash
python src/tasks/generate_cars.py --group acrylates --mdlm-report outputs/primary/T05/all/seed_0/mdlm-seed0/training.json --run-id cars-acrylates0
python modal/generate_cars.py --group acrylates --mdlm-report outputs/primary/T05/all/seed_0/mdlm-seed0/training.json --run-id cars-acrylates0
```

Repeat with `chain_extenders` or `isocyanates` and `--seed 0`, `1`, or `2`, using a distinct run ID. [CARS settings](configs/cars.yaml) target 100 unique successes within 1,000 root attempts. Local and Modal use their own artifacts. Use `--checkpoint` instead of `--mdlm-report` to select an explicit manifest; `--smoke` requires a smoke MDLM.

After T01 and the T04 Llama download, run the separate grammar-only AR replication:

```bash
python src/tasks/generate_cars.py --config configs/replication.yaml --group acrylates --run-id cars-ar-acrylates0
python modal/generate_cars.py --config configs/replication.yaml --group acrylates --run-id cars-ar-acrylates0
```

[Replication settings](configs/replication.yaml) pin Llama, BF16, temperature 1, top-p 1, top-k 0, 256 new tokens, and the tokenizer's EOS stopping rule. Valid outputs use canonical molecule identity; grammar-only successes without an RDKit identity use exact raw SMILES. Prompt exclusion and uniqueness are selection rules outside the reward/trie.

After T01 and review, verify CARS mass and sampling checks locally or on Modal; these use toy models and their own small tokenizers:

```bash
python src/tasks/verify.py --phase cars
python modal/verify.py --phase cars
```

Both generation scripts show `tqdm` progress and print a short result. Runs save attempts, selected samples, probability traces, `generation.json`, a checked generation manifest, and `cars_state.pt` under `outputs/<suite>/<cars_or_cars_ar>/<group>/seed_<seed>/<run-id>/`. Repeat the original generation command with `--resume` and identical inputs to restore trie, RNG, selection, and journal offsets; Modal commits checkpoints to its Volume. Unattained targets retain their counts and stopping reason.

CARS uses the full-vocabulary left-to-right q_z policy for MDLM, preserving known coordinates. Each attempt freezes trie mass, then learns proven grammar failures and terminal zero-reward leaves. Root constraints are prepared before recording initial mass. At the configured node cap, new-node learning pauses while existing exclusions remain active. The MDLM adapter, joint reward, bounded trie, and probability/cost records are project adaptations; the policy value is V_q, without an ideal V* claim.

## 9. Generate baseline, MDM-VGB and hybrid samples

After T01 and T05, run the unguided reference baseline locally or on Modal:

```bash
python src/tasks/generate_base.py --group acrylates --mdlm-report outputs/primary/T05/all/seed_0/mdlm-seed0/training.json --run-id base-acrylates0
python modal/generate_base.py --group acrylates --mdlm-report outputs/primary/T05/all/seed_0/mdlm-seed0/training.json --run-id base-acrylates0
```

After T07, run MDM-VGB with its selected learned verifier:

```bash
python src/tasks/generate_mdm_vgb.py --group acrylates --mdlm-report outputs/primary/T05/all/seed_0/mdlm-seed0/training.json --verifier-report outputs/primary/T07/all/seed_0/verifier-seed0/training.json --run-id mdm-vgb-acrylates0
python modal/generate_mdm_vgb.py --group acrylates --mdlm-report outputs/primary/T05/all/seed_0/mdlm-seed0/training.json --verifier-report outputs/primary/T07/all/seed_0/verifier-seed0/training.json --run-id mdm-vgb-acrylates0
```

After T01 and T05, run the same search kernel with CARS values:

```bash
python src/tasks/generate_hybrid.py --group acrylates --mdlm-report outputs/primary/T05/all/seed_0/mdlm-seed0/training.json --run-id hybrid-acrylates0
python modal/generate_hybrid.py --group acrylates --mdlm-report outputs/primary/T05/all/seed_0/mdlm-seed0/training.json --run-id hybrid-acrylates0
```

[Base settings](configs/base.yaml) use 128 reverse steps. [MDM-VGB](configs/mdm_vgb.yaml) and [hybrid](configs/hybrid.yaml) share search settings: L_f=8, L_b=32, K=8 sampled candidates, B=1, lambda=0, N=16. N multiplies editable length to bound search steps; every actual forward, including inner CARS and fallback, counts toward `protocol.max_model_nfe`. Momentum is optional: set both `search.momentum=true` and `search.chi=1.0`.

Search stops at its nN limit by default, following the QM9 setting. Enable a paid forward completion with `--set search.fallback_forward=true`; its steps and NFE are recorded separately.

Hybrid uses fixed m=16 and averages pre-attempt mass times terminal reward, retaining failures. Terminal values are exact; the guidance floor applies to incomplete states. Learned values retain their rollout training target, while CARS targets V_q. The reference estimators report policy and uncertainty; no ideal-value claim is made.

Repeat for each group/seed with distinct run IDs. Use `--checkpoint` or `--verifier-checkpoint` for explicit manifests. `--smoke` requires smoke checkpoints. All scripts print short results with `tqdm`; outputs, attempts, candidate/value traces and resume state live under `outputs/<suite>/<method>/<group>/seed_<seed>/<run-id>/`. Repeat the same command with `--resume`; search resumes between steps, baseline resumes between root rollouts, and Modal commits checkpoints to its Volume.

After review, run the finite value and integration checks locally or on Modal; these need no trained checkpoints:

```bash
python src/tasks/verify.py --phase values
python modal/verify.py --phase values
python src/tasks/verify.py --phase integration
python modal/verify.py --phase integration
```

## 10. Evaluate learned and CARS values

After T03, T05 and T07, evaluate both providers on independent test states locally or on Modal:

```bash
python src/tasks/evaluate_value.py --mdlm-report outputs/primary/T05/all/seed_0/mdlm-seed0/training.json --verifier-report outputs/primary/T07/all/seed_0/verifier-seed0/training.json --run-id value-final0
python modal/evaluate_value.py --mdlm-report outputs/primary/T05/all/seed_0/mdlm-seed0/training.json --verifier-report outputs/primary/T07/all/seed_0/verifier-seed0/training.json --run-id value-final0
```

[Settings](configs/value_eval.yaml) select up to 16 source molecules per group and mask fractions 0/0.25/0.5/0.75/1. Verifier train/validation identities and exact states are excluded; duplicate states count once. Missing group/depth coverage is reported.

Tuning uses both `--set value_eval.purpose=tuning` and `--set value_eval.split=validation`. A fixed state-ID partition prevents overlap with final test states across seeds.

Both providers receive the same NFE/time caps. Fresh CARS queries use fixed m=16 with three repetitions. Actual MDLM calls, verifier calls and time are reported separately. Independent references use 512 completions in batches of 32 per policy, with uncertainty intervals. MAE/MSE and calibration are grouped by class and mask depth; wide intervals are flagged.

Rollout-trained verifier predictions against `q_z` are labeled cross-policy diagnostics. Set `value_eval.require_q_trained_verifier=true` to require a matching `q_z` training target. Reverse-reference results are reported separately.

States, checked shards and `reference_values.parquet` are under `data/value_eval/<suite>/<purpose>/seed_<seed>/<run-id>/`; metrics are under `outputs/<suite>/T13/all/seed_<seed>/<run-id>/metrics.json`.

Use `--checkpoint`/`--verifier-checkpoint` for explicit manifests and `--smoke` with smoke checkpoints. Repeat the same command with `--resume` to continue. Modal commits checkpoints; interrupted queries replay deterministically. Recorded interruption cost is retained; hard-kill cost is unavailable.

After review, run the value checks locally or on Modal:

```bash
python src/tasks/verify.py --phase values
python modal/verify.py --phase values
```

## 11. Evaluate molecules and publish reports

Update local dependencies for the pinned RetroStar adapter; Modal rebuilds its image from the same package settings:

```bash
pip install -e ".[dev,modal]"
```

After generation stops, evaluate one run locally or on Modal:

```bash
python src/tasks/evaluate.py --generation-run outputs/primary/cars/acrylates/seed_0/cars-acrylates0 --run-id metrics-cars0
python modal/evaluate.py --generation-run outputs/primary/cars/acrylates/seed_0/cars-acrylates0 --run-id metrics-cars0
```

[Evaluation settings](configs/evaluation.yaml) keep root attempts, all completed outputs, search candidates and selected samples separate. Validity/membership use all completed outputs; uniqueness/novelty use canonical molecules. Diversity uses Morgan radius 2/2048 bits, with radius 3 as sensitivity. Selected validity is conditional on selection.

If RetroStar is needed, fetch only its pinned T04 artifacts before evaluation:

```bash
python src/tasks/fetch_models.py --set 'models.download=[retrostar]' --run-id retrostar
python modal/fetch_models.py --set 'models.download=[retrostar]' --run-id retrostar
```

RetroStar counts valid outputs with a top-1 reaction probability > 0.4 over all completed outputs; it also reports the valid-only denominator. This is a one-step proxy. Missing models produce `unavailable`, never a zero score. QED/Lipinski are excluded.

T14 checks source configuration, checkpoint/tokenizer/reward provenance and journals. Metrics and a checked manifest are under `outputs/<suite>/T14/all/seed_<seed>/<run-id>/`; a metric copy is saved under the source run's `evaluations/<run-id>/`. Full candidate visits are exported to `full_candidates.jsonl`. Attempts and samples remain unchanged. Missing or invalid inputs produce explicit failed evaluation artifacts; stopped runs retain attained counts.

After evaluating each selected run, publish CSV, PNG and Markdown reports locally or on Modal:

```bash
python src/tasks/report.py --evaluation-manifest outputs/primary/T14/all/seed_0/metrics-cars0/metrics_manifest.json --run-id report-primary
python modal/report.py --evaluation-manifest outputs/primary/T14/all/seed_0/metrics-cars0/metrics_manifest.json --run-id report-primary
```

Repeat `--evaluation-manifest` for other runs. Add T13 inputs with `--value-manifest data/value_eval/primary/final/seed_0/value-final0/manifest.json`. Local and Modal read their own artifacts. Use `--smoke` or `--set suite=replication` for the matching suite.

[Report settings](configs/report.yaml) accept explicit `expected_generation_runs` to flag absent evaluations. Reports under `reports/<suite>/<run-id>/` separate molecular quality, value error/calibration and cost; retain timeouts; and pair seeds only with matching backbone, reward, protocol and caps. Seed bootstrap intervals are unavailable with one seed. Frontiers use observed costs; inner completions count once. Verifier calls remain separate, and adjusted NFE requires measured FLOPs. V_q/rollout diagnostics do not establish ideal V*.

After review, run the metrics checks locally or on Modal without downloading reaction models:

```bash
python src/tasks/verify.py --phase metrics
python modal/verify.py --phase metrics
```

Both tasks show `tqdm` progress and print short results. Repeat an interrupted command with `--resume` and identical inputs; source checksums must match the saved snapshot. Modal commits report artifacts to its Volume and prints their paths.

## 12. Run and lock the comparison matrix

Complete T01–T07 using sections 4–7 first. T08–T11 generate samples, T12 verifies artifacts, T13 evaluates values, T14 evaluates molecules, T15 coordinates comparisons, and T16 publishes reports. T15 calls existing generation/evaluation/report tasks; it checks dependencies before dispatch and never trains or downloads models.

Run the validation sweep with the same T05 and T07 checkpoints locally or on Modal:

```bash
python src/tasks/compare.py --purpose tuning --mdlm-report outputs/primary/T05/all/seed_0/mdlm-seed0/training.json --verifier-report outputs/primary/T07/all/seed_0/verifier-seed0/training.json --run-id compare-validation
python modal/compare.py --purpose tuning --mdlm-report outputs/primary/T05/all/seed_0/mdlm-seed0/training.json --verifier-report outputs/primary/T07/all/seed_0/verifier-seed0/training.json --run-id compare-validation
```

[Comparison settings](configs/comparison.yaml) cover the three methods plus baseline, all groups, seeds 0/1/2, and separate NFE/time allowances. Validation sweeps N=[1,2,4,8,16] and m=[4,8,16,32]. Shared baseline/CARS jobs run once; each N/m cell gets a report. Inner hybrid calls count toward the same allowance.

Review the validation reports, then lock your chosen cell locally or on Modal; replace `N16_m16` with the reviewed variant ID:

```bash
python src/tasks/compare.py --lock-from outputs/primary/T15/all/seed_0/compare-validation/comparison_manifest.json --select-variant N16_m16 --run-id comparison-lock
python modal/compare.py --lock-from outputs/primary/T15/all/seed_0/compare-validation/comparison_manifest.json --select-variant N16_m16 --run-id comparison-lock
```

Run final trials from the checked selection locally or on Modal:

```bash
python src/tasks/compare.py --lock-manifest outputs/primary/T15/all/seed_0/comparison-lock/selection_manifest.json --run-id compare-final
python modal/compare.py --lock-manifest outputs/primary/T15/all/seed_0/comparison-lock/selection_manifest.json --run-id compare-final
```

The lock fixes source/configuration hashes, checkpoint/tokenizer/benchmark contents, RetroStar availability, shared search settings and budgets. It supplies the selected N/m and checkpoints; changed inputs require a new validation run. Validation uses logical seed + 2³¹; final uses the listed seed. These are independent generation trials on the same prompts, not held-out molecules. T14 comparison keys keep both purposes separate.

Choose seeds, budgets and shared kernel settings before validation using repeated overrides; repeat them for final if they differ from defaults:

```bash
python src/tasks/compare.py --purpose tuning --mdlm-report outputs/primary/T05/all/seed_0/mdlm-seed0/training.json --verifier-report outputs/primary/T07/all/seed_0/verifier-seed0/training.json --set 'seeds=[3,4,5]' --set 'comparison.parameters.search={N: 16, lambda: 0.0}' --run-id comparison-custom
python modal/compare.py --purpose tuning --mdlm-report outputs/primary/T05/all/seed_0/mdlm-seed0/training.json --verifier-report outputs/primary/T07/all/seed_0/verifier-seed0/training.json --set 'seeds=[3,4,5]' --set 'comparison.parameters.search={N: 16, lambda: 0.0}' --run-id comparison-custom
```

Use `--checkpoint`/`--verifier-checkpoint` with the actual `best_manifest` paths from T05/T07 reports for explicit checkpoints. Replace the whole `comparison.parameters.search` mapping to add shared kernel overrides. Optional `comparison.value_manifests` must point to completed T13 manifests with matching purpose, split and shared models. T15 does not launch T13 reference queries.

Run the separate smoke matrix with smoke checkpoints locally or on Modal:

```bash
python src/tasks/compare.py --smoke --mdlm-report outputs/smoke/T05/all/seed_0/mdlm-smoke0/training.json --verifier-report outputs/smoke/T07/all/seed_0/verifier-smoke0/training.json --run-id compare-smoke
python modal/compare.py --smoke --mdlm-report outputs/smoke/T05/all/seed_0/mdlm-smoke0/training.json --verifier-report outputs/smoke/T07/all/seed_0/verifier-smoke0/training.json --run-id compare-smoke
```

For AR replication, fetch pinned Llama through T04 in section 6, then start its separate validation matrix locally or on Modal:

```bash
python src/tasks/compare.py --set suite=replication --purpose tuning --run-id compare-ar-validation
python modal/compare.py --set suite=replication --purpose tuning --run-id compare-ar-validation
```

Lock variant `ar` and run final using the same lock/final commands with `--set suite=replication` and replication artifact paths. Model access is needed only for T04: accept the Llama license and configure the `smiles-model-access` Modal Secret containing `HF_TOKEN`, as in section 6. Comparison workers read existing artifacts offline.

Resume an interrupted matrix with identical arguments and its original run ID locally or on Modal:

```bash
python src/tasks/compare.py --lock-manifest outputs/primary/T15/all/seed_0/comparison-lock/selection_manifest.json --run-id compare-final --resume
python modal/compare.py --lock-manifest outputs/primary/T15/all/seed_0/comparison-lock/selection_manifest.json --run-id compare-final --resume
```

T15 saves `job_plan.json`, `job_results.json` and `comparison_manifest.json` under `outputs/<suite>/T15/all/seed_0/<run-id>/`. Each child has its own run directory; cell reports are under `reports/<suite>/<run-id>-<variant>-r-<digest>/`. Timeouts and budget stops keep actual counts, including fewer than 100 samples, and are not rerun to fill the target. Failed jobs remain in the audit; resume continues unfinished work. Durable interrupted model cost is retained; unrecorded hard-kill work is unavailable.

Local artifacts and Modal's `smiles-artifacts` Volume are separate. Modal uses a CPU coordinator, method-specific GPU workers, an evaluation worker and a CPU report worker; each commits before downstream reads. Copy the complete dependency/run tree when moving artifacts between environments, and use the same project-relative `--artifact-root` layout. All entrypoints log progress and short results through `tqdm`.

The matrix, fixed allowances, validation streams and explicit selection lock are project experiment design. Primary uses the shared MDLM and combined reward; replication retains Llama and grammar-only reward. Read sections 8–11 and [provenance](THIRD_PARTY_NOTICES.md) for paper differences and the limits of V_q and measured cost.

After review, run the comparison and existing kernel checks locally or on Modal:

```bash
python src/tasks/verify.py --phase integration
python modal/verify.py --phase integration
```
