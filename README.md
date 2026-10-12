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
