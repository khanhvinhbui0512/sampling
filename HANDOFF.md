# Agent handoff: CARS and MDM-VGB on QM9

Last updated: 2026-10-12. Project owner: Nam.

## 1. Read this first

This repository contains source written through **Step 14** of
`implementation.md`. It does not contain a verified end-to-end experiment or a
demonstration that CARS improves MDM-VGB. Implementation and experimental
validation are separate milestones.

The latest request is to save this handoff and push the existing work to GitHub.
The preceding Step 15 request was interrupted before implementation;
`implementation.md` currently defines Steps 1–14 only. Do not invent Step 15
or start production jobs from that interrupted request.

Read these files next:

1. [implementation.md](implementation.md): file order, complete source tree,
   artifact paths, gates and outstanding review items.
2. [PLAN.md](PLAN.md): scientific targets and research design; section 12
   reconciles the plan with the current implementation.
3. [PAPER_CHECK.md](PAPER_CHECK.md): matches and differences from the papers,
   known execution evidence and limits of any reproduction claim.
4. [README.md](README.md): English Miniconda setup and numbered task commands;
   sections 13–14 cover benchmark orchestration and the shared local runner.
5. [modal/README.md](modal/README.md): task pairs, Volume, resources and workflow.
6. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): source provenance, licenses,
   fingerprints and independently implemented CARS components.

Earlier README sections retain the notes written when each step was implemented.
Some consequently say a later step was not implemented *at that time*. Use this
handoff and the latest implementation notes for the combined picture, and actual
completed manifests for runtime evidence.

## 2. Research objective and targets

There are two required deliverables:

- Compare **standalone CARS adapted to QM9** with **MDM-VGB on QM9**, using a
  common backbone, tokenizer, reward/calibration and declared compute protocol.
- Replace MDM-VGB's value-verifier slot with a **CARS value estimator**, comparing
  against matched MC and neural controls while keeping the outer VGB kernel fixed.

QM9 is the dataset. Molecules are represented by SMILES token sequences, not by a
separate "CARS dataset". Original CARS molecular experiments use other models,
chemical classes and stopping rules; this project is an adaptation.

The following targets must remain distinct:

| Name | Meaning |
| --- | --- |
| `V_rho(z)` | Expected terminal reward under fixed left-to-right completion `q_rho(. | z)`, preserving every observed token. |
| Conditional value of `pi_rho` | Conditional value under the joint defined by left-to-right generation from the all-mask root; arbitrary masks can require importance correction. |
| Native-conditional `V*` | `P_native(Y matches z, reward=1) / P_native(Y matches z)`, under the declared native **root-trajectory** terminal joint. A fresh native rollout from `z` does not automatically have this conditional law. |
| Upstream neural label target | Native 128-step trajectory snapshots labelled by terminal reward; kept separate from the normalized native32 conditional target. |

The native bridge estimates the numerator with frozen, pruned trajectory
proposals and the denominator with independent unpruned native root draws.
Insufficient denominator evidence produces an unavailable estimate, not a fake
zero. The matched neural control uses masks chosen independently of root-sampled
terminal tokens/reward. Native feasibility on QM9 remains unverified.

Standalone CARS adapts its trie while generating. Value queries split discovery
from independent fixed-count estimation after freezing the trie. Completed values
are pinned within a run; bounded resident tries and optional JSON persistence are
implemented. Disk persistence is disabled by default. Benchmark workers have
cold run/attempt scopes, with online reuse inside an attempt and no cross-seed
or retry cache reuse.

## 3. Implementation history

The user requested one implementation step at a time, then review, normally with
an explicit instruction **not to run code**. Only separately authorized executions
listed in section 4 have recorded evidence.

| Step | Source written / purpose |
| --- | --- |
| 1 | Miniconda/package setup, relative-path/config helpers, state/value/cost contracts, RNG and immutable I/O. |
| 2 | QM9 dataset/tokenizer preparation, split audit, chemistry/reward and local/Modal preparation task. |
| 3 | Local UDLM architecture and MDLM backbone training/checkpoint/resume. |
| 4 | Native/rho BASE sampling, reference definitions and independent Pass@95 calibration. |
| 5 | MC/interface, rollout collection and neural verifier training/resume. |
| 6 | BoN, VGR, VGB and Momentum libraries with nested accounting and reserved completion. |
| 7 | Independent CARS trie, conservative oracle, reweighted proposal and toy tests. |
| 8 | Frozen-trie rho value estimation, exact-context cache and RS/ARS/CARS generation. |
| 9 | Native trajectory reference/CARS N/D bridge, matched MC and independent-mask neural labels. |
| 10 | Run factory, single-config generation, locked value panels and toy/QM9 value checks. |
| 11 | CPU artifact evaluator, uncertainty, audit and figures; selected Modal report downloads. |
| 12 | Offline verification runner, real pytest logs/status and separate Modal Volume readback. |
| 13 | Locked pilot/final matrix, CPU coordinator/GPU workers, immutable attempts, unit-level resume and retry ledger. |
| 14 | Shared local `run.py`, documentation/source-tree reconciliation and verify/benchmark image allowlists. |

The full verification suite and production inference/benchmark tasks have not
been run. Source existence and static review do not establish a passed gate.

## 4. Recorded execution evidence

These facts come from earlier work documented in README/PAPER_CHECK. The remote
Volume and job state were **not rechecked** while preparing this handoff.

- QM9 preparation completed on Modal on 2026-10-12: 127,190 train rows, 6,695 test
  rows, vocabulary size 35 and all 133,885 rows audited. Assets manifest:
  `workspace/data/qm9/qm9-v1/attempts/3d6c18b413394019953f839048b57df2/manifest.json`.
  [Recorded preparation job](https://modal.com/apps/reinforcement/main/ap-0EZnIR6PoV3xkA88Djqc7Q).
- Four targeted local path regression checks were recorded as passing. This was
  not a full suite run and does not verify the subsequently written pipeline.
- Backbone training attempt `8c39744ea4b74ddfaefa952f9c700d25` committed
  5,000/10,000-step checkpoints before a client `StreamTerminatedError`.
  [Recorded training job](https://modal.com/apps/reinforcement/main/ap-nnWdjDTKXpzWMKp28La2Ts).
  Completion of 25,000 steps and the job's later state are unknown. Check existing
  remote artifacts before authorizing a replacement training run.
- No completed calibration/threshold, verifier training, QM9 value validation,
  pilot, final benchmark or evaluation report is established by this handoff.
- Steps 13–14 were written without running Python, task help, tests or Modal.
  Static file/whitespace checks are not runtime verification.

The project downloads QM9 data and tokenizer assets, not pretrained backbone
weights. The backbone is trained from scratch with the MDLM objective using the
local UDLM-QM9 architecture, or resumed from an explicitly selected compatible
checkpoint. The paper describes a 92.4M backbone; runtime checkpoint manifests
record the actual parameter count and identities.

## 5. Important settings and paper differences

- MDM-VGB paper/upstream defaults: length 32, `L_f=8`, `L_b=32`, `K=8`, `N=16`,
  `lambda=0`, `chi=1`; native inference has 32 steps, upstream verifier rollout
  collection has 128. These do not imply all other defaults are paper settings.
- A common independent BASE 95th-percentile QED threshold defines Pass@95.
  Calibration size 10,000, seed 2026, linear quantile and bootstrap settings are
  project choices. The upstream numeric threshold is not reused.
- Tokenizer IDs use the published contiguous vocabulary 0..34. The old custom
  tokenizer's duplicate special-token offset is corrected; shifted-ID checkpoints
  are incompatible. Strict terminal decoding, counted final denoise, Pass@95
  terminal acceptance and reserved completion can change upstream behavior.
- The VGB `n*N` walk cap and the hard **total sequence-work** cap are different.
  The default 528 sequence cap funds native BoN but must be reviewed/profiled for
  nested MC/CARS guidance. Never silently reduce fixed draws to fit a budget.
- CARS value estimators, sample splitting, native N/D bridge, uncertainty and
  exact-context caching are project additions, not an estimator provided by the
  original CARS paper. Do not inherit exact sampling/mixing guarantees by name.
- Pilot/final seeds and output counts remain unset. Earlier suggestions of a
  128-output pilot or 3 seeds x 1,000 outputs were not verified paper settings.
  The evaluator's 1,000-molecule diversity cap is a separate project cost cap.
- Morgan radius 2 / 2,048 bits follows the CARS paper; released CARS code uses
  radius 3. Every requested generation slot stays in the relevant denominators,
  including failed/invalid/unfinished slots. Estimator draws are never deduplicated
  or treated as generated molecules.

See PAPER_CHECK.md for the detailed comparison rather than claiming a literal
reproduction of either paper's tables.

## 6. Code and execution map

- `run.py`: standard-library local dispatcher for 11 existing runners; forwards
  task arguments, tqdm/stdout/stderr and exit status, with no cloud import/job.
- Root task runners: `prepare_assets.py`, `train_base.py`, `calibrate.py`,
  `collect_rollouts.py`, `train_verifier.py`, `generate.py`,
  `prepare_value_panel.py`, `check_values.py`, `verify.py`, `benchmark.py`,
  `evaluate.py`. Each has `modal/<task>.py` and `modal/run_<task>.sh`.
- `src/`: `qm9`, `models`, `sampling`, `cars`, `verifiers`, `training`,
  `experiments`, `evaluation`, plus shared contracts/config/RNG/I/O/budget.
- `configs/`: plain one-QM9 YAML dictionaries. Counts still `null` are deliberately
  unprofiled, not safe fallback defaults.
- `tests/`: offline fixtures, probability/accounting/cache/training/runner/evaluator
  regressions; full suite is pending execution.

Artifacts use relative `workspace/` paths. Modal Volume
`nam-cars-vgb-qm9-artifacts` mounts that tree at `/root/workspace`. Local workspace
and remote Volume are separate stores; a matching path does not transfer data.
Completed manifests and registered hashes establish usable artifacts. Each new
attempt is immutable; interrupted/failed attempts remain available for audit.

`modal/run_calibrate.sh` submits with `-d`. Its shell exit status describes job
submission; inspect the remote job and require its completed calibration manifest
before downstream work. Other task shells use attached CLI invocation.

Benchmark orchestration writes under `workspace/results/benchmarks/`; generation,
value checks and evaluation use `results/raw`, `results/value_checks` and
`results/summary` inside workspace. Select explicit final worker manifests and
the evaluation config snapshot from the plan. Keep `attempts.jsonl` for retry
costs; unregistered/in-flight work can remain unknown and partial costs are lower
bounds. Value-check costs include independent reference MC work.

## 7. Known blockers and next work

1. **Calibration artifact mismatch, not fixed:**
   `src/experiments/benchmark.py` requests calibration `protocol.json`, but
   `src/qm9/calibration.py` registers `sampling_protocol.json`. The mismatch was
   discovered during Step 14 static reconciliation and left unchanged because
   that request covered Step 14 only. Align the reader/writer and add a meaningful
   regression check when a fix is authorized.
2. Full verification for current source, native enumeration/feasibility and
   evaluator validation remain pending. Verify/benchmark images now include root
   `run.py`; source hashes must match the evidence supplied to benchmark gates.
3. Production backbone completion, calibration, both neural control checkpoints
   and a locked native panel still need confirmed manifests.
4. Native denominator reliability/ESS and total nested work must be profiled on
   QM9. Unavailable estimates and failed requested output slots must remain in
   reports; favorable conditional errors alone do not establish success.
5. `configs/benchmark.yaml` remains `locked: false`, with seed/count/input/gate
   manifests unset. CARS profiling/readiness/native gates remain false. Do not
   flip them merely because files exist. Final requires a ready pilot with the
   same scientific contract and current full verification.
6. Dependency versions have no verified lock. QM9 importance benchmarking,
   order sensitivity, distillation and editing are not completed experiments.

The first next action is the user's review/next instruction, not an automatic
training, verification or benchmark run.

## 8. Persistent user preferences

- Vietnamese conversation; English main README and short numbered workflow
  sections. Update only relevant sections; commands in code blocks with a
  one-line description. No "Current status" section.
- Implement the requested step only and stop for review. Do not execute code,
  help, tests or Modal unless the user authorizes execution. When asked to run one
  exact command, run only that task and address its error.
- Read the supplied papers/upstream code before scientific decisions. State
  whether a setting/method matches a paper, upstream code or is a project change.
- Tell the user about missing/unreadable required inputs first; ask at most two
  clarification questions. Do not guess missing manifests, counts or evidence.
- Use Miniconda, project-relative paths and script entrypoints; no module-form
  Python commands or pathlib. Runtime never imports/installs/mounts/uploads
  `resource/`; archives remain reference-only and are ignored by Git.
- Keep code simple for one QM9 dataset. `src` is the package; avoid another
  `src/cars_vgb` layer or an extensible configuration framework.
- Modal files follow the provided App/Image/Volume template with the hardcoded
  CUDA tag `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`, Python 3.11 and explicit
  resources. No `modal/__init__.py`; no wrapper imports another Modal entrypoint.
- tqdm belongs to processing code, not an extra bar around Modal remote calls.
  Task scripts run from the repo root and print a short result.
- Use a descriptive `nam_...` Git branch. Push source/docs/configs/tests; keep
  credentials, reference archives, datasets, weights and runtime artifacts out
  of commits.

## 9. Git snapshot

Remote: `https://github.com/khanhvinhbui0512/sampling`.
Implementation branch intended for this snapshot:
`nam_cars_vgb_qm9_implementation`, based on `nam_cars_vgb_qm9_plan`
at planning commit `7a3ce40c`. Use Git history for the resulting commit identity
and remote publication status rather than assuming push success from this file.

This publication task creates this handoff and packages the existing work; it
does not implement another research step or run the experiment pipeline.
