# Third-party references

Steps 1 and 2 contain newly written infrastructure, QM9 preparation, and reward handling. Steps 3–5 adapt the MIT-licensed MDM-VGB architecture, MDLM objective, training schedule, native QM9 sampling, rollout snapshots and neural verifier into the project's own package, as listed below. Runtime never imports, installs, mounts, or uploads `resource/`. Reference archives are development inputs excluded from the Python package. The archives do not provide a verified upstream Git revision; SHA-256 fingerprints identify the inspected files.

| Reference | Local source | License / use |
| --- | --- | --- |
| [Constrained Adaptive Rejection Sampling](https://arxiv.org/abs/2510.01902) | `resource/Constrained Adaptive Rejection Sampling.pdf`; `resource/cars-main/` | Algorithm reference only. No LICENSE/COPYING file was found in the supplied code archive. Code reuse is pending license verification. |
| [VGB for Masked Diffusion Model](https://arxiv.org/abs/2606.28301) | `resource/VGB for Masked Diffusion Model Efficient Test-time Scaling for Reward Satisfaction and Sample Editing.pdf`; `resource/MDM-VGB-main/` | Code archive: MIT, copyright (c) 2026 Kijung Jeon. Architecture, QM9 training and native sampling adaptations are listed below. The code license does not establish a license for the paper or datasets. |

Step 7's `src/cars/trie.py`, `oracle.py`, `proposal.py` and toy tests are independently written from the CARS paper's mathematical algorithm. The supplied trie, logits processor and grammar wrapper were inspected for comparison; no upstream CARS source or grammar was copied. This step adds no runtime dependency on that archive. Lark is a development dependency for the project's new finite toy grammar only; no molecular grammar is supplied or claimed complete.

Step 8's estimator, cache, verifier adapter, rejection sampler and tests are independent project implementations. The CARS paper Section 3 supplies the failure-cone/reweighting algorithm and RS/ARS/CARS update strategies. Fixed-m sample splitting, the V_rho estimator, a scaled conditional Hoeffding interval and exact-context persistence are project additions; they are not the paper's native-MDM V* estimator. No CARS source is copied, and no runtime dependency on resource is added. Step 8 code/tests have not been run.

## Inspected source fingerprints

| File under `resource/` | SHA-256 |
| --- | --- |
| `Constrained Adaptive Rejection Sampling.pdf` | `4d91751338b7b2a283017bee1f1e0ba32664da841e48d9af43fd3587ad18a3e2` |
| `VGB for Masked Diffusion Model Efficient Test-time Scaling for Reward Satisfaction and Sample Editing.pdf` | `0624bc09dedcb6ceb4e51b723fb196d57ac56d017fe344ad17e4249e035c4a3d` |
| `MDM-VGB-main/LICENSE` | `33d779be2daea20a7c0d9e245fe6b0f63ee0165c7c31b5cad0bd1dbade584c50` |
| `MDM-VGB-main/pyproject.toml` | `0ee8b8e5805f581133e312b1f107047c8135924c1b8bdb7a6e9b3c1bdb07f458` |
| `MDM-VGB-main/prepare_assets.py` | `3df736ab738d8e9feb0068572aa9f5eab50c036e2a19d4cf1018037599d85caa` |
| `MDM-VGB-main/src/tasks/qm9/task.py` | `486e9203f1caf95bf4cb573c5299e70bbe6765cd1c760eba3ef763c97eabf03c` |
| `MDM-VGB-main/src/tasks/qm9/train_base_model.py` | `f9158700ae8f4b9ba1c2307b756dc55256ac2bbde7cdcd2a1dc8b7140771beb7` |
| `MDM-VGB-main/src/tasks/qm9/mdlm_source/configuration_udlm.py` | `28d84e355b4431e4c6836252fa8a9676f7345d57f318af36287fddcec406b500` |
| `MDM-VGB-main/src/tasks/qm9/mdlm_source/modeling_udlm.py` | `edbf9a7bb2b32c2c0857dcecdd774d2d8a0b24a733a11396a1bde9fecb596fd7` |
| `MDM-VGB-main/src/tasks/qm9/mdlm_source/config.json` | `209c31fb9d6e386398a4115a35d312bd793ef9007e900256eaf1e0b9cd79fc99` |
| `MDM-VGB-main/configs/qm9/qm9_base_model_training.yaml` | `77f901591e8157db99d8b7bcc76ed7c27d9dc905226a6ff0503135654b6726d0` |
| `MDM-VGB-main/src/tasks/qm9/harness.py` | `994f9fdbc61120413ad16fa2b6383e306d96dfb325d189670ef41da5f42fd1f1` |
| `MDM-VGB-main/configs/qm9/qm9_inference.yaml` | `20cd4a1f86fc5ce2bc1885d449275279bfaccfe7cd9217c43e17e28179d9846a` |
| `MDM-VGB-main/src/tasks/qm9/verifier.py` | `74250889e48a4e22d84fe94b25689eb1898767e04bc3288c247f3f7ff1b765b4` |
| `MDM-VGB-main/src/tasks/qm9/train_verifier.py` | `247c2c60baa408b37f86bcf3f79373091f88f9e0346bf4bc54a322b16980972a` |
| `MDM-VGB-main/configs/qm9/qm9_rollout.yaml` | `9501089a3874c9fbf859eaa47d9d09d7a66ae2a9d0ebd82afc689e0d08a814c9` |
| `MDM-VGB-main/configs/qm9/qm9_verifier_training.yaml` | `abaee427cffa646c2040efb40fd25f611303252ab22c8558a9ed022d9d233be7` |
| `MDM-VGB-main/src/algorithms/vgb_candidates.py` | `a613782f667919ddfe4c6e9394ab06e3dda82fecf6836b9943ab808fc9f1a99b` |
| `MDM-VGB-main/src/algorithms/vgb_terminal.py` | `d6ab024b3f72e3dccb8ecfe69c2719f3bc18852a8cae0580f2825af8c14d996e` |
| `MDM-VGB-main/src/algorithms/vgb.py` | `5c3d215719c953e736ff096c9979ffbac73d5ca4bc02e4c02f84f27c7d5b60ed` |
| `MDM-VGB-main/src/algorithms/vgb_momentum.py` | `7fb5d044bf570287af59e2d11f3623f0f4b338a9000510e77e1f231e9a7d1c36` |
| `MDM-VGB-main/src/algorithms/vgr.py` | `aca1740ef51744e0a920ab052e6fb21b01701850308bae1cffb18dd6825ac109` |
| `MDM-VGB-main/src/algorithms/algorithm_utils.py` | `5f7cc7fedfb83a113aa541ff9271d1e098e3b0bb13896af9573b363ed4d849c2` |
| `cars-main/README.md` | `59bf4f4ca93c5c4918fdf22b2fae94bacc7e48c0fb958e1a8a7b9b6df2091b66` |
| `cars-main/cars/oracle_trie.py` | `612857d146bf6c52fb59327b1e68f9a320be73fe3c053fd18c896331f4047e2c` |
| `cars-main/cars/logits_process.py` | `c294bd4f1593b08cfa9fcf0393f1948fc02ee3f951a1e9a9b10f8e50b3f6cf90` |

## Dependency and reuse policy

Source adaptations retain copyright/SPDX notices in the destination files, with the full MIT text below:

| Original path under `MDM-VGB-main/` | Project destination | Changes |
| --- | --- | --- |
| `src/tasks/qm9/mdlm_source/configuration_udlm.py` | `src/models/configuration_udlm.py` | QM9 defaults, dimension validation, no external `auto_map`, explicit untied weights, reject unused CFG. |
| `src/tasks/qm9/mdlm_source/modeling_udlm.py` | `src/models/modeling_udlm.py` | Keep parameter names, rotary Q/K attention, AdaLN and zero initialization. Remove private JIT flags, unused helpers/CFG, and internal CUDA autocast; recompute short rotary tables and let the caller control precision. |
| `src/tasks/qm9/harness.py` | `src/models/backbone.py` | Adapt log-linear noise and SUBS; use float32 log-probabilities, local strict model loading and real-forward instrumentation. Native sampling is listed separately below. |
| `src/tasks/qm9/harness.py` | `src/sampling/native.py` | Adapt log-linear schedule, absorbing posterior, known-token copy, and unchanged-batch caching. Add explicit final denoise, budget handling and caller-selected pre-transition snapshots. Use float64 CPU probabilities/multinomial rather than the upstream GPU exponential race, and adjacent schedule endpoints rather than float32 `t-dt`. Cache only time-independent models. |
| `src/tasks/qm9/train_base_model.py` | `src/training/base.py` | Adapt batch sampling, MDLM loss and warmup/cosine schedule. Use saved split IDs, one GPU, and add optimizer/scheduler/RNG checkpoints, provenance, failure markers and resume. |
| `configs/qm9/qm9_base_model_training.yaml`; `src/tasks/qm9/mdlm_source/config.json` | `configs/base_training.yaml` | Plain QM9 settings; full vocabulary and length come from prepared assets. No general task/distributed framework. |
| `src/tasks/qm9/harness.py`; `src/tasks/qm9/task.py`; `configs/qm9/qm9_rollout.yaml` | `src/qm9/rollouts.py`; `configs/rollouts.yaml` | Retain interior snapshot selection, shared terminal labels, native 128 steps/batch 256/five snapshots and the all-train-example unconditional draw count. Add rho trajectories, binary labels tied to calibration, grouped validation, shard/RNG IDs and safe immutable artifacts. |
| `src/tasks/qm9/verifier.py` | `src/verifiers/neural.py` | Retain token Transformer architecture and parameter names. Sigmoid only; actual vocabulary/length; safe local weight loading, target/backbone/calibration checks, terminal anchoring and accounting. Remove generic activation/legacy pickle/distributed paths. |
| `src/tasks/qm9/train_verifier.py`; `configs/qm9/qm9_verifier_training.yaml` | `src/training/verifier.py`; `configs/verifier_training.yaml` | Retain MSE, AdamW, clipping, epoch shuffling and source YAML defaults. Add strict rollout-group validation, epoch checkpoints with optimizer/RNG, immutable attempts and resume. Single GPU, final epoch, float32. |
| `src/algorithms/vgb_candidates.py` | `src/sampling/candidates.py` | Singleton shortlists, replacement draws, multiplicities, position scaling and common depth rescaling. Use paper terminal-gate exception, no epsilon floors; group reservation and full-state value freeze. |
| `src/algorithms/vgb_terminal.py`; `src/algorithms/vgb.py` | `src/sampling/terminal.py`; `src/sampling/vgb.py` | Shared finite walk/leaf remask concepts. Pass@95 acceptance, no synthetic remask NFE, reserved unguided BASE completion, fixed original context, cooperative deadlines, trace/RNG records. No runtime dependency on upstream code. |
| `src/algorithms/vgr.py`; `src/algorithms/vgb_momentum.py`; `src/algorithms/vgb_terminal.py` | `src/sampling/vgr.py`; `src/sampling/momentum.py` | Shared candidate/terminal implementation; VGR empirical reference fallback; Momentum Definition 4.3 stays by default and explicit upstream jump option. No distributed/task-general framework. |
| `configs/qm9/qm9_inference.yaml` | `configs/mdm_vgb.yaml` | Retain QM9 singleton L_f/L_b/K/lambda/chi and BoN N defaults. Distinguish paper steps, project total-work cap, Pass@95 stopping, BASE completion and momentum-stay option. |

The fingerprint table identifies these originals. `src/sampling/completion.py`, `reference.py`, `base.py`, the toy fixture, and `src/qm9/calibration.py` are project implementations. No CARS source was copied. q_rho/pi_rho and calibration size/seed/bootstrap settings are project choices; the BASE calibration definition comes from MDM-VGB Appendix F.

API references are [Transformers models](https://huggingface.co/docs/transformers/main_classes/model), [configurations](https://huggingface.co/docs/transformers/main_classes/configuration), [NumPy quantiles](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html), [Torch categorical draws](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html), [TransformerEncoder](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html), [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) and [Python NormalDist](https://docs.python.org/3.11/library/statistics.html#statistics.NormalDist). Steps 3–5 have not been executed; adaptations are not a claim of bitwise reproduction or verified compatibility.

`src/verifiers/interface.py` and `monte_carlo.py` are project implementations, with no CARS source reuse. MC draw/batch/confidence defaults, Wilson uncertainty, rollout group split and resume policy are declared project choices. The [Wilson interval formula](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm) is computed directly; no third-party interval implementation is copied.

Step 6's `src/sampling/bon.py` and `tests/test_vgb.py` are project implementations. BoN follows MDM-VGB Algorithm 1; optional QED ranking and first-index ties are explicit. Adapted sampling files retain the upstream MIT copyright/SPDX notice; the source hashes and destinations are listed above. Small `required_work` methods were added to existing verifier adapters for whole-group reservation. Step 6 code/tests have not been run, and no CARS code is included.

Step 2 follows the MDM-VGB dataset/tokenizer IDs, seeded 5% split, sequence length, and QED definition. It adds runtime revision pinning, immutable manifests, canonical train identities, and strict token validation instead of upstream's string-based special-token deletion. Pass@95 requires independent BASE calibration as defined in the paper; the upstream numeric threshold is not reused. CPU/memory/time limits are unprofiled project choices, not paper settings.

`src/qm9/tokenizer.py` implements the regex splitting, concatenation, and CLS/SEP framing described by the [pinned QM9 tokenizer source](https://huggingface.co/yairschiff/qm9-tokenizer/blob/7e50af1babe9455a42aa64cb4ebbe43851990ea3/tokenizer.py), using the standard Hugging Face Tokenizers backend. It uses the contiguous IDs 0..34 in the [published vocabulary](https://huggingface.co/yairschiff/qm9-tokenizer/blob/7e50af1babe9455a42aa64cb4ebbe43851990ea3/vocab.json). **Difference from the old Python class:** that class adds five to a vocabulary which already includes special tokens; this project fixes that extra offset. Checkpoints using the old shifted IDs are not compatible. The custom Hub Python class is not imported. Preparation preserves the source tokenizer configuration, saves a portable `tokenizer.json`, and checks encoded IDs against the regex plus published vocabulary for every audited QM9 row. BOS/EOS aliases identify the existing CLS/SEP tokens without adding vocabulary entries.

The source comparison and its limitations are recorded in [PAPER_CHECK.md](PAPER_CHECK.md). Each task reads a plain YAML file; Modal resources are fixed directly in each task file.

The supplied archive does not pin Hub revisions. `configs/qm9.yaml` uses the explicit selector `main`; preparation resolves each selector once and records the actual commit, reported asset license, vocabulary, software versions, and content hashes. The requested Modal preparation completed on 2026-10-12 and saved these records with its completed assets manifest; see README section 2. Base training subsequently committed intermediate 5,000/10,000-step checkpoints before a client connection error; final completion is unverified. Sampling remains unrun.

API references for the independent implementation: [Modal images](https://modal.com/docs/sdk/py/latest/Image), [Modal Volumes](https://modal.com/docs/guide/volumes), [Hugging Face Hub client](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api), [dataset splits](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset.train_test_split), and [RDKit SMILES](https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html).

Python 3.11 follows the CARS README's tested interpreter. The QM9 dependencies follow the MDM-VGB package and QM9 code; unrelated DNA/protein dependencies are omitted. Modal, plotting, and development tools are separate extras. Upstream does not supply a tested dependency lock; the declarations here are not a claim that a particular resolved environment has been verified. Record resolved versions with each environment/run before experiments.

For any later copied file, add its verified revision or hash, original path, destination, retained copyright/license, and changes here before use. For CARS, write independent code until permission to copy is established. Checkpoint, tokenizer, and dataset terms must be recorded when those assets are selected; they are not covered by this source notice.

Step 9's native trajectory reference/trie, N/D and importance estimators,
matched-target adapters and independent-mask control are new project code,
with no CARS source copied. They call the existing MIT-adapted native posterior;
the upstream sampler and outer kernel are unchanged. Existing rollout/training
adaptations retain their MIT notices. The trajectory integration, sample
splitting, Hoeffding ratio bounds and posthoc mask protocol are not supplied
experiments from either paper. Step 9 code/tests have not been executed.

## MDM-VGB code license

MIT License

Copyright (c) 2026 Kijung Jeon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
