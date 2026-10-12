# Paper comparison for implemented Steps 1–13

Reviewed by reading the supplied papers and repositories. On 2026-10-12, the
requested QM9 asset preparation completed on Modal: 127,190 train rows, 6,695
test rows, 35 vocabulary entries, and all 133,885 source rows audited. Four
targeted path regression checks also passed locally. See README section 2 for
the completed run and manifest. These checks do not reproduce model results.

The project has code for infrastructure, QM9 preparation, terminal reward,
the local MDLM architecture, base training/resume, native/rho BASE sampling,
root-joint likelihood, calibration, fixed-count MC-rho, rollout collection,
learned verifier training/resume, and BoN/VGR/VGB/Momentum baseline libraries.
Base training was also attempted on Modal on 2026-10-12. Attempt
`8c39744ea4b74ddfaefa952f9c700d25` committed checkpoints at steps 5,000 and
10,000 before the client lost its connection with `StreamTerminatedError`.
Completion of all 25,000 steps and the remote job's subsequent state are
unverified. See the [training run](https://modal.com/apps/reinforcement/main/ap-nnWdjDTKXpzWMKp28La2Ts).
Sampling/calibration/verifier tasks remain unrun. Step 6 and Step 7 tests have
not been executed. Steps 8–9 add adaptive rho sampling/value libraries and a
native conditional bridge with matched MC/neural controls; their tests and QM9
feasibility checks have not been executed.

Step 10 adds the run factory, generation/state-panel/value-check entrypoints
and local/Modal pairs. Its code and regression tests have not been run.

Step 11 adds the CPU artifact evaluator, metric/seed tables, value-reference
diagnostics, quality/cost/cache plots and selected Modal report downloads.
Its source and hand-checkable regression tests have **not been run**; the
evaluator gate remains pending. Step 10 receives provenance metadata only
(panel manifest path, MC settings, execution versions/device/precision/hardware).

| Component | Comparison with the original material |
| --- | --- |
| Evaluation denominator and QED | **Same MDM-VGB paper definitions:** fixed independent BASE Pass@95 threshold, valid-molecule mean QED, validity and unique Pass@95. **Project addition:** every requested slot remains in Pass@95/validity/unique Pass@95, including failed/unfinished outputs; zero-filled QED is separate from valid-only QED. Undefined conditional metrics are null. Failed/incomplete attempts are audited, not silently selected away or imputed as a valid comparison. |
| Molecular uniqueness and novelty | **Same released MDM-VGB code convention:** unique valid canonical identities / valid outputs; unique identities outside TRAIN / valid outputs. **Additional diagnostic:** novelty / unique valid identities. Prepared canonical TRAIN identities replace exemplars; this is a QM9 comparison, unlike CARS's industrial chemical classes. |
| Morgan diversity | **Same CARS paper Appendix H.4 settings:** radius 2, 2,048 bits, average pairwise Tanimoto distance. **Differs from released CARS code:** its `InternalDiversity.get_diversity` uses radius 3. **Project addition:** chirality explicitly false; seeded uniform subsampling caps 1,000 completed valid OUTPUTS with duplicates retained, and records settings/indices. The cap does not set benchmark output count. Neither deduplicated successful molecules nor estimator draws define this metric population. |
| Cost evaluation | **Same paper distinction:** raw NFE counts actual base calls; learned verifier adjusted NFE requires measured forward FLOPs. **Project additions:** sequence work/batch-size sets/phase costs, query and sampling wall vs setup, partial failed-attempt cost lower bounds, cold/warm current costs vs historical unique cache-origin costs. Origins can overlap cold ledgers and are never an additive charge. No parameter-count FLOP proxy or original CARS 100-unique-molecule stopping experiment is claimed. |
| Uncertainty and artifact comparison | **Project additions:** immutable explicit attempt selection, hash/target/model/calibration/config/execution grouping; equal-seed bootstrap with no seed CI for one seed; duplicate-seed suppression; rollout-origin cluster bootstrap on retained panels. QM9 MC references keep intervals and N/D/ESS diagnostics and are not exact V*. Error metrics condition explicitly on available pairs, with unavailable states and availability retained. Toy enumeration remains separate. These procedures are not claimed as either paper's original statistical protocol. |
| Dataset and tokenizer | **Same as upstream code:** `yairschiff/qm9` and `yairschiff/qm9-tokenizer`, from `MDM-VGB-main/prepare_assets.py`. The paper names QM9 but does not pin these Hub revisions. |
| Train/test counts | **Same specified counts as the paper:** 127,190 / 6,695, Appendix F.1. Verified by the completed Modal preparation on 2026-10-12. |
| Split procedure | **Same as upstream code:** a shuffled 5% held-out split with seed 42, using `Dataset.train_test_split`. Splitting original row IDs preserves the corresponding molecule membership and order. The paper gives counts, not this seed. Exact split identity still depends on the downloaded revision and library versions. |
| Sequence length and tokenization | **Same specified length:** 32, Appendix F.3, Table 7. Right padding/truncation and added special tokens follow upstream. Step 3 trains on the original `canonical_smiles` column, including all padded/special-token positions in the loss and using no attention mask, as upstream does. It uses saved train row IDs, not the canonical deduplicated novelty list. Step 2's audit uses `smiles`; training separately rejects MASK/UNK in the canonical column. |
| Vocabulary | **Same size as Table 7: 35**, including five special tokens, using the published `vocab.json` IDs 0..34. **Different from the old custom Python class:** it adds another five to IDs already containing special tokens, producing gaps; this project fixes that duplicate offset. The supplied verifier-training YAML's size 40 is not used. Existing checkpoints with shifted IDs are incompatible. |
| Tokenizer implementation | **Same upstream SMILES regex, concatenation and CLS/SEP framing; different backend and corrected IDs.** Preparation writes a standard fast `tokenizer.json` and preserves the source config. CLS/SEP also have BOS/EOS aliases. Every audited source row is checked against the regex plus the published vocabulary; custom Hub Python code is not imported. This check does not establish bitwise reproduction of the old custom class. |
| Continuous terminal score | **Same definition as the paper:** QED for valid molecules and 0 for invalid molecules, Appendix F.1. Heavy-atom count is a diagnostic; molecules are not rejected solely for exceeding nine heavy atoms. |
| SMILES parsing and terminal tokens | **Different from upstream code; not specified in this detail by the paper.** Upstream deletes MASK/UNK/PAD text and truncates at EOS. This implementation requires a complete state, rejects UNK and misplaced special tokens, and allows only PAD after EOS/PAD. It also disables RDKit molecule names and CXSMILES. These choices can change validity and Pass@95; results must identify this decoding policy. |
| Scoring errors | **Different from upstream code:** invalid SMILES receives score 0, but a QED calculation error after successful parsing raises an error here. Upstream catches scoring exceptions and returns 0. The paper does not specify this software-error handling. |
| Pass@95 threshold | **Same calibration definition as the paper:** 95th percentile of an independent BASE calibration set, Appendix F, with invalid scores retained as zero. Step 4 writes the generation/calibration code; no actual threshold exists until it is run. **Different from supplied code/config:** its fixed `0.5614854131280362` and optional training-dataset percentile are not reused. |
| Canonical train identities | **Project addition:** store canonical isomeric SMILES of valid train molecules for later novelty measurement. Upstream stores raw train SMILES before comparing with canonical generated SMILES. Raw rows are not filtered or deduplicated, so this addition does not alter the training split. |
| Revision pinning, manifests, hashes and attempts | **Project additions:** resolve Hub selectors to commit SHAs at execution time, record versions/hashes, and retain separate completed/failed attempts. These support reproducibility and do not define a new sampling method. The completed asset manifest records the resolved commits. |
| Modal and dependency setup | **Project choices:** requested CUDA image, Python 3.11 and persistent Volume. Preparation completed using 2 CPUs / 8,192 MiB / 3,600 seconds. Training uses one RTX PRO 6000 / 4 CPUs / 32,768 MiB / 24 hours, with checkpoint commits. Appendix F reports RTX PRO 6000 and H200, but does not prescribe these task resource limits. Training limits remain unprofiled and dependencies unlocked; the preparation environment is recorded in its manifest. |
| Base architecture | **Same design as upstream's QM9 MDLM:** 12 blocks, hidden 768, 12 attention heads, condition dimension 128, dropout 0.1, rotary Q/K, AdaLN and zero-initialized gates/head; `time_conditioning=False`. Parameter names are retained for strict loading. Paper F.1 reports 92.4M; the actual count is recorded when training runs and has not been runtime-checked. Architecture files now live in `src/models`; no external code loading. |
| Base training objective and schedule | **Same specified defaults as upstream code, not all stated in the paper:** seed 1; independent `random.sample` batches of 256 from train rows; 25,000 steps; log-linear masking with epsilon 0.001; clamp time to at least 0.001; weighted SUBS loss `mean(token_loss / t)`; AdamW lr 3e-4, weight decay 0, default betas/epsilon; 1,000 warmup steps, cosine to 3e-6; clip norm 1; save every 5,000. Single GPU only here; upstream also supports distributed training. |
| Precision and implementation | **Deliberate differences from upstream:** one caller-controlled bf16 autocast scope on CUDA, float32 LayerNorm and SUBS log-probabilities, float32 on CPU. Upstream nests a float32 CUDA autocast scope for blocks and bf16 for the final layer. Private JIT settings and unused CFG helpers are removed; rotary tables are recomputed. Mathematical architecture/objective are retained; numerical/performance equivalence is not established. |
| Training checkpoints and resume | **Project additions:** model/config/tokenizer plus optimizer, scheduler, step, Python/NumPy/Torch/CUDA RNG, actual vocabulary, and config/asset/software/source hashes. Each checkpoint gets a completion marker last; resume creates a new attempt and requires the same environment and schedule. Upstream saves model weights without full resume state. The Modal training attempt committed intermediate 5,000/10,000-step checkpoints; the final checkpoint and resume behavior are unverified. |
| Native BASE transition | **Same mathematical schedule/posterior as upstream QM9 harness:** log-linear corruption `m(t)=(1-0.001)t`; a masked site stays masked with probability `m(s)/m(t)` and otherwise follows the non-MASK model prediction; revealed tokens are copied. Default 32 steps and batch 256 follow `qm9_inference.yaml`, not the separate 128-step rollout setting. Explicit schedule sigma is passed to the adapter. |
| Native implementation and completion | **Declared differences from upstream:** float64 CPU probabilities/categorical draws instead of GPU exponential-race draws; use adjacent time-grid endpoints to avoid subtraction rounding; snapshots are caller-selected pre-transition steps. Reuse a forward cache only for an unchanged batch with `time_conditioning=False`; completed batches need no forwards. Default final denoise removes residual MASK at the finite endpoint `1e-5`, with at most one extra counted forward. Upstream stops at the endpoint without this final pass. Policy IDs distinguish these choices; bitwise reproduction is not claimed. |
| q_rho and pi_rho | **New project reference:** q_rho reveals leftmost masks using every known token as context. pi_rho is the joint obtained from the all-MASK root; its likelihood always replays root prefixes. They can give different conditional laws for states with future observations. Step 4 implements likelihood/log-weight primitives and writes exact toy enumeration tests; it does not implement MC/CARS/importance estimators or establish native-reference consistency. |
| Calibration size and uncertainty | **Project choices, not paper settings:** 10,000 draws, seed 2026 in a calibration-only RNG namespace, linear quantile, 1,000 percentile-bootstrap replicates, 95% diagnostic interval. No filtering, deduplication, invalid retry, or dataset percentile; unfinished draws fail rather than changing the denominator. Sampling and bootstrap use separate streams. Samples, settings, IDs, costs, RNG, uncertainty and hashes are saved. Bootstrap coverage is not guaranteed, particularly with ties. |
| Learned verifier architecture | **Same as supplied QM9 code:** token and learned position embeddings; hidden 256, four Transformer layers, four heads, feedforward 1,024, GELU, dropout 0.1, mean pooling over every position, LayerNorm/MLP head and sigmoid. No PAD attention mask. Actual prepared vocabulary/length are used. Paper reports 3.24M parameters; the actual count is recorded at runtime and has not been checked. |
| Verifier objective and optimization | **Same defaults as source YAML/trainer:** binary shaped Pass@95 labels, MSE, AdamW lr 1e-4, weight decay 0, batch 2,048, seed 1, 200 epochs, CPU randperm shuffling and gradient clip 1. Uses float32 as upstream does for this verifier. Single GPU here; no distributed trainer. The paper's regression target equals V* only under Appendix B.1's reference-consistency condition. |
| Rollout count and snapshots | **Same as supplied rollout code/config:** 128 native diffusion steps, batch 256, five interior pre-transition snapshots, each sharing its terminal label. Upstream uses all train examples, giving 127,190 unconditional draws; the count is from code and dataset size, not an independently specified paper sample budget. Step 4's final denoise and numeric differences still apply. **New:** explicit per-shard/group IDs, safe tensor artifacts, binary labels locked to the independent calibration. |
| Native versus rho training target | **Explicit distinction added by this project:** native labels estimate the conditional mean of terminal reward under sampled 128-step trajectories, marginalizing snapshot time. This is not a fresh 128-step completion kernel at every state, and is not silently reused as the 32-step native target. Rho trajectory labels use the fixed left-to-right policy and have the same target ID as MC-rho. Current rho training states are prefixes; arbitrary-mask generalization remains unverified. No native/reference consistency claim follows from either regression. |
| Validation and resume | **Project additions:** rollout seed 1, 5% validation grouped by complete rollout, split seed 42; disjoint shard partitions; reject missing shards, mixed targets, duplicate groups, missing splits and snapshot/terminal mismatches. Upstream train loader selects train rows and has no held-out validation; it can fall back to all rows when no train split exists. This trainer rejects that case. Immutable checkpoints every ten epochs save optimizer, full RNG, data/source/environment/target hashes and epoch resume. Final epoch is used without best-validation selection. |
| MC-rho | **New project baseline:** fixed 128 draws, draw batch 256, binary mean, plug-in standard error and 95% Wilson interval. These are not paper QM9 MC hyperparameters. Repeated draws retain multiplicity; the entire query's sequence work is reserved before sampling. A shortage returns value=None, without replacing it with zero or changing the draw count. Wilson coverage is approximate. |
| Terminal anchoring and query accounting | **Same anchoring principle as Appendix F:** both MC and neural adapters replace complete-leaf predictions with exact binary reward, without a learned forward. **Project additions:** ordered results, actual base/verifier batched forwards and sequence counts, separate oracle counts, strict checkpoint/target checks. A shared neural forward is charged once to the first result in its chunk. No FLOP ratio is inferred from parameter counts. |
| Baseline settings | **Same as upstream QM9 defaults:** singleton blocks, random forward positions, low-confidence backward positions, L_f=8, L_b=32, K=8 draws with replacement, lambda=0, chi=1, BoN N=16. Appendix F.3 Table 7 supports the shortlist sizes; N=16 is the supplied code default, not a claim about every paper experiment. Step limit 512 is n×N for n=32. The independent total-work cap of 528 base sequences is a project choice: 16×(32+1) reserves all native BoN draws including final denoise. It includes nested queries/completion and is not a paper NFE setting. |
| Shortlisted edge weights | **Same as Definition 4.2:** s[k,1]=1/C(n−1,k), forward current gate and backward parent gate, and the terminal exception omitting those gates on edges into leaves. **Same practical convention as upstream:** replace token probabilities by sampled count/K, multiply by available/selected position counts, and rescale both depth coefficients together. K is not top-k truncation. **Differs from upstream:** no 1e-8 value floor, and terminal-child forward weights omit the current gate even when lambda>0. Low-confidence selection uses raw model logits from the same forward; no second forward or SUBS-pinned known-token confidence is used. Shortlisting remains a heuristic, not the full-neighborhood theorem. |
| BoN and VGR | **Same as Algorithms 1–2:** BoN scores all N complete draws, default binary tau ranking; VGR uses forward value resampling and empirical reference fallback when weights vanish. **Project choices:** first-index ties, optional explicitly labelled QED ranking, whole-group work reservation and no partial-N result. Duplicate draws count separately. VGR returns its complete leaf even if it fails the binary reward, as a completed but unsuccessful output. |
| Terminal and exhaustion policies | **Same binary objective as the paper:** exact Pass@95 anchoring. **Differs from upstream QM9:** validity alone does not accept a leaf. Failed leaves use the upstream-style uniform singleton remask, with no NFE charged for that pure operation. Zero-mass VGB states use a labelled uniform remask or completion escape; this adds support only through that explicit policy, not epsilon weights. **Follows the paper's unguided completion principle:** the selected BASE policy finishes remaining masks at the step/work limit. **Differs from upstream forced completion:** no shortlisted verifier guidance during fallback. Early success returns one output, whereas Algorithms 3–4 describe collecting leaves over a fixed walk. |
| Momentum | **Same as Definition 4.3 by default:** forward/backward motion, switch mass opposite−chi·min(F,B), and stay mass chi·min(F,B), normalized together. **Differs from upstream directed_choices:** stays are retained by default. Explicit include_stay=false removes stay mass and renormalizes moves/switches, matching that upstream convention. Boundary direction resets, failed-leaf escapes and early stopping remain practical policies; no mixing/stationarity claim is made for these finite runs. |
| Guidance budgeting and records | **Project additions:** reserve worst-case unguided completion before guidance, then reserve every entire candidate-value group using each adapter's required_work. MC never silently lowers its draw count; unavailable values remain unavailable. Values are frozen by full masked state within one run; trace, value records, seed, RNG states and momentum direction are returned for later persistence. Position draws use Python RNG and categorical draws use float64 CPU Torch RNG; this is not upstream GPU/seed bitwise reproduction. Pure backtracks, stays and switches add no base forward unless a fresh candidate-logit computation actually occurs. Cooperative deadlines mark failure, including completion finishing after the deadline, but do not interrupt an in-flight GPU call. These records are not mid-walk resume support. |
| CARS trie and surviving mass | **Same mathematical mechanism as CARS Section 3, equations (1)-(3):** raw next-token probabilities, zero mass on certified failure cones, mass one for unknown subtrees, and recursive weighted surviving mass. Step 7 writes an independent log-space implementation with duplicate/overlap handling. **Project additions:** full masked-state identity, ancestor-probability checks, explicit node limit, frozen snapshots and JSON round-trip contract. No numerical test has run. |
| CARS oracle on QM9 | **Same soundness requirement; different oracle/task:** pruning requires that every completion in the cone has binary reward zero. Default leaf-only pruning is labelled ARS, not full branch-pruning CARS. Optional token constraints provide an incomplete oracle for this project's strict terminal decoder, keeping all future observations at their original positions. No partial RDKit/QED rejection is used. The supplied CARS functional-group grammars do not cover all QM9 target successes and are not adopted. Toy grammar verification is written but unrun; no full QM9 SMILES grammar is claimed. |
| Frozen CARS proposal | **Same law as CARS:** outside failure cones, `Q_W(y|z)=q_rho(y|z)/B_W`; the local transition is raw probability times child surviving mass, normalized by parent mass. The paper discusses freezing the trie. **Project additions:** a read-only proposal API for fixed-horizon masked states, log likelihood/cost records and separate underflow/exhaustion status. Unknown suffix probabilities use counted model forwards without modifying the frozen trie. No success retry occurs in this primitive. CPU float64 categorical arithmetic follows the project's completion convention; exact real-arithmetic identities are subject to floating-point error in software. |
| CARS on QM9 and approximation of V* | **New integration, not an original CARS experiment.** CARS Appendix H uses autoregressive models with few-shot examples and grammars for acrylates, chain extenders and isocyanates. It does not report QM9 masked diffusion or this value estimator. Step 8 writes discovery/estimation/adaptive sampling targeting `V_rho`. No equivalence to native `V*` or experimental success is claimed. |
| CARS value estimator | **Project addition derived from the CARS proposal law:** independent discovery/estimation RNGs, frozen W and fixed-m pre-acceptance draws give `B_W * mean(tau)`. Every failure/duplicate is retained. A scaled Hoeffding interval has conservative conditional coverage under exact sampling and sound pruning; software behavior is unverified. A zero-success estimate retains a positive upper bound; certified support exhaustion alone gives exact zero. Numerical underflow and incomplete queries remain unavailable. Whole-query reservations count discovery and estimation work. |
| CARS cache and verifier adapter | **Project additions:** source/context-bound optional immutable JSON snapshots, bounded resident trie LRU, and compact values pinned for the whole run. A cache hit retains origin uncertainty/provenance but charges zero fresh draws/NFE. Disk I/O is not NFE. The adapter uses the existing ordered verifier interface and exact leaf anchoring, with an explicit rho target; it does not alter the outer kernel or native decoding. Cold/warm comparisons must report origin costs separately. |
| RS/ARS/CARS generation | **Same update mechanism as CARS Section 3/Algorithm 1:** sample under current W, then learn only certified failures for later attempts. This implementation's ARS is strictly leaf-only. CARS additionally checks supported sibling continuations at visited prefixes, including successful paths; QM9 token constraints remain an incomplete oracle. **Project differences:** fixed-horizon rho completion, conservative whole-attempt admission, explicit finite attempt caps and freeze-on-capacity. Tries survive calls within the same exact context. No finite-budget exact-output or speedup result has been verified. |
| CARS experimental settings | **Unresolved project settings, not paper hyperparameters:** discovery attempts/work, fixed estimation count and node cap are null in `configs/cars.yaml`, pending profiling. The file is not a benchmark preset. Lark is a dev-only dependency for finite toy grammar enumeration; QM9 runtime does not import it. |

The locked zero-threshold convention is `complete AND valid AND QED >= threshold`,
matching this project's PLAN and the upstream validity gate. At threshold zero,
invalid samples still fail. This explicitly differs from the paper's literal
`R >= threshold` formula for invalid samples with `R=0`. The convention is saved
in each calibration protocol; no ambiguity is deferred to evaluation.

Calibration's Modal wrapper uses one RTX PRO 6000, 4 CPUs, 32,768 MiB, and a
six-hour timeout as unprofiled project choices. Collection and verifier training
use the same GPU/CPU/memory with a 24-hour timeout, also unprofiled. Only the
asset-preparation task, a partial base-training attempt and four local path
regression checks have run; enumeration, generation, calibration, verifier
training, and the remaining tests are unrun.

## Scope alignment with the requested project

The required outputs are (1) a QM9 comparison of MDM-VGB and CARS adapted to
the same QM9 backbone, and (2) CARS used inside MDM-VGB's value-verifier slot.
CARS's original molecular experiment is not QM9; its functional-group grammars
and autoregressive LLM results cannot be presented as an existing QM9 baseline.

The second output must address the conditional value under a locked native
reference. For the terminal law pi_native of the selected root sampling
protocol, `V*_native(z)=P_native(C(z),tau=1)/P_native(C(z))`, when the denominator
is positive. The current rho code is a diagnostic path, not a substitute for
this objective. A native sampler newly started from z is not automatically the
native root joint conditioned on C(z).

| Required part | What the current source establishes |
| --- | --- |
| QM9 data/backbone | Assets prepared; base training attempted with intermediate checkpoints. Final training completion is unverified. |
| MDM-VGB baseline | Libraries written, with declared upstream differences. Numerical reproduction and generation tests are unrun. |
| Standalone CARS-QM9 | Step 8 writes RS/ARS/CARS-rho adaptive sampling libraries; Step 10 adds generation entrypoints. The QM9 comparison remains unrun, with attempt/node caps unset. |
| CARS verifier inside VGB | Step 8 writes the fixed-m rho estimator/cache and `src/verifiers/cars.py` adapter. It connects through the existing interface/ValueTable; numerical integration tests are written but unrun. This is an intermediate V_rho path. |
| Native V* approximation | Step 9 writes native root trajectory scoring, an independent trajectory trie, frozen CARS numerator/unpruned denominator estimator, native MC, native cache/adapter and an independent-mask neural control. Enumeration/integration tests are written but unrun. QM9 costs/variance and denominators remain unprofiled; the native gate has not passed. |

PLAN sections 4.3/5.3 and implementation Step 9 now make the native-reference
bridge mandatory. The written trajectory numerator/denominator design avoids
substituting a left-to-right likelihood for native likelihood; it is a project
design, not an implementation from either paper. Enumeration and QM9 feasibility
must be checked before use. A ratio with an unresolved denominator is unavailable,
not a certified zero. Failure to complete this bridge leaves the integration
objective unfinished even if rho experiments succeed.

The upstream neural snapshot verifier remains a separate reproduction reference.
For an estimator comparison with the same native conditional target, the plan
adds a neural control using native-root terminals and independently chosen masks.
This control is a project addition: independence gives the desired population
conditional regression target, while finite data/model approximation error still
requires measurement. Snapshot labels from 128-step trajectories are not
silently re-labelled as exact 32-step native conditional labels.

The initial scope audit changed planning only. Steps 8–9 now have source and
tests written; no Step 8/9 code, tests or Modal jobs have been executed.

Step 9 differs from both papers: it serializes native parallel draws with fixed
pre-step logits, retains reveal timing/MASK loops, and uses the normalized
32-step final-denoise endpoint already declared for this project's baseline.
It does not change the baseline sampler or VGB kernel. Compatibility cones and
exact terminal failures are sound numerator exclusions; partial RDKit/QED is
not used to prune. Unpruned independent denominator draws retain failed
compatible molecules. Discovery, numerator and denominator use separate RNG
streams and a whole-query reservation. Fixed-work Hoeffding ratio bounds,
minimum compatible hits, unavailable results, optional native snapshots and
independent-mask neural labels are project additions. N/D has finite-sample
bias; a correct target alone gives no stationary-law guarantee. Serial path
sampling and rare compatible events may be infeasible on QM9; budget/counts
remain unset until profiling. Importance diagnostics require global support
and a pre-established weight bound, not the maximum observed weight. Serial
path draws preserve the selected transition law but consume RNG differently
from batched baseline sampling; equal seeds do not imply identical molecules.

## Step 9 static review (2026-10-12)

Review of the equations and source found no additional target-law mismatch:
the frozen proposal estimates the native numerator, while unpruned root draws
estimate the denominator, retaining reward-zero leaves and multiplicity.
This is source inspection, not an executed correctness check.

- **Numerical support:** a positive child cone can underflow during categorical
  exponentiation even when root surviving mass is ordinary. Native proposal
  draws now reject that case; the estimator reports unavailable and releases
  its reservation. It does not silently remove the edge or change the target.
- **ESS:** squaring small representable importance weights could produce zero
  and falsely report ESS=0. ESS now uses weights divided by their compatible
  maximum. Ratio weights and denominator uncertainty are unchanged.
- **Trie context:** direct failure learning now verifies reference settings and
  full calibration metadata, including a changed threshold under the same ID.
  Snapshots retain that metadata; incompatible snapshots are rejected.
- **Temporary memory:** sibling candidates are streamed rather than all full
  prefixes being materialized before the node cap is checked. Long paths still
  incur replay and trie costs; this does not establish QM9 feasibility.

New regression cases cover squared-weight underflow, positive-edge underflow,
unavailable numerical results/reservation release, and context drift. All are
unrun. Enumeration, fixed-panel uncertainty/availability and QM9 profiling
remain required. Measure end-to-end query wall time, including CPU trie work;
`CostRecord.elapsed_seconds` measures instrumented calls only. Also compare
serial and batched native transition probabilities at the chosen precision:
matching mathematical formulas does not establish equal numerical kernels.
No Step 10 source, model jobs or benchmark results were produced in this review.

## Step 10 runner and panel design

The runner assembles existing samplers; it does not alter successful candidate
weights, the backbone or the completion law when swapping native neural/MC/CARS
verifiers. Upstream 128-step snapshot labels have a separate neural-only target.
The normalized native32 endpoint and finite-budget baseline changes documented
above remain project variants, so this is not a reproduction of published tables.
Unresolved values now end a sample with an explicit failure and release its
completion reservation; they are not converted to zero. Work-cap refusal keeps
the existing reserved-completion behavior. All requested output slots are kept.

State-panel strata and first-occurrence selection are new project choices,
fixed without consulting compared values or filtering validity/reward. BASE
terminal posthoc masks are distinguished from actual native/walk snapshots.
CARS and BoN source runs are excluded to avoid constructing the baseline panel
from their selected outputs. This does not make a finite panel representative
of all states; its sources, order, strata and hash are persisted for review.

QM9 comparisons use an independent fixed-count MC reference with denominator,
uncertainty and availability diagnostics, not exact ground truth. CPU toy mode
enumerates whole parallel native steps independently of the trajectory cursor;
rho mode distinguishes fresh rho completion from the root-joint conditional.
Enumeration work and reference work are recorded separately from estimator work.
Synchronized query wall time includes trie/cache processing; setup (including
warm loading), source-run costs and artifact writing remain separately labelled.
Warm snapshots are checked against their completed source manifests and exact
estimator context; original costs remain provenance, without being charged again.

Source/config/regression tests are written but **unrun**. No enumeration gate
flags were changed, no QM9 budgets were invented, and no benchmark/evaluation
results were generated. Step 9 feasibility and the Step 10 small-run gate remain
unmet; Step 11 is not implemented.

## Step 12 verification design (written, not executed)

This step adds project tests and runners; it does not reproduce an experimental
procedure or a result table from either paper. It changes no generation kernel,
native reference, calibration rule or production draw/budget setting.

- MDM-VGB Appendix F defines raw NFE using realized model forwards. New tests
  compare counters against independent network-call records and a real tiny
  model hook, including nested MC/CARS, importance replay, batching and reserved
  completion. Pure backtracking and cached logits incur no invented forward.
  Reservations and failure charging are project accounting safeguards.
- CARS Appendix F discusses freezing a trie. Frozen-proposal tests retain that
  distinction; the fixed-count value estimator, uncertainty bounds and native
  conditional bridge remain project additions. A small unresolved denominator
  stays unavailable rather than being turned into a zero value.
- The offline integration uses three prepared fixture rows, two training steps,
  four calibration draws and two requested outputs per test run. These are test
  sizes, not QM9 paper settings or pilot/benchmark sizes. It checks resume state,
  checkpoint loading, batch order, shared targets, artifact flow and evaluator
  rejection of inconsistent metadata. Existing tests cover neural training.
- A disposable copy excludes `resource/`; a fresh process imports local modules,
  checks a model config, runs finite toy values/evaluation and builds a wheel.
  Static checks inspect runtime imports/load paths and Modal upload allowlists.
  The original reference directory is never moved or modified.
- Local/Modal verification saves real pytest logs/status when executed. Default
  execution is CPU; CUDA smoke needs an explicit request. A separate Modal
  function reloads and verifies committed log hashes, also for failed suites.

All new Step 12 tests and runners are **unexecuted**. No verification artifact,
numeric result, Modal readback success, benchmark approval or passed scientific
gate is claimed by this implementation.

## Step 13 controlled benchmark (written, not executed)

The matrix retains both required deliverables: standalone CARS adapted to QM9
versus the VGB upstream-label control, and VGB with matched native-conditional
neural/MC/CARS values. BASE/BoN are native controls; BASE-rho/RS/ARS stay labelled
diagnostics. A separate job compares all three native estimators on the same
locked panel per seed. No dataset/model training or calibration is performed
by the benchmark coordinator.

MDM-VGB Appendix F uses a shared backbone/reward, independent BoN rollouts and
an n*N walk cap. Existing defaults N=16, n=32, L_f=8, L_b=32, K=8, lambda=0
remain in the inference config. The common hard sequence-work cap additionally
counts nested estimator work and reserved completion; it is a project change,
not the paper's walk-step budget. The coordinator does not adjust these settings
or estimator draw counts to make a method finish.

CARS Appendix H.4 measures calls to obtain 100 valid unique molecules, with its
own timeout and three trials on industrial molecular classes. This project
requests the same declared output slots and seed list for every QM9 method,
retains unsuccessful slots and uses QED/Pass@95. Those tasks, stopping rules and
sample counts differ; pilot 128 or 3x1,000 outputs are not copied paper settings.
Benchmark seed/output counts are deliberately unset until explicitly chosen.

Cold method/seed boundaries preserve online trie reuse within a worker attempt
while preventing cross-seed leakage. Config/source/input hashes, snapshots,
dispatch records and separate immutable worker attempts are project workflow
controls. Resume skips verified completed units, preserves failed-attempt traces
and known cost lower bounds, and refuses ambiguous duplicate completed units.
Failed in-flight or unregistered work can remain unknown. Value-job work includes
the independent MC reference; it must not be interpreted as generation NFE.

Pilot readiness checks actual completion/guidance/common-pair availability,
without selecting configurations by reward or claiming value accuracy. A final
run requires a ready compatible pilot, current full verification and declared
native/profiling gates. This is a project readiness rule, not a paper theorem or
statistical acceptance test. Only reviewed final manifests should go to the
Step 11 evaluator; the retry ledger must accompany any end-to-end cost claim.

Step 13 config/coordinator/local and Modal runners/tests are **unexecuted**.
`locked` remains false, no missing manifest is invented, no numeric budget is
silently profiled, and no pilot, benchmark result or verification gate is claimed.

## Required checks before interpreting experiments

The present implementation is a project variant, not a verified reproduction
of the published tables. These conditions remain unmet; documentation of a
difference does not make its experimental effect negligible.

1. **Separate reproduction from the new method.** Validate a baseline against
   the supplied QM9 implementation under its actual decoding, terminal,
   completion, Momentum and inference settings. This project changes those
   behaviors; following a paper equation does not establish agreement with the
   released experimental code. A result gap from the paper cannot be attributed
   to CARS while these factors also differ.
2. **Compare estimators of the same quantity.** The main integration comparison
   needs CARS/MC/neural controls for the same native conditional target, reward
   and calibrated threshold. Rho experiments use matching rho controls and
   remain diagnostic. Keep the upstream trajectory-labelled neural result
   separate. On the supplied toy example with future observations, `V_rho=0.5`
   while the conditional value under `pi_rho` is `0.7`. Correcting to pi_rho
   does not establish equality to the native reference. The paper's Proposition
   B.1 explicitly requires reference consistency; Step 9 must address this gap.
3. **Hold the generation mechanism fixed when replacing the verifier.** Use
   the same frozen checkpoint/tokenizer, initial states, forward/backward
   shortlist, completion and stopping policies, Momentum convention, decoder,
   reward, and threshold across paired verifier comparisons. Neural-rho is
   currently trained on prefixes; arbitrary-mask performance must be measured
   on a fixed panel, not assumed from architecture or training objective. The
   main adapter must feed values into the same VGB candidate scoring path;
   swapping the outer generator as well would test a different question.
4. **Use feasible, measured compute budgets.** The Step 6 hard cap of 528 base
   sequences funds BoN, but one cold MC query with 128 draws and 31 masks needs
   3,968 sequence-forwards under the current policy. An entire candidate group
   costs more. Whole-query reservations can therefore prevent guidance from
   running. Profile before fixing budgets; record query refusals/fallback rate,
   actual calls, batch sizes, sequence work, verifier cost, oracle cost and wall
   time. A model call and a sequence-forward are different units. Show a matched
   cost frontier rather than treating equal outer step counts as equal compute.
5. **Verify pruning and estimation independently.** Execute the Step 7
   enumeration tests before enabling branch pruning. Token certificates here
   rely on the strict decoder and must be re-proven if that decoder changes.
   In Step 8, freeze discovery before fresh fixed-count estimation, keep failed
   draws and multiplicity, and distinguish `B_W` from success mass. Validate
   against the same target with uncertainty; a high Pass@95 result alone does
   not prove value accuracy or the paper's stationary distribution.

## Sources inspected

- [MDM-VGB paper](<resource/VGB for Masked Diffusion Model Efficient Test-time Scaling for Reward Satisfaction and Sample Editing.pdf>): Appendix F, printed pages 67–70; F.1 QM9 task and F.3 Table 7.
- [CARS paper](<resource/Constrained Adaptive Rejection Sampling.pdf>): Section 3, Algorithm 1, equations (1)-(3), incomplete-oracle discussion; Appendix F freezing discussion and Appendix H molecular generation setup.
- [Upstream CARS trie](resource/cars-main/cars/oracle_trie.py), [logits processor](resource/cars-main/cars/logits_process.py), and [grammar wrapper](resource/cars-main/llguidance_grammar_recognizer.py). Read for comparison only; Step 7 source is independently written, with no upstream CARS source copied.
- [Upstream asset preparation](resource/MDM-VGB-main/prepare_assets.py).
- [Upstream QM9 split, reward and decoding](resource/MDM-VGB-main/src/tasks/qm9/task.py).
- [Upstream training tokenization](resource/MDM-VGB-main/src/tasks/qm9/train_base_model.py).
- [Upstream model configuration](resource/MDM-VGB-main/src/tasks/qm9/mdlm_source/config.json) and [architecture](resource/MDM-VGB-main/src/tasks/qm9/mdlm_source/modeling_udlm.py).
- [Upstream base training settings](resource/MDM-VGB-main/configs/qm9/qm9_base_model_training.yaml) and [SUBS/noise definitions](resource/MDM-VGB-main/src/tasks/qm9/harness.py).
- [Upstream inference settings](resource/MDM-VGB-main/configs/qm9/qm9_inference.yaml) and [verifier training settings](resource/MDM-VGB-main/configs/qm9/qm9_verifier_training.yaml).
- [Upstream rollout settings](resource/MDM-VGB-main/configs/qm9/qm9_rollout.yaml), [collection](resource/MDM-VGB-main/src/tasks/qm9/task.py), [verifier architecture](resource/MDM-VGB-main/src/tasks/qm9/verifier.py) and [trainer](resource/MDM-VGB-main/src/tasks/qm9/train_verifier.py).
- MDM-VGB paper Definitions 4.2–4.3, Appendix A Algorithms 1–4, and Appendix F completion/budget definitions.
- [Upstream shortlist weights](resource/MDM-VGB-main/src/algorithms/vgb_candidates.py), [terminal/direction handling](resource/MDM-VGB-main/src/algorithms/vgb_terminal.py), [VGB](resource/MDM-VGB-main/src/algorithms/vgb.py), [Momentum](resource/MDM-VGB-main/src/algorithms/vgb_momentum.py), [VGR](resource/MDM-VGB-main/src/algorithms/vgr.py) and [value utilities](resource/MDM-VGB-main/src/algorithms/algorithm_utils.py).

These are development references only. Runtime code does not import or mount
`resource/`.
