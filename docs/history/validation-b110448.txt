# Adopted compact backend: current evidence

This release uses CLI defaults `CONSTANT_DIVISION=1`, `UNROLL=1`, and
`SHARED_LOADS=compact`; the `EMMY_PTX_` prefix applies to each. All compiler
backend modules remain byte-identical to reviewed source `34129e9`. The Python
API and explicit old-policy outputs remain unchanged. Blake approved adoption
on 2026-10-25 after Fable review 072 and Opus re-check 073.

Reviewed hosted batches for these exact enabled policies:

| Batch | Cases | Requests | Result |
|---|---|---:|---|
| 064 | Five matmul/attention fixtures, including flat controls and an nvcc comparison | 36 | All outputs within original tolerances; performance gates passed |
| 068 | Rectangular and odd-tail matmul | 12 | 30,720 checked values passed; rectangular 1.507x, tail gate pass only |
| 069 | Nine shipped typed/copy/tensor examples | 18 | 760 logical values bit-exact |
| 071 | Eight signed-math/reduction/copy/runtime witnesses | 16 | 5,942 logical values bit-exact |

Opus independently decoded all four batches with no blocker or required
correction. Total coverage is 15 corpus fixtures plus nine examples, not all 429
changed corpus outputs. 069 and 071 used one iteration with zero warmups and
are correctness evidence only. i64 uses exact i32-word transport, not direct
service i64 JSON. Tensor inputs remain small exact values. The half-reduction
result applies to its frozen summation order. One or two schedules do not prove
race freedom or behavior on hidden inputs.

Within batch 064, baseline/compact means (microseconds) were main matmul
31.272/15.480, staged rectangular 29.360/15.008, asymmetric attention
23.192/15.024, small-K 3.400/3.344, and split-K 5.040/5.008. The matched nvcc
main-matmul mean was 15.032. Only two request medians per version were sampled;
compact versus vector was unresolved. These compare the supplied CUDA schedule,
not cuBLAS or a peak-throughput implementation. Do not multiply ratios across batches.

Offline 066 assembled/disassembled 862 distinct baseline/candidate PTX texts
for 438 changed fixture outputs: no spills or static instruction-count increases,
but 40 fixtures increased local register counts. Those counts are not hosted
register counts or measured occupancy. The 070 replay reproduced 12,602 equal
sampled executions, 0 mismatches and 984 validated load groups. Its 574 skipped
runs map to 37 fixtures, unsupported in both versions, all default differences
limited to constant division. Skips are not passes. Local models and assembly
are distinct from hardware evidence.

Successor validation on frozen code commit `03707bd` completed offline on
macOS, Linux ARM64 and Linux x86-64: 381 tests on each platform. Both fresh Linux
installs passed `pip check`, all 440 default CLI output hashes, and the three
portable example/historical-evidence commands. Containers used cached images
and hashed wheels with networking disabled. The later documentation-only
successor preserves every executable file from this tested archive.

The local policy bridge checked 440 default CLI outputs against frozen compact
PTX, 440 explicit-off outputs against the previous backend, and 5,280 API
comparisons across six policy modes with both backend settings. All outputs and
550 matched refusal cases agree, including 2,586 prior corpus-mode hashes.
These are offline identity checks, not additional hardware runs. The default CLI
also reproduces every one of the 24 recorded hardware kernel payloads using its
exact frozen metadata. All 28 files under `ptxgen/` match `34129e9` byte for byte.
Fable's encompassing review 072 of extracted `2c5d165` independently reproduced
381 tests, the 440 default/rollback outputs, unchanged plain API behavior and all
24 hardware PTX payloads. It found no implementation blocker and required only
correction of stale status text in README, FEATURES and DESIGN. This docs-only
successor `96c6d3a` made those corrections without changing executable files.
Opus re-check 073 verified all 357 archive files, the five-document-only delta
and the corrected wording, closing R1 with no blocker or required correction.
This adopted release updates only documentation from `96c6d3a`; all executable
files, tests, dependencies, examples and sealed replay packets are unchanged.

The historical records below describe earlier sources and policies. They remain
preserved and should not be interpreted as extra hardware coverage for this
candidate. Successor-specific portability and default-policy checks are recorded
in the review packet. The final adoption record carries those results by file
identity; the adoption step adds no hardware results or performance measurements.

---

# Validation record

## Qualified judging candidate — 2026-10-24

This branch starts at `83348ec`; it is not the adopted `8f7f887` release.
Its three tensor examples now have hosted correctness results, described below;
there is no tensor-core performance claim. This packaging revision changes no
compiler, test, dependency or example code from qualified `264f72a`.

- 349 local tests pass under the socket guard, including 10 tensor-fragment
  tests. Those tests cover 18 changed-input/stride/K-depth cases, unsupported
  fields, lifetimes, bounds, participation, policy combinations and broken controls.
- All 431 existing fixtures match `83348ec` across six policy combinations:
  2,586 pairwise comparisons of PTX bytes or refusal outcomes, with no changes.
  This carries forward artifact identity; it does not add hardware coverage.
- Nine bundled examples compile. The three MMA examples also match the offline
  emitted-instruction model; six typed/async examples get reference consistency
  checks, not simulated kernel execution.
- Local CUDA 12.9.1 `ptxas -arch=sm_120` accepts all nine examples with zero spills.
  The three MMA cases lower to one, one and two `HMMA.16816.F32` instructions,
  using 32, 34 and 40 registers for the basic, padded and K=32 cases respectively.
- Validation caught and corrected `.s32` packed operands: MMA requires `.b32`.
  The instruction model now checks this type distinction and has a broken-type
  control. Model success alone was insufficient to establish assembler validity.

Evidence is kept outside the submission source at
`evidence/judging_features_049/` in the research workspace. The bundled examples
and tests are sufficient to reproduce the portable checks without that folder.
Independent Opus review of frozen source `d993c88` found no implementation
blocker. It reproduced the 349 tests, 431 hashes and 2,586 comparisons from
extracted archives. Its separately authored forward-map executor checked
3,000 generated programs in 12,000 modeled executions, including 974 programs
with repeated MMA accumulation; 57 targeted probes matched their expected
acceptance or refusal. Local assembly accepted 89 generated/example files with
zero spills, including 83 tensor kernels. These are offline checks, not GPU
results or exhaustive numerical coverage.

That review required an explicit reachability disclosure: none of the 15 public
problem IRs or 431 prior corpus inputs uses the new fragment statements, and
Emmy's upstream passes have not been shown to emit our supported form. README
and FEATURES now state this. Opus closed the correction in review 053.
Hardware qualification 055 and its independent receipt review are complete.
Fable's encompassing review 056 of extracted archive `4ea1fd6` completed
on 2026-10-24 with no blocker and no required correction. This documentation-only
successor records that result; release adoption remains Blake's decision.

The exact `d993c88` archive also passed fresh Linux ARM64 and emulated x86-64
checks: 349 tests, 431 CLI hashes and all nine example hashes on each platform.
Opus inspected those records but did not repeat the Linux runs. This successor
changes documentation plus the check_examples module docstring; compiler,
tests, example artifacts and executable checker statements are unchanged.
See [FEATURES.md](FEATURES.md) for the precise supported contract.

## Encompassing review 056 — 2026-10-24

Fable reviewed the extracted `4ea1fd6` archive offline under the socket guard,
with no blocker and no required correction. The review independently reproduced
349 tests, 431 frozen CLI outputs and all three portable commands from a relocated
archive and empty working directory. It also checked Git visibility, decoy input
surroundings, compiler import/read paths, and the source and walkthrough claims.

Both replay checkers were read and exercised with 76 re-sealed mutations:
74 were refused with ValueError, one missing echo failed with KeyError, and one
performance value within the original tolerance passed as designed. This was a
code and saved-evidence review, not new hardware qualification. The 2,586 six-mode
comparisons, Linux records and earlier model/assembly checks carry forward by
source identity; they were not rerun in review 056.

The full review is `reviews/fable-judging-release-056-review.txt` in the research
workspace, with evidence under `evidence/fable_judging_056/`. Optional checker
hardening O2/O3 is outside this documentation-only successor. Both inner packet
READMEs remain sealed preparation snapshots; the outer evidence guide records
the completed review without changing any checker, manifest or receipt.
Hardware evidence, numerical limits and all refusal boundaries are unchanged.

## Completed qualification 055

Following Blake's explicit approval, core completed 12 requests and passed a
separate offline receipt check before synchronization completed 26. All 19
compiles and 19 runs passed: zero warmups, one iteration, no retries or errors.
Both one-use allowances are consumed. Opus independently rebuilt all payloads
and decoded all 11,262 returned values without the runner or Codex's checker.

| Witnesses | Result | Boundary |
|---|---|---|
| `mma_tile`, `mma_padded`, `mma_k32` | Bit-exact | Small exact-integer FP16 lane mapping and accumulation |
| `m_rectangular` | Maximum absolute error 5.7220458984375e-6 | Original rtol=atol=1e-3 |
| `async_lane_f16`, `async_lane_f32` | Bit-exact, including tails | One frozen launch each |
| 13 synchronization witnesses | Bit-exact | One launch each; not race-freedom proof |

All compiles returned `sm_120`; the protocol does not attest the physical GPU
model. Metadata, IR, PTX, inputs, original references, tool hashes and clean
source matched the approved plans. Signed zeros match. The padded halo and
copy tails rely on documented zeroed output allocation. `sync_12`'s all-zero
reference cannot distinguish stores from a no-op on hardware; Opus's offline
sentinel probe separately verified all 256 stores. The roughly 15–33 us
single-iteration timings are recorded but support no performance claim.

These 19 results cover the 16 changed outputs still pending after 046, plus
three new tensor examples. With the five measured in 046, all 21 outputs
changed by `83348ec` have hardware results. This does not put all 431 offline
fixtures or all nine examples on hardware. The six scalar/copy examples did
not run. No general FP16 rounding, subnormal/NaN or throughput claim follows.

All original payloads, 38 raw receipts and Opus's unchanged result are bundled
under `validation_records/qualification-055/`. Its local checker replays saved
results; it performs zero requests and is not an independent peer decoder.
Fable read both portable replay checkers and exercised 76 re-sealed mutations
during review 056. Hardware evidence and the checker code are unchanged.

## Earlier release evidence

The hardware and original portability evidence below describe executable source
`77d6b818e1c8dcf251e3fb6fd41192149e93ad4d`, carried into the reviewed release
`c15a1fd` by documentation-only changes. The adopted readability release `8f7f887` is a separate
source revision; its carried-forward checks are described below. Earlier checkpoint documents
remain in Git history and the external evidence archive.

## Adopted readability release — historical checks

- All 318 existing tests and 10 new comparison-helper tests pass locally.
- All 431 default CLI outputs reproduce the frozen hashes byte for byte.
- Six Python API policy combinations across those fixtures match the reviewed
  release in 2,586 comparisons, including 266 matching refusals.
- Seventy AST comparisons check extracted executable bodies against the original
  source after explicit receiver and delegation normalization.
- Nineteen relocated/renamed cases with misleading adjacent reference and PTX
  files remain unchanged apart from the requested entry rename; compilation
  reads only the IR and metadata and performs no external operation.

These checks are offline. No new GPU request was made. The comparison helper's
new network behavior is tested with fake responses only, including failure at
every request position, interruption accounting, and the request cap. This does
not independently qualify that helper on the live service. Fable independently reviewed the extracted `48ef7f6` archive with no blocker
or required correction. Its two-archive sweep matched all 3,633 comparisons
across 519 IRs and seven policy combinations, including 359 matching refusals;
it also reproduced the 328 tests and 431 outputs and passed 48 injected helper
failures. These are offline equivalence checks, not new hardware coverage. Fresh installs from an exact source
archive also pass all 328 tests and reproduce all 431 CLI hashes on Linux ARM64
and emulated Linux x86-64. Both use cached hashed wheels, a socket guard and
network-disabled containers. The final archive differs from that tested archive
only in this validation document, which records the completed results.

## Reviewed release evidence

| Check | Recorded result | What it establishes |
|---|---|---|
| Local regression suite | 318 passing tests | Tested contracts, arithmetic and refusal behavior |
| Exact-archive Linux ARM64 rehearsal | 318 tests; 431 CLI hashes match | Installation and output reproducibility on that platform |
| Exact-archive emulated Linux x86-64 rehearsal | 318 tests; 431 CLI hashes match | Same checks under emulation, not the hidden evaluator |
| Latest hosted qualification | 15 bounded witnesses pass | Those exact kernels, inputs and launches on `sm_120` |
| Independent generated-probe audit | Demonstrated findings closed by Opus | No remaining demonstrated in-scope gap in that finite probe set |

An instruction model is not GPU execution. Assembler acceptance is not a
numerical result. An unchanged PTX hash carries previous evidence only with
matching IR, metadata, inputs and launch. None of these counts is language
coverage, hidden-set success, or a universal numerical bound.

## Latest typed qualification

The reviewed 15-case plan preserved original references and tolerances. The
first approved batch stopped after 28 requests: 14 successful compiles and
13 passing runs, followed by a run-stage service error for `power_i64`.
The response was `Python int too large to convert to C long`, without outputs.
All submitted integers were valid signed64 values. Intermediate floating-point
coercion reproduces that exception locally, but the server implementation and
execution phase are unknown. No retry or remaining case was sent.

| Witnesses | Transport | Result |
|---|---|---|
| `boolean_and`, `boolean_or`, `boolean_not` | Direct f32 buffers | Exact outputs |
| `power_i32` | Direct i32 buffers | Exact outputs |
| `remainder_f32_finite` | Direct f32 buffers | Exact outputs, including zero signs |
| `loaded_cta_stride`, `loaded_warp_stride`, `loaded_cta_boolean` | Direct i32 buffers | Exact outputs |
| `cta_exchange`, `warp_exchange`, `partial_permuted_tree` | Direct f32 buffers | Exact outputs |
| `floor_f16_finite` | Direct f16 buffers | Exact outputs, including zero signs |
| `remainder_f64_finite` | Direct f64 buffers | Exact outputs, including zero signs and subnormals |
| `power_i64`, `power_i64_alias` | Paired i32 words with unchanged buffer bytes | Exact reconstructed i64 outputs |

A separately reviewed and approved transport plan completed exactly four
requests for the two i64 cases: one compile and one run each, zero warmups,
one iteration, no retries or health probes. Original IR, PTX, inputs, references,
launches and buffer byte counts were unchanged. Each i64 buffer was represented
as low-word-first i32 pairs; the original kernel still performed i64 loads,
arithmetic and stores. Reconstructed outputs matched all 256 values per case,
including 72 values beyond 2**53 and signed64 extrema. This qualifies the kernels
through adapted byte transport; it does not establish direct i64 JSON support.

Across these two batches, 32 requests produced 15 passing witness results and
one retained service error. The compiles report `sm_120`; the protocol does not
attest the physical GPU model. Single-iteration times are not benchmarks.
There is no persistent device-buffer pipeline claim. Both approvals are consumed.

## Local arithmetic, scheduling and memory evidence

The tests and external models cover scalar widths, aliases, negative
intermediates, runtime strides, lexical/fold scopes, boolean composition,
collective membership, shared-memory layout, argument order and zero ownership.
Broken controls are retained where used to test that a checker detects a fault.

Examples of additional model evidence, not blanket hardware qualification:

- Floating divmod: 548,864 scalar comparisons against NumPy, including sampled
  bit patterns, all binary16 dividends for selected divisors, signed zeros and
  special values. Nine whole kernels also passed two CPU thread orders.
- Large logical Tiles: selected CTAs around signed32/unsigned32 boundaries
  for a logical size of 4,295,098,369. The full huge grid was not executed.
- Boolean audit correction: eleven fresh generated kernels passed original
  references in both model orders; six supplementary generated cases followed.
- Dependent ABI/zeroing: four modeled pipeline runs and four broken controls.
  These do not establish device-resident multi-kernel service execution.
- Binary64 math, trigonometric and power probes: sampled numerical evidence,
  including high-precision comparisons where recorded, not an all-input ULP
  guarantee or a device-throughput measurement for every new path.

Opus independently reviewed the shape guards, typed boolean provenance,
collective proofs, generated refusal cases and numerical conflict dispositions.
Five authored stress-reference conflicts were classified as explicit typed-IR
rounding versus higher-precision references. Original failing evidence and
reference tolerances remain unchanged. This does not establish which convention
an unseen reference will use. Arbitrary memory-derived tree coordinates,
unproved convergence and other proof-envelope cases may still refuse.

## Performance evidence

The tables in this section compare against our own earlier backend. Separate
experiments also compared selected kernels against nvcc compiling the organizers'
supplied CUDA; these are not cuBLAS or Triton comparisons. The completed
attention pair measured 90.776 microseconds for our released compiler and 74.376
for nvcc (ours took 1.2205 times as long). Earlier matmul/nvcc data was partial.
There is no clock locking, reliable confidence interval or hidden score.

### Integer index reuse — historical comparison

The five-pair family batch used 30 serial requests and checked all 20 run
outputs. Both variants used 40 registers. The original staged matmul measured
2.3782×; three related launch sizes measured approximately 2.281×, 2.284× and
2.281×. The split-K control measured 1.0079×, effectively flat.

At the index-reuse release checkpoint, PTX for the original staged matmul,
staged 96×192 variant and split-K control matched candidate requests 02, 08
and 26 of that saved batch exactly. Later alignment and loop-layout changes
alter the matmul outputs; those older timings are historical evidence, not
measurements of the current candidate. The transfer experiments support the measured loop at those launch
sizes, up to 128 CTAs, not a universal shape/performance claim. The selector
also admits the flat control through load-staging div/rem reuse. It is a scope
restriction and cannot predict speedup for arbitrary new kernels.

The broader initial optimization screen failed its two-win adoption criterion.
The subsequent loop policy followed a narrower investigation and separate
family test; it does not retroactively turn the broad screen into a pass.
Unrestricted reuse and strict+loop/tree+loop combinations remain experimental.

### Corrected loop layout — candidate experiment 046

Candidate `83348ec` was compared with alignment candidate `9b588d9` in the same
five-case experiment. Candidate `d993c88` emits those five PTX files unchanged.
All 30 requests completed (10 compiles, 20 runs), and all 1,157,456 returned
values passed the original tolerances. The counts include repeated runs.

| Case | Baseline µs | Candidate µs | Baseline / candidate |
|---|---:|---:|---:|
| Main 512×512 matmul | 39.536 | 31.376 | 1.2601× |
| Staged rectangle | 39.464 | 29.248 | 1.3493× |
| Asymmetric attention | 23.616 | 23.096 | 1.0225× |
| Small-K matmul | 3.504 | 3.488 | 1.0046× |
| Split-K control, identical PTX | 4.904 | 4.904 | 1.0000× |

Each entry is the mean of two request medians, with 2 warmups and 20 iterations
per request, on the shared service. Only within-batch ratios are meaningful.
The small differences are not statistically established wins. Baseline code
varied between batches: staged rectangle +5.07%, asymmetric attention −6.23%.
Do not multiply these ratios by historical FMA or index-reuse ratios.

The preceding motion candidate `050a620` passed numerical checks but FAILED the
performance gate: asymmetric attention was 0.802× and small matmul was 0.656×.
It was not adopted. Local assembly showed the arithmetic motion disrupted
unrolling; moving counter initialization back beside its loop header restored
local unrolling. The corrected candidate removed both measured slowdowns in
046 without kernel-specific rules. Hosted timings fit that account, but we do
not have hosted machine code proving the same cause there.

The five results initially left 16 changed outputs unqualified. Completed
qualification 055 now covers those three copy-width and 13 synchronization
witnesses, plus three tensor examples. It adds correctness evidence only.
No tensor-core performance result exists.

A portable offline replay of all five pairs is included at
`validation_records/performance-046/`. Opus verified that all receipts,
payloads and the original decode match its earlier independent evidence.
Fable reviewed this Codex-authored replay checker in review 056;
its successful replay is not a second independently authored decoder.

### FMA contraction (historical comparison)

A four-workload ABBA comparison used 24 serial requests, two warmups and 20
iterations per run. All 16 output checks passed. Mean-of-medians speedups were:

| Workload | Baseline µs | Candidate µs | Speedup |
|---|---:|---:|---:|
| Matmul 512 | 102.968 | 101.000 | 1.0195× |
| Rectangular matmul | 4.992 | 4.976 | 1.0032× |
| Attention | 115.344 | 90.768 | 1.2708× |
| Attention 128 | 53.888 | 51.856 | 1.0392× |

The geometric mean was 1.0780×. Registers were unchanged. Nine remaining
changed fixtures subsequently passed a consolidated 18-request qualification.
At that milestone, 13 of 185 artifacts changed and 172 retained their bytes.
These are historical counts, not 185 newly executed current kernels. Later
Accum-only eligibility preserved those retained default outputs; subsequent
index reuse has its own measurements above. Do not multiply the two reported
speedups into an unmeasured combined score.

Current auto contraction is restricted to typed Accum additions and positive
absolute tolerance. Strict mode remains available. Neither reduction provenance
nor positive atol bounds arbitrary cancellation or overflow error.

### Rejected or disabled experiments

Adding `.maxntid` passed a four-kernel correctness comparison but measured a
0.998129× geometric mean, with unchanged registers and larger cubins. It was
reverted. Short-tree lowering passed a limited correctness screen but failed
performance gates; it remains disabled. All changed opt-in artifacts have not
been hardware-qualified. Flat and failed results are retained.

## Reproduction and evidence map

Follow the README for dependencies and the separate organizer checkout. The
historical 318-test Linux rehearsals used the exact archive with SHA256
`fed588c47cbb06e31e6094e590e7f5bc380fcfe71f14766f9bd3dff04348e76a`.
The final documentation archive has a separate identity; its unchanged
executable sources carry this platform evidence through a recorded byte bridge.

The complete 046 and 055 saved experiments are bundled; see
[the offline evidence guide](validation_records/README.md). The following
older research records remain outside the submission repository and are not
required for the portable example/replay commands:

| Workspace path | Evidence |
|---|---|
| `evidence/boolean_bitwise_027/` | Source freeze, 318 tests, 431 CLI identities, two Linux rehearsals |
| `evidence/typed_qualification_order_030/` | Reviewed 15-case plan and unchanged original artifacts |
| `evidence/typed_qualification_live_031/` | 28 actual requests, 13 passing outputs, retained service error |
| `evidence/i64_word_runner_033/` | Frozen transport plan, model and adversarial runner checks |
| `evidence/opus_i64_word_runner_001/` | Independent runner/byte/model review |
| `evidence/opus_i64_word_live_001/` | Four actual requests and independent i64 decoder |
| `evidence/final_package_034/` | Final documentation/source bridge and receipt recheck |
| `evidence/index_family_run_001/` | Saved family timing and numerical receipts |
| `evidence/runs/fma_001/`, `evidence/runs/fma_qualification_001/` | Historical FMA comparison and qualification |
| `reviews/opus-boolean-bitwise-correction-review.txt` | Final implementation-audit disposition |

Earlier checkpoints remain recoverable from Git history and sealed workspace
records. Fable's final encompassing review assessed the extracted `4ea1fd6`
archive; this successor changes only documentation and needs an identity and
wording re-check before adoption.
No publication, hidden-evaluator result, prize placement or universal correctness
claim follows from this record.
