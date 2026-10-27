# Validation of the submitted implementation

Start with the current records, not the historical review chronology:

```sh
python validation_records/current/check.py
```

This command sends no requests. It verifies preserved source hashes, regenerates
24 tested PTX files and recomputes results from all 82 saved service responses.
[Packet contents and method](validation_records/current/README.md) ·
[source identity](docs/PROVENANCE.md) · [historical record](docs/history/README.md).

## Current hardware coverage

| Batch | Cases | Compile / run responses | Result |
|---|---:|---:|---|
| [064](validation_records/current/064/plan.json) | 5 | 12 / 24 | Matmul/attention comparisons and controls; 2,206,032 returned values within original tolerances |
| [068](validation_records/current/068/plan.json) | 2 | 4 / 8 | Rectangular and odd-tail matmul; 30,720 returned values within original tolerances |
| [069](validation_records/current/069/plan.json) | 9 | 9 / 9 | All shipped examples; 760 logical values bit-exact, including signed zeros |
| [071](validation_records/current/071/plan.json) | 8 | 8 / 8 | Signed indexing, typed sums, copies, runtime loops and branches; 5,942 logical values bit-exact |

That is **15 corpus fixtures plus nine examples**, not GPU coverage of all 440
offline inputs. The default CLI regenerates the exact current PTX recorded in
all four batches. Cases 069/071 used zero warmups and one iteration, so their
timings are not benchmarks. Both i64 cases used low/high i32-word transport to
preserve bits; this does not validate direct i64 JSON serialization.

The three tensor examples test lane mapping, padded stores and repeated K-tile
accumulation on small exact-integer FP16 inputs. They do not establish general
FP16 rounding, throughput, or upstream generation of that explicit fragment form.
One-run copy and synchronization tests do not prove race freedom. Hidden inputs
remain untested. Original metadata and tolerance values are preserved.

## Measured performance

These means use two request medians per variant, two warmups and 20 iterations per
request. They are from a shared service without clock locking or confidence
intervals. Each row compares versions **inside one batch**; do not multiply
ratios from different batches into an unmeasured total.

| Batch / case | Preceding backend µs | Current compact µs | Ratio | Interpretation |
|---|---:|---:|---:|---|
| 064 main 512×512 matmul | 31.272 | 15.480 | 2.020× | Clear measured improvement |
| 064 staged rectangle | 29.360 | 15.008 | 1.956× | Clear measured improvement |
| 064 asymmetric attention | 23.192 | 15.024 | 1.544× | Clear measured improvement |
| 064 small-K | 3.400 | 3.344 | 1.017× | Effectively flat |
| 064 split-K control | 5.040 | 5.008 | 1.006× | Effectively flat |
| 068 rectangle | 4.872 | 3.232 | 1.507× | Both current samples below both baseline samples |
| 068 small-K tail | 3.664 | 3.408 | 1.075× | Baseline spread exceeds mean difference; no reliable speedup claim |

Within 064, nvcc on the organizers' supplied main-matmul CUDA averaged **15.032 µs**.
Our compact mean is about **3% higher**, with substantial sample spread. The vector
variant averaged 15.384 µs, too close to distinguish from compact here. These
are matched supplied-schedule comparisons, **not cuBLAS or a hardware-peak claim**.
[Exact medians, CUDA source and build details](validation_records/current/README.md#main-matmul-comparison)
are bundled. The checker recalculates means and ratios from the raw response text.

The adopted backend combines exact constant-division lowering, bounded unrolling
and shared-load grouping/rebasing. This batch does not isolate each pass's share
of the improvement. Split-K and small-K controls were retained rather than hidden.

## Offline checks and what can be reproduced

| Check | Recorded scope | Reproduction here |
|---|---|---|
| Regression suite | 381 tests on macOS, Linux ARM64 and Linux x86-64 | README command with separate organizer checkout and dev requirements |
| Public CLI contract | 15 public IRs compile, including entry names | `compile.py` on the organizer's public inputs |
| Default/rollback identity | 440 inputs each; 431 corpus + nine examples | Development census; not all generated corpus inputs are shipped |
| Current hardware payload identity | 24 current default PTX artifacts | `validation_records/current/check.py` regenerates all 24 |
| Backend assembly census | 862 distinct baseline/candidate PTX texts | Development record; offline assembly, not device execution |
| Sampled semantic model | 12,602 matching executions, 984 validated load groups | Development record; 574 skipped runs across 37 fixtures are not passes |

The publication review reran all 381 tests, the three previously advertised
portable example/historical replay commands, and all 15 public CLI invocations
from outside the checkout. These used the existing macOS Python environment,
with socket connections blocked. They were not new Linux installs or GPU tests.
The added current checker and its adversarial tests are publication tooling,
separate from the preserved 381-test implementation suite.

Linux results and the 440-output census were recorded before publication.
Executable identity preserves their relevance, but a hash check is not a new
execution. The [provenance manifest](docs/source-provenance.json) verifies the
preserved implementation, original tests/tools, examples and historical packets.

## Numerical and coverage boundaries

- Guarded FMA contraction is enabled only for eligible accumulation patterns and
  metadata tolerances. Positive absolute tolerance is a policy guard, not a bound
  for arbitrary cancellation, overflow or reduction inputs.
- f32 exponential/logarithmic/reciprocal-square-root paths use GPU approximate
  instructions; activation formulas build on these. Accuracy evidence is fixture
  tolerances and sampled bit checks, not correct rounding. Explicit f64, trig and
  power lowerings also have sampled numerical evidence, not universal libm claims.
- Typed IR rounding can differ from a higher-precision reference. Original
  tolerance and reference-conflict records remain in the historical narrative.
- Unsupported or unproved schedules refuse. Refusal is safer than invented
  semantics, but it can reduce coverage of legal hidden inputs. The compiler is
  not a complete malformed-IR validator.
- The historical all-zero `sync_12` witness cannot distinguish stores from a
  no-op on zero-initialized memory; only its offline model checked those stores.
- Historical short-tree and arithmetic-motion candidates failed performance gates;
  their failures remain recorded. The later signed-division experiment also stayed
  outside the adopted implementation after its primary timings were flat.

## Evidence integrity and review roles

Raw responses are unsigned participant-held records. Packet hashes detect local
byte changes; they do not authenticate the service or establish external audit.
Codex implemented the compiler and this new portable checker. Opus separately
checked the original current-batch receipts; Fable reviewed release candidates.
Those are AI development/review roles, not organizer endorsements. The original
reviews do not silently extend to new publication tooling. See
[provenance](docs/PROVENANCE.md) for roles and internal IDs.

Older 046/055 replay packets remain unchanged and are explicitly historical.
Their results must not be mistaken for current default measurements. Detailed
old counts, internal review chronology, failures and workspace references are
preserved under [development history](docs/history/README.md).
