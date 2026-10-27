# Features a reviewer can inspect

The backend translates typed Emmy IR into PTX. These examples are small,
reproducible entry points into the implementation. No example name or input
value selects a compiler fast path. The examples are authored here using Emmy's
IR constructors and serializer; they do not use its renderer or CUDA backend.

## Start in a clean clone

With Python 3.12+ and the pinned dependencies installed:

```sh
python -m pip install -r requirements.txt
python tools/check_examples.py
python compile.py examples/mma_tile/tensor_tile.ir.json > /tmp/tensor_tile.ptx
python tools/explain.py examples/mma_tile/tensor_tile.ir.json --report /tmp/decisions.json > /tmp/explained.ptx
python -m unittest discover -s tests -p test_tensor_fragments.py -v
```

These commands require no GPU, CUDA toolkit, organizer checkout or network
after dependency installation. `check_examples.py` labels what it checked:
the tensor examples execute in a limited instruction model, while typed-add
and async-copy examples get compilation and reference consistency checks.
Neither is a hardware run. `explain.py` uses explicit default auto policies
with constant-division, unrolling and compact shared loads enabled, matching
the CLI defaults. It writes transformation decisions beside PTX stdout. These
decisions are not a prediction of speedup or a report of every lowering.
For the tensor example, integer reuse is disabled and constant divisions are
lowered. The sidecar does not explain MMA lowering. To inspect a selected optimization,
run it on an organizer matmul input with that input's adjacent metadata.

The full regression suite also needs the separate organizer checkout and the
development dependencies, as described in the README. All bundled inputs and
expected arrays are local. The two generators in `tools/make_*examples.py`
can recreate them into a new directory. Metadata contains the exact input data,
not a seed that produces a different distribution. Input NPZ files duplicate
that data for offline checking.

| Capability | Example / focused tests | Evidence and limits |
|---|---|---|
| FP16, FP32, FP64, i32, i64 arithmetic | `examples/add_*`, `tests/test_scalar_types.py`, `test_wide_math.py` | Existing typed lowering. All five typed-add examples passed hardware qualification in batch 069 under this default backend. i64 example values exceed 2^53 and must not pass through float32/float64 comparison. Earlier hosted i64 evidence used paired i32 transport. |
| Async global-to-shared copy | `examples/async_copy`, `tests/test_alignment.py`, `test_swizzles.py`, `test_barrier_lifetimes.py` | Existing `cp.async` path, explicit commit/wait/barrier, 16-byte transfers in this example. The example passed batch 069; additional half-tail and f64 cross-boundary cases passed 071. This is not TMA or a race-freedom proof. |
| Dense tensor-core operation | `examples/mma_tile`, `mma_padded`, `mma_k32`, `tests/test_tensor_fragments.py` | New FP16-input/FP32-accumulator `mma.sync` path. Offline models, local sm_120 assembly and all three hosted examples pass. Small exact-integer inputs only; no throughput claim. |
| Integer index reuse and loop-invariant motion | `ptxgen/index_reuse.py`, `loop_invariants.py`, their tests | Retains the base candidate's generic transformations. Existing case evidence is bounded, and the reuse selector can also select a flat control. |
| Exact integer lowering and shared-load grouping | `ptxgen/constant_division.py`, `loop_unroll.py`, `shared_loads.py`; `tests/test_backend_gap.py`, `test_shared_loads.py` | Generic address/dependency proofs and bounded loop expansion. CLI defaults enable the measured compact combination; explicit off switches preserve the preceding backend. |
| Collective correctness | `tests/test_warp_participation.py`, `test_named_barriers.py`, `test_runtime_schedule.py` | Explicit membership, lifetime and ordering proofs with conservative refusals. These tests do not prove every possible schedule safe. |

## Tensor-core scope

The new module is `ptxgen/fragments.py`. It lowers this real IR chain:

1. `RegFragment`: allocate four packed registers for A, two for B, four FP32
   accumulator registers for C; initialize C to zero.
2. `LdmatrixLoad(staged=False)`: load each lane's FP16 values from global memory
   and pack pairs into `.b32` registers. This documented global-load variant
   does **not** emit the shared-memory `ldmatrix` instruction.
3. `MmaSyncPtx`: issue `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
4. `RegStore`: distribute the four accumulator values owned by each lane to
   the FP32 output buffer.

The first scope supports one complete 32-thread CTA, straight-line fragment-only
code, row-major rank-two physical buffers, constant tile origins and complete
16×8×16 atoms. Padded leading dimensions and nonzero origins are supported;
repeated loads and MMA operations accumulate additional K tiles. The example
with K=32 exercises that accumulation. No ordinary FP32 matmul is silently
converted to FP16. The IR must explicitly request the tensor-core operation.
This backend interprets global-load `ldm=0` as the physical row stride and
`src_index` as a `(row, column)` tile origin; interoperability of these choices
with upstream tensor lowering is not yet established.
None of the 15 public problem IRs or 431 prior corpus inputs contains these
fragment statements; m01/m02 still take the scalar FP32 path. The three bundled
tensor demonstrations are authored through Emmy's typed IR constructors.
Emmy's upstream passes have not been shown to emit this supported one-warp,
global-load, constant-origin, straight-line form. Supporting these explicit
statements does not establish coverage of Emmy-generated tensor-core kernels.

The compiler refuses shared staging, swizzles, transposes, fragment arrays,
partial tiles, guards, mixed scalar statements, nested control flow, multiple
CTAs, epilogues, atomics and other fragment data types/layouts. It checks load
bounds, roles, declaration/load order, launch participation and overlapping
output stores. This is a small intrinsic backend path, not a tiled GEMM library.

Packing follows NVIDIA's
[PTX 8.8 fragment specification](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float).
The separately specified checker decodes the emitted instructions and reconstructs
logical matrices using inverse ownership maps. It detects swapped halfwords,
bad accumulator stores, duplicate output owners and incorrectly typed packed
registers. Its numerical claims cover small integer-valued FP16 inputs whose
dot products are exact in FP32. It does not simulate tensor-core rounding,
subnormals, nonfinite values, scheduling or performance.

## Current qualification and review

The default backend has hardware results for 15 corpus kernels and all
nine shipped examples across batches 064/068/069/071. All nine examples passed
069 bit-for-bit, including signed zeros and the i64 word-transport check. The
three tensor examples cover padded output and K=32 accumulation with small exact
values. These results establish those inputs and launches, not all-domain
rounding, throughput or freedom from races.

Fable's encompassing review 072 of extracted archive `2c5d165` reproduced
381 tests, 440 default and rollback outputs, and all 24 hardware PTX payloads.
It found no implementation blocker and required correction of stale status
text. Opus review 073 of the documentation-only successor `96c6d3a` closed
that correction with no blocker or required correction. This release preserves
all executable files from the reviewed candidate.
The backend modules remain byte-identical to reviewed `34129e9`. Most of the
429 changed corpus outputs have not run on hardware. Earlier 055/056 results
remain historical evidence, as labeled in VALIDATION.md.
Blake approved adoption on 2026-10-25. All hosted allowances are consumed.

BF16/FP8/FP4 and TMA remain unsupported. TMA needs a descriptor ABI and lifetime
contract, not just an instruction spelling. Adding extra formats without sound
conversion, packing and validation would weaken the submission.
