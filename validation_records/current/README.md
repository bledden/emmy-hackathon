# Current compiler: recorded hardware evidence

```sh
python validation_records/current/check.py
python -m unittest discover -s validation_records/current -p 'test_check.py' -v
```

Install the root requirements first. The replay runs offline, using only local
files, NumPy and the compiler. No CUDA, GPU, organizer checkout or service access
is required. It sends **zero requests**, has no execution flag and cannot replay
a live batch. `--skip-compile` skips only the 24 local compiler invocations.

## What is included

| Batch | Cases | Compile / run responses | Values rechecked | Purpose |
|---|---:|---:|---:|---|
| 064 | 5 | 12 / 24 | 2,206,032 | Current compact backend vs preceding backend; vector and nvcc controls for main matmul |
| 068 | 2 | 4 / 8 | 30,720 | Rectangular and odd-tail transfer cases |
| 069 | 9 | 9 / 9 | 760 | All shipped scalar, copy and tensor examples |
| 071 | 8 | 8 / 8 | 5,942 | Signed floor/remainder, f16/i64 sums, copies, runtime loops and branches |
| Total | **24** | **33 / 49** | **2,243,454** | **82 saved responses**, including repeated timing runs |

Every case includes IR, original metadata, expected arrays and the PTX actually
sent. Explicit input arrays are included where the original packet had them;
timing cases preserve the exact seed-based input descriptors in metadata and
request bodies. The saved responses and payload files retain their original bytes.
`origins.json` records their original relative locations and SHA-256 hashes.

`plan.json` files preserve ordered requests, launch bodies and case names. The
publication plan omits the original machine's approval-state path, runner setup
and stale approval prose; it is **not a runnable authorization**. Typed-batch
bodies come from the separate pre-run Opus construction. The original manifest
hashes remain recorded for provenance. No live transport or approval state is bundled.

## What the checker verifies

1. The packet file set/hashes and the 351 preserved implementation/evidence hashes.
2. All 24 current CLI outputs equal the tested current PTX byte for byte.
3. Every response matches its planned request, compile-to-run binary binding,
   `sm_120` compile architecture, entry name and echoed launch configuration.
4. Output names, shapes, dtypes, truncation, numeric representation and values.
   Performance uses original tolerances. All 069/071 results additionally require
   bit-exact equality, including signed zero. i64 is reconstructed with integer
   arithmetic from low/high i32 words; values never round through float64.
5. Timing sample order, register reports, means and same-batch speedup ratios.

The JSON output is recomputed from receipts, not copied from a saved summary.
The compiler does not read any references or responses. The checker can emit
its result to a file **outside this sealed folder** if desired:

```sh
python validation_records/current/check.py > /tmp/emmy-current-results.json
```

## Main matmul comparison

Batch 064 recorded these request medians in microseconds:

| Variant | Median 1 | Median 2 | Mean |
|---|---:|---:|---:|
| Preceding backend | 31.360 | 31.184 | 31.272 |
| Current compact backend | 15.888 | 15.072 | 15.480 |
| Vector variant | 15.392 | 15.376 | 15.384 |
| nvcc on organizer CUDA | 15.056 | 15.008 | 15.032 |

`31.272 / 15.480 = 2.020155…`; current/nvcc time is `1.029803…`.
The original CUDA and build description are in `064/payloads/m01_matmul_512/`
and `064/nvcc-build.json`. They document nvcc 12.9.86, `-O3 --fmad=true`, no
fast-math, and PTX 8.8 for the nvcc baseline. The replay does not rebuild nvcc.

Two medians per variant on a shared GPU do not establish a confidence interval.
Compact and vector are not distinguishable from these samples. This is the same
supplied schedule, not a cuBLAS or peak-performance comparison. Keep ratios within
their batch. Small-K, split-K and the tail case establish no reliable speed gain.

## Boundaries and provenance

These records cover 15 corpus fixtures and nine examples, not all 440 offline
inputs or hidden programs. Tensor inputs are small exact integers. Single-run
copy/synchronization cases do not prove race freedom. The service's direct i64
JSON path is not validated; the unchanged kernel used a byte-preserving i32 wire
representation. Single-iteration 069/071 times are not performance evidence.

This is a **new Codex-authored checker**, with adversarial tests, not a new Opus or
Fable review. It reproduces results already independently decoded during development.
Original compiler reviews apply only through unchanged source identity. Hashes
check local integrity; neither they nor these unsigned HTTP records authenticate
the hardware service independently. See [provenance](../../docs/PROVENANCE.md).
