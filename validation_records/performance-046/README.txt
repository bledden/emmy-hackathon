Recorded performance comparison — portable offline replay, batch 046

Run: python check.py
Dependency: NumPy (already included in the compiler's pinned dependencies).
The command sends no requests, imports no compiler/runner, and needs no GPU.
It reads all 30 saved HTTP records and checks their bodies, compile identity,
launch echoes, every returned value against the original fixture tolerance,
timing arithmetic and original gates. Its results must agree with the saved
independent Opus decoder, whose result is included unchanged.

This checker is newly authored by Codex. Agreement with the earlier decoder
is a consistency check, not a new independent peer review or a new GPU run.
The enclosing hashes detect accidental changes. They do not authenticate the
records as a statement signed by the organizers. Nothing in this packet is
authorization to contact the hosted runner. No request-sending code is included.

All five pairs from the experiment are included, including flat controls.
Baseline: 9b588d9 (alignment improvement); candidate: 83348ec (loop correction).
The later d993c88 tensor-feature candidate preserves these five PTX outputs.
The new tensor-core examples are NOT part of this timing experiment.
The currently adopted release remains 8f7f887 at preparation time.

Mean of two request medians per version, microseconds:
  m01:          39.536 -> 31.376, 1.2601x
  staged_rect:  39.464 -> 29.248, 1.3493x
  a_asymmetric: 23.616 -> 23.096, 1.0225x
  m_smallk:      3.504 ->  3.488, 1.0046x
  m02:           4.904 ->  4.904, 1.0000x (same PTX both versions)

These are within-batch ratios against our own preceding compiler. They are
not nvcc, cuBLAS, Triton, hidden-set results or statistical confidence intervals.
Each timing request used 2 warmups and 20 iterations. The two returned medians
are too few to estimate a reliable confidence interval or remove shared-service
noise. Do not multiply these ratios by wins from other batches.

Context that must accompany these results:
The preceding candidate 050a620 passed numerical checks but FAILED the same
performance gate in batch 045: asymmetric attention was 0.802x and small matmul
was 0.656x. It was not adopted. Local NVIDIA assembly showed that the new loop
layout inhibited unrolling. A general counter-initialization placement fix
restored local unrolling; corrected candidate 83348ec passed batch 046. This
is consistent with, but does not prove, the hosted machine-code explanation.
No kernel-name or measured-time exception was introduced. Baseline timings
drifted between batches (staged +5.07%, asymmetric attention -6.23%), so do not
attribute differences between separate batches to compiler changes alone.

Recorded numerical coverage: 1,157,456 returned elements over 20 runs of five
cases, not that many distinct kernels or inputs. All original tolerances pass.
The remaining changed outputs and new tensor feature need their own hardware
qualification; this packet says nothing about those cases or general MMA speed.

payloads/manifest.json is the ORIGINAL pre-run plan, kept byte-identical, so its
state still says NOT AUTHORIZED. Execution was subsequently authorized and
completed once; authorization and consumed-lock records remain in the research
workspace. No fresh execution is authorized by a copy of any historical record.

meta.json records original seeded inputs and launch metadata; expected.npz
contains the original references. This replay checks recorded inputs/outputs,
not the server's hidden RNG implementation. The payload's kernel.ir.json and
PTX are included for inspection. The compiler never reads this evidence bundle
to emit a kernel. packet-manifest.json binds every bundled artifact by SHA256.

This is a staging packet for release preparation, not yet reviewed or published.
