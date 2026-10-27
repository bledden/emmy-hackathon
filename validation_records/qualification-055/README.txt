Recorded qualification055 — OFFLINE REPLAY ONLY

From the repository root: python validation_records/qualification-055/check.py
Requires NumPy. No compiler, hosted runner, CUDA, GPU or network call is used.
The checker reads original payloads and all38 saved request/response records,
reconstructs bodies, checks exact typed transport and original tolerances, and
compares its summary with the included independently authored Opus decode.
It writes nothing. Packet hashes detect changed bytes, not third-party authenticity.

Blake approved core12 then synchronization26 only after core passed and its
receipts were checked. Both completed once:19compiles,19runs,11,262values,
18bit-exact cases. Rectangular matmul max error5.7220458984375e-6 at original
rtol=atol=1e-3. All compiles reported sm_120. No retries,0warmups,1iteration.
These timings are not benchmarks. Both allowances are consumed; this packet
contains no request-sending program and is not permission to run anything live.

Cases:3tensor examples,3copy-width witnesses,13synchronization witnesses.
MMA evidence covers small exact-integer inputs, not general FP16 rounding or
throughput. Each copy/sync case ran once, not a proof of race freedom. Halos and
tails rely on zeroed allocation. sync_12's all-zero hardware output cannot show
stores occurred; the independent CPU sentinel model did show256stores.

Source264f72a remains the executable base of this packaging candidate. All21
outputs changed by83348ec have hardware results across046and055; this is not
all431offline fixtures or all9examples. The6scalar/copy examples were not run.

payloads/*/manifest.json and receipts/*/approved-manifest.json are unchanged
historical records. Their absolute shared-state paths and tool hashes describe
the completed execution, not dependencies needed for this offline replay.
original-decode.json and PEER-REVIEW.txt are unchanged Opus artifacts. The
portable check.py is adapted from Codex's separate decoder and still needs
Fable's final code review; replay agreement is not a new independent decoder.
