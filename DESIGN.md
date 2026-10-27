# How the compiler works

For current measured results and source identity, see [VALIDATION.md](VALIDATION.md)
and [provenance](docs/PROVENANCE.md). Numbered reviews and earlier commit IDs below
are historical development references, not external certifications or prerequisites.

This project implements the hackathon's Kernel IR → PTX translation in Python.
Emmy 0.3.11 loads the typed IR through `Graph.from_dict`; it does not render or
compile the output. The entry point is `python compile.py <kernel>.ir.json`.
Successful compilation writes PTX to stdout. Unsupported semantics raise a
clear error rather than returning a partially generated kernel.

## A short code walkthrough

1. **Load and bind.** `compiler.compile_file` reads the typed graph and adjacent
   metadata, identifies the one kernel, and checks which launch describes it.
   It resolves the tolerance-dependent policy before creating an emitter.
2. **Establish the machine model.** `Emitter.compile` calls the launch helpers
   to validate geometry, bind buffer parameters and derive each thread's logical
   coordinates. Register allocation and all proof facts belong to this emitter.
3. **Translate statements.** `Emitter.body` walks the supplied schedule.
   `Emitter.statement` dispatches to named handlers: loads/stores, assignments,
   loops, conditionals and collectives. Handlers preserve types and lexical
   scope. Pure IR analyses and participation checks justify synchronization;
   they refuse unsupported cases rather than assuming every thread participates.
4. **Lower arithmetic and optimize.** Arithmetic delegates to typed numerical
   functions. Final assembly optionally contracts eligible accumulation pairs
   and reuses integer address expressions. The reuse pass tracks value identity,
   invalidates overwritten results, and clears facts at control-flow boundaries.
5. **Return one kernel.** The assembler emits the declarations, instructions and
   header. The CLI writes the completed ASCII PTX, or reports an error.

For the demo, show the small `Emitter.compile` driver, then follow one `Load`
through `load_statement` and one repeated division through `index_reuse.py`.
The contribution is a direct, typed backend with conservative synchronization
proofs and measured address-arithmetic reuse. It preserves Emmy's schedule;
it does not search for a new tiling or claim to outperform nvcc.

## Readability refactor boundaries

The entry point, pure analyses, binary32 formulas and launch assembly were
separated by responsibility. The large statement switch now delegates its
long cases to named handlers in the same state-owning class. Numerical formulas,
proof predicates, register allocation order and optimization order are preserved.
No generic pass framework or mutable state forwarding layer was added.

The emitter remains the largest module because scope, memory and synchronization
share state. Moving those fields into separate owners would be a semantic
redesign, not needed to explain the compilation path. Tests and byte comparisons
are supplemented by an AST comparison of extracted executable bodies; unchanged
outputs on a finite corpus alone would not establish equivalence on unseen IR.

## Experimental short shared-memory trees

`short_tree=True` selects a narrowly guarded alternative implementation of
`TreeHalve`. It does not change the IR schedule, kernel ABI, grid, or block.
Eligibility requires a flat tree of length 4, 8, 16, or 32; distinct allocated
shared buffers; an immediately preceding uncounted full-CTA barrier zero; and
a combine closed over its state, partners, and locally defined temporaries.
The original coordinate/writer, scope, and collective-uniformity checks still
run. A combine depending on another per-thread value cannot migrate safely and
falls back. Named/counted barriers and segmented trees also fall back.

The first physical warp participates in every shuffle. Only the first `length`
lanes load shared slots; the remaining lanes receive unused zero values. At
each descending distance all partner components are captured, then only lanes
below that distance perform the original ordered assignments. No reassociation
or new floating-point approximation is introduced by this pass. The existing
FMA policy remains a separate subsequent transformation.

All shuffle instructions use the full physical-warp mask. The branch selecting
`tid.x < 32` is explicitly `bra.uni`: its condition is equal within each warp,
including warps that skip the entire region. This fact prevents the tested
assembler from emitting additional divergent-collective paths. The per-level
merge conditions are not marked uniform.

Lanes below `length / 2` write their final state back to their own slots. These
are exactly the slots the original tree writes, including intermediate values
that a later shared load could observe. Writing only the root would be wrong.
Other slots remain unchanged. No lane reads another lane's shared slot in the
new region: communication uses registers, so no extra shared-memory fence is
assumed from `shfl.sync`. One final full-CTA barrier publishes the writebacks
before every original consumer loads the root.

The test-only symbolic executor checks actual emitted instructions against an
ordered-halving oracle, all scratch slots, shuffle membership, data dependencies,
and a conservative shared-access race check. It also checks the real coupled
softmax combine. This is a bounded structural check, not a PTX/SASS proof or
a hardware floating-point emulator. Hardware qualification is still required.

## Experimental exact integer reuse

The opt-in `loop` policy first computes the full exact-reuse result and its
replacement report. It recognizes direct `bra` / `bra.uni` targets, including
predicated branches, and records each backward edge's textual span. It selects
the full result if an actual `div.s32` or `rem.s32` replacement occurs strictly
inside at least one such span. Otherwise it returns the original stream.
Reports distinguish considered edits from applied edits. Duplicate labels,
unresolved targets or unsupported branch spellings decline selection.

This gate is a structural scoping rule, not a profitability predictor or a control-flow dominance,
reachability, trip-count or performance proof. Full-pass correctness is
independent of selection. When selected, edits outside the span are retained;
this preserves the already measured full-pass PTX. No problem names, launch
dimensions, file paths, floating-point values or benchmark timings are queried.
The policy was developed after the broad screen failed and the loop mechanism
was investigated; it does not retroactively change that screen's acceptance.
The measured-flat m02 control remains selected by its load-staging div/rem,
not its FMA loop. No rule is added to exclude it after observing the result.
The CLI defaults to index `auto`. After resolving FMA eligibility,
auto delegates to `loop` only if FMA is enabled and short-tree is disabled;
otherwise it disables index reuse and records the reason in the optional report.
This is an evidence boundary, not a mathematical dependence on FMA. Explicit
`loop` and full mode keep their previous semantics. The Python API default is
still False. Hardware evidence covers exact corpus outputs under guarded FMA
with short-tree disabled, not every switch combination or hidden kernel.

The optional `index_reuse` pass runs on our own emitted instructions after the
existing FMA pass. It recognizes a small explicit set of integer operations.
Each expression key contains the exact opcode and ordered operand identities;
an input register receives a new identity whenever it is redefined. Result
holders are tracked separately, so an overwritten representative cannot be
used even if its expression remains in the table. A surviving duplicate result
can still be used. Inputs are resolved before assigning the destination, which
also permits safe in-place assignments.

Repeated expressions become `mov.b32` or `mov.b64` to the original destination.
No instruction is moved or removed, and floating-point operations and memory
accesses stay in place. Width-matched copies preserve register bits. The pass
does not infer nonnegativity, simplify ranges, reorder operands, propagate
memory values, or change the numerical/FMA policy.

Labels, predicated instructions, branches, barriers, shuffle/async operations
and unknown instruction forms clear all knowledge. An explicit table describes
known single-destination instructions; these invalidate integer destinations
but are not themselves reused. Floating loads retain integer facts only when
all destination registers are floating registers, including vector loads.
This is a restricted emitter pass, not a general PTX parser or verifier.

Signed division/remainder reuse requires a literal divisor in 1..2^31-1;
zero, negative and variable divisors are left alone and clear facts. The exact
signed opcode is retained, so the pass does not choose between floor and
truncation conventions for negative inputs. Literal shift counts must be 0..31.
The reuse pass does not choose these semantics; the emitter now corrects signed
index division before reuse, as described below.

An independent finite-width test executor checks transformed programs, register
overwrites, loops, memory traces, and both floor/truncation division models.
A separate evidence script audits every changed full-corpus instruction using
symbolic expression DAGs and fresh definitions. Neither is a proof of arbitrary
PTX behavior or of NVIDIA's final machine-code transformations. Target assembly
and subsequent numerical/performance qualification are separate gates.

## The contract comes first

The kernel body describes operations and their scheduled thread layout. The
adjacent `meta.json` supplies the entry's parameter order, buffer descriptions,
launch dimensions and dynamic shared-memory allocation. The emitter checks
these agree before using them. Reused entry names are resolved to the matching
launch site rather than blindly selecting the first metadata entry.

The supplied schedule is preserved. This is a backend, not a replacement
scheduler or an attempt to rediscover tiling. Output remains PTX 8.0 targeting
sm_89; hosted qualification assembled it for sm_120. No alternate compiler or
precomputed PTX participates in translation.

## Mapping work to threads

The prologue loads pointer parameters, converts them to global addresses, and
reads physical thread and CTA identifiers. It forms a global logical index
from the prescribed block size. A guard excludes excess logical threads and
entire unused CTAs where necessary. Tile coordinates are decoded from that
index using the declared dimensions; supported raster layouts adjust that
mapping explicitly.

Loop instructions preserve the provided iteration structure. An `Accum` is
initialized by its owning loop and updated at each fold. Explicit axes identify
an outer owner across nested drains; legacy empty axes select the nearest loop.
A `seed=False` loop updates existing state rather than resetting it.

## Registers have lexical scope

A name-to-register cache alone is insufficient: the same spelling may denote
different variables inside a loop or conditional. The emitter tracks visibility
separately from cached register allocation. Entering a scope snapshots visible
names; leaving it restores that visibility.

An explicit inner `Init` gets a fresh register and restores the outer binding
on exit. For example, an outer value of 7 and an inner `Init` of 2 must still
produce 7 after the inner block. Nested Let/Assign/Load shadows likewise allocate local destinations after
reading their operands. Expired sibling register caches do not impose a type on
a new binding. Direct owned-fold/axis redeclarations and collective-local
collisions remain refused. An ordinary `Assign` inside a loop cannot overwrite an
outer fold merely because both use the same name. Same-scope updates remain
supported.

Collective state names are intentional in-place updates. Their shuffled or
loaded partners and merge temporaries are scoped to the collective; they cannot
escape or overwrite a distinct outer binding. This distinction allows coupled
reductions while preventing accidental state aliasing.

## Memory and collective participation

Addresses use declared buffer shapes and integer index expressions. Grouped
loads and stores use vector instructions when alignment is proven; otherwise
the emitter uses scalar accesses. Shared-memory declarations honor the supplied
allocation, with checks for capacity. The supported asynchronous copy operations
map to PTX copy, commit and wait instructions. Atomic output updates follow the
IR; launch-time zeroing remains the runner's metadata responsibility.

A barrier or shuffle cannot safely execute with arbitrary subsets of its
required participants. The emitter checks the whole enclosing control flow.
Conditional collectives require predicates proven constant across a CTA.
Collective loops require a proven uniform trip count and non-wrapping index
progression. Unknown cases refuse compilation.

Shuffle masks derive from physical participation, not mutable IR names such
as `lane`. Partial warps must contain complete aligned shuffle groups. Shared
trees use a checked physical writer for each slot and broadcast the final
state. The supplied combine program handles scalar and coupled states alike.

## Numerical behavior and boundaries

Elementwise operations preserve their supported declared scalar widths; typed
f16/f32/f64/i32/i64 paths are described below. Numerically sensitive
activations received separate finite-value and bit-oriented boundary checks;
correct special-value classes do not imply a universal ULP guarantee. All
published qualification claims refer to concrete fixtures and tolerances.

Guarded FMA contraction runs on our own emitted instruction stream. It accepts
only adjacent, unpredicated `mul.rn.f32` and `add.rn.f32` with register operands
and a product temporary appearing exactly twice in the whole stream: once as
the multiply destination and once as an add input. Extra definitions or uses,
including an in-place multiply or a loop-carried temporary, prevent fusion.
The add destination may alias an input because FMA reads operands before writing.
No memory operation, label, barrier or other instruction is crossed. Register
declarations remain valid even when a product temporary becomes unused.

`EMMY_PTX_CONTRACT_FMA=auto` enables this only for additions originating in
an `Accum`, with explicit finite nonnegative metadata tolerances and positive
`atol`. The emitter records instruction positions before optimization; names and
labels cannot grant eligibility. Ordinary `Assign(add)` remains separately
rounded. Exact, relative-only or missing tolerances use the baseline. Mode `1`
retains broader adjacent multiply/add contraction with eligible metadata. The
CLI default is `auto`; mode `0` preserves separate rounding. The lower-level
Python API still defaults to `contract_fma=False`. Selection is independent of
problem names or measured speed.
One rounding replaces two, so cancellation, overflow and extreme-value outcomes
can change. Tolerance eligibility is not a proof of numerical acceptance.
Literal-bearing approximation formulas are outside this version's scope.
A legal typed pointwise witness gives 0 with separate rounding and -0.015625
with broad fusion at atol=rtol=1e-6. Auto now agrees with separate rounding there.
The public pointwise reference uses NumPy f32 arithmetic; public matmul references
use f64 products/accumulation. This supports the narrower policy, but neither
positive tolerance nor reduction provenance proves arbitrary output accuracy.
At that milestone all 185 retained default outputs stayed byte-identical after
this narrowing; later changes and current evidence are tracked in VALIDATION.md.

General integer expressions use finite-width arithmetic; global Tile decoding
uses the wide path described below when flattened coordinates require it. The pinned typed Expr evaluator
floors both `/` and `//`, and `%` follows the divisor's sign. PTX truncates. For
opposite signs and nonzero remainder the emitter therefore changes q to q-1,
or remainder to remainder+divisor. It elides that correction only where interval
facts prove matching signs. Constants, immutable tile coordinates, non-wrapping
induction ranges and integer Let values can contribute facts. Grouped raster
coordinates retain their numeric bounds because rasterization is a bijection
within the same dimensions; they remain excluded from physical-neighbour and
CTA-uniformity proofs. These are separate proof domains. Arithmetic whose
intermediate range may wrap produces no fact; bound-register writes invalidate
old facts, shadowing uses distinct registers and loop-axis facts expire on exit.
Known zero divisors are refused. If INT_MIN/-1 quotient overflow cannot be ruled
out the expression is refused. Runtime zero divisors are outside the valid domain;
no Python-style exception behavior is claimed. Other arithmetic remains finite
width, not an implementation of unbounded Python integers.

At that milestone all 185 retained outputs used identical PTX after this
correction. Three
legal typed witnesses with negative intermediates and in-bounds final addresses
previously produced 44/44/52 wrong outputs and now match the typed evaluator in
both modeled thread orders. This paragraph records that local milestone; see
VALIDATION.md for current independent review and bounded hardware evidence.

Conditional fold exports use definite-initialization analysis across general
statement sequences and nested branches, as described below. Partially initialized
paths refuse; no neutral result is invented. The compiler is not a complete IR
validator: orphan Accum updating an existing binding remains a malformed-input
limitation.

Small proven constant f32 powers use explicitly rounded multiplication chains.
Other f32 powers use the runtime lowering described below. Integer powers use
width-preserving modular arithmetic for the supported nonnegative domain.
None is a promise of bitwise equality with every library implementation.

Sine and cosine use a standard-library-only full-range lowering. The exact f32
significand is multiplied by the 256-bit fixed-point constant floor(2/pi*2^256).
The quadrant plus 64 fractional bits determine a remainder in [-pi/4,pi/4].
Truncation of the reciprocal contributes less than 2^-128 turns at FLT_MAX;
fraction extraction contributes less than 2^-64 turns. Small |x|<0.5 bypasses
reduction. f64 Taylor polynomials then produce a single f32 result, with explicit
sign restoration and nonfinite classification. No libdevice, alternate compiler,
or source-rendering call participates in translation. This implementation uses
f64 internal registers; a separate direct f64 lowering supports f64 inputs and
outputs. Its sampled
accuracy evidence is not a universal correctly-rounded proof. Device numerical
accuracy and throughput remain unmeasured for this new path.

Grouped rasterization may cross physical CTA boundaries only for independent
scalar work. Kernels containing shared memory, asynchronous copies, barriers or
collectives retain the whole-CTA constraint. Tests execute the emitted integer
mapping over incomplete groups, both orientations and outer/inner dimensions;
a generated asymmetric attention case exercises the newly accepted mapping.
These paths have local model and assembly evidence. The current independent
review and exact hardware-qualified witnesses are listed in VALIDATION.md.

## How changes earn their place

Local checks establish source identity, scope behavior, metadata contracts and
exact emitted PTX. GPU qualification uses frozen inputs, references and runner
policy, with recorded requests and stop-on-failure behavior. A hardware pass
against an incorrect reference is not useful evidence: three such historical
scope fixtures were excluded and replaced with legal, independently reviewed
cases.

The optional launch-bound experiment changed only a `.maxntid` directive.
Across 185 artifacts, removing that line reproduced the qualified PTX exactly.
A bounded four-kernel comparison passed correctness but did not improve register
counts or indicative timings, so the change was rejected. See `VALIDATION.md`.

FMA passed a separate four-case performance gate and a consolidated nine-case
correctness qualification. The narrowed version preserves all four measured
PTX files exactly, so narrowing did not require another timing campaign. It is
integrated initially as an opt-in feature. After Opus independently verified the
evidence, the eligibility rule was tightened to require positive absolute
tolerance and the CLI default became `auto`. Both retained 185-artifact indexes
were checked again offline; their PTX bytes did not change.

The resulting claim is deliberately limited: a reproducible direct emitter,
a broad retained fixture set, explicit refusal boundaries, and evidence for
its implemented behavior. Hidden-test success, a leaderboard position and an
nvcc performance comparison have not been established.

## General conditional merges and proven uniform aliases

A sequence-level analysis tracks definitely initialized fold results. It combines
nested conditional paths by intersection (or the selected literal arm), recognizes
zero-trip seeded loops, and removes ordinary declarations from fold provenance.
Each branch is emitted under its actual lexical scope; finalized results copy to
merge registers before exit. No prebound placeholder can make an early read legal.
Outer bindings retain ordinary shadow semantics, including inside nested branches.

Let uniformity is proved from current register bindings and immutable CTA-wide
coordinates. A loop index is uniform only if its start and step are uniform within the
relevant group; equal trip counts alone are insufficient. Accum and collective mutations
kill carried facts before the loop, including predicates ahead of the mutation
on subsequent iterations. Register writes kill facts; shadow bindings do not
transfer their facts to an outer binding. Read-only global loads can now carry group agreement as described below.

## Runtime power fallback

Small proven integer exponents retain the previous rounded multiplication chain.
All other f32 powers use our own scalar PTX lowering in power.py. Bitwise dispatch
handles zero, infinities, NaNs, negative bases and exponent parity before numerical
evaluation. A subnormal base is normalized exactly by multiplying by 2**24. The
mantissa is reduced around one, avoiding cancellation for x close to one.
The logarithm uses the atanh series through degree31; exp uses a degree18 Taylor
polynomial after splitting y*log2(x) into an integer exponent and a fractional
part. f64 intermediates protect the f32 output accuracy. This is an accuracy-first
fallback, not a performance claim or an exhaustive correct-rounding guarantee.
No libdevice call or existing compiler performs the translation.


## Typed scalar storage and computation

Buffer element width determines byte offsets, vector alignment, shared allocation,
and memory instructions. Values carry their PTX kind independently of their buffer.
Loads and writes make dtype conversions explicit; a double or integer is never
silently routed through f32. Integer arithmetic wraps at its declared width.
Signed elementwise floor division/remainder correct truncated PTX division and
handle MIN/-1 without executing its undefined quotient. Index-expression range
proofs remain a separate, stricter domain. Shift counts are clipped before an
s64-to-s32 conversion so large counts cannot wrap into a small shift.

Half arithmetic widens exact operands and rounds each operation back to half.
Division, sqrt and reciprocal use f64 intermediates. A collective uses its
explicit state dtype and its unstamped combine Assigns inherit that dtype; an
explicit Assign dtype still takes precedence. Sixty-four-bit shuffles move both
32-bit halves; half shuffles pack into a 32-bit register and unpack afterwards.
The optional short-tree optimization remains f32-only, with ordinary typed
TreeHalve as the fallback.

Conditional exports inspect each arm's finalized visible binding, not every
same-spelled Accum nested within it. Only reachable arms constrain the exported
type when the predicate is a literal. Literal truth uses that literal's actual
precision, including half underflow and signed integer wrap.

The pinned generator sometimes declares f32 reduction accumulators for i64/f64
inputs, and half arithmetic in output epilogues. The compiler obeys those IR
choices. Its separate test records retain discrepancies against high-level
references; an IR-specific oracle is not presented as a hidden-evaluator pass.


## Direct binary64 math

The scalar dispatcher selects the f64 math lowering before the f32 paths.
Exp range reduction uses split ln(2), a degree-20 polynomial and two normal
power-of-two scales; the last scale performs subnormal rounding. Log normalizes
the significand exactly (including subnormals) and evaluates the atanh series
through degree31. Pow retains the low logarithm term and the rounded product's
FMA residual before exp2. That residual is essential at the largest finite
result and half-minimum-subnormal boundaries. Special bases/exponents and odd
integer powers are handled explicitly without an overflowing integer cast.

Sigmoid/SiLU/softplus use stable exponential branches; tanh has a small-input
polynomial to avoid cancellation. Erf uses its power series below 1.5 and an
erfc continued fraction above it. GELU retains the negative erfc tail instead
of subtracting nearly equal values. Formula references: NIST DLMF
[7.6](https://dlmf.nist.gov/7.6) and [7.9](https://dlmf.nist.gov/7.9).

Binary64 trig uses a 1152-bit 2/pi constant and exact integer significand
multiplication. It extracts a quadrant and 128 fractional bits; complementing
those bits before conversion avoids cancellation near the next quadrant.
Degree21/22 sine/cosine polynomials evaluate the reduced interval. A separate
Decimal Gauss-Legendre calculation checks the stored constant in the tests.

These explicit f64 FMA instructions implement compensation, not contraction
of two user operations. Every other operation preserves its specified rounding
steps. No libdevice call, PTX cache, problem-name selection or external compiler
performs the translation. The math is accuracy-oriented; sampled relative
error checks are not proofs over all inputs or compositions, and device
execution/performance remain unqualified.


## Synchronization-domain proofs

Conditional and loop proofs distinguish a region containing only WarpShuffle
and warp=True Sync from one containing TreeHalve or cross-warp Sync. Every
ancestor uses that domain; a nested CTA barrier cannot hide inside an otherwise
warp-only branch. A physical warp index is warp-uniform. Immutable mixed-radix
coordinates are also warp-uniform when their boundaries cannot split a physical
warp; this requires a multiple-of-32 block for global coordinate alignment.
Raster overrides and coordinates written anywhere in the body are excluded.

Let aliases and loop induction values carry separate warp/CTA facts tied to
register lifetimes. Binding writes, Accum updates and backedge-carried collective
state invalidate these facts. The backedge also invalidates integer intervals,
so a step that starts constant but is later mutated is not accepted as static.

A positive singleton s32 interval proves a symbolic constant step. Collective
loops require either a common non-wrapping trip count over the whole interval,
or start/end/step values uniform in their synchronization domain. A widened
terminal-increment guard exits before wrap when no static bound suffices. Different warps may then take different numbers of iterations
for warp-only operations. Counted named barriers and named trees additionally
use the exact cohort and phase proofs described below.


## Exact counted named barriers

The emitter tracks exact s32 values by physical thread for immutable launch
coordinates and Let aliases. These are register-lifetime facts; memory loads
are never guessed, and binding/Accum/backedge writes invalidate them. Integer
expression evaluation follows signed32 wrap and floor division, declining
unsupported or undefined expressions. Global tile coordinates contribute exact
facts only when their mixed-radix period divides the physical block size.
The physical warp index uses floor(tid/32), including short final warps.

For a region containing only warp operations and named counted Syncs, a static
predicate partitions the current physical-thread set. Every partition must
contain whole warps. A counted Sync requires its declared count to equal the
current set; nested masks intersect instead of resetting to the whole block.
An unknown CTA-uniform predicate retains the potential set because all its
members enter or skip together. A statically dead branch has no phase.

Each barrier ID owns a fixed participant set between proven full-CTA rendezvous. Every enclosing loop either has
a prior common-trip proof, or exact non-wrapping counts equal for that set.
Different groups may have different counts, but individual members of one group
may not. Thus each group sees the same lexical barrier sequence and iteration
phase; ID reuse cannot accidentally join disjoint or overlapping groups.
Named trees use the same exact-cohort proof; uncounted barriers retain the
stricter CTA control proof.

The barrier-lifetime analysis below permits ownership changes after proven
full-CTA rendezvous. Changes without that ordering still decline. It does not infer arbitrary runtime masks
or producer/consumer protocols. General coverage remains open; the proof is not
relaxed merely because the IR supplies a numeric count. PTX semantics reference:
[NVIDIA bar/barrier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier).


## Software shared-memory XOR layouts

B32/B64/B128 transform the flattened destination element index as
`e ^ (((e >> shift) & mask) << 3)`. The default shift is6 for every mode;
masks are1,3,7 respectively. A `<mode>@<shift>` spelling accepts shifts0..31.
The pinned IR specifies this mapping in element units. The buffer's actual
2/4/8-byte width is applied after the transform; it is not substituted into
a different hardware byte-address formula. Unknown modes refuse explicitly.

Each value of a Write uses its own logical index before the transform. A run
of1/2/4/8 aligned elements is provably contiguous if the shift is at least the
log2 of that width: the XOR source and destination fields then remain above
every varying bit. Only those runs use the ordinary vector-store path; other
runs compute each address separately, including crossings between chunks.
Atomic writes share the same address transform and retain typed atomic emission.

CpAsyncCopy is a contiguous-byte operation. Compose lane_index with its base,
flatten, transform that START once, and copy nbytes contiguously there. Alignment
proofs remain valid because XOR leaves element bits0..2 untouched. Existing
4-byte splitting and synchronous fallbacks preserve this start-plus-payload
contract. A consumer must use the matching physical addresses; an unaligned
copy is not a vector of separately swizzled elements.

Semantics were taken from pinned Emmy0.3.11 data-field comments for Write,
CpAsyncCopy and LDMATRIX_SWIZZLE_XOR. Parsing and PTX address emission are ours;
no renderer, existing translation helper or TMA operation is invoked. The
separate V70A/V70B coordinate layouts are implemented as described below.


## Auxiliary logical cells and physical synchronization

Pinned Tile semantics define physical block size as compute plus auxiliary
threads. Grid coverage, signed-index overflow checks, axis decode and cooperative
raster groups use compute cells. The emitted gid is cta*compute + tid%compute.
Raw tid, builtin warp/lane, full barriers and named populations stay physical.
Without auxiliary threads the prior PTX is byte-identical.

CTA-uniform axis facts use compute periods. Warp-uniform axis facts require both
physical and compute sizes to align to 32, so wrapping cannot split a warp's
coordinate group. Exact local-axis facts repeat over the compute period; builtin
warp does not wrap. The full mixed-radix numeric ranges remain unchanged.

An explicitly named TreeHalve is allowed inside an exact whole-warp cohort.
For each (segment, inner slot), select the lowest actual participating thread
with those immutable coordinates. Every required slot must have a representative.
Coalesce contiguous representative ids into predicate ranges; each tree level
intersects that predicate with its active segment range. This supports auxiliary
bands whose representatives do not begin at physical thread zero, and aliased
compute/auxiliary coordinates without duplicate writers. Counts and enclosing
loop phases use the same barrier checks as Sync. The short-tree optimization is
disabled for partial groups. Full-group trees preserve previous emission.

These proofs do not infer arbitrary runtime cohorts, prove shared memory was
populated on every path. Barrier-id changes additionally require the lifetime
proof below. Invalid missing
population/synchronization remains the IR producer's error; concrete witnesses
check those obligations with the existing race-checking CPU model.


## Barrier lifetimes across structured control flow

The emitter records a proof trace of barrier IDs, exact physical cohorts,
CTA-uniform alternatives and loop trip bounds. It does not insert barriers or
change PTX. A separate forward analysis tracks every possible preceding owner
of an ID since the last full-CTA rendezvous. Using an ID with a different possible
owner refuses. A valid full-CTA barrier clears prior ownership only after checking
its own ID for a possible collision. A named full-population barrier qualifies.

CTA-uniform branches join both outgoing states, retaining the incoming state
when one arm does not fence. Warp-divergent arms can execute concurrently, so
they retain shared ownership across arms; they cannot contain a full-CTA fence.
Known dead arms and statically zero-trip cohorts contribute no phases.

Loops retain entry state if they may execute zero times. One iteration checks
entry ownership; a second checks the backedge. This suffices because each ID's
transfer is either a constant owner set C, or its incoming set union C: barrier
events assign/clear, branch joins take union, and nested loops retain these forms.
The second valid transfer therefore has the same exit state. Static trip counts
are saturated at two, so compile work does not grow with a million-trip loop.
Unknown correlated runtime branches can still cause conservative refusals.

NVIDIA's PTX barrier contract reinitializes a completed barrier for reuse. The
full-CTA rendezvous proves that all earlier cohort phases have finished before
any subsequent cohort can enter. This analysis covers synchronous bar.sync only;
it does not infer an arrive/wait protocol or introduce asynchronous barriers.
Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier


## Uniform quotients of mixed-radix coordinates

An immutable coordinate is `(gid // S) % E`. Dividing by a positive static D is
constant within a synchronization group if the quotient-change boundaries S*D
and wrap boundaries S*E both align to group starts. E<=D makes the quotient zero
everywhere and needs no alignment. Group size is logical compute size for CTAs;
for warps it is32, with both physical and logical sizes aligned to32 so auxiliary
wrapping cannot split a warp. This is a sufficient proof, not a general symbolic
uniformity solver.

Decode descriptors are keyed by live s32 register identity. Direct s32 Let aliases
copy the descriptor before declaration changes bindings; ordinary overwrites,
Accum updates and loop backedges kill it. Lexical shadow exit restores only the
outer binding's facts. Raster-overridden axes and axes written anywhere in the
body receive no initial descriptor. General offsets and arbitrary arithmetic do
not inherit decode facts. Divisors need positive singleton s32 bounds.

The proof propagates through existing expression uniformity, so comparisons,
boolean combinations, ternaries and Let-bound quotient results can use it.
No instruction or barrier is inserted; already accepted PTX stays byte-identical.

### V70 coordinate permutations

V70A maps row bits into a four-element column group: output low row bits are
`r2 xor c3`, `r3 xor r4 xor c4`, `r0`, `r1`; higher row bits pass through.
The column group selects a stride of four times the slab row count. V70B
exchanges row bits 0–1 with column bits 3–4 and XORs the original row bits
into the relocated column group. Leading slab offsets are preserved. This
independent integer PTX lowering follows the public Volta layout contract,
not an Emmy renderer or another compiler. Public reference:
https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/layout/tensor_op_multiplicand_sm70.h

Write values are mapped independently. CpAsyncCopy maps only the payload
start. V70A preserves four consecutive elements, V70B eight; transformed
alignment must also account for the slab column stride. When a requested
16-byte instruction lacks the proof, aligned four-byte copies or scalar
loads/stores preserve the same contiguous byte transport.

### Exact facts across integer SSA arithmetic

Pure s32 Assign operations carry interval, uniformity and per-physical-thread
facts when their operands are known. Facts are captured before destination
shadowing and invalidated by the same binding/backedge rules as Let facts.
Shift participation uses a u32 count clamped to32, with signed right shift.
This extends proof coverage without weakening barrier membership/count checks.

### Runtime noncollective schedules and dormant metadata

Runtime s32 strides are accepted for loops without any nested collective. The
loop checks positive progress and tests the terminal addition in s64 before
updating its s32 induction variable. Unknown steps never grant interval facts
or inherited uniformity to that variable. The stricter collective trip proof
is unchanged. Pinned Window.base/bound are structurally retained metadata, not
execution bounds; they are ignored. Explicit local extent/start/end still apply.

### Exact tree coordinates and guarded runtime collective schedules

Runtime s32 steps may differ between synchronization groups when start, end
and step are uniform inside each group. Nonpositive-step and widened terminal
exits are uniform under the same proof. Carried updates still kill input facts
before checking the loop, and equal trip counts do not imply equal induction
values. A constant step also receives a terminal guard when necessary.

TreeHalve first retains the existing mixed-radix writer path. Otherwise exact
register facts supply each participant's segment and inner slot. The compiler
requires a present writer for every shared slot and selects the lowest physical
thread per slot; no memory values are guessed. The selected writers execute
each halving level, followed by the declared barrier, and every participant
loads the final value for its own validated inner slot. This supports aliases,
permutations and partial named cohorts without changing the mathematical tree.

### Integer power domain

For an explicit signed integer result, sign uses predicates on the original
operand; power copies both operands and performs modular repeated squaring.
Each multiplication keeps the low32/64 bits, so all intermediate truncations
are algebraically equivalent to exponentiation modulo2**width. A nonnegative
signed exponent terminates after at most31/63 bit steps. Input copies protect
either destination alias. A negative exponent exits with0 as an invalid-domain
guard, not as a specified integer reciprocal. Default Assign dtype inference
still follows Emmy's existing rule; this adds only explicit integer lowering.

### Boolean SSA and V70 shape validation

Boolean Assign results use canonical s32 zero/one. Source kinds drive comparisons:
exact f16-to-f32 widening, f64 for mixed floating/integer operands, and the wider
signed kind for integer pairs. Mixed integer/float comparisons follow numeric
promotion and can therefore round large integers, as NumPy does. Arguments and
uniformity facts are captured before rebinding the destination; the destination
receives only the new [0,1] interval and applicable group uniformity facts. Logical
truth tests preserve nonzero fractions and NaNs. No bool storage type is added.

The V70 mapper validates slab dimensions before emitting offsets. V70A permits
rows divisible by16 and columns divisible by4. V70B permits rows divisible by4
and columns divisible by32 or exactly8. Dimensions must be positive. This same
check covers Write and CpAsyncCopy, including scalar fallback. These conditions
exclude the collision/out-of-slab shapes found by independent review.

### Floating divmod

The emitter first handles zero divisors, NaNs and infinite dividends. For finite
magnitudes it doubles the divisor until near half the dividend, then walks the
binary scales downward, subtracting whenever the remainder is at least that
scale. Each accepted subtraction is between nearby floating values; scaling
by two is exact until the finite bounds stop the loop. The downward walk stops
below the original divisor. An infinite divisor bypasses the finite walk.
No flush-to-zero modifier is used. The two loops depend on exponent separation,
not the integer value of the quotient (at most a few thousand scale steps for
binary64 extremes).

The resulting fmod has the dividend's sign. A nonzero remainder with the wrong
sign gains one divisor, and the quotient loses one. A zero remainder receives
the divisor's sign. Quotient adjustment uses (a-fmod)/b, then floor and the
half-unit correction for rounding noise; a zero quotient gets the sign of a/b.
Half follows float32 intermediate arithmetic, matching the tested NumPy half
ufunc convention. Separate results are written only after every operand read,
so either source may alias the destination. This is a correctness-oriented
fallback; no speed advantage or universal GPU error claim is made.

### Wide global Tile coordinates

Use mul.wide.u32 for CTA×logical-threads, then an unsigned64 add of the local
thread. Widen if the launch could exceed2**31 threads or Tile size exceeds
2**31-1; a wide tail guard runs before all axis decoding and memory access.
Each axis division/remainder is unsigned64, then narrowed only after proving
0<=coordinate<extent<=2**31. CUDA grid.x is bounded to2**31-1 and blocks to1024,
so the flat product fits comfortably in64 bits. Grouped raster arithmetic also
uses64 bits; its group is capped to the grouped extent (an equivalent single
stripe when the original group is larger), avoiding oversized intermediates.
The existing uniformity/physical-thread facts now derive from these exact
bounded coordinates as well. The small-grid path is byte-identical.

### Agreement from read-only global inputs

The KernelOp/runner buffer ABI supplies distinct named argument buffers. An
input absent from this kernel's outputs and all Write/ZeroPrologue destinations
is immutable during the kernel. If every index expression is uniform within a
CTA (or physical warp), each corresponding vector element is loaded from the
same address and its converted register value agrees within that group. Capture
these facts before declaring destinations, including self-shadowing addresses.
No numeric value/range or exact per-thread mapping is inferred from input data.
Shared loads, mutable inputs and varying addresses do not receive this proof.
Register rebinding and loop-carried updates still kill the resulting facts.

### Experimental induction alignment proof

The backend extends power-of-two alignment facts to literal remainder and
immutable strided induction variables. For either floor or truncating remainder,
`a % b = a - q*b` preserves the common low zero bits of `a` and `b`. Signed
32-bit wrapping also preserves those bits. A progression begins at `start` and
adds a static positive `step`, so its proven divisor is
`gcd(multiple(start), step, 2**32)`; the start is never assumed to be zero.
Facts are keyed by current registers, killed on writes/rebinding and removed on
loop exit. Any body write to the induction variable disables the fact for the
whole loop, including uses before the write on a later iteration. Dynamic steps
and unknown bindings decline the proof. Existing swizzle alignment proofs still
apply. This changes memory instruction width only when both addresses prove
alignment; it does not change indices, scheduling, arithmetic or tolerances.
The earlier 046/055 batches covered the 21 outputs changed by `83348ec`;
review 056 concerned that earlier candidate. For the current compact backend,
hardware coverage is 15 corpus kernels plus nine examples, while most of the
429 changed corpus outputs have only offline evidence. See VALIDATION.md.

### Experimental invariant integer expressions

Before an innermost loop, evaluate repeated s32 division/remainder expressions
whose variable bindings are visible and never written anywhere in that loop.
The expression tree may contain only s32 literals, those bindings and wrapping
addition/subtraction/multiplication or division/remainder by a positive literal.
No loads, floating-point expressions or unknown divisors move. Division by zero
and the signed overflow pair are excluded, so evaluating this arithmetic on a
zero-trip path is safe. Expressions retain the emitter's existing signed
semantics; there is no reassociation or approximate division.

The lexical cache is scoped to that loop and discarded on exit. An inner loop's
preheader executes anew on each outer iteration, so outer induction/carried
values are recomputed. All nested declarations, conditional writes, reductions
and collective state destinations participate in the conservative write set.
We do not select by problem name, shape or observed speedup. This is a generic
motion rule, not a claim that motion improves every kernel: extra live values
can increase registers. Review 056 covered the earlier candidate; review 072 of
`2c5d165` found no implementation blocker. Opus 073 closed its status-text
correction in `96c6d3a`; Blake approved adoption on 2026-10-25.

The induction-register initialization remains immediately before the loop
header, after the hoisted expressions. Those expressions cannot reference the
new induction binding, so this ordering preserves semantics. Local sm_120
assembly showed that interposing invariant arithmetic between initialization
and the header prevented full unrolling of two short loops. Restoring this
canonical layout restored unrolling without kernel-specific rules or forced
unroll factors. The five-pair 046 comparison subsequently measured the
corrected layout; see VALIDATION for within-batch ratios and shared-service
limits. Hosted timings fit, but do not prove, the local assembly explanation.


### Explicit tensor fragments

`ptxgen/fragments.py` owns a small one-warp FP16/FP32 intrinsic path. Launch
preflight establishes a complete 32-thread CTA and a straight-line fragment-only
body. RegFragment declarations allocate packed `.b32` operands and zeroed FP32
accumulators. Global-mode LdmatrixLoad validates a complete rank-two tile and
packs lane-owned FP16 pairs; MmaSyncPtx emits the collective m16n8k16 instruction;
RegStore writes the four FP32 values each lane owns. Fragment bindings are kept
separate from ordinary scalar bindings, with role/readiness checks and explicit
store effects in read-only-buffer analysis.

Constant origins, padded strides and repeated K tiles are supported; shared
staging, swizzles, tails, nested control flow and mixed scalar bodies are refused.
This path does not recognize ordinary scalar matmul or change its precision.
The authored examples are typed Emmy IR, but no public/corpus input reaches this
path and upstream generation of the accepted form is unestablished. See
[FEATURES.md](FEATURES.md) for the precise contract and NVIDIA lane-map reference.


## Adopted compact CLI default (post-071)

The backend modules are unchanged from reviewed `34129e9`. The CLI supplies
`constant_division=True`, `unroll_loops=True`, `shared_loads='compact'` by default.
Explicit environment settings override each independently. The API retains all
previous defaults, and disabling the three options recovers previous CLI bytes.
Neither FMA eligibility nor automatic reuse selection changes. Exact-tolerance
inputs therefore retain strict floating-point behavior while the exact integer
and proved shared-memory transformations may run. Experimental short-tree mode
still requires an explicit request; no profitability is inferred for its combinations.

The sequence first resolves the existing numerical/reuse policy, then applies
the backend transformations. Constant division preserves signed truncation and
the existing floor correction. Shared-load proofs are live from pointer
definition and invalidated by pointer/dependency writes and control boundaries.
Unrolling retains original statement and accumulator ordering. These eligibility
rules do not inspect kernel names, references or recorded timings.

The explanation and example tools select the same explicit policies as the CLI.
The bundled tensor model now includes the exact three-operation update reviewed
in 067: signed shift, subtraction and unsigned-shift masking. This changes only
an offline example checker, not generated PTX. Tensor model scope stays limited
to the small exact-valued fixture domain.
