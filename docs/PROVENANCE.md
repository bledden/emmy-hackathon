# Source identity and AI-assisted development

## What is in this repository

The first publication commit,
[`bec6c644e6377a7e459c2b06c71c273045b3c1c3`](https://github.com/bledden/emmy-hackathon/commit/bec6c644e6377a7e459c2b06c71c273045b3c1c3),
is a clean source snapshot. Its Git tree is
`5776accd79b769839d2a38a450584444d4f7f13e`: all 357 original files and modes equal
the locally adopted release. Subsequent publication work improves documentation,
diagrams and portable evidence checks, without changing the compiler implementation.

The [source manifest](source-provenance.json) gives SHA-256 hashes for 351 preserved
files: compiler, original tests/tools, dependencies, examples and sealed historical
records. The six original narrative documents are excluded because publication
edits or relocates them. Added publication documents/checkers are also outside
that historical review boundary. Run:

```sh
python validation_records/current/check.py
```

This verifies those preserved file hashes, the new packet's integrity manifest,
all recorded results, and current default PTX identity for 24 hardware cases.
The replay checker is new Codex-authored publication tooling. Its tests exercise
corrupted records; it is not the original Opus decoder and has not received a new
independent agent review. Original implementation review conclusions apply by
unchanged bytes, not by pretending the new tooling was part of the old review.

## Internal IDs are archival identifiers

The following IDs refer to the development repository, **not reachable commits in
this clean publication history**. They are retained to explain old records, not
as links a reviewer must resolve:

| Internal ID | Meaning |
|---|---|
| `34129e9` | Reviewed backend modules used by current hardware batches |
| `2c5d165` | Reviewed CLI defaults, tools and regression tests |
| `96c6d3a` | Documentation correction; executable bytes unchanged |
| `b110448` | Adopted release; publication snapshot has the identical Git tree |

The adopted archive SHA-256 was
`eabfbfa9ece903ad707b8572e975d8f002ec980975977775560272b2f3128eaf`.
The archive itself and the private research workspace are not prerequisites.
The source manifest and reproduced PTX establish the local byte relationships;
they do not constitute organizer signatures or independent authentication of the
hardware service. Saved HTTP responses are unsigned participant-held records.

## Who did what

- **Blake:** directed scope, challenged performance against external baselines,
  constrained shared GPU use, approved bounded runs and selected the release.
- **Codex:** implementation, experiment preparation/execution, packaging and this
  portable replay checker.
- **Opus:** separately authored counterexamples, reference checks and receipt
  decoders; for example, exposed stale pointer facts in a load-merging pass.
- **Fable:** encompassing source/archive reviews at release checkpoints.

Opus and Fable are AI reviewers. Their findings are development evidence, not an
external audit, organizer endorsement or guarantee of correctness. Separate
probes and observable counterexamples helped, but agents may share assumptions.
The compiler's code and reproducible checks are what a judge should assess.

## Historical material

[Development history](history/README.md) preserves prior plans, failed candidates
and review chronology. Workspace paths in that material describe where work
happened; current commands resolve relative to this repository. Frozen historical
046/055 packets remain unchanged. No approval record or server runner is needed
to inspect the new current packet, and it performs no network calls.
