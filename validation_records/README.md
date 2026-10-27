# Inspect the evidence offline

## Current implementation: start here

```sh
python validation_records/current/check.py
```

The [current packet](current/README.md) contains **all 82 saved responses** from
064/068/069/071, their original payload artifacts and references. The checker
recomputes timings and output comparisons, checks preserved source hashes, and
regenerates all **24 tested PTX files** through the current CLI. It does not send
requests or execute GPU code. Both i64 examples use exact paired-i32 decoding.

The current packet supports the headline 2.020× main-matmul comparison and the
15-corpus-plus-nine-examples hardware statement. It does not establish coverage
of all 440 offline inputs, hidden sets, race freedom or general FP16 rounding.

## Historical packets

These remain byte-for-byte unchanged. They describe earlier backend versions,
not additional measurements of the current default:

```sh
python validation_records/performance-046/check.py
python validation_records/qualification-055/check.py
```

- [046](performance-046/README.txt): five earlier timing pairs and 30 responses.
- [055](qualification-055/README.txt): 19 earlier witnesses and 38 responses.

Their inner READMEs and manifests are sealed historical records. References to
pending reviews or absent absolute paths describe their preparation snapshots;
those paths are not required to replay them. See [history](../docs/history/README.md).

## What a replay proves

It checks the internal consistency and preserved bytes of supplied artifacts and
recomputes results from saved service responses. It is **not a new GPU run** or
an organizer-signed attestation. The compiler never reads these references or
responses during translation. See [provenance](../docs/PROVENANCE.md) for the AI
roles, internal commit identities and the new checker's review boundary.
