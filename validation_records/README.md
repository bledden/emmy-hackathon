# Saved evidence, replayed offline

These folders contain recorded data, not a GPU runner. The compiler does not
read them when translating IR. Install the root requirements first, then run:

```sh
python validation_records/performance-046/check.py
python validation_records/qualification-055/check.py
```

- [Performance 046](performance-046/README.txt): all five comparison pairs,
  including the flat control, original payloads and all 30 saved responses.
  This compares two earlier scalar-backend versions, not tensor cores or nvcc.
- [Qualification 055](qualification-055/README.txt): all 19 correctness witnesses
  and 38 saved responses, including three tensor examples and the copy/sync
  cases that completed qualification. One iteration each; these are not benchmark results.

Both commands are local replays and perform zero network requests. The newly
packaged checkers are Codex-authored; each agrees with a separately authored,
unchanged Opus result. Fable's encompassing review 056 of extracted archive
`4ea1fd6` completed on 2026-10-24 with no blocker or required correction. Fable
read both checker programs, exercised 76 re-sealed mutations, and reproduced
349 tests, 431 outputs and all three portable commands offline. Those figures
describe the earlier archive, not the current default backend.
Replaying these packets adds no hardware evidence and is not another
independently authored decoder.
Hashes bind the recorded bytes; the organizer did not sign these records.

Both packets are preserved byte-for-byte from their preparation snapshots.
Their inner READMEs are covered by the unchanged packet-manifest.json hashes.
Statements there that review or later qualification is pending describe those
historical snapshots: qualification 055 and review 056 are now complete.
Current status is in the root VALIDATION.md. Blake adopted the compact backend
after Fable 072 and Opus 073 on 2026-10-25. These preserved packets still describe
earlier versions. Historical absolute paths inside original manifests need not
exist to replay saved data.
