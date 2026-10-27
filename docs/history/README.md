# Development history

This folder preserves the pre-publication investigation. It is **historical**, not
setup instructions or the status of the current submission. Some internal commits,
workspace paths and numbered review packets mentioned in these documents are not
included in this clean repository history. They are not needed to run the compiler
or the packaged evidence checks.

- [Original validation narrative](validation-b110448.txt): earlier candidates,
  qualification, failed experiments and review chronology, preserved verbatim.
- [Backend investigation](backend-investigation.txt): the original experimental
  plan and its status updates, preserved verbatim. Later status takes precedence.

For current results use [VALIDATION.md](../../VALIDATION.md); for identity and AI
assistance use [PROVENANCE.md](../PROVENANCE.md). The older sealed 046/055 packets
remain in `validation_records/` for historical replay; their files are unchanged.

Later, an exact signed-division experiment reduced static instruction counts but
failed its main timing gates. It stayed outside the adopted implementation. No
speedup from that rejected experiment is claimed by this submission.
