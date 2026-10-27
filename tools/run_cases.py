"""Run official submit.py and retain its output plus compiler/source identities.

This calls the organizer's network runner. Run batches serially; timings are
indicative only. The evidence directory must be outside the submission repo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    upstream = args.upstream.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    source_hashes = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*.py"))}
    (args.out / "source-hashes.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    results = []
    for case in args.cases:
        folder = Path(case) if Path(case).is_dir() else upstream / "problems" / case
        stem = folder.name
        ptx_hashes = {}
        for ir in sorted(folder.glob("*.ir.json")):
            built = subprocess.run([sys.executable, str(root / "compile.py"), str(ir)],
                                   capture_output=True, text=True)
            if built.returncode == 0:
                output = args.out / (stem + "_" + ir.name.replace(".ir.json", ".ptx"))
                output.write_text(built.stdout)
                ptx_hashes[ir.name] = sha(output)
        command = [sys.executable, str(upstream / "submit.py"), str(folder.resolve()),
                   "--compiler", str(root / "compile.py")]
        timestamp = datetime.now(timezone.utc).isoformat()
        run = subprocess.run(command, capture_output=True, text=True, timeout=360)
        record = {"case": stem, "timestamp_utc": timestamp, "command": command,
                  "exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr,
                  "ptx_sha256": ptx_hashes, "layer": "run", "timing": "indicative, not benchmark"}
        (args.out / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
        results.append(record)
        print(stem, "exit", run.returncode, flush=True)
        print(run.stdout + run.stderr, end="", flush=True)
    (args.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return int(any(r["exit_code"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
