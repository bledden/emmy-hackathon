"""Compile offline and write an optimization sidecar, keeping stdout pure PTX."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptxgen.compiler import compile_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ir', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = []
    ptx = compile_file(args.ir, contract_fma='auto', index_reuse='auto',
                           constant_division=True, unroll_loops=True, shared_loads='compact', optimization_report=report)
    args.report.write_text(json.dumps({
        'description': 'Offline policy and transformation decisions; not measured speedup.',
        'ir_sha256': hashlib.sha256(args.ir.read_bytes()).hexdigest(),
        'meta_sha256': hashlib.sha256((args.ir.parent/'meta.json').read_bytes()).hexdigest(),
        'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(),
        'policy': {'contract_fma': 'auto', 'index_reuse': 'auto', 'short_tree': False,
                   'constant_division': True, 'unroll_loops': True, 'shared_loads': 'compact'},
        'decisions': report}, indent=2)+'\n')
    sys.stdout.write(ptx)


if __name__ == '__main__':
    main()
