"""Probe NaN propagation using i32 readback of f32 output bits.

No NaN crosses the runner's JSON interface: the compiler receives typed literal
IR locally, and the runner returns integer output words from the same allocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import requests
from emmy.compiler.dtype import F32
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt.leaves import Assign, Let
from make_coverage_variants import template, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--expect', choices=['propagate', 'suppress'], required=True)
    p.add_argument('--server', default='http://runner.riftstack.ai:48701')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    compiler = a.compiler.resolve()
    hashes = {str(f.relative_to(compiler.parent)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in compiler.parent.rglob('*.py')}
    (a.out / 'source-hashes.json').write_text(json.dumps(hashes, indent=2))
    results = []
    cases = [('minimum', True), ('minimum', False), ('maximum', True),
             ('maximum', False), ('relu', True), ('finite_control', False)]
    for operation, left_nan in cases:
        g, m, node, tile = template(a.upstream)
        name = operation + ('_left' if left_nan else '_right')
        folder = a.out / name
        folder.mkdir()
        op = 'minimum' if operation == 'finite_control' else operation
        x = float('nan') if left_nan else 2.0
        y = 1.0 if left_nan or operation == 'finite_control' else float('nan')
        body = (Let('x', Literal(x), F32), Let('y', Literal(y), F32),
                Assign('result', ElementwiseImpl(op), ('x',) if op == 'relu' else ('x', 'y'), F32),
                write('add', Var('a0'), ('result',)))
        node.op = replace(node.op, name='probe_' + name, body=(replace(tile, body=body),))
        k = m['kernels'][0]
        k['entry'] = node.op.name
        ir = folder / (node.op.name + '.ir.json')
        ir.write_text(json.dumps(g.to_dict(), default=repr))
        (folder / 'meta.json').write_text(json.dumps(m))
        ptx = subprocess.check_output([sys.executable, str(compiler), str(ir)], text=True)
        (folder / 'kernel.ptx').write_text(ptx)
        def post(endpoint, payload):
            response = requests.post(a.server + endpoint, json=payload, timeout=(10, 120))
            response.raise_for_status()
            result = response.json()
            (folder / (endpoint[1:] + '.json')).write_text(json.dumps(
                {'request': payload, 'response': result}, indent=2))
            if not result.get('ok'):
                raise RuntimeError(result)
            return result
        compiled = post('/compile', {'ptx': ptx})
        # Same 4-byte allocation and pointer ABI; observe the bits as integers.
        run_args = [dict(arg, dtype='i32') if arg.get('output') else arg for arg in k['args']]
        run = post('/run', dict(cubin_id=compiled['cubin_id'], entry=k['entry'],
                               args=run_args, grid=k['grid'], block=k['block'],
                               smem_bytes=0, zero_outputs=[], warmup=0, iters=1, max_return=128))
        output = run['outputs']['add']
        assert not output.get('truncated')
        values = np.asarray(output['values'], dtype=np.int32).view(np.float32)
        if operation == 'finite_control':
            passed = bool(np.all(values == np.float32(1)))
        else:
            passed = bool(np.all(np.isnan(values))) if a.expect == 'propagate' else bool(np.all(np.isfinite(values)))
        row = dict(case=name, pass_expected_behavior=passed,
                   nan_count=int(np.isnan(values).sum()), output_count=int(values.size))
        results.append(row)
        print(json.dumps(row), flush=True)
    (a.out / 'summary.json').write_text(json.dumps(results, indent=2))
    assert all(hashlib.sha256((compiler.parent / f).read_bytes()).hexdigest() == h for f, h in hashes.items())
    return int(not all(r['pass_expected_behavior'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
