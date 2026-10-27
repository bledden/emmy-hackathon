"""GPU i32 readback: classify nonfinite results, zero signs and subnormals."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import requests
from emmy.compiler.dtype import F32
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt.leaves import Assign, Let
from make_activation_variants import OPS, reference, select_literal
from make_coverage_variants import template, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--server', default='http://runner.riftstack.ai:48701')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    compiler = a.compiler.resolve()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {str(f.relative_to(compiler.parent)): sha(f) for f in compiler.parent.rglob('*.py')}
    (a.out / 'source-hashes.json').write_text(json.dumps(hashes, indent=2))
    tiny, smallest = np.finfo(np.float32).tiny, np.nextafter(np.float32(0), np.float32(1))
    values = np.asarray([-np.inf, -np.finfo(np.float32).max, -100, -20, -1,
                         -0.06250001, -0.0625, -1e-8, -tiny, -smallest, -0.0, 0.0,
                         smallest, tiny, 1e-8, 0.0625, 0.06250001,
                         np.nextafter(np.float32(1), np.float32(0)), 1,
                         np.nextafter(np.float32(1), np.float32(2)), 20, 100,
                         np.finfo(np.float32).max, np.inf, np.nan], dtype=np.float32)
    summary = []
    for op in OPS:
        folder = a.out / op
        folder.mkdir()
        g, m, node, tile = template(a.upstream)
        body = [Let('x', select_literal(values), F32),
                Assign('result', ElementwiseImpl(op), ('x',), F32),
                write('add', Var('a0'), ('result',))]
        node.op = replace(node.op, name='probe_activation_' + op, body=(replace(tile, body=body),))
        k = m['kernels'][0]
        k['entry'] = node.op.name
        ir = folder / (k['entry'] + '.ir.json')
        ir.write_text(json.dumps(g.to_dict(), default=repr))
        (folder / 'meta.json').write_text(json.dumps(m))
        ptx = subprocess.check_output([sys.executable, compiler, ir], text=True)
        (folder / 'kernel.ptx').write_text(ptx)
        expected = reference(op, np.resize(values, 128))
        np.savez(folder / 'expected.npz', add=expected, input=np.resize(values, 128))
        def post(endpoint, payload):
            response = requests.post(a.server + endpoint, json=payload, timeout=(10, 120))
            response.raise_for_status()
            result = response.json()
            (folder / (endpoint[1:] + '.json')).write_text(json.dumps({'request': payload, 'response': result}, indent=2))
            if not result.get('ok'):
                raise RuntimeError(result)
            return result
        compiled = post('/compile', {'ptx': ptx})
        run_args = [dict(arg, dtype='i32') if arg.get('output') else arg for arg in k['args']]
        run = post('/run', dict(cubin_id=compiled['cubin_id'], entry=k['entry'], args=run_args,
                               grid=k['grid'], block=k['block'], smem_bytes=0, zero_outputs=[],
                               warmup=0, iters=1, max_return=128))
        output = run['outputs']['add']
        assert not output.get('truncated')
        words = np.asarray(output['values'], dtype=np.int32)
        got = words.view(np.float32)
        finite = np.isfinite(expected)
        zero = finite & (expected == 0)
        subnormal = finite & (np.abs(expected) < tiny) & ~zero
        checks = {'nan_class': bool(np.array_equal(np.isnan(got), np.isnan(expected))),
                  'positive_inf': bool(np.array_equal(np.isposinf(got), np.isposinf(expected))),
                  'negative_inf': bool(np.array_equal(np.isneginf(got), np.isneginf(expected))),
                  'finite_values': bool(np.allclose(got[finite], expected[finite], rtol=3e-6, atol=2e-7)),
                  'zero_bits': bool(np.array_equal(words[zero], expected.view(np.int32)[zero]))}
        # exp-based sigmoid/silu may differ by one subnormal ulp. Other ops
        # here preserve or correctly round their subnormal outputs exactly.
        ulps = np.abs(words[subnormal].astype(np.int64) - expected.view(np.int32)[subnormal].astype(np.int64))
        checks['subnormal_bits'] = bool(np.all(ulps <= (1 if op in ('sigmoid', 'silu') else 0)))
        row = {'case': op, 'pass': all(checks.values()), 'checks': checks,
               'output_count': len(got), 'subnormal_count': int(subnormal.sum()),
               'nan_count': int(np.isnan(got).sum())}
        summary.append(row)
        print(json.dumps(row), flush=True)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    assert all(sha(compiler.parent / f) == h for f, h in hashes.items())
    return int(not all(row['pass'] for row in summary))


if __name__ == '__main__':
    raise SystemExit(main())
