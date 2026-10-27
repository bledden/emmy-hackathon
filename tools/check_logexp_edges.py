"""GPU i32 readback: sign bits, exp2 ULPs, and logarithm domain/accuracy."""
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

from make_coverage_variants import template
from make_logexp_variants import OPS, TOLERANCES, literal_body, reference


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
    summary = []
    for op in OPS:
        folder = a.out / op
        folder.mkdir()
        g, m, node, tile = template(a.upstream)
        body, values = literal_body(op, nonfinite=True)
        node.op = replace(node.op, name='probe_logexp_' + op, body=(replace(tile, body=body),))
        k = m['kernels'][0]
        k['entry'] = node.op.name
        ir = folder / (k['entry'] + '.ir.json')
        ir.write_text(json.dumps(g.to_dict(), default=repr))
        (folder / 'meta.json').write_text(json.dumps(m))
        ptx = subprocess.check_output([sys.executable, compiler, ir], text=True)
        (folder / 'kernel.ptx').write_text(ptx)
        x = np.resize(values, 128)
        expected = reference(op, x)
        np.savez(folder / 'expected.npz', add=expected, input=x)
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
        checks = {'nan_class': bool(np.array_equal(np.isnan(got), np.isnan(expected))),
                  'positive_inf': bool(np.array_equal(np.isposinf(got), np.isposinf(expected))),
                  'negative_inf': bool(np.array_equal(np.isneginf(got), np.isneginf(expected))),
                  'finite_class': bool(np.array_equal(np.isfinite(got), finite)),
                  'zero_bits': bool(np.array_equal(words[zero], expected.view(np.int32)[zero]))}
        max_ulp = None
        if op == 'sign':
            checks['exact_finite_bits'] = bool(np.array_equal(words[finite], expected.view(np.int32)[finite]))
        elif op == 'exp2':
            nonzero = finite & ~zero
            # Positive results have monotonically ordered bit patterns.
            distances = np.abs(words[nonzero].astype(np.int64) - expected.view(np.int32)[nonzero].astype(np.int64))
            max_ulp = int(distances.max(initial=0))
            checks['within_two_ulp'] = bool(np.all(distances <= 2))
            units = (x == 0)
            checks['zero_input_one_bits'] = bool(np.array_equal(words[units], expected.view(np.int32)[units]))
        else:
            rtol, atol = TOLERANCES[op]
            # f64 error arithmetic avoids accidental f32 underflow in checks.
            checks['finite_values'] = bool(np.allclose(got[finite].astype(np.float64),
                                                       expected[finite].astype(np.float64), rtol=rtol, atol=atol))
        row = {'case': op, 'pass': all(checks.values()), 'checks': checks,
               'output_count': len(got), 'input_table_entries': len(values),
               'max_finite_nonzero_ulp': max_ulp,
               'zero_count': int(zero.sum()), 'nan_count': int(np.isnan(expected).sum()),
               'infinity_count': int(np.isinf(expected).sum()),
               'subnormal_count': int((finite & (np.abs(expected) < np.finfo(np.float32).tiny) & ~zero).sum())}
        summary.append(row)
        print(json.dumps(row), flush=True)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    assert all(sha(compiler.parent / f) == h for f, h in hashes.items())
    return int(not all(row['pass'] for row in summary))


if __name__ == '__main__':
    raise SystemExit(main())
