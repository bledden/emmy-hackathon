"""GPU i32 readback: exact non-NaN bits for rounding and f32 where."""
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
from make_selection_variants import OPS, literal_body, reference


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
        node.op = replace(node.op, name='probe_selection_' + op, body=(replace(tile, body=body),))
        k = m['kernels'][0]
        k['entry'] = node.op.name
        ir = folder / (k['entry'] + '.ir.json')
        ir.write_text(json.dumps(g.to_dict(), default=repr))
        (folder / 'meta.json').write_text(json.dumps(m))
        ptx = subprocess.check_output([sys.executable, compiler, ir], text=True)
        (folder / 'kernel.ptx').write_text(ptx)
        arrays = [np.resize(x, 128) for x in values]
        expected = reference(op, arrays)
        np.savez(folder / 'expected.npz', add=expected,
                 **{f'input{i}': x for i, x in enumerate(arrays)})
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
        non_nan = ~np.isnan(expected)
        checks = {'nan_class': bool(np.array_equal(np.isnan(got), np.isnan(expected))),
                  'exact_non_nan_bits': bool(np.array_equal(words[non_nan], expected.view(np.int32)[non_nan]))}
        row = {'case': op, 'pass': all(checks.values()), 'checks': checks,
               'output_count': len(got), 'distinct_inputs': len(values[0]),
               'zero_count': int((expected == 0).sum()),
               'nan_count': int(np.isnan(expected).sum()),
               'infinity_count': int(np.isinf(expected).sum()),
               'subnormal_count': int(((np.abs(expected) < np.finfo(np.float32).tiny) & (expected != 0)).sum())}
        summary.append(row)
        print(json.dumps(row), flush=True)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    assert all(sha(compiler.parent / f) == h for f, h in hashes.items())
    return int(not all(row['pass'] for row in summary))


if __name__ == '__main__':
    raise SystemExit(main())
