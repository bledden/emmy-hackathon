"""Prepare softplus fixtures and PTX locally. This module makes no network calls."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from emmy.compiler.dtype import F32
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt.leaves import Assign, Let
from make_activation_variants import select_literal
from make_coverage_variants import emit, read, template, write
from make_variants import inputs

RTOL, ATOL, TAIL_RTOL = 1e-6, 2e-7, 1.2e-5


def reference(x):
    return np.logaddexp(0.0, np.asarray(x, dtype=np.float32).astype(np.float64)).astype(np.float32)


def edges(nonfinite=False):
    small = np.nextafter(np.float32(0), np.float32(1))
    tiny, maximum = np.finfo(np.float32).tiny, np.finfo(np.float32).max
    values = [-maximum, -1e20, -104, -103.9, -103.3, -100, -90, -87.5, -87,
              -80, -65, -50, -20, -17.5, -17, -10, -3.28, -2, -1, -0.5,
              -tiny, -small, -0.0, 0.0, small, tiny, 0.5, 1, 2, 10,
              17, 17.5, 20, 88.72, 100, 1e20, maximum]
    for point in (-104, -103.9, -17.33, -24 * np.log(2), 0, 17.33):
        values.extend([np.nextafter(np.float32(point), np.float32(-np.inf)),
                       np.float32(point),
                       np.nextafter(np.float32(point), np.float32(np.inf))])
    if nonfinite:
        values += [-np.inf, np.inf, np.nan]
    return np.asarray(values, dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    for kind in ('random', 'edges', 'bits'):
        count, size = (128, 128) if kind == 'bits' else (257, 274)
        g, m, node, tile = template(a.upstream, size=size, cells=count)
        m['rtol'], m['atol'] = RTOL, ATOL
        m['kernels'][0]['zero_outputs'] = ['add']
        if kind == 'random':
            x = np.asarray(inputs(m)['x0'][:count] * np.float32(8), dtype=np.float32)
            body = [read(('x',), 'x0', (Var('a0'),), F32),
                    Let('scale', Literal(8.0), F32),
                    Assign('x', ElementwiseImpl('multiply'), ('x', 'scale'), F32)]
        else:
            table = edges(kind == 'bits')
            x = np.resize(table, count)
            body = [Let('x', select_literal(table), F32)]
        # Exercise both in-place and distinct destination assignment.
        result = 'result' if kind == 'edges' else 'x'
        body += [Assign(result, ElementwiseImpl('softplus'), ('x',), F32),
                 write('add', Var('a0'), (result,))]
        expected = np.zeros(size, dtype=np.float32)
        expected[:count] = reference(x)
        name = 'softplus_' + kind
        emit(a.out, name, g, m, node, tile, body, expected)
        folder = a.out / name
        np.save(folder / 'input.npy', x)
        ir = next(folder.glob('*.ir.json'))
        ptx = subprocess.check_output([sys.executable, a.compiler.resolve(), ir.resolve()])
        (folder / 'kernel.ptx').write_bytes(ptx)
    files = {str(f.relative_to(a.out)): hashlib.sha256(f.read_bytes()).hexdigest()
             for f in sorted(a.out.rglob('*')) if f.is_file()}
    sources = {str(f.relative_to(a.compiler.parent)): hashlib.sha256(f.read_bytes()).hexdigest()
               for f in sorted(a.compiler.parent.rglob('*.py'))}
    (a.out / 'manifest.json').write_text(json.dumps({
        'state': 'prepared locally; NO hardware execution',
        'files': files, 'sources': sources, 'rtol': RTOL, 'atol': ATOL,
        'tail_rtol': TAIL_RTOL, 'cases': ['softplus_random', 'softplus_edges', 'softplus_bits'],
        'maximum_http_requests': 6, 'warmup': 0, 'iters': 1, 'automatic_retries': 0,
    }, indent=2) + '\n')
    print('Prepared 3 kernels locally; no network requests.')


if __name__ == '__main__':
    main()
