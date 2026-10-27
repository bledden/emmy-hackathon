"""Prepare erf/GELU fixtures and PTX locally; no HTTP client or execution batch."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
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

OPS = ('erf', 'gelu', 'gelu_tanh')
RTOL, ATOL, TAIL_RTOL = 3e-6, 2e-7, 1.5e-5
TAIL_WINDOWS = {'gelu': [-10, -2], 'gelu_tanh': [-6, -2]}


def reference(op, values):
    x = np.asarray(values, dtype=np.float32).astype(np.float64)
    with np.errstate(all='ignore'):
        if op == 'erf':
            result = np.vectorize(math.erf, otypes=[float])(x)
        elif op == 'gelu':
            result = 0.5 * x * np.vectorize(math.erfc, otypes=[float])(-x / math.sqrt(2))
        else:
            y = 2 * math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)
            e = np.exp(-np.abs(y))
            result = np.where(y < 0, x * e / (1 + e), x / (1 + e))
        return result.astype(np.float32)


def edges(op, nonfinite=False):
    small = np.nextafter(np.float32(0), np.float32(1))
    tiny, maximum = np.finfo(np.float32).tiny, np.finfo(np.float32).max
    values = [-maximum, -1e20, -20, -14.5, -14.2, -14, -13.2, -13, -11.6,
              -11, -10, -9, -8.5, -8, -6, -5.5, -5, -4, -3, -2, -1,
              -tiny, -small, -0., 0., small, tiny, 1, 2, 3, 4, 6, 10, 1e20, maximum]
    threshold = math.sqrt(0.5) if op == 'gelu' else 0.5
    for sign in (-1, 1):
        t = np.float32(sign * threshold)
        values += [np.nextafter(t, np.float32(-np.inf)), t, np.nextafter(t, np.float32(np.inf))]
    if nonfinite:
        values += [-np.inf, np.inf, np.nan]
    return np.asarray(values, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cases = []
    for op in OPS:
        for kind in ('random', 'edges', 'bits'):
            count, size = (128, 128) if kind == 'bits' else (257, 274)
            graph, meta, node, tile = template(a.upstream, size=size, cells=count)
            meta['rtol'], meta['atol'] = RTOL, ATOL
            meta['kernels'][0]['zero_outputs'] = ['add']
            if kind == 'random':
                scale = 3.0 if op == 'erf' else 4.0
                x = np.asarray(inputs(meta)['x0'][:count] * np.float32(scale), dtype=np.float32)
                body = [read(('x',), 'x0', (Var('a0'),), F32),
                        Let('scale', Literal(scale), F32),
                        Assign('x', ElementwiseImpl('multiply'), ('x', 'scale'), F32)]
            else:
                x = np.resize(edges(op, kind == 'bits'), count)
                body = [Let('x', select_literal(edges(op, kind == 'bits')), F32)]
            destination = 'result' if kind == 'edges' else 'x'
            body += [Assign(destination, ElementwiseImpl(op), ('x',), F32),
                     write('add', Var('a0'), (destination,))]
            expected = np.zeros(size, dtype=np.float32)
            expected[:count] = reference(op, x)
            name = op + '_' + kind
            emit(a.out, name, graph, meta, node, tile, body, expected)
            folder = a.out / name
            np.save(folder / 'input.npy', x)
            ir = next(folder.glob('*.ir.json'))
            ptx = subprocess.check_output([sys.executable, a.compiler.resolve(), ir.resolve()])
            (folder / 'kernel.ptx').write_bytes(ptx)
            cases.append({'name': name, 'operation': op, 'kind': kind,
                          'computed_outputs': count, 'untouched_zero_halo': size-count,
                          'tail_window': TAIL_WINDOWS.get(op),
                          'future_transport': 'i32 bit readback' if kind == 'bits' else 'f32'})
    files = {str(f.relative_to(a.out)): hashlib.sha256(f.read_bytes()).hexdigest()
             for f in sorted(a.out.rglob('*')) if f.is_file()}
    sources = {str(f.relative_to(a.compiler.parent)): hashlib.sha256(f.read_bytes()).hexdigest()
               for f in sorted(a.compiler.parent.rglob('*.py'))}
    manifest = {'state': 'CPU preparation only; NOT a hosted execution authorization',
                'files': files, 'sources': sources, 'cases': cases,
                'rtol': RTOL, 'atol': ATOL, 'tail_rtol': TAIL_RTOL, 'tail_atol': 0,
                'nan_checks': 'classification only, not sign or payload',
                'subnormal_ulp_budget': None,
                'boundary_note': 'Finite-input exact underflow zeros are unresolved hardware checks; do not widen after failure.'}
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Prepared 9 fixtures locally. No network requests or GPU execution.')


if __name__ == '__main__':
    main()
