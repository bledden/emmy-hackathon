"""Independent f64 references for f32 sign and base-2/base-10 functions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from emmy.compiler.dtype import F32
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt.leaves import Assign, Let

from make_activation_variants import select_literal
from make_coverage_variants import emit, read, template, write
from make_variants import inputs

OPS = ('sign', 'exp2', 'log2', 'log10')
# Fixed before GPU validation. lg2's near-one error bound is absolute 2^-22.
TOLERANCES = {'sign': (0, 0), 'exp2': (1e-6, 1e-44),
              'log2': (1e-6, 3e-7), 'log10': (1e-6, 2e-7)}


def reference(op, values):
    with np.errstate(all='ignore'):
        return getattr(np, op)(np.asarray(values, dtype=np.float32).astype(np.float64)).astype(np.float32)


def edge_values(op, nonfinite=False):
    small = np.nextafter(np.float32(0), np.float32(1))
    tiny, maximum = np.finfo(np.float32).tiny, np.finfo(np.float32).max
    if op == 'sign':
        values = [-maximum, -1e20, -1, -0.5, -tiny, -small, -0.0,
                  0.0, small, tiny, 0.5, 1, 1e20, maximum]
    elif op == 'exp2':
        values = [-maximum, -1e20, -151, -150, -149.5, -149, -148,
                  -127, -126, -125.5, -100, -16, -1, -tiny, -small, -0.0,
                  0.0, small, tiny, 0.5, 1, 16, 100, 127,
                  np.nextafter(np.float32(128), np.float32(0))]
        for point in (-150, -149, -126, 0, 1):
            values.extend([np.nextafter(np.float32(point), np.float32(-np.inf)),
                           np.nextafter(np.float32(point), np.float32(np.inf))])
    else:
        values = [small, np.nextafter(small, np.float32(1)), tiny, 1e-30, 1e-20,
                  1e-10, 0.1, 0.5, 0.75, 1, 1.5, 2, 10, 100, 1e10, 1e20, maximum]
        for point in (0.5, 1, 2, 10):
            values.extend([np.nextafter(np.float32(point), np.float32(0)),
                           np.nextafter(np.float32(point), np.float32(np.inf))])
    if nonfinite:
        values += [-np.inf, np.inf, np.nan]
        if op == 'exp2':
            values += [128, np.nextafter(np.float32(128), np.float32(np.inf)), 129, 1e20, maximum]
        elif op in ('log2', 'log10'):
            values += [-0.0, 0.0, -small, -tiny, -1, -maximum]
    return np.asarray(values, dtype=np.float32)


def literal_body(op, nonfinite=False):
    values = edge_values(op, nonfinite)
    return [Let('x', select_literal(values), F32),
            Assign('x', ElementwiseImpl(op), ('x',), F32),
            write('add', Var('a0'), ('x',))], values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    count, size = 257, 274
    for op in OPS:
        for kind in ('random', 'edges'):
            g, m, node, tile = template(a.upstream, size=size, cells=count)
            m['kernels'][0]['zero_outputs'] = ['add']
            m['rtol'], m['atol'] = TOLERANCES[op]
            if kind == 'edges':
                body, values = literal_body(op)
                x = np.resize(values, count)
            else:
                x = inputs(m)['x0'][:count]
                body = [read(('x',), 'x0', (Var('a0'),), F32)]
                if op in ('log2', 'log10'):
                    body += [Assign('x', ElementwiseImpl('abs'), ('x',), F32),
                             Let('half', Literal(0.5), F32),
                             Assign('x', ElementwiseImpl('add'), ('x', 'half'), F32)]
                    x = np.asarray(np.abs(x) + np.float32(0.5), dtype=np.float32)
                elif op == 'exp2':
                    body += [Let('scale', Literal(32.0), F32),
                             Assign('x', ElementwiseImpl('multiply'), ('x', 'scale'), F32)]
                    x = np.asarray(x * np.float32(32), dtype=np.float32)
                body += [Assign('x', ElementwiseImpl(op), ('x',), F32),
                         write('add', Var('a0'), ('x',))]
            expected = np.zeros(size, dtype=np.float32)
            expected[:count] = reference(op, x)
            assert np.isfinite(expected).all(), (op, kind)
            emit(a.out, 'logexp_' + op + '_' + kind, g, m, node, tile, body, expected)


if __name__ == '__main__':
    main()
