"""Exact f32 selection/rounding references, including ties and range edges."""
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

OPS = ('floor', 'ceil', 'trunc', 'rint', 'where')


def edge_values(nonfinite=False):
    small = np.nextafter(np.float32(0), np.float32(1))
    tiny, maximum = np.finfo(np.float32).tiny, np.finfo(np.float32).max
    values = [-maximum, -1e20, -2**31, -2**24, -2**23, -3.5, -2.5, -1.5,
              -1, -0.5, -0.0, 0.0, 0.5, 1, 1.5, 2.5, 3.5, 2**23, 2**24,
              2**31, 1e20, maximum, -tiny, -small, small, tiny]
    for point in (-2.5, -1.0, -0.5, 0.5, 1.0, 2.5, 2**23):
        values.extend([np.nextafter(np.float32(point), np.float32(-np.inf)),
                       np.nextafter(np.float32(point), np.float32(np.inf))])
    if nonfinite:
        values += [-np.inf, np.inf, np.nan]
    return np.asarray(values, dtype=np.float32)


def operands(nonfinite=False):
    # Different rotations put zeros/subnormals/nonfinite values in each arm
    # as well as in the condition. Selection must preserve the chosen bits.
    x = edge_values(nonfinite)
    values = (x, np.roll(x, 11), np.roll(x, 23))
    if nonfinite:
        # Explicitly select NaN from BOTH arms and ignore NaN in the other
        # arm. Rotations alone can accidentally leave every NaN unselected.
        extra = ([0.0, -0.0, np.nan, 1.0, -1.0, 0.0],
                 [7.0, np.nan, -0.0, np.nan, 3.0, np.nan],
                 [np.nan, np.nan, np.inf, 5.0, np.nan, -0.0])
        values = tuple(np.concatenate((array, np.asarray(tail, dtype=np.float32)))
                       for array, tail in zip(values, extra))
    return values


def reference(op, values):
    with np.errstate(all='ignore'):
        return np.asarray(np.where(*values) if op == 'where'
                          else getattr(np, op)(values[0]), dtype=np.float32)


def literal_body(op, nonfinite=False, destination='x'):
    names = ('x', 'y', 'z') if op == 'where' else ('x',)
    values = operands(nonfinite)[:len(names)]
    body = [Let(name, select_literal(array), F32) for name, array in zip(names, values)]
    body += [Assign(destination, ElementwiseImpl(op), names, F32),
             write('add', Var('a0'), (destination,))]
    return body, values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    count, size = 257, 274
    for op in OPS:
        kinds = ('random', 'edges_x', 'edges_y', 'edges_z') if op == 'where' else ('random', 'edges_x')
        for kind in kinds:
            g, m, node, tile = template(a.upstream, size=size, cells=count)
            m['kernels'][0]['zero_outputs'] = ['add']
            m['rtol'], m['atol'] = 0, 0
            if kind.startswith('edges'):
                body, arrays = literal_body(op, destination=kind[-1])
                arrays = [np.resize(x, count) for x in arrays]
            else:
                data = inputs(m)
                body = [read(('x',), 'x0', (Var('a0'),), F32)]
                arrays = [data['x0'][:count]]
                names = ('x',)
                if op == 'where':
                    body += [read(('y',), 'x1', (Var('a0'),), F32),
                             Let('z', Literal(-7.25), F32)]
                    arrays += [data['x1'][:count], np.full(count, -7.25, dtype=np.float32)]
                    names = ('x', 'y', 'z')
                body += [Assign('x', ElementwiseImpl(op), names, F32),
                         write('add', Var('a0'), ('x',))]
            expected = np.zeros(size, dtype=np.float32)
            expected[:count] = reference(op, arrays)
            emit(a.out, 'selection_' + op + '_' + kind, g, m, node, tile, body, expected)


if __name__ == '__main__':
    main()
