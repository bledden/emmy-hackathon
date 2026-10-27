"""Independent NumPy expectations for six f32 unary operations."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from emmy.compiler.dtype import F32
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, TernaryExpr, Var
from emmy.compiler.ir.stmt.leaves import Assign, Let

from make_coverage_variants import binary, emit, lit, read, template, write
from make_variants import inputs

OPS = ('rsqrt', 'sigmoid', 'silu', 'tanh', 'log', 'square')


def reference(op, values):
    x = np.asarray(values, dtype=np.float32).astype(np.float64)
    with np.errstate(all='ignore'):
        if op == 'rsqrt':
            y = 1 / np.sqrt(x)
        elif op == 'log':
            y = np.log(x)
        elif op == 'square':
            y = x * x
        elif op == 'tanh':
            y = np.tanh(x)
        else:
            # NumPy mathematical reference, independent of PTX approximations.
            y = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
            if op == 'silu':
                y = x * y
        return y.astype(np.float32)


def select_literal(values):
    index = binary('%', Var('a0'), lit(len(values)))
    value = Literal(float(values[-1]), 'float')
    for i in reversed(range(len(values) - 1)):
        value = TernaryExpr(binary('==', index, lit(i)), Literal(float(values[i]), 'float'), value)
    return value


def edge_values(op):
    if op in ('rsqrt', 'log'):
        return np.asarray([np.nextafter(np.float32(0), np.float32(1)),
                           np.finfo(np.float32).tiny, *np.logspace(-35, 35, 19),
                           np.nextafter(np.float32(1), np.float32(0)), 1,
                           np.nextafter(np.float32(1), np.float32(2)),
                           np.finfo(np.float32).max], dtype=np.float32)
    values = [-1e18, -100, -20, -5, -1, -0.06250001, -0.0625, -0.06249999,
              -1e-4, -1e-8, -1e-22, -1e-38, -1e-45, -0.0, 0.0,
              1e-45, 1e-38, 1e-22, 1e-8, 1e-4, 0.06249999, 0.0625, 0.06250001,
              1, 5, 20, 100, 1e18]
    return np.asarray(values, dtype=np.float32)


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
            if kind == 'edges':
                values = edge_values(op)
                body = [Let('x', select_literal(values), F32)]
                x = np.resize(values, count)
            else:
                x = inputs(m)['x0'][:count]
                body = [read(('x',), 'x0', (Var('a0'),), F32)]
                if op in ('rsqrt', 'log'):
                    body += [Assign('x', ElementwiseImpl('multiply'), ('x', 'x'), F32),
                             Let('one', Literal(1.0), F32),
                             Assign('x', ElementwiseImpl('add'), ('x', 'one'), F32)]
                    x = np.asarray(x * x + np.float32(1), dtype=np.float32)
            result = 'x' if kind == 'random' else 'result'
            body += [Assign(result, ElementwiseImpl(op), ('x',), F32),
                     write('add', Var('a0'), (result,))]
            expected = np.zeros(size, dtype=np.float32)
            expected[:count] = reference(op, x)
            m['rtol'], m['atol'] = 3e-6, 2e-7
            emit(a.out, 'activation_' + op + '_' + kind, g, m, node, tile, body, expected)


if __name__ == '__main__':
    main()
