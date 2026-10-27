"""Expression fixtures with independent NumPy expectations and finite readback.

Special values are local IR literals, never JSON runner inputs or outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from emmy.compiler.dtype import F32
from emmy.compiler.ir.expr import Literal, TernaryExpr, Var
from emmy.compiler.ir.stmt.blocks import Cond
from emmy.compiler.ir.stmt.leaves import Let

from make_coverage_variants import binary, emit, lit, read, template, write
from make_variants import inputs


COMPARISONS = [('eq', '==', np.equal), ('ne', '!=', np.not_equal),
               ('lt', '<', np.less), ('le', '<=', np.less_equal),
               ('gt', '>', np.greater), ('ge', '>=', np.greater_equal)]


def table(index, values):
    result = Literal(float(values[-1]))
    for j in reversed(range(len(values) - 1)):
        result = TernaryExpr(binary('==', index, lit(j)), Literal(float(values[j])), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    i, x, y = Var('a0'), Var('x'), Var('y')
    load = (read(('x',), 'x0', (i,), F32), read(('y',), 'x1', (i,), F32))
    indices = np.arange(257)

    for label, op, compare in COMPARISONS:
        g, m, node, tile = template(args.upstream, size=257, cells=257)
        data = inputs(m)
        # Equal lanes prevent eq/ne from passing through an all-false/all-true shortcut.
        other = np.where(indices % 3 == 0, data['x0'], data['x1'])
        body = (*load, Let('other', TernaryExpr(binary('==', binary('%', i, lit(3)), lit(0)), x, y)),
                Let('flag', binary(op, x, Var('other'))),
                Let('result', TernaryExpr(Var('flag'), Literal(3.0), Literal(-2.0))),
                write('add', i, ('result',)))
        expected = np.where(compare(data['x0'], other), 3.0, -2.0)
        emit(args.out, 'float_compare_' + label, g, m, node, tile, body, expected)

    g, m, node, tile = template(args.upstream, size=257, cells=257)
    data = inputs(m)
    value = binary('-', binary('+', binary('*', x, y), x), y)
    expected = ((data['x0'] * data['x1']) + data['x0']) - data['x1']
    emit(args.out, 'float_expression_arithmetic', g, m, node, tile,
         (*load, Let('result', value), write('add', i, ('result',))), expected)

    g, m, node, tile = template(args.upstream, size=257, cells=257)
    data = inputs(m)
    # Cond-local names do not escape. Store within each arm instead of
    # inventing an implicit phi/merge through the emitter's register cache.
    body = (*load, Cond(binary('<', x, y),
                        (Let('merged', binary('+', x, y)), write('add', i, ('merged',))),
                        (Let('merged', binary('-', x, y)), write('add', i, ('merged',)))))
    expected = np.where(data['x0'] < data['x1'], data['x0'] + data['x1'], data['x0'] - data['x1'])
    emit(args.out, 'float_cond_branch_stores', g, m, node, tile, body, expected)

    g, m, node, tile = template(args.upstream, size=257, cells=257)
    # Materialized integer booleans participate in address arithmetic.
    flag1 = binary('&&', binary('<', i, lit(200)), binary('==', binary('%', i, lit(3)), lit(0)))
    flag2 = binary('||', binary('>=', i, lit(64)), binary('==', i, lit(0)))
    body = (Let('f1', flag1), Let('f2', flag2), Let('index', binary('+', Var('f1'), Var('f2'))),
            read(('result',), 'x0', (Var('index'),), F32), write('add', i, ('result',)))
    selected = ((indices < 200) & (indices % 3 == 0)).astype(int) + ((indices >= 64) | (indices == 0))
    emit(args.out, 'boolean_let_index', g, m, node, tile, body, inputs(m)['x0'][selected])

    g, m, node, tile = template(args.upstream, size=257, cells=257)
    tiny = np.nextafter(np.float32(0), np.float32(1))
    special = np.array([0., -0., tiny, -tiny, np.inf, -np.inf, np.nan, 1.], dtype=np.float32)
    body = (Let('special', table(binary('%', i, lit(8)), special)),
            Let('result', TernaryExpr(Var('special'), Literal(1.0), Literal(-1.0))),
            write('add', i, ('result',)))
    emit(args.out, 'float_truth_special', g, m, node, tile, body,
         np.where(special[indices % 8] != 0, 1.0, -1.0))

    g, m, node, tile = template(args.upstream, size=257 * 6, cells=257)
    left = np.array([np.nan, 1., np.nan, 0., -0., np.inf, -np.inf, tiny], dtype=np.float32)
    right = np.array([1., np.nan, np.nan, -0., 0., np.inf, np.inf, -tiny], dtype=np.float32)
    body = [Let('left', table(binary('%', i, lit(8)), left)),
            Let('right', table(binary('%', i, lit(8)), right))]
    expected = np.empty((257, 6), dtype=np.float32)
    for j, (label, op, compare) in enumerate(COMPARISONS):
        body += [Let(label, TernaryExpr(binary(op, Var('left'), Var('right')), Literal(1.0), Literal(0.0))),
                 write('add', binary('+', binary('*', i, lit(6)), lit(j)), (label,))]
        expected[:, j] = compare(left[indices % 8], right[indices % 8])
    emit(args.out, 'float_compare_special', g, m, node, tile, body, expected.reshape(-1))

    g, m, node, tile = template(args.upstream, size=257, cells=257)
    data = inputs(m)
    condition = binary('&&', binary('>', binary('*', x, x), Literal(0.25)),
                       binary('%', i, lit(2)))
    body = (*load, Let('result', TernaryExpr(condition, Literal(1.5), Literal(-2.0))),
            write('add', i, ('result',)))
    expected = np.where((data['x0'] * data['x0'] > np.float32(.25)) & (indices % 2 != 0), 1.5, -2.0)
    emit(args.out, 'nested_float_integer_predicate', g, m, node, tile, body, expected)

    # The old predicate path silently truncated these literals to integer 1.
    # Exercise both immediate operands and every comparison without a Let flag.
    g, m, node, tile = template(args.upstream, size=257 * 6, cells=257)
    body = []
    expected = np.empty((257, 6), dtype=np.float32)
    for j, (label, op, compare) in enumerate(COMPARISONS):
        body += [Let(label, TernaryExpr(binary(op, Literal(1.25), Literal(1.75)),
                                       Literal(1.0), Literal(0.0)), F32),
                 write('add', binary('+', binary('*', i, lit(6)), lit(j)), (label,))]
        expected[:, j] = compare(np.float32(1.25), np.float32(1.75))
    emit(args.out, 'float_literal_predicates', g, m, node, tile, body, expected.reshape(-1))


if __name__ == '__main__':
    main()
