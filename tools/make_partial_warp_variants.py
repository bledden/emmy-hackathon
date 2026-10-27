"""Partial-warp shuffles with independent NumPy subgroup expectations."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import WarpShuffle
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Let

from make_coverage_variants import binary, emit, lit, read, template, write
from make_variants import inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    configs = [('one_thread', 1, 1, 'plain'),
               ('three_threads', 3, 1, 'plain'),
               ('full_width_one', 32, 1, 'plain'),
               ('eight_width2', 8, 2, 'plain'),
               ('eight_width4', 8, 4, 'plain'),
               ('eight_width8', 8, 8, 'plain'),
               ('sixteen_width16', 16, 16, 'plain'),
               ('twentyfour_width8', 24, 8, 'plain'),
               ('forty_width8', 40, 8, 'plain'),
               ('fortyeight_coupled', 48, 16, 'coupled'),
               ('fiftysix_loop', 56, 8, 'loop'),
               ('seventytwo_predicated', 72, 8, 'predicated'),
               ('forty_shadow_names', 40, 8, 'shadow'),
               ('forty_mixed_widths', 40, 8, 'mixed')]
    add, maximum = ElementwiseImpl('add'), ElementwiseImpl('maximum')
    rows = 3
    for name, threads, width, mode in configs:
        rounds = 3 if mode == 'loop' else 1
        two = mode in ('coupled', 'mixed')
        output_width = 2 if two else 1
        size = rows * threads * max(rounds, output_width) + 17
        g, m, node, tile = template(args.upstream, size=size, cells=rows * threads)
        tile = replace(tile, axes=(Axis('row', Dim(rows)), Axis('local', Dim(threads))),
                       block_threads=threads)
        k = m['kernels'][0]
        # One wholly unused CTA exercises the existing early-exit guard too.
        k['block'], k['grid'], k['zero_outputs'] = [threads, 1, 1], [rows + 1, 1, 1], ['add']
        local, row = Var('local'), Var('row')
        iteration = Var('round') if rounds > 1 else lit(0)
        index = binary('+', binary('*', binary('+', binary('*', row, lit(rounds)), iteration), lit(threads)), local)
        body = [read(('partial',), 'x0', (index,), F32)]
        if mode == 'predicated':
            # An explicit value selection keeps the result in this scope.
            # This tests selection before shuffle, not divergent reconvergence.
            body += [Assign('negated', ElementwiseImpl('negative'), ('partial',), F32),
                     Assign('squared', ElementwiseImpl('multiply'), ('partial', 'partial'), F32),
                     Let('partial', TernaryExpr(binary('==', binary('%', local, lit(2)), lit(0)),
                                               Var('negated'), Var('squared')), F32)]
        if mode == 'shadow':
            body += [Let('warp', lit(777)), Let('lane', lit(31))]
        if two:
            body.append(Assign('secondary', ElementwiseImpl('copy'), ('partial',), F32))
        primary_combine = Assign('partial', add, ('partial', 'peer'), F32)
        if mode == 'coupled':
            secondary_combine = Assign('secondary', maximum, ('secondary', 'other'), F32)
            body.append(WarpShuffle(('partial', 'secondary'), ('peer', 'other'),
                                    (primary_combine, secondary_combine), width))
        else:
            body.append(WarpShuffle(('partial',), ('peer',), (primary_combine,), width))
            if mode == 'mixed':
                body.append(WarpShuffle(('secondary',), ('other',),
                                        (Assign('secondary', add, ('secondary', 'other'), F32),), 2))
        if rounds > 1:
            body += [Accum('total', 'partial', add, F32, axes=('round',))]
            body = [Loop(Axis('round', Dim(rounds)), body)]
        outputs = ('total',) if rounds > 1 else (('partial', 'secondary') if two else ('partial',))
        destination = binary('*', binary('+', binary('*', row, lit(threads)), local), lit(output_width))
        body.append(write('add', destination, outputs))

        data = inputs(m)['x0'][:rows * rounds * threads].reshape(rows, rounds, threads)
        if mode == 'predicated':
            data = np.where(np.arange(threads) % 2 == 0, -data, data * data)
        slab = data.astype(np.float64).reshape(rows, rounds, threads // width, width)
        primary = np.repeat(slab.sum(axis=(1, 3)), width, axis=1)
        expected = np.zeros(size, dtype=np.float32)
        if two:
            if mode == 'coupled':
                secondary = np.repeat(slab.max(axis=(1, 3)), width, axis=1)
            else:
                secondary = np.repeat(data.astype(np.float64).reshape(rows, threads // 2, 2).sum(axis=2), 2, axis=1)
            expected[:rows * threads * 2] = np.stack((primary, secondary), axis=2).ravel()
        else:
            expected[:rows * threads] = primary.ravel()
        m['rtol'], m['atol'] = 3e-5, 3e-5
        emit(args.out, 'partial_' + name, g, m, node, tile, body, expected)


if __name__ == '__main__':
    main()
