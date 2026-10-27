"""Collective loops with distinct starts and provably equal trip counts."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init

from make_coverage_variants import binary, emit, lit, read, template, write
from make_variants import inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    # name, rows, threads, stop, subgroup, shared, nested, explicit stop
    configs = [('uniform_shuffle_rows', 3, 128, 512, 32, False, False, False),
               ('uniform_tree_rows', 3, 128, 512, 32, True, False, False),
               ('uniform_subgroup_rows', 2, 128, 384, 8, False, False, False),
               ('uniform_nested', 2, 128, 256, 32, True, True, False),
               ('uniform_zero', 2, 128, 0, 32, True, False, False),
               ('uniform_explicit_stop', 3, 128, 512, 32, True, False, True),
               ('uniform_long', 2, 256, 4096, 32, True, False, False)]
    add = ElementwiseImpl('add')
    combine = (Assign('partial', add, ('partial', 'peer'), F32),)
    for name, rows, threads, stop, subgroup, shared, nested, explicit in configs:
        row_size = max(threads, stop)
        g, m, node, tile = template(args.upstream, size=rows * row_size, cells=rows * threads)
        tile = replace(tile, axes=(Axis('row', Dim(rows)), Axis('column', Dim(threads))),
                       block_threads=threads)
        k = m['kernels'][0]
        k['block'], k['grid'], k['zero_outputs'] = [threads, 1, 1], [rows, 1, 1], ['add']
        base = binary('*', Var('row'), lit(row_size))
        body = [read(('partial',), 'x0', (binary('+', base, Var('i')),), F32),
                WarpShuffle(('partial',), ('peer',), combine, subgroup)]
        if shared:
            body += [Cond(binary('==', Var('lane'), lit(0)),
                          (write('scratch', Var('warp'), ('partial',)),)),
                     Sync(), TreeHalve(('scratch',), ('partial',), ('peer',),
                                       combine, threads // 32, 'warp')]
        body.append(Accum('total', 'partial', add, F32, axes=('i',)))
        if shared:
            body.append(Sync())  # Broadcast consumers complete before reuse.
        if nested:
            body = [Loop(Axis('repeat', Dim(2)), body, seed=False)]
        loop = StridedLoop(Axis('i', Dim(stop + int(explicit))), Var('column'),
                           lit(threads), body, end=lit(stop) if explicit else None,
                           seed=stop != 0)
        prefix = [Smem('scratch', (threads // 32,), 'f32', 16)] if shared else []
        if stop == 0:
            prefix.append(Init('total', 7.0, dtype=F32))
        body = [*prefix, loop, write('add', binary('+', base, Var('column')), ('total',))]
        data = inputs(m)['x0'].astype(np.float64).reshape(rows, row_size)
        expected = np.zeros_like(data)
        if stop == 0:
            expected[:, :threads] = 7
        elif shared:
            expected[:, :threads] = data.sum(axis=1, keepdims=True) * (2 if nested else 1)
        else:
            sums = data.reshape(rows, stop // threads, threads // subgroup, subgroup).sum(axis=(1, 3))
            expected[:, :threads] = np.repeat(sums, subgroup, axis=1)
        m['rtol'], m['atol'] = 3e-5, 3e-5
        emit(args.out, name, g, m, node, tile, body, expected.reshape(-1))

    # Mixed-radix coordinates: each warp traverses its own 96-value segment
    # using a lane coordinate in [0,31], and broadcasts its three-round sum.
    rows, warps, width = 3, 4, 96
    g, m, node, tile = template(args.upstream, size=rows * warps * width, cells=rows * 128)
    tile = replace(tile, axes=(Axis('row', Dim(rows)), Axis('warp_column', Dim(warps)),
                              Axis('lane_column', Dim(32))))
    m['kernels'][0]['zero_outputs'] = ['add']
    segment = binary('+', binary('*', Var('row'), lit(warps)), Var('warp_column'))
    base = binary('*', segment, lit(width))
    loop = StridedLoop(Axis('i', Dim(width)), Var('lane_column'), lit(32),
                       (read(('partial',), 'x0', (binary('+', base, Var('i')),), F32),
                        WarpShuffle(('partial',), ('peer',), combine, 32),
                        Accum('total', 'partial', add, F32, axes=('i',))))
    expected = np.zeros((rows, warps, width), dtype=np.float64)
    data = inputs(m)['x0'].astype(np.float64).reshape(rows, warps, width)
    expected[:, :, :32] = data.sum(axis=2, keepdims=True)
    m['rtol'], m['atol'] = 3e-5, 3e-5
    emit(args.out, 'uniform_mixed_radix', g, m, node, tile,
         (loop, write('add', binary('+', base, Var('lane_column')), ('total',))),
         expected.reshape(-1))


if __name__ == '__main__':
    main()
