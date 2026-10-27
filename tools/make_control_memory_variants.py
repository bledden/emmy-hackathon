"""Typed IR witnesses for branch, seed, tree and async-copy semantics.

Expected arrays come from NumPy and direct mathematical descriptions, never
from emitted PTX or Emmy's renderer.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import (CpAsyncCommit, CpAsyncCopy, CpAsyncWait,
                                       Smem, Sync, TreeHalve, WarpShuffle)
from emmy.compiler.ir.stmt.blocks import Cond, Loop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let

from make_coverage_variants import binary, emit, lit, read, template, write
from make_variants import inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    add, maximum = ElementwiseImpl('add'), ElementwiseImpl('maximum')

    # Both arms execute within each warp, plus a partial final CTA.
    g, m, node, tile = template(args.upstream, size=257, cells=257)
    i = Var('a0')
    predicate = binary('==', binary('%', i, lit(3)), lit(0))
    body = (read(('x',), 'x0', (i,), F32), read(('y',), 'x1', (i,), F32),
            Cond(predicate,
                 (Assign('plus', add, ('x', 'y'), F32), write('add', i, ('plus',))),
                 (Assign('minus', ElementwiseImpl('subtract'), ('x', 'y'), F32),
                  write('add', i, ('minus',)))))
    x = inputs(m)
    result = np.where(np.arange(257) % 3 == 0, x['x0'] + x['x1'], x['x0'] - x['x1'])
    emit(args.out, 'conditional_else_tail', g, m, node, tile, body, result)

    # A boolean expression selects a reversed or original integer index, then a
    # float ternary selects one of two loaded SSA values. Every index stays valid.
    g, m, node, tile = template(args.upstream, size=257, cells=257)
    condition = binary('or', binary('<', i, lit(64)), binary('>=', i, lit(192)))
    body = (Let('chosen_index', TernaryExpr(condition, binary('-', lit(256), i), i)),
            read(('x',), 'x0', (Var('chosen_index'),), F32),
            read(('y',), 'x1', (i,), F32),
            Let('chosen_float', TernaryExpr(binary('==', binary('%', i, lit(2)), lit(0)),
                                           Var('x'), Var('y'))),
            write('add', i, ('chosen_float',)))
    x = inputs(m)
    indices = np.arange(257)
    selected = np.where((indices < 64) | (indices >= 192), 256 - indices, indices)
    result = np.where(indices % 2 == 0, x['x0'][selected], x['x1'])
    emit(args.out, 'ternary_index_and_float', g, m, node, tile, body, result)

    # Broadcast the full result to every lane, not just the tree's writer lanes.
    for warps, two_state in [(1, False), (2, False), (8, False), (16, False), (8, True)]:
        threads = 32 * warps
        width = 2 if two_state else 1
        g, m, node, tile = template(args.upstream, size=threads * width, cells=threads)
        tile = replace(tile, block_threads=threads)
        m['kernels'][0]['block'], m['kernels'][0]['grid'] = [threads, 1, 1], [1, 1, 1]
        states = ('sum', 'max') if two_state else ('sum',)
        peers = tuple(s + '_other' for s in states)
        buffers = tuple(s + '_shared' for s in states)
        combines = tuple(Assign(s, maximum if s == 'max' else add, (s, p), F32)
                         for s, p in zip(states, peers))
        body = [read(('value',), 'x0', (i,), F32)]
        body += [Assign(s, ElementwiseImpl('copy'), ('value',), F32) for s in states]
        body.append(WarpShuffle(states, peers, combines, 32))
        body += [Smem(b, (warps,), 'f32', 16) for b in buffers]
        body.append(Cond(binary('==', Var('lane'), lit(0)),
                         tuple(write(b, Var('warp'), (s,)) for b, s in zip(buffers, states))))
        body += [Sync(barrier_id=3, count=threads),
                 TreeHalve(buffers, states, peers, combines, warps, 'warp',
                           barrier_id=3, barrier_count=threads),
                 write('add', binary('*', i, lit(width)), states)]
        data = inputs(m)['x0'][:threads].astype(np.float64)
        expected = np.tile([data.sum(), data.max()] if two_state else [data.sum()], threads)
        m['rtol'], m['atol'] = 2e-5, 2e-5
        emit(args.out, f'tree_{warps}_warps' + ('_two_state' if two_state else ''),
             g, m, node, tile, body, expected)

    # Each thread produces one distinct chunk; consumers read another thread's
    # chunk after commit, wait and CTA synchronization. Offset1 tests fallback.
    configs = [(w, offset, False) for w in (1, 2, 4) for offset in (0, 1)] + [(2, 0, True)]
    for width, offset, dynamic in configs:
        size = 128 * width + 2 * offset
        g, m, node, tile = template(args.upstream, size=size)
        destination = binary('+', binary('*', i, lit(width)), lit(offset))
        source = binary('+', binary('*', binary('-', lit(127), i), lit(width)), lit(offset))
        names = tuple(f'v{j}' for j in range(width))
        body = [Smem('stage', (size,), 'f32', 16),
                CpAsyncCopy('stage', (destination,), 'x0', (destination,), nbytes=4 * width),
                CpAsyncCommit(), CpAsyncWait(0), Sync(),
                read(names, 'stage', (source,), F32), write('add', destination, names)]
        k = m['kernels'][0]
        k['zero_outputs'] = ['add']
        if dynamic:
            k['smem_bytes'] = (size * 4 + 15) // 16 * 16
        expected = np.zeros(size, dtype=np.float32)
        data = inputs(m)['x0'][offset:offset + 128 * width]
        expected[offset:offset + 128 * width] = data.reshape(128, width)[::-1].reshape(-1)
        name = f'async_{width * 4}_bytes_offset_{offset}' + ('_dynamic' if dynamic else '')
        emit(args.out, name, g, m, node, tile, body, expected)

    # seed=True resets only the loop owning the fold; False preserves prior state.
    for mode in ('preseeded', 'inner_reset', 'outer_reset'):
        g, m, node, tile = template(args.upstream)
        axes = ('outer',) if mode == 'outer_reset' else ('inner',)
        fold = Accum('state', 'value', add, F32, axes=axes)
        inner = Loop(Axis('inner', Dim(3)), (fold,), seed=mode == 'inner_reset')
        outer = Loop(Axis('outer', Dim(2)), (inner,), seed=mode == 'outer_reset')
        body = [read(('value',), 'x0', (i,), F32), Init('state', 7.0, dtype=F32),
                outer, write('add', i, ('state',))]
        if mode == 'inner_reset':
            # The inner fold exports into the outer body only. Observe it
            # there, with no outer declaration of the same spelling.
            body = [read(('value',), 'x0', (i,), F32),
                    Loop(Axis('outer', Dim(2)), (inner, write('add', i, ('state',))), seed=False)]
        x = inputs(m)['x0'].astype(np.float64)
        expected = 7 + 6 * x if mode == 'preseeded' else (3 * x if mode == 'inner_reset' else 6 * x)
        m['rtol'], m['atol'] = 2e-6, 2e-6
        name = 'seed_inner_local_reset' if mode == 'inner_reset' else 'seed_' + mode
        emit(args.out, name, g, m, node, tile, body, expected)


if __name__ == '__main__':
    main()
