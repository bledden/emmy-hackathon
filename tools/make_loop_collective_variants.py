"""Repeated collective reductions with independent NumPy expected arrays."""
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
    # name, warps, subgroup width, shared tree, loop form, two states
    configs = [('warp32_loop', 4, 32, False, 'loop', False),
               ('warp8_strided', 4, 8, False, 'strided', False),
               ('warp32_zero', 4, 32, False, 'zero', False),
               ('warp32_nested', 4, 32, False, 'nested', False),
               ('tree4_loop', 4, 32, True, 'loop', False),
               ('tree8_strided', 8, 32, True, 'strided', False),
               ('tree1_loop', 1, 32, True, 'loop', False),
               ('tree4_nested', 4, 32, True, 'nested', False),
               ('tree4_two_state', 4, 32, True, 'loop', True)]
    add, maximum = ElementwiseImpl('add'), ElementwiseImpl('maximum')
    for name, warps, subgroup, shared, form, two in configs:
        threads = 32 * warps
        rounds = 6 if form == 'nested' else (0 if form == 'zero' else 3)
        size = max(1, rounds) * threads
        g, m, node, tile = template(args.upstream, size=size, cells=threads)
        tile = replace(tile, block_threads=threads)
        k = m['kernels'][0]
        k['block'], k['grid'], k['zero_outputs'] = [threads, 1, 1], [1, 1, 1], ['add']
        i, step = Var('a0'), Var('step')
        if form == 'strided':
            iteration = binary('//', binary('-', step, lit(1)), lit(2))
        elif form == 'nested':
            iteration = binary('+', binary('*', Var('outer'), lit(3)), step)
        else:
            iteration = step
        index = binary('+', binary('*', iteration, lit(threads)), i)
        states = ('partial', 'peak') if two else ('partial',)
        peers = tuple(s + '_peer' for s in states)
        buffers = tuple(s + '_shared' for s in states)
        combines = tuple(Assign(s, maximum if s == 'peak' else add, (s, p), F32)
                         for s, p in zip(states, peers))
        body = [read(('input_value',), 'x0', (index,), F32)]
        body += [Assign(s, ElementwiseImpl('copy'), ('input_value',), F32) for s in states]
        body.append(WarpShuffle(states, peers, combines, subgroup))
        if shared:
            body += [Cond(binary('==', Var('lane'), lit(0)),
                          tuple(write(b, Var('warp'), (s,)) for b, s in zip(buffers, states))),
                     Sync(), TreeHalve(buffers, states, peers, combines, warps, 'warp')]
        owner = 'outer' if form == 'nested' else 'step'
        body.append(Accum('total', 'partial', add, F32, axes=(owner,)))
        if two:
            body.append(Accum('total_peak', 'peak', maximum, F32, axes=(owner,)))
        if shared:
            # Every consumer finishes the broadcast before any next writer reuses it.
            body.append(Sync())
        if form == 'strided':
            loop = StridedLoop(Axis('step', Dim(7)), lit(1), lit(2), body, end=lit(7))
        elif form == 'nested':
            loop = Loop(Axis('outer', Dim(2)),
                        (Loop(Axis('step', Dim(3)), body, seed=False),))
        else:
            loop = Loop(Axis('step', Dim(rounds)), body, seed=form != 'zero')
        prefix = [Smem(b, (warps,), 'f32', 16) for b in buffers] if shared else []
        if form == 'zero':
            prefix.append(Init('total', 7.0, dtype=F32))
        outputs = ('total', 'total_peak') if two else ('total',)
        kernel_body = [*prefix, loop, write('add', binary('*', i, lit(len(outputs))), outputs)]
        expected = np.zeros(size, dtype=np.float32)
        data = inputs(m)['x0'].astype(np.float64)
        if form == 'zero':
            expected[:threads] = 7.0
        elif shared:
            result = [data.sum(), data.max()] if two else [data.sum()]
            expected[:threads * len(outputs)] = np.tile(result, threads)
        else:
            # Sum over rounds and subgroup members, broadcast to each subgroup lane.
            sums = data.reshape(rounds, threads // subgroup, subgroup).sum(axis=(0, 2))
            expected[:threads] = np.repeat(sums, subgroup)
        m['rtol'], m['atol'] = 2e-5, 2e-5
        emit(args.out, name, g, m, node, tile, kernel_body, expected)


if __name__ == '__main__':
    main()
