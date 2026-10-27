"""Independent flat and segmented shared-tree witnesses; no renderer calls."""
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
from emmy.compiler.ir.kernel.ir import Smem, Sync, TreeHalve
from emmy.compiler.ir.stmt.blocks import Cond, Loop
from emmy.compiler.ir.stmt.leaves import Accum, Assign

from make_coverage_variants import binary, emit, lit, read, template, write
from make_variants import inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    # name, segments, inner slots, copies, transposed, two states, rounds, reduction length
    configs = [('tree_flat128', 128, 1, 1, False, False, 1, 128),
               ('tree_flat256', 256, 1, 1, False, False, 1, 256),
               ('tree_flat_one', 128, 1, 1, False, False, 1, 1),
               ('tree_warp_inner_lane', 4, 32, 1, False, False, 1, 4),
               ('tree_segment8x16', 8, 16, 1, False, False, 1, 8),
               ('tree_transposed32x4', 32, 4, 1, True, False, 1, 32),
               ('tree_duplicate_coordinates', 8, 8, 2, False, False, 1, 8),
               ('tree_segment_coupled', 4, 32, 1, False, True, 1, 4),
               ('tree_segment_loop', 8, 16, 1, False, False, 3, 8),
               ('tree_flat_warp_name', 128, 1, 1, False, False, 1, 128),
               ('tree_segment_named_barrier', 8, 16, 1, False, False, 1, 8)]
    add, maximum = ElementwiseImpl('add'), ElementwiseImpl('maximum')
    rows = 2
    for name, segments, slots, copies, transpose, two, rounds, length in configs:
        threads = segments * slots * copies
        output_width = 2 if two else 1
        size = rows * threads * max(rounds, output_width)
        g, m, node, tile = template(args.upstream, size=size, cells=rows * threads)
        segment_name = 'warp' if name == 'tree_flat_warp_name' else 'segment'
        inner_axes = [Axis(segment_name, Dim(segments))]
        if slots > 1:
            slot_axis = Axis('slot', Dim(slots))
            inner_axes = [slot_axis, *inner_axes] if transpose else [*inner_axes, slot_axis]
        if copies > 1:
            inner_axes.insert(0, Axis('copy', Dim(copies)))
        tile = replace(tile, axes=(Axis('row', Dim(rows)), *inner_axes), block_threads=threads)
        k = m['kernels'][0]
        k['block'], k['grid'], k['zero_outputs'] = [threads, 1, 1], [rows, 1, 1], ['add']
        segment = Var(segment_name)
        slot = Var('slot') if slots > 1 else lit(0)
        physical = (binary('+', binary('*', slot, lit(segments)), segment) if transpose
                    else binary('+', binary('*', segment, lit(slots)), slot))
        if copies > 1:
            physical = binary('+', binary('*', Var('copy'), lit(segments * slots)), physical)
        row_base = binary('*', Var('row'), lit(threads * rounds))
        iteration = Var('round') if rounds > 1 else lit(0)
        source = binary('+', row_base, binary('+', binary('*', iteration, lit(threads)), physical))
        flat = binary('+', binary('*', segment, lit(slots)), slot)
        states = ('partial', 'peak') if two else ('partial',)
        peers = tuple(s + '_peer' for s in states)
        buffers = tuple(s + '_shared' for s in states)
        combine = tuple(Assign(s, maximum if s == 'peak' else add, (s, peer), F32)
                        for s, peer in zip(states, peers))
        body = [read(('partial',), 'x0', (source,), F32)]
        if two:
            body.append(Assign('peak', ElementwiseImpl('copy'), ('partial',), F32))
        stores = tuple(write(b, flat, (s,)) for b, s in zip(buffers, states))
        body += ([Cond(binary('==', Var('copy'), lit(0)), stores)] if copies > 1 else list(stores))
        body.append(Sync())
        tid = 'warp' if name == 'tree_warp_inner_lane' else segment_name
        inner = (('lane' if name == 'tree_warp_inner_lane' else 'slot', slots)
                 if slots > 1 else None)
        named = name == 'tree_segment_named_barrier'
        body.append(TreeHalve(buffers, states, peers, combine, length, tid,
                              barrier_id=3 if named else 0,
                              barrier_count=threads if named else None, inner=inner))
        if rounds > 1:
            body += [Accum('total', 'partial', add, F32, axes=('round',)), Sync()]
            body = [Loop(Axis('round', Dim(rounds)), body)]
        outputs = ('total',) if rounds > 1 else states
        destination = binary('*', binary('+', binary('*', Var('row'), lit(threads)), physical), lit(output_width))
        body += [write('add', destination, outputs)]
        body = [*(Smem(b, (segments * slots,), 'f32', 16) for b in buffers), *body]

        data = inputs(m)['x0'].astype(np.float64)[:rows * rounds * threads].reshape(rows, rounds, copies, threads // copies)
        expected = np.zeros(size, dtype=np.float64)
        for row in range(rows):
            slab = data[row, :, 0].reshape((rounds, slots, segments) if transpose else (rounds, segments, slots))
            if transpose:
                slab = slab.transpose(0, 2, 1)
            total = slab[:, :length, :].sum(axis=(0, 1))
            peak = slab[:, :length, :].max(axis=(0, 1))
            for thread in range(threads):
                local = thread % (segments * slots)
                slot_number = local // segments if transpose else local % slots
                out = (row * threads + thread) * output_width
                expected[out] = total[slot_number]
                if two:
                    expected[out + 1] = peak[slot_number]
        m['rtol'], m['atol'] = 3e-5, 3e-5
        emit(args.out, name, g, m, node, tile, body, expected)


if __name__ == '__main__':
    main()
