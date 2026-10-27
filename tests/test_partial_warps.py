from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from emmy.compiler.dtype import F32
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.kernel.ir import Sync, WarpShuffle
from emmy.compiler.ir.stmt.leaves import Assign
from make_coverage_variants import emit, template
from ptxgen.compiler import CompileError, compile_file

TASK = Path(os.environ.get('EMMY_TASK_ROOT', ROOT.parent / 'upstream/emmy-hackathon-2026'))


class PartialWarpTests(unittest.TestCase):
    def build(self, threads, widths, *, cells=None, sync=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        out = Path(temporary.name)
        g, m, node, tile = template(TASK, cells=threads if cells is None else cells)
        tile = replace(tile, block_threads=threads)
        m['kernels'][0].update(block=[threads, 1, 1], grid=[2, 1, 1])
        # Keep the original input loads and output but reduce their sum.
        body = list(tile.body[:-1])
        state = tile.body[-1].values[0]
        combine = Assign(state, ElementwiseImpl('add'), (state, 'peer'), F32)
        body += [WarpShuffle((state,), ('peer',), (combine,), w) for w in widths]
        if sync:
            body.append(Sync())
        body.append(tile.body[-1])
        emit(out, 'test', g, m, node, tile, body, [])
        return compile_file(next(out.rglob('*.ir.json')))

    def test_short_and_final_warp_masks(self):
        for threads, width in [(8, 8), (24, 8), (40, 8), (48, 16), (72, 4)]:
            with self.subTest(threads=threads, width=width):
                ptx = self.build(threads, [width])
                self.assertIn(hex((1 << (threads % 32)) - 1), ptx)
                self.assertNotIn('activemask', ptx)
                self.assertEqual('selp.b32' in ptx, threads > 32)
        ptx = self.build(64, [32])
        self.assertIn('0xffffffff;', ptx)
        self.assertNotIn('selp.b32', ptx)

    def test_incomplete_groups_and_invalid_widths_refuse(self):
        for threads, widths in [(8, [16]), (24, [16]), (40, [8, 16]),
                                (56, [32]), (40, [0]), (40, [3]), (64, [64])]:
            with self.subTest(threads=threads, widths=widths):
                with self.assertRaisesRegex(CompileError, 'shuffle'):
                    self.build(threads, widths)

    def test_width_one_is_identity(self):
        for threads in (1, 3, 32, 33):
            with self.subTest(threads=threads):
                self.assertNotIn('shfl.sync', self.build(threads, [1]))

    def test_partial_cta_and_nonshuffle_collectives_remain_refused(self):
        with self.assertRaisesRegex(CompileError, 'partial-block collectives'):
            self.build(40, [8], cells=39)
        with self.assertRaisesRegex(CompileError, 'only shuffle collectives'):
            self.build(40, [8], sync=True)


if __name__ == '__main__':
    unittest.main()
