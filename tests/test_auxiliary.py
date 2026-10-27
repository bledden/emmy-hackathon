"""Logical-cell coverage and physical participant obligations."""
from dataclasses import replace
import json
import unittest
from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr as B, Literal as L, Var as V
from emmy.compiler.ir.kernel.ir import Sync, Smem, TreeHalve
from emmy.compiler.ir.stmt.blocks import Cond
from emmy.compiler.ir.stmt.leaves import Assign
import test_uniform_conditions as fixtures
from ptxgen.compiler import CompileError, compile_file


class AuxiliaryTests(unittest.TestCase):
    stage = fixtures.UniformConditionTests.stage

    def kernel(self, body=(), compute=128, aux=64, physical=192, cells=256, grid=2):
        axes = (Axis('row', Dim(cells // compute)), Axis('unit', Dim(compute)))
        p = self.stage(lambda t: replace(t, axes=axes, block_threads=compute,
                                       aux_threads=aux, body=body), 'p00_scalar_add')
        meta = json.loads(p.with_name('meta.json').read_text())
        meta['kernels'][0].update(block=[physical, 1, 1], grid=[grid, 1, 1])
        p.with_name('meta.json').write_text(json.dumps(meta))
        return p

    def test_physical_full_barrier_and_logical_row_uniformity(self):
        p = self.kernel((Cond(V('row'), (Sync(),)),))
        self.assertIn('bar.sync 0;', compile_file(p))

    def test_wrapped_logical_axis_selects_exact_physical_cohort(self):
        # unit<64 selects 64 compute + 64 auxiliary threads, not 64 total.
        condition = B('<', V('unit'), L(64, 'int'))
        self.assertIn('bar.sync 1, 128', compile_file(self.kernel(
            (Cond(condition, (Sync(barrier_id=1, count=128),)),))))
        with self.assertRaisesRegex(CompileError, 'barrier count'):
            compile_file(self.kernel((Cond(condition, (Sync(barrier_id=1, count=64),)),)))

    def test_compute_and_producer_warp_bands(self):
        condition = B('<', V('warp'), L(4, 'int'))
        code = compile_file(self.kernel((Cond(condition, (Sync(barrier_id=1, count=128),),
                                              (Sync(barrier_id=2, count=64),)),)))
        self.assertIn('bar.sync 1, 128', code)
        self.assertIn('bar.sync 2, 64', code)
        with self.assertRaises(CompileError):
            compile_file(self.kernel((Cond(condition, (Sync(),)),)))

    def test_invalid_auxiliary_geometry(self):
        for args in (dict(aux=-1), dict(aux=True), dict(aux=192),
                     dict(physical=160), dict(grid=1), dict(compute=128.0)):
            with self.subTest(args=args), self.assertRaises((CompileError, TypeError)):
                compile_file(self.kernel(**args))

    def test_partial_warp_boundary_does_not_gain_uniform_axis_proof(self):
        p = self.kernel((Cond(V('unit'), (Sync(warp=True),)),),
                        compute=48, aux=16, physical=64, cells=96)
        with self.assertRaises(CompileError):
            compile_file(p)

    def tree_body(self, predicate, length=128, count=128):
        tree = TreeHalve(('s',), ('v',), ('partner',),
                         (Assign('v', 'add', ('v', 'partner')),), length, 'unit',
                         barrier_id=1, barrier_count=count)
        return (Smem('s', (128,)), Cond(predicate, (tree,)))

    def test_named_tree_compute_band(self):
        code = compile_file(self.kernel(self.tree_body(B('<', V('warp'), L(4, 'int')))))
        self.assertEqual(code.count('bar.sync 1, 128;'), 7)

    def test_named_tree_aux_band_uses_present_writers(self):
        code = compile_file(self.kernel(self.tree_body(B('>=', V('warp'), L(4, 'int')),
                                                      length=64, count=64)))
        self.assertEqual(code.count('bar.sync 1, 64;'), 6)

    def test_named_tree_missing_slots_refuses(self):
        with self.assertRaisesRegex(CompileError, 'misses a shared slot'):
            compile_file(self.kernel(self.tree_body(B('>=', V('warp'), L(4, 'int')),
                                                   length=128, count=64)))

    def test_noncontiguous_named_tree_participants(self):
        # Lower and auxiliary aliases together still have one selected writer.
        code = compile_file(self.kernel(self.tree_body(B('<', V('unit'), L(64, 'int')),
                                                      length=64, count=128)))
        self.assertEqual(code.count('bar.sync 1, 128;'), 6)


if __name__ == '__main__':
    unittest.main()
