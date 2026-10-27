"""Finite coordinate ownership and convergent, guarded collective loops."""
from dataclasses import replace
import unittest
from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal as L, Var as V, BinaryExpr as B
from emmy.compiler.ir.stmt.blocks import StridedLoop, Loop
from emmy.compiler.ir.stmt.leaves import Let, Assign, Accum, Load
from emmy.compiler.ir.kernel.ir import Sync, Smem, TreeHalve
from ptxgen.compiler import compile_file, CompileError
import test_uniform_conditions as fixtures

I = lambda n: L(n, 'int')

class CollectiveCoordinateTests(unittest.TestCase):
    stage = fixtures.UniformConditionTests.stage

    def compile(self, body):
        return compile_file(self.stage(lambda t: replace(t, body=body)))

    def loop(self, step, warp=False, start=0, stop=17):
        return StridedLoop(Axis('i', Dim(stop)), I(start), step,
                           (Sync(warp=warp),))

    def tree(self, name='coord', inner=None):
        return TreeHalve(('s',), ('state',), ('other',),
                         (Assign('state', 'add', ('state', 'other')),),
                         4, name, inner=inner)

    def test_cta_varying_group_uniform_runtime_step(self):
        code = self.compile((self.loop(B('+', V('a0'), I(1))),))
        self.assertIn('bar.sync', code)
        self.assertIn('setp.gt.s64', code)

    def test_warp_uniform_step_has_only_warp_scope(self):
        step = B('+', V('warp'), I(1))
        self.compile((self.loop(step, warp=True),))
        with self.assertRaisesRegex(CompileError, 'proven uniform'):
            self.compile((self.loop(step),))

    def test_lane_varying_steps_are_unproved(self):
        for body in ((self.loop(B('+', V('lane'), I(1)), warp=True),),
                     (Load('step', 'x', (I(0), V('lane')), dtype='i32'), self.loop(V('step')))):
            with self.assertRaisesRegex(CompileError, 'proven uniform'):
                self.compile(body)

    def test_static_terminal_overflow_uses_guard(self):
        for body in ((self.loop(I(4), start=2147483644, stop=2147483647),),
                     (StridedLoop(Axis('i', Dim(2147483647)), I(2147483644), I(4), ()),)):
            self.assertIn('setp.gt.s64', self.compile(body))

    def test_carried_step_fact_cannot_survive_backedge(self):
        body = (Let('step', I(1)), Let('delta', V('lane')),
                StridedLoop(Axis('i', Dim(17)), I(0), V('step'),
                            (Sync(), Accum('step', 'delta', axes=('i',), dtype='i32')), seed=False))
        with self.assertRaisesRegex(CompileError, 'proven uniform'):
            self.compile(body)

    def test_tree_alias_and_permutations(self):
        for expr in (V('warp'), B('-', I(3), V('warp')), B('^', V('warp'), I(3))):
            self.compile((Smem('s', (4,)), Let('coord', expr), self.tree()))

    def test_tree_assign_alias_and_permuted_inner(self):
        body = (Smem('s', (16,)), Assign('coord', 'copy', ('warp',), dtype='i32'),
                Let('slot', B('^', B('%', V('lane'), I(4)), I(3))),
                self.tree(inner=('slot', 4)))
        self.compile(body)

    def test_tree_missing_and_invalid_slots(self):
        for expr in (I(0), B('*', V('warp'), I(2)), B('-', V('warp'), I(1))):
            with self.assertRaisesRegex(CompileError, 'shared slot'):
                self.compile((Smem('s', (4,)), Let('coord', expr), self.tree()))

    def test_tree_inner_must_cover_independent_slots(self):
        for expr in (V('warp'), I(4), I(-1)):
            with self.assertRaisesRegex(CompileError, 'shared slot'):
                self.compile((Smem('s', (16,)), Let('coord', V('warp')), Let('slot', expr),
                              self.tree(inner=('slot', 4))))

    def test_tree_facts_expire_after_redefinition_and_backedge(self):
        prefix = (Smem('s', (4,)), Let('coord', V('warp')))
        for body in ((Load('coord', 'x', (I(0), I(0)), dtype='i32'), self.tree()),
                     (Let('delta', I(1)), Loop(Axis('i', Dim(2)),
                       (self.tree(), Accum('coord', 'delta', axes=('i',), dtype='i32')), seed=False))):
            with self.assertRaisesRegex(CompileError, 'exact coordinate values'):
                self.compile(prefix + body)

if __name__ == '__main__':
    unittest.main()
