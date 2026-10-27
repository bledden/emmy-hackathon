"""Participation proofs distinguish warp operations from cross-warp barriers."""
from dataclasses import replace
import unittest
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Sync, WarpShuffle
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Let
import test_uniform_conditions as fixtures
from ptxgen.compiler import CompileError, compile_file


class WarpParticipationTests(unittest.TestCase):
    stage = fixtures.UniformConditionTests.stage

    def compile_body(self, body):
        return compile_file(self.stage(lambda tile: replace(tile,body=body)))

    def test_warp_conditions_and_aliases_allow_warp_sync(self):
        for expr in (Var('warp'),BinaryExpr('<',Var('warp'),Literal(2,'int'))):
            self.assertIn('bar.warp.sync',self.compile_body((Cond(expr,(Sync(warp=True),)),)))
        self.assertIn('bar.warp.sync',self.compile_body((Let('w',Var('warp')),Cond(Var('w'),(Sync(warp=True),)),)))

    def test_nested_warp_conditions_allow_shuffle(self):
        shuffle=WarpShuffle(('v',),('peer',),(Assign('v',ElementwiseImpl('add'),('v','peer'),F32),),32)
        code=self.compile_body((Let('v',Literal(1.),F32),Cond(Var('warp'),(Cond(Var('a0'),(shuffle,Sync(warp=True))),))))
        self.assertIn('shfl.sync',code)

    def test_cross_warp_and_lane_divergence_still_refuse(self):
        for body in ((Cond(Var('warp'),(Sync(),)),),
                     (Cond(Var('warp'),(Sync(warp=True),Cond(Literal(1,'int'),(Sync(),)))),),
                     (Cond(Var('lane'),(Sync(warp=True),)),),
                     (Cond(Var('warp'),(Sync(barrier_id=1,count=64),)),)):
            with self.assertRaisesRegex(CompileError,'collective inside conditional|barrier count'):
                self.compile_body(body)

    def test_warp_alias_mutation_and_shadowing_do_not_leak(self):
        bodies=[(Let('w',Var('warp')),Let('w',Var('lane')),Cond(Var('w'),(Sync(warp=True),))),
                (Let('w',Var('lane')),Cond(Literal(1,'int'),(Let('w',Var('warp')),)),Cond(Var('w'),(Sync(warp=True),))),
                (Let('warp',Var('lane')),Cond(Var('warp'),(Sync(warp=True),)))]
        for body in bodies:
            with self.assertRaisesRegex(CompileError,'collective inside conditional'):self.compile_body(body)

    def test_warp_alias_backedge_fact_is_killed(self):
        body=(Let('flag',Literal(1.),F32),Let('delta',Literal(1.),F32),
              Loop(Axis('i',Dim(2)),(Cond(Var('flag'),(Sync(warp=True),)),
                                    Accum('flag','delta',axes=('i',))),seed=False))
        with self.assertRaisesRegex(CompileError,'collective inside conditional'):self.compile_body(body)

    def test_warp_uniform_loop_bounds_can_vary_between_warps(self):
        for start,end in ((Var('warp'),None),(Literal(0,'int'),BinaryExpr('+',Var('warp'),Literal(1,'int')))):
            body=(StridedLoop(Axis('i',Dim(4)),start,Literal(1,'int'),
                              (Cond(Var('i'),(Sync(warp=True),)),),end=end),)
            self.assertIn('bar.warp.sync',self.compile_body(body))

    def test_warp_bound_cannot_guard_cta_or_vary_within_warp(self):
        for start,sync in ((Var('warp'),Sync()),(Var('lane'),Sync(warp=True))):
            body=(StridedLoop(Axis('i',Dim(32)),start,Literal(1,'int'),(sync,)),)
            with self.assertRaisesRegex(CompileError,'proven uniform'):self.compile_body(body)

    def test_symbolic_static_step_and_stop(self):
        body=(Let('step',BinaryExpr('+',Literal(1,'int'),Literal(1,'int'))),
              Let('stop',Literal(8,'int')),
              StridedLoop(Axis('i',Dim(8)),Literal(0,'int'),Var('step'),(Sync(),),end=Var('stop')))
        self.assertIn('bar.sync',self.compile_body(body))

    def test_step_mutation_and_nonpositive_step_refuse(self):
        for value in (0,-1):
            with self.assertRaisesRegex(CompileError,'positive static'):
                self.compile_body((Let('step',Literal(value,'int')),StridedLoop(Axis('i',Dim(4)),Literal(0,'int'),Var('step'),()),))
        body=(Let('step',Literal(1,'int')),Let('delta',Literal(1,'int')),
              StridedLoop(Axis('i',Dim(8)),Literal(0,'int'),Var('step'),
                          (Sync(),Accum('step','delta',axes=('i',),dtype='i32')),seed=False))
        with self.assertRaisesRegex(CompileError,'proven uniform'):self.compile_body(body)

    def test_float_step_is_not_a_static_integer_step(self):
        with self.assertRaisesRegex(CompileError,'requires s32'):
            self.compile_body((StridedLoop(Axis('i',Dim(4)),Literal(0,'int'),Literal(1.5),(Sync(warp=True),)),))

    def test_coordinate_uniform_per_warp_but_not_per_cta(self):
        def change(tile,sync):
            return replace(tile,axes=(Axis('row',Dim(256)),Axis('col',Dim(32))),body=(Cond(Var('row'),(sync,)),))
        self.assertIn('bar.warp.sync',compile_file(self.stage(lambda t:change(t,Sync(warp=True)))))
        with self.assertRaisesRegex(CompileError,'CTA-uniform'):
            compile_file(self.stage(lambda t:change(t,Sync())))

    def test_terminal_overflow_is_guarded_for_warp_uniform_loops(self):
        start=BinaryExpr('+',Literal(2147483644,'int'),Var('warp'))
        body=(StridedLoop(Axis('i',Dim(2147483647)),start,Literal(4,'int'),(Sync(warp=True),)),)
        self.assertIn('setp.gt.s64',self.compile_body(body))


if __name__=='__main__':unittest.main()
