"""Count, cohort and phase obligations for named counted barriers."""
from dataclasses import replace
import unittest
from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr as B, Literal as L, Var as V
from emmy.compiler.ir.kernel.ir import Sync
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Let, Accum
import test_uniform_conditions as fixtures
from ptxgen.compiler import CompileError, compile_file
from ptxgen.participation import trips, values, whole_warps


class NamedBarrierTests(unittest.TestCase):
    stage = fixtures.UniformConditionTests.stage

    def compile_body(self, body):
        return compile_file(self.stage(lambda tile: replace(tile,body=body)))

    def test_exact_two_warp_group(self):
        code=self.compile_body((Cond(B('<',V('warp'),L(2,'int')),(Sync(barrier_id=1,count=64),)),))
        self.assertIn('bar.sync 1, 64;',code)

    def test_nested_and_complement_groups(self):
        body=(Cond(B('<',V('warp'),L(3,'int')),
                   (Cond(B('<',V('warp'),L(2,'int')),(Sync(barrier_id=1,count=64),),
                         (Sync(barrier_id=2,count=32),)),),
                   (Sync(barrier_id=3,count=32),)),)
        code=self.compile_body(body)
        for expected in ('bar.sync 1, 64','bar.sync 2, 32','bar.sync 3, 32'):self.assertIn(expected,code)

    def test_alias_predicates_and_noncontiguous_warps(self):
        body=(Let('w',B('&',V('warp'),L(1,'int'))),Cond(B('==',V('w'),L(0,'int')),
              (Sync(barrier_id=1,count=64),)))
        self.assertIn('bar.sync 1, 64',self.compile_body(body))

    def test_wrong_count_and_lane_divergence_refuse(self):
        for pred,count in ((B('<',V('warp'),L(2,'int')),32),(B('<',V('lane'),L(16,'int')),64),
                           (L(1,'int'),64)):
            with self.assertRaises(CompileError):self.compile_body((Cond(pred,(Sync(barrier_id=1,count=count),)),))

    def test_id_reuse_by_disjoint_or_overlapping_groups_refuses(self):
        for second in (B('>=',V('warp'),L(2,'int')),B('!=',V('warp'),L(0,'int'))):
            count=64 if second.op=='>=' else 96
            body=(Cond(B('<',V('warp'),L(2,'int')),(Sync(barrier_id=1,count=64),)),
                  Cond(second,(Sync(barrier_id=1,count=count),)))
            with self.assertRaisesRegex(CompileError,'different participant groups'):self.compile_body(body)

    def test_same_group_can_use_repeated_phases(self):
        sync=Sync(barrier_id=1,count=64)
        body=(Cond(B('<',V('warp'),L(2,'int')),(sync,Loop(Axis('i',Dim(4)),(sync,sync)),sync)),)
        self.assertEqual(self.compile_body(body).count('bar.sync 1, 64'),4)

    def test_per_group_loop_counts_can_differ(self):
        # Warps0/1 do4 trips and warps2/3 do3; each barrier cohort agrees.
        loop=StridedLoop(Axis('i',Dim(4)),B('//',V('warp'),L(2,'int')),L(1,'int'),
             (Cond(B('<',V('warp'),L(2,'int')),(Sync(barrier_id=1,count=64),),
                   (Sync(barrier_id=2,count=64),)),))
        code=self.compile_body((loop,));self.assertIn('bar.sync 2, 64',code)

    def test_unequal_phase_counts_refuse(self):
        loop=StridedLoop(Axis('i',Dim(4)),V('warp'),L(1,'int'),
             (Cond(B('<',V('warp'),L(2,'int')),(Sync(barrier_id=1,count=64),)),))
        with self.assertRaisesRegex(CompileError,'different loop phase counts'):self.compile_body((loop,))

    def test_inner_loop_cannot_hide_outer_phase_mismatch(self):
        body=(Cond(B('<',V('warp'),L(2,'int')),
              (StridedLoop(Axis('i',Dim(4)),V('warp'),L(1,'int'),
                   (Loop(Axis('j',Dim(2)),(Sync(barrier_id=1,count=64),)),)),)),)
        with self.assertRaisesRegex(CompileError,'different loop phase counts'):self.compile_body(body)

    def test_mutation_and_lexical_shadow_do_not_keep_exact_facts(self):
        body=(Let('w',V('warp')),Let('w',V('lane')),Cond(B('<',V('w'),L(2,'int')),(Sync(barrier_id=1,count=64),)))
        with self.assertRaises(CompileError):self.compile_body(body)
        body=(Let('w',V('lane')),Cond(L(1,'int'),(Let('w',V('warp')),)),
              Cond(B('<',V('w'),L(2,'int')),(Sync(barrier_id=1,count=64),)))
        with self.assertRaises(CompileError):self.compile_body(body)

    def test_backedge_invalidates_exact_predicate(self):
        body=(Let('w',V('warp')),Let('one',L(1,'int')),
              Loop(Axis('i',Dim(2)),(Cond(B('<',V('w'),L(2,'int')),(Sync(barrier_id=1,count=64),)),
                   Accum('w','one',axes=('i',),dtype='i32')),seed=False))
        with self.assertRaises(CompileError):self.compile_body(body)

    def test_static_dead_branch_has_no_barrier_phase(self):
        for literal in (L(0,'int'),L(-0.0),L(1e-50)):
            self.assertIn('bar.sync 1, 64',self.compile_body((Cond(literal,(Sync(barrier_id=1,count=64),)),)))

    def test_full_barrier_cannot_hide_in_partial_region(self):
        with self.assertRaisesRegex(CompileError,'collective inside conditional'):
            self.compile_body((Cond(B('<',V('warp'),L(2,'int')),
                       (Sync(barrier_id=1,count=64),Cond(L(1,'int'),(Sync(),)))),))

    def test_unknown_cta_predicate_keeps_group_ownership(self):
        body=(Cond(V('a0'),(Cond(B('<',V('warp'),L(2,'int')),(Sync(barrier_id=1,count=64),)),)),)
        self.assertIn('bar.sync 1, 64',self.compile_body(body))

    def test_all_four_warp_cohorts_and_counts(self):
        for mask in range(1,16):
            pred=L(0,'int')
            for w in range(4):
                if mask & (1<<w):pred=B('or',pred,B('==',V('warp'),L(w,'int')))
            for count in (32,64,96,128):
                body=(Cond(pred,(Sync(barrier_id=1,count=count),)),)
                if count==mask.bit_count()*32:
                    self.assertIn(f'bar.sync 1, {count}',self.compile_body(body))
                else:
                    with self.assertRaisesRegex(CompileError,'barrier count'):self.compile_body(body)

    def test_all_cohorts_against_independent_phase_counts(self):
        for mask in range(1,16):
            pred=L(0,'int');members=[]
            for w in range(4):
                if mask & (1<<w):
                    pred=B('or',pred,B('==',V('warp'),L(w,'int')));members.append(w)
            body=(StridedLoop(Axis('i',Dim(4)),V('warp'),L(1,'int'),
                  (Cond(pred,(Sync(barrier_id=1,count=32*len(members)),)),)),)
            counts=[len(range(w,4)) for w in members]
            if len(set(counts))==1:self.assertIn('bar.sync',self.compile_body(body))
            else:
                with self.assertRaisesRegex(CompileError,'phase counts'):self.compile_body(body)

    def test_exact_integer_evaluation_matches_signed_wrap(self):
        x=(2147483647,-2147483648,-7,7);y=(1,1,3,-3)
        self.assertEqual(values(B('+',V('x'),V('y')),{'x':x,'y':y},4),(-2147483648,-2147483647,-4,4))
        self.assertEqual(values(B('//',V('x'),V('y')),{'x':x,'y':y},4),(2147483647,-2147483648,-3,-3))
        self.assertIsNone(values(B('/',L(1,'int'),L(0,'int')),{},4))
        self.assertFalse(whole_warps(frozenset(range(63))));self.assertTrue(whole_warps(frozenset(range(64))))
        self.assertIsNone(trips((2147483646,),(2147483647,),2))


if __name__=='__main__':unittest.main()
