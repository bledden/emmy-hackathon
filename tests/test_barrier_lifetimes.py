"""Ownership joins, missing fences and loop backedges, independently enumerated."""
from dataclasses import replace
import itertools
import random
import unittest
from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr as B, Literal as L, Var as V
from emmy.compiler.ir.kernel.ir import Sync
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
import test_uniform_conditions as fixtures
from ptxgen.compiler import CompileError, compile_file
from ptxgen.barrier_lifetimes import validate


def group(which, identifier=1):
    return Cond(B('<' if which == 0 else '>=', V('warp'), L(2, 'int')),
                (Sync(barrier_id=identifier, count=64),))


class BarrierLifetimeTests(unittest.TestCase):
    stage = fixtures.UniformConditionTests.stage

    def compile_body(self, body):
        return compile_file(self.stage(lambda t: replace(t, body=body)))

    def test_full_cta_fence_allows_reassignment(self):
        code=self.compile_body((group(0),Sync(),group(1)))
        self.assertEqual(code.count('bar.sync 1, 64;'),2)

    def test_named_full_population_also_fences(self):
        self.compile_body((group(0),Sync(barrier_id=2,count=128),group(1)))

    def test_fence_cannot_collide_with_preceding_partial_phase(self):
        with self.assertRaisesRegex(CompileError,'different participant groups'):
            self.compile_body((group(0),Sync(barrier_id=1,count=128),group(1)))

    def test_uniform_choice_joins_both_paths(self):
        self.compile_body((group(0),Cond(V('a0'),(Sync(),),(Sync(),)),group(1)))
        for yes,no in (((Sync(),),()),((),(Sync(),))):
            with self.assertRaisesRegex(CompileError,'different participant groups'):
                self.compile_body((group(0),Cond(V('a0'),yes,no),group(1)))

    def test_alternative_cta_groups_do_not_conflict(self):
        self.compile_body((Cond(V('a0'),(group(0),),(group(1),)),Sync(),group(0)))
        with self.assertRaises(CompileError):
            self.compile_body((Cond(V('a0'),(group(0),),(group(1),)),group(0)))

    def test_simultaneous_warp_arms_still_conflict(self):
        with self.assertRaises(CompileError):
            self.compile_body((Cond(B('<',V('warp'),L(2,'int')),
                (Sync(barrier_id=1,count=64),),(Sync(barrier_id=1,count=64),)),))

    def test_zero_trip_loop_does_not_fence(self):
        with self.assertRaises(CompileError):
            self.compile_body((group(0),Loop(Axis('i',Dim(0)),(Sync(),)),group(1)))
        self.compile_body((group(0),Loop(Axis('i',Dim(1)),(Sync(),)),group(1)))

    def test_maybe_zero_trip_loop_does_not_fence(self):
        loop=StridedLoop(Axis('i',Dim(1)),V('a0'),L(1,'int'),(Sync(),))
        with self.assertRaises(CompileError):
            self.compile_body((group(0),loop,group(1)))

    def test_backedge_ownership_is_checked(self):
        body=(group(0),Sync(),group(1))
        self.compile_body((Loop(Axis('i',Dim(1)),body),))
        with self.assertRaises(CompileError):
            self.compile_body((Loop(Axis('i',Dim(2)),body),))
        self.compile_body((Loop(Axis('i',Dim(1000000)),(*body,Sync())),))

    def test_nested_and_conditional_backedges(self):
        inner=Loop(Axis('j',Dim(2)),(group(0),Sync(),group(1),Sync()))
        self.compile_body((Loop(Axis('i',Dim(3)),(inner,)),))
        with self.assertRaises(CompileError):
            self.compile_body((Loop(Axis('i',Dim(3)),(group(0),Sync(),group(1),Cond(V('a0'),(Sync(),)))),))

    def test_static_dead_paths_do_not_own_ids(self):
        self.compile_body((group(0),Cond(L(0,'int'),(group(1),)),group(0)))
        self.compile_body((group(0),Loop(Axis('i',Dim(0)),(group(1),)),group(0)))

    def test_enumerated_paths_match_abstract_transfer(self):
        # Independent concrete trace expansion, including independently chosen
        # branches on every loop iteration. No compiler inference is reused.
        a,b=frozenset(range(64)),frozenset(range(64,128))
        full=a|b;rng=random.Random(9431)
        primitives=[('barrier',1,a,False),('barrier',1,b,False),
                    ('barrier',2,a,False),('barrier',0,full,True),
                    ('barrier',1,full,True)]
        def expand(events):
            paths=[[]]
            for e in events:
                if e[0]=='barrier':part=[[e]]
                elif e[0]=='choice':part=[p for arm in e[1] for p in expand(arm)]
                else:
                    _,body,skip,n=e;part=[];one=expand(body)
                    for count in (range(n+1) if skip else (n,)):
                        for product in itertools.product(one,repeat=count):
                            part.append([e for seq in product for e in seq])
                paths=[a+b for a in paths for b in part]
            return paths
        def concrete(path):
            owners={}
            for _,identifier,cohort,full in path:
                if identifier in owners and owners[identifier]!=cohort:return False
                if full:owners.clear()
                else:owners[identifier]=cohort
            return True
        for _ in range(2000):
            events=[]
            for _ in range(3):
                pair=[rng.choice(primitives) for _ in range(2)]
                choice=('choice',([pair[0]],[pair[1]]))
                events.append(rng.choice([pair[0],choice,('loop',[choice],bool(rng.randrange(2)),rng.randrange(4))]))
            expected=all(concrete(p) for p in expand(events))
            try:validate(events);actual=True
            except ValueError:actual=False
            self.assertEqual(actual,expected,events)


if __name__=='__main__':unittest.main()
