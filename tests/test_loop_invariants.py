"""Compare emitted memory traces with motion enabled/disabled, including wrap."""
from types import SimpleNamespace
from unittest.mock import patch
import random
import unittest
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import I32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr as B, Literal as L, Var as V
from emmy.compiler.ir.stmt.blocks import Cond, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Let, Write
from ptxgen.compiler import Emitter
from ptxgen.loop_invariants import candidates, key
from index_model import execute


def setup():
    e = Emitter(None, SimpleNamespace(op=None), {'block':[32,1,1]})
    e.buffers['buf'] = ('%rd900', (128,), 'global')
    e.buffer_kinds['buf'] = 's32'
    return e, {name:e.bind(name,'s32') for name in ('x','y')}


def loop(expr, n=5, extra=()):
    return StridedLoop(Axis('i', Dim(n)), L(0,'int'), L(1,'int'),
                      (Let('v',expr), Write('buf',(V('i'),),values=('v',),value_dtype=I32), *extra), seed=False)


class LoopInvariantTests(unittest.TestCase):
    def compare(self, build, inputs):
        a, ar = setup()
        with patch('ptxgen.loop_invariants.candidates', return_value=[]):
            build(a)
        b, br = setup();build(b)
        self.assertFalse(a.loop_expression_cache);self.assertFalse(b.loop_expression_cache)
        for x,y in inputs:
            av={ar['x']:x,ar['y']:y,'%rd900':4096}
            bv={br['x']:x,br['y']:y,'%rd900':4096}
            ra=execute(a.code,av);rb=execute(b.code,bv)
            self.assertEqual(ra[1:],rb[1:],(x,y))
        return a.code,b.code

    def test_negative_wrap_and_zero_trip_are_equivalent(self):
        rng=random.Random(450045)
        inputs=[(x,1) for x in [-2**31,-2**31+1,-1,0,1,2**31-1]]
        inputs += [(rng.randrange(-2**31,2**31),1) for _ in range(300)]
        expr=B('%',B('//',B('*',V('x'),L(8,'int')),L(64,'int')),L(3,'int'))
        for n in (0,1,5):
            a,b=self.compare(lambda e:e.loop(loop(expr,n)),inputs)
            # Hoisted arithmetic is before the first loop label.
            first=next(i for i,s in enumerate(b) if s.strip().endswith(':'))
            self.assertTrue(any('div.s32' in s or 'rem.s32' in s for s in b[:first]))
            self.assertFalse(any('div.s32' in s or 'rem.s32' in s for s in b[first:]))

    def test_late_or_conditional_write_disables_motion(self):
        update=Accum('x','y',dtype='i32')
        for extra in [(update,),(Cond(V('y'),(update,)),)]:
            a,b=self.compare(lambda e:e.loop(loop(B('%',V('x'),L(3,'int')),extra=extra)),[(x,y) for x in (-8,0,9) for y in (0,1)])
            self.assertEqual(a,b)

    def test_nested_loop_recomputes_at_each_outer_iteration(self):
        inner=loop(B('%',V('x'),L(3,'int')))
        outer=StridedLoop(Axis('outer',Dim(4)),L(0,'int'),L(1,'int'),(inner,Accum('x','y',dtype='i32')),seed=False)
        self.compare(lambda e:e.loop(outer),[(-8,1),(0,1),(9,-1)])

    def test_binding_replaced_between_sibling_loops(self):
        def build(e):
            e.loop(loop(B('%',V('x'),L(3,'int'))))
            e.statement(Let('x',L(17,'int')))
            e.loop(loop(B('%',V('x'),L(3,'int'))))
        self.compare(build,[(-8,1),(0,1),(9,-1)])

    def test_unsafe_divisors_and_unbound_locals_decline(self):
        for divisor in (L(0,'int'),L(-1,'int'),V('y')):
            self.assertIsNone(key(B('%',V('x'),divisor)))
        self.assertEqual(candidates((Let('v',B('%',V('x'),L(3,'int'))),),set()),[])
        self.assertEqual(candidates((Let('v',B('%',V('i'),L(3,'int'))),),{'x'}),[])


if __name__=='__main__':unittest.main()
