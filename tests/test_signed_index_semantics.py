"""Signed floor semantics, non-wrapping proofs and binding fact lifetime."""
import random
from types import SimpleNamespace
import unittest
from emmy.compiler.ir.expr import BinaryExpr as B, Literal as L, Var as V
from emmy.compiler.ir.stmt.leaves import Let
from emmy.compiler.ir.stmt.blocks import Cond,StridedLoop
from emmy.compiler.ir.axis import Axis
from emmy.compiler.dim import Dim
from ptxgen.compiler import CompileError,Emitter
from ptxgen.integer_semantics import interval,MIN,MAX
from index_model import execute


def literal(n):return L(n,'int')
def signed(n):return (n+2**31)%2**32-2**31
def emitter():return Emitter(None,SimpleNamespace(op=None),{})

class SignedIndexTests(unittest.TestCase):
    def test_all_signs_exact_and_inexact_against_python(self):
        rng=random.Random(20261022)
        pairs=[(a,b) for a in (-MAX,-19,-6,-1,0,1,6,19,MAX) for b in (MIN,-19,-3,-1,1,3,19,MAX)]
        pairs += [(rng.randint(-MAX,MAX),rng.choice([-1,1])*rng.randint(1,MAX)) for _ in range(1500)]
        for op in ('/','//','%'):
            e=emitter();a=e.bind('a','s32');b=e.bind('b','s32');e.integer_ranges[a]=(-MAX,MAX);e.integer_ranges[b]=(MIN,MAX)
            d=e.expression(B(op,V('a'),V('b')))
            for x,y in pairs:
                got=execute(e.code,{a:x,b:y})[0][d]
                self.assertEqual(signed(got),x%y if op=='%' else x//y,(op,x,y))

    def test_domain_errors_and_int_min_valid_positive_divisors(self):
        for op in ('/','//','%'):
            for a,b in ((4,0),(MIN,-1)):
                with self.assertRaises(CompileError):emitter().expression(B(op,literal(a),literal(b)))
            for b in (1,2,3,MAX):
                e=emitter();d=e.expression(B(op,literal(MIN),literal(b)))
                regs=execute(e.code,{})[0];got=regs[d] if d.startswith('%') else int(d)
                self.assertEqual(signed(got),MIN%b if op=='%' else MIN//b)

    def test_nonnegative_proof_keeps_original_instruction(self):
        for op,opcode in (('/','div'),('//','div'),('%','rem')):
            e=emitter();r=e.bind('a','s32');e.integer_ranges[r]=(0,128)
            e.expression(B(op,V('a'),literal(3)))
            self.assertEqual(len(e.code),1);self.assertIn(opcode+'.s32',e.code[0])

    def test_wrapping_intermediate_cannot_claim_nonnegative(self):
        e=emitter();r=e.bind('a','s32');e.integer_ranges[r]=(0,2)
        expr=B('+',V('a'),literal(MAX));self.assertIsNone(e.index_interval(expr))
        d=e.expression(B('/',expr,literal(3)))
        for a in (0,1,2):self.assertEqual(signed(execute(e.code,{r:a})[0][d]),signed(a+MAX)//3)

    def test_binding_reassignment_and_conditional_shadow_restore(self):
        e=emitter();e.body((Let('a',literal(5)),Let('a',B('-',literal(0),V('a'))),
            Cond(literal(1),(Let('a',literal(7)),Let('inner',B('/',V('a'),literal(2))))),
            Let('outer',B('/',V('a'),literal(2)))))
        regs=execute(e.code,{})[0]
        self.assertEqual(signed(regs[e.values['inner'][0]]),3)
        self.assertEqual(signed(regs[e.values['outer'][0]]),-3)

    def test_negative_induction_bounds_and_expired_fact(self):
        e=emitter();loop=StridedLoop(Axis('j',Dim(5)),start=literal(-4),end=literal(5),step=literal(2),body=(Let('q',B('/',V('j'),literal(3))),))
        e.statement(loop);r=e.values['j'][0]
        self.assertNotIn(r,e.integer_ranges)
        self.assertEqual(signed(execute(e.code,{})[0][e.values['q'][0]]),1)
        e.statement(Let('j',literal(-5)));d=e.expression(B('/',V('j'),literal(3)))
        self.assertEqual(signed(execute(e.code,{})[0][d]),-2)

    def test_interval_bounds_cover_enumerated_values(self):
        rng=random.Random(22)
        for _ in range(1000):
            a=sorted(rng.sample(range(-30,31),2));b=sorted(rng.sample(range(-30,31),2))
            for op in ('+','-','*','/','//','%'):
                expr=B(op,V('a'),V('b'));bounds=interval(expr,{'a':tuple(a),'b':tuple(b)})
                if op in ('/','//','%') and b[0]<=0<=b[1]:self.assertIsNone(bounds);continue
                self.assertIsNotNone(bounds)
                for x in (a[0],sum(a)//2,a[1]):
                    for y in (b[0],sum(b)//2,b[1]):
                        result=expr.eval({'a':x,'b':y});self.assertLessEqual(bounds[0],result);self.assertLessEqual(result,bounds[1])
        self.assertIsNone(interval(B('*',literal(MAX),literal(2)),{}))

if __name__=='__main__':unittest.main()
