"""Experimental passes: exact integer semantics and ordered loop traces."""
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import I32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr as B, Literal as L, Var as V
from emmy.compiler.ir.stmt.blocks import Loop, Cond
from emmy.compiler.ir.stmt.leaves import Accum, Let, Write
from ptxgen.compiler import Emitter
from ptxgen.constant_division import emit, lower
from ptxgen.loop_unroll import expand_annotated_loops
from index_model import execute


def emitter():
    return Emitter(None, SimpleNamespace(op=None), {'block':[32,1,1]})


def signed(n):
    return (n+2**31) % 2**32 - 2**31


class ConstantDivisionTests(unittest.TestCase):
    def test_boundaries_multiples_and_random_all_signs(self):
        rng = random.Random(590059)
        divisors = [2**i for i in range(1,31)] + [3,5,7,9,17,31,63,65,127,129,65535,65537,2**30-1,2**30+1,2**31-1]
        divisors += [rng.randrange(2,2**31) for _ in range(100)]
        for d in divisors:
            e=emitter(); n=e.bind('n','s32')
            q=emit(e,n,d); r=emit(e,n,d,remainder=True)
            values=[-2**31,-2**31+1,-1,0,1,2**31-2,2**31-1]
            for k in (1,2,3,(2**31-1)//d):
                values += [signed(sign*(k*d+delta)) for sign in (-1,1) for delta in (-1,0,1)]
            values += [rng.randrange(-2**31,2**31) for _ in range(200)]
            for value in values:
                regs,_,_=execute(e.code,{n:value & 0xffffffff})
                expected=abs(value)//d * (-1 if value<0 else 1)
                self.assertEqual(signed(regs[q]),expected,(value,d))
                self.assertEqual(signed(regs[r]),value-expected*d,(value,d))

    def test_floor_correction_still_matches_python(self):
        for d in (2,3,7,64,65537,2**31-1):
            e=emitter();n=e.bind('n','s32')
            q=e.expression(B('//',V('n'),L(d,'int')))
            r=e.expression(B('%',V('n'),L(d,'int')))
            candidate=lower(e.code,e.reg)
            for value in (-2**31,-129,-65,-1,0,1,65,129,2**31-1):
                regs,_,_=execute(candidate,{n:value & 0xffffffff})
                self.assertEqual(signed(regs[q]),value//d)
                self.assertEqual(signed(regs[r]),value%d)

    def test_unknown_negative_and_zero_divisors_keep_old_lowering(self):
        for d in ('%r999','0','-1','-3'):
            a=emitter();a.integer('/', '%r998', d)
            self.assertEqual(a.code,lower(a.code,a.reg))

    def test_destination_alias_and_predicated_instruction(self):
        e=emitter();n=e.bind('n','s32')
        for opcode in ('div','rem'):
            code=[f'    {opcode}.s32 {n}, {n}, 7;']
            candidate=lower(code,e.reg)
            for value in (-2**31,-8,0,8,2**31-1):
                initial={n:value & 0xffffffff}
                self.assertEqual(execute(code,initial)[0][n],execute(candidate,initial)[0][n])
            predicated=['    @%p0 '+code[0].strip()]
            self.assertEqual(predicated,lower(predicated,e.reg))


class AnnotatedLoopTests(unittest.TestCase):
    def make_loop(self,n,unroll=True):
        e=emitter();e.unroll_loops=True
        e.buffers['buf']=('%rd900',(128,),'global');e.buffer_kinds['buf']='s32'
        total=e.bind('total','s32')
        body=(Let('v',B('+',V('total'),V('i'))),
              Write('buf',(V('i'),),values=('v',),value_dtype=I32),
              Accum('total','v',dtype='i32'))
        e.loop(Loop(Axis('i',Dim(n)),body,unroll=unroll,seed=False))
        return e,total

    def test_ordered_stores_and_carried_wrap_are_identical(self):
        for n in (2,3,8,64):
            e,total=self.make_loop(n)
            report=[];candidate=expand_annotated_loops(e.code,e.unroll_regions,report)
            self.assertEqual(report[0]['loops'],1)
            for start in (-2**31,-19,0,1,2**31-1):
                initial={total:start & 0xffffffff,'%rd900':4096}
                self.assertEqual(execute(e.code,initial),execute(candidate,initial))

    def test_expansion_requires_hint_and_bounded_trip_count(self):
        for n,hint in ((0,True),(1,True),(65,True),(4,False)):
            e,_=self.make_loop(n,hint)
            self.assertEqual(e.code,expand_annotated_loops(e.code,e.unroll_regions))

    def test_control_flow_collectives_opaque_and_axis_writes_decline(self):
        e,_=self.make_loop(4);begin,_,index,_,_=e.unroll_regions[0]
        pos=e.code.index('    '+begin+':')+3
        for line in ('bar.sync 0;','cp.async.wait_all;','call target;',
                     'mystery.op %r999, 1;', f'mov.s32 {index}, 3;',
                     '@%p999 bra other;', 'other:'):
            code=list(e.code);code.insert(pos,'    '+line)
            self.assertEqual(code,expand_annotated_loops(code,e.unroll_regions))

    def test_size_budget_declines_without_partial_expansion(self):
        e,_=self.make_loop(64)
        with patch('ptxgen.loop_unroll.MAX_EXPANDED_LINES',10):
            self.assertEqual(e.code,expand_annotated_loops(e.code,e.unroll_regions))


if __name__=='__main__':unittest.main()
