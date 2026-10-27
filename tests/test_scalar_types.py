"""Typed lowering regressions, including integers that cannot survive f32.

Integer PTX is executed by the existing instruction interpreter and compared
with Python fixed-width arithmetic, independently of the emission rules.
"""
import itertools
import random
import struct
from types import SimpleNamespace
import unittest

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.stmt.blocks import Cond, Loop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.kernel.ir import WarpShuffle
from ptxgen.compiler import CompileError, Emitter, definite_conditional_folds
from ptxgen.scalars import literal, promoted_kind
from index_model import execute


def emitter():
    return Emitter(None, SimpleNamespace(op=None), {})


def signed(value, width):
    return (value + (1 << (width - 1))) % (1 << width) - (1 << (width - 1))


class ScalarTypeTests(unittest.TestCase):
    def test_integer_arithmetic_against_fixed_width_reference(self):
        rng = random.Random(47281)
        for width in (32, 64):
            kind = 's' + str(width)
            limit = 1 << (width-1)
            values = [-limit, -limit+1, -2**24-1, -7, -1, 0, 1, 7, 2**24+1, limit-1]
            pairs = list(itertools.product(values, repeat=2))
            pairs += [(rng.randrange(-limit, limit), rng.randrange(-limit, limit)) for _ in range(300)]
            operations = {'add': lambda a,b:a+b, 'subtract':lambda a,b:a-b,
                          'multiply':lambda a,b:a*b, 'floor_divide':lambda a,b:a//b,
                          'remainder':lambda a,b:a%b, 'bitwise_and':lambda a,b:a&b,
                          'bitwise_or':lambda a,b:a|b, 'bitwise_xor':lambda a,b:a^b}
            for op, reference in operations.items():
                e = emitter(); a,b,d = [e.reg(kind) for _ in range(3)]
                e.arithmetic(d,op,[a,b])
                for x,y in pairs:
                    if y == 0 and op in ('floor_divide','remainder'):continue
                    actual = signed(execute(e.code,{a:x,b:y})[0][d],width)
                    self.assertEqual(actual,signed(reference(x,y),width),(kind,op,x,y))

    def test_integer_shift_counts_clip_before_narrowing(self):
        for width in (32,64):
            kind='s'+str(width)
            for op in ('left_shift','right_shift'):
                e=emitter();a,b,d=[e.reg(kind) for _ in range(3)]
                e.arithmetic(d,op,[a,b])
                counts=[-1,0,1,width-1,width,width+1,(1<<(width-1))-1]
                if width==64:counts += [2**32,2**32+1]
                for x,n in itertools.product((-7,0,7,-(1<<(width-1))),counts):
                    effective=n if 0<=n<width else width
                    expected=(x << effective) if op=='left_shift' else (x >> effective)
                    actual=execute(e.code,{a:x,b:n})[0][d]
                    self.assertEqual(signed(actual,width),signed(expected,width),(width,op,x,n))

    def test_bit_count_is_count_of_magnitude_including_signed_minimum(self):
        for width in (32,64):
            e=emitter();a,d=[e.reg('s'+str(width)) for _ in range(2)]
            e.arithmetic(d,'bitwise_count',[a])
            for x in (-(1<<(width-1)),-13,-1,0,1,13,(1<<(width-1))-1):
                self.assertEqual(execute(e.code,{a:x})[0][d],abs(x).bit_count())

    def test_i64_binary_expressions_preserve_high_bits(self):
        for op,expected in (('+',2**45+7),('*',(2**45+4)*3),('//',(2**45+4)//3),('%',(2**45+4)%3)):
            e=emitter()
            e.body((Let('a',Literal(2**45+4,'i64')),Let('b',Literal(3,'i64')),
                    Let('c',BinaryExpr(op,Var('a'),Var('b')))))
            self.assertEqual(e.values['c'][1],'s64')
            self.assertEqual(execute(e.code,{})[0][e.read('c')],expected)

    def test_literals_preserve_storage_precision(self):
        self.assertEqual(literal(2**53+1,'s64'),str(2**53+1))
        v=1.0+2**-40
        self.assertEqual(struct.unpack('>d',bytes.fromhex(literal(v,'f64')[2:]))[0],v)
        for x in (0.0,-0.0,2**-24,1.0,65504.0):
            self.assertEqual(int(literal(x,'f16')),int.from_bytes(struct.pack('>e',x),'big'))

    def test_half_ternary_uses_bit_selection_and_typed_predicate(self):
        e=emitter()
        e.body((Let('half',TernaryExpr(Literal(1,'int'),Literal(1.0,'f16'),Literal(2.0,'f16'))),
                Let('flag',BinaryExpr('<',Var('half'),Literal(2.0,'f16')))))
        self.assertTrue(any('selp.b16 ' in line for line in e.code))
        self.assertFalse(any('selp.f16 ' in line for line in e.code))

    def test_literal_condition_uses_its_own_precision(self):
        fold=Loop(Axis('i',Dim(1)),(Accum('sum','v',dtype='f64',axes=('i',)),))
        for dtype,value,truth in [('f64',1e-300,True),('f32',1e-300,False),
                                 ('f16',1e-9,False),('f16',2**-24,True),
                                 ('i64',2**32,True),('i32',2**32,False)]:
            c=Cond(Literal(value,dtype),(fold,),())
            self.assertEqual(definite_conditional_folds(c),{'sum'} if truth else set())

    def test_cond_export_ignores_differently_typed_inner_shadow(self):
        outer=Loop(Axis('i',Dim(1)),(Accum('total','v',dtype='i64',axes=('i',)),))
        shadow=Cond(Literal(1,'int'),(Init('total',0.0,dtype='f64'),
                    Loop(Axis('j',Dim(1)),(Accum('total','w',dtype='f64',axes=('j',)),),seed=False)))
        e=emitter()
        e.body((Let('v',Literal(2**40+1,'i64')),Let('w',Literal(2.0,'f64')),
                Cond(Literal(1,'int'),(outer,shadow),(outer,))))
        self.assertEqual(e.values['total'][1],'s64')
        self.assertIn('mov.s64', '\n'.join(e.code))

    def test_cond_export_disagrees_across_live_paths(self):
        e=emitter()
        left=Loop(Axis('i',Dim(1)),(Accum('sum','x',dtype='i64',axes=('i',)),))
        right=Loop(Axis('i',Dim(1)),(Accum('sum','y',dtype='f64',axes=('i',)),))
        with self.assertRaisesRegex(CompileError,'conditional fold types disagree'):
            e.body((Let('x',Literal(1,'i64')),Let('y',Literal(1.0,'f64')),
                    Let('flag',Literal(1,'int')),Cond(Var('flag'),(left,),(right,))))

    def test_constant_cond_exports_only_reachable_arm_type(self):
        for flag in (0,1):
            e=emitter()
            left=Loop(Axis('i',Dim(1)),(Accum('sum','x',dtype='i64',axes=('i',)),))
            right=Loop(Axis('i',Dim(1)),(Accum('sum','y',dtype='f64',axes=('i',)),))
            e.body((Let('x',Literal(1,'i64')),Let('y',Literal(1.0,'f64')),
                    Cond(Literal(flag,'int'),(left,),(right,))))
            self.assertEqual(e.values['sum'][1],'s64' if flag else 'f64')

    def test_declared_collective_boundary_converts_without_bit_reinterpretation(self):
        e=emitter();e.body((Init('acc',1.0,dtype='f16'),))
        e.state_as('acc','f32')
        self.assertEqual(e.values['acc'][1],'f32')
        self.assertIn('cvt.f32.f16','\n'.join(e.code))

    def test_unstamped_collective_combine_uses_collective_dtype(self):
        for dtype,kind in (('f16','f16'),('f64','f64'),('i64','s64')):
            e=Emitter(None,SimpleNamespace(op=None),{'block':[32,1,1]})
            e.thread=e.reg('s32')
            e.body((Init('acc',1,dtype=dtype),WarpShuffle(
                state=('acc',),state_b=('partner',),dtype=dtype,length=32,
                combine_states=(Assign('acc',ElementwiseImpl('add'),('acc','partner')),))))
            self.assertEqual(e.values['acc'][1],kind)
            self.assertNotIn('partner',e.visible)

    def test_default_assign_promotion_is_distinct_from_explicit_dtype(self):
        self.assertEqual(promoted_kind('multiply',['s32','s64']),'s64')
        self.assertEqual(promoted_kind('divide',['s64','s64']),'f32')
        self.assertEqual(promoted_kind('add',['f16','f16']),'f16')
        self.assertEqual(promoted_kind('add',['f64','f64']),'f32')

    def test_bitcast_moves_bits_without_numerical_conversion(self):
        for source,target,width in [('i32','f32',32),('f32','i32',32),
                                    ('i64','f64',64),('f64','i64',64)]:
            e=emitter()
            e.body((Init('source',1,dtype=source),
                    Assign('bits',ElementwiseImpl('bitcast'),('source',),target)))
            self.assertIn(f'mov.b{width} {e.read("bits")}, {e.read("source")};',
                          '\n'.join(e.code))
            self.assertNotIn('cvt.', '\n'.join(e.code))
        e=emitter()
        with self.assertRaisesRegex(CompileError,'equal storage widths'):
            e.body((Init('source',1,dtype='i32'),
                    Assign('bits',ElementwiseImpl('bitcast'),('source',),'f64')))


if __name__=='__main__':unittest.main()
