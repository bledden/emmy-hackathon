"""Full-range f32 trig: actual emitted operations and independent libm reference."""
from decimal import Decimal,localcontext
from types import SimpleNamespace
import unittest
import numpy as np
from ptxgen.compiler import Emitter,CompileError
from ptxgen.trigonometry import TWO_OVER_PI
from trig_model import execute


def program(op,alias=False):
    e=Emitter(None,SimpleNamespace(op=None),{});x=e.reg('f32');d=x if alias else e.reg('f32')
    e.arithmetic(d,op,[x]);return e.code,d


def check(code,d,x,op):
    y=execute(code,x,d)
    with np.errstate(invalid='ignore'):
        expected=getattr(np,op)(x.astype(np.float64)).astype(np.float32)
    finite=np.isfinite(expected)
    # The same-sign outputs permit an integer ULP distance. NaN payload is free.
    assert np.array_equal(np.signbit(y[finite]),np.signbit(expected[finite]))
    distance=np.abs(y[finite].view(np.int32).astype(np.int64)-expected[finite].view(np.int32).astype(np.int64))
    assert np.all(distance<=1), (op,int(distance.max()),x[finite][distance.argmax()])
    assert np.array_equal(np.isnan(y),np.isnan(expected))
    return int(distance.max(initial=0))


class TrigonometryTests(unittest.TestCase):
    def test_reciprocal_table_independently_derived(self):
        with localcontext() as ctx:
            ctx.prec=110
            def atan(z):
                value=Decimal(0);power=z
                for k in range(200):
                    value+=power/Decimal(2*k+1);power*=-z*z
                return value
            pi=16*atan(Decimal(1)/5)-4*atan(Decimal(1)/239)
            self.assertEqual(int(2/pi*Decimal(2)**256),TWO_OVER_PI)

    def test_random_bits_all_exponents_and_alias(self):
        rng=np.random.default_rng(20261022)
        bits=rng.integers(0,2**32,20000,dtype=np.uint32)
        x=bits.view(np.float32)
        for op in ('sin','cos'):
            for alias in (False,True):
                code,d=program(op,alias);check(code,d,x,op)

    def test_boundaries_zero_subnormal_quadrants_and_extremes(self):
        bits=np.array([0,0x80000000,1,0x80000001,0x7fffff,0x807fffff,
                       0x800000,0x3effffff,0x3f000000,0x3f000001,
                       0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00001,
                       0x7f800001,0xff800001],dtype=np.uint32)
        # Mantissa boundaries at every exponent plus adjacent f32 values to
        # many quarter-turns. References reduce independently through libm.
        e=np.arange(1,255,dtype=np.uint32)<<23
        boundaries=np.concatenate([e,e-1,e+1,e+0x7fffff])
        quarters=(np.arange(-4096,4097)*np.pi/2).astype(np.float32)
        x=np.concatenate([bits.view(np.float32),boundaries.view(np.float32),
                          (-boundaries.view(np.float32)),quarters,
                          np.nextafter(quarters,np.float32(np.inf)),
                          np.nextafter(quarters,np.float32(-np.inf))])
        for op in ('sin','cos'):
            code,d=program(op);check(code,d,x,op)

    def test_arity_and_no_unbounded_approximate_instruction(self):
        e=Emitter(None,SimpleNamespace(op=None),{})
        for op in ('sin','cos'):
            with self.assertRaisesRegex(CompileError,'bad arity'):e.arithmetic('%f1',op,[])
            code,_=program(op)
            self.assertFalse(any('.approx' in line for line in code))
            self.assertTrue(any('mul.wide.u32' in line for line in code))

if __name__=='__main__':unittest.main()
