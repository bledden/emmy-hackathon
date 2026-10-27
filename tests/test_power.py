"""Execute emitted pow PTX against an independent float64 power reference."""
from types import SimpleNamespace
import unittest
import numpy as np
from ptxgen.compiler import Emitter
from ptxgen.power import emit_power
from trig_model import execute


def run(x,y,alias=0):
    e=Emitter(None,SimpleNamespace(op=None),{})
    xr=e.reg('f32');yr=e.reg('f32');dst=xr if alias==1 else yr if alias==2 else e.reg('f32')
    emit_power(e,dst,xr,yr)
    with np.errstate(all='ignore'):
        return execute(e.code,x,dst,{yr:y})


class PowerTests(unittest.TestCase):
    def compare(self,x,y,alias=0):
        x=np.asarray(x,dtype=np.float32);y=np.asarray(y,dtype=np.float32)
        actual=run(x,y,alias)
        with np.errstate(all='ignore'):expected=np.power(x.astype(np.float64),y.astype(np.float64)).astype(np.float32)
        np.testing.assert_array_equal(np.isnan(actual),np.isnan(expected))
        np.testing.assert_array_equal(np.isinf(actual),np.isinf(expected))
        valid=~np.isnan(expected)
        np.testing.assert_array_equal(np.signbit(actual[valid]),np.signbit(expected[valid]))
        # One ULP allows the unavoidable near-midpoint disagreements of a finite
        # polynomial, while preserving subnormals (no blanket absolute epsilon).
        a=(actual[valid].view(np.uint32)&0x7fffffff).astype(np.int64)
        b=(expected[valid].view(np.uint32)&0x7fffffff).astype(np.int64)
        self.assertLessEqual(int(np.max(np.abs(a-b),initial=0)),1)

    def test_special_cartesian_product_and_destination_aliases(self):
        values=np.array([0.,-0.,1.,-1.,2.,-2.,.5,-.5,np.inf,-np.inf,np.nan,
            2**-149,-2**-149,np.finfo(np.float32).max,-np.finfo(np.float32).max,
            3.,-3.,1.5,-1.5,2**24-1,-(2**24-1),2**24,-2**24],dtype=np.float32)
        x,y=np.meshgrid(values,values)
        for alias in (0,1,2):self.compare(x.ravel(),y.ravel(),alias)

    def test_random_bit_patterns_and_near_one_large_exponents(self):
        rng=np.random.default_rng(260422)
        x=rng.integers(0,2**32,50000,dtype=np.uint32).view(np.float32)
        y=rng.integers(0,2**32,50000,dtype=np.uint32).view(np.float32)
        self.compare(x,y)
        x=rng.integers(0x3f7ff000,0x3f801000,20000,dtype=np.uint32).view(np.float32)
        y=rng.uniform(-1e7,1e7,len(x)).astype(np.float32)
        self.compare(x,y)

    def test_integer_negative_fractional_and_underflow_boundaries(self):
        rng=np.random.default_rng(5522)
        x=rng.integers(1,0x7f800000,20000,dtype=np.uint32).view(np.float32)
        y=rng.uniform(-10,10,len(x)).astype(np.float32)
        self.compare(x,y)
        self.compare(-x,rng.integers(-50,51,len(x)).astype(np.float32))
        exponents=np.array([-150.,-149.,-126.,0.,127.,128.],dtype=np.float32)
        y=np.concatenate([np.nextafter(exponents,-np.inf),exponents,np.nextafter(exponents,np.inf)])
        self.compare(np.full(len(y),2.,dtype=np.float32),y)
        self.compare(np.full(len(y),.5,dtype=np.float32),-y)


if __name__=='__main__':unittest.main()
