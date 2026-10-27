"""Execute emitted binary64 math against independent NumPy/libm references."""
from types import SimpleNamespace
import unittest
from decimal import Decimal, localcontext
import numpy as np
from ptxgen.compiler import Emitter, CompileError
from wide_model import execute
from scipy.special import erf, erfc, expit


def run(op, inputs, alias=False, extra=None):
    e=Emitter(None,SimpleNamespace(op=None),{})
    x=e.reg('f64');args=[x];initial={}
    if extra is not None:
        y=e.reg('f64');args.append(y);initial[y]=extra
    dst=y if alias == 'rhs' else x if alias else e.reg('f64')
    e.arithmetic(dst,op,args)
    with np.errstate(all='ignore'):
        return execute(e.code,inputs,dst,initial)


class WideMathTests(unittest.TestCase):
    def check(self,op,x,ref,rtol=2e-13,atol=0):
        with np.errstate(all='ignore'):expected=ref(x)
        for alias in (False,True):
            actual=run(op,x,alias)
            np.testing.assert_array_equal(np.isnan(actual),np.isnan(expected))
            np.testing.assert_array_equal(np.isposinf(actual),np.isposinf(expected))
            np.testing.assert_array_equal(np.isneginf(actual),np.isneginf(expected))
            finite=np.isfinite(expected)
            # A single subnormal ULP is an absolute representational limit,
            # not the broad epsilon that could hide flushing tiny outputs.
            np.testing.assert_allclose(actual[finite],expected[finite],rtol=rtol,atol=max(atol,np.nextafter(0.,1.)))

    def test_exp_range_and_boundaries(self):
        rng=np.random.default_rng(601)
        boundaries=np.array([-746.,-745.1332191019411,-708.3964185322641,0.,709.782712893384,710.])
        x=np.concatenate([rng.uniform(-746,710,30000),boundaries,
                          np.nextafter(boundaries,-np.inf),np.nextafter(boundaries,np.inf),
                          [np.inf,-np.inf,np.nan,0.,-0.,2**-1074,-2**-1074]])
        self.check('exp',x,np.exp)
        self.check('exp_fast',x,np.exp)

    def test_exp2_including_every_binary64_exponent(self):
        boundaries=np.arange(-1075.,1025.)
        x=np.concatenate([boundaries,np.nextafter(boundaries,-np.inf),np.nextafter(boundaries,np.inf),
                          np.random.default_rng(602).uniform(-1076,1025,30000),[np.inf,-np.inf,np.nan]])
        self.check('exp2',x,np.exp2,rtol=3e-15)

    def test_log_full_range_and_near_one(self):
        rng=np.random.default_rng(603)
        bits=rng.integers(1,0x7ff0000000000000,40000,dtype=np.uint64)
        near=rng.integers(0x3fefffffffff0000,0x3ff0000000010000,10000,dtype=np.uint64)
        exponents=np.ldexp(np.ones(2098),np.arange(-1074,1024))
        x=np.concatenate([bits.view(np.float64),near.view(np.float64),exponents,
                          [-1.,0.,-0.,1.,np.inf,-np.inf,np.nan]])
        for op,ref in [('log',np.log),('log2',np.log2),('log10',np.log10)]:
            self.check(op,x,ref,rtol=2e-14)

    def test_sign_rsqrt_specials_and_subnormal_values(self):
        values=np.array([-np.inf,-4.,-2**-1074,-0.,0.,2**-1074,1.,4.,np.finfo(float).max,np.inf,np.nan])
        self.check('sign',values,np.sign,rtol=0)
        self.check('rsqrt',values,lambda x:1/np.sqrt(x),rtol=3e-15)
        with np.errstate(all='ignore'):
            np.testing.assert_array_equal(np.signbit(run('rsqrt',values)),np.signbit(1/np.sqrt(values)))

    def test_arities_refuse(self):
        for op in ('sign','rsqrt','exp','exp_fast','exp2','log','log2','log10','sigmoid','silu','tanh','softplus','erf','gelu','gelu_tanh','sin','cos','pow'):
            for arity in ((0,1,3) if op == 'pow' else (0,2)):
                e=Emitter(None,SimpleNamespace(op=None),{});args=[e.reg('f64') for _ in range(arity)]
                with self.assertRaisesRegex(CompileError,'bad arity'):e.arithmetic(e.reg('f64'),op,args)

    def test_activations_across_ranges_and_cancellation(self):
        rng=np.random.default_rng(604)
        x=np.concatenate([rng.uniform(-700,700,10000),np.linspace(-10,10,10001),
                          rng.uniform(-1e-15,1e-15,1000),[0.,-0.,2**-1074,-2**-1074,np.inf,-np.inf,np.nan]])
        self.check('sigmoid',x,expit)
        self.check('silu',x,lambda x:x*expit(x))
        self.check('softplus',x,lambda x:np.logaddexp(0,x))
        self.check('tanh',x,np.tanh,rtol=4e-15)

    def test_gaussian_families_and_negative_tails(self):
        x=np.concatenate([np.linspace(-37,37,10001),np.linspace(-2.2,2.2,3001),
                          [0.,-0.,2**-1074,-2**-1074,np.inf,-np.inf,np.nan]])
        self.check('erf',x,erf,rtol=8e-15)
        self.check('gelu',x,lambda x:.5*x*erfc(-x/np.sqrt(2)),rtol=8e-13)
        # scipy.expit is independent of the compiler's split-exp implementation.
        y=np.linspace(-12,12,8001)
        self.check('gelu_tanh',y,lambda x:x*expit(2*np.sqrt(2/np.pi)*(x+.044715*x*x*x)),rtol=2e-12)

    def test_trig_reciprocal_constant_independently_regenerated(self):
        from ptxgen.wide_trig import TWO_OVER_PI, TWO_OVER_PI_BITS
        # Gauss-Legendre iteration, independently of the stored hexadecimal
        # constant and the mpmath computation used when it was authored.
        with localcontext() as ctx:
            ctx.prec=420
            a=Decimal(1);b=1/Decimal(2).sqrt();t=Decimal(1)/4;p=Decimal(1)
            for _ in range(12):
                an=(a+b)/2;b=(a*b).sqrt();t-=p*(a-an)**2;a=an;p*=2
            pi=(a+b)**2/(4*t)
            self.assertEqual(int(2/pi*Decimal(2)**TWO_OVER_PI_BITS),TWO_OVER_PI)

    def test_trig_full_binary64_range(self):
        rng=np.random.default_rng(606)
        x=rng.integers(0,2**64,25000,dtype=np.uint64).view(np.float64)
        multiples=np.pi/2*np.arange(-2000,2001)
        powers=np.ldexp(np.ones(2098),np.arange(-1074,1024))
        x=np.concatenate([x,multiples,np.nextafter(multiples,-np.inf),np.nextafter(multiples,np.inf),
                          powers,-powers,[np.inf,-np.inf,np.nan,-0.]])
        for op,ref in [('sin',np.sin),('cos',np.cos)]:self.check(op,x,ref,rtol=3e-14)
        self.assertTrue(np.signbit(run('sin',np.array([-0.])))[0])

    def test_pow_specials_and_full_range(self):
        values=np.array([0.,-0.,1.,-1.,2.,-2.,.5,-.5,np.inf,-np.inf,np.nan,
                         2**-1074,-2**-1074,np.finfo(float).max,-np.finfo(float).max,
                         3.,-3.,.25,-.25,2**53-1,-(2**53-1),2**53,-2**53,1024.,-1075.])
        x,y=np.meshgrid(values,values);rng=np.random.default_rng(607)
        x=np.concatenate([x.ravel(),np.exp(rng.uniform(-700,700,20000)),
                          rng.integers(0x3fefffffffff0000,0x3ff0000000010000,10000,dtype=np.uint64).view(np.float64)])
        y=np.concatenate([y.ravel(),rng.uniform(-10,10,20000),rng.uniform(-1e13,1e13,10000)])
        with np.errstate(all='ignore'):expected=np.power(x,y)
        for alias in (False,True,'rhs'):
            actual=run('pow',x,alias,extra=y)
            np.testing.assert_array_equal(np.isnan(actual),np.isnan(expected))
            np.testing.assert_array_equal(np.isinf(actual),np.isinf(expected))
            valid=~np.isnan(expected)
            np.testing.assert_array_equal(np.signbit(actual[valid]),np.signbit(expected[valid]))
            np.testing.assert_allclose(actual,expected,rtol=5e-13,atol=np.nextafter(0.,1.),equal_nan=True)

    def test_pow_compensated_overflow_and_underflow_boundaries(self):
        largest=np.finfo(float).max
        x=np.array([largest,largest,np.nextafter(largest,0.),2.,2.,.5,.5,1.+2**-52])
        y=np.array([1.,np.nextafter(1.,0.),np.nextafter(1.,0.),-1075.,np.nextafter(-1075.,0.),1075.,np.nextafter(1075.,0.),2**40])
        with np.errstate(all='ignore'):expected=np.power(x,y)
        for alias in (False,True,'rhs'):
            actual=run('pow',x,alias,extra=y)
            np.testing.assert_array_equal(np.isinf(actual),np.isinf(expected))
            np.testing.assert_array_equal(actual==0,expected==0)
            np.testing.assert_allclose(actual,expected,rtol=5e-15,atol=np.nextafter(0.,1.))


if __name__=='__main__':unittest.main()
