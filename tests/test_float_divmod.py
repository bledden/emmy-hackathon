from types import SimpleNamespace
import unittest
import numpy as np
from ptxgen.compiler import Emitter,CompileError
from divmod_model import execute


def run(op,T,pairs,alias=None):
    kind={np.float16:'f16',np.float32:'f32',np.float64:'f64'}[T]
    e=Emitter(None,SimpleNamespace(op=None),{});a=e.bind('a',kind);b=e.bind('b',kind)
    d=a if alias=='a' else b if alias=='b' else e.reg(kind)
    e.arithmetic(d,op,[a,b]);x=np.asarray([p[0] for p in pairs],T);y=np.asarray([p[1] for p in pairs],T)
    got=execute(e.code,{a:x,b:y},d)
    with np.errstate(all='ignore'):expected=(np.floor_divide if op=='floor_divide' else np.remainder)(x,y)
    equal=(got==expected)|(np.isnan(got)&np.isnan(expected));equal &= (got!=0)|(np.signbit(got)==np.signbit(expected))
    if not equal.all():raise AssertionError((T,op,alias,[(x[i],y[i],got[i],expected[i]) for i in np.flatnonzero(~equal)[:8]]))


class FloatDivmodTests(unittest.TestCase):
    def test_specials_subnormals_and_extremes(self):
        for T in (np.float16,np.float32,np.float64):
            f=np.finfo(T);vals=[0.,-0.,1.,-1.,3.,-3.,.1,-.1,np.inf,-np.inf,np.nan,f.max,-f.max,f.tiny,-f.tiny,f.smallest_subnormal,-f.smallest_subnormal]
            pairs=[(a,b) for a in vals for b in vals]
            for op in ('floor_divide','remainder','mod'):run(op,T,pairs)

    def test_random_bit_patterns(self):
        rng=np.random.default_rng(23023)
        for T,U in ((np.float16,np.uint16),(np.float32,np.uint32),(np.float64,np.uint64)):
            pairs=rng.integers(0,np.iinfo(U).max,size=(512,2),dtype=U).view(T)
            for op in ('floor_divide','remainder'):run(op,T,pairs)

    def test_rounded_quotient_boundaries_and_destination_aliases(self):
        pairs=[(a,b) for a,b in [(1.,.1),(1.,.2),(3.,.1),(6.,.2),(-1.,.1),(-3.,.1),(1.,-np.inf),(0.,-0.),(-0.,1.),(np.nan,1.)]]
        for T in (np.float16,np.float32,np.float64):
            for op in ('floor_divide','remainder','mod'):
                for alias in ('a','b'):run(op,T,pairs,alias)

    def test_wrong_arities_refused(self):
        for kind in ('f16','f32','f64'):
            for op in ('floor_divide','remainder','mod'):
                e=Emitter(None,SimpleNamespace(op=None),{});a=e.bind('a',kind)
                with self.assertRaisesRegex(CompileError,'bad arity'):e.arithmetic(a,op,[a])

if __name__=='__main__':unittest.main()
