"""Execute emitted integer operations against Python exact modular arithmetic."""
import random
import unittest
from types import SimpleNamespace
from ptxgen.compiler import Emitter, CompileError
from index_model import execute

def signed(value, width):
    return (value + (1 << (width - 1))) % (1 << width) - (1 << (width - 1))

class IntegerMathTests(unittest.TestCase):
    def build(self, op, width, alias=None):
        e = Emitter(None, SimpleNamespace(op=None), {})
        a,b,d = [e.reg('s'+str(width)) for _ in range(3)]
        dest = [a,b,d][alias if alias is not None else 2]
        e.arithmetic(dest, op, [a] if op == 'sign' else [a,b])
        return e,a,b,dest

    def test_sign_including_minimum_and_in_place(self):
        for width in (32,64):
            for alias in (None,0):
                e,a,b,d = self.build('sign',width,alias)
                for x in (-(1<<(width-1)),-2**24-1,-1,0,1,2**24+1,(1<<(width-1))-1):
                    got = signed(execute(e.code,{a:x})[0][d],width)
                    self.assertEqual(got,(x>0)-(x<0))

    def test_power_boundary_and_random_modular_values(self):
        rng = random.Random(19023)
        for width in (32,64):
            limit=1<<(width-1)
            pairs=[(a,b) for a in (-limit,-limit+1,-3,-1,0,1,3,2**24+1,limit-1)
                   for b in (0,1,2,3,7,width-1,width,width+1,limit-1)]
            pairs += [(rng.randrange(-limit,limit),rng.randrange(limit)) for _ in range(200)]
            for alias in (None,0,1):
                e,a,b,d=self.build('pow',width,alias)
                for x,y in pairs:
                    got=signed(execute(e.code,{a:x,b:y})[0][d],width)
                    self.assertEqual(got,signed(pow(x,y,1<<width),width),(width,x,y,alias))

    def test_negative_power_exponent_has_bounded_invalid_domain_exit(self):
        for width in (32,64):
            e,a,b,d=self.build('pow',width)
            for y in (-1,-(1<<(width-1))):
                self.assertEqual(execute(e.code,{a:3,b:y})[0][d],0)

    def test_bad_arity_refuses(self):
        for width in (32,64):
            for op,args in (('sign',[]),('sign',[1,2]),('pow',[1]),('pow',[1,2,3])):
                e=Emitter(None,SimpleNamespace(op=None),{});d=e.reg('s'+str(width))
                with self.assertRaisesRegex(CompileError,'bad arity'):
                    e.arithmetic(d,op,[str(x) for x in args])

if __name__=='__main__':unittest.main()
