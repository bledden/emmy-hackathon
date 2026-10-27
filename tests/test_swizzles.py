"""Software swizzles use the pinned element-index mapping, not byte swizzles."""
import random
from types import SimpleNamespace
import unittest
from emmy.compiler.dtype import I32
from emmy.compiler.ir.expr import Literal as L, Var as V
from emmy.compiler.ir.stmt.leaves import Write
from ptxgen.compiler import Emitter, CompileError
from ptxgen.swizzles import parse
from index_model import execute


def emitter(kind='s32',shared=True):
    e=Emitter(None,SimpleNamespace(op=None),{})
    e.buffers['buf']=('%rd900',(4096,), 'shared' if shared else 'global')
    e.buffer_kinds['buf']=kind
    return e


class SwizzleTests(unittest.TestCase):
    def test_modes_and_invalid_spellings(self):
        self.assertIsNone(parse(None));self.assertIsNone(parse('NONE'))
        for mode,mask in [('B32',1),('B64',3),('B128',7)]:
            self.assertEqual(parse(mode),(6,mask))
            for shift in (0,4,7,31):self.assertEqual(parse(f'{mode}@{shift}'),(shift,mask))
        for mode in ('V70A','V70B'): self.assertEqual(parse(mode),mode)
        for mode in ('b128','B256','B128@','B32@-1','B64@32','B128@1.0','NONE@6','V70A@6',42):
            with self.assertRaises(ValueError):parse(mode)

    def test_emitted_address_matches_independent_bit_permutation(self):
        rng=random.Random(9009)
        samples=list(range(1024))+[rng.randrange(2**31) for _ in range(100)]
        for mode,bits in [('B32',1),('B64',2),('B128',3)]:
            for shift in (0,4,6,7,31):
                for kind,size in [('f16',2),('f32',4),('f64',8),('s32',4),('s64',8)]:
                    e=emitter(kind);index=e.bind('i','s32');p,_=e.address('buf',(V('i'),),swizzle=parse(f'{mode}@{shift}'))
                    for value in samples:
                        # Toggle each target bit separately: no helper from the pass.
                        expected=value
                        for bit in range(bits):expected ^= ((value>>(shift+bit))&1)<<(3+bit)
                        got=execute(e.code,{'%rd900':4096,index:value})[0][p]
                        self.assertEqual(got,4096+size*expected,(mode,shift,kind,value))

    def test_vector_crossings_permute_each_logical_value(self):
        for mode in ('B32','B64','B128','B128@0','B128@7'):
            e=emitter();initial={'%rd900':4096};names=tuple(f'v{i}' for i in range(17))
            for i,name in enumerate(names):initial[e.bind(name,'s32')]=100+i
            e.statement(Write('buf',(L(63,'int'),),values=names,value_dtype=I32,swizzle=mode))
            _,memory,trace=execute(e.code,initial)
            shift,mask=parse(mode)
            for i in range(17):
                logical=63+i;expected=logical^(((logical>>shift)&mask)<<3)
                self.assertEqual(memory[4096+4*expected],100+i)
            self.assertEqual(len(trace),17)

    def test_none_is_identical_and_global_swizzles_refuse(self):
        a,b=emitter(),emitter()
        a.address('buf',(L(7,'int'),));b.address('buf',(L(7,'int'),),swizzle=parse('NONE'))
        self.assertEqual(a.code,b.code)
        with self.assertRaisesRegex(CompileError,'shared destination'):
            emitter(shared=False).address('buf',(L(7,'int'),),swizzle=parse('B128'))

    def test_aligned_vector_retained_only_with_a_contiguity_proof(self):
        for mode,vector in [('B128',True),('B64@7',True),('B128@0',False)]:
            e=emitter();names=tuple(f'v{i}' for i in range(4))
            for name in names:e.bind(name,'s32')
            e.statement(Write('buf',(L(64,'int'),),values=names,value_dtype=I32,swizzle=mode))
            self.assertEqual(any('st.shared.v4.s32' in line for line in e.code),vector)

    def test_atomic_write_uses_each_transformed_address(self):
        e=emitter();e.bind('v','s32')
        e.statement(Write('buf',(L(64,'int'),),values=('v',),value_dtype=I32,swizzle='B128',atomic=True))
        self.assertTrue(any('atom.shared.add.s32' in line for line in e.code))
        self.assertTrue(any('xor.b32' in line for line in e.code))


if __name__=='__main__':unittest.main()
