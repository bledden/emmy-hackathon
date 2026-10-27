"""Independent coordinate references for the two software Volta layouts."""
from types import SimpleNamespace
import unittest
from emmy.compiler.dtype import I32,F16
from emmy.compiler.ir.expr import Literal as L,Var as V
from emmy.compiler.ir.stmt.leaves import Write
from emmy.compiler.ir.kernel.ir import CpAsyncCopy
from ptxgen.compiler import Emitter,CompileError
from ptxgen.swizzles import parse
from index_model import execute


def reference(flat, shape, mode):
    rows,cols=shape[-2:];slab,local=divmod(flat,rows*cols);r,c=divmod(local,cols)
    if mode=='V70A':
        tile,rr=divmod(r,16);cg,cc=divmod(c,4)
        rr=(rr%4)*4+((rr//4)^(2*(tile%2)))
        rr ^= (cg//2)%4
        off=cg*rows*4+(tile*16+rr)*4+cc
    else:
        rt,rr=divmod(r,4);ct,cc=divmod(c,32);cg,ce=divmod(cc,8)
        off=(rt*4+cg)*cols+ct*32+(rr^cg)*8+ce
    return slab*rows*cols+off


def emitter(shape,kind='s32'):
    e=Emitter(None,SimpleNamespace(op=None),{})
    e.buffers['s']=('%rd900',shape,'shared');e.buffer_kinds['s']=kind
    return e


class VoltaLayoutTests(unittest.TestCase):
    def test_complete_maps_and_leading_slabs(self):
        for mode in ('V70A','V70B'):
            for shape in ((16,32),(2,32,64),(3,48,96)):
                count=1
                for d in shape:count*=d
                mapped=[reference(i,shape,mode) for i in range(count)]
                self.assertEqual(sorted(mapped),list(range(count)))
                # Full smallest tile, plus all boundaries and deterministic samples.
                samples=range(count) if count==512 else sorted(set(range(0,count,31))|{count-1,shape[-1]-1,shape[-1]})
                for kind,size in (('f16',2),('s32',4),('f64',8)):
                    e=emitter(shape,kind);v=e.bind('i','s32')
                    idx=[];stride=count
                    from emmy.compiler.ir.expr import BinaryExpr as B
                    for d in shape:
                        stride//=d;idx.append(B('%',B('//',V('i'),L(stride,'int')),L(d,'int')))
                    p,_=e.address('s',tuple(idx),swizzle=parse(mode))
                    for i in samples:
                        self.assertEqual(execute(e.code,{'%rd900':4096,v:i})[0][p],4096+size*mapped[i])

    def test_vectors_cross_group_boundaries_in_multiple_slabs(self):
        for mode in ('V70A','V70B'):
            for start in (3,7,17,515,1041):
                shape=(3,16,32);e=emitter(shape);initial={'%rd900':4096}
                names=tuple(f'v{k}' for k in range(11))
                for k,n in enumerate(names):initial[e.bind(n,'s32')]=100+k
                slab,local=divmod(start,512);row,col=divmod(local,32)
                e.statement(Write('s',tuple(L(x,'int') for x in (slab,row,col)),values=names,value_dtype=I32,swizzle=mode))
                _,memory,_=execute(e.code,initial)
                for k in range(11):self.assertEqual(memory[4096+4*reference(start+k,shape,mode)],100+k)

    def test_async_transformed_alignment_and_contiguous_payload(self):
        for mode,nbytes,offset,expected_chunk in [('V70A',16,0,4),('V70A',8,0,8),('V70B',16,0,16),('V70A',16,1,None)]:
            e=emitter((32,64),'f16');e.buffers['x']=('%rd901',(4096,),'global');e.buffer_kinds['x']='f16'
            e.statement(CpAsyncCopy('s',(L(4,'int'),L(offset,'int')),'x',(L(offset,'int'),),nbytes=nbytes,swizzle=mode))
            copies=[s for s in e.code if s.strip().startswith('cp.async.')]
            if expected_chunk is None:self.assertFalse(copies)
            else:
                self.assertEqual(len(copies),nbytes//expected_chunk)
                self.assertTrue(all(s.endswith(f', {expected_chunk};') for s in copies))

    def test_invalid_rank_and_global_destination_refuse(self):
        for mode in ('V70A','V70B'):
            with self.assertRaisesRegex(CompileError,'two shared slab dimensions'):
                emitter((1024,)).address('s',(L(0,'int'),),swizzle=parse(mode))
            e=emitter((32,32));e.buffers['s']=('%rd900',(32,32),'global')
            with self.assertRaisesRegex(CompileError,'shared destination'):
                e.address('s',(L(0,'int'),L(0,'int')),swizzle=parse(mode))

    def test_atomic_targets_mapped_element(self):
        e=emitter((32,32));e.bind('v','s32')
        e.statement(Write('s',(L(5,'int'),L(17,'int')),values=('v',),value_dtype=I32,swizzle='V70B',atomic=True))
        self.assertTrue(any('atom.shared.add.s32' in s for s in e.code))

    def test_nonbijective_shared_shapes_refuse(self):
        for mode,shape in [('V70A',(8,32)),('V70A',(20,16)),
                           ('V70B',(16,16)),('V70B',(32,48))]:
            with self.assertRaisesRegex(CompileError,'complete bijective'):
                emitter(shape).address('s',(L(0,'int'),L(0,'int')),swizzle=parse(mode))
            e=emitter(shape);e.bind('value','s32')
            with self.assertRaisesRegex(CompileError,'complete bijective'):
                e.statement(Write('s',(L(0,'int'),L(0,'int')),values=('value',),value_dtype=I32,swizzle=mode))
            e=emitter(shape);e.buffers['x']=('%rd901',(4096,),'global');e.buffer_kinds['x']='s32'
            with self.assertRaisesRegex(CompileError,'complete bijective'):
                e.statement(CpAsyncCopy('s',(L(0,'int'),L(0,'int')),'x',(L(0,'int'),),nbytes=16,swizzle=mode))

    def test_eight_column_v70b_is_bijective(self):
        shape=(8,8);e=emitter(shape);v=e.bind('i','s32')
        from emmy.compiler.ir.expr import BinaryExpr as B
        p,_=e.address('s',(B('//',V('i'),L(8,'int')),B('%',V('i'),L(8,'int'))),swizzle=parse('V70B'))
        got=[execute(e.code,{'%rd900':4096,v:i})[0][p] for i in range(64)]
        self.assertEqual(sorted(got),list(range(4096,4352,4)))
        self.assertEqual(got,[4096+4*reference(i,shape,'V70B') for i in range(64)])
