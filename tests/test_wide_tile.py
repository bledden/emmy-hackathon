"""Decode large logical grids at selected boundaries without launching GPUs."""
from dataclasses import replace
from pathlib import Path
import json,math,os,random,unittest
from emmy.compiler.graph import Graph
from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from ptxgen.compiler import Emitter,CompileError
from index_model import execute
TASK=Path(os.environ.get('EMMY_TASK_ROOT',Path(__file__).resolve().parents[2]/'upstream/emmy-hackathon-2026'))


def build(shape,aux=0,raster=None,grid=None):
    path=next((TASK/'problems/p00_scalar_add').glob('*.ir.json'));g=Graph.from_dict(json.loads(path.read_text()));n=next(n for n in g.nodes.values() if type(n.op).__name__=='KernelOp')
    tile=replace(n.op.body[0],axes=tuple(Axis(f'i{i}',Dim(x)) for i,x in enumerate(shape)),body=(),block_threads=128,aux_threads=aux,raster_axes=('i0','i1') if raster else None,raster_group=raster[0] if raster else None,raster_orient=raster[1] if raster else 'm')
    n.op=replace(n.op,body=(tile,));meta=json.loads(path.with_name('meta.json').read_text())['kernels'][0];meta.update(block=[128+aux,1,1],grid=[grid or math.ceil(math.prod(shape)/128),1,1]);e=Emitter(g,n,meta);e.compile()
    code=[line for line in e.code if not line.strip().startswith(('ld.param.','cvta.'))]
    return e,code,meta


def decode(shape,gid,raster=None):
    result=[];stride=math.prod(shape)
    for size in shape:stride//=size;result.append(gid//stride%size)
    if raster:
        group,orient=raster;m,n=shape[:2];sub=gid//math.prod(shape[2:])%(m*n)
        first_size,second_size=(m,n) if orient=='m' else (n,m)
        start=sub//(group*second_size)*group;active=min(group,first_size-start);local=sub%(group*second_size)
        a,b=start+local%active,local//active
        result[:2]=[a,b] if orient=='m' else [b,a]
    return result


class WideTileTests(unittest.TestCase):
    def check(self,shape,aux=0,raster=None,grid=None):
        e,code,meta=build(shape,aux,raster,grid);total=math.prod(shape);ctas=meta['grid'][0]
        chosen={0,ctas-1,ctas//2,min(ctas-1,2**31//128-1),min(ctas-1,2**31//128),min(ctas-1,2**32//128)}
        chosen.update(random.Random(24023).randrange(ctas) for _ in range(10))
        for cta in sorted(chosen):
            for tid in (0,1,31,127,*((128,159) if aux else ())):
                regs,_,_=execute(code,{'%tid.x':tid,'%ctaid.x':cta,'%ntid.x':128+aux})
                gid=cta*128+tid%128
                names=[e.read(f'i{i}','s32') for i in range(len(shape))]
                if gid>=total:self.assertTrue(all(name not in regs for name in names))
                else:self.assertEqual([regs[name] for name in names],decode(shape,gid,raster))
        self.assertTrue(any('mul.wide.u32' in s for s in code))

    def test_wide_decode_crosses_signed_and_unsigned32_boundaries(self):
        for shape in ((2**24+1,128),(65537,65537),(2**31,),(4096,3840,256)):
            self.check(shape)

    def test_overlaunch_small_tile_uses_wide_tail_guard(self):
        self.check((1024,),grid=2**24+1)

    def test_auxiliary_band_wraps_before_widening(self):
        self.check((2**24+1,128),aux=32)

    def test_wide_raster_both_orientations_and_ragged_stripes(self):
        for raster in ((3,'m'),(7,'n'),(2**40,'m')):self.check((65537,65537),raster=raster)

    def test_unrepresentable_axis_or_cuda_grid_refuses(self):
        for shape,grid in [((2**31+1,),None),((128,),2**31)]:
            with self.assertRaises(CompileError):build(shape,grid=grid)

if __name__=='__main__':unittest.main()
