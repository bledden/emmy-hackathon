"""Prove group-aligned quotient and wrap boundaries; reject stale facts."""
from dataclasses import replace
import json,math,random,unittest
from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr as B, Literal as L, Var as V
from emmy.compiler.ir.kernel.ir import Sync
from emmy.compiler.ir.stmt.blocks import Cond,Loop
from emmy.compiler.ir.stmt.leaves import Let,Accum,Load
import test_uniform_conditions as fixtures
from ptxgen.compiler import CompileError,compile_file
from ptxgen.uniformity import quotient_uniform


class QuotientUniformityTests(unittest.TestCase):
    stage=fixtures.UniformConditionTests.stage

    def make(self,body,shape=(('flat',384),),compute=128,aux=0):
        p=self.stage(lambda t:replace(t,axes=tuple(Axis(n,Dim(s)) for n,s in shape),
                     block_threads=compute,aux_threads=aux,body=body),'p00_scalar_add')
        meta=json.loads(p.with_name('meta.json').read_text());meta['kernels'][0].update(
            block=[compute+aux,1,1],grid=[math.prod(s for _,s in shape)//compute,1,1])
        p.with_name('meta.json').write_text(json.dumps(meta));return p

    def guarded(self,d=128,name='flat',warp=False,**args):
        return self.make((Cond(B('//',V(name),L(d,'int')),(Sync(warp=warp),)),),**args)

    def test_flat_quotients_and_nested_predicates(self):
        for op in ('/','//'):
            q=B(op,V('flat'),L(128,'int'))
            compile_file(self.make((Cond(B('==',B('%',q,L(2,'int')),L(1,'int')),(Sync(),)),)))

    def test_warp_domain_does_not_become_cta_domain(self):
        compile_file(self.guarded(32,warp=True))
        with self.assertRaises(CompileError):compile_file(self.guarded(32))

    def test_non_power_of_two_compute_and_auxiliary_groups(self):
        compile_file(self.guarded(96,shape=(('flat',384),),compute=96,aux=32))
        compile_file(self.guarded(128,aux=64))

    def test_inner_stride_can_align_quotient(self):
        compile_file(self.guarded(16,name='outer',shape=(('outer',48),('inner',8))))

    def test_coordinate_wrap_boundary_must_align(self):
        with self.assertRaises(CompileError):
            compile_file(self.guarded(64,name='x',shape=(('outer',2),('x',96)),compute=64))

    def test_constant_quotient_needs_no_group_alignment(self):
        compile_file(self.guarded(16,name='inner',shape=(('outer',24),('inner',16))))

    def test_direct_aliases_and_symbolic_constant_divisor(self):
        body=(Let('copy',V('flat')),Let('again',V('copy')),Let('d',L(128,'int')),
              Let('q',B('//',V('again'),V('d'))),Cond(V('q'),(Sync(),)))
        compile_file(self.make(body))

    def test_rebinding_and_loads_discard_decode(self):
        for change in (Let('copy',V('lane')),Load('copy','x0',(V('lane'),),dtype='i32')):
            with self.assertRaises(CompileError):
                compile_file(self.make((Let('copy',V('flat')),change,
                    Cond(B('//',V('copy'),L(128,'int')),(Sync(),)))))

    def test_lexical_shadow_restores_only_outer_fact(self):
        q=B('//',V('copy'),L(128,'int'))
        compile_file(self.make((Let('copy',V('flat')),
            Cond(L(1,'int'),(Let('copy',V('lane')),)),Cond(q,(Sync(),)))))
        with self.assertRaises(CompileError):
            compile_file(self.make((Let('copy',V('lane')),
                Cond(L(1,'int'),(Let('copy',V('flat')),)),Cond(q,(Sync(),)))))

    def test_backedge_update_discards_alias_and_divisor_proofs(self):
        for name in ('copy','d'):
            body=(Let('copy',V('flat')),Let('d',L(128,'int')),Let('one',L(1,'int')),
                Loop(Axis('i',Dim(2)),(Cond(B('//',V('copy'),V('d')),(Sync(),)),
                     Accum(name,'one',axes=('i',),dtype='i32')),seed=False))
            with self.assertRaises(CompileError):compile_file(self.make(body))

    def test_axis_writes_and_float_divisors_do_not_qualify(self):
        with self.assertRaises(CompileError):
            compile_file(self.make((Let('flat',V('lane')),Cond(B('//',V('flat'),L(128,'int')),(Sync(),)))))
        with self.assertRaises(CompileError):
            compile_file(self.make((Cond(B('//',V('flat'),L(128.0)),(Sync(),)),)))

    def test_offset_can_destroy_alignment(self):
        with self.assertRaises(CompileError):
            compile_file(self.make((Cond(B('//',B('+',V('flat'),L(1,'int')),L(128,'int')),(Sync(),)),)))

    def test_raster_override_is_not_original_decode(self):
        p=self.stage(lambda t:replace(t,axes=(Axis('m',Dim(2)),Axis('n',Dim(3)),Axis('flat',Dim(128))),
                    raster_axes=('m','n'),raster_group=2,body=(Cond(B('//',V('m'),L(2,'int')),(Sync(),)),)))
        # The old mixed-radix fact must not be reused after raster assignment.
        with self.assertRaises(CompileError):compile_file(p)

    def test_boundary_proof_against_full_phase_cycle(self):
        rng=random.Random(23109)
        for _ in range(1500):
            stride=rng.randrange(1,10);extent=rng.randrange(1,25);d=rng.randrange(1,25)
            logical=rng.choice((16,32,48,64,96,128));physical=logical+rng.choice((0,16,32,64))
            for warp in (False,True):
                if not quotient_uniform(stride,extent,d,logical,physical,warp=warp):continue
                for cta in range(math.lcm(stride*extent,logical)//logical):
                    values=[(((cta*logical+t%logical)//stride)%extent)//d for t in range(physical)]
                    groups=[values[t:t+32] for t in range(0,physical,32)] if warp else [values]
                    self.assertTrue(all(len(set(g))==1 for g in groups),
                                    (stride,extent,d,logical,physical,warp,cta))


if __name__=='__main__':unittest.main()
