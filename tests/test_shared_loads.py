"""Address bit-vector equivalence and adversarial load-motion boundaries."""
from types import SimpleNamespace
import random
import unittest
from pathlib import Path
from emmy.compiler.ir.expr import BinaryExpr as B, Literal as L, Var as V
from ptxgen.compiler import Emitter, CompileError, compile_file
from ptxgen.shared_loads import combine, specialize_proof, rebase
from ptxgen.loop_unroll import expand_annotated_loops
from index_model import execute


def setup(mode='off',shape=(4096,)):
    e=Emitter(None,SimpleNamespace(op=None),{'block':[32,1,1]})
    e.shared_loads=mode
    e.buffers['sm']=('%rd900',shape,'shared');e.buffer_kinds['sm']='f32'
    regs={name:e.bind(name,'s32') for name in ('x','y')}
    e.integer_ranges.update({regs['x']:(0,31),regs['y']:(0,31)})
    return e,regs


class SharedAddressTests(unittest.TestCase):
    def test_invalid_policy_is_refused_before_reading_input(self):
        for mode in ('yes','full','v4','',None,True):
            with self.assertRaisesRegex(CompileError,'invalid shared-load mode'):
                compile_file(Path('/nonexistent-shared-load-policy-test'),shared_loads=mode)

    def compare(self,expr,values,shape=(4096,),offset=0):
        a,ar=setup(shape=shape);b,br=setup('address',shape)
        ap,_=a.address('sm',expr,offset);bp,_=b.address('sm',expr,offset)
        for x,y in values:
            av={'%rd900':4096,ar['x']:x&0xffffffff,ar['y']:y&0xffffffff}
            bv={'%rd900':4096,br['x']:x&0xffffffff,br['y']:y&0xffffffff}
            self.assertEqual(execute(a.code,av)[0][ap],execute(b.code,bv)[0][bp],(x,y))
        return a,b,bp

    def test_affine_cancellation_and_adjacent_offsets(self):
        block=B('*',V('x'),L(64,'int'))
        expr=B('-',B('+',block,B('*',V('y'),L(2,'int'))),block)
        for offset in (0,1):
            a,b,p=self.compare((expr,),[(x,y) for x in range(32) for y in range(32)],offset=offset)
            self.assertIn(p,b.shared_address_info)
            proof=b.shared_address_info[p]
            self.assertEqual(proof['alignment'],8)
            self.assertEqual(proof['offset'],offset*4)

    def test_overflowing_canceled_terms_preserve_modular_semantics(self):
        large=B('*',V('x'),L(2**30,'int'))
        expr=B('-',B('+',large,V('y')),large)
        a,b,p=self.compare((expr,),[(x,y) for x in range(32) for y in (0,1,31)])
        self.assertIn(p,b.shared_address_info)

    def test_multidimensional_flattening_and_floor_atoms(self):
        expr=(B('%',V('x'),L(3,'int')),B('+',B('*',V('y'),L(2,'int')),L(1,'int')))
        a,b,p=self.compare(expr,[(x,y) for x in range(32) for y in range(32)],shape=(3,64))
        self.assertIn(p,b.shared_address_info)

    def test_unknown_ranges_and_out_of_bounds_decline(self):
        for expr in (V('x'),B('-',V('x'),L(1,'int')),B('*',V('x'),L(2**30,'int'))):
            e,regs=setup('address')
            if isinstance(expr,V):e.integer_ranges.clear()
            p,_=e.address('sm',(expr,))
            self.assertNotIn(p,e.shared_address_info)

    def test_canceled_invalid_division_is_still_refused(self):
        bad=B('//',V('x'),L(0,'int'))
        for mode in ('off','address'):
            e,_=setup(mode)
            with self.assertRaises(CompileError):e.address('sm',(B('-',bad,bad),))

    def test_positive_offset_must_not_cross_signed_boundary(self):
        e,regs=setup('address',shape=(2**31+64,))
        e.integer_ranges[regs['x']]=(2**31-2,2**31-1)
        pointer,_=e.address('sm',(B('+',V('x'),L(1,'int')),))
        self.assertNotIn(pointer,e.shared_address_info)


class SharedLoadMotionTests(unittest.TestCase):
    def proof(self,offset=0,alignment=16,key=('buffer',())):
        return {'key':key,'offset':offset,'alignment':alignment,'dependencies':frozenset(('%r0','%rd900'))}

    def code(self,middle=()):
        return ['    add.u64 %rd0, %rd900, 0;', '    add.u64 %rd1, %rd900, 4;',
                '    ld.shared.f32 %f0, [%rd0];',*['    '+x for x in middle],
                '    ld.shared.f32 %f1, [%rd1];']

    def test_pair_moves_before_independent_arithmetic(self):
        code=self.code(('add.rn.f32 %f2, %f0, %f3;',))
        proofs={'%rd0':self.proof(),'%rd1':self.proof(4)}
        out=combine(code,proofs)
        self.assertEqual(out,code[:2]+['    ld.shared.v2.f32 {%f0, %f1}, [%rd0];',code[3]])

    def test_side_effect_control_flow_and_register_hazards_decline(self):
        blockers=('st.shared.f32 [%rd2], %f3;','st.global.f32 [%rd2], %f3;',
                  'atom.shared.add.f32 %f3, [%rd2], %f4;','bar.sync 0;',
                  'cp.async.wait_all;','L0:','@%p0 bra L0;',
                  'ld.volatile.shared.f32 %f3, [%rd2];','mystery %r8, 2;',
                  'add.rn.f32 %f2, %f1, %f3;','mov.f32 %f1, %f3;',
                  'mov.s32 %r0, 3;','add.u64 %rd900, %rd900, 4;')
        proofs={'%rd0':self.proof(),'%rd1':self.proof(4)}
        for blocker in blockers:
            code=self.code((blocker,))
            self.assertEqual(combine(code,proofs),code,blocker)

    def test_alignment_gap_and_different_bases_decline(self):
        code=self.code()
        for a,b in ((self.proof(alignment=4),self.proof(4)),
                    (self.proof(4),self.proof(8)),
                    (self.proof(),self.proof(8)),
                    (self.proof(),self.proof(4,key=('other',())))):
            self.assertEqual(combine(code,{'%rd0':a,'%rd1':b}),code)

    def test_four_loads_preserve_symbolic_arithmetic_and_read_set(self):
        code=[];proofs={};initial={'%f10':('accumulator',),'%rd900':4096}
        memory={4096+i*4:('bits',i) for i in range(4)}
        for i in range(4):
            code += [f'    add.u64 %rd{i}, %rd900, {i*4};',
                     f'    ld.shared.f32 %f{i}, [%rd{i}];',
                     f'    fma.rn.f32 %f10, %f{i}, %f{i}, %f10;']
            proofs[f'%rd{i}']=self.proof(i*4);initial[f'%rd{i}']=4096+i*4
        optimized=combine(code,proofs)
        self.assertIn('v4.f32',optimized[1])
        a=execute(code,initial,memory,symbolic_float=True)
        b=execute(optimized,initial,memory,symbolic_float=True)
        self.assertEqual(a[0]['%f10'],b[0]['%f10'])
        self.assertCountEqual(a[2],b[2])

    def test_induction_specialization_keeps_nested_dependencies(self):
        direct=('register','%r7');nested=('binary','%',direct,('literal',3))
        proof={'key':('%rd900',((direct,1),(nested,64))), 'offset':4,
               'alignment':4,'term_alignment':{direct:4,nested:256},
               'dependencies':frozenset(('%rd900','%r7'))}
        result=specialize_proof(proof,'%r7',2,{})
        self.assertEqual(result['offset'],12)
        self.assertEqual(result['alignment'],16)
        self.assertIn('%r7',result['dependencies'])

    def test_cross_iteration_temporaries_and_live_out_values(self):
        code=['    mov.s32 %r0, 0;', '    L0:',
              '    setp.ge.s32 %p0, %r0, 4;', '    @%p0 bra L1;',
              '    mul.wide.s32 %rd0, %r0, 4;',
              '    add.u64 %rd1, %rd900, %rd0;',
              '    ld.shared.f32 %f0, [%rd1];',
              '    fma.rn.f32 %f1, %f0, %f0, %f1;',
              '    add.s32 %r0, %r0, 1;', '    bra L0;', '    L1:']
        proof={'key':('%rd900',((('register','%r0'),1),)), 'offset':0,
               'alignment':4,'term_alignment':{('register','%r0'):4},
               'dependencies':frozenset(('%rd900','%r0'))}
        for live_out in (False,True):
            original=code+(['    add.rn.f32 %f2, %f0, %f1;'] if live_out else [])
            e=Emitter(None,SimpleNamespace(op=None),{'block':[32,1,1]})
            e.counts.update({'s32':1,'u64':901,'f32':3,'pred':1})
            kinds={'%rd0':'u64','%rd1':'u64','%f0':'f32'}
            proofs={'%rd1':proof}
            expanded=expand_annotated_loops(original,[('L0','L1','%r0','%p0',4)],
                      allocate=e.reg,register_kinds=kinds,address_proofs=proofs)
            candidate=combine(expanded,proofs)
            initial={'%rd900':4096,'%f1':('initial',)}
            memory={4096+i*4:('input',i) for i in range(4)}
            a=execute(original,initial,memory,symbolic_float=True)
            b=execute(candidate,initial,memory,symbolic_float=True)
            self.assertEqual(a[0]['%f1'],b[0]['%f1'])
            self.assertCountEqual(a[2],b[2])
            if live_out:self.assertEqual(a[0]['%f2'],b[0]['%f2'])
            else:self.assertTrue(any('v4.f32' in line for line in candidate))

    def test_rebase_pointer_values_and_invalidation(self):
        proofs={'%rd0':self.proof(),'%rd1':self.proof(4)}
        prefix=['    mul.wide.s32 %rd2, %r0, 4;',
                '    add.u64 %rd0, %rd900, %rd2;']
        suffix=['    add.s32 %r1, %r0, 1;',
                '    mul.wide.s32 %rd3, %r1, 4;',
                '    add.u64 %rd1, %rd900, %rd3;']
        original=prefix+suffix
        rewritten=rebase(original,proofs)
        self.assertEqual(rewritten[-1],'    add.u64 %rd1, %rd0, 4;')
        for value in (0,1,31,1024):
            initial={'%r0':value,'%rd900':4096}
            self.assertEqual(execute(original,initial)[0]['%rd1'],execute(rewritten,initial)[0]['%rd1'])
        for mutation in ('mov.s32 %r0, 3;', 'mov.u64 %rd900, 8192;',
                         'mov.u64 %rd0, 3;', 'bar.sync 0;', 'L2:',
                         'st.shared.f32 [%rd4], %f0;', '@%p0 mov.s32 %r0, 1;'):
            code=prefix+['    '+mutation]+suffix
            self.assertEqual(rebase(code,proofs),code,mutation)

    def lifetime_case(self, mutation=()):
        code=['    mul.wide.s32 %rd101, %r0, 4;',
              '    add.u64 %rd102, %rd900, %rd101;',
              '    add.u64 %rd103, %rd102, 4;',
              *['    '+line for line in mutation],
              '    mul.wide.s32 %rd104, %r0, 4;',
              '    add.u64 %rd105, %rd900, %rd104;',
              '    add.u64 %rd106, %rd105, 0;',
              '    ld.shared.f32 %f10, [%rd106];',
              '    ld.shared.f32 %f11, [%rd103];']
        key=('%rd900',((('register','%r0'),1),))
        proofs={'%rd103':self.proof(4,key=key),'%rd106':self.proof(key=key)}
        return code,proofs

    def test_g1_dependency_rewritten_before_first_load_declines(self):
        code,proofs=self.lifetime_case(('add.s32 %r0, %r0, 8;',))
        self.assertEqual(combine(code,proofs),code)
        initial={'%r0':0,'%rd900':4096}
        memory={4096+i*4:('bits',i) for i in range(16)}
        self.assertEqual(execute(code,initial,memory,symbolic_float=True),
                         execute(combine(code,proofs),initial,memory,symbolic_float=True))

    def test_g1_without_dependency_rewrite_still_merges(self):
        code,proofs=self.lifetime_case()
        candidate=combine(code,proofs)
        self.assertIn('    ld.shared.v2.f32 {%f10, %f11}, [%rd106];',candidate)
        initial={'%r0':0,'%rd900':4096};memory={4096:('a',),4100:('b',)}
        self.assertEqual(execute(code,initial,memory,symbolic_float=True),
                         execute(candidate,initial,memory,symbolic_float=True))

    def test_pointer_rewrite_cannot_reactivate_static_proof(self):
        for rewrite in ('add.u64 %rd103, %rd103, 16;',
                        'mov.b64 %rd103, %rd900;',
                        'mov.u64 %rd103, 0;'):
            code,proofs=self.lifetime_case((rewrite,))
            self.assertEqual(combine(code,proofs),code,rewrite)

    def test_boundary_before_first_load_clears_older_pointer_facts(self):
        for boundary in ('L2:', 'bra L2;', '@%p0 mov.s32 %r0, 4;',
                         'mystery %r0, 4;', 'bar.sync 0;'):
            code,proofs=self.lifetime_case((boundary,))
            self.assertEqual(combine(code,proofs),code,boundary)

    def test_pointer_proof_without_definition_is_not_available(self):
        code=['    ld.shared.f32 %f0, [%rd0];','    ld.shared.f32 %f1, [%rd1];']
        self.assertEqual(combine(code,{'%rd0':self.proof(),'%rd1':self.proof(4)}),code)


if __name__=='__main__':unittest.main()
