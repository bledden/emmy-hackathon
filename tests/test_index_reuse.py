"""Adversarial value-lifetime, control-flow and bit-vector checks for index reuse."""
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ptxgen.index_reuse import reuse_integer_expressions
from ptxgen.compiler import compile_file, CompileError
from index_model import execute
TASK = Path(os.environ.get('EMMY_TASK_ROOT',ROOT.parent/'upstream/emmy-hackathon-2026'))


def lines(*text): return ['    '+s for s in text]


class IndexReuseTests(unittest.TestCase):
    def assert_equivalent(self, code, initial, memory=None):
        report=[]
        optimized=reuse_integer_expressions(code,report)
        for rounding in ('truncate','floor'):
            self.assertEqual(execute(code,initial,memory,rounding),
                             execute(optimized,initial,memory,rounding))
        return optimized,report[0]

    def test_reuses_transitive_values_and_wide_addresses_without_moving_memory(self):
        code=lines('mul.lo.s32 %r1, %r0, 16;', 'mul.wide.s32 %rd1, %r1, 4;',
                   'add.u64 %rd2, %rd0, %rd1;', 'ld.global.f32 %f0, [%rd2];',
                   'mul.lo.s32 %r2, %r0, 16;', 'mul.wide.s32 %rd3, %r2, 4;',
                   'add.u64 %rd4, %rd0, %rd3;', 'ld.global.f32 %f1, [%rd4];',
                   'st.global.f32 [%rd4], %f0;')
        out,report=self.assert_equivalent(code,{'%r0':3,'%rd0':1024},{1216:123})
        self.assertEqual(report['replacements'],3)
        self.assertIn('    mov.b64 %rd4, %rd2;',out)
        self.assertEqual([x for x in code if '.global.' in x],[x for x in out if '.global.' in x])

    def test_operand_overwrite_invalidates_expression(self):
        code=lines('add.s32 %r2, %r0, %r1;', 'mov.s32 %r0, 9;', 'add.s32 %r3, %r0, %r1;')
        out,report=self.assert_equivalent(code,{'%r0':2,'%r1':3})
        self.assertEqual(out,code);self.assertEqual(report['replacements'],0)
        # Negative control: a stale representative demonstrably changes the answer.
        bad=code[:-1]+['    mov.b32 %r3, %r2;']
        self.assertNotEqual(execute(code,{'%r0':2,'%r1':3}),execute(bad,{'%r0':2,'%r1':3}))

    def test_overwritten_representative_is_not_used_but_surviving_copy_is(self):
        for survivor in (False,True):
            code=lines('add.s32 %r2, %r0, %r1;')
            if survivor: code+=lines('add.s32 %r3, %r0, %r1;')
            code+=lines('mov.s32 %r2, 99;', 'add.s32 %r4, %r0, %r1;')
            out,report=self.assert_equivalent(code,{'%r0':2,'%r1':3})
            self.assertEqual(report['replacements'],2 if survivor else 0)
            if survivor:self.assertEqual(out[-1],'    mov.b32 %r4, %r3;')

    def test_inplace_destination_reads_old_inputs_first(self):
        code=lines('add.s32 %r2, %r0, %r1;', 'add.s32 %r0, %r0, %r1;',
                   'add.s32 %r3, %r0, %r1;')
        out,report=self.assert_equivalent(code,{'%r0':0xffffffff,'%r1':2})
        self.assertEqual(report['replacements'],1)
        self.assertEqual(out[1],'    mov.b32 %r0, %r2;')
        self.assertEqual(out[2],code[2])

    def test_predicates_branches_loops_and_barriers_are_boundaries(self):
        for boundary in ('@%p0 mov.s32 %r0, 4;', 'bra L1;', 'bra.uni L1;',
                         'bar.sync 0;', 'bar.warp.sync 0xffffffff;',
                         'shfl.sync.bfly.b32 %r0, %r1, 1, 31, 0xffffffff;',
                         'cp.async.wait_all;', 'mystery.write %r0, %r1;'):
            code=lines('add.s32 %r2, %r0, %r1;',boundary,'add.s32 %r3, %r0, %r1;')
            self.assertEqual(reuse_integer_expressions(code),code)
        code=lines('mov.s32 %r0, 0;', 'mov.s32 %r1, 0;', 'add.s32 %r2, %r0, 7;')
        code+=['L1:']+lines('add.s32 %r2, %r0, 7;', 'add.s32 %r3, %r0, 7;',
                 'add.s32 %r1, %r1, %r3;', 'add.s32 %r0, %r0, 1;',
                 'setp.lt.s32 %p0, %r0, 8;', '@%p0 bra L1;')
        out,report=self.assert_equivalent(code,{})
        self.assertEqual(report['replacements'],1)
        for initial in (0,1):
            branch=lines('add.s32 %r2, %r0, 7;', '@%p0 bra L1;', 'mov.s32 %r0, 19;')
            branch+=['L1:']+lines('add.s32 %r3, %r0, 7;')
            self.assert_equivalent(branch,{'%r0':2,'%p0':initial})

    def test_loads_bitcasts_and_multidestination_instructions_do_not_leave_stale_values(self):
        for change in ('ld.global.u32 %r0, [%rd0];', 'ld.param.u64 %rd0, [param_0];',
                       'mov.b32 %r0, %f0;', 'cvt.u32.u64 %r0, %rd0;',
                       'selp.s32 %r0, 1, 2, %p0;',
                       'ld.global.v2.f32 {%r0, %r1}, [%rd0];',
                       'setp.eq.s32 %p0|%p1, %r0, %r1;'):
            op='add.u64' if 'ld.param' in change else 'add.s32'
            dest,src=('%rd1','%rd0') if 'ld.param' in change else ('%r2','%r0')
            code=lines(f'{op} {dest}, {src}, 7;',change,f'{op} {dest}, {src}, 7;')
            self.assertEqual(reuse_integer_expressions(code),code)
        code=lines('add.s32 %r2, %r0, 7;', 'ld.global.u32 %r0, [%rd0];',
                   'add.s32 %r3, %r0, 7;')
        self.assert_equivalent(code,{'%r0':2,'%rd0':1024},{1024:17})

    def test_width_type_and_operand_order_are_part_of_identity(self):
        code=lines('sub.s32 %r2, %r0, %r1;', 'sub.s32 %r3, %r1, %r0;',
                   'mul.lo.s32 %r4, %r0, %r1;', 'mul.wide.s32 %rd1, %r0, %r1;',
                   'shr.s32 %r5, %r0, 2;', 'shr.u32 %r6, %r0, 2;')
        out,report=self.assert_equivalent(code,{'%r0':0x80000000,'%r1':13})
        self.assertEqual(report['replacements'],0)

    def test_division_and_shift_guards(self):
        for opcode in ('div.s32','rem.s32'):
            for divisor in ('0','-1','%r1','2147483648'):
                code=lines(f'{opcode} %r2, %r0, {divisor};',f'{opcode} %r3, %r0, {divisor};')
                self.assertEqual(reuse_integer_expressions(code),code)
            for number in (0,1,0x80000000,0xffffffff):
                code=lines(f'{opcode} %r2, %r0, 7;',f'{opcode} %r3, %r0, 7;')
                _,report=self.assert_equivalent(code,{'%r0':number})
                self.assertEqual(report['replacements'],1)
        for shift in ('-1','32','%r1'):
            code=lines(f'shl.b32 %r2, %r0, {shift};',f'shl.b32 %r3, %r0, {shift};')
            self.assertEqual(reuse_integer_expressions(code),code)

    def test_randomized_finite_width_programs(self):
        rng=random.Random(70031)
        ops=('add.s32','sub.s32','mul.lo.s32','and.b32','or.b32','xor.b32',
             'shl.b32','shr.s32','shr.u32','div.s32','rem.s32','mul.wide.s32','add.u64')
        replacements=0
        for trial in range(250):
            initial={f'%r{i}':rng.getrandbits(32) for i in range(12)}
            initial.update({f'%rd{i}':rng.getrandbits(64) for i in range(8)})
            code=[]
            for step in range(24):
                op=rng.choice(ops);wide=op=='add.u64'
                operands=[f'%rd{i}' for i in range(8)] if wide else [f'%r{i}' for i in range(12)]
                a,b=rng.choices(operands,k=2)
                if op.startswith(('div.','rem.')):b=str(rng.choice((1,3,7,2147483647)))
                elif op.startswith(('shl.','shr.')):b=str(rng.choice((0,1,16,31)))
                dest=f'%rd{rng.randrange(8)}' if wide or '.wide.' in op else f'%r{rng.randrange(12)}'
                code+=lines(f'{op} {dest}, {a}, {b};')
                if rng.random()<.7:
                    second=f'%rd{rng.randrange(8)}' if dest.startswith('%rd') else f'%r{rng.randrange(12)}'
                    code+=lines(f'{op} {second}, {a}, {b};')
                if rng.random()<.3:code+=lines(f'mov.s32 %r{rng.randrange(12)}, {rng.randrange(-100,100)};')
            _,report=self.assert_equivalent(code,initial)
            replacements+=report['replacements']
        self.assertGreater(replacements,1000)

    def test_cli_opt_in_api_validation_and_fma_stream(self):
        path=next((TASK/'problems/m01_matmul_512').glob('*.ir.json'))
        baseline=compile_file(path,contract_fma='auto')
        report=[]
        candidate=compile_file(path,contract_fma='auto',index_reuse=True,optimization_report=report)
        self.assertGreater(report[0]['replacements'],0)
        self.assertEqual(candidate,'\n'.join(reuse_integer_expressions(baseline.splitlines()))+'\n')
        for old,new in zip(baseline.splitlines(),candidate.splitlines()):
            if old!=new:self.assertRegex(new,r'^    mov\.b(?:32|64) %rd?\d+, %rd?\d+;$')
        for flag in ('0','1','yes'):
            p=subprocess.run([sys.executable,str(ROOT/'compile.py'),str(path)],capture_output=True,text=True,
                env={**os.environ,'EMMY_PTX_CONSTANT_DIVISION':'0','EMMY_PTX_UNROLL':'0','EMMY_PTX_SHARED_LOADS':'off','EMMY_PTX_INDEX_REUSE':flag,'EMMY_PTX_CONTRACT_FMA':'auto'})
            if flag=='yes':self.assertNotEqual(p.returncode,0);self.assertFalse(p.stdout)
            else:self.assertEqual(p.stdout,candidate if flag=='1' else baseline);self.assertEqual(p.returncode,0,p.stderr)
        for mode in (1,'1',None):
            with self.assertRaisesRegex(CompileError,'invalid index-reuse mode'):compile_file(path,index_reuse=mode)


if __name__=='__main__':unittest.main()
