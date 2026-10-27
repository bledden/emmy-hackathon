from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emmy.compiler.dtype import F32
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.kernel.ir import Sync, TreeHalve
from emmy.compiler.ir.stmt.leaves import Assign
from emmy.compiler.ir.stmt.leaves import Let
from emmy.compiler.ir.stmt.blocks import Cond
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.graph import Graph
from ptxgen.compiler import Emitter, compile_file
from short_tree_model import execute

TASK = Path(os.environ.get('EMMY_TASK_ROOT', ROOT.parent/'upstream/emmy-hackathon-2026'))


def make_tree(n, coupled=False, operation='add'):
    states = ('a', 'b') if coupled else ('a',)
    peers = tuple(s + '_peer' for s in states)
    if coupled:
        # Deliberately order-sensitive and coupled: b reads the updated a.
        combine = (Assign('a', ElementwiseImpl('subtract'), ('a', 'b_peer'), F32),
                   Assign('b', ElementwiseImpl('add'), ('a', 'a_peer'), F32))
    else:
        combine = (Assign('a', ElementwiseImpl(operation), ('a', 'a_peer'), F32),)
    return TreeHalve(tuple(s+'_shared' for s in states), states, peers, combine, n, 'slot')


def emit_tree(stmt, fast, *, ready=True, classic=False):
    block = stmt.length * 32 if classic else max(32, stmt.length * 2)
    e = Emitter(None, SimpleNamespace(op=None), {'block':[block,1,1]})
    e.short_tree = fast
    e.optimization_report = []
    e.thread = '%r900'
    e.values = {'slot':('%r901','s32'), 'lane':('%r902','s32')}
    e.visible = set(e.values)
    e.thread_layouts = {'slot':(32 if classic else 1, stmt.length), 'lane':(1,32)}
    if classic:
        stmt = replace(stmt, tid_var='warp')
        e.values['warp'] = e.values['slot']; e.visible.add('warp')
        e.thread_layouts['warp'] = (32,stmt.length)
    for i, (buf, state) in enumerate(zip(stmt.bufs, stmt.state)):
        e.buffers[buf] = (f'%rd{900+i}', (stmt.length,), 'shared')
        e.bind(state)
    for name in ('outer',):
        e.bind(name)
    initial = []
    for tid in range(block):
        regs = {'%r900':tid, '%r901':tid//32 if classic else tid%stmt.length, '%r902':tid%32}
        for i in range(len(stmt.state)):regs[f'%rd{900+i}'] = 4096*(i+1)
        for name,(register,kind) in e.values.items():
            if kind == 'f32':regs[register] = ('old_state', name, tid)
        initial.append(regs)
    memory = {4096*(c+1)+4*i:('input',c,i) for c in range(len(stmt.state))
              for i in range(-1,stmt.length+1)}
    with e.lexical_scope():e.tree(stmt, input_ready=ready)
    return e, initial, memory


class ShortTreeTests(unittest.TestCase):
    def stage_public(self, change):
        source = next((TASK/'problems/r04_row_sum').glob('*.ir.json'))
        graph = Graph.from_dict(json.loads(source.read_text()))
        node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
        node.op = replace(node.op, body=(change(node.op.body[0]),))
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name)/source.name
        path.write_text(json.dumps(graph.to_dict(), default=repr))
        path.with_name('meta.json').write_bytes(source.with_name('meta.json').read_bytes())
        return path

    def test_actual_emitted_tree_matches_ordered_oracle_and_scratch(self):
        for n in (4,8,16,32):
            for coupled, op in ((False,'add'),(False,'subtract'),(False,'minimum'),(True,'add')):
                stmt=make_tree(n,coupled,op)
                baseline, regs, mem=emit_tree(stmt,False)
                fast, fregs, fmem=emit_tree(stmt,True)
                # Independent direct halving over abstract tuples, not PTX.
                expected=dict(mem)
                d=n//2
                while d:
                    old=dict(expected)
                    for i in range(d):
                        a,ap=old[4096+4*i],old[4096+4*(i+d)]
                        if coupled:
                            bp=old[8192+4*(i+d)]
                            a=('sub.rn.f32',a,bp)
                            expected[4096+4*i]=a
                            expected[8192+4*i]=('add.rn.f32',a,ap)
                        else:
                            opcode={'add':'add.rn.f32','subtract':'sub.rn.f32','minimum':'min.NaN.f32'}[op]
                            expected[4096+4*i]=(opcode,a,ap)
                    d//=2
                for seed in (0,17,813):
                    with self.subTest(n=n,coupled=coupled,op=op,seed=seed):
                        bresult,bmem=execute(baseline.code,regs,mem,seed)
                        fresult,fmem=execute(fast.code,fregs,mem,seed)
                        self.assertEqual(bmem,expected)
                        self.assertEqual(fmem,expected)
                        for c,name in enumerate(stmt.state):
                            for lane in fresult:self.assertEqual(lane[fast.values[name][0]],expected[4096*(c+1)])
                            for lane in bresult:self.assertEqual(lane[baseline.values[name][0]],expected[4096*(c+1)])

    def test_classic_warp_writers_can_migrate_to_first_lanes(self):
        stmt=make_tree(4,True)
        for enabled in (False,True):
            e,regs,mem=emit_tree(stmt,enabled,classic=True)
            result,shared=execute(e.code,regs,mem,seed=73)
            if not enabled: expected=shared
            else:self.assertEqual(shared,expected)
            for c,name in enumerate(stmt.state):
                self.assertTrue(all(t[e.values[name][0]]==shared[4096*(c+1)] for t in result))

    def test_real_softmax_combine_preserves_every_operation_and_scratch_slot(self):
        from ptxgen.compiler import walk
        from ptxgen.optimizations import contract_fma
        source=next((TASK/'problems/f02_row_softmax').glob('*.ir.json'))
        graph=Graph.from_dict(json.loads(source.read_text()))
        node=next(n for n in graph.nodes.values() if type(n.op).__name__=='KernelOp')
        tree=next(s for s in walk(node.op.body) if isinstance(s,TreeHalve))
        for n in (4,8,16,32):
            current=replace(tree,length=n,tid_var='slot')
            base,regs,mem=emit_tree(current,False)
            fast,fregs,fmem=emit_tree(current,True)
            for transform in (lambda code:code,contract_fma):
                _,expected=execute(transform(base.code),regs,mem,seed=31)
                actual,shared=execute(transform(fast.code),fregs,fmem,seed=97)
                self.assertEqual(expected,shared)
                for c,name in enumerate(current.state):
                    self.assertTrue(all(t[fast.values[name][0]]==shared[4096*(c+1)] for t in actual))

    def test_bad_mask_and_root_only_writeback_are_detected(self):
        e,regs,mem=emit_tree(make_tree(8),True)
        bad=[s.replace(', 0xffffffff;', ', 0xfffffffe;') for s in e.code]
        with self.assertRaisesRegex(AssertionError,'mask|participants|deadlock'):
            execute(bad,regs,mem)
        # The last predicate is the writeback guard; force only root to write.
        bad=list(e.code)
        candidates=[i for i,s in enumerate(bad) if 'setp.lt.s32' in s and s.endswith(', 4;')]
        bad[candidates[-1]]=bad[candidates[-1]].replace(', 4;', ', 1;')
        _,actual=execute(bad,regs,mem)
        _,expected=execute(e.code,regs,mem)
        self.assertNotEqual(actual,expected)

    def test_coupled_assignment_reordering_is_detected(self):
        e,regs,mem=emit_tree(make_tree(8,True),True)
        bad=list(e.code)
        first=next(i for i,s in enumerate(bad) if 'sub.rn.f32' in s)
        # b must read the newly computed a, not its pre-combine value.
        self.assertIn('add.rn.f32',bad[first+1])
        bad[first],bad[first+1]=bad[first+1],bad[first]
        _,actual=execute(bad,regs,mem)
        _,expected=execute(e.code,regs,mem)
        self.assertNotEqual(actual,expected)

    def test_real_ir_requires_immediate_full_cta_publication(self):
        variants=((), (Sync(warp=True),), (Sync(count=128),), (Sync(barrier_id=1),),
                  (Sync(), Let('unrelated',Literal(1,'int'))),
                  (Cond(Literal(1,'int'),(Sync(),)),))
        for replacement in variants:
            def change(tile):
                return replace(tile, body=tuple(x for s in tile.body
                    for x in (replacement if isinstance(s,Sync) else (s,))))
            path=self.stage_public(change); report=[]
            self.assertEqual(compile_file(path),compile_file(path,short_tree=True,optimization_report=report))
            self.assertEqual(report[0]['reason'],'requires immediately preceding full-CTA Sync')

    def test_real_uniform_branch_allows_fast_path_but_divergence_refuses(self):
        def wrap(tile,condition):
            storage=tuple(s for s in tile.body if type(s).__name__=='Smem')
            work=tuple(s for s in tile.body if type(s).__name__!='Smem')
            return replace(tile,body=(*storage,Cond(condition,work)))
        report=[]
        compile_file(self.stage_public(lambda t:wrap(t,Var('a0'))),short_tree=True,optimization_report=report)
        self.assertTrue(report[0]['applied'])
        from ptxgen.compiler import CompileError
        with self.assertRaisesRegex(CompileError,'CTA-uniform'):
            compile_file(self.stage_public(lambda t:wrap(t,Var('lane'))),short_tree=True)

    def test_outer_reads_and_missing_sync_fall_back_byte_identically(self):
        stmt=make_tree(4)
        outer=replace(stmt,combine_states=(Assign('a',ElementwiseImpl('add'),('a','outer'),F32),))
        for current,ready in ((stmt,False),(outer,True)):
            base,_,_=emit_tree(current,False,ready=ready)
            fast,_,_=emit_tree(current,True,ready=ready)
            self.assertEqual(base.code,fast.code)
            self.assertFalse(fast.optimization_report[0]['applied'])

    def test_length_and_named_barrier_fallback(self):
        for stmt in (make_tree(2),replace(make_tree(4),barrier_id=2,barrier_count=32)):
            base,_,_=emit_tree(stmt,False)
            fast,_,_=emit_tree(stmt,True)
            self.assertEqual(base.code,fast.code)
            self.assertFalse(fast.optimization_report[0]['applied'])
        e,_,_=emit_tree(make_tree(4),False)
        self.assertEqual(e.short_tree_reason(replace(make_tree(4),inner=('slot',4)),True),'segmented tree')

    def test_public_default_and_opt_in_cli_and_report(self):
        path=next((TASK/'problems/r04_row_sum').glob('*.ir.json'))
        report=[]
        base=compile_file(path,contract_fma='auto')
        fast=compile_file(path,contract_fma='auto',short_tree=True,optimization_report=report)
        self.assertEqual(base.count('bar.sync'),3)
        self.assertEqual(fast.count('bar.sync'),2)
        self.assertEqual(len(report),1)
        self.assertTrue(report[0]['applied'])
        for mode,expected in (('0',base),('1',fast)):
            env=dict(os.environ,EMMY_PTX_CONSTANT_DIVISION='0',EMMY_PTX_UNROLL='0',EMMY_PTX_SHARED_LOADS='off',EMMY_PTX_SHORT_TREE=mode,EMMY_PTX_CONTRACT_FMA='auto')
            p=subprocess.run([sys.executable,str(ROOT/'compile.py'),str(path)],capture_output=True,text=True,env=env)
            self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(p.stdout,expected);self.assertFalse(p.stderr)
        env=dict(os.environ,EMMY_PTX_CONSTANT_DIVISION='0',EMMY_PTX_UNROLL='0',EMMY_PTX_SHARED_LOADS='off',EMMY_PTX_SHORT_TREE='yes')
        p=subprocess.run([sys.executable,str(ROOT/'compile.py'),str(path)],capture_output=True,text=True,env=env)
        self.assertNotEqual(p.returncode,0);self.assertFalse(p.stdout)


if __name__=='__main__':unittest.main()
