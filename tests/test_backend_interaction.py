"""The two optional exact passes compose without changing independent modes."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ptxgen.compiler import compile_file,walk
from ptxgen.index_reuse import reuse_integer_expressions
from ptxgen.optimizations import contract_fma
from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel.ir import TreeHalve
from short_tree_model import execute
from test_short_tree import emit_tree,make_tree,TASK


class BackendInteractionTests(unittest.TestCase):
    def test_reuse_after_actual_tree_including_softmax(self):
        path=next((TASK/'problems/f02_row_softmax').glob('*.ir.json'))
        graph=Graph.from_dict(json.loads(path.read_text()))
        node=next(n for n in graph.nodes.values() if type(n.op).__name__=='KernelOp')
        softmax=next(s for s in walk(node.op.body) if isinstance(s,TreeHalve))
        for n in (4,8,16,32):
            for tree in (make_tree(n,True),replace(softmax,length=n,tid_var='slot')):
                e,regs,mem=emit_tree(tree,True)
                for code in (e.code,contract_fma(e.code)):
                    optimized=reuse_integer_expressions(code)
                    self.assertEqual(execute(code,regs,mem,seed=13),execute(optimized,regs,mem,seed=47))

    def test_cli_modes_and_report_composition(self):
        for case in ('f02_row_softmax','m01_matmul_512'):
            path=next((TASK/'problems'/case).glob('*.ir.json'))
            for tree in (False,True):
                for reuse in (False,True):
                    report=[]
                    ptx=compile_file(path,contract_fma='auto',short_tree=tree,index_reuse=reuse,optimization_report=report)
                    base=compile_file(path,contract_fma='auto',short_tree=tree)
                    expected='\n'.join(reuse_integer_expressions(base.splitlines()))+'\n' if reuse else base
                    self.assertEqual(ptx,expected)
                    if reuse:self.assertEqual(report[-1]['pass'],'index_reuse')
                    p=subprocess.run([sys.executable,str(ROOT/'compile.py'),str(path)],capture_output=True,text=True,
                        env={**os.environ,'EMMY_PTX_CONSTANT_DIVISION':'0','EMMY_PTX_UNROLL':'0','EMMY_PTX_SHARED_LOADS':'off','EMMY_PTX_SHORT_TREE':str(int(tree)),
                             'EMMY_PTX_INDEX_REUSE':str(int(reuse)),'EMMY_PTX_CONTRACT_FMA':'auto'})
                    self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(p.stdout,ptx);self.assertFalse(p.stderr)


if __name__=='__main__':unittest.main()
