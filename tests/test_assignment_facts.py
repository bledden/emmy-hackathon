"""Exact participation facts must follow pure SSA assignments, not source names."""
from dataclasses import replace
import random,unittest
from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr as B,Literal as L,Var as V
from emmy.compiler.ir.kernel.ir import Sync
from emmy.compiler.ir.stmt.blocks import Cond,Loop
from emmy.compiler.ir.stmt.leaves import Let,Assign,Accum
import test_uniform_conditions as fixtures
from ptxgen.compiler import compile_file,CompileError
from ptxgen.participation import values
from index_model import execute
from test_swizzles import emitter

class AssignmentFactTests(unittest.TestCase):
 stage=fixtures.UniformConditionTests.stage
 def compile(self,body):return compile_file(self.stage(lambda t:replace(t,body=body)))
 def test_shift_cohort(self):
  p=B('==',B('>>',V('warp'),L(1,'int')),L(0,'int'))
  self.assertIn('bar.sync 1, 64',self.compile((Cond(p,(Sync(1,64),)),)))
 def test_assign_alias_chain_and_static_step(self):
  body=(Let('one',L(1,'int')),Assign('group','right_shift',('warp','one'),dtype='i32'),
        Assign('copy','copy',('group',),dtype='i32'),
        Cond(B('==',V('copy'),L(0,'int')),(Sync(1,64),)))
  self.assertIn('bar.sync 1, 64',self.compile(body))
 def test_self_shadow_reads_old_fact(self):
  body=(Let('copy',V('warp')),Let('one',L(1,'int')),Assign('copy','right_shift',('copy','one'),dtype='i32'),
        Cond(B('==',V('copy'),L(0,'int')),(Sync(1,64),)))
  self.assertIn('bar.sync 1, 64',self.compile(body))
 def test_lane_divergence_and_wrong_counts_still_refuse(self):
  for src,count in [('lane',64),('warp',32)]:
   with self.assertRaises(CompileError):
    self.compile((Let('one',L(1,'int')),Assign('group','right_shift',(src,'one'),dtype='i32'),
      Cond(B('==',V('group'),L(0,'int')),(Sync(1,count),))))
 def test_backedge_kills_captured_assign_facts(self):
  body=(Let('one',L(1,'int')),Assign('group','right_shift',('warp','one'),dtype='i32'),
    Loop(Axis('i',Dim(2)),(Cond(B('==',V('group'),L(0,'int')),(Sync(1,64),)),
      Accum('group','one',dtype='i32',axes=('i',))),seed=False))
  with self.assertRaises(CompileError):self.compile(body)
 def test_shift_facts_match_emitted_fixed_width_results(self):
  rng=random.Random(16016)
  for op in ('<<','>>'):
   e=emitter();a=e.bind('a','s32');b=e.bind('b','s32');expr=B(op,V('a'),V('b'));out=e.expression(expr)
   for x,n in [(x,n) for x in (-2**31,-1,0,1,2**31-1) for n in (-1,0,1,31,32,33)]+[(rng.randrange(-2**31,2**31),rng.randrange(0,40)) for _ in range(1000)]:
    expected=execute(e.code,{a:x,b:n})[0][out]
    expected=(expected+2**31)%2**32-2**31
    self.assertEqual(values(expr,{'a':(x,),'b':(n,)},1),(expected,))
