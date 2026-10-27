"""Dormant window metadata and noncollective positive runtime strides."""
from dataclasses import replace
import unittest
from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis,Window
from emmy.compiler.ir.expr import Literal as L,Var as V
from emmy.compiler.ir.stmt.blocks import StridedLoop
from emmy.compiler.ir.stmt.leaves import Load,Let,Accum
from emmy.compiler.ir.kernel.ir import Sync
from ptxgen.compiler import compile_file,CompileError
import test_uniform_conditions as fixtures

class RuntimeScheduleTests(unittest.TestCase):
 stage=fixtures.UniformConditionTests.stage
 def compile(self,body):return compile_file(self.stage(lambda t:replace(t,body=body)))
 def test_tile_window_is_dormant_identity_metadata(self):
  a=self.stage(lambda t:t)
  b=self.stage(lambda t:replace(t,axes=tuple(replace(ax,window=Window(parent=ax,base=L(99,'int'),bound=L(-5,'int'),partition=True)) for ax in t.axes)))
  self.assertEqual(compile_file(a),compile_file(b))
 def test_loop_window_does_not_override_explicit_bounds(self):
  def body(window):return (Let('one',L(1.)),StridedLoop(Axis('i',Dim(8),window),L(2,'int'),L(2,'int'),(Accum('sum','one',axes=('i',)),),end=L(7,'int')))
  self.assertEqual(self.compile(body(None)),self.compile(body(Window(base=L(50,'int'),bound=L(100,'int')))))
 def test_runtime_step_emits_positive_and_overflow_guards(self):
  body=(Load('stride','x',(L(0,'int'),L(0,'int')),dtype='i32'),Let('one',L(1.)),StridedLoop(Axis('i',Dim(8)),L(0,'int'),V('stride'),(Accum('sum','one',axes=('i',)),)))
  code=self.compile(body);self.assertIn('setp.le.s32',code);self.assertIn('setp.gt.s64',code)
 def test_lane_varying_runtime_steps_cannot_enter_collective_loops(self):
  with self.assertRaisesRegex(CompileError,'proven uniform'):
   self.compile((Load('stride','x',(L(0,'int'),V('lane')),dtype='i32'),StridedLoop(Axis('i',Dim(8)),L(0,'int'),V('stride'),(Sync(),))))
 def test_static_nonpositive_steps_remain_refused(self):
  for step in (0,-1):
   with self.assertRaisesRegex(CompileError,'positive'):
    self.compile((StridedLoop(Axis('i',Dim(8)),L(0,'int'),L(step,'int'),()),))
