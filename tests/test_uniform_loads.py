"""Read-only identical addresses establish group agreement, not numeric values."""
from dataclasses import replace
import unittest
from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal as L,Var as V
from emmy.compiler.ir.stmt.leaves import Load,Write,Let,Assign
from emmy.compiler.ir.stmt.blocks import StridedLoop,Cond
from emmy.compiler.ir.kernel.ir import Sync,Smem
from ptxgen.compiler import compile_file,CompileError
import test_uniform_conditions as fixtures
I=lambda x:L(x,'int')

class UniformLoadTests(unittest.TestCase):
    stage=fixtures.UniformConditionTests.stage
    def compile(self,body):return compile_file(self.stage(lambda t:replace(t,body=body)))
    def loop(self,warp=False):return StridedLoop(Axis('i',Dim(6)),I(0),V('step'),(Sync(warp=warp),))

    def test_scalar_and_row_address_allow_cta_collectives(self):
        for index in ((I(0),I(0)),(V('a0'),I(0))):
            self.assertIn('bar.sync',self.compile((Load('step','x',index,dtype='i32'),self.loop())))

    def test_warp_address_proves_only_warp_scope(self):
        load=Load('step','x',(I(0),V('warp')),dtype='i32')
        self.assertIn('bar.warp.sync',self.compile((load,self.loop(True))))
        with self.assertRaisesRegex(CompileError,'proven uniform'):self.compile((load,self.loop()))

    def test_mutable_or_shared_source_does_not_gain_facts(self):
        read=Load('step','x',(I(0),I(0)),dtype='i32')
        for body in ((read,self.loop(),Write('x',(I(0),I(0)),values=('step',))),
                     (Smem('s',(128,)),Load('step','s',(I(0),),dtype='i32'),self.loop())):
            with self.assertRaisesRegex(CompileError,'proven uniform'):self.compile(body)

    def test_rebinding_and_lane_addresses_kill_uniformity(self):
        load=Load('step','x',(I(0),I(0)),dtype='i32')
        for change in (Load('step','x',(I(0),V('lane')),dtype='i32'),Assign('step','copy',('lane',),dtype='i32')):
            with self.assertRaisesRegex(CompileError,'proven uniform'):self.compile((load,change,self.loop()))

    def test_capture_index_before_self_shadow_and_boolean_condition(self):
        body=(Let('step',V('a0')),Load('step','x',(V('step'),I(0)),dtype='i32'),Let('zero',I(0)),Assign('mask','greater',('step','zero'),dtype='bool'),Cond(V('mask'),(Sync(),)))
        self.assertIn('bar.sync',self.compile(body))

if __name__=='__main__':unittest.main()
