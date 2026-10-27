"""Boolean results must compare the original values, not rounded integers."""
from types import SimpleNamespace
import unittest
import numpy as np
from emmy.compiler.ir.stmt.leaves import Assign, Let
from emmy.compiler.ir.expr import Literal, Var
from ptxgen.compiler import Emitter, CompileError
import trig_model, wide_model

class BooleanSSATests(unittest.TestCase):
    def test_comparisons_preserve_numeric_operands_and_nan_rules(self):
        refs={op:getattr(np,op) for op in ('equal','not_equal','less','less_equal','greater','greater_equal')}
        for kind,T,model in [('f32',np.float32,trig_model),('f64',np.float64,wide_model)]:
            x=np.array([-np.inf,-1.5,-.1,-0.,0.,.1,1.5,np.inf,np.nan],dtype=T)
            y=np.array([0.,-1.,-.2,0.,-0.,.2,1.,np.inf,np.nan],dtype=T)
            for op,ref in refs.items():
                e=Emitter(None,SimpleNamespace(op=None),{});a=e.bind('a',kind);b=e.bind('b',kind)
                e.statement(Assign('mask',op,('a','b'),dtype='bool'))
                actual=model.execute(e.code,x,e.read('mask'),{b:y})
                np.testing.assert_array_equal(actual,ref(x,y))
                self.assertFalse(any('cvt.rzi.s32' in s for s in e.code))

    def test_logical_ops_truth_test_nonzero_floats(self):
        x=np.array([0.,-0.,.2,-.2,np.nan,np.inf],np.float32)
        y=np.array([1.,0.,0.,-.1,0.,np.nan],np.float32)
        for op in ('logical_and','logical_or','logical_xor','logical_not'):
            e=Emitter(None,SimpleNamespace(op=None),{});a=e.bind('a');b=e.bind('b')
            names=('a',) if op=='logical_not' else ('a','b')
            e.statement(Assign('mask',op,names,dtype='bool'))
            actual=trig_model.execute(e.code,x,e.read('mask'),{b:y})
            np.testing.assert_array_equal(actual,getattr(np,op)(x) if len(names)==1 else getattr(np,op)(x,y))

    def test_boolean_copy_and_where_remain_canonical(self):
        for op in ('copy','where'):
            e=Emitter(None,SimpleNamespace(op=None),{});a=e.bind('a');b=e.bind('b');c=e.bind('c')
            names=('a',) if op=='copy' else ('a','b','c')
            e.statement(Assign('mask',op,names,dtype='bool'))
            x=np.array([0.,.1,np.nan],np.float32);y=np.array([.2,0.,1.],np.float32);z=np.array([0.,1.,0.],np.float32)
            actual=trig_model.execute(e.code,x,e.read('mask'),{b:y,c:z})
            expected=x.astype(bool) if op=='copy' else np.where(x.astype(bool),y.astype(bool),z.astype(bool))
            np.testing.assert_array_equal(actual,expected)

    def test_comparison_result_can_be_floating(self):
        e=Emitter(None,SimpleNamespace(op=None),{});a=e.bind('a');b=e.bind('b')
        e.statement(Assign('a','greater',('a','b'),dtype='f32'))
        x=np.array([.1,-.1,np.nan],np.float32);y=np.zeros(3,np.float32)
        np.testing.assert_array_equal(trig_model.execute(e.code,x,e.read('a'),{b:y}),np.greater(x,y).astype(np.float32))

    def test_uniform_fact_does_not_survive_rebinding(self):
        e=Emitter(None,SimpleNamespace(op=None),{})
        e.body((Let('a',Literal(.5)),Let('b',Literal(0.)),Assign('mask','greater',('a','b'),dtype='bool')))
        self.assertTrue(e.is_uniform(Var('mask')))
        e.bind('a');e.statement(Assign('mask','greater',('a','b'),dtype='bool'))
        self.assertFalse(e.is_uniform(Var('mask')))

    def test_boolean_bitwise_spellings_and_aliases(self):
        x=np.array([-1.,0.,1.,np.nan],np.float32);y=np.array([0.,1.,0.,np.nan],np.float32)
        for op in ('bitwise_and','bitwise_or','bitwise_xor','bitwise_not'):
            e=Emitter(None,SimpleNamespace(op=None),{});a=e.bind('a');b=e.bind('b')
            e.statement(Assign('left','greater',('a','b'),dtype='bool'))
            e.statement(Assign('right','not_equal',('a','b'),dtype='bool'))
            names=('left',) if op=='bitwise_not' else ('left','right')
            e.statement(Assign('left',op,names,dtype='bool'))
            expected=getattr(np,op)(x>y) if len(names)==1 else getattr(np,op)(x>y,x!=y)
            np.testing.assert_array_equal(trig_model.execute(e.code,x,e.read('left'),{b:y}),expected)

    def test_integer_rebinding_cannot_retain_boolean_type(self):
        e=Emitter(None,SimpleNamespace(op=None),{})
        e.body((Let('a',Literal(1.,'float')),Let('b',Literal(0.,'float')),Assign('mask','greater',('a','b'),dtype='bool')))
        e.statement(Let('mask',Literal(1,'int')))
        with self.assertRaisesRegex(CompileError,'boolean bitwise operands'):
            e.statement(Assign('out','bitwise_not',('mask',),dtype='bool'))

    def test_unknown_boolean_operation_refuses_without_guessing(self):
        e=Emitter(None,SimpleNamespace(op=None),{});e.bind('a');e.bind('b')
        with self.assertRaisesRegex(CompileError,'unsupported boolean SSA'):
            e.statement(Assign('mask','add',('a','b'),dtype='bool'))

if __name__=='__main__':unittest.main()
