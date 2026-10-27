"""Adversarial checks for the new offline replay tool; never contact a service."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location('current_replay',Path(__file__).with_name('check.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)/'packet'
        shutil.copytree(c.HERE,cls.root,ignore=shutil.ignore_patterns('__pycache__'))
        cls.batches=c.read_json(cls.root/'index.json')['batches']

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_packet_and_source_identity(self):
        c.verify_seal(self.root)
        self.assertEqual(c.verify_source(c.REPO),351)

    def test_all_saved_results_and_headline_numbers(self):
        expected={'064':2206032,'068':30720,'069':760,'071':5942}
        for batch in self.batches:
            with self.subTest(batch=batch['id']):
                r=c.replay_batch(self.root,batch,False)
                self.assertEqual(sum(x['values_checked'] for x in r['cases'].values()),expected[batch['id']])
                if batch['id']=='064':
                    m=r['cases']['m01_matmul_512']
                    self.assertEqual(m['mean_us'],{'baseline':31.272,'vector':15.384,'compact':15.480,'nvcc':15.032})
                if batch['mode']=='exact':
                    self.assertTrue(all(v['values_checked']==v['bit_exact_values'] for v in r['cases'].values()))

    def mutate_record(self,batch,number,change):
        plan=c.read_json(self.root/batch['id']/'plan.json')
        p=self.root/batch['id']/plan['requests'][number-1]['receipt'];original=p.read_bytes()
        data=c.strict_json(gzip.decompress(original).decode() if p.suffix=='.gz' else original.decode())
        change(data);raw=json.dumps(data).encode()
        p.write_bytes(gzip.compress(raw) if p.suffix=='.gz' else raw)
        try:
            with self.assertRaises(ValueError):c.replay_batch(self.root,batch,False)
        finally:p.write_bytes(original)

    def test_status_failure_first_and_last_record_each_batch(self):
        for batch in self.batches:
            for n in [1,batch['requests']]:
                with self.subTest(batch=batch['id'],request=n):
                    self.mutate_record(batch,n,lambda r:r.update(status_code=500))

    def test_last_response_and_payload_corruptions(self):
        batch=self.batches[-1]
        def response_change(field,value):
            def change(r):
                d=c.strict_json(r['response_text']);d[field]=value;r['response_text']=json.dumps(d)
            return change
        changes=[response_change('ok',False),response_change('iters',True),response_change('outputs',{}),
                 lambda r:r['request'].update(cubin_id='0'*64)]
        for i,change in enumerate(changes):
            with self.subTest(change=i):self.mutate_record(batch,batch['requests'],change)

    def test_wrong_arch_and_ptx(self):
        def arch(r):
            d=c.strict_json(r['response_text']);d['arch']='sm_89';r['response_text']=json.dumps(d)
        for change in [arch,lambda r:r['request'].update(ptx='ret;')]:
            self.mutate_record(self.batches[0],1,change)

    def test_packet_hash_rejects_edit_and_extra_file(self):
        p=self.root/'064/plan.json';old=p.read_bytes();p.write_bytes(old+b' ')
        try:
            with self.assertRaises(ValueError):c.verify_seal(self.root)
        finally:p.write_bytes(old)
        extra=self.root/'unexpected.txt';extra.write_text('extra')
        try:
            with self.assertRaises(ValueError):c.verify_seal(self.root)
        finally:extra.unlink()

    def test_unsafe_paths_and_duplicate_keys(self):
        for name in ['../escape','/tmp/escape']:
            with self.subTest(path=name),self.assertRaises(ValueError):c.safe_path(self.root,name)
        for value in ['{"a":1,"a":2}','{"v":NaN}','{"v":Infinity}']:
            with self.subTest(json=value),self.assertRaises(ValueError):c.strict_json(value)

    def test_lossy_float_and_integer_tokens(self):
        for value in [True,float('inf'),0.1]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                c.decode({'dtype':'f16','shape':[1],'truncated':False,'values':[value]},'f16',[1])
        for value in [True,2**31,1.5,float(2**63-1)]:
            with self.subTest(value=value),self.assertRaises(ValueError):c.integer(value,32 if value!=float(2**63-1) else 64)

    def test_output_headers_and_extra_outputs(self):
        base={'dtype':'f32','shape':[1],'truncated':False,'values':[1.0]}
        for field,value in [('shape',[1.0]),('dtype','f64'),('truncated',True),('values',[])]:
            bad={**base,field:value}
            with self.subTest(field=field),self.assertRaises(ValueError):c.decode(bad,'f32',[1])

    def test_negative_zero_is_not_exact_positive_zero(self):
        k={'args':[{'name':'out','dtype':'f32','shape':[1],'output':True}]}
        r={'outputs':{'out':{'dtype':'f32','shape':[1],'truncated':False,'values':[-0.0]}}}
        with self.assertRaises(ValueError):c.compare_outputs(r,k,{'out':np.array([0.],np.float32)},{'atol':0.,'rtol':0.},True)

    def test_i64_words_preserve_low_bits_and_extremes(self):
        values=[2**53+1,-2**63,2**63-1,-1,0,1]
        k={'entry':'k','grid':[1,1,1],'block':[32,1,1],'smem_bytes':0,'zero_outputs':[],
           'args':[{'name':'x','dtype':'i64','shape':[6],'data':values},
                   {'name':'out','dtype':'i64','shape':[6],'output':True}]}
        body,words=c.wire_payload(k,0,1);self.assertTrue(words)
        raw=body['args'][0]['data'];self.assertEqual(len(raw),12)
        r={'outputs':{'out':{'dtype':'i32','shape':[12],'truncated':False,'values':raw}}}
        refs={'out':np.array(values,dtype=np.int64)}
        self.assertEqual(c.compare_outputs(r,k,refs,{'atol':0,'rtol':0},True)[:2],(6,6))
        r['outputs']['out']['values'][0]+=1
        with self.assertRaises(ValueError):c.compare_outputs(r,k,refs,{'atol':0,'rtol':0},True)

    def test_tolerance_cannot_hide_large_error(self):
        k={'args':[{'name':'out','dtype':'f32','shape':[1],'output':True}]}
        r={'outputs':{'out':{'dtype':'f32','shape':[1],'truncated':False,'values':[2.0]}}}
        with self.assertRaises(ValueError):c.compare_outputs(r,k,{'out':np.array([1.],np.float32)},{'atol':1e-3,'rtol':1e-3},False)


if __name__=='__main__':unittest.main()
