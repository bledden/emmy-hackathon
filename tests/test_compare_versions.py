"""Offline failure probes for the optional finite-f32 comparison helper."""
import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import requests

from tools import compare_versions as runner


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = self.root / 'case'
        self.case.mkdir()
        self.kernel = dict(entry='k', arg_order=['x', 'y'], grid=[1, 1, 1], block=[32, 1, 1],
                           smem_bytes=0, zero_outputs=[], args=[
                               dict(name='x', dtype='f32', shape=[2], data=[1, 2]),
                               dict(name='y', dtype='f32', shape=[2], output=True)])
        self.meta = dict(kernels=[self.kernel], rtol=0, atol=0)
        self.save()
        self.ref = {'y': np.array([1, 2], dtype=np.float32)}
        np.savez(self.case / 'expected.npz', **self.ref)
        (self.case / 'k.ir.json').write_text('{}')
        self.compiler = self.root / 'compile.py'
        self.compiler.write_text('print(".version 8.0")')
        self.out = self.root / 'out'
        self.argv = [str(self.case), '--baseline', str(self.compiler), '--candidate', str(self.compiler),
                     '--out', str(self.out)]
        self.body = {k: self.kernel[k] for k in ('entry', 'grid', 'block', 'smem_bytes')}
        self.body['iters'] = 50
        self.result = dict(self.body, ok=True, median_us=1.0, outputs={
            'y': dict(dtype='f32', shape=[2], truncated=False, values=[1.0, 2.0])})

    def save(self):
        (self.case / 'meta.json').write_text(json.dumps(self.meta))

    def service(self, failure_at=None, failure=None):
        session = Mock()
        def post(url, **kwargs):
            body = kwargs['json']
            self.assertFalse(kwargs['allow_redirects'])
            self.assertEqual(kwargs['timeout'], (10, 300))
            if session.post.call_count == failure_at:
                if isinstance(failure, BaseException):
                    raise failure
                data = failure
            elif url.endswith('/compile'):
                data = dict(ok=True, arch='sm_120', cubin_id='cubin', entry_points=['k'])
            else:
                data = copy.deepcopy(self.result)
            return Mock(status_code=200, text=json.dumps(data), json=lambda: data)
        session.post.side_effect = post
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=False)
        return session

    def test_preparation_never_constructs_network_session(self):
        with patch.object(runner.requests, 'Session') as session:
            runner.main(self.argv)
        session.assert_not_called()
        self.assertEqual(len(list(self.out.glob('*.ptx'))), 2)
        self.assertFalse(list(self.out.glob('request-*')))

    def test_all_cases_preflight_before_network(self):
        # Even an unsupported second case stops before the first compile request.
        other = self.root / 'bad'
        other.mkdir()
        self.kernel['args'][0]['dtype'] = 'i64'
        (other / 'meta.json').write_text(json.dumps(self.meta))
        with patch.object(runner.requests, 'Session') as session:
            with self.assertRaisesRegex(ValueError, 'f32 buffers only'):
                runner.main([str(self.case), str(other), *self.argv[1:], '--execute'])
        session.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_reference_dtype_and_completeness_are_checked(self):
        for ref in ({'y': np.array([2**53+1, 2], dtype=np.int64)},
                    {'y': np.array([1, 2], dtype=np.float64)},
                    {'y': np.array([1, 2], dtype=np.float32), 'extra': np.zeros(1)},
                    {'y': np.array([np.nan, 2], dtype=np.float32)}):
            with self.subTest(ref=list(ref)):
                np.savez(self.case / 'expected.npz', **ref)
                with self.assertRaises(ValueError):
                    runner.load_case(self.case)

    def test_success_is_exactly_two_compiles_and_six_runs(self):
        session = self.service()
        with patch.object(runner.requests, 'Session', return_value=session):
            runner.main(self.argv + ['--execute'])
        self.assertEqual(session.post.call_count, 8)
        self.assertFalse(session.trust_env)
        paths = [call.args[0].rsplit('/', 1)[-1] for call in session.post.call_args_list]
        self.assertEqual(paths, ['compile'] * 2 + ['run'] * 6)
        self.assertEqual(len(list(self.out.glob('request-*'))), 8)

    def test_failures_at_every_request_stop_without_retry(self):
        for position in range(1, 9):
            with self.subTest(position=position):
                out = self.root / f'failed-{position}'
                argv = self.argv[:-1] + [str(out), '--execute']
                session = self.service(position, {'ok': False, 'stage': 'busy'})
                with patch.object(runner.requests, 'Session', return_value=session):
                    with self.assertRaisesRegex(RuntimeError, 'without retry'):
                        runner.main(argv)
                self.assertEqual(session.post.call_count, position)
                self.assertEqual(len(list(out.glob('request-*'))), position)
                self.assertEqual(json.loads((out / 'failure.json').read_text())['attempts'], position)

    def test_network_errors_and_interrupts_are_counted_and_recorded(self):
        for index, error in enumerate((requests.Timeout('timeout'), KeyboardInterrupt())):
            out = self.root / f'error-{index}'
            out.mkdir()
            session = self.service(1, error)
            recorder = runner.Recorder('http://invalid.test', out, 8, session)
            with self.assertRaises(type(error)):
                recorder.post('/compile', {})
            self.assertEqual(recorder.attempts, 1)
            with gzip.open(out / 'request-001.json.gz', 'rt') as f:
                self.assertIn(type(error).__name__, json.load(f)['error'])

    def test_cap_prevents_ninth_request(self):
        self.out.mkdir()
        session = self.service()
        recorder = runner.Recorder('http://invalid.test', self.out, 8, session)
        for _ in range(8):
            recorder.post('/compile', {})
        with self.assertRaisesRegex(RuntimeError, 'cap reached'):
            recorder.post('/compile', {})
        self.assertEqual(session.post.call_count, 8)

    def test_response_errors_fail_even_on_last_run(self):
        wrong = copy.deepcopy(self.result)
        wrong['outputs']['y']['values'] = [1, 3]
        session = self.service(8, wrong)
        with patch.object(runner.requests, 'Session', return_value=session):
            with self.assertRaisesRegex(ValueError, 'incorrect timed output'):
                runner.main(self.argv + ['--execute'])
        self.assertEqual(session.post.call_count, 8)
        self.assertFalse((self.out / 'summary.json').exists())
        self.assertEqual(json.loads((self.out / 'failure.json').read_text())['attempts'], 8)

    def test_response_validation_cannot_round_away_mismatch(self):
        for value in (1.000000001, True, '1', float('inf')):
            with self.subTest(value=value):
                result = copy.deepcopy(self.result)
                result['outputs']['y']['values'][0] = value
                with self.assertRaises(ValueError):
                    runner.check_run(result, self.body, self.ref, self.meta)

    def test_echo_outputs_metadata_and_timing_are_checked(self):
        changes = [lambda r: r.update(iters=True), lambda r: r.update(median_us=-1),
                   lambda r: r['outputs'].update(extra=r['outputs']['y']),
                   lambda r: r['outputs']['y'].update(dtype='i64'),
                   lambda r: r['outputs']['y'].update(shape=[1, 2]),
                   lambda r: r['outputs']['y'].update(truncated=True)]
        for change in changes:
            result = copy.deepcopy(self.result)
            change(result)
            with self.assertRaises(ValueError):
                runner.check_run(result, self.body, self.ref, self.meta)
