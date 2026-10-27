"""Offline tests of the small hosted batch's request limits and stop conditions."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_softplus_batch as runner


class BatchTests(unittest.TestCase):
    def run_fake(self, fail=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / 'batch'
            batch.mkdir()
            cases = ['softplus_random', 'softplus_edges', 'softplus_bits']
            x = np.array([-2, -100], dtype=np.float32)
            ref = np.logaddexp(0., x.astype(np.float64)).astype(np.float32)
            for case in cases:
                folder = batch / case
                folder.mkdir()
                (folder / 'kernel.ptx').write_text('offline dummy; never sent')
                (folder / 'meta.json').write_text(json.dumps({'kernels': [{
                    'entry': case, 'args': [{'name': 'add', 'output': True, 'dtype': 'f32', 'shape': [2]}],
                    'grid': [1, 1, 1], 'block': [128, 1, 1], 'smem_bytes': 0, 'zero_outputs': ['add']}]}))
                np.savez(folder / 'expected.npz', add=ref)
                np.save(folder / 'input.npy', x)
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {'files': {str(p.relative_to(batch)): sha(p) for p in batch.rglob('*') if p.is_file()},
                        'sources': {}, 'cases': cases, 'rtol': 1e-6, 'atol': 2e-7, 'tail_rtol': 1.2e-5}
            path = batch / 'manifest.json'
            path.write_text(json.dumps(manifest))
            calls = []

            def fake_post(url, **kwargs):
                calls.append((url, kwargs))
                if fail == 'busy':
                    result = {'ok': False, 'stage': 'busy'}
                elif url.endswith('/compile'):
                    result = {'ok': True, 'cubin_id': 'offline'}
                else:
                    payload = kwargs['json']
                    values = ref.view(np.int32) if payload['args'][0]['dtype'] == 'i32' else ref
                    result = {'ok': True, 'outputs': {'add': {'values': values.tolist(), 'truncated': False}}}
                    if fail == 'wrong_output':
                        result['outputs']['add']['values'] = [42, 42]
                response = unittest.mock.Mock(status_code=200, text=json.dumps(result))
                response.json.return_value = result
                return response

            argv = ['runner', '--batch', str(batch), '--out', str(root / 'out'),
                    '--approved-manifest-sha256', sha(path)]
            if fail == 'changed':
                (batch / cases[0] / 'kernel.ptx').write_text('changed')
            with patch.object(sys, 'argv', argv), patch.object(runner.requests, 'post', fake_post), contextlib.redirect_stdout(io.StringIO()):
                if fail in ('busy', 'changed'):
                    with self.assertRaises((RuntimeError, ValueError)):
                        runner.main()
                else:
                    self.assertEqual(runner.main(), 1 if fail else 0)
            return calls

    def test_exact_six_requests_and_single_execution(self):
        calls = self.run_fake()
        self.assertEqual(len(calls), 6)
        for index, (url, kwargs) in enumerate(calls):
            self.assertTrue(url.endswith('/compile' if index % 2 == 0 else '/run'))
            self.assertFalse(kwargs['allow_redirects'])
            if index % 2:
                self.assertEqual(kwargs['json']['warmup'], 0)
                self.assertEqual(kwargs['json']['iters'], 1)

    def test_busy_stops_without_retry(self):
        self.assertEqual(len(self.run_fake('busy')), 1)

    def test_failed_output_stops_batch(self):
        self.assertEqual(len(self.run_fake('wrong_output')), 2)

    def test_changed_artifact_stops_before_network(self):
        self.assertEqual(len(self.run_fake('changed')), 0)
