"""One explicitly approved, frozen softplus batch; at most six HTTP requests.

Do not invoke until Blake approves this specific batch. No retries, warmups,
benchmark loops, regeneration, or other cases. Stops on the first failure.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import requests


def check_output(kind, got, expected, x, manifest):
    finite = np.isfinite(expected)
    checks = {
        'nan_class': bool(np.array_equal(np.isnan(got), np.isnan(expected))),
        'positive_inf': bool(np.array_equal(np.isposinf(got), np.isposinf(expected))),
        'negative_inf': bool(np.array_equal(np.isneginf(got), np.isneginf(expected))),
        'finite_class': bool(np.array_equal(np.isfinite(got), finite)),
        'finite_values': bool(np.allclose(got[finite].astype(np.float64), expected[finite].astype(np.float64),
                                         rtol=manifest['rtol'], atol=manifest['atol'])),
    }
    if kind == 'softplus_bits':
        zeros = finite & (expected == 0)
        checks['zero_bits'] = bool(np.array_equal(got.view(np.uint32)[zeros], expected.view(np.uint32)[zeros]))
        tail = (x >= -87) & (x <= -2)
        checks['negative_tail_relative'] = bool(tail.any() and np.allclose(
            got[tail].astype(np.float64), expected[tail].astype(np.float64),
            rtol=manifest['tail_rtol'], atol=0))
        # A loose absolute tolerance would accept total loss of subnormal tails.
        tiny = finite & (expected > 0) & (expected < np.finfo(np.float32).tiny)
        distance = np.abs(got.view(np.uint32)[tiny].astype(np.int64) -
                          expected.view(np.uint32)[tiny].astype(np.int64))
        # 160 ULP covers exponent-argument rounding (~1e-5 * 2^23) near
        # the normal boundary plus ex2 error; fixed before hardware execution.
        checks['subnormal_within_160_ulp'] = bool(tiny.any() and np.all(distance <= 160))
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--approved-manifest-sha256', required=True,
                   help='Hash of the exact batch Blake approved; not a substitute for his approval')
    p.add_argument('--server', default='http://runner.riftstack.ai:48701')
    a = p.parse_args()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = a.batch / 'manifest.json'
    if sha(manifest_path) != a.approved_manifest_sha256:
        raise ValueError('Manifest differs from approved batch')
    manifest = json.loads(manifest_path.read_text())
    for name, expected_hash in manifest['files'].items():
        if sha(a.batch / name) != expected_hash:
            raise ValueError('Changed batch file: ' + name)
    root = Path(__file__).resolve().parents[1]
    for name, expected_hash in manifest['sources'].items():
        if sha(root / name) != expected_hash:
            raise ValueError('Changed source: ' + name)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'approved-manifest.json').write_bytes(manifest_path.read_bytes())
    serial, summary = 0, []

    def post(endpoint, payload):
        nonlocal serial
        if serial >= 6:
            raise RuntimeError('Six-request cap reached')
        serial += 1
        record = {'endpoint': endpoint, 'request': payload}
        path = a.out / f'request-{serial:02d}.json'
        path.write_text(json.dumps(record, indent=2))
        # requests does not retry these POSTs; redirects are disabled too.
        response = requests.post(a.server + endpoint, json=payload,
                                 timeout=(10, 120), allow_redirects=False)
        record['status_code'] = response.status_code
        record['response_text'] = response.text
        path.write_text(json.dumps(record, indent=2))
        if response.status_code != 200:
            raise RuntimeError('HTTP failure; no retry')
        result = response.json()
        if not result.get('ok') or result.get('stage') == 'busy':
            raise RuntimeError('Runner failure/busy; no retry')
        return result

    for name in manifest['cases']:
        folder = a.batch / name
        meta = json.loads((folder / 'meta.json').read_text())
        k = meta['kernels'][0]
        compiled = post('/compile', {'ptx': (folder / 'kernel.ptx').read_text()})
        bit_readback = name == 'softplus_bits'
        args = [dict(arg, dtype='i32') if bit_readback and arg.get('output') else arg for arg in k['args']]
        expected = np.load(folder / 'expected.npz')['add']
        body = {key: k[key] for key in ('entry', 'grid', 'block', 'smem_bytes', 'zero_outputs')}
        body.update(cubin_id=compiled['cubin_id'], args=args, warmup=0, iters=1, max_return=expected.size)
        run = post('/run', body)
        output = run['outputs']['add']
        if output.get('truncated'):
            raise RuntimeError('Truncated output; no retry')
        got = np.asarray(output['values'], dtype=np.int32 if bit_readback else np.float32)
        if bit_readback:
            got = got.view(np.float32)
        if got.shape != expected.shape:
            raise RuntimeError('Unexpected output shape; no retry')
        checks = check_output(name, got, expected, np.load(folder / 'input.npy'), manifest)
        row = {'case': name, 'pass': all(checks.values()), 'checks': checks}
        summary.append(row)
        (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(row), flush=True)
        if not row['pass']:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
