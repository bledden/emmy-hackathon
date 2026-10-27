"""Bounded comparisons for finite f32, single-kernel fixtures only.

Prepares both compilers locally by default. --execute sends two compilations
and six interleaved runs per case. Stops on the first error; never retries.
This development helper is not the typed qualification or submission runner.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

SEQUENCE = ('baseline', 'candidate', 'candidate', 'baseline', 'baseline', 'candidate')
REQUESTS_PER_CASE = 2 + len(SEQUENCE)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(folder):
    """Refuse unsupported types and incomplete references before any request."""
    meta = json.loads((folder / 'meta.json').read_text())
    if len(meta['kernels']) != 1:
        raise ValueError('comparison supports single-kernel fixtures only')
    kernel = meta['kernels'][0]
    args = kernel['args']
    if any(a.get('dtype') != 'f32' or a.get('scalar') for a in args):
        raise ValueError('comparison supports f32 buffers only; use typed qualification for other types')
    names = [a['name'] for a in args]
    if len(set(names)) != len(names) or kernel['arg_order'] != names:
        raise ValueError('ambiguous or mismatched argument order')
    for a in args:
        if not isinstance(a['shape'], list) or not all(type(n) is int and n > 0 for n in a['shape']):
            raise ValueError('invalid buffer shape')
    for key in ('rtol', 'atol'):
        t = meta.get(key)
        if type(t) not in (int, float) or not math.isfinite(t) or t < 0:
            raise ValueError('tolerances must be finite nonnegative numbers')
    with np.load(folder / 'expected.npz', allow_pickle=False) as archive:
        expected = dict(archive)
    outputs = {a['name']: a for a in args if a.get('output')}
    if not outputs or set(expected) != set(outputs):
        raise ValueError('references must cover exactly the output buffers')
    for name, ref in expected.items():
        if ref.dtype != np.dtype('float32') or list(ref.shape) != outputs[name]['shape']:
            raise ValueError('reference dtype/shape must match f32 output metadata')
        if not np.isfinite(ref).all():
            raise ValueError('comparison supports finite references only')
    return meta, kernel, expected


class Recorder:
    """Count attempts before sending; preserve failures and enforce a hard cap."""

    def __init__(self, server, out, cap, session):
        self.server, self.out, self.cap, self.session = server.rstrip('/'), out, cap, session
        self.attempts = 0

    def post(self, endpoint, body):
        if self.attempts >= self.cap:
            raise RuntimeError('request cap reached; no further requests')
        self.attempts += 1
        record = {'utc': datetime.now(timezone.utc).isoformat(), 'endpoint': endpoint,
                  'request': body}
        try:
            response = self.session.post(self.server + endpoint, json=body,
                                         timeout=(10, 300), allow_redirects=False)
            record.update(status_code=response.status_code, response_text=response.text)
            if response.status_code != 200:
                raise RuntimeError(f'runner HTTP {response.status_code}; stopped without retry')
            data = response.json()
            if not isinstance(data, dict) or data.get('ok') is not True or data.get('stage') == 'busy':
                raise RuntimeError('runner refused or busy; stopped without retry')
            return data
        except BaseException as error:
            record['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            with gzip.open(self.out / f'request-{self.attempts:03d}.json.gz', 'wt') as stream:
                json.dump(record, stream)


def check_run(result, body, expected, meta):
    for key in ('entry', 'grid', 'block', 'smem_bytes', 'iters'):
        # JSON spelling also distinguishes bools/floats from integer echoes.
        if json.dumps(result.get(key)) != json.dumps(body[key]):
            raise ValueError(f'run echo mismatch: {key}')
    outputs = result.get('outputs')
    if not isinstance(outputs, dict) or set(outputs) != set(expected):
        raise ValueError('returned outputs differ from reference set')
    for name, ref in expected.items():
        out = outputs[name]
        if out.get('truncated') is not False or out.get('dtype') != 'f32' or out.get('shape') != list(ref.shape):
            raise ValueError('truncated or mismatched output metadata')
        values = out.get('values')
        if not isinstance(values, list) or len(values) != ref.size or any(type(v) not in (int, float) for v in values):
            raise ValueError('invalid output values')
        # Check the JSON numbers without rounding them to f32 first: rounding
        # could hide an incorrect result, just as it did for large i64 values.
        got = np.asarray(values, dtype=np.float64).reshape(ref.shape)
        if not np.isfinite(got).all() or not np.allclose(got, ref, rtol=meta['rtol'], atol=meta['atol']):
            raise ValueError(f'incorrect timed output: {name}')
    timing = result.get('median_us')
    if type(timing) not in (int, float) or not math.isfinite(timing) or timing <= 0:
        raise ValueError('invalid timing')
    return timing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='+', type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--server', default='http://runner.riftstack.ai:48701')
    parser.add_argument('--execute', action='store_true', help='send the bounded comparison after local preparation')
    args = parser.parse_args(argv)
    # Validate the whole batch first, including later cases.
    cases = [(folder, *load_case(folder)) for folder in args.cases]
    args.out.mkdir(parents=True, exist_ok=False)
    compilers = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve()}
    identities = {label: {str(p.relative_to(path.parent)): digest(p)
                         for p in sorted(path.parent.rglob('*.py'))}
                  for label, path in compilers.items()}
    (args.out / 'source-hashes.json').write_text(json.dumps(identities, indent=2))
    cap = REQUESTS_PER_CASE * len(cases)
    (args.out / 'protocol.json').write_text(json.dumps({
        'created_utc': datetime.now(timezone.utc).isoformat(), 'warmup': 5, 'iters': 50,
        'sequence_per_case': SEQUENCE, 'cases': [str(p.resolve()) for p in args.cases],
        'request_cap': cap, 'retries': 0, 'execute': args.execute,
        'scope': 'finite f32 buffers, single-kernel fixtures only',
        'retention_gate': 'all outputs correct; >=5% median improvement in at least one case; no case >5% median regression',
        'limits': 'shared runner; no clock locking or L2 flush; indicative only',
    }, indent=2))
    prepared = []
    for index, (folder, meta, kernel, expected) in enumerate(cases):
        ir = folder / (kernel['entry'] + '.ir.json')
        sources = {}
        for label, compiler in compilers.items():
            result = subprocess.run([sys.executable, compiler, ir], capture_output=True, text=True, check=True)
            (args.out / f'{index:03d}-{folder.name}-{label}.ptx').write_text(result.stdout)
            sources[label] = result.stdout
        prepared.append((folder, meta, kernel, expected, sources))
    for label, compiler in compilers.items():
        current = {str(p.relative_to(compiler.parent)): digest(p) for p in compiler.parent.rglob('*.py')}
        if current != identities[label]:
            raise ValueError('compiler sources changed during preparation')
    if not args.execute:
        print(f'Prepared {len(cases)} cases; zero requests. --execute would send at most {cap} requests.')
        return
    summary = []
    with requests.Session() as session:
        session.trust_env = False
        recorder = Recorder(args.server, args.out, cap, session)
        try:
            for folder, meta, kernel, expected, sources in prepared:
                compiled = {}
                for label, ptx in sources.items():
                    build = recorder.post('/compile', {'ptx': ptx})
                    if (build.get('arch') != 'sm_120' or kernel['entry'] not in build.get('entry_points', [])
                            or not isinstance(build.get('cubin_id'), str) or not build['cubin_id']):
                        raise ValueError('compile identity mismatch')
                    compiled[label] = build
                measured = {label: [] for label in compilers}
                for label in SEQUENCE:
                    body = {key: kernel[key] for key in ('entry', 'grid', 'block', 'smem_bytes', 'zero_outputs', 'args')}
                    body.update(cubin_id=compiled[label]['cubin_id'], warmup=5, iters=50,
                                max_return=sum(ref.size for ref in expected.values()))
                    result = recorder.post('/run', body)
                    measured[label].append(check_run(result, body, expected, meta))
                medians = {label: statistics.median(values) for label, values in measured.items()}
                row = {'case': str(folder), 'request_medians_us': measured, 'median_us': medians,
                       'candidate_over_baseline': medians['candidate'] / medians['baseline'],
                       'registers': {label: c.get('registers') for label, c in compiled.items()}, 'correct': True}
                summary.append(row)
                print(json.dumps(row), flush=True)
        except BaseException as error:
            (args.out / 'failure.json').write_text(json.dumps({
                'attempts': recorder.attempts, 'error': f'{type(error).__name__}: {error}',
                'completed_cases': summary}, indent=2))
            raise
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    passes = (any(r['candidate_over_baseline'] <= .95 for r in summary) and
              all(r['candidate_over_baseline'] <= 1.05 for r in summary))
    print('Indicative retention gate:', 'PASS' if passes else 'NOT MET', flush=True)


if __name__ == '__main__':
    main()
