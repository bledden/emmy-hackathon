"""Offline replay only. Reads saved HTTP records; imports no compiler or runner.

Requires NumPy. The packet manifest binds bytes, not third-party authenticity.
All ratios are calculated from one batch, never multiplied across experiments.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(root):
    root = Path(root)
    seal = json.loads((root / 'packet-manifest.json').read_text())
    for name, digest in seal['files'].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), 'Unsafe artifact path')
        require(sha(path) == digest, f'Artifact hash: {name}')
    payloads = root / 'payloads'
    manifest = json.loads((payloads / 'manifest.json').read_text())
    require(sha(payloads / 'manifest.json') == seal['original_manifest_sha256'], 'Original manifest hash')
    for name, digest in manifest['files'].items():
        require(sha(payloads / name) == digest, f'Original payload hash: {name}')
    require(manifest['source']['baseline']['commit'] == seal['baseline_commit'], 'Baseline source')
    require(manifest['source']['candidate']['commit'] == seal['candidate_commit'], 'Candidate source')
    cases = manifest['cases']
    policy = manifest['policy']
    records = sorted((root / 'receipts').glob('request-*.json.gz'))
    require([p.name for p in records] == [f'request-{i:03d}.json.gz' for i in range(1, 31)], 'Exactly 30 ordered receipts')
    require(len(cases) == 5 and policy['maximum_requests'] == 30, 'Five-case scope')
    require(policy['timing_order'] == ['baseline', 'candidate', 'candidate', 'baseline'], 'ABBA order')
    require(policy['timing_warmup'] == 2 and policy['timing_iters'] == 20, 'Timing settings')
    results, position, values_checked = {}, 0, 0
    for case in cases:
        folder = payloads / case
        metadata = json.loads((folder / 'meta.json').read_text())
        kernel = metadata['kernels'][0]
        with np.load(folder / 'expected.npz', allow_pickle=False) as data:
            expected = {name: data[name] for name in data.files}
        atol, rtol = metadata['atol'], metadata['rtol']
        require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in (atol, rtol)), 'Original tolerances')
        cubins, registers = {}, {}
        timings = {'baseline': [], 'candidate': []}
        sequence = [('/compile', 'baseline'), ('/compile', 'candidate')]
        sequence += [('/run', v) for v in policy['timing_order']]
        for endpoint, variant in sequence:
            record = json.loads(gzip.decompress(records[position].read_bytes()))
            position += 1
            label = f'{case}/{variant}/request-{position}'
            require(record['endpoint'] == endpoint and record['status_code'] == 200 and 'error' not in record, label + ': transport')
            response = json.loads(record['response_text'])
            require(response.get('ok') is True, label + ': ok')
            if endpoint == '/compile':
                require(record['request'] == {'ptx': (folder / f'{variant}.ptx').read_text()}, label + ': compile payload')
                require(response.get('arch') == 'sm_120' and response.get('entry_points') == [kernel['entry']], label + ': compile identity')
                require(isinstance(response.get('cubin_id'), str) and response['cubin_id'], label + ': cubin id')
                cubins[variant] = response['cubin_id']
                registers[variant] = response.get('registers')
                continue
            body = {key: kernel[key] for key in ('entry', 'grid', 'block', 'smem_bytes', 'zero_outputs', 'args')}
            body.update(cubin_id=cubins[variant], warmup=2, iters=20, max_return=sum(a.size for a in expected.values()))
            require(record['request'] == body, label + ': run payload')
            require(response.get('entry') == kernel['entry'], label + ': entry')
            for name in ('grid', 'block'):
                value = response.get(name)
                require(isinstance(value, list) and all(type(x) is int for x in value) and value == kernel[name], label + ': ' + name)
            for name in ('smem_bytes', 'iters'):
                require(type(response.get(name)) is int and response[name] == body[name], label + ': ' + name)
            require(set(response.get('outputs', {})) == set(expected), label + ': output names')
            for name, reference in expected.items():
                output = response['outputs'][name]
                require(reference.dtype == np.float32 and np.isfinite(reference).all(), label + ': finite f32 reference')
                require(output.get('dtype') == 'f32' and output.get('shape') == list(reference.shape) and output.get('truncated') is False, label + ': output format')
                tokens = output['values']
                require(len(tokens) == reference.size and all(type(v) in (int, float) for v in tokens), label + ': value count/types')
                got = np.asarray(tokens, dtype=np.float64)
                require(np.isfinite(got).all(), label + ': nonfinite value')
                require(np.array_equal(got.astype(np.float32).astype(np.float64), got), label + ': lossy f32 JSON')
                want = reference.astype(np.float64).ravel()
                require(np.all(np.abs(got - want) <= atol + rtol * np.abs(want)), label + ': original tolerance')
                values_checked += got.size
            median = response.get('median_us')
            require(type(median) in (int, float) and math.isfinite(median) and median > 0, label + ': timing')
            timings[variant].append(median)
        means = {v: sum(samples) / len(samples) for v, samples in timings.items()}
        results[case] = {'medians_us': timings, 'mean_us': means,
                         'baseline_over_candidate': means['baseline'] / means['candidate'],
                         'registers': registers}
    ratios = {case: row['baseline_over_candidate'] for case, row in results.items()}
    gates = {'m01_ge_1_10': ratios['m01_matmul_512'] >= 1.10,
             'every_case_ge_0_95': all(r >= .95 for r in ratios.values())}
    require(position == 30 and values_checked == 1157456, 'Original coverage counts')
    require(all(gates.values()), 'Performance gates')
    original_decode = json.loads((root / 'original-decode.json').read_text())
    require(not original_decode['problems'] and original_decode['values_checked'] == values_checked, 'Earlier independent decoder counts')
    for case, row in results.items():
        earlier = original_decode['complete_pairs'][case]
        require(row['medians_us'] == original_decode['medians_us'][case], 'Earlier decoder medians')
        require(row['baseline_over_candidate'] == earlier['baseline_time_over_candidate_time'], 'Earlier decoder ratio')
    return {'offline': True, 'network_requests': 0, 'recorded_requests': position,
            'returned_values_checked': values_checked, 'cases': results, 'gates': gates,
            'limits': 'Two request medians per variant on a shared service; directional only. Saved records are not independently authenticated.'}


if __name__ == '__main__':
    print(json.dumps(replay(Path(__file__).resolve().parent), indent=2))
