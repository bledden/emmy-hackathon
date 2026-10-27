"""Offline receipt check, independently implemented from run_typed/typed_io.
Only the approved055 f16/f32 groups; imports neither runner nor compiler.
"""
from pathlib import Path
from fractions import Fraction
import hashlib, json, math, sys
import numpy as np
HERE = Path(__file__).resolve().parent
PLAN = HERE / 'payloads'
HASHES = {'core': '7d27e6159dbcdea9abd612754a7f0d6962afe3ce346cf9d605b5299741de516d', 'synchronization': 'e54081be24c0f0b5e55fc6275fe04489675322edee94aa768db2358c40bb1daa'}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def canonical(x):
    return json.dumps(x, sort_keys=True, allow_nan=False, separators=(',', ':'))

def check(group, receipts):
    batch = PLAN / ('review-' + group)
    mf = batch / 'manifest.json'
    if not sha(mf) == HASHES[group]:
        raise ValueError('manifest identity')
    m = json.loads(mf.read_text())
    if not all((sha(batch / p) == h for p, h in m['files'].items())):
        raise ValueError('payload identity')
    if not (receipts / 'approved-manifest.json').read_bytes() == mf.read_bytes():
        raise ValueError('Receipt contract mismatch')
    count = 2 * len(m['cases'])
    if not sorted((p.name for p in receipts.glob('request-*.json'))) == [f'request-{i:02d}.json' for i in range(1, count + 1)]:
        raise ValueError('Receipt contract mismatch')
    summary = json.loads((receipts / 'summary.json').read_text())
    if not (summary['state'] == 'passed' and type(summary['requests_attempted']) is int and (summary['requests_attempted'] == count)):
        raise ValueError('Receipt contract mismatch')
    if not (summary['maximum_http_requests'] == count and summary['candidate'] == m['source']['commit']):
        raise ValueError('Receipt contract mismatch')
    rows = []
    total = 0
    for ci, name in enumerate(m['cases']):
        d = batch / name
        meta = json.loads((d / 'meta.json').read_text())
        k = meta['kernels'][0]
        pair = [json.loads((receipts / f'request-{2 * ci + i:02d}.json').read_text()) for i in (1, 2)]
        for rec in pair:
            if not (rec['status_code'] == 200 and 'error' not in rec):
                raise ValueError('Receipt contract mismatch')
        compiled = json.loads(pair[0]['response_text'])
        ran = json.loads(pair[1]['response_text'])
        if not (pair[0]['endpoint'] == '/compile' and pair[0]['request'] == {'ptx': (d / 'kernel.ptx').read_text()}):
            raise ValueError('Receipt contract mismatch')
        if not (compiled['ok'] is True and compiled.get('stage') != 'busy'):
            raise ValueError('Receipt contract mismatch')
        if not (compiled['arch'] == 'sm_120' and compiled['entry_points'] == [k['entry']]):
            raise ValueError('Receipt contract mismatch')
        if not (isinstance(compiled['cubin_id'], str) and compiled['cubin_id']):
            raise ValueError('Receipt contract mismatch')
        body = {key: k[key] for key in ('entry', 'grid', 'block', 'smem_bytes', 'zero_outputs')}
        body.update(cubin_id=compiled['cubin_id'], warmup=0, iters=1)
        body['args'] = []
        with np.load(d / 'inputs.npz', allow_pickle=False) as inputs:
            for a in k['args']:
                item = {key: a[key] for key in ('name', 'dtype', 'shape')}
                if a.get('output'):
                    item['output'] = True
                else:
                    item['data'] = inputs[a['name']].ravel().tolist()
                body['args'].append(item)
        with np.load(d / 'expected.npz', allow_pickle=False) as refs:
            body['max_return'] = sum((refs[n].size for n in refs.files))
            if not (pair[1]['endpoint'] == '/run' and canonical(pair[1]['request']) == canonical(body)):
                raise ValueError('exact request body')
            if not (ran['ok'] is True and ran.get('stage') != 'busy'):
                raise ValueError('Receipt contract mismatch')
            for key in ('entry', 'grid', 'block', 'smem_bytes', 'iters'):
                if not canonical(ran[key]) == canonical(body[key]):
                    raise ValueError('exact response echo')
            if not (type(ran['median_us']) in (int, float) and math.isfinite(ran['median_us']) and (ran['median_us'] >= 0)):
                raise ValueError('Receipt contract mismatch')
            if not set(ran['outputs']) == set(refs.files):
                raise ValueError('Receipt contract mismatch')
            outputs = {}
            for a in k['args']:
                if not a.get('output'):
                    continue
                ref = refs[a['name']]
                o = ran['outputs'][a['name']]
                if not a['dtype'] in ('f16', 'f32'):
                    raise ValueError('decoder scope')
                dtype = np.dtype('float16' if a['dtype'] == 'f16' else 'float32')
                if not (ref.dtype == dtype and ref.shape == tuple(a['shape'])):
                    raise ValueError('Receipt contract mismatch')
                if not (o['dtype'] == a['dtype'] and canonical(o['shape']) == canonical(a['shape']) and (o['truncated'] is False)):
                    raise ValueError('Receipt contract mismatch')
                raw = o['values']
                if not (isinstance(raw, list) and len(raw) == ref.size):
                    raise ValueError('Receipt contract mismatch')
                if not all((type(x) in (int, float) and math.isfinite(x) for x in raw)):
                    raise ValueError('Receipt contract mismatch')
                with np.errstate(all='ignore'):
                    got = np.asarray(raw, dtype=dtype).reshape(ref.shape)
                if not np.isfinite(got).all():
                    raise ValueError('Receipt contract mismatch')
                if not all((float(v) == raw[j] for j, v in enumerate(got.ravel()))):
                    raise ValueError('no lossy output conversion')
                rtol, atol = (Fraction(meta['rtol']), Fraction(meta['atol']))
                if not (rtol >= 0 and atol >= 0):
                    raise ValueError('Receipt contract mismatch')
                for v, w in zip(got.ravel(), ref.ravel()):
                    if not abs(Fraction(float(v)) - Fraction(float(w))) <= atol + rtol * abs(Fraction(float(w))):
                        raise ValueError('value outside original tolerance')
                    if w == 0:
                        if not np.signbit(v) == np.signbit(w):
                            raise ValueError('zero sign')
                outputs[a['name']] = {'values': ref.size, 'max_abs_error': float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64)))), 'bit_identical': got.tobytes() == ref.tobytes(), 'zero_reference_count': int(np.sum(ref == 0)), 'original_rtol': meta['rtol'], 'original_atol': meta['atol']}
                total += ref.size
        rows.append({'case': name, 'arch': compiled['arch'], 'outputs': outputs, 'single_run_us_not_benchmark': ran['median_us']})
    return {'group': group, 'manifest_sha256': HASHES[group], 'requests': count, 'cases': rows, 'values_checked': total, 'all_pass': True, 'requests_by_checker': 0, 'limits': 'One run per frozen case. Exact small-integer MMA; no speed/race/all-domain claim. Same-author separate decoder, not peer review.'}


def replay():
    seal = json.loads((HERE / 'packet-manifest.json').read_text())
    for name, expected in seal['files'].items():
        path = (HERE / name).resolve()
        if not path.is_relative_to(HERE.resolve()) or sha(path) != expected:
            raise ValueError('Packet identity mismatch: ' + name)
    reports = {group: check(group, HERE / 'receipts' / group) for group in HASHES}
    peer = json.loads((HERE / 'original-decode.json').read_text())
    total = sum(r['values_checked'] for r in reports.values())
    exact = 0
    for group, report in reports.items():
        prior = peer['groups'][group]
        if prior['pass'] is not True or prior['request_files'] != report['requests']:
            raise ValueError('Peer group mismatch')
        if [c['case'] for c in prior['cases']] != [c['case'] for c in report['cases']]:
            raise ValueError('Peer case list mismatch')
        for case, old in zip(report['cases'], prior['cases']):
            exact += all(v['bit_identical'] for v in case['outputs'].values())
            if old['pass_'] is not True or set(old['checks']) != set(case['outputs']):
                raise ValueError('Peer output set mismatch')
            for name, result in case['outputs'].items():
                before = old['checks'][name]
                if before['bitwise_equal'] != result['bit_identical'] or before['zeros'] != result['zero_reference_count']:
                    raise ValueError('Peer value summary mismatch')
                if 'max_abs_error' in before and before['max_abs_error'] != result['max_abs_error']:
                    raise ValueError('Peer error summary mismatch')
    if total != peer['values_decoded'] or exact != peer['exact_cases'] or peer['all_pass'] is not True:
        raise ValueError('Peer total mismatch')
    return {'offline_replay': True, 'http_requests': 0, 'recorded_requests': 38,
            'values_checked': total, 'bit_exact_cases': exact, 'groups': reports,
            'limits': 'Saved one-run correctness evidence, not benchmarks, race proof or all-domain accuracy.'}

if __name__ == '__main__':
    print(json.dumps(replay(), indent=2))
