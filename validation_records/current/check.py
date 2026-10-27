"""Replay recorded 064/068/069/071 responses without any network or GPU calls.

New publication checker, authored by Codex. It is not the original Opus decoder
and has not received an independent agent review. Requires only NumPy plus the
normal compiler dependencies for the optional/default current-PTX regeneration.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DTYPES = {'f16': np.dtype('<f2'), 'f32': np.dtype('<f4'), 'f64': np.dtype('<f8'),
          'i32': np.dtype('<i4'), 'i64': np.dtype('<i8')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(text):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'Duplicate JSON key: ' + k)
            result[k] = v
        return result
    def nonfinite(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)


def read_json(path):
    return strict_json(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_json(a, b):
    # Distinguish true from 1, and integer launch fields from float spellings.
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def safe_path(root, name):
    p = root / name
    require(not Path(name).is_absolute() and '..' not in Path(name).parts
            and p.resolve().is_relative_to(root.resolve()), 'Unsafe path: ' + name)
    return p


def verify_seal(root):
    seal = read_json(root / 'packet-manifest.json')
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and str(p.relative_to(root)) != 'packet-manifest.json'}
    require(actual == set(seal['files']), 'Packet file set changed')
    for name, expected in seal['files'].items():
        require(digest(safe_path(root, name)) == expected, 'Packet hash: ' + name)


def verify_source(repo):
    provenance = read_json(repo / 'docs/source-provenance.json')
    for name, expected in provenance['files'].items():
        require(digest(safe_path(repo, name)) == expected, 'Reviewed source hash: ' + name)
    return len(provenance['files'])


def integer(v, bits):
    require(type(v) in (int, float), 'Integer token is not numeric')
    if type(v) is float:
        require(math.isfinite(v) and v.is_integer() and abs(v) <= 2**53 - 1,
                'Lossy integer JSON token')
    v = int(v)
    require(-(2**(bits-1)) <= v < 2**(bits-1), 'Integer outside declared width')
    return v


def decode(output, dtype, shape):
    require(dtype in DTYPES, 'Unknown output dtype')
    require(equal_json(output.get('shape'), shape) and output.get('dtype') == dtype
            and output.get('truncated') is False, 'Output type/shape/truncation')
    raw = output.get('values')
    require(type(raw) is list and len(raw) == math.prod(shape), 'Output value count')
    if dtype.startswith('i'):
        vals = [integer(x, DTYPES[dtype].itemsize * 8) for x in raw]
    else:
        require(all(type(x) in (int, float) for x in raw), 'Non-numeric floating token')
        vals = np.asarray(raw, dtype=np.float64)
        require(np.isfinite(vals).all(), 'Nonfinite floating output')
        with np.errstate(all='ignore'):
            rounded = vals.astype(DTYPES[dtype])
        require(np.isfinite(rounded).all() and np.array_equal(rounded.astype(np.float64), vals),
                'Lossy floating JSON token')
    return np.asarray(vals, dtype=DTYPES[dtype]).reshape(shape)


def wire_payload(kernel, warmup, iters):
    body = {k: copy.deepcopy(kernel[k]) for k in ('entry', 'grid', 'block', 'smem_bytes', 'zero_outputs', 'args')}
    words = all(a['dtype'] == 'i64' for a in kernel['args'])
    if words:
        args = []
        for a in kernel['args']:
            wire = {'name': a['name'], 'dtype': 'i32', 'shape': [2 * math.prod(a['shape'])]}
            if a.get('output'):
                wire['output'] = True
            else:
                values = [integer(v, 64) for v in a['data']]
                wire['data'] = []
                for v in values:
                    for w in (v & 0xffffffff, (v >> 32) & 0xffffffff):
                        wire['data'].append(w - 2**32 if w >= 2**31 else w)
            args.append(wire)
        body['args'] = args
    body.update(warmup=warmup, iters=iters,
                max_return=sum(math.prod(a['shape']) for a in body['args'] if a.get('output')))
    return body, words


def compare_outputs(response, kernel, references, metadata, exact):
    outputs = response.get('outputs')
    out_args = {a['name']: a for a in kernel['args'] if a.get('output')}
    require(type(outputs) is dict and set(outputs) == set(references) == set(out_args), 'Output names')
    atol, rtol = metadata['atol'], metadata['rtol']
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in (atol, rtol)), 'Invalid tolerance')
    words = all(a['dtype'] == 'i64' for a in kernel['args'])
    count, exact_values, worst_error = 0, 0, 0.0
    for name, ref in references.items():
        a = out_args[name]
        require(ref.shape == tuple(a['shape']) and ref.dtype == DTYPES[a['dtype']], 'Reference dtype/shape')
        require(np.isfinite(ref).all(), 'Nonfinite reference')
        if words:
            halves = decode(outputs[name], 'i32', [2 * ref.size]).reshape(-1).tolist()
            values = []
            for lo, hi in zip(halves[::2], halves[1::2]):
                n = (hi & 0xffffffff) << 32 | (lo & 0xffffffff)
                values.append(n - 2**64 if n >= 2**63 else n)
            got = np.asarray(values, dtype='<i8').reshape(ref.shape)
        else:
            got = decode(outputs[name], a['dtype'], a['shape'])
        if ref.dtype.kind == 'i' or exact:
            require(got.tobytes() == ref.tobytes(), 'Exact values or signed zeros differ: ' + name)
            exact_values += ref.size
        else:
            # Timing cases are f32; float64 comparison avoids rounding raw JSON
            # into a passing f32 value. No integer or extreme f64 path uses it.
            require(ref.dtype == DTYPES['f32'], 'Timing comparison requires f32')
            g, e = got.astype(np.float64), ref.astype(np.float64)
            err = np.abs(g - e)
            require(np.all(err <= atol + rtol * np.abs(e)), 'Outside original tolerance: ' + name)
            exact_values += int(((g == e) & (np.signbit(g) == np.signbit(e))).sum())
            worst_error = max(worst_error, float(err.max(initial=0)))
        count += ref.size
    return int(count), int(exact_values), worst_error


def replay_batch(root, batch, compile_current=True):
    folder = root / batch['id']; plan = read_json(folder / 'plan.json')
    require(plan['batch'] == batch['id'] and plan['mode'] == batch['mode'], 'Batch identity')
    cases = {c['name']: c for c in plan['cases']}
    require(len(cases) == batch['cases'], 'Case count')
    require(len(plan['requests']) == batch['requests'], 'Request count')
    require([q['number'] for q in plan['requests']] == list(range(1, batch['requests']+1)), 'Request order')
    want_receipts = {q['receipt'] for q in plan['requests']}
    actual_receipts = {'receipts/'+p.name for p in (folder/'receipts').iterdir() if p.is_file()}
    require(want_receipts == actual_receipts, 'Receipt file set')
    cubins, times, registers, orders, compiled, rows = {}, {}, {}, {}, set(), {}
    data = {}
    regenerated = 0
    for name, case in cases.items():
        d = folder / 'payloads' / name; meta = read_json(d/'meta.json')
        require(len(meta['kernels']) == 1, 'Expected one kernel')
        k = meta['kernels'][0]
        with np.load(d/'expected.npz', allow_pickle=False) as z:
            refs = {key:z[key].copy() for key in z.files}
        data[name] = (d, meta, k, refs)
        rows[name] = {'values_checked':0, 'bit_exact_values':0, 'max_abs_error':0.0}
        if compile_current:
            env = {k:v for k,v in os.environ.items() if not k.startswith('EMMY_PTX_')}
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            p = subprocess.run([sys.executable, '-B', str(REPO/'compile.py'), str(d/'kernel.ir.json')],
                               capture_output=True, env=env, timeout=30, cwd=folder)
            artifact = 'compact.ptx' if batch['mode']=='timing' else 'kernel.ptx'
            require(p.returncode == 0 and not p.stderr and p.stdout == (d/artifact).read_bytes(),
                    'Current CLI no longer reproduces tested PTX: '+name)
            regenerated += 1
    for q in plan['requests']:
        require(q['case'] in cases and q['variant'] in cases[q['case']]['variants'], 'Unknown case/variant')
        d, meta, k, refs = data[q['case']]
        f = safe_path(folder, q['receipt']); raw = gzip.decompress(f.read_bytes()) if f.suffix=='.gz' else f.read_bytes()
        receipt = strict_json(raw.decode())
        require(set(receipt) in ({'endpoint','request','status_code','response_text'},
                               {'utc','endpoint','request','status_code','response_text'}), 'Receipt fields')
        require(type(receipt['status_code']) is int and receipt['status_code']==200, 'HTTP status')
        require(receipt['endpoint'] == q['endpoint'], 'Endpoint order')
        response = strict_json(receipt['response_text'])
        require(response.get('ok') is True, 'Service reported failure')
        key = q['case']+':'+q['variant']
        body = copy.deepcopy(q['body'])
        if q['endpoint'] == '/compile':
            ptx_name = q['variant']+'.ptx' if batch['mode']=='timing' else 'kernel.ptx'
            require(equal_json(body, {'ptx':(d/ptx_name).read_text()}), 'Planned PTX mismatch')
            require(equal_json(receipt['request'], body), 'Sent PTX mismatch')
            require(q['bind_cubin']==key and key not in compiled, 'Duplicate/wrong compile binding')
            require(response.get('arch')=='sm_120' and response.get('entry_points')==[k['entry']], 'Compile arch/entry')
            cid=response.get('cubin_id')
            require(type(cid) is str and re.fullmatch('[0-9a-f]{64}',cid), 'Invalid cubin ID')
            regs=response.get('registers')
            require(type(regs) is dict and set(regs)=={k['entry']} and type(regs[k['entry']]) is int
                    and regs[k['entry']]>=0, 'Invalid registers')
            cubins[key]=cid; registers[key]=regs[k['entry']]; compiled.add(key)
        else:
            require(q['endpoint']=='/run' and key in compiled, 'Run before compile')
            require(body.pop('cubin_id')=={'from_compile':key}, 'Plan compile/run binding')
            expected, _ = wire_payload(k,plan['warmup'],plan['iters'])
            require(equal_json(body,expected), 'Planned launch/input payload differs from metadata')
            body['cubin_id']=cubins[key]
            require(equal_json(receipt['request'],body), 'Sent run payload mismatch')
            for field in ('entry','grid','block','smem_bytes','iters'):
                require(equal_json(response.get(field),body[field]), 'Run echo: '+field)
            median=response.get('median_us')
            require(type(median) in (int,float) and math.isfinite(median) and median>0, 'Invalid timing')
            times.setdefault(key,[]).append(median)
            orders.setdefault(q['case'],[]).append(q['variant'])
            n, exact_n, err=compare_outputs(response,k,refs,meta,batch['mode']=='exact')
            rows[q['case']]['values_checked']+=n
            rows[q['case']]['bit_exact_values']+=exact_n
            rows[q['case']]['max_abs_error']=max(rows[q['case']]['max_abs_error'],err)
    for name,case in cases.items():
        require(orders[name]==case['run_order'], 'Variant run order')
        require(all(name+':'+v in compiled for v in case['variants']), 'Missing variant')
        rows[name]['registers']={v:registers[name+':'+v] for v in case['variants']}
        if batch['mode']=='timing':
            medians={v:times[name+':'+v] for v in case['variants']}
            require(all(len(t)==2 for t in medians.values()), 'Two request medians per variant required')
            means={v:sum(Fraction(str(x)) for x in t)/len(t) for v,t in medians.items()}
            rows[name].update(medians_us=medians,mean_us={v:float(m) for v,m in means.items()},
                             speedup_over_baseline={v:float(means['baseline']/m) for v,m in means.items()})
        else:
            require(len(times[name+':current'])==1, 'One qualification run required')
    return {'requests':len(plan['requests']),'current_ptx_regenerated':regenerated,'cases':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-compile',action='store_true',help='Replay receipts without regenerating the 24 current PTX files.')
    args=parser.parse_args()
    verify_seal(HERE); protected=verify_source(REPO)
    index=read_json(HERE/'index.json')
    require([b['id'] for b in index['batches']]==['064','068','069','071'], 'Batch set')
    result={b['id']:replay_batch(HERE,b,not args.skip_compile) for b in index['batches']}
    require(sum(b['requests'] for b in result.values())==82, 'Total receipt count')
    print(json.dumps({'network_requests':0,'gpu_runs':0,'protected_files_verified':protected,'batches':result},indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
