"""Compile bundled examples offline; execute tensor examples in the limited model.

Compilation is not hardware validation. Typed/async examples are compilation
smoke tests with recomputed reference arrays, not PTX executions.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptxgen.compiler import compile_file
from fragment_model import execute


def main():
    root = Path(__file__).resolve().parents[1]
    rows = []
    for folder in sorted((root/'examples').iterdir()):
        if not folder.is_dir():
            continue
        path = next(folder.glob('*.ir.json'))
        ptx = compile_file(path, contract_fma='auto', index_reuse='auto',
                           constant_division=True, unroll_loops=True, shared_loads='compact')
        meta = json.loads((folder/'meta.json').read_text())
        data = dict(np.load(folder/'inputs.npz'))
        expected = dict(np.load(folder/'expected.npz'))
        for arg in meta['kernels'][0]['args']:
            if not arg.get('output'):
                # Float64 would lose low bits of i64; preserve the stored dtype.
                np.testing.assert_array_equal(np.asarray(arg['data'], dtype=data[arg['name']].dtype)
                                              .reshape(arg['shape']), data[arg['name']])
        if folder.name.startswith('mma_'):
            actual = execute(ptx, data, meta)
            level = 'emitted PTX lane/ownership model (exact integer inputs only)'
        else:
            actual = {'result': data['source'] if folder.name == 'async_copy'
                      else data['left'] + data['right']}
            level = 'compile smoke + reference consistency (no PTX execution)'
        for name, ref in expected.items():
            assert actual[name].dtype == ref.dtype
            np.testing.assert_array_equal(actual[name], ref)
        rows.append({'example': folder.name, 'validation': level,
                     'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest()})
    print(json.dumps({'network_requests': 0, 'gpu_runs': 0, 'examples': rows}, indent=2))


if __name__ == '__main__':
    main()
