"""Generate portable examples for existing typed arithmetic and async copy."""
import argparse
import json
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import get
from emmy.compiler.graph import Graph
from emmy.compiler.tensor import Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt.leaves import Assign
from emmy.compiler.ir.kernel.ir import (
    KernelOp, Tile, Load, Write, Smem, CpAsyncCopy, CpAsyncCommit, CpAsyncWait, Sync)


def example(dtype='f32', *, asynchronous=False):
    kind = get(dtype)
    count = 128 if asynchronous else 32
    graph = Graph()
    names = ('source',) if asynchronous else ('left', 'right')
    for name in names:
        graph.add_node(InputOp(), [], Tensor(name, (Dim(count),), kind))
    index = Var('cell')
    if asynchronous:
        assert dtype == 'f32'
        offset = (BinaryExpr('*', index, Literal(4, 'int')),)
        regs = ('v0', 'v1', 'v2', 'v3')
        body = (Smem('stage', (count,), 'f32', 16),
                CpAsyncCopy('stage', offset, 'source', offset, nbytes=16),
                CpAsyncCommit(), CpAsyncWait(0), Sync(),
                Load(input='stage', index=offset, names=regs, dtype=kind),
                Write(output='result', index=offset, values=regs, value_dtype=kind))
    else:
        body = (Load('a', 'left', (index,), dtype=kind), Load('b', 'right', (index,), dtype=kind),
                Assign('sum', ElementwiseImpl('add'), ('a', 'b'), kind),
                Write('result', (index,), 'sum', value_dtype=kind))
    entry = 'async_copy' if asynchronous else 'typed_add'
    graph.add_node(KernelOp((Tile((Axis('cell', Dim(32)),), body),), name=entry), list(names),
                   Tensor('result', (Dim(count),), kind))
    if asynchronous:
        data = {'source': np.arange(count, dtype=np.float32) / 8}
        expected = data['source'].copy()
    else:
        data = {'left': np.arange(count, dtype=kind.np), 'right': np.ones(count, dtype=kind.np)}
        if dtype == 'i64': data['left'] += 2**53
        if dtype == 'f64': data['right'] *= 2**-40
        if dtype == 'f16': data['left'] *= np.float16(1/8)
        expected = data['left'] + data['right']
    args = [{'name': name, 'dtype': dtype, 'shape': [count], 'data': values.tolist()}
            for name, values in data.items()]
    args.append({'name': 'result', 'dtype': dtype, 'shape': [count], 'output': True})
    meta = {'id': entry, 'rtol': 0.0, 'atol': 0.0, 'kernels': [{
        'entry': entry, 'node': 'result', 'grid': [1,1,1], 'block': [32,1,1],
        'smem_bytes': 0, 'zero_outputs': [], 'arg_order': [a['name'] for a in args], 'args': args}]}
    return graph, meta, data, {'result': expected}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cases = {f'add_{d}': {'dtype': d} for d in ('f16','f32','f64','i32','i64')}
    cases['async_copy'] = {'asynchronous': True}
    for name, options in cases.items():
        folder = args.out/name
        folder.mkdir(parents=True, exist_ok=False)
        graph, meta, data, expected = example(**options)
        entry = meta['kernels'][0]['entry']
        (folder/f'{entry}.ir.json').write_text(json.dumps(graph.to_dict(), default=repr, indent=2)+'\n')
        (folder/'meta.json').write_text(json.dumps(meta, indent=2)+'\n')
        np.savez(folder/'inputs.npz', **data)
        np.savez(folder/'expected.npz', **expected)
