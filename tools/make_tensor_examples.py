"""Construct small typed TensorCoreIR examples; no renderer or network use."""
import argparse
import json
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.tensor import Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.kernel.ir import (
    KernelOp, Tile, RegFragment, LdmatrixLoad, MmaSyncPtx, RegStore)
from emmy.compiler.dtype import F16 as f16, F32 as f32


def example(*, chunks=1, padded=False, seed=17, names=('left', 'right', 'result')):
    """FP16 integer operands keep the reference exactly representable in FP32."""
    a, b, c = names
    k = chunks * 16
    ar, ac, br, bc, cr, cc = (1, 3, 2, 1, 2, 3) if padded else (0,) * 6
    shapes = {a: (ar+16, ac+k+int(padded)), b: (br+k, bc+8+int(padded)),
              c: (cr+16, cc+8+int(padded))}
    graph = Graph()
    for name in (a, b):
        graph.add_node(InputOp(), [], Tensor(name, tuple(Dim(d) for d in shapes[name]), f16))
    atom = (16, 8, 16)
    body = [RegFragment('fa', 'a', atom, f16), RegFragment('fb', 'b', atom, f16),
            RegFragment('fc', 'c', atom, f32)]
    lit = lambda r, c: (Literal(r, 'int'), Literal(c, 'int'))
    for chunk in range(chunks):
        body += [LdmatrixLoad('fa', a, lit(ar, ac+chunk*16), 'a', staged=False),
                 LdmatrixLoad('fb', b, lit(br+chunk*16, bc), 'b', staged=False),
                 MmaSyncPtx('fc', 'fa', 'fb', atom)]
    body.append(RegStore(c, lit(cr, cc), 'fc', atom))
    tile = Tile((Axis('thread_cell', Dim(32)),), tuple(body))
    entry = 'tensor_tile'
    graph.add_node(KernelOp((tile,), name=entry), [a, b],
                   Tensor(c, tuple(Dim(d) for d in shapes[c]), f32))
    args = [{'name': name, 'dtype': 'f32' if name == c else 'f16',
             'shape': list(shapes[name]), **({'output': True} if name == c else {})}
            for i, name in enumerate(names)]
    meta = {'id': 'tensor_tile', 'rtol': 0.0, 'atol': 0.0,
            'kernels': [{'entry': entry, 'node': c, 'block': [32, 1, 1], 'grid': [1, 1, 1],
                         'smem_bytes': 0, 'zero_outputs': [], 'arg_order': list(names), 'args': args}]}
    rng = np.random.default_rng(seed)
    data = {name: rng.integers(-4, 5, shapes[name]).astype(np.float16) for name in (a, b)}
    for arg in args:
        if not arg.get('output'):
            arg['data'] = data[arg['name']].ravel().tolist()
    expected = np.zeros(shapes[c], dtype=np.float32)
    expected[cr:cr+16, cc:cc+8] = (
        data[a][ar:ar+16, ac:ac+k].astype(np.int64) @
        data[b][br:br+k, bc:bc+8].astype(np.int64)).astype(np.float32)
    return graph, meta, data, {c: expected}


def save(folder, **kwargs):
    folder.mkdir(parents=True, exist_ok=False)
    graph, meta, inputs, expected = example(**kwargs)
    (folder / 'tensor_tile.ir.json').write_text(json.dumps(graph.to_dict(), default=repr, indent=2)+'\n')
    (folder / 'meta.json').write_text(json.dumps(meta, indent=2)+'\n')
    np.savez(folder / 'inputs.npz', **inputs)
    np.savez(folder / 'expected.npz', **expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    for name, kwargs in {'mma_tile': {}, 'mma_padded': {'padded': True},
                         'mma_k32': {'chunks': 2}}.items():
        save(args.out / name, **kwargs)
