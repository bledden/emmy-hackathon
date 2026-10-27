"""Execute emitted arithmetic locally to catch alias, tail and special-value bugs.

This models ex2 using NumPy, not the GPU approximation; hardware is a separate gate.
"""
from collections import defaultdict
from pathlib import Path
import struct
import sys
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptxgen.compiler import Emitter


def simulate(x, alias=True, perturb=0):
    emitter = Emitter.__new__(Emitter)
    emitter.register_kinds = {}
    emitter.code, emitter.counts = [], defaultdict(int)
    source = emitter.reg('f32')
    dest = source if alias else emitter.reg('f32')
    emitter.arithmetic(dest, 'softplus', [source])
    registers = {source: np.asarray(x, dtype=np.float32)}

    def read(token):
        if token.startswith('0f'):
            return np.float32(struct.unpack('!f', bytes.fromhex(token[2:]))[0])
        return registers[token]

    with np.errstate(all='ignore'):
        for line in emitter.code:
            opcode, rest = line.strip().rstrip(';').split(' ', 1)
            dst, *args = rest.split(', ')
            a = [read(arg) for arg in args]
            if opcode == 'abs.f32':
                value = np.abs(a[0])
            elif opcode == 'mul.rn.f32':
                value = a[0] * a[1]
            elif opcode == 'add.rn.f32':
                value = a[0] + a[1]
            elif opcode == 'div.rn.f32':
                value = a[0] / a[1]
            elif opcode == 'ex2.approx.f32':
                value = np.exp2(a[0].astype(np.float64)).astype(np.float32)
                if perturb:
                    bits = value.view(np.uint32).astype(np.int64)
                    # Perturb finite arguments including results that rounded to zero;
                    # preserve ISA-defined ex2(-inf)=+0 and NaN class.
                    changed = np.maximum(bits + perturb, 0).astype(np.uint32).view(np.float32)
                    value = np.where(np.isfinite(a[0]), changed, value)
            elif opcode == 'setp.le.f32':
                value = a[0] <= a[1]
            elif opcode == 'setp.gt.f32':
                value = a[0] > a[1]
            elif opcode == 'selp.f32':
                value = np.where(a[2], a[0], a[1])
            else:
                raise AssertionError('Unmodelled emitted instruction: ' + opcode)
            registers[dst] = value if opcode.startswith('setp.') else np.asarray(value, dtype=np.float32)
    return registers[dest]


class SoftplusTests(unittest.TestCase):
    def test_emitted_alias_and_special_values(self):
        x = np.array([-np.inf, -np.finfo(np.float32).max, -103.9, -100,
                      -87, -20, -2, -0., 0., 2, 100, np.finfo(np.float32).max, np.inf, np.nan], dtype=np.float32)
        got = simulate(x)
        other = simulate(x, alias=False)
        ref = np.logaddexp(0., x.astype(np.float64)).astype(np.float32)
        np.testing.assert_array_equal(got.view(np.uint32), other.view(np.uint32))
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
        np.testing.assert_array_equal(np.isinf(got), np.isinf(ref))
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=2e-7)
        self.assertEqual(int(got[2].view(np.uint32)), 1, 'smallest tail must survive')
        self.assertEqual(int(got[0].view(np.uint32)), 0)

    def test_emitted_accuracy_and_tail_under_ex2_perturbation(self):
        x = np.unique(np.concatenate([np.linspace(-120, 120, 100001),
                                     np.linspace(-87, -2, 20001),
                                     [-24 * np.log(2)]]).astype(np.float32))
        ref = np.logaddexp(0., x.astype(np.float64)).astype(np.float32)
        tail = (x >= -87) & (x <= -2)
        for delta in (-2, 0, 2):
            with self.subTest(ex2_ulp=delta):
                got = simulate(x, perturb=delta).astype(np.float64)
                np.testing.assert_allclose(got, ref.astype(np.float64), rtol=1e-6, atol=2e-7)
                np.testing.assert_allclose(got[tail], ref[tail].astype(np.float64), rtol=1.2e-5, atol=0)


if __name__ == '__main__':
    unittest.main()
