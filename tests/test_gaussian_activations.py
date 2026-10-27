"""Interpret emitted arithmetic; ex2 is perturbed, never executed on a GPU."""
from collections import defaultdict
from fractions import Fraction
import math
from pathlib import Path
import struct
import sys
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptxgen.compiler import Emitter

OPS = ('erf', 'gelu', 'gelu_tanh')


def reference(op, x):
    x = np.asarray(x, dtype=np.float32).astype(np.float64)
    with np.errstate(all='ignore'):
        if op == 'erf':
            return np.vectorize(math.erf, otypes=[float])(x).astype(np.float32)
        if op == 'gelu':
            return (0.5 * x * np.vectorize(math.erfc, otypes=[float])(
                -x / math.sqrt(2))).astype(np.float32)
        y = 2 * math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)
        e = np.exp(-np.abs(y))
        return np.where(y < 0, x * e / (1 + e), x / (1 + e)).astype(np.float32)


def fma32(a, b, c):
    # The f32 product is exact in f64. TwoSum retains the addition residual,
    # which resolves an f64-to-f32 double-rounding tie. Checked below against
    # exact rational arithmetic, including deliberate midpoint perturbations.
    a, b, c = [np.asarray(v, dtype=np.float32).astype(np.float64) for v in (a, b, c)]
    product = a * b
    total = product + c
    delta = total - product
    residual = (product - (total - delta)) + (c - delta)
    rounded = total.astype(np.float32)
    lower = np.nextafter(rounded, np.float32(-np.inf))
    upper = np.nextafter(rounded, np.float32(np.inf))
    rounded64 = rounded.astype(np.float64)
    tie_up = np.isfinite(rounded64) & (total == (rounded64 + upper.astype(np.float64)) / 2) & (residual > 0)
    tie_down = np.isfinite(rounded64) & (total == (rounded64 + lower.astype(np.float64)) / 2) & (residual < 0)
    return np.where(tie_up, upper, np.where(tie_down, lower, rounded)).astype(np.float32)


def simulate(op, x, alias=True, perturb=0):
    emitter = Emitter.__new__(Emitter)
    emitter.register_kinds = {}
    emitter.code, emitter.counts = [], defaultdict(int)
    source = emitter.reg('f32')
    destination = source if alias else emitter.reg('f32')
    emitter.arithmetic(destination, op, [source])
    registers = {source: np.asarray(x, dtype=np.float32)}

    def read(token):
        if token.startswith('0f'):
            return np.float32(struct.unpack('!f', bytes.fromhex(token[2:]))[0])
        return registers[token]

    with np.errstate(all='ignore'):
        for line in emitter.code:
            opcode, rest = line.strip().rstrip(';').split(' ', 1)
            dst, *arguments = rest.split(', ')
            a = [read(v) for v in arguments]
            if opcode == 'mov.f32': value = a[0]
            elif opcode == 'abs.f32': value = np.abs(a[0])
            elif opcode == 'mul.rn.f32': value = a[0] * a[1]
            elif opcode == 'add.rn.f32': value = a[0] + a[1]
            elif opcode == 'sub.rn.f32': value = a[0] - a[1]
            elif opcode == 'div.rn.f32': value = a[0] / a[1]
            elif opcode == 'fma.rn.f32': value = fma32(*a)
            elif opcode == 'copysign.f32': value = np.copysign(a[1], a[0])
            elif opcode == 'setp.lt.f32': value = a[0] < a[1]
            elif opcode == 'selp.f32': value = np.where(a[2], a[0], a[1])
            elif opcode == 'ex2.approx.f32':
                value = np.exp2(a[0].astype(np.float64)).astype(np.float32)
                if perturb:
                    bits = value.view(np.uint32).astype(np.int64)
                    shifted = np.maximum(bits + perturb, 0).astype(np.uint32).view(np.float32)
                    value = np.where(np.isfinite(a[0]), shifted, value)
            else: raise AssertionError('Unmodelled instruction: ' + opcode)
            registers[dst] = value if opcode.startswith('setp.') else np.asarray(value, dtype=np.float32)
    return registers[destination], '\n'.join(emitter.code)


class GaussianActivationTests(unittest.TestCase):
    def test_fma_model_against_rational_rounding(self):
        rng = np.random.default_rng(63)
        triples = np.ldexp(rng.uniform(-2, 2, (500, 3)), rng.integers(-60, 60, (500, 3))).astype(np.float32)
        midpoint = [[np.float32(1 + 2**-23), 1.5, sign * 2**-100] for sign in (-1, 1)]
        triples = np.vstack([triples, np.asarray(midpoint, dtype=np.float32)])
        got = fma32(*triples.T)
        for row, actual in zip(triples, got):
            a, b, c = [Fraction(float(v)) for v in row]
            exact = a * b + c
            near = np.float32(float(exact))
            candidates = [near, np.nextafter(near, np.float32(-np.inf)),
                          np.nextafter(near, np.float32(np.inf))]
            expected = min(candidates, key=lambda v: (abs(Fraction(float(v)) - exact), int(v.view(np.uint32)) & 1))
            self.assertEqual(actual.view(np.uint32), expected.view(np.uint32))

    def test_specials_aliasing_and_instruction_contract(self):
        s, maximum, tiny = np.nextafter(np.float32(0), np.float32(1)), np.finfo(np.float32).max, np.finfo(np.float32).tiny
        x = np.array([-np.inf, -maximum, -1e20, -20, -14.5, -14.2, -10, -8, -5,
                      -1, -tiny, -s, -0., 0., s, tiny, 1, 5, 20, 1e20, maximum, np.inf, np.nan], dtype=np.float32)
        for op in OPS:
            with self.subTest(op=op):
                got, code = simulate(op, x)
                other, _ = simulate(op, x, alias=False)
                ref = reference(op, x)
                np.testing.assert_array_equal(got.view(np.uint32), other.view(np.uint32))
                np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
                np.testing.assert_array_equal(np.isinf(got), np.isinf(ref))
                # Exact zeros here are checks of the unperturbed CPU model only.
                exact = (ref == 0) | np.isinf(ref) | np.isinf(x) & ~np.isnan(ref)
                np.testing.assert_array_equal(got[exact].view(np.uint32), ref[exact].view(np.uint32))
                np.testing.assert_allclose(got, ref, rtol=3e-6, atol=2e-7)
                self.assertEqual(code.count('ex2.approx.f32'), 1)
                self.assertNotIn('lg2', code)
                self.assertNotIn('tanh.approx', code)
                self.assertNotIn('.ftz', code)

    def test_accuracy_and_negative_tails_under_ex2_perturbation(self):
        rng = np.random.default_rng(2026)
        values = [np.linspace(-20, 20, 100001), np.linspace(-1, 1, 20001),
                  np.linspace(-15, -8, 20001), 4 * rng.normal(size=20000)]
        for threshold in (0.5, math.sqrt(0.5)):
            t = np.float32(threshold)
            around = np.array([np.nextafter(t, np.float32(-np.inf)), t, np.nextafter(t, np.float32(np.inf))])
            values.extend([around, -around])
        x = np.unique(np.concatenate(values).astype(np.float32))
        for op in OPS:
            ref = reference(op, x).astype(np.float64)
            tail = (x >= (-10 if op == 'gelu' else -6)) & (x <= -2)
            for shift in (-2, 0, 2):
                with self.subTest(op=op, ex2_ulp=shift):
                    got, _ = simulate(op, x, perturb=shift)
                    np.testing.assert_allclose(got.astype(np.float64), ref, rtol=3e-6, atol=2e-7)
                    if op != 'erf':
                        np.testing.assert_allclose(got[tail].astype(np.float64), ref[tail], rtol=1.5e-5, atol=0)


if __name__ == '__main__':
    unittest.main()
