"""Pinned Emmy scalar aliases reuse existing qualified lowering paths."""
from collections import defaultdict
import unittest

from emmy.compiler.ir.elementwise import ElementwiseImpl
from ptxgen.compiler import CompileError, Emitter


class ArithmeticAliasTests(unittest.TestCase):
    def emit(self, operation, arguments, destination='%f1'):
        emitter = Emitter.__new__(Emitter)
        emitter.register_kinds = {}
        emitter.code = []
        emitter.counts = defaultdict(int, f32=2)
        emitter.arithmetic(destination, operation, arguments)
        return emitter.code

    def test_alias_matches_existing_lowering_with_and_without_destination_alias(self):
        for alias, original in (('pad', 'copy'), ('exp_fast', 'exp')):
            self.assertEqual(ElementwiseImpl(alias).arity, 1)
            for destination in ('%f0', '%f1'):
                self.assertEqual(self.emit(alias, ['%f0'], destination),
                                 self.emit(original, ['%f0'], destination))

    def test_bad_arity_is_not_silently_ignored(self):
        for operation in ('pad', 'exp_fast'):
            for arguments in ([], ['%f0', '%f1']):
                with self.assertRaisesRegex(CompileError, 'bad arity'):
                    self.emit(operation, arguments)


if __name__ == '__main__':
    unittest.main()
