"""The loop policy selects existing exact PTX without name/shape lookup."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptxgen.compiler import compile_file, CompileError
from ptxgen.index_reuse import reuse_integer_expressions, reuse_loop_integer_expressions
from index_model import execute

TASK = Path(os.environ.get('EMMY_TASK_ROOT', ROOT.parent/'upstream/emmy-hackathon-2026'))


def loop(divisor=3, operand='%r0', target='again'):
    return ['    mov.s32 %r0, 0;', '    mov.s32 %r9, 0;', target+':',
            f'    div.s32 %r1, {operand}, {divisor};',
            f'    div.s32 %r2, {operand}, {divisor};',
            '    add.s32 %r9, %r9, %r2;', '    add.s32 %r0, %r0, 1;',
            '    setp.lt.s32 %p0, %r0, 8;', f'    @%p0 bra {target};']


class IndexLoopPolicyTests(unittest.TestCase):
    def select(self, code):
        report = []
        result = reuse_loop_integer_expressions(code, report)
        return result, report[0]

    def test_selected_output_is_exactly_full_pass_and_executes_identically(self):
        for divisor in (1, 3, 7, 2147483647):
            code = loop(divisor)
            result, report = self.select(code)
            self.assertTrue(report['selected'])
            self.assertEqual(result, reuse_integer_expressions(code))
            for rounding in ('truncate', 'floor'):
                self.assertEqual(execute(result, {}, None, rounding), execute(code, {}, None, rounding))

    def test_register_and_label_renaming_do_not_change_selection(self):
        code = loop().copy()
        renamed = [line.replace('%r0', '%r100').replace('%r1,', '%r101,')
                   .replace('%r2', '%r102').replace('%p0', '%p77')
                   .replace('again', 'some_other_label') for line in code]
        self.assertTrue(self.select(renamed)[1]['selected'])
        self.assertEqual(self.select(renamed)[1]['replacements'], self.select(code)[1]['replacements'])

    def test_straight_line_and_forward_branch_leave_all_code_unchanged(self):
        straight = loop()[3:5]
        for code in (straight, ['    bra later;']+straight+['later:'],
                     ['    bra end;', 'again:']+straight+['end:']):
            result, report = self.select(code)
            self.assertEqual(result, code)
            self.assertFalse(report['selected'])
            self.assertEqual(report['replacements'], 0)
            self.assertEqual(report['changes'], [])

    def test_loop_with_only_add_reuse_or_div_reuse_outside_loop_is_skipped(self):
        add_loop = [line.replace('div.s32', 'add.s32') for line in loop()]
        for code in (add_loop, loop()[3:5]+add_loop):
            result, report = self.select(code)
            self.assertEqual(result, code)
            self.assertFalse(report['selected'])

    def test_barriers_predication_overwrites_and_exceptional_division_do_not_trigger(self):
        for instruction in ('    bar.sync 0;', '    mov.s32 %r0, 99;',
                            '    @%p1 mov.s32 %r0, 1;'):
            code = loop(); code.insert(4, instruction)
            self.assertFalse(self.select(code)[1]['selected'])
        for divisor in (0, -1, '%r7'):
            self.assertFalse(self.select(loop(divisor))[1]['selected'])

    def test_remainder_and_nested_loop_are_selected_without_double_counting(self):
        code = [line.replace('div.s32', 'rem.s32') for line in loop()]
        code = ['outer:']+code+['    bra outer;']
        _, report = self.select(code)
        self.assertTrue(report['selected'])
        self.assertEqual(len(report['loop_divrem_lines']), 1)
        self.assertEqual(len(report['backward_edges']), 2)

    def test_unresolved_duplicate_or_unsupported_branch_declines_selection(self):
        for code in (loop()+['again:'], loop()+['    bra missing;'],
                     loop()+['    bra.indirect %r3;']):
            self.assertEqual(self.select(code)[0], code)
            self.assertFalse(self.select(code)[1]['selected'])

    def test_selected_kernel_retains_full_pass_edits_outside_loop(self):
        code = ['    add.s32 %r7, %r5, 1;', '    add.s32 %r8, %r5, 1;']+loop()
        result, report = self.select(code)
        self.assertTrue(report['selected'])
        self.assertEqual(report['replacements'], 2)
        self.assertEqual(result, reuse_integer_expressions(code))

    def test_cli_modes_api_validation_and_default_are_preserved(self):
        for case, selected in (('m01_matmul_512', True), ('m02_matmul_rect', True),
                               ('a01_attention', False), ('p01_vector_add', False)):
            path = next((TASK/'problems'/case).glob('*.ir.json'))
            expected = compile_file(path, contract_fma='auto', index_reuse=selected)
            report = []
            self.assertEqual(compile_file(path, contract_fma='auto', index_reuse='loop',
                                          optimization_report=report), expected)
            self.assertEqual(report[-1]['selected'], selected)
            for mode in (None, '0', '1', 'loop', 'auto', ''):
                env = {k:v for k,v in os.environ.items() if k != 'EMMY_PTX_INDEX_REUSE'}
                env.update(EMMY_PTX_CONSTANT_DIVISION='0',EMMY_PTX_UNROLL='0',EMMY_PTX_SHARED_LOADS='off',EMMY_PTX_CONTRACT_FMA='auto', EMMY_PTX_SHORT_TREE='0')
                if mode is not None: env['EMMY_PTX_INDEX_REUSE'] = mode
                p = subprocess.run([sys.executable, str(ROOT/'compile.py'), str(path)],
                                   capture_output=True, text=True, env=env)
                if mode == '':
                    self.assertNotEqual(p.returncode, 0); self.assertFalse(p.stdout)
                else:
                    want = expected if mode in (None, 'auto', 'loop') else compile_file(path, contract_fma='auto', index_reuse=mode == '1')
                    self.assertEqual(p.returncode, 0, p.stderr); self.assertFalse(p.stderr)
                    self.assertEqual(p.stdout, want)
        for mode in ('AUTO', '1', 1, None):
            with self.assertRaises(CompileError): compile_file(path, index_reuse=mode)


if __name__ == '__main__': unittest.main()
