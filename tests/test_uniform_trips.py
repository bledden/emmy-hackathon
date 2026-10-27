"""Check the proof against direct integer-loop execution, including boundaries."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptxgen.compiler import S32_MAX, S32_MIN, uniform_trip_count


class UniformTripTests(unittest.TestCase):
    def test_small_intervals_against_execution(self):
        for low in range(-5, 6):
            for high in range(low, 7):
                for stop in range(-7, 12):
                    for step in range(1, 8):
                        counts = {len(range(start, stop, step))
                                  for start in range(low, high + 1)}
                        expected = counts.pop() if len(counts) == 1 else None
                        self.assertEqual(uniform_trip_count(low, high, stop, step), expected)

    def test_signed_boundary_and_terminal_increment(self):
        cases = [(S32_MAX - 7, S32_MAX - 4, S32_MAX - 3, 4, 1),
                 (S32_MAX - 5, S32_MAX - 5, S32_MAX, 4, None),
                 (S32_MIN, S32_MIN + 3, S32_MIN + 8, 4, 2),
                 (S32_MIN, S32_MIN, S32_MAX, S32_MAX, None),
                 (S32_MAX, S32_MAX, S32_MAX, 1, 0),
                 (0, 127, 512, 128, 4), (0, 127, 257, 128, None),
                 (0, 127, 0, 128, 0), (0, 127, 17, 128, None),
                 (0, 0, S32_MAX + 1, 1, None),
                 (0, 0, 1, S32_MAX + 1, None),
                 (S32_MIN - 1, 0, 1, 1, None)]
        for low, high, stop, step, expected in cases:
            with self.subTest(bounds=(low, high, stop, step)):
                self.assertEqual(uniform_trip_count(low, high, stop, step), expected)


if __name__ == '__main__':
    unittest.main()
