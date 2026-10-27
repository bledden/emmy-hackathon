"""Optional numerical transformations of our own emitted instruction stream."""
from collections import Counter
import re

REGISTER = r"%f[0-9]+"
OPERAND = REGISTER
MUL = re.compile(rf"    mul\.rn\.f32 ({REGISTER}), ({OPERAND}), ({OPERAND});")
ADD = re.compile(rf"    add\.rn\.f32 ({REGISTER}), ({OPERAND}), ({OPERAND});")


def contract_fma(code, *, allowed_adds=None):
    """Fuse adjacent mul/add with a globally single-use temporary.

    allowed_adds optionally restricts fusion to instruction positions marked
    by the typed emitter (auto mode uses Accum-add provenance). None means the
    explicit broad opt-in policy; an empty set permits no transformations.

    This is deliberately not exact-rounding preservation: the product is no
    longer rounded before addition. It must be selected explicitly. No FTZ,
    subtraction, motion across instructions, or predicated fusion is performed.
    All operands must be registers. Literal-bearing approximation formulas are
    intentionally outside this version's numerical qualification scope.
    Whole-stream token counts reject any other definition/use of the temporary,
    including loop-carried, in-place, vector, and differently typed uses.
    """
    counts = Counter(re.findall(r"%[A-Za-z_][A-Za-z_0-9]*", "\n".join(code)))
    output, index = [], 0
    while index < len(code):
        mul = MUL.fullmatch(code[index])
        add = ADD.fullmatch(code[index + 1]) if index + 1 < len(code) else None
        if mul and add and (allowed_adds is None or index + 1 in allowed_adds):
            temporary, a, b = mul.groups()
            destination, left, right = add.groups()
            # Two occurrences = this definition and exactly one add operand.
            # A matching destination or a multiply input would add an occurrence.
            if counts[temporary] == 2 and (left == temporary) != (right == temporary):
                c = right if left == temporary else left
                output.append(f"    fma.rn.f32 {destination}, {a}, {b}, {c};")
                index += 2
                continue
        output.append(code[index])
        index += 1
    return output
