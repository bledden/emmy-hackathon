"""Stateful lowering of typed statements to PTX.

Owns registers, lexical bindings, memory and participation facts. Numerical
formulas and pure IR analyses live in their corresponding modules.
"""
from __future__ import annotations

import math
import re
import struct
from collections import defaultdict
from contextlib import contextmanager

from .errors import CompileError, require, identifier
from .ir_analysis import (
    COLLECTIVES, S32_MIN, S32_MAX,
    extent,
    walk,
    loop_accumulators,
    written_names,
    definite_body_folds,
    literal_truth,
    definite_conditional_folds,
    warp_only_collectives,
    counted_region,
    uniform_trip_count,
    uniform_tile_names,
    uniform_expression,
    guaranteed_multiple,
)
from .scalars import BYTES, KINDS, PREFIX, instruction_type, literal as scalar_literal, f32

COMPARISONS = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
LOGICAL = {"and": "and", "or": "or", "&&": "and", "||": "or"}
FLOAT_EXPRESSIONS = {"+": "add", "-": "subtract", "*": "multiply"}


class Emitter:
    def __init__(self, graph, node, meta):
        self.graph, self.node, self.kernel, self.meta = graph, node, node.op, meta
        self.code = []
        self.declarations = []
        self.counts = defaultdict(int)
        self.values = {}  # name -> (register, PTX scalar type)
        self.visible = set()  # cached registers do not imply lexical visibility
        self.inherited_scopes = []
        self.combine_updates = frozenset()
        self.combine_kind = None
        self.binding_scopes = []  # prior bindings restored when a lexical scope exits
        self.shadow_scopes = []  # (allow declarations to shadow, protected loop bindings)
        self.boolean_registers = set()  # canonical bool SSA, distinct from integer zero/one
        self.integer_ranges = {}  # current register facts; killed on binding writes
        self.integer_multiples = {}  # s32 induction-register low-zero-bit proofs
        self.loop_expression_cache = {}  # only proven immutable, total s32 expressions
        self.coordinate_ranges = {}  # only immutable, non-wrapping Tile coordinates
        self.coordinate_decodes = {}  # register -> (stride, extent, logical, physical)
        self.uniform_names = set()  # immutable coordinates constant within each CTA
        self.warp_uniform_names = set()
        self.warp_uniform_registers = set()
        self.thread_values = {}  # register -> exact s32 values by physical thread
        launch = meta.get('block') or [0]
        size = launch[0]
        self.active_threads = (frozenset(range(size)) if type(size) is int and 0 < size <= 1024
                               else frozenset())  # None later means unproved
        self.barrier_trace = []  # structured control-flow proof, no emitted code
        self.barrier_loops = []  # enclosing loop trip proofs
        self.uniform_registers = set()  # proven Let results and loop induction values
        self.thread_layouts = {}  # immutable coordinate -> (thread stride, extent)
        self.readonly_global_inputs = frozenset()
        self.buffers = {}  # name -> (base register, shape, address space)
        self.buffer_kinds = {}
        self.register_kinds = {}
        self.labels = 0
        self.dynamic_smem_used = 0
        self.dynamic_smem_align = 16
        self.contract_fma = False
        self.fma_reduction_only = False
        self.fma_accumulation_lines = set()
        self.short_tree = False
        self.index_reuse = False
        self.unroll_loops = False
        self.constant_division = False
        self.shared_loads = 'off'
        self.shared_address_info = {}
        self.unroll_regions = []
        self.optimization_report = None

    def compile(self):
        """Validate → bind → map threads → lower statements → assemble PTX."""
        from . import launch

        tile, block, grid, logical, shape, cells, wide_id = launch.validate_launch(self)
        from .fragments import preflight
        preflight(self, tile, block, grid, cells)
        parameters = launch.bind_parameters(self)
        exit_label = launch.emit_coordinates(
            self, tile, block, grid, logical, shape, cells, wide_id)
        self.body(tile.body)
        launch.finish_body(self, tile, exit_label)
        return launch.render(self, parameters)


    def emit(self, instruction):
        self.code.append("    " + instruction)

    def reg(self, kind):
        prefix = PREFIX[kind]
        index = self.counts[kind]
        self.counts[kind] += 1
        register = f"%{prefix}{index}"
        self.register_kinds[register] = kind
        return register

    def label(self):
        self.labels += 1
        return f"L{self.labels}"

    def bind(self, name, kind="f32"):
        if name in self.values:
            register, old_kind = self.values[name]
            if kind == old_kind:
                self.integer_ranges.pop(register, None)
                self.integer_multiples.pop(register, None)
                self.boolean_registers.discard(register)
                self.uniform_registers.discard(register)
                self.warp_uniform_registers.discard(register)
                self.thread_values.pop(register, None)
                self.coordinate_decodes.pop(register, None)
                self.visible.add(name)
                return register
            # A cached register from an expired sibling scope is not a binding.
            require(name not in self.visible, f"type changed for {name}: {old_kind} to {kind}")
        register = self.reg(kind)
        self.values[name] = (register, kind)
        self.visible.add(name)
        return register

    def declare(self, name, kind="f32"):
        if (self.inherited_scopes and name in self.inherited_scopes[-1]
                and name not in self.combine_updates):
            allowed, protected = self.shadow_scopes[-1]
            require(allowed and name not in protected,
                    f"unsupported lexical shadowing: {name}")
            # Resolve operands before this call, then create a scope-local
            # destination. Later assignments in the same body update that local.
            self.binding_scopes[-1].setdefault(name, self.values.get(name))
            self.values.pop(name, None)
            self.inherited_scopes[-1].discard(name)
        return self.bind(name, kind)

    def read(self, name, kind=None):
        require(name in self.visible, f"value used before definition: {name} (possibly out of scope)")
        register, actual = self.values[name]
        require(kind is None or kind == actual, f"type mismatch for {name}")
        return register

    def is_uniform(self, expression, *, warp=False):
        names = self.uniform_names | (self.warp_uniform_names if warp else set()) | {
            name for name in self.visible
            if self.values[name][0] in (self.uniform_registers | (self.warp_uniform_registers if warp else set()))
        }
        def quotient(expr):
            if expr.op not in ('/', '//') or type(expr.left).__name__ != 'Var':
                return False
            name = expr.left.name
            if name not in self.visible or self.values[name][1] != 's32':
                return False
            decode = self.coordinate_decodes.get(self.values[name][0])
            bounds = self.index_interval(expr.right)
            if (decode is None or bounds is None or bounds[0] != bounds[1] or
                    not 0 < bounds[0] <= S32_MAX or self.expression_kind(expr.right) != 's32'):
                return False
            from .uniformity import quotient_uniform
            stride, size, logical, physical = decode
            return quotient_uniform(stride, size, bounds[0], logical, physical, warp=warp)
        return uniform_expression(expression, names, quotient)

    def physical_values(self, expression):
        from .participation import values
        return values(expression, {name: self.thread_values[register]
                                   for name, (register, kind) in self.values.items()
                                   if name in self.visible and kind == 's32' and
                                   register in self.thread_values}, self.meta.get('block', [0])[0])

    def dtype(self, dtype):
        name = getattr(dtype, "name", dtype)
        require(name in KINDS, f"unsupported dtype {name}")
        return KINDS[name]

    def value_as(self, name, kind):
        from .scalars import convert
        return convert(self, self.read(name), self.values[name][1], kind)

    def state_as(self, name, kind):
        """A typed collective consumes and returns state in its declared dtype.

        The generator can widen a half local maximum to an f32 collective.
        Make that boundary conversion explicit rather than reinterpreting bits
        or treating the old register's type as the new collective's type.
        """
        value = self.value_as(name, kind)
        self.values[name] = (value, kind)
        return value

    def integer(self, op, left, right):
        # Local integer identities keep emitted indexing readable. Arithmetic
        # constants wrap like the 32-bit PTX registers, not Python integers.
        if re.fullmatch(r"-?\d+", left) and re.fullmatch(r"-?\d+", right) and op in ("+", "-", "*"):
            a, b = int(left), int(right)
            value = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b}[op]()
            return str((value + 2**31) % 2**32 - 2**31)
        if (op in ("+", "-") and right == "0") or (op in ("*", "/", "//") and right == "1"):
            return left
        if (op == "+" and left == "0") or (op == "*" and left == "1"):
            return right
        if op == "*" and "0" in (left, right) or op == "%" and right == "1":
            return "0"
        destination = self.reg("s32")
        opcode = {"+": "add", "-": "sub", "*": "mul.lo", "/": "div", "//": "div",
                  "%": "rem", "&": "and", "|": "or", "^": "xor", ">>": "shr",
                  "<<": "shl"}.get(op)
        require(opcode is not None, f"unsupported integer operator {op}")
        suffix = "b32" if op in ("&", "|", "^", "<<") else "s32"
        self.emit(f"{opcode}.{suffix} {destination}, {left}, {right};")
        return destination

    def index_interval(self, expr):
        """Bounds only where every arithmetic intermediate fits signed 32 bits."""
        from .integer_semantics import interval
        return interval(expr, {name: self.integer_ranges[register]
                              for name, (register, kind) in self.values.items()
                              if name in self.visible and kind == "s32" and
                              register in self.integer_ranges})

    def floor_index_operation(self, op, left_expr, right_expr):
        left_bounds, right_bounds = self.index_interval(left_expr), self.index_interval(right_expr)
        require(right_bounds != (0, 0), "integer division/remainder by zero")
        if right_bounds is None or right_bounds[0] <= -1 <= right_bounds[1]:
            require(left_bounds is not None and left_bounds[0] > S32_MIN,
                    "integer division may overflow INT_MIN / -1")
        left, right = self.expression(left_expr), self.expression(right_expr)
        same_sign = (left_bounds is not None and right_bounds is not None and
                     ((left_bounds[0] >= 0 and right_bounds[0] > 0) or
                      (left_bounds[1] <= 0 and right_bounds[1] < 0)))
        if same_sign:
            return self.integer(op, left, right)
        # PTX truncates; Emmy's typed evaluator floors. A nonzero remainder
        # with opposite operand signs needs q-1, or remainder+divisor.
        remainder = self.integer("%", left, right)
        different, nonzero, adjust = [self.reg("pred") for _ in range(3)]
        signs, correction, result = [self.reg("s32") for _ in range(3)]
        self.emit(f"xor.b32 {signs}, {left}, {right};")
        self.emit(f"setp.lt.s32 {different}, {signs}, 0;")
        self.emit(f"setp.ne.s32 {nonzero}, {remainder}, 0;")
        self.emit(f"and.pred {adjust}, {different}, {nonzero};")
        if op == "%":
            self.emit(f"selp.s32 {correction}, {right}, 0, {adjust};")
            self.emit(f"add.s32 {result}, {remainder}, {correction};")
        else:
            quotient = self.integer("/", left, right)
            self.emit(f"selp.s32 {correction}, 1, 0, {adjust};")
            self.emit(f"sub.s32 {result}, {quotient}, {correction};")
        return result

    def expression(self, expression, kind="s32"):
        if kind == 's32' and self.loop_expression_cache:
            from .loop_invariants import key
            cached = self.loop_expression_cache.get(key(expression))
            if cached is not None:
                return cached
        tag = type(expression).__name__
        if tag == "Var":
            return self.read(expression.name, kind)
        if tag == "Literal":
            return scalar_literal(expression.value, kind)
        if tag == "BinaryExpr":
            require(self.expression_kind(expression) == kind,
                    "BinaryExpr result type disagrees with requested type")
            if expression.op in COMPARISONS or expression.op in LOGICAL:
                predicate = self.predicate(expression)
                out = self.reg("s32")
                self.emit(f"selp.s32 {out}, 1, 0, {predicate};")
                return out
            if kind in ("f16", "f32", "f64"):
                args = [self.expression(e, kind) for e in (expression.left, expression.right)]
                out = self.reg(kind)
                self.arithmetic(out, FLOAT_EXPRESSIONS[expression.op], args)
                return out
            if kind == "s64":
                operation = {"+": "add", "-": "subtract", "*": "multiply",
                             "/": "floor_divide", "//": "floor_divide", "%": "remainder",
                             "&": "bitwise_and", "|": "bitwise_or", "^": "bitwise_xor",
                             "<<": "left_shift", ">>": "right_shift"}.get(expression.op)
                require(operation is not None, f"unsupported integer operator {expression.op}")
                args = [self.expression(e, kind) for e in (expression.left, expression.right)]
                out = self.reg(kind)
                self.arithmetic(out, operation, args)
                return out
            if expression.op in ("/", "//", "%"):
                return self.floor_index_operation(expression.op, expression.left, expression.right)
            return self.integer(expression.op, self.expression(expression.left),
                                self.expression(expression.right))
        if tag == "TernaryExpr":
            predicate = self.predicate(expression.cond)
            left = self.expression(expression.if_true, kind)
            right = self.expression(expression.if_false, kind)
            out = self.reg(kind)
            self.emit(f"selp.{instruction_type(kind)} {out}, {left}, {right}, {predicate};")
            return out
        raise CompileError(f"unsupported expression {tag}")

    def expression_kind(self, expression):
        """Infer an untyped Let from its value, keeping index bindings integral."""
        tag = type(expression).__name__
        if tag == "Var":
            self.read(expression.name)
            return self.values[expression.name][1]
        if tag == "Literal":
            dtype = getattr(expression.dtype, "name", expression.dtype)
            if dtype in ("int", "i32"):
                return "s32"
            return self.dtype(dtype)
        if tag == "BinaryExpr":
            left = self.expression_kind(expression.left)
            right = self.expression_kind(expression.right)
            if expression.op in LOGICAL:
                return "s32"  # Logical operands are independently truth-tested.
            require(left == right, "BinaryExpr operands have different types")
            if expression.op in COMPARISONS:
                return "s32"  # A comparison used as a value is canonical 0 or 1.
            require(left not in ("f16", "f32", "f64") or expression.op in FLOAT_EXPRESSIONS,
                    f"unsupported float BinaryExpr operator {expression.op}")
            return left
        if tag == "TernaryExpr":
            left = self.expression_kind(expression.if_true)
            require(left == self.expression_kind(expression.if_false),
                    "untyped Let ternary arms have different types")
            return left
        raise CompileError(f"cannot infer untyped Let expression {tag}")

    def predicate(self, expression):
        tag = type(expression).__name__
        if tag == "BinaryExpr":
            op = expression.op
            result = self.reg("pred")
            comparison = COMPARISONS.get(op)
            if comparison:
                kind = self.expression_kind(expression.left)
                require(kind == self.expression_kind(expression.right),
                        "comparison operands have different types")
                left, right = (self.expression(e, kind)
                               for e in (expression.left, expression.right))
                if kind == "f16":
                    # Widen exactly, including literal operands. PTX half
                    # comparisons do not accept every scalar literal form.
                    from .scalars import convert
                    left, right = [convert(self, v, kind, "f32") for v in (left, right)]
                    kind = "f32"
                # PTX ne is ordered (false for NaN); source != is true for NaN.
                if kind in ("f16", "f32", "f64") and op == "!=":
                    comparison = "neu"
                self.emit(f"setp.{comparison}.{kind} {result}, {left}, {right};")
                return result
            if op in LOGICAL:
                left, right = self.predicate(expression.left), self.predicate(expression.right)
                self.emit(f"{LOGICAL[op]}.pred {result}, {left}, {right};")
                return result
        kind = self.expression_kind(expression)
        value = self.expression(expression, kind)
        if kind == "f16":
            from .scalars import convert
            value = convert(self, value, kind, "f32")
            kind = "f32"
        result = self.reg("pred")
        comparison, zero = (("neu", scalar_literal(0, kind))
                            if kind in ("f16", "f32", "f64") else ("ne", "0"))
        self.emit(f"setp.{comparison}.{kind} {result}, {value}, {zero};")
        return result

    def swizzle_layout(self, mode):
        from .swizzles import parse
        try:
            return parse(mode)
        except ValueError as error:
            raise CompileError(str(error)) from error

    def address(self, name, indices, offset=0, *, swizzle=None):
        require(name in self.buffers, f"unknown buffer {name}")
        base, shape, space = self.buffers[name]
        require(len(indices) == len(shape) or (not indices and math.prod(shape) == 1),
                f"index rank mismatch for {name}")
        flat = "0"
        for index, size in zip(indices, shape):
            flat = self.integer("+", self.integer("*", flat, str(size)), self.expression(index))
        if offset:
            flat = self.integer("+", flat, str(offset))
        if swizzle is not None:
            require(space == "shared", "software swizzle requires a shared destination")
            from .swizzles import emit
            try:
                flat = emit(self, flat, swizzle, shape)
            except ValueError as error:
                raise CompileError(str(error)) from error
        byte_offset = self.reg("u64")
        size = BYTES[self.buffer_kinds.get(name, "f32")]
        self.emit(f"mul.wide.s32 {byte_offset}, {flat}, {size};")
        pointer = self.reg("u64")
        self.emit(f"add.u64 {pointer}, {base}, {byte_offset};")
        if self.shared_loads != 'off' and swizzle is None:
            from .shared_loads import address
            canonical = address(self, name, indices, offset)
            if canonical is not None:
                return canonical
        return pointer, space

    def at_shared(self, name, index):
        base, _, space = self.buffers[name]
        require(space == "shared", f"expected shared buffer {name}")
        offset, ptr = self.reg("u64"), self.reg("u64")
        size = BYTES[self.buffer_kinds.get(name, "f32")]
        self.emit(f"mul.wide.s32 {offset}, {index}, {size};")
        self.emit(f"add.u64 {ptr}, {base}, {offset};")
        return ptr

    def vector_aligned(self, name, indices, width):
        _, shape, _ = self.buffers[name]
        divisor = 0
        for index, size in zip(indices, shape):
            divisor = math.gcd(divisor * size, self.index_multiple(index))
        return divisor % width == 0

    def index_multiple(self, expression):
        return guaranteed_multiple(expression, {
            name: self.integer_multiples[register]
            for name, (register, kind) in self.values.items()
            if name in self.visible and kind == 's32' and register in self.integer_multiples
        })

    def arithmetic(self, destination, op, args):
        kind = self.register_kinds.get(destination, "f32")
        if kind in ('f16', 'f32', 'f64') and op in ('floor_divide', 'remainder', 'mod'):
            from .float_divmod import emit
            return emit(self, destination, op, args, kind)
        if kind != "f32":
            from .scalars import arithmetic
            return arithmetic(self, destination, op, args, kind)
        from .float32_math import emit_math
        return emit_math(self, destination, op, args)

    @contextmanager
    def lexical_scope(self, enabled=True, *, allow_shadow=True, protected=()):
        # Init is a declaration, unlike an Accum update or a seeded drain.
        # Restore lexical visibility and shadowed declaration bindings. Register
        # caching is independent of scope. Owner-loop folds are bound before
        # entering the body; seed=False and explicit outer drains update them.
        if enabled:
            self.binding_scopes.append({})
            previous_visible = self.visible.copy()
            self.inherited_scopes.append(previous_visible.copy())
            allowed = allow_shadow and all(s[0] for s in self.shadow_scopes)
            self.shadow_scopes.append((allowed, frozenset(protected)))
        try:
            yield
        finally:
            if enabled:
                self.visible = previous_visible
                self.inherited_scopes.pop()
                self.shadow_scopes.pop()
                for name, previous in self.binding_scopes.pop().items():
                    if previous is None:
                        self.values.pop(name, None)
                    else:
                        self.values[name] = previous

    def body(self, body, scoped=False, updates=(), protected=(), combine_kind=None):
        previous_updates = self.combine_updates
        previous_kind = self.combine_kind
        self.combine_updates = frozenset(updates)
        self.combine_kind = combine_kind
        try:
            with self.lexical_scope(scoped, protected=protected):
                previous = None
                prior = []
                for statement in body:
                    ready = (type(previous).__name__ == "Sync" and
                             not previous.warp and previous.barrier_id == 0 and
                             previous.count is None)
                    self.statement(statement, tree_input_ready=ready, prior=prior)
                    prior.append(statement)
                    previous = statement
        finally:
            self.combine_updates = previous_updates
            self.combine_kind = previous_kind

    def local_scalar_constant(self, name, prior):
        """Resolve a recent literal/scalar ConstantOp load, without crossing control.

        This is deliberately not general constant propagation. A later definition
        kills the fact, and a memory constant must agree with the launch data and
        have no transform chain or writes anywhere in this kernel.
        """
        value = None
        for stmt in reversed(prior):
            tag = type(stmt).__name__
            if tag in ("Loop", "StridedLoop", "Cond", "WarpShuffle", "TreeHalve"):
                return None
            if tag == "Load" and name in stmt.names:
                source = self.graph.nodes.get(stmt.input)
                if (len(stmt.names) != 1 or source is None or
                        type(source.op).__name__ != "ConstantOp" or source.op.load_ops or
                        len(source.outputs) != 1 or
                        math.prod(extent(n) for n in source.outputs[0].shape) != 1 or
                        not all(type(e).__name__ == "Literal" and e.value == 0 for e in stmt.index) or
                        any(type(s).__name__ == "Write" and s.output == stmt.input
                            for s in walk(self.kernel.body))):
                    return None
                value = source.op.value
                arg = next((a for a in self.meta["args"] if a["name"] == stmt.input), {})
                data = arg.get("data")
                if (type(value) not in (float, int) or not isinstance(data, list) or
                        len(data) != 1 or type(data[0]) not in (float, int) or
                        not math.isfinite(value) or not math.isfinite(data[0]) or
                        abs(value) > float.fromhex('0x1.fffffep127') or
                        abs(data[0]) > float.fromhex('0x1.fffffep127') or
                        f32(value) != f32(data[0])):
                    return None
                break
            if tag in ("Let", "Assign", "Init", "Accum") and stmt.name == name:
                if tag == "Let" and type(stmt.value).__name__ == "Literal":
                    value = stmt.value.value
                elif tag == "Init":
                    value = stmt.identity
                else:
                    return None
                break
        if (type(value) not in (float, int) or not math.isfinite(value) or
                abs(value) > float.fromhex('0x1.fffffep127')):
            return None
        return struct.unpack(">f", bytes.fromhex(f32(value)[2:]))[0]

    def positive_integer_power(self, destination, base, exponent):
        require(exponent is not None and exponent.is_integer() and 0 <= exponent <= 8,
                "pow requires a local compile-time integer exponent in [0, 8]")
        power = int(exponent)
        if power == 0:
            self.emit(f"mov.f32 {destination}, {f32(1)};")
            return
        value = base
        for _ in range(power - 1):
            temporary = self.reg("f32")
            self.emit(f"mul.rn.f32 {temporary}, {value}, {base};")
            value = temporary
        # A destination alias must not overwrite the original base mid-chain.
        self.emit(f"mov.f32 {destination}, {value};")

    def statement(self, stmt, *, tree_input_ready=False, prior=()):
        tag = type(stmt).__name__
        if tag == "Load":
            self.load_statement(stmt)
        elif tag == "Write":
            self.write_statement(stmt)
        elif tag == "Assign":
            self.assign_statement(stmt, prior)
        elif tag == "Let":
            self.let_statement(stmt)
        elif tag == "Init":
            kind = self.dtype(stmt.dtype)
            if self.binding_scopes:
                scope = self.binding_scopes[-1]
                if stmt.name not in scope:
                    scope[stmt.name] = self.values.get(stmt.name)
                self.values.pop(stmt.name, None)
                self.inherited_scopes[-1].discard(stmt.name)
            self.emit(f"mov.{instruction_type(kind)} {self.bind(stmt.name, kind)}, {scalar_literal(stmt.identity, kind)};")
        elif tag == "Accum":
            kind = self.dtype(stmt.dtype)
            destination = self.read(stmt.name, kind)
            self.boolean_registers.discard(destination)
            self.uniform_registers.discard(destination)
            self.warp_uniform_registers.discard(destination)
            self.thread_values.pop(destination, None)
            self.coordinate_decodes.pop(destination, None)
            self.integer_ranges.pop(destination, None)
            self.integer_multiples.pop(destination, None)
            arguments = [self.value_as(stmt.base if stmt.base is not None else stmt.name, kind),
                         self.value_as(stmt.value, kind)]
            if kind == "f32" and stmt.op.name == "add":
                self.fma_accumulation_lines.add(len(self.code))
            self.arithmetic(destination, stmt.op.name, arguments)
        elif tag in ("Loop", "StridedLoop"):
            self.loop(stmt)
        elif tag == "Cond":
            self.conditional(stmt)
        elif tag == "Smem":
            self.shared_memory(stmt)
        elif tag == "CpAsyncCopy":
            self.async_copy(stmt)
        elif tag == "CpAsyncCommit":
            self.emit("cp.async.commit_group;")
        elif tag == "CpAsyncWait":
            require(isinstance(stmt.group, int) and 0 <= stmt.group <= 7,
                    "async wait group must be in [0, 7]")
            self.emit(f"cp.async.wait_group {stmt.group};")
        elif tag == "Sync":
            self.barrier(stmt.barrier_id, stmt.count, stmt.warp)
        elif tag == "WarpShuffle":
            # Partner registers belong to the collective, just like its
            # merge temporaries. Only the carried state is updated outside.
            with self.lexical_scope(allow_shadow=False):
                self.shuffle(stmt)
        elif tag == "TreeHalve":
            with self.lexical_scope(allow_shadow=False):
                self.tree(stmt, input_ready=tree_input_ready)
        elif tag in ("RegFragment", "LdmatrixLoad", "MmaSyncPtx", "RegStore"):
            from .fragments import statement
            statement(self, stmt)
        else:
            raise CompileError(f"unsupported statement {tag}")

    def load_statement(self, stmt):
        kind = self.dtype(stmt.dtype)
        storage = self.buffer_kinds.get(stmt.input, "f32")
        pointer, space = self.address(stmt.input, stmt.index)
        # Read-only same-address loads agree within the proven group. Capture
        # this before a destination can shadow one of its index operands.
        immutable = space == 'global' and stmt.input in self.readonly_global_inputs
        uniform = immutable and all(self.is_uniform(i) for i in stmt.index)
        warp_uniform = immutable and all(self.is_uniform(i, warp=True) for i in stmt.index)
        # Resolve addresses before declarations, preserving self-shadow reads.
        registers = [self.declare(n, kind) for n in stmt.names]
        loaded = registers if kind == storage else [self.reg(storage) for _ in registers]
        self.memory("ld", space, pointer, loaded,
                    self.vector_aligned(stmt.input, stmt.index, len(registers)), kind=storage)
        if kind != storage:
            from .scalars import convert
            for destination, source in zip(registers, loaded):
                value = convert(self, source, storage, kind)
                self.emit(f"mov.{instruction_type(kind)} {destination}, {value};")
        if uniform:
            self.uniform_registers.update(registers)
        if warp_uniform:
            self.warp_uniform_registers.update(registers)

    def write_statement(self, stmt):
        self.dtype(stmt.value_dtype)
        layout = self.swizzle_layout(stmt.swizzle)
        if layout is not None:
            # Write is an elementwise layout: transform each logical value,
            # including vectors crossing an eight-element chunk boundary.
            kind = self.buffer_kinds.get(stmt.output, "f32")
            registers = [self.value_as(n, kind) for n in stmt.values]
            width = len(registers)
            # Within an aligned power-of-two run, XOR is a constant
            # translation if its source and destination fields are above
            # every varying bit. Preserve the vector path in that case.
            contiguous = (not isinstance(layout, str) and width in (1, 2, 4, 8) and
                          layout[0] >= width.bit_length() - 1 and
                          self.vector_aligned(stmt.output, stmt.index, width))
            if contiguous:
                pointer, space = self.address(stmt.output, stmt.index, swizzle=layout)
                if stmt.atomic:
                    self.atomic_add(space, pointer, registers, kind)
                else:
                    self.memory("st", space, pointer, registers, kind=kind)
                return
            for offset, register in enumerate(registers):
                pointer, space = self.address(stmt.output, stmt.index, offset, swizzle=layout)
                if stmt.atomic:
                    self.atomic_add(space, pointer, [register], kind)
                else:
                    self.memory("st", space, pointer, [register], kind=kind)
            return
        pointer, space = self.address(stmt.output, stmt.index)
        kind = self.buffer_kinds.get(stmt.output, "f32")
        registers = [self.value_as(n, kind) for n in stmt.values]
        if stmt.atomic:
            self.atomic_add(space, pointer, registers, kind)
        else:
            self.memory("st", space, pointer, registers,
                        self.vector_aligned(stmt.output, stmt.index, len(registers)), kind=kind)

    def assign_statement(self, stmt, prior):
        from .scalars import promoted_kind
        for name in stmt.args:
            self.read(name)
        boolean = getattr(stmt.dtype, 'name', stmt.dtype) == 'bool'
        kind = ('s32' if boolean else self.dtype(stmt.dtype) if stmt.dtype is not None else
                self.combine_kind or
                promoted_kind(stmt.op.name, [self.values[n][1] for n in stmt.args]))
        from .booleans import emit_assign
        if emit_assign(self, stmt, kind, boolean):
            return
        require(not boolean, f'unsupported boolean SSA operation {stmt.op.name}')
        # Capture facts from the OLD bindings before a self-shadowing Assign
        # declares its destination. Only exact s32 operations qualify.
        assigned_facts = None
        if kind == 's32' and all(self.values[n][1] == 's32' for n in stmt.args):
            from emmy.compiler.ir.expr import BinaryExpr, Var
            operators = {'add': '+', 'subtract': '-', 'multiply': '*',
                         'floor_divide': '//', 'remainder': '%', 'mod': '%',
                         'bitwise_and': '&', 'bitwise_or': '|', 'bitwise_xor': '^',
                         'left_shift': '<<', 'right_shift': '>>'}
            expression = (Var(stmt.args[0]) if stmt.op.name == 'copy' and len(stmt.args) == 1
                else BinaryExpr(operators[stmt.op.name], *(Var(n) for n in stmt.args))
                if stmt.op.name in operators and len(stmt.args) == 2 else None)
            if expression is not None:
                assigned_facts = (self.index_interval(expression), self.is_uniform(expression),
                                  self.is_uniform(expression, warp=True), self.physical_values(expression))
        if stmt.op.name == "bitcast":
            require(len(stmt.args) == 1, "bad arity for bitcast")
            source = self.read(stmt.args[0])
            source_kind = self.values[stmt.args[0]][1]
            require(BYTES[source_kind] == BYTES[kind],
                    "bitcast requires equal storage widths")
            destination = self.declare(stmt.name, kind)
            self.emit(f"mov.b{8 * BYTES[kind]} {destination}, {source};")
            return
        arguments = [self.value_as(n, kind) for n in stmt.args]
        if stmt.op.name == "pow" and kind == "f32":
            require(len(arguments) == 2, "bad arity for pow")
            exponent = self.local_scalar_constant(stmt.args[1], prior)
            destination = self.declare(stmt.name)
            if exponent is not None and exponent.is_integer() and 0 <= exponent <= 8:
                self.positive_integer_power(destination, arguments[0], exponent)
            else:
                from .power import emit_power
                emit_power(self, destination, *arguments)
        else:
            self.arithmetic(self.declare(stmt.name, kind), stmt.op.name, arguments)
        if assigned_facts is not None:
            bounds, uniform, warp_uniform, physical = assigned_facts
            destination = self.read(stmt.name, kind)
            if bounds is not None:
                self.integer_ranges[destination] = bounds
            if uniform:
                self.uniform_registers.add(destination)
            if warp_uniform:
                self.warp_uniform_registers.add(destination)
            if physical is not None:
                self.thread_values[destination] = physical

    def let_statement(self, stmt):
        dtype = getattr(stmt.dtype, "name", stmt.dtype)
        if dtype is None:
            kind = self.expression_kind(stmt.value)
        elif dtype in ("int", "i32"):
            kind = "s32"
        else:
            kind = self.dtype(dtype)
        bounds = self.index_interval(stmt.value) if kind == "s32" else None
        uniform = self.is_uniform(stmt.value)
        warp_uniform = self.is_uniform(stmt.value, warp=True)
        physical = self.physical_values(stmt.value) if kind == "s32" else None
        decode = (self.coordinate_decodes.get(self.read(stmt.value.name, 's32'))
                  if kind == 's32' and type(stmt.value).__name__ == 'Var' and
                  self.expression_kind(stmt.value) == 's32' else None)
        value = self.expression(stmt.value, kind)
        destination = self.declare(stmt.name, kind)
        self.emit(f"mov.{instruction_type(kind)} {destination}, {value};")
        if bounds is not None:
            self.integer_ranges[destination] = bounds
        if uniform:
            self.uniform_registers.add(destination)
        if warp_uniform:
            self.warp_uniform_registers.add(destination)
        if physical is not None:
            self.thread_values[destination] = physical
        if decode is not None:
            self.coordinate_decodes[destination] = decode

    def conditional(self, stmt):
        from .participation import whole_warps
        region = (*stmt.body, *stmt.else_body)
        physical = self.physical_values(stmt.cond)
        constant_truth = literal_truth(stmt.cond)
        if constant_truth is not None:
            physical = (int(constant_truth),) * self.meta.get("block", [0])[0]
        parent_threads = self.active_threads
        cta_uniform = self.is_uniform(stmt.cond)
        if physical is not None and parent_threads is not None:
            yes_threads = frozenset(t for t in parent_threads if physical[t])
            no_threads = parent_threads - yes_threads
        else:
            yes_threads = no_threads = parent_threads if cta_uniform else None
        counted_proof = (counted_region(region) and yes_threads is not None and
                         no_threads is not None and whole_warps(yes_threads) and
                         whole_warps(no_threads))
        require(not any(type(s).__name__ in COLLECTIVES for s in walk(region)) or
                self.is_uniform(stmt.cond, warp=warp_only_collectives(region)) or counted_proof,
                "collective inside conditional requires a proven " +
                ("warp-uniform" if warp_only_collectives(region) else "CTA-uniform or exact counted-participation") + " predicate")
        otherwise, end = self.label(), self.label()
        predicate = self.predicate(stmt.cond)
        # Do not invent a neutral value for a skipped arm. Only bindings
        # definitely seeded on the chosen path may become visible outside.
        # Existing outer names keep their normal lexical shadow semantics.
        export_names = sorted(definite_conditional_folds(stmt) - self.visible)
        # Inspect the finalized visible binding in each arm. Descendant
        # folds with the same spelling may be unrelated lexical shadows.
        export_kinds, exports = {}, {}
        def arm(body, reachable, participants):
            # Keep branch bindings lexical. Copy a finalized result before
            # scope exit rather than making a fictitious outer binding that
            # changes shadowing in nested conditionals or reads before seed.
            defined = definite_body_folds(body)
            self.active_threads = participants
            with self.lexical_scope():
                self.body(body)
                for name in export_names:
                    if reachable and name in defined:
                        actual = self.values[name][1]
                        if name not in exports:
                            export_kinds[name] = actual
                            exports[name] = self.reg(actual)
                        require(actual == export_kinds[name],
                                f"conditional fold types disagree: {name}")
                        destination = exports[name]
                        kind = export_kinds[name]
                        self.emit(f"mov.{instruction_type(kind)} {destination}, {self.read(name, kind)};")
        self.emit(f"@!{predicate} bra {otherwise};")
        with self.capture_barriers() as yes_events:
            arm(stmt.body, constant_truth is not False, yes_threads)
        self.emit(f"bra {end};")
        self.emit(otherwise + ":")
        with self.capture_barriers() as no_events:
            arm(stmt.else_body, constant_truth is not True, no_threads)
        self.active_threads = parent_threads
        if constant_truth is not None:
            self.barrier_trace.extend(yes_events if constant_truth else no_events)
        elif cta_uniform:
            self.barrier_trace.append(('choice', (yes_events, no_events)))
        else:
            # Both physical cohorts may execute concurrently. Full-CTA
            # barriers cannot occur here; retain ownership across arms.
            self.barrier_trace.extend((*yes_events, *no_events))
        self.emit(end + ":")
        for name, register in exports.items():
            self.values[name] = (register, export_kinds[name])
            self.visible.add(name)

    def shared_memory(self, stmt):
        kind = self.dtype(stmt.dtype)
        require(stmt.name not in self.buffers, f"duplicate shared memory {stmt.name}")
        shape = tuple(extent(n) for n in stmt.extents)
        size = BYTES[kind] * math.prod(shape)
        require(size > 0, "empty shared buffer")
        name = "sm_" + identifier(stmt.name)
        align = max(16, stmt.align or 16)
        require(align & (align - 1) == 0, "shared alignment must be power of two")
        pointer = self.reg("u64")
        if self.meta["smem_bytes"]:
            offset = (self.dynamic_smem_used + align - 1) // align * align
            self.dynamic_smem_used = offset + size
            self.dynamic_smem_align = max(self.dynamic_smem_align, align)
            require(self.dynamic_smem_used <= self.meta["smem_bytes"],
                    "shared declarations exceed launch dynamic shared memory")
            self.emit(f"mov.u64 {pointer}, sm_dynamic;")
            if offset:
                self.emit(f"add.u64 {pointer}, {pointer}, {offset};")
        else:
            self.declarations.append(f"    .shared .align {align} .b8 {name}[{size}];")
            self.emit(f"mov.u64 {pointer}, {name};")
        self.buffers[stmt.name] = (pointer, shape, "shared")
        self.buffer_kinds[stmt.name] = kind

    def memory(self, instruction, space, pointer, registers, aligned=True, *, kind="f32"):
        count = len(registers)
        require(count > 0, "empty memory operation")
        size = BYTES[kind]
        if count not in (1, 2, 4) or count * size > 16 or (not aligned and count > 1):
            # Vector instructions require alignment; scalarization preserves the
            # IR's consecutive-value semantics when we cannot prove it.
            for offset, register in enumerate(registers):
                element_pointer = self.reg("u64")
                self.emit(f"add.u64 {element_pointer}, {pointer}, {offset * size};")
                self.memory(instruction, space, element_pointer, [register], kind=kind)
            return
        vector = "" if count == 1 else f".v{count}"
        value = registers[0] if count == 1 else "{" + ", ".join(registers) + "}"
        args = f"{value}, [{pointer}]" if instruction == "ld" else f"[{pointer}], {value}"
        self.emit(f"{instruction}.{space}{vector}.{instruction_type(kind)} {args};")

    def atomic_add(self, space, pointer, registers, kind):
        for offset, value in enumerate(registers):
            address = pointer
            if offset:
                address = self.reg("u64")
                self.emit(f"add.u64 {address}, {pointer}, {offset * BYTES[kind]};")
            # 64-bit integer add is specified as u64; the low 64-bit result is
            # identical for signed two's-complement arithmetic.
            atom_kind = "u64" if kind == "s64" else kind
            if space == "global" and kind != "f16":
                self.emit(f"red.global.add.{atom_kind} [{address}], {value};")
            else:
                old = self.reg(kind)
                noftz = ".noftz" if kind == "f16" else ""
                self.emit(f"atom.{space}.add{noftz}.{atom_kind} {old}, [{address}], {value};")

    def shuffle_value(self, destination, source, kind, distance, clamp, membermask):
        def shuffle(dst, src):
            self.emit(f"shfl.sync.bfly.b32 {dst}, {src}, {distance}, {clamp}, {membermask};")
        if BYTES[kind] == 4:
            shuffle(destination, source)
        elif BYTES[kind] == 8:
            lo, hi, other_lo, other_hi = [self.reg("s32") for _ in range(4)]
            self.emit(f"mov.b64 {{{lo}, {hi}}}, {source};")
            shuffle(other_lo, lo)
            shuffle(other_hi, hi)
            self.emit(f"mov.b64 {destination}, {{{other_lo}, {other_hi}}};")
        else:
            zero, unused = self.reg("f16"), self.reg("f16")
            packed, other = self.reg("s32"), self.reg("s32")
            self.emit(f"mov.b16 {zero}, 0;")
            self.emit(f"mov.b32 {packed}, {{{source}, {zero}}};")
            shuffle(other, packed)
            self.emit(f"mov.b32 {{{destination}, {unused}}}, {other};")

    @contextmanager
    def capture_barriers(self):
        parent, events = self.barrier_trace, []
        self.barrier_trace = events
        try:
            yield events
        finally:
            self.barrier_trace = parent

    def loop(self, stmt):
        self.validate_axis(stmt.axis)
        is_strided = type(stmt).__name__ == "StridedLoop"
        # A carried update later in the body also affects earlier predicates on
        # the next iteration. Lexical declarations shadow; Accum/collectives
        # can mutate existing registers and must kill facts before the backedge.
        for child in walk(stmt.body):
            updates = ((child.name,) if type(child).__name__ == 'Accum' else
                       getattr(child, 'state', ()))
            for name in updates:
                if name in self.visible:
                    self.boolean_registers.discard(self.values[name][0])
                    self.uniform_registers.discard(self.values[name][0])
                    self.warp_uniform_registers.discard(self.values[name][0])
                    self.integer_ranges.pop(self.values[name][0], None)
                    self.integer_multiples.pop(self.values[name][0], None)
                    self.thread_values.pop(self.values[name][0], None)
                    self.coordinate_decodes.pop(self.values[name][0], None)
        step_bounds = self.index_interval(stmt.step) if is_strided else (1, 1)
        collective = any(type(s).__name__ in COLLECTIVES for s in walk(stmt.body))
        static_step = step_bounds is not None and step_bounds[0] == step_bounds[1]
        require(not is_strided or self.expression_kind(stmt.step) == "s32",
                "loop step requires s32")
        require(not static_step or 0 < step_bounds[0] <= S32_MAX,
                "loop step must be positive; positive static step required for collective loops")
        step_value = step_bounds[0] if static_step else 0
        bounds = self.index_interval(stmt.start) if is_strided else (0, 0)
        stop_bounds = (self.index_interval(stmt.end) if is_strided and stmt.end is not None
                       else (extent(stmt.axis.extent), extent(stmt.axis.extent)))
        # The increment is reached only with index < stop. Its upper bound
        # therefore comes from stop even when the initial index is unknown.
        safe_progress = (static_step and stop_bounds is not None and
                         S32_MIN <= stop_bounds[0] <= stop_bounds[1] <= S32_MAX and
                         stop_bounds[1] - 1 + step_value <= S32_MAX)
        from .participation import trips, per_warp
        block = self.meta.get('block', [0])[0]
        starts = self.physical_values(stmt.start) if is_strided else (0,) * block
        stops = (self.physical_values(stmt.end) if is_strided and stmt.end is not None
                 else (extent(stmt.axis.extent),) * block)
        exact_trips = trips(starts, stops, step_value)
        loop_proven = True
        if collective:
            warp = warp_only_collectives(stmt.body)
            uniform_bounds = (not is_strided or self.is_uniform(stmt.start, warp=warp)) and (
                not is_strided or stmt.end is None or self.is_uniform(stmt.end, warp=warp)) and (
                not is_strided or self.is_uniform(stmt.step, warp=warp))
            # Equal numeric bounds may prove equal counts even for varying starts.
            equal_counts = (bounds is not None and stop_bounds is not None and
                            stop_bounds[0] == stop_bounds[1] and
                            uniform_trip_count(*bounds, stop_bounds[0], step_value) is not None)
            # Alternatively each participating group has identical start/end.
            # Different warps/CTAs may then execute different numbers of trips.
            # Group-uniform bounds and step imply the same exits, including
            # the nonpositive-step and widened terminal-increment guards.
            # No interval guess about a runtime value is needed for this proof.
            loop_proven = equal_counts or uniform_bounds
            counted_proof = (counted_region(stmt.body) and self.active_threads is not None and
                             per_warp(exact_trips, self.active_threads))
            require(stmt.axis.name not in written_names(stmt.body) and
                    (loop_proven or counted_proof),
                    "collective loop requires a proven uniform, non-wrapping trip count")
        # Initialize only folds owned by this loop, before its first iteration.
        accumulators = list(loop_accumulators(stmt.body, stmt.axis.name))
        operations = {}
        for accumulator in accumulators:
            previous = operations.setdefault(accumulator.name, accumulator.op.name)
            require(previous == accumulator.op.name,
                    f"mixed reduction operations for {accumulator.name}")
        if stmt.seed:
            seen = set()
            for accumulator in accumulators:
                if accumulator.name in seen:
                    continue
                seen.add(accumulator.name)
                require(accumulator.op.name in ("add", "multiply", "maximum", "minimum"),
                        "unsupported reduction identity")
                # Honor the typed operator's identity. In particular, Emmy's
                # finite max sentinel keeps empty online-softmax lanes from
                # evaluating (-inf)-(-inf) during coupled-state merges.
                kind = self.dtype(accumulator.dtype)
                identity = accumulator.op.identity
                if kind in ("s32", "s64") and accumulator.op.name in ("maximum", "minimum"):
                    bits = 8 * BYTES[kind]
                    identity = -(1 << (bits-1)) if accumulator.op.name == "maximum" else (1 << (bits-1))-1
                self.emit(f"mov.{instruction_type(kind)} {self.declare(accumulator.name, kind)}, {scalar_literal(identity, kind)};")
        start = self.expression(stmt.start) if is_strided else "0"
        step = self.expression(stmt.step) if is_strided else "1"
        stop = self.expression(stmt.end) if is_strided and stmt.end is not None else str(extent(stmt.axis.extent))
        require(stmt.axis.name not in self.visible,
                f"unsupported lexical shadowing of loop axis: {stmt.axis.name}")
        uniform_index = (not is_strided or
                         (self.is_uniform(stmt.start) and self.is_uniform(stmt.step)))
        uniform_index &= stmt.axis.name not in written_names(stmt.body)
        warp_index = (not is_strided or
                      (self.is_uniform(stmt.start, warp=True) and self.is_uniform(stmt.step, warp=True)))
        warp_index &= stmt.axis.name not in written_names(stmt.body)
        # Capture the initial expression before binding the induction variable.
        # A progression preserves gcd(start, step, 2**32). Only immutable axes
        # and static positive steps qualify; body writes invalidate the proof
        # even when the write appears after a use on the first iteration.
        index_multiple = (math.gcd(self.index_multiple(stmt.start), step_value, 2**32)
                          if is_strided and static_step and
                          stmt.axis.name not in written_names(stmt.body) else 1)
        index = self.bind(stmt.axis.name, "s32")
        if index_multiple > 1:
            self.integer_multiples[index] = index_multiple
        if uniform_index:
            self.uniform_registers.add(index)
        if warp_index:
            self.warp_uniform_registers.add(index)
        from .loop_invariants import candidates, key
        previous_cache = self.loop_expression_cache
        self.loop_expression_cache = dict(previous_cache)
        immutable_s32 = {name for name in self.visible if self.values[name][1] == 's32'}
        immutable_s32.discard(stmt.axis.name)
        for expression in candidates(stmt.body, immutable_s32):
            expression_key = key(expression)
            if expression_key not in self.loop_expression_cache:
                value = self.expression(expression)
                self.loop_expression_cache[expression_key] = value
        # Keep induction initialization next to the header. Interposing the
        # hoisted arithmetic can make ptxas miss constant-trip unrolling.
        self.emit(f"mov.s32 {index}, {start};")
        begin, end = self.label(), self.label()
        self.emit(begin + ":")
        predicate = self.reg("pred")
        if not static_step:
            # Group-uniform collective loops take this exit together. For all
            # loops, avoid spinning on a nonpositive runtime step.
            self.emit(f"setp.le.s32 {predicate}, {step}, 0;")
            self.emit(f"@{predicate} bra {end};")
        self.emit(f"setp.ge.s32 {predicate}, {index}, {stop};")
        self.emit(f"@{predicate} bra {end};")
        start_bounds = self.index_interval(stmt.start) if is_strided else (0, 0)
        stop_bounds = (self.index_interval(stmt.end) if is_strided and stmt.end is not None
                       else (extent(stmt.axis.extent), extent(stmt.axis.extent)))
        if (start_bounds is not None and stop_bounds is not None and step_value > 0 and
                max(start_bounds[1], stop_bounds[1] - 1) + step_value <= S32_MAX and
                S32_MIN <= start_bounds[0] <= max(start_bounds[1], stop_bounds[1] - 1)):
            self.integer_ranges[index] = (start_bounds[0], max(start_bounds[1], stop_bounds[1] - 1))
        self.barrier_loops.append((exact_trips, loop_proven))
        try:
            with self.capture_barriers() as events:
                self.body(stmt.body, scoped=True, protected=(*operations, stmt.axis.name))
        finally:
            self.loop_expression_cache = previous_cache
            self.barrier_loops.pop()
            self.integer_ranges.pop(index, None)
            self.integer_multiples.pop(index, None)
            self.uniform_registers.discard(index)
            self.warp_uniform_registers.discard(index)
        # Saturate at two: one iteration checks entry; two check the backedge.
        if exact_trips is not None and self.active_threads is not None:
            counts = [exact_trips[t] for t in self.active_threads]
            low, high = (min(counts), max(counts)) if counts else (0, 0)
        elif static_step and start_bounds is not None and stop_bounds is not None:
            low = max(0, (stop_bounds[0] - start_bounds[1] + step_value - 1) // step_value)
            high = max(0, (stop_bounds[1] - start_bounds[0] + step_value - 1) // step_value)
        else:
            low, high = 0, 2
        self.barrier_trace.append(('loop', events, low == 0, min(high, 2)))
        if not safe_progress:
            current, increment, following = [self.reg('s64') for _ in range(3)]
            self.emit(f"cvt.s64.s32 {current}, {index};")
            self.emit(f"cvt.s64.s32 {increment}, {step};")
            self.emit(f"add.s64 {following}, {current}, {increment};")
            self.emit(f"setp.gt.s64 {predicate}, {following}, {S32_MAX};")
            self.emit(f"@{predicate} bra {end};")
        self.emit(f"add.s32 {index}, {index}, {step};")
        self.emit(f"bra {begin};")
        self.emit(end + ":")
        if (self.unroll_loops and not is_strided and stmt.unroll is True and
                safe_progress and not collective and
                stmt.axis.name not in written_names(stmt.body)):
            self.unroll_regions.append((begin, end, index, predicate, extent(stmt.axis.extent)))
        self.visible.discard(stmt.axis.name)

    def barrier(self, barrier_id=0, count=None, warp=False):
        require(isinstance(barrier_id, int) and 0 <= barrier_id < 16, "invalid barrier id")
        if warp:
            require(count is None, "counted warp barrier unsupported")
            self.emit("bar.warp.sync 0xffffffff;")
        else:
            require(count is None or (isinstance(count, int) and count > 0 and
                    count % 32 == 0 and count <= self.meta["block"][0]),
                    "invalid counted CTA barrier")
            from .participation import same_on, whole_warps
            participants = self.active_threads
            # None is unproved, not an empty branch. An empty static branch
            # executes no barrier and contributes no phase or group ownership.
            require(participants is not None, "unproved counted barrier participants")
            if participants:
                expected = self.meta['block'][0] if count is None else count
                require(len(participants) == expected and whole_warps(participants),
                        "barrier count does not match exact whole-warp participants")
                for trip_counts, proven in self.barrier_loops:
                    require((trip_counts is None and proven) or same_on(trip_counts, participants),
                            "named barrier participants execute different loop phase counts")
                # Statically zero-trip cohorts execute no phase in this loop.
                if not any(trips is not None and all(trips[t] == 0 for t in participants)
                           for trips, _ in self.barrier_loops):
                    self.barrier_trace.append(('barrier', barrier_id, participants,
                                               len(participants) == self.meta['block'][0]))
            suffix = "" if count is None else f", {int(count)}"
            self.emit(f"bar.sync {barrier_id}{suffix};")

    def async_copy(self, stmt):
        from emmy.compiler.ir.expr import BinaryExpr

        require(stmt.nbytes in (4, 8, 16), "invalid asynchronous copy size")
        layout = self.swizzle_layout(stmt.swizzle)
        indices = stmt.smem_index
        if stmt.lane_index is not None:
            require(len(indices) == len(stmt.lane_index) == 2,
                    "lane-offset async copies require two-dimensional shared memory")
            indices = tuple(BinaryExpr("+", base, lane)
                            for base, lane in zip(indices, stmt.lane_index))
        # CpAsyncCopy copies one contiguous byte payload at the transformed
        # START address. It is not a vector of independently permuted writes.
        # Modern XOR preserves low bits; V70 requires a separate proof of
        # transformed-start alignment before selecting an async copy width.
        destination, dst_space = self.address(stmt.smem, indices, swizzle=layout)
        source, src_space = self.address(stmt.src, stmt.src_index)
        require(dst_space == "shared" and src_space == "global",
                "async copy must go from global to shared memory")
        src_kind = self.buffer_kinds.get(stmt.src, "f32")
        dst_kind = self.buffer_kinds.get(stmt.smem, "f32")
        require(src_kind == dst_kind, "async copy buffer element types disagree")
        size = BYTES[src_kind]
        require(stmt.nbytes % size == 0, "async copy splits a buffer element")
        from .swizzles import preserves_alignment
        shape = self.buffers[stmt.smem][1]
        aligned = (preserves_alignment(layout, shape, stmt.nbytes // size) and
                   self.vector_aligned(stmt.smem, indices, stmt.nbytes // size) and
                   self.vector_aligned(stmt.src, stmt.src_index, stmt.nbytes // size))
        four_aligned = (size >= 4 or
                       (preserves_alignment(layout, shape, 4 // size) and
                        self.vector_aligned(stmt.smem, indices, 4 // size) and
                        self.vector_aligned(stmt.src, stmt.src_index, 4 // size)))
        if not four_aligned:
            values = [self.reg(src_kind) for _ in range(stmt.nbytes // size)]
            self.memory("ld", "global", source, values, aligned=False, kind=src_kind)
            self.memory("st", "shared", destination, values, aligned=False, kind=dst_kind)
            return
        chunk = stmt.nbytes if aligned else 4
        for offset in range(0, stmt.nbytes, chunk):
            dst, src = destination, source
            if offset:
                dst, src = self.reg("u64"), self.reg("u64")
                self.emit(f"add.u64 {dst}, {destination}, {offset};")
                self.emit(f"add.u64 {src}, {source}, {offset};")
            dst32 = self.reg("s32")
            self.emit(f"cvt.u32.u64 {dst32}, {dst};")
            cache = "cg" if chunk == 16 else "ca"
            self.emit(f"cp.async.{cache}.shared.global [{dst32}], [{src}], {chunk};")

    def validate_axis(self, axis):
        # Pinned Window base/bound are dormant identity metadata. The local
        # extent and StridedLoop start/end alone determine executed coordinates.
        return None

    def raster(self, tile, gid, shape):
        """Apply a bijective grouped traversal; preserve CTA-local cooperation."""
        if tile.raster_group is None:
            return
        require(tile.raster_axes is not None and len(tile.raster_axes) == 2,
                "grouped raster requires two block axes")
        names = [a.name for a in tile.axes]
        m_name, n_name = tile.raster_axes
        require(m_name in names and n_name in names, "unknown raster axis")
        mi, ni = names.index(m_name), names.index(n_name)
        require(ni == mi + 1, "raster block axes must be consecutive")
        inner = math.prod(shape[ni + 1:])
        cooperative = any(type(s).__name__ in
                          (COLLECTIVES | {"Smem", "CpAsyncCopy", "CpAsyncCommit", "CpAsyncWait"})
                          for s in walk(tile.body))
        # Independent scalar work may put several raster groups in one CTA.
        # Its logical-coordinate permutation does not change participation.
        # Shared memory, asynchronous staging and collectives still require
        # whole-CTA groups: those depend on physical neighbours and barriers.
        require(inner == (getattr(tile, 'block_threads', None) or self.meta["block"][0]) or not cooperative,
                "raster must preserve whole CTAs for cooperative work")
        group = tile.raster_group
        require(isinstance(group, int) and group > 0, "invalid raster group")
        require(tile.raster_orient in ("m", "n"), "invalid raster orientation")
        m, n = shape[mi], shape[ni]
        wide = self.register_kinds.get(gid) == 'u64'
        if wide:
            from .wide_tile import arithmetic, narrow
            integer = lambda op, a, b: arithmetic(self, op, a, b)
        else:
            integer = self.integer
        sub = integer("%", integer("/", gid, str(inner)), str(m * n))
        first_name, second_name, first_size, second_size = (
            (m_name, n_name, m, n) if tile.raster_orient == "m" else
            (n_name, m_name, n, m))
        if wide:
            group = min(group, first_size)  # equivalent stripe, bounded product
        stripe = group * second_size
        first = integer("*", integer("/", sub, str(stripe)), str(group))
        remaining = integer("-", str(first_size), first)
        active = self.reg("u64" if wide else "s32")
        self.emit(f"min.{'u64' if wide else 's32'} {active}, {remaining}, {group};")
        local = integer("%", sub, str(stripe))
        first_index = integer("+", first, integer("%", local, active))
        second_index = integer("/", local, active)
        if wide:
            first_index, second_index = narrow(self, first_index), narrow(self, second_index)
        self.emit(f"mov.s32 {self.bind(first_name, 's32')}, {first_index};")
        self.emit(f"mov.s32 {self.bind(second_name, 's32')}, {second_index};")

    def shuffle(self, stmt):
        kind = self.dtype(stmt.dtype)
        for name in stmt.state:
            self.state_as(name, kind)
        self.validate_combine(stmt)
        length = stmt.length
        require(length in (1, 2, 4, 8, 16, 32), "shuffle width must be a power of two <=32")
        block = self.meta["block"][0]
        remainder = block % 32
        require(remainder % length == 0, "partial warp must contain complete shuffle groups")
        if length == 1:
            return
        # Every live CTA thread reaches this instruction (checked at the Tile
        # and at every control-flow ancestor). Name exactly those physical
        # lanes, independent of incidental scheduling or rebound IR lane/warp.
        membermask = "0xffffffff"
        if remainder:
            membermask = hex((1 << remainder) - 1)
            if block > 32:
                last, mask = self.reg("pred"), self.reg("s32")
                self.emit(f"setp.ge.s32 {last}, {self.thread}, {block - remainder};")
                self.emit(f"selp.b32 {mask}, {membermask}, 0xffffffff, {last};")
                membermask = mask
        clamp = ((32 - length) << 8) | 31
        distance = length // 2
        while distance:
            # All partner values must be captured before coupled state is updated.
            for name, partner in zip(stmt.state, stmt.state_b):
                self.shuffle_value(self.declare(partner, kind), self.read(name, kind),
                                   kind, distance, clamp, membermask)
            self.body(stmt.combine_states, scoped=True, updates=stmt.state, combine_kind=kind)
            distance //= 2

    def validate_combine(self, stmt):
        require(bool(stmt.state) and len(stmt.state) == len(stmt.state_b),
                "reduction state arity mismatch")
        names = (*stmt.state, *stmt.state_b)
        require(len(names) == len(set(names)), "reduction state and partner names must be distinct")
        # The typed contract is tuple[Assign,...], not an arbitrary statement
        # region. Tree combines execute only on the writers of active nodes.
        require(all(type(s).__name__ == "Assign" for s in stmt.combine_states),
                "reduction combine supports only Assign statements")

    def short_tree_reason(self, stmt, input_ready):
        """Eligibility for migrating a flat tree's merge work to one warp.

        The immediate full-CTA Sync is deliberately a narrow dominance rule.
        Reading a per-thread outer value in the merge would change its meaning
        when the original writer's work migrates to another physical lane.
        """
        if self.dtype(stmt.dtype) != "f32":
            return "short-tree optimization is scoped to f32"
        if not input_ready:
            return "requires immediately preceding full-CTA Sync"
        if stmt.inner is not None:
            return "segmented tree"
        if not 4 <= stmt.length <= 32:
            return "length outside 4..32"
        if stmt.barrier_id != 0 or stmt.barrier_count is not None:
            return "named or counted barrier"
        if self.meta["block"][0] < 32:
            return "requires a complete physical warp"
        available = set(stmt.state) | set(stmt.state_b)
        for assignment in stmt.combine_states:
            if any(name not in available for name in assignment.args):
                return "combine reads an outer value"
            available.add(assignment.name)
        return None

    def warp_tree(self, stmt):
        """Preserve the ordered halving tree and every final shared slot.

        The whole first physical warp enters every shuffle. Only length lanes
        load shared state; the others carry unused zero values. Full participation
        avoids the assembler's additional paths for partial member masks.
        Capturing every component before the merge preserves coupled state.
        """
        participating, finish = self.reg("pred"), self.label()
        self.emit(f"setp.lt.s32 {participating}, {self.thread}, 32;")
        # tid < 32 selects a whole physical warp, so this branch is uniform
        # within every warp. Preserve that fact for the PTX assembler.
        self.emit(f"@!{participating} bra.uni {finish};")
        loads = self.reg("pred")
        self.emit(f"setp.lt.s32 {loads}, {self.thread}, {stmt.length};")
        pointers = []
        for buffer, name in zip(stmt.bufs, stmt.state):
            pointer = self.at_shared(buffer, self.thread)
            pointers.append(pointer)
            self.emit(f"mov.f32 {self.bind(name)}, 0f00000000;")
            self.emit(f"@{loads} ld.shared.f32 {self.bind(name)}, [{pointer}];")
        mask = "0xffffffff"
        clamp = ((32 - stmt.length) << 8) | 31
        distance = stmt.length // 2
        while distance:
            for name, partner in zip(stmt.state, stmt.state_b):
                self.emit(f"shfl.sync.bfly.b32 {self.declare(partner)}, {self.read(name)}, "
                          f"{distance}, {clamp}, {mask};")
            active, done = self.reg("pred"), self.label()
            self.emit(f"setp.lt.s32 {active}, {self.thread}, {distance};")
            self.emit(f"@!{active} bra {done};")
            self.body(stmt.combine_states, scoped=True, updates=stmt.state, combine_kind="f32")
            self.emit(done + ":")
            distance //= 2
        # The original tree writes precisely slots [0, length/2). Later levels
        # overwrite a prefix; each lane now holds that slot's final value.
        writer, published = self.reg("pred"), self.label()
        self.emit(f"setp.lt.s32 {writer}, {self.thread}, {stmt.length // 2};")
        self.emit(f"@!{writer} bra {published};")
        for pointer, name in zip(pointers, stmt.state):
            self.emit(f"st.shared.f32 [{pointer}], {self.read(name)};")
        self.emit(published + ":")
        self.emit(finish + ":")
        self.barrier()
        for buffer, name in zip(stmt.bufs, stmt.state):
            pointer = self.at_shared(buffer, "0")
            self.emit(f"ld.shared.f32 {self.bind(name)}, [{pointer}];")

    def tree(self, stmt, *, input_ready=False):
        kind = self.dtype(stmt.dtype)
        for name in stmt.state:
            if name in self.visible:
                self.state_as(name, kind)
        self.validate_combine(stmt)
        require(stmt.length > 0 and stmt.length & (stmt.length - 1) == 0, "tree length must be power of two")
        require(len(stmt.bufs) == len(stmt.state) == len(stmt.state_b), "tree state arity mismatch")
        require(len(set(stmt.bufs)) == len(stmt.bufs), "tree components require distinct shared buffers")
        index = self.read(stmt.tid_var, "s32")
        mapped = (stmt.tid_var in self.thread_layouts and
                  (stmt.inner is None or stmt.inner[0] in self.thread_layouts))
        tid_stride, tid_size = self.thread_layouts.get(stmt.tid_var, (0, 0))
        if mapped:
            require(stmt.length <= tid_size, "tree length exceeds participating coordinates")
        inner, scale, inner_stride = "0", 1, 0
        if stmt.inner is not None:
            inner_name, scale = stmt.inner
            inner_stride, inner_size = self.thread_layouts.get(inner_name, (0, scale))
            require(type(scale) is int and scale > 0 and (not mapped or scale == inner_size),
                    "tree inner scale must match coordinate extent")
            inner = self.read(inner_name, "s32")
        block = self.meta["block"][0]
        participants = self.active_threads
        require(participants is not None, "tree needs exact participants")
        partial = participants != frozenset(range(block))
        explicit_leaders = partial or not mapped
        leaders = set()
        if explicit_leaders and participants:
            # Logical coordinates can alias in an auxiliary band. Choose one
            # actual participating thread per slot, never an absent low tid.
            tid_values = self.thread_values.get(index)
            inner_values = self.thread_values.get(inner) if stmt.inner is not None else (0,) * block
            require(tid_values is not None and inner_values is not None,
                    "tree index needs an immutable CTA-local coordinate or exact coordinate values")
            require(all(tid_values[t] >= 0 and 0 <= inner_values[t] < scale for t in participants),
                    "tree coordinates contain invalid shared slots")
            require(stmt.length * scale <= len(participants),
                    "tree participant group misses a shared slot")
            for segment in range(stmt.length):
                for slot in range(scale):
                    candidates = [t for t in sorted(participants)
                                  if tid_values[t] == segment and inner_values[t] == slot]
                    require(candidates, "tree participant group misses a shared slot")
                    leaders.add(candidates[0])
        # Choose one physical writer per (segment, slot). Verify the inverse
        # mapping exhaustively over the small, static CTA before emitting it.
        for segment in range(stmt.length if mapped else 0):
            for slot in range(scale):
                writer = segment * tid_stride + slot * inner_stride
                require(writer < block and (writer // tid_stride) % tid_size == segment and
                        (stmt.inner is None or (writer // inner_stride) % scale == slot),
                        "tree coordinates do not define independent writer slots")
        for buffer in stmt.bufs:
            require(buffer in self.buffers and self.buffers[buffer][2] == "shared" and
                    math.prod(self.buffers[buffer][1]) >= stmt.length * scale and
                    self.buffer_kinds.get(buffer, "f32") == kind,
                    "tree shared buffer is missing or too small")
        if self.short_tree and not explicit_leaders:
            reason = self.short_tree_reason(stmt, input_ready)
            if self.optimization_report is not None:
                self.optimization_report.append({"pass": "short_tree", "length": stmt.length,
                    "components": len(stmt.state), "applied": reason is None,
                    "reason": reason or "flat closed combine after full-CTA Sync"})
            if reason is None:
                self.warp_tree(stmt)
                return
        def offset(segment):
            return self.integer("+", self.integer("*", segment, str(scale)), inner)
        classic = (mapped and stmt.tid_var == "warp" and stmt.inner is None and
                   self.thread_layouts["warp"] == (32, block // 32) and
                   self.thread_layouts.get("lane") == (1, 32))
        writer = (None if classic or not mapped else self.integer("+", self.integer("*", index, str(tid_stride)),
                                                   self.integer("*", inner, str(inner_stride))))
        selected = None
        if explicit_leaders:
            selected = self.reg("pred")
            self.emit(f"setp.eq.s32 {selected}, 0, 1;")
            # Coalesce contiguous leader runs to avoid a predicate per lane.
            pending = sorted(leaders)
            while pending:
                start = end = pending.pop(0)
                while pending and pending[0] == end + 1:
                    end = pending.pop(0)
                lo, hi, member = self.reg("pred"), self.reg("pred"), self.reg("pred")
                self.emit(f"setp.ge.s32 {lo}, {self.thread}, {start};")
                self.emit(f"setp.le.s32 {hi}, {self.thread}, {end};")
                self.emit(f"and.pred {member}, {lo}, {hi};")
                self.emit(f"or.pred {selected}, {selected}, {member};")
        distance = stmt.length // 2
        while distance:
            active, leader, both = self.reg("pred"), self.reg("pred"), self.reg("pred")
            self.emit(f"setp.lt.s32 {active}, {index}, {distance};")
            if explicit_leaders:
                leader = selected
            elif classic:
                self.emit(f"setp.eq.s32 {leader}, {self.read('lane')}, 0;")
            else:
                self.emit(f"setp.eq.s32 {leader}, {self.thread}, {writer};")
            self.emit(f"and.pred {both}, {active}, {leader};")
            end = self.label()
            self.emit(f"@!{both} bra {end};")
            partner_index = self.integer("+", index, str(distance))
            pointers = []
            for buffer, name, partner in zip(stmt.bufs, stmt.state, stmt.state_b):
                left = self.at_shared(buffer, offset(index))
                right = self.at_shared(buffer, offset(partner_index))
                self.memory("ld", "shared", left, [self.bind(name, kind)], kind=kind)
                self.memory("ld", "shared", right, [self.declare(partner, kind)], kind=kind)
                pointers.append(left)
            self.body(stmt.combine_states, scoped=True, updates=stmt.state, combine_kind=kind)
            for ptr, name in zip(pointers, stmt.state):
                self.memory("st", "shared", ptr, [self.read(name, kind)], kind=kind)
            self.emit(end + ":")
            self.barrier(stmt.barrier_id, stmt.barrier_count)
            distance //= 2
        # Every thread receives the result, including consumers in a later pass.
        for buffer, name in zip(stmt.bufs, stmt.state):
            pointer = self.at_shared(buffer, inner)
            self.memory("ld", "shared", pointer, [self.bind(name, kind)], kind=kind)
