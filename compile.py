"""Compile one Emmy Kernel IR using the adjacent problem launch metadata."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from ptxgen.compiler import CompileError, compile_file


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python compile.py path/to/kernel.ir.json", file=sys.stderr)
        return 2
    try:
        mode = os.environ.get("EMMY_PTX_CONTRACT_FMA", "auto")
        if mode not in ("0", "1", "auto"):
            raise CompileError("EMMY_PTX_CONTRACT_FMA must be 0, 1, or auto")
        short_tree = os.environ.get("EMMY_PTX_SHORT_TREE", "0")
        if short_tree not in ("0", "1"):
            raise CompileError("EMMY_PTX_SHORT_TREE must be 0 or 1")
        reuse = os.environ.get("EMMY_PTX_INDEX_REUSE", "auto")
        if reuse not in ("0", "1", "loop", "auto"):
            raise CompileError("EMMY_PTX_INDEX_REUSE must be 0, 1, loop, or auto")
        unroll = os.environ.get("EMMY_PTX_UNROLL", "1")
        if unroll not in ("0", "1"):
            raise CompileError("EMMY_PTX_UNROLL must be 0 or 1")
        division = os.environ.get("EMMY_PTX_CONSTANT_DIVISION", "1")
        if division not in ("0", "1"):
            raise CompileError("EMMY_PTX_CONSTANT_DIVISION must be 0 or 1")
        shared_loads = os.environ.get("EMMY_PTX_SHARED_LOADS", "compact")
        if shared_loads not in ('off', 'address', 'vector', 'compact'):
            raise CompileError('EMMY_PTX_SHARED_LOADS must be off, address, vector, or compact')
        source = compile_file(Path(sys.argv[1]), contract_fma="auto" if mode == "auto" else mode == "1",
                              shared_loads=shared_loads,
                              constant_division=division == "1",
                              unroll_loops=unroll == "1", short_tree=short_tree == "1", index_reuse=reuse if reuse in ("loop", "auto") else reuse == "1")
    except Exception as exc:
        print(f"compile error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
