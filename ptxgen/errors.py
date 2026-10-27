"""Compiler refusals and PTX identifier validation."""
import re


class CompileError(ValueError):
    pass



def require(condition, message):
    if not condition:
        raise CompileError(message)



def identifier(name):
    require(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name), f"invalid identifier {name!r}")
    return name
