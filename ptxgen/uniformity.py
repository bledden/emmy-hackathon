"""Uniformity of quotient boundaries in immutable mixed-radix coordinates."""


def quotient_uniform(stride, extent, divisor, logical, physical, *, warp=False):
    """Sufficient proof for floor(((gid // stride) % extent) / divisor).

    A coordinate changes quotient at stride*divisor boundaries and wraps at
    stride*extent boundaries. Neither may split a synchronization group. With
    auxiliary threads gid wraps within the logical compute interval, while warp
    boundaries remain physical. Constant-zero quotients need no alignment.
    """
    if min(stride, extent, divisor, logical, physical) <= 0:
        return False
    if extent <= divisor:
        return True
    if warp and (logical % 32 or physical % 32):
        return False
    group = 32 if warp else logical
    return stride * divisor % group == 0 and stride * extent % group == 0
