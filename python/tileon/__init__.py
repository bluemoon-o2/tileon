"""isort:skip_file"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.1"

from .runtime import (
    KernelInterface,
    reinterpret,
    TileonTensor,
    OutOfResources,
    InterpreterError,
    MockTensor,
)
from .runtime.jit import constexpr_function, jit
from .errors import TileonError

from . import language
from . import testing

must_use_result = language.core.must_use_result

__all__ = [
    "__version__",
    "TileonError",
    "knobs",
    "testing",
    "must_use_result",
    "cdiv",
    "next_power_of_2",
    "jit",
    "constexpr_function",
]


@constexpr_function
def cdiv(x: int, y: int):
    return (x + y - 1) // y


@constexpr_function
def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n
