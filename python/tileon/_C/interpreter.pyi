"""
Tilen Interpreter API
"""
try:
    import numpy
except ImportError:
    numpy = None
from __future__ import annotations
import collections.abc
import typing
__all__: list[str] = ['MEM_SEMANTIC', 'RMW_OP', 'atomic_cas', 'atomic_rmw', 'convert_float', 'load', 'parallel_launch', 'store']
class MEM_SEMANTIC:
    """
    Memory semantic for atomic operations.
    
    Members:
    
      ACQUIRE_RELEASE : Acquire and release memory
    
      ACQUIRE : Acquire memory
    
      RELEASE : Release memory
    
      RELAXED : Relaxed memory semantic
    """
    ACQUIRE: typing.ClassVar[MEM_SEMANTIC]  # value = <MEM_SEMANTIC.ACQUIRE: 1>
    ACQUIRE_RELEASE: typing.ClassVar[MEM_SEMANTIC]  # value = <MEM_SEMANTIC.ACQUIRE_RELEASE: 0>
    RELAXED: typing.ClassVar[MEM_SEMANTIC]  # value = <MEM_SEMANTIC.RELAXED: 3>
    RELEASE: typing.ClassVar[MEM_SEMANTIC]  # value = <MEM_SEMANTIC.RELEASE: 2>
    __members__: typing.ClassVar[dict[str, MEM_SEMANTIC]]  # value = {'ACQUIRE_RELEASE': <MEM_SEMANTIC.ACQUIRE_RELEASE: 0>, 'ACQUIRE': <MEM_SEMANTIC.ACQUIRE: 1>, 'RELEASE': <MEM_SEMANTIC.RELEASE: 2>, 'RELAXED': <MEM_SEMANTIC.RELAXED: 3>}
    @typing.overload
    def __eq__(self, other: MEM_SEMANTIC) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: MEM_SEMANTIC) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class RMW_OP:
    """
    RMW operation enumeration class: specifies the type of atomic RMW operation
    
    Members:
    
      AND : AND operation
    
      OR : OR operation
    
      XOR : XOR operation
    
      ADD : ADD operation
    
      FADD : FADD operation
    
      MAX : MAX operation
    
      MIN : MIN operation
    
      UMAX : UMAX operation
    
      UMIN : UMIN operation
    
      XCHG : XCHG operation
    """
    ADD: typing.ClassVar[RMW_OP]  # value = <RMW_OP.ADD: 0>
    AND: typing.ClassVar[RMW_OP]  # value = <RMW_OP.AND: 2>
    FADD: typing.ClassVar[RMW_OP]  # value = <RMW_OP.FADD: 1>
    MAX: typing.ClassVar[RMW_OP]  # value = <RMW_OP.MAX: 6>
    MIN: typing.ClassVar[RMW_OP]  # value = <RMW_OP.MIN: 7>
    OR: typing.ClassVar[RMW_OP]  # value = <RMW_OP.OR: 3>
    UMAX: typing.ClassVar[RMW_OP]  # value = <RMW_OP.UMAX: 9>
    UMIN: typing.ClassVar[RMW_OP]  # value = <RMW_OP.UMIN: 8>
    XCHG: typing.ClassVar[RMW_OP]  # value = <RMW_OP.XCHG: 5>
    XOR: typing.ClassVar[RMW_OP]  # value = <RMW_OP.XOR: 4>
    __members__: typing.ClassVar[dict[str, RMW_OP]]  # value = {'AND': <RMW_OP.AND: 2>, 'OR': <RMW_OP.OR: 3>, 'XOR': <RMW_OP.XOR: 4>, 'ADD': <RMW_OP.ADD: 0>, 'FADD': <RMW_OP.FADD: 1>, 'MAX': <RMW_OP.MAX: 6>, 'MIN': <RMW_OP.MIN: 7>, 'UMAX': <RMW_OP.UMAX: 9>, 'UMIN': <RMW_OP.UMIN: 8>, 'XCHG': <RMW_OP.XCHG: 5>}
    @typing.overload
    def __eq__(self, other: RMW_OP) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: RMW_OP) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def atomic_cas(ptr: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], cmp: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]], val: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]], mask: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], order: MEM_SEMANTIC) -> numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]:
    """
        Perform compare-and-swap operation on memory addresses based on mask.
    
        Args:
            ptr (numpy.ndarray): Memory addresses to perform compare-and-swap operation.
            cmp (numpy.ndarray): Values to compare.
            val (numpy.ndarray): Values to swap.
            mask (numpy.ndarray): Mask to apply.
            order (MemSemantic): Memory order for atomic operation.
    """
def atomic_rmw(rmw_op: RMW_OP, ptr: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], val: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]], mask: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], order: MEM_SEMANTIC) -> numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]:
    """
        Perform RMW operation on memory addresses based on mask.
    
        Args:
            rmw_op (RMWOp): RMW operation to perform.
            ptr (numpy.ndarray): Memory addresses to perform RMW operation.
            val (numpy.ndarray): Values to perform RMW operation.
            mask (numpy.ndarray): Mask to apply.
            order (MemSemantic): Memory order for atomic operation.
    """
def convert_float(input: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], in_w: typing.SupportsInt | typing.SupportsIndex, in_m: typing.SupportsInt | typing.SupportsIndex, in_b: typing.SupportsInt | typing.SupportsIndex, out_w: typing.SupportsInt | typing.SupportsIndex, out_m: typing.SupportsInt | typing.SupportsIndex, out_b: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.uint64]:
    """
        Convert floating-point numbers between different representations.
    
        Args:
            input (numpy.ndarray): Input array of floating-point numbers.
            in_w (int): Width of the input floating-point number representation.
            in_m (int): Mantissa bits of the input floating-point number representation.
            in_b (int): Bias of the input floating-point number representation.
            out_w (int): Width of the output floating-point number representation.
            out_m (int): Mantissa bits of the output floating-point number representation.
            out_b (int): Bias of the output floating-point number representation.
    """
def load(ptr: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], mask: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], other: numpy.ndarray | None, dtype: typing.Any) -> numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]:
    """
        Load data from memory addresses or fallback array based on mask.
    
        Args:
            ptr (numpy.ndarray): Memory addresses to load from.
            mask (numpy.ndarray): Mask to apply.
            other (numpy.ndarray): Fallback array to load from.
            dtype (numpy.dtype): Data type of the output array.
    """
def parallel_launch(fn: collections.abc.Callable, grid_dim: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], builder: typing.Any) -> None:
    """
        Launch kernel in parallel.
    
        Args:
            fn (function): The kernel function to launch.
            grid_dim (list): The grid dimension.
            builder (InterpreterBuilder): The interpreter builder.
    """
def store(ptr: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], value: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]], mask: typing.Annotated[numpy.typing.ArrayLike, numpy.bool]) -> None:
    """
        Store data to memory addresses based on mask.
    
        Args:
            ptr (numpy.ndarray): Memory addresses to store to.
            value (numpy.ndarray): Values to store.
            mask (numpy.ndarray): Mask to apply.
    """
