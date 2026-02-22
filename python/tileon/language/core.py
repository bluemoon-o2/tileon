from __future__ import annotations

import math
import builtins
import inspect
import warnings
from enum import Enum
from contextlib import contextmanager
from functools import partial, wraps, cached_property
import typing
from typing import Union, Callable, List, Sequence, TypeVar, Optional, Tuple, Any
from dataclasses import dataclass

from .. import knobs
from .._utils import (
    TILEON_MAX_TENSOR_NUMEL,
    deprecated,
    tuple_create,
    validate_block_shape,
    get_primitive_bitwidth,
)
from ..runtime.jit import JITCallable
from .._C import ir as _ir


class ir:
    """
    Base class of types that exist in the tileon IR.
    """
    builder: Any
    value: Any
    PROPAGATE_NAN = _ir.PROPAGATE_NAN


PropagateNan = _ir.PROPAGATE_NAN

# -----------------------------------------------------------------------------
# Type Definition
# -----------------------------------------------------------------------------


class const:
    """
    A type annotation class to mark pointers to constant data.

    The `store` function rejects const pointers. Constness is part
    of the pointer type (follow Tileon's type consistency rules:
    e.g., a function can't return both const and non-const pointers).
    """
    ...


class _type:
    """
    Base class of types that exist in the tileon IR.
    """

    def __eq__(self, other) -> bool:
        raise NotImplementedError("Types must implement __eq__")

    def __ne__(self, other) -> bool:
        return not (self == other)

    def mangle(self) -> str:
        """
        Return a string that mangles the type into a unique identifier.
        """
        raise NotImplementedError(
            f"NYI: Type mangling for type {self.__class__}")

    def _unflatten_ir(self, handles: list, cursor: int) -> Tuple[_value, int]:
        """
        Build a frontend value with the current dtype.

        Args:
            handles: The list of mlir handles to wrap.
            cursor: The index of the first handle relevant to this value.

        Returns:
            The frontend value and the updated cursor position.
        """
        raise NotImplementedError

    def _flatten_ir_types(self, builder, out: list) -> None:
        """
        Flatten the type into a sequence of mlir types,
        which are appended to the output list.

        Args:
            builder: The mlir builder.
            out: The list of mlir types to append to.
        """
        raise NotImplementedError


class _value:
    """
    Base class of values that exist in the tileon IR (i.e. not constexprs).
    """
    type: _type

    def _set_name(self, builder, name: str) -> None:
        """
        Set the name of the value in the IR.

        Args:
            builder: The mlir builder.
            name: The name to set.
        """
        raise NotImplementedError

    def _flatten_ir(self, handles: list) -> None:
        """
        Flatten frontend value into a sequence of mlir handles,
        which are appended to the output list

        Args:
            handles: The list of mlir handles to append to.
        """
        raise NotImplementedError


class constexpr_t(_type):
    """Type of constexpr values."""

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, constexpr_t) and self.value == other.value

    def __repr__(self) -> str:
        return f"constexpr_t({self.value})"

    def __hash__(self):
        return hash(self.value)

    def mangle(self) -> str:
        if hasattr(self.value, "mangle"):
            val = self.value.mangle()
        else:
            val = repr(self.value)
        return f"c{val}"

    def _flatten_ir_types(self, builder, out: list) -> None:
        return

    def _unflatten_ir(self, handles: list, cursor: int) -> Tuple[_value, int]:
        return constexpr(self.value), cursor


def _unwrap_if_constexpr(x):
    """
    Unwraps the constexpr value if it is a constexpr.

    Args:
        x: The input to unwrap.

    Returns:
        The unwrapped value if x is a constexpr.
        Otherwise, returns x.
    """
    if isinstance(x, list):
        return [_unwrap_if_constexpr(i) for i in x]
    if isinstance(x, builtins.tuple):
        return tuple_create([_unwrap_if_constexpr(i) for i in x], dtype=x)
    if isinstance(x, tuple):
        return tuple([_unwrap_if_constexpr(i) for i in x], dtype=x.type)
    return x.value if isinstance(x, constexpr) else x


def _unwrap_shape(shape: Sequence[Union[int, constexpr]]) -> List[constexpr]:
    """
    Unwraps the shape to a list of constexpr values.

    Args:
        shape: The input shape to unwrap.

    Returns:
        A list of constexpr values.
    """
    shape = _unwrap_if_constexpr(shape)
    return [_unwrap_if_constexpr(s) for s in shape]


def _normalize_tuple(x):
    """
    Normalizes the tuple x to a constexpr tuple.

    Args:
        x: The input tuple to normalize.

    Returns:
        A constexpr tuple.
    """
    normalized_tuple = _unwrap_if_constexpr(x)
    if isinstance(normalized_tuple, (list, builtins.tuple)):
        normalized_tuple = tuple(normalized_tuple)
    return normalized_tuple


class constexpr(_value):
    """This class is used to store a value that is known at compile-time."""

    def __init__(self, value):
        while isinstance(value, constexpr):
            value = value.value
        self.value = value
        self.type = constexpr_t(value)

    def __repr__(self) -> str:
        return f"constexpr({self.value})"

    def __hash__(self):
        return hash((self.value, self.type))

    def _set_name(self, builder: ir.builder, name: str) -> None:
        return

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        return

    def __index__(self):
        return self.value

    # We call `_unwrap_if_constexpr` to support interpreter mode.
    def __add__(self, other):
        return constexpr(self.value + _unwrap_if_constexpr(other))

    def __radd__(self, other):
        return constexpr(_unwrap_if_constexpr(other) + self.value)

    def __sub__(self, other):
        return constexpr(self.value - _unwrap_if_constexpr(other))

    def __rsub__(self, other):
        return constexpr(_unwrap_if_constexpr(other) - self.value)

    def __mul__(self, other):
        return constexpr(self.value * _unwrap_if_constexpr(other))

    def __mod__(self, other):
        return constexpr(self.value % _unwrap_if_constexpr(other))

    def __rmul__(self, other):
        return constexpr(_unwrap_if_constexpr(other) * self.value)

    def __truediv__(self, other):
        return constexpr(self.value / _unwrap_if_constexpr(other))

    def __rtruediv__(self, other):
        return constexpr(_unwrap_if_constexpr(other) / self.value)

    def __floordiv__(self, other):
        return constexpr(self.value // _unwrap_if_constexpr(other))

    def __rfloordiv__(self, other):
        return constexpr(_unwrap_if_constexpr(other) // self.value)

    def __gt__(self, other):
        return constexpr(self.value > _unwrap_if_constexpr(other))

    def __rgt__(self, other):
        return constexpr(_unwrap_if_constexpr(other) > self.value)

    def __ge__(self, other):
        return constexpr(self.value >= _unwrap_if_constexpr(other))

    def __rge__(self, other):
        return constexpr(_unwrap_if_constexpr(other) >= self.value)

    def __lt__(self, other):
        return constexpr(self.value < _unwrap_if_constexpr(other))

    def __rlt__(self, other):
        return constexpr(_unwrap_if_constexpr(other) < self.value)

    def __le__(self, other):
        return constexpr(self.value <= _unwrap_if_constexpr(other))

    def __rle__(self, other):
        return constexpr(_unwrap_if_constexpr(other) <= self.value)

    def __eq__(self, other):
        return constexpr(self.value == _unwrap_if_constexpr(other))

    def __ne__(self, other):
        return constexpr(self.value != _unwrap_if_constexpr(other))

    def __neg__(self):
        return constexpr(-self.value)

    def __bool__(self):
        return bool(self.value)

    def __and__(self, other):
        return constexpr(self.value & _unwrap_if_constexpr(other))

    def logical_and(self, other):
        return constexpr(self.value and _unwrap_if_constexpr(other))

    def __or__(self, other):
        return constexpr(self.value | _unwrap_if_constexpr(other))

    def __xor__(self, other):
        return constexpr(self.value ^ _unwrap_if_constexpr(other))

    def logical_or(self, other):
        return constexpr(self.value or _unwrap_if_constexpr(other))

    def __pos__(self):
        return constexpr(+self.value)

    def __invert__(self):
        return constexpr(~self.value)

    def __pow__(self, other):
        return constexpr(self.value**_unwrap_if_constexpr(other))

    def __rpow__(self, other):
        return constexpr(_unwrap_if_constexpr(other)**self.value)

    def __rshift__(self, other):
        return constexpr(self.value >> _unwrap_if_constexpr(other))

    def __lshift__(self, other):
        return constexpr(self.value << _unwrap_if_constexpr(other))

    def __not__(self):
        return constexpr(not self.value)

    def __iter__(self):
        return iter(self.value)

    def __call__(self, *args, **kwds):
        return self.value(*args, **kwds)

    def __getitem__(self, *args):
        args = (_unwrap_if_constexpr(x) for x in _normalize_tuple(args))
        return self.value.__getitem__(*args)


CONSTEXPR_0 = constexpr(0)
CONSTEXPR_1 = constexpr(1)


class dtype(_type):
    """
    A class represents the data type.
    """

    SINT_TYPES = ['int8', 'int16', 'int32', 'int64']
    UINT_TYPES = ['int1', 'uint8', 'uint16', 'uint32', 'uint64']
    FP_TYPES = [
        'fp8e4b15', 'fp8e4nv', 'fp8e4b8', 'fp8e5', 'fp8e5b16', 'fp16', 'bf16',
        'fp32', 'fp64'
    ]
    STANDARD_FP_TYPES = ['fp16', 'bf16', 'fp32', 'fp64']
    OTHER_TYPES = ['void']

    class SIGNEDNESS(Enum):
        SIGNED = 0
        UNSIGNED = 1

    class KIND(Enum):
        BOOLEAN = 0
        INTEGRAL = 1
        FLOATING = 2

    def __init__(self, name):
        """
        Create a dtype object.

        Args:
            name: The name of the dtype.
        """
        name = _unwrap_if_constexpr(name)
        self.name = name
        self.primitive_bitwidth = get_primitive_bitwidth(name)
        self.itemsize = self.primitive_bitwidth // 8

        if name in dtype.SINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.SIGNED
            self.int_bitwidth = self.primitive_bitwidth
        elif name in dtype.UINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.UNSIGNED
            self.int_bitwidth = self.primitive_bitwidth
        elif name in dtype.FP_TYPES:
            if name == 'fp8e4b15':
                self.fp_mantissa_width = 3
                self.exponent_bias = 15
            elif name == 'fp8e4nv':
                self.fp_mantissa_width = 3
                self.exponent_bias = 7
            elif name == 'fp8e4b8':
                self.fp_mantissa_width = 3
                self.exponent_bias = 8
            elif name == 'fp8e5':
                self.fp_mantissa_width = 2
                self.exponent_bias = 15
            elif name == 'fp8e5b16':
                self.fp_mantissa_width = 2
                self.exponent_bias = 16
            elif name == 'fp16':
                self.fp_mantissa_width = 10
                self.exponent_bias = 15
            elif name == 'bf16':
                self.fp_mantissa_width = 7
                self.exponent_bias = 127
            elif name == 'fp32':
                self.fp_mantissa_width = 23
                self.exponent_bias = 127
            elif name == 'fp64':
                self.fp_mantissa_width = 52
                self.exponent_bias = 1023
            else:
                raise RuntimeError(f'Unsupported floating-point type {name}')

    def __str__(self):
        return self.name

    def is_fp8(self):
        return 'fp8' in self.name

    def is_fp8e4nv(self):
        return self.name == 'fp8e4nv'

    def is_fp8e4b8(self):
        return self.name == 'fp8e4b8'

    def is_fp8e4b15(self):
        return self.name == 'fp8e4b15'

    def is_fp8e5(self):
        return self.name == 'fp8e5'

    def is_fp8e5b16(self):
        return self.name == 'fp8e5b16'

    def is_fp16(self):
        return self.name == 'fp16'

    def is_bf16(self):
        return self.name == 'bf16'

    def is_fp32(self):
        return self.name == 'fp32'

    def is_fp64(self):
        return self.name == 'fp64'

    def is_int1(self):
        return self.name == 'int1'

    def is_int8(self):
        return self.name == 'int8'

    def is_int16(self):
        return self.name == 'int16'

    def is_int32(self):
        return self.name == 'int32'

    def is_int64(self):
        return self.name == 'int64'

    def is_uint8(self):
        return self.name == 'uint8'

    def is_uint16(self):
        return self.name == 'uint16'

    def is_uint32(self):
        return self.name == 'uint32'

    def is_uint64(self):
        return self.name == 'uint64'

    def is_floating(self):
        """
        Check if the dtype is floating-point.

        Returns:
            True if the dtype is floating-point, False otherwise.
        """
        return self.name in dtype.FP_TYPES

    def is_standard_floating(self):
        """
        Check if the dtype is standard floating-point.

        Returns:
            True if the dtype is standard floating-point, False otherwise.
        """
        return self.name in dtype.STANDARD_FP_TYPES

    def is_int_signed(self):
        """
        Check if the dtype is signed integer.

        Returns:
            True if the dtype is signed integer, False otherwise.
        """
        return self.name in dtype.SINT_TYPES

    def is_int_unsigned(self):
        """
        Check if the dtype is unsigned integer.

        Returns:
            True if the dtype is unsigned integer, False otherwise.
        """
        return self.name in dtype.UINT_TYPES

    def is_int(self):
        """
        Check if the dtype is integer.

        Returns:
            True if the dtype is integer, False otherwise.
        """
        return self.name in dtype.SINT_TYPES + dtype.UINT_TYPES

    def is_bool(self):
        """
        Check if the dtype is boolean.

        Returns:
            True if the dtype is boolean, False otherwise.
        """
        return self.is_int1()

    def kind(self):
        """
        Get the kind of the dtype.

        The kind of the dtype is determined by the following rules:
        - Boolean: int1
        - Integer: int8, int16, int32, int64, uint8, uint16, uint32, uint64
        - Floating-point: fp8e4nv, fp8e4b8, fp8e4b15, fp8e5, fp8e5b16, fp16, bf16, fp32, fp64

        Returns:
            The kind of the dtype.
        """
        if self.is_bool():
            return dtype.KIND.BOOLEAN
        elif self.is_int():
            return dtype.KIND.INTEGRAL
        else:
            assert self.is_floating()
            return dtype.KIND.FLOATING

    def get_int_max_value(self):
        """
        Get the maximum value of the dtype.

        Returns:
            The maximum value of the dtype.
        """
        if self.is_int_signed():
            return 2**(self.int_bitwidth - 1) - 1
        if self.is_int_unsigned():
            return 2**self.int_bitwidth - 1
        assert False

    def get_int_min_value(self):
        """
        Get the minimum value of the dtype.

        Returns:
            The minimum value of the dtype.
        """
        if self.is_int_signed():
            return -2**(self.int_bitwidth - 1)
        if self.is_int_unsigned():
            return 0
        assert False

    @staticmethod
    def is_void():
        raise RuntimeError("Not implemented")

    @staticmethod
    def is_block():
        return False

    @staticmethod
    def is_ptr():
        return False

    @staticmethod
    def is_const():
        return False

    def __eq__(self, other) -> bool:
        other = _unwrap_if_constexpr(other)
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash((self.name, ))

    @property
    def scalar(self):
        """
        Get the scalar dtype.

        Returns:
            The scalar dtype.
        """
        return self

    def _flatten_ir_types(self, builder, out: list) -> None:
        """
        Flatten the dtype to a list of IR types.

        Args:
            builder: The IR builder.
            out: The list to store the flattened IR types.
        """
        out.append(self.to_ir(builder))

    def to_ir(self, builder):
        """
        Convert the dtype to an IR type.

        Args:
            builder: The IR builder.

        Returns:
            The IR type.
        """
        if self.name.startswith("fp8"):
            if (hasattr(builder, "options")
                    and self.name not in builder.options.supported_fp8):
                raise ValueError(
                    f'type {self} not supported in this architecture. '
                    f'The supported fp8 dtypes are {builder.options.supported_fp8}'
                )

        if self.name == 'void':
            return builder.get_void_t()
        elif self.name == 'int1':
            return builder.get_int1_t()
        elif self.name in ('int8', 'uint8'):
            return builder.get_int8_t()
        elif self.name in ('int16', 'uint16'):
            return builder.get_int16_t()
        elif self.name in ('int32', 'uint32'):
            return builder.get_int32_t()
        elif self.name in ('int64', 'uint64'):
            return builder.get_int64_t()
        elif self.name == 'fp8e5':
            return builder.get_fp8e5_t()
        elif self.name == 'fp8e5b16':
            return builder.get_fp8e5b16_t()
        elif self.name == 'fp8e4nv':
            return builder.get_fp8e4nv_t()
        elif self.name == 'fp8e4b8':
            return builder.get_fp8e4b8_t()
        elif self.name == 'fp8e4b15':
            return builder.get_fp8e4b15_t()
        elif self.name == 'fp16':
            return builder.get_half_t()
        elif self.name == 'bf16':
            return builder.get_bf16_t()
        elif self.name == 'fp32':
            return builder.get_float_t()
        elif self.name == 'fp64':
            return builder.get_double_t()
        raise ValueError(f'fail to convert {self} to ir type')

    def codegen_name(self):
        """
        Get the codegen name of the dtype.

        Returns:
            The codegen name of the dtype.
        """
        if self.name.startswith("fp"):
            return "float" + self.name[2:]
        elif self.name.startswith("bf"):
            return "bfloat" + self.name[2:]
        else:
            return self.name

    @property
    def cache_key_part(self) -> str:
        """See cache_key_part() in tileon.cc."""
        return self.name

    def __repr__(self):
        """Output of repr needs to be an evaluatable expression"""
        return f'tileon.language.{self.codegen_name()}'

    def _unflatten_ir(self, handles: list, cursor: int) -> Tuple[_value, int]:
        """
        Unflatten the dtype from a list of IR values.

        Args:
            handles: The list of IR values.
            cursor: The cursor to the current position in the list.

        Returns:
            A tuple of the unflattened value and the new cursor position.
        """
        return tensor(handles[cursor], self), cursor + 1

    def mangle(self) -> str:
        """
        Get the mangled name of the dtype.

        Returns:
            The mangled name of the dtype.
        """
        if self.is_int():
            SIGNED = dtype.SIGNEDNESS.SIGNED
            prefix = 'i' if self.int_signedness == SIGNED else 'u'
            return prefix + str(self.int_bitwidth)
        if self.is_floating():
            return str(self)
        if self.is_void():
            return 'V'
        return super().mangle()

    def with_element_t(self, element_t: dtype):
        """
        Create a new dtype with the same name but with a different element type.

        Args:
            element_t: The new element type.

        Returns:
            A new dtype with the same name but with a different element type.
        """
        assert not self.is_block()
        return element_t


def is_dtype(type_str: str) -> bool:
    """
    Check if the type_str is a valid dtype.

    Args:
        type_str: The type string to check.

    Returns:
        True if the type_str is a valid dtype, False otherwise.
    """
    return type_str in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES


# Declare an alias for dtype class.
_DtypeClass = dtype


class pointer_t(dtype):
    """The dtype for pointer."""

    def __init__(self,
                 element_t: dtype,
                 address_space: int = 1,
                 const: bool = False):
        """
        Create a new pointer dtype.

        Args:
            element_t: The element type of the pointer.
            address_space: The address space of the pointer.
            const: Whether the pointer is const.
        """
        element_t = _unwrap_if_constexpr(element_t)
        assert isinstance(
            element_t, dtype
        ), f'element_t expected `dtype`, got `{type(element_t).__name__}`.'
        self.element_t = element_t
        self.address_space = address_space
        self.const = const
        self.name = f'pointer<{element_t}>' if not const else f'const_pointer<{element_t}>'

    def to_ir(self, builder):
        return builder.get_ptr_t(self.element_t.to_ir(builder),
                                 self.address_space)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def is_ptr(self):
        return True

    def is_const(self):
        return self.const

    def __eq__(self, other) -> bool:
        other = _unwrap_if_constexpr(other)
        if not isinstance(other, pointer_t):
            return False
        return (self.element_t == other.element_t
                and self.address_space == other.address_space
                and self.const == other.const)

    @property
    def scalar(self):
        return self

    def mangle(self) -> str:
        return f"P{self.element_t.mangle()}"


class block_t(dtype):
    """The dtype for block."""

    def __init__(self, element_t: dtype, shape: List):
        """
        Create a new block dtype.

        Args:
            element_t: The element type of the block.
            shape: The shape of the block.
        """
        self.element_t = element_t

        assert (isinstance(
            shape, (list, tuple)
        ), f'shape has type `{type(shape).__name__}`; expected `list` or `tuple`.'
                )

        # shape can be empty ([]) when an input is a 0D tensor.
        self.shape = tuple(_unwrap_shape(shape))
        assert self.shape, '0d block_type is forbidden'

        self.numel = validate_block_shape(self.shape)
        self.name = f'<{self.shape}, {self.element_t}>'

    def to_ir(self, builder):
        """
        Get the IR type of the block dtype.

        Args:
            builder: The IR builder.

        Returns:
            The IR type of the block dtype.
        """
        return builder.get_block_t(self.element_t.to_ir(builder), self.shape)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def is_block(self):
        return True

    def get_block_shape(self) -> Tuple[int]:
        return self.shape

    def with_element_t(self, scalar_t: dtype) -> block_t:
        """
        Create a new block dtype with the same shape but with a different element type.

        Args:
            scalar_t: The new element type.

        Returns:
            A new block dtype with the same shape but with a different element type.
        """
        return block_t(scalar_t, self.shape)

    def __eq__(self, other) -> bool:
        """
        Check if the block dtype is equal to the other block dtype.

        Args:
            other: The other block dtype.

        Returns:
            True if the block dtype is equal to the other block dtype, False otherwise.
        """
        other = _unwrap_if_constexpr(other)
        if not isinstance(other, block_t):
            return False
        return self.element_t == other.element_t and self.shape == other.shape

    @property
    def scalar(self):
        """
        Get the element type of the block.

        Returns:
            The element type of the block.
        """
        return self.element_t

    @property
    def nbytes(self):
        """
        Get the number of bytes of the block.

        Returns:
            The number of bytes of the block.
        """
        return self.numel * (self.element_t.primitive_bitwidth // 8)

    def mangle(self) -> str:
        """
        Mangle the block dtype.

        Returns:
            The mangled string.
        """
        elt = self.scalar.mangle()
        shape = '_'.join(map(str, self.shape))
        return f'{elt}S{shape}S'


class tuple_t(_type):
    """The dtype for tuple."""

    def __init__(self,
                 types: Sequence[dtype],
                 fields: Optional[Sequence[str]] = None):
        self.types = types
        self.fields = fields

    @cached_property
    def name(self):
        """
        Get the name of the tuple dtype.

        Returns:
            The name of the tuple dtype.
        """
        if self.fields is None:
            return '[' + ','.join(str(v) for v in self.types) + ']'
        return '[' + ','.join(
            [f"{k}:{v}" for k, v in zip(self.fields, self.types)]) + ']'

    def __str__(self):
        return self.name

    def __iter__(self):
        return iter(self.types)

    def _flatten_ir_types(self, builder, out: List):
        """
        Flatten the IR types of the tuple dtype.

        Args:
            builder: The IR builder.
            out: The list to store the flattened IR types.
        """
        for ty in self.types:
            ty._flatten_ir_types(builder, out)

    def __getitem__(self, index: int) -> dtype:
        return self.types[index]

    def __eq__(self, other):
        return (type(self) is type(other) and self.types == other.types
                and self.fields == other.fields)

    def _unflatten_ir(self, handles: List, cursor: int) -> Tuple[tuple, int]:
        """
        Unflatten the IR values of the tuple dtype.

        Args:
            handles: The list of IR values.
            cursor: The cursor to the current position in the list.

        Returns:
            A tuple of the unflattened IR values and the updated cursor.
        """
        values = []
        for t in self.types:
            value, cursor = t._unflatten_ir(handles, cursor)
            values.append(value)
        return tuple(values, type=self), cursor

    def mangle(self):
        """
        Mangle the tuple dtype.

        Returns:
            The mangled string.
        """
        return 'T' + '_'.join(ty.mangle() for ty in self.types) + 'T'


def _type_for_tuple(values: Sequence[Any], fields=None) -> tuple_t:
    """
    Get the dtype for a tuple of values.

    Args:
        values: The values of the tuple.
        fields: The fields of the tuple.

    Returns:
        The dtype for the tuple.
    """
    return tuple_t([
        constexpr_t(x) if isinstance(x, (int, float, dtype)) else x.type
        for x in values
    ], fields)


class tuple(_value):
    """A tuple class in Tileon."""

    def __init__(self, x: Sequence, dtype: Optional[tuple_t] = None):
        self.values = [i for i in x]
        if isinstance(dtype, tuple_t):
            self.type = dtype
        elif dtype is not None:
            self.type = tuple_t(dtype)
        else:
            self.type = _type_for_tuple(self.values)

    def __getitem__(self, idx: constexpr):
        if isinstance(idx, int):
            idx = constexpr(idx)
        if isinstance(idx, constexpr):
            return self.values[idx]
        else:
            assert isinstance(idx, (slice, builtins.slice))
            return tuple(self.values[idx.start:idx.stop:idx.step])

    def __getattr__(self, name):
        fields = self.type.fields
        if fields is None or name not in fields:
            raise AttributeError(f"'tuple' object has no attribute {name}")
        return self.values[fields.index(name)]

    # TODO: remove
    def _setitem(self, idx, value):
        idx = _unwrap_if_constexpr(idx)
        assert isinstance(idx, int)
        self.values[idx] = value
        self.type = _type_for_tuple(self.values, self.type.fields)

    def __add__(self, other):
        other = _normalize_tuple(other)
        return tuple(self.values + other.values)

    def __mul__(self, other):
        assert isinstance(other, constexpr)
        return tuple(self.values * other.value)

    def __eq__(self, other):
        other = _normalize_tuple(other)
        return constexpr(self.values == other.values)

    def __hash__(self):
        return hash(builtins.tuple(self.values))

    def __str__(self):
        return str([str(x) for x in self.values])

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def _set_name(self, builder, name: str) -> None:
        """
        Set the name of the tuple dtype.

        Args:
            builder: The IR builder.
            name: The name to set.
        """
        fields = self.type.fields
        if fields is not None:
            for field, v in zip(fields, self.values):
                v._set_name(builder, f"{name}.{field}")
        else:
            for i, v in enumerate(self.values):
                v._set_name(builder, f"{name}.{i}")

    def _flatten_ir(self, handles: List):
        """
        Flatten the IR values of the tuple dtype.

        Args:
            handles: The list of IR values.
        """
        for v in self.values:
            v._flatten_ir(handles)

    def __repr__(self):
        return f"({', '.join(repr(x) for x in self.values)})"


class slice_t(dtype):
    """The dtype for slice."""

    def __init__(self):
        self.name = 'slice_t'


class slice:
    """A slice class in Tileon."""

    def __init__(self,
                 start: Optional[constexpr] = None,
                 stop: Optional[constexpr] = None,
                 step: Optional[constexpr] = None):
        self.start = start
        self.stop = stop
        self.step = step
        self.type = slice_t()


# scalar types
void = dtype('void')
int1 = dtype('int1')
int8 = dtype('int8')
int16 = dtype('int16')
int32 = dtype('int32')
int64 = dtype('int64')
uint8 = dtype('uint8')
uint16 = dtype('uint16')
uint32 = dtype('uint32')
uint64 = dtype('uint64')
float8e5 = dtype('fp8e5')
float8e5b16 = dtype('fp8e5b16')
float8e4nv = dtype('fp8e4nv')
float8e4b8 = dtype('fp8e4b8')
float8e4b15 = dtype('fp8e4b15')
float16 = dtype('fp16')
bfloat16 = dtype('bf16')
float32 = dtype('fp32')
float64 = dtype('fp64')
# pointer types
pi32_t = pointer_t(int32)


def get_int_dtype(bitwidth: int, signed: bool) -> dtype:
    """
    Get the dtype for an integer type.

    Args:
        bit_width: The bitwidth of the integer.
        signed: Whether the integer is signed.

    Returns:
        The dtype for the integer.
    """
    if bitwidth == 1:
        return int1
    elif bitwidth == 8 and signed:
        return int8
    elif bitwidth == 8 and not signed:
        return uint8
    elif bitwidth == 16 and signed:
        return int16
    elif bitwidth == 16 and not signed:
        return uint16
    elif bitwidth == 32 and signed:
        return int32
    elif bitwidth == 32 and not signed:
        return uint32
    elif bitwidth == 64 and signed:
        return int64
    elif bitwidth == 64 and not signed:
        return uint64
    else:
        raise ValueError(
            f'Unsupported bitwidth {bitwidth} and signedness {signed}')


# -----------------------------------------------------------------------------
# Tensor & Builtin Functions
# -----------------------------------------------------------------------------

TILEON_BUILTIN = "__tileon_builtin__"


def is_builtin(fn: Callable) -> bool:
    """Returns True if the function is a builtin function.

    Args:
        fn: The function to check.

    Returns:
        True if the function is a builtin function.
    """
    return getattr(fn, TILEON_BUILTIN, False)


T = TypeVar("T", bound=Callable)


def builtin(fn: T) -> T:
    """
    A decorator that marks a function as a builtin.

    Args:
        fn: The function to decorate.

    Returns:
        The decorated function.
    """
    assert callable(fn), "builtin decorator must be applied to a function"

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "_semantic" not in kwargs or kwargs["_semantic"] is None:
            raise ValueError(
                "Did you forget to add @tileon.jit ? "
                "(`_semantic` argument must be provided outside of JIT functions.)"
            )
        return fn(*args, **kwargs)

    setattr(wrapper, TILEON_BUILTIN, True)
    wrapper.signature = inspect.signature(fn)

    return wrapper


class tensor(_value):
    """
    Represents an N-dimensional array of values or pointers.

    `tensor` is the fundamental data structure in Tileon programs. Most
    functions in `tileon.language` operate on and return tensors.

    Most of the named member functions here are duplicates of the free functions
    in `tileon.language`. For example, `tileon.language.sqrt(x)` is
    equivalent to `x.sqrt()`.

    `tensor` also defines most of the magic/dunder methods, so you can
    write `x+y`, `x << 2`, etc.
    """

    def __init__(self, handle, type: dtype):
        """
        Creates a tensor.

        Args:
            handle: The IR value handle.
            type: The dtype of the tensor.
        """
        super().__init__()
        # IR handle
        self.handle = handle
        # Block shape
        self.shape = type.shape if type.is_block() else ()
        self.numel = constexpr(math.prod(self.shape))
        self.type = type  # Tensor type (can be block_type)
        self.dtype = type.scalar  # dtype is scalar type
        self.shape = tuple([constexpr(s) for s in self.shape])

    def _set_name(self, builder, name: str) -> None:
        """
        Sets the name of the tensor.

        Args:
            builder: The IR builder.
            name: The name to set.
        """
        self.handle.set_loc(
            builder.create_name_loc(name, self.handle.get_loc()))

    def _flatten_ir(self, handles: List) -> None:
        """
        Flattens the IR handle of the tensor.

        Args:
            handles: The list of IR handles to append to.
        """
        handles.append(self.handle)

    def __str__(self) -> str:
        return str(self.dtype) + '(' + ', '.join(str(s)
                                                 for s in self.shape) + ')'

    @builtin
    def __add__(self, other, _semantic=None):
        return add(self, other, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __radd__(self, other, _semantic=None):
        return add(other, self, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __sub__(self, other, _semantic=None):
        return sub(self, other, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __rsub__(self, other, _semantic=None):
        return sub(other, self, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __mul__(self, other, _semantic=None):
        return mul(self, other, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __rmul__(self, other, _semantic=None):
        return mul(other, self, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __truediv__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.truediv(self, other)

    @builtin
    def __rtruediv__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.truediv(other, self)

    @builtin
    def __floordiv__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.floordiv(self, other)

    @builtin
    def __rfloordiv__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.floordiv(other, self)

    @builtin
    def __mod__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.mod(self, other)

    @builtin
    def __rmod__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.mod(other, self)

    # unary operators
    @builtin
    def __neg__(self, _semantic=None):
        return _semantic.minus(self)

    @builtin
    def __invert__(self, _semantic=None):
        return _semantic.invert(self)

    # bitwise operators
    @builtin
    def __and__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.and_(self, other)

    @builtin
    def __rand__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.and_(other, self)

    @builtin
    def __or__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.or_(self, other)

    @builtin
    def __ror__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.or_(other, self)

    @builtin
    def __xor__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.xor_(self, other)

    @builtin
    def __rxor__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.xor_(other, self)

    @builtin
    def __lshift__(self, other, _semantic=None):
        check_bit_width(self, other)
        other = _unwrap_if_constexpr(other)
        return _semantic.shl(self, other)

    @builtin
    def __rlshift__(self, other, _semantic=None):
        check_bit_width(other, self)
        other = _unwrap_if_constexpr(other)
        return _semantic.shl(other, self)

    @builtin
    def __rshift__(self, other, _semantic=None):
        check_bit_width(self, other)
        other = _unwrap_if_constexpr(other)
        if self.dtype.is_int_signed():
            return _semantic.ashr(self, other)
        else:
            return _semantic.lshr(self, other)

    @builtin
    def __rrshift__(self, other, _semantic=None):
        check_bit_width(other, self)
        other = _unwrap_if_constexpr(other)
        if self.dtype.is_int_signed():
            return _semantic.ashr(other, self)
        else:
            return _semantic.lshr(other, self)

    # >
    @builtin
    def __gt__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.greater_than(self, other)

    @builtin
    def __rgt__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.greater_than(other, self)

    # >=
    @builtin
    def __ge__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.greater_equal(self, other)

    @builtin
    def __rge__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.greater_equal(other, self)

    # <
    @builtin
    def __lt__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.less_than(self, other)

    @builtin
    def __rlt__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.less_than(other, self)

    # <=
    @builtin
    def __le__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.less_equal(self, other)

    @builtin
    def __rle__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.less_equal(other, self)

    # ==
    @builtin
    def __eq__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.equal(self, other)

    @builtin
    def __req__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.equal(other, self)

    @builtin
    def __ne__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.not_equal(self, other)

    @builtin
    def __rne__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.not_equal(other, self)

    @builtin
    def logical_and(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.logical_and(self, other)

    @builtin
    def logical_or(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.logical_or(self, other)

    # __not__ isn't actually a magic method in python
    # but our ASTVisitor handles it
    @builtin
    def __not__(self, _semantic=None):
        return _semantic.not_(self)

    @builtin
    def __getitem__(self, slices, _semantic=None):
        """
        Get a slice of the tensor.

        Only supports dimension expansion (unsqueeze/expand_dims)

        Args:
            slices: The slice to get (None, (None, slice(None)), [None], etc.).

        Returns:
            The sliced tensor.
        """
        if slices is None or isinstance(slices,
                                        (builtins.slice, slice, constexpr)):
            slices = [slices]
        if isinstance(slices, tuple):
            slices = slices.values
        ret = self
        for dim, sl in enumerate(slices):
            if _unwrap_if_constexpr(sl) is None:
                ret = _semantic.expand_dims(ret, dim)
            elif (isinstance(sl, (builtins.slice, slice)) and all(
                    _unwrap_if_constexpr(x) is None
                    for x in (sl.start, sl.stop, sl.step))):
                pass
            else:
                raise ValueError(f"unsupported tensor index: {sl}")
        return ret

    @property
    def T(self):
        """Transposes a 2D tensor."""
        raise RuntimeError("Transposition must be created by the AST Visitor")

    @builtin
    def to(self,
           dtype: dtype,
           fp_downcast_rounding: Optional[str] = None,
           bitcast: bool = False,
           _semantic=None):
        """
        Alias for :py:func:`tensor.cast`.

        Args:
            dtype: The dtype to cast to.
            fp_downcast_rounding: The rounding mode to use for floating point downcasts.
            bitcast: Whether to perform a bitcast.

        Returns:
            The cast tensor.
        """
        return cast(self,
                    dtype,
                    fp_downcast_rounding,
                    bitcast,
                    _semantic=_semantic)

    # Type stubs for functions added by the _tensor_member_fn decorator.
    def broadcast_to(self, *shape) -> tensor:
        ...

    def trans(self, *dims) -> tensor:
        ...

    def permute(self, *dims) -> tensor:
        ...

    def split(self) -> tuple[tensor, tensor]:
        ...

    def view(self, *shape) -> tensor:
        ...

    def reshape(self, *shape) -> tensor:
        ...

    def expand_dims(self, axis) -> tensor:
        ...

    def cast(self, dtype, fp_downcast_rounding=None, bitcast=False) -> tensor:
        ...

    def store(self,
              value,
              mask=None,
              boundary_check=(),
              cache_modifier="",
              eviction_policy="") -> tensor:
        ...

    def advance(self, offsets) -> tensor:
        ...

    def atomic_cas(self, cmp, val, sem=None, scope=None) -> tensor:
        ...

    def atomic_xchg(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_add(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_max(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_min(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_and(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_or(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_xor(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def exp(self) -> tensor:
        ...

    def log(self) -> tensor:
        ...

    def cos(self) -> tensor:
        ...

    def sin(self) -> tensor:
        ...

    def sqrt(self) -> tensor:
        ...

    def rsqrt(self) -> tensor:
        ...

    def abs(self) -> tensor:
        ...

    def reduce(self, axis, combine_fn, keep_dims=False) -> tensor:
        ...

    def associative_scan(self, axis, combine_fn, reverse=False) -> tensor:
        ...

    def gather(self, indices, axis) -> tensor:
        ...

    def histogram(self, num_bins) -> tensor:
        ...

    def cdiv(self, div) -> tensor:
        ...

    def sigmoid(self) -> tensor:
        ...

    def softmax(self,
                dim=None,
                keep_dims=False,
                ieee_rounding=False) -> tensor:
        ...

    def ravel(self) -> tensor:
        ...

    def max(self,
            axis=None,
            return_indices=False,
            return_indices_tie_break_left=True,
            keep_dims=False) -> tensor:
        ...

    def argmax(self, axis, tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def min(self,
            axis=None,
            return_indices=False,
            return_indices_tie_break_left=True,
            keep_dims=False) -> tensor:
        ...

    def argmin(self, axis, tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def sum(self, axis=None, keep_dims=False, dtype=None) -> tensor:
        ...

    def xor_sum(self, axis=None, keep_dims=False) -> tensor:
        ...

    def reduce_or(self, axis=None, keep_dims=False) -> tensor:
        ...

    def cumsum(self, axis=0, reverse=False) -> tensor:
        ...

    def cumprod(self, axis=0, reverse=False) -> tensor:
        ...

    def sort(self,
             dim: constexpr = None,
             descending: constexpr = CONSTEXPR_0) -> tensor:
        ...

    def flip(self, dim=None) -> tensor:
        ...


@builtin
def to_tensor(x, _semantic=None):
    """
    Convert a value to a tensor.

    Args:
        x: The value to convert.
    """
    return _semantic.to_tensor(x)


def check_bit_width(value, shift_value):
    """
    Check if the shift value exceeds the bitwidth of the value.

    Args:
        value: The tensor value.
        shift_value: The shift value.
    """
    bitwidth = value.type.scalar.primitive_bitwidth
    if shift_value.value >= bitwidth:
        warnings.warn(
            f"Value {shift_value.value} exceeds the maximum bitwidth ({bitwidth}) "
            f"for type '{value.dtype}'. This may result in undefined behavior."
        )


class _tensor_descriptor_t(_type):
    """
    A tensor descriptor type.
    """

    def __init__(self, block_t: block_t):
        self.block_t = block_t

    def _unflatten_ir(self, handles: List,
                      cursor: int) -> Tuple[_tensor_descriptor, int]:
        """
        Unflatten the IR value of the tensor descriptor.

        Args:
            handles: The list of IR values.
            cursor: The cursor position.

        Returns:
            The tensor descriptor value and the updated cursor position.
        """
        value = _tensor_descriptor(handles[cursor], self.block_t)
        return value, cursor + 1

    def _flatten_ir_types(self, builder, out: List) -> None:
        """
        Flatten the IR types of the tensor descriptor.

        Args:
            builder: The IR builder.
            out: The list of IR types.
        """
        is_signed = self.block_t.element_t.is_int_signed()
        out.append(
            builder.create_tensor_descriptor_t(self.block_t.to_ir(builder),
                                               is_signed))

    def __str__(self) -> str:
        return f"tensor_descriptor<{self.block_t}>"

    def __eq__(self, other) -> bool:
        if type(other) is not type(self):
            return False
        return self.block_t == other.block_t

    def __neq__(self, other) -> bool:
        return not (self == other)

    def mangle(self) -> str:
        return f"TD{self.block_t.mangle()}"


class _tensor_descriptor(_value):
    """"
    A tensor descriptor with unknown shape and strides
    """

    def __init__(self, handle, block_t: block_t):
        super().__init__()
        self.handle = handle
        self.type = _tensor_descriptor_t(block_t)

    def _set_name(self, builder, name: str) -> None:
        """
        Set the name of the tensor descriptor.

        Args:
            builder: The IR builder.
            name: The name of the tensor descriptor.
        """
        self.handle.set_loc(
            builder.create_name_loc(name, self.handle.get_loc()))

    def _flatten_ir(self, handles: List) -> None:
        """
        Flatten the IR value of the tensor descriptor.

        Args:
            handles: The list of IR values.
        """
        handles.append(self.handle)

    @property
    def block_t(self):
        return self.type.block_t

    @property
    def block_shape(self):
        return self.type.block_t.shape

    @property
    def dtype(self):
        return self.type.block_t.element_t

    def __str__(self) -> str:
        return str(self.type)

    @builtin
    def load(self,
             offsets: Sequence[constexpr | tensor],
             _semantic=None) -> tensor:
        """
        Load a block from the descriptor starting at the given element offsets.

        Values outside of the tensor bounds will be filled with zeros.

        :note: Offset must be a multiple of 16-bytes
        """
        return _semantic.descriptor_load(self, offsets, "", "")

    @builtin
    def store(self,
              offsets: Sequence[constexpr | tensor],
              value: tensor,
              _semantic=None) -> tensor:
        """
        Store a block from the descriptor starting at the given element offsets.

        Values outside of the tensor bounds will be ignored.

        :note: Offset must be a multiple of 16-bytes
        """
        return _semantic.descriptor_store(self, value, offsets)

    @builtin
    def atomic_add(self,
                   offsets: Sequence[constexpr | tensor],
                   value: tensor,
                   _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_add(self, value, offsets)

    @builtin
    def atomic_min(self,
                   offsets: Sequence[constexpr | tensor],
                   value: tensor,
                   _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_min(self, value, offsets)

    @builtin
    def atomic_max(self,
                   offsets: Sequence[constexpr | tensor],
                   value: tensor,
                   _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_max(self, value, offsets)

    @builtin
    def atomic_and(self,
                   offsets: Sequence[constexpr | tensor],
                   value: tensor,
                   _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_and(self, value, offsets)

    @builtin
    def atomic_or(self,
                  offsets: Sequence[constexpr | tensor],
                  value: tensor,
                  _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_or(self, value, offsets)

    @builtin
    def atomic_xor(self,
                   offsets: Sequence[constexpr | tensor],
                   value: tensor,
                   _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_xor(self, value, offsets)

    @builtin
    def gather(self, *args, _semantic=None) -> tensor:
        """Gather multiple descriptors worth of data"""
        assert len(
            args
        ) == 2, f"descriptor gather only supports 2D indexing, but got {len(args)}"
        x_offsets = args[0]
        y_offset = args[1]
        return _semantic.descriptor_gather(self, x_offsets, y_offset, "", "")

    @builtin
    def scatter(self, value, *args, _semantic=None) -> tensor:
        """Scatter multiple descriptors worth of data"""
        assert len(
            args
        ) == 2, f"descriptor scatter only supports 2D indexing, but got {len(args)}"
        x_offsets = args[0]
        y_offset = args[1]
        return _semantic.descriptor_scatter(self, value, x_offsets, y_offset)


class tensor_descriptor_t(_tensor_descriptor_t):
    """
    A tensor descriptor with shape and strides
    """

    def __init__(self, block_t: block_t, shape_t: tuple_t, strides_t: tuple_t):
        self.block_t = block_t
        self.shape_t = shape_t
        self.strides_t = strides_t

    def _unflatten_ir(self, handles: List,
                      cursor: int) -> Tuple[_tensor_descriptor, int]:
        """
        Unflatten the IR value of the tensor descriptor.

        Args:
            handles: The list of IR values.
            cursor: The cursor of the IR value.

        Returns:
            The tensor descriptor and the cursor.
        """
        handle = handles[cursor]
        cursor += 1
        shape, cursor = self.shape_t._unflatten_ir(handles, cursor)
        strides, cursor = self.strides_t._unflatten_ir(handles, cursor)
        shape = shape.values
        strides = strides.values
        value = tensor_descriptor(handle, shape, strides, self.block_t)
        return value, cursor

    def _flatten_ir_types(self, builder, out: List) -> None:
        """
        Flatten the IR types of the tensor descriptor.

        Args:
            builder: The IR builder.
            out: The list of IR types.
        """
        super()._flatten_ir_types(builder, out)
        self.shape_t._flatten_ir_types(builder, out)
        self.strides_t._flatten_ir_types(builder, out)

    def __eq__(self, other):
        return (super().__eq__(other) and self.shape_t == other.shape_t
                and self.strides_t == other.strides_t)


class tensor_descriptor(_tensor_descriptor):
    """
    A descriptor representing a tensor in global memory.
    """

    def __init__(self, handle, shape: List[tensor], strides: List[tensor],
                 block_t: block_t):
        """
        Create a tensor descriptor.

        Args:
            handle: The IR handle.
            shape: The global shape of the tensor.
            strides: The global strides of the tensor.
            block_t: The block type of the tensor.
        """
        # IR handle
        super().__init__(handle, block_t)
        # Global shape
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.type = tensor_descriptor_t(
            block_t,
            shape_t=self.shape.type,
            strides_t=self.strides.type,
        )

    def _set_name(self, builder, name: str) -> None:
        """
        Set the name of the tensor descriptor.

        Args:
            builder: The IR builder.
            name: The name of the tensor descriptor.
        """
        super()._set_name(builder, name)
        self.shape._set_name(builder, name + ".shape")
        self.strides._set_name(builder, name + ".stride")

    def _flatten_ir(self, handles: List) -> None:
        """
        Flatten the IR value of the tensor descriptor.

        Args:
            handles: The list of IR values.
        """
        super()._flatten_ir(handles)
        self.shape._flatten_ir(handles)
        self.strides._flatten_ir(handles)


# -----------------------------------------------------------------------------
# Aggregate Type
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _aggregate_t(_type):
    """A generic base type for all Tileon aggregate types.

    This class contains a reference to the original user-defined Python class
    and a list of class fields with their Tileon types.
    """

    base_cls: type
    fields: List[Tuple[str, _type]]

    def _unflatten_ir(self, handles: List[ir.value],
                      cursor: int) -> Tuple[ir.value, int]:
        """
        Unflatten the IR value of the aggregate type.

        Args:
            handles: The list of IR values.
            cursor: The cursor of the IR value.

        Returns:
            The aggregate value and the cursor.
        """
        instance = self.base_cls._get_instance()
        for name, ty in self.fields:
            value, cursor = ty._unflatten_ir(handles, cursor)
            setattr(instance, name, value)
        return instance, cursor

    def _flatten_ir_types(self, builder, out: List) -> None:
        """
        Flatten the IR types of the aggregate type.

        Args:
            builder: The IR builder.
            out: The list of IR types.
        """
        for name, ty in self.fields:
            ty._flatten_ir_types(builder, out)

    def mangle(self) -> str:
        name = f"{self.base_cls.__module__}.{self.base_cls.__qualname__}"
        fields = [ty.mangle() for (name, ty) in self.fields]
        return f"{name}<{', '.join(fields)}>"


def _wrap_init_args(x):
    """
    Wrap the init arguments of the aggregate type.

    Args:
        x: The init arguments.

    Returns:
        The wrapped init arguments.
    """
    if isinstance(x, tuple):
        from tileon.compiler.code_generator import _apply_to_tuple_values
        return _apply_to_tuple_values(x, _wrap_init_args)
    if isinstance(x, builtins.tuple):
        wrapped = builtins.tuple(_wrap_init_args(i) for i in x)
        fields = getattr(x, "_fields", None)
        t = tuple_t([v.type for v in wrapped], fields)
        return tuple(wrapped, t)
    if isinstance(x, _value):
        return x
    return constexpr(x)


def _aggregate(cls):
    """
    Create a Tileon aggregate type.

    Args:
        cls: The user-defined Python class.

    Returns:
        The Tileon aggregate type.
    """
    init = cls.__dict__.get("__init__", None)
    if init is None:
        field_names = builtins.tuple(cls.__annotations__.keys())

        def init(self, *args, **kwargs):
            if len(args) > len(field_names):
                raise TypeError(
                    f"{cls.__name__}.__init__() takes {len(field_names) + 1} positional arguments "
                    f"but {len(args) + 1} were given")

            for idx, name in enumerate(field_names):
                if idx < len(args):
                    if name in kwargs:
                        raise TypeError(
                            f"{cls.__name__}.__init__() got multiple values for argument '{name}'"
                        )
                    value = args[idx]
                elif name in kwargs:
                    value = kwargs.pop(name)
                else:
                    raise TypeError(
                        f"{cls.__name__}.__init__() missing required argument: '{name}'"
                    )

                value = _wrap_init_args(value)
                setattr(self, name, value)

            if kwargs:
                unexpected = next(iter(kwargs))
                raise TypeError(
                    f"{cls.__name__}.__init__() got an unexpected keyword argument '{unexpected}'"
                )

        init.__tileon_builtin__ = True

    # Define the wrapped Tileon value type.
    class aggregate_value(_value):
        __tileon_builtin__ = True
        __tileon_aggregate__ = True
        __annotations__ = cls.__annotations__

        @classmethod
        def _get_instance(this_cls):
            return super().__new__(this_cls)

        def __new__(this_cls,
                    *args,
                    _semantic=None,
                    _generator=None,
                    **kwargs):
            # Call into the user-defined constructor.
            instance = this_cls._get_instance()
            extra_kwargs = {}
            if isinstance(init, JITCallable):
                raise ValueError(
                    f"{cls.__name__}.__init__ cannot be a @tileon.jit function"
                )
            else:
                if "_semantic" in inspect.signature(init).parameters:
                    extra_kwargs["_semantic"] = _semantic
                if "_generator" in inspect.signature(init).parameters:
                    extra_kwargs["_generator"] = _generator
            init(instance, *args, **extra_kwargs, **kwargs)

            # Require that the user-defined constructor initialized all fields.
            for name in cls.__annotations__.keys():
                if not hasattr(instance, name):
                    raise AttributeError(
                        f"constructor for {cls.__name__} did not initialize attribute '{name}'"
                    )

            return instance

        # Only allow setting attributes defined in the class annotations.
        def __setattr__(self, name, value):
            if name not in cls.__annotations__:
                raise AttributeError(
                    f"{cls.__name__} has no attribute '{name}'")
            if not isinstance(value, cls.__annotations__[name]):
                raise TypeError(
                    f"Expected {cls.__annotations__[name]} for attribute '{name}', got {type(value)}"
                )
            super().__setattr__(name, value)

        def _set_name(self, builder: ir.builder, name: str) -> None:
            """
            Set the name of the aggregate value.

            Args:
                builder: The IR builder.
                name: The name of the aggregate value.
            """
            for key_name in cls.__annotations__.keys():
                getattr(self, key_name)._set_name(builder,
                                                  f"{name}.{key_name}")

        def _flatten_ir(self, handles: List[ir.value]) -> None:
            """
            Flatten the IR values of the aggregate value.

            Args:
                handles: The list of IR values.
            """
            for name in cls.__annotations__.keys():
                getattr(self, name)._flatten_ir(handles)

        @property
        def type(self):
            """
            Get the type of the aggregate value.

            Returns:
                The type of the aggregate value.
            """
            return _aggregate_t(aggregate_value,
                                [(name, getattr(self, name).type)
                                 for name in cls.__annotations__.keys()])

    hash_attrs = [init]

    for name, member in inspect.getmembers(cls):
        if inspect.isfunction(member) or inspect.ismethod(
                member) or isinstance(member, JITCallable):
            if name != "__init__":
                setattr(aggregate_value, name, member)
                hash_attrs.append(member)

    aggregate_value.hash_attrs = hash_attrs
    aggregate_value.__name__ = cls.__name__
    aggregate_value.__module__ = cls.__module__
    aggregate_value.__qualname__ = cls.__qualname__
    aggregate_value.__doc__ = cls.__doc__

    return aggregate_value


# -----------------------------------------------------------------------------
# SPMD Programming Model
# -----------------------------------------------------------------------------


@builtin
def program_id(axis, _semantic=None):
    """Returns the ID of the current program instance along the given :code:`axis`.

    Args:
        axis (int): The axis of the 3D launch grid to retrieve the program ID for.
            Must be an integer value of 0, 1, or 2.

    Returns:
        int: The program ID of the current kernel instance along the specified axis.
    """
    axis = _unwrap_if_constexpr(axis)
    return _semantic.program_id(axis)


@builtin
def num_programs(axis, _semantic=None):
    """
    Returns the number of program instances launched along the given :code:`axis`.

    Args:
        axis (int): The axis of the 3D launch grid. Must be 0, 1 or 2.

    Returns:
        int: The number of program instances launched along the specified axis.
    """
    axis = _unwrap_if_constexpr(axis)
    return _semantic.num_programs(axis)


# -----------------------------------------------------------------------------
# Block Initialization
# -----------------------------------------------------------------------------


@builtin
def arange(start, end, _semantic=None):
    start = _unwrap_if_constexpr(start)
    end = _unwrap_if_constexpr(end)
    return _semantic.arange(start, end)


arange.__doc__ = f"""
    Returns contiguous values within the half-open interval :code:`[start,
    end)`.  :code:`end - start` must be less than or equal to
    :code:`TILEON_MAX_TENSOR_NUMEL = {TILEON_MAX_TENSOR_NUMEL}`

    Args:
        start (int): Start of the interval. Must be a power of two.
        end (int): End of the interval. Must be a power of two greater than
            :code:`start`.

    Returns:
        tensor: A tensor of shape :code:`(end - start,)` containing the values
            :code:`[start, start + 1, ..., end - 1]`.
"""


def _shape_check_impl(shape):
    """
    Check the shape of the block.

    Args:
        shape (tuple of ints): Shape of the tensor.

    Returns:
        tuple of ints: The shape of the tensor.
    """
    shape = _unwrap_shape(shape)
    validate_block_shape(shape)
    return shape


@builtin
def full(shape, value, dtype, _semantic=None):
    """
    Returns a tensor filled with the scalar value for the given :code:`shape` and :code:`dtype`.

    Args:
        shape (tuple of ints): Shape of the new array, e.g., (8, 16) or (8, )
        value: A scalar value to fill the array with
        dtype (tl.dtype): Data type of the new array, e.g., :code:`tl.float16`

    Returns:
        tensor: A tensor of shape :code:`shape` filled with the scalar value :code:`value`.
    """
    shape = _shape_check_impl(shape)
    value = _unwrap_if_constexpr(value)
    dtype = _unwrap_if_constexpr(dtype)
    return _semantic.full(shape, value, dtype)


def tensor_member(fn: T) -> T:
    """
    Decorator that adds this free function as a member fn on class tensor.

    When called as a member function on class tensor, the first argument to `fn`
    is `self`, i.e. the tensor object.

    If there are multiple decorators on a function, you probably want this one
    to be the highest one (i.e. furthest from the function's `def`), so it's
    applied last.

    Unfortunately you still need to add a type stub to the body of class tensor
    in order for pytype to know about it.

    Args:
        fn: The function to decorate.

    Returns:
        The decorated function.
    """
    assert callable(fn), (
        "tensor_member decorator must be applied to a function, "
        f"but got {type(fn)}"
    )

    orig_sig = inspect.signature(fn)
    sig_len = len(orig_sig.parameters.keys() - {"_semantic", "_generator"})
    assert sig_len > 0, (
        "tensor_member decorator must be applied to a function with "
        f"at least one argument, but got {sig_len}")

    if sig_len > 1:
        args1, args2 = "...", ", ..."
    else:
        args1, args2 = "", ""

    if not fn.__doc__:
        fn.__doc__ = ""
    fn.__doc__ += f"""\n
    This function can also be called as a member function on :py:class:`tensor`,
    as :code:`x.{fn.__name__}({args1})` instead of :code:`{fn.__name__}(x{args2})`.
    """

    new_params = list(orig_sig.parameters.values())
    new_params[0] = new_params[0].replace(name='self')
    new_sig = orig_sig.replace(parameters=new_params)

    if isinstance(fn, JITCallable):
        setattr(tensor, fn.__name__, fn)
    else:

        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapper.__signature__ = new_sig
        wrapper.signature = new_sig
        wrapper.__doc__ = f"Forwards to :py:func:`{fn.__name__}` free function"

        if is_builtin(fn):
            setattr(wrapper, TILEON_BUILTIN, True)

        setattr(tensor, fn.__name__, wrapper)

    return fn


# -----------------------------------------------------------------------------
# Shape Manipulation
# -----------------------------------------------------------------------------


@builtin
def broadcast(input, other, _semantic=None):
    """
    Tries to broadcast the two given blocks to a common compatible shape.

    Args:
        input (tensor): The first input tensor.
        other (tensor): The second input tensor.

    Returns:
        tensor: A tensor of shape :code:`(max(input.shape, other.shape))`
            containing the values :code:`input` and :code:`other` broadcasted
            to a common compatible shape.
    """
    return _semantic.broadcast_impl_value(input, other)


def _unwrap_iterable(x):
    """
    Unwraps the iterable with one element.

    Args:
        x: The input to unwrap.

    Returns:
        The unwrapped iterable if x has one element and x[0] is iterable.
        Otherwise, returns x.
    """
    if len(x) == 1:
        # We use try/except instead of `collections.abc.Iterable`
        # to work with constexpr.
        try:
            iter(x[0])
            return x[0]
        except TypeError:
            pass

    return x


@tensor_member
@builtin
def broadcast_to(input, *shape, _semantic=None):
    """
    Tries to broadcast the given tensor to a new :code:`shape`.

    Args:
        input (tensor): The input tensor.
        shape (tuple of ints): The desired shape.

    Returns:
        tensor: A tensor of shape :code:`shape` containing the values
            :code:`input` broadcasted to the new shape.

    Note:
        :code:`shape` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        broadcast_to(x, (32, 32))
        broadcast_to(x, 32, 32)
    """
    shape = _shape_check_impl(_unwrap_iterable(shape))
    return _semantic.broadcast_impl_shape(input, shape)


@tensor_member
@builtin
def transpose(input: tensor, *dims, _semantic=None):
    """
    Permutes the dimensions of a tensor.

    If the parameter :code:`dims` is not specified, the function defaults to
    swapping the last two axes, thereby performing an (optionally batched)
    2D transpose.

    Args:
        input (tensor): The input tensor.
        dims (tuple of ints): The desired ordering of dimensions.  For example,
            :code:`(2, 1, 0)` reverses the order dims in a 3D tensor.

    Returns:
        tensor: A tensor of shape :code:`input.shape` with the dimensions
            permuted as specified by :code:`dims`.

    Note:
        :code:`dims` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        transpose(x, (2, 1, 0))
        transpose(x, 2, 1, 0)

    :py:func:`permute` is equivalent to this function, except it doesn't
    have the special case when no permutation is specified.
    """
    dims = _unwrap_iterable(dims)
    if not dims:
        n = len(input.shape)
        if n < 2:
            raise ValueError(
                "tl.transpose invoked with a 0- or 1-dimensional tensor")
        dims = list(builtins.range(n - 2)) + [n - 1, n - 2]
    return _semantic.permute(input, dims)


@tensor_member
@builtin
def permute(input, *dims, _semantic=None):
    """
    Permutes the dimensions of a tensor.

    Args:
        input (tensor): The input tensor.
        dims (tuple of ints): The desired ordering of dimensions.  For example,
            :code:`(2, 1, 0)` reverses the order dims in a 3D tensor.

    Returns:
        tensor: A tensor of shape :code:`input.shape` with the dimensions
            permuted as specified by :code:`dims`.

    Note:
        :code:`dims` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        permute(x, (2, 1, 0))
        permute(x, 2, 1, 0)

    :py:func:`transpose` is equivalent to this function, except when
    :code:`dims` is empty, it tries to swap the last two axes.
    """
    dims = _unwrap_iterable(dims)
    return _semantic.permute(input, dims)


def _wrap_axis(axis: int, ndim: int) -> int:
    """
    Wraps the axis to be in the range [-ndim, ndim).

    Args:
        axis (int): The axis to wrap.
        ndim (int): The number of dimensions.

    Returns:
        int: The wrapped axis.
    """
    if not (-ndim <= axis < ndim):
        raise ValueError(
            f"invalid axis {axis}. Expected {-ndim} <= axis < {ndim}")

    return axis if axis >= 0 else axis + ndim


@builtin
def cat(input, other, dim=0, can_reorder=False, _semantic=None):
    """
    Concatenate the given blocks

    Args:
        input (tensor): The first input tensor.
        other (tensor): The second input tensor.
        can_reorder (bool): Compiler hint. If true, the compiler is
            allowed to reorder elements while concatenating inputs.  Only use if the
            order does not matter (e.g., result is only used in reduction ops).
        dim (int): The dimension to concatenate along (used when can_reorder is False).

    Returns:
        tensor: A tensor of shape :code:`(max(input.shape, other.shape))`
            containing the values :code:`input` and :code:`other` broadcasted
            to a common compatible shape.
    """
    if can_reorder:
        return _semantic.cat(input, other, can_reorder)

    rank = len(input.shape)
    assert rank == len(
        other.shape
    ), f"tensors must have the same rank, got {rank} and {len(other.shape)}"
    dim = _wrap_axis(_unwrap_if_constexpr(dim), rank)
    assert all(
        input.shape[i] == other.shape[i] for i in builtins.range(rank)
        if i != dim
    ), (f"tensor dims must match except in the concat dimension {dim}, got {input.shape} and {other.shape}"
        )

    # Join introduces a new minor dim; move it before the concat dim and merge.
    c = join(input, other, _semantic=_semantic)
    order = list(builtins.range(rank))
    order.insert(dim, rank)
    c = permute(c, order, _semantic=_semantic)
    new_shape = list(input.shape)
    new_shape[dim] = input.shape[dim] + other.shape[dim]
    return reshape(c, new_shape, _semantic=_semantic)


@builtin
def join(a, b, _semantic=None):
    """
    Join the given tensors in a new, minor dimension.

    For example, given two tensors of shape (4,8), produces a new tensor of
    shape (4,8,2).  Given two scalars, returns a tensor of shape (2).

    The two inputs are broadcasted to be the same shape.

    If you want to join more than two elements, you can use multiple calls to
    this function.  This reflects the constraint in Tileon that tensors must
    have power-of-two sizes.

    join is the inverse of split.

    Args:
        a (tensor): The first input tensor.
        b (tensor): The second input tensor.

    Returns:
        tensor: A tensor of shape :code:`(max(a.shape, b.shape))`
            containing the values :code:`a` and :code:`b` broadcasted
            to a common compatible shape.
    """
    return _semantic.join(a, b)


def _unsplat(x, _semantic=None):
    """
    Convert a single-element tensor to a scalar.

    Args:
        x (tensor): The input tensor.

    Returns:
        tensor: A scalar tensor.
    """
    if len(x.shape) == 0:
        return x
    numel = 1
    for d in x.shape:
        numel *= d
    assert numel == 1, "can only unsplat single-element tensors"
    return _semantic.unsplat(x)


@tensor_member
@builtin
def split(x, _semantic=None) -> tuple[tensor, tensor]:
    """
    Split a tensor in two along its last dim, which must have size 2.

    For example, given a tensor of shape (4,8,2), produces two tensors of shape
    (4,8).  Given a tensor of shape (2), returns two scalars.

    If you want to split into more than two pieces, you can use multiple calls
    to this function (probably plus calling reshape).  This reflects the
    constraint in Tileon that tensors must have power-of-two sizes.

    split is the inverse of join.

    Args:
        x (tensor): The input tensor.

    Returns:
        tuple[tensor, tensor]: A tuple of two tensors.
    """
    # If len(x.shape) == 1, i.e. x.shape == [2], we should return two scalars.
    # But _semantic.split can only handle returning tensors.  Work around this by
    # expanding the input to shape [1,2] and then reducing the result.
    is_dim_1 = len(x.shape) == 1
    if is_dim_1:
        x = _semantic.expand_dims(x, 0)

    out_lhs, out_rhs = _semantic.split(x)

    if is_dim_1:
        # Currently `reduce` is the best way to convert a tensor of shape [1] to a scalar.
        out_lhs = _unsplat(out_lhs, _semantic=_semantic)
        out_rhs = _unsplat(out_rhs, _semantic=_semantic)

    return out_lhs, out_rhs


@tensor_member
@builtin
def reshape(input, *shape, can_reorder=False, _semantic=None):
    """
    Returns a tensor with the same number of elements as input but with the
    provided shape.

    :param input: The input tensor.
    :type input: Block
    :param shape: The new shape.

    :code:`shape` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        reshape(x, (32, 32))
        reshape(x, 32, 32)
    """
    shape = _shape_check_impl(_unwrap_iterable(shape))
    if len(shape) == 0:
        return _unsplat(input, _semantic=_semantic)
    return _semantic.reshape(input, shape, can_reorder)


@deprecated(
    "tl.view is deprecated, please use reshape with can_reorder=True.", )
@tensor_member
@builtin
def view(input, *shape, _semantic=None):
    """
    Returns a tensor with the same elements as `input` but a different shape.
    The order of the elements may not be preserved.

    Args:
        input (tensor): The input tensor.
        shape (tuple[int] | int): The desired shape.

    :code:`shape` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        view(x, (32, 32))
        view(x, 32, 32)
    """
    shape = _shape_check_impl(_unwrap_iterable(shape))
    return _semantic.reshape(input, shape, can_reorder=True)


@tensor_member
@builtin
def item(input, _semantic=None):
    """
    Converts a single-element tensor into a scalar.

    Args:
        input (tensor): The input tensor.

    Returns:
        tensor: A scalar tensor.
    """
    return _unsplat(input, _semantic=_semantic)


@tensor_member
@builtin
def expand_dims(input, axis, _semantic=None):
    """
    Expand the shape of a tensor, by inserting new length-1 dimensions.

    Axis indices are with respect to the resulting tensor, so
    ``result.shape[axis]`` will be 1 for each axis.

    Args:
        input (tensor): The input tensor.
        axis (int | Sequence[int]): The indices to add new axes.

    Returns:
        tensor: A tensor with the same elements as `input` but with the
            specified axes expanded.

    """
    input = _semantic.to_tensor(input)
    axis = _unwrap_if_constexpr(axis)
    axes = list(axis) if isinstance(axis, (Sequence, tuple)) else [axis]
    new_ndim = len(input.shape) + len(axes)
    axes = [_wrap_axis(_unwrap_if_constexpr(d), new_ndim) for d in axes]

    if len(set(axes)) != len(axes):
        raise ValueError(
            f"expand_dims received duplicate axes, normalized axes = {axes}")

    ret = input
    for a in sorted(axes):
        ret = _semantic.expand_dims(ret, a)
    return ret


@tensor_member
@builtin
def cast(input,
         dtype: dtype,
         fp_downcast_rounding: Optional[str] = None,
         bitcast: bool = False,
         _semantic=None):
    """
    Casts a tensor to the given :code:`dtype`.

    Args:
        input (tensor): The input tensor.
        dtype (dtype): The target data type.
        fp_downcast_rounding (str, optional): The rounding mode for downcasting
            floating-point values. This parameter is only used when self is a
            floating-point tensor and dtype is a floating-point type with a
            smaller bitwidth. Supported values are :code:`"rtne"` (round to
            nearest, ties to even) and :code:`"rtz"` (round towards zero).
        bitcast (bool, optional): If true, the tensor is bitcasted to the given
            :code:`dtype`, instead of being numerically casted.

    Returns:
        tensor: A tensor with the same elements as `input` but with the
            specified dtype.
    """
    input = _semantic.to_tensor(input)
    dtype = _unwrap_if_constexpr(dtype)
    fp_downcast_rounding = _unwrap_if_constexpr(fp_downcast_rounding)
    bitcast = _unwrap_if_constexpr(bitcast)
    if bitcast:
        return _semantic.bitcast(input, dtype)
    return _semantic.cast(input, dtype, fp_downcast_rounding)


# -----------------------------------------------------------------------------
# Linear Algebra
# -----------------------------------------------------------------------------


@builtin
def dot(input,
        other,
        acc=None,
        input_precision=None,
        allow_tf32=None,
        max_num_imprecise_acc=None,
        out_dtype=float32,
        _semantic=None):
    """
    Returns the matrix product of two blocks.

    The two blocks must both be two-dimensional or three-dimensional and have compatible inner dimensions.
    For three-dimensional blocks, `tl.dot` performs the batched matrix product,
    where the first dimension of each block represents the batch dimension.

    Note:
        When using TF32 precision, the float32 inputs may be truncated to TF32 format (19-bit floating point)
        without rounding which may bias the result. For best results, you must round to TF32 explicitly, or load
        the data using `TensorDescriptor` with `round_f32_to_tf32=True`.

    Args:
        input (tensor): The first tensor to be multiplied, must be 2D or 3D with scalar-type
            in {:code:`int8`, :code:`float8_e5m2`, :code:`float16`, :code:`bfloat16`, :code:`float32`}
        other (tensor): The second tensor to be multiplied, must be 2D or 3D with scalar-type
            in {:code:`int8`, :code:`float8_e5m2`, :code:`float16`, :code:`bfloat16`, :code:`float32`}
        acc (tensor, optional): The accumulator tensor. If not None, the result is added to this tensor.
            Must be 2D or 3D with scalar-type in {:code:`float16`, :code:`float32`, :code:`int32`}
        input_precision (str, optional): How to exercise the Tensor Cores for f32 x f32. If
            not None, must be one of {:code:`"tf32"`, :code:`"tf32x3"`, :code:`"ieee"`}.
            Default: :code:`"tf32"`.
        max_num_imprecise_acc (int, optional): The maximum number of accumulators that can be
            accumulated in fp32 precision.
        out_dtype (dtype, optional): The output data type. Default: :code:`float32`.
    """
    assert input_precision is None or allow_tf32 is None, "Only one of input_precision and allow_tf32 can be specified"
    if input_precision is None:
        supports_tf32 = "tf32" in _semantic.builder.options.allowed_dot_input_precisions
        input_precision = (knobs.language.fp32_default or "tf32"
                           if supports_tf32 and
                           (allow_tf32 or allow_tf32 is None) else "ieee")

    input_precision = _unwrap_if_constexpr(input_precision)
    out_dtype = _unwrap_if_constexpr(out_dtype)
    max_num_imprecise_acc = _unwrap_if_constexpr(max_num_imprecise_acc)
    acc = _unwrap_if_constexpr(acc)

    a_shape = list(input.shape)
    b_shape = list(other.shape)
    assert len(a_shape) == len(
        b_shape) >= 2, "input and other must have equal ranks >= 2"
    assert a_shape[:
                   -2] == b_shape[:
                                  -2], "input and other must have equal batch shapes"
    assert a_shape[-1] == b_shape[
        -2], "input and other must have equal reduction dimensions"

    # compute shape of accumulator:
    c_shape = a_shape[:-1] + [b_shape[-1]]
    if acc is not None:
        assert list(acc.shape) == c_shape, "accumulator shape is incompatible"
    rank = len(c_shape)

    if rank >= 4:
        batch_size = 1
        for i in builtins.range(rank - 2):
            batch_size *= c_shape[i]
        input = _semantic.reshape(input, [batch_size] + a_shape[-2:],
                                  can_reorder=False)
        other = _semantic.reshape(other, [batch_size] + b_shape[-2:],
                                  can_reorder=False)
        if acc is not None:
            acc = _semantic.reshape(acc, [batch_size] + c_shape[-2:],
                                    can_reorder=False)

    res = _semantic.dot(input, other, acc, input_precision,
                        max_num_imprecise_acc, out_dtype)

    if rank >= 4:
        res = _semantic.reshape(res, c_shape, can_reorder=False)

    assert list(res.shape) == c_shape, "output shape is unexpected"
    return res


@builtin
def dot_scaled(lhs,
               lhs_scale,
               lhs_format,
               rhs,
               rhs_scale,
               rhs_format,
               acc=None,
               fast_math=False,
               lhs_k_pack=True,
               rhs_k_pack=True,
               out_dtype=float32,
               _semantic=None):
    """
    Returns the matrix product of two blocks in microscaling format.

    lhs and rhs use microscaling formats described here:
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

    Software emulation enables targeting hardware architectures without native microscaling
    operation support. Right now for such case, microscaled lhs/rhs are upcasted to
    :code:`bf16` element type beforehand for dot computation, with one exception:
    for AMD CDNA3 specifically, if one of the inputs is of :code:`fp16` element type,
    the other input is also upcasted to :code:`fp16` element type instead.
    This behavior is experimental and may be subject to change in the future.

    Args:
        lhs(tensor): The first tensor to be multiplied. It is a 2D tensor representing fp4, fp8 or bf16 elements.
            Fp4 elements are packed into uint8 inputs with the first element in lower bits.
            Fp8 are stored as uint8 or the corresponding fp8 type.
        lhs_scale(tensor): Scale factor for lhs tensor. Shape should be [M, K//group_size] when lhs is [M, K],
            where group_size is 32 if scales type are `e8m0`.
        lhs_format(str): format of the lhs tensor. Available formats:
            {:code:`e2m1`, :code:`e4m3`, :code:`e5m2`, :code:`bf16`, :code:`fp16`}.
        rhs(tensor): The second tensor to be multiplied. It is a 2D tensor representing fp4, fp8 or bf16 elements.
            Fp4 elements are packed into uint8 inputs with the first element in lower bits.
            Fp8 are stored as uint8 or the corresponding fp8 type.
        rhs_scale(tensor): Scale factor for rhs tensor. Shape should be [N, K//group_size] where rhs is [K, N].
            Important: Do NOT transpose rhs_scale
        rhs_format(str): format of the rhs tensor. Available formats:
            {:code:`e2m1`, :code:`e4m3`, :code:`e5m2`, :code:`bf16`, :code:`fp16`}.
        acc(tensor, optional): The accumulator tensor. If not None, the result is added to this tensor.
        fast_math(bool): If true, fast math is enabled. Default: :code:`False`.
        lhs_k_pack(bool): If false, the lhs tensor is packed into uint8 along M dimension. Default: :code:`True`.
        rhs_k_pack(bool): If false, the rhs tensor is packed into uint8 along N dimension. Default: :code:`True`.
        out_dtype(dtype): The output data type. Default: :code:`float32`.
    """
    out_dtype = _unwrap_if_constexpr(out_dtype)
    acc = _unwrap_if_constexpr(acc)
    assert out_dtype == float32, "Only float32 is supported for out_dtype at the moment"
    return _semantic.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale,
                                rhs_format, acc, fast_math, lhs_k_pack,
                                rhs_k_pack, out_dtype)


# -----------------------------------------------------------------------------
# Non-Atomic Memory Operations
# -----------------------------------------------------------------------------


@builtin
def load(pointer,
         mask=None,
         other=None,
         boundary_check=(),
         padding_option="",
         cache_modifier="",
         eviction_policy="",
         volatile=False,
         _semantic=None):
    """
    Return a tensor of data whose values are loaded from memory at location defined by `pointer`:

        (1) If `pointer` is a single element pointer, a scalar is be loaded.  In
            this case:

            - `mask` and `other` must also be scalars,
            - `other` is implicitly typecast to `pointer.dtype.element_t`, and
            - `boundary_check` and `padding_option` must be empty.

        (2) If `pointer` is an N-dimensional tensor of pointers, an
            N-dimensional tensor is loaded.  In this case:

            - `mask` and `other` are implicitly broadcast to `pointer.shape`,
            - `other` is implicitly typecast to `pointer.dtype.element_t`, and
            - `boundary_check` and `padding_option` must be empty.

        (3) If `pointer` is a block pointer defined by `make_block_ptr`, a
            tensor is loaded.  In this case:

            - `mask` and `other` must be `None`, and
            - `boundary_check` and `padding_option` can be specified to control the behavior of out-of-bound access.

    Args:
        pointer: Pointer to the data to be loaded. Can be either a `triton.PointerType`
            instance or a block with dtype set to `triton.PointerType`.
        mask: Optional mask tensor of type `triton.int1`. If `mask[idx]` is False,
            the data at address `pointer[idx]` will not be loaded. Must be `None`
            when using block pointers.
        other: Optional block tensor. If `mask[idx]` is False, `other[idx]` will be
            returned instead of the data from `pointer[idx]`.
        boundary_check: Optional tuple of integers indicating which dimensions
            should undergo boundary checking (to prevent out-of-bounds access).
        padding_option: String specifying the padding value for out-of-bounds access,
            must be one of {"", "zero", "nan"}:
            - "": Undefined padding value (default)
            - "zero": Pad with 0 for out-of-bounds addresses
            - "nan": Pad with NaN for out-of-bounds addresses
        cache_modifier: Optional string to modify cache behavior in NVIDIA PTX,
            must be one of {"", ".ca", ".cg", ".cv"}:
            - "": Default cache behavior
            - ".ca": Cache at all levels (L1/L2/etc.)
            - ".cg": Cache at global level (L2 and below, not L1)
            - ".cv": Do not cache, force re-fetch from global memory
            See NVIDIA PTX documentation for cache operators for more details:
            https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
        eviction_policy: Optional string to modify the cache eviction policy in NVIDIA PTX.
        volatile: Optional boolean to enable/disable the `volatile` option in
            NVIDIA PTX (controls memory consistency semantics).
    """
    # `mask` and `other` can be constexpr
    mask = _unwrap_if_constexpr(mask)
    other = _unwrap_if_constexpr(other)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    if other is not None:
        other = _semantic.to_tensor(other)
    padding_option = _unwrap_if_constexpr(padding_option)
    cache_modifier = _unwrap_if_constexpr(cache_modifier)
    eviction_policy = _unwrap_if_constexpr(eviction_policy)
    volatile = _unwrap_if_constexpr(volatile)
    return _semantic.load(pointer, mask, other, boundary_check, padding_option,
                          cache_modifier, eviction_policy, volatile)


@builtin
def load_tensor_descriptor(desc: _tensor_descriptor,
                           offsets: Sequence[constexpr | tensor],
                           _semantic=None) -> tensor:
    """
    Load a block of data from a tensor descriptor.

    Args:
        desc: The tensor descriptor to load from.
        offsets: A sequence of offsets to load from.

    Returns:
        A tensor of data loaded from the descriptor.
    """
    return desc.load(offsets, _semantic=_semantic)


@builtin
def store_tensor_descriptor(desc: _tensor_descriptor,
                            offsets: Sequence[constexpr | tensor],
                            value: tensor,
                            _semantic=None) -> tensor:
    """
    Store a block of data to a tensor descriptor.

    Args:
        desc: The tensor descriptor to store to.
        offsets: A sequence of offsets to store to.
        value: The tensor of data to store.

    Returns:
        The tensor descriptor.
    """
    return desc.store(offsets, value, _semantic=_semantic)


@tensor_member
@builtin
def store(pointer,
          value,
          mask=None,
          boundary_check=(),
          cache_modifier="",
          eviction_policy="",
          _semantic=None):
    """
    Store a tensor of data into memory locations defined by `pointer`.

        (1) If `pointer` is a single element pointer, a scalar is stored.  In
            this case:

            - `mask` must also be scalar, and
            - `boundary_check` and `padding_option` must be empty.

        (2) If `pointer` is an N-dimensional tensor of pointers, an
            N-dimensional block is stored.  In this case:

            - `mask` is implicitly broadcast to `pointer.shape`, and
            - `boundary_check` must be empty.

        (3) If `pointer` is a block pointer defined by `make_block_ptr`, a block
            of data is stored.  In this case:

            - `mask` must be None, and
            - `boundary_check` can be specified to control the behavior of out-of-bound access.

    `value` is implicitly broadcast to `pointer.shape` and typecast to `pointer.dtype.element_t`.

    Args:
        pointer: triton.PointerType or block of dtype=triton.PointerType
            The memory location where the elements of `value` are stored.
        value: Block
            The tensor of elements to be stored.
        mask: block of triton.int1, optional
            If `mask[idx]` is False, the element `value[idx]` will not be stored at the address
            `pointer[idx]`.
        boundary_check: tuple of integers, optional
            Tuple of integers indicating which dimensions should perform boundary checking
            (prevents out-of-bounds memory writes).
        cache_modifier: str, optional
            Cache behavior modifier for NVIDIA PTX, must be one of {"", ".wb", ".cg", ".cs", ".wt"}:
            - "": Default cache behavior
            - ".wb": Cache write-back (all coherent cache levels)
            - ".cg": Cache global (cache at L2 and below, excludes L1)
            - ".cs": Cache streaming (optimized for sequential memory access)
            - ".wt": Cache write-through (write directly to global memory, bypass L1 cache)
            See NVIDIA PTX cache operators for details:
            https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
        eviction_policy: str, optional
            Cache eviction policy for NVIDIA PTX, must be one of {"", "evict_first", "evict_last"}:
            - "": Default eviction policy (hardware-defined)
            - "evict_first": Evict cached data first when cache capacity is reached
            - "evict_last": Evict cached data last when cache capacity is reached
    """
    # `value` can be constexpr
    value = _semantic.to_tensor(value)
    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    cache_modifier = _unwrap_if_constexpr(cache_modifier)
    eviction_policy = _unwrap_if_constexpr(eviction_policy)
    return _semantic.store(pointer, value, mask, boundary_check,
                           cache_modifier, eviction_policy)


@builtin
def make_block_ptr(base: tensor,
                   shape,
                   strides,
                   offsets,
                   block_shape,
                   order,
                   _semantic=None):
    """
    Returns a pointer to a block in a parent tensor.

    Args:
        base: The base pointer to the parent tensor
        shape: The shape of the parent tensor
        strides: The strides of the parent tensor
        offsets: The offsets to the block
        block_shape: The shape of the block
        order: The order of the original data format
    """
    return _semantic.make_block_ptr(base, shape, strides, offsets, block_shape,
                                    order)


def must_use_result(func: Union[Callable, str],
                    flag: Union[bool, str] = True) -> Callable:
    """A wrapper function that marks the result of the decorated function as must be used.

    Args:
        func: The function to decorate.
        flag: If True, the result must be used. If False, the result must not be used.
              If a string, it will be used as the error message.
    """
    if isinstance(func, str):
        return lambda x: must_use_result(x, flag=func)
    func._must_use_result = flag
    return func


@must_use_result(
    "Note that tl.advance does not have any side effects. To move the block pointer, "
    "you need to assign the result of tl.advance to a variable.")
@tensor_member
@builtin
def advance(base, offsets, _semantic=None):
    """
    Advance a block pointer

    Args:
        base: the block pointer to advance
        offsets: the offsets to advance, a tuple by dimension
    """
    return _semantic.advance(base, offsets)


@builtin
def make_tensor_descriptor(
    base: tensor,
    shape: List[tensor],
    strides: List[tensor],
    block_shape: List[constexpr],
    padding_option="zero",
    _semantic=None,
) -> tensor_descriptor:
    """Make a tensor descriptor object

    Args:
        base: the base pointer of the tensor, must be 16-byte aligned
        shape: A list of non-negative integers representing the tensor shape
        strides: A list of tensor strides. Leading dimensions must be multiples
            of 16-byte strides and the last dimension must be contiguous.
        block_shape: The shape of block to be loaded/stored from global memory

    Note:
        On NVIDIA GPUs with TMA support, this will result in a TMA descriptor object
        and loads and stores from the descriptor will be backed by the TMA hardware.

        Currently only 2-5 dimensional tensors are supported.

    Example:
        Example of using `tl.make_tensor_descriptor` for in-place absolute value computation
        with TMA-accelerated memory access:

        ```python
        @tileon.jit
        def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
            desc = tl.make_tensor_descriptor(
                in_out_ptr,
                shape=[M, N],
                strides=[N, 1],
                block_shape=[M_BLOCK, N_BLOCK],
            )

            moffset = tl.program_id(0) * M_BLOCK
            noffset = tl.program_id(1) * N_BLOCK

            value = desc.load([moffset, noffset])
            desc.store([moffset, noffset], tl.abs(value))

        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        tileon.set_allocator(alloc_fn)

        M, N = 256, 256
        x = torch.randn(M, N, device="cuda")
        M_BLOCK, N_BLOCK = 32, 32
        grid = (M / M_BLOCK, N / N_BLOCK)
        inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)
        ```
    """

    padding_option = _unwrap_if_constexpr(padding_option)
    return _semantic.make_tensor_descriptor(base, shape, strides, block_shape,
                                            padding_option)


# -----------------------------------------------------------------------------
# Atomic Memory Operations
# -----------------------------------------------------------------------------


def add_atomic_docstr(name: str, has_cmp: bool = False) -> Callable[[T], T]:
    """Add a docstring to an atomic function.

    Args:
        name: The name of the atomic operation.
        has_cmp: Whether the atomic operation has a compare-and-swap operation.
    """

    def _decorator(func: T) -> T:
        docstr = f"""
    Performs an atomic {name} at the memory location specified by :code:`pointer`.

    Return the data stored at :code:`pointer` before the atomic operation.

    Args:
        pointer(tileon.PointerDType): The memory locations to operate on
    """
        if has_cmp:
            docstr += """
        cmp(pointer.dtype.element_t): The values expected to be found in the atomic object
    """
        docstr += """
        val(pointer.dtype.element_t): The values with which to perform the atomic operation
        sem(str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire",
        "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided,
        the function defaults to using "acq_rel" semantics.
        scope(str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation.
        Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block),
        or "sys" (stands for "SYSTEM"). The default value is "gpu".
    """
        func.__doc__ = docstr
        return func

    return _decorator


@tensor_member
@builtin
@add_atomic_docstr("compare-and-swap", has_cmp=True)
def atomic_cas(pointer, cmp, val, sem=None, scope=None, _semantic=None):
    cmp = _semantic.to_tensor(cmp)
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    return _semantic.atomic_cas(pointer, cmp, val, sem, scope)


@tensor_member
@builtin
@add_atomic_docstr("exchange")
def atomic_xchg(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_xchg(pointer, val, mask, sem, scope)


@tensor_member
@builtin
@add_atomic_docstr("add")
def atomic_add(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_add(pointer, val, mask, sem, scope)


@tensor_member
@builtin
@add_atomic_docstr("max")
def atomic_max(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_max(pointer, val, mask, sem, scope)


@tensor_member
@builtin
@add_atomic_docstr("min")
def atomic_min(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_min(pointer, val, mask, sem, scope)


@tensor_member
@builtin
@add_atomic_docstr("logical and")
def atomic_and(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_and(pointer, val, mask, sem, scope)


@tensor_member
@builtin
@add_atomic_docstr("logical or")
def atomic_or(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_or(pointer, val, mask, sem, scope)


@tensor_member
@builtin
@add_atomic_docstr("logical xor")
def atomic_xor(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_xor(pointer, val, mask, sem, scope)


# -----------------------------------------------------------------------------
# Conditioning
# -----------------------------------------------------------------------------


@builtin
def where(condition, x, y, _semantic=None):
    """
    Returns a tensor of elements from either :code:`x` or :code:`y`, depending on :code:`condition`.

    Note that :code:`x` and :code:`y` are always evaluated regardless of the value of :code:`condition`.

    If you want to avoid unintended memory operations, use the :code:`mask` arguments in `triton.load` and `triton.store` instead.

    The shape of :code:`x` and :code:`y` are both broadcast to the shape of :code:`condition`.
    :code:`x` and :code:`y` must have the same data type.

    Args:
        condition(Block of triton.bool): When True (nonzero), yield x, otherwise yield y.
        x: values selected at indices where condition is True.
        y: values selected at indices where condition is False.
    """
    condition = _semantic.to_tensor(condition)
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return _semantic.where(condition, x, y)


# -----------------------------------------------------------------------------
# Math
# -----------------------------------------------------------------------------


@builtin
def add(x, y, sanitize_overflow: constexpr = True, _semantic=None):
    """
    Computes the element-wise addition of :code:`x` and :code:`y`.

    Args:
        x: the first input tensor
        y: the second input tensor
        sanitize_overflow(bool): whether to sanitize overflow.
    """
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return _semantic.add(x, y, sanitize_overflow)


@builtin
def sub(x, y, sanitize_overflow: constexpr = True, _semantic=None):
    """
    Computes the element-wise subtraction of :code:`x` and :code:`y`.

    Args:
        x: the first input tensor
        y: the second input tensor
        sanitize_overflow(bool): whether to sanitize overflow.
    """
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return _semantic.sub(x, y, sanitize_overflow)


@builtin
def mul(x, y, sanitize_overflow: constexpr = True, _semantic=None):
    """
    Computes the element-wise multiplication of :code:`x` and :code:`y`.

    Args:
        x: the first input tensor
        y: the second input tensor
        sanitize_overflow(bool): whether to sanitize overflow.
    """
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return _semantic.mul(x, y, sanitize_overflow)


@builtin
def _promote_bf16_to_f32(t: tensor, _semantic=None):
    """
    Promotes bfloat16 to float32 if the scalar type is bfloat16.
    Otherwise, returns the tensor as is.

    Args:
        t: Block
            Input tensor.
    """
    scalar_t = t.type.scalar
    # hardware doesn't support FMAX, FMIN, CMP for bfloat16
    if scalar_t is bfloat16:
        return t.to(float32, _semantic=_semantic)
    return t


@builtin
def minimum(x,
            y,
            propagate_nan: constexpr = PropagateNan.NONE,
            _semantic=None):
    """
    Computes the element-wise minimum of x and y.

    Args:
        x: Block
            First input tensor.
        y: Block
            Second input tensor.
        propagate_nan: tl.PropagateNan
            Whether to propagate NaN values.

    See Also:
        tl.PropagateNan: NaN propagation behavior enum.
    """
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    x = _promote_bf16_to_f32(x, _semantic=_semantic)
    y = _promote_bf16_to_f32(y, _semantic=_semantic)
    propagate_nan = _unwrap_if_constexpr(propagate_nan)
    return _semantic.minimum(x, y, propagate_nan)


@builtin
def maximum(x,
            y,
            propagate_nan: constexpr = PropagateNan.NONE,
            _semantic=None):
    """
    Computes the element-wise maximum of :code:`x` and :code:`y`.

    Args:
        x: Block
            First input tensor.
        y: Block
            Second input tensor.
        propagate_nan: tl.PropagateNan
            Whether to propagate NaN values.

    See Also:
        tl.PropagateNan: NaN propagation behavior enum.
    """
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    x = _promote_bf16_to_f32(x, _semantic=_semantic)
    y = _promote_bf16_to_f32(y, _semantic=_semantic)
    propagate_nan = _unwrap_if_constexpr(propagate_nan)
    return _semantic.maximum(x, y, propagate_nan)


@builtin
def clamp(x,
          min,
          max,
          propagate_nan: constexpr = PropagateNan.NONE,
          _semantic=None):
    """
    Clamps the input tensor :code:`x` within the range [min, max].
    Behavior when :code:`min` > :code:`max` is undefined.

    Args:
        x(Block): the input tensor
        min(Block): the lower bound for clamping
        max(Block): the upper bound for clamping
        propagate_nan(tl.PropagateNan): whether to propagate NaN values. Applies only to the :code:`x` tensor.
        If either :code:`min` or :code:`max` is NaN, the result is undefined.

    See Also:
        tl.PropagateNan: NaN propagation behavior enum.
    """
    x = _semantic.to_tensor(x)
    min = _semantic.to_tensor(min)
    max = _semantic.to_tensor(max)
    x = _promote_bf16_to_f32(x, _semantic=_semantic)
    min = _promote_bf16_to_f32(min, _semantic=_semantic)
    max = _promote_bf16_to_f32(max, _semantic=_semantic)
    propagate_nan = _unwrap_if_constexpr(propagate_nan)
    return _semantic.clamp(x, min, max, propagate_nan)


# -----------------------------------------------------------------------------
# Reductions
# -----------------------------------------------------------------------------


def add_reduction_docstr(
    name: str,
    return_indices_arg: Optional[str] = None,
    tie_break_arg: Optional[str] = None,
    dtype_arg: Optional[str] = None
) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = f"""
    Returns the {name} of all elements in the :code:`input` tensor along the provided :code:`axis`

    The reduction operation should be associative and commutative.

    Args:
        input(Tensor): the input values
        axis(Optional[int]): the dimension along which the reduction should be done. If None, reduce all dimensions
        keep_dims(Optional[bool]): if true, keep the reduced dimensions with length 1
        """
        if return_indices_arg is not None:
            docstr += f"""
        {return_indices_arg}(bool): if true, return index corresponding to the {name} value
        """
        if tie_break_arg is not None:
            docstr += f"""
        {tie_break_arg}(bool): if true, in case of a tie (i.e., multiple elements have the same {name} value), return the left-most index for values that aren't NaN
        """
        if dtype_arg is not None:
            docstr += f"""
        {dtype_arg}(tl.dtype): the desired data type of the returned tensor. If specified, the input tensor
            is casted to :code:`{dtype_arg}` before the operation is performed. This is useful for preventing
            data overflows. If not specified, integer and bool dtypes are upcasted to :code:`tl.int32` and float
            dtypes are upcasted to at least :code:`tl.float32`.
        """

        func.__doc__ = docstr
        return func

    return _decorator


@contextmanager
def insertion_guard(builder):
    """
    A context manager that guards the insertion point of the builder.
    """
    ip = builder.get_insertion_point()
    yield
    builder.restore_insertion_point(ip)


@tensor_member
@builtin
def reduce(
    input,
    axis,
    combine_fn,
    keep_dims=False,
    _semantic=None,
    _generator=None
):
    """Applies the combine_fn to all elements in :code:`input` tensors along the provided :code:`axis`

    Args:
        input(Tensor): the input tensor, or tuple of tensors
        axis(Optional[int]): the dimension along which the reduction should be done. If None, reduce all dimensions
        combine_fn(Callable): a function to combine two groups of scalar tensors (must be marked with @tileon.jit)
        keep_dims(bool): if true, keep the reduced dimensions with length 1

    Returns:
        Tensor: the reduced tensor
    """
    if isinstance(input, tensor):
        return reduce((input, ),
                      axis,
                      combine_fn,
                      keep_dims=keep_dims,
                      _semantic=_semantic,
                      _generator=_generator)[0]

    def make_combine_region(reduce_op):
        param_types = [t.type.scalar for t in input] * 2
        region = reduce_op.get_region(0)
        builder = _semantic.builder
        with insertion_guard(builder):
            to_ir = lambda T: T.to_ir(builder)
            block = builder.create_block_with_parent(
                region, list(map(to_ir, param_types)))
            args = [
                tensor(block.arg(i), ty) for i, ty in enumerate(param_types)
            ]
            results = _generator.call_JitFunction(combine_fn, args, kwargs={})
            if isinstance(results, tensor):
                handles = [results.handle]
            else:
                handles = [r.handle for r in results]
            builder.create_reduce_ret(*handles)

    def expand_ndims(t, ndims):
        for _ in builtins.range(ndims):
            t = expand_dims(t, 0, _semantic=_semantic)
        return t

    axis = _unwrap_if_constexpr(axis)
    keep_dims = _unwrap_if_constexpr(keep_dims)
    if axis is not None:
        axis = _wrap_axis(axis, len(input[0].shape))
    ret = _semantic.reduction(input, axis, make_combine_region)
    if keep_dims:
        if axis is not None:
            ret = tuple(expand_dims(t, axis, _semantic=_semantic) for t in ret)
        else:
            ret = tuple(expand_ndims(t, len(input[0].shape)) for t in ret)
    return ret


@builtin
def _reduce_with_indices(input,
                         axis,
                         combine_fn,
                         keep_dims=False,
                         _semantic=None,
                         _generator=None):
    """Applies the combine_fn to all elements in :code:`input` tensors along the provided :code:`axis`,
    and returns the indices of the elements that contribute to the reduction.

    Args:
        input(Tensor): the input tensor, or tuple of tensors
        axis(Optional[int]): the dimension along which the reduction should be done. If None, reduce all dimensions
        combine_fn(Callable): a function to combine two groups of scalar tensors (must be marked with @tileon.jit)
        keep_dims(bool): if true, keep the reduced dimensions with length 1
    """
    axis = _unwrap_if_constexpr(axis)
    n = input.shape[axis]
    index = arange(0, n, _semantic=_semantic)

    if len(input.shape) > 1:
        # Broadcast index across the non-reduced axes
        axes_to_expand = [
            constexpr(d) for d in builtins.range(len(input.shape))
        ]
        del axes_to_expand[axis]
        index = expand_dims(index, axes_to_expand, _semantic=_semantic)
        index = broadcast_to(index, input.shape, _semantic=_semantic)

    rvalue, rindices = reduce((input, index),
                              axis,
                              combine_fn,
                              keep_dims=keep_dims,
                              _semantic=_semantic,
                              _generator=_generator)
    return rvalue, rindices


# -----------------------------------------------------------------------------
# Scans
# -----------------------------------------------------------------------------


def add_scan_docstr(name: str, dtype_arg: str = None) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = f"""
    Returns the {name} of all elements in the :code:`input` tensor along the provided :code:`axis`

    Args:
        input(Tensor): the input values
        axis(int): the dimension along which the scan should be done
        reverse(bool): if true, the scan is performed in the reverse direction
    """

        if dtype_arg is not None:
            docstr += f"""
        {dtype_arg}(tl.dtype): the desired data type of the returned tensor. If specified, the input tensor
            is casted to :code:`{dtype_arg}` before the operation is performed. If not specified, small integer
            types (< 32 bits) are upcasted to prevent overflow. Note that :code:`tl.bfloat16` inputs are automatically
            promoted to :code:`tl.float32`.
        """

        func.__doc__ = docstr
        return func

    return _decorator


@tensor_member
@builtin
def associative_scan(input,
                     axis,
                     combine_fn,
                     reverse=False,
                     _semantic=None,
                     _generator=None):
    """Applies the combine_fn to each elements with a carry in :code:`input` tensors along
    the provided :code:`axis` and update the carry

    Args:
        input(Tensor): the input tensor, or tuple of tensors
        axis(int): the dimension along which the reduction should be done
        combine_fn(Callable): a function to combine two groups of scalar tensors (must be marked with @tileon.jit)
        reverse(bool): whether to apply the associative scan in the reverse direction along axis
    """
    if isinstance(input, tensor):
        return associative_scan((input, ),
                                axis,
                                combine_fn,
                                reverse,
                                _semantic=_semantic,
                                _generator=_generator)[0]

    def make_combine_region(scan_op):
        param_types = [t.type.scalar for t in input] * 2
        region = scan_op.get_region(0)
        builder = _semantic.builder
        with insertion_guard(builder):
            to_ir = lambda T: T.to_ir(builder)
            block = builder.create_block_with_parent(
                region, list(map(to_ir, param_types)))
            args = [
                tensor(block.arg(i), ty) for i, ty in enumerate(param_types)
            ]
            results = _generator.call_JitFunction(combine_fn, args, kwargs={})
            if isinstance(results, tensor):
                handles = [results.handle]
            else:
                handles = [r.handle for r in results]
            builder.create_scan_ret(*handles)

    axis = _unwrap_if_constexpr(axis)
    if axis is not None:
        axis = _wrap_axis(axis, len(input[0].shape))
    return _semantic.associative_scan(input, axis, make_combine_region,
                                      reverse)


@tensor_member
@builtin
def histogram(input, num_bins, mask=None, _semantic=None, _generator=None):
    """computes an histogram based on input tensor with num_bins bins, the bins have a width of 1 and start at 0.

    Args:
        input(Tensor): the input tensor
        num_bins(int): number of histogram bins
        mask(Optional[Block of `triton.int1`]): if `mask[idx]` is false, exclude `input[idx]` from histogram
    """
    num_bins = _unwrap_if_constexpr(num_bins)
    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    return _semantic.histogram(input, num_bins, mask)


@tensor_member
@builtin
def gather(src, index, axis, _semantic=None):
    """Gather from a tensor along a given dimension.

    Args:
        src(Tensor): the source tensor
        index(Tensor): the index tensor
        axis(int): the dimension to gather along

    Returns:
        Tensor: the gathered tensor
    """
    src = _unwrap_if_constexpr(src)
    index = _unwrap_if_constexpr(index)
    axis = _unwrap_if_constexpr(axis)
    return _semantic.gather(src, index, axis)


@builtin
def map_elementwise(
    scalar_fn: Callable[..., Tuple[tensor, ...]],
    *tensors: tensor,
    pack: int = 1,
    _semantic=None,
    _generator=None,
):
    """
    Map a scalar function over a tensor.

    The input tensors :code:`tensors` are implicitly broadcasted to the same shape.

    This may be useful in allowing control flow over single elements in a tensor,
    for example a multi-branch function where one branch is more expensive. With
    :code:`tl.where` you are forced to calculate both sides of the branch, but
    with an if we only execute one side.

    Example:
        ```python
        @tileon.jit
        def selu_scalar(x, alpha):
            if x > 0:
                return a
            else:
                return alpha * (tl.exp(x) - 1)

        @tileon.jit
        def selu(x, alpha):
            return tl.map_elementwise(selu_scalar, x, alpha)
        ```

    Args:
        scalar_fn(Callable[..., Tuple[tensor, ...]]): the function to map over
        tensors(Tensor): the input tensors
        pack(int): the number of elements to be processed by one function call

    Returns:
        Tensor: the mapped tensor
    """
    # Build the block for the nested region first to discover the return types
    assert pack >= 1, "pack must be at least 1"
    in_scalar_tys = [t.type.scalar for t in tensors]
    builder = _semantic.builder
    block = builder.new_block()
    scalar_args = []
    original_loc = builder.get_loc()
    for i, ty in enumerate(in_scalar_tys):
        for j in builtins.range(pack):
            block.add_argument_at(ty.to_ir(builder), original_loc)
            scalar_args.append(tensor(block.arg(i * pack + j), ty))

    with insertion_guard(builder):
        builder.set_insertion_point_to_start(block)
        scalar_results = _generator.call_JitFunction(scalar_fn,
                                                     scalar_args,
                                                     kwargs={})

        is_single = isinstance(scalar_results, tensor)
        if is_single:
            scalar_results = scalar_results,

        handles = [r.handle for r in scalar_results]
        builder.set_loc(original_loc)
        builder.create_map_elementwise_ret(handles)

    fn_result_types = [x.type for x in scalar_results]
    scalar_result_types = fn_result_types
    if pack > 1:
        scalar_result_types = fn_result_types[::pack]
        for offset in builtins.range(1, pack):
            assert scalar_result_types == fn_result_types[
                offset::pack], "type mismatch in unpacked results"

    def make_elementwise_region(elementwise_op):
        region = elementwise_op.get_region(0)
        region.push_back(block)

    builder.set_loc(original_loc)
    result = _semantic.map_elementwise(tensors, scalar_result_types, pack,
                                       make_elementwise_region)
    return result[0] if is_single else result


# -----------------------------------------------------------------------------
# Compiler Hint Ops
# -----------------------------------------------------------------------------


@builtin
def debug_barrier(_semantic=None):
    """
    Insert a barrier to synchronize all threads in a block.
    """
    return _semantic.debug_barrier()


def _unwrap_hint_values(values):
    """
    Check that the values are constexpr[int] and return the values as a list of int.
    """
    if isinstance(values, constexpr):
        values = [values]
    for i, t in enumerate(values):
        assert isinstance(
            t, constexpr), f"values element {i} must have type `constexpr`"
        assert isinstance(t.value, int), (
            f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(t.value)}]`"
        )
    return [x.value for x in values]


@builtin
def multiple_of(input, values, _semantic=None):
    """
    Let the compiler know that the values in :code:`input` are all multiples of :code:`value`.
    """
    values = _unwrap_hint_values(values)
    return _semantic.multiple_of(input, values)


@builtin
def max_contiguous(input, values, _semantic=None):
    """
    Let the compiler know that the `value` first values in :code:`input` are contiguous.
    """
    values = _unwrap_hint_values(values)
    return _semantic.max_contiguous(input, values)


@builtin
def max_constancy(input, values, _semantic=None):
    """
    Let the compiler know that the `value` first values in :code:`input` are constant.

    e.g. if :code:`values` is [4], then each group of 4 values in :code:`input` should all be equal,
    for example [0, 0, 0, 0, 1, 1, 1, 1].
    """
    values = _unwrap_hint_values(values)
    return _semantic.max_constancy(input, values)


@builtin
def assume(cond, _semantic=None):
    """
    Allow compiler to assume the :code:`cond` is True.
    """
    return _semantic.assume(_semantic.to_tensor(cond))


# -----------------------------------------------------------------------------
# Debugging Functions
# -----------------------------------------------------------------------------


@builtin
def static_print(*values,
                 sep: str = " ",
                 end: str = "\n",
                 file=None,
                 flush=False,
                 _semantic=None):
    """
    Print the values at compile time.
    The parameters are the same as the builtin :code:`print`.

    Example:
        Calling the Python builtin :code:`print` is not the same as calling this,
        it instead maps to :code:`device_print`, which has special requirements for the arguments.

        ```python
        tl.static_print(f"BLOCK_SIZE={BLOCK_SIZE}")
        ```
    """
    ...


@builtin
def static_assert(cond, msg="", _semantic=None):
    '''
    Assert the condition at compile time.
    Does not require that the :code:`TRITON_DEBUG` environment variable is set.

    Example:
        ```python
        tl.static_assert(BLOCK_SIZE == 1024)
        ```
    '''
    ...


@builtin
def device_print(prefix, *args, hex=False, _semantic=None):
    """
    Print the values at runtime from the device.  String formatting does not work for runtime values, so you should
    provide the values you want to print as arguments.  The first value must be a string, all following values must
    be scalars or tensors.

    Calling the Python builtin :code:`print` is the same as calling this function, and the requirements
    for the arguments will match this function (not the normal requirements for :code:`print`).

    Example:
        ```python
        tl.device_print("pid", pid)
        print("pid", pid)
        ```

        On CUDA, printfs are streamed through a buffer of limited size (on one host,
        we measured the default as 6912 KiB, but this may not be consistent across
        GPUs and CUDA versions).  If you notice some printfs are being dropped, you
        can increase the buffer size by calling

        ```python
        triton.runtime.driver.active.utils.set_printf_fifo_size(size_bytes)
        ```

        CUDA may raise an error if you try to change this value after running a
        kernel that uses printfs.  The value set here may only affect the current
        device (so if you have multiple GPUs, you'd need to call it multiple times).

    Args:
        prefix: a prefix to print before the values. This is required to be a string literal.
        args: the values to print. They can be any tensor or scalar.
        hex: print all values as hex instead of decimal
    """
    import string
    prefix = _unwrap_if_constexpr(prefix)
    assert isinstance(prefix, str), f"{prefix} is not string"
    b_ascii = True
    for ch in prefix:
        if ch not in string.printable:
            b_ascii = False
            break
    assert b_ascii, f"{prefix} is not an ascii string"
    new_args = []
    for arg in args:
        new_args.append(_semantic.to_tensor(arg))
    return _semantic.device_print(prefix, new_args, hex)


@builtin
def device_assert(cond, msg="", mask=None, _semantic=None):
    """
    Assert the condition at runtime from the device.
    Requires that the environment variable :code:`TILEON_DEBUG`
    is set to a value besides :code:`0` in order for this to have any effect.

    Using the Python :code:`assert` statement is the same as calling this function, except that the second argument
    must be provided and must be a string, e.g. :code:`assert pid == 0, "pid != 0"`.  The environment variable must
    be set for this :code:`assert` statement to have any effect.

    Example:
        ```python
        tl.device_assert(pid == 0)
        assert pid == 0, f"pid != 0"
        ```

    Args:
        cond: the condition to assert. This is required to be a boolean tensor.
        msg: the message to print if the assertion fails. This is required to be a string literal.
    """
    msg = _unwrap_if_constexpr(msg)
    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    return _semantic.device_assert(_semantic.to_tensor(cond), msg, mask)


@builtin
def inline_asm_elementwise(asm: str,
                           constraints: str,
                           args: Sequence,
                           dtype: Union[dtype, Sequence[dtype]],
                           is_pure: bool,
                           pack: int,
                           _semantic=None):
    """Executes inline assembly element-wise over one or more input tensors.

    This function is functionally equivalent to a `map` operation where the applied
    function is inline assembly (e.g., NVIDIA PTX for CUDA GPUs). Input tensors are
    implicitly broadcasted to a common shape before processing.

    Key Behavior Notes:
        - Each invocation of the inline assembly processes `pack` elements at a time;
            the specific set of input elements assigned to a block is unspecified.
        - Input elements smaller than 4 bytes are packed into 4-byte registers for
            assembly execution.
        - The `dtype` parameter cannot be empty: the inline assembly must return at
            least one tensor (even if unused). A dummy tensor of arbitrary type can be
            returned as a workaround (no performance cost if unused).
        - `dtype` can be a tuple of types, in which case the output is a tuple of
            tensors matching the specified types.

    Examples:
        Execute NVIDIA PTX assembly to process uint8/float32 tensors (4 elements at a time):
        ```python
        @tileon.jit
        def kernel(A, B, C, D, BLOCK: tl.constexpr):
            a = tl.load(A + tl.arange(0, BLOCK))  # uint8 tensor
            b = tl.load(B + tl.arange(0, BLOCK))  # float32 tensor

            # For each (a,b) pair in (a_tensor, b_tensor):
            # 1. Convert `a` (uint8) to int32 (`ai`)
            # 2. Convert `ai` to float32
            # 3. Compute max of the float32 `ai` and `b`
            # 4. Return the int32 `ai` and max result (process 4 elements at once)
            (c, d) = tl.inline_asm_elementwise(
                asm=\"\"\"
                    {
                        // Unpack `a` into `ai` (4 uint8 elements → 4 int32 registers)
                        .reg .b8 tmp<4>;
                        mov.b32 {tmp0, tmp1, tmp2, tmp3}, $8;
                        cvt.u32.u8 $0, tmp0;
                        cvt.u32.u8 $1, tmp1;
                        cvt.u32.u8 $2, tmp2;
                        cvt.u32.u8 $3, tmp3;
                    }
                    // Convert int32 `ai` to float32
                    cvt.rn.f32.s32 $4, $0;
                    cvt.rn.f32.s32 $5, $1;
                    cvt.rn.f32.s32 $6, $2;
                    cvt.rn.f32.s32 $7, $3;
                    // Compute max of float32 `ai` and `b`
                    max.f32 $4, $4, $9;
                    max.f32 $5, $5, $10;
                    max.f32 $6, $6, $11;
                    max.f32 $7, $7, $12;
                    \"\"\",
                constraints=(
                    # 8 output registers:
                    #   $0=ai0, $1=ai1, $2=ai2, $3=ai3,
                    #   $4=m0,  $5=m1,  $6=m2,  $7=m3
                    \"=r,=r,=r,=r,=r,=r,=r,=r,\"
                    # 5 input registers:
                    #   $8=a (packed 4 uint8 elements),
                    #   $9=b0, $10=b1, $11=b2, $12=b3
                    \"r,r,r,r,r\"),
                args=[a, b],
                dtype=(tl.int32, tl.float32),
                is_pure=True,
                pack=4,
            )
            tl.store(C + tl.arange(0, BLOCK), c)
            tl.store(D + tl.arange(0, BLOCK), d)
        ```

    Args:
        asm: str
            Inline assembly code to execute. Must match the assembly format of the
            target hardware (e.g., PTX for NVIDIA GPUs).
        constraints: str
            Assembly constraints in LLVM format, defining input/output register
            mappings (see https://llvm.org/docs/LangRef.html#inline-asm-constraint-string).
        args: list[tensor]
            Input tensors whose values are passed to the assembly block. Tensors are
            implicitly broadcasted to the same shape before processing.
        dtype: type or tuple[type]
            Element type(s) of the returned tensor(s) (e.g., tl.int32, (tl.int32, tl.float32)).
            Cannot be empty (assembly must return at least one tensor).
        is_pure: bool
            If True, the compiler assumes the assembly block has no side effects (enables
            additional optimizations).
        pack: int
            Number of elements processed by a single invocation of the inline assembly
            (batch size for assembly execution).

    Returns:
        tensor or tuple[tensor]
            Single tensor (if `dtype` is a single type) or tuple of tensors (if `dtype`
            is a tuple) with the specified element types. The output shape matches the
        broadcasted shape of the input tensors.
    """
    asm = _unwrap_if_constexpr(asm)
    constraints = _unwrap_if_constexpr(constraints)
    pack = _unwrap_if_constexpr(pack)
    is_pure = _unwrap_if_constexpr(is_pure)

    try:
        iter(dtype)
        has_multiple_outputs = True
    except TypeError:
        has_multiple_outputs = False
        dtype = (dtype, )

    dtype = typing.cast(Sequence[_DtypeClass], dtype)

    res_tys = dtype
    if dispatch_args := [_semantic.to_tensor(arg) for arg in args]:
        bin_op_type_checking = partial(
            _semantic.binary_op_type_checking_impl,
            arithmetic_check=False,
            allow_lhs_ptr=True,
            allow_rhs_ptr=True,
        )
        broadcast_arg = dispatch_args[0]
        # Get the broadcast shape over all the arguments
        for item in dispatch_args:
            _, broadcast_arg = bin_op_type_checking(item, broadcast_arg)
        if broadcast_arg.shape:
            # Change the shape of each argument based on the broadcast shape
            for i, item in enumerate(dispatch_args):
                dispatch_args[i], _ = bin_op_type_checking(item, broadcast_arg)
            res_tys = [broadcast_arg.type.with_element_t(dt) for dt in dtype]
    handles = [t.handle for t in dispatch_args]
    builder = _semantic.builder
    call = builder.create_inline_asm(asm, constraints, handles,
                                     [ty.to_ir(builder) for ty in res_tys],
                                     is_pure, pack)

    if not has_multiple_outputs:
        return tensor(call.get_result(0), res_tys[0])
    return tuple(
        tensor(call.get_result(i), ty) for i, ty in enumerate(res_tys))


# -----------------------------------------------------------------------------
#  Special Iterators
# -----------------------------------------------------------------------------


class static_range(_value):
    """
    Iterator that counts upward forever.

    Example:
        ```python
        @tileon.jit
        def kernel(...):
            for i in tl.static_range(10):
                ...
        ```

    Note: This is a special iterator used to implement similar semantics to Python's :code:`range`
        in the context of :code:`tileon.jit` functions. In addition, it also guides the compiler to
        unroll the loop aggressively.

    Args:
        arg1: the start value.
        arg2: the end value.
        step: the step value.
    """

    def __init__(self,
                 arg1: constexpr,
                 arg2: Optional[constexpr] = None,
                 step: Optional[constexpr] = None):
        assert isinstance(
            arg1, constexpr
        ), f"{arg1} used as tl.static_range start value is not a constexpr"
        if step is None:
            self.step = CONSTEXPR_1
        else:
            assert isinstance(
                step, constexpr
            ), f"{step} used as tl.static_range step value is not a constexpr"
            self.step = step
        if arg2 is None:
            self.start = CONSTEXPR_0
            self.end = arg1
        else:
            assert isinstance(
                arg2, constexpr
            ), f"{arg2} used as tl.static_range end value is not a constexpr"
            self.start = arg1
            self.end = arg2

    def __iter__(self):
        raise RuntimeError(
            "static_range can only be used in @tileon.jit'd functions")

    def __next__(self):
        raise RuntimeError(
            "static_range can only be used in @tileon.jit'd functions")


class range(_value):
    """
    Iterator that counts upward forever.

    Example:
        ```python
        @tileon.jit
        def kernel(...):
            for i in tl.range(10, num_stages=3):
                ...
        ```

    Note:
        This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
        :code:`tileon.jit` functions. In addition, it allows user to pass extra attributes to the compiler.

    Args:
        arg1: the start value.
        arg2: the end value.
        step: the step value.
        num_stages: pipeline the loop into this many stages (so there are :code:`num_stages` iterations of
            the loop in flight at once).

            Note this is subtly different than passing :code:`num_stages` as a
            kernel argument.  The kernel argument only pipelines loads that feed
            into :code:`dot` operations, while this attribute tries to pipeline most
            (though not all) loads in this loop.
        loop_unroll_factor: Tells the Triton IR level loop unroller how many
            times to unroll a for loop that this range is used with. Less than 2 for
            this value implies no unrolling.
        disallow_acc_multi_buffer: If true, prevent the accumulator of the dot operation in the
            loop to be multi-buffered, if applicable.
        flatten: automatically flatten the loop nest starting at this loop to
            create a single flattened loop. The compiler will try to pipeline the
            flattened loop which can avoid stage stalling.
        warp_specialize: Enable automatic warp specialization on the loop.
            The compiler will attempt to partition memory, MMA, and vector
            operations in the loop into separate async partitions. This will
            increase the total number of warps required by the kernel.
        disable_licm: Tells the compiler it shouldn't hoist loop invariant
            code outside the loop. This is often useful to avoid creating long liveranges
            within a loop.

            Note that warp specialization is only supported on Blackwell GPUs and
            only works on simple matmul loops. Support for arbitrary loops will be
            expanded over time.
    """

    def __init__(
        self,
        arg1: constexpr,
        arg2: Optional[constexpr] = None,
        step: Optional[constexpr] = None,
        num_stages=None,
        loop_unroll_factor=None,
        disallow_acc_multi_buffer: bool = False,
        flatten: bool = False,
        warp_specialize: bool = False,
        disable_licm: bool = False,
    ):
        if step is None:
            self.step = CONSTEXPR_1
        else:
            self.step = step
        if arg2 is None:
            self.start = CONSTEXPR_0
            self.end = arg1
        else:
            self.start = arg1
            self.end = arg2
        self.num_stages = num_stages
        self.loop_unroll_factor = loop_unroll_factor
        self.disallow_acc_multi_buffer = disallow_acc_multi_buffer
        self.flatten = flatten
        self.warp_specialize = warp_specialize
        self.disable_licm = disable_licm

    def __iter__(self):
        raise RuntimeError(
            "tl.range can only be used in @triton.jit'd functions")

    def __next__(self):
        raise RuntimeError(
            "tl.range can only be used in @triton.jit'd functions")


class condition(_value):
    """
    While loop condition wrapper.

    Example:
        ```python
        @tileon.jit
        def kernel(...):
            while tl.condition(c, disable_licm)
                ...
        ```

    Note:
        This is a special wrapper used to annotate while loops in the context of
        :code:`tileon.jit` functions. It allows user to pass extra attributes to the compiler.

    Args:
        arg1: the condition value.
        disable_licm: Tells the compiler it shouldn't hoist loop invariant
            code outside the loop. This is often useful to avoid creating long liveranges within a loop.
    """

    def __init__(self, arg1, disable_licm: bool = False):
        self.condition = arg1
        self.disable_licm = disable_licm


# -----------------------------------------------------------------------------
# Extern functions
# -----------------------------------------------------------------------------


def dispatch(func: Callable, lib_name: str, lib_path: str, args: list,
             arg_type_symbol_dict: dict, ret_type: dtype, is_pure: bool,
             _semantic):
    """
    Dispatch a function to a library.

    Args:
        func: the function to dispatch
        lib_name: the name of the library
        lib_path: the path of the library
        args: the arguments of the function
        arg_type_symbol_dict: the type of the arguments
        ret_type: the type of the return value
        is_pure: whether the function is pure
        _semantic: the semantic of the function
    """
    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(f"length of input args does not match."
                         f"Expect {len(args)}, got {num_args}")

    arg_types = []
    arg_list = []
    for arg in args:
        if isinstance(arg, tensor):
            arg_types.append(arg.dtype)
            arg_list.append(arg.handle)
        else:
            arg_types.append(type(arg))
            arg_list.append(arg)
    arg_types = tuple(arg_types)

    if arg_types not in arg_type_symbol_dict:
        raise ValueError(
            f"input arg type does not match."
            f"Expect one of {arg_type_symbol_dict.keys()}, got {arg_types}")
    else:
        symbol = arg_type_symbol_dict[arg_types][0]
        builder = _semantic.builder
        return tensor(
            func(lib_name, lib_path, symbol, arg_list, ret_type.to_ir(builder),
                 is_pure), ret_type)


@builtin
def extern_elementwise(lib_name: str,
                       lib_path: str,
                       args: list,
                       arg_type_symbol_dict: dict,
                       is_pure: bool,
                       _semantic=None):
    """
    Dispatch an elementwise function to a library.

    Args:
        lib_name: the name of the library
        lib_path: the path of the library
        args: the arguments of the function
        arg_type_symbol_dict: the type of the arguments
        is_pure: whether the function is pure
    Returns:
        the return value of the function
    """
    dispatch_args = args.copy()
    all_scalar = True
    arg_types = []
    for i in builtins.range(len(dispatch_args)):
        dispatch_args[i] = _semantic.to_tensor(dispatch_args[i])
        arg_types.append(dispatch_args[i].dtype)
        if dispatch_args[i].type.is_block():
            all_scalar = False

    arg_types = tuple(arg_types)
    ret_type = arg_type_symbol_dict[arg_types][1]
    if len(arg_types) > 0:
        arithmetic_check = True
        # If there's a type tuple that is not supported by the library, we will do arithmetic check
        if arg_types in arg_type_symbol_dict:
            arithmetic_check = False
        broadcast_arg = dispatch_args[0]
        # Get the broadcast shape over all the arguments
        for item in dispatch_args:
            _, broadcast_arg = _semantic.binary_op_type_checking_impl(
                item, broadcast_arg, arithmetic_check=arithmetic_check)
        # Change the shape of each argument based on the broadcast shape
        for i in builtins.range(len(dispatch_args)):
            dispatch_args[i], _ = _semantic.binary_op_type_checking_impl(
                dispatch_args[i],
                broadcast_arg,
                arithmetic_check=arithmetic_check)
        if not all_scalar:
            ret_type = broadcast_arg.type.with_element_t(ret_type)
    func = _semantic.builder.create_extern_elementwise
    return dispatch(func, lib_name, lib_path, dispatch_args,
                    arg_type_symbol_dict, ret_type, is_pure, _semantic)


def binary_op_type_legalization(lhs, rhs, _semantic):
    """
    Convert both operands to a single common type.

    Args:
        lhs: the left operand
        rhs: the right operand
        _semantic: the semantic of the function
    Returns:
        the return value of the function
    """
    return _semantic.binary_op_type_checking_impl(lhs, rhs)


def extern(fn: Callable):
    """
    A decorator for external functions.

    Args:
        fn: the function to decorate
    Returns:
        the decorated function
    """
    return builtin(fn)


_NOTHING = object()


def is_negative_zero(x):
    """
    Check if the value is negative zero.

    Args:
        x: the value to check
    Returns:
        True if the value is negative zero, False otherwise
    """
    return x == 0.0 and math.copysign(1.0, x) < 0


@builtin
def builtin_max(*args, propagate_nan=_NOTHING, _semantic=None):
    """
    Get the maximum value of the arguments.

    Args:
        args: the arguments to get the maximum value
        propagate_nan: whether to propagate NaN values
    Returns:
        the maximum value of the arguments
    """
    args = _unwrap_if_constexpr(args)
    is_constexpr = all(not isinstance(x, _value) for x in args)
    if is_constexpr:
        assert propagate_nan is _NOTHING, "propagate_nan is not supported on `builtin_max`"
        assert not any(math.isnan(x) for x in args)
        assert not any(is_negative_zero(x) for x in args)
        return constexpr(builtins.max(_unwrap_if_constexpr(args)))

    if propagate_nan is _NOTHING:
        propagate_nan = PropagateNan.NONE
    else:
        warnings.warn(
            "passing propagate_nan to `builtin_max` is deprecated, use `tl.minimum` instead"
        )

    assert len(args) >= 2, "min requires at least 2 values"
    max_val = args[0]
    for arg in args[1:]:
        max_val = maximum(max_val,
                          arg,
                          propagate_nan=propagate_nan,
                          _semantic=_semantic)
    if max_val.type.is_block():
        warnings.warn(
            "`builtin_max` on non-scalar tensor values is deprecated, use `tl.maximum` instead"
        )
    return max_val


@builtin
def builtin_min(*args, propagate_nan=_NOTHING, _semantic=None):
    """
    Get the minimum value of the arguments.

    Args:
        args: the arguments to get the minimum value
        propagate_nan: whether to propagate NaN values
    Returns:
        the minimum value of the arguments
    """
    args = _unwrap_if_constexpr(args)
    is_constexpr = all(not isinstance(x, _value) for x in args)
    if is_constexpr:
        assert propagate_nan is _NOTHING, "propagate_nan is not supported on `builtin_min`"
        assert not any(math.isnan(x) for x in args)
        assert not any(is_negative_zero(x) for x in args)
        return constexpr(builtins.min(_unwrap_if_constexpr(args)))

    if propagate_nan is _NOTHING:
        propagate_nan = PropagateNan.NONE
    else:
        warnings.warn(
            "passing propagate_nan to `builtin_min` is deprecated, use `tl.minimum` instead"
        )

    assert len(args) >= 2, "min requires at least 2 values"
    min_val = args[0]
    for arg in args[1:]:
        min_val = minimum(min_val,
                          arg,
                          propagate_nan=propagate_nan,
                          _semantic=_semantic)
    if min_val.type.is_block():
        warnings.warn(
            "`builtin_min` on non-scalar tensor values is deprecated, use `tl.minimum` instead"
        )
    return min_val
