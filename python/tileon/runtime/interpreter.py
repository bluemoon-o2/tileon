from __future__ import annotations
import ast
import math
import textwrap
import inspect
import threading
from functools import partial
from typing import Tuple, List, Dict, Callable, TypeVar, Optional, Any

import numpy as np

import tileon
import tileon.knobs as knobs
import tileon.language as tl
import dataclasses
from dataclasses import dataclass

from tileon.language.semantic import TileonSemantic
from tileon.runtime.jit import JITCallable, KernelInterface, TileonTensor
from .errors import InterpreterError
from .._C import interpreter as _interpreter
from .._C import ir as _ir
from .._utils import tuple_create, _normalize_t, validate_block_shape, canonicalize_dtype

T = TypeVar("T", bound=Callable)


def _validate_np_data_size(np_array: np.ndarray, tl_dtype: tl.dtype):
    """
    Validate if the numpy array data size is compatible with the tileon dtype.
    """
    if isinstance(tl_dtype, tl.pointer_t):
        return True

    np_dtype_bitwidth = np_array.itemsize * 8
    tl_dtype_bitwidth = tl_dtype.primitive_bitwidth

    # numpy lowest itemsize is at least 8 bits
    if tl_dtype_bitwidth < 8:
        tl_dtype_bitwidth = 8

    if np_dtype_bitwidth > tl_dtype_bitwidth:
        return False
    return True


@dataclass
class TensorHandle:
    """
    A handle for tensor (numpy array).

    Attributes:
        data: numpy array
        dtype: tileon type, either pointer_type or scalar_type.
        attr: a dictionary of attributes
    """
    data: np.ndarray
    dtype: tl.dtype
    attr: Dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not _validate_np_data_size(self.data, self.dtype):
            raise ValueError(
                f"numpy data itemsize ({self.data.itemsize * 8} bits) exceeds dtype primitive_bitwidth "
                f"({self.dtype.primitive_bitwidth} bits) for tileon type {self.dtype}"
            )

    def __bool__(self):
        return bool(self.data.all())

    @property
    def element_t(self):
        """
        Get the element type of the tensor.
        """
        dtype = self.dtype
        while hasattr(dtype, "element_t"):
            dtype = dtype.element_t
        return dtype

    def clone(self):
        """Clone the tensor handle."""
        return TensorHandle(self.data.copy(), self.dtype)

    def set_attr(self, key, value):
        """Set the attribute of the tensor."""
        self.attr[key] = value


class BlockPointerHandle:
    """A handle for block pointer tensor."""

    def __init__(
        self, 
        base: TensorHandle,
        shape: List[TensorHandle],
        strides: List[TensorHandle],
        offsets: List[TensorHandle],
        block_shape: List[int],
        order: List[int]
    ):
        self.base = base
        self.shape = shape
        self.strides = strides
        self.offsets = offsets
        self.block_shape = block_shape
        self.order = order

    def materialize_pointers(self, boundary_check: List[int]):
        """
        Materialize the block pointer tensor.

        Args:
            boundary_check: a list of dimensions to check boundary.

        Returns:
            A tuple of (pointer tensor handle, mask tensor handle).
        """
        n_bytes = self.base.element_t.primitive_bitwidth // 8
        ptrs_data = np.broadcast_to(self.base.data, self.block_shape)
        masks = np.ones(self.block_shape, dtype=bool)
        for dim in range(len(self.block_shape)):
            bcast_dims = [1] * len(self.block_shape)
            bcast_dims[dim] = self.block_shape[dim]
            off = (self.offsets[dim].data + np.arange(self.block_shape[dim])).reshape(bcast_dims)
            ptrs_data = ptrs_data + (n_bytes * off * self.strides[dim].data).astype(np.uint64)
            if dim in boundary_check:
                masks = masks & (off < self.shape[dim].data) & (off >= 0)
        ptrs_handle = TensorHandle(ptrs_data, self.base.dtype.scalar)
        return ptrs_handle, masks


class TensorDescHandle:
    """A handle for tensor descriptor tensor."""

    def __init__(
        self,
        base: TensorHandle,
        shape: List[TensorHandle],
        strides: List[TensorHandle],
        block_shape: List[int],
        padding: List[Tuple[int, int]]
    ):
        self.base = base
        self.ndim = len(shape)
        self.shape = shape
        self.strides = strides
        self.block_shape = block_shape
        self.padding = padding

    def validate(self):
        """Validate the tensor descriptor."""
        assert self.base.data.item() % 16 == 0, "base must be 16-byte aligned"
        assert len(self.strides) == self.ndim
        assert len(self.block_shape) == self.ndim
        assert self.ndim >= 1, "descriptor cannot be 0 dimensional"

        itemsize = self.base.dtype.element_t.primitive_bitwidth // 8
        for stride in self.strides[:-1]:
            byte_stride = stride.data.item() * itemsize
            assert byte_stride % 16 == 0, "stride must be 16-byte aligned"
        assert self.strides[-1].data.item() == 1, "last dim must be contiguous"

    def materialize_pointers(self, offsets: List[TensorHandle]):
        """
        Materialize the block pointer tensor.

        Args:
            offsets: a list of offset tensors.

        Returns:
            A tuple of (pointer tensor handle, mask tensor handle).
        """
        assert len(offsets) == self.ndim, "number of offsets must match number of dimensions"
        itemsize = self.base.dtype.element_t.primitive_bitwidth // 8
        assert (offsets[-1].data * itemsize ) % 16 == 0, "block offset start must be 16-byte aligned"
        ptrs_data = np.broadcast_to(self.base.data, self.block_shape)
        masks = np.ones(self.block_shape, dtype=bool)
        for dim in range(len(self.block_shape)):
            bcast_dims = [1] * len(self.block_shape)
            bcast_dims[dim] = self.block_shape[dim]
            off = (offsets[dim].data + np.arange(self.block_shape[dim])).reshape(bcast_dims)
            ptrs_data = ptrs_data + (itemsize * off * self.strides[dim].data).astype(np.uint64)
            masks = masks & (0 <= off) & (off < self.shape[dim].data)
        assert ptrs_data.dtype == np.uint64, "pointer tensor must be uint64"
        ptrs_handle = TensorHandle(ptrs_data, self.base.dtype.scalar)
        return ptrs_handle, masks


@dataclass
class TensorDescriptor:
    """
    A descriptor for a tensor.

    Args:
        base: The base tensor.
        shape: The shape of the tensor.
        strides: The strides of the tensor.
        block_shape: The block shape of the tensor.
        padding: The padding option for the tensor. Defaults to "zero".
        round_f32_to_tf32: Whether to round float32 to tf32. Defaults to False.
    """
    base: Any
    shape: List[int]
    strides: List[int]
    block_shape: List[int]
    padding: str = "zero"
    round_f32_to_tf32: bool = False

    def __post_init__(self):
        rank = len(self.shape)
        assert len(self.strides) == rank, f"rank mismatch: {self}"
        assert len(self.block_shape) == rank, f"rank mismatch: {self}"
        assert rank > 0, "rank must not be zero"
        assert rank <= 5, "rank cannot be more than 5"
        ty = type(self.base)
        if ty.__name__ not in ("FakeTensor", "FunctionalTensor"):
            assert self.base.data_ptr() % 16 == 0, "base must be 16-byte aligned"
        validate_block_shape(self.block_shape)
        elem_bytes = self.base.dtype.itemsize
        for stride in self.strides[:-1]:
            assert (stride * elem_bytes) % 16 == 0, "strides must be 16-byte aligned"
        for shape_dim in self.shape:
            assert shape_dim > 0, "shape must be positive"
        assert self.strides[-1] == 1, "Last dimension must be contiguous"
        assert self.padding == "zero" or self.padding == "nan", "Illegal value for padding"
        if self.padding == "nan":
            assert self.base.dtype.is_floating_point, "Padding option `nan` is only supported for floating point tensors"
        if self.round_f32_to_tf32:
            dtype_name = canonicalize_dtype(self.base.dtype)
            assert dtype_name == "fp32", "round_f32_to_tf32 is only supported for float32 tensors"

    @staticmethod
    def from_tensor(tensor: Any, block_shape: List[int], padding="zero", round_f32_to_tf32=False):
        return TensorDescriptor(tensor, tensor.shape, tensor.stride(), block_shape, padding, round_f32_to_tf32)


@dataclass(frozen=True)
class InterpreterOptions:
    """Options for the interpreter."""

    backend_name: str = "interpreter"
    extern_libs: Optional[dict] = None
    debug: bool = False
    sanitize_overflow: bool = True
    arch: Optional[str] = None
    supported_fp8_dtypes: Tuple[str, ...] = ("fp8e5", "fp8e5b16", "fp8e4nv", "fp8e4b8", "fp8e4b15")
    deprecated_fp8_dot_operand_dtypes: Tuple[str, ...] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str, ...] = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default: int = 0


def _get_signed_np_dtype(dtype: np.dtype):
    """Get the signed numpy dtype for the unsigned numpy dtype."""
    if dtype == np.uint8:
        return np.int8
    if dtype == np.uint16:
        return np.int16
    if dtype == np.uint32:
        return np.int32
    if dtype == np.uint64:
        return np.int64
    return dtype


def _get_np_dtype(dtype: tl.dtype):
    """Get the numpy dtype for the tileon dtype."""
    if isinstance(dtype, tl.pointer_t):
        return np.dtype(np.uint64)
    np_types = {
        tl.int1: np.dtype(bool),
        tl.float16: np.dtype(np.float16),
        tl.float32: np.dtype(np.float32),
        tl.float64: np.dtype(np.float64),
        tl.int8: np.dtype(np.int8),
        tl.uint8: np.dtype(np.uint8),
        tl.int16: np.dtype(np.int16),
        tl.uint16: np.dtype(np.uint16),
        tl.int32: np.dtype(np.int32),
        tl.uint32: np.dtype(np.uint32),
        tl.int64: np.dtype(np.int64),
        tl.uint64: np.dtype(np.uint64),
        # bfloat16 types are stored as uint16
        tl.bfloat16: np.dtype(np.uint16),
        # float8 types are stored as uint8
        tl.float8e5: np.dtype(np.uint8),
        tl.float8e5b16: np.dtype(np.uint8),
        tl.float8e4nv: np.dtype(np.uint8),
        tl.float8e4b8: np.dtype(np.uint8),
        tl.float8e4b15: np.dtype(np.uint8),
    }
    if isinstance(dtype, tl.block_t):
        if isinstance(dtype.element_t, tl.pointer_t):
            return np.dtype(np.uint64)
        return np_types[dtype.element_t]
    return np_types[dtype]


def _convert_float(input, input_dtype: tl.dtype, output_dtype: tl.dtype, rounding_mode):
    """
    Convert float input to float output with specified rounding mode.

    Args:
        input: the input tensor.
        input_dtype: the dtype of the input tensor.
        output_dtype: the dtype of the output tensor.
        rounding_mode: the rounding mode.

    Returns:
        The output tensor.
    """
    input_uint_dtype = getattr(np, f"uint{input_dtype.primitive_bitwidth}")
    output_unint_dtype = getattr(np, f"uint{output_dtype.primitive_bitwidth}")
    input_bin = np.frombuffer(input.tobytes(), dtype=input_uint_dtype)
    sign = (input_bin >> (input_dtype.primitive_bitwidth - 1)) & 0x01
    input_exponent_width = input_dtype.primitive_bitwidth - input_dtype.fp_mantissa_width - 1
    output_exponent_width = output_dtype.primitive_bitwidth - output_dtype.fp_mantissa_width - 1
    significand = input_bin & ((1 << input_dtype.fp_mantissa_width) - 1)
    bias_input = input_dtype.exponent_bias
    bias_output = output_dtype.exponent_bias
    exponent = ((input_bin >> input_dtype.fp_mantissa_width) & ((1 << input_exponent_width) - 1)).astype(np.int32)
    subnormal_index = exponent == 0
    if np.any(subnormal_index):
        # subnormal repr: ((-1.0)**sign) * (2.0**(1 - exp_bias)) * (2^(m0) + 2^(m1) + ... + 2^(mn))
        # where m0, m1, ..., mn are the 1-bit of the mantissa
        # convert it to normal repr: ((-1.0)**sign) * (2.0**(1 + m0 - exp_bias)) * (1 + 2^(m1 - m0) + ... + 2^(mn - m0))
        bit_pos = np.zeros_like(input_bin, dtype=np.int32)
        # Find the most significant bit of the mantissa in the significand
        for i in range(input_dtype.fp_mantissa_width):
            bit_index = ((significand >> i) & 0x01)
            # pos should be >= 1
            bit_pos[bit_index == 1] = input_dtype.fp_mantissa_width - i
        zero_significand_index = significand == 0
        exponent[subnormal_index] = 1 - bit_pos[subnormal_index]
        # 0 significand and subnormal should be treated as 0
        exponent[zero_significand_index & subnormal_index] = bias_input - bias_output
        significand[subnormal_index] = (
            (significand[subnormal_index] << bit_pos[subnormal_index]) & ((1 << input_dtype.fp_mantissa_width) - 1)
        )
    # Prevent overflow and underflow
    exponent_output = np.maximum(0, np.minimum((exponent - bias_input + bias_output), (1 << output_exponent_width) - 1))
    exponent_output = exponent_output.astype(output_unint_dtype)
    sign_output = sign.astype(output_unint_dtype)
    if input_dtype.primitive_bitwidth > output_dtype.primitive_bitwidth:  # Downcast
        shift = input_dtype.fp_mantissa_width - output_dtype.fp_mantissa_width
        significand_output = (
            (significand >> shift) & ((1 << output_dtype.fp_mantissa_width) - 1)
        )
        if rounding_mode == _ir.ROUNDING_MODE.RTNE:  # Round to nearst even
            cutoff_mask = (1 << (shift - 1))
            sticky_mask = cutoff_mask - 1
            cutoff = (significand & cutoff_mask) > 0
            sticky = (significand & sticky_mask) > 0
            lsb = (significand_output & 1) > 0
            round_up = cutoff & (sticky | lsb)
            
            # Apply rounding only to normal results (exponent_output > 0)
            # Subnormals are handled in the subnormal block with higher precision
            normal_mask = exponent_output > 0
            significand_output = significand_output + (round_up & normal_mask)

            # Handle overflow
            overflow = significand_output >= (1 << output_dtype.fp_mantissa_width)
            if np.any(overflow):
                significand_output[overflow] = 0
                exponent_output[overflow] += 1
                # Cap at max exponent to avoid wrapping into sign bit (though it implies Inf)
                max_exp = (1 << output_exponent_width) - 1
                exponent_output[overflow] = np.minimum(exponent_output[overflow], max_exp)
                
        significand_output = significand_output.astype(output_unint_dtype)
    else:  # Upcast
        significand_output = (
            (significand.astype(output_unint_dtype) << (output_dtype.fp_mantissa_width - input_dtype.fp_mantissa_width)) & ((1 << output_dtype.fp_mantissa_width) - 1)
        )
    subnormal_index = exponent_output == 0
    if np.any(subnormal_index):  # underflow
        # normal repr: ((-1.0)**sign) * (2.0**(exp - exp_bias_input)) * (1 + 2^(m0) + 2^(m1) + ... + 2^(mn))
        # where m0, m1, ..., mn are the 1-bit of the mantissa
        # shift = (1 - exp_bias_output) - (exp - exp_bias_input)
        # convert it to subnormal repr: ((-1.0)**sign) * (2.0**(1 - exp_bias_output)) * (2^(-shift) + 2^(m0 - shift) + 2^(m1 - shift) + ... + 2^(mn - shift))
        exponent = ((input_bin >> input_dtype.fp_mantissa_width) & ((1 << input_exponent_width) - 1)).astype(np.int32)
        non_zero_exponent_index = exponent != 0
        # If the original exponent is not zero, we still need to shift the significand and consider the 1.0 part in mantissa
        subnormal_index = subnormal_index & non_zero_exponent_index
        
        # Calculate full precision mantissa for subnormal case
        shift = np.zeros_like(input_bin, dtype=np.int32)
        shift[subnormal_index] = (1 - bias_output) - (exponent[subnormal_index] - bias_input)
        
        # Calculate shift amount relative to input significand position
        shift_amt = (input_dtype.fp_mantissa_width - output_dtype.fp_mantissa_width) + shift[subnormal_index]
        
        # Reconstruct full mantissa (implicit bit + significand bits)
        # implicit bit is at bit 23.
        full_mant = (1 << input_dtype.fp_mantissa_width) | significand[subnormal_index]
        
        # Apply shift
        # shift_amt is at least 13 + 1 = 14.
        res_mant = full_mant >> shift_amt
        
        # Apply rounding (RTNE)
        if rounding_mode == _ir.ROUNDING_MODE.RTNE:
            cutoff_mask = (1 << (shift_amt - 1))
            sticky_mask = cutoff_mask - 1
            cutoff = (full_mant & cutoff_mask) > 0
            sticky = (full_mant & sticky_mask) > 0
            lsb = (res_mant & 1) > 0
            round_up = cutoff & (sticky | lsb)
            res_mant = res_mant + round_up
            
            # Check if rounding caused overflow to normal range
            # If res_mant becomes (1 << output_dtype.fp_mantissa_width), it is normal!
            # It means exp becomes 1, and mant becomes 0.
            # Since exponent_output is 0, the output construction:
            # (exponent_output << mant_width) | significand_output
            # will be 0 | (1 << mant_width), which correctly represents normal number with exp=1, mant=0.
            # So no explicit exponent update is needed.

        significand_output[subnormal_index] = res_mant
    output = (sign_output << (output_dtype.primitive_bitwidth - 1)) | (exponent_output << output_dtype.fp_mantissa_width) | significand_output
    return output.reshape(input.shape)


def _convert_custom_types(
    input, 
    dst_t, 
    fp_downcast_rounding, 
    _semantic
):
    """Convert custom types to fp types."""
    return tl.tensor(
        _semantic.builder.create_fp_to_fp(input.handle, dst_t, fp_downcast_rounding),
        dst_t
    )


def _erf(x):
    """
    Calculate the error function of x.
    """
    return math.erf(x)


def _umulhi_64(a, b):
    """
    Calculate the high 64 bits of the 128-bit product of a and b.
    """
    return (int(a) * int(b)) >> 64


np_erf_fp32 = np.vectorize(_erf, otypes=[np.float32])
np_erf_fp64 = np.vectorize(_erf, otypes=[np.float64])
np_umulhi_u64 = np.vectorize(_umulhi_64, otypes=[np.uint64])


class InterpreterBuilder:
    ir_sem_to_interpreter_sem = {
        _ir.MEM_SEMANTIC.ACQUIRE:
        _interpreter.MEM_SEMANTIC.ACQUIRE,
        _ir.MEM_SEMANTIC.RELEASE:
        _interpreter.MEM_SEMANTIC.RELEASE,
        _ir.MEM_SEMANTIC.RELAXED:
        _interpreter.MEM_SEMANTIC.RELAXED,
        _ir.MEM_SEMANTIC.ACQUIRE_RELEASE:
        _interpreter.MEM_SEMANTIC.ACQUIRE_RELEASE,
    }

    ir_rmw_op_to_interpreter_rmw_op = {
        _ir.ATOMIC_OP.ADD: _interpreter.RMW_OP.ADD,
        _ir.ATOMIC_OP.FADD: _interpreter.RMW_OP.FADD,
        _ir.ATOMIC_OP.MIN: _interpreter.RMW_OP.MIN,
        _ir.ATOMIC_OP.UMIN: _interpreter.RMW_OP.UMIN,
        _ir.ATOMIC_OP.MAX: _interpreter.RMW_OP.MAX,
        _ir.ATOMIC_OP.UMAX: _interpreter.RMW_OP.UMAX,
        _ir.ATOMIC_OP.AND: _interpreter.RMW_OP.AND,
        _ir.ATOMIC_OP.OR: _interpreter.RMW_OP.OR,
        _ir.ATOMIC_OP.XOR: _interpreter.RMW_OP.XOR,
        _ir.ATOMIC_OP.XCHG: _interpreter.RMW_OP.XCHG,
    }

    def __init__(self) -> None:
        self.arch = None
        self.options = InterpreterOptions()
        self.codegen_fns = {}
        self.codegen_fns["convert_custom_types"] = _convert_custom_types
        self.codegen_fns["min_dot_size"] = lambda lhsType, rhsType: (1, 1, 1)
        self._local = threading.local()
        self._grid_dim = (1, 1, 1)
        self._vectorized_nx = None

    def set_vectorized_grid(self, nx):
        self._vectorized_nx = nx
        
    def clear_vectorized_grid(self):
        self._vectorized_nx = None

    @property
    def grid_idx(self):
        return self._local.grid_idx

    @grid_idx.setter
    def grid_idx(self, value):
        self._local.grid_idx = value

    @property
    def grid_dim(self):
        return self._grid_dim

    @grid_dim.setter
    def grid_dim(self, value):
        self._grid_dim = value

    def set_grid_idx(self, x: int, y: int, z: int):
        if not x < self.grid_dim[0]:
            raise ValueError("x >= grid_dim[0]")
        if not y < self.grid_dim[1]:
            raise ValueError("y >= grid_dim[1]")
        if not z < self.grid_dim[2]:
            raise ValueError("z >= grid_dim[2]")
        self.grid_idx = (x, y, z)

    def set_grid_dim(self, nx: int, ny: int, nz: int):
        self.grid_dim = (nx, ny, nz)

    def get_half_t(self):
        return tl.float16

    def get_bf16_t(self):
        return tl.bfloat16

    def get_float_t(self):
        return tl.float32

    def get_double_t(self):
        return tl.float64

    def get_int1_t(self):
        return tl.int1

    def get_int8_t(self):
        return tl.int8

    def get_uint8_t(self):
        return tl.uint8

    def get_int16_t(self):
        return tl.int16

    def get_uint16_t(self):
        return tl.uint16

    def get_int32_t(self):
        return tl.int32

    def get_uint32_t(self):
        return tl.uint32

    def get_int64_t(self):
        return tl.int64

    def get_uint64_t(self):
        return tl.uint64

    def get_fp8e4nv_t(self):
        return tl.float8e4nv

    def get_fp8e4b15_t(self):
        return tl.float8e4b15

    def get_fp8e4b8_t(self):
        return tl.float8e4b8

    def get_fp8e5_t(self):
        return tl.float8e5

    def get_fp8e5b16_t(self):
        return tl.float8e5b16

    def get_ptr_t(self, element_t, address_space):
        return tl.pointer_t(element_t, address_space)

    def get_block_t(self, dtype, shape):
        return tl.block_t(dtype, shape)

    def get_int1(self, value):
        return TensorHandle(np.array([value], dtype=np.bool_), tl.int1)

    def get_uint8(self, value):
        return TensorHandle(np.array([value], dtype=np.uint8), tl.uint8)

    def get_int8(self, value):
        return TensorHandle(np.array([value], dtype=np.int8), tl.int8)

    def get_uint16(self, value):
        return TensorHandle(np.array([value], dtype=np.uint16), tl.uint16)

    def get_int16(self, value):
        return TensorHandle(np.array([value], dtype=np.int16), tl.int16)

    def get_uint32(self, value):
        return TensorHandle(np.array([value], dtype=np.uint32), tl.uint32)

    def get_int32(self, value):
        return TensorHandle(np.array([value], dtype=np.int32), tl.int32)

    def get_uint64(self, value):
        return TensorHandle(np.array([value], dtype=np.uint64), tl.uint64)

    def get_int64(self, value):
        return TensorHandle(np.array([value], dtype=np.int64), tl.int64)

    def get_fp16(self, value):
        return TensorHandle(np.array([value], dtype=np.float16), tl.float16)

    def get_fp32(self, value):
        return TensorHandle(np.array([value], dtype=np.float32), tl.float32)

    def get_fp64(self, value):
        return TensorHandle(np.array([value], dtype=np.float64), tl.float64)

    def get_null_value(self, type):
        return TensorHandle(np.array([0], dtype=_get_np_dtype(type)), type)

    def create_get_program_id(self, axis):
        """
        Tileon's get_program_id instruction returns the ID of the current program in a given axis.

        Args:
            axis (int): The axis to get the program ID for.

        Returns:
            TensorHandle: The ID of the current program in the given axis.
        """
        if self._vectorized_nx is not None:
            if axis == 0:
                # Return (N, 1) array for broadcasting
                return TensorHandle(np.arange(self._vectorized_nx, dtype=np.int32)[:, None], tl.int32)
            else:
                return TensorHandle(np.array([0], dtype=np.int32), tl.int32)

        if self.grid_idx is None:
            raise ValueError("grid_idx is None")
        return TensorHandle(np.array([self.grid_idx[axis]], dtype=np.int32), tl.int32)

    def create_get_num_programs(self, axis):
        """
        Tileon's get_num_programs instruction returns the number of programs in a given axis.

        Args:
            axis (int): The axis to get the number of programs for.

        Returns:
            TensorHandle: The number of programs in the given axis.
        """
        return TensorHandle(np.array([self.grid_dim[axis]], dtype=np.int32), tl.int32)

    def create_masked_load(self, ptrs, mask, other, cache_modifier, eviction_policy, is_volatile):
        """
        Tileon's masked load instruction loads data from memory into a tensor,
        conditionally based on a mask.

        Args:
            ptrs (TensorHandle): The tensor of pointers to load from.
            mask (TensorHandle): The mask tensor.
            other (TensorHandle, optional): The tensor to load into when the mask is False. Defaults to None.
            cache_modifier (str, optional): The cache modifier to use. Defaults to None.
            eviction_policy (str, optional): The eviction policy to use. Defaults to None.
            is_volatile (bool, optional): Whether the load is volatile. Defaults to False.

        Returns:
            TensorHandle: The loaded tensor.
        """
        dtype_tt = ptrs.element_t
        dtype_np = _get_np_dtype(dtype_tt)
        if other is None:
            ret = _interpreter.load(ptrs.data, mask.data, None, dtype_np)
        else:
            ret = _interpreter.load(ptrs.data, mask.data, other.data, dtype_np)
        return TensorHandle(ret, dtype_tt)

    def create_load(self, ptr, _0, _1, is_volatile):
        """
        Tileon's load instruction loads data from memory into a tensor.

        Args:
            ptr (TensorHandle): The tensor of pointers to load from.
            _0 (str, optional): The cache modifier to use. Defaults to None.
            _1 (str, optional): The eviction policy to use. Defaults to None.
            is_volatile (bool, optional): Whether the load is volatile. Defaults to False.

        Returns:
            TensorHandle: The loaded tensor.
        """
        mask = TensorHandle(np.ones_like(ptr.data, dtype=bool), tl.int1)
        return self.create_masked_load(ptr, mask, None, _0, _1, is_volatile)

    def create_masked_store(self, ptrs, value, mask, cache_modifier, eviction_policy):
        """
        Tileon's masked store instruction stores data from a tensor into memory,
        conditionally based on a mask.

        Args:
            ptrs (TensorHandle): The tensor of pointers to store into.
            value (TensorHandle): The tensor of values to store.
            mask (TensorHandle): The mask tensor.
            cache_modifier (str, optional): The cache modifier to use. Defaults to None.
            eviction_policy (str, optional): The eviction policy to use. Defaults to None.

        Returns:
            None
        """
        return _interpreter.store(ptrs.data, value.data, mask.data)

    def create_store(self, ptr, val, _0, _1):
        """
        Tileon's store instruction stores data from a tensor into memory.

        Args:
            ptr (TensorHandle): The tensor of pointers to store into.
            val (TensorHandle): The tensor of values to store.
            _0 (str, optional): The cache modifier to use. Defaults to None.
            _1 (str, optional): The eviction policy to use. Defaults to None.

        Returns:
            None
        """
        mask = TensorHandle(np.ones_like(ptr.data, dtype=bool), tl.int1)
        return self.create_masked_store(ptr, val, mask, None, None)

    def cast_impl(self, src, dst_type):
        """
        Tileon's cast instruction casts data from one type to another.

        Args:
            src (TensorHandle): The tensor to cast.
            dst_type (Type): The type to cast to.

        Returns:
            TensorHandle: The casted tensor.
        """
        src_element_type = src.dtype.scalar
        dst_element_type = dst_type.scalar
        if (
            (src_element_type == tl.bfloat16 and dst_element_type == tl.float32)
            or (src_element_type == tl.float32 and dst_element_type == tl.bfloat16)
        ):
            data = _convert_float(src.data, src_element_type, dst_element_type, None).view(_get_np_dtype(dst_type))
            return TensorHandle(data, dst_type.scalar)
        else:
            return TensorHandle(src.data.astype(_get_np_dtype(dst_type)), dst_type.scalar)

    create_si_to_fp = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_ui_to_fp = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_fp_to_si = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_fp_to_ui = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_fp_ext = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_fp_trunc = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_int_cast = lambda self, src, dst_type, is_signed: self.cast_impl(src, dst_type)

    def create_fp_to_fp(self, src, dst_type, rounding_mode):
        """
        Tileon's fp_to_fp instruction casts floating-point data from one type to another.

        Args:
            src (TensorHandle): The tensor to cast.
            dst_type (Type): The type to cast to.
            rounding_mode (str): The rounding mode to use.

        Returns:
            TensorHandle: The casted tensor.
        """
        src_element_type = src.dtype.scalar
        dst_element_type = dst_type.scalar
        data = _convert_float(src.data, src_element_type, dst_element_type, rounding_mode).view(_get_np_dtype(dst_type))
        return TensorHandle(data, dst_type.scalar)

    def create_bitcast(self, src, dst_type):
        """
        Tileon's bitcast instruction casts data from one type to another.

        Args:
            src (TensorHandle): The tensor to cast.
            dst_type (Type): The type to cast to.

        Returns:
            TensorHandle: The casted tensor.
        """
        return TensorHandle(src.data.view(_get_np_dtype(dst_type)), dst_type.scalar)

    # -----------------------------------------------------------------------------
    # Binary Operators
    # -----------------------------------------------------------------------------

    def binary_op(self, lhs, rhs, op):
        """
        Tileon's binary_op instruction performs a binary operation on two tensors.

        Args:
            lhs (TensorHandle): The left-hand tensor.
            rhs (TensorHandle): The right-hand tensor.
            op (function): The binary operation to perform.

        Returns:
            TensorHandle: The result of the binary operation.
        """
        output = op(lhs.data, rhs.data)
        tl_dtype = lhs.dtype.scalar

        if not _validate_np_data_size(output, tl_dtype):
            output = output.astype(_get_np_dtype(tl_dtype))

        return TensorHandle(output, tl_dtype)

    create_fadd = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.add)
    create_fmul = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.multiply)
    create_fdiv = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.divide)
    create_frem = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.fmod)
    create_fsub = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.subtract)
    create_mul = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.multiply)
    create_precise_divf = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.divide)
    create_sdiv = lambda self, lhs, rhs: self.create_idiv(lhs, rhs)
    create_udiv = lambda self, lhs, rhs: self.create_idiv(lhs, rhs)
    # LLVM has 'numpy.fmod', not 'numpy.remainder', semantics on integer remainders.
    create_srem = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.fmod)
    create_urem = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.fmod)
    create_add = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.add)
    create_sub = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.subtract)
    create_shl = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.left_shift)
    create_lshr = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.right_shift)
    create_minsi = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.minimum)
    create_minui = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.minimum)
    create_minimumf = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.minimum)
    create_minnumf = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.minimum)
    create_maxsi = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.maximum)
    create_maxui = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.maximum)
    create_maximumf = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.maximum)
    create_maxnumf = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.maximum)
    create_icmpSLE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less_equal)
    create_icmpSLT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less)
    create_icmpSGE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater_equal)
    create_icmpSGT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater)
    create_icmpULE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less_equal)
    create_icmpULT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less)
    create_icmpUGE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater_equal)
    create_icmpUGT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater)
    create_icmpEQ = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.equal)
    create_icmpNE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.not_equal)
    create_fcmpOLT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less)
    create_fcmpOGT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater)
    create_fcmpOLE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less_equal)
    create_fcmpOGE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater_equal)
    create_fcmpOEQ = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.equal)
    create_fcmpONE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.not_equal)
    create_fcmpULT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less)
    create_fcmpUGT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater)
    create_fcmpULE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less_equal)
    create_fcmpUGE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater_equal)
    create_fcmpUEQ = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.equal)
    create_fcmpUNE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.not_equal)
    create_and = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.bitwise_and)
    create_xor = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.bitwise_xor)
    create_or = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.bitwise_or)
    create_int_to_ptr = create_bitcast
    create_ptr_to_int = create_bitcast

    def create_idiv(self, lhs, rhs):
        """
        Tileon's idiv instruction performs integer division.

        Note:
            Tileon has IEEE, not numpy/torch, semantics for %, and those carry
            through to //, so we have to use a nonstandard expression to get a
            reference result for //.

        Args:
            lhs (TensorHandle): The left operand.
            rhs (TensorHandle): The right operand.

        Returns:
            TensorHandle: The result of the integer division operation.
        """
        return TensorHandle((lhs.data - np.fmod(lhs.data, rhs.data)) // rhs.data, lhs.dtype.scalar)

    def create_ashr(self, lhs, rhs):
        """
        Tileon's rshift operator depends on the signedness of the left operand.

        Args:
            lhs (TensorHandle): The left operand.
            rhs (TensorHandle): The right operand.

        Returns:
            TensorHandle: The result of the right shift operation.
        """
        # Tileon's rshift operator depends on the signedness of the left operand
        lhs_dtype = _get_signed_np_dtype(lhs.data.dtype)
        rhs_dtype = _get_signed_np_dtype(rhs.data.dtype)
        lhs.data = lhs.data.astype(lhs_dtype)
        rhs.data = rhs.data.astype(rhs_dtype)
        return self.binary_op(lhs, rhs, np.right_shift)

    def create_umulhi(self, lhs, rhs):
        """
        Tileon's umulhi instruction returns the high bits of the product of two unsigned integers.

        Args:
            lhs (TensorHandle): The left operand.
            rhs (TensorHandle): The right operand.

        Returns:
            TensorHandle: The high bits of the product of lhs and rhs.
        """
        dtype = lhs.data.dtype
        if dtype == np.int64 or dtype == np.uint64:
            return TensorHandle(np_umulhi_u64(lhs.data, rhs.data), lhs.dtype.scalar)
        else:
            compute_dtype = getattr(np, f"uint{dtype.itemsize * 8 * 2}")
            lhs_data = lhs.data.astype(compute_dtype)
            rhs_data = rhs.data.astype(compute_dtype)
            ret_data = np.multiply(lhs_data, rhs_data) >> (dtype.itemsize * 8)
            return TensorHandle(ret_data.astype(dtype), lhs.dtype.scalar)

    # -----------------------------------------------------------------------------
    # Ternary Operators
    # -----------------------------------------------------------------------------

    def ternary_op(self, lhs, rhs, other, op):
        """
        Tileon's ternary_op instruction performs a ternary operation on three tensors.

        Args:
            lhs (TensorHandle): The left-hand tensor.
            rhs (TensorHandle): The middle tensor.
            other (TensorHandle): The right-hand tensor.
            op (function): The ternary operation to perform.

        Returns:
            TensorHandle: The result of the ternary operation.
        """
        output = op(lhs.data, rhs.data, other.data)
        tl_dtype = other.dtype.scalar

        if not _validate_np_data_size(output, tl_dtype):
            output = output.astype(_get_np_dtype(tl_dtype))

        return TensorHandle(output, tl_dtype)

    create_clampf = lambda self, arg, lo, hi, propagate_nans: self.ternary_op(arg, lo, hi, np.clip)
    create_select = lambda self, cond, lhs, rhs: self.ternary_op(cond, lhs, rhs, np.where)

    def create_fma(self, x, y, z):
        """
        Tileon's fma instruction performs fused multiply-add operation.

        Args:
            x (TensorHandle): The first operand.
            y (TensorHandle): The second operand.
            z (TensorHandle): The third operand.

        Returns:
            TensorHandle: The result of the fused multiply-add operation.
        """
        return TensorHandle(x.data * y.data + z.data, z.dtype.scalar)

    # -----------------------------------------------------------------------------
    # Unary Operators
    # -----------------------------------------------------------------------------
    
    def unary_op(self, arg, op):
        """
        Tileon's unary_op instruction performs a unary operation on a tensor.

        Args:
            arg (TensorHandle): The input tensor.
            op (function): The unary operation to perform.

        Returns:
            TensorHandle: The result of the unary operation.
        """
        return TensorHandle(op(arg.data), arg.dtype.scalar)

    def create_fabs(self, arg):
        """
        Tileon's fabs instruction returns the absolute value of each element in the input tensor.

        Args:
            arg (TensorHandle): The input tensor.

        Returns:
            TensorHandle: The absolute value of each element in the input tensor.
        """
        # Mask out the sign bit based on the primitive length
        dtype_tt = arg.dtype
        mask_bitwidth = dtype_tt.primitive_bitwidth - 1
        np_uint_dtype = getattr(np, f"uint{dtype_tt.primitive_bitwidth}")
        data = arg.data.view(np_uint_dtype)
        mask = (1 << mask_bitwidth) - 1
        ret = (data & mask).view(_get_np_dtype(dtype_tt))
        return TensorHandle(ret, arg.dtype.scalar)

    create_cos = lambda self, arg: self.unary_op(arg, np.cos)
    create_exp = lambda self, arg: self.unary_op(arg, np.exp)
    create_exp2 = lambda self, arg: self.unary_op(arg, np.exp2)
    create_iabs = lambda self, arg: self.unary_op(arg, np.abs)
    create_floor = lambda self, arg: self.unary_op(arg, np.floor)
    create_ceil = lambda self, arg: self.unary_op(arg, np.ceil)
    create_log = lambda self, arg: self.unary_op(arg, np.log)
    create_log2 = lambda self, arg: self.unary_op(arg, np.log2)
    create_precise_sqrt = lambda self, arg: self.unary_op(arg, np.sqrt)
    create_sqrt = lambda self, arg: self.unary_op(arg, np.sqrt)
    create_sin = lambda self, arg: self.unary_op(arg, np.sin)

    def create_erf(self, arg):
        """
        Tileon's erf instruction returns the error function of each element in the input tensor.

        Args:
            arg (TensorHandle): The input tensor.

        Returns:
            TensorHandle: The error function of each element in the input tensor.
        """
        ret = np_erf_fp32(arg.data) if arg.data.dtype == np.float32 else np_erf_fp64(arg.data)
        return TensorHandle(ret, arg.dtype.scalar)

    def create_rsqrt(self, arg):
        """
        Tileon's rsqrt instruction returns the reciprocal square root of each element in the input tensor.

        Args:
            arg (TensorHandle): The input tensor.

        Returns:
            TensorHandle: The reciprocal square root of each element in the input tensor.
        """
        return TensorHandle(1 / np.sqrt(arg.data), arg.dtype.scalar)

    # -----------------------------------------------------------------------------
    # Tensor Operators
    # -----------------------------------------------------------------------------
    
    def create_reshape(self, arg, shape, allow_reorder):
        """
        Tileon's reshape instruction reshapes the input tensor to the specified shape.

        Args:
            arg (TensorHandle): The input tensor.
            shape (list): The desired shape.
            allow_reorder (bool): Whether to allow reordering of axes.

        Returns:
            TensorHandle: The reshaped tensor.
        """
        return TensorHandle(arg.data.reshape(shape), arg.dtype.scalar)

    def create_trans(self, arg, perm):
        """
        Tileon's trans instruction transposes the input tensor along the specified axes.

        Args:
            arg (TensorHandle): The input tensor.
            perm (list): The permutation of axes.

        Returns:
            TensorHandle: The transposed tensor.
        """
        return TensorHandle(np.transpose(arg.data, perm), arg.dtype.scalar)

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        """
        Tileon's dot instruction performs a dot product between two tensors.

        Args:
            a (TensorHandle): The first input tensor.
            b (TensorHandle): The second input tensor.
            d (TensorHandle): The output tensor.
            input_precision (TileonType): The precision of the input tensors.
            max_num_imprecise_acc (int): The maximum number of imprecise accumulations.

        Returns:
            TensorHandle: The result of the dot product.
        """
        a_data = a.data
        b_data = b.data
        if (
            (a.dtype.primitive_bitwidth == 8 and a.dtype.is_floating())
            or (b.dtype.primitive_bitwidth == 8 and b.dtype.is_floating())
        ):
            a_data = _convert_float(a_data, a.dtype, tl.float16, None).view(np.float16)
            b_data = _convert_float(b_data, b.dtype, tl.float16, None).view(np.float16)
        return TensorHandle(np.matmul(a_data, b_data, dtype=d.data.dtype) + d.data, d.dtype.scalar)

    def create_make_range(self, ret_ty, start, stop):
        """
        Tileon's make_range instruction creates a tensor with values from start to stop (exclusive).

        Args:
            ret_ty (TileonType): The type of the output tensor.
            start (int): The starting value.
            stop (int): The stopping value (exclusive).

        Returns:
            TensorHandle: The tensor with values from start to stop (exclusive).
        """
        return TensorHandle(np.arange(start, stop, dtype=np.int32), tl.int32)

    def create_histogram(self, data, bins, mask):
        """
        Tileon's histogram instruction calculates the histogram of the input tensor.

        Args:
            data (TensorHandle): The input tensor.
            bins (int): The number of bins.
            mask (TensorHandle, optional): The mask tensor. Defaults to None.

        Notes:
            By default np.histogram returns int64 dtype values
            Docs specify that returned dtype is taken based on optional weights.dtype
            This is fix for interpreter cases where for example int32 tensor is being passed
            But unexpectedly int64 values are being returned causing
            tl.store to write 8 bytes instead of 4 bytes which lead to silent data corruption

        Returns:
            TensorHandle: The histogram of the input tensor.
        """
        if mask is None:
            mask = TensorHandle(np.ones_like(data.data, dtype=bool), tl.int1)

        dummy_weights = np.ones_like(data.data, dtype=data.data.dtype)
        # force all masked elements to zero
        data = np.where(mask.data, data.data, np.zeros_like(data.data))
        histogram = np.histogram(data, bins=bins, range=(0, bins), weights=dummy_weights)[0]
        # remove overcounted elements
        histogram[0] -= np.logical_not(mask.data).sum()
        return TensorHandle(histogram, tl.int32)

    def create_gather(self, src, indices, axis):
        """
        Tileon's gather instruction gathers elements from a tensor along a specified axis.

        Args:
            src (TensorHandle): The tensor from which to gather elements.
            indices (TensorHandle): The indices of the elements to gather.
            axis (int): The axis along which to gather elements.

        Returns:
            TensorHandle: The gathered elements.
        """
        return TensorHandle(np.take_along_axis(src.data, indices.data, axis=axis), src.dtype.scalar)

    # -----------------------------------------------------------------------------
    # Pointer Arithmetic
    # -----------------------------------------------------------------------------

    def create_addptr(self, ptr, offset):
        """
        Tileon's addptr instruction adds an offset to a pointer.

        Args:
            ptr (TensorHandle): The pointer to which the offset will be added.
            offset (TensorHandle): The offset to add to the pointer.

        Returns:
            TensorHandle: The pointer with the added offset.
        """
        dtype_tt = ptr.element_t
        element_bitwidth = dtype_tt.primitive_bitwidth
        # int1's bitwidth is 1, but we need to use 8 for pointer arithmetic
        element_bytewidth = max(1, element_bitwidth // 8)
        return TensorHandle(ptr.data + element_bytewidth * offset.data.astype(np.uint64), ptr.dtype)

    def create_tensor_pointer_load(
        self,
        ptr,
        boundary_check,
        padding_option,
        cache_modifier,
        eviction_policy,
        is_volatile
    ):
        """
        Tileon's load instruction loads data from a tensor pointer.

        Args:
            ptr (TensorHandle): The tensor pointer from which to load data.
            boundary_check (bool): Whether to perform boundary checking.
            padding_option (ir.PADDING_OPTION): The padding option to use.
            cache_modifier (ir.CACHE_MODIFIER): The cache modifier to use.
            eviction_policy (ir.EVICTION_POLICY): The eviction policy to use.
            is_volatile (bool): Whether the load is volatile.

        Returns:
            TensorHandle: The loaded data.
        """
        ptrs, masks = ptr.materialize_pointers(boundary_check)
        dtype_tt = ptrs.element_t
        dtype_np = _get_np_dtype(dtype_tt)
        if padding_option is None:
            other = None
        elif padding_option == _ir.PADDING_OPTION.PAD_ZERO:
            other = TensorHandle(np.zeros_like(ptrs.data, dtype=dtype_np), dtype_tt)
        elif padding_option == _ir.PADDING_OPTION.PAD_NAN:
            other = TensorHandle(np.full_like(ptrs.data, float('nan'), dtype=dtype_np), dtype_tt)
        else:
            raise ValueError(f"unsupported padding option {padding_option}")
        return self.create_masked_load(ptrs, masks, other, cache_modifier, eviction_policy, is_volatile)

    def create_tensor_pointer_store(
        self,
        ptr,
        value,
        boundary_check,
        cache_modifier,
        eviction_policy
    ):
        """
        Tileon's store instruction stores data to a tensor pointer.

        Args:
            ptr (TensorHandle): The tensor pointer to which to store data.
            value (TensorHandle): The data to store.
            boundary_check (bool): Whether to perform boundary checking.
            cache_modifier (ir.CACHE_MODIFIER): The cache modifier to use.
            eviction_policy (ir.EVICTION_POLICY): The eviction policy to use.

        Returns:
            None
        """
        ptrs, masks = ptr.materialize_pointers(boundary_check)
        return self.create_masked_store(ptrs, value, masks, cache_modifier, eviction_policy)

    def create_expand_dims(self, arg, axis):
        """
        Tileon's expand_dims instruction expands the dimensions of a tensor.

        Args:
            arg (TensorHandle): The tensor to expand.
            axis (int): The axis along which to expand dimensions.

        Returns:
            TensorHandle: The expanded tensor.
        """
        return TensorHandle(np.expand_dims(arg.data, axis), arg.dtype.scalar)

    def create_broadcast(self, arg, shape):
        """
        Tileon's broadcast instruction broadcasts a tensor to a specified shape.

        Args:
            arg (TensorHandle): The tensor to broadcast.
            shape (tuple): The shape to broadcast the tensor to.

        Returns:
            TensorHandle: The broadcasted tensor.
        """
        return TensorHandle(np.broadcast_to(arg.data, shape), arg.dtype.scalar)

    def create_cat(self, lhs, rhs):
        """
        Tileon's cat instruction concatenates two tensors along a specified axis.

        Args:
            lhs (TensorHandle): The first tensor to concatenate.
            rhs (TensorHandle): The second tensor to concatenate.

        Returns:
            TensorHandle: The concatenated tensor.
        """
        return TensorHandle(np.concatenate([lhs.data, rhs.data]), lhs.dtype.scalar)

    def create_join(self, lhs, rhs):
        """
        Tileon's join instruction joins two tensors along a specified axis.

        Args:
            lhs (TensorHandle): The first tensor to join.
            rhs (TensorHandle): The second tensor to join.

        Notes:
            - The join instruction only supports joining two original tensors into a new one along the last axis.

        Returns:
            TensorHandle: The joined tensor.
        """
        return TensorHandle(np.stack([lhs.data, rhs.data], axis=-1), lhs.dtype.scalar)

    def create_split(self, val):
        """
        Tileon's split instruction splits a tensor along a specified axis.

        Args:
            val (TensorHandle): The tensor to split.

        Notes:
            - The split instruction only supports splitting the original tensor into two along the last axis.

        Returns:
            tuple: A tuple of two TensorHandle objects, each representing a split of the original tensor.
        """
        return (TensorHandle(val.data[..., 0], val.dtype.scalar), TensorHandle(val.data[..., 1], val.dtype.scalar))

    def create_splat(self, ret_ty, arg):
        """
        Tileon's splat instruction splats a tensor to a specified shape.

        Args:
            ret_ty (ir.TensorType): The shape to splat the tensor to.
            arg (TensorHandle): The tensor to splat.

        Returns:
            TensorHandle: The splatted tensor.
        """
        shape = ret_ty.shape
        if self._vectorized_nx is not None:
             target_shape = (self._vectorized_nx, *shape)
             try:
                 return TensorHandle(np.broadcast_to(arg.data, target_shape), arg.dtype.scalar)
             except ValueError:
                 pass

        if isinstance(arg.dtype, tl.block_t):
            return TensorHandle(np.full(shape, arg.data[0], dtype=_get_np_dtype(arg.dtype)), arg.dtype.scalar)
        else:  # scalar
            return TensorHandle(np.full(shape, arg.data, dtype=_get_np_dtype(arg.dtype)), arg.dtype.scalar)

    def create_unsplat(self, arg):
        """
        Tileon's unsplat instruction unsplats a tensor to a specified shape.

        Args:
            arg (TensorHandle): The tensor to unsplat.

        Returns:
            TensorHandle: The unsplatted tensor.
        """
        return TensorHandle(np.full((1, ), arg.data[0], dtype=_get_np_dtype(arg.dtype)), arg.dtype.scalar)

    def create_atomic_cas(self, ptr, cmp, val, sem, scope):
        """
        Tileon's atomic_cas instruction atomically compares and swaps a value in memory.

        Args:
            ptr (TensorHandle): The pointer to the memory location.
            cmp (TensorHandle): The value to compare against.
            val (TensorHandle): The value to swap in if the comparison is successful.
            sem (ir.ATOMIC_SEMANTIC): The atomic semantic to use.
            scope (ir.ATOMIC_SCOPE): The atomic scope to use.

        Returns:
            TensorHandle: The result of the atomic compare and swap operation.
        """
        if sem not in self.ir_sem_to_interpreter_sem:
            raise ValueError(f"unsupported semantic {sem}")
        sem = self.ir_sem_to_interpreter_sem[sem]
        return TensorHandle(_interpreter.atomic_cas(ptr.data, cmp.data, val.data, sem), cmp.dtype.scalar)

    def create_atomic_rmw(self, rmwOp, ptr, val, mask, sem, scope):
        """
        Tileon's atomic_rmw instruction atomically performs a reduction operation on a value in memory.

        Args:
            rmwOp (ir.REDUCE_OP): The reduction operation to perform.
            ptr (TensorHandle): The pointer to the memory location.
            val (TensorHandle): The value to perform the reduction operation on.
            mask (TensorHandle): The mask to apply to the reduction operation.
            sem (ir.ATOMIC_SEMANTIC): The atomic semantic to use.
            scope (ir.ATOMIC_SCOPE): The atomic scope to use.

        Returns:
            TensorHandle: The result of the atomic reduction operation.
        """
        if rmwOp not in self.ir_rmw_op_to_interpreter_rmw_op:
            raise ValueError(f"unsupported rmwOp {rmwOp}")
        if sem not in self.ir_sem_to_interpreter_sem:
            raise ValueError(f"unsupported semantic {sem}")
        rmwOp = self.ir_rmw_op_to_interpreter_rmw_op[rmwOp]
        sem = self.ir_sem_to_interpreter_sem[sem]
        return TensorHandle(_interpreter.atomic_rmw(rmwOp, ptr.data, val.data, mask.data, sem), val.dtype.scalar)

    def create_extern_elementwise(self, libName, libPath, symbol, argList, retType, isPure):
        raise NotImplementedError("extern_elementwise not supported in interpreter mode")

    def create_inline_asm(self, inlineAsm, constraints, values, type, isPure, pack):
        raise NotImplementedError("inline_asm not supported in interpreter mode")

    def create_print(self, prefix, hex, values, isSigned):
        """
        Tileon's print instruction prints the values of tensors to the console.

        Args:
            prefix (str): The prefix to print before the values.
            hex (bool): Whether to print the values in hexadecimal format.
            values (list of TensorHandle): The tensors to print.
            isSigned (bool): Whether to print the values as signed integers.
        """
        msg = f"({self.grid_idx[0]}, {self.grid_idx[1]}, {self.grid_idx[2]})"
        if prefix:
            msg += f" {prefix}"
        if hex:
            np.set_printoptions(formatter={'all': lambda x: f"0x{x:02x}"})
        for value in values:
            print(msg + f" {value.data}")
        if hex:
            np.set_printoptions(formatter=None)

    def create_assert(self, condition, message):
        """
        Tileon's assert instruction asserts that a condition is true.

        Args:
            condition (bool): The condition to assert.
            message (str): The message to print if the condition is false.
        """
        # Interpreter's device_assert function has a different format than Triton's device_assert
        assert condition, f"{message}"

    def create_assume(self, condition):
        """
        Tileon's assume instruction assumes that a condition is true.

        Args:
            condition (bool): The condition to assume.
        """
        assert condition, "Assume failed"

    def create_barrier(self):
        """
        Tileon's barrier instruction synchronizes all threads in a block.

        Note:
            Tileon's barrier applies to each program in a grid, so it's a no-op in the interpreter.
        """
        ...

    def create_make_block_ptr(self, base, shape, strides, offsets, block_shape, order):
        """
        Tileon's make_block_ptr instruction creates a block pointer.

        Args:
            base (TensorHandle): The base tensor.
            shape (list of TensorHandle): The shape of the block.
            strides (list of TensorHandle): The strides of the block.
            offsets (list of TensorHandle): The offsets of the block.
            block_shape (list of int): The shape of the block.
            order (str): The order of the block.

        Returns:
            BlockPointerHandle: The block pointer.
        """
        # Create new offsets to avoid modifying the original
        new_offsets = [offset.clone() for offset in offsets]
        return BlockPointerHandle(base, shape, strides, new_offsets, block_shape, order)

    def create_advance(self, ptr, offsets):
        """
        Tileon's advance instruction advances a block pointer by a given offset.

        Args:
            ptr (BlockPointerHandle): The block pointer to advance.
            offsets (list of TensorHandle): The offsets to advance the block pointer by.

        Returns:
            BlockPointerHandle: The advanced block pointer.
        """
        if len(ptr.offsets) != len(offsets):
            raise ValueError("len(ptr.offsets) != len(offsets)")
        # Create new offsets to avoid modifying the original
        new_offsets = [offset.clone() for offset in ptr.offsets]
        ret = BlockPointerHandle(ptr.base, ptr.shape, ptr.strides, new_offsets, ptr.block_shape, ptr.order)
        for i in range(len(offsets)):
            ret.offsets[i].data += offsets[i].data
        return ret

    def create_make_tensor_descriptor(
        self,
        base: TensorHandle,
        shape: List[TensorHandle],
        strides: List[TensorHandle],
        tensor_shape: List[int],
        is_signed: bool,
        padding: str = "zero"
    ):
        """
        Tileon's make_tensor_descriptor instruction creates a tensor descriptor.

        Args:
            base (TensorHandle): The base tensor.
            shape (list of TensorHandle): The shape of the tensor.
            strides (list of TensorHandle): The strides of the tensor.
            tensor_shape (list of int): The shape of the tensor.
            is_signed (bool): Whether the tensor is signed.
            padding (str, optional): The padding option. Defaults to "zero".

        Returns:
            TensorDescHandle: The tensor descriptor.
        """
        desc = TensorDescHandle(base, shape, strides, tensor_shape, padding)
        desc.validate()
        return desc

    def create_descriptor_load(
        self,
        desc: TensorDescHandle,
        indices: List[TensorHandle],
        cache_modifier,
        eviction_policy
    ):
        """
        Tileon's descriptor_load instruction loads data from a tensor descriptor.

        Args:
            desc (TensorDescHandle): The tensor descriptor.
            indices (list of TensorHandle): The indices to load.
            cache_modifier (str, optional): The cache modifier. Defaults to None.
            eviction_policy (str, optional): The eviction policy. Defaults to None.

        Returns:
            TensorHandle: The loaded data.
        """
        assert isinstance(desc, TensorDescHandle)
        ptrs, mask = desc.materialize_pointers(indices)
        dtype_tt = ptrs.element_t
        dtype_np = _get_np_dtype(dtype_tt)
        padding = desc.padding
        if padding == _ir.PADDING_OPTION.PAD_ZERO:
            other = TensorHandle(np.zeros_like(ptrs.data, dtype=dtype_np), dtype_tt)
        elif padding == _ir.PADDING_OPTION.PAD_NAN:
            other = TensorHandle(np.full_like(ptrs.data, float('nan'), dtype=dtype_np), dtype_tt)
        else:
            raise ValueError(f"unsupported padding {padding}")
        return self.create_masked_load(
            ptrs,
            mask,
            other,
            cache_modifier=cache_modifier,
            eviction_policy=eviction_policy,
            is_volatile=False
        )

    def create_descriptor_store(
        self,
        desc: TensorDescHandle,
        value: TensorHandle,
        indices: List[TensorHandle]
    ):
        """
        Tileon's descriptor_store instruction stores data to a tensor descriptor.

        Args:
            desc (TensorDescHandle): The tensor descriptor.
            value (TensorHandle): The value to store.
            indices (list of TensorHandle): The indices to store.

        Returns:
            None
        """
        ptrs, mask = desc.materialize_pointers(indices)
        return self.create_masked_store(
            ptrs,
            value,
            mask,
            None,
            None
        )

    def create_descriptor_gather(
        self,
        desc: TensorDescHandle,
        x_offsets: TensorHandle,
        y_offset: TensorHandle,
        type
    ):
        """
        Tileon's descriptor_gather instruction gathers data from a tensor descriptor.

        Args:
            desc (TensorDescHandle): The tensor descriptor.
            x_offsets (TensorHandle): The x offsets to gather.
            y_offset (TensorHandle): The y offset to gather.
            type: The type of the gathered data.

        Returns:
            TensorHandle: The gathered data.
        """
        dtype = desc.base.dtype.element_t
        np_dtype = _get_np_dtype(dtype)
        result = np.zeros([x_offsets.data.shape[0], desc.block_shape[-1]],
                          dtype=np_dtype)
        cache_modifier = None
        eviction_policy = None
        for i, x_offset in enumerate(x_offsets.data):
            indices = [TensorHandle(x_offset, tl.int32), y_offset]
            result[i, :] = self.create_descriptor_load(desc, indices, cache_modifier, eviction_policy).data
        return TensorHandle(result, dtype)

    def create_descriptor_scatter(
        self,
        desc: TensorDescHandle,
        value: TensorHandle,
        x_offsets: TensorHandle,
        y_offset: TensorHandle
    ):
        """
        Tileon's descriptor_scatter instruction scatters data to a tensor descriptor.

        Args:
            desc (TensorDescHandle): The tensor descriptor.
            value (TensorHandle): The value to scatter.
            x_offsets (TensorHandle): The x offsets to scatter.
            y_offset (TensorHandle): The y offset to scatter.

        Returns:
            None
        """
        for i, x_offset in enumerate(x_offsets.data):
            slice = TensorHandle(value.data[i], value.dtype)
            indices = [TensorHandle(x_offset, tl.int32), y_offset]
            self.create_descriptor_store(desc, slice, indices)

    def get_all_ones_value(self, type):
        """
        Get a tensor handle with all ones value.

        Args:
            type: The type of the tensor.

        Returns:
            TensorHandle: The tensor handle with all ones value.
        """
        np_type = _get_np_dtype(type)
        if "int" in np_type.name:
            return TensorHandle(np.full(1, -1, dtype=np_type), type.scalar)
        elif np_type == np.bool_:
            return TensorHandle(np.full(1, True, dtype=np_type), type.scalar)
        else:
            raise TypeError(f"unsupported type {type}")


_MISSING = object()
interpreter_builder = InterpreterBuilder()
interpreter_semantic: TileonSemantic = TileonSemantic(interpreter_builder)


class _LangPatchScope:
    """Tracks patched attributes so they can be restored."""

    def __init__(self) -> None:
        self._changes: list[tuple[object, str, object]] = []

    def set_attr(self, obj: object, name: str, value: object) -> None:
        """
        Set an attribute of an object.

        Args:
            obj (object): The object.
            name (str): The name of the attribute.
            value (object): The value of the attribute.
        """
        original = getattr(obj, name, _MISSING)
        self._changes.append((obj, name, original))
        setattr(obj, name, value)

    def restore(self) -> None:
        """Restore the attributes of the objects."""
        while self._changes:
            obj, name, original = self._changes.pop()
            if original is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, original)


def _patch_attr(obj, name, member, builder, scope: _LangPatchScope):
    """Replace a member function with a wrapper that injects the semantic."""
    new_member = (
        lambda *args, member=member, **kwargs: (
            member(*args, **{k: v for k, v in kwargs.items() if k != "_semantic"}, 
            _semantic=interpreter_semantic)
        )
    )
    scope.set_attr(obj, name, new_member)


def _patch_builtin(pkg, builder, scope: _LangPatchScope):
    """Replace all builtin functions in a package with a wrapper that injects the semantic."""
    for name, member in inspect.getmembers(pkg):
        if tl.core.is_builtin(member):
            _patch_attr(pkg, name, member, builder, scope)


def _patch_builtin(pkg, builder, scope: _LangPatchScope):
    """Replace all builtin functions in a package with a wrapper that injects the semantic."""
    for name, member in inspect.getmembers(pkg):
        if tl.core.is_builtin(member):
            _patch_attr(pkg, name, member, builder, scope)


def _patch_lang_tensor(tensor, scope: _LangPatchScope):
    """Replace methods of a tensor with wrappers that inject the semantic."""

    def _get_bool(self):
        data = self.handle.data
        # in tileon, only scalars can be converted to booleans
        # here we need this hack because all scalars are tensors
        return bool(data) if data.size == 1 else True

    def _get_transpose(self):
        handle = TensorHandle(np.transpose(self.handle.data), self.handle.dtype)
        assert self.type.is_block()
        block_shape = list(self.type.shape)
        block_shape[-1], block_shape[-2] = block_shape[-2], block_shape[-1]
        res_ty = tl.core.block_type(self.dtype, block_shape)
        return tl.core.tensor(handle, res_ty)

    scope.set_attr(tensor, "__index__", lambda self: int(self.handle.data.squeeze()))
    scope.set_attr(tensor, "__bool__", lambda self: _get_bool(self))
    scope.set_attr(tensor, "__repr__", lambda self: repr(self.handle.data))
    scope.set_attr(tensor, "__str__", lambda self: str(self.handle.data))
    scope.set_attr(tensor, "T", property(_get_transpose))


class ReduceScanOpInterface:
    """Interface for reduce and scan operations."""

    def __init__(self, axis: int, combine_fn):
        self.axis = axis
        self.combine_fn = combine_fn

    def check_tensor(self, input) -> None:
        """
        Check if input tensor is valid.

        Args:
            input (Tensor): Input tensor.
        """
        axis = self.axis
        for arg in input:
            if not isinstance(arg, tl.core.tensor):
                raise ValueError(f"input must be a tensor, got {type(arg)}")
            shape = arg.shape
            if axis is not None and axis >= len(shape):
                raise ValueError(f"axis {axis} out of bounds for shape {shape}")

    def to_tensor(self, ret, dtype):
        """
        Convert numpy array to tensor.

        Args:
            ret (np.ndarray): Numpy array.
            dtype (DataType): Data type.
        
        Returns:
            Tensor: Tensor.
        """
        np_dtype = _get_np_dtype(dtype)
        if hasattr(ret, "shape") and ret.shape:
            ret = ret.astype(np_dtype)
            ret_type = tl.block_t(dtype, list(ret.shape))
        else:
            ret = np.array([ret], dtype=np_dtype)
            ret_type = dtype
        return tl.core.tensor(TensorHandle(ret, dtype.scalar), ret_type)

    def apply_impl(self, input):
        raise NotImplementedError("apply_impl must be implemented by subclasses")

    def apply(self, input):
        """
        Apply reduce operation.

        Args:
            input (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor.
        """
        if not isinstance(input, tuple):
            return self.apply((input, ))[0]
        self.check_tensor(input)
        ret = self.apply_impl(input)
        return tuple(ret) if isinstance(ret, (list, tuple)) else (ret, )


class ReduceOps(ReduceScanOpInterface):
    """Reduce operations."""

    def __init__(self, axis, combine_fn, keep_dims):
        super().__init__(axis, combine_fn)
        self.keep_dims = keep_dims

    def unravel(self, input, axis):
        """
        Unravel input tensor along axis.

        Args:
            input (Tensor): Input tensor.
            axis (int): Axis to unravel.
        
        Returns:
            tuple: Unraveled input tensor.
        """
        ret = []
        for data in input:
            if axis is not None:
                ret.append(data)
            else:
                axis = 0
                ret.append(self.to_tensor(data.handle.data.flatten(), data.dtype))
        return tuple(ret), axis

    def generic_reduce(self, input):
        """
        Generic reduce operation.

        Args:
            input (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor.
        """
        original_axis = self.axis
        input, axis = self.unravel(input, self.axis)
        input_data = []
        output_data = []
        input_shape = input[0].handle.data.shape
        output_shape = input_shape[0:axis] + input_shape[axis + 1:]
        for arg in input:
            input_data.append(arg.handle.data)
            output_data.append(np.zeros(output_shape, dtype=arg.handle.data.dtype))
        # Reduce on axis
        for i in range(input_data[0].size):
            # Recover input_index from i using input_shape
            input_index = np.unravel_index(i, input_shape)
            output_index = input_index[0:axis] + input_index[axis + 1:]
            input_tuple = tuple(
                self.to_tensor(d[input_index], input[ii].dtype) for ii, d in enumerate(input_data)
            )
            if input_index[axis] == 0:
                # First element
                for j in range(len(output_data)):
                    output_data[j][output_index] = input_tuple[j].handle.data.item()
            else:
                acc_tuple = tuple(
                    self.to_tensor(o[output_index], input[oi].dtype) for oi, o in enumerate(output_data)
                )
                combine_fn_ret = self.combine_fn.fn(*acc_tuple, *input_tuple)
                acc_tuple = (combine_fn_ret, ) if not isinstance(combine_fn_ret, tuple) else combine_fn_ret
                for j in range(len(output_data)):
                    if isinstance(acc_tuple[j], tl.core.tensor):
                        output_data[j][output_index] = acc_tuple[j].handle.data.item()
                    else:
                        output_data[j][output_index] = acc_tuple[j]
        # Pack output
        ret = []
        for i, data in enumerate(output_data):
            if self.keep_dims:
                if original_axis is not None:
                    data = np.expand_dims(data, axis)
                else:
                    for _ in range(len(input_shape)):
                        data = np.expand_dims(data, 0)
            elif original_axis is None:
                # Take a scalar
                data = data.item()
            ret.append(self.to_tensor(data, input[i].dtype))
        return ret

    def min_max(self, input, val_reduce_op, idx_reduce_op=None):
        """
        Min/Max reduction.

        Args:
            input (Tensor): Input tensor.
            val_reduce_op (Callable): Value reduction operator.
            idx_reduce_op (Callable, optional): Index reduction operator. Defaults to None.
        
        Returns:
            Tensor: Output tensor.
        """
        # If input is a tuple, it must be (val, index), and we only take val
        input = input[0] if isinstance(input, tuple) else input
        val = None
        idx = None
        if val_reduce_op:
            val = self.to_tensor(val_reduce_op(input.handle.data, axis=self.axis, keepdims=self.keep_dims), input.dtype)
        if idx_reduce_op:
            idx = self.to_tensor(idx_reduce_op(input.handle.data, axis=self.axis, keepdims=self.keep_dims), tl.int32)
        if val is not None and idx is not None:
            return val, idx
        elif val is not None:
            return val
        elif idx is not None:
            return idx
        else:
            raise ValueError("val_reduce_op and idx_reduce_op are both None")

    def sum(self, input):
        """
        Sum reduction.

        Args:
            input (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor.
        """
        return self.to_tensor(np.sum(input.handle.data, axis=self.axis, keepdims=self.keep_dims), input.dtype)

    def apply_impl(self, input):
        if self.combine_fn == tl.standard._argmin_combine_tie_break_left:
            return self.min_max(input[0], val_reduce_op=np.min, idx_reduce_op=np.argmin)
        elif self.combine_fn == tl.standard._argmax_combine_tie_break_left:
            return self.min_max(input[0], val_reduce_op=np.max, idx_reduce_op=np.argmax)
        elif self.combine_fn == tl.standard._elementwise_max:
            return self.min_max(input[0], val_reduce_op=np.nanmax, idx_reduce_op=None)
        elif self.combine_fn == tl.standard._elementwise_min:
            return self.min_max(input[0], val_reduce_op=np.nanmin, idx_reduce_op=None)
        elif self.combine_fn == tl.standard._sum_combine:
            return self.sum(input[0])
        else:
            # Fall back to the slow mode
            return self.generic_reduce(input)


class ScanOps(ReduceScanOpInterface):
    """Scan operations for associative reduction."""

    def __init__(self, axis, combine_fn, reverse):
        super().__init__(axis, combine_fn)
        self.reverse = reverse

    def cumsum(self, input):
        return [self.to_tensor(np.cumsum(input.handle.data, axis=self.axis), dtype=input.dtype)]
    
    def cumprod(self, input):
        """
        Cumulative product reduction.

        Args:
            input (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor.
        """
        return [self.to_tensor(np.cumprod(input.handle.data, axis=self.axis), dtype=input.dtype)]

    def generic_scan(self, input):
        """
        Generic scan operation.

        Args:
            input (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor.
        """
        input_data = []
        output_data = []
        shape = input[0].handle.data.shape
        for arg in input:
            input_data.append(arg.handle.data)
            output_data.append(np.zeros(shape, dtype=arg.handle.data.dtype))
        # Scan on axis
        for i in range(input_data[0].size):
            # Recover index from i using shape
            index = np.unravel_index(i, shape)
            data = tuple(self.to_tensor(d[index], input[ii].dtype) for ii, d in enumerate(input_data))
            if index[self.axis] == 0:
                # First element
                for j in range(len(output_data)):
                    output_data[j][index] = data[j].handle.data.item()
            else:
                prev_index = tuple(
                    index[i] - 1 if i == self.axis else index[i] for i in range(len(index))
                )
                acc_tuple = tuple(
                    self.to_tensor(o[prev_index], input[oi].dtype) for oi, o in enumerate(output_data)
                )
                combine_fn_ret = self.combine_fn.fn(*acc_tuple, *data)
                acc_tuple = (combine_fn_ret, ) if not isinstance(combine_fn_ret, tuple) else combine_fn_ret
                for j in range(len(output_data)):
                    output_data[j][index] = acc_tuple[j].handle.data.item() if isinstance(
                        acc_tuple[j], tl.core.tensor) else acc_tuple[j]
        # Pack output
        ret = []
        for i, data in enumerate(output_data):
            ret.append(self.to_tensor(data, input[i].dtype))
        return ret

    def apply_impl(self, input):
        new_input = []
        if self.reverse:
            for arg in input:
                new_input.append(self.to_tensor(np.flip(arg.handle.data, axis=self.axis), arg.dtype))
        else:
            new_input = input
        if self.combine_fn == tl.standard._sum_combine:
            ret = self.cumsum(new_input[0])
        elif self.combine_fn == tl.standard._prod_combine:
            ret = self.cumprod(new_input[0])
        else:
            # Fall back to the slow mode
            ret = self.generic_scan(new_input)
        if self.reverse:
            for arg in ret:
                arg.handle.data = np.flip(arg.handle.data, axis=self.axis)
        return ret


def _patch_reduce_scan(scope: _LangPatchScope):
    """Patch the reduce and scan operations."""
    def _new_reduce(input, axis, combine_fn, keep_dims=False, **kwargs):
        return ReduceOps(axis, combine_fn, keep_dims).apply(input)

    def _new_scan(input, axis, combine_fn, reverse=False, **kwargs):
        return ScanOps(axis, combine_fn, reverse).apply(input)

    scope.set_attr(tl, "reduce", _new_reduce)
    scope.set_attr(tl, "associative_scan", _new_scan)
    scope.set_attr(tl.core, "reduce", _new_reduce)
    scope.set_attr(tl.core, "associative_scan", _new_scan)


def _patch_lang_core(lang, scope: _LangPatchScope):
    def _new_to_ir(self, builder):
        """Convert the type to the Tileon IR type."""
        if self.name == 'void':
            return builder.get_void_t()
        elif self.name == 'int1':
            return builder.get_int1_t()
        elif self.name == 'int8':
            return builder.get_int8_t()
        elif self.name == 'uint8':
            return builder.get_uint8_t()
        elif self.name == 'int16':
            return builder.get_int16_t()
        elif self.name == 'uint16':
            return builder.get_uint16_t()
        elif self.name == 'int32':
            return builder.get_int32_t()
        elif self.name == 'uint32':
            return builder.get_uint32_t()
        elif self.name == 'int64':
            return builder.get_int64_t()
        elif self.name == 'uint64':
            return builder.get_uint64_t()
        elif self.name == 'fp8e5':
            return builder.get_fp8e5_t()
        elif self.name == 'fp8e4nv':
            return builder.get_fp8e4nv_t()
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

    # can't just map lang.static_range to `range`, because `tl.static_range`
    # can get `step` passed by keyword
    def _new_range(arg1, arg2=None, step=None, **kwargs):
        """Create a range of integers."""
        if step is None:
            step = 1
        if arg2 is None:
            start, end = 0, arg1
        else:
            start, end = arg1, arg2
        return range(start, end, step)

    def _new_static_assert(cond, msg=""):
        """Assert that the condition is true at compile time."""
        assert cond, msg

    def _set_attr(input, values, name):
        """Set the attribute of the tensor."""
        # skip non tensor types. This may happen for induction variables.
        if not isinstance(input, tl.tensor):
            return input
        values = [values] if not isinstance(values, (list, tuple)) else values
        values = [v.value if isinstance(v, tl.constexpr) else v for v in values]
        if len(values) != max(1, len(input.shape)):
            raise ValueError(f"len(values) != len(input.shape) for {name}")
        input.handle.set_attr(name, values)
        return input

    scope.set_attr(lang, "range", _new_range)
    scope.set_attr(lang, "static_range", _new_range)
    scope.set_attr(lang, "static_assert", _new_static_assert)
    scope.set_attr(lang, "static_print", print)
    scope.set_attr(lang.dtype, "to_ir", _new_to_ir)
    scope.set_attr(lang, "multiple_of", partial(_set_attr, name="tt.divisibility"))
    scope.set_attr(lang, "max_contiguous", partial(_set_attr, name="tt.contiguity"))
    scope.set_attr(lang, "max_constancy", partial(_set_attr, name="tt.constancy"))

    _patch_reduce_scan(scope)


def _patch_lang(fn):
    """
    Patch the language module in the given function.
    """
    scope = _LangPatchScope()
    langs = [value for _, value in fn.__globals__.items() if inspect.ismodule(value) and value in [tl, tl.core]]
    assert len(langs) >= 1, "tileon.language must be visible from within jit'd function"
    for lang in langs:
        _patch_builtin(lang, interpreter_builder, scope)
        _patch_builtin(lang.tensor, interpreter_builder, scope)
        if lang == tl:
            _patch_builtin(lang.math, interpreter_builder, scope)
        _patch_lang_tensor(lang.tensor, scope)
        _patch_lang_core(lang, scope)
    _patch_builtin(tl.core._tensor_descriptor, interpreter_builder, scope)
    return scope


# TODO: wrap everything in tileon tensors
def _implicit_cvt(arg):
    """Implicitly convert a value to a tileon tensor."""

    if isinstance(arg, int):
        ty = tl.str_to_t(tileon.runtime.jit.mangle_type(arg), None)
        dtype = np.int32
        if -2 ** 31 <= arg < 2 ** 31:
            dtype = np.int32
        elif 2 ** 31 <= arg < 2 ** 32:
            dtype = np.uint32
        elif -2 ** 63 <= arg < 2 ** 63:
            dtype = np.int64
        elif 2 ** 63 <= arg < 2 ** 64:
            dtype = np.uint64
        else:
            raise ValueError(f"Unsupported integer value {arg}")
        handle = TensorHandle(np.array([arg], dtype=dtype), ty)
        return tl.tensor(handle, ty)
    if hasattr(arg, "data_ptr"):
        ty = tl.str_to_t(tileon.runtime.jit.mangle_type(arg), None)
        handle = TensorHandle(np.array([arg.data_ptr()], dtype=np.uint64), ty)
        return tl.tensor(handle, ty)
    elif isinstance(arg, tuple):
        return tuple_create(arg, map(_implicit_cvt, arg))
    elif isinstance(arg, TensorDescriptor):
        strides = [_implicit_cvt(s) for s in arg.strides]
        assert arg.strides[-1] == 1
        strides[-1] = tl.constexpr(1)
        return interpreter_semantic.make_tensor_descriptor(
            base=_implicit_cvt(arg.base),
            shape=[_implicit_cvt(s) for s in arg.shape], 
            strides=strides,
            block_shape=[tl.constexpr(b) for b in arg.block_shape],
            padding_option=arg.padding
        )
    return arg


def _unwrap_tensor(t):
    """Unwrap a tileon tensor wrapper."""
    if isinstance(t, TileonTensor):
        return t.base
    return t


def _rewrap_tensor(t, original_tensor):
    """Rewrap a tileon tensor wrapper."""
    if isinstance(original_tensor, TileonTensor):
        return TileonTensor(t, original_tensor.dtype)
    return t


class GridExecutor:
    """Execute a grid of a function."""

    def __init__(self, fn, arg_names, grid, pre_run_hooks=[]):
        """
        Initialize the grid executor.

        Args:
            fn (function): The function to execute.
            arg_names (list[str]): The names of the arguments.
            grid (list[int]): The grid size.
            pre_run_hooks (list[function], optional): The pre-run hooks. Defaults to [].
        """
        self.fn = fn
        self.arg_names = arg_names
        self.grid = grid
        self.pre_run_hooks = pre_run_hooks
        __annotations__ = {
            name: _normalize_t(ty)
            for name, ty in fn.__annotations__.items()
        }
        self.constexprs = [
            name for name in arg_names
            if __annotations__.get(name) == "constexpr"
        ]

    def _init_args_hst(self, args_dev: list, kwargs: dict):
        """Initialize host arguments from device arguments."""
        storages = {}

        def _to_cpu(arg):
            if isinstance(arg, tuple):
                return tuple_create(arg, map(_to_cpu, arg))
            elif isinstance(arg, TensorDescriptor):
                return TensorDescriptor(
                    _to_cpu(arg.base),
                    arg.shape,
                    arg.strides,
                    arg.block_shape,
                    arg.padding,
                    arg.round_f32_to_tf32,
                )
            elif not hasattr(arg, "data_ptr"):
                return arg

            unwrapped_arg = _unwrap_tensor(arg)
            # Copy the tensor to CPU if it's not already on CPU
            if unwrapped_arg.untyped_storage().data_ptr() not in storages:
                storage = unwrapped_arg.untyped_storage()
                storages[storage.data_ptr()] = storage.cpu()

            storage = storages[unwrapped_arg.untyped_storage().data_ptr()]
            cpu_arg = unwrapped_arg.new_empty(0, device='cpu')
            cpu_arg.set_(storage, unwrapped_arg.storage_offset(), unwrapped_arg.size(), unwrapped_arg.stride())
            cpu_arg = _rewrap_tensor(cpu_arg, original_tensor=arg)
            return cpu_arg

        args_hst = [_to_cpu(arg) for arg in args_dev]

        # Process keyword arguments
        kwargs_hst = {}
        for key, value in kwargs.items():
            kwargs_hst[key] = _to_cpu(value)
        return args_hst, kwargs_hst

    def _restore_args_dev(self, args_dev: list, args_hst: list, kwargs: dict,
                          kwargs_hst: dict):
        """Restore device arguments from host arguments."""
        storages = {}

        def _from_cpu(arg_dev, arg_hst):
            if hasattr(arg_dev, "data_ptr"):
                arg_dev, arg_hst = _unwrap_tensor(arg_dev), _unwrap_tensor(arg_hst)
                storages[arg_dev.untyped_storage().data_ptr()] = (
                    arg_dev.untyped_storage(), arg_hst.untyped_storage()
                )
            elif isinstance(arg_dev, tuple):
                for (arg_dev, arg_hst) in zip(arg_dev, arg_hst):
                    _from_cpu(arg_dev, arg_hst)
            elif isinstance(arg_dev, TensorDescriptor):
                _from_cpu(arg_dev.base, arg_hst.base)

        for arg_dev, arg_hst in zip(args_dev, args_hst):
            _from_cpu(arg_dev, arg_hst)

        # Restore keyword arguments
        for key, kwarg_dev in kwargs.items():
            kwarg_hst = kwargs_hst[key]
            _from_cpu(kwarg_dev, kwarg_hst)

        for (arg_dev, arg_hst) in storages.values():
            arg_dev.copy_(arg_hst)

    def __call__(self, *args_dev, **kwargs):
        # filter out keyword arguments that are not in the function signature
        argspec = inspect.getfullargspec(self.fn)
        kwargs = {k: v for k, v in kwargs.items() if k in argspec.args}

        # copy arguments to the host
        args_hst, kwargs_hst = self._init_args_hst(args_dev, kwargs)

        for hook in self.pre_run_hooks:
            hook(*args_hst, **kwargs_hst)

        # remaps core language functions to interpreted ones
        patch_scope = _patch_lang(self.fn)
        try:
            args = inspect.getcallargs(self.fn, *args_hst, **kwargs_hst)
            args = {
                name: arg if name in self.constexprs else _implicit_cvt(arg)
                for name, arg in args.items()
            }

            grid = self.grid(args) if callable(self.grid) else self.grid
            assert len(grid) <= 3, "grid must have at most 3 dimensions"
            grid = grid + (1, ) * (3 - len(grid))

            if grid[1] == 1 and grid[2] == 1:
                interpreter_builder.set_vectorized_grid(grid[0])
                try:
                    self.fn(**args)
                finally:
                    interpreter_builder.clear_vectorized_grid()
            else:
                interpreter_builder.set_grid_dim(*grid)
                try:
                    _interpreter.parallel_launch(lambda: self.fn(**args), list(grid), interpreter_builder)
                except Exception as e:
                    if knobs.runtime.debug:
                        raise
                    raise InterpreterError(repr(e)) from e
        finally:
            patch_scope.restore()
        # copy arguments back to propagate side-effects
        self._restore_args_dev(args_dev, args_hst, kwargs, kwargs_hst)


class ASTTransformer(ast.NodeTransformer):
    """
    AST transformer to rewrite assignments to tensors to use the interpreter semantic.
    """

    def visit_Assign(self, node):
        names = []
        for target in node.targets:
            names += [self.visit(target)]
        if len(names) > 1:
            raise ValueError("Multiple assignments are not supported")
        # x = value -> interpreter_semantic.to_tensor(value, False)
        node.value = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="interpreter_semantic", ctx=ast.Load()), 
                attr="to_tensor",
                ctx=ast.Load()
            ), 
            args=[node.value, ast.Constant(value=False)], 
            keywords=[]
        )
        return node


class FunctionRewriter:
    """
    Function rewriter to rewrite assignments to tensors to use the interpreter semantic.
    """
    ast_transformer = ASTTransformer()

    def __init__(self, fn, **kwargs):
        self.fn = fn
        self.kwargs = kwargs
        self.filename: str = ""
        # Absolute line number in the file
        self.def_file_lineno: int = 0

    def _get_jit_fn_file_line(self) -> Tuple[str, int]:
        from .jit import get_jit_fn_file_line
        return get_jit_fn_file_line(JITCallable(self.fn))

    def _find_def(self, lines: List[str]) -> int:
        def_lineno = 0
        # Line numbers start from 1
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                def_lineno = i + 1
        return def_lineno

    def _prepare_source(self, lines: List[str]) -> str:
        lines = lines[self.def_lineno - 1:]
        src = ''.join(lines)
        return textwrap.dedent(src)

    def _transform_ast(self, src: str) -> ast.AST:
        parsed_ast = ast.parse(src)
        transformed_ast = self.ast_transformer.visit(parsed_ast)
        ast.fix_missing_locations(transformed_ast)
        inc_lineno = self.def_file_lineno - 1
        ast.increment_lineno(transformed_ast, inc_lineno)
        return transformed_ast

    def _compile_and_exec(self, transformed_ast: ast.AST) -> Callable:
        compiled_code = compile(transformed_ast, filename=self.filename, mode='exec')
        local_namespace = {**self.kwargs}
        fn_globals = self.fn.__globals__
        for key, value in globals().items():
            if key not in fn_globals:
                fn_globals[key] = value
        exec(compiled_code, fn_globals, local_namespace)
        return local_namespace[self.fn.__name__]

    def rewrite_ast(self):
        try:
            lines, _ = inspect.getsourcelines(self.fn)
        except Exception:
            # it is dynamically generated function
            return self.fn

        self.filename, self.def_file_lineno = self._get_jit_fn_file_line()
        self.def_lineno = self._find_def(lines)
        src = self._prepare_source(lines)
        transformed_ast = self._transform_ast(src)
        return self._compile_and_exec(transformed_ast)


class FunctionRewriter:
    """
    Function rewriter to rewrite assignments to tensors to use the interpreter semantic.
    """
    ast_transformer = ASTTransformer()

    def __init__(self, fn, **kwargs):
        self.fn = fn
        self.kwargs = kwargs
        self.filename: str = ""
        # Absolute line number in the file
        self.def_file_lineno: int = 0

    def _get_jit_fn_file_line(self) -> Tuple[str, int]:
        from .jit import get_jit_fn_file_line
        return get_jit_fn_file_line(JITCallable(self.fn))

    def _find_def(self, lines: List[str]) -> int:
        def_lineno = 0
        # Line numbers start from 1
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                def_lineno = i + 1
        return def_lineno

    def _prepare_source(self, lines: List[str]) -> str:
        lines = lines[self.def_lineno - 1:]
        src = ''.join(lines)
        return textwrap.dedent(src)

    def _transform_ast(self, src: str) -> ast.AST:
        parsed_ast = ast.parse(src)
        transformed_ast = self.ast_transformer.visit(parsed_ast)
        ast.fix_missing_locations(transformed_ast)
        inc_lineno = self.def_file_lineno - 1
        ast.increment_lineno(transformed_ast, inc_lineno)
        return transformed_ast

    def _compile_and_exec(self, transformed_ast: ast.AST) -> Callable:
        compiled_code = compile(transformed_ast, filename=self.filename, mode='exec')
        local_namespace = {**self.kwargs}
        fn_globals = self.fn.__globals__
        for key, value in globals().items():
            if key not in fn_globals:
                fn_globals[key] = value
        exec(compiled_code, fn_globals, local_namespace)
        return local_namespace[self.fn.__name__]

    def rewrite_ast(self):
        try:
            lines, _ = inspect.getsourcelines(self.fn)
        except Exception:
            # it is dynamically generated function
            return self.fn

        self.filename, self.def_file_lineno = self._get_jit_fn_file_line()
        self.def_lineno = self._find_def(lines)
        src = self._prepare_source(lines)
        transformed_ast = self._transform_ast(src)
        return self._compile_and_exec(transformed_ast)


class InterpretedFunction(KernelInterface[T]):
    """A wrapper for interpreted functions."""

    rewritten_fn: Dict[Callable, Callable] = {}

    def __init__(self, fn, **kwargs):
        self.fn = fn
        self.rewriter = FunctionRewriter(fn, **kwargs)
        self.kwargs = kwargs
        self.pre_run_hooks = []

        signature = inspect.signature(fn)
        self.arg_names = [v.name for v in signature.parameters.values()]

    def run(self, *args, grid, warmup, **kwargs):
        if warmup:
            return
        fn = self.rewrite()
        return GridExecutor(fn, self.arg_names, grid, self.pre_run_hooks)(*args, **kwargs)

    def add_pre_run_hook(self, hook):
        assert callable(hook)
        self.pre_run_hooks.append(hook)

    def rewrite(self):
        if self.fn not in self.rewritten_fn:
            self.rewritten_fn[self.fn] = self.rewriter.rewrite_ast()
        return self.rewritten_fn[self.fn]

    @property
    def __name__(self):
        return self.fn.__name__

    def __call__(self, *args, **kwargs):
        # This is a device function call
        _patch_lang(self.fn)
        fn = self.rewrite()
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            raise InterpreterError(repr(e)) from e
