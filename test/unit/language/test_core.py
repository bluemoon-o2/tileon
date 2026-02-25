# ruff: noqa: F821,F841
import contextlib
import itertools
import re
from typing import Optional
import math
import textwrap
import inspect

import numpy as np
import pytest
import torch
from numpy.random import RandomState

import tileon
import tileon.language as tl

from tileon._internal_testing import (
    integral_dtypes,
    int_dtypes,
    uint_dtypes,
    float_dtypes,
    dtypes,
    is_interpreter,
    numpy_random,
    to_tileon,
)
from tileon.runtime.errors import InterpreterError


def _bitwidth(dtype: str) -> int:
    # ex.: "int64" -> 64
    return int(re.search(r'(\d+)$', dtype).group(1))


def _dtype(dtype: str) -> str:
    # ex.: "int64" -> "int"
    return re.match(r'([a-zA-Z]+)', dtype).group(0)


def patch_kernel(template, to_replace):
    if is_interpreter():
        local_namespace = {}
        src = textwrap.dedent(inspect.getsource(template.fn))
        for k, v in to_replace.items():
            src = src.replace(k, v)
        # Execute in a context that has 'tl' available
        exec(src, template.fn.__globals__, local_namespace)
        return local_namespace[template.fn.__name__]
    else:
        # TODO: Implement JIT patching if needed for non-interpreter backends
        # For now, Tileon seems to primarily use interpreter for testing in this context
        raise NotImplementedError("Patching kernels for JIT is not yet fully supported in this test harness")


def check_type_supported(dtype, device):
    '''
    skip test if dtype is not supported on the current device
    '''
    if is_interpreter():
        if dtype in [tl.bfloat16, "bfloat16", torch.bfloat16]:
            pytest.skip("bfloat16 is not supported in the interpreter")


# generic test functions
def _test_unary(dtype_x, expr, numpy_expr=None, device='cpu'):
    check_type_supported(dtype_x, device)
    SIZE = 128

    # define the kernel / launch-grid
    @tileon.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})

    # inputs
    x = numpy_random(SIZE, dtype_str=dtype_x)
    # avoid log/sqrt of negative numbers
    if 'log' in expr or 'sqrt' in expr:
        x = np.abs(x) + 0.01

    # reference result
    z_ref = eval(expr if numpy_expr is None else numpy_expr)

    # tileon result
    # For interpreter, we can pass numpy arrays directly or torch tensors
    # Using torch tensors to match likely usage
    x_tri = to_tileon(x, device=device, dst_type=dtype_x)
    z_tri = torch.empty_like(x_tri)

    grid = (1, )
    kernel[grid](z_tri, x_tri, SIZE=SIZE)

    # compare
    np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01)


def _test_binary(dtype_x, dtype_y, expr, numpy_expr=None, device='cpu'):
    check_type_supported(dtype_x, device)
    check_type_supported(dtype_y, device)
    SIZE = 128

    @tileon.jit
    def kernel(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})

    # inputs
    x = numpy_random(SIZE, dtype_str=dtype_x)
    y = numpy_random(SIZE, dtype_str=dtype_y)

    # avoid division by zero
    if '/' in expr or '%' in expr:
        y[y == 0] = 1

    # reference result
    z_ref = eval(expr if numpy_expr is None else numpy_expr)

    # tileon result
    x_tri = to_tileon(x, device=device, dst_type=dtype_x)
    y_tri = to_tileon(y, device=device, dst_type=dtype_y)
    z_tri = torch.empty(SIZE, dtype=x_tri.dtype, device=device)  # Simplification: assume result type matches x

    grid = (1, )
    kernel[grid](z_tri, x_tri, y_tri, SIZE=SIZE)

    # compare
    np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype_x", ["float32", "int32"])
def test_unary_op(dtype_x, device="cpu"):
    _test_unary(dtype_x, "x + 1", device=device)
    _test_unary(dtype_x, "-x", device=device)


@pytest.mark.parametrize("dtype_x, dtype_y", [("float32", "float32"), ("int32", "int32")])
def test_binary_op(dtype_x, dtype_y, device="cpu"):
    _test_binary(dtype_x, dtype_y, "x + y", device=device)
    _test_binary(dtype_x, dtype_y, "x - y", device=device)
    _test_binary(dtype_x, dtype_y, "x * y", device=device)
