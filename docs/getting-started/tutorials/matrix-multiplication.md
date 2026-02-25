---
title: Matrix Multiplication
---

# Matrix Multiplication

This tutorial demonstrates how to implement matrix multiplication (GEMM) using Tileon.

## Prerequisites

- Complete the [Vector Addition](vector-add.md) tutorial
- Understanding of tile-based programming

## Introduction

Matrix multiplication is a fundamental operation in deep learning. Tileon provides efficient tile-based matrix multiplication through the `tl.dot` function.

## Basic GEMM Kernel

```python
import torch
import tileon
import tileon.language as tl

@tileon.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, M - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    c = accumulator
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), dtype=torch.float32)
    grid = lambda META: (tl.cdiv(M, META['BLOCK_M']) * tl.cdiv(N, META['BLOCK_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=128, BLOCK_N=256, BLOCK_K=64
    )
    return c
```

## Code Explanation

1. **Tile-based Computation**: The matrix is divided into blocks (tiles) for parallel processing.

2. **2D Grid**: Both `pid_m` and `pid_n` are computed to process a 2D grid of tiles.

3. **Accumulator**: We initialize an accumulator to store partial results.

4. **Loop over K**: The inner loop processes chunks of the K dimension.

5. **tl.dot**: Performs efficient matrix multiplication on tiles.

6. **Masking**: Ensures we don't access out-of-bounds memory.

## Running the Example

```python
a = torch.rand(512, 256)
b = torch.rand(256, 512)
c = matmul(a, b)

# Verify result
expected = torch.matmul(a, b)
assert torch.allclose(c, expected, atol=1e-3)
```

## Performance Tips

1. **Block Size**: Choose BLOCK_M, BLOCK_N, BLOCK_K based on your hardware
2. **Memory Access**: Ensure coalesced memory access patterns
3. **Shared Memory**: Use shared memory for frequently accessed data

## Exercises

1. Add bias vector to the GEMM output
2. Implement a transposed matrix multiplication
3. Optimize for specific block sizes and compare performance
