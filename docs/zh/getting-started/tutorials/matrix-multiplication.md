---
title: 矩阵乘法
---

# 矩阵乘法

本教程演示如何使用 Tileon 实现矩阵乘法 (GEMM)。

## 前置条件

- 完成 [向量加法](vector-add.md) 教程
- 理解基于瓦片的编程

## 简介

矩阵乘法是深度学习中的基本操作。Tileon 通过 `tl.dot` 函数提供高效的基于瓦片的矩阵乘法。

## 基本 GEMM 内核

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

## 代码说明

1. **基于瓦片的计算**: 矩阵被分成块（瓦片）进行并行处理。

2. **2D 网格**: 计算 `pid_m` 和 `pid_n` 以处理 2D 瓦片网格。

3. **累加器**: 我们初始化一个累加器来存储部分结果。

4. **K 维循环**: 内层循环处理 K 维的分块。

5. **`tl.dot`**: 在瓦片上执行高效的矩阵乘法。

6. **掩码**: 确保不会访问越界内存。

## 运行示例

```python
a = torch.rand(512, 256)
b = torch.rand(256, 512)
c = matmul(a, b)

# 验证结果
expected = torch.matmul(a, b)
assert torch.allclose(c, expected, atol=1e-3)
```

## 性能提示

1. **块大小**: 根据硬件选择 BLOCK_M、BLOCK_N、BLOCK_K
2. **内存访问**: 确保合并的内存访问模式
3. **共享内存**: 对频繁访问的数据使用共享内存

## 练习

1. 将偏置向量添加到 GEMM 输出
2. 实现转置矩阵乘法
3. 优化特定块大小并比较性能
