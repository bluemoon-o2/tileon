---
title: 简介
---

# 简介

## 什么是 Tileon?

Tileon 是一个用于并行编程的语言和编译器。它旨在提供一个基于 Python 的编程环境，用于高效编写能够在现代 GPU 硬件上以最大吞吐量运行的自定义 DNN 计算内核。

受 Triton 启发，Tileon 提供了一种直观的方式来编写高性能并行计算内核，同时保持代码的可读性和易用性。

## 核心概念

1. **基于瓦片的编程**: 将数据分成瓦片（块）进行并行处理
2. **即时编译**: 内核使用 `@tileon.jit` 装饰器在运行时编译
3. **基于 Python 的 DSL**: 使用熟悉的 Python 语法编写内核
4. **自动并行化**: 编译器处理并行执行细节

## 为什么选择 Tileon?

- **生产力**: 基于 Python 的 DSL 用于更快的开发
- **性能**: 与 CUDA/C++ 实现相当
- **可读性**: 清晰、易于理解的内核代码
- **灵活性**: 易于定制和扩展

## 安装

请参阅 [安装指南](../../getting-started/installation.md) 获取详细的安装说明。

## 快速示例

```python
import torch
import tileon
import tileon.language as tl

@tileon.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (tileon.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

## 下一步

- [向量加法](../../getting-started/tutorials/vector-add.md) - 您的第一个 Tileon 内核
- [矩阵乘法](../../getting-started/tutorials/matrix-multiplication.md) - GEMM 实现
- [tileon.language](../../python-api/tileon.language.md) - 语言参考
