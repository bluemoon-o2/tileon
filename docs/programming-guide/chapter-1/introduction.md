---
title: Introduction
---

# Introduction

## What is Tileon?

Tileon is a language and compiler for parallel programming. It aims to provide a Python-based programming environment for productively writing custom DNN compute kernels capable of running at maximal throughput on modern GPU hardware.

Inspired by Triton, Tileon offers an intuitive way to write high-performance parallel compute kernels while maintaining readability and ease of use.

## Key Concepts

1. **Tile-based Programming**: Divides data into tiles (blocks) that can be processed in parallel
2. **Just-in-Time Compilation**: Kernels are compiled at runtime using the `@tileon.jit` decorator
3. **Python-based DSL**: Write kernels using familiar Python syntax
4. **Automatic Parallelization**: The compiler handles parallel execution details

## Why Tileon?

- **Productivity**: Python-based DSL for faster development
- **Performance**: Comparable to CUDA/C++ implementations
- **Readability**: Clear, understandable kernel code
- **Flexibility**: Easy to customize and extend

## Installation

See [Installation](../../getting-started/installation.md) for detailed installation instructions.

## Quick Example

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

## Next Steps

- [Vector Addition](../../getting-started/tutorials/vector-add.md) - Your first Tileon kernel
- [Matrix Multiplication](../../getting-started/tutorials/matrix-multiplication.md) - GEMM implementation
- [tileon.language](../../python-api/tileon.language.md) - Language reference
