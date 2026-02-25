---
title: Vector Addition
---

# Vector Addition

This tutorial demonstrates how to write your first Tileon kernel for vector addition.

## Prerequisites

- Tileon installed (see [Installation](../installation.md))
- Basic understanding of Python

## Basic Example

Here's a simple vector addition kernel:

```python
import torch
import tileon
import tileon.language as tl

DEVICE = torch.device("cpu")

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

## Code Explanation

1. **Kernel Definition**: The `@tileon.jit` decorator compiles the kernel to optimized code.

2. **Program ID**: `tl.program_id(axis=0)` returns the unique ID of each program instance.

3. **Index Calculation**: `offsets` computes which elements this program instance processes.

4. **Memory Operations**:
   - `tl.load` reads data from memory with masking for boundary conditions
   - `tl.store` writes results back to memory

5. **Grid Launch**: The `grid` function determines how many program instances to launch.

## Running the Example

```python
x = torch.rand(1024)
y = torch.rand(1024)
z = add(x, y)

# Verify result
assert torch.allclose(z, x + y)
```

## Exercises

1. Modify the kernel to perform vector multiplication instead of addition
2. Change the BLOCK_SIZE and observe the performance difference
3. Add a third input vector and perform element-wise addition of three vectors
