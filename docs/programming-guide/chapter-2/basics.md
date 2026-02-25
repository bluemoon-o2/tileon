---
title: Kernel Programming Basics
---

# Kernel Programming Basics

This chapter covers the fundamentals of writing Tileon kernels.

## Kernel Functions

Tileon kernels are Python functions decorated with `@tileon.jit`:

```python
@tileon.jit
def kernel_name(x_ptr, y_ptr, ..., BLOCK_SIZE: tl.constexpr):
    # Kernel code here
    ...
```

The `@tileon.jit` decorator tells Tileon to compile this function into optimized kernel code.

## Program ID

Each kernel instance is identified by a program ID:

```python
pid = tl.program_id(axis=0)
```

- `axis=0`: First dimension
- `axis=1`: Second dimension
- `axis=2`: Third dimension

## Index Calculation

Calculate which data elements to process:

```python
# Simple 1D case
pid = tl.program_id(axis=0)
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)

# 2D case
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
```

## Memory Operations

Load and store data:

```python
# Load with masking
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

# Store with masking
tl.store(output_ptr + offsets, output, mask=mask)
```

## Masking

Masks ensure safe access to boundary elements:

```python
mask = offsets < n_elements
x = tl.load(ptr + offsets, mask=mask)
```

## Constants

Use `tl.constexpr` for compile-time constants:

```python
BLOCK_SIZE: tl.constexpr = 1024
```

This allows Tileon to optimize the kernel at compile time.

## Grid Launch

Launch kernels with a grid:

```python
grid = lambda meta: (tileon.cdiv(N, meta['BLOCK_SIZE']), )
kernel[grid](x, y, BLOCK_SIZE=1024)
```

The grid function computes how many program instances to launch.

## Data Types

Common data types:

```python
tl.float32
tl.float16
tl.int32
tl.int64
tl.uint32
```

## Vector Operations

Element-wise operations:

```python
# Arithmetic
z = x + y
z = x - y
z = x * y
z = x / y

# Comparison
z = tl.where(cond, x, y)
z = tl.maximum(x, y)
z = tl.minimum(x, y)
```

## Reduction

Reduce across a dimension:

```python
# Sum reduction
result = tl.sum(x, axis=0)

# Max reduction
result = tl.max(x, axis=0)
```
