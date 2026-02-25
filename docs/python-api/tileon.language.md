---
title: tileon.language
---

# tileon.language

## Memory Operations

### load

Load data from memory with optional masking.

```python
x = tl.load(ptr + offsets, mask=mask, other=0.0)
```

### store

Store data to memory with optional masking.

```python
tl.store(ptr + offsets, data, mask=mask)
```

### make_block_ptr

Create a block pointer for efficient memory access.

### tensor_descriptor

Create a tensor descriptor for complex memory layouts.

## Indexing

### arange

Create a range of indices.

```python
offsets = tl.arange(0, BLOCK_SIZE)
```

### program_id

Get the ID of the current program instance.

```python
pid = tl.program_id(axis=0)
```

### range / static_range

Create a static range for iteration.

### full / zeros / zeros_like

Create filled tensors.

```python
data = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
```

## Tensor Operations

### broadcast / broadcast_to

Broadcast tensor to a new shape.

### expand_dims / reshape / slice / split / squeeze / transpose / unsqueeze / view

Tensor shape manipulation operations.

### join / cat

Concatenate tensors.

## Math Operations

### dot

Matrix multiplication for tiles.

```python
c = tl.dot(a, b)
```

### fma

Fused multiply-add.

### add / sub / mul / div

Element-wise arithmetic operations.

## Comparison

### maximum / minimum

Element-wise max/min operations.

### where

Conditional selection.

```python
result = tl.where(condition, x, y)
```

## Reduction

### sum / max / min

Reduction operations across an axis.

```python
result = tl.sum(x, axis=0)
```

### argmax / argmin

Return indices of max/min values.

### reduce

General reduction operation.

## Sorting

### sort / topk

Sorting operations.

## Utility

### cdiv

Ceiling division.

### constexpr

Compile-time constant.

### num_programs

Get total number of program instances.
