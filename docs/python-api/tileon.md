---
title: tileon
---

# tileon

## Core Functions

### jit

JIT compiler decorator for Tileon kernels.

```python
@tileon.jit
def kernel_name(x_ptr, y_ptr, ...):
    ...
```

### cdiv

Ceiling division function.

```python
result = tileon.cdiv(n, block_size)
```

### program_id

Get the ID of the current program instance.

```python
pid = tl.program_id(axis=0)
```

### next_power_of_2

Round up to the next power of 2.

```python
result = tileon.next_power_of_2(n)
```

### testing

Testing and benchmarking utilities module.
