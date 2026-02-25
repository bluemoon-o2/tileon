---
title: tileon.language.random
---

# tileon.language.random

## Random Number Generation

Tileon provides Philox-based random number generation for generating random values in kernels.

### rand

Generate uniform random values in [0, 1).

```python
import tileon.language.random as tl_random

@tileon.jit
def random_kernel(output_ptr, n_elements, seed: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    random_values = tl_random.rand(seed, offsets, n_rounds=10)

    tl.store(output_ptr + offsets, random_values, mask=mask)
```

### randn

Generate values from normal distribution N(0, 1).

### randint

Generate random integer values.

## Utility Functions

- `philox`: Core Philox PRNG implementation
- `uint_to_uniform_float`: Convert uint to uniform float [0, 1)
- `pair_uniform_to_normal`: Convert uniform pair to normal distribution
