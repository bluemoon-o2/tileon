---
title: tileon.language.random
---

# tileon.language.random

## 随机数生成

Tileon 提供基于 Philox 的随机数生成，用于在内核中生成随机值。

### rand

生成 [0, 1) 范围内的均匀随机值。

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

从正态分布 N(0, 1) 生成值。

### randint

生成随机整数值。

## 工具函数

- `philox`: 核心 Philox 伪随机数生成器实现
- `uint_to_uniform_float`: 将无符号整数转换为均匀浮点数 [0, 1)
- `pair_uniform_to_normal`: 将均匀数对转换为正态分布
