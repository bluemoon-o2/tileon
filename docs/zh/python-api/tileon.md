---
title: tileon
---

# tileon

## 核心函数

### jit

Tileon 内核的 JIT 编译器装饰器。

```python
@tileon.jit
def kernel_name(x_ptr, y_ptr, ...):
    ...
```

### cdiv

向上取整除法函数。

```python
result = tileon.cdiv(n, block_size)
```

### program_id

获取当前程序实例的 ID。

```python
pid = tl.program_id(axis=0)
```

### next_power_of_2

向上取整到下一个 2 的幂。

```python
result = tileon.next_power_of_2(n)
```

### testing

测试和基准测试工具模块。
