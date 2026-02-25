---
title: 向量加法
---

# 向量加法

本教程演示如何编写您的第一个 Tileon 内核进行向量加法。

## 前置条件

- 已安装 Tileon（参见 [安装](../installation.md)）
- 具备 Python 基础

## 基本示例

这是一个简单的向量加法内核：

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

## 代码说明

1. **内核定义**: `@tileon.jit` 装饰器将内核编译为优化代码。

2. **程序 ID**: `tl.program_id(axis=0)` 返回每个程序实例的唯一 ID。

3. **索引计算**: `offsets` 计算此程序实例处理哪些元素。

4. **内存操作**:
   - `tl.load` 使用掩码从内存读取数据，处理边界条件
   - `tl.store` 将结果写回内存

5. **网格启动**: `grid` 函数决定启动多少个程序实例。

## 运行示例

```python
x = torch.rand(1024)
y = torch.rand(1024)
z = add(x, y)

# 验证结果
assert torch.allclose(z, x + y)
```

## 练习

1. 修改内核改为向量乘法而不是加法
2. 更改 BLOCK_SIZE 并观察性能差异
3. 添加第三个输入向量并执行三个向量的逐元素加法
