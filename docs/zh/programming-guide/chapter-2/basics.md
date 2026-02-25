---
title: 内核编程基础
---

# 内核编程基础

本章介绍编写 Tileon 内核的基础知识。

## 内核函数

Tileon 内核是用 `@tileon.jit` 装饰的 Python 函数：

```python
@tileon.jit
def kernel_name(x_ptr, y_ptr, ..., BLOCK_SIZE: tl.constexpr):
    # 内核代码
    ...
```

`@tileon.jit` 装饰器告诉 Tileon 将此函数编译成优化的内核代码。

## 程序 ID

每个内核实例都由一个程序 ID 标识：

```python
pid = tl.program_id(axis=0)
```

- `axis=0`: 第一维
- `axis=1`: 第二维
- `axis=2`: 第三维

## 索引计算

计算要处理的数据元素：

```python
# 简单的一维情况
pid = tl.program_id(axis=0)
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)

# 二维情况
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
```

## 内存操作

加载和存储数据：

```python
# 带掩码的加载
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

# 带掩码的存储
tl.store(output_ptr + offsets, output, mask=mask)
```

## 掩码

掩码确保安全访问边界元素：

```python
mask = offsets < n_elements
x = tl.load(ptr + offsets, mask=mask)
```

## 常量

使用 `tl.constexpr` 表示编译时常量：

```python
BLOCK_SIZE: tl.constexpr = 1024
```

这允许 Tileon 在编译时优化内核。

## 网格启动

使用网格启动内核：

```python
grid = lambda meta: (tileon.cdiv(N, meta['BLOCK_SIZE']), )
kernel[grid](x, y, BLOCK_SIZE=1024)
```

网格函数计算要启动多少个程序实例。

## 数据类型

常用数据类型：

```python
tl.float32
tl.float16
tl.int32
tl.int64
tl.uint32
```

## 向量操作

逐元素操作：

```python
# 算术运算
z = x + y
z = x - y
z = x * y
z = x / y

# 比较运算
z = tl.where(cond, x, y)
z = tl.maximum(x, y)
z = tl.minimum(x, y)
```

## 归约操作

跨维度归约：

```python
# 求和归约
result = tl.sum(x, axis=0)

# 最大值归约
result = tl.max(x, axis=0)
```
