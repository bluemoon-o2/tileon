---
title: tileon.language
---

# tileon.language

## 内存操作

### load

从内存加载数据，支持可选掩码。

```python
x = tl.load(ptr + offsets, mask=mask, other=0.0)
```

### store

将数据存储到内存，支持可选掩码。

```python
tl.store(ptr + offsets, data, mask=mask)
```

### make_block_ptr

创建块指针以实现高效的内存访问。

### tensor_descriptor

为复杂内存布局创建张量描述符。

## 索引操作

### arange

创建索引范围。

```python
offsets = tl.arange(0, BLOCK_SIZE)
```

### program_id

获取当前程序实例的 ID。

```python
pid = tl.program_id(axis=0)
```

### range / static_range

创建静态迭代范围。

### full / zeros / zeros_like

创建填充的张量。

```python
data = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
```

## 张量操作

### broadcast / broadcast_to

将张量广播到新形状。

### expand_dims / reshape / slice / split / squeeze / transpose / unsqueeze / view

张量形状操作。

### join / cat

连接张量。

## 数学操作

### dot

瓦片的矩阵乘法。

```python
c = tl.dot(a, b)
```

### fma

融合乘加。

### add / sub / mul / div

逐元素算术运算。

## 比较操作

### maximum / minimum

逐元素最大/最小运算。

### where

条件选择。

```python
result = tl.where(condition, x, y)
```

## 归约操作

### sum / max / min

跨轴归约运算。

```python
result = tl.sum(x, axis=0)
```

### argmax / argmin

返回最大/最小值的索引。

### reduce

通用归约操作。

## 排序操作

### sort / topk

排序操作。

## 工具函数

### cdiv

向上取整除法。

### constexpr

编译时常量。

### num_programs

获取程序实例的总数。
