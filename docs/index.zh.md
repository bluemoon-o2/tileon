---
title: 快速开始
---

<div align="center">
    <p>
        <img src="./assets/tileon-logo.png" alt="Tileon" width="200">
    </p>

<!-- Language Switch -->
<p>
    <a href="./index.md">
        <img src="https://img.shields.io/badge/English-🇺🇸-yellow?style=flat-square" alt="English">
    </a>
    <a href="./index.zh.md">
        <img src="https://img.shields.io/badge/中文-🇨🇳-blue?style=flat-square" alt="中文">
    </a>
</p>

<!-- Platform & Build -->
<p>
    <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python 版本">
    <img src="https://img.shields.io/badge/platform-Win%20|%20Linux-purple" alt="平台">
    <img src="https://img.shields.io/badge/hardware-CPU-green" alt="硬件">
</p>

<!-- Package & Stats -->
<p>
    <img src="https://img.shields.io/badge/License-MIT-green?logo=apache" alt="许可证">
    <img src="https://img.shields.io/github/stars/bluemoon-o2/tileon?style=flat&logo=github&color=yellow" alt="GitHub Stars">
</p>
</div>

## ✨ 特性

- **Python 领域特定语言**: 使用熟悉的 Python 语法编写 GPU 内核
- **基于瓦片 (Tile) 的编程模型**: 通过基于瓦片的计算实现高效并行执行
- **即时编译 (JIT)**: 使用 `@tileon.jit` 装饰器进行动态内核编译
- **内置数学运算**: 全面的数学库 (`tileon.language.math`)
- **随机数生成**: 基于 Philox 的 RNG，支持 `rand`、`randn` 和 `randint` (`tileon.language.random`)
- **自动并行化**: 简单的基于网格的执行模型
- **互操作性**: 与 PyTorch 张量无缝集成

## 🚀 安装

```bash
git clone https://github.com/bluemoon-o2/tileon.git
cd tileon
pip install -e .
```

## ⚡ 快速开始

以下是一个简单的向量加法示例：

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

## 📦 支持的操作

### 线性代数
- **GEMM**: 通用矩阵乘法 (`tl.dot`)
- **向量运算**: 逐元素加法、乘法等

### 神经网络原语
- **Softmax**: 逐行 softmax 计算
- **Flash Attention**: 高效注意力机制
- **块稀疏注意力**: 具有自定义块模式的稀疏注意力

### 数学函数
- **算术**: `add`, `sub`, `mul`, `div`, `fma`
- **比较**: `maximum`, `minimum`, `where`
- **数学**: `exp`, `log`, `sqrt`, `sin`, `cos`, `pow`
- **归约**: `sum`, `max`, `min`, `argmax`

### 随机数生成
- **rand**: 均匀分布 [0, 1)
- **randn**: 正态分布 N(0, 1)
- **randint**: 整数随机值

## 🏗️ 编程模型

### 内核定义

使用 `@tileon.jit` 装饰器定义内核：

```python
@tileon.jit
def kernel_name(x_ptr, y_ptr, ..., BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * 2
    tl.store(y_ptr + offsets, y, mask=mask)
```

### 执行

使用网格规范启动内核：

```python
grid = lambda meta: (tileon.cdiv(N, meta['BLOCK_SIZE']), )
kernel[grid](x, y, BLOCK_SIZE=128)
```

## 🧪 运行测试

```bash
# 运行所有测试
pytest tileon/test/ -v

# 运行特定测试文件
pytest tileon/test/test_kernels.py -v

# 运行基准测试
python tileon/test/test_kernels.py
```

## 📚 API 参考

### 核心函数
- `tileon.jit`: JIT 内核编译装饰器
- `tileon.cdiv`: 向上取整除法
- `tileon.program_id`: 获取当前程序 ID
- `tileon.next_power_of_2`: 向上取整到下一个 2 的幂

### 张量操作 (`tileon.language`)
- `tl.load`: 从内存加载数据
- `tl.store`: 将数据存储到内存
- `tl.arange`: 创建索引范围
- `tl.dot`: 矩阵乘法

### 数学函数 (`tileon.language.math`)
- `tl.exp`, `tl.log`, `tl.sqrt`
- `tl.sin`, `tl.cos`, `tl.pow`

### 随机函数 (`tileon.language.random`)
- `tl_random.rand`: 均匀分布随机数 [0, 1)
- `tl_random.randn`: 正态分布 N(0, 1)
- `tl_random.randint`: 整数随机值
