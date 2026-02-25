---
title: 安装
---

# 安装

## 从源码安装

您可以通过运行以下命令从源码安装 Tileon：

```bash
git clone https://github.com/bluemoon-o2/tileon.git
cd tileon
pip install -e .
```

## 环境要求

- Python 3.10 或更高版本
- PyTorch（用于张量操作）
- NumPy（用于基准测试）

```bash
pip install torch numpy
```

## 测试安装

您可以通过运行测试来验证安装：

```bash
# 运行所有测试
pytest tileon/test/ -v

# 运行特定测试文件
pytest tileon/test/test_kernels.py -v

# 运行基准测试
python tileon/test/test_kernels.py
```

## 验证设置

要验证您的安装是否正常工作：

```python
import torch
import tileon
import tileon.language as tl

# 简单的向量加法
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

# 测试
x = torch.rand(1024)
y = torch.rand(1024)
output = torch.empty(1024)

n_elements = output.numel()
grid = lambda meta: (tileon.cdiv(n_elements, meta['BLOCK_SIZE']), )
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

print("安装成功！")
```
