---
title: Installation
---

# Installation

## From Source

You can install Tileon from source by running the following commands:

```bash
git clone https://github.com/bluemoon-o2/tileon.git
cd tileon
pip install -e .
```

## Requirements

- Python 3.10 or higher
- PyTorch (for tensor operations)
- NumPy (for benchmarking)

```bash
pip install torch numpy
```

## Testing Installation

You can test your installation by running the tests:

```bash
# Run all tests
pytest tileon/test/ -v

# Run specific test file
pytest tileon/test/test_kernels.py -v

# Run benchmark
python tileon/test/test_kernels.py
```

## Verifying Setup

To verify your installation is working correctly:

```python
import torch
import tileon
import tileon.language as tl

# Simple vector addition
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

# Test it
x = torch.rand(1024)
y = torch.rand(1024)
output = torch.empty(1024)

n_elements = output.numel()
grid = lambda meta: (tileon.cdiv(n_elements, meta['BLOCK_SIZE']), )
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

print("Installation successful!")
```
