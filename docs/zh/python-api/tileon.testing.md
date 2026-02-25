---
title: tileon.testing
---

# tileon.testing

## 测试和基准测试

Tileon 提供用于验证正确性和基准测试性能的测试工具。

### Benchmark

基准测试报告配置类。

```python
from tileon.testing import Benchmark

Benchmark(
    x_names=['size'],
    x_vals=[128, 256, 512],
    line_arg='provider',
    line_vals=['tileon', 'torch', 'numpy'],
    ylabel='GFLOPS',
)
```

### do_bench

运行基准测试并返回执行时间。

```python
ms = do_bench(lambda: kernel[grid](...))
```

### perf_report

用于生成性能报告的装饰器。

```python
@tileon.testing.perf_report(
    Benchmark(...)
)
def benchmark_gemm(size, provider):
    ...
```

### assert_close

断言两个值在容差范围内接近。

```python
assert_close(actual, expected, atol=1e-3)
```
