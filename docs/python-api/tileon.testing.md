---
title: tileon.testing
---

# tileon.testing

## Testing and Benchmarking

Tileon provides testing utilities for verifying correctness and benchmarking performance.

### Benchmark

Configuration class for benchmark reporting.

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

Run a benchmark and return execution time.

```python
ms = do_bench(lambda: kernel[grid](...))
```

### perf_report

Decorator for generating performance reports.

```python
@tileon.testing.perf_report(
    Benchmark(...)
)
def benchmark_gemm(size, provider):
    ...
```

### assert_close

Assert that two values are close within tolerance.

```python
assert_close(actual, expected, atol=1e-3)
```
