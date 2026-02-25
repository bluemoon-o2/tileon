"""
Vector Addition
===============

In this tutorial, you will write a simple vector addition using Triton.

In doing so, you will learn about:

* The basic programming model of Triton.

* The `tileon.jit` decorator, which is used to define Triton kernels.

* The best practices for validating and benchmarking your custom ops against native reference implementations.

"""
import torch
import os

import tileon
import tileon.knobs
import tileon.language as tl

DEVICE = torch.device("cpu")

@tileon.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
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
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (tileon.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')


@tileon.testing.perf_report(
    tileon.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 16, 1)] if tileon.knobs.runtime.interpret else [2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['tileon', 'torch', 'numpy'], 
        line_names=['Tileon', 'Torch', 'NumPy'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'tileon':
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    if provider == 'numpy':
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: x.numpy() + y.numpy(), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, save_path=os.path.join(os.path.dirname(__file__), 'figures', 'add'))
