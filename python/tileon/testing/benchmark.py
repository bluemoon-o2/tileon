from __future__ import annotations

from typing import Callable, List, TYPE_CHECKING
from .core import _summarize_statistics

if TYPE_CHECKING:
    import torch


def do_bench_cudagraph(fn: Callable,
                       rep: int = 20,
                       grad_to_none: List["torch.Tensor"] | None = None,
                       quantiles: List[float] | None = None,
                       mode: str = "mean") -> float | List[float]:
    """Benchmark the runtime of the provided function using CUDA graphs.

    Args:
        fn: Function to benchmark.
        rep: Target total repetition time in milliseconds. Defaults to 20.
        grad_to_none: Reset the gradient of the provided tensors to None.
        quantiles: The quantiles to calculate. If None, only the mean will be returned.
        mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all".
            Default is "mean".

    Returns:
        If quantiles is provided, returns the list of quantiles.
        If quantiles is None, returns a single statistic based on mode.

    Raises:
        AssertionError: If an invalid mode is provided.
    """
    import torch
    assert mode in ["min", "max", "mean", "median", "all"]

    with torch.cuda.stream(torch.cuda.Stream()):
        # warmup
        fn()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.detach_()
                x.requires_grad_(True)
                x.grad = None
        # step 1 - we estimate the amount of time the kernel call takes
        # NOTE: this estimate isn't super accurate because the GPU isn't warmed up
        #       at this point but it is probably good enough
        # NOTE: we don't use a graph to estimate the runtime because creating a graph is expensive,
        #       ~300ms on A100, so we default to the same method used in `do_bench` (minus the L2
        #       cache flush).
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            fn()
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5
        # Rewrite to avoid possible division by 0 issues with fast benchmarks
        if estimate_ms == 0:
            n_repeat = 1000
        else:
            n_repeat = max(1, int(rep / estimate_ms))
        # step 2 - construct a cuda graph with `n_repeat`
        # unrolled function calls to minimize host overhead
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                if grad_to_none is not None:
                    for x in grad_to_none:
                        x.grad = None
                fn()
        torch.cuda.synchronize()
        # measure time and return
        ret = []
        n_retries = 10
        for _ in range(n_retries):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            g.replay()
            end_event.record()
            torch.cuda.synchronize()
            ret += [start_event.elapsed_time(end_event) / n_repeat]
        return _summarize_statistics(ret, quantiles, mode)


def do_bench(fn: Callable,
             warmup: int = 25,
             rep: int = 100,
             grad_to_none: List["torch.Tensor"] | None = None,
             quantiles: List[float] | None = None,
             mode: str = "mean") -> float | List[float]:
    """Benchmark the runtime of the provided function.

    By default, returns the specified statistic of :code:`fn` along with
    optional performance quantiles.

    Args:
        fn: Function to benchmark.
        warmup: Target warmup time in milliseconds. Defaults to 25.
        rep: Target total repetition time in milliseconds. Defaults to 100.
        grad_to_none: Reset the gradient of the provided tensors to None.
        quantiles: Performance quantiles to return in addition to the main statistic.
        mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all".
            Default is "mean".

    Returns:
        If quantiles is provided, returns the list of quantiles.
        If quantiles is None, returns a single statistic based on mode.

    Raises:
        AssertionError: If an invalid mode is provided.
    """
    from .. import runtime
    from .. import knobs
    import time
    assert mode in ["min", "max", "mean", "median", "all"]

    if knobs.runtime.interpret:
        # Estimate the runtime of the function
        start = time.perf_counter()
        for _ in range(5):
            fn()
        end = time.perf_counter()
        estimate_ms = (end - start) * 1000 / 5

        # compute number of warmup and repeat
        n_warmup = max(1, int(warmup / estimate_ms)) if estimate_ms > 0 else 10
        n_repeat = max(1, int(rep / estimate_ms)) if estimate_ms > 0 else 100

        # Warm-up
        for _ in range(n_warmup):
            fn()

        times = []
        # Benchmark
        for _ in range(n_repeat):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            start = time.perf_counter()
            fn()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        return _summarize_statistics(times, quantiles, mode)

    di = runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        runtime.driver.active.clear_cache(cache)
        fn()
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        runtime.driver.active.clear_cache(cache)
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, mode)
