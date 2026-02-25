from __future__ import annotations

import functools
import os
import subprocess
from contextlib import contextmanager
from .gpu_utils import nv_smi


def cuda_memcheck(**target_kwargs):
    """Decorator that wraps a test function to run it with cuda-memcheck.

    The test will only be run if the specified keyword arguments match.

    Args:
        **target_kwargs: Keyword arguments that trigger cuda-memcheck when present.

    Returns:
        Decorator function.
    """

    def decorator(test_fn):
        # record the metadata of the test function
        @functools.wraps(test_fn)
        def wrapper(*args, **kwargs):
            import psutil
            ppid_name = psutil.Process(os.getppid()).name()
            # check if the target_kwargs are a subset of the kwargs
            run_cuda_memcheck = target_kwargs.items() <= kwargs.items()
            if run_cuda_memcheck and ppid_name != "cuda-memcheck":
                path = os.path.realpath(test_fn.__globals__["__file__"])
                env = {"PATH": os.environ["PATH"], "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
                assert 'request' in kwargs, "memcheck'ed test must have a (possibly unused) `request` fixture"
                test_id = kwargs['request'].node.callspec.id
                # pytest command to run the test
                cmd = f"{path}::{test_fn.__name__}[{test_id}]"
                out = subprocess.run(["cuda-memcheck", "pytest", "-vs", cmd], capture_output=True, env=env)
                assert out.returncode == 0, "cuda-memcheck returned an error: bounds checking failed"
                assert "ERROR SUMMARY: 0 errors" in str(out.stdout)
            else:
                test_fn(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def set_gpu_clock(ref_sm_clock: int = 1350, ref_mem_clock: int = 1215):
    """Context manager to set the GPU clock to the given reference clock rate.

    The clock rate is set for both the SMs and the memory.
    The original clock settings are restored when exiting the context.

    Args:
        ref_sm_clock: Reference SM clock rate in MHz. Defaults to 1350.
        ref_mem_clock: Reference memory clock rate in MHz. Defaults to 1215.

    Yields:
        Tuple of (tflops, gbps) where tflops is the theoretical TFLOPS and
        gbps is the theoretical DRAM bandwidth in GB/s.

    Raises:
        AssertionError: If the GPU clocks cannot be set to the reference values.
    """
    try:
        subprocess.check_output(["nvidia-smi", "-i", "0", "-pm", "1"])
        subprocess.check_output(["nvidia-smi", "-i", "0", f"--lock-gpu-clocks={ref_sm_clock},{ref_sm_clock}"])
        subprocess.check_output(["nvidia-smi", "-i", "0", f"--lock-memory-clocks={ref_mem_clock},{ref_mem_clock}"])
        cur_sm_clock = nv_smi(["clocks.current.sm"])[0]
        cur_mem_clock = nv_smi(["clocks.current.memory"])[0]
        assert abs(cur_sm_clock - ref_sm_clock) < 10, f"GPU SMs must run at {ref_sm_clock} MHz"
        assert abs(cur_mem_clock - ref_mem_clock) < 10, f"GPU SMs must run at {ref_mem_clock} MHz"
        tflops = 1e-6 * 2 * 108 * 4 * 256 * ref_sm_clock
        gbps = 640 * 2 * ref_mem_clock * 1e-3
        yield tflops, gbps
    finally:
        subprocess.check_output(["nvidia-smi", "-i", "0", "-pm", "0"])
        subprocess.check_output(["nvidia-smi", "-i", "0", "-rgc"])
        subprocess.check_output(["nvidia-smi", "-i", "0", "-rmc"])
