from __future__ import annotations

import sys
import subprocess
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def nv_smi(attrs: List[str]) -> List[int]:
    """Query the NVIDIA GPU metrics using `nvidia-smi`.

    Args:
        attrs: List of attribute names to query.

    Returns:
        List of integer values corresponding to the queried attributes.
    """
    attrs = ','.join(attrs)
    cmd = ['nvidia-smi', '-i', '0', '--query-gpu=' + attrs, '--format=csv,noheader,nounits']
    out = subprocess.check_output(cmd)
    ret = out.decode(sys.stdout.encoding).split(',')
    ret = [int(x) for x in ret]
    return ret


def get_dram_gbps(device: int = None):
    """Get the DRAM bandwidth in GB/s for the given device.

    If no device is specified, the current device is used.

    Args:
        device: Device ID. If None, uses the current device.

    Returns:
        DRAM bandwidth in GB/s.
    """
    from ..runtime import driver
    if device is None:
        device = driver.active.get_device_interface().current_device()
    mem_clock_khz = driver.active.utils.get_device_properties(device)["mem_clock_rate"]  # in kHz
    bus_width = driver.active.utils.get_device_properties(device)["mem_bus_width"]
    bw_gbps = mem_clock_khz * bus_width * 2 / 1e6 / 8  # In GB/s
    return bw_gbps


def get_max_tensorcore_tflops(dtype: "torch.dtype", clock_rate: float, device: int = None) -> float:
    """Get the maximum Tensor Core TFLOPS for the given dtype, clock rate, and device.

    If no device is specified, the current device is used.

    Args:
        dtype: Data type for Tensor Core operations.
        clock_rate: Clock rate in GHz.
        device: Device ID. If None, uses the current device.

    Returns:
        Maximum Tensor Core TFLOPS.

    Raises:
        RuntimeError: If the data type is not supported.
    """
    import torch
    from ..runtime import driver

    if not device:
        device = torch.cuda.current_device()

    num_subcores = driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8:
        assert dtype == torch.float16
        ops_per_sub_core = 256  # 2 4x4x4 Tensor Cores
    else:
        if dtype in [torch.float32, torch.int32]:
            ops_per_sub_core = 256
        elif dtype in [torch.float16, torch.bfloat16, torch.int16]:
            ops_per_sub_core = 512
        elif dtype in [torch.int8]:
            ops_per_sub_core = 1024
        else:
            raise RuntimeError("dtype not supported")
    tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-9
    return tflops


def get_max_simd_tflops(dtype: "torch.dtype", clock_rate: float, device: int = None) -> float:
    """Get the maximum SIMD TFLOPS for the given dtype, clock rate, and device.

    If no device is specified, the current device is used.

    Args:
        dtype: Data type for SIMD operations.
        clock_rate: Clock rate in GHz.
        device: Device ID. If None, uses the current device.

    Returns:
        Maximum SIMD TFLOPS.

    Raises:
        RuntimeError: If the data type is not supported.
    """
    import torch
    from ..runtime import driver

    if not device:
        device = torch.cuda.current_device()

    num_subcores = driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        if dtype == torch.float32:
            ops_per_sub_core = 32  # 2*16
        elif dtype == torch.float16:
            ops_per_sub_core = 64
        else:
            raise RuntimeError("dtype not supported")
    else:
        if dtype == torch.float32:
            ops_per_sub_core = 32
        elif dtype in [torch.float16, torch.bfloat16]:
            ops_per_sub_core = 64
        else:
            raise RuntimeError("dtype not supported")
    tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-9
    return tflops
