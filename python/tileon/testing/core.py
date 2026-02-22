from __future__ import annotations

import math
import statistics
from typing import List, Callable


def assert_close(x: "torch.Tensor" | "np.ndarray" | List[float] | float,
                 y: "torch.Tensor" | "np.ndarray" | List[float] | float,
                 atol: float | Callable[["torch.dtype"], float] = None,
                 rtol: float | Callable[["torch.dtype"], float] = None,
                 err_msg: str = ''):
    """Asserts that two inputs are close within a certain tolerance.

    Args:
        x: The first input to compare. Can be a scalar, list, numpy.ndarray, or torch.Tensor.
        y: The second input to compare. Can be a scalar, list, numpy.ndarray, or torch.Tensor.
        atol: Absolute tolerance for the comparison. Defaults to 1e-2.
            If callable, it takes a torch.dtype and returns the absolute tolerance.
        rtol: Relative tolerance for the comparison. Defaults to 0.
            If callable, it takes a torch.dtype and returns the relative tolerance.
        err_msg: Error message to raise if the assertion fails.

    Raises:
        AssertionError: If inputs are not close within the specified tolerance.
    """
    import torch
    import numpy as np

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)

    atol = 1e-2 if atol is None else atol
    atol = atol(x.dtype) if callable(atol) else atol
    rtol = 0. if rtol is None else rtol
    rtol = rtol(x.dtype) if callable(rtol) else rtol

    # we use numpy because pytorch tends to oom on large tensors
    if x.dtype == torch.bfloat16:
        x = x.float()
    x = x.cpu().detach().numpy()
    if y.dtype == torch.bfloat16:
        y = y.float()
    y = y.cpu().detach().numpy()
    # we handle size==1 case separately as we can
    # provide better error message there
    if x.size > 1 or y.size > 1:
        np.testing.assert_allclose(x,
                                   y,
                                   atol=atol,
                                   rtol=rtol,
                                   equal_nan=True,
                                   err_msg=err_msg)
        return
    if not np.allclose(x, y, atol=atol, rtol=rtol):
        raise AssertionError(
            f'{err_msg} {x} is not close to {y} (atol={atol}, rtol={rtol})')


def _quantile(a: List[float], q: List[float]) -> List[float]:
    """Calculate the quantiles of a list of numbers.

    Args:
        a: List of numeric values.
        q: List of quantile values between 0 and 1.

    Returns:
        List of quantile values corresponding to the input q.

    Raises:
        ValueError: If any quantile value is not in [0, 1].
    """
    n = len(a)
    a = sorted(a)

    def get_quantile(q: float) -> float:
        if not (0 <= q <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(q) for q in q]


def _summarize_statistics(times: List[float],
                          quantiles: List[float] | None = None,
                          mode: str = "mean") -> float | List[float]:
    """Summarize the statistics of a list of times.

    Args:
        times: List of time values.
        quantiles: The quantiles to calculate. If None, only a single statistic is returned.
        mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all".
            Default is "mean".

    Returns:
        If quantiles is provided, returns the list of quantiles.
        If quantiles is None, returns a single statistic based on mode:
            - "all": the entire list of times
            - "min": minimum value
            - "max": maximum value
            - "mean": mean value
            - "median": median value

    Raises:
        ValueError: If an invalid mode is provided.
    """
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if mode == "all":
        return times
    elif mode == "min":
        return min(times)
    elif mode == "max":
        return max(times)
    elif mode == "mean":
        return statistics.mean(times)
    elif mode == "median":
        return statistics.median(times)
    else:
        raise ValueError(f"Invalid mode: {mode}")
