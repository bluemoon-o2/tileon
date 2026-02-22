from __future__ import annotations

from .jit import KernelInterface


class Config:
    """
    An object that represents a possible kernel configuration for the auto-tuner to try.

    :ivar kwargs: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
    :type kwargs: dict[Str, Any]
    :ivar num_warps: the number of warps to use for the kernel when compiled for GPUs. For example, if
                      `num_warps=8`, then each kernel instance will be automatically parallelized to
                      cooperatively execute using `8 * 32 = 256` threads.
    :type num_warps: int
    :ivar num_stages: the number of stages that the compiler should use when software-pipelining loops.
                       Mostly useful for matrix multiplication workloads on SM80+ GPUs.
    :type num_stages: int
    :ivar num_ctas: number of blocks in a block cluster. SM90+ only.
    :type num_ctas: int
    :type maxnreg: Optional[int]
    :ivar maxnreg: maximum number of registers one thread can use.  Corresponds
                       to ptx .maxnreg directive.  Not supported on all platforms.
    :ivar pre_hook: a function that will be called before the kernel is called. Parameters of this
                    function are args.
    :ivar ir_override: filename of a user-defined IR (*.{ttgir|llir|ptx|amdgcn}).
    """

    def __init__(self,
                 kwargs,
                 num_warps=4,
                 num_stages=3,
                 num_ctas=1,
                 maxnreg=None,
                 pre_hook=None,
                 ir_override=None):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_ctas = num_ctas
        self.num_stages = num_stages
        self.maxnreg = maxnreg
        self.pre_hook = pre_hook
        self.ir_override = ir_override

    def __setstate__(self, state):
        self.kwargs = state.get("kwargs", {})
        self.num_warps = state.get("num_warps", 4)
        self.num_stages = state.get("num_stages", 3)
        self.num_ctas = state.get("num_ctas", 1)
        self.maxnreg = state.get("maxnreg", None)
        self.pre_hook = state.get("pre_hook", None)
        self.ir_override = state.get("ir_override", None)

    def all_kwargs(self):
        return {
            **self.kwargs,
            **{
                k: v
                for (k, v) in (
                    ("num_warps", self.num_warps),
                    ("num_ctas", self.num_ctas),
                    ("num_stages", self.num_stages),
                    ("maxnreg", self.maxnreg),
                    ("ir_override", self.ir_override),
                ) if v is not None
            }
        }

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}: {v}")
        res.append(f"num_warps: {self.num_warps}")
        res.append(f"num_ctas: {self.num_ctas}")
        res.append(f"num_stages: {self.num_stages}")
        res.append(f"maxnreg: {self.maxnreg}")
        return ", ".join(res)

    def __hash__(self):
        return hash((*self.all_kwargs().items(), self.pre_hook))

    def __eq__(self, other):
        self_tuple = tuple((
            *self.all_kwargs().items(),
            self.pre_hook,
        ))
        other_tuple = tuple((
            *other.all_kwargs().items(),
            other.pre_hook,
        ))
        return self_tuple == other_tuple


class Heuristics(KernelInterface):

    def __init__(self, fn, arg_names, values):
        self.fn = fn
        self.values = values
        self.arg_names = arg_names

    def run(self, *args, **kwargs):
        for v, heur in self.values.items():
            kwargs[v] = heur({**dict(zip(self.arg_names, args)), **kwargs})
        return self.fn.run(*args, **kwargs)


def heuristics(values):
    """
    Decorator for specifying how the values of certain meta-parameters may be computed.
    This is useful for cases where auto-tuning is prohibitively expensive, or just not applicable.

    .. highlight:: python
    .. code-block:: python

        # smallest power-of-two >= x_size
        @tileon.heuristics(values={'BLOCK_SIZE': lambda args: tileon.next_power_of_2(args['x_size'])})
        @tileon.jit
        def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
            ...
    :param values: a dictionary of meta-parameter names and functions that compute the value of the meta-parameter.
                   each such function takes a list of positional arguments as input.
    :type values: dict[str, Callable[[dict[str, Any]], Any]]
    """

    def decorator(fn):
        return Heuristics(fn, fn.arg_names, values)

    return decorator
