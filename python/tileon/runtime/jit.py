from __future__ import annotations

import re
import ast
import copy
import hashlib
import inspect
import itertools
import threading
import textwrap
from types import ModuleType
from typing import Callable, Generic, Iterable, Optional, TypeAlias, TypeVar, overload, Dict, Any, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .. import knobs
from ..backends import BaseBackend
from .._utils import is_namedtuple
from tileon._C import native_specialize_impl

TILEON_MODULE = "tileon.language"
FUNC_DEF_PATTERN = re.compile(r"^def\s+\w+\s*\(", re.MULTILINE)

T = TypeVar("T", bound=Callable)

GridShape: TypeAlias = Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]]

# -----------------------------------------------------------------------------
# Tensor Wrapper
# -----------------------------------------------------------------------------


class MockTensor:
    """
    A class that mocks a real tensor.

    It is designed to simulate real tensor behavior during kernel warm-up,
    avoiding the overhead of actual tensor operations while maintaining interface consistency.
    """

    def __init__(self, dtype, shape: Optional[list[int]] = None):
        """
        Initialize a MockTensor instance.

        :param dtype: The data type of the real tensor.
        :type dtype: torch.dtype
        :param shape: The shape of the real tensor, defaults to [1] (scalar tensor).
        :type shape: list, optional
        """
        if shape is None:
            shape = [1]
        self.dtype = dtype
        self.shape = shape

    def stride(self):
        """
        Returns the stride of the mock tensor.

        :return: The stride of the mock tensor.
        :rtype: tuple
        """
        strides = [1]
        for size in self.shape[1:]:
            strides.append(strides[-1] * size)
        return tuple(reversed(strides))

    @staticmethod
    def data_ptr():
        """
        Returns the data pointer of the mock tensor.

        :return: The data pointer of the mock tensor.
        :rtype: int
        """
        return 0  # optimistically assumes multiple of 16

    @staticmethod
    def ptr_range():
        """
        Returns the pointer range of the mock tensor.

        :return: The pointer range of the mock tensor.
        :rtype: int
        """
        return 0  # optimistically assumes 32 bit pointer range

    @staticmethod
    def from_torch(t: "torch.Tensor") -> "MockTensor":
        """
        Converts a torch tensor to a MockTensor.

        :param t: The torch tensor to convert.
        :type t: torch.Tensor
        :return: The converted MockTensor.
        :rtype: MockTensor
        """
        if t.__class__.__name__ == "dtype" and t.__module__ == "torch":
            return MockTensor(t)
        return t


class TileonTensor:
    """
    Internal tensor class for Tileon.

    It is a thin wrapper around a base tensor.
    """

    def __init__(self, base, dtype):
        """
        Initialize a TileonTensor instance.

        :param base: The base tensor to wrap.
        :type base: torch.Tensor
        :param dtype: The data type of the tensor.
        :type dtype: torch.dtype
        """
        self.base = base
        self.dtype = dtype
        self.data = base.data
        self.device = base.device
        self.shape = base.shape

    def data_ptr(self) -> int:
        """
        Returns the data pointer of the tensor.

        :return: The data pointer of the tensor.
        :rtype: int
        """
        return self.base.data_ptr()

    def stride(self, *args) -> int:
        """
        Returns the stride of the tensor.

        :return: The stride of the tensor.
        :rtype: int
        """
        return self.base.stride(*args)

    def element_size(self) -> int:
        """
        Returns the element size of the tensor.

        :return: The element size of the tensor.
        :rtype: int
        """
        return self.base.element_size()

    def cpu(self) -> "TileonTensor":
        """
        Moves the tensor to the CPU.

        :return: The tensor on the CPU.
        :rtype: TileonTensor
        """
        return TileonTensor(self.base.cpu(), self.dtype)

    def copy_(self, other: "TileonTensor") -> None:
        """
        Copies the data from another tensor to this tensor.

        :param other: The tensor to copy data from.
        :type other: TileonTensor
        """
        self.base.copy_(other.base)

    def clone(self) -> "TileonTensor":
        """
        Creates a clone of the tensor.

        :return: The cloned tensor.
        :rtype: TileonTensor
        """
        return TileonTensor(self.base.clone(), self.dtype)

    def to(self, device) -> "TileonTensor":
        """
        Moves the tensor to the specified device.

        :param device: The device to move the tensor to.
        :type device: torch.device
        :return: The tensor on the specified device.
        :rtype: TileonTensor
        """
        return TileonTensor(self.base.to(device), self.dtype)

    def new_empty(self, sizes: list[int]) -> "TileonTensor":
        """
        Creates a new empty tensor with the specified sizes.

        :param sizes: The sizes of the new tensor.
        :type sizes: list[int]
        :return: The new empty tensor.
        :rtype: TileonTensor
        """
        return TileonTensor(self.base.new_empty(sizes), self.dtype)

    def __str__(self) -> str:
        """
        Returns the string representation of the tensor with minimal overhead.

        :return: The string representation of the tensor.
        :rtype: str
        """
        s = str(self.base).replace('tensor', 'TileonTensor')
        if s.find('dtype') == -1:
            idx = s.rfind(')')
            s = s[:idx] + ", dtype=" + str(self.dtype) + s[idx:]
        return s


def reinterpret(tensor, dtype) -> TileonTensor:
    if isinstance(tensor, TileonTensor):
        if dtype == tensor.base.dtype:
            return tensor.base
        else:
            return TileonTensor(tensor.base, dtype)
    elif hasattr(tensor, "data_ptr"):
        return TileonTensor(tensor, dtype)
    else:
        raise TypeError(f"Cannot reinterpret a {type(tensor)}.")


def mangle_type(arg, specialize=False):
    """
    Mangles the type of the argument.

    Args:
        arg: The argument whose type is to be mangled.
        specialize (bool, optional): Whether to specialize the type. Defaults to False.

    Returns:
        str: The mangled type string.
    """
    is_const = False
    align = True
    return native_specialize_impl(BaseBackend, arg, is_const, specialize, align)[0]


def get_jit_fn_file_line(fn):
    """
    Get the file name and line number of the JIT-compiled function.

    Args:
        fn: The JIT-compiled function.

    Returns:
        tuple: A tuple containing the file name and line number.
    """
    base_fn = fn
    while not isinstance(base_fn, JITCallable):
        base_fn = base_fn.fn
    file_name = base_fn.fn.__code__.co_filename
    begin_line = base_fn.first_line_number
    for idx, line in enumerate(base_fn.raw_src):
        if line.strip().startswith("def "):
            begin_line += idx
            break
    return file_name, begin_line


# -----------------------------------------------------------------------------
# JIT-Compiling
# -----------------------------------------------------------------------------


class KernelInterface(Generic[T]):
    """Base class for defining interfaces of JIT-compiled kernels."""

    run: T

    def warmup(self, *tensors, grid: GridShape, **kwargs):
        """
        Perform kernel warmup with mocked tensor inputs to avoid cold-start overhead.

        Args:
            *tensors: Input arguments (tensors) to be wrapped for warmup execution
            grid: Grid configuration for kernel launch
            **kwargs: Additional keyword arguments for kernel execution
        """
        return self.run(*map(MockTensor.from_torch, tensors), grid=grid, warmup=True, **kwargs)

    def run(self, *tensors, grid: GridShape, warmup: bool = False, **kwargs):
        """
        Execute the kernel with the given tensors and grid configuration.

        Args:
            *tensors: Input arguments (tensors) for kernel execution
            grid: Grid configuration for kernel launch
            warmup: Whether to perform warmup execution (default: False)
            **kwargs: Additional keyword arguments for kernel execution
        """
        raise NotImplementedError("run not implemented")

    def __getitem__(self, grid: GridShape) -> T:
        """
        Return a callable proxy that memorizes the grid configuration for kernel launch.

        Args:
            grid: Grid configuration for kernel launch

        Returns:
            A callable proxy that executes the kernel with the given grid configuration.
        """
        return lambda *tensors, **kwargs: self.run(*tensors, grid=grid, warmup=False, **kwargs)


class DependenciesFinder(ast.NodeVisitor):
    """AST visitor to find dependencies of a JITFunction.

    This visitor can be used to invalidate a JITFunction's hash when its
    source code or that of its dependencies changes. It also tracks the
    global variables touched by the JITFunction. When launching the kernel,
    we check that these have the same values as they did when we ran this
    visitor. If not, we raise an error.
    """

    def __init__(self, name: str, src: str, globals: Dict[str, Any], nonlocals: Dict[str, Any]):
        """Initialize the DependenciesFinder.

        Args:
            name: Name of the function being analyzed.
            globals: Dictionary of global variables.
            nonlocals: Dictionary of nonlocal variables.
            src: Source code of the function.
        """
        super().__init__()
        self.name = name
        self.hasher = hashlib.sha256(src.encode("utf-8"))

        self.globals = globals
        self.nonlocals = nonlocals
        self.used_global_vals: Dict[Tuple[str, int], Tuple[Any, Dict[str, Any]]] = {}
        self.visiting_arg_default_value = False

        self.supported_python_builtins = {
            'int',
            'float',
            'list',
            'len',
            'max',
            'min',
            'print',
            'range',
            'getattr',
            'isinstance',
        }
        self.supported_modules = {
            TILEON_MODULE,
            "copy",
            "math",
        }

    @property
    def ret(self):
        """Get the SHA-256 hash of the dependencies.

        Returns:
            Hexadecimal string representing the hash.
        """
        return self.hasher.hexdigest()

    def _is_builtin(self, node, func):
        """Check if a function is a Tileon builtin.

        Args:
            node: AST node of the function call.
            func: Function object to check.

        Returns:
            True if the function is a Tileon builtin, False otherwise.
        """
        if inspect.isbuiltin(node.func):
            return True
        module = getattr(func, "__module__", "")
        return module.startswith(TILEON_MODULE)

    def _update_hash(self, func: JITCallable):
        """Update the hash with a called JITCallable.

        Args:
            func: JITCallable to include in the hash.

        Raises:
            RuntimeError: If there's a conflicting value for a global variable.
        """
        assert isinstance(func, JITCallable), f"func must be a JITCallable, got {type(func)}"
        for k in self.used_global_vals.keys() & func.used_global_vals.keys():
            var_name, _ = k
            v1, _ = self.used_global_vals[k]
            v2, _ = func.used_global_vals[k]
            if v1 != v2:
                raise RuntimeError(f"Global variable {var_name} has value {v1} when compiling "
                                   f"{self.name}, but inner kernel {func.__name__} has conflicting "
                                   f"value {v2} from when it was first compiled.  This is not allowed.")
        self.used_global_vals.update(func.used_global_vals)
        func_key = func.cache_key
        func_key += str(getattr(func, "noinline", False))
        self.hasher.update(func_key.encode("utf-8"))

    def record_reference(self, val, var_dict=None, name=None):
        """Record a reference to a value.

        Args:
            val: Value to record a reference to.
            var_dict: Dictionary containing the variable.
            name: Name of the variable.

        Raises:
            RuntimeError: If an unsupported function is referenced.
        """
        from tileon.language.core import constexpr
        if val is None or type(val) is ModuleType:
            return

        if self.visiting_arg_default_value:
            return

        if getattr(val, "__tileon_aggregate__", False):
            self.hasher.update(str(val.__annotations__).encode("utf-8"))
            for attr in val.hash_attrs:
                self.record_reference(attr)
            return

        if getattr(val, "__tileon_builtin__", False):
            return

        if isinstance(val, JITCallable):
            self._update_hash(val)
            return

        if callable(val) and not isinstance(val, type) and not isinstance(val, constexpr):
            raise RuntimeError(f"Unsupported function referenced: {val}")

        if var_dict is not None:
            self.used_global_vals[(name, id(var_dict))] = (copy.deepcopy(val), var_dict)
        return

    def visit_Name(self, node):
        """Visit a Name AST node.

        Args:
            node: Name AST node to visit.

        Returns:
            The value of the name if it's a global/nonlocal variable, None otherwise.
        """
        name = node.id

        if type(node.ctx) is ast.Store:
            return name

        if name in self.local_names:
            return None

        val, var_dict = None, None
        val = self.globals.get(name)
        if val is not None:
            var_dict = self.globals
        else:
            val = self.nonlocals.get(name)
            if val is not None:
                var_dict = self.nonlocals

        if name in self.supported_python_builtins:
            return val

        self.record_reference(val, var_dict, name)
        return val

    def visit_Tuple(self, node):
        """Visit a Tuple AST node.

        Args:
            node: Tuple AST node to visit.

        Returns:
            List of visited elements.
        """
        return [self.visit(elt) for elt in node.elts]

    def visit_Attribute(self, node):
        """Visit an Attribute AST node.

        Args:
            node: Attribute AST node to visit.

        Returns:
            The attribute value if it's not from a supported module, None otherwise.
        """
        lhs = self.visit(node.value)
        while isinstance(lhs, ast.Attribute):
            lhs = self.visit(lhs.value)
        lhs_name = getattr(lhs, "__name__", "")
        if lhs is None or lhs_name in self.supported_modules:
            return None
        ret = getattr(lhs, node.attr)
        self.record_reference(ret)
        return ret

    def visit_FunctionDef(self, node):
        """Visit a FunctionDef AST node.

        Args:
            node: FunctionDef AST node to visit.
        """
        self.local_names = {arg.arg for arg in node.args.args}
        self.generic_visit(node)

    def visit_arguments(self, node):
        """Visit an arguments AST node.

        Args:
            node: arguments AST node to visit.
        """

        def visit_defaults(defaults):
            try:
                assert not self.visiting_arg_default_value
                self.visiting_arg_default_value = True
                for expr in defaults:
                    if expr is not None:
                        self.visit(expr)
            finally:
                self.visiting_arg_default_value = False

        for arg in itertools.chain(node.posonlyargs, node.args, [node.vararg] if node.vararg else [], node.kwonlyargs):
            self.visit(arg)

        visit_defaults(node.kw_defaults)

        if node.kwarg is not None:
            self.visit(node.kwarg)

        visit_defaults(node.defaults)

    def visitAssnTarget(self, node):
        """Visit an assignment target node.

        Args:
            node: Assignment target node to visit.
        """
        target = self.visit(node)
        if isinstance(target, list):
            self.local_names |= set(target)
        else:
            self.local_names.add(target)

    def visit_Assign(self, node):
        """Visit an Assign AST node.

        Args:
            node: Assign AST node to visit.

        Raises:
            TypeError: If simultaneous multiple assignment is used.
        """
        if len(node.targets) != 1:
            raise TypeError("Simultaneous multiple assignment is not supported.")

        self.visitAssnTarget(node.targets[0])
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        """Visit an AnnAssign AST node.

        Args:
            node: AnnAssign AST node to visit.
        """
        self.visitAssnTarget(node.target)
        self.generic_visit(node)

    def visit_For(self, node):
        """Visit a For AST node.

        Args:
            node: For AST node to visit.
        """
        self.visitAssnTarget(node.target)
        self.generic_visit(node)


class JITCallable:
    """Wrapper class for JIT-compiled functions.

    Provides the base functionality for JIT compilation, including:
    - Source code management and parsing
    - Hash computation for caching
    - Global variable dependency tracking
    """

    def __init__(self, fn: Callable):
        """Initialize a JITCallable instance.

        Args:
            fn: Python function to be JIT-compiled.

        Raises:
            ValueError: If the function is not defined in a Python file and source code cannot be retrieved.
        """
        self.fn = fn
        self._fn_name = f"{fn.__module__}.{fn.__qualname__}"
        self.signature = inspect.signature(fn)

        # reuse function docstring
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__qualname__ = fn.__qualname__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__

        try:
            self.raw_src, self.first_line_number = inspect.getsourcelines(fn)
        except OSError as e:
            raise ValueError("@jit functions should be defined in a Python file") from e

        src = textwrap.dedent("".join(self.raw_src))
        self._src = src[FUNC_DEF_PATTERN.search(src).start():]

        self.hash = None
        self._hash_lock = threading.RLock()

        self.used_global_vals: Dict[Tuple[str, int], Tuple[Any, Dict[str, Any]]] = {}

    def _unsafe_update_src(self, new_src):
        """Update the source code.

        This is the only method allowed to modify src.

        Note:
            It is the caller's responsibility to make sure any tileon functions
            that call this function have the `.hash` value reset to None.

        Args:
            new_src: New source code to set.
        """
        self.hash = None
        self._src = new_src

    def _set_src(self, new_src):
        """Prevent direct setting of the src attribute.

        Args:
            new_src: Source code that would be set.

        Raises:
            AttributeError: Always raised to prevent direct setting.
        """
        raise AttributeError("Cannot set attribute 'src' directly. "
                             "Use '_unsafe_update_src()' and manually clear `.hash` of all callers instead.")

    def _get_src(self):
        """Get the source code.

        Returns:
            Source code of the function.
        """
        return self._src

    src = property(fget=_get_src, fset=_set_src)
    """Source code of the JIT-compiled function (read-only).
    """

    @property
    def type(self):
        """Get the constexpr type of the function.

        Returns:
            constexpr type of the function.
        """
        from tileon.language.core import constexpr_t
        return constexpr_t(self)

    def parse(self):
        """Parse the source code into an AST.

        Returns:
            AST Module node containing exactly one FunctionDef node.

        Raises:
            AssertionError: If the AST does not contain exactly one function definition.
        """
        tree = ast.parse(self._src)
        assert isinstance(tree, ast.Module), f"Expected Module, got {type(tree)}"
        assert len(tree.body) == 1, f"Expected 1 function definition, got {len(tree.body)}"
        assert isinstance(tree.body[0], ast.FunctionDef), f"Expected FunctionDef, got {type(tree.body[0])}"
        return tree

    def get_capture_scope(self):
        """Get the combined capture scope including globals and nonlocals.

        Returns:
            Dictionary containing merged globals and nonlocals from the function closure.
        """
        fn = self.fn
        if fn.__closure__ is None:
            return self.__globals__
        nonlocals = {name: cell.cell_contents for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)}
        return self.__globals__ | nonlocals

    @property
    def cache_key(self) -> str:
        """Compute or retrieve the cache key for this JIT-compiled function.

        The cache key is a SHA-256 hash of:
        - Source code
        - Starting line number
        - Dependencies (global variables)
        - constexpr values

        Returns:
            SHA-256 hash string used as cache key.
        """
        with self._hash_lock:
            if self.hash is not None:
                return self.hash

            # avoid infinite recursion
            self.hash = f"recursion:{self._fn_name}"
            nonlocals = inspect.getclosurevars(self.fn).nonlocals
            dependencies_finder = DependenciesFinder(
                name=self._fn_name,
                src=self.src,
                globals=self.__globals__,
                nonlocals=nonlocals,
            )
            dependencies_finder.visit(self.parse())

            # hash the dependencies, line number, and constexpr values
            self.hash = dependencies_finder.ret + str(self.first_line_number)
            self.used_global_vals = dict(sorted(dependencies_finder.used_global_vals.items()))

            from tileon.language.core import constexpr
            self.hash += str([(name, val) for (name, _), (val, _) in self.used_global_vals.items()
                              if isinstance(val, constexpr)])

            self.hash = hashlib.sha256(self.hash.encode("utf-8")).hexdigest()
        return self.hash

    def __hash__(self):
        """Compute hash based on cache key.

        Returns:
            Hash value of the cache key.
        """
        return hash(self.cache_key)

    def _flatten_ir(self, handles) -> None:
        """Flatten intermediate representation (not implemented).

        Args:
            handles: Handles for IR flattening.

        Raises:
            NotImplementedError: Always raised as this method is not implemented.
        """
        raise NotImplementedError("JITCallable._flatten_ir() is not implemented")


class BoundConstexprFunction(JITCallable):
    """JIT-compiled constexpr function bound to an instance.

    This class wraps a constexpr function and binds it to a specific instance.
    When called, it invokes the underlying function with the bound instance as the first argument.
    """

    def __init__(self, instance, fn):
        self.__self__ = instance
        self.__func__ = fn

    @property
    def cache_key(self):
        return self.__func__.cache_key

    def __call__(self, *args, **kwargs):
        return self.__func__(self.__self__, *args, **kwargs)


def compute_cache_key(kernel_key_cache: Dict[Tuple[Any, str], str], specialization, options):
    """Compute the cache key for a JIT function specialization.

    Args:
        kernel_key_cache: Cache to store computed keys.
        specialization: Specialization of the JIT function.
        options: Options for JIT compilation.

    Returns:
        Cache key for the specialized JIT function.
    """
    key = (tuple(specialization), str(options))
    cache_key = kernel_key_cache.get(key, None)
    if cache_key is not None:
        return cache_key

    # Replace JITCallable objects with their hash, so the cache key will change if the src is updated
    def replace_callables(obj):
        if isinstance(obj, list):
            return [replace_callables(arg) for arg in obj]
        elif is_namedtuple(obj):
            results = [replace_callables(arg) for arg in obj]
            return obj.__class__(*results)
        elif isinstance(obj, tuple):
            return tuple(replace_callables(arg) for arg in obj)
        elif isinstance(obj, JITCallable):
            return obj.cache_key
        return obj

    cache_key = str(replace_callables(specialization)) + str(options)
    kernel_key_cache[key] = cache_key
    return cache_key


class ConstexprFunction(JITCallable):
    """JIT-compiled constexpr function.

    This class wraps a constexpr function and compiles it for JIT execution.
    When called, it invokes the underlying function with the provided arguments.
    """

    def __init__(self, fn):
        super().__init__(fn)

    def __get__(self, obj, objclass):
        # Create a bound function to support constexpr_function methods
        if obj is not None:
            return BoundConstexprFunction(obj, self)
        return self

    def __call__(self, *args, _semantic=None, **kwargs):
        from tileon.language.core import _unwrap_if_constexpr, constexpr
        args = [_unwrap_if_constexpr(x) for x in args]
        kwargs = {k: _unwrap_if_constexpr(v) for (k, v) in kwargs.items()}

        res = self.fn(*args, **kwargs)

        if _semantic is None:
            return res

        if knobs.runtime.interpret:
            return res
        return constexpr(res)


def constexpr_function(fn):
    """
    Wraps an arbitrary Python function so that it can be called at
    compile-time on constexpr arguments in a Tileon function and
    returns a constexpr result.
    """
    return ConstexprFunction(fn)


@overload
def jit(fn: T) -> KernelInterface[T]:
    ...


@overload
def jit(
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Callable[[T], KernelInterface[T]]:
    ...


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> KernelInterface[T]:
    """
    Decorator for JIT-compiling a function using the Tileon compiler.

    Note:
        When a jit'd function is called, arguments are implicitly converted to pointers
        if they have a :code:`.data_ptr()` method and a `.dtype` attribute.

        This function will be compiled and run on the GPU. It will only have access to:
           * python primitives,
           * builtins within the tileon package,
           * arguments to this function,
           * other jit'd functions

    Args:
        fn: the function to be jit-compiled

    Returns:
        A KernelInterface object that can be called with arguments.
    """

    def decorator(fn: T) -> KernelInterface[T]:
        assert callable(fn), "jit decorator must be called on a callable object"
        if knobs.runtime.interpret:
            from .interpreter import InterpretedFunction
            return InterpretedFunction(fn,
                                       version=version,
                                       do_not_specialize=do_not_specialize,
                                       do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                                       debug=debug,
                                       noinline=noinline,
                                       repr=repr,
                                       launch_metadata=launch_metadata)

    if fn is not None:
        return decorator(fn)
    else:
        return decorator
