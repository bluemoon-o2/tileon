from __future__ import annotations

import sys
import typing
import warnings
import functools

# Python 3.13.3+ contains a fix for the wrapped __new__
# Breakpoint: https://github.com/python/cpython/pull/132160
if sys.version_info >= (3, 13, 3):
    import warnings
    deprecated = warnings.deprecated
else:

    _T = typing.TypeVar("_T")

    class deprecated:
        """Indicate that a class, function or overload is deprecated.

        When this decorator is applied to an object, the type checker
        will generate a diagnostic on usage of the deprecated object.

        Usage:

            @deprecated("Use B instead")
            class A:
                pass

            @deprecated("Use g instead")
            def f():
                pass

            @overload
            @deprecated("int support is deprecated")
            def g(x: int) -> int: ...
            @overload
            def g(x: str) -> int: ...

        The warning specified by *category* will be emitted at runtime
        on use of deprecated objects. For functions, that happens on calls;
        for classes, on instantiation and on creation of subclasses.
        If the *category* is ``None``, no warning is emitted at runtime.
        The *stacklevel* determines where the
        warning is emitted. If it is ``1`` (the default), the warning
        is emitted at the direct caller of the deprecated object; if it
        is higher, it is emitted further up the stack.
        Static type checker behavior is not affected by the *category*
        and *stacklevel* arguments.

        The deprecation message passed to the decorator is saved in the
        ``__deprecated__`` attribute on the decorated object.
        If applied to an overload, the decorator
        must be after the ``@overload`` decorator for the attribute to
        exist on the overload as returned by ``get_overloads()``.

        See PEP 702 for details.

        """

        def __init__(
            self,
            message: str,
            /,
            *,
            category: typing.Optional[
                typing.Type[Warning]] = DeprecationWarning,
            stacklevel: int = 1,
        ):
            if not isinstance(message, str):
                raise TypeError(
                    "Expected an object of type str for 'message', not "
                    f"{type(message).__name__!r}")
            self.message = message
            self.category = category
            self.stacklevel = stacklevel

        def __call__(self, arg: _T, /) -> _T:
            # Make sure the inner functions created below don't
            # retain a reference to self.
            msg = self.message
            category = self.category
            stacklevel = self.stacklevel
            if category is None:
                arg.__deprecated__ = msg
                return arg
            elif isinstance(arg, type):
                from types import MethodType

                original_new = arg.__new__

                @functools.wraps(original_new)
                def __new__(cls, /, *args, **kwargs):
                    if cls is arg:
                        warnings.warn(msg,
                                      category=category,
                                      stacklevel=stacklevel + 1)
                    if original_new is not object.__new__:
                        return original_new(cls, *args, **kwargs)
                    # Mirrors a similar check in object.__new__.
                    elif cls.__init__ is object.__init__ and (args or kwargs):
                        raise TypeError(f"{cls.__name__}() takes no arguments")
                    else:
                        return original_new(cls)

                arg.__new__ = staticmethod(__new__)

                original_init_subclass = arg.__init_subclass__
                # We need slightly different behavior if __init_subclass__
                # is a bound method (likely if it was implemented in Python)
                if isinstance(original_init_subclass, MethodType):
                    original_init_subclass = original_init_subclass.__func__

                    @functools.wraps(original_init_subclass)
                    def __init_subclass__(*args, **kwargs):
                        warnings.warn(msg,
                                      category=category,
                                      stacklevel=stacklevel + 1)
                        return original_init_subclass(*args, **kwargs)

                    arg.__init_subclass__ = classmethod(__init_subclass__)
                # Or otherwise, which likely means it's a builtin such as
                # object's implementation of __init_subclass__.
                else:

                    @functools.wraps(original_init_subclass)
                    def __init_subclass__(*args, **kwargs):
                        warnings.warn(msg,
                                      category=category,
                                      stacklevel=stacklevel + 1)
                        return original_init_subclass(*args, **kwargs)

                    arg.__init_subclass__ = __init_subclass__

                arg.__deprecated__ = __new__.__deprecated__ = msg
                __init_subclass__.__deprecated__ = msg
                return arg
            elif callable(arg):
                import inspect
                from asyncio import coroutines

                @functools.wraps(arg)
                def wrapper(*args, **kwargs):
                    warnings.warn(msg,
                                  category=category,
                                  stacklevel=stacklevel + 1)
                    return arg(*args, **kwargs)

                if inspect.iscoroutinefunction(arg):
                    # Breakpoint: https://github.com/python/cpython/pull/99247
                    if sys.version_info >= (3, 12):
                        wrapper = inspect.markcoroutinefunction(wrapper)
                    else:
                        wrapper._is_coroutine = coroutines._is_coroutine

                arg.__deprecated__ = wrapper.__deprecated__ = msg
                return wrapper
            else:
                raise TypeError(
                    "@deprecated decorator with non-None category must be applied to "
                    f"a class or callable, not {arg!r}")


if typing.TYPE_CHECKING:
    from .language import core
    IterableType = typing.Union[list[typing.Any], tuple[typing.Any, ...],
                                core.tuple, core.tuple_t]
    ObjPath = typing.Tuple[int, ...]

TILEON_MAX_TENSOR_NUMEL = 1048576


def tuple_create(x: typing.Union[typing.Tuple, typing.NamedTuple],
                 dtype: type):
    """
    Creates a tuple of dtype from x.

    Args:
        dtype: The type of the tuple.
        x: The input to create the tuple from.

    Returns:
        A tuple of dtype.
    """
    # only NamedTuple has "_fields"
    return type(dtype)(*x) if hasattr(dtype, "_fields") else type(dtype)(x)


def get_iterable_path(iterable: IterableType, path: ObjPath):
    """
    Get the value at the given path in the iterable.

    Args:
        iterable: The iterable to get the value from.
        path: The path to the value.

    Returns:
        The value at the given path.
    """
    return functools.reduce(lambda a, idx: a[idx], path, iterable)


def set_iterable_path(iterable: IterableType, path: ObjPath, val: typing.Any):
    from .language import core
    assert len(path) != 0
    prev = iterable if len(path) == 1 else get_iterable_path(
        iterable, path[:-1])
    assert isinstance(prev, core.tuple)
    prev._setitem(path[-1], val)


def is_iterable(x):
    """
    Returns True if x is an iterable.

    Args:
        x: The input to check.

    Returns:
        True if x is an iterable.
    """
    from .language import core
    return isinstance(x, (list, tuple, core.tuple, core.tuple_t))


def apply_with_path(value,
                    fn: typing.Callable[[ObjPath, typing.Any], None],
                    _path=None) -> None:
    if _path is None:
        _path = ()

    if is_iterable(value):
        for idx, item in enumerate(value):
            apply_with_path(item, fn, _path=(*_path, idx))
    else:
        fn(_path, value)


def find_paths_if(
        iterable: typing.Union[IterableType, typing.Any],
        pred: typing.Callable[[ObjPath, typing.Any], bool]) -> list[ObjPath]:
    """
    Finds all paths in the iterable that satisfy the given predicate.

    Args:
        iterable: The iterable to search.
        pred: The predicate function.

    Returns:
        A list of paths that satisfy the predicate.
    """
    ret: dict[ObjPath, None] = {}

    def _impl(path: ObjPath, current: typing.Any):
        if is_iterable(current):
            for idx, item in enumerate(current):
                _impl((*path, idx), item)
        elif pred(path, current):
            ret[path] = None

    _impl((), iterable)

    return list(ret.keys())


def is_power_of_two(x):
    """
    Returns True if x is a power of 2.

    Args:
        x: The input to check.

    Returns:
        True if x is a power of 2.
    """
    return (x & (x - 1)) == 0


def validate_block_shape(shape: typing.Sequence[int]):
    """
    Validates the shape of a block.

    Args:
        shape: The shape of the block.

    Returns:
        The number of elements in the block.
    """
    numel = 1
    for i, d in enumerate(shape):
        if not isinstance(d, int):
            raise TypeError(
                f"Shape element {i} must have type `constexpr(int)`, got `constexpr({type(d)})`"
            )
        if not is_power_of_two(d):
            raise ValueError(f"Shape element {i} must be a power of 2")
        numel *= d

    if numel > TILEON_MAX_TENSOR_NUMEL:
        raise ValueError(
            f"numel ({numel}) exceeds tileon maximum tensor numel ({TILEON_MAX_TENSOR_NUMEL})"
        )
    return numel


type_canonicalisation_dict = {
    # we canonicalise all bools to be unsigned:
    "bool": "u1",
    "int1": "u1",
    "uint1": "u1",
    "i1": "u1",
    # floating-point dtypes:
    "float8e4nv": "fp8e4nv",
    "float8e5": "fp8e5",
    "float8e4b15": "fp8e4b15",
    "float8_e4m3fn": "fp8e4nv",
    "float8e4b8": "fp8e4b8",
    "float8_e4m3fnuz": "fp8e4b8",
    "float8_e5m2": "fp8e5",
    "float8e5b16": "fp8e5b16",
    "float8_e5m2fnuz": "fp8e5b16",
    "half": "fp16",
    "float16": "fp16",
    "bfloat16": "bf16",
    "float": "fp32",
    "float32": "fp32",
    "double": "fp64",
    "float64": "fp64",
    # signed integers:
    "int8": "i8",
    "int16": "i16",
    "int": "i32",
    "int32": "i32",
    "int64": "i64",
    # unsigned integers:
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
    "void": "void",
}

for v in list(type_canonicalisation_dict.values()):
    type_canonicalisation_dict[v] = v


def canonicalize_dtype(dtype):
    dtype_str = str(dtype).split(".")[-1]
    return type_canonicalisation_dict[dtype_str]


def canonicalize_ptr_dtype(dtype, is_const):
    return f"{'*k' if is_const else '*'}{canonicalize_dtype(dtype)}"


BITWIDTH_DICT: typing.Dict[str, int] = {
    **{
        f"u{n}": n
        for n in (1, 8, 16, 32, 64)
    },
    **{
        f"i{n}": n
        for n in (1, 8, 16, 32, 64)
    },
    **{
        f"fp{n}": n
        for n in (16, 32, 64)
    },
    **{
        f"fp8{suffix}": 8
        for suffix in ("e4nv", "e4b15", "e4b8", "e5", "e5b16")
    },
    "bf16": 16,
    "void": 0,
}

for k, v in type_canonicalisation_dict.items():
    BITWIDTH_DICT[k] = BITWIDTH_DICT[v]


def get_primitive_bitwidth(dtype: str) -> int:
    return BITWIDTH_DICT[dtype]


def is_namedtuple(val):
    """
    Returns True if val is a namedtuple.

    Args:
        val: The input to check.

    Returns:
        True if val is a namedtuple.
    """
    return isinstance(val, type) and issubclass(val, tuple) and hasattr(
        val, "_fields")


def convert_to_tuple_if_list(item):
    """
    Recursively convert all lists in the item to tuples.

    Args:
        item: The item to convert.

    Returns:
        The item with all lists converted to tuples.
    """
    if not isinstance(item, list):
        return item

    for i, nested_value in enumerate(item):
        item[i] = convert_to_tuple_if_list(nested_value)

    return tuple(item)


def _normalize_t(ty) -> str:
    """Normalize a type to a string."""
    from .language import core
    if isinstance(ty, str):
        ty = ty.strip()
        if ty.startswith("const "):
            ty = ty.removeprefix("const")
            ty = _normalize_t(ty)
            assert ty.startswith("*")
            return "*k" + ty[1:]
        if ty.endswith("*"):
            return "*" + _normalize_t(ty[:-1])
        if ty.startswith("*"):
            return "*" + _normalize_t(ty[1:])
        if ty.startswith("tl."):
            return _normalize_t(ty.removeprefix("tl."))
    elif isinstance(ty, core.pointer_t):
        return f"*{_normalize_t(ty.element_t)}"
    elif isinstance(ty, core.dtype):
        ty = ty.name
    elif isinstance(ty, type):
        ty = ty.__name__
    else:
        ty = str(ty)
    return type_canonicalisation_dict.get(ty.replace("_t", ""), ty)
