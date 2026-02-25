from __future__ import annotations

import os
import importlib
from contextlib import contextmanager
from typing import Any, Callable, Generator, Generic, Optional, Type, TypeVar, Union, cast, TYPE_CHECKING
from tileon._C import getenv, getenv_bool

if TYPE_CHECKING:
    from .runtime.cache import CacheManager, RemoteCacheBackend

PROPAGATE_ENV: bool = True


class Env:
    """Placeholder for empty environment variables."""
    ...


env = Env()


def set_env(key: str, value: Optional[str]) -> None:
    """
    Set the environment variable `key` to `value`.

    Args:
        key: The name of the environment variable.
        value: The value to set the environment variable to.
            If None, the environment variable is unset.
    """
    if not PROPAGATE_ENV:
        return

    if value is not None:
        os.environ[key] = value
    elif key in os.environ:
        del os.environ[key]


def to_env(val: Any) -> Optional[str]:
    """
    Convert a value to a tuple of strings that can be set as an environment variable.

    Args:
        val: The value to convert.

    Returns:
        A tuple of strings that can be set as an environment variable,
        or None if the value cannot be converted.
    """
    if val is None:
        return None
    t = type(val)
    if t is bool:
        return "1" if val else "0"
    if t is str:
        return val
    if t is int:
        return str(val)
    return None


SetType = TypeVar("SetType")
GetType = TypeVar("GetType")

_NOTHING = object()


class EnvVar(Generic[SetType, GetType]):
    """Base class for environment variables."""

    def __init__(self, key: str) -> None:
        self.key = key

    def __set_name__(self, objclass: Type[object], name: str) -> None:
        self.name = name

    def __get__(self, obj: Optional[object], objclass: Optional[Type[object]]) -> GetType:
        py_val = obj.__dict__.get(self.name, _NOTHING)
        if py_val is _NOTHING:
            return self.get()
        return self.transform(py_val)

    def get(self) -> GetType:
        raise NotImplementedError()

    def __set__(self, obj: object, value: Union[SetType, Env]) -> None:
        if isinstance(value, Env):
            obj.__dict__.pop(self.name, None)
        else:
            obj.__dict__[self.name] = value
            if env_val := to_env(value):
                set_env(self.key, env_val)

    def __delete__(self, obj: object) -> None:
        obj.__dict__.pop(self.name, None)

    def transform(self, val: SetType) -> GetType:
        """Transform the value from the `__dict__` to the desired type."""
        return cast(GetType, val)


class EnvStr(EnvVar[str, str]):
    """String environment variable."""

    def __init__(self, key: str, default: str):
        super().__init__(key)
        self.default = default

    def get(self) -> str:
        return getenv(self.key, self.default)


class EnvStrCallableDefault(EnvVar[str, str]):
    """String environment variable with a callable default factory."""

    def __init__(self, key: str, default_factory: Callable[[], str]):
        super().__init__(key)
        self.default_factory = default_factory

    def get(self) -> str:
        env_val = getenv(self.key)
        if env_val is None:
            return self.default_factory()
        return env_val


class EnvBool(EnvVar[bool, bool]):
    """Boolean environment variable."""

    def __init__(self, key: str, default: bool = False) -> None:
        super().__init__(key)
        self.default = default

    def get(self) -> bool:
        return getenv_bool(self.key, self.default)


class EnvInt(EnvVar[int, int]):
    """Integer environment variable."""

    def __init__(self, key: str, default: int = 0) -> None:
        super().__init__(key)
        self.default = default

    def get(self) -> int:
        val = getenv(self.key)
        if val is None:
            return self.default
        try:
            return int(val)
        except ValueError as e:
            raise RuntimeError(f"Unable to use {self.key}={val}: expected int") from e


class EnvOptStr(EnvVar[Optional[str], Optional[str]]):
    """Optional string environment variable."""

    def get(self) -> Optional[str]:
        return getenv(self.key)


class EnvOptBool(EnvVar[Optional[bool], Optional[bool]]):
    """Optional boolean environment variable."""

    def get(self) -> Optional[bool]:
        return getenv_bool(self.key, None)


ClassType = TypeVar("ClassType")


class EnvClass(Generic[ClassType], EnvVar[Optional[Type[ClassType]], Optional[Type[ClassType]]]):
    """
    Environment variable that reads a class from a string of the form MODULE:CLASS.
    """

    def __init__(self, key: str, type: str) -> None:
        super().__init__(key)
        # We can't pass the type directly to avoid import cycles
        self.type = type

    def get(self) -> Optional[Type[ClassType]]:
        val = getenv(self.key)
        if val is None:
            return None
        comps = val.split(":", 1)
        if len(comps) != 2:
            raise RuntimeError(f"Unable to read {self.key}: '{val}' isn't of the form MODULE:CLASS")
        cls = getattr(importlib.import_module(comps[0]), comps[1])

        if not any((c.__name__ == self.type for c in cls.mro())):
            raise RuntimeError(f"Unable to use '{val}' from {self.key}: not of type '{self.type}'")

        return cast(Type[ClassType], cls)


K = TypeVar("K", bound='Knobs')


class Knobs:
    """Base class for knobs."""

    @property
    def descriptors(self) -> dict[str, EnvVar]:
        """
        Get the environment variable descriptors.
        """
        cls = type(self)
        try:
            return cls.__knob_descriptors_cache  # type: ignore[attr-defined]
        except AttributeError:
            result = {k: v for k, v in cls.__dict__.items() if isinstance(v, EnvVar)}
            cls.__knob_descriptors_cache = result  # type: ignore[attr-defined]
            return result

    @property
    def knobs(self) -> dict[str, Any]:
        """
        Get the current knob values.
        """
        return {k: getattr(self, k) for k in self.descriptors.keys()}

    def copy(self: K) -> K:
        """
        Create a copy of the knobs.
        """
        res = type(self)()
        res.__dict__.update(self.__dict__)
        return res

    def reset(self: K) -> K:
        """
        Reset the knobs to their default values.
        """
        for knob in self.descriptors.keys():
            delattr(self, knob)
        return self

    @contextmanager
    def scope(self) -> Generator[None, None, None]:
        """
        Set the knobs to the current values of the environment variables.
        """
        try:
            initial_env = {knob.key: getenv(knob.key) for knob in self.descriptors.values()}
            orig = dict(self.__dict__)
            yield
        finally:
            self.__dict__.clear()
            self.__dict__.update(orig)

            for k, v in initial_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]


class RuntimeKnobs(Knobs):
    """Runtime knobs."""

    interpret: EnvBool = EnvBool("TILEON_INTERPRET", True)
    debug: EnvBool = EnvBool("TILEON_DEBUG", True)
    jit_post_compile_hook: EnvBool = EnvBool("TILEON_JIT_POST_COMPILE_HOOK")


class LanguageKnobs(Knobs):
    """Language knobs."""

    fp32_default: EnvOptStr = EnvOptStr("TILEON_F32_DEFAULT")
    default_fp_fusion: EnvBool = EnvBool("TILEON_DEFAULT_FP_FUSION", True)


class cache_knobs(Knobs):
    home_dir = EnvStr("TILEON_HOME", os.path.expanduser("~/"))
    dump_dir = EnvStrCallableDefault("TILEON_DUMP_DIR", lambda: cache.get_triton_dir("dump"))
    override_dir = EnvStrCallableDefault("TILEON_OVERRIDE_DIR", lambda: cache.get_triton_dir("override"))
    dir = EnvStrCallableDefault("TILEON_CACHE_DIR", lambda: cache.get_triton_dir("cache"))
    manager_class: EnvClass[CacheManager] = EnvClass("TILEON_CACHE_MANAGER", "CacheManager")
    remote_manager_class: EnvClass[RemoteCacheBackend] = EnvClass("TILEON_REMOTE_CACHE_BACKEND", "RemoteCacheBackend")

    def get_triton_dir(self, dirname: str) -> str:
        return os.path.join(self.home_dir, ".tileon", dirname)


runtime = RuntimeKnobs()
language = LanguageKnobs()
cache = cache_knobs()


def refresh_knobs():
    """
    Refresh knobs from environment variables.
    """
    runtime.interpret = EnvBool("TILEON_INTERPRET")
    runtime.debug = EnvBool("TILEON_DEBUG")
