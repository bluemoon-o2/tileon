"""
Tileon C++ API
"""
from __future__ import annotations
import typing
from . import interpreter
from . import ir
__all__: list[str] = ['get_cache_invalidating_env_vars', 'getenv', 'getenv_bool', 'interpreter', 'ir', 'native_specialize_impl']
def get_cache_invalidating_env_vars() -> dict[str, str]:
    """
        Get a dictionary of environment variables that affect compilation cache.

        Returns:
            dict: A dictionary mapping environment variable names to their values.
                  Boolean-like values are normalized to "true" or "false".
    """
def getenv(name: str, default_val: typing.Any = None) -> typing.Any:
    """
        Get an environment variable.

        Args:
            name (str): The name of the environment variable.
            default_val (object, optional): The value to return if the environment variable is not set. Defaults to None.

        Returns:
            str or object: The value of the environment variable, or default_val if not set.
    """
def getenv_bool(name: str, default_val: typing.Any) -> typing.Any:
    """
        Get an environment variable as a boolean.

        Args:
            name (str): The name of the environment variable.
            default_val (object): The value to return if the environment variable is not set.

        Returns:
            bool or object: True if the environment variable is set to a truthy value ('1', 'y', 'on', 'yes', 'true'),
                            False if it is set to a falsy value, or default_val if not set.
    """
def native_specialize_impl(self: typing.Any, arg: typing.Any, is_const: bool, specialize: bool, align: bool) -> tuple:
    """
            Specialize an argument.

            Args:
                self: The backend instance.
                arg: The argument to specialize.
                is_const: Whether the argument is constant.
                specialize: Whether to specialize on value.
                align: Whether to align.

            Returns:
                A tuple of (type, key) where type is the specialized type and key is the specialized key.
    """
