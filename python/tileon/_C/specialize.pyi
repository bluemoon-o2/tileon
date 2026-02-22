"""
Tilen Specialize API
"""
from __future__ import annotations
import typing
__all__: list[str] = ['native_specialize_impl']
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
