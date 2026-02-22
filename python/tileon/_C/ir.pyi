"""
Tilen IR API
"""
from __future__ import annotations
import typing
__all__: list[str] = ['ATOMIC_OP', 'CACHE_MODIFIER', 'DESCRIPTOR_REDUCE_KIND', 'EVICTION_POLICY', 'INPUT_PRECISION', 'MEM_SEMANTIC', 'MEM_SYNC_SCOPE', 'PADDING_OPTION', 'PROPAGATE_NAN', 'ROUNDING_MODE', 'SCALE_DOT_ELEM_TYPE']
class ATOMIC_OP:
    """
    Atomic operation enumeration class: specifies the type of atomic operation
    
    Members:
    
      AND : AND operation
    
      OR : OR operation
    
      XOR : XOR operation
    
      ADD : ADD operation
    
      FADD : FADD operation
    
      MAX : MAX operation
    
      MIN : MIN operation
    
      UMAX : UMAX operation
    
      UMIN : UMIN operation
    
      XCHG : XCHG operation
    """
    ADD: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.ADD: 0>
    AND: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.AND: 2>
    FADD: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.FADD: 1>
    MAX: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.MAX: 6>
    MIN: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.MIN: 7>
    OR: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.OR: 3>
    UMAX: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.UMAX: 9>
    UMIN: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.UMIN: 8>
    XCHG: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.XCHG: 5>
    XOR: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.XOR: 4>
    __members__: typing.ClassVar[dict[str, ATOMIC_OP]]  # value = {'AND': <ATOMIC_OP.AND: 2>, 'OR': <ATOMIC_OP.OR: 3>, 'XOR': <ATOMIC_OP.XOR: 4>, 'ADD': <ATOMIC_OP.ADD: 0>, 'FADD': <ATOMIC_OP.FADD: 1>, 'MAX': <ATOMIC_OP.MAX: 6>, 'MIN': <ATOMIC_OP.MIN: 7>, 'UMAX': <ATOMIC_OP.UMAX: 9>, 'UMIN': <ATOMIC_OP.UMIN: 8>, 'XCHG': <ATOMIC_OP.XCHG: 5>}
    @typing.overload
    def __eq__(self, other: ATOMIC_OP) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: ATOMIC_OP) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class CACHE_MODIFIER:
    """
    Cache modifier enumeration class: specifies the cache behavior of a tensor/array
    
    Members:
    
      NONE : No cache modifier
    
      CA : Cache all
    
      CG : Cache global
    
      WB : Write back
    
      CS : Cache set
    
      WT : Write through
    
      CV : Cache validate
    """
    CA: typing.ClassVar[CACHE_MODIFIER]  # value = <CACHE_MODIFIER.CA: 1>
    CG: typing.ClassVar[CACHE_MODIFIER]  # value = <CACHE_MODIFIER.CG: 2>
    CS: typing.ClassVar[CACHE_MODIFIER]  # value = <CACHE_MODIFIER.CS: 4>
    CV: typing.ClassVar[CACHE_MODIFIER]  # value = <CACHE_MODIFIER.CV: 6>
    NONE: typing.ClassVar[CACHE_MODIFIER]  # value = <CACHE_MODIFIER.NONE: 0>
    WB: typing.ClassVar[CACHE_MODIFIER]  # value = <CACHE_MODIFIER.WB: 3>
    WT: typing.ClassVar[CACHE_MODIFIER]  # value = <CACHE_MODIFIER.WT: 5>
    __members__: typing.ClassVar[dict[str, CACHE_MODIFIER]]  # value = {'NONE': <CACHE_MODIFIER.NONE: 0>, 'CA': <CACHE_MODIFIER.CA: 1>, 'CG': <CACHE_MODIFIER.CG: 2>, 'WB': <CACHE_MODIFIER.WB: 3>, 'CS': <CACHE_MODIFIER.CS: 4>, 'WT': <CACHE_MODIFIER.WT: 5>, 'CV': <CACHE_MODIFIER.CV: 6>}
    @typing.overload
    def __eq__(self, other: CACHE_MODIFIER) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: CACHE_MODIFIER) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class DESCRIPTOR_REDUCE_KIND:
    """
    Descriptor reduce kind enumeration class: specifies the reduce operation for a descriptor
    
    Members:
    
      ADD : Add reduce kind
    
      MIN : Min reduce kind
    
      MAX : Max reduce kind
    
      INC : Increment reduce kind
    
      DEC : Decrement reduce kind
    
      AND : AND reduce kind
    
      OR : OR reduce kind
    
      XOR : XOR reduce kind
    """
    ADD: typing.ClassVar[DESCRIPTOR_REDUCE_KIND]  # value = <DESCRIPTOR_REDUCE_KIND.ADD: 0>
    AND: typing.ClassVar[DESCRIPTOR_REDUCE_KIND]  # value = <DESCRIPTOR_REDUCE_KIND.AND: 5>
    DEC: typing.ClassVar[DESCRIPTOR_REDUCE_KIND]  # value = <DESCRIPTOR_REDUCE_KIND.DEC: 4>
    INC: typing.ClassVar[DESCRIPTOR_REDUCE_KIND]  # value = <DESCRIPTOR_REDUCE_KIND.INC: 3>
    MAX: typing.ClassVar[DESCRIPTOR_REDUCE_KIND]  # value = <DESCRIPTOR_REDUCE_KIND.MAX: 2>
    MIN: typing.ClassVar[DESCRIPTOR_REDUCE_KIND]  # value = <DESCRIPTOR_REDUCE_KIND.MIN: 1>
    OR: typing.ClassVar[DESCRIPTOR_REDUCE_KIND]  # value = <DESCRIPTOR_REDUCE_KIND.OR: 6>
    XOR: typing.ClassVar[DESCRIPTOR_REDUCE_KIND]  # value = <DESCRIPTOR_REDUCE_KIND.XOR: 7>
    __members__: typing.ClassVar[dict[str, DESCRIPTOR_REDUCE_KIND]]  # value = {'ADD': <DESCRIPTOR_REDUCE_KIND.ADD: 0>, 'MIN': <DESCRIPTOR_REDUCE_KIND.MIN: 1>, 'MAX': <DESCRIPTOR_REDUCE_KIND.MAX: 2>, 'INC': <DESCRIPTOR_REDUCE_KIND.INC: 3>, 'DEC': <DESCRIPTOR_REDUCE_KIND.DEC: 4>, 'AND': <DESCRIPTOR_REDUCE_KIND.AND: 5>, 'OR': <DESCRIPTOR_REDUCE_KIND.OR: 6>, 'XOR': <DESCRIPTOR_REDUCE_KIND.XOR: 7>}
    @typing.overload
    def __eq__(self, other: DESCRIPTOR_REDUCE_KIND) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: DESCRIPTOR_REDUCE_KIND) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class EVICTION_POLICY:
    """
    Eviction policy enumeration class: specifies the policy of evicting a tensor/array from cache
    
    Members:
    
      NORMAL : Normal eviction policy
    
      EVICT_FIRST : Evict first eviction policy
    
      EVICT_LAST : Evict last eviction policy
    """
    EVICT_FIRST: typing.ClassVar[EVICTION_POLICY]  # value = <EVICTION_POLICY.EVICT_FIRST: 1>
    EVICT_LAST: typing.ClassVar[EVICTION_POLICY]  # value = <EVICTION_POLICY.EVICT_LAST: 2>
    NORMAL: typing.ClassVar[EVICTION_POLICY]  # value = <EVICTION_POLICY.NORMAL: 0>
    __members__: typing.ClassVar[dict[str, EVICTION_POLICY]]  # value = {'NORMAL': <EVICTION_POLICY.NORMAL: 0>, 'EVICT_FIRST': <EVICTION_POLICY.EVICT_FIRST: 1>, 'EVICT_LAST': <EVICTION_POLICY.EVICT_LAST: 2>}
    @typing.overload
    def __eq__(self, other: EVICTION_POLICY) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: EVICTION_POLICY) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class INPUT_PRECISION:
    """
    Input precision enumeration class: specifies the precision of input data
    
    Members:
    
      TF32 : TF32 precision
    
      TF32x3 : TF32x3 precision
    
      IEEE : IEEE precision
    
      BF16x3 : BF16x3 precision
    
      BF16x6 : BF16x6 precision
    """
    BF16x3: typing.ClassVar[INPUT_PRECISION]  # value = <INPUT_PRECISION.BF16x3: 3>
    BF16x6: typing.ClassVar[INPUT_PRECISION]  # value = <INPUT_PRECISION.BF16x6: 4>
    IEEE: typing.ClassVar[INPUT_PRECISION]  # value = <INPUT_PRECISION.IEEE: 2>
    TF32: typing.ClassVar[INPUT_PRECISION]  # value = <INPUT_PRECISION.TF32: 0>
    TF32x3: typing.ClassVar[INPUT_PRECISION]  # value = <INPUT_PRECISION.TF32x3: 1>
    __members__: typing.ClassVar[dict[str, INPUT_PRECISION]]  # value = {'TF32': <INPUT_PRECISION.TF32: 0>, 'TF32x3': <INPUT_PRECISION.TF32x3: 1>, 'IEEE': <INPUT_PRECISION.IEEE: 2>, 'BF16x3': <INPUT_PRECISION.BF16x3: 3>, 'BF16x6': <INPUT_PRECISION.BF16x6: 4>}
    @typing.overload
    def __eq__(self, other: INPUT_PRECISION) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: INPUT_PRECISION) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class MEM_SEMANTIC:
    """
    Memory semantic for atomic operations.
    
    Members:
    
      ACQUIRE_RELEASE : Acquire and release memory
    
      ACQUIRE : Acquire memory
    
      RELEASE : Release memory
    
      RELAXED : Relaxed memory semantic
    """
    ACQUIRE: typing.ClassVar[MEM_SEMANTIC]  # value = <MEM_SEMANTIC.ACQUIRE: 1>
    ACQUIRE_RELEASE: typing.ClassVar[MEM_SEMANTIC]  # value = <MEM_SEMANTIC.ACQUIRE_RELEASE: 0>
    RELAXED: typing.ClassVar[MEM_SEMANTIC]  # value = <MEM_SEMANTIC.RELAXED: 3>
    RELEASE: typing.ClassVar[MEM_SEMANTIC]  # value = <MEM_SEMANTIC.RELEASE: 2>
    __members__: typing.ClassVar[dict[str, MEM_SEMANTIC]]  # value = {'ACQUIRE_RELEASE': <MEM_SEMANTIC.ACQUIRE_RELEASE: 0>, 'ACQUIRE': <MEM_SEMANTIC.ACQUIRE: 1>, 'RELEASE': <MEM_SEMANTIC.RELEASE: 2>, 'RELAXED': <MEM_SEMANTIC.RELAXED: 3>}
    @typing.overload
    def __eq__(self, other: MEM_SEMANTIC) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: MEM_SEMANTIC) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class MEM_SYNC_SCOPE:
    """
    Memory synchronization scope enumeration class: specifies the scope of memory synchronization
    
    Members:
    
      GPU : GPU memory scope
    
      CTA : CTA memory scope
    
      SYSTEM : System memory scope
    """
    CTA: typing.ClassVar[MEM_SYNC_SCOPE]  # value = <MEM_SYNC_SCOPE.CTA: 1>
    GPU: typing.ClassVar[MEM_SYNC_SCOPE]  # value = <MEM_SYNC_SCOPE.GPU: 0>
    SYSTEM: typing.ClassVar[MEM_SYNC_SCOPE]  # value = <MEM_SYNC_SCOPE.SYSTEM: 2>
    __members__: typing.ClassVar[dict[str, MEM_SYNC_SCOPE]]  # value = {'GPU': <MEM_SYNC_SCOPE.GPU: 0>, 'CTA': <MEM_SYNC_SCOPE.CTA: 1>, 'SYSTEM': <MEM_SYNC_SCOPE.SYSTEM: 2>}
    @typing.overload
    def __eq__(self, other: MEM_SYNC_SCOPE) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: MEM_SYNC_SCOPE) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PADDING_OPTION:
    """
    Filling option enumeration class: specifies the type of filling value when padding a tensor/array
    
    Members:
    
      PAD_ZERO : Pad with zero
    
      PAD_NAN : Pad with NaN
    """
    PAD_NAN: typing.ClassVar[PADDING_OPTION]  # value = <PADDING_OPTION.PAD_NAN: 1>
    PAD_ZERO: typing.ClassVar[PADDING_OPTION]  # value = <PADDING_OPTION.PAD_ZERO: 0>
    __members__: typing.ClassVar[dict[str, PADDING_OPTION]]  # value = {'PAD_ZERO': <PADDING_OPTION.PAD_ZERO: 0>, 'PAD_NAN': <PADDING_OPTION.PAD_NAN: 1>}
    @typing.overload
    def __eq__(self, other: PADDING_OPTION) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: PADDING_OPTION) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PROPAGATE_NAN:
    """
    NaN propagation strategy enumeration class: specifies how NaN values are propagated in operations
    
    Members:
    
      NONE : Do not propagate NaN
    
      ALL : Propagate all NaN
    """
    ALL: typing.ClassVar[PROPAGATE_NAN]  # value = <PROPAGATE_NAN.ALL: 65535>
    NONE: typing.ClassVar[PROPAGATE_NAN]  # value = <PROPAGATE_NAN.NONE: 0>
    __members__: typing.ClassVar[dict[str, PROPAGATE_NAN]]  # value = {'NONE': <PROPAGATE_NAN.NONE: 0>, 'ALL': <PROPAGATE_NAN.ALL: 65535>}
    @typing.overload
    def __eq__(self, other: PROPAGATE_NAN) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: PROPAGATE_NAN) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ROUNDING_MODE:
    """
    Rounding mode enumeration class: specifies the rounding behavior of a floating-point operation
    
    Members:
    
      RTZ : Round towards zero
    
      RTNE : Round towards nearest even
    """
    RTNE: typing.ClassVar[ROUNDING_MODE]  # value = <ROUNDING_MODE.RTNE: 1>
    RTZ: typing.ClassVar[ROUNDING_MODE]  # value = <ROUNDING_MODE.RTZ: 0>
    __members__: typing.ClassVar[dict[str, ROUNDING_MODE]]  # value = {'RTZ': <ROUNDING_MODE.RTZ: 0>, 'RTNE': <ROUNDING_MODE.RTNE: 1>}
    @typing.overload
    def __eq__(self, other: ROUNDING_MODE) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: ROUNDING_MODE) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SCALE_DOT_ELEM_TYPE:
    """
    Scale dot element type enumeration class: specifies the precision of scale dot product
    
    Members:
    
      E4M3 : E4M3 precision
    
      E5M2 : E5M2 precision
    
      E2M3 : E2M3 precision
    
      E3M2 : E3M2 precision
    
      E2M1 : E2M1 precision
    
      BF16 : BF16 precision
    
      FP16 : FP16 precision
    """
    BF16: typing.ClassVar[SCALE_DOT_ELEM_TYPE]  # value = <SCALE_DOT_ELEM_TYPE.BF16: 5>
    E2M1: typing.ClassVar[SCALE_DOT_ELEM_TYPE]  # value = <SCALE_DOT_ELEM_TYPE.E2M1: 4>
    E2M3: typing.ClassVar[SCALE_DOT_ELEM_TYPE]  # value = <SCALE_DOT_ELEM_TYPE.E2M3: 2>
    E3M2: typing.ClassVar[SCALE_DOT_ELEM_TYPE]  # value = <SCALE_DOT_ELEM_TYPE.E3M2: 3>
    E4M3: typing.ClassVar[SCALE_DOT_ELEM_TYPE]  # value = <SCALE_DOT_ELEM_TYPE.E4M3: 0>
    E5M2: typing.ClassVar[SCALE_DOT_ELEM_TYPE]  # value = <SCALE_DOT_ELEM_TYPE.E5M2: 1>
    FP16: typing.ClassVar[SCALE_DOT_ELEM_TYPE]  # value = <SCALE_DOT_ELEM_TYPE.FP16: 6>
    __members__: typing.ClassVar[dict[str, SCALE_DOT_ELEM_TYPE]]  # value = {'E4M3': <SCALE_DOT_ELEM_TYPE.E4M3: 0>, 'E5M2': <SCALE_DOT_ELEM_TYPE.E5M2: 1>, 'E2M3': <SCALE_DOT_ELEM_TYPE.E2M3: 2>, 'E3M2': <SCALE_DOT_ELEM_TYPE.E3M2: 3>, 'E2M1': <SCALE_DOT_ELEM_TYPE.E2M1: 4>, 'BF16': <SCALE_DOT_ELEM_TYPE.BF16: 5>, 'FP16': <SCALE_DOT_ELEM_TYPE.FP16: 6>}
    @typing.overload
    def __eq__(self, other: SCALE_DOT_ELEM_TYPE) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: SCALE_DOT_ELEM_TYPE) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
