#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_NAMESPACE_BEGIN(IR)

enum class PaddingOption { PAD_ZERO, PAD_NAN };
enum class CacheModifier { NONE, CA, CG, WB, CS, WT, CV };
enum class MemSemantic { ACQUIRE_RELEASE, ACQUIRE, RELEASE, RELAXED };
enum class EvictionPolicy { NORMAL, EVICT_FIRST, EVICT_LAST };
enum class RMWOp { ADD, FADD, AND, OR, XOR, XCHG, MAX, MIN, UMIN, UMAX };
enum class DescriptorReduceKind { ADD, MIN, MAX, INC, DEC, AND, OR, XOR };
enum class MemSyncScope { GPU, CTA, SYSTEM };
enum class RoundingMode { RTZ, RTNE };
enum class PropagateNan { NONE = 0, ALL = 0xFFFF };
enum class InputPrecision { TF32, TF32x3, IEEE, BF16x3, BF16x6 };
enum class ScaleDotElemType { E4M3, E5M2, E2M3, E3M2, E2M1, BF16, FP16 };

PYBIND11_NAMESPACE_END(IR)

using namespace IR;

void init_ir(py::module_ &&m) {
  py::enum_<PaddingOption>(m, "PADDING_OPTION", py::module_local(), 
    "Filling option enumeration class: specifies the type of filling value when padding a tensor/array"
  )
    .value("PAD_ZERO", PaddingOption::PAD_ZERO, "Pad with zero")
    .value("PAD_NAN", PaddingOption::PAD_NAN, "Pad with NaN");

  py::enum_<CacheModifier>(m, "CACHE_MODIFIER", py::module_local(), 
    "Cache modifier enumeration class: specifies the cache behavior of a tensor/array"
  )
    .value("NONE", CacheModifier::NONE, "No cache modifier")
    .value("CA", CacheModifier::CA, "Cache all")
    .value("CG", CacheModifier::CG, "Cache global")
    .value("WB", CacheModifier::WB, "Write back")
    .value("CS", CacheModifier::CS, "Cache set")
    .value("WT", CacheModifier::WT, "Write through")
    .value("CV", CacheModifier::CV, "Cache validate");

  py::enum_<RoundingMode>(m, "ROUNDING_MODE", py::module_local(), 
    "Rounding mode enumeration class: specifies the rounding behavior of a floating-point operation"
  )
    .value("RTZ", RoundingMode::RTZ, "Round towards zero")
    .value("RTNE", RoundingMode::RTNE, "Round towards nearest even");

  py::enum_<MemSemantic>(m, "MEM_SEMANTIC", py::module_local(), 
    "Memory semantic for atomic operations."
  )
    .value("ACQUIRE_RELEASE", MemSemantic::ACQUIRE_RELEASE, "Acquire and release memory")
    .value("ACQUIRE", MemSemantic::ACQUIRE, "Acquire memory")
    .value("RELEASE", MemSemantic::RELEASE, "Release memory")
    .value("RELAXED", MemSemantic::RELAXED, "Relaxed memory semantic");

  py::enum_<MemSyncScope>(m, "MEM_SYNC_SCOPE", py::module_local(), 
    "Memory synchronization scope enumeration class: specifies the scope of memory synchronization"
  )
    .value("GPU", MemSyncScope::GPU, "GPU memory scope")
    .value("CTA", MemSyncScope::CTA, "CTA memory scope")
    .value("SYSTEM", MemSyncScope::SYSTEM, "System memory scope");

  py::enum_<EvictionPolicy>(m, "EVICTION_POLICY", py::module_local(), 
    "Eviction policy enumeration class: specifies the policy of evicting a tensor/array from cache"
  )
    .value("NORMAL", EvictionPolicy::NORMAL, "Normal eviction policy")
    .value("EVICT_FIRST", EvictionPolicy::EVICT_FIRST, "Evict first eviction policy")
    .value("EVICT_LAST", EvictionPolicy::EVICT_LAST, "Evict last eviction policy");

  py::enum_<RMWOp>(m, "ATOMIC_OP", py::module_local(), 
    "Atomic operation enumeration class: specifies the type of atomic operation"
  )
    .value("AND", RMWOp::AND, "AND operation")
    .value("OR", RMWOp::OR, "OR operation")
    .value("XOR", RMWOp::XOR, "XOR operation")
    .value("ADD", RMWOp::ADD, "ADD operation")
    .value("FADD", RMWOp::FADD, "FADD operation")
    .value("MAX", RMWOp::MAX, "MAX operation")
    .value("MIN", RMWOp::MIN, "MIN operation")
    .value("UMAX", RMWOp::UMAX, "UMAX operation")
    .value("UMIN", RMWOp::UMIN, "UMIN operation")
    .value("XCHG", RMWOp::XCHG, "XCHG operation");

  py::enum_<DescriptorReduceKind>(m, "DESCRIPTOR_REDUCE_KIND", py::module_local(), 
    "Descriptor reduce kind enumeration class: specifies the reduce operation for a descriptor"
  )
    .value("ADD", DescriptorReduceKind::ADD, "Add reduce kind")
    .value("MIN", DescriptorReduceKind::MIN, "Min reduce kind")
    .value("MAX", DescriptorReduceKind::MAX, "Max reduce kind")
    .value("INC", DescriptorReduceKind::INC, "Increment reduce kind")
    .value("DEC", DescriptorReduceKind::DEC, "Decrement reduce kind")
    .value("AND", DescriptorReduceKind::AND, "AND reduce kind")
    .value("OR", DescriptorReduceKind::OR, "OR reduce kind")
    .value("XOR", DescriptorReduceKind::XOR, "XOR reduce kind");

  py::enum_<PropagateNan>(m, "PROPAGATE_NAN", py::module_local(), 
    "NaN propagation strategy enumeration class: specifies how NaN values are propagated in operations"
  )
    .value("NONE", PropagateNan::NONE, "Do not propagate NaN")
    .value("ALL", PropagateNan::ALL, "Propagate all NaN");

  py::enum_<InputPrecision>(m, "INPUT_PRECISION", py::module_local(), 
    "Input precision enumeration class: specifies the precision of input data"
  )
    .value("TF32", InputPrecision::TF32, "TF32 precision")
    .value("TF32x3", InputPrecision::TF32x3, "TF32x3 precision")
    .value("IEEE", InputPrecision::IEEE, "IEEE precision")
    .value("BF16x3", InputPrecision::BF16x3, "BF16x3 precision")
    .value("BF16x6", InputPrecision::BF16x6, "BF16x6 precision");

  py::enum_<ScaleDotElemType>(m, "SCALE_DOT_ELEM_TYPE", py::module_local(), 
    "Scale dot element type enumeration class: specifies the precision of scale dot product"
  )
    .value("E4M3", ScaleDotElemType::E4M3, "E4M3 precision")
    .value("E5M2", ScaleDotElemType::E5M2, "E5M2 precision")
    .value("E2M3", ScaleDotElemType::E2M3, "E2M3 precision")
    .value("E3M2", ScaleDotElemType::E3M2, "E3M2 precision")
    .value("E2M1", ScaleDotElemType::E2M1, "E2M1 precision")
    .value("BF16", ScaleDotElemType::BF16, "BF16 precision")
    .value("FP16", ScaleDotElemType::FP16, "FP16 precision");

}
