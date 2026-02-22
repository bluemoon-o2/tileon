#include <atomic>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <type_traits>
#include <omp.h>
#include <algorithm>

#ifdef _MSC_VER
#include <intrin.h>
static int __builtin_clzll(unsigned long long x) {
    unsigned long index;
    if (_BitScanReverse64(&index, x)) {
        return 63 - index;
    }
    return 64;
}
#endif

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_NAMESPACE_BEGIN(Interpreter)

PYBIND11_NAMESPACE_BEGIN(Float16Bits)
// Float16: 1 sign + 5 exponent + 10 significand
constexpr uint16_t SIGN_MASK = 0x8000u;
constexpr uint8_t SIGN_POS = 15;
// Exponent: bit14~bit10 (5 bits)
constexpr uint16_t EXP_MASK = 0x7c00u;
constexpr uint8_t EXP_BITS = 5;
constexpr uint8_t EXP_POS = 10;
constexpr uint16_t EXP_ALL_ZERO = 0x0000u;
constexpr uint8_t EXP_BIAS = 15;
// Significand: bit9~bit0 (10 bits)
constexpr uint16_t SIG_MASK = 0x03ffu;
constexpr uint8_t SIG_BITS = 10;
constexpr uint8_t SIG_POS = 0;
// Implicit bit: bit10 (1 << 10)
constexpr uint16_t IMPLICIT_BIT = 0x0400u;
PYBIND11_NAMESPACE_END(Float16Bits)

PYBIND11_NAMESPACE_BEGIN(Float32Bits)
// Float32: 1 sign + 8 exponent + 23 significand
constexpr uint32_t SIGN_MASK = 0x80000000u;
constexpr uint8_t SIGN_POS = 31;
// Exponent: bit30~bit23 (8 bits)
constexpr uint32_t EXP_MASK = 0x7f800000u;
constexpr uint8_t EXP_BITS = 8;
constexpr uint8_t EXP_POS = 23;
constexpr uint32_t EXP_ALL_ZERO = 0x00000000u;
constexpr uint8_t EXP_BIAS = 127;
constexpr uint32_t EXP_MAX = 0x47800000u;
// Significand: bit22~bit0 (23 bits)
constexpr uint32_t SIG_MASK = 0x007fffffu;
constexpr uint8_t SIG_BITS = 23;
constexpr uint8_t SIG_POS = 0;
// Implicit bit: bit23 (1 << 23)
constexpr uint32_t IMPLICIT_BIT = 0x00800000u;
PYBIND11_NAMESPACE_END(Float32Bits)

PYBIND11_NAMESPACE_BEGIN(Float32BitsToFloat16Bits)
// Shift positions
constexpr uint8_t SIGN_SHIFT = 16;
constexpr uint8_t SIG_SHIFT = 13;
constexpr uint8_t SIG_SUBNORMAL_SHIFT_BASE = 113;
// Thresholds
constexpr uint32_t EXP_OVERFLOW_THRESHOLD = 0x47800000u;
constexpr uint32_t EXP_UNDERFLOW_THRESHOLD = 0x38000000u;
constexpr uint32_t EXP_UNDERFLOW_TO_ZERO_THRESHOLD = 0x33000000u;
// Round to even
constexpr uint32_t ROUND_CHECK_MASK = 0x00003fffu;  // the lower 14 bits are 1
constexpr uint32_t ROUND_ADD_VALUE = 0x00001000u;   // the 13th digit is 1
constexpr uint32_t ROUND_REMAIN_MASK = 0x000007ffu; // the lower 11 bits are 1

PYBIND11_NAMESPACE_END(Float32BitsToFloat16Bits)

struct npy_half {
  uint16_t value;
};

constexpr uint32_t ToFloatBits(uint16_t h) {
  uint16_t h_exp = (h & Float16Bits::EXP_MASK);
  uint32_t f_sgn = ((uint32_t)h & Float16Bits::SIGN_MASK) << Float32BitsToFloat16Bits::SIGN_SHIFT;
  switch (h_exp) {
  case Float16Bits::EXP_ALL_ZERO: { // 0 or subnormal
    uint16_t h_sig = (h & Float16Bits::SIG_MASK);
    if (h_sig == 0) {
      return f_sgn;
    }
    h_sig <<= 1;
    while ((h_sig & Float16Bits::IMPLICIT_BIT) == 0) {
      h_sig <<= 1;
      h_exp++;
    }
    uint32_t f_exp = ((uint32_t)(Float32Bits::EXP_BIAS - Float16Bits::EXP_BIAS - h_exp)) << Float32Bits::EXP_POS;
    uint32_t f_sig = ((uint32_t)(h_sig & Float16Bits::SIG_MASK)) << Float32BitsToFloat16Bits::SIG_SHIFT;
    return f_sgn + f_exp + f_sig;
  }
  case Float16Bits::EXP_MASK: // inf or NaN
    // All-ones exponent and a copy of the significand
    return f_sgn + Float32Bits::EXP_MASK +
           (((uint32_t)(h & Float16Bits::SIG_MASK))
            << Float32BitsToFloat16Bits::SIG_SHIFT);
  default:
    // Just need to adjust the exponent and shift
    return f_sgn +
           (((uint32_t)(h & (Float16Bits::EXP_MASK | Float16Bits::SIG_MASK)) +
             ((Float32Bits::EXP_BIAS - Float16Bits::EXP_BIAS)
              << Float16Bits::EXP_POS))
            << Float32BitsToFloat16Bits::SIG_SHIFT);
  }
}

template <bool raise_overflow = true, bool raise_underflow = true, bool round_even = true>
inline uint16_t FromFloatBits(uint32_t f) {
  uint32_t f_exp, f_sig;
  uint16_t h_sgn, h_exp, h_sig;

  h_sgn = (uint16_t)((f & Float32Bits::SIGN_MASK) >> Float32BitsToFloat16Bits::SIGN_SHIFT);
  f_exp = (f & Float32Bits::EXP_MASK);

  // Exponent overflow/NaN converts to signed inf/NaN
  if (f_exp >= Float32BitsToFloat16Bits::EXP_OVERFLOW_THRESHOLD) {
    if (f_exp == Float32Bits::EXP_MASK) {
      f_sig = (f & Float32Bits::SIG_MASK);
      if (f_sig != 0) {
        // NaN - propagate the flag in the significand
        uint16_t ret = (uint16_t)(Float16Bits::EXP_MASK + (f_sig >> Float32BitsToFloat16Bits::SIG_SHIFT));
        if (ret == Float16Bits::EXP_MASK) {
          ret++;
        } // Defend against NaN payload miss
        return h_sgn + ret;
      } else {
        // signed inf
        return (uint16_t)(h_sgn + Float16Bits::EXP_MASK);
      }
    } else {
      if constexpr (raise_overflow) {
        throw std::overflow_error("overflow to signed inf");
      }
      return (uint16_t)(h_sgn + Float16Bits::EXP_MASK);
    }
  }

  // Exponent underflow converts to a subnormal half or signed zero
  if (f_exp <= Float32BitsToFloat16Bits::EXP_UNDERFLOW_THRESHOLD) {
    // signed zero
    if (f_exp < Float32BitsToFloat16Bits::EXP_UNDERFLOW_TO_ZERO_THRESHOLD) {
      if constexpr (raise_underflow) {
        if ((f & Float32Bits::SIG_MASK) != 0) {
          throw std::underflow_error("underflow to signed zero");
        }
      }
      return h_sgn;
    }
    // subnormal half
    f_exp >>= Float32Bits::EXP_POS;
    f_sig = Float32Bits::IMPLICIT_BIT | (f & Float32Bits::SIG_MASK);
    if constexpr (raise_underflow) {
      if ((f_sig & (((uint32_t)1 << (Float32Bits::EXP_BIAS - 1 - f_exp)) - 1)) != 0) {
        throw std::underflow_error("underflow to subnormal");
      }
    }
    f_sig >>= (Float32BitsToFloat16Bits::SIG_SUBNORMAL_SHIFT_BASE - f_exp);

    if constexpr (round_even) {
      if (((f_sig & Float32BitsToFloat16Bits::ROUND_CHECK_MASK) != Float32BitsToFloat16Bits::ROUND_ADD_VALUE) ||
          (f & Float32BitsToFloat16Bits::ROUND_REMAIN_MASK)) {
        f_sig += Float32BitsToFloat16Bits::ROUND_ADD_VALUE;
      }
    } else {
      f_sig += Float32BitsToFloat16Bits::ROUND_ADD_VALUE;
    }
    h_sig = (uint16_t)(f_sig >> Float32BitsToFloat16Bits::SIG_SHIFT);

    return (uint16_t)(h_sgn + h_sig);
  }

  // Regular case with no overflow or underflow
  h_exp = (uint16_t)((f_exp - Float32BitsToFloat16Bits::EXP_UNDERFLOW_THRESHOLD) >> Float32BitsToFloat16Bits::SIG_SHIFT);
  f_sig = (f & Float32Bits::SIG_MASK);

  if constexpr (round_even) {
    if ((f_sig & Float32BitsToFloat16Bits::ROUND_CHECK_MASK) != Float32BitsToFloat16Bits::ROUND_ADD_VALUE) {
      f_sig += Float32BitsToFloat16Bits::ROUND_ADD_VALUE;
    }
  } else {
    f_sig += Float32BitsToFloat16Bits::ROUND_ADD_VALUE;
  }
  h_sig = (uint16_t)(f_sig >> Float32BitsToFloat16Bits::SIG_SHIFT);

  h_sig += h_exp;
  if constexpr (raise_overflow) {
    if (h_sig == Float16Bits::EXP_MASK) {
      throw std::overflow_error("overflow to signed inf");
    }
  }
  return (uint16_t)(h_sgn + h_sig);
}

template <typename To, typename From>
inline To BitCast(const From &from) noexcept {
  static_assert(sizeof(To) == sizeof(From), "both data types must have the same size");
  static_assert(std::is_trivially_copyable_v<To> && std::is_trivially_copyable_v<From>, "both data types must be trivially copyable");
  To to;
  memcpy(&to, &from, sizeof(from));
  return to;
}

npy_half npy_float_to_half(float f) {
  return {FromFloatBits(BitCast<uint32_t>(f))};
}

float npy_half_to_float(npy_half h) {
  return BitCast<float>(ToFloatBits(h.value));
}

template <int InW, int InM, int InB, int OutW, int OutM, int OutB>
inline uint64_t convert_element(uint64_t in_val) {
  constexpr uint64_t in_mant_mask = (1ULL << InM) - 1;
  constexpr uint64_t out_mant_mask = (1ULL << OutM) - 1;
  constexpr uint64_t in_sign_pos = InW - 1;
  constexpr uint64_t out_sign_pos = OutW - 1;
  constexpr uint64_t in_exp_mask = (1ULL << (InW - InM - 1)) - 1;
  constexpr uint64_t out_exp_max = (1ULL << (OutW - OutM - 1)) - 1;
  constexpr int shift_diff = InM - OutM;

  uint64_t sign = (in_val >> in_sign_pos) & 1;
  uint64_t in_exp = (in_val >> InM) & in_exp_mask;
  uint64_t in_mant = in_val & in_mant_mask;

  constexpr bool in_is_e4m3fn = (InW == 8 && InM == 3 && InB == 7);
  constexpr bool in_is_e4m3fnuz = (InW == 8 && InM == 3 && InB == 8);
  constexpr bool in_is_e5m2 = (InW == 8 && InM == 2 && InB == 15);
  constexpr bool in_is_e5m2fnuz = (InW == 8 && InM == 2 && InB == 16);
  constexpr bool in_is_fnuz = in_is_e4m3fnuz || in_is_e5m2fnuz;
  constexpr bool in_no_inf = in_is_e4m3fn || in_is_fnuz;

  constexpr bool out_is_e4m3fn = (OutW == 8 && OutM == 3 && OutB == 7);
  constexpr bool out_is_e4m3fnuz = (OutW == 8 && OutM == 3 && OutB == 8);
  constexpr bool out_is_e5m2 = (OutW == 8 && OutM == 2 && OutB == 15);
  constexpr bool out_is_e5m2fnuz = (OutW == 8 && OutM == 2 && OutB == 16);
  constexpr bool out_is_fnuz = out_is_e4m3fnuz || out_is_e5m2fnuz;

  bool in_is_nan = false;
  bool in_is_inf = false;
  if constexpr (in_is_e4m3fn) {
    in_is_nan = (in_exp == in_exp_mask && in_mant == in_mant_mask);
  } else if constexpr (in_is_fnuz) {
    in_is_nan = (in_val == 0x80);
  } else {
    if (in_exp == in_exp_mask) {
        in_is_inf = (in_mant == 0);
        in_is_nan = (in_mant != 0);
    }
  }

  if (in_is_nan || in_is_inf) {
    if constexpr (out_is_e4m3fn) {
      if (in_is_nan) return 0x7f;
      return (sign << out_sign_pos) | 0x7f;
    }
    if constexpr (out_is_fnuz) {
      return 0x80;
    }
    if constexpr (out_is_e5m2) {
      if (in_is_nan) return 0x7f;
      return (sign << out_sign_pos) | 0x7c;
    }
    constexpr uint64_t out_exp_bits = OutW - OutM - 1;
    constexpr uint64_t out_exp_all_ones = (1ULL << out_exp_bits) - 1;
    uint64_t out_mant = in_is_nan ? out_mant_mask : 0;
    uint64_t out_sign = in_is_nan ? 0 : (sign << out_sign_pos);
    return out_sign | (out_exp_all_ones << OutM) | out_mant;
  }

  if (in_exp == 0 && in_mant == 0) {
    return out_is_fnuz ? 0 : (sign << out_sign_pos);
  }

  int true_exp;
  uint64_t normalized_mant;

  // Branchless normal/subnormal handling using bitwise ops where possible
  // or optimize the common case (normal) to be first
  if (in_exp != 0 && (in_no_inf || in_exp != in_exp_mask)) {
    // Normal case (Most common)
    true_exp = (int)in_exp - InB;
    normalized_mant = in_mant;
  } else if (in_exp == 0) {
    // Subnormal input
    int lz = __builtin_clzll(in_mant);
    int shift = lz - (64 - InM);
    true_exp = 1 - InB - (shift + 1);
    normalized_mant = (in_mant << (shift + 1)) & in_mant_mask;
  } else {
    // Inf/NaN (handled above but compiler might need help)
    uint64_t out_mant = (in_mant == 0) ? 0 : (1ULL << (OutM - 1));
    return (sign << out_sign_pos) | (out_exp_max << OutM) | out_mant;
  }

  int out_exp_val = true_exp + OutB;
  uint64_t final_exp, final_mant;

  // Check bounds
  if (out_exp_val > 0 && out_exp_val < out_exp_max) {
      // Normal output (Common case)
      final_exp = (uint64_t)out_exp_val;
      if constexpr (shift_diff > 0) {
          // Downsample with rounding
          constexpr uint64_t round_bit = 1ULL << (shift_diff - 1);
          constexpr uint64_t sticky_mask = round_bit - 1;
          bool lsb = (normalized_mant >> shift_diff) & 1;
          bool round = (normalized_mant & round_bit) != 0;
          bool sticky = (normalized_mant & sticky_mask) != 0;
          
          final_mant = normalized_mant >> shift_diff;
          if (round && (lsb || sticky)) {
              final_mant++;
              if (final_mant > out_mant_mask) {
                  final_mant = 0;
                  final_exp++;
              }
          }
      } else {
          final_mant = normalized_mant << (-shift_diff);
      }
  } else if (out_exp_val >= (int64_t)out_exp_max) {
      if constexpr (out_is_e4m3fn) {
          final_exp = out_exp_max;
          final_mant = out_mant_mask;
      } else if constexpr (out_is_fnuz) {
          return 0x80;
      } else {
          final_exp = out_exp_max;
          final_mant = 0;
      }
  } else {
      // Underflow
      int sub_shift = 1 - out_exp_val;
      int total_shift = shift_diff + sub_shift;
      
      final_exp = 0;
      // Add implicit bit
      uint64_t full_mant = normalized_mant | (1ULL << InM);
      
      if (total_shift >= 64) {
          final_mant = 0;
      } else if (total_shift > 0) {
          uint64_t round_bit = 1ULL << (total_shift - 1);
          uint64_t sticky_mask = round_bit - 1;
          bool lsb = (full_mant >> total_shift) & 1;
          bool round = (full_mant & round_bit) != 0;
          bool sticky = (full_mant & sticky_mask) != 0;
          
          final_mant = full_mant >> total_shift;
          if (round && (lsb || sticky)) {
              final_mant++;
          }
      } else {
          final_mant = full_mant << (-total_shift);
      }
  }

  uint64_t result = (sign << out_sign_pos) | (final_exp << OutM) | (final_mant & out_mant_mask);
  if constexpr (out_is_fnuz) {
      if (final_exp == 0 && (final_mant & out_mant_mask) == 0) return 0;
  }
  return result;
}

// Fallback for non-template cases (if any) or simplified call
struct FPDesc {
  int width;
  int mantissa;
  int bias;
};

inline uint64_t convert_element_runtime(uint64_t in_val, const FPDesc& in, const FPDesc& out) {
  const uint64_t in_mant_mask = (1ULL << in.mantissa) - 1;
  const uint64_t out_mant_mask = (1ULL << out.mantissa) - 1;
  const uint64_t in_sign_pos = in.width - 1;
  const uint64_t out_sign_pos = out.width - 1;
  const uint64_t in_exp_mask = (1ULL << (in.width - in.mantissa - 1)) - 1;
  const uint64_t out_exp_max = (1ULL << (out.width - out.mantissa - 1)) - 1;
  const int shift_diff = in.mantissa - out.mantissa;

  uint64_t sign = (in_val >> in_sign_pos) & 1;
  uint64_t in_exp = (in_val >> in.mantissa) & in_exp_mask;
  uint64_t in_mant = in_val & in_mant_mask;
  
  const bool in_is_e4m3fn = (in.width == 8 && in.mantissa == 3 && in.bias == 7);
  const bool in_is_e4m3fnuz = (in.width == 8 && in.mantissa == 3 && in.bias == 8);
  const bool in_is_e5m2 = (in.width == 8 && in.mantissa == 2 && in.bias == 15);
  const bool in_is_e5m2fnuz = (in.width == 8 && in.mantissa == 2 && in.bias == 16);
  const bool in_is_fnuz = in_is_e4m3fnuz || in_is_e5m2fnuz;
  const bool in_no_inf = in_is_e4m3fn || in_is_fnuz;

  const bool out_is_e4m3fn = (out.width == 8 && out.mantissa == 3 && out.bias == 7);
  const bool out_is_e4m3fnuz = (out.width == 8 && out.mantissa == 3 && out.bias == 8);
  const bool out_is_e5m2 = (out.width == 8 && out.mantissa == 2 && out.bias == 15);
  const bool out_is_e5m2fnuz = (out.width == 8 && out.mantissa == 2 && out.bias == 16);
  const bool out_is_fnuz = out_is_e4m3fnuz || out_is_e5m2fnuz;

  bool in_is_nan = false;
  bool in_is_inf = false;
  if (in_is_e4m3fn) {
    in_is_nan = (in_exp == in_exp_mask && in_mant == in_mant_mask);
  } else if (in_is_fnuz) {
    in_is_nan = (in_val == 0x80);
  } else if (in_exp == in_exp_mask) {
    in_is_inf = (in_mant == 0);
    in_is_nan = (in_mant != 0);
  }

  if (in_is_nan || in_is_inf) {
    if (out_is_e4m3fn) {
      if (in_is_nan) return 0x7f;
      return (sign << out_sign_pos) | 0x7f;
    }
    if (out_is_fnuz) {
      return 0x80;
    }
    if (out_is_e5m2) {
      if (in_is_nan) return 0x7f;
      return (sign << out_sign_pos) | 0x7c;
    }
    uint64_t out_exp_bits = out.width - out.mantissa - 1;
    uint64_t out_exp_all_ones = (1ULL << out_exp_bits) - 1;
    uint64_t out_mant = in_is_nan ? out_mant_mask : 0;
    uint64_t out_sign = in_is_nan ? 0 : (sign << out_sign_pos);
    return out_sign | (out_exp_all_ones << out.mantissa) | out_mant;
  }

  if (in_exp == 0 && in_mant == 0) {
    return out_is_fnuz ? 0 : (sign << out_sign_pos);
  }

  int true_exp;
  uint64_t normalized_mant;

  if (in_exp != 0 && (in_no_inf || in_exp != in_exp_mask)) {
    true_exp = (int)in_exp - in.bias;
    normalized_mant = in_mant;
  } else if (in_exp == 0) {
    int lz = __builtin_clzll(in_mant);
    int shift = lz - (64 - in.mantissa);
    true_exp = 1 - in.bias - (shift + 1);
    normalized_mant = (in_mant << (shift + 1)) & in_mant_mask;
  } else {
    uint64_t out_mant = (in_mant == 0) ? 0 : (1ULL << (out.mantissa - 1));
    return (sign << out_sign_pos) | (out_exp_max << out.mantissa) | out_mant;
  }

  int out_exp_val = true_exp + out.bias;
  uint64_t final_exp, final_mant;

  if (out_exp_val > 0 && out_exp_val < out_exp_max) {
      final_exp = (uint64_t)out_exp_val;
      if (shift_diff > 0) {
          uint64_t round_bit = 1ULL << (shift_diff - 1);
          uint64_t sticky_mask = round_bit - 1;
          bool lsb = (normalized_mant >> shift_diff) & 1;
          bool round = (normalized_mant & round_bit) != 0;
          bool sticky = (normalized_mant & sticky_mask) != 0;
          
          final_mant = normalized_mant >> shift_diff;
          if (round && (lsb || sticky)) {
              final_mant++;
              if (final_mant > out_mant_mask) {
                  final_mant = 0;
                  final_exp++;
              }
          }
      } else {
          final_mant = normalized_mant << (-shift_diff);
      }
  } else if (out_exp_val >= (int64_t)out_exp_max) {
      if (out_is_e4m3fn) {
          final_exp = out_exp_max;
          final_mant = out_mant_mask;
      } else if (out_is_fnuz) {
          return 0x80;
      } else {
          final_exp = out_exp_max;
          final_mant = 0;
      }
  } else {
      int sub_shift = 1 - out_exp_val;
      int total_shift = shift_diff + sub_shift;
      
      final_exp = 0;
      uint64_t full_mant = normalized_mant | (1ULL << in.mantissa);
      
      if (total_shift >= 64) {
          final_mant = 0;
      } else if (total_shift > 0) {
          uint64_t round_bit = 1ULL << (total_shift - 1);
          uint64_t sticky_mask = round_bit - 1;
          bool lsb = (full_mant >> total_shift) & 1;
          bool round = (full_mant & round_bit) != 0;
          bool sticky = (full_mant & sticky_mask) != 0;
          
          final_mant = full_mant >> total_shift;
          if (round && (lsb || sticky)) {
              final_mant++;
          }
      } else {
          final_mant = full_mant << (-total_shift);
      }
  }

  uint64_t result = (sign << out_sign_pos) | (final_exp << out.mantissa) | (final_mant & out_mant_mask);
  if (out_is_fnuz && final_exp == 0 && (final_mant & out_mant_mask) == 0) return 0;
  return result;
}

enum class FPType {
    FP32,
    FP16,
    BF16,
    FP64,
    FP8E5,
    FP8E4NV,
    FP8E4B8,
    FP8E4B15,
    FP8E5B16,
    UNKNOWN
};

constexpr FPType get_fp_type(int w, int m, int b) {
    if (w == 32 && m == 23 && b == 127) return FPType::FP32;
    if (w == 16 && m == 10 && b == 15) return FPType::FP16;
    if (w == 16 && m == 7 && b == 127) return FPType::BF16;
    if (w == 64 && m == 52 && b == 1023) return FPType::FP64;
    if (w == 8 && m == 2 && b == 15) return FPType::FP8E5;
    if (w == 8 && m == 3 && b == 7) return FPType::FP8E4NV;
    if (w == 8 && m == 3 && b == 8) return FPType::FP8E4B8;
    if (w == 8 && m == 3 && b == 15) return FPType::FP8E4B15;
    if (w == 8 && m == 2 && b == 16) return FPType::FP8E5B16;
    return FPType::UNKNOWN;
}

#ifdef _MSC_VER
#define PARALLEL_FOR __pragma(omp parallel for)
#else
#define PARALLEL_FOR _Pragma("omp parallel for")
#endif

// Helper macro to generate specialization call
#define GENERATE_CASES(IN_TYPE, IN_W, IN_M, IN_B) \
    case IN_TYPE: \
        switch (out_type) { \
            case FPType::FP32: \
                PARALLEL_FOR \
                for (py::ssize_t i = 0; i < buf_in.size; ++i) dst[i] = convert_element<IN_W, IN_M, IN_B, 32, 23, 127>(src[i]); \
                return result; \
            case FPType::FP16: \
                PARALLEL_FOR \
                for (py::ssize_t i = 0; i < buf_in.size; ++i) dst[i] = convert_element<IN_W, IN_M, IN_B, 16, 10, 15>(src[i]); \
                return result; \
            case FPType::BF16: \
                PARALLEL_FOR \
                for (py::ssize_t i = 0; i < buf_in.size; ++i) dst[i] = convert_element<IN_W, IN_M, IN_B, 16, 7, 127>(src[i]); \
                return result; \
            case FPType::FP64: \
                PARALLEL_FOR \
                for (py::ssize_t i = 0; i < buf_in.size; ++i) dst[i] = convert_element<IN_W, IN_M, IN_B, 64, 52, 1023>(src[i]); \
                return result; \
            case FPType::FP8E5: \
                PARALLEL_FOR \
                for (py::ssize_t i = 0; i < buf_in.size; ++i) dst[i] = convert_element<IN_W, IN_M, IN_B, 8, 2, 15>(src[i]); \
                return result; \
            case FPType::FP8E4NV: \
                PARALLEL_FOR \
                for (py::ssize_t i = 0; i < buf_in.size; ++i) dst[i] = convert_element<IN_W, IN_M, IN_B, 8, 3, 7>(src[i]); \
                return result; \
            case FPType::FP8E4B8: \
                PARALLEL_FOR \
                for (py::ssize_t i = 0; i < buf_in.size; ++i) dst[i] = convert_element<IN_W, IN_M, IN_B, 8, 3, 8>(src[i]); \
                return result; \
            case FPType::FP8E4B15: \
                PARALLEL_FOR \
                for (py::ssize_t i = 0; i < buf_in.size; ++i) dst[i] = convert_element<IN_W, IN_M, IN_B, 8, 3, 15>(src[i]); \
                return result; \
            case FPType::FP8E5B16: \
                PARALLEL_FOR \
                for (py::ssize_t i = 0; i < buf_in.size; ++i) dst[i] = convert_element<IN_W, IN_M, IN_B, 8, 2, 16>(src[i]); \
                return result; \
            default: break; \
        } \
        break;

py::array_t<uint64_t> convert_float(py::array_t<uint64_t> input, int in_w, int in_m, int in_b,
    int out_w, int out_m, int out_b) {

  auto buf_in = input.request();
  auto result = py::array_t<uint64_t>(buf_in.shape);
  auto buf_out = result.request();
  
  const uint64_t* src = static_cast<const uint64_t*>(buf_in.ptr);
  uint64_t* dst = static_cast<uint64_t*>(buf_out.ptr);
  
  // FP32 -> FP16 (Specialized Intrinsic)
  if (in_w == 32 && in_m == 23 && in_b == 127 && out_w == 16 && out_m == 10 && out_b == 15) {
    #pragma omp parallel for
    for (py::ssize_t i = 0; i < buf_in.size; ++i) {
      dst[i] = FromFloatBits<false, false>(static_cast<uint32_t>(src[i]));
    }
    return result;
  }

  // FP16 -> FP32 (Specialized Intrinsic)
  if (in_w == 16 && in_m == 10 && in_b == 15 && out_w == 32 && out_m == 23 && out_b == 127) {
    #pragma omp parallel for
    for (py::ssize_t i = 0; i < buf_in.size; ++i) {
      dst[i] = ToFloatBits(static_cast<uint16_t>(src[i]));
    }
    return result;
  }
  
  FPType in_type = get_fp_type(in_w, in_m, in_b);
  FPType out_type = get_fp_type(out_w, out_m, out_b);

  switch (in_type) {
      GENERATE_CASES(FPType::FP32, 32, 23, 127)
      GENERATE_CASES(FPType::FP16, 16, 10, 15)
      GENERATE_CASES(FPType::BF16, 16, 7, 127)
      GENERATE_CASES(FPType::FP64, 64, 52, 1023)
      GENERATE_CASES(FPType::FP8E5, 8, 2, 15)
      GENERATE_CASES(FPType::FP8E4NV, 8, 3, 7)
      GENERATE_CASES(FPType::FP8E4B8, 8, 3, 8)
      GENERATE_CASES(FPType::FP8E4B15, 8, 3, 15)
      GENERATE_CASES(FPType::FP8E5B16, 8, 2, 16)
      default: break;
  }
  
  FPDesc in_d = {in_w, in_m, in_b};
  FPDesc out_d = {out_w, out_m, out_b};
  #pragma omp parallel for
  for (py::ssize_t i = 0; i < buf_in.size; ++i) {
    dst[i] = convert_element_runtime(src[i], in_d, out_d);
  }
  return result;
}

enum class MemSemantic { ACQUIRE_RELEASE, ACQUIRE, RELEASE, RELAXED };

std::map<MemSemantic, std::memory_order> MemSemantic_MAP = {
    {MemSemantic::ACQUIRE_RELEASE, std::memory_order_acq_rel},
    {MemSemantic::ACQUIRE, std::memory_order_acquire},
    {MemSemantic::RELEASE, std::memory_order_release},
    {MemSemantic::RELAXED, std::memory_order_relaxed},
};

enum class RMWOp { ADD, FADD, AND, OR, XOR, XCHG, MAX, MIN, UMIN, UMAX };

std::mutex atomic_op_guard;

template <typename T>
constexpr bool is_reinterpret_cast_to_atomic_safe =
    std::is_trivially_copyable_v<T> &&
    std::is_trivially_copyable_v<std::atomic<T>> &&
    std::is_standard_layout_v<T> && std::is_standard_layout_v<std::atomic<T>> &&
    sizeof(T) == sizeof(std::atomic<T>) &&
    alignof(T) == alignof(std::atomic<T>);

template <bool is_min, typename T>
T atomic_cmp(T *ptr, T val, std::memory_order order) {
  auto cmp = [](T old, T val) {
    if constexpr (is_min) {
      return old > val;
    } else {
      return old < val;
    }
  };
  T old_val;
  if constexpr (is_reinterpret_cast_to_atomic_safe<T>) {
    std::atomic<T> *atomic_ptr = reinterpret_cast<std::atomic<T> *>(ptr);
    old_val = atomic_ptr->load(order);
    while (cmp(old_val, val)) {
      if (atomic_ptr->compare_exchange_weak(old_val, val, order, order)) {
        break;
      }
    }
  } else {
    const std::lock_guard<std::mutex> lock(atomic_op_guard);
    old_val = *ptr;
    if (cmp(old_val, val)) {
      *ptr = val;
    }
  }
  return old_val;
}

template <typename T> T atomic_fadd(T *loc, T value, std::memory_order order) {
  static_assert(std::is_floating_point<T>::value,
                "T must be a floating-point type");
  T old_value;
  if constexpr (is_reinterpret_cast_to_atomic_safe<T>) {
    T new_value;
    std::atomic<T> *atomic_loc = reinterpret_cast<std::atomic<T> *>(loc);
    old_value = atomic_loc->load(order);
    do {
      new_value = old_value + value;
    } while (
        !atomic_loc->compare_exchange_weak(old_value, new_value, order, order));
  } else {
    const std::lock_guard<std::mutex> lock(atomic_op_guard);
    old_value = *loc;
    *loc = old_value + value;
  }
  return old_value;
}

template <>
npy_half atomic_fadd<npy_half>(npy_half *loc, npy_half value,
                               std::memory_order order) {
  static_assert(sizeof(npy_half) == sizeof(uint16_t),
                "npy_half must be 16 bits");
  static_assert(alignof(npy_half) == alignof(uint16_t),
                "npy_half must be 16-bit aligned");

  if constexpr (is_reinterpret_cast_to_atomic_safe<uint16_t>) {
    std::atomic<uint16_t> *atomic_loc =
        reinterpret_cast<std::atomic<uint16_t> *>(loc);
    uint16_t old_bits = atomic_loc->load(order);
    uint16_t new_bits;

    do {
      npy_half current_half = {old_bits};
      float current_val = npy_half_to_float(current_half);
      float result_val = current_val + npy_half_to_float(value);
      new_bits = npy_float_to_half(result_val).value;

    } while (
        !atomic_loc->compare_exchange_weak(old_bits, new_bits, order, order));

    return {old_bits};
  } else {
    npy_half old_value;
    const std::lock_guard<std::mutex> lock(atomic_op_guard);
    old_value = *loc;
    *loc = npy_float_to_half(npy_half_to_float(old_value) +
                             npy_half_to_float(value));
    return old_value;
  }
}

class AtomicOp {
public:
  AtomicOp(const uint64_t *ptr, size_t numel, std::memory_order order)
      : ptr(ptr), numel(numel), order(order) {}

  void apply() {
    for (size_t i = 0; i < numel; ++i) {
      applyAt(reinterpret_cast<void *>(ptr[i]), i);
    }
  }

  virtual ~AtomicOp() = default;

protected:
  virtual void applyAt(void *, size_t i) = 0;

  const uint64_t *ptr;
  size_t numel;
  std::memory_order order;
};

template <typename DType> class AtomicRMWOpBase : public AtomicOp {
public:
  AtomicRMWOpBase(const uint64_t *ptr, const void *val, void *ret,
                  const bool *mask, size_t numel, std::memory_order order)
      : AtomicOp(ptr, numel, order), val(val), ret(ret), mask(mask) {}

protected:
  void applyAt(void *loc, size_t i) override final {
    if (mask[i]) {
      DType *ptr = static_cast<DType *>(loc);
      *(static_cast<DType *>(ret) + i) =
          applyAtMasked(ptr, *(static_cast<const DType *>(val) + i), order);
    }
  }

  virtual DType applyAtMasked(DType *loc, const DType value,
                              std::memory_order order) = 0;

  const void *val;
  void *ret;
  const bool *mask;
};

template <typename DType, RMWOp Op, typename = void>
class AtomicRMWOp : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::ADD>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    DType old_val;
    if constexpr (is_reinterpret_cast_to_atomic_safe<DType>) {
      std::atomic<DType> *atomic_loc =
          reinterpret_cast<std::atomic<DType> *>(loc);
      old_val = std::atomic_fetch_add_explicit(atomic_loc, value, order);
    } else {
      const std::lock_guard<std::mutex> lock(atomic_op_guard);
      old_val = *loc;
      *loc = *loc + value;
    }
    return old_val;
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::FADD>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    return atomic_fadd(loc, value, order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::AND>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    DType old_val;
    if constexpr (is_reinterpret_cast_to_atomic_safe<DType>) {
      std::atomic<DType> *atomic_loc =
          reinterpret_cast<std::atomic<DType> *>(loc);
      old_val = std::atomic_fetch_and_explicit(atomic_loc, value, order);
    } else {
      const std::lock_guard<std::mutex> lock(atomic_op_guard);
      old_val = *loc;
      *loc = *loc & value;
    }
    return old_val;
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::OR>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    DType old_val;
    if constexpr (is_reinterpret_cast_to_atomic_safe<DType>) {
      std::atomic<DType> *atomic_loc =
          reinterpret_cast<std::atomic<DType> *>(loc);
      old_val = std::atomic_fetch_or_explicit(atomic_loc, value, order);
    } else {
      const std::lock_guard<std::mutex> lock(atomic_op_guard);
      old_val = *loc;
      *loc = *loc | value;
    }
    return old_val;
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::XOR>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    DType old_val;
    if constexpr (is_reinterpret_cast_to_atomic_safe<DType>) {
      std::atomic<DType> *atomic_loc =
          reinterpret_cast<std::atomic<DType> *>(loc);
      old_val = std::atomic_fetch_xor_explicit(atomic_loc, value, order);
    } else {
      const std::lock_guard<std::mutex> lock(atomic_op_guard);
      old_val = *loc;
      *loc = *loc ^ value;
    }
    return old_val;
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op,
                  std::enable_if_t<Op == RMWOp::MAX || Op == RMWOp::UMAX>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    return atomic_cmp</*is_min=*/false>(loc, value, order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op,
                  std::enable_if_t<Op == RMWOp::MIN || Op == RMWOp::UMIN>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    return atomic_cmp</*is_min=*/true>(loc, value, order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::XCHG>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    DType old_val;
    if constexpr (is_reinterpret_cast_to_atomic_safe<DType>) {
      std::atomic<DType> *atomic_loc =
          reinterpret_cast<std::atomic<DType> *>(loc);
      old_val = atomic_loc->exchange(value, order);
    } else {
      const std::lock_guard<std::mutex> lock(atomic_op_guard);
      old_val = *loc;
      *loc = value;
    }
    return old_val;
  }
};

template <typename T>
void atomic_compare_exchange_strong(void *loc, void *expected,
                                    const void *desired, size_t i,
                                    std::memory_order order) {
  T desired_val = *(static_cast<const T *>(desired) + i);
  T *expected_uint = static_cast<T *>(expected) + i;

  if constexpr (is_reinterpret_cast_to_atomic_safe<T>) {
    std::atomic<T> *atomic_loc = reinterpret_cast<std::atomic<T> *>(loc);
    atomic_loc->compare_exchange_strong(*expected_uint, desired_val, order,
                                        order);
  } else {
    const std::lock_guard<std::mutex> lock(atomic_op_guard);
    T *atomic_loc = static_cast<T *>(loc);
    if (*atomic_loc == *expected_uint) {
      *atomic_loc = desired_val;
    } else {
      *expected_uint = *atomic_loc;
    }
  }
}

class AtomicCASOp : public AtomicOp {
public:
  AtomicCASOp(const uint64_t *ptr, void *expected, const void *desired, const bool *mask,
              size_t itemsize, size_t numel, std::memory_order order)
      : AtomicOp(ptr, numel, order), expected(expected), desired(desired), mask(mask),
        itemsize(itemsize) {}

protected:
  void applyAt(void *loc, size_t i) override {
    if (mask && !mask[i]) return;
    // Atomic operations perform bitwise comparison, so it's safe to
    // use number of bytes (itemsize) to determine the type of pointers
    if (itemsize == 1) {
      atomic_compare_exchange_strong<uint8_t>(loc, expected, desired, i, order);
    } else if (itemsize == 2) {
      atomic_compare_exchange_strong<uint16_t>(loc, expected, desired, i,
                                               order);
    } else if (itemsize == 4) {
      atomic_compare_exchange_strong<uint32_t>(loc, expected, desired, i,
                                               order);
    } else if (itemsize == 8) {
      atomic_compare_exchange_strong<uint64_t>(loc, expected, desired, i,
                                               order);
    } else {
      throw std::invalid_argument("Invalid byte size");
    }
  }

private:
  void *expected;
  const void *desired;
  const bool *mask;
  size_t itemsize;
};

template <RMWOp Op> struct OpCreator {
  pybind11::dtype dtype;
  const uint64_t *ptr;
  const void *val;
  void *ret;
  const bool *mask;
  size_t numel;
  std::memory_order order;
  std::unique_ptr<AtomicOp> &atomic_op;

  template <typename T> void create() {
    if (!atomic_op && dtype.is(py::dtype::of<T>())) {
      atomic_op = std::make_unique<AtomicRMWOp<T, Op>>(ptr, val, ret, mask,
                                                       numel, order);
    }
  }
};

template <> template <> void OpCreator<RMWOp::FADD>::create<npy_half>() {
  if (!atomic_op && dtype.char_() == 'e') { // float16
    atomic_op = std::make_unique<AtomicRMWOp<npy_half, RMWOp::FADD>>(
        ptr, val, ret, mask, numel, order);
  }
};

template <RMWOp Op, typename... SupportedDTypes>
std::unique_ptr<AtomicOp>
makeAtomicRMWOp(pybind11::dtype dtype, const uint64_t *ptr, const void *val,
                void *ret, const bool *mask, size_t numel,
                std::memory_order order) {
  // Iterate over all supported data types, make one that matches, and return
  std::unique_ptr<AtomicOp> atomic_op;
  OpCreator<Op> try_make_op{dtype, ptr,   val,   ret,
                            mask,  numel, order, atomic_op};
  (try_make_op.template create<SupportedDTypes>(), ...);
  if (!atomic_op) {
    throw std::invalid_argument("Unsupported data type for RMWOp (dtype: " +
                                dtype.str().cast<std::string>() + ")");
  }
  return atomic_op;
}

py::array atomic_rmw(RMWOp rmw_op, py::array_t<uint64_t> ptr, py::array val,
                     py::array_t<bool> mask, MemSemantic sem) {
  std::memory_order order = MemSemantic_MAP[sem];
  int numel = ptr.size();
  auto shape = std::vector<ptrdiff_t>(ptr.shape(), ptr.shape() + ptr.ndim());
  auto ret_dtype = val.dtype();
  py::array ret(ret_dtype, py::array::ShapeContainer{numel});
  py::array_t<uint64_t> reshaped_ptr = ptr.reshape({numel});
  py::array_t<bool> reshaped_mask = mask.reshape({numel});
  py::array reshaped_val = val.reshape({numel});
  auto *ptr_data = reshaped_ptr.data();
  auto *mask_data = reshaped_mask.data();
  auto *val_data = static_cast<const void *>(reshaped_val.data());
  auto *ret_data = static_cast<void *>(ret.mutable_data());

  std::unique_ptr<AtomicOp> atomic_op;

#define MAKE_ATOMIC_RMW_OP(OP_NAME, ...)                                       \
  case OP_NAME:                                                                \
    atomic_op = makeAtomicRMWOp<OP_NAME, __VA_ARGS__>(                         \
        ret_dtype, ptr_data, val_data, ret_data, mask_data, numel, order);     \
    break;

  switch (rmw_op) {
    MAKE_ATOMIC_RMW_OP(RMWOp::ADD, int32_t, uint32_t, int64_t, uint64_t)
    MAKE_ATOMIC_RMW_OP(RMWOp::FADD, npy_half, float, double)
    MAKE_ATOMIC_RMW_OP(RMWOp::AND, int32_t, uint32_t, int64_t, uint64_t)
    MAKE_ATOMIC_RMW_OP(RMWOp::OR, int32_t, uint32_t, int64_t, uint64_t)
    MAKE_ATOMIC_RMW_OP(RMWOp::XOR, int32_t, uint32_t, int64_t, uint64_t)
    MAKE_ATOMIC_RMW_OP(RMWOp::MAX, int32_t, int64_t)
    MAKE_ATOMIC_RMW_OP(RMWOp::UMAX, uint32_t, uint64_t)
    MAKE_ATOMIC_RMW_OP(RMWOp::MIN, int32_t, int64_t)
    MAKE_ATOMIC_RMW_OP(RMWOp::UMIN, uint32_t, uint64_t)
    MAKE_ATOMIC_RMW_OP(RMWOp::XCHG, int32_t, uint32_t, int64_t, uint64_t)

  default:
    throw std::invalid_argument("Unsupported RMW operation");
  }

#undef MAKE_ATOMIC_RMW_OP
  atomic_op->apply();
  return ret.reshape(shape);
}

py::array atomic_cas(py::array_t<uint64_t> ptr, py::array &cmp, py::array &val, py::array_t<bool> mask, MemSemantic sem) {
  std::memory_order order = MemSemantic_MAP[sem];
  int numel = ptr.size();
  auto shape = std::vector<ptrdiff_t>(ptr.shape(), ptr.shape() + ptr.ndim());
  auto ret_dtype = cmp.dtype();
  py::array ret(ret_dtype, py::array::ShapeContainer{numel});
  py::array_t<uint64_t> reshaped_ptr = ptr.reshape({numel});
  py::array reshaped_cmp = cmp.reshape({numel});
  py::array reshaped_val = val.reshape({numel});
  py::array_t<bool> reshaped_mask = mask.reshape({numel});
  auto itemsize = cmp.itemsize();
  memcpy(static_cast<void *>(ret.mutable_data()), static_cast<const void *>(reshaped_cmp.data()), itemsize * numel);
  AtomicCASOp(reshaped_ptr.data(), ret.mutable_data(), static_cast<const void *>(reshaped_val.data()), reshaped_mask.data(),
      itemsize, numel, order).apply();
  return ret.reshape(shape);
}

py::array load(py::array_t<uint64_t> ptr, py::array_t<bool> mask,
               std::optional<py::array> other, py::object dtype) {
  int numel = ptr.size();
  auto shape = std::vector<ptrdiff_t>(ptr.shape(), ptr.shape() + ptr.ndim());
  py::dtype x_dtype = py::dtype::from_args(dtype);
  py::array x(x_dtype, py::array::ShapeContainer{numel});

  py::array_t<uint64_t> reshaped_ptr = ptr.reshape({numel});
  py::array_t<bool> reshaped_mask = mask.reshape({numel});

  auto x_ptr = static_cast<char *>(x.mutable_data());
  const uint64_t *ptr_data = reshaped_ptr.data();
  const bool *mask_data = reshaped_mask.data();
  const char *other_data = nullptr;
  size_t itemsize = x_dtype.itemsize();

  py::array reshaped_others;
  if (other.has_value()) {
    reshaped_others = other->reshape({numel});
    other_data = static_cast<const char *>(reshaped_others.data());
  } else {
    memset(x_ptr, 0, numel * itemsize);
  }

  // Optimized path for contiguous load
  if (numel > 0) {
      bool contiguous = true;
      uint64_t start_addr = ptr_data[0];
      for (py::ssize_t i = 1; i < numel; ++i) {
          if (ptr_data[i] != start_addr + i * itemsize) {
              contiguous = false;
              break;
          }
      }

      if (contiguous) {
          bool all_masked = true;
          for (py::ssize_t i = 0; i < numel; ++i) {
              if (!mask_data[i]) {
                  all_masked = false;
                  break;
              }
          }
          if (all_masked) {
              memcpy(x_ptr, reinterpret_cast<const void*>(start_addr), numel * itemsize);
              return x.reshape(shape);
          }
      }
  }

  if (numel >= 4096) {
    py::gil_scoped_release release;
    PARALLEL_FOR
    for (py::ssize_t i = 0; i < numel; ++i) {
      if (mask_data[i]) {
        memcpy(x_ptr + i * itemsize, reinterpret_cast<void *>(ptr_data[i]),
               itemsize);
      } else if (other_data) {
        memcpy(x_ptr + i * itemsize, other_data + i * itemsize, itemsize);
      }
    }
  } else {
    for (py::ssize_t i = 0; i < numel; ++i) {
      if (mask_data[i]) {
        memcpy(x_ptr + i * itemsize, reinterpret_cast<void *>(ptr_data[i]),
               itemsize);
      } else if (other_data) {
        memcpy(x_ptr + i * itemsize, other_data + i * itemsize, itemsize);
      }
    }
  }
  return x.reshape(shape);
}

void store(py::array_t<uint64_t> ptr, py::array value, py::array_t<bool> mask) {
  int numel = ptr.size();
  py::array_t<uint64_t> reshaped_ptr = ptr.reshape({numel});
  py::array_t<bool> reshaped_mask = mask.reshape({numel});
  py::array reshaped_value = value.reshape({numel});

  const uint64_t *ptr_data = reshaped_ptr.data();
  const bool *mask_data = reshaped_mask.data();
  const char *value_data = static_cast<const char *>(reshaped_value.data());
  size_t itemsize = value.dtype().itemsize();

  // Optimized path for contiguous store
  if (numel > 0) {
      bool contiguous = true;
      uint64_t start_addr = ptr_data[0];
      for (py::ssize_t i = 1; i < numel; ++i) {
          if (ptr_data[i] != start_addr + i * itemsize) {
              contiguous = false;
              break;
          }
      }

      if (contiguous) {
          bool all_masked = true;
          for (py::ssize_t i = 0; i < numel; ++i) {
              if (!mask_data[i]) {
                  all_masked = false;
                  break;
              }
          }
          if (all_masked) {
              memcpy(reinterpret_cast<void*>(start_addr), value_data, numel * itemsize);
              return;
          }
      }
  }

  if (numel >= 4096) {
    py::gil_scoped_release release;
    PARALLEL_FOR
    for (py::ssize_t i = 0; i < numel; ++i) {
      if (mask_data[i]) {
        memcpy(reinterpret_cast<void *>(ptr_data[i]), value_data + i * itemsize,
               itemsize);
      }
    }
  } else {
    for (py::ssize_t i = 0; i < numel; ++i) {
      if (mask_data[i]) {
        memcpy(reinterpret_cast<void *>(ptr_data[i]), value_data + i * itemsize,
               itemsize);
      }
    }
  }
}

void parallel_launch(py::function fn, std::vector<int> grid_dim, py::object builder) {
  int nx = grid_dim[0];
  int ny = grid_dim[1];
  int nz = grid_dim[2];

  // {
  //   py::gil_scoped_release release;

  //   PARALLEL_FOR
    for (py::ssize_t idx = 0; idx < (py::ssize_t)nx * ny * nz; ++idx) {
      int z = idx % nz;
      int y = (idx / nz) % ny;
      int x = idx / (nz * ny);

      // py::gil_scoped_acquire acquire;
      builder.attr("set_grid_idx")(x, y, z);
      fn();
    }
  // }
}

PYBIND11_NAMESPACE_END(Interpreter)

using namespace Interpreter;

void init_interpreter(py::module_ &&m) {
  py::enum_<MemSemantic>(m, "MEM_SEMANTIC", py::module_local(),
                         "Memory semantic for atomic operations.")
      .value("ACQUIRE_RELEASE", MemSemantic::ACQUIRE_RELEASE,
             "Acquire and release memory")
      .value("ACQUIRE", MemSemantic::ACQUIRE, "Acquire memory")
      .value("RELEASE", MemSemantic::RELEASE, "Release memory")
      .value("RELAXED", MemSemantic::RELAXED, "Relaxed memory semantic");

  py::enum_<RMWOp>(m, "RMW_OP", py::module_local(),
                   "RMW operation enumeration class: specifies the type of "
                   "atomic RMW operation")
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

  m.def("load", &load, "ptr"_a, "mask"_a, "other"_a, "dtype"_a,
        R"pbdoc(
    Load data from memory addresses or fallback array based on mask.

    Args:
        ptr (numpy.ndarray): Memory addresses to load from.
        mask (numpy.ndarray): Mask to apply.
        other (numpy.ndarray): Fallback array to load from.
        dtype (numpy.dtype): Data type of the output array.
    )pbdoc");

  m.def("store", &store, "ptr"_a, "value"_a, "mask"_a,
        R"pbdoc(
    Store data to memory addresses based on mask.

    Args:
        ptr (numpy.ndarray): Memory addresses to store to.
        value (numpy.ndarray): Values to store.
        mask (numpy.ndarray): Mask to apply.
    )pbdoc");

  m.def("atomic_rmw", &atomic_rmw, "rmw_op"_a, "ptr"_a, "val"_a, "mask"_a,
        "order"_a,
        R"pbdoc(
    Perform RMW operation on memory addresses based on mask.

    Args:
        rmw_op (RMWOp): RMW operation to perform.
        ptr (numpy.ndarray): Memory addresses to perform RMW operation.
        val (numpy.ndarray): Values to perform RMW operation.
        mask (numpy.ndarray): Mask to apply.
        order (MemSemantic): Memory order for atomic operation.
    )pbdoc");

  m.def("atomic_cas", &atomic_cas, "ptr"_a, "cmp"_a, "val"_a, "mask"_a, "order"_a,
        R"pbdoc(
    Perform compare-and-swap operation on memory addresses based on mask.

    Args:
        ptr (numpy.ndarray): Memory addresses to perform compare-and-swap operation.
        cmp (numpy.ndarray): Values to compare.
        val (numpy.ndarray): Values to swap.
        mask (numpy.ndarray): Mask to apply.
        order (MemSemantic): Memory order for atomic operation.
    )pbdoc");

  m.def("convert_float", &convert_float, "input"_a, "in_w"_a, "in_m"_a,
        "in_b"_a, "out_w"_a, "out_m"_a, "out_b"_a,
        R"pbdoc(
    Convert floating-point numbers between different representations.

    Args:
        input (numpy.ndarray): Input array of floating-point numbers.
        in_w (int): Width of the input floating-point number representation.
        in_m (int): Mantissa bits of the input floating-point number representation.
        in_b (int): Bias of the input floating-point number representation.
        out_w (int): Width of the output floating-point number representation.
        out_m (int): Mantissa bits of the output floating-point number representation.
        out_b (int): Bias of the output floating-point number representation.
    )pbdoc");

  m.def("parallel_launch", &parallel_launch, "fn"_a, "grid_dim"_a, "builder"_a,
        R"pbdoc(
    Launch kernel in parallel.

    Args:
        fn (function): The kernel function to launch.
        grid_dim (list): The grid dimension.
        builder (InterpreterBuilder): The interpreter builder.
    )pbdoc");
}
