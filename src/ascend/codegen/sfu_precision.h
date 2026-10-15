#ifndef TVM_TL_ASCEND_CODEGEN_SFU_PRECISION_H_
#define TVM_TL_ASCEND_CODEGEN_SFU_PRECISION_H_

/*!
 * \file ascend/codegen/sfu_precision.h
 * \brief Shared per-op SFU precision resolution for the Ascend backends.
 *
 * Both the AscendC and PTO codegens resolve the per-op "precision"
 * annotation with the same priority so a single TileLang program keeps one
 * precision contract across backends:
 *
 *   1. per-op "precision" annotation (Python `precision=` kwarg)
 *   2. `fallback` (legacy: fast_math for vdiv; bare SFU otherwise)
 *
 * The Python side (tilelang/ascend/language/simd.py) is the single source of
 * truth for alias resolution: string aliases are normalized to integer codes
 * (0=hw, 1=exact, 2=ftz_false) before they reach codegen, so no alias table
 * lives here (mirrors the l2_cache_ctrl pattern). LegalizeSimdMerging
 * forwards call annotations, so both the zeroing and the legalized
 * read-modify-write forms keep the annotation.
 */

#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>

namespace tvm {
namespace codegen {

enum class SfuPrecision { kHw, kExact, kKeepSub };

inline SfuPrecision PrecisionFromCode(int code) {
  if (code == 1)
    return SfuPrecision::kExact;
  if (code == 2)
    return SfuPrecision::kKeepSub;
  ICHECK_EQ(code, 0) << "SFU precision annotation must be 0 (hw), 1 (exact), "
                        "or 2 (ftz_false), got "
                     << code;
  return SfuPrecision::kHw;
}

// MODE_MERGING calls are legalized to void calls, so use the explicit result
// dtype supplied by the caller rather than reading op->dtype here.
inline SfuPrecision ResolveSfuPrecision(const tirx::Call &op,
                                        DataType result_dtype,
                                        SfuPrecision fallback) {
  // Precise paths (vdiv_0ulp_ftz_true, *_ftz_false wrappers) are
  // float32-only: non-fp32 keeps the hardware instruction regardless of
  // annotations (historical UsePreciseVdiv contract: "non-fp32 always uses
  // hardware").
  if (!result_dtype.is_float() || result_dtype.bits() != 32) {
    return fallback;
  }
  // Per-op "precision" annotation (int code from the Python side, normalized
  // from the string aliases in tilelang/ascend/language/simd.py -- no alias
  // table here, mirroring the l2_cache_ctrl pattern).
  if (auto p = op->annotations.Get("precision")) {
    const auto *code = p.value().as<tirx::IntImmNode>();
    ICHECK(code != nullptr)
        << "SFU precision annotation must be an integer code, got "
        << p.value();
    return PrecisionFromCode(static_cast<int>(code->value));
  }
  return fallback;
}

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_ASCEND_CODEGEN_SFU_PRECISION_H_
