/*!
 * \file tl/target/sunmmio_utils.h
 * \brief Centralized Sunmmio device-model helpers used by passes and analysis.
 */

#ifndef TVM_TL_TARGET_SUNMMIO_UTILS_H_
#define TVM_TL_TARGET_SUNMMIO_UTILS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>

namespace tvm {
namespace tl {

enum class CommunicationDirections;

// ---------------------------------------------------------------------------
// Sunmmio on-chip SRAM scope identifiers
// ---------------------------------------------------------------------------
constexpr const char *kSunmmioScopeASRAM = "shared.asram";
constexpr const char *kSunmmioScopeWSRAM = "shared.wsram";
constexpr const char *kSunmmioScopeRSRAM = "shared.rsram";

// ---------------------------------------------------------------------------
// Op annotation keys used by the Sunmmio bf16 GEMM legalization pass.
// ---------------------------------------------------------------------------
// Annotation key on CopyNode / AllgatherOpNode whose value is an IntImm
// giving the byte offset added to the source pointer at codegen. Set by the
// LegalizeSunmmioGemm pass to re-stage south-bound A data into the
// destination's north bank for the second bf16 TC half-pass. The value
// propagates to the leaf intrinsic (tl.dma_copy / tl.broadcast_) as a
// trailing positional arg.
constexpr const char *kAttrSrcOffsetByte = "src_offset_byte";

// Reflection field name for the Gemm/GemmPy node member accOffsetByte_. It
// is the Python-visible name registered via def_ro and consumed via
// getattr(node, kFieldAccOffsetByte) in GemmBase. The Sunmmio bf16 GEMM
// legalization pass sets this field on the cloned second-pass Gemm to
// select the second stripe-parity's starting row in the RSRAM accumulator.
constexpr const char *kFieldAccOffsetByte = "accOffsetByte";

struct SunmmioTileProcessorConfig {
  int register_bits;
  int block_height;
  int block_width;
  // Minimum byte-alignment for RSRAM vector memory accesses.
  int rsram_align_bytes;
  // ASRAM north/south bank stripe width in bytes. The bf16 tensor core can
  // only read from the north bank, so legalization of bf16 GEMM duplicates
  // the A-operand writer with a source-pointer offset of this many bytes so
  // that what previously landed in destination south now lands in north.
  int asram_bank_stripe_bytes;
  // Largest bf16 GEMM row count (M-extent) the bf16 tensor core consumes
  // from the ASRAM north bank in a single pass. A bf16 GEMM whose row count
  // does not exceed this fits entirely in the north bank and needs no
  // two-pass legalization.
  int bf16_gemm_single_pass_max_rows;
};

struct SunmmioMeshConfig {
  int nrow;
  int ncol;
};

// Physical mechanism used by a direct Sunmmio data transfer.
enum class SunmmioTransferMechanism {
  kLocalDma,
  kTile,
  kHLink,
  kVLink,
};

// Sending ODMA selected for an asynchronous Sunmmio transfer. Keep this
// TileLang-side enum independent from NPU-IR's numeric enum representation.
enum class SunmmioOdmaUnit {
  kOdma0,
  kOdma1,
};

// TileLang-side hardware completion mask. Keep these values independent from
// NPU-IR's enum representation and translate them explicitly in SUVM codegen.
using SunmmioSyncUnits = uint32_t;
constexpr SunmmioSyncUnits kSunmmioSyncNone = 0;
constexpr SunmmioSyncUnits kSunmmioSyncOdma0 = 1U << 0;
constexpr SunmmioSyncUnits kSunmmioSyncOdma1 = 1U << 1;
constexpr SunmmioSyncUnits kSunmmioSyncTc = 1U << 2;
constexpr SunmmioSyncUnits kSunmmioSyncHlink = 1U << 3;
constexpr SunmmioSyncUnits kSunmmioSyncVlink = 1U << 4;
constexpr SunmmioSyncUnits kSunmmioSyncVector = 1U << 5;
constexpr SunmmioSyncUnits kSunmmioSyncRsram = 1U << 6;

const char *StringifySunmmioOdmaUnit(SunmmioOdmaUnit unit);
PrimExpr MakeSunmmioOdmaUnitExpr(SunmmioOdmaUnit unit);
std::optional<SunmmioOdmaUnit> ParseSunmmioOdmaUnitExpr(const PrimExpr &expr);
std::optional<SunmmioOdmaUnit> GetSunmmioOdmaUnit(const tir::CallNode *call);

SunmmioTileProcessorConfig
GetSunmmioTileProcessorConfig(ffi::Optional<Target> target);
SunmmioTileProcessorConfig GetSunmmioTileProcessorConfig(Target target);
ffi::Array<PrimExpr> GetSunmmioLayoutBlockShape(ffi::Optional<Target> target,
                                                DataType dtype);
ffi::Array<PrimExpr> GetSunmmioLayoutBlockShape(Target target, DataType dtype);
SunmmioMeshConfig GetSunmmioMeshConfig(ffi::Optional<Target> target);
SunmmioMeshConfig GetSunmmioMeshConfig(Target target);

// Return whether Sunmmio can directly transfer between two buffers using the
// requested mechanism. This query covers memory-scope reachability and dtype
// conversion support. Region shape, layout, alignment, and transfer-size
// constraints are checked by their owning passes and by NPU-IR after lowering.
bool SupportsSunmmioDirectTransfer(Target target,
                                   SunmmioTransferMechanism mechanism,
                                   ffi::String src_scope, DataType src_dtype,
                                   ffi::String dst_scope, DataType dst_dtype);

// Return whether a TileLang copy can transfer directly between two buffers.
bool SupportsSunmmioDirectCopy(Target target, ffi::String src_scope,
                               DataType src_dtype, ffi::String dst_scope,
                               DataType dst_dtype);

// Return whether a source can directly feed every communication direction.
// The direction-to-link mapping is target-specific and remains private to the
// Sunmmio capability implementation.
bool SupportsSunmmioDirectCommunication(
    Target target, CommunicationDirections directions, ffi::String src_scope,
    DataType src_dtype, ffi::String dst_scope, DataType dst_dtype);

// Check whether a buffer scope is one of the Sunmmio on-chip SRAM scopes.
inline bool IsSunmmioSramScope(const ffi::String &scope) {
  return scope == kSunmmioScopeASRAM || scope == kSunmmioScopeWSRAM ||
         scope == kSunmmioScopeRSRAM;
}

// Convert an RSRAM byte-alignment requirement into element count.
inline int GetSunmmioRsramAlignmentElems(int rsram_align_bytes,
                                         DataType dtype) {
  if (rsram_align_bytes <= 0) {
    return 1;
  }
  ICHECK_EQ(dtype.lanes(), 1)
      << "Sunmmio RSRAM alignment expects scalar element dtypes, but got "
      << dtype << ".";
  int element_bits = dtype.bits();
  int align_bits = rsram_align_bytes * 8;
  if (align_bits <= element_bits) {
    return 1;
  }
  ICHECK_EQ(align_bits % element_bits, 0)
      << "RSRAM alignment " << rsram_align_bytes
      << " bytes is not divisible by element bit-width " << element_bits << ".";
  return align_bits / element_bits;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_UTILS_H_
