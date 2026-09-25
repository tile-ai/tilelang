/*!
 * \file transfer_analysis.h
 * \brief Transfer semantics shared by scheduling and instruction selection.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_TRANSFER_ANALYSIS_H_
#define TVM_TL_TRANSFORM_COMMON_TRANSFER_ANALYSIS_H_

#include <optional>
#include <tvm/target/target.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

/*! A load, optionally converted and/or zero-filled. This describes values,
 * not an instruction: vector width, alignment and target support are checked
 * by the backend after layout lowering. A data predicate writes zero on its
 * false path; it is not an execution predicate that skips the destination.
 */
struct TransferValue {
  tirx::BufferLoad source;
  ffi::Optional<PrimExpr> zero_fill_predicate;
  bool converts_value = false;
};

std::optional<TransferValue> AnalyzeTransferValue(const PrimExpr &value);

/*! Facts about an entire scheduling unit, without assigning it a stage.
 *
 * A producer can compute values without being a transfer. A transfer can be
 * an async candidate without being profitable to prefetch. Neither property
 * promises that an async instruction will survive physical lowering.
 */
struct TransferSummary {
  bool reads_global = false;
  bool writes_shared = false;
  bool reads_shared_or_local = false;
  bool writes_other = false;
  bool has_global_to_shared = false;
  bool only_transfers = true;
  bool byte_preserving = true;
  bool execution_guard = false;
  bool opaque_effect = false;
  bool tma = false;

  bool IsSimtProducer() const {
    return reads_global && writes_shared && !reads_shared_or_local &&
           !writes_other && !opaque_effect;
  }
  bool IsAsyncCandidate() const {
    return IsSimtProducer() && has_global_to_shared && only_transfers &&
           byte_preserving && !execution_guard && !tma;
  }
};

TransferSummary AnalyzeTransfers(const tirx::Stmt &stmt, const Target &target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_TRANSFER_ANALYSIS_H_
