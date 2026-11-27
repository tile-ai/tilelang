/*!
 * \file tl/backend/common/op/scan.h
 * \brief Shared inclusive scan lowering for GPU backends.
 */

#ifndef TVM_TL_BACKEND_COMMON_OP_SCAN_H_
#define TVM_TL_BACKEND_COMMON_OP_SCAN_H_

#include "op/scan.h"
#include "support/check.h"

#include "op/utils.h"

#include <sstream>

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>

namespace tvm {
namespace tl {
namespace backend {

using namespace tirx;
using namespace ffi;

namespace scan {

template <typename ScanOpNode>
Array<PrimExpr> ResolveScanExtents(const ScanOpNode &op,
                                   arith::Analyzer *analyzer,
                                   const char *pretty_name) {
  arith::Analyzer local_analyzer;
  if (analyzer == nullptr) {
    analyzer = &local_analyzer;
  }

  const Array<Range> &src_ranges = op.srcRegion_->region;
  const Array<Range> &dst_ranges = op.dstRegion_->region;
  const int src_rank = static_cast<int>(src_ranges.size());
  const int dst_rank = static_cast<int>(dst_ranges.size());

  int logical_ndim = op.logical_ndim > 0 ? op.logical_ndim : src_rank;
  ICHECK_GE(src_rank, logical_ndim)
      << pretty_name << " source region rank (" << src_rank
      << ") is smaller than logical_ndim (" << logical_ndim << ").";
  const int src_lead = src_rank - logical_ndim;
  for (int i = 0; i < src_lead; ++i) {
    ICHECK(is_one(src_ranges[i]->extent))
        << pretty_name << " source has non-unit leading dimension at axis " << i
        << " (extent=" << src_ranges[i]->extent
        << ") before logical_ndim=" << logical_ndim
        << ", src_region=" << src_ranges;
  }

  ICHECK_GE(dst_rank, logical_ndim)
      << pretty_name << " destination region rank (" << dst_rank
      << ") is smaller than logical_ndim (" << logical_ndim << ").";
  const int dst_lead = dst_rank - logical_ndim;
  for (int i = 0; i < dst_lead; ++i) {
    ICHECK(is_one(dst_ranges[i]->extent))
        << pretty_name << " destination has non-unit leading dimension at axis "
        << i << " (extent=" << dst_ranges[i]->extent
        << ") before logical_ndim=" << logical_ndim
        << ", dst_region=" << dst_ranges;
  }

  Array<PrimExpr> extents;
  extents.reserve(logical_ndim);
  for (int i = 0; i < logical_ndim; ++i) {
    const PrimExpr &src_ext = src_ranges[src_lead + i]->extent;
    const PrimExpr &dst_ext = dst_ranges[dst_lead + i]->extent;
    // Only reject provably mismatched extents so runtime-guarded dynamic slices
    // remain legal.
    ICHECK(!analyzer->CanProve(src_ext != dst_ext))
        << pretty_name << " source and destination extent mismatch at logical "
        << "axis " << i << ": src=" << src_ext << ", dst=" << dst_ext;
    extents.push_back(src_ext);
  }
  return extents;
}

template <typename ScanOpNode>
Stmt LowerSharedScan(const ScanOpNode &op, const LowerArgs &lower_args,
                     arith::Analyzer *analyzer, const char *op_name,
                     const char *pretty_name, const char *symbol_prefix,
                     const char *scan_noun) {
  if (IsFragmentBuffer(op.src) && IsFragmentBuffer(op.dst)) {
    LOG(FATAL) << pretty_name
               << " for fragment not implemented, please raise an issue if "
                  "you need this feature.";
  } else if (IsSharedBuffer(op.src)) {
    ICHECK(IsSharedBuffer(op.dst));
    std::stringstream ss;
    auto threads = lower_args.thread_bounds->extent;
    auto thread_offset = lower_args.thread_bounds->min;
    Array<PrimExpr> args;

    PrimExpr src_ptr = MakeAccessPtrFromRegion(op.srcRegion_, 1);
    PrimExpr dst_ptr = MakeAccessPtrFromRegion(op.dstRegion_, 2);

    Array<PrimExpr> src_extents = ResolveScanExtents(op, analyzer, pretty_name);
    int ndim = static_cast<int>(src_extents.size());

    if (ndim == 1) {
      ICHECK_EQ(op.dim, 0) << scan_noun
                           << " over a 1D buffer only supports dim = 0.";
      ss << "tl::" << symbol_prefix << "1D<" << threads << ", "
         << (op.reverse ? "true" : "false") << ", " << thread_offset
         << ">::run";
      args = {StringImm(ss.str()), src_ptr, dst_ptr, src_extents[0]};
    } else if (ndim == 2) {
      ss << "tl::" << symbol_prefix << "2D<" << threads << ", " << op.dim
         << ", " << (op.reverse ? "true" : "false") << ", " << thread_offset
         << ">::run";
      PrimExpr src_stride = op.src->shape[op.src->shape.size() - 1];
      PrimExpr dst_stride = op.dst->shape[op.dst->shape.size() - 1];
      args = {StringImm(ss.str()), src_ptr,    dst_ptr,   src_extents[0],
              src_extents[1],      src_stride, dst_stride};
    } else {
      LOG(FATAL) << pretty_name
                 << " currently supports only 1D or 2D buffers, got " << ndim
                 << "D.";
    }
    return Evaluate(Call(op.dst->dtype, builtin::call_extern(), args));
  } else {
    ICHECK(false) << "Cannot lower " << op_name << " for " << op.src.scope()
                  << " and " << op.dst.scope();
  }

  return Stmt();
}

inline Stmt LowerCumSum(const CumSumOpNode &op, const LowerArgs &lower_args,
                        arith::Analyzer *analyzer) {
  return LowerSharedScan(op, lower_args, analyzer, "cumsum", "CumSum", "CumSum",
                         "Cumulative sum");
}

inline Stmt LowerCumMax(const CumMaxOpNode &op, const LowerArgs &lower_args,
                        arith::Analyzer *analyzer) {
  return LowerSharedScan(op, lower_args, analyzer, "cummax", "CumMax", "CumMax",
                         "Cumulative maximum");
}

} // namespace scan

} // namespace backend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_OP_SCAN_H_
