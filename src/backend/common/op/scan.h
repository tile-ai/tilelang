/*!
 * \file tl/backend/common/op/scan.h
 * \brief Shared inclusive scan lowering for GPU backends.
 */

#ifndef TVM_TL_BACKEND_COMMON_OP_SCAN_H_
#define TVM_TL_BACKEND_COMMON_OP_SCAN_H_

#include "op/scan.h"
#include "support/check.h"

#include "op/utils.h"

#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>

#include <sstream>

namespace tvm {
namespace tl {
namespace backend {

using namespace tirx;
using namespace ffi;

namespace scan {

template <typename ScanOpNode>
Stmt LowerSharedScan(const ScanOpNode &op, const LowerArgs &T,
                     const char *op_name, const char *pretty_name,
                     const char *symbol_prefix, const char *scan_noun) {
  if (IsFragmentBuffer(op.src) && IsFragmentBuffer(op.dst)) {
    LOG(FATAL) << pretty_name
               << " for fragment not implemented, please raise an issue if "
                  "you need this feature.";
  } else if (IsSharedBuffer(op.src)) {
    ICHECK(IsSharedBuffer(op.dst));
    std::stringstream ss;
    auto threads = T.thread_bounds->extent;
    Array<PrimExpr> args;

    PrimExpr src_ptr = MakeAccessPtrFromRegion(op.srcRegion_, 1);
    PrimExpr dst_ptr = MakeAccessPtrFromRegion(op.dstRegion_, 2);

    Array<PrimExpr> src_extents;
    for (const auto &range : op.srcRegion_->region) {
      src_extents.push_back(range->extent);
    }
    int ndim = static_cast<int>(src_extents.size());

    if (ndim == 1) {
      ICHECK_EQ(op.dim, 0) << scan_noun
                           << " over a 1D buffer only supports dim = 0.";
      ss << "tl::" << symbol_prefix << "1D<" << threads << ", "
         << (op.reverse ? "true" : "false") << ">::run";
      args = {StringImm(ss.str()), src_ptr, dst_ptr, src_extents[0]};
    } else if (ndim == 2) {
      ss << "tl::" << symbol_prefix << "2D<" << threads << ", " << op.dim
         << ", " << (op.reverse ? "true" : "false") << ">::run";
      args = {StringImm(ss.str()), src_ptr, dst_ptr, src_extents[0],
              src_extents[1]};
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

inline Stmt LowerCumSum(const CumSumOpNode &op, const LowerArgs &T,
                        arith::Analyzer *) {
  return LowerSharedScan(op, T, "cumsum", "CumSum", "CumSum", "Cumulative sum");
}

inline Stmt LowerCumMax(const CumMaxOpNode &op, const LowerArgs &T,
                        arith::Analyzer *) {
  return LowerSharedScan(op, T, "cummax", "CumMax", "CumMax",
                         "Cumulative maximum");
}

} // namespace scan

} // namespace backend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_OP_SCAN_H_
