/*!
 * \file tl/backend/cuda/op/cumsum.cc
 * \brief CUDA implementation for tl.cumsum lowering.
 */

#include "op/reduce.h"

#include "op/utils.h"
#include "target/utils.h"
#include <tvm/tir/builtin.h>

#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

namespace cuda {

struct CumSum {
  static Stmt Lower(const CumSumOpNode &op, const LowerArgs &T,
                    arith::Analyzer *) {
    if (IsFragmentBuffer(op.src) && IsFragmentBuffer(op.dst)) {
      LOG(FATAL) << "CumSum for fragment not implemented, please raise an "
                    "issue if you need this feature.";
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
        ICHECK_EQ(op.dim, 0)
            << "Cumulative sum over a 1D buffer only supports dim = 0.";
        ss << "tl::CumSum1D<" << threads << ", "
           << (op.reverse ? "true" : "false") << ">::run";
        args = {StringImm(ss.str()), src_ptr, dst_ptr, src_extents[0]};
      } else if (ndim == 2) {
        ss << "tl::CumSum2D<" << threads << ", " << op.dim << ", "
           << (op.reverse ? "true" : "false") << ">::run";
        args = {StringImm(ss.str()), src_ptr, dst_ptr, src_extents[0],
                src_extents[1]};
      } else {
        LOG(FATAL) << "CumSum currently supports only 1D or 2D buffers, got "
                   << ndim << "D.";
      }
      return Evaluate(Call(op.dst->dtype, builtin::call_extern(), args));
    } else {
      ICHECK(false) << "Cannot lower cumsum for " << op.src.scope() << " and "
                    << op.dst.scope();
    }

    return Stmt();
  }
};

} // namespace cuda

namespace {

bool MatchCudaCumSumTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaCumSum() {
  RegisterCumSumImpl(CumSumImpl{
      "cuda.CumSum",
      MatchCudaCumSumTarget,
      cuda::CumSum::Lower,
  });
  return true;
}

const bool cuda_cumsum_registered = RegisterCudaCumSum();

} // namespace

} // namespace tl
} // namespace tvm
