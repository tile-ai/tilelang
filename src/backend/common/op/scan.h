/*!
 * \file tl/backend/common/op/scan.h
 * \brief Shared inclusive scan lowering for GPU backends.
 */

#ifndef TVM_TL_BACKEND_COMMON_OP_SCAN_H_
#define TVM_TL_BACKEND_COMMON_OP_SCAN_H_

#include "backend/common/target_utils.h"
#include "op/scan.h"
#include "support/check.h"

#include "op/utils.h"

#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>

#include <sstream>
#include <unordered_set>

namespace tvm {
namespace tl {
namespace backend {

using namespace tirx;
using namespace ffi;

namespace scan {

inline bool
ExprDependsOnAxis(const PrimExpr &expr, const Var &axis,
                  const Map<Var, PrimExpr> &bind_var_to_expr,
                  arith::Analyzer *analyzer,
                  std::unordered_set<const VarNode *> *visited_bind_vars) {
  PrimExpr simplified = analyzer->Simplify(expr);
  return UsesVar(simplified, [&](const VarNode *var_node) {
    if (var_node == axis.get()) {
      return true;
    }
    Var var = GetRef<Var>(var_node);
    if (!bind_var_to_expr.count(var) ||
        !visited_bind_vars->insert(var_node).second) {
      return false;
    }
    return ExprDependsOnAxis(bind_var_to_expr[var], axis, bind_var_to_expr,
                             analyzer, visited_bind_vars);
  });
}

inline bool RegionDependsOnAxis(const BufferRegion &region, const Var &axis,
                                const Map<Var, PrimExpr> &bind_var_to_expr,
                                arith::Analyzer *analyzer) {
  for (const auto &range : region->region) {
    std::unordered_set<const VarNode *> visited_bind_vars;
    if (ExprDependsOnAxis(range->min, axis, bind_var_to_expr, analyzer,
                          &visited_bind_vars) ||
        ExprDependsOnAxis(range->extent, axis, bind_var_to_expr, analyzer,
                          &visited_bind_vars)) {
      return true;
    }
  }
  return false;
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

    Array<PrimExpr> src_extents;
    for (const auto &range : op.srcRegion_->region) {
      src_extents.push_back(range->extent);
    }
    int ndim = static_cast<int>(src_extents.size());

    Array<Var> guarded_thread_axes;
    if (TargetIsCuda(lower_args.target) &&
        !TargetIsCuTeDSL(lower_args.target)) {
      const int64_t *num_threads = as_const_int(threads);
      CHECK(num_threads != nullptr && *num_threads >= 32 &&
                (ndim == 1 || *num_threads % 32 == 0),
            ValueError)
          << "T." << op_name << " on CUDA shared memory requires "
          << (ndim == 1 ? "at least 32 block threads"
                        : "at least 32 block threads and a multiple of 32 "
                          "for 2D scans")
          << "; got " << threads << ".";

      for (const IterVar &thread_axis : lower_args.thread_bindings) {
        if (thread_axis->thread_tag != "threadIdx.y" &&
            thread_axis->thread_tag != "threadIdx.z") {
          continue;
        }
        const int64_t *extent = as_const_int(thread_axis->dom->extent);
        if (extent == nullptr || *extent <= 1) {
          continue;
        }
        CHECK(!RegionDependsOnAxis(op.srcRegion_, thread_axis->var,
                                   lower_args.bind_var_to_expr, analyzer) &&
                  !RegionDependsOnAxis(op.dstRegion_, thread_axis->var,
                                       lower_args.bind_var_to_expr, analyzer),
              ValueError)
            << "T." << op_name
            << " on CUDA shared memory does not support scan regions that "
               "depend on extra threadIdx.y/z axes; got dependency on "
            << thread_axis->thread_tag << ".";
        guarded_thread_axes.push_back(thread_axis->var);
      }
    }

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
    Stmt lowered = Evaluate(Call(op.dst->dtype, builtin::call_extern(), args));
    if (!guarded_thread_axes.empty()) {
      PrimExpr guard = EQ(guarded_thread_axes[0], 0);
      for (size_t i = 1; i < guarded_thread_axes.size(); ++i) {
        guard = And(guard, EQ(guarded_thread_axes[i], 0));
      }
      lowered = IfThenElse(guard, lowered);
    }
    return lowered;
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
