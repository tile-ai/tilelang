/*!
 * \file tl/cuda/op/atomic_add.cc
 * \brief CUDA implementation for tl.atomic_add lowering.
 */

#include "op/atomic_add.h"
#include "backend/common/op/atomic_reduce.h"
#include "cuda/stubs/cuda.h"
#include "op/prim.h"
#include "support/check.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/runtime/logging.h>

#include "backend/common/target_utils.h"
#include "cuda/op/builtin.h"
#include "cuda/op/copy.h"
#include "cuda/op/tma_layout.h"
#include "layout/layout.h"
#include "op/utils.h"
#include "span_utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"

#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace cuda {

namespace {

bool IsValidTMAReduceAddDtype(PrimType dtype) {
  if (!dtype.IsScalar()) {
    return false;
  }
  if (dtype.MatchesElementType(kDLBfloat, 16)) {
    return true;
  }
  if (dtype.MatchesCode(kDLFloat)) {
    return dtype.bits() == 16 || dtype.bits() == 32;
  }
  if (dtype.MatchesCode(kDLInt)) {
    return dtype.bits() == 32;
  }
  return dtype.MatchesCode(kDLUInt) &&
         (dtype.bits() == 32 || dtype.bits() == 64);
}

struct TMAAtomicAddPlan {
  int tensor_map_dtype;
};

struct TMAAtomicAddAnalysis {
  std::optional<TMAAtomicAddPlan> plan;
  std::string reason;
};

TMAAtomicAddAnalysis UnsupportedTMAAtomicAdd(std::string reason) {
  return {std::nullopt, std::move(reason)};
}

TMAAtomicAddAnalysis AnalyzeTMAAtomicAdd(const AtomicAddNode &op, Target target,
                                         const Layout &shared_layout) {
  if (!target.defined() || !TargetHasBulkCopy(target)) {
    std::ostringstream reason;
    reason << "TMA atomic add requires a CUDA target with TMA support (SM90+), "
              "but got target "
           << target;
    return UnsupportedTMAAtomicAdd(reason.str());
  }

  if (!op.src.has_value() || !IsSharedBuffer(op.src.value()) ||
      !IsGlobalBuffer(op.dst)) {
    return UnsupportedTMAAtomicAdd(
        "TMA atomic add requires shared source and global destination tensors");
  }
  if (op.GetMemoryOrder() != 0) {
    return UnsupportedTMAAtomicAdd(
        "TMA atomic add only supports relaxed memory ordering");
  }

  if (op.src.value().dtype() != op.dst.dtype()) {
    std::ostringstream reason;
    reason << "TMA atomic add between buffer " << op.src.value()->name
           << " and " << op.dst->name << " requires matching dtypes, but got "
           << op.src.value().dtype() << " and " << op.dst.dtype();
    return UnsupportedTMAAtomicAdd(reason.str());
  }

  PrimType dtype = op.dst.dtype();
  if (!IsValidTMAReduceAddDtype(dtype)) {
    std::ostringstream reason;
    reason << "TMA atomic add does not support dtype " << dtype
           << "; supported scalar dtypes are float16, bfloat16, float32, "
              "int32, uint32, and uint64";
    return UnsupportedTMAAtomicAdd(reason.str());
  }

  TMASharedLayoutAnalysis layout_analysis =
      AnalyzeTMASharedLayout(shared_layout, dtype);
  if (!layout_analysis.encoding.has_value()) {
    std::ostringstream reason;
    reason << "TMA atomic add cannot encode the shared layout for buffer "
           << op.src.value()->name << ": " << layout_analysis.reason;
    return UnsupportedTMAAtomicAdd(reason.str());
  }

  TMAAtomicAddPlan plan{to_CUtensorMapDataType(dtype)};
  return {std::move(plan), ""};
}

Layout MakeTMAAtomicAddSharedLayout(const TensorVar &shared_tensor,
                                    const Array<Range> &region) {
  ICHECK_GE(shared_tensor.shape().size(), 2U)
      << "TMA atomic add layout inference requires a rank-2+ shared buffer";
  int ndim = static_cast<int>(shared_tensor.shape().size());
  auto mat_stride = GetConstInt(shared_tensor.shape()[ndim - 2]);
  auto mat_continuous = GetConstInt(shared_tensor.shape()[ndim - 1]);
  ICHECK(mat_stride.has_value() && mat_continuous.has_value())
      << "TMA atomic add requires constant innermost shared-buffer shapes, "
         "but got "
      << shared_tensor.shape();

  Layout inferred_layout =
      MakeTmaLinearLayout(shared_tensor.shape(), region, shared_tensor.dtype());
  int element_bits = shared_tensor.dtype().bits();
  if ((element_bits == 16 || element_bits == 32) && *mat_stride % 8 == 0) {
    int vector_size = 128 / element_bits;
    if (*mat_continuous % (vector_size * 8) == 0) {
      inferred_layout = MakeFullBankSwizzleLayout(shared_tensor);
    } else if (*mat_continuous % (vector_size * 4) == 0) {
      inferred_layout = MakeHalfBankSwizzleLayout(shared_tensor);
    } else if (*mat_continuous % (vector_size * 2) == 0) {
      inferred_layout = MakeQuarterBankSwizzleLayout(shared_tensor);
    }
  }

  TMASharedLayoutAnalysis layout_analysis =
      AnalyzeTMASharedLayout(inferred_layout, shared_tensor.dtype());
  ICHECK(layout_analysis.encoding.has_value())
      << "Internal error: inferred TMA atomic-add layout is not encodable: "
      << layout_analysis.reason;
  return inferred_layout;
}

} // namespace

struct AtomicAdd {
  static Stmt LowerSIMT(const AtomicAddNode &op, const LowerArgs &lower_args,
                        sym::Analyzer analyzer) {
    return backend::AtomicReduce::Lower(op, lower_args, analyzer);
  }

  static LayoutMap InferLayout(const AtomicAddNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    if (!op.UseTMA()) {
      return backend::AtomicReduce::InferLayout(op, layout_args, level);
    }

    Map<TensorVar, Layout> result_map;
    TensorVar shared_tensor = op.src.value();
    Array<Range> shared_range = op.src_range;
    bool is_tma_1d = shared_range.size() == 1;
    if (is_tma_1d) {
      return result_map;
    }

    if (level == InferLevel::kFree &&
        !layout_args.layout_map.count(shared_tensor)) {
      result_map.Set(shared_tensor,
                     MakeTMAAtomicAddSharedLayout(shared_tensor, shared_range));
    }

    return result_map;
  }

  static Stmt Lower(const AtomicAddNode &op, const LowerArgs &lower_args,
                    sym::Analyzer analyzer) {
    if (!op.UseTMA()) {
      return LowerSIMT(op, lower_args, analyzer);
    }

    // For AtomicAdd with TMA: src is shared memory, dst is global memory.
    TensorVar shared_tensor = op.src.value();
    TensorVar global_tensor = op.dst;
    Array<Range> shared_range = op.src_range;
    Array<Range> global_range = op.dst_range;

    // The reduce dtype whitelist is stricter than plain TMA copies.
    Layout shared_layout = MakeLinearLayout(shared_tensor.shape());
    if (lower_args.layout_map.count(shared_tensor)) {
      shared_layout = lower_args.layout_map.at(shared_tensor);
    }
    TMAAtomicAddAnalysis analysis =
        AnalyzeTMAAtomicAdd(op, lower_args.target, shared_layout);
    ICHECK(analysis.plan.has_value())
        << analysis.reason
        << SpanHintSuffix({op.dst->loc, op.src.value()->loc});

    TMABulkCopyAnalysis bulk_analysis = AnalyzeTMABulkCopy(
        lower_args, global_tensor, shared_tensor, global_range, shared_range);
    ICHECK(bulk_analysis.plan.has_value())
        << "TMA atomic add cannot lower the copy: " << bulk_analysis.reason
        << SpanHintSuffix({op.dst->loc, op.src.value()->loc});
    const TMABulkCopyPlan &plan = bulk_analysis.plan.value();

    TMADesc desc = plan.desc;
    desc.data_type = analysis.plan.value().tensor_map_dtype;
    desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
    desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);
    shared_tensor = plan.shared_tensor;

    Call create_descriptor =
        Call(PtrType(tirx::TensorMapType()), create_tma_descriptor(),
             desc.EncodeCallArgs());

    auto op_annotations = op.annotations;
    op_annotations.erase("use_tma");

    auto make_copy = [&](std::optional<PrimExpr> rest_idx) {
      Array<Expr> args;
      args.reserve(desc.rank + 4);
      args.push_back(create_descriptor);
      args.push_back(plan.SharedPointer(rest_idx, kAccessRead));
      for (auto coord : plan.TmaCoords(rest_idx))
        args.push_back(coord);
      args.push_back(PrimExpr(1)); // reduce (add)
      args.push_back(PrimExpr(0)); // eviction policy
      return Evaluate(Call(TupleType(Array<Type>{}), tma_store(), args,
                           DictAttrs(op_annotations)));
    };

    Stmt tma_reduce = plan.EmitInstructions(make_copy);

    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(tma_reduce);
    seq.push_back(
        Evaluate(Call(TupleType(Array<Type>{}), tma_store_arrive(), {})));
    seq.push_back(Evaluate(Call(TupleType(Array<Type>{}), tma_store_wait(),
                                {IntImm(PrimType::Int(32), 0),
                                 tl::MakeConst(PrimType::Bool(), true)})));
    return If(EQ(lower_args.thread_index, lower_args.thread_bounds->begin),
              SeqStmt(std::move(seq)));
  }
};

} // namespace cuda

namespace {

bool MatchCudaAtomicAddTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaAtomicAdd() {
  RegisterAtomicAddImpl(AtomicAddImpl{
      "cuda.AtomicAdd",
      MatchCudaAtomicAddTarget,
      cuda::AtomicAdd::InferLayout,
      cuda::AtomicAdd::Lower,
  });
  return true;
}

const bool cuda_atomic_add_registered = RegisterCudaAtomicAdd();

} // namespace

} // namespace tl
} // namespace tvm
