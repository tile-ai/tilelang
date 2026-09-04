/*!
 * \file tl/cuda/op/atomic_add.cc
 * \brief CUDA implementation for tl.atomic_add lowering.
 */

#include "op/atomic_add.h"
#include "support/check.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/cast.h>
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

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

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

bool UseTMA(const AtomicAddNode &op) {
  if (auto val = op.annotations.Get("use_tma")) {
    if (auto int_val = val->as<IntImmNode>()) {
      if (int_val->value != 0) {
        ICHECK(!op.src_value.defined())
            << "TMA is not supported when using TiledAtomicAdd with PrimExpr "
               "as value.";
        return true;
      }
    }
  }
  return false;
}

bool IsValidTMAReduceAddDtype(DataType dtype) {
  if (!dtype.is_scalar()) {
    return false;
  }
  if (dtype.is_bfloat16()) {
    return true;
  }
  if (dtype.is_float()) {
    return dtype.bits() == 16 || dtype.bits() == 32;
  }
  if (dtype.is_int()) {
    return dtype.bits() == 32;
  }
  return dtype.is_uint() && (dtype.bits() == 32 || dtype.bits() == 64);
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

  if (op.src->dtype != op.dst->dtype) {
    std::ostringstream reason;
    reason << "TMA atomic add between buffer " << op.src->name << " and "
           << op.dst->name << " requires matching dtypes, but got "
           << op.src->dtype << " and " << op.dst->dtype;
    return UnsupportedTMAAtomicAdd(reason.str());
  }

  DataType dtype = op.dst->dtype;
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
           << op.src->name << ": " << layout_analysis.reason;
    return UnsupportedTMAAtomicAdd(reason.str());
  }

  TMAAtomicAddPlan plan{to_CUtensorMapDataType(dtype)};
  return {std::move(plan), ""};
}

Layout MakeTMAAtomicAddSharedLayout(const Buffer &shared_tensor,
                                    const Array<Range> &region) {
  ICHECK_GE(shared_tensor->shape.size(), 2U)
      << "TMA atomic add layout inference requires a rank-2+ shared buffer";
  int ndim = static_cast<int>(shared_tensor->shape.size());
  const int64_t *mat_stride = as_const_int(shared_tensor->shape[ndim - 2]);
  const int64_t *mat_continuous = as_const_int(shared_tensor->shape[ndim - 1]);
  ICHECK(mat_stride != nullptr && mat_continuous != nullptr)
      << "TMA atomic add requires constant innermost shared-buffer shapes, "
         "but got "
      << shared_tensor->shape;

  Layout inferred_layout = MakeTmaLinearLayout(shared_tensor->shape, region);
  int element_bits = shared_tensor->dtype.bits();
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
      AnalyzeTMASharedLayout(inferred_layout, shared_tensor->dtype);
  ICHECK(layout_analysis.encoding.has_value())
      << "Internal error: inferred TMA atomic-add layout is not encodable: "
      << layout_analysis.reason;
  return inferred_layout;
}

Array<IterVar> MakeIterVars(const AtomicAddNode &op) {
  Array<IterVar> loop_vars;
  size_t idx = 0;
  for (size_t i = 0; i < op.dst_range.size(); i++) {
    if (is_one(op.dst_range[i]->extent)) {
      continue;
    }
    Var var = Var(std::string{char('i' + idx)}, op.dst_range[i]->extent->dtype);
    idx++;
    loop_vars.push_back(
        {Range(0, op.dst_range[i]->extent), var, IterVarType::kDataPar});
  }

  if (loop_vars.empty()) {
    Var var = Var("i");
    loop_vars.push_back({Range(0, 1), var, IterVarType::kDataPar});
  }

  return loop_vars;
}

Array<PrimExpr> MakeIndices(const AtomicAddNode &op, const Array<IterVar> &ivs,
                            int src_dst) {
  Array<PrimExpr> indices;
  Array<Range> ranges = src_dst == 0 ? op.src_range : op.dst_range;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent)) {
      indices.push_back(ranges[i]->min);
    } else {
      indices.push_back(ranges[i]->min + ivs[idx]->var);
      idx++;
    }
  }

  ICHECK(idx == ivs.size() || (idx == 0 && ivs.size() == 1))
      << "Unmatched indices: idx = " << idx << ", ivs.size() = " << ivs.size()
      << ", dst name = " << op.dst->name;
  return indices;
}

PrimExpr MakePredicate(const AtomicAddNode &op, arith::Analyzer *analyzer,
                       const Array<IterVar> &ivs, Array<PrimExpr> extents,
                       int src_dst) {
  Array<Range> ranges = src_dst == 0 ? op.src_range : op.dst_range;
  Array<PrimExpr> cond_list;
  ICHECK(extents.size() == ranges.size()) << extents << " " << ranges;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent)) {
      continue;
    }
    PrimExpr cond = ranges[i]->min + ivs[idx]->var < extents[i];
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    cond = ranges[i]->min + ivs[idx]->var >= 0;
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    idx++;
  }
  if (cond_list.empty()) {
    return {};
  }
  PrimExpr cond = cond_list[0];
  for (size_t i = 1; i < cond_list.size(); i++) {
    cond = And(cond, cond_list[i]);
  }
  return cond;
}

For MakeSIMTLoop(const AtomicAddNode &op, arith::Analyzer *analyzer) {
  Array<IterVar> loop_vars = MakeIterVars(op);
  ICHECK(!loop_vars.empty()) << "MakeIterVars in AtomicOp should not return "
                                "empty vars (at least 1 var)";

  for (const auto &iv : loop_vars) {
    analyzer->Bind(iv->var, iv->dom);
  }

  ICHECK(loop_vars.size() <= op.dst_range.size())
      << "loop_vars.size() = " << loop_vars.size()
      << ", dst_range.size() = " << op.dst_range.size()
      << ", dst = " << op.dst->name;

  Array<PrimExpr> dst_indices = MakeIndices(op, loop_vars, 1);
  Array<PrimExpr> new_args;

  PrimExpr dst_predicate =
      MakePredicate(op, analyzer, loop_vars, op.dst->shape, 1);

  PrimExpr src_value_arg;

  if (!op.src_value.defined()) {
    ICHECK(loop_vars.size() <= op.src_range.size())
        << "loop_vars.size() = " << loop_vars.size()
        << ", src_range.size() = " << op.src_range.size()
        << ", src = " << op.src->name << ", dst = " << op.dst->name;

    Array<PrimExpr> src_indices = MakeIndices(op, loop_vars, 0);
    PrimExpr src_predicate =
        MakePredicate(op, analyzer, loop_vars, op.src->shape, 0);
    src_value_arg = BufferLoad(op.src, src_indices);
  } else {
    src_value_arg = op.src_value;
  }

  if (src_value_arg->dtype != op.dst->dtype) {
    src_value_arg = Cast(op.dst->dtype, src_value_arg);
  }

  DataType idx_dtype =
      dst_indices.empty() ? DataType::Int(32) : dst_indices[0].dtype();
  PrimExpr dst_ptr =
      Call(DataType::Handle(), tl::access_ptr(),
           {BufferLoad(op.dst, dst_indices), make_const(idx_dtype, 1),
            make_const(DataType::Int(32), 3)});

  new_args.push_back(dst_ptr);
  new_args.push_back(src_value_arg);
  new_args.push_back(op.GetMemoryOrder());

  auto annotations = op.annotations;
  annotations.erase("use_tma");
  Call atomicadd_call =
      tvm::tirx::Call(op.dst->dtype, op.GetElemOp(), new_args, annotations);

  Stmt body = tvm::tirx::Evaluate(atomicadd_call);

  for (int i = loop_vars.size() - 1; i >= 0; i--) {
    Map<String, ObjectRef> loop_annotations;
    if (i == 0) {
      if (annotations.count(attr::kCoalescedWidth)) {
        loop_annotations.Set(attr::kCoalescedWidth,
                             annotations.Get(attr::kCoalescedWidth).value());
      }
    }

    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body, std::nullopt, loop_annotations);
  }
  return Downcast<For>(body);
}

LayoutMap InferSIMTLayout(const AtomicAddNode &op,
                          const LayoutInferArgs &layout_args, InferLevel) {
  if (IsFragmentBuffer(op.src) && IsFragmentBuffer(op.dst)) {
    if (layout_args.layout_map.count(op.src) &&
        layout_args.layout_map.count(op.dst)) {
      Layout src_layout = layout_args.layout_map.at(op.src);
      Layout dst_layout = layout_args.layout_map.at(op.dst);
      ICHECK(StructuralEqual()(src_layout, dst_layout))
          << "AtomicAdd requires src and dst to have the same layout, but got "
          << "src layout: " << src_layout << ", dst layout: " << dst_layout
          << " for src buffer: " << op.src->name
          << ", dst buffer: " << op.dst->name
          << SpanHintSuffix({op.dst->span, op.src->span});
    }
  }
  return {};
}

} // namespace

struct AtomicAdd {
  static Stmt LowerSIMT(const AtomicAddNode &op, const LowerArgs &lower_args,
                        arith::Analyzer *analyzer) {
    auto simt_loop = MakeSIMTLoop(op, analyzer);
    auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
    auto par_op = ParallelOp(fused_loop);
    std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                      InferLevel::kFree};
    for (auto level : levels) {
      par_op->InferLayout({lower_args.target,
                           lower_args.thread_bounds,
                           lower_args.layout_map,
                           analyzer,
                           lower_args.buffer_remap,
                           {}},
                          level);
    }
    auto loop_layout = par_op->GetLoopLayout();
    return LowerParallelLoop(
        fused_loop, loop_layout, lower_args.thread_index, analyzer,
        lower_args.layout_map, par_op->GetPredicate(lower_args.thread_index),
        /*parallel_loop=*/true, par_op->LoopLayoutRequiresPaddingGuard());
  }

  static LayoutMap InferLayout(const AtomicAddNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    if (!UseTMA(op)) {
      return InferSIMTLayout(op, layout_args, level);
    }

    Map<Buffer, Layout> result_map;
    Buffer shared_tensor = op.src;
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
                    arith::Analyzer *analyzer) {
    if (!UseTMA(op)) {
      return LowerSIMT(op, lower_args, analyzer);
    }

    // For AtomicAdd with TMA: src is shared memory, dst is global memory.
    Buffer shared_tensor = op.src;
    Buffer global_tensor = op.dst;
    Array<Range> shared_range = op.src_range;
    Array<Range> global_range = op.dst_range;

    // The reduce dtype whitelist is stricter than plain TMA copies.
    Layout shared_layout = MakeLinearLayout(shared_tensor->shape);
    if (lower_args.layout_map.count(shared_tensor)) {
      shared_layout = lower_args.layout_map.at(shared_tensor);
    }
    TMAAtomicAddAnalysis analysis =
        AnalyzeTMAAtomicAdd(op, lower_args.target, shared_layout);
    ICHECK(analysis.plan.has_value())
        << analysis.reason << SpanHintSuffix({op.dst->span, op.src->span});

    TMABulkCopyAnalysis bulk_analysis = AnalyzeTMABulkCopy(
        lower_args, global_tensor, shared_tensor, global_range, shared_range);
    ICHECK(bulk_analysis.plan.has_value())
        << "TMA atomic add cannot lower the copy: " << bulk_analysis.reason
        << SpanHintSuffix({op.dst->span, op.src->span});
    const TMABulkCopyPlan &plan = bulk_analysis.plan.value();

    TMADesc desc = plan.desc;
    desc.data_type = analysis.plan.value().tensor_map_dtype;
    desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
    desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);
    shared_tensor = plan.shared_tensor;

    Call create_descriptor = Call(DataType::Handle(), create_tma_descriptor(),
                                  desc.EncodeCallArgs());

    PrimExpr total_elements = IntImm(DataType::Int(32), plan.box_size);

    auto op_annotations = op.annotations;
    op_annotations.erase("use_tma");

    auto make_copy = [&](std::optional<PrimExpr> rest_idx) {
      Array<PrimExpr> args;
      args.reserve(desc.rank + 4);
      args.push_back(create_descriptor);
      args.push_back(shared_tensor.access_ptr(1, DataType::Handle(), 1,
                                              plan.SharedOffset(rest_idx),
                                              total_elements));
      for (auto coord : plan.TmaCoords(rest_idx))
        args.push_back(coord);
      args.push_back(1); // reduce (add)
      args.push_back(0); // eviction policy
      return Evaluate(
          Call(DataType::Handle(), tma_store(), args, op_annotations));
    };

    Stmt tma_reduce = plan.EmitInstructions(make_copy);

    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(tma_reduce);
    seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_arrive(), {})));
    seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_wait(),
                                {IntImm(DataType::Int(32), 0), Bool(true)})));
    return IfThenElse(
        EQ(lower_args.thread_index, lower_args.thread_bounds->min),
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
