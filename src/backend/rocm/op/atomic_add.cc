/*!
 * \file tl/backend/rocm/op/atomic_add.cc
 * \brief ROCm implementation for tl.atomic_add lowering.
 */

#include "op/atomic_add.h"

#include "layout/layout.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "target/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace rocm {

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
      tvm::tir::Call(op.dst->dtype, op.GetElemOp(), new_args, annotations);

  Stmt body = tvm::tir::Evaluate(atomicadd_call);

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

LayoutMap InferSIMTLayout(const AtomicAddNode &op, const LayoutInferArgs &T,
                          InferLevel) {
  if (IsFragmentBuffer(op.src) && IsFragmentBuffer(op.dst)) {
    if (T.layout_map.count(op.src) && T.layout_map.count(op.dst)) {
      Layout src_layout = T.layout_map.at(op.src);
      Layout dst_layout = T.layout_map.at(op.dst);
      ICHECK(StructuralEqual()(src_layout, dst_layout))
          << "AtomicAdd requires src and dst to have the same layout, but got "
          << "src layout: " << src_layout << ", dst layout: " << dst_layout
          << " for src buffer: " << op.src->name
          << ", dst buffer: " << op.dst->name;
    }
  }
  return {};
}

} // namespace

struct AtomicAdd {
  static Stmt LowerSIMT(const AtomicAddNode &op, const LowerArgs &T,
                        arith::Analyzer *analyzer) {
    auto simt_loop = MakeSIMTLoop(op, analyzer);
    auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
    auto par_op = ParallelOp(fused_loop);
    std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                      InferLevel::kFree};
    for (auto level : levels) {
      par_op->InferLayout({T.target,
                           T.thread_bounds,
                           T.layout_map,
                           analyzer,
                           false,
                           T.buffer_remap,
                           {}},
                          level);
    }
    auto loop_layout = par_op->GetLoopLayout();
    return LowerParallelLoop(fused_loop, loop_layout, T.thread_var, analyzer,
                             T.layout_map, par_op->GetPredicate(T.thread_var));
  }

  static LayoutMap InferLayout(const AtomicAddNode &op,
                               const LayoutInferArgs &T, InferLevel level) {
    ICHECK(!UseTMA(op))
        << "TMA atomic_add is only supported by the CUDA backend";
    return InferSIMTLayout(op, T, level);
  }

  static Stmt Lower(const AtomicAddNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    ICHECK(!UseTMA(op))
        << "TMA atomic_add is only supported by the CUDA backend";
    return LowerSIMT(op, T, analyzer);
  }
};

} // namespace rocm

namespace {

bool MatchROCmAtomicAddTarget(Target target) { return TargetIsRocm(target); }

bool RegisterROCmAtomicAdd() {
  RegisterAtomicAddImpl(AtomicAddImpl{
      "rocm.AtomicAdd",
      MatchROCmAtomicAddTarget,
      rocm::AtomicAdd::InferLayout,
      rocm::AtomicAdd::Lower,
  });
  return true;
}

const bool rocm_atomic_add_registered = RegisterROCmAtomicAdd();

} // namespace

} // namespace tl
} // namespace tvm
