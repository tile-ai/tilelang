/*!
 * \file tl/op/atomic_add.cc
 *
 * Define elment-wise operators.
 */

#include "atomic_add.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../target/utils.h"
#include "../transform/atomicadd_vectorize.h"
#include "../transform/common/loop_fusion_utils.h"
#include "../transform/loop_partition.h"
#include "builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

static int GetArchInt(Target target) {
  int arch_int = 0;
  auto s = target->GetAttr<String>("arch");
  ICHECK(s.defined());
  const char *arch_str = s.value().c_str();
  if (arch_str[0] == 's' && arch_str[1] == 'm' && arch_str[2] == '_') {
    arch_int = atoi(&arch_str[3]);
  } else {
    arch_int = 0;
  }
  return arch_int;
}

AtomicAdd::AtomicAdd(Array<PrimExpr> args, BufferMap vmap) : args_(args) {
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto expr = args[i];
    auto call = expr.as<CallNode>();
    ICHECK(call);
    auto region = RegionOp(call->args, vmap);
    rgs[i] = region.GetRanges();
    bf[i] = region.GetBuffer();
  }
  std::tie(this->src, this->dst) = std::tie(bf[0], bf[1]);
  std::tie(this->src_range, this->dst_range) = std::tie(rgs[0], rgs[1]);
  if (args.size() >= 3) {
    coalesced_width = Downcast<IntImm>(args[2]);
  }
}

Array<IterVar> AtomicAdd::MakeIterVars() const {
  Array<IterVar> loop_vars;
  size_t idx = 0;
  for (size_t i = 0; i < src_range.size(); i++) {
    if (is_one(src_range[i]->extent))
      continue;
    Var var = Var(std::string{char('i' + idx)}, src_range[i]->extent->dtype);
    idx++;
    loop_vars.push_back(
        {Range(0, src_range[i]->extent), var, IterVarType::kDataPar});
  }
  return loop_vars;
}

// ivs: itervars returned by MakeIterVars()
// src_dst: 0 for src_indices, 1 for dst_indices
Array<PrimExpr> AtomicAdd::MakeIndices(const Array<IterVar> &ivs,
                                       int src_dst) const {
  Array<PrimExpr> indices;
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent))
      indices.push_back(ranges[i]->min);
    else {
      indices.push_back(ranges[i]->min + ivs[idx]->var);
      idx++;
    }
  }
  ICHECK(idx == ivs.size())
      << "idx = " << idx << ", ivs.size() = " << ivs.size()
      << "src name = " << src->name << ", dst name = " << dst->name;
  return indices;
}

PrimExpr AtomicAdd::MakePredicate(arith::Analyzer *analyzer,
                                  const Array<IterVar> &ivs,
                                  Array<PrimExpr> extents, int src_dst) const {
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;
  Array<PrimExpr> cond_list;
  ICHECK(extents.size() == ranges.size()) << extents << " " << ranges;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent))
      continue;
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
  if (cond_list.empty())
    return {};
  else {
    PrimExpr cond = cond_list[0];
    for (size_t i = 1; i < cond_list.size(); i++)
      cond = And(cond, cond_list[i]);
    return cond;
  }
}

For AtomicAdd::MakeSIMTLoop(arith::Analyzer *analyzer) const {
  Array<IterVar> loop_vars = MakeIterVars();
  bool is_scalar = loop_vars.size() == 0;
  if (is_scalar) {
    return For(Var("i"), 0, 1, ForKind::kSerial,
               BufferStore(dst, BufferLoad(src, {0}), {0}));
  }

  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);

  ICHECK(loop_vars.size() <= src_range.size())
      << "loop_vars.size() = " << loop_vars.size()
      << ", src_range.size() = " << src_range.size() << ", src = " << src->name
      << ", dst = " << dst->name;

  ICHECK(loop_vars.size() <= dst_range.size())
      << "loop_vars.size() = " << loop_vars.size()
      << ", dst_range.size() = " << dst_range.size() << ", src = " << src->name
      << ", dst = " << dst->name;

  Array<PrimExpr> src_indices = MakeIndices(loop_vars, 0);
  Array<PrimExpr> dst_indices = MakeIndices(loop_vars, 1);

  PrimExpr src_predicate = MakePredicate(analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate = MakePredicate(analyzer, loop_vars, dst->shape, 1);

  Array<PrimExpr> new_args;
  new_args.push_back(StringImm("AtomicAdd"));

  PrimExpr src_value = BufferLoad(src, src_indices);
  if (src->dtype != dst->dtype)
    src_value = Cast(dst->dtype, src_value);
  if (src_predicate.defined())
    src_value = if_then_else(src_predicate, src_value, make_zero(dst->dtype));

  PrimExpr dst_value = BufferLoad(dst, dst_indices);
  if (dst_predicate.defined())
    dst_value = if_then_else(dst_predicate, dst_value, make_zero(dst->dtype));

  Call address_of_value =
      tvm::tir::Call(DataType::Handle(), builtin::address_of(), {dst_value});

  new_args.push_back(address_of_value);
  new_args.push_back(src_value);

  Call atomicadd_call =
      tvm::tir::Call(dst->dtype, builtin::call_extern(), new_args);

  Stmt body = tvm::tir::Evaluate(atomicadd_call);

  for (int i = loop_vars.size() - 1; i >= 0; i--) {
    Map<String, ObjectRef> annotations = {};
    if (coalesced_width.defined()) {
      annotations.Set("coalesced_width", coalesced_width);
    }

    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body, std::nullopt, annotations);
  }
  return Downcast<For>(body);
}

Stmt AtomicAdd::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  Target target = T.target;
  bool is_cpu_target = target->GetTargetDeviceType() == kDLCPU;
  auto simt_loop = MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
  For vectorized_thread_loop;
  auto par_op = std::make_unique<ParallelOp>(fused_loop);

  if (!is_cpu_target) {
    std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                      InferLevel::kFree};
    for (auto level : levels) {
      par_op->InferLayout(
          {T.target, T.thread_bounds, T.layout_map, T.buffer_remap}, level);
    }
    auto loop_layout = par_op->GetLoopLayout();
    Var thread_var = T.thread_var;
    Range thread_bounds = T.thread_bounds;
    auto thread_loop =
        PartitionLoop(par_op->GetRoot(), T.thread_var, analyzer, loop_layout);
    vectorized_thread_loop = VectorizeAtomicAdd(
        thread_loop, thread_var, thread_bounds, GetArchInt(target));
  }

  if (par_op->GetPredicate(T.thread_var).defined()) {
    return IfThenElse(par_op->GetPredicate(T.thread_var).value(),
                      vectorized_thread_loop);
  }

  return vectorized_thread_loop;
}

LayoutMap AtomicAdd::InferLayout(const LayoutInferArgs &T, InferLevel level) {
  if (par_op_ == nullptr) {
    arith::Analyzer analyzer;
    par_op_ = std::make_unique<ParallelOp>(MakeSIMTLoop(&analyzer));
  }
  if (T.layout_map.count(src) && T.layout_map.count(dst)) {
    if (src.scope() == "local.fragment" && dst.scope() == "local.fragment") {
      const FragmentNode *src_layout = T.layout_map[src].as<FragmentNode>();
      const FragmentNode *dst_layout = T.layout_map[dst].as<FragmentNode>();
      if (src_layout && dst_layout) {
        ICHECK(src_layout->IsEqual(dst_layout, true))
            << "Get different layout for " << src << " and " << dst
            << "\nLHS = " << src_layout->DebugOutput()
            << "\nRHS = " << dst_layout->DebugOutput()
            << "\nYou may need to use a shared memory to transform the layout";
      }
    }
  }
  return par_op_->InferLayout(T, level);
}

TIR_REGISTER_TL_OP(AtomicAdd, atomicadd)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// TVM_REGISTER_OP("tl.atomicadd")
//     .set_num_inputs(2)
//     .add_argument("ref", "Buffer", "The destination buffer")
//     .add_argument("val", "Expr", "The value to be added atomically");

} // namespace tl
} // namespace tvm