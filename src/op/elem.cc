/*!
 * \file tl/op/elem.cc
 *
 * Define elment-wise operators.
 */

#include "elem.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../target/utils.h"
#include "../transform/common/loop_fusion_utils.h"
#include "../transform/common/loop_parallel_transform_utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"
#include "builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

Fill::Fill(Array<PrimExpr> args, BufferMap vmap) {
  ObjectPtr<FillNode> node = make_object<FillNode>();

  if (args[0]->IsInstance<BufferLoadNode>()) {
    auto buffer_load = Downcast<BufferLoad>(args[0]);
    for (const auto &index : buffer_load->indices) {
      if (const auto *ramp = index.as<RampNode>()) {
        CHECK(ramp->stride.as<IntImmNode>()->value == 1)
            << "Only stride 1 ramps are supported";
        const auto *lanes = ramp->lanes.as<IntImmNode>();
        CHECK(lanes)
            << "Scalable vectors not supported in BufferRegion conversion";
        node->region.push_back(Range::FromMinExtent(ramp->base, ramp->lanes));
      } else {
        node->region.push_back(Range::FromMinExtent(index, 1));
      }
    }
    node->dst = buffer_load->buffer;
  } else {
    node->dst = vmap[GetVarFromAccessPtr(args[0])];
    for (int i = 0; i < node->dst->shape.size(); i++) {
      node->region.push_back(Range(0, node->dst->shape[i]));
    }
  }

  if (args[1]->dtype != node->dst->dtype) {
    node->value = Cast(node->dst->dtype, args[1]);
  } else {
    node->value = args[1];
  }

  ICHECK(node->region.size() == node->dst->shape.size())
      << "region size = " << node->region.size()
      << " != " << node->dst->shape.size();
  for (int i = 0; i < node->region.size(); i++) {
    // bound check if region is static
    if (node->region[i]->min.as<IntImm>()) {
      int64_t min = Downcast<IntImm>(node->region[i]->min)->value;
      ICHECK_GE(min, 0) << "region[" << i << "] = " << min << " < 0";
    }
    if (node->region[i]->extent.as<IntImm>()) {
      int64_t extent = Downcast<IntImm>(node->region[i]->extent)->value;
      ICHECK_LE(extent, Downcast<IntImm>(node->dst->shape[i])->value)
          << "region[" << i << "] = " << extent << " > " << node->dst->shape[i];
    }
  }
  data_ = std::move(node);
}

TileOperator FillNode::Clone() const {
  auto op = make_object<FillNode>(*this);
  return Fill(op);
}

For FillNode::MakeSIMTLoop(arith::Analyzer *analyzer) const {
  int ndim = dst->shape.size();
  Array<IterVar> loop_vars;
  Array<PrimExpr> dst_indices;
  for (int i = 0; i < ndim; i++) {
    Var var = Var(std::string{char('i' + i)}, region[i]->extent->dtype);
    loop_vars.push_back({region[i], var, IterVarType::kDataPar});
    dst_indices.push_back(var);
  }
  Stmt body = BufferStore(dst, value, dst_indices);
  for (int i = ndim - 1; i >= 0; i--) {
    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body);
  }
  return Downcast<For>(body);
}

Stmt FillNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (dst.scope() == "local.fragment") {
    auto par_op = ParallelOp(MakeSIMTLoop(analyzer));
    par_op->InferLayout({T.target, T.thread_bounds, T.layout_map},
                        InferLevel::kFree);
    par_op->InferLayout({T.target, T.thread_bounds, T.layout_map},
                        InferLevel::kFree);
    auto thread_loop = PartitionLoop(par_op->GetRoot(), T.thread_var, analyzer,
                                     par_op->GetLoopLayout());
    auto vectorized_thread_loop = VectorizeLoop(thread_loop);
    if (par_op->GetPredicate(T.thread_var).defined()) {
      return IfThenElse(par_op->GetPredicate(T.thread_var).value(),
                        vectorized_thread_loop);
    }
    return vectorized_thread_loop;
  } else if (dst.scope() == "local") {
    auto init_loop = MakeSIMTLoop(analyzer);
    auto vectorized_thread_loop = VectorizeLoop(init_loop);
    return vectorized_thread_loop;
  } else if (dst.scope() == "shared.dyn" || dst.scope() == "shared") {
    auto par_op = ParallelOp(MakeSIMTLoop(analyzer));
    par_op->InferLayout({T.target, T.thread_bounds, T.layout_map},
                        InferLevel::kFree);
    auto thread_loop = PartitionLoop(par_op->GetRoot(), T.thread_var, analyzer,
                                     par_op->GetLoopLayout());
    auto vectorized_thread_loop = VectorizeLoop(thread_loop);
    if (par_op->GetPredicate(T.thread_var).defined()) {
      return IfThenElse(par_op->GetPredicate(T.thread_var).value(),
                        vectorized_thread_loop);
    }
    return vectorized_thread_loop;
  } else {
    LOG(FATAL) << "Unsupported scope " << dst.scope();
  }
}

LayoutMap FillNode::InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const {
  return {};
}

TIR_REGISTER_TL_OP(Fill, fill)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm