// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/reduce.cc
 *
 * Define reduce operator.
 */

#include "reduce.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>

#include "../layout/utils.h"
#include "../transform/loop_partition.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;

ReduceOp::ReduceOp(Array<PrimExpr> args, BufferMap vmap) {
  src = vmap[GetVarFromAccessPtr(args[0])];
  dst = vmap[GetVarFromAccessPtr(args[1])];
  String reduce_type = args[2].as<StringImm>().value()->value;
  dim = args[3].as<IntImm>().value()->value;
  if (reduce_type == "sum")
    type = ReduceType::kSum;
  else if (reduce_type == "abssum")
    type = ReduceType::kAbsSum;
  else if (reduce_type == "absmax")
    type = ReduceType::kAbsMax;
  else if (reduce_type == "max")
    type = ReduceType::kMax;
  else if (reduce_type == "min")
    type = ReduceType::kMin;
  else
    ICHECK(0) << "Unknown reduce type: " << reduce_type;
  clear = args[4].as<Bool>().value();
}

PrimExpr ReduceOp::MakeInitValue() const {
  switch (type) {
  case ReduceType::kSum:
    return make_zero(dst->dtype);
  case ReduceType::kAbsSum:
    return make_zero(dst->dtype);
  case ReduceType::kMax:
    return make_const(dst->dtype, -INFINITY);
  case ReduceType::kMin:
    return make_const(dst->dtype, INFINITY);
  case ReduceType::kAbsMax:
    return make_const(dst->dtype, 0);
  default:
    ICHECK(0);
  }
}

PrimExpr ReduceOp::MakeReduce(const PrimExpr &a, const PrimExpr &b) const {
  PrimExpr lhs = a, rhs = b;
  if (lhs->dtype != rhs->dtype) {
    rhs = Cast(lhs->dtype, rhs);
  }
  switch (type) {
  case ReduceType::kSum:
    return lhs + rhs;
  case ReduceType::kAbsSum:
    return lhs + Max(rhs, -rhs);
  case ReduceType::kMax:
    return Max(lhs, rhs);
  case ReduceType::kMin:
    return Min(lhs, rhs);
  case ReduceType::kAbsMax:
    return Max(Max(lhs, rhs), -Min(lhs, rhs));
  default:
    ICHECK(0);
    return PrimExpr(0);
  }
}

std::string ReduceOp::MakeCodegenReducer() const {
  switch (type) {
  case ReduceType::kSum:
    return "tl::SumOp";
  case ReduceType::kAbsSum:
    return "tl::SumOp";
  case ReduceType::kMax:
    return "tl::MaxOp";
  case ReduceType::kMin:
    return "tl::MinOp";
  case ReduceType::kAbsMax:
    return "tl::MaxOp";
  default:
    ICHECK(0);
    return "";
  }
}

Stmt ReduceOp::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  ICHECK(this->src.scope() == "local.fragment" &&
         this->dst.scope() == "local.fragment")
      << "Reduce for shared memory not implemented.";
  auto src_buffer = T.buffer_remap[this->src];
  auto dst_buffer = T.buffer_remap[this->dst];
  Fragment src_layout = T.layout_map[this->src].as<Fragment>().value();
  Fragment dst_layout = T.layout_map[this->dst].as<Fragment>().value();
  ICHECK(src_layout->InputDim() == dst_layout->InputDim() + 1);
  Array<IterVar> dst_vars;
  for (size_t i = 0; i < dst_layout->InputDim(); i++) {
    Var var = Var(std::string{char('i' + i)});
    dst_vars.push_back(IterVar(Range(0, dst_layout->InputShape()[i]), var,
                               IterVarType::kDataPar));
  }
  Array<IterVar> src_vars = dst_vars;
  src_vars.insert(src_vars.begin() + this->dim,
                  {Range(0, src_layout->InputShape()[this->dim]), Var("rv"),
                   IterVarType::kDataPar});
  Array<PrimExpr> src_indices = src_layout->Forward(
      src_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
  Array<PrimExpr> dst_indices = dst_layout->Forward(
      dst_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));

  Array<Stmt> stmts;

  bool require_init = this->clear;
  // sum op must be cleared
  if (this->type == ReduceType::kSum) {
    require_init = true;
  } else if (this->type == ReduceType::kAbsSum) {
    require_init = true;
  }

  Buffer clear_buffer = dst_buffer;
  bool need_duplicate = false;
  if (this->type == ReduceType::kSum && !this->clear) {
    need_duplicate = true;
  } else if (this->type == ReduceType::kAbsSum && !this->clear) {
    need_duplicate = true;
  }

  if (need_duplicate) {
    // Create a new buffer with same shape and dtype as dst_buffer
    clear_buffer = decl_buffer(dst_buffer->shape, dst_buffer->dtype,
                               dst_buffer->name + "_clear",
                               GetPtrStorageScope(dst_buffer->data));
  }

  // make reduce-init stmt
  if (require_init)
    stmts.push_back(
        BufferStore(clear_buffer, this->MakeInitValue(), dst_indices));

  // make thread-local reduce
  Array<PrimExpr> src_indice_compressed;
  Array<IterVar> src_var_compressed;
  for (size_t i = 0; i < src_layout->OutputDim(); i++) {
    PrimExpr expr;
    IterVar var;
    std::tie(expr, var) = CompressIterator(src_indices[i], src_vars,
                                           src_vars[this->dim]->var, analyzer);
    src_indice_compressed.push_back(expr);
    src_var_compressed.push_back(var);
  }
  Stmt reduce_local = BufferStore(
      clear_buffer,
      this->MakeReduce(BufferLoad(clear_buffer, dst_indices),
                       BufferLoad(src_buffer, src_indice_compressed)),
      dst_indices);
  for (int i = src_layout->OutputDim() - 1; i >= 0; i--) {
    reduce_local =
        For(src_var_compressed[i]->var, 0, src_var_compressed[i]->dom->extent,
            ForKind::kUnrolled, reduce_local, NullOpt,
            {{tir::attr::pragma_unroll_explicit, Bool(false)}});
  }
  stmts.push_back(reduce_local);

  // make inter-thread reduce
  PrimExpr src_thread = src_layout->ForwardThread(
      src_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }), {});
  auto iter_sum =
      arith::NormalizeToIterSum(src_thread, ToVMap(src_vars), analyzer);
  for (const auto &iter_split : iter_sum->args) {
    auto mark = iter_split->source->source.as<Var>();
    ICHECK(mark.defined());
    if (mark.value().same_as(src_vars[this->dim]->var)) {
      auto scale = as_const_int(iter_split->scale);
      auto extent = as_const_int(iter_split->extent);
      ICHECK(scale != nullptr && extent != nullptr);
      if (*extent == 1)
        continue;
      int reducing_threads = (*extent) * (*scale);
      std::stringstream ss;

      bool has_arch = T.target->attrs.count("arch") > 0;
      if (has_arch && Downcast<String>(T.target->attrs["arch"]) == "sm_90") {
        ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", "
           << reducing_threads << ", " << (*scale) << ">::run_hopper";
      } else {
        ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", "
           << reducing_threads << ", " << (*scale) << ">::run";
      }
      Array<PrimExpr> thread_reduce_args = {
          StringImm(ss.str()), BufferLoad(clear_buffer, dst_indices)};
      if (reducing_threads >= 32) {
        PrimExpr workspace = T.AddWorkspace(
            *as_const_int(T.thread_bounds->extent), clear_buffer->dtype);
        thread_reduce_args.push_back(workspace);
      }
      auto call =
          Call(clear_buffer->dtype, builtin::call_extern(), thread_reduce_args);
      stmts.push_back(BufferStore(clear_buffer, call, dst_indices));
    }
  }
  Stmt reduce_interthread = BufferStore(
      clear_buffer, BufferLoad(clear_buffer, dst_indices), dst_indices);

  // copy clear_buffer to dst_buffer
  if (need_duplicate) {
    // if is reduce sum, we should add a copy from clear_buffer to dst_buffer
    if (this->type == ReduceType::kSum) {
      stmts.push_back(BufferStore(dst_buffer,
                                  Add(BufferLoad(dst_buffer, dst_indices),
                                      BufferLoad(clear_buffer, dst_indices)),
                                  dst_indices));
    } else if (this->type == ReduceType::kAbsSum) {
      stmts.push_back(BufferStore(dst_buffer,
                                  Add(BufferLoad(dst_buffer, dst_indices),
                                      BufferLoad(clear_buffer, dst_indices)),
                                  dst_indices));
    } else {
      ICHECK(false) << "Unsupported reduce type: " << (int)this->type;
    }
  }
  // make the outer spatial loop
  Stmt body = stmts.size() > 1 ? SeqStmt(stmts) : stmts[0];
  for (int i = dst_layout->InputDim() - 1; i >= 0; i--) {
    body = For(dst_vars[i]->var, 0, dst_vars[i]->dom->extent,
               ForKind::kParallel, body);
  }

  body = PartitionLoop(Downcast<For>(body), T.thread_var, analyzer, dst_layout);
  if (need_duplicate) {
    body = Allocate(clear_buffer->data, clear_buffer->dtype,
                    clear_buffer->shape, const_true(), body);
  }
  return body;
}

LayoutMap ReduceOp::InferLayout(const LayoutInferArgs &T, InferLevel level) {
  if (level >= InferLevel::kStrict)
    return {};
  if (src.scope() == "local.fragment" && dst.scope() == "local.fragment" &&
      T.layout_map.count(src) && !T.layout_map.count(dst)) {
    auto src_layout = T.layout_map[src].as<Fragment>().value();

    PrimExpr indice_rep_extent = src->shape[dim];
    PrimExpr src_rep_extent = src_layout->ReplicateExtent();
    PrimExpr dest_buffer_rep_extent = indice_rep_extent * src_rep_extent;

    Array<PrimExpr> fwd;
    for (int i = 0; i < static_cast<int>(src->shape.size()); i++) {
      if (i == dim) {
        fwd.push_back(FloorMod(ReplicationPlaceholder(), indice_rep_extent));
      } else if (i < dim) {
        fwd.push_back(InputPlaceholder(i));
      } else if (i > dim) {
        fwd.push_back(InputPlaceholder(i - 1));
      }
    }
    auto thd = src_layout->ForwardThread(
        fwd, FloorDiv(ReplicationPlaceholder(), indice_rep_extent));
    Fragment dst_layout =
        Fragment(dst->shape, {}, thd, dest_buffer_rep_extent, NullOpt)
            ->CondenseReplicateVar();
    return {{dst, dst_layout}};
  }
  return {};
}

TIR_REGISTER_TL_OP(ReduceOp, reduce)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

CumSumOp::CumSumOp(Array<PrimExpr> args, BufferMap vmap) {
  /*
    CumSum arguments:
      src: input buffer
      dst: output buffer
      dim: dimension to cumsum
      reverse: whether to cumsum in reverse order
   */
  CHECK_EQ(args.size(), 4);
  src = vmap[GetVarFromAccessPtr(args[0])];
  dst = vmap[GetVarFromAccessPtr(args[1])];
  dim = args[2].as<IntImm>().value()->value;
  reverse = args[3].as<Bool>().value();
  CHECK_LT(dim, static_cast<int>(src->shape.size()));
}

Stmt CumSumOp::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (this->src.scope() == "local.fragment" &&
      this->dst.scope() == "local.fragment") {
    LOG(FATAL) << "CumSum for fragment not implemented, please raise an issue "
                  "if you need this feature.";
  } else if (this->src.scope() == "shared.dyn" ||
             this->src.scope() == "shared") {
    ICHECK(this->dst.scope() == "shared.dyn" || this->dst.scope() == "shared");
    std::stringstream ss;
    auto threads = T.thread_bounds->extent;
    ss << "tl::CumSum2D<" << threads << ", " << dim << ", "
       << (reverse ? "true" : "false") << ">::run";
    Array<PrimExpr> args = {StringImm(ss.str()), src.access_ptr(1),
                            dst.access_ptr(3)};
    for (int i = 0; i < src->shape.size(); i++) {
      args.push_back(src->shape[i]);
    }
    return Evaluate(Call(dst->dtype, builtin::call_extern(), args));
  } else {
    ICHECK(false) << "Cannot lower cumsum for " << this->src.scope() << " and "
                  << this->dst.scope();
  }

  return Stmt();
}

LayoutMap CumSumOp::InferLayout(const LayoutInferArgs &T, InferLevel level) {
  return {};
}

TIR_REGISTER_TL_OP(CumSumOp, cumsum)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
} // namespace tl
} // namespace tvm