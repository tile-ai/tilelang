/*!
 * \file tl/backend/metal/op/reduce.cc
 * \brief Metal implementation for tl.reduce replicated-fragment lowering.
 */

#include "backend/common/op/reduce.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace metal {

struct Reduce {
  static Stmt Lower(const ReduceOpNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    if (op.nan_propagate &&
        (op.dst->dtype.is_float16() || op.dst->dtype.is_bfloat16())) {
      LOG(FATAL) << "ReduceOp: nan_propagate=True for fp16/bf16 "
                    "max/min/absmax is only supported on CUDA targets "
                    "(requires __hmax_nan/__hmin_nan intrinsics). Target was: "
                 << T.target->str();
    }

    auto get_buffer = [&](const Buffer &buf) {
      if (T.buffer_remap.count(buf)) {
        return T.buffer_remap[buf];
      }
      return buf;
    };

    auto src_scope = op.src.scope();
    auto dst_scope = op.dst.scope();
    if (src_scope != "local.fragment" || dst_scope != "local.fragment") {
      LOG(FATAL) << "Reduce for buffers in scope (" << src_scope << ", "
                 << dst_scope << ") is not implemented.";
    }

    auto src_buffer = get_buffer(op.src);
    auto dst_buffer = get_buffer(op.dst);
    auto src_layout = T.layout_map[op.src].as<Fragment>().value();
    auto dst_layout = T.layout_map[op.dst].as<Fragment>().value();
    auto src_dim = src_layout->InputDim();
    auto dst_dim = dst_layout->InputDim();

    bool can_use_replicated_path = src_layout->IsCompletedReplicated() &&
                                   dst_layout->IsCompletedReplicated() &&
                                   !src_buffer->data.same_as(dst_buffer->data);
    if (!can_use_replicated_path) {
      LOG(FATAL) << "Metal ReduceOp requires completed replicated fragment "
                    "layouts for distinct source and destination buffers.";
    }

    auto is_1d_reduce = src_dim == dst_dim && dst_dim == 1;
    if (is_1d_reduce) {
      ICHECK(is_one(dst_layout->OutputShape().back()))
          << "Reduce for scalar not implemented.";
    } else {
      ICHECK_EQ(src_dim, dst_dim + 1) << "Reduce dimension mismatch.";
    }

    Array<IterVar> dst_vars;
    for (size_t i = 0; i < dst_dim; ++i) {
      Var var = Var(std::string{char('i' + i)});
      dst_vars.push_back(IterVar(Range(0, dst_layout->InputShape()[i]), var,
                                 IterVarType::kDataPar));
    }

    Array<IterVar> src_vars;
    if (!is_1d_reduce) {
      src_vars = dst_vars;
    }
    Range reduce_dom(0, src_layout->InputShape()[op.dim]);
    IterVar reduce_iv(reduce_dom, Var("rv"), IterVarType::kDataPar);
    src_vars.insert(src_vars.begin() + op.dim, reduce_iv);

    auto src_indices = src_layout->Forward(
        src_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
    auto dst_indices = dst_layout->Forward(
        dst_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));

    Array<PrimExpr> src_indice_compressed;
    Array<IterVar> src_var_compressed;
    for (size_t i = 0; i < src_layout->OutputDim(); ++i) {
      auto [expr, var] = CompressIterator(src_indices[i], src_vars,
                                          src_vars[op.dim]->var, analyzer);
      src_indice_compressed.push_back(expr);
      src_var_compressed.push_back(var);
    }

    PrimExpr init_value = op.clear ? backend::reduce::MakeInitValue(op)
                                   : BufferLoad(dst_buffer, dst_indices);
    Stmt init = BufferStore(dst_buffer, init_value, dst_indices);
    Stmt reduce_local =
        BufferStore(dst_buffer,
                    backend::reduce::MakeReduce(
                        op, 1, BufferLoad(dst_buffer, dst_indices),
                        BufferLoad(src_buffer, src_indice_compressed))
                        .value(),
                    dst_indices);
    for (int i = static_cast<int>(src_layout->OutputDim()) - 1; i >= 0; --i) {
      reduce_local =
          For(src_var_compressed[i]->var, 0, src_var_compressed[i]->dom->extent,
              ForKind::kSerial, reduce_local, std::nullopt,
              {{tirx::attr::pragma_unroll_explicit, Bool(false)}});
    }

    Stmt body = SeqStmt({init, reduce_local});
    for (int i = static_cast<int>(dst_vars.size()) - 1; i >= 0; --i) {
      body = For(dst_vars[i]->var, 0, dst_vars[i]->dom->extent,
                 ForKind::kSerial, body, std::nullopt,
                 {{tirx::attr::pragma_unroll_explicit, Bool(false)}});
    }
    return body;
  }
};

} // namespace metal

namespace {

bool MatchMetalReduceTarget(Target target) { return TargetIsMetal(target); }

bool RegisterMetalReduce() {
  RegisterReduceImpl(ReduceImpl{
      "metal.Reduce",
      MatchMetalReduceTarget,
      metal::Reduce::Lower,
  });
  return true;
}

const bool metal_reduce_registered = RegisterMetalReduce();

} // namespace

} // namespace tl
} // namespace tvm
