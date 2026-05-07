/*!
 * \file tl/backend/common/op/reduce.h
 * \brief Shared tl.reduce AllReduce lowering for GPU backends.
 */

#ifndef TVM_TL_BACKEND_COMMON_OP_REDUCE_H_
#define TVM_TL_BACKEND_COMMON_OP_REDUCE_H_

#include "op/reduce.h"

#include "layout/layout.h"
#include "layout/utils.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "tir/transforms/ir_utils.h"
#include "transform/loop_partition.h"

#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace tvm {
namespace tl {
namespace backend {

using namespace tir;

namespace reduce {

inline Array<PrimExpr> InputPlaceholders(size_t n) {
  Array<PrimExpr> result;
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    result.push_back(InputPlaceholder(i));
  }
  return result;
}

inline Fragment ComputeReducerLayout(const Fragment &src_layout, int dim) {
  PrimExpr src_rep_extent = src_layout->ReplicateExtent();
  PrimExpr indice_rep_extent = src_layout->InputShape()[dim];
  PrimExpr reducer_rep_extent = indice_rep_extent * src_rep_extent;

  auto fwd = InputPlaceholders(src_layout->InputDim() - 1);
  fwd.insert(fwd.begin() + dim,
             FloorMod(ReplicationPlaceholder(), indice_rep_extent));

  auto thd = src_layout->ForwardThread(
      fwd, FloorDiv(ReplicationPlaceholder(), indice_rep_extent));

  auto reducer_shape = src_layout->InputShape();
  reducer_shape.erase(reducer_shape.begin() + dim);
  if (reducer_shape.empty()) {
    reducer_shape.push_back(1);
  }

  return Fragment(reducer_shape, {}, thd, reducer_rep_extent, std::nullopt)
      ->CondenseReplicateVar()
      ->BindThreadRange(src_layout->ThreadRange());
}

inline int64_t SignedMin(int bits) {
  if (bits >= 64) {
    return std::numeric_limits<int64_t>::min();
  }
  return -(static_cast<int64_t>(1) << (bits - 1));
}

inline int64_t SignedMax(int bits) {
  if (bits >= 64) {
    return std::numeric_limits<int64_t>::max();
  }
  return (static_cast<int64_t>(1) << (bits - 1)) - 1;
}

inline uint64_t UnsignedMax(int bits) {
  if (bits >= 64) {
    return std::numeric_limits<uint64_t>::max();
  }
  return (static_cast<uint64_t>(1) << bits) - 1;
}

inline PrimExpr MakeInitValue(const ReduceOpNode &op) {
  auto dst_dtype = op.dst->dtype;
  auto is_int = dst_dtype.is_int();
  bool is_uint = dst_dtype.is_uint();
  auto bits = dst_dtype.bits();

  if (op.type->isSum() || op.type->isAbsSum()) {
    return make_zero(op.dst->dtype);
  } else if (op.type->isMax()) {
    if (is_int) {
      return make_const(op.dst->dtype, SignedMin(bits));
    } else if (is_uint) {
      return make_const(op.dst->dtype, 0);
    } else {
      return make_const(op.dst->dtype, -INFINITY);
    }
  } else if (op.type->isMin()) {
    if (is_int) {
      return make_const(op.dst->dtype, SignedMax(bits));
    } else if (is_uint) {
      return make_const(op.dst->dtype, UnsignedMax(bits));
    } else {
      return make_const(op.dst->dtype, INFINITY);
    }
  } else if (op.type->isAbsMax()) {
    return make_const(op.dst->dtype, 0);
  } else if (op.type->isBitAnd()) {
    if (is_int) {
      return make_const(op.dst->dtype, -1);
    } else if (is_uint) {
      return make_const(op.dst->dtype, UnsignedMax(bits));
    } else {
      return make_const(op.dst->dtype, -INFINITY);
    }
  } else if (op.type->isBitOr() || op.type->isBitXor()) {
    return make_zero(op.dst->dtype);
  }
  LOG(FATAL) << "Unsupported reduce type: " << op.type->type;
  return PrimExpr();
}

inline PrimExpr MakeReduce(const ReduceOpNode &op, const PrimExpr &acc,
                           const PrimExpr &b) {
  PrimExpr rhs = b;
  if (acc->dtype != rhs->dtype) {
    rhs = Cast(acc->dtype, rhs);
  }
  const bool use_nan_op = op.nan_propagate && (acc.dtype().is_float16() ||
                                               acc.dtype().is_bfloat16());
  if (op.type->isSum()) {
    return acc + rhs;
  } else if (op.type->isAbsSum()) {
    return acc + Max(rhs, -rhs);
  } else if (op.type->isMax()) {
    if (use_nan_op) {
      return Call(acc.dtype(), tl::max_nan(), {acc, rhs});
    }
    return Max(acc, rhs);
  } else if (op.type->isMin()) {
    if (use_nan_op) {
      return Call(acc.dtype(), tl::min_nan(), {acc, rhs});
    }
    return Min(acc, rhs);
  } else if (op.type->isAbsMax()) {
    if (use_nan_op) {
      return Call(acc.dtype(), tl::max_nan(), {acc, tvm::abs(rhs)});
    }
    return Max(acc, tvm::abs(rhs));
  } else if (op.type->isBitAnd()) {
    return acc & rhs;
  } else if (op.type->isBitOr()) {
    return acc | rhs;
  } else if (op.type->isBitXor()) {
    return acc ^ rhs;
  }
  LOG(FATAL) << "Unsupported reduce type: " << op.type->type;
  return PrimExpr();
}

inline std::string MakeCodegenReducer(const ReduceOpNode &op) {
  const bool use_nan_op = op.nan_propagate && (op.dst->dtype.is_float16() ||
                                               op.dst->dtype.is_bfloat16());
  if (op.type->isSum() || op.type->isAbsSum()) {
    return "tl::SumOp";
  } else if (op.type->isMax()) {
    return use_nan_op ? "tl::MaxOpNan" : "tl::MaxOp";
  } else if (op.type->isMin()) {
    return use_nan_op ? "tl::MinOpNan" : "tl::MinOp";
  } else if (op.type->isAbsMax()) {
    return use_nan_op ? "tl::MaxOpNan" : "tl::MaxOp";
  } else if (op.type->isBitAnd()) {
    return "tl::BitAndOp";
  } else if (op.type->isBitOr()) {
    return "tl::BitOrOp";
  } else if (op.type->isBitXor()) {
    return "tl::BitXorOp";
  }
  LOG(FATAL) << "Unsupported reduce type: " << op.type->type;
  return "";
}

inline PrimExpr MakeUpdate(const ReduceOpNode &op, PrimExpr dst_val,
                           PrimExpr src_val) {
  if (op.type->isSum() || op.type->isAbsSum()) {
    return dst_val + src_val;
  } else if (op.type->isBitAnd()) {
    return op.clear ? src_val : bitwise_and(dst_val, src_val);
  } else if (op.type->isBitOr()) {
    return bitwise_or(dst_val, src_val);
  } else if (op.type->isBitXor()) {
    return bitwise_xor(dst_val, src_val);
  } else if (op.type->isMax() || op.type->isAbsMax()) {
    return Max(dst_val, src_val);
  } else if (op.type->isMin()) {
    return Min(dst_val, src_val);
  }
  LOG(FATAL) << "Unsupported reduce type: " << op.type->type;
  return PrimExpr();
}

} // namespace reduce

template <typename Impl> struct ReduceLowerer {
  static Stmt Lower(const ReduceOpNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    if (op.nan_propagate &&
        (op.dst->dtype.is_float16() || op.dst->dtype.is_bfloat16()) &&
        !Impl::SupportsFp16Bf16NanReduce(T.target)) {
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

    if (src_scope == "local.fragment" && dst_scope == "local.fragment") {
      auto src_buffer = get_buffer(op.src);
      auto dst_buffer = get_buffer(op.dst);
      auto src_layout = T.layout_map[op.src].as<Fragment>().value();
      auto dst_layout = T.layout_map[op.dst].as<Fragment>().value();
      auto red_layout = reduce::ComputeReducerLayout(src_layout, op.dim);
      auto src_dim = src_layout->InputDim();
      auto dst_dim = dst_layout->InputDim();

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
      auto red_indices = red_layout->Forward(
          dst_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));

      Array<Stmt> stmts;

      auto require_init = op.clear;
      if (op.type->isSum() || op.type->isAbsSum() || op.type->isBitAnd() ||
          op.type->isBitOr() || op.type->isBitXor()) {
        require_init = true;
      }

      auto clear_buffer = dst_buffer;
      auto need_duplicate = false;
      auto need_update = false;
      if ((op.type->isSum() || op.type->isAbsSum()) && !op.clear) {
        need_duplicate = true;
        need_update = true;
      } else if (op.type->isBitAnd() && !op.clear) {
        need_duplicate = true;
        need_update = true;
      } else if ((op.type->isBitOr() || op.type->isBitXor()) && !op.clear) {
        need_duplicate = true;
        need_update = true;
      } else if ((op.type->isMax() || op.type->isMin() ||
                  op.type->isAbsMax()) &&
                 !op.clear) {
        need_duplicate = true;
        need_update = true;
      }

      if (!analyzer->CanProve(dst_layout->ReplicateExtent() ==
                              red_layout->ReplicateExtent())) {
        need_duplicate = true;
      }
      ICHECK(!analyzer->CanProve(dst_layout->ReplicateExtent() >
                                 red_layout->ReplicateExtent()))
          << "Inconsistent layouts between src and dst in ReduceOp: "
          << "dst_layout=" << dst_layout << "red_layout=" << red_layout;

      if (need_duplicate) {
        clear_buffer = decl_buffer(red_layout->OutputShape(), dst_buffer->dtype,
                                   dst_buffer->name + "_clear",
                                   GetPtrStorageScope(dst_buffer->data));
      }
      if (require_init ||
          (need_duplicate &&
           (op.type->isMax() || op.type->isMin() || op.type->isAbsMax()))) {
        stmts.push_back(
            BufferStore(clear_buffer, reduce::MakeInitValue(op), red_indices));
      }

      Array<PrimExpr> src_indice_compressed;
      Array<IterVar> src_var_compressed;
      for (size_t i = 0; i < src_layout->OutputDim(); ++i) {
        auto [expr, var] = CompressIterator(src_indices[i], src_vars,
                                            src_vars[op.dim]->var, analyzer);
        src_indice_compressed.push_back(expr);
        src_var_compressed.push_back(var);
      }

      Stmt reduce_local = BufferStore(
          clear_buffer,
          reduce::MakeReduce(op, BufferLoad(clear_buffer, red_indices),
                             BufferLoad(src_buffer, src_indice_compressed)),
          red_indices);

      for (int i = static_cast<int>(src_layout->OutputDim()) - 1; i >= 0; --i) {
        reduce_local = For(src_var_compressed[i]->var, 0,
                           src_var_compressed[i]->dom->extent,
                           ForKind::kUnrolled, reduce_local, std::nullopt,
                           {{tir::attr::pragma_unroll_explicit, Bool(false)}});
      }
      stmts.push_back(reduce_local);

      auto src_thread = src_layout->ForwardThread(
          src_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }), {});
      auto iter_sum =
          arith::NormalizeToIterSum(src_thread, ToVMap(src_vars), analyzer);

      const int batch = op.batch;
      if (batch > 1) {
        int64_t N_total = 1;
        for (const auto &s : clear_buffer->shape) {
          const int64_t *p = as_const_int(s);
          ICHECK(p != nullptr) << "ReduceOp: batch > 1 requires compile-time "
                                  "constant output shape";
          N_total *= *p;
        }
        CHECK_LE(batch, N_total)
            << "ReduceOp: batch=" << batch
            << " exceeds per-thread output element count N=" << N_total;
        CHECK_EQ(N_total % batch, 0) << "ReduceOp: batch=" << batch
                                     << " must evenly divide N=" << N_total;
      }

      bool use_batch = batch > 1;

      auto make_dst_loop = [&](Stmt body, const Array<IterVar> &vars) -> Stmt {
        for (int i = static_cast<int>(vars.size()) - 1; i >= 0; --i) {
          body = For(vars[i]->var, 0, vars[i]->dom->extent, ForKind::kParallel,
                     body);
        }
        body = PartitionLoop(Downcast<For>(body), T.thread_var, analyzer,
                             red_layout);
        body = PragmaUnrollLoop(Downcast<For>(body));
        return body;
      };

      auto make_fresh_dst_vars = [&](const std::string &suffix)
          -> std::tuple<Array<IterVar>, Array<PrimExpr>, Array<PrimExpr>> {
        Array<IterVar> vars;
        for (size_t i = 0; i < dst_dim; ++i) {
          Var v(std::string{char('i' + i)} + suffix);
          vars.push_back(IterVar(Range(0, dst_layout->InputShape()[i]), v,
                                 IterVarType::kDataPar));
        }
        auto d_idx = dst_layout->Forward(
            vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
        auto r_idx = red_layout->Forward(
            vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
        return {vars, d_idx, r_idx};
      };

      if (use_batch) {
        Stmt pre_body = stmts.size() > 1 ? SeqStmt(stmts) : stmts[0];
        pre_body = make_dst_loop(pre_body, dst_vars);

        Array<Stmt> phases;
        phases.push_back(pre_body);

        for (const auto &iter_split : iter_sum->args) {
          auto mark = iter_split->source->source.template as<Var>();
          if (!mark) {
            continue;
          }
          if (!mark.value().same_as(src_vars[op.dim]->var)) {
            continue;
          }
          auto scale = as_const_int(iter_split->scale);
          auto extent = as_const_int(iter_split->extent);
          ICHECK(scale != nullptr && extent != nullptr);
          if (*extent == 1) {
            continue;
          }

          int reducing_threads = (*extent) * (*scale);
          auto thread_offset = T.thread_bounds->min;

          std::string allreduce = Impl::MakeBatchAllReduce(
              reduce::MakeCodegenReducer(op), reducing_threads, *scale,
              thread_offset, T.thread_bounds->extent, batch, reducing_threads,
              T.target);

          PrimExpr workspace;
          bool need_workspace = reducing_threads > 32;
          if (need_workspace) {
            int ws_size = reducing_threads * batch;
            workspace = T.AddWorkspace(ws_size, clear_buffer->dtype);
          }

          int64_t N_total = 1;
          for (const auto &s : clear_buffer->shape) {
            N_total *= *as_const_int(s);
          }
          int num_chunks = static_cast<int>(N_total / batch);

          int buf_ndim = static_cast<int>(clear_buffer->shape.size());
          std::vector<int64_t> buf_shape_vals;
          for (const auto &s : clear_buffer->shape) {
            buf_shape_vals.push_back(*as_const_int(s));
          }
          std::vector<int64_t> buf_strides(buf_ndim, 1);
          for (int d = buf_ndim - 2; d >= 0; d--) {
            buf_strides[d] = buf_strides[d + 1] * buf_shape_vals[d + 1];
          }

          for (int chunk = 0; chunk < num_chunks; chunk++) {
            int64_t flat_offset = static_cast<int64_t>(chunk) * batch;
            Array<PrimExpr> chunk_indices;
            for (int d = 0; d < buf_ndim; d++) {
              int64_t idx = (flat_offset / buf_strides[d]) % buf_shape_vals[d];
              chunk_indices.push_back(Integer(idx));
            }
            PrimExpr ptr = Call(DataType::Handle(), builtin::address_of(),
                                {BufferLoad(clear_buffer, chunk_indices)});

            Array<PrimExpr> args = {StringImm(allreduce), ptr};
            if (need_workspace) {
              args.push_back(workspace);
            }
            phases.push_back(Evaluate(
                Call(DataType::Handle(), builtin::call_extern(), args)));
          }
        }

        if (need_duplicate) {
          auto [post_vars, post_dst_idx, post_red_idx] =
              make_fresh_dst_vars("_p");

          PrimExpr predicate = Bool(true);
          {
            auto dst_th = post_dst_idx;
            dst_th.push_back(T.thread_var);
            auto inv = dst_layout->Inverse()->Forward(dst_th);
            inv.pop_back();
            for (int i = 0; i < static_cast<int>(dst_layout->InputDim()); i++) {
              predicate = predicate && (inv[i] == post_vars[i]->var);
            }
            predicate = analyzer->Simplify(predicate);
          }

          PrimExpr update =
              need_update
                  ? reduce::MakeUpdate(op, BufferLoad(dst_buffer, post_dst_idx),
                                       BufferLoad(clear_buffer, post_red_idx))
                  : BufferLoad(clear_buffer, post_red_idx);
          auto store = BufferStore(dst_buffer, update, post_dst_idx);
          Stmt post_body;
          if (analyzer->CanProve(predicate)) {
            post_body = store;
          } else {
            post_body = IfThenElse(predicate, store);
          }
          phases.push_back(make_dst_loop(post_body, post_vars));
        }

        Stmt body = phases.size() > 1 ? SeqStmt(phases) : phases[0];
        if (need_duplicate) {
          body = Allocate(clear_buffer->data, clear_buffer->dtype,
                          clear_buffer->shape, const_true(), body);
        }
        return body;
      }

      for (const auto &iter_split : iter_sum->args) {
        auto mark = iter_split->source->source.template as<Var>();
        if (!mark) {
          continue;
        }
        if (mark.value().same_as(src_vars[op.dim]->var)) {
          auto scale = as_const_int(iter_split->scale);
          auto extent = as_const_int(iter_split->extent);
          ICHECK(scale != nullptr && extent != nullptr);
          if (*extent == 1) {
            continue;
          }

          int reducing_threads = (*extent) * (*scale);
          auto thread_offset = T.thread_bounds->min;
          std::string allreduce = Impl::MakeScalarAllReduce(
              reduce::MakeCodegenReducer(op), reducing_threads, *scale,
              thread_offset, T.thread_bounds->extent, T.target);
          Array<PrimExpr> thread_reduce_args = {
              StringImm(allreduce), BufferLoad(clear_buffer, red_indices)};
          if (reducing_threads > 32) {
            int workspace_size =
                static_cast<int>(*as_const_int(T.thread_bounds->extent));
            PrimExpr workspace =
                T.AddWorkspace(workspace_size, clear_buffer->dtype);
            thread_reduce_args.push_back(workspace);
          }
          auto call = Call(clear_buffer->dtype, builtin::call_extern(),
                           thread_reduce_args);
          stmts.push_back(BufferStore(clear_buffer, call, red_indices));
        }
      }

      PrimExpr predicate = Bool(true);
      {
        auto dst_th_indices = dst_indices;
        dst_th_indices.push_back(T.thread_var);
        auto inv = dst_layout->Inverse()->Forward(dst_th_indices);
        inv.pop_back();
        for (int i = 0; i < static_cast<int>(dst_layout->InputDim()); i++) {
          predicate = predicate && (inv[i] == dst_vars[i]->var);
        }
        predicate = analyzer->Simplify(predicate);
      }
      if (need_duplicate) {
        PrimExpr update =
            need_update
                ? reduce::MakeUpdate(op, BufferLoad(dst_buffer, dst_indices),
                                     BufferLoad(clear_buffer, red_indices))
                : BufferLoad(clear_buffer, red_indices);
        auto store = BufferStore(dst_buffer, update, dst_indices);
        if (analyzer->CanProve(predicate)) {
          stmts.push_back(store);
        } else {
          stmts.push_back(IfThenElse(predicate, store));
        }
      }

      auto body = stmts.size() > 1 ? SeqStmt(stmts) : stmts[0];
      for (int i = static_cast<int>(dst_layout->InputDim()) - 1; i >= 0; --i) {
        body = For(dst_vars[i]->var, 0, dst_vars[i]->dom->extent,
                   ForKind::kParallel, body);
      }

      if (dst_layout->InputDim() > 0) {
        body = PartitionLoop(Downcast<For>(body), T.thread_var, analyzer,
                             red_layout);
        body = PragmaUnrollLoop(Downcast<For>(body));
      } else {
        auto guard = (T.thread_var == T.thread_bounds->min);
        body = IfThenElse(guard, body);
      }

      if (need_duplicate) {
        body = Allocate(clear_buffer->data, clear_buffer->dtype,
                        clear_buffer->shape, const_true(), body);
      }
      return body;
    }

    LOG(FATAL) << "Reduce for buffers in scope (" << src_scope << ", "
               << dst_scope << ") is not implemented.";
    return Stmt();
  }
};

} // namespace backend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_OP_REDUCE_H_
