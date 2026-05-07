/*!
 * \file tl/backend/cuda/op/reduce.cc
 * \brief CUDA implementation for tl.reduce AllReduce lowering.
 */

#include "op/reduce.h"

#include "layout/layout.h"
#include "layout/utils.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "target/utils.h"
#include "tir/transforms/ir_utils.h"
#include "transform/loop_partition.h"

#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <tuple>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace cuda {

namespace {

Array<PrimExpr> InputPlaceholders(size_t n) {
  Array<PrimExpr> result;
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    result.push_back(InputPlaceholder(i));
  }
  return result;
}

Fragment ComputeReducerLayout(const Fragment &src_layout, int dim) {
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

int64_t SignedMin(int bits) {
  if (bits >= 64) {
    return std::numeric_limits<int64_t>::min();
  }
  return -(static_cast<int64_t>(1) << (bits - 1));
}

int64_t SignedMax(int bits) {
  if (bits >= 64) {
    return std::numeric_limits<int64_t>::max();
  }
  return (static_cast<int64_t>(1) << (bits - 1)) - 1;
}

uint64_t UnsignedMax(int bits) {
  if (bits >= 64) {
    return std::numeric_limits<uint64_t>::max();
  }
  return (static_cast<uint64_t>(1) << bits) - 1;
}

int GetPreferedVectorizedSize(DataType dt) {
  if (dt.is_bfloat16() || dt.is_float16())
    return 2;
  return 1;
}

PrimExpr MakeInitValue(const ReduceOpNode &op, int vsize = 1) {
  auto dst_dtype = op.dst->dtype;
  auto is_int = dst_dtype.is_int();
  bool is_uint = dst_dtype.is_uint();
  auto bits = dst_dtype.bits();

  PrimExpr scalar;
  if (op.type->isSum() || op.type->isAbsSum()) {
    scalar = make_zero(op.dst->dtype);
  } else if (op.type->isMax()) {
    if (is_int) {
      scalar = make_const(op.dst->dtype, SignedMin(bits));
    } else if (is_uint) {
      scalar = make_const(op.dst->dtype, 0);
    } else {
      scalar = make_const(op.dst->dtype, -INFINITY);
    }
  } else if (op.type->isMin()) {
    if (is_int) {
      scalar = make_const(op.dst->dtype, SignedMax(bits));
    } else if (is_uint) {
      scalar = make_const(op.dst->dtype, UnsignedMax(bits));
    } else {
      scalar = make_const(op.dst->dtype, INFINITY);
    }
  } else if (op.type->isAbsMax()) {
    scalar = make_const(op.dst->dtype, 0);
  } else if (op.type->isBitAnd()) {
    if (is_int) {
      scalar = make_const(op.dst->dtype, -1);
    } else if (is_uint) {
      scalar = make_const(op.dst->dtype, UnsignedMax(bits));
    } else {
      scalar = make_const(op.dst->dtype, -INFINITY);
    }
  } else if (op.type->isBitOr() || op.type->isBitXor()) {
    scalar = make_zero(op.dst->dtype);
  } else {
    LOG(FATAL) << "Unsupported reduce type: " << op.type->type;
    scalar = PrimExpr();
  }

  if (vsize <= 1)
    return scalar;
  return Broadcast(scalar, vsize);
}

std::optional<PrimExpr> MakeReduce(const ReduceOpNode &op, int vsize,
                                   const PrimExpr &acc, const PrimExpr &b) {
  if (vsize == 1) {
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
      return use_nan_op ? Call(acc.dtype(), tl::max_nan(), {acc, rhs})
                        : PrimExpr(Max(acc, rhs));
    } else if (op.type->isMin()) {
      return use_nan_op ? Call(acc.dtype(), tl::min_nan(), {acc, rhs})
                        : PrimExpr(Min(acc, rhs));
    } else if (op.type->isAbsMax()) {
      auto abs_rhs = Max(rhs, -rhs);
      return use_nan_op ? Call(acc.dtype(), tl::max_nan(), {acc, abs_rhs})
                        : PrimExpr(Max(acc, abs_rhs));
    } else if (op.type->isBitAnd()) {
      return acc & rhs;
    } else if (op.type->isBitOr()) {
      return acc | rhs;
    } else if (op.type->isBitXor()) {
      return acc ^ rhs;
    }
    LOG(FATAL) << "Unsupported reduce type: " << op.type->type;
    return std::nullopt;
  }

  if (vsize != 2)
    return std::nullopt;

  if (op.type->isSum()) {
    return Call(acc.dtype(), tl::add2(), {acc, b});
  } else if (op.type->isAbsSum()) {
    return Call(acc.dtype(), tl::add2(),
               {acc, Call(acc.dtype(), tl::abs2(), {b})});
  } else if (op.type->isMax()) {
    return Call(acc.dtype(), op.nan_propagate ? tl::max2_nan() : tl::max2(),
               {acc, b});
  } else if (op.type->isMin()) {
    return Call(acc.dtype(), op.nan_propagate ? tl::min2_nan() : tl::min2(),
               {acc, b});
  } else if (op.type->isAbsMax()) {
    return Call(acc.dtype(), op.nan_propagate ? tl::max2_nan() : tl::max2(),
               {acc, Call(acc.dtype(), tl::abs2(), {b})});
  }
  return std::nullopt;
}

std::optional<std::string> MakeCodegenReducer(const ReduceOpNode &op,
                                              int vsize = 1) {
  const bool use_nan_op = op.nan_propagate && (op.dst->dtype.is_float16() ||
                                               op.dst->dtype.is_bfloat16());

  auto base = [&]() -> std::string {
    if (op.type->isSum() || op.type->isAbsSum())
      return "tl::SumOp";
    if (op.type->isMax())
      return use_nan_op ? "tl::MaxOpNan" : "tl::MaxOp";
    if (op.type->isMin())
      return use_nan_op ? "tl::MinOpNan" : "tl::MinOp";
    if (op.type->isAbsMax())
      return use_nan_op ? "tl::MaxOpNan" : "tl::MaxOp";
    if (op.type->isBitAnd())
      return "tl::BitAndOp";
    if (op.type->isBitOr())
      return "tl::BitOrOp";
    if (op.type->isBitXor())
      return "tl::BitXorOp";
    LOG(FATAL) << "Unsupported reduce type: " << op.type->type;
    return "";
  }();

  if (vsize <= 1)
    return base;

  if (vsize == 2) {
    if (op.dst->dtype.is_bfloat16())
      return base + "_bf16x2";
    if (op.dst->dtype.is_float16())
      return base + "_fp16x2";
  }
  return std::nullopt;
}

PrimExpr MakeUpdate(const ReduceOpNode &op, PrimExpr dst_val,
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

} // namespace

struct Reduce {
  static bool SupportsFp16Bf16NanReduce(Target target) {
    return TargetIsCuda(target);
  }

  static Stmt Lower(const ReduceOpNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    if (op.nan_propagate &&
        (op.dst->dtype.is_float16() || op.dst->dtype.is_bfloat16()) &&
        !SupportsFp16Bf16NanReduce(T.target)) {
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
      auto red_layout = ComputeReducerLayout(src_layout, op.dim);
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

      // make thread-local reduce
      Array<PrimExpr> src_indice_compressed;
      Array<IterVar> src_var_compressed;
      for (size_t i = 0; i < src_layout->OutputDim(); ++i) {
        auto [expr, var] = CompressIterator(src_indices[i], src_vars,
                                            src_vars[op.dim]->var, analyzer);
        src_indice_compressed.push_back(expr);
        src_var_compressed.push_back(var);
      }

      bool can_pack = false;
      bool need_pack_buffer = false;
      bool need_batch_pack_buffer = false;
      Buffer clear_buffer_packed;
      Buffer clear_batch_pack_buffer;
      {
        int vsize = GetPreferedVectorizedSize(clear_buffer->dtype);
        if (vsize > 1 && !src_var_compressed.empty()) {
          auto *ext = src_var_compressed.back()->dom->extent.as<IntImmNode>();
          if (ext && ext->value >= vsize && ext->value % vsize == 0) {
            can_pack = true;
            DataType vec_dtype = clear_buffer->dtype.with_lanes(vsize);
            clear_buffer_packed =
                decl_buffer(red_layout->OutputShape(), vec_dtype,
                            clear_buffer->name + "_pack",
                            GetPtrStorageScope(clear_buffer->data));
            need_pack_buffer = true;

            Array<Stmt> local_body;

            if (require_init ||
                (need_duplicate && (op.type->isMax() || op.type->isMin() ||
                                    op.type->isAbsMax()))) {
              local_body.push_back(BufferStore(clear_buffer_packed,
                                              MakeInitValue(op, vsize),
                                              red_indices));
            }

            const auto *ext_int =
                as_const_int(src_var_compressed.back()->dom->extent);
            int64_t inner_extent = *ext_int;
            PrimExpr halved_extent = Integer(inner_extent / vsize);

            auto &inner_var = src_var_compressed.back();

            PrimExpr ramp_base =
                Substitute(src_indice_compressed.back(),
                           {{inner_var->var, inner_var->var * Integer(2)}});
            src_indice_compressed.Set(
                src_indice_compressed.size() - 1,
                Ramp(ramp_base, IntImm(DataType::Int(32), 1), vsize));

            auto src_load = BufferLoad(src_buffer, src_indice_compressed);
            auto *src_writer = src_load.CopyOnWrite();
            src_writer->dtype = vec_dtype;

            Stmt reduce_local = BufferStore(
                clear_buffer_packed,
                MakeReduce(op, vsize,
                           BufferLoad(clear_buffer_packed, red_indices),
                           src_load)
                    .value(),
                red_indices);

            reduce_local =
                For(inner_var->var, 0, halved_extent, ForKind::kUnrolled,
                    reduce_local, std::nullopt,
                    {{tir::attr::pragma_unroll_explicit, Bool(false)}});

            for (int i = static_cast<int>(src_layout->OutputDim()) - 2; i >= 0;
                 --i) {
              reduce_local =
                  For(src_var_compressed[i]->var, 0,
                      src_var_compressed[i]->dom->extent, ForKind::kUnrolled,
                      reduce_local, std::nullopt,
                      {{tir::attr::pragma_unroll_explicit, Bool(false)}});
            }
            local_body.push_back(reduce_local);

            auto acc_vec = BufferLoad(clear_buffer_packed, red_indices);
            auto lane0 = Shuffle::ExtractElement(acc_vec, 0);
            auto lane1 = Shuffle::ExtractElement(acc_vec, 1);
            auto scalar_result = MakeReduce(op, 1, lane0, lane1).value();
            local_body.push_back(
                BufferStore(clear_buffer, scalar_result, red_indices));

            stmts.push_back(SeqStmt(local_body));
          }
        }
      }

      if (!can_pack) {
        if (require_init ||
            (need_duplicate && (op.type->isMax() || op.type->isMin() ||
                                op.type->isAbsMax()))) {
          stmts.push_back(
              BufferStore(clear_buffer, MakeInitValue(op), red_indices));
        }

        Stmt reduce_local = BufferStore(
            clear_buffer,
            MakeReduce(op, 1, BufferLoad(clear_buffer, red_indices),
                       BufferLoad(src_buffer, src_indice_compressed))
                .value(),
            red_indices);

        for (int i = static_cast<int>(src_layout->OutputDim()) - 1; i >= 0;
             --i) {
          reduce_local = For(src_var_compressed[i]->var, 0,
                             src_var_compressed[i]->dom->extent,
                             ForKind::kUnrolled, reduce_local, std::nullopt,
                             {{tir::attr::pragma_unroll_explicit, Bool(false)}});
        }
        stmts.push_back(reduce_local);
      }

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
        // ================================================================
        // Batched AllReduce path — three phases:
        //   1. Loop: init + thread-local reduce
        //   2. Flat: batched AllReduce (single butterfly pass for all values)
        //   3. Loop: copy-back (only when need_duplicate)
        // ================================================================

        // Phase 1: pre-reduce loop
        Stmt pre_body = stmts.size() > 1 ? SeqStmt(stmts) : stmts[0];
        pre_body = make_dst_loop(pre_body, dst_vars);

        Array<Stmt> phases;
        phases.push_back(pre_body);

        // Phase 2: batched AllReduce call(s).
        for (const auto &iter_split : iter_sum->args) {
          auto mark = iter_split->source->source.as<Var>();
          if (!mark)
            continue;
          if (!mark.value().same_as(src_vars[op.dim]->var))
            continue;
          auto scale = as_const_int(iter_split->scale);
          auto extent = as_const_int(iter_split->extent);
          ICHECK(scale != nullptr && extent != nullptr);
          if (*extent == 1)
            continue;

          int reducing_threads = (*extent) * (*scale);
          auto thread_offset = T.thread_bounds->min;
          std::stringstream ss;

          int vsize = GetPreferedVectorizedSize(clear_buffer->dtype);
          bool can_batch_pack =
              vsize > 1 && batch >= vsize && batch % vsize == 0;
          int eff_batch = can_batch_pack ? (batch / vsize) : batch;

          std::string reducer =
              MakeCodegenReducer(op, can_batch_pack ? vsize : 1).value();

          if (TargetHasSMVersionGE(T.target, 90)) {
            auto all_threads = T.thread_bounds->extent;
            ss << "tl::AllReduce<" << reducer << ", " << reducing_threads
               << ", " << (*scale) << ", " << thread_offset
               << ", tl::NamedBarrier<" << all_threads << ">, " << eff_batch
               << ", " << reducing_threads << ">::run_batch";
          } else {
            ss << "tl::AllReduce<" << reducer << ", " << reducing_threads
               << ", " << (*scale) << ", " << thread_offset
               << ", tl::SyncThreadsBarrier, " << eff_batch << ", "
               << reducing_threads << ">::run_batch";
          }

          DataType ws_dtype = can_batch_pack
                                  ? clear_buffer->dtype.with_lanes(vsize)
                                  : clear_buffer->dtype;
          PrimExpr workspace;
          bool need_workspace = reducing_threads > 32;
          if (need_workspace) {
            int ws_size = reducing_threads * eff_batch;
            workspace = T.AddWorkspace(ws_size, ws_dtype);
          }

          int64_t N_total = 1;
          for (const auto &s : clear_buffer->shape)
            N_total *= *as_const_int(s);
          int num_chunks = static_cast<int>(N_total / batch);

          int buf_ndim = static_cast<int>(clear_buffer->shape.size());
          std::vector<int64_t> buf_shape_vals;
          for (const auto &s : clear_buffer->shape)
            buf_shape_vals.push_back(*as_const_int(s));
          std::vector<int64_t> buf_strides(buf_ndim, 1);
          for (int d = buf_ndim - 2; d >= 0; d--)
            buf_strides[d] = buf_strides[d + 1] * buf_shape_vals[d + 1];

          std::string template_str = ss.str();

          if (can_batch_pack) {
            int K = vsize;
            int packed_batch = batch / K;

            Buffer pack_buf =
                decl_buffer({Integer(packed_batch)},
                            clear_buffer->dtype.with_lanes(K),
                            clear_buffer->name + "_pack",
                            GetPtrStorageScope(clear_buffer->data));

            need_batch_pack_buffer = true;
            clear_batch_pack_buffer = pack_buf;

            for (int chunk = 0; chunk < num_chunks; chunk++) {
              int64_t flat_offset = (int64_t)chunk * batch;

              // --- Pack loop ---
              Var pack_j("pack_j");
              PrimExpr base = Integer(flat_offset);
              PrimExpr scaled = pack_j * K;

              Array<PrimExpr> idx_a, idx_b;
              PrimExpr fa = base + scaled;
              PrimExpr fb = base + scaled + Integer(1);
              for (int d = 0; d < buf_ndim; d++) {
                idx_a.push_back(FloorMod(FloorDiv(fa, Integer(buf_strides[d])),
                                         Integer(buf_shape_vals[d])));
                idx_b.push_back(FloorMod(FloorDiv(fb, Integer(buf_strides[d])),
                                         Integer(buf_shape_vals[d])));
              }
              auto a_load = BufferLoad(clear_buffer, idx_a);
              auto b_load = BufferLoad(clear_buffer, idx_b);
              Stmt pack_body = BufferStore(
                  pack_buf, Shuffle({a_load, b_load}, {0, 1}), {pack_j});
              Stmt pack_loop =
                  For(pack_j, 0, packed_batch, ForKind::kUnrolled, pack_body);
              phases.push_back(pack_loop);

              // --- AllReduce on packed buffer ---
              PrimExpr packed_ptr =
                  Call(DataType::Handle(), builtin::address_of(),
                       {BufferLoad(pack_buf, {Integer(0)})});
              Array<PrimExpr> args = {StringImm(template_str), packed_ptr};
              if (need_workspace)
                args.push_back(workspace);
              phases.push_back(Evaluate(
                  Call(DataType::Handle(), builtin::call_extern(), args)));

              // --- Unpack loop ---
              Var unpack_j("unpack_j");
              PrimExpr ubase = Integer(flat_offset);
              PrimExpr uscaled = unpack_j * K;
              Array<PrimExpr> uidx_a, uidx_b;
              PrimExpr ufa = ubase + uscaled;
              PrimExpr ufb = ubase + uscaled + Integer(1);
              for (int d = 0; d < buf_ndim; d++) {
                uidx_a.push_back(
                    FloorMod(FloorDiv(ufa, Integer(buf_strides[d])),
                             Integer(buf_shape_vals[d])));
                uidx_b.push_back(
                    FloorMod(FloorDiv(ufb, Integer(buf_strides[d])),
                             Integer(buf_shape_vals[d])));
              }
              auto packed_val = BufferLoad(pack_buf, {unpack_j});
              Stmt unpack_body = SeqStmt({
                  BufferStore(clear_buffer,
                              Shuffle::ExtractElement(packed_val, 0), uidx_a),
                  BufferStore(clear_buffer,
                              Shuffle::ExtractElement(packed_val, 1), uidx_b),
              });
              Stmt unpack_loop = For(unpack_j, 0, packed_batch,
                                     ForKind::kUnrolled, unpack_body);
              phases.push_back(unpack_loop);
            }
          } else {
            for (int chunk = 0; chunk < num_chunks; chunk++) {
              int64_t flat_offset = (int64_t)chunk * batch;
              Array<PrimExpr> chunk_indices;
              for (int d = 0; d < buf_ndim; d++) {
                int64_t idx =
                    (flat_offset / buf_strides[d]) % buf_shape_vals[d];
                chunk_indices.push_back(Integer(idx));
              }
              PrimExpr ptr = Call(DataType::Handle(), builtin::address_of(),
                                  {BufferLoad(clear_buffer, chunk_indices)});

              Array<PrimExpr> args = {StringImm(template_str), ptr};
              if (need_workspace)
                args.push_back(workspace);
              phases.push_back(Evaluate(
                  Call(DataType::Handle(), builtin::call_extern(), args)));
            }
          }
        }

        // Phase 3: copy-back (only when a temp buffer was used)
        if (need_duplicate) {
          auto [post_vars, post_dst_idx, post_red_idx] =
              make_fresh_dst_vars("_p");

          PrimExpr predicate = Bool(true);
          {
            auto dst_th = post_dst_idx;
            dst_th.push_back(T.thread_var);
            auto inv = dst_layout->Inverse()->Forward(dst_th);
            inv.pop_back();
            for (int i = 0; i < static_cast<int>(dst_layout->InputDim()); i++)
              predicate = predicate && (inv[i] == post_vars[i]->var);
            predicate = analyzer->Simplify(predicate);
          }

          PrimExpr update =
              need_update
                  ? MakeUpdate(op, BufferLoad(dst_buffer, post_dst_idx),
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
        if (need_pack_buffer) {
          body =
              Allocate(clear_buffer_packed->data, clear_buffer_packed->dtype,
                       clear_buffer_packed->shape, const_true(), body);
        }
        if (need_batch_pack_buffer) {
          body = Allocate(clear_batch_pack_buffer->data,
                          clear_batch_pack_buffer->dtype,
                          clear_batch_pack_buffer->shape, const_true(), body);
        }
        return body;

      } else {
        // ================================================================
        // Original scalar AllReduce path.
        // ================================================================
        for (const auto &iter_split : iter_sum->args) {
          auto mark = iter_split->source->source.as<Var>();
          if (!mark)
            continue;
          if (mark.value().same_as(src_vars[op.dim]->var)) {
            auto scale = as_const_int(iter_split->scale);
            auto extent = as_const_int(iter_split->extent);
            ICHECK(scale != nullptr && extent != nullptr);
            if (*extent == 1)
              continue;

            int reducing_threads = (*extent) * (*scale);
            auto thread_offset = T.thread_bounds->min;
            std::string allreduce = MakeScalarAllReduce(
                MakeCodegenReducer(op).value(), reducing_threads, *scale,
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
                  ? MakeUpdate(op, BufferLoad(dst_buffer, dst_indices),
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
        for (int i = static_cast<int>(dst_layout->InputDim()) - 1; i >= 0;
             --i) {
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
        if (need_pack_buffer) {
          body =
              Allocate(clear_buffer_packed->data, clear_buffer_packed->dtype,
                       clear_buffer_packed->shape, const_true(), body);
        }
        return body;
      }
    }

    LOG(FATAL) << "Reduce for buffers in scope (" << src_scope << ", "
               << dst_scope << ") is not implemented.";
    return Stmt();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset,
                                         PrimExpr all_threads, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (TargetHasSMVersionGE(target, 90)) {
      ss << ", tl::NamedBarrier<" << all_threads << ">";
    }
    ss << ">::run";
    return ss.str();
  }
};

} // namespace cuda

namespace {

bool MatchCudaReduceTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaReduce() {
  RegisterReduceImpl(ReduceImpl{
      "cuda.Reduce",
      MatchCudaReduceTarget,
      cuda::Reduce::Lower,
  });
  return true;
}

const bool cuda_reduce_registered = RegisterCudaReduce();

} // namespace

} // namespace tl
} // namespace tvm
