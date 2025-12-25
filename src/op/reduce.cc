/*!
 * \file tl/op/reduce.cc
 * \brief Implementation of reduction operators
 */

#include "reduce.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>

#include "../layout/layout.h"
#include "../layout/utils.h"
#include "../op/parallel.h"
#include "../target/utils.h"
#include "../transform/loop_partition.h"
#include "tir/transforms/ir_utils.h"
#include "tvm/tir/stmt.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

// NormalizeToBufferRegion moved to src/op/utils.{h,cc}

// MakeAccessPtrFromRegion moved to src/op/utils.{h,cc}

ReduceOp::ReduceOp(Array<PrimExpr> args) {
  ObjectPtr<ReduceOpNode> node = tvm::ffi::make_object<ReduceOpNode>();
  // Accept BufferRegion/BufferLoad for src/dst
  node->srcRegion_ = NormalizeToBufferRegion(args[0]);
  node->dstRegion_ = NormalizeToBufferRegion(args[1]);
  node->src = node->srcRegion_->buffer;
  node->dst = node->dstRegion_->buffer;
  std::string reduce_type = args[2].as<StringImm>().value()->value;
  node->dim = args[3].as<IntImm>().value()->value;
  node->type = ReduceType(reduce_type);
  node->clear = args[4].as<Bool>().value();
  data_ = std::move(node);
}

TileOperator ReduceOpNode::Clone() const {
  auto op = tvm::ffi::make_object<ReduceOpNode>(*this);
  return ReduceOp(op);
}

TileOperator CumSumOpNode::Clone() const {
  auto op = tvm::ffi::make_object<CumSumOpNode>(*this);
  return CumSumOp(op);
}

PrimExpr ReduceOpNode::MakeInitValue() const {
  auto dst_dtype = dst->dtype;
  auto is_int = dst_dtype.is_int();
  bool is_uint = dst_dtype.is_uint();
  auto bits = dst_dtype.bits();

  if (type->isSum()) {
    return make_zero(dst->dtype);
  } else if (type->isAbsSum()) {
    return make_zero(dst->dtype);
  } else if (type->isMax()) {
    if (is_int) {
      return make_const(dst->dtype, -(1 << (bits - 1)));
    } else if (is_uint) {
      return make_const(dst->dtype, 0);
    } else {
      return make_const(dst->dtype, -INFINITY);
    }
  } else if (type->isMin()) {
    if (is_int) {
      return make_const(dst->dtype, (1 << (bits - 1)) - 1);
    } else if (is_uint) {
      return make_const(dst->dtype, (1 << bits) - 1);
    } else {
      return make_const(dst->dtype, INFINITY);
    }
  } else if (type->isAbsMax()) {
    return make_const(dst->dtype, 0);
  } else if (type->isBitAnd()) {
    if (is_int) {
      return make_const(dst->dtype, -1);
    } else if (is_uint) {
      return make_const(dst->dtype, (1 << bits) - 1);
    } else {
      // Should not arrive here
      return make_const(dst->dtype, -INFINITY);
    }
  } else if (type->isBitOr()) {
    return make_zero(dst->dtype);
  } else if (type->isBitXor()) {
    return make_zero(dst->dtype);
  } else {
    LOG(FATAL) << "Unsupported reduce type: " << type->type;
    return PrimExpr();
  }
}

PrimExpr ReduceOpNode::MakeReduce(const PrimExpr &lhs,
                                  const PrimExpr &b) const {
  PrimExpr rhs = b;
  if (lhs->dtype != rhs->dtype) {
    rhs = Cast(lhs->dtype, rhs);
  }
  if (type->isSum()) {
    return lhs + rhs;
  } else if (type->isAbsSum()) {
    return lhs + Max(rhs, -rhs);
  } else if (type->isMax()) {
    return Max(lhs, rhs);
  } else if (type->isMin()) {
    return Min(lhs, rhs);
  } else if (type->isAbsMax()) {
    return Max(tvm::abs(lhs), tvm::abs(rhs));
  } else if (type->isBitAnd()) {
    return lhs & rhs;
  } else if (type->isBitOr()) {
    return lhs | rhs;
  } else if (type->isBitXor()) {
    return lhs ^ rhs;
  } else {
    LOG(FATAL) << "Unsupported reduce type: " << type->type;
  }
}

std::string ReduceOpNode::MakeCodegenReducer() const {
  if (type->isSum()) {
    return "tl::SumOp";
  } else if (type->isAbsSum()) {
    return "tl::SumOp";
  } else if (type->isMax()) {
    return "tl::MaxOp";
  } else if (type->isMin()) {
    return "tl::MinOp";
  } else if (type->isAbsMax()) {
    return "tl::MaxOp";
  } else if (type->isBitAnd()) {
    return "tl::BitAndOp";
  } else if (type->isBitOr()) {
    return "tl::BitOrOp";
  } else if (type->isBitXor()) {
    return "tl::BitXorOp";
  } else {
    LOG(FATAL) << "Unsupported reduce type: " << type->type;
    return "";
  }
}

/**
 * @brief Lower the Reduce operator to a TIR statement.
 *
 * Lowers a ReduceOpNode operating on fragment-scoped buffers into a sequence of
 * TIR statements implementing: optional initialization, thread-local reduction
 * (unrolled inner loops), inter-thread reduction via a runtime AllReduce call
 * (Hopper-specific `run_hopper` variant when TargetIsHopper(T.target) is true),
 * and an optional accumulation or copy back to the destination buffer when a
 * temporary clear buffer is used.
 *
 * Behavior notes:
 * - Only supports src and dst in "local.fragment" scope; otherwise it checks
 *   and aborts with "Reduce for shared memory not implemented.".
 * - Supports both 1D reductions (scalar output) and reductions along a single
 *   extra dimension; validates layout dimensionality consistency.
 * - If `clear` is set (or for sum/abssum reductions), an initial value is
 *   written to the clear buffer; for non-clearing sum/abssum a duplicate
 *   temporary buffer is allocated and accumulated back into dst after
 * reduction.
 * - Performs iterator compression for local reduction loops using `analyzer`.
 * - Detects parallel thread splitting from the normalized iterator sum and
 *   emits a call to a templated `tl::AllReduce<...>::run` (or `run_hopper`)
 *   via `builtin::call_extern`. For sufficiently large reducing thread counts
 *   (>= 32) a workspace is allocated via T.AddWorkspace and passed to the
 *   AllReduce call.
 * - The final body is wrapped in parallel loops over the destination spatial
 *   dimensions and partitioned by the lowering thread variable. If a temporary
 *   clear buffer is used, it is allocated for the body.
 *
 * @param T Lowering context providing buffer and layout maps, thread bounds,
 *          target information, thread variable, and workspace allocation
 * helper.
 * @param analyzer Analyzer used for iterator compression and arithmetic
 * normalization.
 * @return Stmt Lowered TIR statement implementing the reduction.
 */
Stmt ReduceOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  auto get_buffer = [&](const Buffer &buf) {
    if (T.buffer_remap.count(buf))
      return T.buffer_remap[buf];
    return buf;
  };

  auto src_scope = this->src.scope();
  auto dst_scope = this->dst.scope();

  if (src_scope == "local.fragment" && dst_scope == "local.fragment") {

    Buffer src_buffer = get_buffer(this->src);
    Buffer dst_buffer = get_buffer(this->dst);
    Fragment src_layout = T.layout_map[this->src].as<Fragment>().value();
    Fragment dst_layout = T.layout_map[this->dst].as<Fragment>().value();
    size_t src_dim = src_layout->InputDim();
    size_t dst_dim = dst_layout->InputDim();

    bool is_1d_reduce = src_dim == dst_dim && dst_dim == 1;

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
    Range reduce_dom(0, src_layout->InputShape()[this->dim]);
    IterVar reduce_iv(reduce_dom, Var("rv"), IterVarType::kDataPar);
    src_vars.insert(src_vars.begin() + this->dim, reduce_iv);

    Array<PrimExpr> src_indices = src_layout->Forward(
        src_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
    Array<PrimExpr> dst_indices = dst_layout->Forward(
        dst_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));

    Array<Stmt> stmts;

    bool require_init = this->clear;
    if (this->type->isSum() || this->type->isAbsSum() ||
        this->type->isBitAnd() || this->type->isBitOr() ||
        this->type->isBitXor()) {
      require_init = true;
    }

    Buffer clear_buffer = dst_buffer;
    bool need_duplicate = false;
    if ((this->type->isSum() || this->type->isAbsSum()) && !this->clear) {
      need_duplicate = true;
    } else if (this->type->isBitAnd() && !this->clear) {
      need_duplicate = true;
    } else if ((this->type->isBitOr() || this->type->isBitXor()) &&
               !this->clear) {
      need_duplicate = true;
    }

    if (need_duplicate) {
      // Create a new buffer with same shape and dtype as dst_buffer
      clear_buffer = decl_buffer(dst_buffer->shape, dst_buffer->dtype,
                                 dst_buffer->name + "_clear",
                                 GetPtrStorageScope(dst_buffer->data));
    }
    // make reduce-init stmt
    if (require_init) {
      stmts.push_back(
          BufferStore(clear_buffer, this->MakeInitValue(), dst_indices));
    }

    // make thread-local reduce
    Array<PrimExpr> src_indice_compressed;
    Array<IterVar> src_var_compressed;
    for (size_t i = 0; i < src_layout->OutputDim(); ++i) {
      PrimExpr expr;
      IterVar var;
      std::tie(expr, var) = CompressIterator(
          src_indices[i], src_vars, src_vars[this->dim]->var, analyzer);
      src_indice_compressed.push_back(expr);
      src_var_compressed.push_back(var);
    }

    Stmt reduce_local = BufferStore(
        clear_buffer,
        this->MakeReduce(BufferLoad(clear_buffer, dst_indices),
                         BufferLoad(src_buffer, src_indice_compressed)),
        dst_indices);

    for (int i = static_cast<int>(src_layout->OutputDim()) - 1; i >= 0; --i) {
      reduce_local =
          For(src_var_compressed[i]->var, 0, src_var_compressed[i]->dom->extent,
              ForKind::kUnrolled, reduce_local, std::nullopt,
              {{tir::attr::pragma_unroll_explicit, Bool(false)}});
    }
    stmts.push_back(reduce_local);

    PrimExpr src_thread = src_layout->ForwardThread(
        src_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }), {});
    auto iter_sum =
        arith::NormalizeToIterSum(src_thread, ToVMap(src_vars), analyzer);
    for (const auto &iter_split : iter_sum->args) {
      auto mark = iter_split->source->source.as<Var>();
      ICHECK(mark) << "Not a normalized iterator: " << iter_split->source;
      if (mark.value().same_as(src_vars[this->dim]->var)) {
        auto scale = as_const_int(iter_split->scale);
        auto extent = as_const_int(iter_split->extent);
        ICHECK(scale != nullptr && extent != nullptr);
        if (*extent == 1)
          continue;

        int reducing_threads = (*extent) * (*scale);
        std::stringstream ss;

        auto thread_offset = T.thread_bounds->min;
        if (TargetIsHopper(T.target) || TargetIsSm100(T.target)) {
          auto all_threads = T.thread_bounds->extent;
          ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", "
             << reducing_threads << ", " << (*scale) << ", " << thread_offset
             << ", " << all_threads << ">::run_hopper";
        } else {
          ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", "
             << reducing_threads << ", " << (*scale) << ", " << thread_offset
             << ">::run";
        }
        Array<PrimExpr> thread_reduce_args = {
            StringImm(ss.str()), BufferLoad(clear_buffer, dst_indices)};
        if (reducing_threads >= 32) {
          PrimExpr workspace = T.AddWorkspace(
              *as_const_int(T.thread_bounds->extent), clear_buffer->dtype);
          thread_reduce_args.push_back(workspace);
        }
        auto call = Call(clear_buffer->dtype, builtin::call_extern(),
                         thread_reduce_args);
        stmts.push_back(BufferStore(clear_buffer, call, dst_indices));
      }
    }

    if (need_duplicate) {
      PrimExpr src_val = BufferLoad(clear_buffer, dst_indices);
      PrimExpr dst_val = BufferLoad(dst_buffer, dst_indices);
      PrimExpr update;
      if (this->type->isSum() || this->type->isAbsSum()) {
        update = dst_val + src_val;
      } else if (this->type->isBitAnd()) {
        update = this->clear ? src_val : bitwise_and(dst_val, src_val);
      } else if (this->type->isBitOr()) {
        update = bitwise_or(dst_val, src_val);
      } else if (this->type->isBitXor()) {
        update = bitwise_xor(dst_val, src_val);
      } else {
        LOG(FATAL) << "Unsupported reduce type: " << this->type->type;
      }
      stmts.push_back(BufferStore(dst_buffer, update, dst_indices));
    }

    Stmt body = stmts.size() > 1 ? SeqStmt(stmts) : stmts[0];
    for (int i = static_cast<int>(dst_layout->InputDim()) - 1; i >= 0; --i) {
      body = For(dst_vars[i]->var, 0, dst_vars[i]->dom->extent,
                 ForKind::kParallel, body);
    }

    if (dst_layout->InputDim() > 0) {
      body = PartitionLoop(Downcast<For>(body), T.thread_var, analyzer,
                           dst_layout);
    } else {
      PrimExpr guard = (T.thread_var == T.thread_bounds->min);
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

LayoutMap ReduceOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  if (level >= InferLevel::kStrict)
    return {};

  if (IsFragmentBuffer(src) && IsFragmentBuffer(dst) &&
      T.layout_map.count(src)) {
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

    // Ensure the thread count is divisible by the replicate extent.
    // Otherwise, we cannot infer a valid fragment<->fragment layout.
    {
      arith::Analyzer analyzer;
      PrimExpr num_threads = T.thread_bounds->extent;
      // Though the dest_buffer_rep_extent will be compressed at
      // CondenseReplicateVar, we need to check the divisibility here to avoid
      // the issue that the thread count is not divisible by the replicate
      // extent.
      if (!analyzer.CanProve(FloorMod(num_threads, dest_buffer_rep_extent) ==
                             0) &&
          !analyzer.CanProve(FloorMod(dest_buffer_rep_extent, num_threads) ==
                             0)) {
        ICHECK(false) << "ReduceOp fragment layout inference failed: "
                         "num_threads % replicate_extent != 0. "
                      << "This mapping requires the block's thread count to be "
                         "divisible by the "
                      << "replicate extent. "
                      << "Try one of: (1) choose a thread block size divisible "
                         "by replicate_extent; "
                      << "(2) pick a different reduce dimension or adjust the "
                         "source fragment layout; "
                      << "Details: num_threads=" << num_threads
                      << ", replicate_extent=" << indice_rep_extent
                      << ", src=" << src << ", dst=" << dst;
      }
    }

    Fragment dst_layout =
        Fragment(dst->shape, {}, thd, dest_buffer_rep_extent, std::nullopt)
            ->CondenseReplicateVar()
            ->BindThreadRange(T.thread_bounds);

    if (!T.layout_map.count(dst))
      return {{dst, dst_layout}};
    else {
      // Check if computed layout is compatible with existing: the existing one
      // must strictly contains the computed layout
      auto orig_dst_layout =
          T.layout_map.Get(dst).value().as<Fragment>().value();
      ICHECK(dst_layout->InputDim() == orig_dst_layout->InputDim());
      Array<PrimExpr> indices;
      indices.reserve(dst_layout->InputDim());
      arith::Analyzer inner_analyzer;
      for (int i = 0; i < dst_layout->InputDim(); ++i) {
        auto x = InputPlaceholder(i);
        indices.push_back(x);
        // should be literal - literal = 0, any analyzer will work
        ICHECK(is_zero(inner_analyzer.Simplify(
            dst_layout->InputShape()[i] - orig_dst_layout->InputShape()[i])));
        inner_analyzer.Bind(x, Range(0, dst_layout->InputShape()[i]));
      }

      ICHECK(as_const_int(dst_layout->ReplicateExtent()));
      ICHECK(as_const_int(src_layout->ReplicateExtent()));
      auto dst_rep = *as_const_int(dst_layout->ReplicateExtent());
      auto src_rep = *as_const_int(src_layout->ReplicateExtent());
      if (dst_rep < src_rep ||
          !ProveFragmentContains(orig_dst_layout, dst_layout, indices, indices,
                                 inner_analyzer)) {
        std::ostringstream oss;
        oss << "Layout may conflict with ReduceOp for buffer " << dst << " vs. "
            << src << "\nLHS = " << src_layout->DebugOutput()
            << "\nRHS = " << orig_dst_layout->DebugOutput()
            << "\nYou may need to use a shared memory to transform the "
               "layout";
        throw LayoutConflictException(oss.str());
      }

      if (dst_rep > src_rep) {
        return {{dst, dst_layout}};
      }
    }
  }
  return {};
}

TIR_REGISTER_TL_TILE_OP(ReduceOp, reduce)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// Normalize "Buffer" to BufferRegion. Use the shape of the buffer as the
// ranges.
static BufferRegion ConvertBufferToBufferRegion(const Buffer &buf) {
  Array<Range> ranges;
  for (PrimExpr extent : buf->shape) {
    ranges.push_back(Range(IntImm(extent->dtype, 0), extent));
  }
  return BufferRegion(buf, ranges);
}

CumSumOp::CumSumOp(Array<PrimExpr> args) {
  /// CumSum constructor arguments:
  /// - src: input buffer
  /// - dst: output buffer
  /// - dim: dimension to cumsum
  /// - reverse: whether to cumsum in reverse order
  CHECK_EQ(args.size(), 4);
  ObjectPtr<CumSumOpNode> node = tvm::ffi::make_object<CumSumOpNode>();
  // node->src = vmap[GetVarFromAccessPtr(args[0])];
  // node->dst = vmap[GetVarFromAccessPtr(args[1])];
  node->srcRegion_ = NormalizeToBufferRegion(args[0]);
  node->dstRegion_ = NormalizeToBufferRegion(args[1]);
  node->src = node->srcRegion_->buffer;
  node->dst = node->dstRegion_->buffer;
  node->dim = args[2].as<IntImm>().value()->value;
  node->reverse = args[3].as<Bool>().value();
  CHECK_LT(node->dim, static_cast<int>(node->src->shape.size()))
      << "The dim of cumsum should be less than the number of dimensions. Got "
         "dim="
      << node->dim << ", but src has " << node->src->shape.size() << " dims.";

  data_ = std::move(node);
}

Stmt CumSumOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (IsFragmentBuffer(this->src) && IsFragmentBuffer(this->dst)) {
    LOG(FATAL) << "CumSum for fragment not implemented, please raise an issue "
                  "if you need this feature.";
  } else if (this->src.scope() == "shared.dyn" ||
             this->src.scope() == "shared") {
    ICHECK(this->dst.scope() == "shared.dyn" || this->dst.scope() == "shared");
    std::stringstream ss;
    auto threads = T.thread_bounds->extent;
    Array<PrimExpr> args;

    // Build access pointers from regions locally
    PrimExpr srcPtr = MakeAccessPtrFromRegion(srcRegion_, 1);
    PrimExpr dstPtr = MakeAccessPtrFromRegion(dstRegion_, 2);

    // Use region extents instead of buffer shape for correct slice handling
    Array<PrimExpr> src_extents;
    for (const auto &range : srcRegion_->region) {
      src_extents.push_back(range->extent);
    }
    int ndim = static_cast<int>(src_extents.size());

    if (ndim == 1) {
      ICHECK_EQ(dim, 0) << "Cumulative sum over a 1D buffer only supports dim "
                           "= 0.";
      ss << "tl::CumSum1D<" << threads << ", " << (reverse ? "true" : "false")
         << ">::run";
      args = {StringImm(ss.str()), srcPtr, dstPtr, src_extents[0]};
    } else if (ndim == 2) {
      ss << "tl::CumSum2D<" << threads << ", " << dim << ", "
         << (reverse ? "true" : "false") << ">::run";
      args = {StringImm(ss.str()), srcPtr, dstPtr, src_extents[0],
              src_extents[1]};
    } else {
      LOG(FATAL) << "CumSum currently supports only 1D or 2D buffers, got "
                 << ndim << "D.";
    }
    return Evaluate(Call(dst->dtype, builtin::call_extern(), args));
  } else {
    ICHECK(false) << "Cannot lower cumsum for " << this->src.scope() << " and "
                  << this->dst.scope();
  }

  return Stmt();
}

LayoutMap CumSumOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  // Only infer layout in strict mode
  if (level != InferLevel::kStrict) {
    return {};
  }

  LayoutMap result_map;

  auto make_linear_layout = [](const Buffer &buf) -> Layout {
    return makeLinearLayout(buf->shape);
  };

  auto check_or_set_linear_layout = [&](const Buffer &buf) {
    if (!IsSharedBuffer(buf))
      return;

    Layout linear_layout = make_linear_layout(buf);
    if (T.layout_map.count(buf)) {
      // Check if existing layout is linear
      Layout existing = T.layout_map.Get(buf).value().as<Layout>().value();
      ICHECK(StructuralEqual()(existing, linear_layout))
          << "CumSum requires linear layout for shared buffer " << buf->name
          << ", but got non-linear layout.";
    } else {
      result_map.Set(buf, linear_layout);
    }
  };

  check_or_set_linear_layout(src);
  check_or_set_linear_layout(dst);

  return result_map;
}

TIR_REGISTER_TL_TILE_OP(CumSumOp, cumsum)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  ReduceOpNode::RegisterReflection();
  CumSumOpNode::RegisterReflection();
  ReduceTypeNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
