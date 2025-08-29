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
#include "../op/parallel.h"
#include "../transform/loop_partition.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;

/**
 * @brief Construct a ReduceOp from raw TL arguments and a buffer mapping.
 *
 * Interprets `args` and `vmap` to populate an internal ReduceOpNode:
 * - args[0]: access pointer for the source buffer
 * - args[1]: access pointer for the destination buffer
 * - args[2]: string literal specifying the reduce type: "sum", "abssum",
 *            "absmax", "max", or "min"
 * - args[3]: integer literal for the reduction dimension (axis)
 * - args[4]: boolean literal indicating whether to clear/init the destination
 *
 * The constructor resolves the access pointers via `vmap`, maps the reduce
 * type string to the ReduceType enum, assigns the reduction dimension and
 * clear flag, and stores the constructed node in `data_`. An invalid reduce
 * type triggers a fatal check.
 *
 * @param args Array of TL prim-expr arguments as described above.
 * @param vmap Mapping from variables (from access pointers) to Buffer objects.
 */
ReduceOp::ReduceOp(Array<PrimExpr> args, BufferMap vmap) {
  ObjectPtr<ReduceOpNode> node = make_object<ReduceOpNode>();
  node->src = vmap[GetVarFromAccessPtr(args[0])];
  node->dst = vmap[GetVarFromAccessPtr(args[1])];
  std::string reduce_type = args[2].as<StringImm>().value()->value;
  node->dim = args[3].as<IntImm>().value()->value;
  if (reduce_type == "sum")
    node->type = ReduceType::kSum;
  else if (reduce_type == "abssum")
    node->type = ReduceType::kAbsSum;
  else if (reduce_type == "absmax")
    node->type = ReduceType::kAbsMax;
  else if (reduce_type == "max")
    node->type = ReduceType::kMax;
  else if (reduce_type == "min")
    node->type = ReduceType::kMin;
  else
    ICHECK(0) << "Unknown reduce type: " << reduce_type;
  node->clear = args[4].as<Bool>().value();
  data_ = std::move(node);
}

/**
 * @brief Create a copy of this ReduceOpNode wrapped as a TileOperator.
 *
 * Returns a new TileOperator holding a freshly allocated ReduceOpNode constructed
 * as a copy of this node.
 *
 * @return TileOperator A tile operator that owns the cloned ReduceOpNode.
 */
TileOperator ReduceOpNode::Clone() const {
  auto op = make_object<ReduceOpNode>(*this);
  return ReduceOp(op);
}

/**
 * @brief Create a deep copy of this CumSum op node wrapped as a TileOperator.
 *
 * Returns a new TileOperator whose underlying CumSumOpNode is a copy of
 * the current node. Useful for cloning operators when building or
 * transforming computation graphs.
 *
 * @return TileOperator A TileOperator containing a copy of this node.
 */
TileOperator CumSumOpNode::Clone() const {
  auto op = make_object<CumSumOpNode>(*this);
  return CumSumOp(op);
}

/**
 * @brief Create the initial accumulator value for the destination buffer based on reduction type.
 *
 * Returns the PrimExpr representing the initial value stored in the destination accumulator
 * before any source elements are combined. The returned value depends on the destination
 * dtype and the node's reduction type:
 * - kSum, kAbsSum: zero of the destination dtype.
 * - kMax: minimum representable value for signed integers, zero for unsigned integers,
 *   and -INFINITY for floating point.
 * - kMin: maximum representable value for signed integers, all-ones (max) for unsigned integers,
 *   and +INFINITY for floating point.
 * - kAbsMax: zero of the destination dtype.
 *
 * The function will abort (ICHECK failure) if the reduction type is unrecognized.
 *
 * @return PrimExpr initial value appropriate for `dst->dtype` and `type`.
 */
PrimExpr ReduceOpNode::MakeInitValue() const {
  auto dst_dtype = dst->dtype;
  auto is_int = dst_dtype.is_int();
  bool is_uint = dst_dtype.is_uint();
  auto bits = dst_dtype.bits();

  switch (type) {
  case ReduceType::kSum:
    return make_zero(dst->dtype);
  case ReduceType::kAbsSum:
    return make_zero(dst->dtype);
  case ReduceType::kMax:
    if (is_int) {
      return make_const(dst->dtype, -(1 << (bits - 1)));
    } else if (is_uint) {
      return make_const(dst->dtype, 0);
    } else {
      return make_const(dst->dtype, -INFINITY);
    }
  case ReduceType::kMin:
    if (is_int) {
      return make_const(dst->dtype, (1 << (bits - 1)) - 1);
    } else if (is_uint) {
      return make_const(dst->dtype, (1 << bits) - 1);
    } else {
      return make_const(dst->dtype, INFINITY);
    }
  case ReduceType::kAbsMax:
    return make_const(dst->dtype, 0);
  default:
    ICHECK(0);
  }
}

/**
 * @brief Combine two scalar expressions according to this node's reduction type.
 *
 * Casts the right operand to the left operand's dtype if they differ, then returns
 * the reduction of `a` and `b` using the operator specified by `type`:
 * - kSum: `a + b`
 * - kAbsSum: `a + max(b, -b)`
 * - kMax: `max(a, b)`
 * - kMin: `min(a, b)`
 * - kAbsMax: `max(max(a, b), -min(a, b))`
 *
 * @param a Left-hand operand (result dtype drives the output dtype).
 * @param b Right-hand operand (will be cast to `a`'s dtype if needed).
 * @return PrimExpr The combined expression with dtype equal to `a.dtype`.
 *
 * @note The function DCHECKs/ICHECKs on an unknown/unsupported reduction type.
 */
PrimExpr ReduceOpNode::MakeReduce(const PrimExpr &a, const PrimExpr &b) const {
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

/**
 * @brief Map the reduction type to the codegen reducer name used by external ALL-Reduce/CUDA helpers.
 *
 * Returns the string identifier of the code-generation reducer corresponding to this
 * ReduceOpNode's `type`. Mapping:
 * - kSum, kAbsSum -> "tl::SumOp"
 * - kMax, kAbsMax -> "tl::MaxOp"
 * - kMin -> "tl::MinOp"
 *
 * The function terminates with a check failure if `type` is unknown.
 *
 * @return std::string Reducer name used by codegen extern calls.
 */
std::string ReduceOpNode::MakeCodegenReducer() const {
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

/**
 * @brief Lower the Reduce operator node to a TIR statement.
 *
 * Lowers a ReduceOpNode that targets fragment-local buffers into a sequence of
 * TIR statements implementing: per-thread local reduction, inter-thread
 * AllReduce (when needed), and final writeback (with an optional duplicate
 * clear buffer to avoid in-place conflicts). Supports reduction kinds
 * (sum/abs-sum/max/min/abs-max) and handles layout-driven index mapping and
 * loop partitioning to thread axes.
 *
 * @param T Lowering context providing buffer remapping, layout map, target and
 *          thread bounds, and workspace allocation helper. Must contain
 *          fragment-local mappings for both src and dst.
 * @param analyzer Symbolic analyzer used to simplify and compress iterators.
 * @return Stmt The constructed TIR statement implementing the reduction.
 *
 * Preconditions:
 * - src and dst buffers must be in "local.fragment" scope.
 * - The layouts must have compatible input/output dimensions for the
 *   specified reduction axis.
 *
 * Failure modes:
 * - The function uses ICHECK to enforce unsupported scopes, dimension
 *   mismatches, unknown reduction types, and other invariants; violations
 *   will trigger a fatal check failure.
 */
Stmt ReduceOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  ICHECK(this->src.scope() == "local.fragment" &&
         this->dst.scope() == "local.fragment")
      << "Reduce for shared memory not implemented.";
  auto src_buffer = T.buffer_remap[this->src];
  auto dst_buffer = T.buffer_remap[this->dst];
  Fragment src_layout = T.layout_map[this->src].as<Fragment>().value();
  Fragment dst_layout = T.layout_map[this->dst].as<Fragment>().value();
  size_t src_dim = src_layout->InputDim();
  size_t dst_dim = dst_layout->InputDim();

  bool is_1d_reduce = src_dim == dst_dim && dst_dim == 1;

  if (is_1d_reduce) {
    ICHECK(is_one(dst_layout->OutputShape().back()))
        << "Reduce for scalar not implemented.";
  } else {
    ICHECK(src_dim == dst_dim + 1) << "Reduce dimension mismatch.";
  }

  Array<IterVar> dst_vars;
  for (size_t i = 0; i < dst_dim; i++) {
    Var var = Var(std::string{char('i' + i)});
    dst_vars.push_back(IterVar(Range(0, dst_layout->InputShape()[i]), var,
                               IterVarType::kDataPar));
  }
  Array<IterVar> src_vars;
  if (!is_1d_reduce) {
    src_vars = dst_vars;
  }
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
            ForKind::kUnrolled, reduce_local, std::nullopt,
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
    ICHECK(mark) << "Not a normalized iterator: " << iter_split->source;
    if (mark.value().same_as(src_vars[this->dim]->var)) {
      auto scale = as_const_int(iter_split->scale);
      auto extent = as_const_int(iter_split->extent);
      ICHECK(scale != nullptr && extent != nullptr);
      if (*extent == 1)
        continue;

      int reducing_threads = (*extent) * (*scale);
      std::stringstream ss;

      bool has_arch = T.target->attrs.count("arch") > 0;
      auto thread_offset = T.thread_bounds->min;
      if (has_arch && Downcast<String>(T.target->attrs["arch"]) == "sm_90") {
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

/**
 * @brief Infer a layout mapping for the destination buffer of a Reduce operator.
 *
 * When inference level is below `kStrict`, and both source and destination buffers
 * live in `local.fragment` with a known source fragment layout, this computes a
 * candidate destination Fragment layout that accounts for replication over the
 * reduction dimension and binds thread ranges from `T.thread_bounds`.
 *
 * Behavior:
 * - Constructs a destination Fragment whose replicate extent equals
 *   src.shape[dim] * src_fragment.ReplicateExtent(), and whose threading is
 *   derived from the source fragment with the reduction dimension folded out.
 * - If no layout exists for `dst` in `T.layout_map`, returns a map {dst -> inferred}.
 * - If `dst` already has a layout, validates that the existing layout strictly
 *   contains the computed layout (shapes match and fragment containment holds).
 *   If compatible but the computed replicate extent is larger, returns the new layout.
 * - In all other cases (strict inference level, unsupported scopes, or no src layout),
 *   returns an empty map.
 *
 * @param T Layout inference context containing `layout_map` and `thread_bounds`.
 * @param level Inference strictness; no inference is performed at or above `kStrict`.
 * @return LayoutMap A mapping for `dst` to an inferred Fragment layout, or empty.
 * @throws LayoutConflictException if an existing `dst` layout conflicts with the
 *         computed layout (not containable or incompatible replication extents).
 */
LayoutMap ReduceOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  if (level >= InferLevel::kStrict)
    return {};
  if (src.scope() == "local.fragment" && dst.scope() == "local.fragment" &&
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

TIR_REGISTER_TL_OP(ReduceOp, reduce)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

/**
 * @brief Construct a CumSumOp from a list of arguments and a buffer map.
 *
 * Expects args to contain exactly four PrimExprs in this order:
 *  0: access pointer to source buffer (src),
 *  1: access pointer to destination buffer (dst),
 *  2: integer dimension to perform the cumulative sum along (dim),
 *  3: boolean flag indicating whether to compute the cumsum in reverse (reverse).
 *
 * The constructor resolves src and dst from the provided BufferMap and stores
 * the parsed dim and reverse values on the node. It verifies that args.size()
 * == 4 and that dim is a valid axis for the source buffer shape.
 *
 * @param args Array of PrimExpr as described above.
 */
CumSumOp::CumSumOp(Array<PrimExpr> args, BufferMap vmap) {
  /*
    CumSum arguments:
      src: input buffer
      dst: output buffer
      dim: dimension to cumsum
      reverse: whether to cumsum in reverse order
   */
  CHECK_EQ(args.size(), 4);
  ObjectPtr<CumSumOpNode> node = make_object<CumSumOpNode>();
  node->src = vmap[GetVarFromAccessPtr(args[0])];
  node->dst = vmap[GetVarFromAccessPtr(args[1])];
  node->dim = args[2].as<IntImm>().value()->value;
  node->reverse = args[3].as<Bool>().value();
  CHECK_LT(node->dim, static_cast<int>(node->src->shape.size()));
  data_ = std::move(node);
}

/**
 * @brief Lower the CumSum operator to TIR.
 *
 * Produces a TIR statement implementing cumulative sum depending on buffer scopes:
 * - For shared/shared.dyn scopes: emits an extern call to `tl::CumSum2D<threads, dim, reverse>::run`
 *   with arguments [function_name, src.access_ptr(1), dst.access_ptr(3), src.shape...]. The number
 *   of threads is taken from `T.thread_bounds->extent`. Returns an Evaluate(Call(...)) statement.
 * - For local.fragment scopes on both src and dst: fatal error (not implemented).
 * - For any other scope combinations: fails with an assertion.
 *
 * The `analyzer` parameter is accepted for interface compatibility but is not used by this lowering.
 *
 * @param T Lowering arguments (provides thread bounds and other lowering context).
 * @return Stmt A TIR statement representing the lowered cumulative-sum operation.
 */
Stmt CumSumOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
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

/**
 * @brief Layout inference for CumSum operator.
 *
 * CumSum does not perform any layout inference; this function always returns
 * an empty mapping. The operator's lowering expects shared-memory semantics
 * and layout decisions are handled elsewhere.
 *
 * @param T Layout inference inputs (buffers, existing layouts, etc.).
 * @param level Inference strictness level (unused).
 * @return LayoutMap Empty map indicating no inferred layouts.
 */
LayoutMap CumSumOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  return {};
}

TIR_REGISTER_TL_OP(CumSumOp, cumsum)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
} // namespace tl
} // namespace tvm