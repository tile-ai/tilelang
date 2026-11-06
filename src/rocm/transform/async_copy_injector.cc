/*!
 * \brief Inject eligible global->shared copies into ROCm async-copy intrinsics.
 * \file rocm/transform/async_copy_injector.cc
 */
#include "cuda/op/builtin.h"
#include "ir/lexical_scope.h"
#include "ir/loop_math.h"
#include "support/check.h"
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "ir/annotations.h"
#include "ir/tensor_utils.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "rocm/op/builtin.h"
#include "rocm/transform/async_copy_injector.h"
#include <tvm/ir/prim/op.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;
using namespace ffi;

class ROCmAsyncCopyInjector : public StmtExprMutator {
public:
  explicit ROCmAsyncCopyInjector(bool async_without_async_commit_wait)
      : async_without_async_commit_wait_(async_without_async_commit_wait) {}

  using StmtExprMutator::Mutate_;

  bool InjectedROCmAsyncCopy() const { return injected_rocm_async_copy_; }

  Stmt Finalize(Stmt body) {
    if (!pending_sync_copies_ || UseExplicitAsyncSemantics()) {
      pending_sync_copies_ = false;
      uncommitted_sync_copies_ = false;
      return body;
    }

    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(body);
    AppendSyncVisibility(&seq, uncommitted_sync_copies_);
    pending_sync_copies_ = false;
    uncommitted_sync_copies_ = false;
    return SeqStmt(seq);
  }

  UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    // Track nested vectorized loop extents so we can decide whether an
    // element-wise copy has a legal final cp.async width after later loop
    // vectorization:
    //   for v in T.vectorized(k): tl.ptx_cp_async(dst, src, elem_count)
    // => tl.ptx_cp_async(dst_base, src_base, elem_count * k)
    //
    // TileLang records logical element counts in tl.ptx_cp_async. HIP codegen
    // derives the final ROCm byte width is derived later from the access_ptr
    // dtype, so subbyte dtypes such as int4/fp4/int2/int1 remain representable
    // here.
    int previous_vectorized_lanes = current_vectorized_lanes_;
    bool pushed_vectorized_loop = false;
    if (op->kind == ForKind::kVectorized) {
      PrimExpr extent = analyzer_->Simplify(LoopExtent(op));
      const auto *extent_imm = extent.as<IntImmObj>();
      ICHECK(extent_imm)
          << "Vectorized loops must have constant extent, but got "
          << LoopExtent(op);
      auto constant = extent_imm->value.as<int64_t>();
      TVM_FFI_CHECK(constant && *constant >= 0 &&
                        *constant <= std::numeric_limits<int>::max(),
                    ValueError);
      int lanes = static_cast<int>(*constant);
      if (lanes > 1 && current_vectorized_lanes_ <=
                           std::numeric_limits<int>::max() / lanes) {
        current_vectorized_lanes_ *= lanes;
        active_vectorized_loops_.push_back({op->var, lanes});
        pushed_vectorized_loop = true;
      }
    }
    Stmt stmt =
        StmtExprMutator::Mutate_(op, mode).ValueOrUnchanged(GetRef<Stmt>(op));
    if (pushed_vectorized_loop) {
      active_vectorized_loops_.pop_back();
    }
    current_vectorized_lanes_ = previous_vectorized_lanes;

    // A later iteration can consume copies left pending by an earlier one.
    // Flush the backedge of a mixed loop; a pure copy loop has no consumers
    // and retains its single hoisted synchronization after the loop.
    if (!UseExplicitAsyncSemantics() && pending_sync_copies_) {
      For loop = stmt.as_or_throw<For>();
      if (!AnalyzeCopyRegion(loop->body).is_pure_copy_region) {
        Array<Stmt> body = loop->body->seq;
        AppendSyncVisibility(&body, uncommitted_sync_copies_);
        loop.CopyOnWrite()->body = SeqStmt(body);
        pending_sync_copies_ = false;
        uncommitted_sync_copies_ = false;
        stmt = loop;
      }
    }
    return stmt;
  }

  Optional<Stmt> TryInjectROCmAsyncCopy(
      const TensorLoadObj *load, const TensorStoreObj *store,
      bool predicated = false,
      const PrimExpr &predicate_value = PrimExpr(ffi::UnsafeInit{})) {
    // Pipeline:
    // 1) Analyze source/destination indices and transfer width eligibility.
    // 2) Build tl.ptx_cp_async with scalar/vectorized base offsets when the
    //    eventual ROCm byte width is representable.
    std::optional<CopyIndexInfo> index_info = PrepareCopyIndexInfo(load, store);
    if (!index_info.has_value()) {
      return Optional<Stmt>();
    }

    if (index_info->index_lanes == 1) {
      if (current_vectorized_lanes_ > 1 &&
          !HasContiguousVectorizedOffsets(index_info->src_index,
                                          index_info->dst_index)) {
        return Optional<Stmt>();
      }
      return MakeCPAsyncStmtFromLoads(
          store,
          /*dst_base_load=*/
          TensorLoad(store->dest.as_or_throw<TensorVar>(), store->indices),
          /*src_base_load=*/
          TensorLoad(load->source.as_or_throw<TensorVar>(), load->indices),
          /*num_elems=*/index_info->per_access_num_elems, predicated,
          predicate_value);
    }

    Optional<Array<PrimExpr>> src_base_indices =
        ExtractVectorBaseIndices(load->indices);
    Optional<Array<PrimExpr>> dst_base_indices =
        ExtractVectorBaseIndices(store->indices);
    if (!src_base_indices.has_value() || !dst_base_indices.has_value()) {
      // If we can't extract base indices from vectorized accesses, fall back.
      if (predicated) {
        LOG(WARNING)
            << "Cannot extract base indices from vectorized accesses for "
               "predicated cp.async; falling back to regular buffer store/load";
      }
      return Optional<Stmt>();
    }
    return MakeCPAsyncStmtFromLoads(
        store,
        /*dst_base_load=*/
        TensorLoad(store->dest.as_or_throw<TensorVar>(),
                   dst_base_indices.value()),
        /*src_base_load=*/
        TensorLoad(load->source.as_or_throw<TensorVar>(),
                   src_base_indices.value()),
        /*num_elems=*/index_info->per_access_num_elems, predicated,
        predicate_value);
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    if (UseExplicitAsyncSemantics()) {
      return StmtExprMutator::Mutate_(op, mode).ValueOrUnchanged(
          GetRef<Stmt>(op));
    }

    // Insert commit+wait at statement boundaries to preserve synchronous
    // semantics for normal global->shared TensorStore copies.
    //
    // Important: avoid flushing inside inner loop bodies just because there
    // are trailing no-op statements (e.g., Evaluate(0)) after the injected
    // cp.async. Instead, treat "pure copy region" statements as part of the
    // copy run and only flush right before the next non-copy statement.
    Array<Stmt> out;
    out.reserve(op->seq.size() + 2);

    CopySyncState sync_state{pending_sync_copies_, uncommitted_sync_copies_};
    pending_sync_copies_ = false;
    uncommitted_sync_copies_ = false;

    for (const Stmt &stmt : op->seq) {
      VisitedStmtInfo visited_info = VisitAndAnalyzeStmt(stmt);
      bool stmt_is_pure_copy_region = visited_info.analysis.is_pure_copy_region;

      // Before we execute a non-copy statement, we must preserve synchronous
      // semantics for injected cp.async stores by making the data visible.
      if (sync_state.open_copy_region && !stmt_is_pure_copy_region) {
        AppendSyncVisibility(&out, sync_state.uncommitted_transfers);
        sync_state.open_copy_region = false;
        sync_state.uncommitted_transfers = false;
      }

      // If we are carrying uncommitted injected cp.async into an explicit wait,
      // ensure they are committed so the wait actually covers them.
      if (sync_state.open_copy_region && sync_state.uncommitted_transfers &&
          visited_info.analysis.may_wait) {
        out.push_back(MakeCommitGroupStmt());
        sync_state.uncommitted_transfers = false;
      }

      out.push_back(visited_info.visited);

      // These guarantees apply to copies entering the statement. Copies
      // injected inside it have already been analyzed in execution order and
      // may occur after its last commit or wait, so merge their exit state
      // last.
      if (visited_info.analysis.guarantees_commit) {
        sync_state.uncommitted_transfers = false;
      }
      if (visited_info.analysis.guarantees_wait_all) {
        sync_state.open_copy_region = false;
        sync_state.uncommitted_transfers = false;
      }
      if (visited_info.opens_copy_region) {
        sync_state.open_copy_region = true;
        sync_state.uncommitted_transfers =
            sync_state.uncommitted_transfers ||
            visited_info.has_uncommitted_transfers;
      }
    }

    pending_sync_copies_ = sync_state.open_copy_region;
    uncommitted_sync_copies_ = sync_state.uncommitted_transfers;

    if (out.empty()) {
      return Evaluate(0);
    }
    if (out.size() == 1) {
      return out[0];
    }
    return SeqStmt(out);
  }

  UnchangedOr<Stmt> Mutate_(const IfObj *op, InplaceMode mode) final {
    if (UseExplicitAsyncSemantics()) {
      return StmtExprMutator::Mutate_(op, mode).ValueOrUnchanged(
          GetRef<Stmt>(op));
    }

    // Treat branches as separate control flow paths. We propagate pending
    // synchronous copies into both branches (they occur before the branch),
    // but do not let mutations in one branch affect the other.
    bool pending_before = pending_sync_copies_;
    bool uncommitted_before = uncommitted_sync_copies_;

    pending_sync_copies_ = pending_before;
    uncommitted_sync_copies_ = uncommitted_before;
    SeqStmt then_case =
        Mutate(op->then_case, mode).ValueOrUnchanged(op->then_case);
    bool pending_then = pending_sync_copies_;
    bool uncommitted_then = uncommitted_sync_copies_;

    bool pending_else = pending_before;
    bool uncommitted_else = uncommitted_before;
    Optional<SeqStmt> else_case;
    if (op->else_case.has_value()) {
      pending_sync_copies_ = pending_before;
      uncommitted_sync_copies_ = uncommitted_before;
      else_case = Mutate(op->else_case.value(), mode)
                      .ValueOrUnchanged(op->else_case.value());
      pending_else = pending_sync_copies_;
      uncommitted_else = uncommitted_sync_copies_;
    }

    pending_sync_copies_ = pending_then || pending_else;
    uncommitted_sync_copies_ = uncommitted_then || uncommitted_else;

    if (then_case.same_as(op->then_case) &&
        (!else_case.has_value() ||
         else_case.value().same_as(op->else_case.value()))) {
      return GetRef<Stmt>(op);
    }
    return If(op->condition, then_case, else_case);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj *store,
                            InplaceMode mode) final {
    if (!IsSharedBuffer(store->dest.as_or_throw<TensorVar>())) {
      return StmtExprMutator::Mutate_(store, mode)
          .ValueOrUnchanged(GetRef<Stmt>(store));
    }

    Optional<PrimExpr> predicate = std::nullopt;
    const TensorLoadObj *load =
        MatchZeroFillBufferLoad(store->value, &predicate);
    if (load) {
      Optional<Stmt> injected = TryInjectROCmAsyncCopy(
          load, store, predicate.has_value(),
          predicate.has_value() ? predicate.value()
                                : PrimExpr(ffi::UnsafeInit{}));
      if (injected.has_value()) {
        injected_rocm_async_copy_ = true;
        if (!UseExplicitAsyncSemantics()) {
          pending_sync_copies_ = true;
          uncommitted_sync_copies_ = true;
        }
        return injected.value();
      }
    }

    return StmtExprMutator::Mutate_(store, mode)
        .ValueOrUnchanged(GetRef<Stmt>(store));
  }

private:
  bool UseExplicitAsyncSemantics() const {
    return async_without_async_commit_wait_;
  }

  // A copy candidate represented after flattening source/destination indexing.
  struct CopyIndexInfo {
    PrimExpr src_index{ffi::UnsafeInit{}};
    PrimExpr dst_index{ffi::UnsafeInit{}};
    int index_lanes{1};
    int per_access_num_elems{0};
  };

  // Synchronization state for injected cp.async runs carried across statements.
  struct CopySyncState {
    bool open_copy_region{false};
    bool uncommitted_transfers{false};
  };

  struct ActiveVectorizedLoop {
    PrimVar loop_var;
    int extent;
  };

  // ---- Copy candidate analysis helpers ----
  static const TensorLoadObj *
  MatchZeroFillBufferLoad(const PrimExpr &value,
                          Optional<PrimExpr> *predicate) {
    if (const auto *load = value.as<TensorLoadObj>()) {
      return load;
    }

    const auto *call = value.as<CallObj>();
    if (!call || !call->op.same_as(prim::if_then_else_op()) ||
        !IsZeroBitPattern(call->args[2].as_or_throw<PrimExpr>())) {
      return nullptr;
    }

    const TensorLoadObj *load = MatchZeroFillBufferLoad(
        call->args[1].as_or_throw<PrimExpr>(), predicate);
    if (load == nullptr) {
      return nullptr;
    }

    *predicate =
        predicate->has_value()
            ? Optional<PrimExpr>(And(call->args[0].as_or_throw<PrimExpr>(),
                                     predicate->value()))
            : Optional<PrimExpr>(call->args[0].as_or_throw<PrimExpr>());
    return load;
  }

  static Optional<PrimExpr>
  FlattenToLinearOffset(const TensorVar &buf, const Array<PrimExpr> &indices) {
    auto physical = buf.ty()->ElemOffset(indices);
    if (physical.size() != 1)
      return std::nullopt;
    return physical[0];
  }

  std::optional<CopyIndexInfo>
  PrepareCopyIndexInfo(const TensorLoadObj *load, const TensorStoreObj *store) {
    if (!IsGlobalBuffer(load->source.as_or_throw<TensorVar>())) {
      return std::nullopt;
    }

    Optional<PrimExpr> src_index_opt = FlattenToLinearOffset(
        load->source.as_or_throw<TensorVar>(), load->indices);
    Optional<PrimExpr> dst_index_opt = FlattenToLinearOffset(
        store->dest.as_or_throw<TensorVar>(), store->indices);
    if (!src_index_opt.has_value() || !dst_index_opt.has_value()) {
      return std::nullopt;
    }

    PrimExpr src_index = src_index_opt.value();
    PrimExpr dst_index = dst_index_opt.value();
    if (src_index.ty().lanes() != dst_index.ty().lanes()) {
      // Not a straightforward vectorized copy; skip.
      return std::nullopt;
    }

    const int index_lanes = src_index.ty().lanes();
    const int value_lanes = load->ty.as_or_throw<PrimType>().lanes();
    if (value_lanes > 1 && index_lanes > 1 && value_lanes != index_lanes) {
      // Mismatched vector lane representations; be conservative.
      return std::nullopt;
    }

    const int effective_lanes = std::max(value_lanes, index_lanes);
    const int per_access_bits =
        effective_lanes * load->ty.as_or_throw<PrimType>().bits();
    const int total_bits = static_cast<int>(per_access_bits) *
                           static_cast<int>(current_vectorized_lanes_);
    // The ROCm async-copy marker is byte-granular. `tl.ptx_cp_async` stores
    // logical element counts, but we still need to know that the eventual
    // vectorized transfer can map to a legal byte width without over-copying
    // packed subbyte data.
    if (total_bits % 8 != 0) {
      return std::nullopt;
    }
    const int total_bytes = total_bits / 8;
    if (!IsValidCPAsyncTransferBytes(total_bytes)) {
      return std::nullopt;
    }

    CopyIndexInfo info;
    info.src_index = src_index;
    info.dst_index = dst_index;
    info.index_lanes = index_lanes;
    info.per_access_num_elems = effective_lanes;
    return info;
  }

  static PrimExpr ExtractVectorBase(const PrimExpr &index) {
    if (index.ty().lanes() == 1) {
      return index;
    }
    if (const auto *broadcast = index.as<BroadcastObj>()) {
      return broadcast->value;
    }
    if (const auto *ramp = index.as<RampObj>()) {
      if (!IsOne(ramp->stride)) {
        return PrimExpr(ffi::UnsafeInit{});
      }
      return ramp->base;
    }

    const auto *add = index.as<AddObj>();
    if (!add) {
      return PrimExpr(ffi::UnsafeInit{});
    }

    // Common pattern after flattening a vectorized N-D buffer access:
    //   (broadcast(base_offset) + ramp(vec_base, 1, lanes))
    // or its commuted form:
    //   (ramp(vec_base, 1, lanes) + broadcast(base_offset))
    const PrimExpr &lhs = add->a;
    const PrimExpr &rhs = add->b;
    if (const auto *lhs_ramp = lhs.as<RampObj>()) {
      if (!IsOne(lhs_ramp->stride)) {
        return PrimExpr(ffi::UnsafeInit{});
      }
      if (const auto *rhs_broadcast = rhs.as<BroadcastObj>()) {
        return prim::Add(lhs_ramp->base, rhs_broadcast->value);
      }
    }
    if (const auto *rhs_ramp = rhs.as<RampObj>()) {
      if (!IsOne(rhs_ramp->stride)) {
        return PrimExpr(ffi::UnsafeInit{});
      }
      if (const auto *lhs_broadcast = lhs.as<BroadcastObj>()) {
        return prim::Add(rhs_ramp->base, lhs_broadcast->value);
      }
    }
    return PrimExpr(ffi::UnsafeInit{});
  }

  static Optional<Array<PrimExpr>>
  ExtractVectorBaseIndices(const Array<PrimExpr> &indices) {
    Array<PrimExpr> base_indices;
    base_indices.reserve(indices.size());
    for (const PrimExpr &index : indices) {
      PrimExpr base = ExtractVectorBase(index);
      if (!base.defined()) {
        return Optional<Array<PrimExpr>>();
      }
      base_indices.push_back(base);
    }
    return base_indices;
  }

  static Expr MakeAccessPtrFromLoad(const TensorLoad &base_load, int extent,
                                    int rw_mask) {
    return MakeTensorAccessPtr(base_load, IntImm::Int32(extent), rw_mask);
  }

  static Optional<Stmt>
  MakeCPAsyncStmtFromLoads(const TensorStoreObj *store,
                           const TensorLoad &dst_base_load,
                           const TensorLoad &src_base_load, int num_elems,
                           bool predicated, const PrimExpr &predicate_value) {
    Expr dst_access_ptr =
        MakeAccessPtrFromLoad(dst_base_load, num_elems, /*rw_mask=*/2);
    Expr src_access_ptr =
        MakeAccessPtrFromLoad(src_base_load, num_elems, /*rw_mask=*/1);

    Array<Expr> cp_async_args;
    if (predicated) {
      cp_async_args = {dst_access_ptr, src_access_ptr, PrimExpr(num_elems),
                       predicate_value};
    } else {
      cp_async_args = {dst_access_ptr, src_access_ptr, PrimExpr(num_elems)};
    }
    return Evaluate(Call(PrimType::Void(), tvm::tl::ptx_cp_async(),
                         cp_async_args, {}, {}, store->loc));
  }

  static Stmt MakeCommitGroupStmt() {
    return Evaluate(Call(PrimType::Void(), Op::Get("tl.ptx_commit_group"), {}));
  }

  static Stmt MakeWaitGroupStmt(int n) {
    return Evaluate(Call(PrimType::Void(), Op::Get("tl.ptx_wait_group"),
                         {IntImm(PrimType::Int(32), n)}));
  }

  // ---- Vectorized-offset contiguity helpers ----
  static bool TryGetConstInt64(const PrimExpr &expr, int64_t *value) {
    if (const auto *imm = expr.as<IntImmObj>()) {
      if (auto constant = imm->value.as<int64_t>()) {
        *value = *constant;
        return true;
      }
    }
    return false;
  }

  bool HasUnitStrideForVectorizedLoop(const PrimExpr &expr,
                                      const ActiveVectorizedLoop &loop) {
    PrimExpr prev =
        analyzer_->Simplify(tirx::SubstituteWithDataTypeLegalization(
            expr,
            [values = ffi::Map<Var, PrimExpr>{{loop.loop_var,
                                               IntImm(loop.loop_var.ty(), 0)}}](
                const Var &var) -> ffi::Optional<PrimExpr> {
              return values.Get(var);
            }));

    int64_t stride = 0;
    for (int value = 1; value < loop.extent; ++value) {
      PrimExpr curr =
          analyzer_->Simplify(tirx::SubstituteWithDataTypeLegalization(
              expr,
              [values =
                   ffi::Map<Var, PrimExpr>{
                       {loop.loop_var, IntImm(loop.loop_var.ty(), value)}}](
                  const Var &var) -> ffi::Optional<PrimExpr> {
                return values.Get(var);
              }));
      PrimExpr delta = analyzer_->Simplify(curr - prev);
      int64_t delta_value = 0;
      if (!TryGetConstInt64(delta, &delta_value)) {
        return false;
      }
      if (value == 1) {
        stride = delta_value;
      } else if (delta_value != stride) {
        return false;
      }
      prev = curr;
    }

    return stride == 1;
  }

  bool HasContiguousVectorizedOffsets(const PrimExpr &src_index,
                                      const PrimExpr &dst_index) {
    for (const auto &loop : active_vectorized_loops_) {
      if (!HasUnitStrideForVectorizedLoop(src_index, loop) ||
          !HasUnitStrideForVectorizedLoop(dst_index, loop)) {
        return false;
      }
    }
    return true;
  }

  // ---- Copy-region synchronization analysis helpers ----
  struct CopyRegionAnalysis {
    bool is_pure_copy_region = true;
    bool may_wait = false;
    bool guarantees_commit = false;
    bool guarantees_wait_all = false;
  };

  struct VisitedStmtInfo {
    Stmt visited{ffi::UnsafeInit{}};
    CopyRegionAnalysis analysis;
    bool opens_copy_region{false};
    bool has_uncommitted_transfers{false};
  };

  static CopyRegionAnalysis
  MergeCopyRegionAnalysis(CopyRegionAnalysis a, const CopyRegionAnalysis &b) {
    a.is_pure_copy_region = a.is_pure_copy_region && b.is_pure_copy_region;
    a.may_wait = a.may_wait || b.may_wait;
    a.guarantees_commit = a.guarantees_commit || b.guarantees_commit;
    a.guarantees_wait_all = a.guarantees_wait_all || b.guarantees_wait_all;
    return a;
  }

  CopyRegionAnalysis AnalyzeCopyRegion(const Stmt &stmt) {
    CopyRegionAnalysis out;
    if (!stmt.defined()) {
      return out;
    }
    if (const auto *seq = stmt.as<SeqStmtObj>()) {
      for (const Stmt &s : seq->seq) {
        out = MergeCopyRegionAnalysis(out, AnalyzeCopyRegion(s));
      }
      return out;
    }
    if (const auto *ite = stmt.as<IfObj>()) {
      CopyRegionAnalysis then_info = AnalyzeCopyRegion(ite->then_case);
      CopyRegionAnalysis else_info;
      if (ite->else_case.has_value()) {
        else_info = AnalyzeCopyRegion(ite->else_case.value());
      }
      out.is_pure_copy_region =
          SideEffect(ite->condition) <= CallEffectKind::kPure &&
          then_info.is_pure_copy_region && else_info.is_pure_copy_region;
      out.may_wait = then_info.may_wait || else_info.may_wait;
      // A missing else is an executable path with no synchronization.
      out.guarantees_commit =
          then_info.guarantees_commit && else_info.guarantees_commit;
      out.guarantees_wait_all =
          then_info.guarantees_wait_all && else_info.guarantees_wait_all;
      return out;
    }
    if (const auto *eval = stmt.as<EvaluateObj>()) {
      if (eval->value.as<IntImmObj>() != nullptr) {
        return out;
      }
      const auto *call = eval->value.as<CallObj>();
      if (!call) {
        out.is_pure_copy_region = false;
        return out;
      }
      if (call->op.same_as(tl::ptx_cp_async_bytes()) ||
          call->op.same_as(tl::ptx_cp_async())) {
        return out;
      }
      if (call->op.same_as(tl::ptx_commit_group())) {
        out.guarantees_commit = true;
        return out;
      }
      if (call->op.same_as(tl::ptx_wait_group())) {
        out.may_wait = true;
        // wait_group(n > 0) may leave the newest injected copies pending.
        out.guarantees_wait_all =
            IsZero(analyzer_->Simplify(call->args[0].as_or_throw<PrimExpr>()));
        return out;
      }
      out.is_pure_copy_region = false;
      return out;
    }
    if (stmt.as<BindObj>()) {
      CopyRegionAnalysis out;
      out.is_pure_copy_region = false;
      return out;
    }
    if (const auto *region = stmt.as<RegionStmtObj>()) {
      // Scope wrappers are transparent only when their explicit operands are
      // pure. An operator region can read memory itself before its body.
      if (region->op->name != "tl.attr" &&
          !region->op.same_as(tl::lexical_scope_op()) &&
          region->op->name != "tirx.launch_thread") {
        out.is_pure_copy_region = false;
        return out;
      }
      for (const Expr &arg : region->args) {
        if (auto primitive = arg.as<PrimExpr>();
            primitive && SideEffect(*primitive) > CallEffectKind::kPure)
          out.is_pure_copy_region = false;
      }
      return MergeCopyRegionAnalysis(out, AnalyzeCopyRegion(region->body));
    }
    if (const auto *loop = stmt.as<ForObj>()) {
      out.is_pure_copy_region =
          SideEffect(loop->begin) <= CallEffectKind::kPure &&
          SideEffect(loop->end) <= CallEffectKind::kPure &&
          SideEffect(loop->step) <= CallEffectKind::kPure;
      out = MergeCopyRegionAnalysis(out, AnalyzeCopyRegion(loop->body));
      // Merely containing a wait or commit does not establish a boundary if
      // this loop can execute zero times. Keep may_wait for pre-commit safety.
      bool nonempty = analyzer_->CanProve(loop->begin < loop->end) &&
                      analyzer_->CanProve(loop->step > 0);
      out.guarantees_commit = out.guarantees_commit && nonempty;
      out.guarantees_wait_all = out.guarantees_wait_all && nonempty;
      return out;
    }
    out.is_pure_copy_region = false;
    return out;
  }

  VisitedStmtInfo VisitAndAnalyzeStmt(const Stmt &stmt) {
    pending_sync_copies_ = false;
    uncommitted_sync_copies_ = false;

    Stmt visited = Mutate(stmt).ValueOrUnchanged(stmt);
    VisitedStmtInfo out;
    out.visited = visited;
    out.analysis = AnalyzeCopyRegion(visited);
    out.opens_copy_region = pending_sync_copies_;
    out.has_uncommitted_transfers = uncommitted_sync_copies_;
    return out;
  }

  // ---- Synchronization emission helpers ----
  void AppendSyncVisibility(Array<Stmt> *seq, bool include_commit) const {
    if (include_commit) {
      seq->push_back(MakeCommitGroupStmt());
    }
    seq->push_back(MakeWaitGroupStmt(0));
  }

  // Note: AnalyzeCopyRegion replaces both the old `IsPureCopyRegion` and
  // `SummarizeAsyncIntrinsics` helpers to avoid redundant traversals.

  bool async_without_async_commit_wait_{false};
  int current_vectorized_lanes_{1};
  std::vector<ActiveVectorizedLoop> active_vectorized_loops_;
  sym::Analyzer analyzer_;
  bool injected_rocm_async_copy_{false};
  bool pending_sync_copies_{false};
  bool uncommitted_sync_copies_{false};
};

ROCmAsyncCopyInjectResult
InjectROCmAsyncCopy(const Stmt &body, bool async_without_async_commit_wait) {
  auto injector =
      make_object<ROCmAsyncCopyInjector>(async_without_async_commit_wait);
  Stmt injected = injector->Mutate(body).ValueOrUnchanged(body);
  return {injector->Finalize(injected), injector->InjectedROCmAsyncCopy()};
}

} // namespace tl
} // namespace tvm
