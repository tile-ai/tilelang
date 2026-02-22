/*!
 * \file inject_fence_proxy.cc
 * \brief Inject proxy fences between generic and async proxies (sm90+)
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <cstdint>
#include <utility>

#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;
using runtime::StorageRank;
using runtime::StorageScope;
using tvm::transform::PassContext;

namespace {

/*!
 * \brief Proxy state tracking for Hopper's proxy switching rules.
 *
 * We track the *possible* last proxy state at a program point, because control
 * flow (if/loops) can merge different execution paths.
 *
 * - None   : no relevant proxy activity / state reset.
 * - Generic: last relevant op used the generic proxy path.
 * - Async  : last relevant op used the async proxy path.
 */
class ProxyStateSet {
public:
  static ProxyStateSet None() { return ProxyStateSet(kNone); }
  static ProxyStateSet Generic() { return ProxyStateSet(kGeneric); }
  static ProxyStateSet Async() { return ProxyStateSet(kAsync); }

  bool MayBeNone() const { return bits_ & kNone; }
  bool MayBeGeneric() const { return bits_ & kGeneric; }
  bool MayBeAsync() const { return bits_ & kAsync; }

  ProxyStateSet Union(ProxyStateSet other) const {
    return ProxyStateSet(bits_ | other.bits_);
  }
  ProxyStateSet &UnionInplace(ProxyStateSet other) {
    bits_ |= other.bits_;
    return *this;
  }

  bool operator==(const ProxyStateSet &other) const {
    return bits_ == other.bits_;
  }
  bool operator!=(const ProxyStateSet &other) const {
    return bits_ != other.bits_;
  }

private:
  explicit ProxyStateSet(uint8_t bits) : bits_(bits) {}

  static constexpr uint8_t kNone = 1 << 0;
  static constexpr uint8_t kGeneric = 1 << 1;
  static constexpr uint8_t kAsync = 1 << 2;
  uint8_t bits_{kNone};
};

enum class ProxyEvent : uint8_t {
  kNone,    // does not affect proxy state
  kGeneric, // generic proxy activity
  kAsync,   // async proxy activity
  kNeutral, // barrier/reset (e.g., fence.proxy.async)
};

inline bool IsFenceProxyAsyncCall(const CallNode *call) {
  return call && call->op.same_as(fence_proxy_async());
}

inline bool IsTMAStoreCall(const CallNode *call) {
  return call && call->op.same_as(tma_store());
}

inline bool IsTMAStoreArriveCall(const CallNode *call) {
  return call && call->op.same_as(tma_store_arrive());
}

inline bool IsTMAStoreWaitCall(const CallNode *call) {
  return call && call->op.same_as(tma_store_wait());
}

inline const CallNode *GetEvaluateCall(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    return eval->value.as<CallNode>();
  }
  return nullptr;
}

inline bool IsTMAStoreStmt(const Stmt &stmt) {
  return IsTMAStoreCall(GetEvaluateCall(stmt));
}

inline bool IsTMAStoreArriveStmt(const Stmt &stmt) {
  return IsTMAStoreArriveCall(GetEvaluateCall(stmt));
}

inline bool IsTMAStoreWaitStmt(const Stmt &stmt) {
  return IsTMAStoreWaitCall(GetEvaluateCall(stmt));
}

// Identify async intrinsics emitted by TileLang or TVM that require a fence
// when they follow generic proxies.
bool IsAsyncIntrinsic(const CallNode *call) {
  if (call == nullptr) {
    return false;
  }

  // TileLang async intrinsics
  if (call->op.same_as(tma_load()) || call->op.same_as(tma_load_im2col()) ||
      call->op.same_as(tma_store()) || call->op.same_as(tma_store_arrive()) ||
      call->op.same_as(tma_store_wait()) || call->op.same_as(ptx_wgmma_ss()) ||
      call->op.same_as(ptx_wgmma_rs()) ||
      call->op.same_as(ptx_tcgen05_mma_ss()) ||
      call->op.same_as(ptx_tcgen05_mma_ts())) {
    return true;
  }

  // PTX async copy intrinsics on SM90+ (cp.async.bulk family).
  if (call->op.same_as(builtin::ptx_cp_async_bulk())) {
    return true;
  }

  // wgmma async intrinsics
  if (call->op.same_as(tl_gemm()) || call->op.same_as(tl_gemm_sp())) {
    return true;
  }

  return false;
}

// Known ops that must be treated as generic proxies (e.g. ldmatrix/stmatrix).
bool IsKnownGeneric(const CallNode *call) {
  if (call == nullptr) {
    return false;
  }
  // Note: cp.async (classic) is *not* part of the async proxy on Hopper; treat
  // it as generic proxy traffic for fence injection purposes.
  return call->op.same_as(ptx_ldmatrix()) || call->op.same_as(ptx_stmatrix()) ||
         call->op.same_as(ptx_cp_async()) ||
         call->op.same_as(builtin::ptx_cp_async());
}

inline bool IsSharedPointerVar(const VarNode *var) {
  if (var == nullptr) {
    return false;
  }
  const auto *ptr_type = var->type_annotation.as<PointerTypeNode>();
  if (ptr_type == nullptr) {
    return false;
  }
  auto scope = StorageScope::Create(ptr_type->storage_scope);
  return scope.rank == StorageRank::kShared;
}

inline bool ExprContainsSharedPointerVar(const PrimExpr &expr) {
  bool has_shared = false;
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (has_shared) {
      return;
    }
    if (const auto *var = node.as<VarNode>()) {
      if (IsSharedPointerVar(var)) {
        has_shared = true;
      }
    }
  });
  return has_shared;
}

// Some "generic proxy" shared-memory stores are expressed as opaque calls
// (e.g. call_extern) that take pointers into shared memory via tvm_access_ptr
// or address_of.  Recognize these so we do not accidentally treat them as async
// and miss a required fence before the next async-proxy op.
bool CallMayWriteSharedMemory(const CallNode *call) {
  if (call == nullptr) {
    return false;
  }

  bool writes_shared = false;
  PostOrderVisit(tvm::ffi::GetRef<Call>(call), [&](const ObjectRef &node) {
    if (writes_shared) {
      return;
    }
    const auto *c = node.as<CallNode>();
    if (c == nullptr) {
      return;
    }

    if (c->op.same_as(builtin::tvm_access_ptr())) {
      // tvm_access_ptr(dtype, base_ptr, offset, extent, rw_mask)
      if (c->args.size() != 5) {
        // Unexpected signature; be conservative if we still see a shared base.
        if (c->args.size() >= 2 && ExprContainsSharedPointerVar(c->args[1])) {
          writes_shared = true;
        }
        return;
      }

      if (!ExprContainsSharedPointerVar(c->args[1])) {
        return;
      }

      // rw_mask: read(1), write(2), read/write(3).
      const auto *mask_imm = c->args[4].as<IntImmNode>();
      if (mask_imm == nullptr) {
        // Unknown mask; could include writes.
        writes_shared = true;
        return;
      }
      if (mask_imm->value & 2) {
        writes_shared = true;
      }
      return;
    }

    if (c->op.same_as(builtin::address_of())) {
      // address_of(BufferLoad(...)) returns a pointer that may be used for a
      // write in an opaque call.
      if (c->args.size() != 1) {
        return;
      }
      const auto *load = c->args[0].as<BufferLoadNode>();
      if (load == nullptr) {
        return;
      }
      auto scope = StorageScope::Create(GetPtrStorageScope(load->buffer->data));
      if (scope.rank == StorageRank::kShared) {
        writes_shared = true;
      }
      return;
    }
  });

  return writes_shared;
}

// Ops that should *not* be considered generic/async proxy traffic for the
// purpose of injecting fence.proxy.async.
bool IsNonProxyIntrinsic(const CallNode *call) {
  if (call == nullptr) {
    return false;
  }

  // Thread/barrier synchronization and barrier init fencing do not represent
  // shared-memory proxy traffic.
  if (call->op.same_as(builtin::tvm_storage_sync()) ||
      call->op.same_as(builtin::ptx_init_barrier_thread_count()) ||
      call->op.same_as(ptx_fence_barrier_init()) ||
      call->op.same_as(mbarrier_wait_parity()) ||
      call->op.same_as(mbarrier_expect_tx()) ||
      call->op.same_as(builtin::ptx_arrive_barrier()) ||
      call->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
      call->op.same_as(builtin::ptx_wait_barrier()) ||
      call->op.same_as(builtin::ptx_commit_group()) ||
      call->op.same_as(builtin::ptx_wait_group()) ||
      call->op.same_as(builtin::ptx_cp_async_barrier()) ||
      call->op.same_as(ptx_cp_async_barrier_noinc())) {
    return true;
  }

  // Warpgroup primitives (WGMMA scheduling) do not change shared-memory proxy
  // state; we want generic traffic before them to still require a fence before
  // the subsequent WGMMA.
  if (call->op.same_as(warpgroup_arrive()) ||
      call->op.same_as(warpgroup_commit_batch()) ||
      call->op.same_as(warpgroup_wait()) ||
      call->op.same_as(warpgroup_fence_operand())) {
    return true;
  }

  // Descriptor initialization only materializes register/local metadata and
  // does not represent shared-memory proxy traffic itself.
  if (call->op.same_as(initialize_wgmma_descriptor()) ||
      call->op.same_as(initialize_tcgen05_descriptor())) {
    return true;
  }

  // Register allocation hints are orthogonal to proxy state.
  if (call->op.same_as(set_max_nreg()) || call->op.same_as(no_set_max_nreg())) {
    return true;
  }

  return false;
}

ProxyEvent ClassifyCallProxyEvent(const CallNode *call) {
  if (call == nullptr) {
    return ProxyEvent::kNone;
  }
  if (IsFenceProxyAsyncCall(call)) {
    return ProxyEvent::kNeutral;
  }
  if (IsNonProxyIntrinsic(call)) {
    return ProxyEvent::kNone;
  }
  if (IsAsyncIntrinsic(call)) {
    return ProxyEvent::kAsync;
  }
  if (IsKnownGeneric(call)) {
    return ProxyEvent::kGeneric;
  }
  if (CallMayWriteSharedMemory(call)) {
    return ProxyEvent::kGeneric;
  }

  // Default: unknown/external ops do not affect proxy state. If you introduce a
  // new async-proxy intrinsic, add it to IsAsyncIntrinsic.
  return ProxyEvent::kNone;
}

inline void AppendFlattened(Array<Stmt> *out, const Stmt &stmt) {
  if (!stmt.defined()) {
    return;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &s : seq->seq) {
      out->push_back(s);
    }
    return;
  }
  out->push_back(stmt);
}

inline Stmt MakeFenceProxyAsyncStmt() {
  return Evaluate(Call(DataType::Handle(), fence_proxy_async(), {}));
}

inline Stmt MakeTMAStoreArriveStmt() {
  return Evaluate(Call(DataType::Handle(), tma_store_arrive(), {}));
}

inline Stmt MakeTMAStoreWaitStmt() {
  return Evaluate(Call(DataType::Handle(), tma_store_wait(), {}));
}

/*!
 * \brief Stateful rewriter that injects fence.proxy.async and normalizes
 *        tma_store synchronization.
 *
 * The key property is that we traverse statements in execution order and keep
 * a running (may-)state of the last proxy kind. Whenever we are about to issue
 * an async-proxy instruction with a possible preceding generic-proxy state, we
 * inject a fence.proxy.async right before that async instruction.
 */
class ProxyFenceRewriter : public StmtExprMutator {
public:
  static PrimFunc Apply(PrimFunc f) {
    if (!f->body.defined()) {
      return f;
    }
    ProxyFenceRewriter rewriter;
    // Start in the reset/unknown proxy state.
    auto res = rewriter.RewriteWithState(f->body, ProxyStateSet::None());
    f.CopyOnWrite()->body = res.stmt;
    return f;
  }

private:
  struct RewriteResult {
    Stmt stmt;
    ProxyStateSet out_state;
  };

  RewriteResult RewriteWithState(const Stmt &stmt, ProxyStateSet in_state) {
    ProxyStateSet saved = current_state_;
    current_state_ = in_state;
    Stmt mutated = VisitStmt(stmt);
    ProxyStateSet out_state = current_state_;
    current_state_ = saved;
    return {std::move(mutated), out_state};
  }

  Stmt InjectFenceIfNeededAndUpdateState(const Stmt &async_stmt) {
    if (current_state_.MayBeGeneric()) {
      // Transitioning from generic->async: insert a proxy fence.
      Array<Stmt> seq{MakeFenceProxyAsyncStmt(), async_stmt};
      current_state_ = ProxyStateSet::Async();
      return SeqStmt(std::move(seq));
    }
    current_state_ = ProxyStateSet::Async();
    return async_stmt;
  }

  Stmt ApplyProxyEvent(const Stmt &stmt, ProxyEvent event) {
    switch (event) {
    case ProxyEvent::kNone:
      return stmt;
    case ProxyEvent::kNeutral:
      current_state_ = ProxyStateSet::None();
      return stmt;
    case ProxyEvent::kGeneric:
      current_state_ = ProxyStateSet::Generic();
      return stmt;
    case ProxyEvent::kAsync:
      return InjectFenceIfNeededAndUpdateState(stmt);
    }
    return stmt;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    seq_depth_++;
    Array<Stmt> seq;
    seq.reserve(op->seq.size());

    for (int i = 0; i < static_cast<int>(op->seq.size()); ++i) {
      const Stmt &original = op->seq[i];
      const bool is_tma_store = IsTMAStoreStmt(original);
      const bool has_arrive = is_tma_store &&
                              (i + 1 < static_cast<int>(op->seq.size())) &&
                              IsTMAStoreArriveStmt(op->seq[i + 1]);
      const bool has_wait = is_tma_store &&
                            (i + 2 < static_cast<int>(op->seq.size())) &&
                            IsTMAStoreWaitStmt(op->seq[i + 2]);

      Stmt mutated = VisitStmt(original);
      AppendFlattened(&seq, mutated);

      // TMA stores must be followed by the arrive/wait pair. Inject it here so
      // we can avoid duplicates when the user already provided the handshake.
      if (is_tma_store && !(has_arrive && has_wait)) {
        AppendFlattened(&seq, VisitStmt(MakeTMAStoreArriveStmt()));
        AppendFlattened(&seq, VisitStmt(MakeTMAStoreWaitStmt()));
      }
    }

    seq_depth_--;
    if (seq.size() == 1) {
      return seq[0];
    }
    return SeqStmt(std::move(seq));
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    const auto *eval = stmt.as<EvaluateNode>();
    const auto *call = eval->value.as<CallNode>();

    // Standalone tma_store (not within a SeqStmt) still needs arrive/wait.
    if (seq_depth_ == 0 && IsTMAStoreCall(call)) {
      Array<Stmt> seq;
      if (current_state_.MayBeGeneric()) {
        seq.push_back(MakeFenceProxyAsyncStmt());
      }
      seq.push_back(stmt);
      seq.push_back(MakeTMAStoreArriveStmt());
      seq.push_back(MakeTMAStoreWaitStmt());
      current_state_ = ProxyStateSet::Async();
      return SeqStmt(std::move(seq));
    }

    ProxyEvent event = ClassifyCallProxyEvent(call);
    return ApplyProxyEvent(stmt, event);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    auto scope = StorageScope::Create(GetPtrStorageScope(op->buffer->data));
    if (scope.rank == StorageRank::kShared) {
      current_state_ = ProxyStateSet::Generic();
    }
    return stmt;
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = tvm::ffi::GetRef<Block>(op);
    auto *n = block.CopyOnWrite();
    // Block executes init (if any) before body.
    if (op->init.defined()) {
      n->init = VisitStmt(op->init.value());
    }
    n->body = VisitStmt(op->body);
    return block;
  }

  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    // A block realize is conditional on its predicate.
    ProxyStateSet entry = current_state_;
    auto block_res = RewriteWithState(op->block, entry);
    current_state_ = block_res.out_state;

    // If the predicate can be false, the block may not execute.
    PrimExpr predicate = VisitExpr(op->predicate);
    if (!is_one(predicate)) {
      current_state_ = current_state_.Union(entry);
    }

    Array<PrimExpr> iter_values;
    iter_values.reserve(op->iter_values.size());
    for (const PrimExpr &v : op->iter_values) {
      iter_values.push_back(VisitExpr(v));
    }
    return BlockRealize(iter_values, predicate,
                        Downcast<Block>(block_res.stmt));
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr cond = VisitExpr(op->condition);
    ProxyStateSet entry = current_state_;

    auto then_res = RewriteWithState(op->then_case, entry);

    Stmt else_stmt;
    ProxyStateSet else_out = entry;
    if (op->else_case.defined()) {
      auto else_res = RewriteWithState(op->else_case.value(), entry);
      else_stmt = else_res.stmt;
      else_out = else_res.out_state;
    }

    current_state_ = then_res.out_state.Union(else_out);
    return IfThenElse(cond, then_res.stmt, else_stmt);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    PrimExpr min = VisitExpr(op->min);
    PrimExpr extent = VisitExpr(op->extent);

    ProxyStateSet entry = current_state_;

    // Compute a conservative loop-header state that covers the first and all
    // subsequent iterations: S = entry ∪ Transfer(body, S).
    ProxyStateSet header = entry;
    RewriteResult body_res{op->body, entry};

    for (int iter = 0; iter < 8; ++iter) {
      body_res = RewriteWithState(op->body, header);
      ProxyStateSet next_header = entry.Union(body_res.out_state);
      if (next_header == header) {
        break;
      }
      header = next_header;
    }

    // Determine whether the loop may execute zero times.
    bool may_be_zero = true;
    if (const auto *imm = extent.as<IntImmNode>()) {
      may_be_zero = imm->value == 0;
    }

    current_state_ =
        may_be_zero ? entry.Union(body_res.out_state) : body_res.out_state;

    return For(op->loop_var, min, extent, op->kind, body_res.stmt,
               op->thread_binding, op->annotations);
  }

  Stmt VisitStmt_(const WhileNode *op) final {
    PrimExpr cond = VisitExpr(op->condition);
    ProxyStateSet entry = current_state_;

    // While may execute zero or more times; use the same header fixpoint.
    ProxyStateSet header = entry;
    RewriteResult body_res{op->body, entry};
    for (int iter = 0; iter < 8; ++iter) {
      body_res = RewriteWithState(op->body, header);
      ProxyStateSet next_header = entry.Union(body_res.out_state);
      if (next_header == header) {
        break;
      }
      header = next_header;
    }

    current_state_ = entry.Union(body_res.out_state);
    return While(cond, body_res.stmt);
  }

  ProxyStateSet current_state_{ProxyStateSet::None()};
  int seq_depth_{0};
};

} // namespace

tvm::transform::Pass InjectFenceProxy() {
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    f = ProxyFenceRewriter::Apply(f);
    return f;
  };
  return tir::transform::CreatePrimFuncPass(pass_func, 0, "tl.InjectFenceProxy",
                                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectFenceProxy", InjectFenceProxy);
}

} // namespace tl
} // namespace tvm
