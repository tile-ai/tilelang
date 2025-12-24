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

#include <unordered_map>
#include <utility>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;
using tvm::transform::PassContext;

// Tracks what kind of proxy activity a statement performs so we can decide when
// to inject fences while traversing the IR.
enum class ProxyKind : uint8_t {
  kUnknown,
  kGeneric,
  kAsync,
  kMixed,
  kNeutral, // Acts as a barrier and resets proxy state (e.g., fence
            // instructions)
};

namespace {

inline bool IsAsync(ProxyKind kind) { return kind == ProxyKind::kAsync; }
inline bool IsGeneric(ProxyKind kind) { return kind == ProxyKind::kGeneric; }

// Merge two proxy kinds to represent the aggregate behaviour of a compound
// node.
inline ProxyKind CombineProxy(ProxyKind a, ProxyKind b) {
  if (a == ProxyKind::kUnknown)
    return b;
  if (b == ProxyKind::kUnknown)
    return a;
  if (a == ProxyKind::kNeutral)
    return b;
  if (b == ProxyKind::kNeutral)
    return a;
  if (a == b)
    return a;
  return ProxyKind::kMixed;
}

// We only need a fence when transitioning from generic operations to async
// ones.
inline bool NeedsFence(ProxyKind prev, ProxyKind curr) {
  if (prev == ProxyKind::kUnknown || curr == ProxyKind::kUnknown)
    return false;
  if (prev == ProxyKind::kNeutral || curr == ProxyKind::kNeutral)
    return false;
  if (prev == ProxyKind::kMixed || curr == ProxyKind::kMixed)
    return false;
  return IsGeneric(prev) && IsAsync(curr);
}

inline bool IsFenceCall(const CallNode *call) {
  return call && call->op.same_as(fence_proxy_async());
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
      call->op.same_as(tma_store_wait()) ||
      call->op.same_as(ptx_cp_async_barrier_noinc()) ||
      call->op.same_as(ptx_wgmma_ss()) || call->op.same_as(ptx_wgmma_rs())) {
    return true;
  }

  // PTX async copy intrinsics
  if (call->op.same_as(builtin::ptx_cp_async()) ||
      call->op.same_as(builtin::ptx_cp_async_barrier()) ||
      call->op.same_as(builtin::ptx_cp_async_bulk())) {
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
  return call->op.same_as(ptx_ldmatrix()) || call->op.same_as(ptx_stmatrix()) ||
         call->op.same_as(initialize_wgmma_descriptor()) ||
         call->op.same_as(initialize_tcgen05_descriptor());
}

ProxyKind ProxyFromAttrValue(const ObjectRef &value) {
  if (const auto *str = value.as<StringImmNode>()) {
    if (str->value == "async") {
      return ProxyKind::kAsync;
    }
    if (str->value == "generic") {
      return ProxyKind::kGeneric;
    }
    if (str->value == "neutral") {
      return ProxyKind::kNeutral;
    }
  }
  return ProxyKind::kUnknown;
}

// TMA stores must be followed by the arrive/wait pair. We rewrite them as part
// of the pass to guarantee the proper synchronization semantics.
class TMAStoreSyncInjector : public StmtExprMutator {
public:
  static PrimFunc Apply(PrimFunc f) {
    if (!f->body.defined()) {
      return f;
    }
    auto injector = TMAStoreSyncInjector();
    f.CopyOnWrite()->body = injector(f->body);
    return f;
  }

private:
  Stmt operator()(const Stmt &stmt) { return StmtExprMutator::VisitStmt(stmt); }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    Stmt mutated = StmtExprMutator::VisitStmt_(op);
    const auto *node = mutated.as<EvaluateNode>();
    if (const auto *call = node->value.as<CallNode>()) {
      if (call->op.same_as(tma_store())) {
        Array<Stmt> seq;
        seq.push_back(mutated);
        seq.push_back(
            Evaluate(Call(DataType::Handle(), tma_store_arrive(), {})));
        seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_wait(), {})));
        return SeqStmt(std::move(seq));
      }
    }
    return mutated;
  }
};

// Main pass: track the proxy state while walking the IR and inject fences when
// switching from generic to async proxies.
class ProxyFenceInjector : public StmtMutator {
public:
  static PrimFunc Apply(PrimFunc f) {
    if (!f->body.defined()) {
      return f;
    }
    ProxyFenceInjector injector;
    f.CopyOnWrite()->body = injector.VisitStmt(f->body);
    return f;
  }

private:
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> seq;
    seq.reserve(op->seq.size());

    ProxyKind sequence_kind = ProxyKind::kUnknown;
    ProxyKind prev_kind = ProxyKind::kUnknown;

    for (const Stmt &stmt : op->seq) {
      Stmt new_stmt = VisitStmt(stmt);
      ProxyKind current_kind = GetProxyKind(new_stmt);

      if (!seq.empty() && NeedsFence(prev_kind, current_kind)) {
        Stmt fence = MakeFenceStmt();
        seq.push_back(fence);
        prev_kind = GetProxyKind(fence);
      }

      seq.push_back(new_stmt);
      sequence_kind = CombineProxy(sequence_kind, current_kind);
      prev_kind = current_kind;
    }

    Stmt result = seq.size() == 1 ? seq[0] : SeqStmt(std::move(seq));
    SetProxyKind(result, sequence_kind);
    return result;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    const auto *evaluate = stmt.as<EvaluateNode>();
    ProxyKind kind = ProxyKind::kGeneric;

    if (const auto *call = evaluate->value.as<CallNode>()) {
      if (IsFenceCall(call)) {
        kind = ProxyKind::kNeutral;
      } else if (IsAsyncIntrinsic(call)) {
        kind = ProxyKind::kAsync;
      } else if (IsKnownGeneric(call)) {
        kind = ProxyKind::kGeneric;
      } else {
        // We can now treat extern as Generic, since gemm and gemm_sp are never
        // represented as call_extern nodes. They are call_intrin nodes and will
        // be handled by IsAsyncIntrinsic above.
        kind = ProxyKind::kGeneric;
      }
    }

    SetProxyKind(stmt, kind);
    return stmt;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    SetProxyKind(stmt, ProxyKind::kGeneric);
    return stmt;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    const auto *node = stmt.as<IfThenElseNode>();
    ProxyKind kind = GetProxyKind(node->then_case);
    if (node->else_case.defined()) {
      kind = CombineProxy(kind, GetProxyKind(node->else_case.value()));
    }
    SetProxyKind(stmt, kind);
    return stmt;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    const auto *node = stmt.as<AttrStmtNode>();
    ProxyKind body_kind = GetProxyKind(node->body);
    SetProxyKind(stmt, body_kind);
    return stmt;
  }

  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    const auto *node = stmt.as<BlockRealizeNode>();
    SetProxyKind(stmt, GetProxyKind(node->block));
    return stmt;
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    const auto *node = stmt.as<BlockNode>();
    ProxyKind kind = ProxyKind::kUnknown;
    if (node->init.defined()) {
      kind = CombineProxy(kind, GetProxyKind(node->init.value()));
    }
    kind = CombineProxy(kind, GetProxyKind(node->body));
    SetProxyKind(stmt, kind);
    return stmt;
  }

  Stmt VisitStmt_(const ForNode *op) final { return VisitSingleBody(op); }
  Stmt VisitStmt_(const LetStmtNode *op) final { return VisitSingleBody(op); }
  Stmt VisitStmt_(const AssertStmtNode *op) final {
    return VisitSingleBody(op);
  }
  Stmt VisitStmt_(const WhileNode *op) final { return VisitSingleBody(op); }

  template <typename NodeType> Stmt VisitSingleBody(const NodeType *op) {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    const auto *node = stmt.as<NodeType>();
    ProxyKind body_kind = GetProxyKind(node->body);
    SetProxyKind(stmt, body_kind);
    return stmt;
  }

  void SetProxyKind(const Stmt &stmt, ProxyKind kind) {
    proxy_map_[stmt.get()] = kind;
  }

  ProxyKind GetProxyKind(const Stmt &stmt) const {
    if (!stmt.defined()) {
      return ProxyKind::kUnknown;
    }
    auto it = proxy_map_.find(stmt.get());
    if (it == proxy_map_.end()) {
      return ProxyKind::kUnknown;
    }
    return it->second;
  }

  Stmt MakeFenceStmt() {
    Stmt fence = Evaluate(Call(DataType::Handle(), fence_proxy_async(), {}));
    SetProxyKind(fence, ProxyKind::kNeutral);
    return fence;
  }

  std::unordered_map<const StmtNode *, ProxyKind> proxy_map_;
};

} // namespace

tvm::transform::Pass InjectFenceProxy() {
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    f = TMAStoreSyncInjector::Apply(f);
    f = ProxyFenceInjector::Apply(f);
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
