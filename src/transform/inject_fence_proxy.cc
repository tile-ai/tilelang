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
  // NOTE(wt): We only need to inject fences before tma_store and WGMMA,
  // since tma_load and WGMMA contain implicit proxy fence after them
  if (call->op.same_as(tma_store()) ||  
      call->op.same_as(ptx_wgmma_ss()) || call->op.same_as(ptx_wgmma_rs())) {
    return true;
  }

  if (call->op.same_as(tl_gemm()) || call->op.same_as(tl_gemm_sp())) {
    // determine whether async wgmma is utilized
    std::ostringstream oss;
    oss << call->args[0].as<StringImmNode>()->value;
    return oss.str().find("wgmma") != std::string::npos;
  }

  return false;
}

// Known ops that must be treated as generic proxies (e.g. ldmatrix/stmatrix).
bool IsKnownGeneric(const CallNode *call) {
  if (call == nullptr) {
    return false;
  }
  return call->op.same_as(ptx_ldmatrix()) || call->op.same_as(ptx_stmatrix()) ||
         call->op.same_as(initialize_descriptor());
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
    ProxyKind entry_kind = ProxyKind::kUnknown; // Entry kind for parent-level checking
    ProxyKind prev_entry_kind = prev_stmt_kind_; // Entry kind from before this SeqStmt

    for (size_t i = 0; i < op->seq.size(); ++i) {
      const Stmt &stmt = op->seq[i];
      Stmt new_stmt = VisitStmt(stmt);
      
      // Get the entry kind of the visited statement
      ProxyKind current_entry_kind = GetEntryKind(new_stmt);
      // Get the exit kind
      ProxyKind current_exit_kind = prev_stmt_kind_;

      if (i == 0) {
        // First statement: record its entry kind for parent-level checking
        entry_kind = current_entry_kind;
        
        // Check if fence needed before this SeqStmt (parent will handle it)
        // We skip fence injection here
      } else {
        // Subsequent statements: check if fence needed between prev and current
        if (NeedsFence(prev_kind, current_entry_kind)) {
          Stmt fence = MakeFenceStmt();
          seq.push_back(fence);
          prev_kind = GetProxyKind(fence);
        }
      }

      seq.push_back(new_stmt);
      
      // Use GetProxyKind for the combined kind stored in the map
      ProxyKind stmt_combined_kind = GetProxyKind(new_stmt);
      sequence_kind = CombineProxy(sequence_kind, stmt_combined_kind);
      
      // Update prev_kind to the exit kind for next iteration
      prev_kind = current_exit_kind;
    }

    Stmt result = seq.size() == 1 ? seq[0] : SeqStmt(std::move(seq));
    SetProxyKind(result, sequence_kind);
    
    // Set entry kind of this SeqStmt (for parent's fence checking)
    SetEntryKind(result, entry_kind);
    
    // prev_stmt_kind_ already contains the exit kind (last statement's exit kind)
    
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
        // Remaining intrinsic and extern are marked as Generic.
        // We can now all extern as Generic, since gemm and gemm_sp are never
        // represented as call_extern nodes. They are call_intrin nodes and will
        // be handled by IsAsyncIntrinsic above.
        kind = ProxyKind::kGeneric;
      }
    }

    SetProxyKind(stmt, kind);
    SetEntryKind(stmt, kind); // Entry kind = exit kind for single statements
    prev_stmt_kind_ = kind;
    return stmt;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    SetProxyKind(stmt, ProxyKind::kGeneric);
    SetEntryKind(stmt, ProxyKind::kGeneric);
    prev_stmt_kind_ = ProxyKind::kGeneric;
    return stmt;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    const auto *node = stmt.as<IfThenElseNode>();
    ProxyKind kind = GetProxyKind(node->then_case);
    ProxyKind entry_kind = GetEntryKind(node->then_case);
    if (node->else_case.defined()) {
      kind = CombineProxy(kind, GetProxyKind(node->else_case.value()));
      entry_kind = CombineProxy(entry_kind, GetEntryKind(node->else_case.value()));
    }
    SetProxyKind(stmt, kind);
    SetEntryKind(stmt, entry_kind);
    prev_stmt_kind_ = kind;
    return stmt;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    const auto *node = stmt.as<AttrStmtNode>();
    ProxyKind body_kind = GetProxyKind(node->body);
    ProxyKind body_entry_kind = GetEntryKind(node->body);
    SetProxyKind(stmt, body_kind);
    SetEntryKind(stmt, body_entry_kind);
    prev_stmt_kind_ = body_kind;
    return stmt;
  }

  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    const auto *node = stmt.as<BlockRealizeNode>();
    ProxyKind kind = GetProxyKind(node->block);
    ProxyKind entry_kind = GetEntryKind(node->block);
    SetProxyKind(stmt, kind);
    SetEntryKind(stmt, entry_kind);
    prev_stmt_kind_ = kind;
    return stmt;
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    const auto *node = stmt.as<BlockNode>();
    ProxyKind kind = ProxyKind::kUnknown;
    ProxyKind entry_kind = ProxyKind::kUnknown;
    if (node->init.defined()) {
      kind = CombineProxy(kind, GetProxyKind(node->init.value()));
      entry_kind = CombineProxy(entry_kind, GetEntryKind(node->init.value()));
    }
    kind = CombineProxy(kind, GetProxyKind(node->body));
    if (entry_kind == ProxyKind::kUnknown) {
      // If no init, use body's entry kind
      entry_kind = GetEntryKind(node->body);
    } else {
      // If init exists, combine with body's combined kind
      entry_kind = CombineProxy(entry_kind, GetProxyKind(node->body));
    }
    SetProxyKind(stmt, kind);
    SetEntryKind(stmt, entry_kind);
    prev_stmt_kind_ = kind;
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
    ProxyKind body_entry_kind = GetEntryKind(node->body);
    SetProxyKind(stmt, body_kind);
    SetEntryKind(stmt, body_entry_kind); // Propagate entry kind from body
    prev_stmt_kind_ = body_kind;
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

  void SetEntryKind(const Stmt &stmt, ProxyKind kind) {
    entry_map_[stmt.get()] = kind;
  }

  ProxyKind GetEntryKind(const Stmt &stmt) const {
    if (!stmt.defined()) {
      return ProxyKind::kUnknown;
    }
    auto it = entry_map_.find(stmt.get());
    if (it == entry_map_.end()) {
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
  std::unordered_map<const StmtNode *, ProxyKind> entry_map_;
  ProxyKind prev_stmt_kind_ = ProxyKind::kUnknown;
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

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectFenceProxy", InjectFenceProxy);
});

} // namespace tl
} // namespace tvm
