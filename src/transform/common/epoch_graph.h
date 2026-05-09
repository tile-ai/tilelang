/*
 * Sync-Epoch Graph for shared memory lifetime analysis.
 *
 * Phase 4 design (see daily-work dev notes ~1699). Replaces the legacy
 * 1-D `[start, end]` linear-interval lifetime model with a 2-D
 * `(epoch_id, per-epoch access set)` model that can express
 * "sync-isolated dead sub-intervals" inside an outer live interval.
 *
 * This header is intentionally header-only to match the style of the
 * neighboring `shared_access_analysis.h` and to avoid CMake churn.
 *
 * S1.2 scope:
 *   - Data structures: SyncEpoch / EpochEdge / Scope / EpochGraph
 *   - EpochGraphBuilder (StmtVisitor) that recognizes
 *       Evaluate(Call(builtin::tvm_storage_sync, "shared")) and
 *       Evaluate(Call(builtin::ptx_wait_group, depth)) as epoch
 *       boundaries, and handles For loop_body/loop_back/loop_exit and
 *       IfThenElse branch/branch_join edges.
 *   - EpochGraph::Dump emits `[MSMA-EPOCH-GRAPH]` lines (per-epoch
 *     filling will be extended in S1.3).
 *
 * The builder is intentionally conservative:
 *   - cp_async_wait (`ptx_wait_group`) is treated as a hard sync.
 *   - IfThenElse join takes the union of branch tails (no per-branch
 *     mutual-exclusion alias relaxation here; that is Phase 5).
 *   - PerEpochAccess is declared but NOT populated in S1.2; it will
 *     be filled in S1.3 once the graph is connected to the access
 *     analysis pipeline.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_EPOCH_GRAPH_H_
#define TVM_TL_TRANSFORM_COMMON_EPOCH_GRAPH_H_

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm::tl::epoch_graph {

/*! \brief Reason a sync edge separates two epochs. */
enum class SyncKind : uint8_t {
  kNone = 0,        //!< Boundary at scope start/end (no actual sync stmt).
  kHard = 1,        //!< __syncthreads / tvm_storage_sync("shared").
  kCpAsyncWait = 2, //!< ptx_wait_group / ptx_cp_async_barrier.
  kLoopBack = 3,    //!< Synthetic boundary on a loop back-edge.
  kLoopExit = 4,    //!< Synthetic boundary on a loop exit-edge.
  kBranchJoin = 5,  //!< Synthetic boundary at IfThenElse join.
};

inline const char *SyncKindName(SyncKind k) {
  switch (k) {
  case SyncKind::kNone:
    return "none";
  case SyncKind::kHard:
    return "hard";
  case SyncKind::kCpAsyncWait:
    return "cp_async_wait";
  case SyncKind::kLoopBack:
    return "loop_back";
  case SyncKind::kLoopExit:
    return "loop_exit";
  case SyncKind::kBranchJoin:
    return "branch_join";
  }
  return "?";
}

/*! \brief Kind of edge in the epoch graph. */
enum class EdgeKind : uint8_t {
  kSeq = 0,        //!< Sequential adjacency inside one scope.
  kBranch = 1,     //!< Parent → first epoch of then/else branch.
  kBranchJoin = 2, //!< Last epoch of then/else → join epoch in parent.
  kLoopBody = 3,   //!< Parent → first epoch of For body scope.
  kLoopBack = 4,   //!< Last body epoch → first body epoch.
  kLoopExit = 5,   //!< Last body epoch → first epoch after For.
};

inline const char *EdgeKindName(EdgeKind k) {
  switch (k) {
  case EdgeKind::kSeq:
    return "seq";
  case EdgeKind::kBranch:
    return "branch";
  case EdgeKind::kBranchJoin:
    return "branch_join";
  case EdgeKind::kLoopBody:
    return "loop_body";
  case EdgeKind::kLoopBack:
    return "loop_back";
  case EdgeKind::kLoopExit:
    return "loop_exit";
  }
  return "?";
}

/*! \brief Kind of control-flow scope. */
enum class ScopeKind : uint8_t {
  kRoot = 0,
  kForBody = 1,
  kIfThen = 2,
  kIfElse = 3,
};

inline const char *ScopeKindName(ScopeKind k) {
  switch (k) {
  case ScopeKind::kRoot:
    return "root";
  case ScopeKind::kForBody:
    return "for_body";
  case ScopeKind::kIfThen:
    return "if_then";
  case ScopeKind::kIfElse:
    return "if_else";
  }
  return "?";
}

/*! \brief One sync-isolated stmt segment. */
struct SyncEpoch {
  int epoch_id = -1;
  int scope_id = -1;
  /*! \brief Boundary kinds (purely descriptive). */
  SyncKind kind_in = SyncKind::kNone;
  SyncKind kind_out = SyncKind::kNone;
  /*! \brief First / last leaf stmt observed in this epoch (may be null). */
  const tir::StmtNode *span_lo = nullptr;
  const tir::StmtNode *span_hi = nullptr;
  /*! \brief Edge ids in EpochGraph::edges. */
  std::vector<int> in_edges;
  std::vector<int> out_edges;
};

/*! \brief Directed edge between two epochs. */
struct EpochEdge {
  int edge_id = -1;
  int from = -1;
  int to = -1;
  EdgeKind kind = EdgeKind::kSeq;
};

/*! \brief Control-flow scope (root, loop body, branch). */
struct Scope {
  int scope_id = -1;
  int parent_scope_id = -1;
  ScopeKind kind = ScopeKind::kRoot;
  /*! \brief Anchor stmt: ForNode for kForBody, IfThenElseNode for branches. */
  const tir::StmtNode *anchor = nullptr;
  /*! \brief Ordered list of epoch ids in this scope. */
  std::vector<int> epoch_ids;
};

/*!
 * \brief Per-(buffer, epoch) access summary. Populated in S1.3.
 *
 * `byte_range` is the union of byte intervals touched by all accesses
 * to the buffer within the epoch; an absent value means "unknown" and
 * must be treated as "covers the whole buffer" by callers.
 */
struct PerEpochAccess {
  bool touched = false;
  bool has_read = false;
  bool has_write = false;
  std::optional<std::pair<int64_t, int64_t>> byte_range;
};

/*! \brief The whole epoch graph for one PrimFunc body. */
struct EpochGraph {
  std::vector<SyncEpoch> epochs;
  std::vector<EpochEdge> edges;
  std::vector<Scope> scopes;
  /*! \brief stmt → epoch_id index for *leaf* stmts only. */
  std::unordered_map<const Object *, int> stmt_to_epoch;

  int RootScopeId() const { return scopes.empty() ? -1 : 0; }

  /*! \brief Lookup the epoch a stmt belongs to, or -1. */
  int EpochOf(const Object *s) const {
    auto it = stmt_to_epoch.find(s);
    return it == stmt_to_epoch.end() ? -1 : it->second;
  }

  /*! \brief Render a human-readable dump prefixed with `[MSMA-EPOCH-GRAPH]`. */
  void Dump(std::ostream &os) const {
    os << "[MSMA-EPOCH-GRAPH] scopes=" << scopes.size()
       << " epochs=" << epochs.size() << " edges=" << edges.size() << "\n";
    for (const Scope &sc : scopes) {
      os << "[MSMA-EPOCH-GRAPH] scope #" << sc.scope_id
         << " kind=" << ScopeKindName(sc.kind)
         << " parent=" << sc.parent_scope_id
         << " anchor=" << static_cast<const void *>(sc.anchor) << " epochs=[";
      for (size_t i = 0; i < sc.epoch_ids.size(); ++i) {
        if (i)
          os << ",";
        os << sc.epoch_ids[i];
      }
      os << "]\n";
    }
    for (const SyncEpoch &e : epochs) {
      os << "[MSMA-EPOCH-GRAPH] epoch #" << e.epoch_id
         << " scope=" << e.scope_id << " in=" << SyncKindName(e.kind_in)
         << " out=" << SyncKindName(e.kind_out)
         << " in_edges=" << e.in_edges.size()
         << " out_edges=" << e.out_edges.size() << "\n";
    }
    for (const EpochEdge &ed : edges) {
      os << "[MSMA-EPOCH-GRAPH] edge #" << ed.edge_id << " " << ed.from
         << " -> " << ed.to << " kind=" << EdgeKindName(ed.kind) << "\n";
    }
  }

  std::string DumpStr() const {
    std::ostringstream oss;
    Dump(oss);
    return oss.str();
  }
};

/*!
 * \brief Build an EpochGraph from a Stmt (typically a PrimFunc body).
 *
 * Usage:
 *   EpochGraphBuilder builder;
 *   EpochGraph g = builder.Build(stmt);
 */
class EpochGraphBuilder : public tir::StmtVisitor {
public:
  EpochGraph Build(const tir::Stmt &body) {
    g_ = EpochGraph{};
    cur_scope_id_ =
        OpenScope(ScopeKind::kRoot, /*parent=*/-1, /*anchor=*/nullptr);
    cur_epoch_id_ = OpenEpoch(cur_scope_id_, SyncKind::kNone);
    this->VisitStmt(body);
    CloseEpoch(cur_epoch_id_, SyncKind::kNone);
    return std::move(g_);
  }

protected:
  // Top-level dispatch hook: every visited Stmt gets attached to the
  // currently-open epoch *before* the type-specific visitor runs (which
  // may itself open/close epochs for control-flow nodes). This guarantees
  // that *every* stmt has a defined `EpochOf`.
  void VisitStmt(const tir::Stmt &s) final {
    if (s.defined()) {
      AttachLeaf(s.get());
    }
    tir::StmtVisitor::VisitStmt(s);
  }

  // ---- Stmt overrides ---------------------------------------------------

  void VisitStmt_(const tir::EvaluateNode *op) final {
    if (const auto *call = op->value.as<tir::CallNode>()) {
      if (IsHardSync(call)) {
        BoundaryAt(op, SyncKind::kHard);
        return;
      }
      if (IsCpAsyncWait(call)) {
        BoundaryAt(op, SyncKind::kCpAsyncWait);
        return;
      }
    }
  }

  void VisitStmt_(const tir::ForNode *op) final {
    int parent_scope = cur_scope_id_;
    int parent_epoch = cur_epoch_id_;
    int body_scope = OpenScope(ScopeKind::kForBody, parent_scope, op);
    int body_first = OpenEpoch(body_scope, SyncKind::kNone);
    AddEdge(parent_epoch, body_first, EdgeKind::kLoopBody);
    cur_scope_id_ = body_scope;
    cur_epoch_id_ = body_first;
    this->VisitStmt(op->body);
    int body_last = cur_epoch_id_;
    CloseEpoch(body_last, SyncKind::kLoopBack);
    AddEdge(body_last, body_first, EdgeKind::kLoopBack);
    // Restore parent scope and open a fresh epoch on loop_exit.
    cur_scope_id_ = parent_scope;
    int after = OpenEpoch(parent_scope, SyncKind::kLoopExit);
    AddEdge(body_last, after, EdgeKind::kLoopExit);
    cur_epoch_id_ = after;
  }

  void VisitStmt_(const tir::IfThenElseNode *op) final {
    int parent_scope = cur_scope_id_;
    int parent_epoch = cur_epoch_id_;

    // then branch
    int then_scope = OpenScope(ScopeKind::kIfThen, parent_scope, op);
    int then_first = OpenEpoch(then_scope, SyncKind::kNone);
    AddEdge(parent_epoch, then_first, EdgeKind::kBranch);
    cur_scope_id_ = then_scope;
    cur_epoch_id_ = then_first;
    this->VisitStmt(op->then_case);
    int then_last = cur_epoch_id_;
    CloseEpoch(then_last, SyncKind::kBranchJoin);

    // else branch (may be absent)
    int else_last = -1;
    if (op->else_case.defined()) {
      int else_scope = OpenScope(ScopeKind::kIfElse, parent_scope, op);
      int else_first = OpenEpoch(else_scope, SyncKind::kNone);
      AddEdge(parent_epoch, else_first, EdgeKind::kBranch);
      cur_scope_id_ = else_scope;
      cur_epoch_id_ = else_first;
      this->VisitStmt(op->else_case.value());
      else_last = cur_epoch_id_;
      CloseEpoch(else_last, SyncKind::kBranchJoin);
    }

    // join
    cur_scope_id_ = parent_scope;
    int join = OpenEpoch(parent_scope, SyncKind::kBranchJoin);
    AddEdge(then_last, join, EdgeKind::kBranchJoin);
    if (else_last >= 0) {
      AddEdge(else_last, join, EdgeKind::kBranchJoin);
    } else {
      // No else branch: the parent path also reaches the join.
      AddEdge(parent_epoch, join, EdgeKind::kBranchJoin);
    }
    cur_epoch_id_ = join;
  }

  // SeqStmt: visit children sequentially; epoch boundaries inside are
  // produced by EvaluateNode handlers, so just chain.
  void VisitStmt_(const tir::SeqStmtNode *op) final {
    for (const tir::Stmt &s : op->seq) {
      this->VisitStmt(s);
    }
  }

  // For all other stmt kinds, recurse via the default StmtVisitor traversal.
  // The blanket `VisitStmt` override above already attached them as leaves.

private:
  static bool IsHardSync(const tir::CallNode *call) {
    if (!call)
      return false;
    if (!call->op.same_as(tir::builtin::tvm_storage_sync()))
      return false;
    if (call->args.empty())
      return false;
    const auto *scope_str = call->args[0].as<tir::StringImmNode>();
    return scope_str != nullptr && scope_str->value == "shared";
  }
  static bool IsCpAsyncWait(const tir::CallNode *call) {
    if (!call)
      return false;
    return call->op.same_as(tir::builtin::ptx_wait_group()) ||
           call->op.same_as(tir::builtin::ptx_cp_async_barrier());
  }

  int OpenScope(ScopeKind kind, int parent, const tir::StmtNode *anchor) {
    int sid = static_cast<int>(g_.scopes.size());
    Scope sc;
    sc.scope_id = sid;
    sc.parent_scope_id = parent;
    sc.kind = kind;
    sc.anchor = anchor;
    g_.scopes.push_back(std::move(sc));
    return sid;
  }

  int OpenEpoch(int scope_id, SyncKind kind_in) {
    int eid = static_cast<int>(g_.epochs.size());
    SyncEpoch e;
    e.epoch_id = eid;
    e.scope_id = scope_id;
    e.kind_in = kind_in;
    g_.epochs.push_back(std::move(e));
    g_.scopes.at(scope_id).epoch_ids.push_back(eid);
    return eid;
  }

  void CloseEpoch(int epoch_id, SyncKind kind_out) {
    g_.epochs.at(epoch_id).kind_out = kind_out;
  }

  int AddEdge(int from, int to, EdgeKind kind) {
    int id = static_cast<int>(g_.edges.size());
    EpochEdge ed;
    ed.edge_id = id;
    ed.from = from;
    ed.to = to;
    ed.kind = kind;
    g_.edges.push_back(ed);
    g_.epochs.at(from).out_edges.push_back(id);
    g_.epochs.at(to).in_edges.push_back(id);
    return id;
  }

  /*! \brief Open a new epoch in the current scope right after a sync stmt. */
  void BoundaryAt(const tir::StmtNode *sync_stmt, SyncKind kind) {
    AttachLeaf(sync_stmt); // sync stmt belongs to the *outgoing* epoch
    CloseEpoch(cur_epoch_id_, kind);
    int prev = cur_epoch_id_;
    int next = OpenEpoch(cur_scope_id_, kind);
    AddEdge(prev, next, EdgeKind::kSeq);
    cur_epoch_id_ = next;
  }

  /*! \brief Record that `s` is a leaf stmt of the current epoch. */
  void AttachLeaf(const tir::StmtNode *s) {
    if (!s)
      return;
    g_.stmt_to_epoch.emplace(s, cur_epoch_id_);
    SyncEpoch &e = g_.epochs.at(cur_epoch_id_);
    if (e.span_lo == nullptr)
      e.span_lo = s;
    e.span_hi = s;
  }

  EpochGraph g_;
  int cur_scope_id_ = -1;
  int cur_epoch_id_ = -1;
};

// ---------------------------------------------------------------------------
// Per-epoch liveness analysis
// ---------------------------------------------------------------------------
//
// Liveness is the *intersection* of two dataflow problems:
//
//   1. Backward "use-reaches": standard liveness
//        live_out[E] = ⋃_{succ ∈ out(E)} live_in[succ]
//        live_in[E]  = use[E] ∪ (live_out[E] − def[E])
//
//   2. Forward "def-reaches": does some def reach E from above?
//        reach_in[E]  = ⋃_{pred ∈ in(E)} reach_out[pred]
//        reach_out[E] = reach_in[E] ∪ def[E]
//      (no kill: shared memory keeps the value until next def, which
//      simply replaces it; either way some def has reached.)
//
// Effective liveness:
//        live_in[E]  &= reach_in[E]
//        live_out[E] &= reach_out[E]
//
// The intersection is necessary because IfThenElse without `else` adds a
// "skip" edge from parent to join. Backward dataflow alone then propagates
// liveness from a use *past* the conditionally-guarded def, all the way
// back to epoch 0. For shared memory there is no meaningful pre-state, so
// the buffer cannot be live before any def reaches it. The forward
// reaching-def pass clamps liveness to the correct interval.
//
// `kLoopBack` edges are followed in *both* directions naturally: backward
// liveness propagates uses to prior iterations; forward reaching-defs
// propagates writes to later iterations. Loop-carried buffers (e.g. a
// cp_async pipeline writing in iter k, reading in iter k+1) remain live
// across the loop tail.

/*! \brief Read/write summary for one (buffer, epoch) cell. */
struct EpochAccess {
  bool def = false; //!< buffer is written in this epoch.
  bool use = false; //!< buffer is read in this epoch.
};

/*! \brief Liveness summary for one (buffer, epoch) cell. */
struct EpochLiveness {
  bool live_in = false;
  bool live_out = false;
  bool Live() const { return live_in || live_out; }
};

/*!
 * \brief Compute per-(buffer, epoch) liveness via backward dataflow.
 *
 * The buffer key type is left as a template parameter so callers can use
 * `std::string`, `const tir::VarNode*`, etc. without forcing a copy.
 *
 * Complexity: O(B · (E + edges) · iters) where iters is bounded by graph
 * depth in practice. Each buffer is solved independently.
 */
template <typename BufferId, typename Hash = std::hash<BufferId>,
          typename Eq = std::equal_to<BufferId>>
inline std::unordered_map<BufferId, std::unordered_map<int, EpochLiveness>,
                          Hash, Eq>
ComputePerEpochLiveness(
    const EpochGraph &g,
    const std::unordered_map<BufferId, std::unordered_map<int, EpochAccess>,
                             Hash, Eq> &access) {
  std::unordered_map<BufferId, std::unordered_map<int, EpochLiveness>, Hash, Eq>
      result;
  const int num_epochs = static_cast<int>(g.epochs.size());
  for (const auto &kv : access) {
    const BufferId &buf = kv.first;
    const auto &per_epoch = kv.second;
    // Per-buffer working arrays indexed by epoch_id.
    std::vector<EpochAccess> A(num_epochs);
    std::vector<EpochLiveness> L(num_epochs);
    for (const auto &p : per_epoch) {
      if (p.first >= 0 && p.first < num_epochs) {
        A[p.first] = p.second;
      }
    }
    // ---- Forward reaching-def fixed point. ----
    // R_in[e] = OR over preds of R_out[pred]; R_out[e] = R_in[e] || def[e].
    std::vector<bool> R_in(num_epochs, false);
    std::vector<bool> R_out(num_epochs, false);
    {
      std::vector<bool> in_fwl(num_epochs, false);
      std::vector<int> fwl;
      for (int e = 0; e < num_epochs; ++e) {
        if (A[e].def) {
          R_out[e] = true;
          fwl.push_back(e);
          in_fwl[e] = true;
        }
      }
      while (!fwl.empty()) {
        int e = fwl.back();
        fwl.pop_back();
        in_fwl[e] = false;
        for (int eid : g.epochs[e].out_edges) {
          int succ = g.edges[eid].to;
          if (!R_in[succ]) {
            R_in[succ] = true;
            bool new_out = R_in[succ] || A[succ].def;
            if (new_out && !R_out[succ]) {
              R_out[succ] = true;
              if (!in_fwl[succ]) {
                fwl.push_back(succ);
                in_fwl[succ] = true;
              }
            } else if (!in_fwl[succ]) {
              // R_in changed; succ's successors may need refresh even if
              // R_out is unchanged (e.g. def already true here).
              fwl.push_back(succ);
              in_fwl[succ] = true;
            }
          }
        }
      }
    }
    // ---- Backward use-reaches fixed point. ----
    // Worklist of epochs whose live_out may need recomputation.
    // Seed with epochs that have a use (these can immediately set
    // their own live_in to true).
    std::vector<bool> in_wl(num_epochs, false);
    std::vector<int> wl;
    for (int e = 0; e < num_epochs; ++e) {
      if (A[e].use) {
        wl.push_back(e);
        in_wl[e] = true;
      }
    }
    while (!wl.empty()) {
      int e = wl.back();
      wl.pop_back();
      in_wl[e] = false;
      // live_out[e] = ⋃ live_in[succ]
      bool new_live_out = false;
      for (int eid : g.epochs[e].out_edges) {
        int succ = g.edges[eid].to;
        if (L[succ].live_in) {
          new_live_out = true;
          break;
        }
      }
      // live_in[e] = use[e] ∪ (live_out[e] − def[e])
      bool new_live_in = A[e].use || (new_live_out && !A[e].def);
      bool changed_in = (new_live_in != L[e].live_in);
      bool changed_out = (new_live_out != L[e].live_out);
      L[e].live_in = new_live_in;
      L[e].live_out = new_live_out;
      if (changed_in || changed_out) {
        for (int eid : g.epochs[e].in_edges) {
          int pred = g.edges[eid].from;
          if (!in_wl[pred]) {
            wl.push_back(pred);
            in_wl[pred] = true;
          }
        }
      }
    }
    auto &dst = result[buf];
    for (int e = 0; e < num_epochs; ++e) {
      // Intersect backward use-reach with forward def-reach.
      L[e].live_in = L[e].live_in && R_in[e];
      L[e].live_out = L[e].live_out && R_out[e];
      if (L[e].Live()) {
        dst[e] = L[e];
      }
    }
  }
  return result;
}

/*! \brief Render liveness as `[MSMA-EPOCH-LIVE] epoch=N buf=B in=0/1 out=0/1`.
 */
template <typename BufferId, typename Hash, typename Eq>
inline void DumpEpochLiveness(
    std::ostream &os,
    const std::unordered_map<BufferId, std::unordered_map<int, EpochLiveness>,
                             Hash, Eq> &liveness) {
  // Group by epoch for readability.
  std::map<int, std::vector<std::pair<BufferId, EpochLiveness>>> by_epoch;
  for (const auto &kv : liveness) {
    for (const auto &p : kv.second) {
      by_epoch[p.first].push_back({kv.first, p.second});
    }
  }
  for (const auto &kv : by_epoch) {
    for (const auto &p : kv.second) {
      os << "[MSMA-EPOCH-LIVE] epoch=" << kv.first << " buf=" << p.first
         << " in=" << (p.second.live_in ? 1 : 0)
         << " out=" << (p.second.live_out ? 1 : 0) << "\n";
    }
  }
}

} // namespace tvm::tl::epoch_graph

#endif // TVM_TL_TRANSFORM_COMMON_EPOCH_GRAPH_H_
