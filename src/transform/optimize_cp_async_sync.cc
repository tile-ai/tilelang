/*!
 * \file optimize_cp_async_sync.cc
 * \brief Optimize explicit cp.async synchronization intrinsics.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <optional>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace transform {

class CPAsyncSyncOptimizer : public StmtExprMutator {
public:
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> visited;
    visited.reserve(op->seq.size());
    for (const Stmt &stmt : op->seq) {
      visited.push_back(this->VisitStmt(stmt));
    }

    enum class UncommittedState { kUnknown, kZero, kNonZero };

    UncommittedState uncommitted_state = UncommittedState::kUnknown;
    std::optional<int> last_wait_n;
    bool last_wait_dynamic = false;
    std::optional<int> outstanding_committed_groups = 0;

    Array<Stmt> simplified;
    simplified.reserve(visited.size());
    for (size_t stmt_idx = 0; stmt_idx < visited.size(); ++stmt_idx) {
      const Stmt &stmt = visited[stmt_idx];
      Stmt current = stmt;
      if (const auto *loop = current.as<ForNode>()) {
        bool has_following_wait = false;
        for (size_t j = stmt_idx + 1; j < visited.size(); ++j) {
          AsyncIntrinSummary summary = SummarizeAsyncIntrinsics(visited[j]);
          if (summary.wait > 0) {
            has_following_wait = true;
            break;
          }
        }
        if (has_following_wait) {
          current = MaybeRelaxLoopFirstWait(Downcast<For>(current));
        }
      }

      ClassifiedStmt cls = ClassifySimpleAsyncStmt(current);
      switch (cls.kind) {
      case AsyncStmtKind::kCPAsync:
        uncommitted_state = UncommittedState::kNonZero;
        simplified.push_back(current);
        break;
      case AsyncStmtKind::kCommit: {
        if (uncommitted_state == UncommittedState::kZero) {
          // Proven redundant commit: no cp.async issued since the last commit.
          break;
        }
        bool commit_has_new_cpasync =
            (uncommitted_state == UncommittedState::kNonZero);
        simplified.push_back(current);
        uncommitted_state = UncommittedState::kZero;
        if (outstanding_committed_groups.has_value() &&
            commit_has_new_cpasync) {
          outstanding_committed_groups =
              outstanding_committed_groups.value() + 1;
        } else {
          outstanding_committed_groups = std::nullopt;
        }
        last_wait_n.reset();
        last_wait_dynamic = false;
        break;
      }
      case AsyncStmtKind::kWaitStatic:
        if (!last_wait_dynamic && last_wait_n.has_value() &&
            cls.wait_n >= *last_wait_n) {
          // A weaker (or equal) wait is redundant when no commit happened in
          // between.
          break;
        }
        simplified.push_back(current);
        last_wait_n = cls.wait_n;
        last_wait_dynamic = false;
        if (outstanding_committed_groups.has_value()) {
          outstanding_committed_groups =
              std::min(outstanding_committed_groups.value(), cls.wait_n);
        }
        break;
      case AsyncStmtKind::kWaitDynamic:
        simplified.push_back(current);
        last_wait_n.reset();
        last_wait_dynamic = true;
        outstanding_committed_groups = std::nullopt;
        break;
      case AsyncStmtKind::kOther:
        simplified.push_back(current);
        if (ContainsAsyncIntrinsics(current)) {
          AsyncIntrinSummary summary = SummarizeAsyncIntrinsics(current);
          if (summary.cp_async > 0 && summary.commit == 0 &&
              summary.wait == 0) {
            // Preserve pending cp.async state across cp.async-only wrappers
            // (e.g. prologue loops before a standalone commit).
            uncommitted_state = UncommittedState::kNonZero;
            break;
          }
          // Cross this unknown boundary conservatively.
          uncommitted_state = UncommittedState::kUnknown;
          last_wait_n.reset();
          last_wait_dynamic = false;
          outstanding_committed_groups = std::nullopt;
        }
        break;
      }
    }

    if (simplified.empty()) {
      return Evaluate(0);
    }
    if (simplified.size() == 1) {
      return simplified[0];
    }
    return SeqStmt(simplified);
  }

private:
  enum class AsyncStmtKind {
    kOther,
    kCPAsync,
    kCommit,
    kWaitStatic,
    kWaitDynamic
  };

  struct ClassifiedStmt {
    AsyncStmtKind kind{AsyncStmtKind::kOther};
    int wait_n{0};
  };

  struct AsyncIntrinSummary {
    int cp_async = 0;
    int commit = 0;
    int wait = 0;
  };

  Stmt MakeStaticWaitStmtLike(const Stmt &stmt, int new_wait_n) const {
    const auto *eval = stmt.as<EvaluateNode>();
    if (!eval) {
      return stmt;
    }
    const auto *call = eval->value.as<CallNode>();
    if (!call || !IsWaitCall(call)) {
      return stmt;
    }

    DataType wait_dtype =
        call->args.empty() ? DataType::Int(32) : call->args[0].dtype();
    Array<PrimExpr> args{make_const(wait_dtype, new_wait_n)};
    return Evaluate(
        Call(call->dtype, call->op, args, call->annotations, call->span));
  }

  Stmt RewriteWaitStaticInSimpleWrapper(const Stmt &stmt, int new_wait_n,
                                        bool *changed) const {
    ClassifiedStmt cls = ClassifySimpleAsyncStmt(stmt);
    if (cls.kind != AsyncStmtKind::kWaitStatic) {
      return stmt;
    }

    if (const auto *eval = stmt.as<EvaluateNode>()) {
      const auto *call = eval->value.as<CallNode>();
      if (call && IsWaitCall(call)) {
        *changed = true;
        return MakeStaticWaitStmtLike(stmt, new_wait_n);
      }
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      Stmt new_body =
          RewriteWaitStaticInSimpleWrapper(let->body, new_wait_n, changed);
      if (*changed) {
        return LetStmt(let->var, let->value, new_body, let->span);
      }
      return stmt;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      Stmt new_body =
          RewriteWaitStaticInSimpleWrapper(attr->body, new_wait_n, changed);
      if (*changed) {
        return AttrStmt(attr->node, attr->attr_key, attr->value, new_body,
                        attr->span);
      }
      return stmt;
    }
    if (const auto *iff = stmt.as<IfThenElseNode>()) {
      if (!iff->else_case.defined()) {
        Stmt then_case = RewriteWaitStaticInSimpleWrapper(iff->then_case,
                                                          new_wait_n, changed);
        if (*changed) {
          return IfThenElse(iff->condition, then_case, Stmt(), iff->span);
        }
      }
      return stmt;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        Stmt inner =
            RewriteWaitStaticInSimpleWrapper(seq->seq[0], new_wait_n, changed);
        if (*changed) {
          return SeqStmt({inner});
        }
      }
      return stmt;
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      Stmt inner =
          RewriteWaitStaticInSimpleWrapper(block->body, new_wait_n, changed);
      if (*changed) {
        Block new_block = Downcast<Block>(stmt);
        BlockNode *n = new_block.CopyOnWrite();
        n->body = inner;
        return new_block;
      }
      return stmt;
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        Stmt inner = RewriteWaitStaticInSimpleWrapper(realize->block->body,
                                                      new_wait_n, changed);
        if (*changed) {
          Block block = realize->block;
          BlockNode *n = block.CopyOnWrite();
          n->body = inner;
          return BlockRealize(realize->iter_values, realize->predicate, block,
                              realize->span);
        }
      }
      return stmt;
    }

    return stmt;
  }

  Stmt MaybeRelaxLoopFirstWait(const For &loop) const {
    if (!loop.defined() || loop->kind != ForKind::kSerial) {
      return loop;
    }
    const auto *seq = loop->body.as<SeqStmtNode>();
    if (!seq) {
      return loop;
    }

    Array<Stmt> body = seq->seq;
    int cp_async_before_wait = 0;
    int commit_before_wait = 0;
    bool changed = false;
    for (int i = 0, n = static_cast<int>(body.size()); i < n; ++i) {
      ClassifiedStmt cls = ClassifySimpleAsyncStmt(body[i]);
      if (cls.kind == AsyncStmtKind::kCPAsync) {
        ++cp_async_before_wait;
        continue;
      }
      if (cls.kind == AsyncStmtKind::kCommit) {
        ++commit_before_wait;
        continue;
      }
      if (cls.kind == AsyncStmtKind::kWaitStatic) {
        if (cls.wait_n == 0 && cp_async_before_wait > 0 &&
            commit_before_wait > 0) {
          bool changed_wait = false;
          body.Set(i,
                   RewriteWaitStaticInSimpleWrapper(body[i], 1, &changed_wait));
          changed = changed || changed_wait;
        }
        break;
      }
      if (cls.kind == AsyncStmtKind::kWaitDynamic) {
        break;
      }
      if (cls.kind == AsyncStmtKind::kOther &&
          ContainsAsyncIntrinsics(body[i])) {
        AsyncIntrinSummary summary = SummarizeAsyncIntrinsics(body[i]);
        if (summary.cp_async > 0 && summary.commit == 0 && summary.wait == 0) {
          cp_async_before_wait += summary.cp_async;
          continue;
        }
        if (summary.cp_async == 0 && summary.commit > 0 && summary.wait == 0) {
          commit_before_wait += summary.commit;
          continue;
        }
        break;
      }
    }

    if (!changed) {
      return loop;
    }
    For new_loop = loop;
    ForNode *n = new_loop.CopyOnWrite();
    n->body = body.size() == 1 ? body[0] : SeqStmt(body);
    return new_loop;
  }

  bool IsCPAsyncCall(const CallNode *call) const {
    return call && (call->op.same_as(builtin::ptx_cp_async()) ||
                    call->op.same_as(tl::ptx_cp_async()));
  }

  bool IsCommitCall(const CallNode *call) const {
    return call && call->op.same_as(builtin::ptx_commit_group());
  }

  bool IsWaitCall(const CallNode *call) const {
    return call && call->op.same_as(builtin::ptx_wait_group());
  }

  bool ContainsAsyncIntrinsics(const Stmt &stmt) const {
    bool found = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (found) {
        return;
      }
      const auto *call = node.as<CallNode>();
      if (!call) {
        return;
      }
      if (IsCPAsyncCall(call) || IsCommitCall(call) || IsWaitCall(call)) {
        found = true;
      }
    });
    return found;
  }

  AsyncIntrinSummary SummarizeAsyncIntrinsics(const Stmt &stmt) const {
    AsyncIntrinSummary summary;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      const auto *call = node.as<CallNode>();
      if (!call) {
        return;
      }
      if (IsCPAsyncCall(call)) {
        ++summary.cp_async;
      } else if (IsCommitCall(call)) {
        ++summary.commit;
      } else if (IsWaitCall(call)) {
        ++summary.wait;
      }
    });
    return summary;
  }

  ClassifiedStmt ClassifySimpleAsyncStmt(const Stmt &stmt) const {
    if (const auto *let = stmt.as<LetStmtNode>()) {
      return ClassifySimpleAsyncStmt(let->body);
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return ClassifySimpleAsyncStmt(attr->body);
    }
    if (const auto *iff = stmt.as<IfThenElseNode>()) {
      if (!iff->else_case.defined()) {
        return ClassifySimpleAsyncStmt(iff->then_case);
      }
      return {};
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        return ClassifySimpleAsyncStmt(seq->seq[0]);
      }
      return {};
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return ClassifySimpleAsyncStmt(block->body);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        return ClassifySimpleAsyncStmt(realize->block->body);
      }
      return {};
    }

    const auto *eval = stmt.as<EvaluateNode>();
    if (!eval) {
      return {};
    }
    const auto *call = eval->value.as<CallNode>();
    if (!call) {
      return {};
    }
    if (IsCPAsyncCall(call)) {
      return {AsyncStmtKind::kCPAsync, 0};
    }
    if (IsCommitCall(call)) {
      return {AsyncStmtKind::kCommit, 0};
    }
    if (IsWaitCall(call)) {
      if (!call->args.empty()) {
        if (const auto *imm = call->args[0].as<IntImmNode>()) {
          return {AsyncStmtKind::kWaitStatic, static_cast<int>(imm->value)};
        }
      }
      return {AsyncStmtKind::kWaitDynamic, 0};
    }
    return {};
  }
};

tvm::transform::Pass OptimizeCPAsyncSync() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = CPAsyncSyncOptimizer()(std::move(fptr->body));
    return f;
  };
  return tvm::tir::transform::CreatePrimFuncPass(pass_func, 0,
                                                 "tl.OptimizeCPAsyncSync", {});
}

// Backward-compatible alias.
tvm::transform::Pass SimplifyCPAsyncSync() { return OptimizeCPAsyncSync(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.OptimizeCPAsyncSync",
                        OptimizeCPAsyncSync);
  refl::GlobalDef().def("tl.transform.SimplifyCPAsyncSync",
                        SimplifyCPAsyncSync);
}

} // namespace transform
} // namespace tl
} // namespace tvm
