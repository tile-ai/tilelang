/*!
 * \file if_stmt_binding.cc
 * \brief Bind the If Stmt to each Stmt in SeqStmt
 */

#include "support/check.h"
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "../op/operator.h"
#include "../op/parallel.h"
#include "common/bind_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

class IfStmtBindingRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f, bool inline_replayable_binds) {
    auto rewriter = IfStmtBindingRewriter(inline_replayable_binds);
    for (const auto &[_, buffer] : f->buffer_map) {
      rewriter.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  explicit IfStmtBindingRewriter(bool inline_replayable_binds)
      : inline_replayable_binds_(inline_replayable_binds) {}

  class IfStmtAccessCollector : public StmtExprVisitor {
  public:
    explicit IfStmtAccessCollector(Map<Var, Buffer> buffer_data_to_buffer)
        : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)) {}

    std::pair<Array<BufferRegion>, Array<BufferRegion>>
    Collect(const Stmt &stmt) {
      this->VisitStmt(stmt);
      return {std::move(reads_), std::move(writes_)};
    }

  private:
    void AddRead(const Buffer &buffer) {
      reads_.push_back(BufferRegion::FullRegion(buffer));
    }

    void AddWrite(const Buffer &buffer) {
      writes_.push_back(BufferRegion::FullRegion(buffer));
    }

    void AddAccess(const Buffer &buffer, int access_mask) {
      if (access_mask & kAccessRead) {
        AddRead(buffer);
      }
      if (access_mask & kAccessWrite) {
        AddWrite(buffer);
      }
    }

    void VisitStmt_(const BufferStoreNode *op) final {
      AddWrite(op->buffer);
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const BufferLoadNode *op) final {
      AddRead(op->buffer);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const CallNode *op) final {
      if (auto tile_op = ParseOperator(GetRef<Call>(op)); tile_op.defined()) {
        if (const auto *parallel = tile_op.as<ParallelOpNode>()) {
          this->VisitStmt(parallel->GetRoot());
        } else {
          AccessRegions access = tile_op->GetAccessRegions();
          reads_.insert(reads_.end(), access.reads.begin(), access.reads.end());
          writes_.insert(writes_.end(), access.writes.begin(),
                         access.writes.end());
        }
        StmtExprVisitor::VisitExpr_(op);
        return;
      }
      if (op->op.same_as(builtin::address_of())) {
        if (const auto *load = op->args[0].as<BufferLoadNode>()) {
          AddRead(load->buffer);
        } else if (const auto *var_node = op->args[0].as<VarNode>()) {
          auto it = buffer_data_to_buffer_.find(GetRef<Var>(var_node));
          if (it != buffer_data_to_buffer_.end()) {
            AddRead((*it).second);
          }
        }
      } else if (op->op.same_as(tl::access_ptr())) {
        ICHECK_EQ(op->args.size(), 3U);
        const auto *load = op->args[0].as<BufferLoadNode>();
        ICHECK(load) << "tl.access_ptr base must be a BufferLoad";
        AddAccess(load->buffer, GetConservativeAccessMask(op->args[2]));
        for (const PrimExpr &index : load->indices) {
          this->VisitExpr(index);
        }
        if (load->predicate.defined()) {
          this->VisitExpr(load->predicate.value());
        }
        this->VisitExpr(op->args[1]);
        this->VisitExpr(op->args[2]);
        return;
      } else if (op->op.same_as(builtin::tvm_access_ptr())) {
        if (op->args.size() > 1) {
          if (const auto *var_node = op->args[1].as<VarNode>()) {
            auto it = buffer_data_to_buffer_.find(GetRef<Var>(var_node));
            if (it != buffer_data_to_buffer_.end()) {
              const int access_mask =
                  op->args.size() == 5U ? GetConservativeAccessMask(op->args[4])
                                        : kAccessReadWrite;
              AddAccess((*it).second, access_mask);
            }
          }
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    Map<Var, Buffer> buffer_data_to_buffer_;
    Array<BufferRegion> reads_;
    Array<BufferRegion> writes_;
  };

  static Stmt MakeSeq(Array<Stmt> seq, Span span = Span()) {
    ICHECK(!seq.empty());
    return seq.size() == 1 ? seq[0] : SeqStmt(std::move(seq), span);
  }

  std::pair<Array<BufferRegion>, Array<BufferRegion>>
  CollectStmtAccessRegions(const Stmt &stmt) const {
    IfStmtAccessCollector collector(buffer_data_to_buffer_);
    return collector.Collect(stmt);
  }

  BufferSet CollectWriteBuffers(const Array<Stmt> &stmts) const {
    BufferSet write_buffers;
    for (const Stmt &stmt : stmts) {
      auto [_, writes] = CollectStmtAccessRegions(stmt);
      for (const BufferRegion &write : writes) {
        write_buffers.insert(write->buffer);
      }
    }
    return write_buffers;
  }

  bool IsReplayableBindStmt(const Stmt &stmt,
                            const BufferSet &write_buffers) const {
    auto [reads, _] = CollectStmtAccessRegions(stmt);
    return IsReplayableScalarBind(stmt, reads, write_buffers);
  }

  PrimExpr
  RewriteWithReplayableBinds(const PrimExpr &expr,
                             const Map<Var, PrimExpr> &replayable_binds) const {
    return replayable_binds.empty() ? expr
                                    : tirx::Substitute(expr, replayable_binds);
  }

  Stmt
  RewriteWithReplayableBinds(const Stmt &stmt,
                             const Map<Var, PrimExpr> &replayable_binds) const {
    return replayable_binds.empty() ? stmt
                                    : tirx::Substitute(stmt, replayable_binds);
  }

  // Substituting accepted binds into a whole statement can still expand an
  // expression many times over (`x = x + x` chains under nested guards).
  // Abort the substitution as soon as the rewritten statement exceeds this
  // budget and let the caller fall back to the legacy binding for the guard.
  class ExpansionBudgetExceeded : public std::exception {
  public:
    const char *what() const noexcept override {
      return "replayable bind expansion budget exceeded";
    }
  };

  class BudgetedReplaySubstitute : public tirx::StmtExprMutator {
  public:
    BudgetedReplaySubstitute(const Map<Var, PrimExpr> &subs, int64_t limit)
        : subs_(subs), limit_(limit) {}

    PrimExpr VisitExpr(const PrimExpr &expr) final {
      if (++count_ > limit_) {
        throw ExpansionBudgetExceeded();
      }
      return tirx::StmtExprMutator::VisitExpr(expr);
    }

    PrimExpr VisitExpr_(const VarNode *op) final {
      auto it = subs_.find(GetRef<Var>(op));
      if (it != subs_.end()) {
        return VisitExpr((*it).second);
      }
      return GetRef<PrimExpr>(op);
    }

  private:
    const Map<Var, PrimExpr> &subs_;
    int64_t count_{0};
    int64_t limit_;
  };

  // Inlining a bind duplicates its value into every use. A chain such as
  // `x = x + x` therefore doubles the expression on each step and would grow
  // exponentially (issue #3700). Keep replayable inlining only while the
  // rewritten value stays small; once the budget is exceeded, the remaining
  // binds are emitted in the guarded lexical scope instead.
  static constexpr int64_t kMaxReplayableNodes = 128;
  static constexpr int64_t kMaxReplayableStatementNodes = 1024;

  class ReplayableNodeCounter : public tirx::ExprVisitor {
  public:
    explicit ReplayableNodeCounter(int64_t limit) : limit_(limit) {}

    void VisitExpr(const PrimExpr &expr) final {
      if (count_ > limit_) {
        return;
      }
      ++count_;
      tirx::ExprVisitor::VisitExpr(expr);
    }

    int64_t count() const { return count_; }

  private:
    int64_t count_{0};
    int64_t limit_;
  };

  static bool WithinReplayableBudget(const PrimExpr &value) {
    ReplayableNodeCounter counter(kMaxReplayableNodes);
    counter.VisitExpr(value);
    return counter.count() <= kMaxReplayableNodes;
  }

  Stmt RewriteBindWithReplayableBinds(
      const BindNode *bind, const Map<Var, PrimExpr> &replayable_binds) const {
    PrimExpr value = RewriteWithReplayableBinds(bind->value, replayable_binds);
    if (value.same_as(bind->value)) {
      return GetRef<Stmt>(bind);
    }
    return Bind(bind->var, value, bind->span);
  }

  Stmt GuardStmt(const PrimExpr &condition, const Stmt &stmt, Span span) const {
    return IfThenElse(condition, stmt, Stmt(), span);
  }

  Stmt GuardStmts(const Array<Stmt> &stmts, const PrimExpr &condition,
                  Span span) const {
    ICHECK(!stmts.empty());
    if (stmts.size() == 1) {
      return GuardStmt(condition, stmts[0], span);
    }

    Array<Stmt> guarded_stmts;
    PrimExpr guard_condition = condition;
    Var condition_snapshot("if_condition", condition.dtype(), span);
    Stmt condition_bind = Bind(condition_snapshot, condition, span);
    auto [condition_reads, _] = CollectStmtAccessRegions(condition_bind);
    const BufferSet write_buffers = CollectWriteBuffers(stmts);
    // Replay a read-only condition when no guarded body can change its inputs.
    // Otherwise preserve the original if's single condition evaluation.
    if (!IsReplayableScalarBind(condition_bind, condition_reads,
                                write_buffers)) {
      guarded_stmts.push_back(condition_bind);
      guard_condition = condition_snapshot;
    }
    for (const Stmt &stmt : stmts) {
      guarded_stmts.push_back(GuardStmt(guard_condition, stmt, span));
    }
    return MakeSeq(std::move(guarded_stmts), span);
  }

  Stmt BindIfStmtLegacy(const Stmt &body, const PrimExpr &condition,
                        Span span) const {
    if (auto seq_stmt = body.as<SeqStmtNode>()) {
      Array<Stmt> guarded_bodies;
      const size_t n = seq_stmt->seq.size();
      size_t i = 0;
      for (; i < n && !seq_stmt->seq[i].as<BindNode>(); ++i) {
        guarded_bodies.push_back(seq_stmt->seq[i]);
      }

      // A direct Bind is emitted as a C/CUDA declaration. Keep it and the
      // following statements in one lexical block, matching old LetStmt scope
      // semantics.
      if (i < n) {
        Array<Stmt> bind_scope;
        for (; i < n; ++i) {
          bind_scope.push_back(seq_stmt->seq[i]);
        }
        guarded_bodies.push_back(MakeSeq(std::move(bind_scope), span));
      }
      return GuardStmts(guarded_bodies, condition, span);
    }
    return GuardStmt(condition, body, span);
  }

  Stmt BindIfStmtWithReplayableBindInlining(const Stmt &body,
                                            const PrimExpr &condition,
                                            Span span) const {
    if (!inline_replayable_binds_) {
      return BindIfStmtLegacy(body, condition, span);
    }
    try {
      return ReplayableBindSeq(body, condition, span);
    } catch (const ExpansionBudgetExceeded &) {
      return BindIfStmtLegacy(body, condition, span);
    }
  }

  Stmt ReplayableBindSeq(const Stmt &body, const PrimExpr &condition,
                         Span span) const {
    auto seq_stmt = body.as<SeqStmtNode>();
    if (seq_stmt == nullptr) {
      return GuardStmt(condition, body, span);
    }

    const BufferSet write_buffers = CollectWriteBuffers(seq_stmt->seq);
    Map<Var, PrimExpr> replayable_binds;
    Array<Stmt> guarded_bodies;
    Array<Stmt> bind_scope;
    bool in_bind_scope = false;

    for (const Stmt &stmt : seq_stmt->seq) {
      if (!in_bind_scope) {
        if (const auto *bind = stmt.as<BindNode>()) {
          if (IsReplayableBindStmt(stmt, write_buffers)) {
            PrimExpr inlined =
                RewriteWithReplayableBinds(bind->value, replayable_binds);
            if (WithinReplayableBudget(inlined)) {
              replayable_binds.Set(bind->var, std::move(inlined));
              continue;
            }
            bind_scope.push_back(Bind(bind->var, inlined, bind->span));
            in_bind_scope = true;
            continue;
          }
          bind_scope.push_back(
              RewriteBindWithReplayableBinds(bind, replayable_binds));
          in_bind_scope = true;
          continue;
        }
        guarded_bodies.push_back(BudgetedReplaySubstitute(
            replayable_binds, kMaxReplayableStatementNodes)(stmt));
        continue;
      }
      bind_scope.push_back(BudgetedReplaySubstitute(
          replayable_binds, kMaxReplayableStatementNodes)(stmt));
    }

    if (!bind_scope.empty()) {
      guarded_bodies.push_back(MakeSeq(std::move(bind_scope), span));
    }
    if (guarded_bodies.empty()) {
      return Evaluate(0, span);
    }
    return GuardStmts(guarded_bodies, condition, span);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    auto condition = VisitExpr(op->condition);
    auto then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case = op->else_case;
    if (else_case.defined()) {
      Stmt new_else = VisitStmt(else_case.value());
      return IfThenElse(condition, then_case, new_else, op->span);
    }
    ICHECK(then_case.defined()) << "then_case must be defined";
    ICHECK(!else_case.defined()) << "else_case must be undefined";

    return BindIfStmtWithReplayableBindInlining(then_case, condition, op->span);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> seq;
    std::vector<std::pair<Var, Optional<Buffer>>> old_bindings;

    auto register_buffer = [&](const Buffer &buffer) {
      old_bindings.emplace_back(buffer->data,
                                buffer_data_to_buffer_.Get(buffer->data));
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    };

    for (auto stmt : op->seq) {
      Stmt new_stmt = VisitStmt(stmt);
      seq.push_back(new_stmt);
      if (const auto *alloc = new_stmt.as<AllocBufferNode>()) {
        register_buffer(alloc->buffer);
      } else if (const auto *decl = new_stmt.as<DeclBufferNode>()) {
        register_buffer(decl->buffer);
      }
    }

    for (auto it = old_bindings.rbegin(); it != old_bindings.rend(); ++it) {
      if (it->second.defined()) {
        buffer_data_to_buffer_.Set(it->first, it->second.value());
      } else {
        buffer_data_to_buffer_.erase(it->first);
      }
    }
    return SeqStmt(std::move(seq), op->span);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    std::vector<std::pair<Var, Optional<Buffer>>> old_bindings;
    for (const Buffer &buffer : op->alloc_buffers) {
      old_bindings.emplace_back(buffer->data,
                                buffer_data_to_buffer_.Get(buffer->data));
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }

    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));

    for (auto it = old_bindings.rbegin(); it != old_bindings.rend(); ++it) {
      if (it->second.defined()) {
        buffer_data_to_buffer_.Set(it->first, it->second.value());
      } else {
        buffer_data_to_buffer_.erase(it->first);
      }
    }
    return block;
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  bool inline_replayable_binds_{true};
};

using namespace tirx::transform;
tvm::transform::Pass IfStmtBinding() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    bool inline_replayable_binds =
        ctx->GetConfig<Bool>(kIfStmtBindingInlineReplayableBinds, Bool(true))
            .value();
    return IfStmtBindingRewriter::Substitute(f, inline_replayable_binds);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.IfStmtBinding", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.IfStmtBinding", IfStmtBinding);
}

} // namespace tl
} // namespace tvm
