/*!
 * \file if_stmt_binding.cc
 * \brief Bind the If Stmt to each Stmt in SeqStmt
 */

#include "support/check.h"
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "pipeline/access_analysis.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

class IfStmtBindingRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f, bool inline_replayable_binds) {
    Target target;
    if (auto target_attr = f->GetAttr<Target>(tvm::attr::kTarget)) {
      target = target_attr.value();
    }
    auto rewriter = IfStmtBindingRewriter(target, inline_replayable_binds);
    for (const auto &[_, buffer] : f->buffer_map) {
      rewriter.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  IfStmtBindingRewriter(Target target, bool inline_replayable_binds)
      : target_(std::move(target)),
        inline_replayable_binds_(inline_replayable_binds) {}

  static Stmt MakeSeq(Array<Stmt> seq) {
    ICHECK(!seq.empty());
    return seq.size() == 1 ? seq[0] : SeqStmt(std::move(seq));
  }

  std::pair<Array<BufferRegion>, Array<BufferRegion>>
  CollectStmtAccessRegions(const Stmt &stmt) const {
    auto collector = BufferRegionCollector(buffer_data_to_buffer_, target_);
    collector(stmt);
    return {collector.GetReads(), collector.GetWrites()};
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

  Stmt BindIfStmtLegacy(const Stmt &body, const PrimExpr &condition,
                        Span span) const {
    if (auto seq_stmt = body.as<SeqStmtNode>()) {
      Array<Stmt> seq;
      const size_t n = seq_stmt->seq.size();
      size_t i = 0;
      for (; i < n && !seq_stmt->seq[i].as<BindNode>(); ++i) {
        seq.push_back(GuardStmt(condition, seq_stmt->seq[i], span));
      }

      // A direct Bind is emitted as a C/CUDA declaration. Keep it and the
      // following statements in one lexical block, matching old LetStmt scope
      // semantics.
      if (i < n) {
        Array<Stmt> bind_scope;
        for (; i < n; ++i) {
          bind_scope.push_back(seq_stmt->seq[i]);
        }
        seq.push_back(
            GuardStmt(condition, MakeSeq(std::move(bind_scope)), span));
      }
      return MakeSeq(std::move(seq));
    }
    return GuardStmt(condition, body, span);
  }

  Stmt BindIfStmtWithReplayableBindInlining(const Stmt &body,
                                            const PrimExpr &condition,
                                            Span span) const {
    if (!inline_replayable_binds_) {
      return BindIfStmtLegacy(body, condition, span);
    }
    auto seq_stmt = body.as<SeqStmtNode>();
    if (seq_stmt == nullptr) {
      return GuardStmt(condition, body, span);
    }

    const BufferSet write_buffers = CollectWriteBuffers(seq_stmt->seq);
    Map<Var, PrimExpr> replayable_binds;
    Array<Stmt> guarded_stmts;
    Array<Stmt> bind_scope;
    bool in_bind_scope = false;

    for (const Stmt &stmt : seq_stmt->seq) {
      if (!in_bind_scope) {
        if (const auto *bind = stmt.as<BindNode>()) {
          if (IsReplayableBindStmt(stmt, write_buffers)) {
            replayable_binds.Set(bind->var, RewriteWithReplayableBinds(
                                                bind->value, replayable_binds));
            continue;
          }
          bind_scope.push_back(
              RewriteBindWithReplayableBinds(bind, replayable_binds));
          in_bind_scope = true;
          continue;
        }
        guarded_stmts.push_back(GuardStmt(
            condition, RewriteWithReplayableBinds(stmt, replayable_binds),
            span));
        continue;
      }
      bind_scope.push_back(RewriteWithReplayableBinds(stmt, replayable_binds));
    }

    if (!bind_scope.empty()) {
      guarded_stmts.push_back(
          GuardStmt(condition, MakeSeq(std::move(bind_scope)), span));
    }
    if (guarded_stmts.empty()) {
      return Evaluate(0, span);
    }
    return MakeSeq(std::move(guarded_stmts));
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
    return SeqStmt(std::move(seq));
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
  Target target_;
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
