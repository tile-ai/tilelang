/*!
 * \file if_stmt_binding.cc
 * \brief Merge the If Stmt in SeqStmt
 */

#include "merge_if_stmt.h"
#include "support/check.h"

#include <unordered_set>

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

using BufferDataVarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

struct BufferAccesses {
  BufferDataVarSet reads;
  BufferDataVarSet writes;
};

class BufferAccessCollector : public StmtExprVisitor {
public:
  static BufferAccesses Collect(const Array<Stmt> &statements) {
    BufferAccesses accesses;
    BufferAccessCollector collector(&accesses);
    for (const Stmt &stmt : statements) {
      collector(stmt);
    }
    return accesses;
  }

  static BufferAccesses Collect(const PrimExpr &expr) {
    BufferAccesses accesses;
    BufferAccessCollector collector(&accesses);
    collector(expr);
    return accesses;
  }

private:
  static constexpr int kReadAccessMask = 1;
  static constexpr int kWriteAccessMask = 2;

  explicit BufferAccessCollector(BufferAccesses *accesses)
      : accesses_(accesses) {}

  void AddAccess(const Var &data, const PrimExpr &mask) {
    const auto *flag = mask.as<IntImmNode>();
    if (!flag || (flag->value & kReadAccessMask) != 0) {
      accesses_->reads.insert(data);
    }
    if (!flag || (flag->value & kWriteAccessMask) != 0) {
      accesses_->writes.insert(data);
    }
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    accesses_->writes.insert(op->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    accesses_->reads.insert(op->buffer->data);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5U);
      const auto *data = op->args[1].as<VarNode>();
      ICHECK(data) << "tvm_access_ptr data argument must be a Var";
      AddAccess(GetRef<Var>(data), op->args[4]);
    } else if (op->op.same_as(tl::access_ptr())) {
      ICHECK_EQ(op->args.size(), 3U);
      const auto *load = op->args[0].as<BufferLoadNode>();
      ICHECK(load) << "tl.access_ptr base must be a BufferLoad";
      AddAccess(load->buffer->data, op->args[2]);
    } else if (op->op.same_as(builtin::address_of())) {
      ICHECK_EQ(op->args.size(), 1U);
      const auto *load = op->args[0].as<BufferLoadNode>();
      ICHECK(load) << "address_of argument must be a BufferLoad";
      // Without an access mask, conservatively assume the pointer may be read
      // or written by its enclosing call.
      accesses_->reads.insert(load->buffer->data);
      accesses_->writes.insert(load->buffer->data);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  BufferAccesses *accesses_;
};

class MergeIfStmtRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    f.CopyOnWrite()->body = MergeIfStmtRewriter::Apply(f->body);
    return f;
  }

  static Stmt Apply(Stmt stmt) {
    auto rewriter = MergeIfStmtRewriter();
    return rewriter(stmt);
  }

private:
  MergeIfStmtRewriter() = default;

  void FlattenAppend(const Stmt &s, Array<Stmt> *out) {
    if (const auto *seq = s.as<SeqStmtNode>()) {
      for (const Stmt &e : seq->seq) {
        FlattenAppend(e, out);
      }
    } else {
      out->push_back(s);
    }
  }

  bool BodyWritesConditionBuffer(const PrimExpr &condition,
                                 const Array<Stmt> &prior_bodies) {
    BufferAccesses body_accesses = BufferAccessCollector::Collect(prior_bodies);
    if (body_accesses.writes.empty()) {
      return false;
    }

    BufferAccesses condition_accesses =
        BufferAccessCollector::Collect(condition);
    for (const Var &read : condition_accesses.reads) {
      if (body_accesses.writes.count(read)) {
        return true;
      }
    }
    return false;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    // First, recursively flatten nested SeqStmt so that
    //   SeqStmt{ if, SeqStmt{ if, SeqStmt{ if } } }
    // becomes a single-level sequence of [if, if, if].
    Array<Stmt> flat_seq;
    for (const Stmt &stmt : op->seq) {
      Stmt new_stmt = this->VisitStmt(stmt);
      FlattenAppend(new_stmt, &flat_seq);
    }

    // Then, merge consecutive IfThenElse (without else) that share the same
    // condition.
    Array<Stmt> new_seq;
    PrimExpr current_condition;
    Array<Stmt> current_if_bodies;

    for (const Stmt &stmt : flat_seq) {
      if (const auto *if_node = stmt.as<IfThenElseNode>()) {
        if (!if_node->else_case.defined()) {
          if (current_condition.defined() &&
              ExprDeepEqual()(current_condition, if_node->condition) &&
              !BodyWritesConditionBuffer(current_condition,
                                         current_if_bodies)) {
            current_if_bodies.push_back(if_node->then_case);
            continue;
          } else {
            if (!current_if_bodies.empty()) {
              auto if_stmt =
                  IfThenElse(current_condition,
                             current_if_bodies.size() == 1
                                 ? current_if_bodies[0]
                                 : this->VisitStmt(SeqStmt(current_if_bodies)),
                             Stmt());
              new_seq.push_back(if_stmt);
              current_if_bodies.clear();
            }

            current_condition = if_node->condition;
            current_if_bodies.push_back(if_node->then_case);
            continue;
          }
        }
      }

      if (!current_if_bodies.empty()) {
        auto if_stmt =
            IfThenElse(current_condition,
                       current_if_bodies.size() == 1
                           ? current_if_bodies[0]
                           : this->VisitStmt(SeqStmt(current_if_bodies)),
                       Stmt());
        new_seq.push_back(if_stmt);
        current_condition = PrimExpr();
        current_if_bodies.clear();
      }

      new_seq.push_back(stmt);
    }

    if (!current_if_bodies.empty()) {
      auto if_stmt =
          IfThenElse(current_condition,
                     current_if_bodies.size() == 1
                         ? current_if_bodies[0]
                         : this->VisitStmt(SeqStmt(current_if_bodies)),
                     Stmt());
      new_seq.push_back(if_stmt);
    }

    return new_seq.size() == 1 ? new_seq[0] : SeqStmt(new_seq);
  }
};

PrimFunc MergeIfStmtSubstitute(PrimFunc &f) {
  return MergeIfStmtRewriter::Substitute(f);
}

Stmt ApplyMergeIfStmt(Stmt stmt) { return MergeIfStmtRewriter::Apply(stmt); }

using namespace tirx::transform;
tvm::transform::Pass MergeIfStmt() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return MergeIfStmtRewriter::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MergeIfStmt", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.MergeIfStmt", MergeIfStmt);
}

} // namespace tl
} // namespace tvm
