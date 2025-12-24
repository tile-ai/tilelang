/*!
 * \file annotate_warp_group_reg_alloc.cc
 * \brief Annotate warp group reg alloc for warp specialization
 */

#include "warp_specialized_rewriter.h"
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

class SetMaxNRegCollector : public StmtExprVisitor {
public:
  static Array<IntImm> Collect(const PrimFunc &f) {
    SetMaxNRegCollector collector;
    collector(f->body);
    if (collector.warp_specialized_) {
      return Array<IntImm>({});
    }
    return collector.has_no_set_max_nreg_
               ? Array<IntImm>({IntImm(DataType::Int(32), -1),
                                IntImm(DataType::Int(32), -1)})
               : collector.nreg_;
  }

private:
  void VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      if (call->op.same_as(set_max_nreg())) {
        auto reg_hint = call->args[0].as<IntImmNode>()->value;
        auto is_inc = call->args[1].as<IntImmNode>()->value;
        ICHECK(reg_hint <= 240 && reg_hint >= 24)
            << "Invalid reg hint: " << reg_hint;
        ICHECK(is_inc == 0 || is_inc == 1) << "Invalid is_inc: " << is_inc;

        // producer should decrease register hint while consumer should increase
        // register hint
        nreg_.Set(is_inc, IntImm(DataType::Int(32), reg_hint));
      } else if (call->op.same_as(no_set_max_nreg())) {
        has_no_set_max_nreg_ = true;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == attr::kCustomWarpSpecialization) {
      warp_specialized_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  Array<IntImm> nreg_{IntImm(DataType::Int(32), 0),
                      IntImm(DataType::Int(32), 0)};
  bool has_no_set_max_nreg_ = false;
  bool warp_specialized_ = false;
};

class SimtCopyDetector : public StmtExprVisitor {
public:
  static bool Detect(const Stmt &stmt) {
    SimtCopyDetector detector;
    detector.VisitStmt(stmt);
    return detector.has_simt_copy_;
  }

private:
  void VisitStmt_(const BufferStoreNode *op) final {
    auto scope =
        runtime::StorageScope::Create(GetPtrStorageScope(op->buffer->data));
    if (scope.to_string() != "global") {
      has_simt_copy_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool has_simt_copy_{false};
};

class SetMaxNRegInjector : public StmtExprMutator {
public:
  static PrimFunc Inject(PrimFunc f) {
    auto T = SetMaxNRegInjector();
    T.nreg_ = SetMaxNRegCollector::Collect(f);
    if (T.nreg_.empty()) {
      return f;
    }
    f.CopyOnWrite()->body = T(f->body);
    return f;
  }

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      if (call->op.same_as(no_set_max_nreg())) {
        // Remove the original set_max_nreg calls as they will be re-inserted
        // at appropriate locations
        return Evaluate(0);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent &&
        Downcast<IterVar>(op->node)->thread_tag == "threadIdx.x") {
      thread_iv_ = Downcast<IterVar>(op->node);
      need_update_thread_extent_ = false;
      AttrStmt attr_stmt = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
      if (need_update_thread_extent_) {
        thread_iv_.CopyOnWrite()->dom = {0, updated_thread_extent_.value()};
        attr_stmt.CopyOnWrite()->node = thread_iv_;
        attr_stmt.CopyOnWrite()->value = updated_thread_extent_.value();
      }
      thread_iv_ = {};
      return attr_stmt;
    } else if (op->attr_key == attr::kWarpSpecializationScope) {
      auto if_then_else = Downcast<IfThenElse>(op->body);
      if (!if_then_else.defined()) {
        return StmtExprMutator::VisitStmt_(op);
      }
      auto producer_body = if_then_else->then_case;
      Optional<Stmt> consumer_body = if_then_else->else_case;
      // In some degenerate warp-specialized patterns (e.g., producer-only),
      // the consumer body may be absent. Handle gracefully by only annotating
      // the producer side when consumer is missing.

      auto dec_reg = nreg_[0].as<IntImmNode>()->value;
      auto inc_reg = nreg_[1].as<IntImmNode>()->value;

      auto inc_reg_stmt = Evaluate(0);
      auto dec_reg_stmt = Evaluate(0);

      // Only inject if we have valid register hints and no SIMT copy
      bool has_simt_copy = SimtCopyDetector::Detect(producer_body);

      if (dec_reg == 0 && inc_reg == 0 && !has_simt_copy) {
        auto inc_reg_num = IntImm(DataType::Int(32), 240);
        auto dec_reg_num = IntImm(DataType::Int(32), 24);
        inc_reg_stmt = Evaluate(
            Call(DataType::Handle(), set_max_nreg(), {inc_reg_num, 1}));
        dec_reg_stmt = Evaluate(
            Call(DataType::Handle(), set_max_nreg(), {dec_reg_num, 0}));
      }

      // Inject register setting statements
      Array<Stmt> producer_stmts;
      producer_stmts.push_back(dec_reg_stmt);
      producer_stmts.push_back(producer_body);
      auto new_producer_body = SeqStmt(producer_stmts);

      Stmt new_if_stmt;
      if (consumer_body.defined()) {
        Array<Stmt> consumer_stmts;
        consumer_stmts.push_back(inc_reg_stmt);
        consumer_stmts.push_back(consumer_body.value());
        auto new_consumer_body = SeqStmt(consumer_stmts);
        new_if_stmt = IfThenElse(if_then_else->condition, new_producer_body,
                                 new_consumer_body);
      } else {
        // No consumer branch; keep the if-then form.
        new_if_stmt = IfThenElse(if_then_else->condition, new_producer_body);
      }

      auto new_attr = AttrStmt(op->node, op->attr_key, op->value, new_if_stmt);
      return new_attr;
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Array<IntImm> nreg_;
  IterVar thread_iv_;
  Optional<PrimExpr> updated_thread_extent_;
  bool need_update_thread_extent_ = false;
};

using namespace tir::transform;

tvm::transform::Pass AnnotateWarpGroupRegAlloc() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const PassContext &ctx) -> PrimFunc {
    return SetMaxNRegInjector::Inject(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AnnotateWarpGroupRegAlloc", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnnotateWarpGroupRegAlloc",
                        AnnotateWarpGroupRegAlloc);
}

} // namespace tl
} // namespace tvm
