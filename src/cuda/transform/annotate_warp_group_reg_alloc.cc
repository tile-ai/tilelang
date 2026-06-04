/*!
 * \file annotate_warp_group_reg_alloc.cc
 * \brief Annotate warp group reg alloc for warp specialization
 */

#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "op/builtin.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"
#include <functional>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

bool IsManualWarpSpecializationScope(const AttrStmtNode *op) {
  return op->attr_key == "warp_specialize";
}

Evaluate MakeSetMaxNRegCall(int reg_count, int is_inc) {
  return Evaluate(Call(DataType::Handle(), set_max_nreg(),
                       {IntImm(DataType::Int(32), reg_count),
                        IntImm(DataType::Int(32), is_inc)}));
}

template <typename F>
Stmt RewriteWarpSpecializationBody(const Stmt &stmt, F &&rewrite_if,
                                   bool *rewrote) {
  if (*rewrote) {
    return stmt;
  }

  if (const auto *if_node = stmt.as<IfThenElseNode>()) {
    *rewrote = true;
    return rewrite_if(GetRef<IfThenElse>(if_node));
  }

  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    Array<Stmt> new_seq;
    bool changed = false;
    for (const auto &sub_stmt : seq->seq) {
      Stmt rewritten =
          RewriteWarpSpecializationBody(sub_stmt, rewrite_if, rewrote);
      changed = changed || !rewritten.same_as(sub_stmt);
      new_seq.push_back(rewritten);
    }
    if (!changed) {
      return stmt;
    }
    return new_seq.size() == 1 ? new_seq[0] : SeqStmt(new_seq);
  }

  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    Stmt new_body =
        RewriteWarpSpecializationBody(attr->body, rewrite_if, rewrote);
    if (new_body.same_as(attr->body)) {
      return stmt;
    }
    return AttrStmt(attr->node, attr->attr_key, attr->value, new_body);
  }

  if (stmt.as<BindNode>()) {
    return stmt;
  }

  if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
    const SBlock &block = realize->block;
    Stmt new_body =
        RewriteWarpSpecializationBody(block->body, rewrite_if, rewrote);
    if (new_body.same_as(block->body)) {
      return stmt;
    }
    SBlock new_block(block->iter_vars, block->reads, block->writes,
                     block->name_hint, new_body, block->init,
                     block->alloc_buffers, block->match_buffers,
                     block->annotations);
    return SBlockRealize(realize->iter_values, realize->predicate, new_block);
  }

  if (const auto *block = stmt.as<SBlockNode>()) {
    Stmt new_body =
        RewriteWarpSpecializationBody(block->body, rewrite_if, rewrote);
    if (new_body.same_as(block->body)) {
      return stmt;
    }
    return SBlock(block->iter_vars, block->reads, block->writes,
                  block->name_hint, new_body, block->init, block->alloc_buffers,
                  block->match_buffers, block->annotations);
  }

  return stmt;
}

} // namespace

class SetMaxNRegCollector : public StmtExprVisitor {
public:
  struct Result {
    Array<IntImm> nreg;
    bool preserve_explicit_set_max_nreg{false};
  };

  static Result Collect(const PrimFunc &f) {
    SetMaxNRegCollector collector;
    collector(f->body);
    if (collector.warp_specialized_) {
      return {Array<IntImm>({}), true};
    }
    Array<IntImm> nreg = collector.has_no_set_max_nreg_
                             ? Array<IntImm>({IntImm(DataType::Int(32), -1),
                                              IntImm(DataType::Int(32), -1)})
                             : collector.nreg_;
    return {nreg, false};
  }

private:
  void VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      if (call->op.same_as(set_max_nreg())) {
        return;
      } else if (call->op.same_as(annotate_producer_reg_dealloc())) {
        auto reg_hint = call->args[0].as<IntImmNode>()->value;
        ICHECK(reg_hint <= 240 && reg_hint >= 24)
            << "Invalid reg hint: " << reg_hint;
        nreg_.Set(0, IntImm(DataType::Int(32), reg_hint));
      } else if (call->op.same_as(annotate_consumer_reg_alloc())) {
        auto reg_hint = call->args[0].as<IntImmNode>()->value;
        ICHECK(reg_hint <= 240 && reg_hint >= 24)
            << "Invalid reg hint: " << reg_hint;
        nreg_.Set(1, IntImm(DataType::Int(32), reg_hint));
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
  void VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::ptx_cp_async()) ||
          call->op.same_as(tl::ptx_cp_async())) {
        has_simt_copy_ = true;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

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
    SetMaxNRegCollector::Result result = SetMaxNRegCollector::Collect(f);
    T.nreg_ = result.nreg;
    T.preserve_explicit_set_max_nreg_ = result.preserve_explicit_set_max_nreg;
    if (T.nreg_.empty()) {
      return f;
    }
    f.CopyOnWrite()->body = T(f->body);
    return f;
  }

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      if (!preserve_explicit_set_max_nreg_ &&
          call->op.same_as(set_max_nreg())) {
        return StmtExprMutator::VisitStmt_(op);
      }
      if (in_manual_warp_specialization_ &&
          call->op.same_as(annotate_producer_reg_dealloc())) {
        auto reg_hint = call->args[0].as<IntImmNode>()->value;
        ICHECK(reg_hint <= 240 && reg_hint >= 24)
            << "Invalid reg hint: " << reg_hint;
        return MakeSetMaxNRegCall(reg_hint, 0);
      }
      if (in_manual_warp_specialization_ &&
          call->op.same_as(annotate_consumer_reg_alloc())) {
        auto reg_hint = call->args[0].as<IntImmNode>()->value;
        ICHECK(reg_hint <= 240 && reg_hint >= 24)
            << "Invalid reg hint: " << reg_hint;
        return MakeSetMaxNRegCall(reg_hint, 1);
      }
      if (call->op.same_as(annotate_producer_reg_dealloc()) ||
          call->op.same_as(annotate_consumer_reg_alloc()) ||
          call->op.same_as(no_set_max_nreg())) {
        // Remove annotations after they have been consumed by this pass.
        return Evaluate(0);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent &&
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
    } else if (IsManualWarpSpecializationScope(op)) {
      bool old_in_manual_ws = in_manual_warp_specialization_;
      in_manual_warp_specialization_ = true;
      AttrStmt attr_stmt = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
      in_manual_warp_specialization_ = old_in_manual_ws;
      return attr_stmt;
    } else if (op->attr_key == attr::kWarpSpecializationScope) {
      bool rewrote_ws_body = false;
      auto rewrite_if = [&](const IfThenElse &if_then_else) -> Stmt {
        auto producer_body = if_then_else->then_case;
        Optional<Stmt> consumer_body = if_then_else->else_case;
        // In some degenerate warp-specialized patterns (e.g., producer-only),
        // the consumer body may be absent. Handle gracefully by only
        // annotating the producer side when consumer is missing.

        auto dec_reg = nreg_[0].as<IntImmNode>()->value;
        auto inc_reg = nreg_[1].as<IntImmNode>()->value;

        auto inc_reg_stmt = Evaluate(0);
        auto dec_reg_stmt = Evaluate(0);

        // Default hints stay conservative: skip auto-injection when producer
        // contains SIMT copy-like statements. Explicit user hints should still
        // be honored even in that case.
        bool has_simt_copy = SimtCopyDetector::Detect(producer_body);
        bool has_explicit_hints = dec_reg != 0 || inc_reg != 0;

        if (dec_reg != -1 && inc_reg != -1 &&
            (has_explicit_hints || !has_simt_copy)) {
          int final_dec_reg = has_explicit_hints ? dec_reg : 24;
          int final_inc_reg = has_explicit_hints ? inc_reg : 240;
          bool inject_set_max_nreg = true;

          // The hard-coded 24/240 default pair only fits the canonical
          // 1-producer + 2-consumer warpgroup layout (<=384 threads):
          //   128*24 + 256*240 = 64512 max regs/SM.
          // Wider warp-specialized layouts over-subscribe the register file
          // and the consumer's setmaxnreg.inc deadlocks. When no explicit
          // hint is given, derive the producer/consumer split from the
          // warp-specialization guard and pick a consumer request so that
          // producer_threads*dec + consumer_threads*inc <= 64512; only a
          // handful of layouts have a known-good pair, any other layout falls
          // back to no register reallocation at all.
          if (!has_explicit_hints) {
            int producer_threads = -1;
            int total_threads = -1;
            if (const auto *lt = if_then_else->condition.as<LTNode>()) {
              if (const auto *imm = lt->b.as<IntImmNode>()) {
                producer_threads = static_cast<int>(imm->value);
              }
            }
            if (thread_iv_.defined()) {
              if (const auto *imm = thread_iv_->dom->extent.as<IntImmNode>()) {
                total_threads = static_cast<int>(imm->value);
              }
            }
            int producer_wg =
                producer_threads > 0 ? producer_threads / 128 : -1;
            int consumer_wg = (total_threads > producer_threads)
                                  ? (total_threads - producer_threads) / 128
                                  : -1;
            if (producer_wg == 1 && consumer_wg == 2) {
              // 1P + 2C (384 threads): 128*24 + 256*240 = 64512. Original pair.
            } else if (producer_wg == 2 && consumer_wg == 2) {
              // 2P + 2C (512 threads): 256*24 + 256*224 = 63488 <= 64512.
              final_inc_reg = 224;
              LOG(WARNING)
                  << "AnnotateWarpGroupRegAlloc: 2-producer + 2-consumer "
                     "warp-specialized layout (512 threads) cannot use the "
                     "default consumer register allocation of 240; lowering "
                     "to 24/224 to fit the 64512 max regs/SM. This may reduce "
                     "occupancy/performance — consider an explicit "
                     "T.annotate_consumer_reg_alloc() hint or a different "
                     "warpgroup layout.";
            } else if (producer_wg == 1 && consumer_wg == 3) {
              // 1P + 3C (512 threads): 128*24 + 384*160 = 64512.
              final_inc_reg = 160;
              LOG(WARNING)
                  << "AnnotateWarpGroupRegAlloc: 1-producer + 3-consumer "
                     "warp-specialized layout (512 threads) cannot use the "
                     "default consumer register allocation of 240; lowering "
                     "to 24/160 to fit the 64512 max regs/SM. This may reduce "
                     "occupancy/performance — consider an explicit "
                     "T.annotate_consumer_reg_alloc() hint or a different "
                     "warpgroup layout.";
            } else {
              // Unknown / unsupported layout: skip register reallocation
              // entirely rather than risk over-subscribing the register file.
              inject_set_max_nreg = false;
              LOG(WARNING)
                  << "AnnotateWarpGroupRegAlloc: unsupported warp-specialized "
                     "warpgroup layout (producer="
                  << producer_threads << ", total=" << total_threads
                  << " threads); skipping setmaxnreg register reallocation to "
                     "avoid register-file over-subscription. This may reduce "
                     "occupancy/performance — consider an explicit "
                     "T.annotate_producer_reg_dealloc()/"
                     "T.annotate_consumer_reg_alloc() hint.";
            }
          }
          if (inject_set_max_nreg) {
            dec_reg_stmt =
                Evaluate(Call(DataType::Handle(), set_max_nreg(),
                              {IntImm(DataType::Int(32), final_dec_reg),
                               IntImm(DataType::Int(32), 0)}));
            inc_reg_stmt =
                Evaluate(Call(DataType::Handle(), set_max_nreg(),
                              {IntImm(DataType::Int(32), final_inc_reg),
                               IntImm(DataType::Int(32), 1)}));
          }
        }

        Array<Stmt> producer_stmts;
        producer_stmts.push_back(dec_reg_stmt);
        producer_stmts.push_back(producer_body);
        auto new_producer_body = SeqStmt(producer_stmts);

        if (consumer_body.defined()) {
          Array<Stmt> consumer_stmts;
          consumer_stmts.push_back(inc_reg_stmt);
          consumer_stmts.push_back(consumer_body.value());
          auto new_consumer_body = SeqStmt(consumer_stmts);
          return IfThenElse(if_then_else->condition, new_producer_body,
                            new_consumer_body);
        }

        return IfThenElse(if_then_else->condition, new_producer_body);
      };

      Stmt new_body =
          RewriteWarpSpecializationBody(op->body, rewrite_if, &rewrote_ws_body);
      if (!rewrote_ws_body) {
        return StmtExprMutator::VisitStmt_(op);
      }
      return AttrStmt(op->node, op->attr_key, op->value, new_body);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Array<IntImm> nreg_;
  bool preserve_explicit_set_max_nreg_{false};
  bool in_manual_warp_specialization_{false};
  IterVar thread_iv_;
  Optional<PrimExpr> updated_thread_extent_;
  bool need_update_thread_extent_ = false;
};

using namespace tirx::transform;

tvm::transform::Pass AnnotateWarpGroupRegAlloc() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const PassContext &ctx) -> PrimFunc {
    return SetMaxNRegInjector::Inject(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AnnotateWarpGroupRegAlloc", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.cuda.transform.AnnotateWarpGroupRegAlloc",
                        AnnotateWarpGroupRegAlloc);
}

} // namespace tl
} // namespace tvm
