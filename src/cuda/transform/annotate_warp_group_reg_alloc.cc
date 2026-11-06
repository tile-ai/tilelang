/*!
 * \file annotate_warp_group_reg_alloc.cc
 * \brief Consume register hints inside native warp-specialization scopes.
 */

#include "cuda/op/builtin.h"
#include "ir/annotations.h"
#include "ir/lexical_scope.h"
#include "op/builtin.h"
#include "transform/common/warp_specialize.h"
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
using namespace tirx;
using namespace prim;
using namespace ffi;

namespace {

// Only the first branch under the specialization marker divides producer
// and consumer roles. Preserve Bind definitions and every enclosing native
// region, including its body parameters and results.
template <typename F>
Stmt RewriteWarpSpecializationBody(const Stmt &stmt, F &&rewrite_if,
                                   bool *rewrote) {
  if (*rewrote)
    return stmt;
  if (const auto *branch = stmt.as<IfObj>()) {
    *rewrote = true;
    return rewrite_if(GetRef<If>(branch));
  }
  if (const auto *sequence = stmt.as<SeqStmtObj>()) {
    Array<Stmt> statements;
    bool changed = false;
    for (const auto &statement : sequence->seq) {
      Stmt next = RewriteWarpSpecializationBody(statement, rewrite_if, rewrote);
      changed |= !next.same_as(statement);
      statements.push_back(next);
    }
    return changed ? Stmt(SeqStmt(statements, sequence->loc)) : stmt;
  }
  if (const auto *region = stmt.as<RegionStmtObj>()) {
    Stmt body =
        RewriteWarpSpecializationBody(region->body, rewrite_if, rewrote);
    if (!body.same_as(region->body))
      return RegionStmt(region->op, region->args, region->body_params,
                        region->attrs, body.as_or_throw<SeqStmt>(),
                        region->result_vars, region->loc);
  }
  return stmt;
}

class SetMaxNRegCollector : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  int producer = 0;
  int consumer = 0;
  bool disabled = false;
  bool explicit_schedule = false;

private:
  Optional<VisitInterrupt> Visit_(const CallObj *call) final {
    if (call->op.same_as(annotate_producer_reg_dealloc()) ||
        call->op.same_as(annotate_consumer_reg_alloc())) {
      const auto *immediate = call->args[0].as<IntImmObj>();
      TVM_FFI_CHECK(immediate && immediate->value >= 24 &&
                        immediate->value <= 240,
                    ValueError)
          << "Invalid register hint: " << call->args[0];
      if (call->op.same_as(annotate_producer_reg_dealloc()))
        producer = immediate->value.as<int>().value();
      else
        consumer = immediate->value.as<int>().value();
    } else if (call->op.same_as(no_set_max_nreg())) {
      disabled = true;
    }
    return StmtExprVisitor::Visit_(call);
  }

  Optional<VisitInterrupt> Visit_(const RegionStmtObj *region) final {
    if (region->op->name == "tl.attr" &&
        ReadAnnotations(region).at("key").cast<String>() ==
            attr::kCustomWarpSpecialization)
      explicit_schedule = true;
    return StmtExprVisitor::Visit_(region);
  }
};

class SimtCopyDetector : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  bool found = false;

private:
  Optional<VisitInterrupt> Visit_(const CallObj *call) final {
    if (call->op.same_as(ptx_cp_async()) ||
        call->op.same_as(ptx_cp_async_bytes()))
      found = true;
    return StmtExprVisitor::Visit_(call);
  }

  Optional<VisitInterrupt> Visit_(const TensorStoreObj *store) final {
    const auto *type = store->dest->ty.as<TensorTypeObj>();
    TVM_FFI_CHECK(type, TypeError);
    if (type->scope.name() != "global")
      found = true;
    return StmtExprVisitor::Visit_(store);
  }
};

class SetMaxNRegInjector : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  SetMaxNRegInjector(int producer, int consumer, bool disabled,
                     bool preserve_explicit)
      : producer_(producer), consumer_(consumer), disabled_(disabled),
        preserve_explicit_(preserve_explicit) {}

private:
  UnchangedOr<Stmt> Mutate_(const EvaluateObj *evaluate,
                            InplaceMode mode) final {
    if (const auto *call = evaluate->value.as<CallObj>()) {
      if (call->op.same_as(annotate_producer_reg_dealloc()) ||
          call->op.same_as(annotate_consumer_reg_alloc()) ||
          call->op.same_as(no_set_max_nreg()))
        return SeqStmt(Array<Stmt>{}, evaluate->loc);
    }
    // Raw set_max_nreg calls are execution operations, not annotations.
    return StmtExprMutator::Mutate_(evaluate, mode);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *region,
                            InplaceMode mode) final {
    Optional<PrimExpr> saved_threads = thread_extent_;
    if (region->op.same_as(launch_thread_op()) &&
        region->args[0].as_or_throw<StringImm>()->value == "threadIdx.x")
      thread_extent_ = region->args[1].as_or_throw<PrimExpr>();

    // Consume annotations in nested bodies before adding the execution calls.
    RegionStmt rewritten = StmtExprMutator::Mutate_(region, mode)
                               .ValueOrUnchanged(GetRef<RegionStmt>(region))
                               .as_or_throw<RegionStmt>();
    String key;
    if (region->op->name == "tl.attr")
      key = ReadAnnotations(region).at("key").cast<String>();
    if (key == attr::kWarpSpecializationScope && !preserve_explicit_) {
      bool found = false;
      auto body = RewriteWarpSpecializationBody(
          rewritten->body, [&](const If &branch) { return Annotate(branch); },
          &found);
      if (found)
        rewritten =
            RegionStmt(rewritten->op, rewritten->args, rewritten->body_params,
                       rewritten->attrs, body.as_or_throw<SeqStmt>(),
                       rewritten->result_vars, rewritten->loc);
    }
    thread_extent_ = saved_threads;
    // Shared-memory planning, synchronization and register annotation are
    // the last consumers of these metadata scopes, even when register
    // inference is disabled. Keep the original definition boundary.
    if (key == attr::kWarpSpecializationScope ||
        key == attr::kCustomWarpSpecialization) {
      TVM_FFI_CHECK(region->body_params.empty() && region->result_vars.empty(),
                    ValueError);
      return MakeLexicalScope(rewritten->body, region->loc);
    }
    return rewritten;
  }

  Stmt Annotate(const If &branch) {
    if (disabled_)
      return branch;
    auto detector = make_object<SimtCopyDetector>();
    detector->Visit(branch->then_case);
    const bool explicit_hint = producer_ != 0 || consumer_ != 0;
    if (!explicit_hint && detector->found)
      return branch;

    int decrease = explicit_hint ? producer_ : 24;
    int increase = explicit_hint ? consumer_ : 240;
    if (!explicit_hint) {
      int producer_threads = -1, total_threads = -1;
      if (const auto *less = branch->condition.as<LTObj>())
        if (const auto *bound = less->b.as<IntImmObj>())
          producer_threads = bound->value.as<int>().value_or(-1);
      if (thread_extent_)
        if (const auto *bound = thread_extent_.value().as<IntImmObj>())
          total_threads = bound->value.as<int>().value_or(-1);
      int producer_groups = producer_threads > 0 ? producer_threads / 128 : -1;
      int consumer_groups = total_threads > producer_threads
                                ? (total_threads - producer_threads) / 128
                                : -1;
      // Keep the source default pairs. These known layouts fit the 64512
      // registers/SM budget; other layouts retain their original allocation.
      if (producer_groups == 1 && consumer_groups <= 2) {
        // 1P+1C and 1P+2C use the original 24/240 pair.
      } else if (producer_groups == 2 && consumer_groups == 2) {
        increase = 224;
        LOG(WARNING) << "AnnotateWarpGroupRegAlloc: using 24/224 for the "
                        "2-producer + 2-consumer warpgroup register budget";
      } else if (producer_groups == 1 && consumer_groups == 3) {
        increase = 160;
        LOG(WARNING) << "AnnotateWarpGroupRegAlloc: using 24/160 for the "
                        "1-producer + 3-consumer warpgroup register budget";
      } else {
        LOG(WARNING) << "AnnotateWarpGroupRegAlloc: unsupported warpgroup "
                        "layout; preserving the original register allocation";
        return branch;
      }
    }
    auto prepend = [&](SeqStmt body, int count, int is_allocation) {
      Array<Stmt> statements{
          Evaluate(Call(PrimType::Void(), set_max_nreg(),
                        {IntImm::Int32(count), IntImm::Int32(is_allocation)},
                        {}, {}, branch->loc),
                   branch->loc)};
      for (const auto &statement : body->seq)
        statements.push_back(statement);
      return SeqStmt(statements, body->loc);
    };
    Optional<SeqStmt> consumer;
    if (branch->else_case)
      consumer = prepend(*branch->else_case, increase, 1);
    return If(branch->condition, prepend(branch->then_case, decrease, 0),
              consumer, branch->loc);
  }

  int producer_, consumer_;
  bool disabled_, preserve_explicit_;
  Optional<PrimExpr> thread_extent_;
};
} // namespace

tvm::transform::Pass
AnnotateWarpGroupRegAlloc(bool enable_register_inference = true) {
  auto pass = [enable_register_inference](tirx::Function function, IRModule,
                                          tvm::transform::PassContext) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto collector = make_object<SetMaxNRegCollector>();
    collector->Visit(function->body);
    // An explicit WSSchedule already emitted register operations inside its
    // role scopes; the legacy inference pass must preserve those operations.
    auto injector = make_object<SetMaxNRegInjector>(
        collector->producer, collector->consumer,
        collector->disabled || !enable_register_inference,
        collector->explicit_schedule);
    auto body = function->body;
    auto result = injector->Mutate(body).ValueOrUnchanged(body);
    if (!result.same_as(body))
      function.CopyOnWrite()->body = result;
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.AnnotateWarpGroupRegAlloc");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def_packed(
      "tl.cuda.transform.AnnotateWarpGroupRegAlloc",
      [](PackedArgs args, Any *result) {
        TVM_FFI_CHECK(args.size() <= 1, TypeError)
            << "AnnotateWarpGroupRegAlloc accepts optional "
               "enable_register_inference";
        *result = AnnotateWarpGroupRegAlloc(
            args.size() == 0 ? true : args[0].cast<bool>());
      });
}
} // namespace tvm::tl
