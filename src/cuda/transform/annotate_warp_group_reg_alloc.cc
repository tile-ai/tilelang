#include "compat/expr.h"
#include "compat/visit.h"
#include "cuda/op/builtin.h"
#include "op/utils.h"

#include <tvm/tirx/op/region.h>
#include <tvm/tirx/transform.h>

#include <array>
#include <functional>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

bool HasMarker(const RegionStmtNode *region, const char *name) {
  return region->attrs->dict.count(name) != 0;
}

class RegisterHints : public StmtExprVisitor {
public:
  std::array<int64_t, 2> counts{0, 0};
  bool disabled = false;
  bool custom = false;

private:
  Optional<VisitInterrupt> Visit_(const RegionStmtNode *node) final {
    custom |= HasMarker(node, attr::kCustomWarpSpecialization);
    return StmtExprVisitor::Visit_(node);
  }

  Optional<VisitInterrupt> Visit_(const CallNode *node) final {
    if (node->op.same_as(no_set_max_nreg()))
      disabled = true;
    for (size_t index = 0; index < counts.size(); ++index) {
      const Op &operation = index == 0 ? annotate_producer_reg_dealloc()
                                       : annotate_consumer_reg_alloc();
      if (!node->op.same_as(operation))
        continue;
      auto count = GetConstInt(node->args[0].as_or_throw<PrimExpr>());
      TVM_FFI_CHECK(count && *count >= 24 && *count <= 240 && *count % 8 == 0,
                    ValueError)
          << "Warpgroup register hint must be a multiple of 8 between 24 and "
             "240";
      counts[index] = *count;
    }
    return StmtExprVisitor::Visit_(node);
  }
};

bool HasSIMTCopy(const Stmt &statement) {
  bool found = false;
  PostOrderVisit(statement, [&](const ObjectRef &object) {
    if (const auto *store = object.as<TensorStoreNode>()) {
      if (auto tensor = store->dest.as<TensorVar>())
        found |= !IsGlobalBuffer(*tensor);
    }
    if (const auto *call = object.as<CallNode>()) {
      if (auto operation = call->op.as<Op>()) {
        found |= operation.value()->name == "tl.ptx_cp_async" ||
                 operation.value()->name == "tirx.s_tir.cp_async_raw";
      }
    }
  });
  return found;
}

Stmt RewriteFirstBranch(const Stmt &statement,
                        const std::function<Stmt(const IfNode *)> &rewrite,
                        bool *rewritten) {
  if (*rewritten)
    return statement;
  if (const auto *branch = statement.as<IfNode>()) {
    *rewritten = true;
    return rewrite(branch);
  }
  if (const auto *sequence = statement.as<SeqStmtNode>()) {
    Array<Stmt> statements;
    for (const Stmt &child : sequence->seq) {
      statements.push_back(RewriteFirstBranch(child, rewrite, rewritten));
    }
    return *rewritten ? Stmt(SeqStmt(statements, sequence->loc)) : statement;
  }
  if (const auto *region = statement.as<RegionStmtNode>();
      region &&
      (region->op.same_as(scope_op()) || region->op->name == "tl.scope")) {
    Stmt body = RewriteFirstBranch(region->body, rewrite, rewritten);
    if (*rewritten) {
      RegionStmt result = GetRef<RegionStmt>(region);
      result.CopyOnWrite()->body = SeqStmt(body);
      return result;
    }
  }
  return statement;
}

Stmt ReconfigureRegisters(int64_t count, bool increase) {
  return Evaluate(Call(PrimType::Void(), set_max_nreg(),
                       {IntImm::Int32(count), IntImm::Int32(increase)}));
}

class RegisterInjector : public StmtExprMutator {
public:
  explicit RegisterInjector(const RegisterHints &hints)
      : counts_(hints.counts), disabled_(hints.disabled) {}

private:
  UnchangedOr<Stmt> Mutate_(const EvaluateNode *node, InplaceMode mode) final {
    if (const auto *call = node->value.as<CallNode>();
        call && (call->op.same_as(annotate_producer_reg_dealloc()) ||
                 call->op.same_as(annotate_consumer_reg_alloc()) ||
                 call->op.same_as(no_set_max_nreg()))) {
      return SeqStmt({});
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

  Stmt RewriteBranch(const IfNode *branch) const {
    if (disabled_)
      return GetRef<Stmt>(branch);
    bool explicit_hints = counts_[0] != 0 || counts_[1] != 0;
    if (!explicit_hints && HasSIMTCopy(branch->then_case))
      return GetRef<Stmt>(branch);
    int64_t producer_registers = counts_[0] ? counts_[0] : 24;
    int64_t consumer_registers = counts_[1] ? counts_[1] : 240;
    Optional<int64_t> producer_threads;
    if (const auto *condition = branch->condition.as<LTNode>();
        condition && thread_ && condition->a.same_as(*thread_)) {
      producer_threads = GetConstInt(condition->b);
    }
    if (!explicit_hints) {
      if (!producer_threads || !thread_extent_ || *producer_threads <= 0 ||
          *producer_threads % 128 || *thread_extent_ % 128 ||
          *thread_extent_ <= *producer_threads) {
        return GetRef<Stmt>(branch);
      }
      int64_t producers = *producer_threads / 128;
      int64_t consumers = (*thread_extent_ - *producer_threads) / 128;
      if (producers == 1 && consumers <= 2) {
        consumer_registers = 240;
      } else if (producers == 2 && consumers == 2) {
        consumer_registers = 224;
      } else if (producers == 1 && consumers == 3) {
        consumer_registers = 160;
      } else {
        LOG(WARNING)
            << "AnnotateWarpGroupRegAlloc: unsupported warpgroup split; "
               "skipping automatic register reallocation";
        return GetRef<Stmt>(branch);
      }
    }
    if (producer_threads && thread_extent_ && branch->else_case) {
      TVM_FFI_CHECK(*producer_threads * producer_registers +
                            (*thread_extent_ - *producer_threads) *
                                consumer_registers <=
                        64512,
                    ValueError)
          << "Warp-specialized register hints exceed the CTA register budget";
    }
    SeqStmt producer(
        {ReconfigureRegisters(producer_registers, false), branch->then_case});
    Optional<SeqStmt> consumer;
    if (branch->else_case) {
      consumer = SeqStmt(
          {ReconfigureRegisters(consumer_registers, true), *branch->else_case});
    }
    return If(branch->condition, producer, consumer, branch->loc);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                            InplaceMode mode) final {
    auto previous_thread = thread_;
    auto previous_extent = thread_extent_;
    if (node->op.same_as(launch_thread_op()) &&
        node->args[0].as_or_throw<StringImm>()->value == "threadIdx.x") {
      thread_ = node->body_params[0].as_or_throw<PrimVar>();
      thread_extent_ = GetConstInt(node->args[1].as_or_throw<PrimExpr>());
    }
    RegionStmt result = StmtExprMutator::Mutate_(node, mode)
                            .ValueOrUnchanged(GetRef<Stmt>(node))
                            .as_or_throw<RegionStmt>();
    if (HasMarker(node, attr::kWarpSpecializationScope)) {
      bool rewritten = false;
      Stmt body = RewriteFirstBranch(
          result->body,
          [this](const IfNode *branch) { return RewriteBranch(branch); },
          &rewritten);
      result.CopyOnWrite()->body = SeqStmt(body);
    }
    thread_ = previous_thread;
    thread_extent_ = previous_extent;
    return result;
  }

  std::array<int64_t, 2> counts_;
  bool disabled_;
  Optional<PrimVar> thread_;
  Optional<int64_t> thread_extent_;
};
} // namespace

tvm::transform::Pass AnnotateWarpGroupRegAlloc() {
  return tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext) -> Optional<tirx::Function> {
        if (!function->body)
          return function;
        auto hints = make_object<RegisterHints>();
        hints->Visit(*function->body);
        if (hints->custom)
          return function;
        auto injector = make_object<RegisterInjector>(*hints);
        SeqStmt body = *function->body;
        function.CopyOnWrite()->body =
            injector->Mutate(body).ValueOrUnchanged(body);
        return function;
      },
      0, "tl.cuda.AnnotateWarpGroupRegAlloc");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.cuda.transform.AnnotateWarpGroupRegAlloc",
                              AnnotateWarpGroupRegAlloc);
}
} // namespace tvm::tl
