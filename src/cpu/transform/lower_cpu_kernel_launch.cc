/*!
 * \file lower_cpu_kernel_launch.cc
 * \brief Prepare native CPU grids and per-launch OpenMP thread counts.
 */

#include "cpu/op/builtin.h"
#include "ir/annotations.h"
#include "ir/lexical_scope.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <unordered_set>
#include <vector>

namespace tvm::tl {
namespace {
using namespace ffi;

bool IsGrid(const ForObj *node) {
  return node != nullptr && node->annotations.count(attr::kCPUGridDim);
}

const ForObj *SingleGrid(const SeqStmt &body) {
  return body->seq.size() == 1 ? body->seq[0].as<ForObj>() : nullptr;
}

// Source alloc_buffer belonged to an SBlock-managed allocation pool. The old
// CPU storage planner attached that pool outside serial control flow, while
// native alloc_tensor has an explicit lexical lifetime. Preserve the source
// contract at this boundary, then let MaterializeCPUParallelGrid prove when a
// shared allocation can be privatized. Never infer this origin from scope
// names.
class SourceBlockAllocationPlanner {
public:
  SeqStmt Run(const SeqStmt &body) {
    auto result = Rewrite(body);
    return Prepend(result.lifted, SeqStmt(result.stmt));
  }

private:
  using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
  struct Result {
    Stmt stmt;
    Array<Bind> lifted;
  };
  enum class AllocationBarrier { kNone, kAll, kLocal, kSharedOrGlobal };
  static constexpr const char *kSourceBlockAlloc = "tl.source_block_alloc";

  static SeqStmt Prepend(const Array<Bind> &allocations, SeqStmt body) {
    if (allocations.empty())
      return body;
    Array<Stmt> sequence;
    for (const auto &allocation : allocations)
      sequence.push_back(allocation);
    sequence.push_back(body);
    return SeqStmt(sequence, body->loc);
  }

  static bool UsesAny(const Bind &allocation, const VarSet &variables) {
    for (const Var &var : UndefinedVars(allocation))
      if (variables.count(var))
        return true;
    return false;
  }

  static bool PureType(const Type &type) {
    bool pure = true;
    StructuralWalk<WalkOrder::kPreOrder>(
        type, [&](const ExprObj *expr) -> Expected<WalkResult> {
          pure &= SideEffect(GetRef<Expr>(expr)) == CallEffectKind::kPure;
          return WalkResult::Skip();
        });
    return pure;
  }

  static bool IsSharedOrGlobal(const Bind &allocation) {
    auto tensor = allocation->var->ty.as<tirx::TensorType>();
    if (!tensor.has_value())
      return false;
    const std::string scope = tensor.value()->scope.name();
    return scope == "global" || scope.rfind("global.", 0) == 0 ||
           scope == "shared" || scope.rfind("shared.", 0) == 0;
  }

  static bool IsLexicalAnnotation(const Stmt &stmt) {
    const auto *evaluate = stmt.as<EvaluateObj>();
    const auto *call = evaluate ? evaluate->value.as<CallObj>() : nullptr;
    if (!call || !IsAnnotationCall(call))
      return false;
    auto fields = ReadAnnotations(call);
    return fields.count(attr::kLexicalAllocScope);
  }

  // A definition in this sequence must still dominate an allocation that uses
  // it, including variables inside TensorType.shape and layout expressions.
  // Keep such an allocation just before its original containing statement.
  Result RewriteSequence(const SeqStmtObj *op) {
    Array<Stmt> sequence;
    Array<Bind> lifted;
    VarSet defined;
    for (size_t i = 0; i < op->seq.size(); ++i) {
      const Stmt &element = op->seq[i];
      if (IsLexicalAnnotation(element)) {
        // sblock_attr annotates this sequence's suffix. Keep the statement
        // for LowerSourceAnnotations, and place its local pool after it. The
        // lexical depth must already be active while visiting nested parallel
        // scopes so that they do not claim this local pool first.
        Array<Stmt> suffix;
        for (size_t j = i + 1; j < op->seq.size(); ++j)
          suffix.push_back(op->seq[j]);
        ++lexical_scope_depth_;
        Result child = Rewrite(SeqStmt(suffix, op->loc));
        --lexical_scope_depth_;
        SeqStmt body = KeepInside(&child, {}, AllocationBarrier::kLocal);
        for (const Bind &allocation : child.lifted) {
          if (UsesAny(allocation, defined))
            sequence.push_back(allocation);
          else
            lifted.push_back(allocation);
        }
        sequence.push_back(element);
        sequence.push_back(body);
        break;
      }
      Result child = Rewrite(element);
      for (const Bind &allocation : child.lifted) {
        if (UsesAny(allocation, defined))
          sequence.push_back(allocation);
        else
          lifted.push_back(allocation);
      }
      sequence.push_back(child.stmt);
      if (const auto *bind = element.as<BindObj>())
        defined.insert(bind->var);
      else if (const auto *region = element.as<RegionStmtObj>())
        for (const Var &var : region->result_vars)
          defined.insert(var);
    }
    return {SeqStmt(sequence, op->loc), lifted};
  }

  // Place dependent allocations after a loop/region's parameter definitions.
  // Explicit parallel loops and native lexical scopes own their private pools.
  static SeqStmt KeepInside(Result *result, const VarSet &parameters,
                            AllocationBarrier barrier) {
    Array<Bind> inner, outer;
    for (const Bind &allocation : result->lifted) {
      bool shared = IsSharedOrGlobal(allocation);
      bool anchored = barrier == AllocationBarrier::kAll ||
                      (barrier == AllocationBarrier::kLocal && !shared) ||
                      (barrier == AllocationBarrier::kSharedOrGlobal && shared);
      if (UsesAny(allocation, parameters) || anchored)
        inner.push_back(allocation);
      else
        outer.push_back(allocation);
    }
    result->lifted = std::move(outer);
    return Prepend(inner, SeqStmt(result->stmt));
  }

  Result Rewrite(const Stmt &stmt) {
    if (const auto *op = stmt.as<SeqStmtObj>())
      return RewriteSequence(op);
    if (const auto *op = stmt.as<BindObj>()) {
      const auto *call = op->value.as<CallObj>();
      if (!call || !call->op.same_as(tirx::alloc_tensor_op()) ||
          !call->attrs.has_value())
        return {stmt, {}};
      auto attrs = call->attrs.value().as<DictAttrs>();
      if (!attrs.has_value() || !attrs.value()->dict.count(kSourceBlockAlloc))
        return {stmt, {}};
      auto values = attrs.value()->dict;
      auto origin = values.at(kSourceBlockAlloc).as<IntImm>();
      TVM_FFI_CHECK(origin.has_value() && origin.value()->value == 1,
                    ValueError)
          << kSourceBlockAlloc << " must be the source allocation marker";
      values.erase(kSourceBlockAlloc);
      Bind allocation(op->var,
                      Call(call->ty, call->op, call->args, DictAttrs(values),
                           call->ty_args, call->loc),
                      op->loc);
      // Explicit placement and effectful type evaluation are anchored. The
      // marker is consumed even when a declaration cannot leave its position.
      if (!call->args.empty() || !PureType(op->var->ty))
        return {allocation, {}};
      return {SeqStmt({}), {allocation}};
    }
    if (const auto *op = stmt.as<ForObj>()) {
      bool parallel = op->kind == ForKind::kParallel;
      bool owns_execution_pool = parallel && execution_scope_depth_ == 0;
      execution_scope_depth_ += parallel;
      Result child = Rewrite(op->body);
      execution_scope_depth_ -= parallel;
      auto barrier =
          owns_execution_pool
              ? (lexical_scope_depth_ != 0 ? AllocationBarrier::kSharedOrGlobal
                                           : AllocationBarrier::kAll)
              : AllocationBarrier::kNone;
      auto body = KeepInside(&child, VarSet{op->var}, barrier);
      return {For(op->var, op->begin, op->end, op->step, op->kind, body,
                  op->annotations, op->loc),
              child.lifted};
    }
    if (const auto *op = stmt.as<IfObj>()) {
      Result then_result = Rewrite(op->then_case);
      Optional<SeqStmt> else_body;
      if (op->else_case.has_value()) {
        Result else_result = Rewrite(op->else_case.value());
        else_body = SeqStmt(else_result.stmt);
        for (const Bind &allocation : else_result.lifted)
          then_result.lifted.push_back(allocation);
      }
      return {If(op->condition, SeqStmt(then_result.stmt), else_body, op->loc),
              then_result.lifted};
    }
    if (const auto *op = stmt.as<WhileObj>()) {
      Result child = Rewrite(op->body);
      return {While(op->condition, SeqStmt(child.stmt), op->loc), child.lifted};
    }
    if (const auto *op = stmt.as<RegionStmtObj>()) {
      auto barrier = AllocationBarrier::kAll;
      bool source_lexical = false, execution_scope = false;
      if (IsLexicalScope(op)) {
        source_lexical = IsExplicitAllocationScope(op);
        // A pre-lowered explicit scope owns local storage; shared/global
        // pools still belong to the surrounding execution scope. Ordinary
        // regions retain their full definition boundary.
        if (source_lexical)
          barrier = AllocationBarrier::kLocal;
      } else if (op->op->name == "tl.attr") {
        auto fields =
            DecodeAnnotations(
                op->args[0].as_or_throw<tvm::Tuple>(),
                op->attrs.GetAttr<Array<Any>>("annotation_schema").value())
                .cast<Map<Any, Any>>();
        const std::string key = fields.at("key").cast<String>();
        source_lexical = key == attr::kLexicalAllocScope;
        execution_scope = key == "thread_extent" || key == "virtual_thread" ||
                          key.rfind("pragma_", 0) == 0;
        if (source_lexical) {
          barrier = AllocationBarrier::kLocal;
        } else if (execution_scope && execution_scope_depth_ == 0) {
          barrier = lexical_scope_depth_ != 0
                        ? AllocationBarrier::kSharedOrGlobal
                        : AllocationBarrier::kAll;
        } else {
          barrier = AllocationBarrier::kNone;
        }
      }
      // The old planner's outermost execution scope owns shared/global pools.
      // The nearest explicit lexical scope overrides that placement for
      // local-like storage, including when it surrounds a parallel loop.
      lexical_scope_depth_ += source_lexical;
      execution_scope_depth_ += execution_scope;
      Result child = Rewrite(op->body);
      execution_scope_depth_ -= execution_scope;
      lexical_scope_depth_ -= source_lexical;
      VarSet parameters(op->body_params.begin(), op->body_params.end());
      auto body = KeepInside(&child, parameters, barrier);
      return {RegionStmt(op->op, op->args, op->body_params, op->attrs, body,
                         op->result_vars, op->loc),
              child.lifted};
    }
    return {stmt, {}};
  }

  int execution_scope_depth_ = 0;
  int lexical_scope_depth_ = 0;
};

// Inspect only this launch's metadata, never the header of a nested launch.
class CPULaunchAnnotationExtractor : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  Optional<int64_t> num_threads;

  UnchangedOr<Stmt> Mutate_(const ForObj *, InplaceMode) final {
    return Unchanged();
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    if (op->op->name != "tl.attr")
      return StmtExprMutator::Mutate_(op, mode);
    auto values =
        DecodeAnnotations(
            op->args[0].as_or_throw<tvm::Tuple>(),
            op->attrs.GetAttr<Array<Any>>("annotation_schema").value())
            .cast<Map<Any, Any>>();
    if (values.at("key").cast<String>() != "tl.kernel")
      return StmtExprMutator::Mutate_(op, mode);
    auto hints = values.at("value").cast<Map<Any, Any>>();
    if (!hints.count(attr::kCPUNumThreads))
      return Unchanged();
    const auto &value = hints.at(attr::kCPUNumThreads);
    int64_t count;
    if (const auto *imm = value.as<IntImmObj>()) {
      TVM_FFI_CHECK(imm->value.as<int64_t>().has_value(), ValueError)
          << attr::kCPUNumThreads << " is out of range";
      count = imm->value.as<int64_t>().value();
    } else {
      TVM_FFI_CHECK(value.as<int64_t>().has_value(), ValueError)
          << attr::kCPUNumThreads << " must be a positive integer";
      count = value.cast<int64_t>();
    }
    TVM_FFI_CHECK(count > 0 && count <= std::numeric_limits<int32_t>::max(),
                  ValueError)
        << attr::kCPUNumThreads << " must be a positive int32 integer";
    TVM_FFI_CHECK(!num_threads.has_value(), ValueError)
        << "A CPU launch cannot have multiple thread-count headers";
    num_threads = count;
    hints.erase(attr::kCPUNumThreads);
    if (hints.empty())
      return op->body;
    values.Set("value", hints);
    auto encoded = EncodeAnnotations(values);
    return RegionStmt(op->op, {encoded.payload}, op->body_params,
                      DictAttrs({{"schema_version", 1},
                                 {"annotation_schema", encoded.schema}}),
                      op->body, op->result_vars, op->loc);
  }
};

class CPUKernelLaunchLowerer : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit CPUKernelLaunchLowerer(bool parallel) : parallel_(parallel) {}

  UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    if (!IsGrid(op))
      return StmtExprMutator::Mutate_(op, mode);
    std::vector<const ForObj *> chain{op};
    while (IsGrid(SingleGrid(chain.back()->body)))
      chain.push_back(SingleGrid(chain.back()->body));
    auto extractor = make_object<CPULaunchAnnotationExtractor>();
    SeqStmt body = extractor->Mutate(chain.back()->body)
                       .ValueOrUnchanged(chain.back()->body);
    body = Mutate(body, mode).ValueOrUnchanged(body);
    for (size_t i = chain.size(); i-- > 0;) {
      const auto *loop = chain[i];
      auto annotations = loop->annotations;
      if (!parallel_)
        annotations.erase(attr::kCPUGridDim);
      if (i == 0 && extractor->num_threads.has_value())
        annotations.Set(attr::kCPUNumThreads, extractor->num_threads.value());
      body = SeqStmt(For(loop->var, loop->begin, loop->end, loop->step,
                         ForKind::kDefault, body, annotations, loop->loc));
    }
    return body;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    TVM_FFI_CHECK(op->op->name != "tl.launch_kernel", ValueError)
        << "LowerCPUKernelLaunch must run after MaterializeKernelLaunch(false, "
           "false)";
    Stmt rewritten =
        StmtExprMutator::Mutate_(op, mode).ValueOrUnchanged(GetRef<Stmt>(op));
    // A zero-dimensional launch has no grid loop to receive the hint. Drop
    // only its own thread-count hint; the body still executes exactly once.
    auto extractor = make_object<CPULaunchAnnotationExtractor>();
    return extractor->Mutate(rewritten).ValueOrUnchanged(rewritten);
  }

private:
  bool parallel_;
};
} // namespace

tvm::transform::Pass LowerCPUKernelLaunch() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext ctx) {
    if (!(!function->GetAttr(tvm::attr::kDeclaration).value_or(false)))
      return function;
    const bool parallel = ctx->GetConfig<bool>(kCPUParallel).value_or(false);
    auto rewriter = ffi::make_object<CPUKernelLaunchLowerer>(parallel);
    auto body = function->body;
    auto planned = SourceBlockAllocationPlanner().Run(body);
    auto result = rewriter->Mutate(planned).ValueOrUnchanged(planned);
    if (!result.same_as(body))
      function.CopyOnWrite()->body = result;
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.LowerCPUKernelLaunch");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cpu.transform.LowerCPUKernelLaunch",
                                   LowerCPUKernelLaunch);
}
} // namespace tvm::tl
