/*!
 * \file storage_rewrite.cc
 * \brief TileLang allocation policy without an upstream planner policy fork.
 */
#include "../ir/loop_math.h"
#include "../op/builtin.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl::transform {
namespace {
using namespace tvm::tirx;
using namespace tvm::prim;

// A tensor that escapes through an address, view, annotation, call, or other
// opaque use is never a candidate. Direct scalar/vector loads and stores are
// tracked without treating their tensor operands as opaque uses.
class TensorAccesses final : public StmtExprVisitor {
public:
  std::unordered_set<const VarObj *> used;
  std::unordered_set<const VarObj *> escaped;

  ffi::Optional<VisitInterrupt> Visit_(const VarObj *node) final {
    if (def_region_kind() == kTVMFFIDefRegionKindNone &&
        node->ty.as<TensorTypeObj>()) {
      used.insert(node);
      escaped.insert(node);
    }
    return StmtExprVisitor::Visit_(node);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj *node) final {
    used.insert(node->source.as_or_throw<TensorVar>().get());
    return Visit(node->indices);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj *node) final {
    used.insert(node->dest.as_or_throw<TensorVar>().get());
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(node->indices));
    return Visit(node->value);
  }
};

// This is deliberately an optimization verifier, not a general memory planner.
// Require one store, pointwise source loads, and a provably single-visit index.
// In particular, equal source/destination indices alone are insufficient: a
// loop repeatedly writing B[0] = A[0] + 1 must not turn into repeated
// increments.
class PointwiseValueVerifier final : public StmtExprVisitor {
public:
  PointwiseValueVerifier(const TensorStoreObj *store, const VarObj *source)
      : store_(store), source_(source) {}
  bool safe = true;
  bool reads_source = false;

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj *node) final {
    if (node->source.same_as(store_->dest))
      safe = false;
    if (node->source.get() == source_) {
      reads_source = true;
      if (node->indices.size() != store_->indices.size()) {
        safe = false;
      } else {
        for (size_t i = 0; i < node->indices.size(); ++i)
          safe &= ExprDeepEqual()(node->indices[i], store_->indices[i]);
      }
    }
    return StmtExprVisitor::Visit_(node);
  }

private:
  const TensorStoreObj *store_;
  const VarObj *source_;
};

const StmtObj *SingleStatement(const StmtObj *stmt) {
  while (stmt->IsInstance<SeqStmtObj>()) {
    const auto *sequence = static_cast<const SeqStmtObj *>(stmt);
    if (sequence->seq.size() != 1)
      return nullptr;
    stmt = sequence->seq[0].get();
  }
  return stmt;
}

bool IsPointwiseUpdate(const Stmt &statement, const VarObj *destination,
                       const VarObj *source) {
  const StmtObj *stmt = SingleStatement(statement.get());
  if (!stmt)
    return false;
  const ForObj *loop =
      stmt->IsInstance<ForObj>() ? static_cast<const ForObj *>(stmt) : nullptr;
  if (loop) {
    stmt = SingleStatement(loop->body.get());
    if (!stmt)
      return false;
  }
  const auto *store = stmt->IsInstance<TensorStoreObj>()
                          ? static_cast<const TensorStoreObj *>(stmt)
                          : nullptr;
  if (!store || store->dest.get() != destination || store->indices.size() != 1)
    return false;
  if (loop) {
    // The native loop always uses var < end. A negative step or an update
    // that wraps can revisit an index even when the source/destination indices
    // match. Require a monotone, nonwrapping traversal before coalescing.
    sym::Analyzer analyzer;
    if (loop->var.ty() != loop->begin.ty() ||
        !CanUseLoopBounds(loop->begin, loop->end, loop->step, analyzer.get()) ||
        !store->indices[0].same_as(loop->var))
      return false;
  }
  auto values = ffi::make_object<PointwiseValueVerifier>(store, source);
  values->Visit(store->value);
  return values->safe && values->reads_source;
}

class InplaceAllocationRewriter final : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  InplaceAllocationRewriter(const tirx::Function &function,
                            std::unordered_set<const VarObj *> escaped)
      : escaped_(std::move(escaped)) {
    if (auto alignment = function->attrs->dict.Get("tl.smem_alignment_map")) {
      if (auto names = alignment->as<ffi::Map<ffi::String, ffi::Any>>())
        for (const auto &[name, value] : *names)
          constrained_names_.insert(name);
    }
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *node, InplaceMode mode) final {
    // Recurse before examining this lexical sequence. Never move an allocation
    // across a loop, thread domain, conditional, or explicit scope boundary.
    auto rewritten = StmtExprMutator::Mutate_(node, mode)
                         .ValueOrUnchanged(ffi::GetRef<Stmt>(node))
                         .as_or_throw<SeqStmt>();
    struct Allocation {
      const BindObj *binding;
      size_t definition;
      size_t first = std::numeric_limits<size_t>::max();
      size_t last = 0;
    };
    std::vector<Allocation> allocations;
    for (size_t i = 0; i < rewritten->seq.size(); ++i) {
      const auto *binding = rewritten->seq[i].as<BindObj>();
      if (binding && IsCandidate(binding))
        allocations.push_back({binding, i});
    }
    if (allocations.size() < 2)
      return rewritten;
    for (size_t i = 0; i < rewritten->seq.size(); ++i) {
      auto accesses = ffi::make_object<TensorAccesses>();
      accesses->Visit(rewritten->seq[i]);
      for (auto &allocation : allocations) {
        if (accesses->used.count(allocation.binding->var.get())) {
          allocation.first = std::min(allocation.first, i);
          allocation.last = i;
        }
      }
    }
    for (size_t dst = 1; dst < allocations.size(); ++dst) {
      const auto &destination = allocations[dst];
      if (destination.first <= destination.definition ||
          destination.first == std::numeric_limits<size_t>::max())
        continue;
      for (size_t src = 0; src < dst; ++src) {
        const auto &source = allocations[src];
        if (source.last != destination.first ||
            source.definition >= destination.definition ||
            !ffi::StructuralEqual()(source.binding->var->ty,
                                    destination.binding->var->ty) ||
            !IsPointwiseUpdate(rewritten->seq[destination.first],
                               destination.binding->var.get(),
                               source.binding->var.get()))
          continue;
        Var replacement = source.binding->var;
        if (auto previous = replacements_.Get(replacement))
          replacement = *previous;
        replacements_.Set(destination.binding->var, replacement);
        break;
      }
    }
    if (replacements_.empty())
      return rewritten;
    // Traverse the same sequence with the proven replacements; removed Bind
    // definitions do not create aliases that could overwrite the remap table.
    return StmtExprMutator::Mutate_(rewritten.get(), InplaceMode::kDisallow)
        .ValueOrUnchanged(static_cast<const Stmt &>(rewritten));
  }

  UnchangedOr<Stmt> Mutate_(const BindObj *node, InplaceMode mode) final {
    if (replacements_.count(node->var))
      return Evaluate(IntImm::Int32(0), node->loc);
    return StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Expr> Mutate_(const VarObj *node, InplaceMode) final {
    if (auto replacement = replacements_.Get(ffi::GetRef<Var>(node)))
      return *replacement;
    return ffi::Unchanged();
  }

private:
  bool IsCandidate(const BindObj *binding) const {
    if (escaped_.count(binding->var.get()) ||
        constrained_names_.count(binding->var->name))
      return false;
    const auto *call = binding->value.as<CallObj>();
    if (!call || !call->op.same_as(tirx::alloc_tensor_op()) ||
        !call->args.empty())
      return false;
    if (call->attrs) {
      const auto *annotations = call->attrs.as<DictAttrsObj>();
      if (!annotations || !annotations->dict.empty())
        return false;
    }
    const auto *type = binding->var->ty.as<TensorTypeObj>();
    if (!type || type->shape.size() != 1 || type->layout ||
        type->dtype.IsScalableVector() || type->dtype.lanes() != 1)
      return false;
    const auto *size = type->shape[0].as<IntImmObj>();
    if (!size || size->value <= 0)
      return false;
    const auto &scope = type->scope.name();
    // Shared/global arrays can become one physical allocation. Local arrays
    // keep their separate-register contract; a scalar temporary with no address
    // uses may be eliminated without introducing a register array or pointer.
    return scope == "shared" || scope == "shared.dyn" || scope == "global" ||
           (scope == "local" && size->value == 1);
  }

  std::unordered_set<const VarObj *> escaped_;
  std::unordered_set<ffi::String> constrained_names_;
  ffi::Map<Var, Var> replacements_;
};
} // namespace

tvm::transform::Pass StorageRewrite() {
  auto rewrite = [](tirx::Function function, IRModule,
                    tvm::transform::PassContext context) {
    if (!context->GetConfig<bool>(kStorageRewriteDetectInplace).value_or(false))
      return function;
    auto accesses = ffi::make_object<TensorAccesses>();
    accesses->Visit(function);
    auto inplace = ffi::make_object<InplaceAllocationRewriter>(
        function, std::move(accesses->escaped));
    auto body = function->body;
    auto rewritten = inplace->Mutate(body).ValueOrUnchanged(body);
    if (!rewritten.same_as(body))
      function.CopyOnWrite()->body = rewritten;
    return function;
  };
  // Lifetime reuse/packing belong to MergeSharedMemoryAllocations; lexical
  // allocation lifting belongs to its backend owner. Preserve scalar element
  // types for TileLang's vector-access emitters instead of running the native
  // planner's unconditional PointerValueTypeRewrite.
  return tirx::transform::CreateFunctionPass(rewrite, 0, "tl.StorageRewrite");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.transform.StorageRewrite", StorageRewrite)
      .def("tl.transform.PointerValueTypeRewrite",
           tirx::transform::PointerValueTypeRewrite);
}
} // namespace tvm::tl::transform
