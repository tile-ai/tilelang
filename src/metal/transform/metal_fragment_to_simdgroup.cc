/*!
 * \file metal_fragment_to_simdgroup.cc
 * \brief Preserve logical GEMM regions until opaque per-warp storage lowering.
 */
#include "ir/alias_analysis.h"
#include "ir/annotations.h"
#include "metal/target_utils.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/stmt_functor.h>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

#include <limits>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

constexpr const char *kPhysicalElements =
    "tl.metal_simdgroup_physical_elements";

class AccumulatorCollector : public StmtExprVisitor {
public:
  explicit AccumulatorCollector(const PhysicalRootAnalysis *roots)
      : roots_(roots) {}
  using StmtExprVisitor::Visit_;

  Optional<VisitInterrupt> Visit_(const RegionStmtObj *region) final {
    int64_t old_threads = threads_;
    if (region->op->name == "tirx.launch_thread" && region->args.size() == 2 &&
        region->args[0].as_or_throw<StringImm>()->value == "threadIdx.x") {
      auto extent = region->args[1].as<IntImmObj>();
      TVM_FFI_CHECK(extent && extent->value.as<int64_t>(), ValueError)
          << "Metal SIMD-group GEMM requires a static thread extent";
      threads_ = extent->value.as<int64_t>().value();
    }
    auto result = StmtExprVisitor::Visit_(region);
    threads_ = old_threads;
    return result;
  }

  Optional<VisitInterrupt> Visit_(const CallObj *call) final {
    auto op = call->op.as<Op>();
    if (op && op.value()->name == "tl.tileop.gemm") {
      auto arguments = DecodeTileOpArguments(call).operands;
      TVM_FFI_CHECK_GE(arguments.size(), 3U, ValueError);
      auto region = arguments[2].as<TensorRegion>();
      if (region) {
        auto tensor = region.value()->source.as<TensorVar>();
        if (tensor && tensor.value().scope().name() == "local.fragment") {
          TVM_FFI_CHECK_GT(threads_, 0, ValueError);
          TVM_FFI_CHECK_EQ(threads_ % 32, 0, ValueError)
              << "Metal SIMD-group GEMM requires complete 32-thread SIMD "
                 "groups";
          auto root = roots_->Root(*tensor);
          TVM_FFI_CHECK(root, ValueError)
              << "Cannot resolve Metal accumulator storage";
          int64_t warps = threads_ / 32;
          if (auto previous = root_warps.Get(*root))
            TVM_FFI_CHECK_EQ(*previous, warps, ValueError)
                << "One Metal accumulator cannot use conflicting warp counts";
          root_warps.Set(*root, warps);
        }
      }
    }
    return StmtExprVisitor::Visit_(call);
  }

  Map<Var, int64_t> root_warps;

private:
  const PhysicalRootAnalysis *roots_;
  int64_t threads_{32};
};

class AccumulatorPromoter : public StmtExprMutator {
public:
  Map<TensorVar, int64_t> physical_elements;
  using StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const BindObj *binding, InplaceMode mode) final {
    auto tensor = binding->var.as<TensorVar>();
    if (!tensor || !physical_elements.count(*tensor))
      return StmtExprMutator::Mutate_(binding, mode);
    const auto *call = binding->value.as<CallObj>();
    if (!call)
      return StmtExprMutator::Mutate_(binding, mode);
    TVM_FFI_CHECK(call->op.same_as(alloc_tensor_op()) ||
                      call->op.same_as(decl_tensor_op()),
                  ValueError)
        << "Metal accumulator storage must be an explicit allocation or tensor "
           "view";
    auto type = make_object<TensorTypeObj>(*tensor.value().ty().get());
    type->scope = runtime::MemoryScope::Get("metal.simdgroup");
    TensorVar replacement(tensor.value()->name, TensorType(type),
                          tensor.value()->loc);
    Array<Expr> operands;
    for (const Expr &operand : call->args)
      operands.push_back(Mutate(operand, mode).ValueOrUnchanged(operand));
    Map<String, Any> attributes;
    if (call->attrs)
      attributes = call->attrs.value().as_or_throw<DictAttrs>()->dict;
    attributes.Set(kPhysicalElements, physical_elements.at(*tensor));
    Expr value = Call(replacement.ty(), call->op, operands,
                      DictAttrs(attributes), {replacement.ty()}, call->loc);
    VarRemapSet(*tensor, replacement);
    return Bind(replacement, value, binding->loc);
  }
};

tirx::Function Promote(tirx::Function function) {
  auto target = function->GetAttr<Target>(tvm::attr::kTarget);
  if (!target || !TargetIsMetal(*target) ||
      function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return function;
  PhysicalRootAnalysis roots(function);
  auto collector = make_object<AccumulatorCollector>(&roots);
  collector->Visit(function->body);
  if (collector->root_warps.empty())
    return function;
  auto promoter = make_object<AccumulatorPromoter>();
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      function->body,
      [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
        auto tensor = node.as<TensorVar>();
        if (!tensor || tensor.value().scope().name() != "local.fragment")
          return ffi::WalkResult::Advance();
        auto root = roots.Root(*tensor);
        if (!root)
          return ffi::WalkResult::Advance();
        auto warps = collector->root_warps.Get(*root);
        if (!warps)
          return ffi::WalkResult::Advance();
        BigInt elements = 1;
        for (const PrimExpr &dim : tensor.value().shape()) {
          const auto *extent = dim.as<IntImmObj>();
          TVM_FFI_CHECK(extent && extent->value > 0, ValueError)
              << "Metal SIMD-group accumulators require positive static shapes";
          elements = elements * extent->value;
        }
        auto count = elements.as<int64_t>();
        TVM_FFI_CHECK(count &&
                          *warps <= std::numeric_limits<int64_t>::max() / 64,
                      ValueError)
            << "Metal accumulator storage size exceeds int64";
        TVM_FFI_CHECK_EQ(*count % (64 * *warps), 0, ValueError)
            << "Each warp must own complete 8x8 Metal accumulator matrices";
        promoter->physical_elements.Set(*tensor, *count / *warps);
        return ffi::WalkResult::Advance();
      });
  SeqStmt body =
      promoter->Mutate(function->body).ValueOrUnchanged(function->body);
  function.CopyOnWrite()->body = std::move(body);
  return function;
}
} // namespace

namespace transform {
tvm::transform::Pass MetalFragmentToSimdgroup() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    return Promote(std::move(function));
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.MetalFragmentToSimdgroup");
}
} // namespace transform

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.MetalFragmentToSimdgroup",
                              transform::MetalFragmentToSimdgroup);
}
} // namespace tvm::tl
