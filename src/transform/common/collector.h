/*!
 * \file collector.h
 * \brief Read hardware-thread definitions from native launch regions.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_COLLECTOR_H_
#define TVM_TL_TRANSFORM_COMMON_COLLECTOR_H_

#include "thread_sync_types.h"
#include <optional>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <vector>

namespace tvm::tl {

inline std::vector<NativeThreadBinding>
GetRegionThreadBindings(const RegionStmtObj *region) {
  std::vector<NativeThreadBinding> result;
  if (region->op.same_as(tirx::launch_thread_op())) {
    PrimExpr extent = region->args[1].as_or_throw<PrimExpr>();
    result.push_back({region->body_params[0].as_or_throw<PrimVar>(),
                      Range::FromMinExtent(IntImm(extent.ty(), 0), extent),
                      region->args[0].as_or_throw<StringImm>()->value});
  } else if (region->op->name == "tl.launch_kernel") {
    auto begins = region->args[0].as_or_throw<Tuple>()->fields;
    auto ends = region->args[1].as_or_throw<Tuple>()->fields;
    auto threads = region->args[3].as_or_throw<Tuple>()->fields;
    for (size_t i = 0; i < ends.size(); ++i) {
      result.push_back({region->body_params[i].as_or_throw<PrimVar>(),
                        Range(begins[i].as_or_throw<PrimExpr>(),
                              ends[i].as_or_throw<PrimExpr>()),
                        std::string("blockIdx.") + "xyz"[i]});
    }
    for (size_t i = 0; i < threads.size(); ++i) {
      PrimExpr extent = threads[i].as_or_throw<PrimExpr>();
      result.push_back(
          {region->body_params[ends.size() + i].as_or_throw<PrimVar>(),
           Range::FromMinExtent(IntImm(extent.ty(), 0), extent),
           std::string("threadIdx.") + "xyz"[i]});
    }
  }
  return result;
}

class ThreadTagChecker : public tirx::StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;

  static bool HasOnlyThreadIdxX(const tirx::Function &function) {
    auto checker = ffi::make_object<ThreadTagChecker>();
    if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      checker->Visit(function->body);
    return checker->is_valid_;
  }

  static std::optional<NativeThreadBinding> GetThreadVar(const Stmt &body) {
    auto checker = ffi::make_object<ThreadTagChecker>();
    checker->Visit(body);
    return checker->thread_var_;
  }

private:
  ffi::Optional<ffi::VisitInterrupt> Visit_(const RegionStmtObj *op) final {
    for (const auto &binding : GetRegionThreadBindings(op)) {
      if (binding.tag == "threadIdx.x")
        thread_var_ = binding;
      if ((binding.tag == "threadIdx.y" || binding.tag == "threadIdx.z") &&
          !prim::IsOne(binding.dom->extent))
        is_valid_ = false;
    }
    return StmtExprVisitor::Visit_(op);
  }
  std::optional<NativeThreadBinding> thread_var_;
  bool is_valid_{true};
};

} // namespace tvm::tl

#endif // TVM_TL_TRANSFORM_COMMON_COLLECTOR_H_
