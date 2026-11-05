/*!
 * \file testing/cpp/native_constraint_probe.cc
 * \brief Exercise the real loop collector and two-instance renaming protocol.
 */
#include "transform/common/constr_visitor.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl {
namespace {
class LoopContextProbe : public ConstrVisitor {
public:
  Result Visit_(const TensorStoreObj *) override {
    context = GetConstrSet();
    return std::nullopt;
  }
  ConstrSet context;
};
} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.testing.RenamedLoopInstances",
      [](tirx::Function function, Var index, bool distinct_coordinates) {
        auto collector = ffi::make_object<LoopContextProbe>();
        collector->Visit(function->body);
        ffi::Map<Var, PrimExpr> left_map, right_map;
        auto left = collector->context.RenameFrom("_left", left_map, index);
        auto right = collector->context.RenameFrom("_right", right_map, index);
        auto joined = left.Merge(right);
        if (distinct_coordinates)
          joined.AddConstr(left_map.at(index) != right_map.at(index));
        PrimVar unrelated("unrelated", PrimType::Int(32));
        return joined.CanProve(unrelated < IntImm(PrimType::Int(32), 0));
      });
}
} // namespace tvm::tl
