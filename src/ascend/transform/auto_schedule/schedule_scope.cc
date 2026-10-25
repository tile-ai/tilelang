#include "schedule_scope.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/op.h>

namespace tvm::tl::ascend {
namespace {

ffi::Array<Var> ScheduleBodyParameters(const CallNode *) { return {}; }

void ValidateScheduleScope(const RegionStmtNode *region) {
  TVM_FFI_CHECK(!ScheduleMarkerKey(region).empty(), ValueError)
      << "An Ascend schedule scope requires a scheduling marker";
  auto values = region->attrs.GetAttr<ffi::Array<Expr>>(kScheduleYield)
                    .value_or(ffi::Array<Expr>{});
  TVM_FFI_CHECK_EQ(values.size(), region->result_vars.size(), ValueError)
      << "Scheduled result and yield counts must match";
  ffi::Map<Var, Var> replacements;
  for (size_t index = 0; index < values.size(); ++index) {
    replacements.Set(values[index].as_or_throw<Var>(),
                     region->result_vars[index]);
  }
  auto remapper = ffi::make_object<ScheduleVarRemapper>(replacements);
  for (size_t index = 0; index < values.size(); ++index) {
    TVM_FFI_CHECK(values[index].as<Var>().has_value(), ValueError)
        << "Scheduled yields must be lexical variable definitions";
    auto type =
        remapper->Mutate(values[index]->ty).ValueOrUnchanged(values[index]->ty);
    TVM_FFI_CHECK(ffi::StructuralEqual()(type, region->result_vars[index]->ty),
                  TypeError)
        << "Scheduled result and yield types must match";
  }
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.ascend.schedule_scope",
        "Ascend scheduled region with explicit lexical results.")
      .signature(sig::call_attrs<DictAttrsNode>())
      .set_native_fn_attr<&ScheduleBodyParameters>(
          tvm::op_attr::kRegionGetBodyParams)
      .set_native_fn_attr<&ValidateScheduleScope>(
          tvm::op_attr::kRegionValidate);
}

} // namespace tvm::tl::ascend
