#include "compat/visit.h"
#include "op/reducer.h"

#include <tvm/tirx/transform.h>

namespace tvm::tl {

namespace {

void CheckAnnotations(const ffi::Map<ffi::String, ffi::Any> &annotations) {
  TVM_FFI_CHECK(!annotations.count(attr::kReducerInfoV2), InternalError)
      << "reducer_info_v2 survived past ReducerPlanAndMaterialize";
  TVM_FFI_CHECK(!annotations.count(attr::kParallelMultiplicity), InternalError)
      << "unconsumed tl.parallel_multiplicity marker survived past "
         "PartitionLoop";
  if (auto layouts = annotations.Get(attr::kLayoutMap)) {
    for (const auto &[tensor, layout] : layouts.value().cast<LayoutMap>()) {
      TVM_FFI_CHECK(!IsReducerV2Buffer(tensor), InternalError)
          << "partial reducer layout survived past ReducerPlanAndMaterialize";
    }
  }
}

void CheckFunction(const tirx::Function &function) {
  CheckAnnotations(function->attrs->dict);
  if (!function->body) {
    return;
  }
  PostOrderVisit(function->body.value(), [](const ffi::ObjectRef &object) {
    if (auto tensor = object.as<tirx::TensorVar>()) {
      TVM_FFI_CHECK(!IsReducerV2Buffer(tensor.value()), InternalError)
          << "reducer " << tensor.value()
          << " (scope local.reducer) survived past ReducerPlanAndMaterialize";
    }
    if (const auto *call = object.as<CallNode>()) {
      TVM_FFI_CHECK(!call->op.same_as(ReducerInitOp::Get()) &&
                        !call->op.same_as(reducer_update()) &&
                        !call->op.same_as(FinalizeReducerV2Op::Get()),
                    InternalError)
          << "first-class reducer op " << call->op
          << " survived past ReducerPlanAndMaterialize";
      if (call->attrs) {
        if (auto attrs = call->attrs.value().as<DictAttrs>()) {
          CheckAnnotations(attrs.value()->dict);
        }
      }
    }
    if (const auto *region = object.as<RegionStmtNode>()) {
      TVM_FFI_CHECK(region->op->name != attr::kParallelMultiplicity,
                    InternalError)
          << "unconsumed tl.parallel_multiplicity marker survived past "
             "PartitionLoop";
      CheckAnnotations(region->attrs->dict);
    }
    if (const auto *loop = object.as<ForNode>()) {
      CheckAnnotations(loop->annotations);
    }
  });
}

} // namespace

tvm::transform::Pass VerifyReducerConsumed() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    CheckFunction(function);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.VerifyReducerConsumed");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.VerifyReducerConsumed",
                                   VerifyReducerConsumed);
}

} // namespace tvm::tl
