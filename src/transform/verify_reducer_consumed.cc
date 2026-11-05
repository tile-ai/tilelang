/*!
 * \file verify_reducer_consumed.cc
 * \brief Reject any first-class reducer contract left after materialization.
 */
#include "common/native_reducer_utils.h"

#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
void CheckMetadata(const reducer_native::Metadata &metadata) {
  TVM_FFI_CHECK(!metadata.count(attr::kReducerInfoV2), ValueError)
      << "reducer_info_v2 annotation survived past ReducerPlanAndMaterialize";
  if (auto layouts = metadata.Get(attr::kLayoutMap)) {
    for (const auto &[tensor, layout] : reducer_native::ReadLayouts(*layouts))
      TVM_FFI_CHECK(!IsReducerV2Buffer(tensor) &&
                        !layout.as<PartialFragmentNode>(),
                    ValueError)
          << "Partial reducer layout survived past ReducerPlanAndMaterialize "
             "for "
          << tensor;
  }
}

void CheckFunction(const tirx::Function &function) {
  if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return;
  VisitNativePostOrder(function->body, [&](const ffi::ObjectRef &object) {
    if (auto tensor = object.as<tirx::TensorVar>()) {
      TVM_FFI_CHECK(!IsReducerV2Buffer(*tensor), ValueError)
          << "local.reducer tensor survived past ReducerPlanAndMaterialize: "
          << *tensor;
    } else if (const auto *call = object.as<CallObj>()) {
      TVM_FFI_CHECK(!call->op.same_as(ReducerInitOp::Get()) &&
                        !call->op.same_as(reducer_update()) &&
                        !call->op.same_as(FinalizeReducerV2Op::Get()),
                    ValueError)
          << "First-class reducer op survived past ReducerPlanAndMaterialize: "
          << call->op;
      if (reducer_native::IsMetadataCall(call))
        CheckMetadata(ReadNativeAnnotations(call));
    } else if (const auto *region = object.as<RegionStmtObj>();
               region && region->op->name == "tl.attr") {
      auto fields = ReadNativeAnnotations(region);
      auto key = fields.at("key").as_or_throw<ffi::String>();
      TVM_FFI_CHECK(key != attr::kParallelMultiplicity, ValueError)
          << "Unconsumed tl.parallel_multiplicity marker survived past "
             "PartitionLoop";
      if (key == "tl.kernel")
        CheckMetadata(reducer_native::StringMap(fields.at("value")));
      else if (reducer_native::IsReducerMetadataKey(key))
        CheckMetadata({{key, fields.at("value")}});
    }
  });
}
} // namespace

tvm::transform::Pass VerifyReducerConsumed() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
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
