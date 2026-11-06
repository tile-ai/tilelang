/*!
 * \file ascend/ir/block.cc
 * \brief Ascend execution domains over native TIRx statements.
 */

#include "block.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>

namespace tvm::tl::ascend {
namespace {

void ValidateExecutionAttrs(const DictAttrs &attrs) {
  auto name = attrs.GetAttr<ffi::String>("name");
  TVM_FFI_CHECK(name && (*name == "CUBE" || *name == "VECTOR" ||
                         *name == "SIMT_VF" || *name == "SIMD_VF"),
                ValueError)
      << "tl.ascend_block name must be CUBE, VECTOR, SIMT_VF or SIMD_VF";
  const bool vf = *name == "SIMT_VF" || *name == "SIMD_VF";
  for (const auto &[key, value] : attrs->dict) {
    if (key == "name")
      continue;
    const bool vector_count = *name == "VECTOR" && key == "vector_count";
    TVM_FFI_CHECK(vector_count || (vf && (key == "tl.vf_source_index" ||
                                          key == "tl.vf_latency")),
                  ValueError)
        << "Unknown Ascend execution attribute: " << key;
    auto count = value.try_cast<int64_t>();
    if (const auto *literal = value.as<IntImmObj>())
      count = literal->value.as<int64_t>();
    TVM_FFI_CHECK(count.has_value(), ValueError)
        << "Ascend execution attribute must be a constant integer: " << key;
    TVM_FFI_CHECK(vector_count ? (*count == 1 || *count == 2) : *count >= 0,
                  ValueError)
        << "Invalid Ascend execution attribute " << key << ": " << *count;
  }
}

ffi::Array<Var> ExecutionBodyParams(const CallObj *call) {
  TVM_FFI_CHECK(call->args.empty() && call->ty_args.empty(), ValueError)
      << "tl.ascend_block has no operands or type arguments";
  ValidateExecutionAttrs(call->attrs.value().as_or_throw<DictAttrs>());
  return {};
}

void ValidateExecutionRegion(const RegionStmtObj *region) {
  TVM_FFI_CHECK(region->args.empty() && region->body_params.empty() &&
                    region->result_vars.empty(),
                ValueError)
      << "tl.ascend_block has no operands, body parameters or results";
  ValidateExecutionAttrs(region->attrs);
}

} // namespace

const Op &ascend_block_op() {
  static const Op op = Op::Get("tl.ascend_block");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.ascend_block",
        "An Ascend core or vector-function execution domain.")
      .signature(sig::call_attrs<DictAttrsObj>())
      .set_native_fn_attr<&ExecutionBodyParams>(op_attr::kRegionGetBodyParams)
      .set_native_fn_attr<&ValidateExecutionRegion>(op_attr::kRegionValidate);
}

} // namespace tvm::tl::ascend
