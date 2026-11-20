#pragma once

#include <optional>
#include <tvm/tirx/stmt_functor.h>

#include "../../support/check.h"
#include "../op/builtin.h"

namespace tvm {
namespace tl {
namespace ascend {

// LowerOpaqueBlock preserves this frontend launch annotation as an AttrStmt.
constexpr const char *kAivCount = "pragma_aiv_count";
// Only frontend MixedKernel / Vector frames request logical AIV grouping.
constexpr const char *kAivGroup = "pragma_aiv_group";

// Share the existing PTO Cube-operation fallback with AscendC. This prevents
// a flat Cube body without a CUBE block from becoming a grouped Vector body.
inline bool IsCubeCall(const tirx::CallNode *call) {
  return call->op.same_as(tl::ascend_gemm_l1()) ||
         call->op.same_as(tl::ascend_blockscaled_gemm_l1()) ||
         call->op.same_as(tl::ascend_mad()) ||
         call->op.same_as(tl::ascend_mad_mx()) ||
         call->op.same_as(tl::ascend_copy_gm_to_cbuf()) ||
         call->op.same_as(tl::ascend_fill_l1()) ||
         call->op.same_as(tl::ascend_load_cbuf_to_ca()) ||
         call->op.same_as(tl::ascend_load_cbuf_to_cb()) ||
         call->op.same_as(tl::ascend_load_ca_sf()) ||
         call->op.same_as(tl::ascend_load_cb_sf()) ||
         call->op.same_as(tl::ascend_copy_ubuf_to_cbuf()) ||
         call->op.same_as(tl::ascend_copy_matrix_cc_to_ub()) ||
         call->op.same_as(tl::ascend_copy_matrix_cc_to_gm());
}

// A grouped Vector-only kernel executes its complete body on the AIV side.
// Preserve existing block scopes, but make them logical regions inside one
// physical Vector section. Thread bindings are deliberately left unchanged.
inline tirx::Stmt MakeGroupedVectorBody(const tirx::Stmt &body, int count) {
  auto logical_regions = tirx::IRTransform(
      body, nullptr,
      ffi::Function::FromTyped([](ffi::ObjectRef node) -> ffi::ObjectRef {
        if (const auto *block = node.as<tirx::SBlockNode>();
            block && block->name_hint == "VECTOR") {
          auto result = ffi::GetRef<tirx::SBlock>(block);
          result.CopyOnWrite()->name_hint = "VECTOR_BODY";
          return result;
        }
        return node;
      }),
      ffi::Array<ffi::String>{"tirx.SBlock"});
  return tirx::SBlock({}, {}, {}, "VECTOR", logical_regions, std::nullopt, {},
                      {}, {{"vector_count", Integer(count)}});
}

inline std::optional<int> ReadAivCount(const tirx::SBlockNode *block,
                                       const char *key) {
  auto value = block->annotations.Get(key);
  if (!value.has_value())
    return std::nullopt;
  PrimExpr expression = value.value().cast<PrimExpr>();
  const auto *count = expression.as<IntImmNode>();
  ICHECK(count && (count->value == 1 || count->value == 2))
      << key << " must be the constant integer 1 or 2 in block '"
      << block->name_hint << "', got " << value.value();
  return static_cast<int>(count->value);
}

// Read the count selected by the frontend without querying hardware again.
inline int GetAivCount(const tirx::Stmt &body) {
  std::optional<int> selected;
  tirx::PostOrderVisit(body, [&](const ffi::ObjectRef &node) {
    const auto *attr = node.as<tirx::AttrStmtNode>();
    if (!attr || attr->attr_key != kAivCount)
      return;
    const auto *count = attr->value.as<IntImmNode>();
    ICHECK(count && (count->value == 1 || count->value == 2))
        << "Invalid frontend AIV count";
    ICHECK(!selected.has_value() || selected.value() == count->value)
        << "Conflicting frontend AIV counts";
    selected = static_cast<int>(count->value);
  });
  ICHECK(selected.has_value()) << "Missing frontend AIV count before CodeGen";
  return selected.value();
}

} // namespace ascend
} // namespace tl
} // namespace tvm
