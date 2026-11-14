#pragma once

#include <optional>
#include <tvm/tirx/stmt_functor.h>

#include "../../support/check.h"

namespace tvm {
namespace tl {
namespace ascend {

// LowerOpaqueBlock preserves this frontend launch annotation as an AttrStmt.
constexpr const char *kAivCount = "pragma_aiv_count";

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

// CodeGen reads the selected launch count; it never normalizes the body.
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
