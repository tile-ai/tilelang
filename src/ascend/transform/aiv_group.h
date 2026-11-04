#pragma once

#include <optional>
#include <tvm/ffi/function.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include "../../support/check.h"

namespace tvm {
namespace tl {
namespace ascend {

// LowerOpaqueBlock carries pragma annotations through the removed kernel root.
constexpr const char *kDefaultAivCount = "pragma_aiv_default_count";

inline std::optional<int> ReadAivCount(const tirx::SBlockNode *block,
                                       const char *key) {
  auto value = block->annotations.Get(key);
  if (!value.has_value())
    return std::nullopt;
  // Annotation values may be an FFI integer or an existing PrimExpr.
  PrimExpr expression = value.value().cast<PrimExpr>();
  const auto *count = expression.as<IntImmNode>();
  ICHECK(count && (count->value == 1 || count->value == 2))
      << key << " must be the constant integer 1 or 2 in block '"
      << block->name_hint << "', got " << value.value();
  return static_cast<int>(count->value);
}

inline int ResolveAivCount(std::optional<int> requested = std::nullopt) {
  if (requested.has_value()) {
    ICHECK(requested.value() == 1 || requested.value() == 2)
        << "AIV group count must be 1 or 2, got " << requested.value();
  }
  // Query hardware internally. Zero or a missing callback means unknown.
  std::optional<int> limit;
  if (auto query = ffi::Function::GetGlobal(
          "tilelang.ascend._query_device_aiv_per_aic")) {
    int64_t ratio = (*query)().cast<int64_t>();
    ICHECK(ratio == 0 || ratio == 1 || ratio == 2)
        << "Internal device AIV capability must be 0 (unknown), 1 or 2";
    if (ratio != 0)
      limit = static_cast<int>(ratio);
  }
  // Explicit kernel request, then current device ratio, then offline default.
  int count = requested.value_or(limit.value_or(2));
  if (limit.has_value()) {
    ICHECK_LE(count, limit.value())
        << "AIV group count=" << count
        << " exceeds the current device capability " << limit.value();
  }
  return count;
}

// One collector for frontend validation and the existing CodeGen pre-scans.
// Requests remain distinct from counts selected by a previous compiler stage.
struct AivGroupInfo {
  std::optional<int> vector_count;
  std::optional<int> resolved_count;
  std::optional<int> default_count;

  void Observe(const ffi::ObjectRef &node) {
    if (const auto *block = node.as<tirx::SBlockNode>()) {
      if (block->name_hint == "VECTOR")
        Merge(ReadAivCount(block, "vector_count"), &vector_count,
              "Conflicting explicit T.Vector counts");
      Merge(ReadAivCount(block, "tl.resolved_aiv_count"), &resolved_count,
            "Conflicting resolved AIV group counts");
      Merge(ReadAivCount(block, kDefaultAivCount), &default_count,
            "Conflicting default AIV group counts");
    } else if (const auto *attr = node.as<tirx::AttrStmtNode>()) {
      if (attr->attr_key == kDefaultAivCount) {
        const auto *value = attr->value.as<IntImmNode>();
        ICHECK(value && (value->value == 1 || value->value == 2))
            << "Invalid default AIV group count";
        Merge(static_cast<int>(value->value), &default_count,
              "Conflicting default AIV group counts");
      }
    }
  }

  int MixedCount() const {
    auto count = resolved_count.has_value() ? resolved_count : default_count;
    ICHECK(count.has_value())
        << "AIV count must be prepared by ResolveAivGroup before CodeGen";
    return count.value();
  }

private:
  static void Merge(std::optional<int> value, std::optional<int> *selected,
                    const char *message) {
    if (!value.has_value())
      return;
    ICHECK(!selected->has_value() || selected->value() == value.value())
        << message << ": " << selected->value_or(-1) << " versus "
        << value.value();
    *selected = value;
  }
};

// Bind an already selected mixed-group count. This helper neither infers
// core types nor creates a new outer sid domain.
class AivDomainNormalizer : public tirx::StmtMutator {
public:
  AivDomainNormalizer(int count, ffi::Optional<tirx::Var> sid = std::nullopt,
                      bool unify_sid = true)
      : count_(count), sid_(sid), unify_sid_(unify_sid) {}

private:
  tirx::Stmt VisitStmt_(const tirx::AttrStmtNode *op) final {
    const auto *iv = op->node.as<tirx::IterVarNode>();
    if (op->attr_key != tirx::attr::thread_extent || !iv ||
        iv->thread_tag != "cthread")
      return StmtMutator::VisitStmt_(op);
    if (sid_.has_value() && unify_sid_) {
      if (iv->var.same_as(sid_.value()))
        return VisitStmt(op->body);
      ffi::Map<tirx::Var, PrimExpr> replacements;
      replacements.Set(iv->var, sid_.value());
      return VisitStmt(tirx::Substitute(op->body, replacements));
    }
    auto previous = sid_;
    sid_ = iv->var;
    tirx::Stmt body = VisitStmt(op->body);
    sid_ = previous;
    if (body.same_as(op->body) && tirx::is_const_int(op->value, count_) &&
        iv->dom.defined() && tirx::is_zero(iv->dom->min) &&
        tirx::is_const_int(iv->dom->extent, count_))
      return ffi::GetRef<tirx::Stmt>(op);
    tirx::IterVar thread(Range(0, count_), iv->var, iv->iter_type,
                         iv->thread_tag);
    return tirx::AttrStmt(thread, op->attr_key, Integer(count_), body,
                          op->span);
  }
  tirx::Stmt VisitStmt_(const tirx::SBlockNode *op) final {
    if (op->name_hint != "VECTOR")
      return StmtMutator::VisitStmt_(op);
    auto requested = ReadAivCount(op, "vector_count");
    ICHECK(!requested.has_value() || requested.value() == count_)
        << "Conflicting AIV group settings: explicit T.Vector count "
        << "does not match resolved kernel count " << count_;
    auto block = Downcast<tirx::SBlock>(StmtMutator::VisitStmt_(op));
    if (requested == count_ &&
        ReadAivCount(op, "tl.resolved_aiv_count") == count_)
      return block;
    auto *writer = block.CopyOnWrite();
    writer->annotations.Set("vector_count", Integer(count_));
    writer->annotations.Set("tl.resolved_aiv_count", Integer(count_));
    return block;
  }
  int count_;
  ffi::Optional<tirx::Var> sid_;
  bool unify_sid_;
};

} // namespace ascend
} // namespace tl
} // namespace tvm
