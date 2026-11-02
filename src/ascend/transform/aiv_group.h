#pragma once

#include <optional>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/ffi/function.h>
#include "../../support/check.h"

namespace tvm { namespace tl { namespace ascend {
inline int ResolveAivCount(std::optional<int> requested = std::nullopt) {
  // Query hardware internally. Zero or a missing callback means unknown.
  std::optional<int> limit;
  if (auto query = ffi::Function::GetGlobal(
          "tilelang.ascend._query_device_aiv_per_aic")) {
    int64_t ratio = (*query)().cast<int64_t>();
    ICHECK(ratio == 0 || ratio == 1 || ratio == 2)
        << "Internal device AIV capability must be 0 (unknown), 1 or 2";
    if (ratio != 0) limit = static_cast<int>(ratio);
  }
  if (requested.has_value()) {
    ICHECK(requested.value() == 1 || requested.value() == 2)
        << "AIV group count must be 1 or 2, got " << requested.value();
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
    if (op->attr_key != tirx::attr::thread_extent || !iv || iv->thread_tag != "cthread")
      return StmtMutator::VisitStmt_(op);
    if (sid_.has_value() && unify_sid_) {
      ffi::Map<tirx::Var, PrimExpr> replacements;
      replacements.Set(iv->var, sid_.value());
      return VisitStmt(tirx::Substitute(op->body, replacements));
    }
    auto previous = sid_;
    sid_ = iv->var;
    tirx::Stmt body = VisitStmt(op->body);
    sid_ = previous;
    tirx::IterVar thread(Range(0, count_), iv->var, iv->iter_type, iv->thread_tag);
    return tirx::AttrStmt(thread, op->attr_key, Integer(count_), body, op->span);
  }
  tirx::Stmt VisitStmt_(const tirx::SBlockNode *op) final {
    if (op->name_hint == "VECTOR") {
      if (auto value = op->annotations.Get("vector_count")) {
        const auto *imm = value.value().as<IntImmNode>();
        ICHECK(imm && imm->value == count_)
            << "Conflicting AIV group settings: explicit T.Vector count "
            << "does not match resolved kernel count " << count_;
      }
    }
    auto block = Downcast<tirx::SBlock>(StmtMutator::VisitStmt_(op));
    if (op->name_hint == "VECTOR") {
      block.CopyOnWrite()->annotations.Set("vector_count", Integer(count_));
      block.CopyOnWrite()->annotations.Set("tl.resolved_aiv_count", Integer(count_));
    }
    return block;
  }
  int count_;
  ffi::Optional<tirx::Var> sid_;
  bool unify_sid_;
};

// An explicit Vector-only group is a request bounded by device capacity,
// not an assertion that its count equals the hardware default.
inline std::optional<int> ExplicitVectorCount(const tirx::Stmt &body) {
  std::optional<int> requested;
  tirx::PostOrderVisit(body, [&](const ffi::ObjectRef &node) {
    const auto *block = node.as<tirx::SBlockNode>();
    if (!block || block->name_hint != "VECTOR") return;
    if (auto value = block->annotations.Get("vector_count")) {
      int count = Downcast<IntImm>(value.value().cast<PrimExpr>())->value;
      ICHECK(!requested.has_value() || requested.value() == count)
          << "Conflicting explicit T.Vector counts";
      requested = count;
    }
  });
  return requested;
}

// CodeGen calls this only after its existing mixed-kernel classification.
inline int ResolveMixedCodegenCount(const tirx::Stmt &body) {
  std::optional<int> selected;
  tirx::PostOrderVisit(body, [&](const ffi::ObjectRef &node) {
    const auto *block = node.as<tirx::SBlockNode>();
    if (!block) return;
    if (auto value = block->annotations.Get("tl.resolved_aiv_count")) {
      int count = Downcast<IntImm>(value.value().cast<PrimExpr>())->value;
      ICHECK(!selected.has_value() || selected.value() == count)
          << "Conflicting resolved AIV group counts: previously selected "
          << selected.value_or(-1) << ", but block '" << block->name_hint
          << "' has tl.resolved_aiv_count=" << count;
      selected = count;
    }
  });
  return ResolveAivCount(selected);
}
}}} // namespace tvm::tl::ascend
