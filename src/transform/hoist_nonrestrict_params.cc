/*
 * Hoist tl.non_restrict_params block annotation(s) to PrimFunc attribute.
 *
 * Previously, we only looked at the root block. This version recursively
 * scans all blocks, unions any tl.non_restrict_params entries it finds,
 * merges with any existing PrimFunc-level attribute, then writes the
 * deduplicated result back to the PrimFunc attrs. This makes annotation
 * placement within the function body flexible for frontends.
 */
#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace ffi;
using namespace tvm::tirx;

class NonRestrictCollector : public StmtVisitor {
public:
  void Collect(const Stmt &stmt) { VisitStmt(stmt); }

  Array<Var> Result() const {
    Array<Var> out;
    out.reserve(collected_.size());
    for (const Var &v : collected_)
      out.push_back(v);
    return out;
  }

private:
  static std::string NormalizeName(const std::string &s) {
    if (s.size() >= 8 && s.rfind("_handle") == s.size() - 7) {
      return s.substr(0, s.size() - 7);
    }
    return s;
  }

  void MaybeInsert(const Var &v) {
    if (!v.defined())
      return;
    const VarNode *p = v.get();
    if (seen_ptr_.count(p))
      return;
    // Also dedup by normalized name to be robust w.r.t recreated Vars
    std::string norm = NormalizeName(v->name_hint);
    if (seen_name_.count(norm))
      return;
    seen_ptr_.insert(p);
    seen_name_.insert(std::move(norm));
    collected_.push_back(v);
  }

  void VisitStmt_(const SBlockNode *op) final {
    auto it = op->annotations.find(attr::kNonRestrictParams);
    if (it != op->annotations.end()) {
      if (const auto *arr = (*it).second.as<ArrayObj>()) {
        // Downcast directly to Array<Var> for convenience
        Array<Var> vars = tvm::Downcast<Array<Var>>((*it).second);
        for (const Var &v : vars) {
          MaybeInsert(v);
        }
      }
    }
    // Recurse into child statements
    StmtVisitor::VisitStmt_(op);
  }

  std::vector<Var> collected_;
  std::unordered_set<const VarNode *> seen_ptr_;
  std::unordered_set<std::string> seen_name_;
};

static PrimFunc HoistNonRestrictParams(PrimFunc f) {
  if (!f.defined())
    return f;

  NonRestrictCollector collector;
  collector.Collect(f->body);
  Array<Var> from_blocks = collector.Result();

  // Merge with any existing PrimFunc-level attribute if present
  if (auto opt_existing = f->GetAttr<Array<Var>>(attr::kNonRestrictParams)) {
    for (const Var &v : opt_existing.value()) {
      // Reuse the collector's dedup logic by temporarily constructing a new
      // collector Alternatively, do a small inline dedup mirroring MaybeInsert
      // Here we inline a simplified pointer-based dedup plus name-based
      // fallback
      bool exists = false;
      for (const Var &cur : from_blocks) {
        if (cur.get() == v.get() || cur->name_hint == v->name_hint) {
          exists = true;
          break;
        }
      }
      if (!exists)
        from_blocks.push_back(v);
    }
  }

  if (from_blocks.empty())
    return f;

  return WithAttr(std::move(f), attr::kNonRestrictParams,
                  std::move(from_blocks));
}

namespace transform {

tvm::transform::Pass HoistNonRestrictParams() {
  auto pass_func = [](PrimFunc f, const IRModule &,
                      const tvm::transform::PassContext &) {
    return tvm::tl::HoistNonRestrictParams(std::move(f));
  };
  return tvm::tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.HoistNonRestrictParams", {});
}

} // namespace transform

} // namespace tl
} // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.HoistNonRestrictParams",
                        tvm::tl::transform::HoistNonRestrictParams);
}
