/*!
 * \file constr_set_analysis.cc
 * \brief Python-visible probe for ConstrSet merge/populate semantics.
 *
 * `tl.analysis.ConstrSetsProve` builds two constraint sets from structured
 * entry specs, merges them, and reports whether the merged set proves the
 * goal. It exists so the bind-before-use normalization in
 * `ConstrSet::Merge` and the predicate expansion in `Constr::Populate`
 * (src/transform/common/constr_visitor.h) stay testable from Python
 * without staging a full pass pipeline around them.
 *
 * Entry specs, one array per entry (kind as StringImm):
 *   ["pred",   PrimExpr]                     -- branch/assert predicate
 *   ["assume", PrimExpr]                     -- trusted assume predicate
 *   ["bind",   Var, PrimExpr]                -- flat `v = expr` bind
 *   ["range",  Var, PrimExpr, PrimExpr]      -- `v in [min, min+extent)`
 */
#include "common/constr_visitor.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

Constr ParseConstr(const Array<ObjectRef> &spec) {
  ICHECK(!spec.empty()) << "ConstrSetsProve: empty entry spec";
  const auto *kind_imm = spec[0].as<StringImmNode>();
  ICHECK(kind_imm != nullptr)
      << "ConstrSetsProve: entry kind must be a StringImm";
  const std::string kind = kind_imm->value;
  if (kind == "pred" || kind == "assume") {
    ICHECK_EQ(spec.size(), 2U)
        << "ConstrSetsProve: [\"" << kind << "\", PrimExpr] expects 2 fields";
    return Constr(Downcast<PrimExpr>(spec[1]), /*is_assume=*/kind == "assume");
  }
  if (kind == "bind") {
    ICHECK_EQ(spec.size(), 3U)
        << "ConstrSetsProve: [\"bind\", Var, PrimExpr] expects 3 fields";
    return Constr(Downcast<Var>(spec[1]), Downcast<PrimExpr>(spec[2]));
  }
  if (kind == "range") {
    ICHECK_EQ(spec.size(), 4U)
        << "ConstrSetsProve: [\"range\", Var, min, extent] expects 4 fields";
    return Constr(Downcast<Var>(spec[1]),
                  Range::FromMinExtent(Downcast<PrimExpr>(spec[2]),
                                       Downcast<PrimExpr>(spec[3])));
  }
  LOG(FATAL) << "ConstrSetsProve: unknown entry kind '" << kind
             << "' (expected pred/assume/bind/range)";
  return Constr();
}

ConstrSet ParseSet(const Array<Array<ObjectRef>> &specs) {
  ConstrSet set;
  set.constrs_.reserve(specs.size());
  for (const Array<ObjectRef> &spec : specs) {
    set.constrs_.push_back(ParseConstr(spec));
  }
  return set;
}

bool ConstrSetsProve(Array<Array<ObjectRef>> first,
                     Array<Array<ObjectRef>> second, PrimExpr goal) {
  return ParseSet(first).Merge(ParseSet(second)).CanProve(goal);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.analysis.ConstrSetsProve", ConstrSetsProve);
}

} // namespace
} // namespace tl
} // namespace tvm
