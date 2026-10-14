/*!
 * \file pto_index_analysis.cc
 * \brief Low-level read-only index/address analysis shared by the PTO
 * Parallel layout planning and the Verify/Vectorize consumers
 * (see pto_index_analysis.h).
 */

#include "pto_index_analysis.h"

#include <tvm/ir/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <functional>
#include <limits>
#include <sstream>
#include <utility>

namespace tvm {
namespace tl {
namespace pto {

using namespace tirx;
using ffi::GetRef;
using ffi::Map;
using ffi::ObjectRef;

namespace {

// Substitute `from` -> `to` inside an expression tree.
class VarSubstituter : public ExprMutator {
public:
  VarSubstituter(const VarNode *from, PrimExpr to)
      : from_(from), to_(std::move(to)) {}

private:
  PrimExpr VisitExpr_(const VarNode *op) final {
    if (op == from_) {
      return to_;
    }
    return ffi::GetRef<PrimExpr>(op);
  }
  const VarNode *from_;
  PrimExpr to_;
};

} // namespace

PrimExpr SubstituteVar(const PrimExpr &expr, const VarNode *from, PrimExpr to) {
  return VarSubstituter(from, std::move(to))(expr);
}

std::pair<int64_t, int64_t> CheckedPtoPadding(int64_t extent, int64_t lanes) {
  // The PTO lane loop indexes with int32, so any extent above the int32
  // range can never execute; rejecting it here (before the padding
  // addition) also makes an int64 overflow of extent + pad impossible:
  // with extent <= INT32_MAX and lanes <= 256, extent + pad stays far
  // below INT64_MAX.
  constexpr int64_t kMaxIndex = std::numeric_limits<int32_t>::max();
  if (extent > kMaxIndex) {
    std::ostringstream oss;
    oss << "PTO extent " << extent << " exceeds the int32 index space (max "
        << kMaxIndex << ")";
    throw PtoAnalysisError(oss.str());
  }
  const int64_t pad = extent % lanes == 0 ? 0 : lanes - extent % lanes;
  const int64_t padded = extent + pad;
  // Classified re-check: an extent within int32 still gets a distinct
  // diagnostic when the padded value itself crosses the boundary.
  if (padded > kMaxIndex) {
    std::ostringstream oss;
    oss << "PTO padded extent " << padded
        << " exceeds the int32 index space (max " << kMaxIndex << ")";
    throw PtoAnalysisError(oss.str());
  }
  return {pad, padded};
}

Optional<PrimExpr> ResolveIndexAliases(const PrimExpr &index,
                                       const Map<Var, PrimExpr> &bind_env) {
  if (bind_env.empty()) {
    return index;
  }
  // A Bind definition containing a BufferLoad is an SSA
  // *value*, not a coordinate alias — never expand it into a memory read.
  // The check runs at *every* substitution step (not only on the original
  // index) so the value-identity rule propagates through coordinate
  // alias chains (alias = saved; ... A[i + alias - IDX[i]]).
  auto def_is_pure_coordinate = [&](const PrimExpr &def_expr) {
    bool has_load = false;
    PostOrderVisit(def_expr, [&](const ObjectRef &node) {
      if (node.as<BufferLoadNode>()) {
        has_load = true;
      }
    });
    return !has_load;
  };
  constexpr int kMaxDepth = 64;
  // Expression-size budget. The depth cap alone is not enough —
  // every substitution round can double the expression (x1 = i + i;
  // x2 = x1 + x1; ...), so a 26-deep chain already exploded to 2^26 nodes
  // and died minutes later on a raw tvm int32-literal assertion instead of
  // a diagnostic. Count the nodes after each round and reject once the
  // budget is exceeded. Legal chains (plain var aliases, (i + x) - x
  // style cancellations) stay far below the budget and resolve quickly.
  constexpr int64_t kMaxNodes = 4096;
  auto count_nodes = [](const PrimExpr &expr) {
    int64_t nodes = 0;
    PostOrderVisit(expr, [&](const ObjectRef &) { ++nodes; });
    return nodes;
  };
  bool exhausted = true;
  std::function<bool(PrimExpr &, int)> resolve = [&](PrimExpr &expr,
                                                     int depth) -> bool {
    if (depth > kMaxDepth) {
      exhausted = false;
      return false;
    }
    const VarNode *found = nullptr;
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (found == nullptr) {
        if (auto *var_node = node.as<VarNode>()) {
          if (bind_env.count(GetRef<Var>(var_node))) {
            found = var_node;
          }
        }
      }
    });
    if (found == nullptr) {
      return true;
    }
    // Check the definition *before* substituting; a load-valued
    // definition reached through any alias depth is rejected.
    const PrimExpr &def = bind_env[GetRef<Var>(found)];
    if (!def_is_pure_coordinate(def)) {
      exhausted = false;
      return false;
    }
    expr = SubstituteVar(expr, found, def);
    if (count_nodes(expr) > kMaxNodes) {
      std::ostringstream oss;
      oss << "bind alias resolution exceeded the expression-size budget ("
          << kMaxNodes
          << " nodes); the alias chain expands too much "
             "(e.g. doubling definitions like x = y + y)";
      throw PtoAnalysisError(oss.str());
    }
    return resolve(expr, depth + 1);
  };
  PrimExpr result = index;
  try {
    if (!resolve(result, 0) || !exhausted) {
      return Optional<PrimExpr>();
    }
  } catch (const ffi::Error &err) {
    // A tvm-internal failure during substitution (e.g. an int32
    // literal overflowing while an expression is rebuilt) must surface as
    // an analysis diagnostic, not a raw Check failed crash.
    throw PtoAnalysisError(
        std::string("bind alias resolution hit a tvm-internal error: ") +
        err.what());
  }
  return result;
}

bool IsSupportedAddressExpr(const PrimExpr &expr) {
  // Addresses may contain coordinate arithmetic, constants, casts
  // and pure intrinsics — anything whose per-chunk evaluation is
  // equivalent. Opaque/extern calls (call_extern is kOpaque in TVM) have
  // no lowering rule; evaluating them once per chunk instead of per
  // logical point changes the program, so they are rejected. A load
  // inside an index makes the address data-dependent: the alias resolver
  // already rejects Bind-introduced loaded indices, and a direct load
  // must follow the same rule.
  class AddressCheck : public ExprVisitor {
  public:
    bool supported = true;
    void VisitExpr_(const CallNode *op) final {
      if (!supported) {
        return;
      }
      const auto *op_node = op->op.as<OpNode>();
      std::string name = op_node != nullptr ? op_node->name : "call";
      // Pure index intrinsics with defined semantics may appear; only
      // known-pure builtins are currently allowed.
      // call_extern/call_pure_extern/call_llvm_intrin etc. are rejected:
      // their effect is opaque to this pass.
      if (name == "tir.tvm_access_ptr" || name == "tl.access_ptr" ||
          name == "tir.address_of") {
        ExprVisitor::VisitExpr_(op);
        return;
      }
      supported = false;
    }
    void VisitExpr_(const BufferLoadNode *) final { supported = false; }
    void VisitExprDefault_(const ffi::Object *) final { /* fine */ }
  };
  AddressCheck checker;
  checker(expr);
  return checker.supported;
}

} // namespace pto
} // namespace tl
} // namespace tvm
