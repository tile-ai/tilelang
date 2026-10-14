/*!
 * \file pto_index_analysis.h
 * \brief Low-level read-only index/address analysis shared by the PTO
 * Parallel layout planning (src/op/parallel.cc) and the Verify/Vectorize
 * consumers (src/ascend/transform/parallel_to_pto_utils.cc).
 *
 * These helpers live in src/op so both layers can depend on them without
 * an op -> ascend/transform reverse include. They are the single
 * implementation of the bind-alias resolution rules (cycle/depth failure,
 * expression-size budget, loaded-value identity) and the address-node
 * whitelist; planning and consuming code must not re-derive their own
 * variants.
 */

#ifndef TVM_TL_OP_PTO_INDEX_ANALYSIS_H_
#define TVM_TL_OP_PTO_INDEX_ANALYSIS_H_

#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <stdexcept>
#include <string>

namespace tvm {
namespace tl {
namespace pto {

using ffi::Map;
using ffi::Optional;
using tirx::Var;
using tirx::VarNode;

/*! Error type thrown by the shared analysis on unsupported input; both
 * passes translate it into their own diagnostics. */
class PtoAnalysisError : public std::runtime_error {
public:
  explicit PtoAnalysisError(const std::string &msg) : std::runtime_error(msg) {}
};

/*! Substitute `from` -> `to` inside an expression tree. */
PrimExpr SubstituteVar(const PrimExpr &expr, const VarNode *from, PrimExpr to);

/*! Resolve Bind aliases inside an index expression recursively with a
 * visited set (cycle-safe) and a depth guard that *rejects* instead of
 * silently stopping. Returns nullopt when a chain exceeds the
 * resolution limit; the caller must reject the access. */
Optional<PrimExpr> ResolveIndexAliases(const PrimExpr &index,
                                       const Map<Var, PrimExpr> &bind_env);

/*! Check whether an address/index expression only uses supported
 * nodes: coordinate arithmetic over loop/bound vars, constants,
 * casts and pure intrinsics are allowed; loads inside the index
 * (data-dependent addresses) and opaque or side-effecting calls
 * (call_extern etc.) are rejected because collapsing them to one
 * evaluation per chunk changes the program's evaluation behavior. */
bool IsSupportedAddressExpr(const PrimExpr &expr);

/*! Compute the padded fragment-input extent P = ceil(E / lanes) * lanes with
 * the shared overflow guard. The final execution space is int32-bounded
 * (the PTO lane loop uses int32 indices), so an extent above the int32
 * range is rejected before the padding addition could overflow int64.
 * Returns the pair (pad, padded); throws PtoAnalysisError with a
 * classified reason on rejection. The callers translate the error into
 * their own diagnostics. */
std::pair<int64_t, int64_t> CheckedPtoPadding(int64_t extent, int64_t lanes);

} // namespace pto
} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_PTO_INDEX_ANALYSIS_H_
