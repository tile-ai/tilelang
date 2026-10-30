/*!
 * \file pto_parallel_layout.cc
 * \brief PTO lane-layout construction for T.Parallel units.
 */

#include "pto_parallel_layout.h"

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pto_index_analysis.h"

namespace tvm {
namespace tl {
namespace pto {

using namespace tirx;
using ffi::Array;
using ffi::Map;
using ffi::Optional;

namespace {

// Collect the T.Parallel loop variables in the same traversal order used by
// the generic parallel operator: outermost first, descending through the whole
// body (including serial loops between parallel layers).
class PtoParallelLoopCollector : public tirx::StmtExprVisitor {
public:
  Array<IterVar> loop_vars;

  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel) {
      loop_vars.push_back(IterVar(Range(op->min, op->extent), op->loop_var,
                                  IterVarType::kDataPar));
    }
    tirx::StmtExprVisitor::VisitStmt_(op);
  }
};

// Pick the vectorized dimension of a two-layer Parallel nest from the buffer
// accesses. Both dimensions are tried as layout candidates under the same
// lane/chunk proof rules the consumers (Verify/Vectorize) use: a candidate is
// valid when every access is continuous (or, for loads, lane-uniform) over its
// padded lane/chunk domain, since a non-divisible selected extent is padded to
// a whole number of lanes. The inner dimension wins when both are valid; no
// valid candidate is a layout error naming the failing access.
class PtoAccessCollector : public tirx::StmtExprVisitor {
public:
  struct Access {
    Buffer buffer;
    Array<PrimExpr> indices;
    bool is_write;
    /*! The definition environment at this access's program point
     * (binds in scope, mirrors the consumers' lexical rules). */
    Map<Var, PrimExpr> bind_env;
  };
  std::vector<Access> accesses;

  void VisitStmt_(const BufferStoreNode *op) final {
    accesses.push_back({op->buffer, op->indices, /*is_write=*/true, bind_env_});
    tirx::StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    accesses.push_back(
        {op->buffer, op->indices, /*is_write=*/false, bind_env_});
    tirx::StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BindNode *op) final {
    // Flat sequence semantics (same as the consumers): the value is
    // visited *before* the var is registered, so a self reference is not
    // in scope; every bind — including load-valued ones — is registered
    // so ResolveIndexAliases can apply its loaded-value identity rule
    // along alias chains.
    tirx::StmtExprVisitor::VisitStmt_(op);
    bind_env_.Set(op->var, op->value);
  }
  void VisitStmt_(const AttrStmtNode *op) final {
    // Nested statement scope (mirror of the consumers' rule):
    // binds defined inside an AttrStmt body must not leak to later
    // siblings, while outer binds stay visible inside.
    Map<Var, PrimExpr> saved = bind_env_;
    tirx::StmtExprVisitor::VisitStmt_(op);
    bind_env_ = std::move(saved);
  }

private:
  Map<Var, PrimExpr> bind_env_;
};

// 2D layout candidate. Both dimensions are tried with their canonical mapping
// — selected coordinate `q * L + lane`, the other coordinate left as-is — over
// the bound lane/chunk/kept ranges. Alias resolution, the loaded-value identity
// rule, the address-node whitelist and the resource limits are the shared
// read-only helpers in pto_index_analysis.h — the same implementation
// AnalyzeBufferAccess uses, so planning and consumption cannot diverge.
// An access passes when its flattened offset is continuous
// (`offset == start + lane`); a load may instead be lane-uniform
// (`offset == start`), while a store must be continuous (lane-uniform
// stores are rejected downstream). A candidate is valid once every access
// proof passes over the padded chunk domain (`q in [0, Q)`, `lane in [0, L)`);
// when both candidates are valid, the selection rules below
// (non-degenerate-dimension preference, then the inner preference) apply.
// Layout-inference code reports construct-time failures as
// LoopLayoutInjectiveException (the type LayoutInference catches to discard
// an attempt), while the shared padding helper speaks PtoAnalysisError;
// translate here so all three call sites share one helper.
std::pair<int64_t, int64_t> CheckedPaddingForInference(int64_t extent,
                                                       int64_t lanes) {
  try {
    return CheckedPtoPadding(extent, lanes);
  } catch (const PtoAnalysisError &err) {
    throw LoopLayoutInjectiveException(err.what());
  }
}

int PtoRequiredVectorDim(const For &root, const Var &iv, const Var &jv,
                         int64_t M, int64_t N, int64_t lanes,
                         std::string *why) {
  PtoAccessCollector collector;
  collector(root->body);

  auto candidate_ok = [&](bool select_inner) -> bool {
    const int64_t sel_extent = select_inner ? N : M;
    const Var &sel_var = select_inner ? jv : iv;
    const Var &kept_var = select_inner ? iv : jv;
    const int64_t kept_extent = select_inner ? M : N;
    // A non-divisible extent is padded to P = ceil(E/L)*L when the
    // fragment is built; the candidate proof below works over the padded
    // domain (Q = P/L chunks), which includes the tail chunk's full lane
    // range, and the analyzer proves continuity over that whole padded
    // domain. What remains a caller guarantee is only the *readability*
    // of the padded range past E on the selected axis (tail loads read
    // it); the candidate neither proves nor rejects that.
    // Shared checked padding (rejects extents above the int32 index space
    // before the addition; the same helper the fragment construction uses).
    const auto [pad, padded_extent] =
        CheckedPaddingForInference(sel_extent, lanes);
    (void)pad;
    const int64_t chunk_count = padded_extent / lanes;
    DataType dt = sel_var.dtype();
    Var lane("probe_lane", dt);
    Var q("probe_q", dt);
    PrimExpr sel_expr = q * IntImm(dt, lanes) + lane;
    for (const auto &access : collector.accesses) {
      // Resolve Bind aliases per index with the shared resolver (cycle /
      // depth / loaded-value failure, 4096-node budget, error conversion)
      // against this access's program-point environment, then apply the
      // shared address-node whitelist — the same order AnalyzeBufferAccess
      // uses.
      Array<PrimExpr> resolved;
      for (const auto &idx : access.indices) {
        Optional<PrimExpr> opt;
        try {
          opt = ResolveIndexAliases(idx, access.bind_env);
        } catch (const PtoAnalysisError &err) {
          *why = err.what();
          return false;
        }
        if (!opt.has_value()) {
          std::ostringstream oss;
          oss << "buffer `" << access.buffer->name
              << "`: bind alias chain in index is too deep, cyclic, or "
                 "depends on a loaded value (data-dependent addresses are "
                 "not yet supported)";
          *why = oss.str();
          return false;
        }
        resolved.push_back(opt.value());
      }
      for (const auto &idx : resolved) {
        if (!IsSupportedAddressExpr(idx)) {
          std::ostringstream oss;
          oss << "buffer `" << access.buffer->name
              << "`: address expression contains a loaded value or an "
                 "unsupported call (loaded values and opaque/extern calls "
                 "cannot appear in addresses; they would be evaluated once "
                 "per chunk instead of per element)";
          *why = oss.str();
          return false;
        }
      }
      // indices under the candidate mapping, then the flattened element
      // offset (same construction as the consumers).
      Array<PrimExpr> indices_after;
      for (const auto &idx : resolved) {
        indices_after.push_back(SubstituteVar(idx, sel_var.get(), sel_expr));
      }
      Array<PrimExpr> raw = access.buffer->ElemOffset(indices_after);
      if (raw.size() != 1) {
        *why = "multi-axis element offsets are not supported";
        return false;
      }
      PrimExpr off = raw[0];
      Array<PrimExpr> start_indices;
      for (const auto &idx : indices_after) {
        start_indices.push_back(SubstituteVar(idx, lane.get(), make_zero(dt)));
      }
      PrimExpr start = access.buffer->ElemOffset(start_indices)[0];
      arith::Analyzer probe;
      probe.Bind(lane, Range::FromMinExtent(make_zero(dt), IntImm(dt, lanes)));
      // In-row chunk range only: q is the per-row chunk index in [0, Q),
      // never the global position (K*Q for a kept extent K). The kept
      // coordinate is separately bounded below, so the proof domain is
      // exactly (q, lane, kept) — the same domain the final consumers'
      // proof uses.
      probe.Bind(q,
                 Range::FromMinExtent(make_zero(dt), IntImm(dt, chunk_count)));
      // The kept coordinate is bounded by its loop extent; the
      // final consumers' proof binds the same fact.
      probe.Bind(kept_var,
                 Range::FromMinExtent(make_zero(dt), IntImm(dt, kept_extent)));
      const bool continuous = probe.CanProveEqual(off, start + lane);
      const bool uniform = probe.CanProveEqual(
          off, probe.Simplify(SubstituteVar(off, lane.get(), make_zero(dt))));
      if (!continuous && !(uniform && !access.is_write)) {
        std::ostringstream oss;
        oss << "buffer `" << access.buffer->name << "` ("
            << (access.is_write ? "store" : "load")
            << ") is neither continuous nor lane-uniform under the "
            << (select_inner ? "inner (j)" : "outer (i)") << " mapping";
        *why = oss.str();
        return false;
      }
    }
    return true;
  };

  std::string inner_why;
  std::string outer_why;
  const bool inner_ok = candidate_ok(/*select_inner=*/true);
  if (!inner_ok) {
    inner_why = *why;
  }
  const bool outer_ok = candidate_ok(/*select_inner=*/false);
  if (!outer_ok) {
    outer_why = *why;
  }
  if (inner_ok && outer_ok && N == 1 && M > 1) {
    // Both dims classify and only the outer is non-degenerate: selecting the
    // inner (extent-1) dim would emit M chunks with
    // one valid lane each, while the outer dim covers the same logical
    // points in ceil(M/L) chunks carrying M valid lanes in total. The
    // candidate proofs above have already checked the actual buffer
    // indices and strides, so the preference only breaks the tie between
    // two already-legal candidates and never overrides an access
    // constraint. The symmetric M == 1 && N > 1 falls through to the
    // inner preference below.
    return 0;
  }
  if (inner_ok) {
    // Inner preference; a degenerate dimension is
    // only an interpretation when its accesses classify under the padded
    // domain, which candidate_ok already enforces.
    return 1;
  }
  if (outer_ok) {
    return 0;
  }
  std::ostringstream oss;
  oss << "no vectorizable dimension: inner (j) -> " << inner_why
      << "; outer (i) -> " << outer_why;
  *why = oss.str();
  return -1;
}

} // namespace

// PTO SIMD lane layout: logical points of the parallel loop are distributed
// over [0, L) lanes. The flattened logical index is
//   1D:      t = i
//   2D:      t = i * N + j (j vectorized) or t = j * M + i (i vectorized)
// with lane = t % L and per-lane chunk position p = t // L; forward_index
// holds p. vectorize_size is fixed at 1 — the per-thread SIMT vector-width
// selection does not apply.
//
// Scope note: this function only *builds* the lane fragment. The full input
// contract (access patterns, dtype, expression whitelist) belongs to
// VerifyParallelToPTO; the checks below are only the minimal conditions
// needed to construct a valid layout, reported as layout-inference errors.
Fragment InferPtoParallelLayout(const For &root, const Range &lane_bounds) {
  PtoParallelLoopCollector loop_collector;
  loop_collector(root);
  const Array<IterVar> &loop_vars = loop_collector.loop_vars;

  const int64_t *lanes_ptr = as_const_int(lane_bounds->extent);
  ICHECK(lanes_ptr != nullptr && *lanes_ptr > 0)
      << "[PTO layout inference] lane count must be a positive compile-time "
         "constant, got "
      << lane_bounds->extent;
  int64_t lanes = *lanes_ptr;

  // Loop extents must be compile-time constants. Keep the
  // int64 value: narrowing here would silently truncate large static
  // extents.
  std::vector<int64_t> extents;
  for (const auto &iv : loop_vars) {
    const int64_t *ext = as_const_int(iv->dom->extent);
    ICHECK(ext != nullptr)
        << "[PTO layout inference] requires compile-time constant loop "
           "extents, got "
        << iv->dom->extent << " for loop " << iv->var->name_hint;
    ICHECK(is_zero(iv->dom->min))
        << "[PTO layout inference] requires loop min = 0, got " << iv->dom->min
        << " for loop " << iv->var->name_hint;
    extents.push_back(*ext);
  }
  ICHECK(loop_vars.size() == 1u || loop_vars.size() == 2u)
      << "[PTO layout inference] supports a 1D T.Parallel or a two-layer "
         "Parallel nest; got "
      << loop_vars.size() << " layers";

  // 1D mapping: t = i, lane = i % L,
  // forward_index = [i // L]. A non-divisible extent E is padded to
  // P = ceil(E/L)*L in the *fragment input space only* — the IterVar extent
  // below is the padded one, while buffer addresses keep using the real
  // extents/strides (the tail's readable range is a caller precondition).
  DataType idx_dtype = loop_vars[0]->var.dtype();
  PrimExpr flattened;
  // The forward vars handed to the Fragment constructor: the selected
  // dimension's IterVar is padded up to P so the thread extent L times the
  // chunk extent P/L covers the padded input space bijectively; the kept
  // dimension keeps its original extent.
  Array<IterVar> frag_vars;
  if (loop_vars.size() == 1u) {
    const int64_t E = extents[0];
    const auto [pad, P] = CheckedPaddingForInference(E, lanes);
    frag_vars = loop_vars;
    if (pad != 0) {
      // Reuse the loop var; only the input-space extent is padded.
      frag_vars.Set(0, IterVar(Range::FromMinExtent(make_zero(idx_dtype),
                                                    IntImm(idx_dtype, P)),
                               loop_vars[0]->var, IterVarType::kDataPar));
    }
    flattened = InputPlaceholder(0);
  } else {
    // 2D mapping: the vectorized dimension is
    // chosen from the buffer accesses (indices + strides), then the
    // logical space is flattened as t = i * P_N + j (j vectorized) or
    // t = j * P_M + i (i vectorized), where P_* is the *padded* selected
    // extent — the flattened global position p lives in [0, K*Q*L) over the
    // padded input space. Both extents stay compile-time constants.
    const int64_t M = extents[0];
    const int64_t N = extents[1];
    const Var &i_var = loop_vars[0]->var;
    const Var &j_var = loop_vars[1]->var;
    std::string candidate_why;
    int required =
        PtoRequiredVectorDim(root, i_var, j_var, M, N, lanes, &candidate_why);
    if (required < 0) {
      // Both candidates failed (an access that is neither continuous nor a
      // lane-uniform load under the padded domain). Report the concrete
      // access and reason — the candidate check already applies the
      // inner-dimension preference, so -1 never means "no constraint".
      std::ostringstream oss;
      oss << "[PTO layout inference] the 2D unit has no vectorizable "
             "dimension (M="
          << M << ", N=" << N << ", L=" << lanes << "): " << candidate_why;
      throw LoopLayoutInjectiveException(oss.str());
    }
    const bool select_inner = required == 1;
    const int64_t sel_extent = select_inner ? N : M;
    const int64_t kept_extent = select_inner ? M : N;
    // Shared checked padding for the selected dimension.
    const auto [pad, P_sel] = CheckedPaddingForInference(sel_extent, lanes);
    // The flattened product K*P_sel indexes the int32 loop-var space; both
    // operands of every t = kept * P_sel + sel evaluation must stay
    // representable.
    if (P_sel > std::numeric_limits<int32_t>::max() ||
        kept_extent > std::numeric_limits<int32_t>::max() / P_sel) {
      std::ostringstream oss;
      oss << "[PTO layout inference] padded flattened extent " << kept_extent
          << "*" << P_sel
          << " overflows the int32 index "
             "space for lanes "
          << lanes;
      throw LoopLayoutInjectiveException(oss.str());
    }
    frag_vars = loop_vars;
    if (pad != 0) {
      frag_vars.Set(select_inner ? 1 : 0,
                    IterVar(Range::FromMinExtent(make_zero(idx_dtype),
                                                 IntImm(idx_dtype, P_sel)),
                            select_inner ? j_var : i_var,
                            IterVarType::kDataPar));
    }
    flattened = select_inner ? InputPlaceholder(0) * IntImm(idx_dtype, P_sel) +
                                   InputPlaceholder(1)
                             : InputPlaceholder(1) * IntImm(idx_dtype, P_sel) +
                                   InputPlaceholder(0);
  }
  PrimExpr lane = FloorMod(flattened, IntImm(idx_dtype, lanes));
  PrimExpr chunk = FloorDiv(flattened, IntImm(idx_dtype, lanes));
  Var rep_var;
  IterVar rep_iter =
      IterVar(Range::FromMinExtent(make_zero(idx_dtype), IntImm(idx_dtype, 1)),
              rep_var, IterVarType::kDataPar);
  auto fragment = Fragment(frag_vars, {chunk}, lane, rep_iter)
                      ->BindThreadRange(lane_bounds);

  // The fragment must not only be forward-injective: Vectorize consumes it
  // through Fragment::InverseWithLevel, so a layout that cannot be inverted
  // with the expected level has to fail here, closest to its construction,
  // not deep inside the vectorize pass.
  try {
    auto inverse_info = fragment->InverseWithLevel(
        /*require_padding_guard=*/false);
    ICHECK(inverse_info.second == arith::IterMapLevel::Bijective)
        << "[PTO layout inference] PTO lane layout must invert at "
           "Bijective level, got "
        << static_cast<int>(inverse_info.second) << " for "
        << fragment->DebugOutput();
  } catch (const ffi::Error &err) {
    std::ostringstream oss;
    oss << "[PTO layout inference] lane layout cannot be inverted by "
           "InverseWithLevel: "
        << fragment->DebugOutput() << "\n  error: " << err.what()
        << "\n  loop AST: " << root;
    throw LoopLayoutInjectiveException(oss.str());
  }

  return fragment;
}

} // namespace pto
} // namespace tl
} // namespace tvm
