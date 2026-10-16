/*!
 * \file parallel_to_pto_utils.cc
 * \brief Shared read-only analysis for VerifyParallelToPTO and
 * VectorizeParallelToPTO.
 */

#include "parallel_to_pto_utils.h"

#include <tvm/arith/pattern.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include "../../layout/utils.h"

#include <functional>
#include <sstream>

namespace tvm {
namespace tl {
namespace pto {

using namespace tirx;
using ffi::Array;
using ffi::GetRef;
using ffi::Map;

std::string AccessPatternToString(AccessPattern pattern) {
  switch (pattern) {
  case AccessPattern::kContinuous:
    return "continuous";
  case AccessPattern::kLaneUniform:
    return "lane-uniform";
  default:
    return "unknown";
  }
}

// SubstituteVar, ResolveIndexAliases and IsSupportedAddressExpr moved to
// src/op/pto_index_analysis.cc: the layout-planning code in
// src/op and the consumers here share that single implementation.

PtoInverseMapping InvertPtoLaneLayout(const For &loop) {
  auto anno = loop->annotations.Get(attr::kPtoParallelLoopLayout);
  if (!anno.has_value()) {
    throw PtoAnalysisError(
        "pto_parallel_loop_layout is missing on the outermost T.Parallel");
  }
  auto fragment_any = anno.value().try_cast<Fragment>();
  if (!fragment_any.has_value()) {
    throw PtoAnalysisError(
        "pto_parallel_loop_layout must be a tl.Fragment annotation");
  }
  Fragment fragment = fragment_any.value();

  // A 2D unit is the outermost For of a two-layer
  // adjacent Parallel nest. The annotation — and the fragment — lives on
  // the outer For and describes the full (i, j) input space; the input
  // coordinate order stays (i, j) whichever dimension is vectorized.
  const ForNode *inner_node = nullptr;
  if (const auto *body_for = loop->body.as<ForNode>()) {
    if (body_for->kind == ForKind::kParallel) {
      inner_node = body_for;
    }
  }
  const bool is_2d = inner_node != nullptr;
  const int expected_dim = is_2d ? 2 : 1;
  if (fragment->InputDim() != expected_dim) {
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout input_size must be " << expected_dim
        << "D for a "
        << (is_2d ? "two-layer nested T.Parallel" : "1D T.Parallel")
        << ", got input dim " << fragment->InputDim();
    throw PtoAnalysisError(oss.str());
  }
  if (fragment->GetForwardIndex().size() != 1) {
    // Extra per-lane position dimensions are a legal-but-unsupported
    // layout shape; reject with a layout diagnostic instead of an internal
    // assertion inside Forward.
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout must have exactly one internal "
           "position dimension, got "
        << fragment->GetForwardIndex().size();
    throw PtoAnalysisError(oss.str());
  }
  if (!is_one(fragment->ReplicateExtent())) {
    throw PtoAnalysisError(
        "pto_parallel_loop_layout requires replicate_size == 1");
  }

  const int64_t *lanes_ptr = as_const_int(fragment->ThreadExtent());
  if (lanes_ptr == nullptr || *lanes_ptr <= 0) {
    throw PtoAnalysisError(
        "pto_parallel_loop_layout thread extent must be a positive constant");
  }
  int64_t lanes = *lanes_ptr;
  if (lanes != 64 && lanes != 128 && lanes != 256) {
    std::ostringstream oss;
    oss << "T.SimdVF lanes must be one of {64, 128, 256}, got " << lanes;
    throw PtoAnalysisError(oss.str());
  }

  // Shape contract: the fragment input shape must match the loop extents
  // (1D: the single extent; 2D: (outer, inner)).
  const int64_t *extent_ptr = as_const_int(loop->extent);
  if (extent_ptr == nullptr) {
    // The helper must not assert on inputs the callers did not
    // pre-validate; a symbolic extent is a layout diagnostic (same rule
    // as the inner extent below).
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout loop extent must be a compile-time "
           "constant, got "
        << loop->extent;
    throw PtoAnalysisError(oss.str());
  }
  int64_t outer_extent = *extent_ptr;
  int64_t inner_extent = 1;
  if (is_2d) {
    const int64_t *inner_extent_ptr = as_const_int(inner_node->extent);
    if (inner_extent_ptr == nullptr) {
      // The helper must not assert on inputs the callers did not
      // pre-validate; a symbolic inner extent is a layout diagnostic.
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout inner loop extent must be a "
             "compile-time constant, got "
          << inner_node->extent;
      throw PtoAnalysisError(oss.str());
    }
    inner_extent = *inner_extent_ptr;
  }
  // Shape contract: the fragment input space is padded on the
  // vectorized dimension. 1D: the single input extent must be the padded
  // extent P = ceil(E / L) * L. 2D: the kept dimension keeps its original
  // loop extent and the selected dimension is padded to P — but which
  // dimension is selected is only known jointly with the canonical mapping
  // after the inversion, so the 2D shape check runs post-inversion. Here
  // the 1D case is checked directly; an unpadded [E] (E % L != 0), an
  // over-padded extent, and (for divisible E) any extent != E are layout
  // diagnostics.
  Array<PrimExpr> input_shape = fragment->InputShape();
  if (!is_2d) {
    if (input_shape.size() != 1u) {
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout input shape " << input_shape
          << " must be 1D for a 1D T.Parallel";
      throw PtoAnalysisError(oss.str());
    }
    const auto [pad_1d, padded_1d] = CheckedPtoPadding(outer_extent, lanes);
    (void)pad_1d;
    const int64_t *frag_extent = as_const_int(input_shape[0]);
    if (frag_extent == nullptr || *frag_extent != padded_1d) {
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout input shape " << input_shape
          << " does not match the padded extent " << padded_1d
          << " (loop extent " << outer_extent << ", lanes " << lanes << "); "
          << "the vectorized dimension must be padded to ceil(E/L)*L";
      throw PtoAnalysisError(oss.str());
    }
  } else {
    if (input_shape.size() != 2u) {
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout input shape " << input_shape
          << " must be 2D for a two-layer nested T.Parallel";
      throw PtoAnalysisError(oss.str());
    }
    // Structural presence check only; the per-dimension contract (kept =
    // original, selected = padded) is a joint check with the canonical
    // mapping below.
    for (const auto &dim : input_shape) {
      if (as_const_int(dim) == nullptr) {
        std::ostringstream oss;
        oss << "pto_parallel_loop_layout input shape " << input_shape
            << " must be compile-time constants, got " << input_shape;
        throw PtoAnalysisError(oss.str());
      }
    }
  }

  // Thread-range contract: [0, L).
  Range thread_range = fragment->ThreadRange();
  if (thread_range.defined()) {
    const int64_t *tr_min = as_const_int(thread_range->min);
    const int64_t *tr_ext = as_const_int(thread_range->extent);
    if (tr_min == nullptr || *tr_min != 0 || tr_ext == nullptr ||
        *tr_ext != lanes) {
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout thread_range must be [0, " << lanes
          << "), got [" << thread_range->min << ", " << thread_range->extent
          << ")";
      throw PtoAnalysisError(oss.str());
    }
  }

  // The fragment is built over int32 placeholder vars (the only
  // dtype LayoutInference produces); a loop var of any other dtype (e.g.
  // int64) crashes the tvm substitution machinery with a raw assertion
  // during inversion. Reject it here with a layout diagnostic instead.
  if (loop->loop_var.dtype() != DataType::Int(32)) {
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout requires an int32 loop variable (the "
           "dtype LayoutInference produces), got "
        << loop->loop_var.dtype();
    throw PtoAnalysisError(oss.str());
  }
  if (is_2d && inner_node->loop_var.dtype() != DataType::Int(32)) {
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout requires an int32 inner loop variable "
           "(the dtype LayoutInference produces), got "
        << inner_node->loop_var.dtype();
    throw PtoAnalysisError(oss.str());
  }

  // Invert through the real inverse layout (single source of truth for the
  // mapping). InverseWithLevel appends the lane (thread) input after the
  // per-lane position input. It raises when the fragment is not a valid
  // bijection — e.g. a hand-built layout whose per-thread position extent
  // does not cover the input space: Layout::InverseWithLevel throws
  // NormalizeIterException when the iteration map has errors, and the
  // iteration-map machinery itself reports tvm errors as ffi::Error. Both are
  // known input failures here, so they become layout diagnostics. Only this
  // call is wrapped, so a genuine implementation error anywhere else still
  // surfaces as one instead of being reported as bad input.
  auto invert = [&]() {
    try {
      return fragment->InverseWithLevel(/*require_padding_guard=*/false);
    } catch (const NormalizeIterException &err) {
      throw PtoAnalysisError(std::string("pto_parallel_loop_layout cannot be "
                                         "inverted: ") +
                             err.what());
    } catch (const ffi::Error &err) {
      throw PtoAnalysisError(std::string("pto_parallel_loop_layout cannot be "
                                         "inverted: ") +
                             err.what());
    }
  };
  auto inverse_info = invert();
  if (inverse_info.second != arith::IterMapLevel::Bijective) {
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout is not bijectively invertible (level "
        << static_cast<int>(inverse_info.second)
        << "): " << fragment->DebugOutput();
    throw PtoAnalysisError(oss.str());
  }
  Layout inverse = inverse_info.first;

  // Forward the (chunk, lane) coordinates through the inverse layout to
  // obtain the logical coordinate expression. The inverse layout has two
  // outputs (the logical coordinate and the replicate coordinate); with
  // replicate_size == 1 the second output is a constant zero and the
  // design-doc convention is to take the logical part only.
  Var chunk_var("pto_chunk", loop->loop_var.dtype());
  Var lane_var("pto_lane", loop->loop_var.dtype());
  Array<PrimExpr> logical = inverse->Forward({chunk_var, lane_var});
  // The output count follows the input dimensionality (1D: (i, replicate);
  // 2D: (i, j, replicate)); each branch below checks its own arity.

  // Only the canonical mappings are accepted. Any other
  // invertible form (e.g. a reversed lane assignment) is rejected instead
  // of being silently replaced with our own formula.
  //   1D:      i = chunk * L + lane
  //   2D (j):  j = (chunk * L + lane) % N, i = (chunk * L + lane) / N
  //   2D (i):  i = (chunk * L + lane) % M, j = (chunk * L + lane) / M
  PtoInverseMapping mapping;
  mapping.lane_var = lane_var;
  mapping.chunk_var = chunk_var;
  mapping.lanes = lanes;

  if (!is_2d) {
    ICHECK_EQ(logical.size(), 2u);
    // The fragment input space is padded to P, so the chunk
    // position range is [0, P/L) — the padded chunk count Q.
    const int64_t padded = CheckedPtoPadding(outer_extent, lanes).second;
    arith::Analyzer probe;
    probe.Bind(chunk_var,
               Range::FromMinExtent(make_zero(chunk_var->dtype),
                                    IntImm(chunk_var->dtype, padded / lanes)));
    probe.Bind(lane_var, Range::FromMinExtent(make_zero(lane_var->dtype),
                                              IntImm(lane_var->dtype, lanes)));
    PrimExpr canonical = chunk_var * IntImm(chunk_var->dtype, lanes) + lane_var;
    if (!probe.CanProveEqual(logical[0], canonical)) {
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout does not use the canonical 1D "
             "mapping i = chunk * L + lane (got "
          << logical[0] << " from " << fragment->DebugOutput() << ")";
      throw PtoAnalysisError(oss.str());
    }
    mapping.logical_extent = outer_extent;
    mapping.padded_extent = padded;
    mapping.chunk_count = padded / lanes;
    mapping.tail_lanes = outer_extent % lanes;
    mapping.index_expr = logical[0];
    return mapping;
  }

  ICHECK_EQ(logical.size(), 3u);
  const int64_t M = outer_extent;
  const int64_t N = inner_extent;
  // The fragment input space is padded on the vectorized dimension, so
  // the flattened position t = kept * P_sel + sel lives in [0, K * P_sel)
  // and the global chunk count is K * ceil(E / L) (each row padded
  // independently — not ceil(K * E / L), which is not equivalent). This
  // probe domain is for the canonical mapping check only; the
  // address-proof domain that AnalyzeBufferAccess uses binds the per-row
  // q range instead.
  // Each interpretation gets its own probe with the exact chunk domain
  // [0, kept * P_sel / L): a domain bound larger than the fragment's real
  // chunk count could make an identity that holds on the real domain
  // unprovable, and a smaller one could admit a mapping that only looks
  // canonical on part of the space. The lane range [0, L) is shared.
  const auto bind_probe = [&](arith::Analyzer *probe, int64_t chunk_bound) {
    probe->Bind(chunk_var,
                Range::FromMinExtent(make_zero(chunk_var->dtype),
                                     IntImm(chunk_var->dtype, chunk_bound)));
    probe->Bind(lane_var, Range::FromMinExtent(make_zero(lane_var->dtype),
                                               IntImm(lane_var->dtype, lanes)));
  };
  PrimExpr t = chunk_var * IntImm(chunk_var->dtype, lanes) + lane_var;
  // The vectorized coordinate is t % P_sel, the kept one t / P_sel — both
  // the modulus and the divisor use the *padded* selected extent. The
  // interpretation is accepted only when it also agrees with the fragment
  // input shape: the kept dimension keeps its original loop extent and
  // the selected dimension is padded up to P_sel (unpadded / over-padded
  // / wrongly-modified kept dims each fail here). A degenerate dimension
  // (extent 1) makes both canonical forms provable; the shape agreement
  // breaks the tie exactly when only one dim was padded, and the inner
  // preference remains for fully divisible shapes.
  const auto matches = [&](bool select_inner) -> bool {
    const int64_t sel_extent = select_inner ? N : M;
    const auto [pad, padded] = CheckedPtoPadding(sel_extent, lanes);
    const int64_t kept_extent = select_inner ? M : N;
    const int64_t *frag_kept = as_const_int(input_shape[select_inner ? 0 : 1]);
    const int64_t *frag_sel = as_const_int(input_shape[select_inner ? 1 : 0]);
    if (frag_kept == nullptr || frag_sel == nullptr ||
        *frag_kept != kept_extent || *frag_sel != padded) {
      return false;
    }
    (void)pad;
    arith::Analyzer probe;
    bind_probe(&probe, kept_extent * padded / lanes);
    const PrimExpr &sel = select_inner ? logical[1] : logical[0];
    const PrimExpr &kept = select_inner ? logical[0] : logical[1];
    PrimExpr sel_imm = IntImm(t.dtype(), padded);
    return probe.CanProveEqual(sel, FloorMod(t, sel_imm)) &&
           probe.CanProveEqual(kept, FloorDiv(t, sel_imm));
  };
  const bool inner_ok = matches(/*select_inner=*/true);
  const bool outer_ok = matches(/*select_inner=*/false);
  if (!inner_ok && !outer_ok) {
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout does not use a canonical 2D mapping "
           "(j = t % P_N, i = t / P_N  or  i = t % P_M, j = t / P_M with "
           "t = chunk * L + lane, P_* = padded selected extent; the kept "
           "dim keeps its loop extent in the fragment shape); got (i, j) = ("
        << logical[0] << ", " << logical[1] << "), shape " << input_shape
        << ", loop extents (" << M << ", " << N << ") from "
        << fragment->DebugOutput();
    throw PtoAnalysisError(oss.str());
  }
  // Both dims matching is degenerate (e.g. M == N == 1); prefer the inner
  // dimension, consistent with the layout-selection rule.
  const bool select_inner = inner_ok;
  const int64_t sel_extent = select_inner ? N : M;
  const auto [pad_sel, P_sel] = CheckedPtoPadding(sel_extent, lanes);
  (void)pad_sel;
  mapping.is_2d = true;
  mapping.outer_var = loop->loop_var;
  mapping.inner_var = inner_node->loop_var;
  mapping.outer_extent = M;
  mapping.inner_extent = N;
  mapping.select_inner = select_inner;
  mapping.logical_extent = sel_extent;
  mapping.padded_extent = P_sel;
  mapping.chunk_count = P_sel / lanes;
  mapping.tail_lanes = sel_extent % lanes;
  mapping.index_expr = t;
  return mapping;
}

MemoryAccess AnalyzeBufferAccess(const Buffer &buffer,
                                 const Array<PrimExpr> &indices, bool is_write,
                                 const Var &loop_var,
                                 const PtoInverseMapping &mapping,
                                 arith::Analyzer *analyzer,
                                 const Map<Var, PrimExpr> &bind_env) {
  MemoryAccess access;
  access.buffer = buffer;
  access.elem_dtype = buffer->dtype;
  access.is_write = is_write;

  // Resolve Bind aliases in the indices first: a bound index var
  // must follow its definition, not be treated as lane-independent. The
  // resolver rejects chains that exceed the limit instead of silently
  // stopping.
  //
  // Only *pure coordinate* aliases (loop vars, arithmetic over
  // loop vars/consts) may be expanded. A Bind whose definition contains a
  // BufferLoad holds an SSA value: re-expanding it into a memory read can
  // cross a same-point store and wrongly cancel (the loaded value and the
  // current buffer state differ). Addresses that depend on loaded values
  // have no supported lowering and are rejected here.
  Array<PrimExpr> resolved;
  for (const auto &idx : indices) {
    auto opt = ResolveIndexAliases(idx, bind_env);
    if (!opt.has_value()) {
      std::ostringstream oss;
      oss << "buffer `" << buffer->name
          << "`: bind alias chain in index is too deep, cyclic, or "
             "depends on a loaded value (data-dependent addresses are "
             "not yet supported)";
      throw PtoAnalysisError(oss.str());
    }
    resolved.push_back(opt.value());
  }
  // Address expressions must only contain supported
  // nodes; opaque calls in addresses have no lowering rule and would be
  // silently collapsed to one evaluation per chunk.
  for (const auto &idx : resolved) {
    if (!IsSupportedAddressExpr(idx)) {
      std::ostringstream oss;
      oss << "buffer `" << buffer->name
          << "`: address expression contains a loaded value or an "
             "unsupported call (loaded values and opaque/extern calls "
             "cannot appear in addresses; they would be evaluated once per "
             "chunk instead of per element)";
      throw PtoAnalysisError(oss.str());
    }
  }

  // indices_after_inverse: substitute loop_var with the inverse mapping's
  // own expression (mapping.index_expr — the single source of truth; do
  // not rebuild chunk * L + lane here).
  Array<PrimExpr> indices_after_inverse;
  for (const auto &idx : resolved) {
    indices_after_inverse.push_back(analyzer->Simplify(
        SubstituteVar(idx, loop_var.get(), mapping.index_expr)));
  }
  access.indices_after_inverse = indices_after_inverse;

  // start_indices: bind lane = 0.
  Array<PrimExpr> start_indices;
  for (const auto &idx : indices_after_inverse) {
    start_indices.push_back(analyzer->Simplify(
        SubstituteVar(idx, mapping.lane_var.get(), make_zero(idx.dtype()))));
  }
  access.start_indices = start_indices;

  // data_offset via ElemOffset (handles strides / elem_offset / versions).
  Array<PrimExpr> elem_offsets = buffer->ElemOffset(indices_after_inverse);
  if (elem_offsets.size() != 1) {
    std::ostringstream oss;
    oss << "buffer `" << buffer->name
        << "`: ElemOffset must return exactly one physical axis, got "
        << elem_offsets.size();
    throw PtoAnalysisError(oss.str());
  }
  access.data_offset = analyzer->Simplify(elem_offsets[0]);

  Array<PrimExpr> start_elem_offsets = buffer->ElemOffset(start_indices);
  ICHECK_EQ(start_elem_offsets.size(), 1u);
  access.start_offset = analyzer->Simplify(start_elem_offsets[0]);

  // Classify with a *complete* proof: continuous means
  // data_offset == start_offset + lane over the whole [0, L) lane range,
  // not a two-point difference. Bind the ranges and require the analyzer
  // to prove the identity; anything it cannot prove is rejected.
  // The lane-uniform check shares this range-bound proof
  // analyzer — with an unbound analyzer a chunk-uniform address such as
  // (i // L) * L cannot be proven equal to its lane=0 substitution (the
  // proof needs 0 <= lane < L), so legal inputs were mis-rejected.
  arith::Analyzer proof;
  proof.Bind(
      mapping.lane_var,
      Range::FromMinExtent(make_zero(mapping.lane_var->dtype),
                           IntImm(mapping.lane_var->dtype, mapping.lanes)));
  // The chunk proof domain is the per-row padded chunk range
  // [0, Q) with Q = P / L on the vectorized dimension — for a 2D unit the
  // *in-row* chunk count, never the global position K*Q (the global
  // flattened position only appears in the Fragment forward/inverse
  // layout checks, not in address proofs).
  proof.Bind(mapping.chunk_var,
             Range::FromMinExtent(
                 make_zero(mapping.chunk_var->dtype),
                 IntImm(mapping.chunk_var->dtype, mapping.chunk_count)));
  // In a 2D unit the kept coordinate is bounded by its own loop
  // extent — a fact the layout candidate already uses when it accepts
  // the unit. Binding it here keeps the final proof domain
  // identical to the candidate's, so range-dependent but legal addresses
  // (e.g. a term multiplied by i // 4 with i in [0, 4)) prove continuous.
  // This adds no facts beyond the loop definitions; it proves nothing
  // about buffer bounds or cross-point dependencies.
  if (mapping.is_2d && mapping.serial_var() != nullptr) {
    const VarNode *serial = mapping.serial_var();
    const int64_t kept_extent =
        mapping.select_inner ? mapping.outer_extent : mapping.inner_extent;
    proof.Bind(GetRef<Var>(serial),
               Range::FromMinExtent(make_zero(serial->dtype),
                                    IntImm(serial->dtype, kept_extent)));
  }
  {
    PrimExpr identity =
        access.data_offset - (access.start_offset + mapping.lane_var);
    if (proof.CanProveEqual(identity, make_zero(identity.dtype()))) {
      access.pattern = AccessPattern::kContinuous;
      return access;
    }
  }

  // Lane-uniform: the offset does not depend on lane at the current
  // program point (version/outer-coordinate indices may stay). Proven
  // over the full lane range with the same bound analyzer.
  {
    PrimExpr at_zero =
        proof.Simplify(SubstituteVar(access.data_offset, mapping.lane_var.get(),
                                     make_zero(access.data_offset.dtype())));
    if (proof.CanProveEqual(access.data_offset, at_zero)) {
      access.pattern = AccessPattern::kLaneUniform;
      return access;
    }
  }

  std::ostringstream oss;
  oss << "buffer `" << buffer->name
      << "`: cannot prove the address is continuous "
         "(data_offset == start + lane over the full lane range) or "
         "lane-uniform; only these two patterns are supported";
  throw PtoAnalysisError(oss.str());
}

bool IsSupportedElementDType(DataType dtype) {
  // No packed sub-byte element types (FP4 etc.), no
  // buffer-element dtypes that already carry vector lanes, and no FP64
  // (the current PTO codegen has no FP64 scalar-type mapping).
  if (dtype.bits() < 8 || dtype.lanes() != 1) {
    // FP4 (e2m1) and every other sub-byte element type land here: an FP4
    // element is 4 bits wide whether packed or not, so a separate FP4 check
    // would be unreachable.
    return false;
  }
  // The PTO codegen only maps 8/16/32-bit scalars; unusual widths (int24) and
  // FP64 (no scalar-type mapping) are rejected for buffers and arithmetic
  // alike by the width rule below, not just inside Cast planning.
  if (!dtype.is_bool() && dtype.bits() != 8 && dtype.bits() != 16 &&
      dtype.bits() != 32) {
    return false;
  }
  return dtype.is_float() || dtype.is_bfloat16() || dtype.is_int() ||
         dtype.is_uint() || dtype.is_bool() || IsSupportedFloat8(dtype);
}

bool IsSupportedFloat8(DataType dtype) {
  // Only these two take part in the FP8 conversion matrix. float8_e4m3 is a
  // distinct TIR dtype from float8_e4m3fn and is not admitted by name
  // similarity; E8M0/HiF8/FNUZ are outside the supported conversion range,
  // while FP4 is a packed sub-byte type with no unified element addressing.
  return dtype.is_float8_e4m3fn() || dtype.is_float8_e5m2();
}

namespace {

// Whitelist check for *value* expressions. A
// BufferLoad's indices belong to address analysis, not value lowering
// the version-dim FloorMod/FloorDiv inside a load index must not
// be rejected here. The value whitelist does not descend into indices.
class WhitelistVisitor : public ExprVisitor {
public:
  std::optional<std::string> unsupported;

  void VisitExpr_(const BufferLoadNode *) final {}
  void VisitExpr_(const VarNode *) final {}
  void VisitExpr_(const IntImmNode *) final {}
  void VisitExpr_(const FloatImmNode *) final {}
  void VisitExpr_(const StringImmNode *) final {
    // String constants are metadata markers only. The base
    // ExprVisitor declares StringImm and accepts it as a leaf, so the
    // VisitExprDefault_ fallback never fired and `Evaluate(StringImm)`
    // slipped through Verify into Vectorize's internal error. Reject it
    // explicitly here, with the same wording as the lane-use analysis.
    if (!unsupported.has_value()) {
      unsupported = "string constants are metadata-only; they cannot "
                    "appear in computation values";
    }
  }
  void VisitExpr_(const AddNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const SubNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const MulNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const DivNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const CastNode *op) final {
    // Only the converted value is a computation value. The base visitor also
    // walks PrimExpr entries of the cast annotations, which would put a
    // StringImm rounding hint in front of the value whitelist and reject an
    // otherwise legal cast ("string constants are metadata-only"). Attribute
    // legality is decided by PlanValueCast, from the raw annotations.
    VisitExpr(op->value);
  }
  // The base ExprVisitor overrides VisitExpr_ for every standard node and
  // silently descends, so non-whitelist nodes never reach
  // VisitExprDefault_. Reject them explicitly.
  void MarkUnsupported(const char *kind) {
    if (!unsupported.has_value()) {
      unsupported = std::string("unsupported expression node `") + kind +
                    "` (only buffer loads, constants, "
                    "scalars, loop indices, Add, Sub, Mul, Div, "
                    "fabs/exp/log/sqrt and Cast)";
    }
  }
  void VisitExpr_(const ModNode *) final { MarkUnsupported("Mod"); }
  void VisitExpr_(const FloorDivNode *) final { MarkUnsupported("FloorDiv"); }
  void VisitExpr_(const FloorModNode *) final { MarkUnsupported("FloorMod"); }
  void VisitExpr_(const MinNode *) final { MarkUnsupported("Min"); }
  void VisitExpr_(const MaxNode *) final { MarkUnsupported("Max"); }
  void VisitExpr_(const EQNode *) final { MarkUnsupported("EQ"); }
  void VisitExpr_(const NENode *) final { MarkUnsupported("NE"); }
  void VisitExpr_(const LTNode *) final { MarkUnsupported("LT"); }
  void VisitExpr_(const LENode *) final { MarkUnsupported("LE"); }
  void VisitExpr_(const GTNode *) final { MarkUnsupported("GT"); }
  void VisitExpr_(const GENode *) final { MarkUnsupported("GE"); }
  void VisitExpr_(const AndNode *) final { MarkUnsupported("And"); }
  void VisitExpr_(const OrNode *) final { MarkUnsupported("Or"); }
  void VisitExpr_(const NotNode *) final { MarkUnsupported("Not"); }
  void VisitExpr_(const SelectNode *) final {
    if (!unsupported.has_value()) {
      unsupported =
          "Select is not vectorizable yet (integer T.abs lowers to Select; "
          "it arrives with the PR-2 Select/cmp support)";
    }
  }
  void VisitExpr_(const RampNode *) final { MarkUnsupported("Ramp"); }
  void VisitExpr_(const BroadcastNode *) final { MarkUnsupported("Broadcast"); }
  void VisitExpr_(const ShuffleNode *) final { MarkUnsupported("Shuffle"); }
  void VisitExpr_(const ReduceNode *) final { MarkUnsupported("Reduce"); }
  void VisitExpr_(const CallNode *op) final {
    if (auto call_op = op->op.as<Op>()) {
      const std::string &name = call_op.value()->name;
      // Hardware-backed unary math. fabs/exp/log/sqrt map to the VMI
      // vabs/vexp/vln/vsqrt instructions; the dtype matrix mirrors the
      // ISA documentation (see IsSupportedVectorUnaryOp).
      if (name == "tirx.fabs" || name == "tirx.exp" || name == "tirx.log" ||
          name == "tirx.sqrt") {
        if (op->args.size() != 1U) {
          if (!unsupported.has_value()) {
            unsupported = name + " expects exactly one operand";
          }
          return;
        }
        const DataType arg = op->args[0].dtype();
        if (!IsSupportedVectorUnaryOp(name, arg)) {
          if (!unsupported.has_value()) {
            std::ostringstream oss;
            oss << arg;
            unsupported = name + " vectorization does not support " +
                          oss.str() +
                          " operands (per the VMI operation matrix)";
          }
          return;
        }
        ExprVisitor::VisitExpr_(op);
        return;
      }
    }
    {
      std::string name = "call";
      if (const auto *op_node = op->op.as<OpNode>()) {
        name = op_node->name;
      }
      if (!unsupported.has_value()) {
        unsupported = "unsupported Call `" + name +
                      "` (only tirx.fabs/exp/log/sqrt are vectorizable)";
      }
    }
  }

  void VisitExprDefault_(const ffi::Object *op) final {
    std::string key = op->GetTypeKey();
    MarkUnsupported(key.c_str());
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Shared SIMD_VF region scan: one implementation for both passes.
// ---------------------------------------------------------------------------
namespace {

void CollectAllParallelDeep(const Stmt &stmt,
                            std::vector<const ForNode *> &out) {
  if (const auto *for_node = stmt.as<ForNode>()) {
    if (for_node->kind == ForKind::kParallel) {
      out.push_back(for_node);
    }
    CollectAllParallelDeep(for_node->body, out);
    return;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const auto &s : seq->seq) {
      CollectAllParallelDeep(s, out);
    }
    return;
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    CollectAllParallelDeep(attr->body, out);
    return;
  }
  if (const auto *block = stmt.as<SBlockNode>()) {
    CollectAllParallelDeep(block->body, out);
    return;
  }
  if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
    CollectAllParallelDeep(realize->block->body, out);
    return;
  }
  if (const auto *ite = stmt.as<IfThenElseNode>()) {
    CollectAllParallelDeep(ite->then_case, out);
    if (ite->else_case.defined()) {
      CollectAllParallelDeep(ite->else_case.value(), out);
    }
    return;
  }
  if (const auto *while_node = stmt.as<WhileNode>()) {
    CollectAllParallelDeep(while_node->body, out);
    return;
  }
}

void DiscoverUnitsShared(const Stmt &stmt, PtoRegionScan *scan) {
  if (const auto *for_node = stmt.as<ForNode>()) {
    if (for_node->kind == ForKind::kParallel) {
      // A two-layer adjacent Parallel nest is one
      // 2D unit; the layout annotation lives on the outermost For. A
      // third Parallel layer (or a Parallel deeper inside) stays
      // unsupported.
      const auto *adjacent_inner = for_node->body.as<ForNode>();
      const bool inner_is_parallel = adjacent_inner != nullptr &&
                                     adjacent_inner->kind == ForKind::kParallel;
      std::vector<const ForNode *> deeper;
      CollectAllParallelDeep(
          inner_is_parallel ? adjacent_inner->body : for_node->body, deeper);
      if (!deeper.empty()) {
        scan->has_unsupported_wrapper = true;
        // Two distinct illegal shapes: a genuine third parallel layer
        // inside an adjacent two-layer nest, or an outer Parallel whose
        // body mixes other statements with an inner Parallel (the nest is
        // not adjacent and cannot form a pure 2D unit).
        scan->issue =
            inner_is_parallel
                ? "three or more nested parallel layers are not yet "
                  "supported"
                : "a T.Parallel whose body mixes other statements with an "
                  "inner T.Parallel (a non-adjacent nest) is not yet "
                  "supported";
      }
      scan->units.push_back(GetRef<For>(for_node));
      return;
    }
    if (for_node->kind == ForKind::kVectorized ||
        for_node->kind == ForKind::kThreadBinding) {
      // Only sequential wrappers may surround a unit: a re-vectorizing loop
      // contradicts the vectorization this conversion already performs, and
      // a thread binding contradicts the no-thread-binding contract of the
      // converting region. A region without T.Parallel under them stays a
      // legitimate hand-written VMI region and is skipped entirely.
      std::vector<const ForNode *> under;
      CollectAllParallelDeep(stmt, under);
      if (!under.empty()) {
        scan->has_unsupported_wrapper = true;
        if (for_node->kind == ForKind::kThreadBinding) {
          std::ostringstream oss;
          oss << "a thread-bound loop (ForKind::kThreadBinding";
          if (const auto *iv = for_node->thread_binding.as<IterVarNode>()) {
            oss << ", thread_binding `" << iv->var->name_hint << "`";
          }
          oss << ") around T.Parallel is not supported: "
                 "thread bindings are not allowed inside a converting "
                 "SIMD_VF block";
          scan->issue = oss.str();
        } else {
          scan->issue =
              "a vectorized loop (ForKind::kVectorized) around T.Parallel is "
              "not yet supported";
        }
      }
      return;
    }
    // A serial loop (with or without an unroll hint) is a sequential
    // wrapper: its iterations run in order and each iteration keeps the
    // unit's semantics, so it stays accepted.
    DiscoverUnitsShared(for_node->body, scan);
    return;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const auto &s : seq->seq) {
      DiscoverUnitsShared(s, scan);
    }
    return;
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    DiscoverUnitsShared(attr->body, scan);
    return;
  }
  if (const auto *block = stmt.as<SBlockNode>()) {
    DiscoverUnitsShared(block->body, scan);
    return;
  }
  if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
    DiscoverUnitsShared(realize->block->body, scan);
    return;
  }
  if (stmt.as<IfThenElseNode>() != nullptr || stmt.as<WhileNode>() != nullptr) {
    // Control-flow wrappers: a Parallel under them is rejected; without
    // one the region may still be a legitimate hand-written VMI region.
    std::vector<const ForNode *> under;
    CollectAllParallelDeep(stmt, under);
    if (!under.empty()) {
      scan->has_unsupported_wrapper = true;
      scan->issue = stmt.as<IfThenElseNode>() != nullptr
                        ? "T.Parallel under a conditional branch is not yet "
                          "supported"
                        : "T.Parallel under a while loop is not yet "
                          "supported";
    }
    return;
  }
}

} // namespace

PtoRegionScan ScanPtoRegion(const Stmt &region) {
  PtoRegionScan scan;
  DiscoverUnitsShared(region, &scan);
  std::vector<const ForNode *> all;
  CollectAllParallelDeep(region, all);
  scan.has_any_parallel = !all.empty();
  if (scan.has_any_parallel && scan.units.empty()) {
    // A Parallel exists but only under an unsupported wrapper.
    scan.has_unsupported_wrapper = true;
    if (scan.issue.empty()) {
      scan.issue = "T.Parallel exists only under an unsupported wrapper";
    }
  }
  return scan;
}

std::string LaneUseToString(LaneUse use) {
  switch (use) {
  case LaneUse::kUniform:
    return "uniform";
  case LaneUse::kVarying:
    return "varying";
  default:
    return "unknown";
  }
}

namespace {

// Shared per-node classification used by both the value mode and the
// metadata mode: value mode whitelists Add/Mul/Cast; metadata
// mode additionally propagates the pure scalar operators and accepts
// StringImm.
LaneUse AnalyzeLaneUseImpl(const PrimExpr &expr, LaneUseContext *ctx,
                           std::string *reason, bool metadata_mode) {
  if (expr.as<IntImmNode>() || expr.as<FloatImmNode>()) {
    return LaneUse::kUniform;
  }
  if (expr.as<StringImmNode>() != nullptr) {
    // String constants are legal metadata (markers); they never
    // appear as computation values (the value whitelist rejects them).
    if (metadata_mode) {
      return LaneUse::kUniform;
    }
    if (reason != nullptr) {
      // Every Unknown carries a concrete reason.
      *reason = "string constants are metadata-only; they cannot appear "
                "in computation values";
    }
    return LaneUse::kUnknown;
  }
  if (const auto *var = expr.as<VarNode>()) {
    if (var == ctx->loop_var.get()) {
      return LaneUse::kVarying;
    }
    if (ctx->serial_var.defined() && var == ctx->serial_var.get()) {
      // The coordinate of the non-vectorized
      // dimension is a uniform scalar at the program point.
      return LaneUse::kUniform;
    }
    if (ctx->bind_env.count(GetRef<Var>(var))) {
      // Use the recorded classification of the bind (SSA identity).
      if (ctx->classified.count(GetRef<Var>(var))) {
        Integer v = ctx->classified[GetRef<Var>(var)];
        return v->value == 0 ? LaneUse::kUniform : LaneUse::kVarying;
      }
      // On-demand analysis with cycle guard + budget.
      if (ctx->visiting.count(var) != 0) {
        if (reason != nullptr) {
          *reason = "bind self/forward reference detected in lane-use "
                    "analysis";
        }
        return LaneUse::kUnknown;
      }
      if (ctx->budget <= 0) {
        if (reason != nullptr) {
          *reason = "lane-use analysis recursion budget exhausted";
        }
        return LaneUse::kUnknown;
      }
      ctx->visiting.insert(var);
      ctx->budget -= 1;
      const PrimExpr &def = ctx->bind_env[GetRef<Var>(var)];
      LaneUse use = AnalyzeLaneUseImpl(def, ctx, reason, metadata_mode);
      ctx->visiting.erase(var);
      if (use == LaneUse::kUnknown) {
        return use;
      }
      // Record only when the definition is fully analyzed; a cycle hit
      // above leaves it unrecorded so later references re-report.
      ctx->classified.Set(GetRef<Var>(var),
                          Integer(use == LaneUse::kUniform ? 0 : 1));
      return use;
    }
    // Only *known* external scalar vars are Uniform; anything
    // else is undefined -> Unknown.
    if (ctx->external_defs.count(var) != 0) {
      // Being in scope is not enough. An external var consumed as
      // a computation value (or as numeric metadata) must be scalar; a
      // vector-typed var has no scalar input contract for this pass and
      // must not be waved through as Uniform. Buffer data pointers are
      // handle-typed (lanes == 1) and keep flowing through the address
      // and Buffer paths.
      if (var->dtype.lanes() != 1) {
        if (reason != nullptr) {
          std::ostringstream oss;
          oss << "variable `" << var->name_hint << "` has vector dtype "
              << var->dtype
              << "; only scalar inputs are supported as computation values";
          *reason = oss.str();
        }
        return LaneUse::kUnknown;
      }
      return LaneUse::kUniform;
    }
    if (reason != nullptr) {
      std::ostringstream oss;
      oss << "variable `" << var->name_hint
          << "` is not defined in this unit or any outer scope";
      // The diagnostic was built but never written back, leaving
      // metadata errors with an empty reason.
      *reason = oss.str();
    }
    return LaneUse::kUnknown;
  }
  if (const auto *load = expr.as<BufferLoadNode>()) {
    try {
      auto access = AnalyzeBufferAccess(
          load->buffer, load->indices, /*is_write=*/false, ctx->loop_var,
          ctx->mapping, ctx->analyzer, ctx->bind_env);
      if (access.pattern == AccessPattern::kLaneUniform) {
        return LaneUse::kUniform;
      }
      return LaneUse::kVarying;
    } catch (const PtoAnalysisError &err) {
      if (reason != nullptr) {
        *reason = err.what();
      }
      return LaneUse::kUnknown;
    }
  }
  // Hardware-backed unary math shares the whitelist's contract (float32
  // tirx.fabs/exp/log/sqrt): recurse into the single operand.
  if (const auto *call = expr.as<CallNode>()) {
    if (auto call_op = call->op.as<Op>()) {
      const std::string &name = call_op.value()->name;
      if (name == "tirx.fabs" || name == "tirx.exp" || name == "tirx.log" ||
          name == "tirx.sqrt") {
        return AnalyzeLaneUseImpl(call->args[0], ctx, reason, metadata_mode);
      }
    }
  }
  // Binary/relational/boolean operators: both value and metadata modes
  // recurse into operands; the *type* checks live in the callers.
  LaneUse lhs, rhs;
  const PrimExpr *a = nullptr;
  const PrimExpr *b = nullptr;
  bool known_op = false;
  if (const auto *add = expr.as<AddNode>()) {
    a = &add->a;
    b = &add->b;
    known_op = true;
  } else if (const auto *sub = expr.as<SubNode>()) {
    a = &sub->a;
    b = &sub->b;
    known_op = true;
  } else if (const auto *mul = expr.as<MulNode>()) {
    a = &mul->a;
    b = &mul->b;
    known_op = true;
  } else if (const auto *div = expr.as<DivNode>()) {
    a = &div->a;
    b = &div->b;
    known_op = true;
  } else if (const auto *mod = expr.as<ModNode>()) {
    a = &mod->a;
    b = &mod->b;
    known_op = metadata_mode;
  } else if (const auto *fdiv = expr.as<FloorDivNode>()) {
    a = &fdiv->a;
    b = &fdiv->b;
    known_op = metadata_mode;
  } else if (const auto *fmod = expr.as<FloorModNode>()) {
    a = &fmod->a;
    b = &fmod->b;
    known_op = metadata_mode;
  } else if (const auto *minn = expr.as<MinNode>()) {
    a = &minn->a;
    b = &minn->b;
    known_op = metadata_mode;
  } else if (const auto *maxx = expr.as<MaxNode>()) {
    a = &maxx->a;
    b = &maxx->b;
    known_op = metadata_mode;
  } else if (const auto *lt = expr.as<LTNode>()) {
    a = &lt->a;
    b = &lt->b;
    known_op = metadata_mode;
  } else if (const auto *len = expr.as<LENode>()) {
    a = &len->a;
    b = &len->b;
    known_op = metadata_mode;
  } else if (const auto *gt = expr.as<GTNode>()) {
    a = &gt->a;
    b = &gt->b;
    known_op = metadata_mode;
  } else if (const auto *ge = expr.as<GENode>()) {
    a = &ge->a;
    b = &ge->b;
    known_op = metadata_mode;
  } else if (const auto *eq = expr.as<EQNode>()) {
    a = &eq->a;
    b = &eq->b;
    known_op = metadata_mode;
  } else if (const auto *ne = expr.as<NENode>()) {
    a = &ne->a;
    b = &ne->b;
    known_op = metadata_mode;
  } else if (const auto *andn = expr.as<AndNode>()) {
    a = &andn->a;
    b = &andn->b;
    known_op = metadata_mode;
  } else if (const auto *orn = expr.as<OrNode>()) {
    a = &orn->a;
    b = &orn->b;
    known_op = metadata_mode;
  }
  if (known_op) {
    lhs = AnalyzeLaneUseImpl(*a, ctx, reason, metadata_mode);
    rhs = AnalyzeLaneUseImpl(*b, ctx, reason, metadata_mode);
    if (lhs == LaneUse::kUnknown || rhs == LaneUse::kUnknown) {
      if (reason != nullptr && reason->empty()) {
        *reason = "unknown operand in binary expression";
      }
      return LaneUse::kUnknown;
    }
    return (lhs == LaneUse::kUniform && rhs == LaneUse::kUniform)
               ? LaneUse::kUniform
               : LaneUse::kVarying;
  }
  if (!metadata_mode) {
    if (const auto *notn = expr.as<NotNode>()) {
      // Not is a value-mode whitelist reject too; keep symmetric.
    }
  } else {
    if (const auto *notn = expr.as<NotNode>()) {
      return AnalyzeLaneUseImpl(notn->a, ctx, reason, metadata_mode);
    }
  }
  if (const auto *cast = expr.as<CastNode>()) {
    return AnalyzeLaneUseImpl(cast->value, ctx, reason, metadata_mode);
  }
  if (reason != nullptr) {
    std::ostringstream oss;
    oss << "unsupported " << (metadata_mode ? "metadata" : "value") << " node `"
        << expr->GetTypeKey() << "` in lane-use analysis";
    *reason = oss.str();
  }
  return LaneUse::kUnknown;
}

} // namespace

LaneUse AnalyzeLaneUse(const PrimExpr &expr, LaneUseContext *ctx,
                       std::string *reason) {
  return AnalyzeLaneUseImpl(expr, ctx, reason, /*metadata_mode=*/false);
}

LaneUse AnalyzeLaneUseMeta(const PrimExpr &expr, LaneUseContext *ctx,
                           std::string *reason) {
  return AnalyzeLaneUseImpl(expr, ctx, reason, /*metadata_mode=*/true);
}

bool IsSupportedVectorUnaryOp(const std::string &tirx_op, DataType dtype) {
  // Per the VMI ISA documentation: vabs admits f16/bf16/f32 (the vabs
  // instruction also takes si8/si16/si32, but the frontend abs() lowers
  // integers to Select — PR-2 territory — and returns unsigned inputs
  // unchanged, so integer dtypes never reach this gate); vexp/vln/vsqrt
  // admit f16/f32 only.
  const int bits = dtype.bits();
  if (tirx_op == "tirx.fabs") {
    return (dtype.is_float() || dtype.is_bfloat16()) &&
           (bits == 16 || bits == 32);
  }
  if (tirx_op == "tirx.exp" || tirx_op == "tirx.log" ||
      tirx_op == "tirx.sqrt") {
    return dtype.is_float() && (bits == 16 || bits == 32);
  }
  return false;
}

bool ExprUsesVar(const PrimExpr &expr, const VarNode *var) {
  bool uses = false;
  PostOrderVisit(expr, [&](const ObjectRef &n) {
    if (n.as<VarNode>() == var) {
      uses = true;
    }
  });
  return uses;
}

bool ContainsRuntimeValue(const PrimExpr &expr) {
  // True when the expression reads any runtime value: a Var (external
  // scalar, serial coordinate, scalar buffer base) or a BufferLoad.
  // A pure IntImm/FloatImm tree is a compile-time constant.
  bool has_runtime = false;
  PostOrderVisit(expr, [&](const ObjectRef &n) {
    if (n.as<VarNode>() != nullptr || n.as<BufferLoadNode>() != nullptr) {
      has_runtime = true;
    }
  });
  return has_runtime;
}

bool IsFloatFamily(DataType dtype) {
  return dtype.is_float() || dtype.is_bfloat16();
}

bool IsSupportedVectorBinaryOp(const std::string &vmi_op, DataType dtype) {
  // Operation-level matrix (per the VMI ISA documentation):
  //   vmul requires i16/i32/f16/bf16/f32; vadd/vsub additionally allow i8;
  //   vdiv admits f16/f32 only (no bf16, no integers).
  int bits = dtype.bits();
  const bool is_int_like = dtype.is_int() || dtype.is_uint();
  const bool is_fp_like =
      (dtype.is_float() || dtype.is_bfloat16()) && (bits == 16 || bits == 32);
  if (vmi_op == "vmul") {
    return (is_int_like && (bits == 16 || bits == 32)) || is_fp_like;
  }
  if (vmi_op == "vadd" || vmi_op == "vsub") {
    return (is_int_like && (bits == 8 || bits == 16 || bits == 32)) ||
           is_fp_like;
  }
  if (vmi_op == "vdiv") {
    return dtype.is_float() && (bits == 16 || bits == 32);
  }
  return false;
}

namespace {

/*! Cast attributes read from a Cast node's backend annotations. The
 * frontend writes them in two shapes — TIR Imm objects (the `T.cast` helper
 * lowers `round`/`sat` to StringImm/IntImm) and plain Python values (a raw
 * `annotations={...}` map) — so both spellings are accepted here. */
struct CastAttributes {
  /*! The `round` annotation was present (even when its value is empty). */
  bool has_round_key = false;
  /*! A non-empty rounding mode was requested. An empty string is the
   * frontend's way of saying "use the backend default", which is not the same
   * as an unusable mode. */
  bool has_round_mode = false;
  std::string round; // raw spelling, e.g. "rn", "R", "A"
  bool has_sat = false;
  bool sat = true;
  bool has_rbits = false;
};

/*! Read the attribute hints of a Cast. Returns false with *why set when an
 * annotation uses a shape this version cannot read; an unknown key is
 * itself a rejection, because silently ignoring an attribute would accept
 * a conversion whose semantics were never checked. */
bool ReadCastAttributes(const ffi::Map<ffi::String, ffi::Any> &annotations,
                        CastAttributes *attrs, std::string *why) {
  for (const auto &[key_ref, value] : annotations) {
    const std::string key = key_ref;
    if (key == "round") {
      attrs->has_round_key = true;
      if (const auto *imm = value.as<StringImmNode>()) {
        attrs->round = imm->value;
      } else if (auto str = value.as<ffi::String>()) {
        attrs->round = std::string(*str);
      } else {
        *why = "annotation `round` must be a string";
        return false;
      }
      attrs->has_round_mode = !attrs->round.empty();
    } else if (key == "sat") {
      attrs->has_sat = true;
      if (const auto *imm = value.as<IntImmNode>()) {
        // Only 0/1 are accepted, in the TIR Imm form as well.
        if (imm->value != 0 && imm->value != 1) {
          *why = "annotation `sat` must be a boolean (got " +
                 std::to_string(imm->value) + ")";
          return false;
        }
        attrs->sat = imm->value != 0;
      } else if (auto flag = value.as<bool>()) {
        attrs->sat = *flag;
      } else if (auto flag = value.as<int64_t>()) {
        // Only 0/1 are accepted: any other integer is a malformed flag, not a
        // saturating request.
        if (*flag != 0 && *flag != 1) {
          *why = "annotation `sat` must be a boolean (got " +
                 std::to_string(*flag) + ")";
          return false;
        }
        attrs->sat = *flag != 0;
      } else {
        *why = "annotation `sat` must be a boolean";
        return false;
      }
    } else if (key == "rbits") {
      attrs->has_rbits = true;
    } else {
      *why = "unknown cast annotation `" + key + "`";
      return false;
    }
  }
  return true;
}

/*! Normalize a rounding spelling to the canonical PTO token. The frontend
 * aliases are PTX-style; the canonical tokens are the ones the FP8
 * conversion policy reasons about. Returns false with *why set for
 * aliases with no PTO meaning (stochastic rounding) and for tokens no
 * conversion in this version accepts. */
bool NormalizeCastRounding(const std::string &raw, std::string *token,
                           std::string *why) {
  if (raw == "rn") {
    *token = "R";
    return true;
  }
  if (raw == "rz") {
    *token = "Z";
    return true;
  }
  if (raw == "rp") {
    *token = "C";
    return true;
  }
  if (raw == "rm") {
    *token = "F";
    return true;
  }
  if (raw == "rs") {
    *why = "stochastic rounding (round=\"rs\") is not supported";
    return false;
  }
  static const std::set<std::string> kCanonical = {"R", "A", "H",
                                                   "Z", "C", "F"};
  if (kCanonical.count(raw) != 0) {
    *token = raw;
    return true;
  }
  *why = "rounding mode \"" + raw + "\" has no PTO vcvt meaning";
  return false;
}

/*! Rounding tokens the FP8 narrowing policy accepts. The conversion
 * instructions the FP8 target exposes are FP32 <-> FP8; the matrix is the
 * intersection of the PTODSL vcvt interface with that support, so the source
 * is always float32 and only the rounding modes below are expressible. */
const std::set<std::string> &Fp8AllowedRoundings() {
  static const std::set<std::string> kAllowed = {"R", "A", "H", "Z"};
  return kAllowed;
}

} // namespace

Optional<CastPlan> PlanValueCast(const CastNode *cast, std::string *reason) {
  DataType from = cast->value.dtype();
  DataType to = cast->dtype;
  auto reject = [&](const std::string &why) {
    if (reason != nullptr) {
      *reason = why;
    }
    return Optional<CastPlan>();
  };
  std::ostringstream pair_os;
  pair_os << "`" << from << " -> " << to << "`";
  const std::string pair = pair_os.str();
  // Dtype legality is a precondition for every action plan,
  // including identity — unsupported types (int24/FP64/...) must not slip
  // through the same-type fast path.
  if (!IsSupportedElementDType(from) || !IsSupportedElementDType(to)) {
    return reject(pair + " is outside the supported element dtype set");
  }

  // Attributes are parsed for every Cast, not just the FP8 ones: the
  // expression whitelist deliberately does not look inside a cast's
  // annotations (attributes are not computation values), so an annotation the
  // selected conversion cannot honour has to be rejected here — for every
  // direction — or it would be dropped silently.
  CastAttributes attrs;
  std::string why;
  if (!ReadCastAttributes(cast->annotations, &attrs, &why)) {
    return reject(pair + ": " + why);
  }
  if (attrs.has_rbits) {
    return reject(pair + ": the rbits operand (stochastic rounding) is not "
                         "supported");
  }
  std::string round_token;
  if (attrs.has_round_mode &&
      !NormalizeCastRounding(attrs.round, &round_token, &why)) {
    return reject(pair + ": " + why);
  }

  if (from == to) {
    // Identity, FP8 or not: no conversion is emitted, so an explicit sat=True
    // is the default and is normalized away; any other explicit attribute
    // would claim a conversion semantic that does not exist here.
    if (attrs.has_sat && !attrs.sat) {
      return reject(pair + ": an identity cast does not accept sat=False");
    }
    if (attrs.has_round_mode) {
      return reject(pair +
                    ": an identity cast does not accept a rounding mode");
    }
    CastPlan identity;
    identity.identity = true;
    return identity;
  }

  const bool from_fp8 = IsSupportedFloat8(from);
  const bool to_fp8 = IsSupportedFloat8(to);
  if (from_fp8 || to_fp8) {
    if (from_fp8 && to_fp8) {
      return reject(pair + ": conversions between the two FP8 types are not "
                           "supported (they are not a reinterpretation)");
    }
    if (from_fp8 && to.is_float() && to.bits() == 32) {
      // Widening: exact, no rounding or saturation involved. An explicit
      // hint here cannot be honoured, so it is rejected rather than dropped.
      // The direction contract rejects an attribute that is *carried* here,
      // even an empty one: widening has no rounding to configure.
      if (attrs.has_round_key) {
        return reject(pair +
                      ": this direction does not accept a rounding mode");
      }
      if (attrs.has_sat) {
        return reject(pair +
                      ": this direction does not accept a saturation mode");
      }
      return CastPlan{};
    }
    if (to_fp8) {
      // The FP8 conversion instructions the target exposes are FP32 <-> FP8;
      // there is no half -> FP8 conversion, so a float16/bfloat16 source is
      // rejected here. This pass never adds a conversion node of its own: a
      // user who wants the float32 route writes it explicitly, so the
      // intermediate rounding is their expressed choice rather than a silent
      // step inserted here.
      if (!(from.is_float() && from.bits() == 32)) {
        return reject(pair + ": the FP8 conversion instructions take a float32 "
                             "source; convert to float32 explicitly first");
      }
      const std::set<std::string> &allowed = Fp8AllowedRoundings();
      // An empty round selects the documented default (R with SAT below).
      const std::string token = attrs.has_round_mode ? round_token : "R";
      if (allowed.count(token) == 0) {
        std::ostringstream oss;
        oss << "rounding mode " << token;
        if (attrs.has_round_mode && attrs.round != token) {
          oss << " (from round=\"" << attrs.round << "\")";
        }
        oss << " is not supported by the FP8 conversion policy for `" << from
            << " -> " << to << "`";
        return reject(oss.str());
      }
      CastPlan plan;
      // The PTO default is R/SAT; pin both explicitly so the generated VMI
      // call never depends on PTODSL defaults.
      plan.rounding = token;
      plan.saturate = (attrs.has_sat && !attrs.sat) ? "NOSAT" : "SAT";
      return plan;
    }
    return reject(pair +
                  " is not a supported FP8 conversion (FP8 -> FP16/BF16 and "
                  "FP8 <-> integer are out of scope; convert through "
                  "float32 explicitly)");
  }

  // Non-FP8 conversion: the plan is fixed by the TIR semantics of the
  // operation, so an explicit annotation must agree with it. `finish` is the
  // one place that decides this, so a hint is accepted exactly when the plan
  // already does what it asks for and rejected otherwise — never dropped.
  auto finish = [&](const CastPlan &plan) -> Optional<CastPlan> {
    if (attrs.has_round_mode) {
      return reject(pair + ": this conversion does not accept a rounding mode");
    }
    if (attrs.has_sat && (attrs.sat || plan.saturate != "NOSAT")) {
      return reject(pair + ": this conversion does not accept an explicit "
                           "saturation flag");
    }
    return plan;
  };

  // Same-width signed/unsigned reinterprets have no validated lowering
  // (VMI vcvt requires a bit-width change); reject.
  if (from.bits() == to.bits() && from.code() == DataType::kInt &&
      to.code() == DataType::kUInt) {
    return reject("same-width int -> uint is a reinterpretation, which VMI "
                  "vcvt does not provide");
  }
  if (from.bits() == to.bits() && from.code() == DataType::kUInt &&
      to.code() == DataType::kInt) {
    return reject("same-width uint -> int is a reinterpretation, which VMI "
                  "vcvt does not provide");
  }
  // No width re-check here: IsSupportedElementDType at function entry already
  // restricts both sides to 8/16/32-bit lane-1 elements (and excludes FP64).
  CastPlan plan;
  // Every integer combination shares one rule: TVM integer casts truncate on
  // narrowing and extend on widening, PTODSL defaults narrowing to saturate,
  // and NOSAT restores the truncating semantics. The same-width int/uint
  // reinterpretations are rejected above.
  if ((from.is_int() || from.is_uint()) && (to.is_int() || to.is_uint())) {
    if (to.bits() < from.bits()) {
      plan.saturate = "NOSAT";
    }
    return finish(plan);
  }
  if (from.is_float() && (to.is_float() || to.is_bfloat16())) {
    if (to.bits() > from.bits() && to.is_float()) {
      // f16 -> f32 widening is exact.
      return finish(plan);
    }
    // f32 -> f16/bf16 narrowing: TIR Cast rounds to nearest; pin R so the
    // VMI vcvt never depends on PTODSL defaults. NOSAT keeps IEEE
    // overflow-to-inf semantics instead of clamping to the largest finite.
    if (from.bits() == 32 && to.bits() == 16) {
      plan.rounding = "R";
      plan.saturate = "NOSAT";
      return finish(plan);
    }
    // The only float narrowing left here is the cross-family f16 <-> bf16
    // step; its rounding semantics are not validated.
    return reject(
        "cross-family f16 <-> bf16 narrowing has no validated rounding "
        "semantics; convert through float32 explicitly");
  }
  // Bfloat16's is_float() is false, so bf16 -> f32 fell into the
  // reject branch below even though it is the same exact widening as
  // f16 -> f32 (and the PTO backend supports the vcvt). Allow exactly
  // this one bfloat combination; every other stays rejected as before.
  if (from.is_bfloat16() && to.is_float() && to.bits() == 32) {
    return finish(plan);
  }
  if (from.is_float() != to.is_float()) {
    // float <-> int conversions need per-combination rounding/overflow
    // validation against PTOAS; currently rejected.
    return reject("float <-> integer conversions need per-combination "
                  "rounding/overflow validation");
  }
  return reject("no proven VMI-equivalent conversion for this dtype pair");
}

std::optional<std::string> FindUnsupportedExprNode(const PrimExpr &expr) {
  WhitelistVisitor visitor;
  visitor(expr);
  return visitor.unsupported;
}

} // namespace pto
} // namespace tl
} // namespace tvm
