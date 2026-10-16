/*!
 * \file parallel_to_pto_utils.h
 * \brief Shared read-only analysis helpers for the PTO Parallel vectorize
 * path. See docs/vectorize_parallel_to_pto_design.md.
 *
 * VerifyParallelToPTO consumes these helpers to check the input contract;
 * VectorizeParallelToPTO re-runs the same analysis to build its VMI code.
 * Keeping one implementation guarantees the two passes classify the same
 * access identically.
 */

#ifndef TVM_TL_ASCEND_TRANSFORM_PARALLEL_TO_PTO_UTILS_H_
#define TVM_TL_ASCEND_TRANSFORM_PARALLEL_TO_PTO_UTILS_H_

#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <optional>
#include <set>
#include <string>

#include "../../layout/layout.h"
#include "../../op/builtin.h"
#include "../../op/pto_index_analysis.h"
#include "../target_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Array;
using ffi::GetRef;
using ffi::Map;
using ffi::ObjectRef;
using ffi::Optional;

namespace pto {

/*!
 * \brief Classification of one buffer access against the lane mapping.
 *
 * The classification describes how the
 * *address* varies over lanes/chunk, not the loaded data.
 */
enum class AccessPattern {
  kUnknown = 0,
  /*! Address = start + lane for lane in [0, L): continuous vector. */
  kContinuous,
  /*! Address independent of lane at the current program point. */
  kLaneUniform,
};

std::string AccessPatternToString(AccessPattern pattern);

/*!
 * \brief Result of the shared single-access analysis.
 *
 * Verify consumes pattern legality; Vectorize additionally consumes the
 * start indices/offset to build tl.access_ptr.
 */
struct MemoryAccess {
  Buffer buffer;
  DataType elem_dtype;
  bool is_write = false;

  /*! Full element offset relative to buffer->data, incl. elem_offset,
   * version dims, outer coordinates and chunk offset (lane term included).
   */
  PrimExpr data_offset;
  /*! indices_after_inverse: buffer indices with the inverse-mapped logical
   * coordinates (lane appears via the lane Var in one index). */
  Array<PrimExpr> indices_after_inverse;
  /*! indices with lane bound to 0: the start element of this vector access.
   */
  Array<PrimExpr> start_indices;
  /*! data_offset with lane bound to 0. */
  PrimExpr start_offset;
  AccessPattern pattern = AccessPattern::kUnknown;
};

// PtoAnalysisError, SubstituteVar, ResolveIndexAliases and
// IsSupportedAddressExpr live in src/op/pto_index_analysis.h (included
// above) so the layout-planning code in src/op and the consumers here
// share one implementation.

/*!
 * \brief The inverse lane mapping of a 1D PTO parallel loop.
 *
 * For a 1D loop of extent E with lanes L (E % L == 0): the fragment is
 * forward_thread = i % L, forward_index = [i // L]. Its inverse recovers
 *   i = p * L + lane
 * where p is the per-lane chunk position.
 */
struct PtoInverseMapping {
  /*! Lane variable of the enclosing SIMD_VF (analysis placeholder). */
  Var lane_var;
  /*! Chunk position variable (per-lane column index). */
  Var chunk_var;
  int64_t lanes = 0;
  /*! Logical extent E of the vectorized dimension (the original loop
   * extent, without padding). */
  int64_t logical_extent = 0;
  /*! Padded extent P = ceil(E / L) * L of the fragment input space on the
   * vectorized dimension. E == P for divisible extents. */
  int64_t padded_extent = 0;
  /*! Chunk count Q = P / L — the per-row chunk range of the vectorized
   * dimension (never the global position K*Q). */
  int64_t chunk_count = 0;
  /*! Tail lanes E % L (0 when E is divisible by L). */
  int64_t tail_lanes = 0;
  /*! Inverse expression of the vectorized coordinate:
   * logical index = chunk * L + lane. */
  PrimExpr index_expr;

  // ------------------------------------------------------------------
  // 2D units. A 2D unit is a two-layer adjacent
  // Parallel nest; the layout annotation sits on the outermost For and
  // the Fragment input coordinates stay (i, j) regardless of which
  // dimension is vectorized.
  // ------------------------------------------------------------------
  /*! True when the unit is a 2D nest. */
  bool is_2d = false;
  /*! Original loop vars of the nest (i = outer, j = inner). */
  Var outer_var;
  Var inner_var;
  /*! Extents of the two iterations (i and j). */
  int64_t outer_extent = 0;
  int64_t inner_extent = 0;
  /*! Vectorized dimension: true selects the inner (j) dim. */
  bool select_inner = true;

  /*! The loop variable substituted into addresses (the vectorized one). */
  const VarNode *vector_var() const {
    if (!is_2d) {
      return nullptr; // 1D: the caller passes the single loop var
    }
    return select_inner ? inner_var.get() : outer_var.get();
  }
  /*! The loop variable kept as a serial loop (2D only). */
  const VarNode *serial_var() const {
    if (!is_2d) {
      return nullptr;
    }
    return select_inner ? outer_var.get() : inner_var.get();
  }
};

/*! Invert the pto_parallel_loop_layout fragment of a Parallel unit.
 * Accepts a 1D Parallel For, or the outermost For of a two-layer adjacent
 * Parallel nest (2D). Throws PtoAnalysisError when the fragment is
 * missing, has an unsupported shape, or cannot be inverted. */
PtoInverseMapping InvertPtoLaneLayout(const For &loop);

/*! Classify one buffer access (load or store indices) against the inverse
 * mapping. \p indices are the buffer indices in the original loop body
 * expressed over the loop variable; \p loop_var is the original parallel
 * loop variable. \p bind_env maps Bind vars in scope to their defining
 * expressions so index aliases are resolved before classification
 * pass an empty map when no binds are in scope.
 *
 * The helper resolves bind aliases, substitutes
 * loop_var -> chunk * L + lane_var, computes ElemOffset, and proves the
 * access pattern with the analyzer over the full lane range (continuous:
 * data_offset == start_offset + lane as an identity; lane-uniform: the
 * offset does not depend on lane). Returns a fully populated MemoryAccess
 * or throws PtoAnalysisError with a reason. */
MemoryAccess AnalyzeBufferAccess(const Buffer &buffer,
                                 const Array<PrimExpr> &indices, bool is_write,
                                 const Var &loop_var,
                                 const PtoInverseMapping &mapping,
                                 arith::Analyzer *analyzer,
                                 const Map<Var, PrimExpr> &bind_env = {});

/*! Check whether a dtype is supported by the unified element
 * addressing convention: the buffer element dtype itself must not carry
 * vector lanes and must not be a packed sub-byte type (FP4 etc.).
 *
 * E4M3FN and E5M2 are supported element types; the other FP8 variants
 * (float8_e4m3 with its own TIR dtype, E8M0, FNUZ, HiF8) are not, even
 * where the PTO codegen would map them to the same target type — only the
 * two dtypes listed in IsSupportedFloat8 take part in the FP8 conversion
 * matrix. */
bool IsSupportedElementDType(DataType dtype);

/*! The two FP8 element types the conversion matrix supports: float8_e4m3fn and
 * float8_e5m2. Deliberately narrower than the codegen's FP8 name mapping:
 * admitting a type here means its conversion directions were validated, not
 * just that a target name exists. */
bool IsSupportedFloat8(DataType dtype);

/*! Cast conversion descriptor.
 * empty() means the combination has no proven-equivalent VMI mapping and
 * must be rejected by Verify. */
struct CastPlan {
  /*! Identity conversion (from == to): keep the original value, emit no
   * vcvt (VMI rejects width-preserving conversions). */
  bool identity = false;
  /*! PTODSL saturate annotation value, e.g. "NOSAT"; empty when the
   * conversion must not rely on PTODSL defaults. */
  std::string saturate;
  /*! PTODSL rounding annotation value, e.g. "Z"; empty when N/A. */
  std::string rounding;
};

/*! Operation-level type support for the vector arithmetic emitted
 * here. VMI vmul only supports i16/i32/f16/bf16/f32
 * elements; vadd/vsub additionally allow i8; vdiv admits f16/f32 only.
 * Unsupported combinations are rejected instead of synthesizing
 * widening chains. */
bool IsSupportedVectorUnaryOp(const std::string &tirx_op, DataType dtype);
bool IsSupportedVectorBinaryOp(const std::string &vmi_op, DataType dtype);
bool ContainsRuntimeValue(const PrimExpr &expr);
bool IsFloatFamily(DataType dtype);

/*! Read-only lane-use classification of a value expression.
 * Shared by the metadata check and the operation type check so both
 * consume the same source semantics. Mirrors the structural conversion
 * the Vectorize pass performs (no new algebraic simplification). */
enum class LaneUse {
  kUniform, ///< stays scalar under the current conversion rules
  kVarying, ///< needs a vector value (incl. the loop index as a value)
  kUnknown, ///< cannot analyze; caller must reject with the reason
};

std::string LaneUseToString(LaneUse use);

/*! Analysis context for AnalyzeLaneUse: the unit's loop var, inverse
 * mapping and analyzer, the bind-definition environment, a cache of
 * already-classified bind vars, the set of known external scalar vars
 * (params / buffer data pointers / outer-scope definitions), and the
 * recursion guards of the classification walk. */
struct LaneUseContext {
  Var loop_var;
  /*! In a 2D unit the coordinate of the non-vectorized dimension
   * is a uniform scalar at the program point. Unset for 1D. */
  Var serial_var;
  PtoInverseMapping mapping;
  arith::Analyzer *analyzer = nullptr;
  Map<Var, PrimExpr> bind_env;
  Map<Var, Integer> classified; // bind var -> 0 uniform / 1 varying
  /*! Known external scalar vars: vars outside the unit that are
   * legitimately defined. A var that is neither in bind_env nor here is
   * *undefined* -> Unknown, not Uniform. */
  std::set<const VarNode *> external_defs;
  /*! Bind vars currently being analyzed (cycle guard). */
  std::set<const VarNode *> visiting;
  /*! Recursion budget for on-demand bind analysis. */
  int budget = 256;
};

/*! Classify \p expr bottom-up. Constants and *known*
 * external scalar vars are Uniform; the loop var as a value is Varying;
 * BufferLoad goes through AnalyzeBufferAccess (uniform load -> Uniform,
 * continuous -> Varying); Add/Mul/Cast propagate operands; original Bind
 * vars use their recorded classification (on-demand analysis is guarded
 * by the visiting set and budget; cycles and exhaustion -> Unknown).
 * \p reason is filled on kUnknown. */
LaneUse AnalyzeLaneUse(const PrimExpr &expr, LaneUseContext *ctx,
                       std::string *reason);

/*! Metadata-mode lane classification: same Var/Bind/BufferLoad
 * rules as AnalyzeLaneUse, but the *node table* is the metadata table —
 * pure scalar Sub/Div/Mod/FloorDiv/FloorMod/Min/Max/comparison/boolean
 * operators and StringImm constants propagate their operands' lane use
 * instead of being rejected by the value whitelist. Unknown nodes still
 * produce kUnknown. */
LaneUse AnalyzeLaneUseMeta(const PrimExpr &expr, LaneUseContext *ctx,
                           std::string *reason);

/*! Decide the conversion attributes for a value Cast. This is the single
 * admission point for every Cast on the vectorize path; Verify and Vectorize
 * both consume its result, so they cannot diverge.
 *
 * Same-width reinterpretation-free identity is impossible in
 * TIR Cast, so the proven set is:
 *  - integer narrowing/widening with truncating semantics: TVM integer
 *    casts truncate on narrowing (codegen_llvm CreateIntCast), PTODSL
 *    needs saturate="NOSAT" for narrowing to keep that semantics;
 *  - float widening (f16->f32, bf16->f32): exact;
 *  - the FP8 matrix: E4M3FN/E5M2 <-> f32. The conversion instructions the
 *    target exposes are FP32 <-> FP8, so a narrowing conversion carries an
 *    explicit rounding and saturation attribute (the FP8 conversion policy in
 *    the implementation file) and a half (f16/bf16) source is rejected rather
 *    than routed through float32 internally. FP8 -> FP16/BF16, FP8 <-> integer,
 *    E4M3FN <-> E5M2 and every other FP8 variant stay out of scope.
 * Everything else (float narrowing, float<->int, unusual bit widths)
 * returns nullopt -> reject.
 *
 * `cast` carries the backend annotations written by the frontend cast
 * helpers (rounding mode, saturation flag, stochastic rbits operand). They
 * are read here — never interpreted by the expression visitors — and an
 * attribute the selected conversion cannot honour is a rejection, not a
 * silently dropped hint. When `reason` is non-null it receives a diagnostic
 * naming the source type, target type and the offending option. */
Optional<CastPlan> PlanValueCast(const CastNode *cast,
                                 std::string *reason = nullptr);

/*! Check whether an expression only uses nodes from the supported
 * whitelist: BufferLoad, constants, scalar Vars, loop
 * indices, Add, Mul, limited Cast, Bind values. Returns nullopt when
 * supported; otherwise a short description of the offending node. */
std::optional<std::string> FindUnsupportedExprNode(const PrimExpr &expr);

/*! Result of scanning a SIMD_VF region for parallel units. Shared by
 * Verify and Vectorize so both passes agree on skip/reject/convert
 * decisions. */
struct PtoRegionScan {
  /*! Sequential 1D parallel units found (outermost For of each chain). */
  std::vector<For> units;
  /*! True when a Parallel exists under a wrapper we cannot currently
   * convert (IfThenElse / While / nested 2D+ parallel). Verify
   * rejects with \p issue; Vectorize treats it as an internal error. */
  bool has_unsupported_wrapper = false;
  std::string issue;
  /*! True when the region contains any T.Parallel at all (directly or
   * under wrappers). A region with none is a pure hand-written VMI
   * region: both passes skip it entirely. */
  bool has_any_parallel = false;
};

/*! Scan a SIMD_VF region body (the SBlock's body, including any scope
 * AttrStmt wrapper) for parallel units. One implementation shared by
 * Verify and Vectorize. */
PtoRegionScan ScanPtoRegion(const Stmt &region);

} // namespace pto
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_PARALLEL_TO_PTO_UTILS_H_
