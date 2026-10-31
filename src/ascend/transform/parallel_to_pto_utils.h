/*!
 * \file parallel_to_pto_utils.h
 * \brief Shared read-only analysis helpers for PTO Parallel vectorization.
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

#include <map>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include "../../layout/layout.h"
#include "../../op/builtin.h"
#include "../target_utils.h"
#include "ascend/layout/pto_index_analysis.h"
#include "ascend/layout/pto_parallel_layout.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Array;
using ffi::GetRef;
using ffi::Map;
using ffi::ObjectRef;
using ffi::Optional;

namespace pto {

/*! Explicit legacy SIMD instructions already operate on vector blocks.
 * Their containing Parallel nest retains its pre-VMI treatment. */
bool UsesExplicitSIMD(const Stmt &stmt);

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
// IsSupportedAddressExpr live in pto_index_analysis.h (included above) so
// PTO lane-layout planning and the consumers here share one implementation.

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

/*! Value category of one value-Cast position.  The kind is supplied by the
 * caller (Verify from AnalyzeLaneUse, Vectorize from the converted input
 * shape); the planner never infers it from lanes. */
enum class CastValueKind {
  kScalar,
  kVectorVmi,
};

/*! Per-Cast planning context.  `logical_lanes` is validation information
 * only; `value_kind` is the dispatch key. */
struct CastContext {
  CastValueKind value_kind = CastValueKind::kScalar;
  int64_t logical_lanes = 1;
};

/*! Scalar value-Cast plan: either keep the input (identity) or rebuild an
 * ordinary scalar Cast with the validated annotations and the original span.
 * No new scalar rounding/saturation mode is introduced. */
enum class ScalarCastAction {
  kKeepInput,
  kRebuildCast,
};

struct ScalarCastPlan {
  ScalarCastAction action = ScalarCastAction::kKeepInput;
  /*! Validated annotations to carry onto the rebuilt Cast (the same shape as
   * CastNode::annotations).  Attributes that are proven to match the
   * existing scalar lowering are normalized away, so this map is empty for
   * the currently admitted directions. */
  ffi::Map<ffi::String, ffi::Any> annotations;
};

/*! One ordered vector conversion action. */
enum class VectorCastOp {
  /*! Same-sign widening: one logical vcvt. */
  kIntegerExtend,
  /*! Same-width signedness change / source-sign re-view: one
   * vinterpret_cast. */
  kIntegerReinterpret,
  /*! Narrowing including cross-sign: one logical vcvt with NOSAT. */
  kIntegerTruncate,
  /*! Real non-integer conversion (f16/bf16/FP8 -> f32, f32 -> f16/bf16/FP8);
   * empty options still mean a conversion, never a no-op. */
  kNonIntegerConvert,
};

struct VectorCastStep {
  VectorCastOp op = VectorCastOp::kIntegerExtend;
  DataType from; // element type consumed by this step
  DataType to;   // element type produced by this step
  /*! PTODSL rounding annotation value, e.g. "R"; empty when N/A. */
  std::string rounding;
  /*! PTODSL saturate annotation value, e.g. "NOSAT"; empty when N/A. */
  std::string saturate;
};

struct VectorCastPlan {
  /*! Ordered actions; an empty list is a proven identity (no conversion).
   * A non-empty list still may be a single real conversion with empty
   * options (that is not an identity). */
  std::vector<VectorCastStep> steps;
};

using CastPlan = std::variant<ScalarCastPlan, VectorCastPlan>;

// Maps a frontend boolean-combine op name (tirx./tir.bitwise_{and,or,not}
// or tir.{And,Or,Not}) to the PTODSL bitwise vector op whose mask register
// overload implements the boolean combine ("vand"/"vor"/"vnot"); returns
// nullptr for any other name.
const char *MaskLogicVmiOpName(const std::string &tir_op_name);

// The tirx.* unary math ops with a VMI mapping (fabs/exp/log/sqrt):
// returns the VMI op name, or nullptr for any other name.
const char *VectorUnaryVmiOpName(const std::string &tirx_op_name);

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

/*! Decide the conversion plan for a value Cast. This is the single admission
 * point for every Cast on the vectorize path; Verify and Vectorize both pass
 * an explicit CastContext and consume the same plan, so they cannot diverge.
 *
 * Dispatch is first scalar vs VMI, then integer vs non-integer:
 *  - scalar: identity KeepInput; integer widening/narrowing and the newly
 *    admitted same-width signedness change rebuild an ordinary scalar Cast;
 *    f16/bf16 <-> f32 keep their existing scalar lowering; non-identity FP8,
 *    float<->int and f16<->bf16 stay rejected;
 *  - VMI integer: identity 0 steps; same-width cross-sign one
 *    vinterpret_cast; same-sign widening one vcvt; cross-sign widening
 *    source-sign widen then same-width vinterpret_cast; narrowing one direct
 *    vcvt NOSAT. All 36 supported pairs are represented; no consumer-layout
 *    feasibility is guessed here;
 *  - VMI non-integer: a real conversion action for f16/bf16/FP8 -> f32
 *    (empty options are not a no-op) and for f32 -> f16/bf16/FP8 with the
 *    existing rounding/saturation policy.
 *
 * `cast` carries the backend annotations written by the frontend cast
 * helpers (rounding mode, saturation flag, stochastic rbits operand). They
 * are read here — never interpreted by the expression visitors — and an
 * attribute the selected conversion cannot honour is a rejection, not a
 * silently dropped hint. When `reason` is non-null it receives a diagnostic
 * naming the category, source/target types, lanes and the offending option. */
Optional<CastPlan> PlanValueCast(const CastNode *cast,
                                 const CastContext &context,
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
  /*! True when the region contains a Parallel requiring lane conversion
   * (directly or under wrappers). Explicit legacy SIMD nests are excluded;
   * regions with no converting Parallel retain their hand-written behavior. */
  bool has_any_parallel = false;
};

/*! Scan a SIMD_VF region body (the SBlock's body, including any scope
 * AttrStmt wrapper) for parallel units. One implementation shared by
 * Verify and Vectorize. */
PtoRegionScan ScanPtoRegion(const Stmt &region);

/*! A statement that carries no work: an Evaluate of an IntImm constant
 * (e.g. a store simplified away into a constant), an empty SeqStmt, or a
 * SeqStmt whose every element is itself a no-op. The shared contract for
 * the "no operation" arm of a conditional branch under T.Parallel — the
 * shape Legalize rewrites as "no else" (the masked-update Select form) and
 * Verify admits as Select-rewritable. */
bool IsNoOpStmt(const Stmt &stmt);

/*! Whether \p value may be substituted for its `Bind` var in a predicate proof.
 * A `Bind` is an SSA snapshot, so only expressions that cannot observe state
 * (constants, vars, arithmetic, comparisons, logic, Select, Min/Max, Cast) are
 * substitutable; a buffer load or a call keeps the var opaque. */
bool IsSafelyReevaluableValue(const PrimExpr &value);

/*! Whether \p predicate is provably true under \p analyzer. The scalar `Bind`
 * values in \p bind_env are resolved first, and the enclosing loop ranges must
 * already be bound into the analyzer. Only a proven-constant-true result
 * counts: anything undecided is not provably true. Dropping a predicate is
 * sound only when it holds for every lane and every iteration. */
bool IsProvablyTruePredicate(const PrimExpr &predicate,
                             arith::Analyzer *analyzer,
                             const Map<Var, PrimExpr> &bind_env);

/*! Collects the target Vars of the handle-valued `Bind` statements of a
 * statement tree (`q = A.data`, `r = q`, a reinterpret cast, ...). A buffer
 * declared over one of these pointers is a view whose storage the lane analysis
 * cannot follow, so Verify rejects every access to such a view.
 *
 * The pointer origin is deliberately not traced: a handle `BufferLoad` yields a
 * pointer value, not the storage that holds it. */
void CollectHandleBindTargets(const Stmt &stmt, std::set<const VarNode *> *out);

/*! What a function knows about where its buffers live. */
struct StorageOrigins {
  /*! Data Vars of real allocations (`AllocBuffer` / block `alloc_buffers`);
   * distinct allocations never overlap. */
  std::set<const VarNode *> allocations;
  /*! Data Vars of parameter buffers. */
  std::set<const VarNode *> parameters;
  /*! Parameter data Vars the caller declared as non-restrict; the source may
   * pass any of them the same storage. */
  std::set<const VarNode *> non_restrict;
  /*! True only when the function explicitly asserts `tir.noalias`. Absent is
   * not a promise: this pass runs before the kernel ABI is materialised, so two
   * parameters may still name one storage. */
  bool no_alias = false;
};

/*! Collects the storage origins of \p body plus the parameter ABI facts:
 * buffer_map entries, the no-alias attribute, and `tl.non_restrict_params`
 * (function attribute or block annotation). */
StorageOrigins CollectStorageOrigins(const Stmt &body,
                                     const Map<Var, Buffer> &buffer_map,
                                     bool no_alias,
                                     const Array<Var> &non_restrict_params);

/*! Whether two accesses may touch the same storage.
 *
 * Same data Var conflicts; two distinct allocations never do; two distinct
 * parameters are independent only under an explicit no-alias contract that does
 * not mark either as non-restrict. Anything else stays conservatively
 * conflicting. Verify and Legalize share this judgment. */
bool MayConflictStorage(const BufferNode *lhs, const BufferNode *rhs,
                        const StorageOrigins &origins);

/*!
 * \brief One relational node kind: how to take its operands, its PTODSL compare
 * name, and how to rebuild it when both operands stay scalar.
 *
 * The (node kind, compare name, scalar semantics) triple lives here once so a
 * new relational kind cannot drift between the verifier's whitelist, the
 * vectorizer's dispatch and its lane-uniform fold.
 */
struct RelationalKind {
  /*! \brief Fill a/b (when non-null) with the node's operands, in the node's
   *  own (a, b) order, and report whether \p e is of this kind. */
  bool (*take)(const PrimExpr &e, PrimExpr *a, PrimExpr *b);
  /*! \brief PTODSL compare name: "eq" / "ne" / "lt" / "le" / "gt" / "ge". */
  const char *cmp;
  /*! \brief Rebuild the comparison for two lane-uniform operands. */
  PrimExpr (*fold)(const PrimExpr &a, const PrimExpr &b);
};

/*!
 * \brief Match \p e against the relational table: on a hit fill a/b (either may
 * be null) with the node's operands and return the entry, else return nullptr.
 */
const RelationalKind *MatchRelational(const PrimExpr &e, PrimExpr *a,
                                      PrimExpr *b);

/*!
 * \brief Look the table up by PTODSL compare name; nullptr when unknown.
 */
const RelationalKind *RelationalByName(const char *cmp);

/*!
 * \brief Collect a flat list of stores from \p stmt (a store, a no-op, or a
 * Seq of those). Returns false when the statement is any other shape.
 */
bool CollectStores(const Stmt &stmt, std::vector<const BufferStoreNode *> *out);
} // namespace pto
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_PARALLEL_TO_PTO_UTILS_H_
