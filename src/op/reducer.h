/*!
 * \file tl/op/reducer.h
 * \brief Reducer v2 first-class ops: reducer_init / reducer_update /
 *        finalize_reducer_v2.
 *
 * A v2 reducer is allocated in the virtual storage scope `local.reducer` and
 * may only be accessed through these three ops. They carry no lowering of
 * their own: the ReducerPlanAndMaterialize pass consumes them (after
 * LayoutInference) and rewrites them into ordinary fragment storage, plain
 * read-modify-write stores guarded by a generic execution-multiplicity
 * marker, and an explicit finalize plan. Backend codegen must never see
 * these ops (enforced by VerifyReducerConsumed).
 */

#ifndef TVM_TL_OP_REDUCER_H_
#define TVM_TL_OP_REDUCER_H_

#include "operator.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace attr {
/*! \brief SBlock annotation: Map<Var, Map<String, Any>> with the key
 *  "op" (String: sum/max/min/bitand/bitor/bitxor). */
constexpr const char *kReducerInfoV2 = "reducer_info_v2";
/*! \brief Legacy (v1) SBlock annotation emitted by
 *  `alloc_reducer(replication=...)`: Map<Var, Map<String, String>> with keys
 *  "op" and "rep". Consumed (and erased) by CanonicalizeLegacyReducer; the
 *  data-race verifier also reads it to exempt legacy reducer stores. Removed
 *  together with the legacy syntax. */
constexpr const char *kReducerInfo = "reducer_info";
/*! \brief Statement marker on a combine store inside a T.Parallel loop:
 *  the side effect must execute once per logical iteration. PartitionLoop
 *  lowers it to a `REP == 0` guard (or strips it when the loop layout has
 *  no replication). The marker is generic: it only describes execution
 *  multiplicity, not reducer semantics. */
constexpr const char *kParallelMultiplicity = "tl.parallel_multiplicity";
} // namespace attr

/*! \brief Combine op kinds supported by reducer v2. The bitwise ops require
 *  an integer dtype (checked when their identity is materialized). */
enum class ReducerV2OpType : int {
  kSum = 0,
  kMax = 1,
  kMin = 2,
  kBitAnd = 3,
  kBitOr = 4,
  kBitXor = 5,
};

/*! \brief Parse a reducer combine-op string
 *  ("sum"/"max"/"min"/"bitand"/"bitor"/"bitxor"). */
ReducerV2OpType ParseReducerV2OpType(const ffi::String &op_str);

/*! \brief Identity element of a combine op for the given dtype. */
PrimExpr ReducerV2Identity(ReducerV2OpType op, DataType dtype);

/*! \brief combine(lhs, rhs) expression for a combine op. */
PrimExpr ReducerV2Combine(ReducerV2OpType op, const PrimExpr &lhs,
                          const PrimExpr &rhs);

/*! \brief True if `buffer` lives in the virtual `local.reducer` scope. */
inline bool IsReducerV2Buffer(const Buffer &buffer) {
  return buffer.defined() && buffer.scope() == "local.reducer";
}

/// T.reducer_init(acc, init=None): open the epoch. The physical partials
/// always start from the combine identity; the optional `init` value is a
/// logical starting value, combined exactly once per logical output at
/// finalize time (so physical replication can never multiply it).
/// args[0] = tl.region(acc, "w"), optional args[1] = init value.
class ReducerInitOpNode : public TileOperatorNode {
public:
  Buffer reducer;
  ffi::Optional<PrimExpr> seed; ///< logical starting value (args[1])
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ReducerInitOp", ReducerInitOpNode,
                                    TileOperatorNode);

  Stmt Lower(const LowerArgs &lower_args,
             arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &layout_args,
                        InferLevel level) const override;
  TileOperator Clone() const override;
  static const Op &Get();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ReducerInitOpNode>()
        .def_ro("reducer", &ReducerInitOpNode::reducer)
        .def_ro("seed", &ReducerInitOpNode::seed);
  }
};

class ReducerInitOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ReducerInitOp, TileOperator,
                                             ReducerInitOpNode);
  TVM_DLL ReducerInitOp(ffi::Array<PrimExpr> args,
                        ffi::Map<ffi::String, ffi::ObjectRef> annotations =
                            ffi::Map<ffi::String, ffi::ObjectRef>());
  static const Op &Get();
};

/// tl.reducer_update(acc[indices], value): contribute `value` to the logical
/// output selected by `indices`, exactly once per dynamic logical iteration.
///
/// Unlike the other epoch ops this is a plain builtin INTRINSIC, not a tile
/// op: it executes once per iteration inside T.Parallel,
/// owns no layout (the enclosing loop and the planner decide physics), and
/// never lowers on its own (ReducerPlanAndMaterialize rewrites it; leftovers
/// are caught by VerifyReducerConsumed). args[0] is a plain BufferLoad
/// `acc[indices]` — an update-target descriptor whose multi-dim indices the
/// planner reads directly, not a read of the reducer (analyses may treat it
/// as a read; updates commute, and VerifyReducerEpoch pins init/finalize to
/// straight-line code, so no cross-statement write ordering is lost).
TVM_DLL const Op &reducer_update();

/*! \brief True for tl.reducer_update calls. Pipeline statement
 *  classification keys on this (through this predicate, so a future
 *  sibling op — e.g. a rescale intrinsic — only needs a new clause here):
 *  the call is compute, not a copy, even though it does not parse as a
 *  tile op. Note the call's write stays hidden from layout inference (the
 *  target buffer deliberately has no layout): the enclosing parallel loop
 *  free-infers like any anchor-less loop, and the layout-order RFC is the
 *  designated home for making that ordering explicit. */
TVM_DLL bool IsReducerUpdateCall(const tirx::CallNode *call);

/*! \brief Parsed form of a tl.reducer_update call. */
struct ReducerUpdateArgs {
  Buffer reducer;
  ffi::Array<PrimExpr> indices; ///< logical output indices
  PrimExpr value;               ///< contribution expression
};

/*! \brief Parse and validate a tl.reducer_update call. */
ReducerUpdateArgs ParseReducerUpdate(const tirx::CallNode *call);

/// T.finalize_reducer(acc, dst): complete the epoch's cross-participant
/// communication and write the logical result into the independent
/// destination fragment. args[0] = tl.region(acc, "rw"),
/// args[1] = tl.region(dst, "w").
class FinalizeReducerV2OpNode : public TileOperatorNode {
public:
  Buffer reducer;
  Buffer dst;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.FinalizeReducerV2Op",
                                    FinalizeReducerV2OpNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &lower_args,
             arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &layout_args,
                        InferLevel level) const override;
  TileOperator Clone() const override;
  static const Op &Get();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FinalizeReducerV2OpNode>()
        .def_ro("reducer", &FinalizeReducerV2OpNode::reducer)
        .def_ro("dst", &FinalizeReducerV2OpNode::dst);
  }
};

class FinalizeReducerV2Op : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(FinalizeReducerV2Op, TileOperator,
                                             FinalizeReducerV2OpNode);
  TVM_DLL
  FinalizeReducerV2Op(ffi::Array<PrimExpr> args,
                      ffi::Map<ffi::String, ffi::ObjectRef> annotations =
                          ffi::Map<ffi::String, ffi::ObjectRef>());
  static const Op &Get();
};

/// tl.finalize_reducer: the MATERIALIZED collective emitted by
/// ReducerPlanAndMaterialize (not user-facing). Performs the plan's
/// cross-participant combine on the reducer's physical partial storage.
/// args[0] = tl.region(storage, "rw"), args[1] = combine op enum; optional
/// args[2] = reducing_threads, args[3] = scale select an explicit narrow
/// collective (default: participant-wide AllReduce derived from the
/// storage layout's replicate extent). Lowering is target-specific via
/// RegisterFinalizeReducerImpl (CUDA/ROCm share the plan contract).
class FinalizeReducerOpNode : public TileOperatorNode {
public:
  tirx::Buffer reducer;
  ReducerV2OpType op;
  // Batch size for batched AllReduce (1 = scalar path, same as T.reduce
  // default).
  int batch{1};
  // Explicit collective plan (reducer v2 narrow plans): flattened
  // (reducing_threads, scale) pairs, one per reduction step. Each step's
  // AllReduce combines `reducing_threads / scale` lanes at stride `scale`.
  // `explicit_plan` distinguishes an explicit empty plan (LocalComplete: no
  // communication) from the legacy wide plan (width derived from the
  // storage layout's ReplicateExtent).
  bool explicit_plan{false};
  ffi::Array<Integer> plan_steps;
  // Optional logical seed, combined into every physical slot exactly once
  // after the collective (all replicas hold the final value by then, so a
  // uniform per-slot combine applies the seed once per logical output).
  ffi::Optional<PrimExpr> seed;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.FinalizeReducerOp",
                                    FinalizeReducerOpNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FinalizeReducerOpNode>()
        .def_ro("reducer", &FinalizeReducerOpNode::reducer)
        .def_ro("op", &FinalizeReducerOpNode::op)
        .def_ro("batch", &FinalizeReducerOpNode::batch)
        .def_ro("explicit_plan", &FinalizeReducerOpNode::explicit_plan)
        .def_ro("plan_steps", &FinalizeReducerOpNode::plan_steps)
        .def_ro("seed", &FinalizeReducerOpNode::seed);
  }

  Stmt Lower(const LowerArgs &lower_args,
             arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &layout_args,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const;
};

using FinalizeReducerTargetPredicate = bool (*)(Target target);

struct FinalizeReducerImpl {
  const char *name;
  FinalizeReducerTargetPredicate match_target;

  Stmt (*lower)(const FinalizeReducerOpNode &op, const LowerArgs &lower_args,
                arith::Analyzer *analyzer);
};

void RegisterFinalizeReducerImpl(FinalizeReducerImpl impl);

class FinalizeReducerOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(FinalizeReducerOp, TileOperator,
                                             FinalizeReducerOpNode);
  TVM_DLL FinalizeReducerOp(ffi::Array<PrimExpr> args,
                            ffi::Map<ffi::String, ffi::ObjectRef> annotations =
                                ffi::Map<ffi::String, ffi::ObjectRef>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_REDUCER_H_
