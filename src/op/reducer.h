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
/*! \brief SBlock annotation: Map<Var, Map<String, Any>> with keys
 *  "op" (String: sum/max/min) and optional "seed" (PrimExpr). */
constexpr const char *kReducerInfoV2 = "reducer_info_v2";
/*! \brief Statement marker on a combine store inside a T.Parallel loop:
 *  the side effect must execute once per logical iteration. PartitionLoop
 *  lowers it to a `REP == 0` guard (or strips it when the loop layout has
 *  no replication). The marker is generic: it only describes execution
 *  multiplicity, not reducer semantics. */
constexpr const char *kParallelMultiplicity = "tl.parallel_multiplicity";
} // namespace attr

/*! \brief Combine op kinds supported by reducer v2 (first version). */
enum class ReducerV2OpType : int { kSum = 0, kMax = 1, kMin = 2 };

/*! \brief Parse a reducer combine-op string ("sum"/"max"/"min"). */
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

/// T.reducer_init(acc): initialize the epoch's physical partials with the
/// combine identity. args[0] = tl.region(acc, "w").
class ReducerInitOpNode : public TileOperatorNode {
public:
  Buffer reducer;
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
    refl::ObjectDef<ReducerInitOpNode>().def_ro("reducer",
                                                &ReducerInitOpNode::reducer);
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

/// T.reducer_update(acc[indices], value): contribute `value` to the logical
/// output selected by `indices`, exactly once per dynamic logical iteration.
/// args[0] = tl.region(acc[indices], "rw") (point region), args[1] = value.
class ReducerUpdateOpNode : public TileOperatorNode {
public:
  Buffer reducer;
  ffi::Array<PrimExpr> indices; ///< logical output indices (region mins)
  PrimExpr value;               ///< contribution expression
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ReducerUpdateOp", ReducerUpdateOpNode,
                                    TileOperatorNode);

  Stmt Lower(const LowerArgs &lower_args,
             arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &layout_args,
                        InferLevel level) const override;
  TileOperator Clone() const override;
  static const Op &Get();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ReducerUpdateOpNode>()
        .def_ro("reducer", &ReducerUpdateOpNode::reducer)
        .def_ro("indices", &ReducerUpdateOpNode::indices)
        .def_ro("value", &ReducerUpdateOpNode::value);
  }
};

class ReducerUpdateOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ReducerUpdateOp, TileOperator,
                                             ReducerUpdateOpNode);
  TVM_DLL ReducerUpdateOp(ffi::Array<PrimExpr> args,
                          ffi::Map<ffi::String, ffi::ObjectRef> annotations =
                              ffi::Map<ffi::String, ffi::ObjectRef>());
  static const Op &Get();
};

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

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_REDUCER_H_
