/*!
 * \file tl/op/deferred_reducer.h
 * \brief First-class deferred reducer operations.
 */

#ifndef TVM_TL_OP_DEFERRED_REDUCER_H_
#define TVM_TL_OP_DEFERRED_REDUCER_H_

#include "operator.h"
#include "reduce.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace attr {

/*! \brief SBlock metadata mapping reducer storage Vars to reducer definitions.
 */
constexpr const char *kReducerInfo = "reducer_info";

/*! \brief Planned combine type attached to first-class reducer calls. */
constexpr const char *kReducerType = "reducer_type";

/*! \brief Optional logical seed attached to planned reducer calls. */
constexpr const char *kReducerSeed = "reducer_seed";

/*! \brief Whether an update needs once-per-logical-iteration lowering. */
constexpr const char *kReducerParallelOnce = "reducer_parallel_once";

/*!
 * \brief Destination Fragment layout proving that every physical replica
 * independently holds a complete partial for its local output slots.
 */
constexpr const char *kReducerLocalCompleteLayout =
    "reducer_local_complete_layout";

/*! \brief Logical output indices retained after physical region rewriting. */
constexpr const char *kReducerLogicalIndices = "reducer_logical_indices";

/*! \brief Statement marker consumed by parallel-loop partitioning. */
constexpr const char *kParallelMultiplicity = "tl.parallel_multiplicity";

/*!
 * \brief Statement marker forcing physical partitioning without suppressing
 * replicated logical iterations.
 */
constexpr const char *kParallelPartitionRequired =
    "tl.parallel_partition_required";

} // namespace attr

class ReducerInfoNode : public ffi::Object {
public:
  ReduceType combine_type;
  ffi::Optional<PrimExpr> seed;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ReducerInfo", ReducerInfoNode,
                                    ffi::Object);

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<ReducerInfoNode>()
        .def_ro("combine_type", &ReducerInfoNode::combine_type)
        .def_ro("seed", &ReducerInfoNode::seed);
  }
};

class ReducerInfo : public ffi::ObjectRef {
public:
  TVM_DLL ReducerInfo(const ffi::String &op,
                      ffi::Optional<PrimExpr> seed = std::nullopt);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ReducerInfo, ffi::ObjectRef,
                                             ReducerInfoNode);
};

class ReducerInitOpNode : public TileOperatorNode {
public:
  Buffer reducer;
  ReduceType combine_type;
  ffi::Optional<Fragment> local_complete_layout;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ReducerInitOp", ReducerInitOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<ReducerInitOpNode>()
        .def_ro("reducer", &ReducerInitOpNode::reducer)
        .def_ro("combine_type", &ReducerInitOpNode::combine_type)
        .def_ro("local_complete_layout",
                &ReducerInitOpNode::local_complete_layout);
  }

  Stmt Lower(const LowerArgs &lower_args,
             arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &layout_args,
                        InferLevel level) const override;
  TileOperator Clone() const override;
  static const Op &Get();
};

class ReducerInitOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ReducerInitOp, TileOperator,
                                             ReducerInitOpNode);
  TVM_DLL ReducerInitOp(ffi::Array<PrimExpr> args,
                        ffi::Map<ffi::String, ffi::ObjectRef> annotations = {});
  static const Op &Get();
};

class ReducerUpdateOpNode : public TileOperatorNode {
public:
  Buffer reducer;
  ffi::Array<PrimExpr> logical_indices;
  PrimExpr contribution;
  ReduceType combine_type;
  bool parallel_once{false};
  ffi::Optional<Fragment> local_complete_layout;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ReducerUpdateOp", ReducerUpdateOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<ReducerUpdateOpNode>()
        .def_ro("reducer", &ReducerUpdateOpNode::reducer)
        .def_ro("logical_indices", &ReducerUpdateOpNode::logical_indices)
        .def_ro("contribution", &ReducerUpdateOpNode::contribution)
        .def_ro("combine_type", &ReducerUpdateOpNode::combine_type)
        .def_ro("parallel_once", &ReducerUpdateOpNode::parallel_once)
        .def_ro("local_complete_layout",
                &ReducerUpdateOpNode::local_complete_layout);
  }

  Stmt Lower(const LowerArgs &lower_args,
             arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &layout_args,
                        InferLevel level) const override;
  TileOperator Clone() const override;
  static const Op &Get();
};

class ReducerUpdateOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ReducerUpdateOp, TileOperator,
                                             ReducerUpdateOpNode);
  TVM_DLL
  ReducerUpdateOp(ffi::Array<PrimExpr> args,
                  ffi::Map<ffi::String, ffi::ObjectRef> annotations = {});
  static const Op &Get();
};

class FinalizeReducerOpNode : public TileOperatorNode {
public:
  Buffer reducer;
  Buffer destination;
  ReduceType combine_type;
  ffi::Optional<PrimExpr> seed;
  ffi::Optional<Fragment> local_complete_layout;
  int batch{1};

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.FinalizeReducerOp",
                                    FinalizeReducerOpNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<FinalizeReducerOpNode>()
        .def_ro("reducer", &FinalizeReducerOpNode::reducer)
        .def_ro("destination", &FinalizeReducerOpNode::destination)
        .def_ro("combine_type", &FinalizeReducerOpNode::combine_type)
        .def_ro("seed", &FinalizeReducerOpNode::seed)
        .def_ro("local_complete_layout",
                &FinalizeReducerOpNode::local_complete_layout)
        .def_ro("batch", &FinalizeReducerOpNode::batch);
  }

  Stmt Lower(const LowerArgs &lower_args,
             arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &layout_args,
                        InferLevel level) const override;
  TileOperator Clone() const override;
  static const Op &Get();
};

class FinalizeReducerOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(FinalizeReducerOp, TileOperator,
                                             FinalizeReducerOpNode);
  TVM_DLL
  FinalizeReducerOp(ffi::Array<PrimExpr> args,
                    ffi::Map<ffi::String, ffi::ObjectRef> annotations = {});
  static const Op &Get();
};

using FinalizeReducerTargetPredicate = bool (*)(Target target);

struct FinalizeReducerImpl {
  const char *name;
  FinalizeReducerTargetPredicate match_target;
  Stmt (*lower)(const FinalizeReducerOpNode &op, const LowerArgs &lower_args,
                arith::Analyzer *analyzer);
};

void RegisterFinalizeReducerImpl(FinalizeReducerImpl impl);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_DEFERRED_REDUCER_H_
