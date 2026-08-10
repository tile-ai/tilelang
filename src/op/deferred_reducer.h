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

/*! \brief Planned physical partial groups attached to init/finalize calls. */
constexpr const char *kReducerPartialPlans = "reducer_partial_plans";

/*! \brief Logical output indices retained after physical region rewriting. */
constexpr const char *kReducerLogicalIndices = "reducer_logical_indices";

/*! \brief Whether an update must retain every inferred physical replica. */
constexpr const char *kReducerPartitionRequired = "reducer_partition_required";

/*! \brief Statement marker consumed by parallel-loop partitioning. */
constexpr const char *kParallelMultiplicity = "tl.parallel_multiplicity";

/*!
 * \brief Statement marker forcing physical partitioning without suppressing
 * replicated logical iterations.
 */
constexpr const char *kParallelPartitionRequired =
    "tl.parallel_partition_required";

/*!
 * \brief Planned reducer update retained until physical-layout lowering.
 *
 * The AttrStmt node is the ReduceType and its value is the logical
 * contribution. Reducer-aware vectorization consumes this marker after buffer
 * layouts and parallel-loop ownership have been materialized.
 */
constexpr const char *kReducerUpdate = "tl.reducer_update";

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

/*!
 * \brief One independently accumulated physical partial group.
 *
 * A canonical group has no partial layout and is finalized across the complete
 * participant range. A projected group uses partial_layout for storage and
 * zero or more thread-reduction steps; an empty step list is the LocalComplete
 * case.
 */
class ReducerPartialPlanNode : public ffi::Object {
public:
  bool canonical{false};
  ffi::Optional<Fragment> partial_layout;
  ffi::Array<Integer> step_extents;
  ffi::Array<Integer> step_scales;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ReducerPartialPlan",
                                    ReducerPartialPlanNode, ffi::Object);

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<ReducerPartialPlanNode>()
        .def_ro("canonical", &ReducerPartialPlanNode::canonical)
        .def_ro("partial_layout", &ReducerPartialPlanNode::partial_layout)
        .def_ro("step_extents", &ReducerPartialPlanNode::step_extents)
        .def_ro("step_scales", &ReducerPartialPlanNode::step_scales);
  }
};

class ReducerPartialPlan : public ffi::ObjectRef {
public:
  TVM_DLL ReducerPartialPlan(bool canonical,
                             ffi::Optional<Fragment> partial_layout,
                             ffi::Array<Integer> step_extents,
                             ffi::Array<Integer> step_scales);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ReducerPartialPlan, ffi::ObjectRef,
                                             ReducerPartialPlanNode);
};

class ReducerInitOpNode : public TileOperatorNode {
public:
  ffi::Array<Buffer> partials;
  ReduceType combine_type;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ReducerInitOp", ReducerInitOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<ReducerInitOpNode>()
        .def_ro("partials", &ReducerInitOpNode::partials)
        .def_ro("combine_type", &ReducerInitOpNode::combine_type);
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
  ffi::Array<PrimExpr> physical_indices;
  PrimExpr contribution;
  ReduceType combine_type;
  bool parallel_once{false};
  bool partition_required{false};

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ReducerUpdateOp", ReducerUpdateOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<ReducerUpdateOpNode>()
        .def_ro("reducer", &ReducerUpdateOpNode::reducer)
        .def_ro("logical_indices", &ReducerUpdateOpNode::logical_indices)
        .def_ro("physical_indices", &ReducerUpdateOpNode::physical_indices)
        .def_ro("contribution", &ReducerUpdateOpNode::contribution)
        .def_ro("combine_type", &ReducerUpdateOpNode::combine_type)
        .def_ro("parallel_once", &ReducerUpdateOpNode::parallel_once)
        .def_ro("partition_required", &ReducerUpdateOpNode::partition_required);
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
  ffi::Array<Buffer> partials;
  ffi::Array<ReducerPartialPlan> partial_plans;
  Buffer destination;
  ReduceType combine_type;
  ffi::Optional<PrimExpr> seed;
  int batch{1};

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.FinalizeReducerOp",
                                    FinalizeReducerOpNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<FinalizeReducerOpNode>()
        .def_ro("partials", &FinalizeReducerOpNode::partials)
        .def_ro("partial_plans", &FinalizeReducerOpNode::partial_plans)
        .def_ro("destination", &FinalizeReducerOpNode::destination)
        .def_ro("combine_type", &FinalizeReducerOpNode::combine_type)
        .def_ro("seed", &FinalizeReducerOpNode::seed)
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
