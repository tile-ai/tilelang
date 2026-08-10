/*!
 * \file tl/op/deferred_reducer.cc
 * \brief First-class deferred reducer operations.
 */

#include "deferred_reducer.h"

#include "region.h"
#include "utils.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/op_attr_types.h>

#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

std::vector<FinalizeReducerImpl> &FinalizeReducerImplRegistry() {
  static std::vector<FinalizeReducerImpl> registry;
  return registry;
}

const FinalizeReducerImpl &ResolveFinalizeReducerImpl(Target target) {
  const FinalizeReducerImpl *matched_impl = nullptr;
  for (const FinalizeReducerImpl &impl : FinalizeReducerImplRegistry()) {
    if (!impl.match_target(target)) {
      continue;
    }
    ICHECK(matched_impl == nullptr)
        << "tl.finalize_reducer found multiple target-specific implementations "
           "for "
        << target->str() << ": " << matched_impl->name << " and " << impl.name;
    matched_impl = &impl;
  }
  ICHECK(matched_impl != nullptr)
      << "tl.finalize_reducer requires a target-specific implementation for "
      << target->str();
  return *matched_impl;
}

Buffer GetLoweredBuffer(const Buffer &buffer, const LowerArgs &lower_args) {
  auto remapped = lower_args.buffer_remap.find(buffer);
  return remapped == lower_args.buffer_remap.end() ? buffer
                                                   : (*remapped).second;
}

ReduceType GetPlannedReduceType(const Map<String, ObjectRef> &annotations) {
  Optional<ObjectRef> value = annotations.Get(attr::kReducerType);
  if (!value.defined()) {
    return ReduceType();
  }
  Optional<StringImm> name = value.value().as<StringImm>();
  ICHECK(name.defined()) << attr::kReducerType << " must be a StringImm";
  return ReduceType(std::string(name.value()->value));
}

bool GetBoolAnnotation(const Map<String, ObjectRef> &annotations,
                       const char *key) {
  Optional<ObjectRef> value = annotations.Get(key);
  if (!value.defined()) {
    return false;
  }
  if (Optional<Bool> boolean = value.value().as<Bool>()) {
    return boolean.value()->value;
  }
  if (Optional<IntImm> integer = value.value().as<IntImm>()) {
    return integer.value()->value != 0;
  }
  return false;
}

Optional<Array<ReducerPartialPlan>>
GetPartialPlans(const Map<String, ObjectRef> &annotations) {
  Optional<ObjectRef> value = annotations.Get(attr::kReducerPartialPlans);
  if (!value.defined()) {
    return std::nullopt;
  }
  return Downcast<Array<ReducerPartialPlan>>(value.value());
}

Array<PrimExpr> GetLogicalIndices(const Map<String, ObjectRef> &annotations) {
  Optional<ObjectRef> value = annotations.Get(attr::kReducerLogicalIndices);
  ICHECK(value.defined())
      << attr::kReducerLogicalIndices
      << " is required by a planned physical reducer update";
  return Downcast<Array<PrimExpr>>(value.value());
}

void CheckFullRegion(const BufferRegion &region, const char *op_name) {
  ICHECK_EQ(region->region.size(), region->buffer->shape.size());
  for (size_t i = 0; i < region->region.size(); ++i) {
    ICHECK(
        is_zero(region->region[i]->min) &&
        StructuralEqual()(region->region[i]->extent, region->buffer->shape[i]))
        << op_name << " requires the full reducer region";
  }
}

void CheckProjectedBuffer(const Buffer &buffer, const Fragment &layout,
                          const char *op_name) {
  ICHECK(StructuralEqual()(buffer->shape, layout->OutputShape()))
      << op_name << " expected projected partial storage shape "
      << layout->OutputShape() << ", got " << buffer->shape;
}

} // namespace

ReducerInfo::ReducerInfo(const String &op, Optional<PrimExpr> seed) {
  ObjectPtr<ReducerInfoNode> node = make_object<ReducerInfoNode>();
  node->combine_type = ReduceType(std::string(op));
  ICHECK(IsBuiltinCommutativeReduceType(node->combine_type));
  node->seed = std::move(seed);
  data_ = std::move(node);
}

ReducerPartialPlan::ReducerPartialPlan(bool canonical,
                                       Optional<Fragment> partial_layout,
                                       Array<Integer> step_extents,
                                       Array<Integer> step_scales) {
  ICHECK_EQ(step_extents.size(), step_scales.size());
  ICHECK(canonical || partial_layout.defined())
      << "A projected reducer partial plan requires a physical layout";
  ICHECK(!canonical || !partial_layout.defined())
      << "A canonical reducer partial plan must use full logical storage";
  ICHECK(!canonical || step_extents.empty())
      << "Canonical participant-wide reduction is derived from the execution "
         "scope, not stored as projected thread steps";
  for (size_t i = 0; i < step_extents.size(); ++i) {
    ICHECK_GT(step_extents[i]->value, 0);
    ICHECK_GT(step_scales[i]->value, 0);
  }
  ObjectPtr<ReducerPartialPlanNode> node =
      make_object<ReducerPartialPlanNode>();
  node->canonical = canonical;
  node->partial_layout = std::move(partial_layout);
  node->step_extents = std::move(step_extents);
  node->step_scales = std::move(step_scales);
  data_ = std::move(node);
}

ReducerInitOp::ReducerInitOp(Array<PrimExpr> args,
                             Map<String, ObjectRef> annotations) {
  ICHECK(!args.empty());
  std::vector<AccessRegion> accesses;
  Array<Buffer> partials;
  accesses.reserve(args.size());
  partials.reserve(args.size());
  for (const PrimExpr &arg : args) {
    AccessRegion access = NormalizeToAccessRegion(arg, kAccessWrite);
    CheckFullRegion(access.region, "T.reducer_init");
    partials.push_back(access.region->buffer);
    accesses.push_back(std::move(access));
  }

  ObjectPtr<ReducerInitOpNode> node = make_object<ReducerInitOpNode>();
  node->partials = std::move(partials);
  node->combine_type = GetPlannedReduceType(annotations);
  node->SetAccessRegions(std::move(accesses));
  data_ = std::move(node);
}

Stmt ReducerInitOpNode::Lower(const LowerArgs &lower_args,
                              arith::Analyzer *) const {
  ICHECK(combine_type.defined())
      << "ReducerInitOp must be planned before lowering";
  Array<Stmt> statements;
  statements.reserve(partials.size());
  for (size_t group = 0; group < partials.size(); ++group) {
    Buffer partial = GetLoweredBuffer(partials[group], lower_args);
    Array<Var> variables;
    Array<PrimExpr> indices;
    variables.reserve(partial->shape.size());
    indices.reserve(partial->shape.size());
    for (size_t i = 0; i < partial->shape.size(); ++i) {
      Var var("reducer_init_" + std::to_string(group) + "_" +
              std::to_string(i));
      variables.push_back(var);
      indices.push_back(var);
    }
    Stmt body = BufferStore(
        partial, MakeReduceIdentity(combine_type, partial->dtype), indices);
    for (int i = static_cast<int>(variables.size()) - 1; i >= 0; --i) {
      body = For(variables[i], 0, partial->shape[i], ForKind::kSerial, body);
    }
    statements.push_back(std::move(body));
  }
  return statements.size() == 1 ? statements[0] : SeqStmt(statements);
}

LayoutMap ReducerInitOpNode::InferLayout(const LayoutInferArgs &,
                                         InferLevel) const {
  return {};
}

TileOperator ReducerInitOpNode::Clone() const {
  return ReducerInitOp(make_object<ReducerInitOpNode>(*this));
}

ReducerUpdateOp::ReducerUpdateOp(Array<PrimExpr> args,
                                 Map<String, ObjectRef> annotations) {
  ICHECK_EQ(args.size(), 2U);
  AccessRegion reducer_access =
      NormalizeToAccessRegion(args[0], kAccessReadWrite);
  Array<PrimExpr> logical_indices;
  Array<PrimExpr> physical_indices;
  for (const Range &range : reducer_access.region->region) {
    ICHECK(is_one(range->extent))
        << "T.reducer_update requires exactly one physical partial element";
    physical_indices.push_back(range->min);
  }
  if (annotations.count(attr::kReducerLogicalIndices)) {
    logical_indices = GetLogicalIndices(annotations);
  } else {
    logical_indices = physical_indices;
  }

  ObjectPtr<ReducerUpdateOpNode> node = make_object<ReducerUpdateOpNode>();
  node->reducer = reducer_access.region->buffer;
  node->logical_indices = std::move(logical_indices);
  node->physical_indices = std::move(physical_indices);
  node->contribution = args[1];
  node->combine_type = GetPlannedReduceType(annotations);
  node->parallel_once =
      GetBoolAnnotation(annotations, attr::kReducerParallelOnce);
  node->partition_required =
      GetBoolAnnotation(annotations, attr::kReducerPartitionRequired);
  ICHECK(!(node->parallel_once && node->partition_required))
      << "A reducer update cannot be both canonicalized and replicated";
  node->SetAccessRegions({reducer_access});
  data_ = std::move(node);
}

Stmt ReducerUpdateOpNode::Lower(const LowerArgs &lower_args,
                                arith::Analyzer *) const {
  ICHECK(combine_type.defined())
      << "ReducerUpdateOp must be planned before lowering";
  Buffer partial = GetLoweredBuffer(reducer, lower_args);
  PrimExpr accumulator = BufferLoad(partial, physical_indices);
  Stmt update = BufferStore(
      partial, MakeReduceCombine(combine_type, accumulator, contribution),
      physical_indices);
  update = AttrStmt(combine_type, attr::kReducerUpdate, contribution,
                    std::move(update));
  if (partition_required) {
    update = AttrStmt(Integer(0), attr::kParallelPartitionRequired, Integer(1),
                      std::move(update));
  } else if (parallel_once) {
    update = AttrStmt(Integer(0), attr::kParallelMultiplicity, Integer(1),
                      std::move(update));
  }
  return update;
}

LayoutMap ReducerUpdateOpNode::InferLayout(const LayoutInferArgs &,
                                           InferLevel) const {
  return {};
}

TileOperator ReducerUpdateOpNode::Clone() const {
  return ReducerUpdateOp(make_object<ReducerUpdateOpNode>(*this));
}

FinalizeReducerOp::FinalizeReducerOp(Array<PrimExpr> args,
                                     Map<String, ObjectRef> annotations) {
  ICHECK_GE(args.size(), 2U);
  Optional<Array<ReducerPartialPlan>> planned = GetPartialPlans(annotations);
  size_t partial_count = planned.defined() ? planned.value().size() : 1U;
  ICHECK_EQ(args.size(), partial_count + 1);

  std::vector<AccessRegion> accesses;
  Array<Buffer> partials;
  accesses.reserve(args.size());
  partials.reserve(partial_count);
  for (size_t i = 0; i < partial_count; ++i) {
    AccessRegion partial_access = NormalizeToAccessRegion(args[i], kAccessRead);
    CheckFullRegion(partial_access.region, "T.finalize_reducer");
    partials.push_back(partial_access.region->buffer);
    accesses.push_back(std::move(partial_access));
  }
  AccessRegion destination_access =
      NormalizeToAccessRegion(args[partial_count], kAccessWrite);
  CheckFullRegion(destination_access.region, "T.finalize_reducer");
  accesses.push_back(destination_access);
  for (const Buffer &partial : partials) {
    ICHECK_EQ(partial->dtype, destination_access.region->buffer->dtype)
        << "T.finalize_reducer partial and destination dtypes must match";
  }
  if (planned.defined()) {
    for (size_t i = 0; i < partial_count; ++i) {
      const ReducerPartialPlan &plan = planned.value()[i];
      if (plan->partial_layout.defined()) {
        CheckProjectedBuffer(partials[i], plan->partial_layout.value(),
                             "T.finalize_reducer");
        ICHECK(StructuralEqual()(destination_access.region->buffer->shape,
                                 plan->partial_layout.value()->InputShape()))
            << "T.finalize_reducer destination shape must match the projected "
               "partial logical shape";
      } else {
        ICHECK(StructuralEqual()(destination_access.region->buffer->shape,
                                 partials[i]->shape))
            << "Canonical partial shape must match the logical destination";
      }
    }
  }

  ObjectPtr<FinalizeReducerOpNode> node = make_object<FinalizeReducerOpNode>();
  node->partials = std::move(partials);
  if (planned.defined()) {
    node->partial_plans = planned.value();
  }
  node->destination = destination_access.region->buffer;
  node->combine_type = GetPlannedReduceType(annotations);
  if (Optional<ObjectRef> seed = annotations.Get(attr::kReducerSeed)) {
    node->seed = Downcast<PrimExpr>(seed.value());
  }
  if (Optional<ObjectRef> batch = annotations.Get("batch")) {
    IntImm value = Downcast<IntImm>(batch.value());
    node->batch = static_cast<int>(value->value);
    ICHECK_GE(node->batch, 1);
  }
  node->SetAccessRegions(std::move(accesses));
  data_ = std::move(node);
}

Stmt FinalizeReducerOpNode::Lower(const LowerArgs &lower_args,
                                  arith::Analyzer *analyzer) const {
  ICHECK(combine_type.defined())
      << "FinalizeReducerOp must be planned before lowering";
  ICHECK_EQ(partials.size(), partial_plans.size())
      << "FinalizeReducerOp must carry one plan per physical partial";
  return ResolveFinalizeReducerImpl(lower_args.target)
      .lower(*this, lower_args, analyzer);
}

LayoutMap FinalizeReducerOpNode::InferLayout(const LayoutInferArgs &,
                                             InferLevel) const {
  return {};
}

TileOperator FinalizeReducerOpNode::Clone() const {
  return FinalizeReducerOp(make_object<FinalizeReducerOpNode>(*this));
}

void RegisterFinalizeReducerImpl(FinalizeReducerImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.lower != nullptr);
  FinalizeReducerImplRegistry().push_back(impl);
}

TIR_REGISTER_TL_TILE_OP(ReducerInitOp, reducer_init)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_TILE_OP(ReducerUpdateOp, reducer_update)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_TILE_OP(FinalizeReducerOp, finalize_reducer)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  ReducerInfoNode::RegisterReflection();
  ReducerPartialPlanNode::RegisterReflection();
  ReducerInitOpNode::RegisterReflection();
  ReducerUpdateOpNode::RegisterReflection();
  FinalizeReducerOpNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
