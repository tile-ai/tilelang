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

void CheckFullRegion(const BufferRegion &region, const char *op_name) {
  ICHECK_EQ(region->region.size(), region->buffer->shape.size());
  for (size_t i = 0; i < region->region.size(); ++i) {
    ICHECK(
        is_zero(region->region[i]->min) &&
        StructuralEqual()(region->region[i]->extent, region->buffer->shape[i]))
        << op_name << " requires the full reducer region";
  }
}

} // namespace

ReducerInfo::ReducerInfo(const String &op, Optional<PrimExpr> seed) {
  ObjectPtr<ReducerInfoNode> node = make_object<ReducerInfoNode>();
  node->combine_type = ReduceType(std::string(op));
  ICHECK(IsBuiltinCommutativeReduceType(node->combine_type));
  node->seed = std::move(seed);
  data_ = std::move(node);
}

ReducerInitOp::ReducerInitOp(Array<PrimExpr> args,
                             Map<String, ObjectRef> annotations) {
  ICHECK_EQ(args.size(), 1U);
  AccessRegion reducer_access = NormalizeToAccessRegion(args[0], kAccessWrite);
  CheckFullRegion(reducer_access.region, "T.reducer_init");

  ObjectPtr<ReducerInitOpNode> node = make_object<ReducerInitOpNode>();
  node->reducer = reducer_access.region->buffer;
  node->combine_type = GetPlannedReduceType(annotations);
  node->SetAccessRegions({reducer_access});
  data_ = std::move(node);
}

Stmt ReducerInitOpNode::Lower(const LowerArgs &lower_args,
                              arith::Analyzer *) const {
  ICHECK(combine_type.defined())
      << "ReducerInitOp must be planned before lowering";
  Buffer partial = GetLoweredBuffer(reducer, lower_args);
  Array<PrimExpr> indices;
  indices.reserve(partial->shape.size());
  Stmt body;
  for (size_t i = 0; i < partial->shape.size(); ++i) {
    indices.push_back(Var("reducer_init_" + std::to_string(i)));
  }
  body = BufferStore(partial, MakeReduceIdentity(combine_type, partial->dtype),
                     indices);
  for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
    body = For(Downcast<Var>(indices[i]), 0, partial->shape[i],
               ForKind::kSerial, body);
  }
  return body;
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
  for (const Range &range : reducer_access.region->region) {
    ICHECK(is_one(range->extent))
        << "T.reducer_update requires exactly one logical output element";
    logical_indices.push_back(range->min);
  }

  ObjectPtr<ReducerUpdateOpNode> node = make_object<ReducerUpdateOpNode>();
  node->reducer = reducer_access.region->buffer;
  node->logical_indices = std::move(logical_indices);
  node->contribution = args[1];
  node->combine_type = GetPlannedReduceType(annotations);
  node->parallel_once =
      GetBoolAnnotation(annotations, attr::kReducerParallelOnce);
  node->SetAccessRegions({reducer_access});
  data_ = std::move(node);
}

Stmt ReducerUpdateOpNode::Lower(const LowerArgs &lower_args,
                                arith::Analyzer *) const {
  ICHECK(combine_type.defined())
      << "ReducerUpdateOp must be planned before lowering";
  Buffer partial = GetLoweredBuffer(reducer, lower_args);
  PrimExpr accumulator = BufferLoad(partial, logical_indices);
  Stmt update = BufferStore(
      partial, MakeReduceCombine(combine_type, accumulator, contribution),
      logical_indices);
  if (parallel_once) {
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
  ICHECK_EQ(args.size(), 2U);
  AccessRegion reducer_access = NormalizeToAccessRegion(args[0], kAccessRead);
  AccessRegion destination_access =
      NormalizeToAccessRegion(args[1], kAccessWrite);
  CheckFullRegion(reducer_access.region, "T.finalize_reducer");
  CheckFullRegion(destination_access.region, "T.finalize_reducer");
  ICHECK_EQ(reducer_access.region->buffer->dtype,
            destination_access.region->buffer->dtype)
      << "T.finalize_reducer reducer and destination dtypes must match";

  ObjectPtr<FinalizeReducerOpNode> node = make_object<FinalizeReducerOpNode>();
  node->reducer = reducer_access.region->buffer;
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
  node->SetAccessRegions({reducer_access, destination_access});
  data_ = std::move(node);
}

Stmt FinalizeReducerOpNode::Lower(const LowerArgs &lower_args,
                                  arith::Analyzer *analyzer) const {
  ICHECK(combine_type.defined())
      << "FinalizeReducerOp must be planned before lowering";
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
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_TILE_OP(ReducerUpdateOp, reducer_update)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_TILE_OP(FinalizeReducerOp, finalize_reducer)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  ReducerInfoNode::RegisterReflection();
  ReducerInitOpNode::RegisterReflection();
  ReducerUpdateOpNode::RegisterReflection();
  FinalizeReducerOpNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
