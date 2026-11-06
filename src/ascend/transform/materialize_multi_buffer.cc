/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*! \file materialize_multi_buffer.cc
 *  \brief Lower prepared logical multi-buffer clocks to physical buffer
 *         versions after synchronization insertion.
 */

#include "ascend/transform/source_scopes.h"
#include "auto_schedule/kernel_rewriter.h"
#include "auto_schedule/multi_buffer.h"
#include "auto_schedule/scheduled_tir.h"
#include "auto_schedule/storage_footprint.h"
#include "auto_schedule/task_analysis.h"
#include "ir/annotations.h"
#include "ir/stmt_functor.h"
#include "op/builtin.h"
#include "op/operator.h"
#include <tvm/ffi/extra/structural_visit.h>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace tvm::tl {
namespace {
using namespace tirx;
using namespace prim;
using namespace ascend;
using ffi::Array;
using ffi::Map;
using ffi::String;
using Variables = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

// A version is one copy of the physical allocation, including holes and the
// native memory-axis origin. Aliases retain their own byte offsets and layouts.
class VersionedTensorRegistry {
public:
  VersionedTensorRegistry(const MultiBufferPlan &plan, const RegionStmt &root)
      : plan_(plan), footprints_(root) {
    std::vector<TensorVar> tensors;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        root,
        [&](const ffi::ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
          if (auto variable = object.as<Var>()) {
            if (auto tensor = variable->as<TensorVar>())
              if (Info(*tensor))
                tensors.push_back(*tensor);
          }
          return ffi::WalkResult::Advance();
        });
    for (const auto &info : plan.Infos()) {
      if (info.NeedsVersionDimension())
        footprints_.PitchBits(info.storage);
    }
    for (const TensorVar &tensor : tensors)
      Resolve(tensor);
  }

  const MultiBufferInfo *Info(const TensorVar &tensor) const {
    const auto *info = plan_.Find(tensor, footprints_.Storage());
    return info && info->NeedsVersionDimension() ? info : nullptr;
  }

  const StorageProvenance &Storage() const { return footprints_.Storage(); }
  PrimExpr PitchBytes(const Var &storage) const {
    return FloorDiv(footprints_.PitchBits(storage), IntImm::Int64(8));
  }

  TensorVar Resolve(const TensorVar &tensor) {
    auto found = replacements_.find(tensor);
    if (found != replacements_.end())
      return found->second;
    const auto *info = Info(tensor);
    if (!info)
      return tensor;
    auto type = footprints_.VersionedType(tensor, info->num_versions);
    TensorVar replacement(tensor.CopyWithType(type));
    replacements_.emplace(tensor, replacement);
    return replacement;
  }

private:
  const MultiBufferPlan &plan_;
  StorageFootprints footprints_;
  std::unordered_map<TensorVar, TensorVar, ObjectPtrHash, ObjectPtrEqual>
      replacements_;
};

class MultiBufferRewriter : public LayoutMapMutator {
public:
  using LayoutMapMutator::Mutate_;
  MultiBufferRewriter(const MultiBufferPlan &plan,
                      VersionedTensorRegistry *tensors)
      : plan_(plan), tensors_(tensors) {}

  ffi::UnchangedOr<Expr> Mutate_(const VarObj *op, InplaceMode mode) final {
    if (auto tensor = ffi::GetRef<Var>(op).as<TensorVar>()) {
      TensorVar result = tensors_->Resolve(*tensor);
      if (!result.same_as(*tensor))
        return result;
    }
    return LayoutMapMutator::Mutate_(op, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    auto tensor = op->var.as<TensorVar>();
    if (!tensor || !tensors_->Info(*tensor))
      return LayoutMapMutator::Mutate_(op, mode);
    TensorVar replacement = tensors_->Resolve(*tensor);
    Expr value = op->value;
    std::optional<ScheduledBinding> scheduled;
    if (const auto *call = value.as<CallObj>(); IsScheduledBind(call)) {
      scheduled = ReadScheduledBinding(call);
      value = scheduled->value;
    }
    if (const auto *call = value.as<CallObj>();
        call && (call->op.same_as(alloc_tensor_op()) ||
                 call->op.same_as(decl_tensor_op()))) {
      Array<Expr> args;
      for (const auto &arg : call->args)
        args.push_back(Mutate(arg, mode).ValueOrUnchanged(arg));
      value = Call(replacement.ty(), call->op, args, call->attrs,
                   {replacement.ty()}, call->loc);
    } else {
      value = Mutate(value, mode).ValueOrUnchanged(value);
    }
    Bind result(replacement, value, op->loc);
    if (scheduled) {
      auto task = RewriteFields(scheduled->task);
      auto encoded = EncodeAnnotations(task);
      Expr payload = encoded.payload;
      payload = Mutate(payload, mode).ValueOrUnchanged(payload);
      auto decoded =
          DecodeAnnotations(payload.as_or_throw<tvm::Tuple>(), encoded.schema)
              .cast<ScheduleUnitMetadata>();
      result = MakeScheduledBinding(result, scheduled->stage, decoded);
    }
    return result;
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *op,
                                     InplaceMode mode) final {
    auto tensor = op->source.as_or_throw<TensorVar>();
    if (const auto *info = tensors_->Info(tensor))
      return TensorLoad(tensors_->Resolve(tensor),
                        Indices(*info, op->indices, mode), op->loc);
    return LayoutMapMutator::Mutate_(op, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreObj *op,
                                 InplaceMode mode) final {
    auto tensor = op->dest.as_or_throw<TensorVar>();
    if (const auto *info = tensors_->Info(tensor))
      return TensorStore(
          tensors_->Resolve(tensor), Indices(*info, op->indices, mode),
          Mutate(op->value, mode).ValueOrUnchanged(op->value), op->loc);
    return LayoutMapMutator::Mutate_(op, mode);
  }

  ffi::UnchangedOr<Expr> Mutate_(const TensorRegionObj *op,
                                 InplaceMode mode) final {
    auto tensor = op->source.as_or_throw<TensorVar>();
    const auto *info = tensors_->Info(tensor);
    if (!info)
      return LayoutMapMutator::Mutate_(op, mode);
    PrimExpr version = Version(*info);
    Array<Range> ranges{Range::FromMinExtent(
        version,
        IntImm(version.ty(),
               broadcast_.count(info->storage) ? info->num_versions : 1))};
    for (const auto &range : op->region)
      ranges.push_back(Range::FromMinExtent(
          Mutate(range->begin, mode).ValueOrUnchanged(range->begin),
          Mutate(range->extent, mode).ValueOrUnchanged(range->extent)));
    return TensorRegion(tensors_->Resolve(tensor), ranges, op->ty, op->loc);
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    if (op->op.same_as(masked_load_op()) || op->op.same_as(masked_store_op())) {
      auto tensor = op->args[0].as_or_throw<TensorVar>();
      if (const auto *info = tensors_->Info(tensor)) {
        Array<Expr> args{tensors_->Resolve(tensor)};
        size_t index_begin = 1;
        if (op->op.same_as(masked_store_op())) {
          args.push_back(
              Mutate(op->args[1], mode).ValueOrUnchanged(op->args[1]));
          index_begin = 2;
        }
        args.push_back(Version(*info));
        for (size_t i = index_begin; i < op->args.size(); ++i)
          args.push_back(
              Mutate(op->args[i], mode).ValueOrUnchanged(op->args[i]));
        return Call(op->ty, op->op, args, op->attrs, op->ty_args, op->loc);
      }
    }
    if (const auto *callee = op->op.as<OpObj>();
        callee && callee == tl::ptr_access().get()) {
      auto address = tensors_->Storage().Address(op->args[0]);
      if (auto root = address.root.as<Var>())
        if (const auto *info = plan_.Find(*root);
            info && info->NeedsVersionDimension()) {
          TVM_FFI_CHECK(address.exact, ValueError);
          Array<Expr> args;
          for (const auto &arg : op->args)
            args.push_back(Mutate(arg, mode).ValueOrUnchanged(arg));
          args.Set(0, Call(args[0]->ty, ptr_byte_offset_op(),
                           {args[0], Version(*info) *
                                         tensors_->PitchBytes(info->storage)}));
          return Call(op->ty, op->op, args, op->attrs, op->ty_args, op->loc);
        }
    }
    return LayoutMapMutator::Mutate_(op, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const EvaluateObj *op,
                                 InplaceMode mode) final {
    if (!IsTaskMetadataStatement(ffi::GetRef<Stmt>(op)))
      return LayoutMapMutator::Mutate_(op, mode);
    auto rewritten = MakeAnnotation(
        RewriteFields(ReadAnnotations(op->value.as<CallObj>())), op->loc);
    if (auto sequence = rewritten.as<SeqStmt>())
      return LayoutMapMutator::Mutate_(sequence->get(), mode)
          .ValueOrUnchanged(*sequence);
    return LayoutMapMutator::Mutate_(rewritten.as_or_throw<Evaluate>().get(),
                                     mode)
        .ValueOrUnchanged(rewritten);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    For loop = LayoutMapMutator::Mutate_(op, mode)
                   .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                   .as_or_throw<For>();
    // Older native producers can still attach storage references directly to
    // For.annotations. The generic native For hook deliberately leaves that
    // static map alone, so rewrite its actual operands explicitly here.
    auto encoded = EncodeAnnotations(RewriteFields(op->annotations));
    Expr payload = encoded.payload;
    payload = Mutate(payload, mode).ValueOrUnchanged(payload);
    loop.CopyOnWrite()->annotations =
        DecodeAnnotations(payload.as_or_throw<tvm::Tuple>(), encoded.schema)
            .cast<ScheduleUnitMetadata>();
    return loop;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    if (IsAscendBlock(op)) {
      auto rewritten = MakeAscendBlock(
          AscendBlockName(op), RewriteFields(AscendBlockAnnotations(op)),
          op->body, op->loc);
      return LayoutMapMutator::Mutate_(rewritten.get(), mode)
          .ValueOrUnchanged(Stmt(rewritten));
    }
    if (IsSemanticTask(op)) {
      auto old_broadcast = broadcast_;
      auto fields = TaskAnnotations(op);
      if (auto fill = fields.Get(kMultiBufferBroadcastFill))
        for (const auto &storage : fill->cast<MultiBufferBroadcastFill>())
          broadcast_.insert(storage);
      auto rewritten = WithTaskAnnotations(ffi::GetRef<RegionStmt>(op),
                                           RewriteFields(fields));
      auto result = LayoutMapMutator::Mutate_(rewritten.get(), mode)
                        .ValueOrUnchanged(Stmt(rewritten));
      broadcast_ = std::move(old_broadcast);
      return result;
    }
    auto attribute = ReadSourceAttribute(op);
    if (!attribute)
      return LayoutMapMutator::Mutate_(op, mode);
    auto old_broadcast = broadcast_;
    Any node = attribute->node;
    Any value = attribute->value;
    if (attribute->key == "tl.loop.annotations" ||
        attribute->key == "tl.kernel") {
      value = RewriteFields(value.cast<ScheduleUnitMetadata>());
    } else if (attribute->key == "tl.sblock") {
      auto fields = value.cast<ScheduleUnitMetadata>();
      if (auto annotations = fields.Get("annotations"))
        fields.Set("annotations",
                   RewriteFields(annotations->cast<ScheduleUnitMetadata>()));
      value = fields;
    } else if (attribute->key == "layout_map") {
      value = RewriteFields({{"layout_map", value}}).at("layout_map");
    }
    auto rewritten =
        MakeAttribute(node, attribute->key, value, op->body, op->loc);
    auto result = LayoutMapMutator::Mutate_(rewritten.get(), mode)
                      .ValueOrUnchanged(Stmt(rewritten));
    broadcast_ = std::move(old_broadcast);
    return result;
  }

private:
  PrimExpr Version(const MultiBufferInfo &info) const {
    if (broadcast_.count(info.storage))
      return IntImm::Int64(0);
    PrimExpr iteration =
        info.UsesCounter()
            ? PrimExpr(TensorLoad(info.counter, {IntImm::Int32(0)}))
            : CalculateIterationCount(info.owners.front().loop);
    return FloorMod(iteration, IntImm(iteration.ty(), info.num_versions));
  }

  Array<PrimExpr> Indices(const MultiBufferInfo &info,
                          const Array<PrimExpr> &indices, InplaceMode mode) {
    Array<PrimExpr> result{Version(info)};
    for (const auto &index : indices)
      result.push_back(Mutate(index, mode).ValueOrUnchanged(index));
    return result;
  }

  ScheduleUnitMetadata RewriteFields(ScheduleUnitMetadata fields) {
    fields.erase(kMultiBufferCounterMap);
    fields.erase(kStorageEpochGuardMap);
    fields.erase(kMultiBufferBroadcastFill);
    if (auto value = fields.Get("layout_map")) {
      LayoutMap result;
      for (const auto &[tensor, layout] : value->cast<LayoutMap>()) {
        auto replacement = tensors_->Resolve(tensor);
        result.Set(replacement, replacement.same_as(tensor)
                                    ? layout
                                    : layout->Expand({replacement.shape()[0]}));
      }
      fields.Set("layout_map", result);
    }
    return fields;
  }

  const MultiBufferPlan &plan_;
  VersionedTensorRegistry *tensors_;
  Variables broadcast_;
};

void RemoveCounterTasks(IRStructure::ChildList *nodes, IRStructure *parent,
                        const Variables &counters) {
  IRStructure::ChildList retained;
  for (const auto &node : *nodes) {
    if (node->IsTask()) {
      auto *task = static_cast<TaskNode *>(node.get());
      if (auto bind = GetFlatTaskBind(task->stmt);
          bind && counters.count(bind->var)) {
        const auto *call = bind->value.as<CallObj>();
        TVM_FFI_CHECK(call && call->op.same_as(alloc_tensor_op()), ValueError)
            << "Single-version counter must have an allocation definition";
        continue;
      }
      bool touches = false;
      for (const auto &counter : counters)
        touches |= task->TouchesStorage(counter);
      if (touches) {
        Stmt body = SingleStatement(task->stmt);
        if (const auto *marker = body.as<RegionStmtObj>();
            IsSemanticTask(marker))
          body = SingleStatement(marker->body);
        while (const auto *guard = body.as<IfObj>()) {
          if (guard->else_case)
            break;
          body = SingleStatement(guard->then_case);
        }
        const auto *store = body.as<TensorStoreObj>();
        TVM_FFI_CHECK(store && counters.count(store->dest.as_or_throw<Var>()),
                      ValueError)
            << "Single-version counter is read by a task that must survive";
        continue;
      }
    } else {
      if (auto *header = node->HeaderTask())
        for (const auto &counter : counters)
          TVM_FFI_CHECK(!header->TouchesStorage(counter), ValueError)
              << "Single-version counter is used by a control header";
      for (auto *list : node->ChildLists())
        RemoveCounterTasks(list, node.get(), counters);
    }
    node->SetParent(parent);
    node->SetIndex(retained.size());
    retained.push_back(node);
  }
  *nodes = std::move(retained);
}

void ElideUnitVersionCounterGroups(ScheduledTIR *scheduled,
                                   const MultiBufferPlan &plan) {
  std::unordered_set<int> retained;
  for (const auto &info : plan.Infos())
    if (info.UsesCounter() && info.NeedsVersionDimension())
      retained.insert(info.counter_group_id);
  Variables counters;
  for (const auto &info : plan.Infos())
    if (info.UsesCounter() && !retained.count(info.counter_group_id))
      counters.insert(info.CounterStorage());
  if (!counters.empty())
    RemoveCounterTasks(&scheduled->tree, nullptr, counters);
}

RegionStmt MaterializeKernel(ScheduledTIR scheduled) {
  auto plan = ReadMultiBufferPlan(
      scheduled.tree, scheduled.metadata.buffer_versions,
      L0StorageGroups(CollectL0SFBindings(scheduled.metadata.kernel_root)));
  ElideUnitVersionCounterGroups(&scheduled, plan);
  VersionedTensorRegistry tensors(plan, scheduled.metadata.kernel_root);
  auto physical = scheduled.metadata.manual_buffer_versions;
  for (const auto &info : plan.Infos()) {
    if (!info.NeedsVersionDimension())
      continue;
    if (auto previous = physical.Get(info.storage))
      TVM_FFI_CHECK_EQ(*previous, info.num_versions, ValueError);
    physical.Set(info.storage, info.num_versions);
  }
  scheduled.metadata.buffer_versions = {};
  scheduled.metadata.buffer_version_modes = {};
  scheduled.metadata.manual_buffer_versions = std::move(physical);
  // The prepared plan refers to current loop nodes until all access clocks
  // have been materialized into native expressions.
  auto retained_tree = scheduled.tree;
  Stmt root = EncodeScheduledTIR(std::move(scheduled));
  auto rewriter = ffi::make_object<MultiBufferRewriter>(plan, &tensors);
  return rewriter->Mutate(root)
      .ValueOrUnchanged(root)
      .as_or_throw<RegionStmt>();
}

} // namespace

tvm::transform::Pass MaterializeMultiBuffer() {
  auto pass = [](tirx::Function function, const IRModule &,
                 const tvm::transform::PassContext &) {
    return ascend::RewriteTilelangKernels(
        std::move(function), "MaterializeMultiBuffer",
        [](const ascend::TilelangKernelContext &context) {
          return MaterializeKernel(ascend::DecodeScheduledTIR(
              context.root, context.outer_ctx, nullptr, {},
              context.parameter_alias_contract));
        });
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.MaterializeMultiBuffer");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.MaterializeMultiBuffer",
                                   MaterializeMultiBuffer);
}
} // namespace tvm::tl
