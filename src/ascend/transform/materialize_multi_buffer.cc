#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "op/prim.h"
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

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "./auto_schedule/kernel_rewriter.h"
#include "./auto_schedule/multi_buffer.h"
#include "./auto_schedule/scheduled_tir.h"
#include "layout/layout.h"
#include "transform/common/storage_size.h"

namespace tvm::tl {

using namespace tirx;
using namespace ascend;

namespace {

class VersionedBufferRegistry {
public:
  VersionedBufferRegistry(const MultiBufferPlan &plan,
                          const std::vector<std::shared_ptr<IRStructure>> &tree)
      : analysis_(tree.empty() ? nullptr : tree.front()->GetStorageAnalysis()) {
    for (const MultiBufferInfo &info : plan.Infos()) {
      if (!info.NeedsVersionDimension())
        continue;
      ICHECK(analysis_);
      auto allocation = analysis_->allocations.Get(info.storage);
      ICHECK(allocation.has_value())
          << "Cannot find the physical allocation for automatic multi-buffer "
          << info.storage;
      storage_versions_.emplace(info.storage, info.num_versions);
      storage_pitch_bits_.emplace(
          info.storage, GetBufferStorageSizeBytes(allocation.value()) * 8);
    }
  }

  ffi::Optional<Var> Storage(const Expr &expression) const {
    if (auto variable = expression.as<Var>()) {
      auto expanded = expanded_storage_.find(variable.value());
      if (expanded != expanded_storage_.end())
        return expanded->second;
    }
    auto access = analysis_ ? analysis_->Resolve(expression) : std::nullopt;
    if (access)
      return access->root;
    return std::nullopt;
  }

  const PrimExpr &PitchBits(const Var &storage) const {
    return storage_pitch_bits_.at(storage);
  }

  TensorVar Resolve(const TensorVar &tensor) {
    if (expanded_storage_.count(tensor))
      return tensor;
    auto exact = replacements_.Get(tensor);
    if (exact)
      return exact.value().as_or_throw<TensorVar>();
    auto storage = Storage(tensor);
    auto versions = storage ? storage_versions_.find(storage.value())
                            : storage_versions_.end();
    if (versions == storage_versions_.end())
      return tensor;
    PrimExpr bits = PitchBits(storage.value());
    PrimExpr width = IntImm(bits.ty(), ElementBits(tensor.dtype()));
    sym::Analyzer analyzer;
    ICHECK(tl::arith::CanProveEqual(analyzer, FloorMod(bits, width), 0))
        << "Multi-buffer pitch is not divisible by alias element width: "
        << tensor;
    PrimExpr pitch = tl::arith::Simplify(analyzer, FloorDiv(bits, width));
    PrimExpr count = IntImm(PrimType::Int(32), versions->second);
    ffi::Array<PrimExpr> shape{count};
    for (const PrimExpr &extent : tensor.shape())
      shape.push_back(extent);
    auto layout = tensor.ty()->layout.value_or(
        tirx::TileLayout::DefaultLayout(tensor.shape()));
    auto tiled = layout.as<tirx::TileLayout>();
    ICHECK(tiled.has_value())
        << "Ascend multi-buffer storage requires a native tiled memory layout";
    ffi::Array<tirx::Iter> dimensions{
        tirx::Iter(count, pitch, tirx::Axis::Get("m"))};
    for (const tirx::Iter &dimension : tiled.value()->shard)
      dimensions.push_back(dimension);
    auto expanded_layout = tirx::TileLayout(dimensions, tiled.value()->replica,
                                            tiled.value()->offset);
    auto type =
        TensorType(shape, tensor.dtype(), tensor.scope(),
                   tensor.ty()->data_alignment, expanded_layout, tensor->loc);
    auto expanded = tensor.CopyWithType(type).as_or_throw<TensorVar>();
    replacements_.Set(tensor, expanded);
    expanded_storage_.emplace(expanded, storage.value());
    return expanded;
  }

  ffi::Any RewriteMetadata(const ffi::Any &value) {
    class Rewriter : public NativeMutator {
    public:
      explicit Rewriter(VersionedBufferRegistry *registry)
          : registry_(registry) {}
      Expr VisitExpr_(const VarNode *node) final {
        if (auto tensor = ffi::GetRef<Var>(node).as<TensorVar>())
          return registry_->Resolve(tensor.value());
        return NativeMutator::VisitExpr_(node);
      }

    private:
      VersionedBufferRegistry *registry_;
    };
    auto rewriter = ffi::make_object<Rewriter>(this);
    DictAttrs wrapped({{"value", value}});
    auto rewritten = rewriter->Mutate(wrapped)
                         .ValueOrUnchanged(wrapped)
                         .as_or_throw<DictAttrs>();
    return rewritten->dict.at("value");
  }

  DictAttrs RewriteAttributes(const DictAttrs &attributes) {
    auto rewritten = RewriteMetadata(attributes).as_or_throw<DictAttrs>();
    if (auto layouts = attributes.GetAttr<ffi::Map<TensorVar, tl::Layout>>(
            attr::kLayoutMap)) {
      ffi::Map<TensorVar, tl::Layout> expanded;
      for (const auto &[tensor, layout] : layouts.value()) {
        TensorVar target = Resolve(tensor);
        expanded.Set(target, target.same_as(tensor)
                                 ? layout
                                 : layout->Expand({target.shape()[0]}));
      }
      rewritten.CopyOnWrite()->dict.Set(attr::kLayoutMap, expanded);
    }
    return rewritten;
  }

private:
  const StorageAliasAnalysis *analysis_;
  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> storage_versions_;
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>
      storage_pitch_bits_;
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> expanded_storage_;
  ffi::Map<Var, Var> replacements_;
};

class MultiBufferAccessRewriter : public NativeMutator {
public:
  MultiBufferAccessRewriter(const MultiBufferPlan &plan,
                            VersionedBufferRegistry *buffers,
                            const MultiBufferBroadcastFill &broadcast_storages)
      : plan_(plan), buffers_(buffers) {
    for (const Var &storage : broadcast_storages) {
      const MultiBufferInfo *info = plan_.Find(storage);
      ICHECK(info) << "Broadcast initialization has no multi-buffer plan";
      if (info->NeedsVersionDimension())
        broadcast_storages_.insert(storage);
    }
  }

  ffi::Any RewriteAttributeNode(const ffi::String &key, const ffi::Any &node) {
    if (CanRewriteMultiBufferAttrNode(key, node))
      return VisitExpr(node.as_or_throw<PrimExpr>());
    return buffers_->RewriteMetadata(node);
  }

  PrimExpr RewriteGuardExpression(const PrimExpr &expression) {
    auto rewritten = VisitExpr(expression);
    ICHECK(pending_views_.empty())
        << "Tensor-valued calls in multi-buffer guards need explicit bindings";
    return rewritten;
  }

private:
  const MultiBufferInfo *Find(const Expr &expression) const {
    auto storage = buffers_->Storage(expression);
    const MultiBufferInfo *info =
        storage ? plan_.Find(storage.value()) : nullptr;
    return info && info->NeedsVersionDimension() ? info : nullptr;
  }

  PrimExpr VersionIndex(const MultiBufferInfo &info) const {
    if (broadcast_storages_.count(info.storage))
      return IntImm::Int32(0);
    PrimExpr iteration =
        info.UsesCounter()
            ? PrimExpr(TensorLoad(info.counter, {IntImm::Int32(0)}))
            : CalculateIterationCount(info.owners.front().loop);
    return tl::CastValue(
        FloorMod(iteration, IntImm(iteration.ty(), info.num_versions)),
        PrimType::Int(32));
  }

  Expr OffsetPointer(const Expr &pointer, const MultiBufferInfo &info) {
    PrimExpr bytes =
        FloorDiv(buffers_->PitchBits(info.storage), IntImm::Int64(8));
    PrimExpr offset = StorageIndex(VersionIndex(info)) * bytes;
    return Call(pointer->ty, tirx::ptr_byte_offset_op(), {pointer, offset});
  }

  Expr SlotView(const TensorVar &tensor, const MultiBufferInfo &info) {
    TensorVar expanded = buffers_->Resolve(tensor);
    Var view(tensor->name + "_slot", tensor.ty(), tensor->loc);
    Expr pointer = OffsetPointer(expanded.data(), info);
    pending_views_.push_back(Bind(
        view,
        Call(tensor.ty(), tirx::decl_tensor_op(), {pointer}, {}, {tensor.ty()}),
        tensor->loc));
    return view;
  }

  Stmt WithViews(Stmt statement) {
    if (pending_views_.empty())
      return statement;
    ffi::Array<Stmt> statements(pending_views_.begin(), pending_views_.end());
    pending_views_.clear();
    statements.push_back(statement);
    return SeqStmt(statements);
  }

  Expr VisitExpr_(const VarNode *node) final {
    Var variable = ffi::GetRef<Var>(node);
    if (auto tensor = variable.as<TensorVar>())
      return buffers_->Resolve(tensor.value());
    Expr rewritten = NativeMutator::VisitExpr_(node);
    if (!definition_ && node->ty.as<PtrTypeNode>())
      if (const MultiBufferInfo *info = Find(variable))
        return OffsetPointer(rewritten, *info);
    return rewritten;
  }

  Stmt VisitStmt_(const BindNode *node) final {
    bool previous = definition_;
    if (buffers_->Storage(node->var))
      definition_ = true;
    auto tensor = node->var.as<TensorVar>();
    const auto *call = node->value.as<CallNode>();
    if (tensor && call &&
        (call->op.same_as(tirx::alloc_tensor_op()) ||
         call->op.same_as(tirx::decl_tensor_op()))) {
      TensorVar variable = buffers_->Resolve(tensor.value());
      auto arguments = Mutate(call->args)
                           .ValueOrUnchanged(call->args)
                           .as_or_throw<ffi::Array<Expr>>();
      auto value = Call(variable.ty(), call->op, arguments, call->attrs,
                        {variable.ty()}, call->loc);
      VarRemapSet(node->var, variable);
      definition_ = previous;
      return WithViews(Bind(variable, value, node->loc));
    }
    Stmt rewritten = NativeMutator::VisitStmt_(node);
    definition_ = previous;
    return WithViews(rewritten);
  }

  PrimExpr VisitExpr_(const TensorLoadNode *node) final {
    const MultiBufferInfo *info = Find(node->source);
    if (!info)
      return NativeMutator::VisitExpr_(node);
    ffi::Array<PrimExpr> indices{VersionIndex(*info)};
    for (const PrimExpr &index : node->indices)
      indices.push_back(VisitExpr(index));
    return TensorLoad(buffers_->Resolve(node->source.as_or_throw<TensorVar>()),
                      indices, node->loc);
  }

  Stmt VisitStmt_(const TensorStoreNode *node) final {
    const MultiBufferInfo *info = Find(node->dest);
    if (!info)
      return WithViews(NativeMutator::VisitStmt_(node));
    ffi::Array<PrimExpr> indices{VersionIndex(*info)};
    for (const PrimExpr &index : node->indices)
      indices.push_back(VisitExpr(index));
    return WithViews(
        TensorStore(buffers_->Resolve(node->dest.as_or_throw<TensorVar>()),
                    indices, VisitExpr(node->value), node->loc));
  }

  ffi::UnchangedOr<Expr> Mutate_(const TensorRegionNode *node,
                                 ffi::InplaceMode mode) final {
    const MultiBufferInfo *info = Find(node->source);
    if (!info)
      return tvm::tl::StmtExprMutator::Mutate_(node, mode);
    int extent =
        broadcast_storages_.count(info->storage) ? info->num_versions : 1;
    ffi::Array<Range> ranges{
        Range::FromMinExtent(VersionIndex(*info), IntImm::Int32(extent))};
    for (const Range &range : node->region)
      ranges.push_back(Range::FromMinExtent(VisitExpr(range->begin),
                                            VisitExpr(range->extent)));
    return NativeTensorRegion(
        buffers_->Resolve(node->source.as_or_throw<TensorVar>()), ranges);
  }

  Expr VisitExpr_(const CallNode *node) final {
    if (node->op.same_as(Op::Get("tl.layout_hint"))) {
      auto tensor = node->args[0].as_or_throw<TensorVar>();
      TensorVar expanded = buffers_->Resolve(tensor);
      auto attributes = node->attrs.value().as_or_throw<DictAttrs>();
      auto layout = attributes.GetAttr<tl::Layout>("layout").value();
      if (!expanded.same_as(tensor))
        attributes.CopyOnWrite()->dict.Set(
            "layout", layout->Expand({expanded.shape()[0]}));
      return Call(node->ty, node->op, {expanded}, attributes, node->ty_args,
                  node->loc);
    }
    if (node->op.same_as(Op::Get("tl.safe_value_hint"))) {
      auto tensor = node->args[0].as_or_throw<TensorVar>();
      return Call(node->ty, node->op,
                  {buffers_->Resolve(tensor), VisitExpr(node->args[1])},
                  node->attrs, node->ty_args, node->loc);
    }
    const MultiBufferInfo *info = Find(ffi::GetRef<Expr>(node));
    if (!definition_ && info && node->ty.as<PtrTypeNode>()) {
      definition_ = true;
      Expr base = NativeMutator::VisitExpr_(node);
      definition_ = false;
      return OffsetPointer(base, *info);
    }
    if (definition_)
      return NativeMutator::VisitExpr_(node);
    ffi::Array<Expr> arguments;
    for (const Expr &argument : node->args) {
      const MultiBufferInfo *argument_info = Find(argument);
      if (auto tensor = argument.as<TensorVar>(); tensor && argument_info)
        arguments.push_back(SlotView(tensor.value(), *argument_info));
      else
        arguments.push_back(VisitExpr(argument));
    }
    return Call(node->ty, node->op, arguments, node->attrs, node->ty_args,
                node->loc);
  }

  Stmt VisitStmt_(const EvaluateNode *node) final {
    return WithViews(NativeMutator::VisitStmt_(node));
  }

  Stmt VisitStmt_(const RegionStmtNode *node) final {
    auto rewritten = NativeMutator::VisitStmt_(node).as_or_throw<RegionStmt>();
    rewritten.CopyOnWrite()->attrs =
        buffers_->RewriteAttributes(rewritten->attrs);
    return rewritten;
  }

  const MultiBufferPlan &plan_;
  VersionedBufferRegistry *buffers_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> broadcast_storages_;
  std::vector<Stmt> pending_views_;
  bool definition_{false};
};

void RewriteGuards(IRStructure *node, MultiBufferAccessRewriter *rewriter) {
  GuardList guards;
  for (const auto &guard : node->GetGuards()) {
    if (guard->IsCondition()) {
      const auto *condition = static_cast<const ConditionGuard *>(guard.get());
      guards.push_back(std::make_unique<ConditionGuard>(
          rewriter->RewriteGuardExpression(condition->condition)));
    } else {
      const auto *attribute = static_cast<const AttributeGuard *>(guard.get());
      guards.push_back(std::make_unique<AttributeGuard>(
          rewriter->RewriteAttributeNode(attribute->key, attribute->node),
          attribute->key, attribute->loc));
    }
  }
  node->SetGuards(std::move(guards));
}

void RewriteScheduledTree(std::vector<std::shared_ptr<IRStructure>> *nodes,
                          const MultiBufferPlan &plan,
                          VersionedBufferRegistry *buffers) {
  for (const auto &node : *nodes) {
    MultiBufferBroadcastFill broadcast;
    if (node->IsTask())
      broadcast =
          GetMultiBufferBroadcastFills(static_cast<TaskNode *>(node.get()));
    auto rewriter =
        ffi::make_object<MultiBufferAccessRewriter>(plan, buffers, broadcast);
    RewriteGuards(node.get(), rewriter.get());
    if (node->IsTask()) {
      auto *task = static_cast<TaskNode *>(node.get());
      task->stmt = rewriter->VisitStmt(task->stmt);
      if (!broadcast.empty())
        ClearMultiBufferBroadcastFills(task);
      continue;
    }
    auto *control = static_cast<ControlNode *>(node.get());
    RewriteScheduledTree(&control->children, plan, buffers);
    For loop = control->control;
    auto *rewritten = loop.CopyOnWrite();
    rewritten->begin = rewriter->RewriteGuardExpression(rewritten->begin);
    rewritten->end = rewriter->RewriteGuardExpression(rewritten->end);
    rewritten->step = rewriter->RewriteGuardExpression(rewritten->step);
    rewritten->annotations.erase(kMultiBufferCounterMap);
    rewritten->annotations.erase(kStorageEpochGuardMap);
    rewritten->annotations =
        buffers->RewriteMetadata(rewritten->annotations)
            .as_or_throw<ffi::Map<ffi::String, ffi::Any>>();
    control->control = std::move(loop);
  }
}

void AddBufferVersion(BufferVersionMap *versions, const Var &storage,
                      int count) {
  if (count <= 1)
    return;
  if (auto existing = versions->Get(storage)) {
    ICHECK_EQ(existing.value(), count)
        << "Conflicting physical buffer versions";
  } else {
    versions->Set(storage, count);
  }
}

void RemoveCounterTasks(
    std::vector<std::shared_ptr<IRStructure>> *nodes, ControlNode *parent,
    const std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> &counters) {
  std::vector<std::shared_ptr<IRStructure>> retained;
  for (const auto &node : *nodes) {
    if (node->IsTask()) {
      auto *task = static_cast<TaskNode *>(node.get());
      auto binding = GetFlatTaskBind(task->stmt);
      if (binding && counters.count(binding.value()->var)) {
        const auto *call = binding.value()->value.as<CallNode>();
        ICHECK(call && call->op.same_as(tirx::alloc_tensor_op()));
        continue;
      }
      auto counter = std::find_if(
          counters.begin(), counters.end(),
          [&](const Var &storage) { return task->TouchesStorage(storage); });
      if (counter != counters.end()) {
        Stmt body = task->stmt;
        while (true) {
          body = SingleScheduleStatement(body);
          if (const auto *region = body.as<RegionStmtNode>();
              region && !ScheduleMarkerKey(region).empty()) {
            body = UnwrapScheduleMarker(region);
          } else if (const auto *condition = body.as<IfNode>();
                     condition && !condition->else_case.has_value()) {
            body = condition->then_case;
          } else {
            break;
          }
        }
        const auto *store = body.as<TensorStoreNode>();
        ICHECK(store && store->dest.same_as(*counter))
            << "Cannot elide single-version counter " << (*counter)->name
            << ": counter task contains additional work";
        ICHECK(SideEffect(store->value) <= CallEffectKind::kReadState)
            << "Cannot elide single-version counter " << (*counter)->name
            << ": counter update contains side effects";
        continue;
      }
    } else {
      auto *control = static_cast<ControlNode *>(node.get());
      for (const Var &counter : counters)
        ICHECK(!control->task->TouchesStorage(counter))
            << "Cannot elide single-version counter " << counter->name
            << ": counter is used in loop control";
      RemoveCounterTasks(&control->children, control, counters);
    }
    node->SetIndex(retained.size());
    node->SetParent(parent);
    retained.push_back(node);
  }
  *nodes = std::move(retained);
}

RegionStmt MaterializeKernel(ScheduledTIR scheduled) {
  MultiBufferPlan plan = ReadMultiBufferPlan(
      scheduled.tree, scheduled.metadata.buffer_versions,
      L0StorageGroups(CollectL0SFBindings(scheduled.metadata.kernel_root)));
  std::unordered_set<int> versioned_groups;
  for (const MultiBufferInfo &info : plan.Infos())
    if (info.UsesCounter() && info.NeedsVersionDimension())
      versioned_groups.insert(info.counter_group_id);
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> counters;
  for (const MultiBufferInfo &info : plan.Infos())
    if (info.UsesCounter() && !versioned_groups.count(info.counter_group_id))
      counters.insert(info.counter);
  if (!counters.empty())
    RemoveCounterTasks(&scheduled.tree, nullptr, counters);
  VersionedBufferRegistry buffers(plan, scheduled.tree);
  RewriteScheduledTree(&scheduled.tree, plan, &buffers);
  BufferVersionMap physical_versions;
  for (const auto &[storage, count] : scheduled.metadata.manual_buffer_versions)
    AddBufferVersion(&physical_versions, storage, count);
  for (const MultiBufferInfo &info : plan.Infos())
    AddBufferVersion(&physical_versions, info.storage, info.num_versions);
  scheduled.metadata.buffer_versions = {};
  scheduled.metadata.manual_buffer_versions =
      buffers.RewriteMetadata(physical_versions)
          .as_or_throw<BufferVersionMap>();
  scheduled.metadata.buffer_aliases =
      buffers.RewriteMetadata(scheduled.metadata.buffer_aliases)
          .as_or_throw<BufferAliasMap>();
  scheduled.metadata.root_conflict_hints =
      buffers.RewriteMetadata(scheduled.metadata.root_conflict_hints)
          .as_or_throw<ffi::Array<ffi::Any>>();
  auto root = scheduled.metadata.kernel_root;
  root.CopyOnWrite()->attrs = buffers.RewriteAttributes(root->attrs);
  scheduled.metadata.kernel_root = root;
  return EncodeScheduledTIR(std::move(scheduled));
}

} // namespace

tvm::transform::Pass MaterializeMultiBuffer() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    return RewriteTilelangKernels(std::move(function), "MaterializeMultiBuffer",
                                  [](const TilelangKernelContext &context) {
                                    return MaterializeKernel(DecodeScheduledTIR(
                                        context.root, context.outer_ctx));
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
