#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "compat/region.h"
#include "op/prim.h"
#include "transform/common/loop_aware_mutator.h"

#include "common/pipeline_async.h"
#include "common/pipeline_replay.h"
#include "common/pipeline_utils.h"
#include "layout/layout.h"

#include <tvm/tirx/stmt_functor_with_analyzer.h>
#include <tvm/tirx/transform.h>

#include <numeric>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

struct PipelineTensor {
  TensorVar original;
  TensorVar storage;
  int depth;
  Optional<Range> written_range = std::nullopt;
};

using PipelineTensorMap = std::unordered_map<const VarNode *, PipelineTensor>;

class PipelineVersionRewriter : public tvm::tl::StmtExprMutator {
public:
  PipelineVersionRewriter(PipelineTensorMap tensors, PipelineTensorMap scalars,
                          PrimExpr iteration)
      : tensors_(std::move(tensors)), scalars_(std::move(scalars)),
        iteration_(std::move(iteration)) {}

private:
  PrimExpr Version(const PipelineTensor &tensor) const {
    return FloorMod(iteration_, tl::MakeConst(iteration_.ty(), tensor.depth));
  }

  Array<PrimExpr> Indices(const PipelineTensor &tensor,
                          const Array<PrimExpr> &indices, InplaceMode mode) {
    Array<PrimExpr> result{Version(tensor)};
    for (const PrimExpr &index : indices)
      result.push_back(Mutate(index, mode).ValueOrUnchanged(index));
    return result;
  }

  TensorRegion Region(const PipelineTensor &tensor) const {
    TVM_FFI_CHECK(!tensor.written_range, ValueError)
        << "Partially versioned pipeline tensors require an explicit accessed "
           "region";
    Array<Range> ranges{Range::FromMinExtent(Version(tensor), PrimExpr(1))};
    for (const PrimExpr &extent : tensor.original.shape())
      ranges.push_back(Range(PrimExpr(0), extent));
    return TensorRegion(tensor.storage, ranges, TensorRegionType());
  }

  UnchangedOr<Expr> Mutate_(const VarNode *node, InplaceMode mode) final {
    if (auto found = scalars_.find(node); found != scalars_.end())
      return TensorLoad(found->second.storage, {Version(found->second)});
    if (auto found = tensors_.find(node); found != tensors_.end())
      return found->second.storage;
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *node, InplaceMode mode) final {
    if (auto found = scalars_.find(node->var.get()); found != scalars_.end()) {
      PrimExpr value = Mutate(node->value, mode)
                           .ValueOrUnchanged(node->value)
                           .as_or_throw<PrimExpr>();
      return TensorStore(found->second.storage, {Version(found->second)}, value,
                         node->loc);
    }
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *node,
                                InplaceMode mode) final {
    if (auto source = node->source.as<Var>())
      if (auto found = tensors_.find(source.value().get());
          found != tensors_.end()) {
        if (found->second.written_range) {
          TVM_FFI_CHECK(node->indices.size() == 1 &&
                            node->indices[0].ty().IsScalar(),
                        ValueError)
              << "Partially versioned pipeline loads require scalar indices";
          PrimExpr index =
              Mutate(node->indices[0], mode).ValueOrUnchanged(node->indices[0]);
          Optional<PrimVar> snapshot;
          if (SideEffect(index) != CallEffectKind::kPure)
            snapshot = PrimVar("pipeline_index", index.ty());
          PrimExpr address = snapshot ? PrimExpr(*snapshot) : index;
          Range written = *found->second.written_range;
          PrimExpr condition =
              address >= tl::CastValue(written->begin, address.ty()) &&
              address <
                  tl::CastValue(written->begin + written->extent, address.ty());
          PrimExpr versioned =
              TensorLoad(found->second.storage,
                         {Version(found->second), address}, node->loc);
          PrimExpr original =
              TensorLoad(found->second.original, {address}, node->loc);
          PrimExpr value = tvm::if_then_else(condition, versioned, original);
          return snapshot ? PrimExpr(Let(*snapshot, index, value, node->loc))
                          : value;
        }
        return TensorLoad(found->second.storage,
                          Indices(found->second, node->indices, mode),
                          node->loc);
      }
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node,
                            InplaceMode mode) final {
    if (auto destination = node->dest.as<Var>())
      if (auto found = tensors_.find(destination.value().get());
          found != tensors_.end())
        return TensorStore(
            found->second.storage, Indices(found->second, node->indices, mode),
            Mutate(node->value, mode).ValueOrUnchanged(node->value), node->loc);
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionNode *node,
                            InplaceMode mode) final {
    if (auto source = node->source.as<Var>())
      if (auto found = tensors_.find(source.value().get());
          found != tensors_.end()) {
        if (found->second.written_range) {
          sym::Analyzer analyzer;
          Range written = *found->second.written_range;
          TVM_FFI_CHECK(
              node->region.size() == 1 &&
                  tl::arith::CanProve(analyzer, node->region[0]->begin >=
                                                    written->begin) &&
                  tl::arith::CanProve(analyzer,
                                      node->region[0]->begin +
                                              node->region[0]->extent <=
                                          written->begin + written->extent),
              ValueError)
              << "Partially versioned pipeline regions must remain inside the "
                 "overwritten interval";
        }
        Array<Range> ranges{
            Range::FromMinExtent(Version(found->second), PrimExpr(1))};
        for (const Range &range : node->region)
          ranges.push_back(Range::FromMinExtent(
              Mutate(range->begin, mode).ValueOrUnchanged(range->begin),
              Mutate(range->extent, mode).ValueOrUnchanged(range->extent)));
        return TensorRegion(found->second.storage, ranges, TensorRegionType(),
                            node->loc);
      }
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Expr> Mutate_(const CallNode *node, InplaceMode mode) final {
    if (node->op.same_as(tensor_data_ptr_op())) {
      auto tensor = node->args[0].as<TensorVar>();
      auto found =
          tensor ? tensors_.find(tensor.value().get()) : tensors_.end();
      if (found != tensors_.end()) {
        TVM_FFI_CHECK(!found->second.written_range, ValueError)
            << "Partially versioned pipeline pointers require explicit tensor "
               "accesses";
        Array<PrimExpr> indices(found->second.original.shape().size(),
                                PrimExpr(0));
        return Call(std::nullopt, address_of_op(),
                    {TensorLoad(found->second.storage,
                                Indices(found->second, indices, mode))});
      }
    }
    Call call = GetRef<Call>(node);
    if (ParseOperator(call).defined()) {
      auto arguments = call->args;
      for (size_t index = 0; index < arguments.size(); ++index)
        if (auto tensor = arguments[index].as<TensorVar>())
          if (auto found = tensors_.find(tensor.value().get());
              found != tensors_.end())
            arguments.Set(index, Region(found->second));
      call.CopyOnWrite()->args = arguments;
      DictAttrs attrs = call->attrs
                            ? call->attrs.value().as_or_throw<DictAttrs>()
                            : DictAttrs();
      if (!attrs->dict.count(attr::kPipelineMbarPhaseExpr)) {
        attrs.CopyOnWrite()->dict.Set(
            attr::kPipelineMbarPhaseExpr,
            FloorMod(iteration_, tl::MakeConst(iteration_.ty(), 2)));
        call.CopyOnWrite()->attrs = attrs;
      }
    }
    return tvm::tl::StmtExprMutator::Mutate_(call.get(), mode)
        .ValueOrUnchanged(call);
  }

  PipelineTensorMap tensors_;
  PipelineTensorMap scalars_;
  PrimExpr iteration_;
};

class PipelineInjector : public LoopAwareMutator {
public:
  PipelineInjector(sym::AnalyzerObj *analyzer, Optional<Target> target,
                   PipelineReadOnlySet readonly)
      : LoopAwareMutator(analyzer), target_(std::move(target)),
        readonly_(std::move(readonly)) {}

private:
  using Base = LoopAwareMutator;

  static Bind Allocate(const TensorVar &tensor,
                       Optional<Attrs> attributes = std::nullopt) {
    return Bind(tensor, Call(tensor.ty(), alloc_tensor_op(), {}, attributes,
                             {tensor.ty()}));
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                            InplaceMode mode) final {
    auto saved = layouts_;
    auto saved_threads = thread_variables_;
    auto saved_constraints = thread_constraints_;
    auto saved_extents = thread_extents_;
    if (auto layouts =
            node->attrs.GetAttr<Map<TensorVar, Layout>>("layout_map"))
      for (const auto &[tensor, layout] : *layouts)
        layouts_.Set(tensor, layout);
    if (node->op.same_as(launch_thread_op())) {
      auto tag = node->args[0].as_or_throw<StringImm>()->value;
      if (tag.find("threadIdx.") == 0) {
        thread_variables_.insert(node->body_params[0].get());
        thread_constraints_.AddConstr(
            node->body_params[0].as_or_throw<PrimVar>(),
            Range(0, node->args[1].as_or_throw<PrimExpr>()));
      }
    }
    if (auto extents =
            node->attrs.GetAttr<Array<PrimExpr>>("tl.launch_threads")) {
      thread_extents_ = *extents;
      for (size_t axis = 0; axis < extents.value().size(); ++axis)
        if (auto variable = logical_threads_.Get(axis)) {
          thread_variables_.insert(variable.value().get());
          thread_constraints_.AddConstr(*variable,
                                        Range(0, extents.value()[axis]));
        }
    }
    auto result = Base::Mutate_(node, mode);
    layouts_ = std::move(saved);
    thread_variables_ = std::move(saved_threads);
    thread_constraints_ = std::move(saved_constraints);
    thread_extents_ = std::move(saved_extents);
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtNode *node, InplaceMode mode) final {
    auto saved = aliases_;
    auto saved_threads = logical_threads_;
    auto saved_constraints = thread_constraints_;
    auto result = Base::Mutate_(node, mode);
    aliases_ = std::move(saved);
    logical_threads_ = std::move(saved_threads);
    thread_constraints_ = std::move(saved_constraints);
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *node, InplaceMode mode) final {
    aliases_.Bind(node);
    if (const auto *call = node->value.as<CallNode>();
        call && call->op.as<OpNode>() &&
        call->op.as_or_throw<Op>()->name == "tl.launch_thread_idx") {
      auto axis = GetConstInt(call->args[0].as_or_throw<PrimExpr>());
      TVM_FFI_CHECK(axis && *axis >= 0 && *axis < 3, ValueError)
          << "Invalid pipeline thread axis";
      PrimVar variable = node->var.as_or_throw<PrimVar>();
      logical_threads_.Set(*axis, variable);
      thread_variables_.insert(variable.get());
      if (*axis < static_cast<int64_t>(thread_extents_.size()))
        thread_constraints_.AddConstr(variable,
                                      Range(0, thread_extents_[*axis]));
      return Base::Mutate_(node, mode);
    }
    if (auto variable = node->var.as<PrimVar>())
      if (auto value = node->value.as<PrimExpr>())
        scalars_.Set(*variable, Substitute(*value, scalars_));
    return Base::Mutate_(node, mode);
  }

  UnchangedOr<Stmt> Mutate_(const ForNode *node, InplaceMode mode) final {
    auto stage_attribute = node->annotations.Get(kPipelineStage);
    auto order_attribute = node->annotations.Get(kPipelineOrder);
    TVM_FFI_CHECK(stage_attribute.has_value() == order_attribute.has_value(),
                  ValueError)
        << "InjectSoftwarePipeline requires matching stage and order "
           "annotations";
    if (!stage_attribute)
      return Base::Mutate_(node, mode);
    For nested = Base::Mutate_(node, mode)
                     .ValueOrUnchanged(GetRef<For>(node))
                     .as_or_throw<For>();
    node = nested.get();
    TVM_FFI_CHECK(node->kind == ForKind::kDefault, ValueError)
        << "Only serial loops can be software pipelined";
    TVM_FFI_CHECK(tl::arith::CanProve(analyzer_, node->step > 0) ||
                      tl::arith::CanProve(analyzer_, node->step < 0),
                  ValueError)
        << "Software-pipelined loop step must have a provably nonzero sign";
    auto replay = PipelineReplayMask(node->body, aliases_, readonly_,
                                     node->annotations, false);
    PipelineBody body =
        AnalyzePipelineBody(node->body, aliases_, thread_constraints_, replay);
    auto schedule_attrs =
        FilterPipelineMetadata(node->annotations, body, node->body->seq.size());
    auto stages =
        PipelineIntegerArray(schedule_attrs.at(kPipelineStage), kPipelineStage);
    auto order = NormalizePipelineOrder(PipelineIntegerArray(
        schedule_attrs.at(kPipelineOrder), kPipelineOrder));
    ValidatePipelineSchedule(body, stages, order);
    auto asynchronous =
        PlanPipelineAsync(body, stages, order, schedule_attrs, target_);
    for (const PipelineStatement &item : body.statements)
      if (const auto *binding = item.statement.as<BindNode>())
        TVM_FFI_CHECK(binding->var.as<PrimVar>() &&
                          binding->value.as<PrimExpr>(),
                      ValueError)
            << "Pointer and tensor aliases must be declared outside a "
               "software-pipelined loop";
    if (stages.empty())
      return SeqStmt({});
    int first_stage = *std::min_element(stages.begin(), stages.end());
    for (int &stage : stages)
      stage -= first_stage;
    int distance = *std::max_element(stages.begin(), stages.end());
    if (distance == 0 && asynchronous.group_last.empty()) {
      Array<Stmt> statements;
      for (const Bind &declaration : body.declarations)
        statements.push_back(declaration);
      for (size_t index : asynchronous.execution)
        statements.push_back(body.statements[index].statement);
      auto attrs = node->annotations;
      for (const char *key :
           {kPipelineStage, kPipelineOrder, kPipelineAsyncProducers,
            kPipelineAsyncProducerGroups, kPipelineReplayableScalarBinds,
            kPipelineTmaCopies, "software_pipeline_async_stages", "num_stages",
            "tl_pipelined_num_stages"})
        attrs.erase(key);
      return For(node->var, node->begin, node->end, node->step, node->kind,
                 SeqStmt(statements), attrs, node->loc);
    }
    PrimExpr iterations = tl::arith::Simplify(
        analyzer_, Max(tl::MakeConst(node->var.ty(), 0),
                       -FloorDiv(node->begin - node->end, node->step)));
    PrimVar clock("pipeline_time", node->var.ty());
    std::unordered_map<const VarNode *, std::pair<int, int>> lifetimes;
    std::unordered_set<const VarNode *> writes, internal;
    for (const Bind &declaration : body.declarations)
      internal.insert(declaration->var.get());
    for (size_t index = 0; index < stages.size(); ++index)
      for (const auto &[storage, mask] : body.statements[index].memory) {
        auto [found, inserted] =
            lifetimes.try_emplace(storage, stages[index], stages[index]);
        found->second.first = std::min(found->second.first, stages[index]);
        found->second.second = std::max(found->second.second, stages[index]);
        if (mask & kAccessWrite)
          writes.insert(storage);
      }
    PipelineTensorMap tensors, scalar_tensors;
    Map<TensorVar, Layout> new_layouts;
    Array<Stmt> prefix, suffix;
    for (const auto &[storage, lifetime] : lifetimes) {
      if (!PipelineNeedsVersion(body, storage, stages))
        continue;
      if (!writes.count(storage) || lifetime.first == lifetime.second)
        continue;
      TensorVar original = GetRef<Var>(storage).as_or_throw<TensorVar>();
      for (const Var &dependency : tvm::UndefinedVars(original.ty()))
        TVM_FFI_CHECK(!dependency.same_as(node->var), ValueError)
            << "Pipeline allocation geometry must not depend on its induction "
               "variable";
      TVM_FFI_CHECK(original.scope().name() != "shared.tmem" &&
                        original.scope().name().find("barrier") == String::npos,
                    ValueError)
          << "TMEM and manually managed barriers require warp-specialization "
             "lifetime scheduling";
      int depth = lifetime.second - lifetime.first + 1;
      TensorVar expanded = ExpandPipelineTensor(original, depth);
      auto partial = PipelinePartialVersionRange(body, storage);
      tensors.emplace(storage,
                      PipelineTensor{original, expanded, depth, partial});
      prefix.push_back(Allocate(expanded));
      if (auto layout = layouts_.Get(original))
        new_layouts.Set(expanded, layout.value()->Expand({PrimExpr(depth)}));
      if (!internal.count(storage)) {
        Array<Range> ranges{Range::FromMinExtent(
            FloorMod(iterations - 1, tl::MakeConst(iterations.ty(), depth)),
            PrimExpr(1))};
        for (const PrimExpr &extent : original.shape())
          ranges.push_back(partial.value_or(Range(PrimExpr(0), extent)));
        TensorRegion destination =
            partial ? TensorRegion(original, {*partial}, TensorRegionType())
                    : NormalizeToBufferRegion(original);
        suffix.push_back(
            Evaluate(Call(std::nullopt, Op::Get("tl.tileop.copy"),
                          {TensorRegion(expanded, ranges, TensorRegionType()),
                           destination})));
      }
    }
    for (const auto &[storage, tensor] : tensors)
      if (tensor.original.scope().name().find("shared") == 0)
        for (const PrimExpr &bound : {node->begin, node->end, node->step})
          for (const Var &variable :
               tvm::UndefinedVars(Substitute(bound, scalars_)))
            TVM_FFI_CHECK(!thread_variables_.count(variable.get()), ValueError)
                << "Shared-memory pipelines require block-uniform loop bounds";
    PostOrderVisit(node->body, [&](const ObjectRef &object) {
      if (const auto *variable = object.as<VarNode>()) {
        MemoryLocation location = aliases_.Resolve(GetRef<Var>(variable));
        if (location.allocation &&
            tensors.count(location.allocation.value().get()))
          TVM_FFI_CHECK(variable == location.allocation.value().get(),
                        ValueError)
              << "Pipeline versioning through an external pointer or tensor "
                 "alias requires an explicit stage-local view";
      }
    });
    for (const Bind &declaration : body.declarations)
      if (!tensors.count(declaration->var.get()))
        prefix.push_back(declaration);
    for (const PipelineStatement &item : body.statements)
      if (const auto *binding = item.statement.as<BindNode>()) {
        PrimVar variable = binding->var.as_or_throw<PrimVar>();
        TensorVar storage(variable->name + "_pipeline",
                          TensorType({PrimExpr(distance + 1)}, variable.ty(),
                                     runtime::MemoryScope::Get("local")));
        scalar_tensors.emplace(variable.get(),
                               PipelineTensor{storage, storage, distance + 1});
        prefix.push_back(Allocate(storage));
      }
    std::vector<size_t> execution(stages.size());
    std::iota(execution.begin(), execution.end(), 0);
    std::sort(
        execution.begin(), execution.end(),
        [&](size_t left, size_t right) { return order[left] < order[right]; });
    Array<Stmt> pipeline;
    for (size_t index : execution) {
      PrimExpr ordinal = clock - tl::MakeConst(clock.ty(), stages[index]);
      PrimExpr iteration = node->begin + ordinal * node->step;
      Stmt statement = Substitute(body.statements[index].statement,
                                  {{node->var, iteration}});
      auto rewriter = make_object<PipelineVersionRewriter>(
          tensors, scalar_tensors, ordinal);
      statement = rewriter->Mutate(statement).ValueOrUnchanged(statement);
      statement = *tl::RenewDef(tirx::Function({}, SeqStmt({statement})))->body;
      int wait = asynchronous.WaitBefore(index, body, stages, order);
      if (wait >= 0)
        pipeline.push_back(PipelineAsyncWait(wait));
      if (asynchronous.producers[index]) {
        auto marker = make_object<PipelineAsyncMarker>(*target_);
        statement = marker->Mutate(statement).ValueOrUnchanged(statement);
        statement =
            RegionStmt(::tvm::tl::scope_op(), {}, {},
                       DictAttrs({{kPipelineCPAsyncScope, IntImm::Bool(true)}}),
                       SeqStmt({statement}));
      }
      pipeline.push_back(
          If(ordinal >= 0 && ordinal < iterations, SeqStmt({statement})));
      if (asynchronous.producers[index] &&
          asynchronous.group_last[asynchronous.groups[index]] == order[index]) {
        if (asynchronous.group_last.size() * (distance + 1) > 8)
          pipeline.push_back(PipelineAsyncWait(7));
        pipeline.push_back(PipelineAsyncCommit());
      }
    }
    auto annotations = node->annotations;
    for (const char *key :
         {kPipelineStage, kPipelineOrder, kPipelineAsyncProducers,
          kPipelineAsyncProducerGroups, kPipelineTmaCopies,
          kPipelineReplayableScalarBinds, "software_pipeline_async_stages",
          "num_stages"})
      annotations.erase(key);
    annotations.Set("tl_pipelined_num_stages", PrimExpr(distance + 1));
    annotations.Set("tl.pipeline_injected", IntImm::Bool(true));
    Stmt pipeline_loop =
        For(clock, tl::MakeConst(clock.ty(), 0), iterations + distance,
            tl::MakeConst(clock.ty(), 1), ForKind::kDefault, SeqStmt(pipeline),
            annotations, node->loc);
    Array<Stmt> active{pipeline_loop};
    if (!asynchronous.group_last.empty())
      active.push_back(PipelineAsyncWait(0));
    active.insert(active.end(), suffix.begin(), suffix.end());
    prefix.push_back(
        If(iterations > 0, SeqStmt(active), std::nullopt, node->loc));
    DictAttrs attributes;
    if (!new_layouts.empty())
      attributes = DictAttrs({{"layout_map", new_layouts}});
    return RegionStmt(::tvm::tl::scope_op(), {}, {}, attributes,
                      SeqStmt(prefix), {}, node->loc);
  }

  MemoryAliases aliases_;
  Map<TensorVar, Layout> layouts_;
  Map<PrimVar, PrimExpr> scalars_;
  std::unordered_set<const VarNode *> thread_variables_;
  ConstrSet thread_constraints_;
  Map<int64_t, PrimVar> logical_threads_;
  Array<PrimExpr> thread_extents_;
  Optional<Target> target_;
  PipelineReadOnlySet readonly_;
};
} // namespace

tvm::transform::Pass InjectSoftwarePipeline() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> Optional<tirx::Function> {
    if (function->body) {
      sym::Analyzer analyzer;
      auto injector = make_object<PipelineInjector>(
          analyzer.get(), function->GetAttr<Target>("target"),
          PipelineReadOnlyParameters(function));
      SeqStmt body = *function->body;
      function.CopyOnWrite()->body =
          injector->Mutate(body).ValueOrUnchanged(body);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.InjectSoftwarePipeline");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.InjectSoftwarePipeline",
                              InjectSoftwarePipeline);
}
} // namespace tvm::tl
