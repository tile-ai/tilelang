#include "common/reducer.h"

#include "backend/common/op/reduce_layout.h"
#include "compat/visit.h"
#include "config.h"
#include "layout/utils.h"
#include "op/copy.h"
#include "op/fill.h"
#include "op/parallel.h"

#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

struct ReducerPlan {
  ReducerPlan(TensorVar tensor, ReducerV2OpType operation)
      : original(tensor),
        storage(ReducerTensor(tensor, tensor->name, "local.fragment")),
        operation(operation) {}
  TensorVar original;
  TensorVar storage;
  ReducerV2OpType operation;
  Optional<TensorVar> destination;
  Optional<TensorVar> seed;
  Optional<TensorVar> packed;
  Optional<PrimExpr> packed_lane;
  Fragment packed_layout;
  Fragment layout;
  size_t update_count = 0;
  int64_t participants = 1;
  int64_t thread_begin = 0;
  PrimType thread_type = PrimType::Int(32);
  bool narrow = false;
  bool initialized = false;
  bool assigned_destination = false;
  std::vector<std::pair<int, int>> steps;
};

using PlanMap = std::unordered_map<const VarNode *, ReducerPlan>;

struct DestinationUse {
  int other = 0;
  int finalized = 0;
  int copied = 0;
  std::vector<TensorVar> consumers;
};

class DestinationUses : public StmtExprVisitor {
public:
  std::unordered_map<const VarNode *, DestinationUse> uses;

  bool Chain(const TensorVar &root, std::vector<TensorVar> *result) const {
    std::unordered_set<const VarNode *> visited;
    std::function<bool(const TensorVar &)> visit =
        [&](const TensorVar &tensor) {
          if (!IsFragmentBuffer(tensor) || !visited.insert(tensor.get()).second)
            return false;
          auto found = uses.find(tensor.get());
          if (found == uses.end())
            return false;
          const DestinationUse &use = found->second;
          bool first = tensor.same_as(root);
          if (use.other || use.finalized != (first ? 1 : 0) ||
              use.copied != (first ? 0 : 1))
            return false;
          result->push_back(tensor);
          for (const TensorVar &consumer : use.consumers) {
            if (!IsGlobalBuffer(consumer) && !visit(consumer))
              return false;
          }
          return true;
        };
    return visit(root);
  }

private:
  Optional<VisitInterrupt> Visit_(const VarNode *node) final {
    if (GetRef<Var>(node).as<TensorVar>())
      ++uses[node].other;
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const BindNode *node) final {
    return Visit(node->value);
  }

  Optional<VisitInterrupt> Visit_(const CallNode *node) final {
    if (node->op.same_as(Op::Get("tl.layout_hint")))
      return std::nullopt;
    if (node->op.same_as(Copy::Get()) && node->args.size() >= 2) {
      TensorVar source = ReducerRegionTensor(node->args[0]);
      TensorVar destination = ReducerRegionTensor(node->args[1]);
      uses[source.get()].consumers.push_back(destination);
      ++uses[destination.get()].copied;
      return std::nullopt;
    }
    if (node->op.same_as(FinalizeReducerV2Op::Get())) {
      ++uses[ReducerRegionTensor(node->args[1]).get()].finalized;
      return std::nullopt;
    }
    return StmtExprVisitor::Visit_(node);
  }
};

class EpochCollector : public StmtExprVisitorWithAnalyzer {
public:
  PlanMap plans;
  LayoutMap layouts;

private:
  Optional<PrimExpr> PackingLane(const ReducerPlan &plan,
                                 const Array<PrimExpr> &indices,
                                 const PrimExpr &value) {
    if (!plan.original.dtype().MatchesElementType(kDLFloat, 16) &&
        !plan.original.dtype().MatchesElementType(kDLBfloat, 16))
      return std::nullopt;
    for (size_t reverse = loops_.size(); reverse > 0; --reverse) {
      size_t root = reverse - 1;
      const ForNode *loop = loops_[root];
      if (loop->kind != ForKind::kParallel ||
          !loop->annotations.count(attr::kParallelLoopLayout))
        continue;
      Fragment layout = loop->annotations.at(attr::kParallelLoopLayout)
                            .as_or_throw<Fragment>();
      Array<PrimVar> variables;
      size_t inner = root;
      while (inner < loops_.size() &&
             loops_[inner]->kind == ForKind::kParallel) {
        variables.push_back(loops_[inner++]->var);
      }
      if (variables.size() != layout->InputDim())
        return std::nullopt;
      for (size_t serial = loops_.size(); serial > inner; --serial) {
        const ForNode *candidate = loops_[serial - 1];
        auto step = GetConstInt(candidate->step);
        auto extent =
            GetConstInt(analyzer_->Simplify(candidate->end - candidate->begin));
        if (candidate->kind == ForKind::kDefault && step && *step == 1 &&
            extent && *extent >= 2 && *extent % 2 == 0) {
          return FloorMod(candidate->var - candidate->begin,
                          MakeConst(candidate->var.ty(), 2));
        }
      }
      ReducerUpdateSiteHint site{layout, variables, indices, value};
      auto analysis = AnalyzeReducerUpdateSite(site, plan.original.shape(),
                                               plan.participants,
                                               plan.thread_begin, analyzer_);
      if (!analysis.narrow_eligible || variables.empty() ||
          analysis.is_output_dim.back())
        return std::nullopt;
      auto extent = GetConstInt(layout->InputShape().back());
      if (!extent)
        return std::nullopt;
      int64_t multiplicity = 1;
      auto steps = backend::reduce::CollectThreadReduceSteps(
          analysis.iter_sum, InputPlaceholder(variables.size() - 1));
      for (const auto &step : steps) {
        if (step.lower_factor % 2 != 0 || step.extent <= 0 ||
            multiplicity > *extent / step.extent)
          return std::nullopt;
        multiplicity *= step.extent;
      }
      int64_t run = *extent / multiplicity;
      if (*extent % multiplicity == 0 && run >= 2 && run % 2 == 0)
        return FloorMod(variables.back(), MakeConst(variables.back().ty(), 2));
      return std::nullopt;
    }
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const ForNode *node) final {
    loops_.push_back(node);
    auto result = StmtExprVisitorWithAnalyzer::Visit_(node);
    loops_.pop_back();
    return result;
  }

  Optional<VisitInterrupt> Visit_(const BindNode *node) final {
    if (auto tensor = node->var.as<TensorVar>();
        tensor && IsReducerV2Buffer(*tensor)) {
      const auto *allocation = node->value.as<CallNode>();
      TVM_FFI_CHECK(allocation && allocation->op.same_as(alloc_tensor_op()),
                    ValueError)
          << "Reducer storage cannot be aliased";
      auto operation = ReducerOperation(allocation);
      TVM_FFI_CHECK(operation, ValueError)
          << "Reducer allocation is missing tl.reducer_op";
      plans.emplace(tensor->get(),
                    ReducerPlan(*tensor, ParseReducerV2OpType(*operation)));
    }
    return StmtExprVisitorWithAnalyzer::Visit_(node);
  }

  Optional<VisitInterrupt> Visit_(const RegionStmtNode *node) final {
    auto previous = thread_;
    if (node->op.same_as(launch_thread_op()) &&
        node->args[0].as_or_throw<StringImm>()->value == "threadIdx.x") {
      thread_ = node->body_params[0].as_or_throw<PrimVar>();
    }
    auto result = StmtExprVisitorWithAnalyzer::Visit_(node);
    thread_ = previous;
    return result;
  }

  Optional<VisitInterrupt> Visit_(const CallNode *node) final {
    if (node->op.same_as(Op::Get("tl.layout_hint"))) {
      TensorVar tensor = node->args[0].as_or_throw<TensorVar>();
      layouts.Set(tensor, node->attrs.value()
                              .as_or_throw<DictAttrs>()
                              .GetAttr<Layout>("layout")
                              .value());
      return std::nullopt;
    }
    if (node->op.same_as(ReducerInitOp::Get())) {
      auto found = plans.find(ReducerRegionTensor(node->args[0]).get());
      TVM_FFI_CHECK(found != plans.end(), ValueError)
          << "reducer_init requires allocated storage";
      ReducerPlan &plan = found->second;
      if (thread_) {
        plan.thread_type = thread_.value().ty();
        auto bound = analyzer_->const_int_bound(*thread_);
        TVM_FFI_CHECK(bound->min_value >= 0 && bound->max_value < 1024,
                      ValueError)
            << "Reducer participant bounds must be statically known";
        plan.thread_begin = bound->min_value;
        plan.participants = bound->max_value - bound->min_value + 1;
      }
      if (node->args.size() > 1 && !plan.seed) {
        plan.seed =
            TensorVar(plan.original->name + "_seed",
                      TensorType({IntImm::Int32(1)}, plan.original.dtype(),
                                 runtime::MemoryScope::Get("local")));
      }
      plan.initialized = true;
      return std::nullopt;
    }
    if (node->op.same_as(FinalizeReducerV2Op::Get())) {
      auto found = plans.find(ReducerRegionTensor(node->args[0]).get());
      TVM_FFI_CHECK(found != plans.end(), ValueError)
          << "finalize_reducer requires allocated storage";
      found->second.destination = ReducerRegionTensor(node->args[1]);
      return std::nullopt;
    }
    if (node->op.same_as(reducer_update())) {
      auto update = ParseReducerUpdate(node);
      ReducerPlan &plan = plans.at(update.reducer.get());
      ++plan.update_count;
      plan.packed_lane = plan.update_count == 1
                             ? PackingLane(plan, update.indices, update.value)
                             : std::nullopt;
    }
    return StmtExprVisitorWithAnalyzer::Visit_(node);
  }

  Optional<PrimVar> thread_;
  std::vector<const ForNode *> loops_;
};

class ReducerMaterializer : public StmtExprMutator {
public:
  ReducerMaterializer(PlanMap plans, LayoutMap layouts,
                      const DestinationUses &uses)
      : plans_(std::move(plans)), layouts_(std::move(layouts)) {
    sym::Analyzer analyzer;
    for (auto &[tensor, plan] : plans_) {
      TVM_FFI_CHECK(plan.initialized && plan.destination, ValueError)
          << "ReducerPlanAndMaterialize requires a complete reducer epoch";
      auto inferred = layouts_.Get(plan.original);
      if (inferred) {
        if (auto partial = inferred->as<PartialFragment>()) {
          plan.layout = partial->AsPostCollective();
          auto combine = GetConstInt(partial.value()->CombineSize());
          plan.narrow = combine && *combine < plan.participants;
          if (plan.narrow)
            plan.steps = partial.value()->CombineSteps();
        }
      }
      auto destination_layout = layouts_.Get(*plan.destination);
      if (plan.narrow && destination_layout) {
        auto fragment = destination_layout->as<Fragment>();
        Array<PrimExpr> coordinates;
        for (size_t axis = 0; axis < plan.original.shape().size(); ++axis) {
          PrimVar coordinate = InputPlaceholder(axis);
          analyzer->Bind(coordinate,
                         Range::FromMinExtent(0, plan.original.shape()[axis]),
                         true);
          coordinates.push_back(coordinate);
        }
        if (!fragment ||
            !ProveFragmentContains(*fragment, plan.layout, coordinates,
                                   coordinates, analyzer)) {
          std::vector<TensorVar> chain;
          bool override = uses.Chain(*plan.destination, &chain);
          for (const TensorVar &tensor : chain) {
            if (tensor.shape().size() != plan.layout->InputShape().size()) {
              override = false;
              break;
            }
            for (size_t axis = 0; axis < tensor.shape().size(); ++axis) {
              if (!analyzer->CanProveEqual(tensor.shape()[axis],
                                           plan.layout->InputShape()[axis]))
                override = false;
            }
          }
          if (override) {
            for (const TensorVar &tensor : chain)
              extra_layouts_.Set(tensor, plan.layout);
          } else {
            plan.narrow = false;
          }
        }
      }
      if (tvm::transform::PassContext::Current()
              ->GetConfig<bool>("tl.reducer_force_baseline")
              .value_or(false))
        plan.narrow = false;
      if (!plan.narrow) {
        int64_t slots = 1;
        for (const PrimExpr &dimension : plan.original.shape()) {
          auto extent = GetConstInt(dimension);
          TVM_FFI_CHECK(extent && *extent > 0 && *extent <= 256 / slots,
                        ValueError)
              << "FullParticipant reducer storage exceeds the 256-slot policy "
                 "limit";
          slots *= *extent;
        }
        Range range = Range::FromMinExtent(
            MakeConst(plan.thread_type, plan.thread_begin),
            MakeConst(plan.thread_type, plan.participants));
        plan.layout = PartialFragment::FullyReplicated(
                          plan.original.shape(),
                          MakeConst(plan.thread_type, plan.participants), range)
                          .AsPostCollective();
        plan.steps.clear();
        if (plan.participants > 1)
          plan.steps.emplace_back(plan.participants, 1);
        std::vector<TensorVar> chain;
        if (uses.Chain(*plan.destination, &chain)) {
          for (const TensorVar &tensor : chain) {
            extra_layouts_.Set(
                tensor,
                PartialFragment::FullyReplicated(
                    tensor.shape(),
                    MakeConst(plan.thread_type, plan.participants), range)
                    .AsPostCollective());
          }
          plan.assigned_destination = true;
        }
      }
      if (!destination_layout) {
        TVM_FFI_CHECK(IsFragmentBuffer(*plan.destination), ValueError)
            << "finalize_reducer destination must be a fragment";
        extra_layouts_.Set(*plan.destination, plan.layout);
        plan.assigned_destination = true;
      }
      if (plan.narrow && plan.packed_lane && plan.layout->OutputDim() == 1) {
        Array<PrimExpr> coordinates;
        for (size_t axis = 0; axis < plan.original.shape().size(); ++axis)
          coordinates.push_back(InputPlaceholder(axis));
        Array<PrimExpr> packed_shape = plan.original.shape();
        packed_shape.push_back(IntImm::Int32(2));
        auto slot = plan.layout->Forward(coordinates)[0];
        PrimExpr lane = InputPlaceholder(coordinates.size());
        plan.packed_layout =
            Fragment(packed_shape, {slot * MakeConst(slot.ty(), 2) + lane},
                     plan.layout->ForwardThread(coordinates,
                                                ReplicationPlaceholder()),
                     plan.layout->ReplicateExtent(), std::nullopt)
                ->BindThreadRange(plan.layout->ThreadRange());
        plan.packed =
            TensorVar(plan.original->name + "_pk",
                      TensorType(packed_shape, plan.original.dtype(),
                                 runtime::MemoryScope::Get("local.fragment"),
                                 plan.original.ty()->data_alignment));
      }
      if (tl_config::ReducerPlanVerboseEnabled()) {
        LOG(INFO) << "[ReducerPlan] `" << tensor->name
                  << "`: " << (plan.narrow ? "narrow" : "FullParticipant")
                  << ", " << plan.steps.size() << " collective steps";
      }
    }
  }

private:
  UnchangedOr<Stmt> Mutate_(const BindNode *node, InplaceMode mode) final {
    if (auto found = plans_.find(node->var.get()); found != plans_.end()) {
      const ReducerPlan &plan = found->second;
      Array<Stmt> statements{AllocateReducerTensor(plan.storage),
                             ReducerLayoutHint(plan.storage, plan.layout)};
      if (plan.packed) {
        statements.push_back(AllocateReducerTensor(*plan.packed));
        statements.push_back(
            ReducerLayoutHint(*plan.packed, plan.packed_layout));
      }
      if (plan.seed)
        statements.push_back(AllocateReducerTensor(*plan.seed));
      return SeqStmt(statements);
    }
    Stmt binding = StmtExprMutator::Mutate_(node, mode)
                       .ValueOrUnchanged(GetRef<Stmt>(node));
    if (auto tensor = node->var.as<TensorVar>()) {
      if (auto layout = extra_layouts_.Get(*tensor)) {
        return SeqStmt({binding, ReducerLayoutHint(*tensor, *layout)});
      }
    }
    return binding;
  }

  Stmt CopyReplicated(const ReducerPlan &plan, const TensorVar &destination) {
    Array<PrimExpr> indices;
    for (size_t axis = 0; axis < destination.shape().size(); ++axis) {
      indices.push_back(PrimVar("finalize_copy_" + std::to_string(axis),
                                destination.shape()[axis].ty()));
    }
    Stmt result =
        TensorStore(destination, indices, TensorLoad(plan.storage, indices));
    for (int axis = static_cast<int>(indices.size()) - 1; axis >= 0; --axis) {
      PrimType index_type = indices[axis].ty();
      result =
          For(indices[axis].as_or_throw<PrimVar>(), MakeConst(index_type, 0),
              destination.shape()[axis], MakeConst(index_type, 1),
              ForKind::kDefault, SeqStmt(result));
    }
    return result;
  }

  Stmt FoldPacked(const ReducerPlan &plan) {
    Array<PrimExpr> indices;
    for (size_t axis = 0; axis < plan.original.shape().size(); ++axis)
      indices.push_back(PrimVar("reducer_fold_" + std::to_string(axis),
                                plan.original.shape()[axis].ty()));
    Array<PrimExpr> first = indices;
    Array<PrimExpr> second = indices;
    first.push_back(IntImm::Int32(0));
    second.push_back(IntImm::Int32(1));
    Stmt result = TensorStore(
        plan.storage, indices,
        ReducerV2Combine(plan.operation, TensorLoad(*plan.packed, first),
                         TensorLoad(*plan.packed, second)));
    for (int axis = static_cast<int>(indices.size()) - 1; axis >= 0; --axis) {
      PrimType index_type = indices[axis].ty();
      Map<String, Any> annotations;
      if (axis == 0)
        annotations.Set(attr::kParallelLoopLayout, plan.layout);
      result =
          For(indices[axis].as_or_throw<PrimVar>(), MakeConst(index_type, 0),
              plan.original.shape()[axis], MakeConst(index_type, 1),
              ForKind::kParallel, SeqStmt(result), annotations);
    }
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateNode *node, InplaceMode mode) final {
    const auto *call = node->value.as<CallNode>();
    if (!call)
      return StmtExprMutator::Mutate_(node, mode);
    if (call->op.same_as(Op::Get("tl.layout_hint"))) {
      TensorVar tensor = call->args[0].as_or_throw<TensorVar>();
      if (plans_.count(tensor.get()))
        return SeqStmt({});
      if (auto layout = extra_layouts_.Get(tensor))
        return ReducerLayoutHint(tensor, *layout);
    }
    if (call->op.same_as(ReducerInitOp::Get())) {
      const ReducerPlan &plan =
          plans_.at(ReducerRegionTensor(call->args[0]).get());
      Array<Stmt> result;
      if (plan.seed) {
        PrimExpr seed =
            ReducerV2Identity(plan.operation, plan.original.dtype());
        if (call->args.size() > 1) {
          seed = Mutate(call->args[1], mode)
                     .ValueOrUnchanged(call->args[1])
                     .as_or_throw<PrimExpr>();
        }
        result.push_back(TensorStore(*plan.seed, {IntImm::Int32(0)}, seed));
      }
      result.push_back(Evaluate(
          Call(PrimType::Void(), Fill::Get(),
               {NormalizeToBufferRegion(plan.packed.value_or(plan.storage)),
                ReducerV2Identity(plan.operation, plan.storage.dtype())})));
      return SeqStmt(result);
    }
    if (call->op.same_as(reducer_update())) {
      auto update = ParseReducerUpdate(call);
      const ReducerPlan &plan = plans_.at(update.reducer.get());
      PrimExpr value =
          Mutate(update.value, mode).ValueOrUnchanged(update.value);
      TensorVar storage = plan.packed.value_or(plan.storage);
      Array<PrimExpr> indices = update.indices;
      if (plan.packed)
        indices.push_back(*plan.packed_lane);
      Stmt store = TensorStore(
          storage, indices,
          ReducerV2Combine(plan.operation, TensorLoad(storage, indices), value),
          node->loc);
      if (plan.narrow)
        return store;
      return RegionStmt(Op::Get(attr::kParallelMultiplicity), {}, {},
                        DictAttrs(), SeqStmt(store));
    }
    if (call->op.same_as(FinalizeReducerV2Op::Get())) {
      const ReducerPlan &plan =
          plans_.at(ReducerRegionTensor(call->args[0]).get());
      Array<Stmt> result;
      if (plan.packed)
        result.push_back(FoldPacked(plan));
      if (!plan.steps.empty() || plan.seed) {
        Array<Expr> operands{NormalizeToBufferRegion(plan.storage),
                             PrimExpr(static_cast<int>(plan.operation))};
        for (const auto &[width, scale] : plan.steps) {
          operands.push_back(PrimExpr(width));
          operands.push_back(PrimExpr(scale));
        }
        auto attributes =
            call->attrs.value_or(DictAttrs()).as_or_throw<DictAttrs>();
        attributes.CopyOnWrite()->dict.Set("plan", IntImm::Int32(1));
        if (plan.seed) {
          attributes.CopyOnWrite()->dict.Set(
              "seed", TensorLoad(*plan.seed, {IntImm::Int32(0)}));
        }
        result.push_back(Evaluate(Call(
            PrimType::Void(), FinalizeReducerOp::Get(), operands, attributes)));
      }
      TensorVar destination = ReducerRegionTensor(call->args[1]);
      if (plan.assigned_destination && !plan.narrow) {
        result.push_back(CopyReplicated(plan, destination));
      } else {
        result.push_back(
            Evaluate(Call(PrimType::Void(), Copy::Get(),
                          {NormalizeToBufferRegion(plan.storage),
                           NormalizeToBufferRegion(destination)})));
      }
      return SeqStmt(result);
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

  PlanMap plans_;
  LayoutMap layouts_;
  LayoutMap extra_layouts_;
};
} // namespace

tvm::transform::Pass ReducerPlanAndMaterialize() {
  return tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext) -> Optional<tirx::Function> {
        if (!function->body)
          return function;
        auto collector = ffi::make_object<EpochCollector>();
        collector->Visit(*function->body);
        if (collector->plans.empty())
          return function;
        auto uses = ffi::make_object<DestinationUses>();
        uses->Visit(*function->body);
        auto materializer = ffi::make_object<ReducerMaterializer>(
            std::move(collector->plans), collector->layouts, *uses);
        SeqStmt body = *function->body;
        function.CopyOnWrite()->body =
            materializer->Mutate(body).ValueOrUnchanged(body);
        return function;
      },
      0, "tl.ReducerPlanAndMaterialize");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.ReducerPlanAndMaterialize",
                                   ReducerPlanAndMaterialize);
}
} // namespace tvm::tl
