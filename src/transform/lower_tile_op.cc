#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "compat/region.h"
#include "op/prim.h"
#include "transform/common/loop_aware_mutator.h"

#include "compat/thread.h"
#include "compat/visit.h"
#include "cuda/op/builtin.h"
#include "cuda/target_utils.h"
#include "cuda/transform/ptx_async_copy_injector.h"
#include "layout/swizzle_mode.h"
#include "layout/utils.h"
#include "lower_tile_op.h"
#include "op/operator.h"
#include "op/utils.h"
#include "span_utils.h"
#include "transform/common/attr.h"
#include "transform/common/mbarrier.h"
#include "transform/common/pipeline_async.h"
#include "transform/loop_partition.h"

#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>
#include <tvm/tirx/transform.h>

#include <deque>
#include <limits>
#include <unordered_set>

namespace tvm::tl {
using namespace ffi;
using namespace tirx;

namespace {

class TensorRebinder : public tvm::tl::StmtExprMutator {
public:
  explicit TensorRebinder(Map<TensorVar, TensorVar> replacements)
      : replacements_(std::move(replacements)) {}

private:
  UnchangedOr<Expr> Mutate_(const VarNode *node, InplaceMode mode) final {
    if (auto tensor = GetRef<Var>(node).as<TensorVar>()) {
      if (auto replacement = replacements_.Get(tensor.value()))
        return replacement.value();
    }
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *node, InplaceMode mode) final {
    if (auto tensor = node->var.as<TensorVar>()) {
      if (auto replacement = replacements_.Get(*tensor)) {
        Expr value = Mutate(node->value, mode).ValueOrUnchanged(node->value);
        const auto *call = value.as<CallNode>();
        TVM_FFI_CHECK(call && (call->op.same_as(alloc_tensor_op()) ||
                               call->op.same_as(decl_tensor_op())),
                      ValueError)
            << "Tensor rebinding requires an allocation or declaration";
        VarRemapSet(node->var, *replacement);
        return Bind(*replacement,
                    Call(replacement.value()->ty, call->op, call->args,
                         call->attrs, {replacement.value()->ty}, call->loc),
                    node->loc);
      }
    }
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }
  Map<TensorVar, TensorVar> replacements_;
};

class BarrierArrivalUpdater : public tvm::tl::StmtExprMutator {
public:
  explicit BarrierArrivalUpdater(Map<Var, PrimExpr> updates)
      : updates_(std::move(updates)) {}

  void CheckConsumed() const {
    TVM_FFI_CHECK(updates_.empty(), ValueError)
        << "Barrier arrival override has no matching annotated allocation";
  }

private:
  UnchangedOr<Stmt> Mutate_(const BindNode *node, InplaceMode mode) final {
    auto updated_count = updates_.Get(node->var);
    if (!updated_count)
      return tvm::tl::StmtExprMutator::Mutate_(node, mode);
    const auto *call = node->value.as<CallNode>();
    TVM_FFI_CHECK(call && call->op.same_as(alloc_tensor_op()) && call->attrs,
                  ValueError)
        << "Barrier arrival override requires an annotated allocation";
    DictAttrs attrs = call->attrs.value().as_or_throw<DictAttrs>();
    auto counts = attrs.GetAttr<Array<PrimExpr>>("barrier_init");
    TVM_FFI_CHECK(counts.has_value(), ValueError)
        << "Barrier initialization metadata is missing";
    attrs.CopyOnWrite()->dict.Set(
        "barrier_init",
        Array<PrimExpr>(counts.value().size(), updated_count.value()));
    updates_.erase(node->var);
    return Bind(
        node->var,
        Call(call->ty, call->op, call->args, attrs, call->ty_args, call->loc),
        node->loc);
  }

  Map<Var, PrimExpr> updates_;
};

int64_t StaticElements(const Array<PrimExpr> &shape, const String &name) {
  int64_t count = 1;
  for (const PrimExpr &extent : shape) {
    auto value = GetConstInt(extent);
    TVM_FFI_CHECK(value && *value > 0, ValueError)
        << "Shared tensor `" << name
        << "` with a layout requires positive compile-time extents";
    TVM_FFI_CHECK(count <= std::numeric_limits<int64_t>::max() / *value,
                  ValueError)
        << "Shared layout allocation size overflows int64";
    count *= *value;
  }
  return count;
}

TensorVar PhysicalTensor(const TensorVar &tensor,
                         const Optional<Layout> &layout) {
  auto scope = IsFragmentBuffer(tensor) ? runtime::MemoryScope::Get("local")
                                        : tensor.scope();
  Array<PrimExpr> shape = tensor.shape();
  if (layout) {
    TVM_FFI_CHECK(tensor.ty()->HasDefaultLayout(), ValueError)
        << "Tensor `" << tensor->name
        << "` cannot own both native and TileLang layouts";
    shape = layout.value()->OutputShape();
    if (IsSharedBuffer(tensor)) {
      int64_t elements = StaticElements(tensor.shape(), tensor->name);
      int64_t physical = StaticElements(shape, tensor->name);
      if (elements > physical) {
        TVM_FFI_CHECK(elements % physical == 0, ValueError)
            << "Shared layout replication must have an integral extent";
        shape.insert(shape.begin(),
                     tl::MakeConst(PrimType::Int(64), elements / physical));
      }
    }
  }
  TensorType type(shape, tensor.dtype(), scope, tensor.ty()->data_alignment,
                  layout ? Optional<tirx::Layout>() : tensor.ty()->layout,
                  tensor->loc);
  return TensorVar(tensor->name, type, tensor->loc);
}

class LowerTileOpPass : public LoopAwareMutator {
public:
  LowerTileOpPass(sym::Analyzer analyzer, Target target,
                  LowerTileOpOptions options = {})
      : LoopAwareMutator(analyzer), target_(std::move(target)),
        options_(std::move(options)) {}

  tirx::Function Run(tirx::Function function) {
    if (!function->body)
      return function;
    MarkTensorParamShapes(function);
    PostOrderVisit(function->body.value(), [this](const ObjectRef &object) {
      if (auto tensor = object.as<TensorVar>())
        tensors_.insert(tensor.value());
      if (const auto *call = object.as<CallNode>();
          call && call->op.as<OpNode>() &&
          call->op.as_or_throw<Op>()->name == "tl.layout_hint") {
        TensorVar tensor = call->args[0].as_or_throw<TensorVar>();
        Layout layout = call->attrs.value()
                            .as_or_throw<DictAttrs>()
                            .GetAttr<Layout>("layout")
                            .value();
        if (auto previous = layouts_.Get(tensor)) {
          TVM_FFI_CHECK(previous.value()->IsEqual(layout.get()), ValueError)
              << "Conflicting layout hints for tensor `" << tensor->name << "`";
        }
        layouts_.Set(tensor, layout);
      }
    });
    for (const Var &parameter : function->params) {
      if (auto tensor = parameter.as<TensorVar>())
        tensors_.insert(tensor.value());
    }
    for (const TensorVar &tensor : tensors_) {
      if (layouts_.count(tensor) || IsFragmentBuffer(tensor)) {
        remap_.Set(tensor, PhysicalTensor(tensor, layouts_.Get(tensor)));
      }
    }
    resources_.emplace_back();
    Stmt body = Rewrite(function->body.value());
    body = FinishResources(body);
    if (!barrier_updates_.empty()) {
      auto updater = make_object<BarrierArrivalUpdater>(barrier_updates_);
      body = updater->Mutate(body).ValueOrUnchanged(body).as_or_throw<Stmt>();
      updater->CheckConsumed();
    }
    Array<Stmt> statements;
    for (const Var &parameter : function->params) {
      if (auto tensor = parameter.as<TensorVar>()) {
        if (auto physical = remap_.Get(tensor.value())) {
          statements.push_back(Bind(
              physical.value(),
              Call(std::nullopt, decl_tensor_op(), {tensor.value().data()},
                   std::nullopt, {physical.value().ty()}, tensor.value()->loc),
              tensor.value()->loc));
        }
      }
    }
    statements.push_back(body);
    SeqStmt aligned_body(statements, function->body.value()->loc);
    Map<TensorVar, TensorVar> aligned_tensors;
    PostOrderVisit(aligned_body, [&](const ObjectRef &object) {
      if (auto tensor = object.as<TensorVar>();
          tensor && IsSharedBuffer(*tensor)) {
        if (auto required = alignments_.Get(tensor.value()->name)) {
          int alignment = GetConstInt(*required).value();
          if (alignment > tensor.value().ty()->data_alignment) {
            TensorType type(tensor.value().shape(), tensor.value().dtype(),
                            tensor.value().scope(), alignment,
                            tensor.value().ty()->layout, tensor.value()->loc);
            aligned_tensors.Set(*tensor, TensorVar(tensor.value()->name, type,
                                                   tensor.value()->loc));
          }
        }
      }
    });
    if (!aligned_tensors.empty()) {
      auto rewriter = make_object<TensorRebinder>(aligned_tensors);
      aligned_body =
          rewriter->Mutate(aligned_body).ValueOrUnchanged(aligned_body);
    }
    function.CopyOnWrite()->body = aligned_body;
    DictAttrs attributes = function->attrs;
    attributes.CopyOnWrite()->dict.erase(attr::kLayoutMap);
    attributes.CopyOnWrite()->dict.Set(kHasTMA, has_tma_);
    if (!alignments_.empty())
      attributes.CopyOnWrite()->dict.Set(kSmemAlignmentMap, alignments_);
    function.CopyOnWrite()->attrs = attributes;
    return function;
  }

private:
  struct Resources {
    Array<Stmt> allocations;
    Optional<TensorVar> barrier;
    Array<PrimExpr> arrive_counts;
  };

  Stmt Rewrite(const Stmt &statement) {
    return Mutate(statement).ValueOrUnchanged(statement).as_or_throw<Stmt>();
  }

  Expr RewriteExpr(const Expr &expression) {
    return Mutate(expression).ValueOrUnchanged(expression).as_or_throw<Expr>();
  }

  PrimExpr RewriteScalar(const PrimExpr &expression) {
    return RewriteExpr(expression).as_or_throw<PrimExpr>();
  }

  Array<PrimExpr> RewriteIndices(const Array<PrimExpr> &indices) {
    return indices.Map(
        [this](const PrimExpr &index) { return RewriteScalar(index); });
  }

  Stmt FinishResources(const Stmt &original) {
    Resources frame = std::move(resources_.back());
    resources_.pop_back();
    Stmt body = original;
    if (frame.barrier) {
      TensorVar complete = CreateMBarrierBuffer(frame.barrier.value()->name,
                                                frame.arrive_counts.size());
      auto rebind = make_object<TensorRebinder>(
          Map<TensorVar, TensorVar>{{frame.barrier.value(), complete}});
      body = rebind->Mutate(body).ValueOrUnchanged(body).as_or_throw<Stmt>();
      frame.allocations.push_back(
          Bind(complete,
               Call(std::nullopt, alloc_tensor_op(), {},
                    DictAttrs({{"barrier_init", frame.arrive_counts}}),
                    {complete.ty()}, original->loc),
               original->loc));
    }
    frame.allocations.push_back(body);
    return SeqStmt(frame.allocations, original->loc);
  }

  Array<PrimExpr> AccessIndices(const TensorVar &tensor,
                                const Array<PrimExpr> &indices) {
    Array<PrimExpr> rewritten = RewriteIndices(indices);
    auto layout = layouts_.Get(tensor);
    if (!layout || !remap_.count(tensor))
      return rewritten;
    if (!in_tma_) {
      Array<PrimExpr> result = layout.value()->Forward(rewritten);
      if (result.size() + 1 == remap_[tensor].shape().size()) {
        result.insert(result.begin(), tl::MakeConst(PrimType::Int(32), 0));
      }
      return result;
    }
    Array<PrimExpr> offset = tensor.ty()->ElemOffset(rewritten);
    TVM_FFI_CHECK_EQ(offset.size(), 1U, ValueError)
        << "TMA pointer requires a memory-only tensor layout";
    PrimExpr remaining = offset[0];
    Array<PrimExpr> result;
    const auto &shape = remap_[tensor].shape();
    for (int axis = static_cast<int>(shape.size()) - 1; axis >= 0; --axis) {
      result.insert(
          result.begin(),
          tl::arith::Simplify(analyzer_, floormod(remaining, shape[axis])));
      remaining =
          tl::arith::Simplify(analyzer_, floordiv(remaining, shape[axis]));
    }
    return result;
  }

  UnchangedOr<Expr> Mutate_(const VarNode *node, InplaceMode mode) final {
    if (auto tensor = GetRef<Var>(node).as<TensorVar>()) {
      if (auto physical = remap_.Get(tensor.value()))
        return physical.value();
    }
    return LoopAwareMutator::Mutate_(node, mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *node, InplaceMode) final {
    TensorVar tensor = node->source.as_or_throw<TensorVar>();
    return TensorLoad(remap_.Get(tensor).value_or(tensor),
                      AccessIndices(tensor, node->indices), node->loc);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node, InplaceMode) final {
    TensorVar tensor = node->dest.as_or_throw<TensorVar>();
    return TensorStore(remap_.Get(tensor).value_or(tensor),
                       AccessIndices(tensor, node->indices),
                       RewriteExpr(node->value).as_or_throw<PrimExpr>(),
                       node->loc);
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionNode *node,
                            InplaceMode mode) final {
    TensorVar tensor = node->source.as_or_throw<TensorVar>();
    if (!layouts_.count(tensor) || !remap_.count(tensor)) {
      return LoopAwareMutator::Mutate_(node, mode);
    }
    auto region = node->region.Map([this](const Range &range) {
      return Range::FromMinExtent(RewriteScalar(range->begin),
                                  RewriteScalar(range->extent), range->loc);
    });
    return TensorRegion(remap_[tensor],
                        layouts_[tensor]->MapRegionBounds(region), node->ty,
                        node->loc);
  }

  UnchangedOr<Expr> Mutate_(const CallNode *node, InplaceMode mode) final {
    if (node->op.same_as(tl::region())) {
      AccessRegion access = NormalizeToAccessRegion(GetRef<Call>(node));
      TensorRegion region = Mutate(access.region, mode)
                                .ValueOrUnchanged(access.region)
                                .as_or_throw<TensorRegion>();
      return Call(node->ty, node->op,
                  {region, IntImm::Int32(access.access_mask)}, node->attrs,
                  node->ty_args, node->loc);
    }
    bool previous_tma = in_tma_;
    const auto *operation = node->op.as<OpNode>();
    if (operation) {
      const String &name = operation->name;
      if (name == "tl.tma_load" || name == "tl.tma_store" ||
          name == "tl.tma_load_im2col" || name == "tl.tma_load_multicast" ||
          name == "tl.tma_load_gather4" || name == "tl.tma_store_scatter4") {
        in_tma_ = true;
        has_tma_ = true;
      }
    }
    if (operation && (node->op.same_as(initialize_wgmma_descriptor()) ||
                      node->op.same_as(initialize_tcgen05_descriptor()))) {
      bool tcgen05 = node->op.same_as(initialize_tcgen05_descriptor());
      auto encoding = GetConstInt(node->args[tcgen05 ? 6 : 2]);
      TVM_FFI_CHECK(encoding && (tcgen05 ? (*encoding == 0 || *encoding == 1 ||
                                            *encoding == 2 || *encoding == 4 ||
                                            *encoding == 6)
                                         : (*encoding >= 0 && *encoding <= 3)),
                    ValueError)
          << "Invalid shared-memory matrix descriptor swizzle encoding";
      int ordinal = *encoding == 0 ? 0
                    : tcgen05 && *encoding == 1
                        ? 3
                        : 4 - *encoding / (tcgen05 ? 2 : 1);
      RecordSharedAlignment(GetVarFromAccessPtr(node->args[1]),
                            SwizzleMode::FromOrdinal(ordinal).SmemAlignment());
    }
    Expr result = LoopAwareMutator::Mutate_(node, mode)
                      .ValueOrUnchanged(GetRef<Expr>(node));
    in_tma_ = previous_tma;
    if (operation &&
        (node->op.same_as(access_ptr()) || node->op.same_as(address_of_op()) ||
         node->op.same_as(tensor_data_ptr_op()))) {
      Call call = result.as_or_throw<Call>();
      Call pointer(std::nullopt, call->op, call->args, call->attrs,
                   call->ty_args, call->loc);
      if (options_.rewrite_access_pointer)
        return options_.rewrite_access_pointer(
            pointer, GetRef<sym::Analyzer>(analyzer_));
      return pointer;
    }
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *node, InplaceMode mode) final {
    bindings_.Set(node->var, node->value);
    if (auto tensor = node->var.as<TensorVar>()) {
      if (auto physical = remap_.Get(tensor.value())) {
        const auto *call = node->value.as<CallNode>();
        TVM_FFI_CHECK(call && (call->op.same_as(alloc_tensor_op()) ||
                               call->op.same_as(decl_tensor_op())),
                      ValueError)
            << "Layout-remapped tensors require an explicit native allocation "
               "or declaration";
        Array<Expr> arguments = call->args.Map(
            [this](const Expr &argument) { return RewriteExpr(argument); });
        Optional<Attrs> attributes = call->attrs;
        return Bind(physical.value(),
                    Call(std::nullopt, call->op, arguments, attributes,
                         {physical.value().ty()}, call->loc),
                    node->loc);
      }
    }
    return LoopAwareMutator::Mutate_(node, mode);
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateNode *node, InplaceMode mode) final {
    const auto *call = node->value.as<CallNode>();
    if (call && call->op.as<OpNode>()) {
      const String &name = call->op.as_or_throw<Op>()->name;
      if (name == "tl.layout_hint")
        return SeqStmt({});
      if (name == "tl.safe_value_hint") {
        safe_values_.Set(call->args[0].as_or_throw<Var>(),
                         call->args[1].as_or_throw<PrimExpr>());
        annotations_.Set(attr::kSafeValueMap, safe_values_);
        return LoopAwareMutator::Mutate_(node, mode);
      }
    }
    TileOperator operation = ParseOperator(GetRef<Stmt>(node), annotations_);
    if (!operation.defined())
      return LoopAwareMutator::Mutate_(node, mode);
    if (options_.rewrite_operator)
      operation = options_.rewrite_operator(operation,
                                            GetRef<sym::Analyzer>(analyzer_));
    LowerArgs arguments{
        target_,
        ComputeThreadBounds(thread_, GetRef<sym::Analyzer>(analyzer_)),
        thread_ ? PrimExpr(thread_.value()) : PrimExpr(0),
        layouts_,
        remap_,
        bindings_};
    arguments.mbar_phase_expr = phases_.empty() ? PrimExpr(0) : phases_.back();
    arguments.mbarrier_buffer = &resources_.back().barrier;
    arguments.cluster_size = cluster_size_;
    arguments.add_workspace =
        [this, location = node->loc](int elements, PrimType dtype) -> Expr {
      TVM_FFI_CHECK(elements > 0, ValueError)
          << "Workspace extent must be positive";
      TensorVar tensor("workspace",
                       TensorType({PrimExpr(elements)}, dtype,
                                  runtime::MemoryScope::Get("shared.dyn"), 16));
      resources_.back().allocations.push_back(
          Bind(tensor,
               Call(std::nullopt, alloc_tensor_op(), {}, DictAttrs(),
                    {tensor.ty()}, location),
               location));
      return MakeAccessPtrFromRegion(NormalizeToBufferRegion(tensor),
                                     kAccessReadWrite);
    };
    arguments.alloc_mbarrier = [this](int arrive_count,
                                      std::optional<std::string> name) {
      TVM_FFI_CHECK(arrive_count > 0, ValueError)
          << "Barrier arrive count must be positive";
      auto &frame = resources_.back();
      if (!frame.barrier)
        frame.barrier = CreateMBarrierBuffer(name.value_or("mbarrier"), 1);
      int index = frame.arrive_counts.size();
      frame.arrive_counts.push_back(PrimExpr(arrive_count));
      return index;
    };
    arguments.update_barrier_arrive = [this](Var tensor, PrimExpr count) {
      barrier_updates_.Set(tensor, count);
    };
    arguments.require_smem_alignment = [this](Var tensor, int alignment) {
      RecordSharedAlignment(tensor, alignment);
    };
    Stmt lowered =
        operation->Lower(arguments, GetRef<sym::Analyzer>(analyzer_));
    StampSubtreeSpans(lowered, node->loc);
    return Rewrite(lowered);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                            InplaceMode mode) final {
    auto previous_thread = thread_;
    auto previous_annotations = annotations_;
    auto previous_safe_values = safe_values_;
    int previous_cluster_size = cluster_size_;
    bool thread_scope =
        node->op.same_as(launch_thread_op()) &&
        node->args[0].as_or_throw<StringImm>()->value == "threadIdx.x";
    if (thread_scope)
      thread_ = node->body_params[0].as_or_throw<PrimVar>();
    bool lexical_scope =
        node->op->name == "tl.scope" || node->op.same_as(::tvm::tl::scope_op());
    bool resource_scope = lexical_scope || thread_scope;
    if (resource_scope)
      resources_.emplace_back();
    if (lexical_scope) {
      for (const auto &[key, value] : node->attrs->dict)
        annotations_.Set(key, value);
      if (auto safe =
              node->attrs.GetAttr<Map<Var, PrimExpr>>(attr::kSafeValueMap)) {
        for (const auto &[tensor, value] : safe.value())
          safe_values_.Set(tensor, value);
      }
      if (auto dimensions =
              node->attrs.GetAttr<Array<PrimExpr>>("cluster_dims")) {
        cluster_size_ = StaticElements(dimensions.value(), "cluster_dims");
      }
    }
    RegionStmt result = LoopAwareMutator::Mutate_(node, mode)
                            .ValueOrUnchanged(GetRef<Stmt>(node))
                            .as_or_throw<RegionStmt>();
    if (resource_scope)
      result.CopyOnWrite()->body = SeqStmt(FinishResources(result->body));
    if (BoolAnnotation(node->attrs->dict, kPipelineCPAsyncScope)) {
      TVM_FFI_CHECK(TargetCudaHasAsyncCopy(target_), ValueError)
          << "Pipeline-managed cp.async requires a compatible CUDA target";
      result.CopyOnWrite()->body =
          SeqStmt(InjectPTXAsyncCopy(result->body, true,
                                     GetRef<sym::Analyzer>(analyzer_))
                      .stmt);
    }
    DictAttrs attributes = result->attrs;
    attributes.CopyOnWrite()->dict.erase(attr::kLayoutMap);
    attributes.CopyOnWrite()->dict.erase(kPipelineCPAsyncScope);
    result.CopyOnWrite()->attrs = attributes;
    if (node->op->name == "tl.scope")
      result.CopyOnWrite()->op = ::tvm::tl::scope_op();
    thread_ = previous_thread;
    annotations_ = previous_annotations;
    safe_values_ = previous_safe_values;
    cluster_size_ = previous_cluster_size;
    return result;
  }

  static bool BoolAnnotation(const Map<String, Any> &annotations,
                             const char *key) {
    auto annotation = annotations.Get(key);
    if (!annotation)
      return false;
    auto value = GetIntAnnotation(annotation.value());
    TVM_FFI_CHECK(value.has_value(), TypeError)
        << "Loop annotation `" << key << "` must be boolean";
    return *value != 0;
  }

  UnchangedOr<Stmt> Mutate_(const ForNode *node, InplaceMode mode) final {
    bool serial = node->kind == ForKind::kDefault;
    if (serial)
      phases_.push_back(tl::arith::Simplify(
          analyzer_, FloorMod(FloorDiv(node->var - node->begin, node->step),
                              tl::MakeConst(node->var.ty(), 2))));
    For loop = LoopAwareMutator::Mutate_(node, mode)
                   .ValueOrUnchanged(GetRef<Stmt>(node))
                   .as_or_throw<For>();
    if (serial)
      phases_.pop_back();
    if (node->kind != ForKind::kParallel ||
        !node->annotations.count(attr::kParallelLoopLayout))
      return loop;
    bool partition = false;
    PostOrderVisit(GetRef<For>(node), [&](const ObjectRef &object) {
      if (const auto *load = object.as<TensorLoadNode>()) {
        partition = partition ||
                    IsFragmentBuffer(load->source.as_or_throw<TensorVar>());
      }
      if (const auto *store = object.as<TensorStoreNode>()) {
        TensorVar tensor = store->dest.as_or_throw<TensorVar>();
        partition =
            partition || IsFragmentBuffer(tensor) || !IsLocalBuffer(tensor);
      }
      if (const auto *call = object.as<CallNode>();
          call && (call->op.same_as(access_ptr()) ||
                   call->op.same_as(address_of_op()))) {
        if (const auto *load = call->args[0].as<TensorLoadNode>()) {
          partition = partition ||
                      !IsLocalBuffer(load->source.as_or_throw<TensorVar>());
        }
      }
    });
    Fragment layout =
        node->annotations.at(attr::kParallelLoopLayout).as_or_throw<Fragment>();
    Optional<PrimExpr> predicate;
    if (auto value = node->annotations.Get(attr::kParallelLoopPredicate))
      predicate = value.value().as_or_throw<PrimExpr>();
    Stmt lowered = LowerParallelLoop(
        loop, layout, thread_ ? PrimExpr(thread_.value()) : PrimExpr(0),
        GetRef<sym::Analyzer>(analyzer_), layouts_, predicate, partition,
        BoolAnnotation(node->annotations,
                       attr::kParallelLoopRequiresPaddingGuard),
        true);
    if (TargetCudaHasAsyncCopy(target_)) {
      bool automatic = tvm::transform::PassContext::Current()
                           ->GetConfig<bool>(kEnableAsyncCopy)
                           .value_or(true);
      bool explicit_async = BoolAnnotation(
          node->annotations, attr::kParallelAsyncWithoutAsyncCommitWait);
      if (BoolAnnotation(node->annotations, attr::kLoopPreferAsync) ||
          (automatic && explicit_async)) {
        lowered = InjectPTXAsyncCopy(lowered, explicit_async,
                                     GetRef<sym::Analyzer>(analyzer_))
                      .stmt;
      }
    }
    StampSubtreeSpans(lowered, node->loc);
    return lowered;
  }

  Target target_;
  LowerTileOpOptions options_;
  std::unordered_set<TensorVar, ObjectPtrHash, ObjectPtrEqual> tensors_;
  LayoutMap layouts_;
  Map<TensorVar, TensorVar> remap_;
  Map<Var, Expr> bindings_;
  Map<Var, PrimExpr> safe_values_;
  Map<String, Any> annotations_;
  Optional<PrimVar> thread_;
  std::deque<Resources> resources_;
  std::vector<PrimExpr> phases_;
  Map<Var, PrimExpr> barrier_updates_;
  void RecordSharedAlignment(const Var &tensor, int alignment) {
    auto previous = alignments_.Get(tensor->name);
    if (!previous || GetConstInt(previous.value()).value_or(0) < alignment) {
      alignments_.Set(tensor->name, IntImm(PrimType::Int(32), alignment));
    }
  }

  Map<String, IntImm> alignments_;
  int cluster_size_{1};
  bool in_tma_{false};
  bool has_tma_{false};
};

} // namespace

tirx::Function LowerTileOperators(tirx::Function function,
                                  LowerTileOpOptions options) {
  auto target = function->GetAttr<Target>("target");
  TVM_FFI_CHECK(target.has_value(), ValueError)
      << "LowerTileOp requires a target attribute";
  sym::Analyzer analyzer;
  return make_object<LowerTileOpPass>(analyzer, target.value(),
                                      std::move(options))
      ->Run(function);
}

namespace transform {
tvm::transform::Pass LowerTileOp() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> Optional<tirx::Function> {
    return LowerTileOperators(std::move(function));
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LowerTileOp");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.LowerTileOp", LowerTileOp);
}
} // namespace transform

} // namespace tvm::tl
