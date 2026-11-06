/*!
 * \file lower_tile_op.cc
 * \brief Lower TileLang operators and logical layouts into native typed IR.
 */
#include "ascend/ir/block.h"
#include "ir/lexical_scope.h"
#include "ir/stmt_functor.h"
#include "layout/expr_utils.h"
#include "op/builtin.h"
#include "support/check.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../config.h"
#include "../ir/tensor_utils.h"
#include "../op/builtin.h"
#include "../op/operator.h"
#include "../op/reducer.h"
#include "../op/utils.h"
#include "../span_utils.h"
#include "common/mbarrier.h"
#include "common/pipeline_utils.h"
#include "cuda/op/builtin.h"
#include "cuda/target_utils.h"
#include "cuda/transform/ptx_async_copy_injector.h"
#include "ir/alias_analysis.h"
#include "ir/annotations.h"
#include "ir/loop_annotations.h"
#include "ir/tensor_utils.h"
#include "loop_partition.h"
#include "loop_vectorize.h"

namespace tvm::tl {
using namespace tirx;
using namespace prim;
using namespace ffi;

namespace {

String OpName(const CallObj *call) {
  auto op = call->op.as<Op>();
  return op ? op.value()->name : String();
}

bool IsTensorMapCall(const CallObj *call) {
  return call->op.same_as(tl::tma_load()) ||
         call->op.same_as(tl::tma_load_im2col()) ||
         call->op.same_as(tl::tma_load_multicast()) ||
         call->op.same_as(tl::tma_store()) ||
         call->op.same_as(tl::tma_load_gather4()) ||
         call->op.same_as(tl::tma_store_scatter4());
}

int64_t ConstantProduct(const Array<PrimExpr> &shape, const TensorVar &tensor) {
  int64_t size = 1;
  for (const auto &extent : shape) {
    auto constant = TryConstInt(extent);
    TVM_FFI_CHECK(constant.has_value(), ValueError)
        << "Shared buffer `" << tensor->name
        << "` has a symbolic layout extent " << extent
        << "; shared layouts require compile-time constant tile extents";
    TVM_FFI_CHECK_GE(*constant, 0, ValueError);
    TVM_FFI_CHECK(*constant == 0 ||
                      size <= std::numeric_limits<int64_t>::max() / *constant,
                  ValueError)
        << "Shared layout extent product overflows int64";
    size *= *constant;
  }
  return size;
}

TensorVar MakePhysicalTensor(const TensorVar &tensor,
                             const Optional<Layout> &layout) {
  Array<PrimExpr> shape =
      layout ? layout.value()->OutputShape() : tensor.shape();
  auto type = make_object<TensorTypeObj>(*tensor.ty().get());
  if (layout) {
    if (IsSharedBuffer(tensor)) {
      int64_t logical = ConstantProduct(tensor.shape(), tensor);
      int64_t physical = ConstantProduct(shape, tensor);
      TVM_FFI_CHECK_GT(physical, 0, ValueError);
      int64_t replicas = logical / physical;
      if (replicas > 1)
        shape.insert(shape.begin(), IntImm::Int64(replicas));
    }
    type->shape = shape;
    // TileLang's layout determines physical coordinates. The prior strided
    // view contributes its base element offset once; retaining its strides as
    // well would apply the logical-to-physical transform twice.
    auto old_pattern = MatchShapeStridePattern(tensor.ty());
    PrimExpr offset = IntImm(tensor.ty()->DefaultIndexType(), 0);
    if (old_pattern) {
      offset = old_pattern.value().get<2>();
    } else {
      // Tile() canonicalizes adjacent contiguous shard iterators. A versioned
      // (stage, row, column) tensor can therefore have a single native memory
      // iterator without having lost its origin. The shape/stride matcher is
      // deliberately stricter than this address-origin query.
      auto native_tile = tensor.ty()->layout.as<TileLayoutObj>();
      TVM_FFI_CHECK(native_tile &&
                        GetRef<TileLayout>(native_tile).IsContiguous(),
                    ValueError)
          << "Layout remapping requires an explicit base offset for " << tensor;
      Array<PrimExpr> zero;
      for (const auto &extent : tensor.shape())
        zero.push_back(IntImm(extent.ty(), 0));
      offset = TensorElementOffset(tensor, zero);
    }
    type->layout = std::nullopt;
    if (!IsZero(offset)) {
      Array<Iter> iters;
      PrimExpr stride = IntImm(offset.ty(), 1);
      for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        iters.insert(iters.begin(), Iter(shape[i], stride, Axis::Get("m")));
        stride = stride * shape[i];
      }
      type->layout = TileLayout(iters, {}, {{Axis::Get("m"), offset}});
    }
  }
  if (IsFragmentBuffer(tensor))
    type->scope = runtime::MemoryScope::Get("local");
  return TensorVar(tensor->name, TensorType(type), tensor->loc);
}

Bind AllocateTensor(const TensorVar &tensor, Location loc) {
  return Bind(tensor,
              Call(tensor.ty(), alloc_tensor_op(), {}, {}, {tensor.ty()}, loc),
              loc);
}

void RetireLoweredLoopHints(Map<String, Any> *metadata) {
  // These directives describe tile/parallel lowering, which is complete at
  // this boundary. Scheduling directives with later owners remain intact.
  for (const char *key :
       {attr::kLoopPreferAsync, attr::kParallelAsyncWithoutAsyncCommitWait,
        attr::kParallelLoopPredicate, attr::kParallelLoopRequiresPaddingGuard,
        "prefer_async", "async_scope", "disable_tma", "disable_async",
        "coalesced_width", "vectorize", "parallel_loop_layout", "predicate",
        "padding_guard", "loop_layout", "tl.ws_op_id"})
    metadata->erase(key);
}

// Layouts have become physical tensor types and rewritten accesses by the
// end of LowerTileOp, including layouts introduced by parallel-loop lowering.
// Retire consumed directives at their original carriers, keeping unrelated
// annotations and definition scopes. No Function completion marker is needed.
class ConsumedTileMetadataRetirer : public LayoutMapMutator {
public:
  using LayoutMapMutator::Mutate_;
  static void CheckConsumed(const Any &layouts) {
    for (const auto &[tensor, layout] : layouts.cast<LayoutMap>()) {
      TVM_FFI_CHECK(!IsReducerV2Buffer(tensor) &&
                        !layout.as<PartialFragmentNode>(),
                    ValueError)
          << "Partial reducer layout survived past ReducerPlanAndMaterialize "
             "for "
          << tensor;
    }
  }
  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    const auto *call = op->value.as<CallObj>();
    if (!call || !IsAnnotationCall(call))
      return LayoutMapMutator::Mutate_(op, mode);
    auto metadata = ReadAnnotations(call);
    if (metadata.empty())
      return SeqStmt(Array<Stmt>{}, op->loc);
    if (!metadata.count(attr::kLayoutMap))
      return ffi::Unchanged();
    CheckConsumed(metadata.at(attr::kLayoutMap));
    metadata.erase(attr::kLayoutMap);
    if (metadata.empty())
      return SeqStmt(Array<Stmt>{}, op->loc);
    return MakeAnnotation(metadata, op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    auto rewritten = LayoutMapMutator::Mutate_(op, mode)
                         .ValueOrUnchanged(GetRef<Stmt>(op))
                         .as_or_throw<RegionStmt>();
    if (op->op->name != "tl.attr")
      return rewritten;
    auto fields = ReadAnnotations(rewritten.get());
    const String key = fields.at("key").cast<String>();
    if (key == attr::kLayoutMap) {
      CheckConsumed(fields.at("value"));
      return RegionStmt(tl::lexical_scope_op(), {}, {}, DictAttrs(),
                        rewritten->body, {}, op->loc);
    }
    if (key == "tl.kernel") {
      auto metadata = fields.at("value").cast<Map<String, Any>>();
      if (metadata.count(attr::kLayoutMap)) {
        CheckConsumed(metadata.at(attr::kLayoutMap));
        metadata.erase(attr::kLayoutMap);
        return MakeAttribute(fields.at("node"), key, metadata, rewritten->body,
                             op->loc);
      }
    }
    if (key == "tl.loop.annotations") {
      auto metadata = fields.at("value").cast<Map<String, Any>>();
      const auto old_size = metadata.size();
      RetireLoweredLoopHints(&metadata);
      if (metadata.empty())
        return MakeLexicalScope(rewritten->body, op->loc);
      if (metadata.size() != old_size)
        return MakeAttribute(fields.at("node"), key, metadata, rewritten->body,
                             op->loc);
    }
    return rewritten;
  }

  UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    auto rewritten = GetRef<For>(op);
    auto annotations = rewritten->annotations;
    RetireLoweredLoopHints(&annotations);
    if (annotations.size() != rewritten->annotations.size())
      rewritten.CopyOnWrite()->annotations = annotations;
    // Dynamic loop directives can be declared after body-local definitions.
    // Rewrite each carrier in place, never aggregate them into the For header.
    rewritten = RewriteLoopAnnotationCarriers(
        rewritten, [](const Map<String, Any> &fields) {
          auto remaining = fields;
          RetireLoweredLoopHints(&remaining);
          return remaining;
        });
    // Traverse after rewriting so empty statements/wrappers are retired by
    // the normal annotation visitors without flattening definition scopes.
    return LayoutMapMutator::Mutate_(rewritten.get(), mode)
        .ValueOrUnchanged(Stmt(rewritten));
  }
};

// Barrier cardinality is known only after all op lowerings have requested
// their slots. Replace the temporary definition and its uses atomically, then
// place the final allocation inside the kernel's real thread scope.
class BarrierInjector : public LayoutMapMutator {
public:
  BarrierInjector(TensorVar tensor, Array<PrimExpr> counts)
      : tensor_(std::move(tensor)), counts_(std::move(counts)) {}
  using LayoutMapMutator::Mutate_;
  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    if (!injected_ && op->op.same_as(launch_thread_op()) &&
        op->args[0].as_or_throw<StringImm>()->value == "threadIdx.x") {
      injected_ = true;
      SeqStmt body(
          {AllocateTensor(tensor_, op->loc),
           MakeAnnotation({{"barrier_init",
                            Map<Var, Array<PrimExpr>>{{tensor_, counts_}}}},
                          op->loc),
           op->body},
          op->loc);
      auto copy = make_object<RegionStmtObj>(*op);
      copy->body = body;
      return Stmt(copy);
    }
    return LayoutMapMutator::Mutate_(op, mode);
  }
  bool injected_{false};

private:
  TensorVar tensor_;
  Array<PrimExpr> counts_;
};

class LowerTileOpPass : public StmtExprMutatorWithAnalyzer {
public:
  explicit LowerTileOpPass(sym::AnalyzerObj *analyzer)
      : StmtExprMutatorWithAnalyzer(analyzer) {}
  using StmtExprMutatorWithAnalyzer::Mutate_;

  static tirx::Function Apply(tirx::Function function) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    sym::Analyzer analyzer;
    auto lowerer = make_object<LowerTileOpPass>(analyzer.get());
    auto target = function->GetAttr<Target>(tvm::attr::kTarget);
    TVM_FFI_CHECK(target.has_value(), ValueError)
        << "LowerTileOp requires a target";
    lowerer->target_ = *target;
    lowerer->aliases_.Collect(function);
    lowerer->CollectLayouts(function->body);
    // The public function keeps its logical tensor parameters and ABI. A
    // physical view in the body shares their storage but carries the chosen
    // layout's physical shape; it does not alter parameter shape checks.
    Array<Stmt> prefix;
    for (const Var &param : function->params) {
      if (auto tensor = param.as<TensorVar>()) {
        if (auto replacement = lowerer->buffer_remap_.Get(*tensor)) {
          prefix.push_back(
              Bind(*replacement,
                   Call(replacement.value().ty(), decl_tensor_op(),
                        {tensor.value().data()}, {}, {replacement.value().ty()},
                        tensor.value()->loc),
                   tensor.value()->loc));
        }
      }
    }
    lowerer->workspace_stack_.emplace_back();
    SeqStmt body =
        lowerer->Mutate(function->body).ValueOrUnchanged(function->body);
    for (const TensorVar &workspace : lowerer->workspace_stack_.back())
      prefix.push_back(AllocateTensor(workspace, function->loc));
    lowerer->workspace_stack_.pop_back();
    prefix.push_back(body);
    body = SeqStmt(prefix, function->loc);
    // Apply pending barrier arrive-count changes after all sites have lowered.
    body = lowerer->RewriteBarrierMetadata(body);
    if (lowerer->mbarrier_count_ > 0) {
      auto old = lowerer->mbarrier_buffer_.value();
      TensorVar final(old->name,
                      TensorType({IntImm::Int32(lowerer->mbarrier_count_)},
                                 old.dtype(), old.ty()->scope,
                                 old.ty()->data_alignment),
                      old->loc);
      auto replacer = make_object<LayoutMapMutator>();
      replacer->VarRemapSet(old, final);
      body = replacer->Mutate(body).ValueOrUnchanged(body);
      Array<PrimExpr> counts;
      for (int n : lowerer->mbarrier_arrive_counts_)
        counts.push_back(IntImm::Int32(n));
      auto injector = make_object<BarrierInjector>(final, counts);
      body = injector->Mutate(body).ValueOrUnchanged(body);
      TVM_FFI_CHECK(injector->injected_, ValueError)
          << "Compiler-generated mbarrier requires a threadIdx.x kernel scope";
    }
    auto retire_metadata = make_object<ConsumedTileMetadataRetirer>();
    body = retire_metadata->Mutate(body).ValueOrUnchanged(body);
    auto node = function.CopyOnWrite();
    node->body = body;
    auto attributes = node->attrs->dict;
    attributes.Set(kHasTMA, lowerer->has_tma_);
    if (!lowerer->smem_alignment_map_.empty())
      attributes.Set(kSmemAlignmentMap, lowerer->smem_alignment_map_);
    node->attrs = DictAttrs(attributes);
    return function;
  }

  UnchangedOr<Expr> Mutate_(const VarObj *op, InplaceMode mode) final {
    if (auto tensor = GetRef<Var>(op).as<TensorVar>()) {
      if (auto replacement = buffer_remap_.Get(*tensor)) {
        // A physical tensor can itself have been rebuilt at its definition
        // after a type child changed. Honor that native binding transaction.
        return StmtExprMutatorWithAnalyzer::Mutate_(replacement->get(), mode)
            .ValueOrUnchanged(Expr(*replacement));
      }
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *op,
                                InplaceMode mode) final {
    TensorVar original = op->source.as_or_throw<TensorVar>();
    // Native Bind rebuilding records the replacement of types containing
    // mutable shape/stride/offset expressions. Ordinary tensor accesses must
    // consume that remap rather than keep the old definition's identity.
    if (!buffer_remap_.count(original) && !layout_map_.count(original))
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    auto indices = MutateIndices(op->indices, mode);
    TensorVar physical = Mutate(original, mode)
                             .ValueOrUnchanged(original)
                             .as_or_throw<TensorVar>();
    if (auto layout = layout_map_.Get(original))
      indices = ForwardIndices(original, indices, *layout);
    return TensorLoad(physical, indices, op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj *op, InplaceMode mode) final {
    TensorVar original = op->dest.as_or_throw<TensorVar>();
    // Native Bind rebuilding records the replacement of types containing
    // mutable shape/stride/offset expressions. Ordinary tensor accesses must
    // consume that remap rather than keep the old definition's identity.
    if (!buffer_remap_.count(original) && !layout_map_.count(original))
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    auto indices = MutateIndices(op->indices, mode);
    TensorVar physical = Mutate(original, mode)
                             .ValueOrUnchanged(original)
                             .as_or_throw<TensorVar>();
    if (auto layout = layout_map_.Get(original))
      indices = ForwardIndices(original, indices, *layout);
    PrimExpr value = Mutate(op->value, mode).ValueOrUnchanged(op->value);
    return TensorStore(physical, indices, value, op->loc);
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionObj *op, InplaceMode mode) final {
    TensorVar original = op->source.as_or_throw<TensorVar>();
    // Native Bind rebuilding records the replacement of types containing
    // mutable shape/stride/offset expressions. Ordinary tensor accesses must
    // consume that remap rather than keep the old definition's identity.
    if (!buffer_remap_.count(original) && !layout_map_.count(original))
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    Array<Range> ranges;
    for (const Range &range : op->region) {
      ranges.push_back(Range::FromMinExtent(
          Mutate(range->begin, mode).ValueOrUnchanged(range->begin),
          Mutate(range->extent, mode).ValueOrUnchanged(range->extent)));
    }
    if (auto layout = layout_map_.Get(original))
      ranges = layout.value()->MapRegionBounds(ranges);
    return TensorRegion(Mutate(original, mode).ValueOrUnchanged(original),
                        ranges, op->ty, op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    bind_var_to_expr_.Set(op->var, op->value);
    if (auto tensor = op->var.as<TensorVar>()) {
      if (auto replacement = buffer_remap_.Get(*tensor)) {
        const auto *call = op->value.as<CallObj>();
        TVM_FFI_CHECK(call && (call->op.same_as(alloc_tensor_op()) ||
                               call->op.same_as(decl_tensor_op())),
                      ValueError)
            << "A layout-changing tensor definition must be alloc_tensor or "
               "decl_tensor";
        Optional<Attrs> attributes = call->attrs;
        if (attributes) {
          auto values = attributes.value().as_or_throw<DictAttrs>()->dict;
          if (values.count("tl.metal_simdgroup_physical_elements")) {
            values.erase("tl.metal_simdgroup_physical_elements");
            attributes = DictAttrs(values);
          }
        }
        Expr value = Call(replacement.value().ty(), call->op, call->args,
                          attributes, {replacement.value().ty()}, call->loc);
        Bind physical(*replacement, value, op->loc);
        // Physical shapes and layout offsets can still contain reads of
        // another tensor that was remapped at its definition. Rebuild the
        // entire binding through the native transaction, so its value type,
        // definition and later uses share the same updated metadata operands.
        auto rebuilt =
            StmtExprMutatorWithAnalyzer::Mutate_(physical.get(), mode)
                .ValueOrUnchanged(Stmt(physical))
                .as_or_throw<Bind>();
        buffer_remap_.Set(*tensor, rebuilt->var.as_or_throw<TensorVar>());
        return rebuilt;
      }
    }
    // The native Bind mutator updates a value's changed PtrType and all uses
    // of that binding. Populate the scalar analyzer only after this
    // transaction.
    auto result = StmtExprMutatorWithAnalyzer::Mutate_(op, mode)
                      .ValueOrUnchanged(GetRef<Stmt>(op))
                      .as_or_throw<Bind>();
    // The native collector records the resulting binding for later queries.
    return result;
  }

  UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    String name = OpName(op);
    if (IsTensorMapCall(op)) {
      has_tma_ = true;
      bool previous = in_tma_context_;
      in_tma_context_ = true;
      auto result = StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
      in_tma_context_ = previous;
      return result;
    }
    if (op->op.same_as(tl::access_ptr()) || op->op.same_as(tl::ptr_access()))
      return RewriteAccess(op, std::nullopt, mode);
    auto operands = DecodeCallOperands(op);
    // The legacy seven-slot target intrinsic carries a separate element
    // offset. Fold it through the logical layout exactly once before emission.
    if ((name == "tirx.ptx_ldmatrix" || name == "tir.ptx_ldmatrix") &&
        operands.size() >= 7) {
      Array<Expr> args = op->args;
      for (size_t i = 0; i < args.size(); ++i) {
        if (i != 5 && i != 6)
          args.Set(i, Mutate(args[i], mode).ValueOrUnchanged(args[i]));
      }
      const auto *pointer = operands[5].as<CallObj>();
      TVM_FFI_CHECK(pointer, ValueError)
          << "ldmatrix requires a typed tensor address";
      args.Set(
          5, RewriteAccess(pointer, operands[6].as_or_throw<PrimExpr>(), mode));
      args.Set(6, IntImm(operands[6].as_or_throw<PrimExpr>().ty(), 0));
      return Call(op->ty, op->op, args, op->attrs, op->ty_args, op->loc);
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    const auto *call = op->value.as<CallObj>();
    if (call && IsAnnotationCall(call)) {
      auto metadata = ReadAnnotations(call);
      if (!metadata.count("tl.source_block_annotation"))
        RecordMetadata(metadata);
      return RewriteAnnotation(metadata, op->loc);
    }
    TileOperator tile = ParseOperator(GetRef<Stmt>(op), block_annotations_);
    if (!tile.defined())
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    LowerArgs args{.target = target_, .inside_simt_vf = inside_simt_vf_};
    args.thread_bounds = CurrentThreadBounds();
    args.thread_index = CurrentThreadIndex();
    args.layout_map = layout_map_;
    args.buffer_remap = buffer_remap_;
    args.bind_var_to_expr = bind_var_to_expr_;
    args.block_annotations = block_annotations_;
    args.mbar_phase_expr = loop_mbar_phase_stack_.empty()
                               ? PrimExpr(0)
                               : loop_mbar_phase_stack_.back();
    args.mbarrier_buffer = &mbarrier_buffer_;
    args.cluster_size = cluster_size_;
    args.add_workspace = [this](int size, PrimType dtype) -> Expr {
      TVM_FFI_CHECK_GE(size, 0, ValueError);
      TensorVar workspace("workspace",
                          TensorType({IntImm::Int32(size)}, dtype,
                                     runtime::MemoryScope::Get("shared.dyn")));
      workspace_stack_.back().push_back(workspace);
      return MakeTensorAccessPtr(TensorLoad(workspace, {0}),
                                 IntImm::Int32(size), kAccessReadWrite);
    };
    args.alloc_mbarrier = [this](int count, std::optional<std::string> name) {
      if (!mbarrier_buffer_)
        mbarrier_buffer_ =
            CreateMBarrierBuffer(name.value_or(injected_mbarrier_name_), 1);
      int id = mbarrier_count_++;
      mbarrier_arrive_counts_.push_back(count);
      return id;
    };
    args.update_barrier_arrive = [this](TensorVar tensor, PrimExpr count) {
      barrier_arrive_updates_.Set(buffer_remap_.Get(tensor).value_or(tensor),
                                  count);
    };
    args.require_smem_alignment = [this](TensorVar tensor, int alignment) {
      auto previous = smem_alignment_map_.Get(tensor->name);
      if (!previous || alignment > previous.value().cast<int64_t>())
        smem_alignment_map_.Set(tensor->name, alignment);
    };
    Stmt lowered = tile->Lower(args, GetSyncedAnalyzer());
    StampSubtreeSpans(lowered, op->loc);
    return Mutate(lowered, mode).ValueOrUnchanged(lowered);
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    auto previous = block_annotations_;
    auto result = StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    block_annotations_ = previous;
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    const bool previous_simt_vf = inside_simt_vf_;
    if (ascend::GetVFKind(op) != ascend::VFKind::kNone)
      inside_simt_vf_ = ascend::GetVFKind(op) == ascend::VFKind::kSimt;
    auto previous_thread = thread_binding_;
    auto previous_loop = loop_annotations_;
    auto previous_block = block_annotations_;
    int previous_cluster = cluster_size_;
    bool workspace_scope = op->op.same_as(tl::lexical_scope_op());
    bool loop_region = false;
    Optional<Map<String, Any>> attr_fields;
    if (op->op.same_as(launch_thread_op()) &&
        op->args[0].as_or_throw<StringImm>()->value == "threadIdx.x") {
      thread_binding_ = op->body_params[0].as_or_throw<PrimVar>();
      // Lowered native functions may have no enclosing tl.kernel annotation.
      // Scratch storage requested by an operation must still be allocated
      // inside the device region, not in the host function before splitting.
      workspace_scope = true;
    }
    if (ascend::IsAscendBlock(op) || IsLexicalScope(op)) {

      RecordMetadata(CollectSourceBlockAnnotations(op->body));
    }
    if (op->op->name == "tl.attr") {
      attr_fields = ReadAnnotations(op);
      String key = attr_fields.value().at("key").cast<String>();
      if (key == "tl.loop.annotations") {
        // LayoutInference adds a nested wrapper around the same loop. Merge
        // its inferred fields with outer user hints; inner values take
        // priority.
        for (const auto &[name, value] :
             attr_fields.value().at("value").cast<Map<String, Any>>())
          loop_annotations_.Set(name, value);
        loop_region = true;
      } else if (key == "tl.kernel") {
        RecordMetadata(
            attr_fields.value().at("value").cast<Map<String, Any>>());
        workspace_scope = true;
      }
      if (key == attr::kLexicalAllocScope) {
        // InlineTileOpFunction carries this scope as a tl.attr region.
        // Workspaces requested inside the helper belong to this lexical body.
        workspace_scope = true;
      } else
        RecordMetadata({{key, attr_fields.value().at("value")}});
      if (key == "tl.kernel")
        RecordMetadata(CollectSourceBlockAnnotations(op->body));
    }
    if (workspace_scope)
      workspace_stack_.emplace_back();
    auto rewritten = StmtExprMutatorWithAnalyzer::Mutate_(op, mode)
                         .ValueOrUnchanged(GetRef<Stmt>(op))
                         .as_or_throw<RegionStmt>();
    if (workspace_scope) {
      Array<Stmt> prefix;
      for (const auto &workspace : workspace_stack_.back())
        prefix.push_back(AllocateTensor(workspace, op->loc));
      workspace_stack_.pop_back();
      prefix.push_back(rewritten->body);
      rewritten.CopyOnWrite()->body = SeqStmt(prefix, op->loc);
    }
    inside_simt_vf_ = previous_simt_vf;
    thread_binding_ = previous_thread;
    loop_annotations_ = previous_loop;
    block_annotations_ = previous_block;
    cluster_size_ = previous_cluster;
    // Tile and parallel-loop directives have now been consumed. Other
    // annotations remain explicit operands for their own consumers.
    if (loop_region) {
      auto fields = ReadAnnotations(rewritten.get());
      auto metadata = fields.at("value").cast<Map<String, Any>>();
      RetireLoweredLoopHints(&metadata);
      if (metadata.empty())
        return MakeLexicalScope(rewritten->body, op->loc);
      return MakeAttribute(fields.at("node"), "tl.loop.annotations", metadata,
                           rewritten->body, op->loc);
    }
    return rewritten;
  }

  UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    auto enclosing = GetSyncedAnalyzer()->Clone();
    bool serial = op->kind == ForKind::kDefault;
    if (serial)
      loop_mbar_phase_stack_.push_back(GetSyncedAnalyzer()->Simplify(floormod(
          floordiv(op->var - op->begin, op->step), IntImm(op->var.ty(), 2))));
    Map<String, Any> metadata = ReadLoopAnnotations(op);
    for (const auto &[key, value] : loop_annotations_)
      metadata.Set(key, value);
    auto previous = loop_annotations_;
    if (op->kind == ForKind::kParallel)
      loop_annotations_ = {};
    For rewritten = StmtExprMutatorWithAnalyzer::Mutate_(op, mode)
                        .ValueOrUnchanged(GetRef<Stmt>(op))
                        .as_or_throw<For>();
    loop_annotations_ = previous;
    if (serial)
      loop_mbar_phase_stack_.pop_back();
    if (op->kind != ForKind::kParallel ||
        !metadata.count(attr::kParallelLoopLayout))
      return rewritten;
    Fragment layout = metadata.at(attr::kParallelLoopLayout).cast<Fragment>();
    Optional<PrimExpr> predicate;
    if (auto value = metadata.Get(attr::kParallelLoopPredicate))
      predicate = value->cast<PrimExpr>();
    bool partition = false;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        GetRef<For>(op),
        [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
          if (const auto *load = node.as<TensorLoadObj>()) {
            if (IsFragmentBuffer(load->source.as_or_throw<TensorVar>()))
              partition = true;
          } else if (const auto *store = node.as<TensorStoreObj>()) {
            auto tensor = store->dest.as_or_throw<TensorVar>();
            if (IsFragmentBuffer(tensor) || !IsLocalBuffer(tensor))
              partition = true;
          } else if (auto expr = node.as<Expr>();
                     expr && expr.value()->ty.as<PtrTypeObj>()) {
            // Opaque calls can write through native addresses, including a
            // pointer bound outside the loop. Those accesses need the same
            // ownership as an explicit nonlocal TensorStore.
            partition |= PointerRequiresPartition(*expr);
          }
          return ffi::WalkResult::Advance();
        });
    Stmt lowered = LowerParallelLoop(
        rewritten, layout, CurrentThreadIndex(), enclosing.get(), layout_map_,
        predicate, partition,
        ReadBoolean(metadata, attr::kParallelLoopRequiresPaddingGuard),
        target_);
    if (TargetCudaHasAsyncCopy(target_)) {
      bool without_wait =
          ReadBoolean(metadata, attr::kParallelAsyncWithoutAsyncCommitWait);
      bool enabled = tvm::transform::PassContext::Current()
                         ->GetConfig<bool>(kEnableAsyncCopy)
                         .value_or(true);
      if (ReadBoolean(metadata, attr::kLoopPreferAsync) ||
          (enabled && without_wait))
        lowered = InjectPTXAsyncCopy(lowered, without_wait).stmt;
    }
    StampSubtreeSpans(lowered, op->loc);
    return lowered;
  }

private:
  static bool ReadBoolean(const Map<String, Any> &metadata, const char *key) {
    auto value = metadata.Get(key);
    if (!value)
      return false;
    if (auto flag = value->as<bool>())
      return *flag;
    if (auto integer = value->as<int64_t>())
      return *integer != 0;
    if (auto expr = value->as<PrimExpr>())
      if (auto constant = TryConstInt(*expr))
        return *constant != 0;
    LOG(WARNING) << "Loop annotation `" << key
                 << "` expects Bool, ignoring override";
    return false;
  }

  void CollectLayouts(const Stmt &body) {
    auto collect = [this](const Map<String, Any> &metadata) {
      if (auto layouts = metadata.Get(attr::kLayoutMap)) {
        for (const auto &[tensor, layout] : layouts->cast<LayoutMap>()) {
          if (auto previous = layout_map_.Get(tensor))
            TVM_FFI_CHECK(previous.value()->IsEqual(layout.get()), ValueError)
                << "Conflicting layouts for tensor " << tensor;
          layout_map_.Set(tensor, layout);
        }
      }
    };
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        body, [&](const ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
          if (const auto *call = object.as<CallObj>();
              call && IsAnnotationCall(call))
            collect(ReadAnnotations(call));
          if (const auto *region = object.as<RegionStmtObj>();
              region && region->op->name == "tl.attr") {
            auto data = ReadAnnotations(region);
            const String key = data.at("key").cast<String>();
            if (key == attr::kLayoutMap)
              collect({{attr::kLayoutMap, data.at("value")}});
            else if (key == "tl.kernel")
              collect(data.at("value").cast<Map<String, Any>>());
          }
          return ffi::WalkResult::Advance();
        });
    for (const auto &[tensor, layout] : layout_map_)
      buffer_remap_.Set(tensor, MakePhysicalTensor(tensor, layout));
    // Metal SIMD-group matrices are opaque per-warp values. Their original
    // tensor shape remains logical through GEMM/copy parsing; only the lowered
    // definitions and uses carry the actual per-warp register capacity.
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        body, [&](const BindObj *binding) -> ffi::Expected<ffi::WalkResult> {
          const auto *call = binding->value.as<CallObj>();
          auto tensor = binding->var.as<TensorVar>();
          if (!tensor || !call || !call->attrs)
            return ffi::WalkResult::Advance();
          auto attrs = call->attrs.value().as_or_throw<DictAttrs>();
          auto elements =
              attrs.GetAttr<int64_t>("tl.metal_simdgroup_physical_elements");
          if (!elements)
            return ffi::WalkResult::Advance();
          TVM_FFI_CHECK_EQ(tensor.value().scope().name(), "metal.simdgroup",
                           ValueError);
          TVM_FFI_CHECK(*elements > 0 && *elements % 64 == 0, ValueError);
          TVM_FFI_CHECK(!layout_map_.count(*tensor), ValueError)
              << "Opaque Metal accumulators cannot also carry a scalar memory "
                 "layout";
          auto type = make_object<TensorTypeObj>(*tensor.value().ty().get());
          type->shape = {IntImm::Int64(*elements)};
          type->layout = std::nullopt;
          buffer_remap_.Set(*tensor,
                            TensorVar(tensor.value()->name, TensorType(type),
                                      tensor.value()->loc));
          return ffi::WalkResult::Advance();
        });
    for (const auto &[identity, tensors] : aliases_.groups)
      for (const auto &tensor : tensors)
        if (IsFragmentBuffer(tensor) && !buffer_remap_.count(tensor))
          buffer_remap_.Set(tensor, MakePhysicalTensor(tensor, std::nullopt));
  }

  void RecordMetadata(const Map<String, Any> &metadata) {
    for (const auto &[key, value] : metadata) {
      if (key == attr::kSafeValueMap) {
        auto previous = block_annotations_.Get(key);
        auto values =
            previous ? NormalizeSafeValueMap(*previous) : Map<Var, PrimExpr>{};
        for (const auto &[tensor, safe] : NormalizeSafeValueMap(value))
          values.Set(tensor, safe);
        block_annotations_.Set(key, values);
      } else if (key == "tl.l0_sf_bindings") {
        // Each allocation emits one explicit binding. Accumulate the current
        // scope's fixed one-to-one map without losing earlier SF handles.
        auto previous = block_annotations_.Get(key);
        auto values =
            previous ? previous->cast<Map<Var, Var>>() : Map<Var, Var>{};
        for (const auto &[sf, data] : value.cast<Map<Var, Var>>()) {
          if (auto existing = values.Get(sf))
            TVM_FFI_CHECK(existing->same_as(data), ValueError)
                << "Conflicting allocation bindings for L0 SF " << sf->name;
          for (const auto &[other_sf, other_data] : values)
            TVM_FFI_CHECK(!other_data.same_as(data) || other_sf.same_as(sf),
                          ValueError)
                << "L0 data " << data->name << " has multiple SF handles";
          values.Set(sf, data);
        }
        block_annotations_.Set(key, values);
      } else
        block_annotations_.Set(key, value);
    }
    if (auto dims = metadata.Get("cluster_dims")) {
      int size = 1;
      for (const auto &value : dims->cast<Array<Any>>()) {
        auto integer = value.as<int64_t>();
        if (!integer)
          integer = TryConstInt(value.cast<PrimExpr>());
        TVM_FFI_CHECK(integer && *integer > 0 &&
                          *integer <= std::numeric_limits<int>::max() / size,
                      ValueError)
            << "cluster_dims must be positive constant dimensions";
        size *= *integer;
      }
      cluster_size_ = size;
    }
  }

  Stmt RewriteAnnotation(Map<String, Any> metadata, Location loc) {
    if (auto values = metadata.Get(attr::kSafeValueMap))
      metadata.Set(attr::kSafeValueMap, NormalizeSafeValueMap(*values));
    Stmt original = MakeAnnotation(metadata, loc);
    // Mixed metadata can produce two declarations. Bypass Evaluate's metadata
    // consumer for each leaf to avoid recursively re-consuming this annotation.
    auto rewrite = [&](const Stmt &statement) -> Stmt {
      auto evaluated = statement.as_or_throw<Evaluate>();
      auto value = StmtExprMutatorWithAnalyzer::Mutate_(
                       evaluated->value.as_or_throw<Call>().get(),
                       InplaceMode::kDisallow)
                       .ValueOrUnchanged(evaluated->value);
      return Evaluate(value, evaluated->loc);
    };
    if (auto sequence = original.as<SeqStmt>()) {
      Array<Stmt> declarations;
      for (const Stmt &statement : (*sequence)->seq)
        declarations.push_back(rewrite(statement));
      return SeqStmt(declarations, loc);
    }
    return rewrite(original);
  }

  SeqStmt RewriteBarrierMetadata(const SeqStmt &body) {
    if (barrier_arrive_updates_.empty())
      return body;
    class Rewriter : public LayoutMapMutator {
    public:
      explicit Rewriter(Map<TensorVar, PrimExpr> updates)
          : updates_(std::move(updates)) {}
      using LayoutMapMutator::Mutate_;
      UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
        const auto *call = op->value.as<CallObj>();
        if (!call || !IsAnnotationCall(call))
          return LayoutMapMutator::Mutate_(op, mode);
        auto metadata = ReadAnnotations(call);
        if (auto value = metadata.Get("barrier_init")) {
          auto counts = value->cast<Map<Var, Array<PrimExpr>>>();
          for (const auto &[tensor, count] : updates_)
            if (auto previous = counts.Get(tensor))
              counts.Set(tensor, previous->Map(
                                     [&](const PrimExpr &) { return count; }));
          metadata.Set("barrier_init", counts);
          return MakeAnnotation(metadata, op->loc);
        }
        return LayoutMapMutator::Mutate_(op, mode);
      }

    private:
      Map<TensorVar, PrimExpr> updates_;
    };
    auto rewriter = make_object<Rewriter>(barrier_arrive_updates_);
    return rewriter->Mutate(body).ValueOrUnchanged(body);
  }

  Array<PrimExpr> MutateIndices(const Array<PrimExpr> &indices,
                                InplaceMode mode) {
    return indices.Map([&](const PrimExpr &index) {
      return Mutate(index, mode).ValueOrUnchanged(index);
    });
  }

  Array<PrimExpr> ForwardIndices(const TensorVar &original,
                                 const Array<PrimExpr> &indices,
                                 const Layout &layout) {
    auto result = layout->Forward(indices);
    auto physical = buffer_remap_.at(original);
    // Replicated shared storage reserves a leading plane; logical accesses
    // address plane zero until the pipeline versioner supplies a plane.
    if (physical.shape().size() == result.size() + 1)
      result.insert(result.begin(), IntImm::Int32(0));
    TVM_FFI_CHECK_EQ(result.size(), physical.shape().size(), ValueError);
    return result;
  }

  Optional<TensorLoad> ResolveAddressLoad(const Expr &expr) const {
    Expr value = expr;
    std::unordered_set<const VarObj *> seen;
    while (const auto *var = value.as<VarObj>()) {
      if (!seen.insert(var).second)
        return std::nullopt;
      auto bound = bind_var_to_expr_.Get(GetRef<Var>(var));
      if (!bound)
        return std::nullopt;
      value = *bound;
    }
    return value.as<TensorLoad>();
  }

  bool PointerRequiresPartition(Expr value) const {
    std::unordered_set<const VarObj *> seen;
    while (true) {
      if (const auto *var = value.as<VarObj>()) {
        if (!seen.insert(var).second)
          break;
        // Use the original bindings: a fragment pointer may already have
        // been rewritten to physical local storage in the mutated body.
        auto bound = aliases_.bindings.Get(GetRef<Var>(var));
        if (!bound)
          bound = bind_var_to_expr_.Get(GetRef<Var>(var));
        if (bound) {
          value = *bound;
          continue;
        }
      }
      if (const auto *call = value.as<CallObj>()) {
        Optional<TensorLoad> load;
        if (call->op.same_as(address_of_op())) {
          // address_of(Var) is the address of that local scalar, even when
          // its value was loaded from a global tensor. Do not resolve its
          // value binding as tl.access_ptr does for an address operand.
          load = call->args[0].as<TensorLoad>();
          if (!load)
            return false;
        } else if (call->op.same_as(tl::access_ptr())) {
          load = ResolveAddressLoad(call->args[0]);
        }
        if (load) {
          auto tensor = load.value()->source.as_or_throw<TensorVar>();
          return IsFragmentBuffer(tensor) ||
                 !IsLocalBuffer(tensor, /*allow_var=*/true);
        }
        if (call->op.same_as(tensor_data_ptr_op())) {
          auto tensor = call->args[0].as_or_throw<TensorVar>();
          return IsFragmentBuffer(tensor) ||
                 !IsLocalBuffer(tensor, /*allow_var=*/true);
        }
        if (call->op.same_as(tl::ptr_access()) ||
            call->op.same_as(ptr_byte_offset_op()) ||
            (call->op.same_as(reinterpret_op()) && call->ty.as<PtrTypeObj>())) {
          value = call->args[0];
          continue;
        }
      }
      break;
    }
    // A raw typed parameter need not have a tensor definition. Its explicit
    // memory scope still distinguishes a thread-local address from storage
    // whose accesses must be distributed across the Parallel iteration set.
    if (auto pointer = value->ty.as<PtrType>()) {
      String scope = pointer.value()->scope.name();
      return scope != "local" && scope != "local.var";
    }
    return false;
  }

  Expr RewriteAccess(const CallObj *call, Optional<PrimExpr> extra,
                     InplaceMode mode) {
    if (call->op.same_as(tl::access_ptr())) {
      auto resolved = ResolveAddressLoad(call->args[0]);
      TVM_FFI_CHECK(resolved.has_value(), ValueError)
          << "tl.access_ptr needs a tensor address";
      TensorLoad load = *resolved;
      auto tensor = load->source.as_or_throw<TensorVar>();
      auto indices = MutateIndices(load->indices, mode);
      if (extra) {
        PrimExpr remaining = Mutate(*extra, mode).ValueOrUnchanged(*extra);
        for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
          indices.Set(i,
                      GetSyncedAnalyzer()->Simplify(
                          indices[i] + floormod(remaining, tensor.shape()[i])));
          remaining = floordiv(remaining, tensor.shape()[i]);
        }
      }
      TensorVar physical = Mutate(tensor, mode)
                               .ValueOrUnchanged(tensor)
                               .as_or_throw<TensorVar>();
      PrimExpr extent =
          Mutate(call->args[1].as_or_throw<PrimExpr>(), mode)
              .ValueOrUnchanged(call->args[1].as_or_throw<PrimExpr>());
      PrimExpr mask =
          Mutate(call->args[2].as_or_throw<PrimExpr>(), mode)
              .ValueOrUnchanged(call->args[2].as_or_throw<PrimExpr>());
      if (in_tma_context_ && layout_map_.count(tensor)) {
        // TMA's descriptor implements the swizzle. Its address remains in the
        // logical tensor's element coordinates, based at the physical storage.
        return Call(physical.ty()->DataPtrType(), Op::Get("tl.ptr_access"),
                    {physical.data(), TensorElementOffset(tensor, indices),
                     extent, mask},
                    {}, {}, call->loc);
      }
      if (auto layout = layout_map_.Get(tensor))
        indices = ForwardIndices(tensor, indices, *layout);
      return Call(physical.ty()->DataPtrType(), call->op,
                  {TensorLoad(physical, indices, load->loc), extent, mask},
                  call->attrs, {}, call->loc);
    }
    if (call->op.same_as(tl::ptr_access())) {
      Expr identity = aliases_.Identity(call->args[0]);
      auto tensor = identity.as<TensorVar>();
      Array<Expr> args = call->args;
      for (size_t i = 0; i < args.size(); ++i)
        args.Set(i, Mutate(args[i], mode).ValueOrUnchanged(args[i]));
      PrimExpr offset = args[1].as_or_throw<PrimExpr>();
      if (extra)
        offset = GetSyncedAnalyzer()->Simplify(offset + *extra);
      if (tensor && layout_map_.count(*tensor) && !in_tma_context_) {
        Array<PrimExpr> indices;
        PrimExpr rest = offset;
        for (int i = static_cast<int>(tensor.value().shape().size()) - 1;
             i >= 0; --i) {
          indices.insert(indices.begin(),
                         floormod(rest, tensor.value().shape()[i]));
          rest = floordiv(rest, tensor.value().shape()[i]);
        }
        auto physical = buffer_remap_.at(*tensor);
        offset = GetSyncedAnalyzer()->Simplify(TensorElementOffset(
            physical,
            ForwardIndices(*tensor, indices, layout_map_.at(*tensor))));
        args.Set(0, physical.data());
      }
      args.Set(1, offset);
      return Call(args[0]->ty.as_or_throw<PtrType>(), call->op, args,
                  call->attrs, {}, call->loc);
    }
    TVM_FFI_CHECK(!extra, ValueError)
        << "Additional ldmatrix offset requires a tensor access contract";
    return Mutate(GetRef<Call>(call), mode)
        .ValueOrUnchanged(GetRef<Call>(call));
  }

  Range CurrentThreadBounds() const {
    return ComputeThreadBounds(thread_binding_, GetAnalyzer());
  }
  PrimExpr CurrentThreadIndex() const {
    return thread_binding_ ? PrimExpr(*thread_binding_) : PrimExpr(0);
  }

  Target target_{UnsafeInit{}};
  TensorAliasGroups aliases_;
  LayoutMap layout_map_;
  Map<TensorVar, TensorVar> buffer_remap_;
  Map<Var, Expr> bind_var_to_expr_;
  bool inside_simt_vf_{false};
  Map<String, Any> block_annotations_;
  Map<String, Any> loop_annotations_;
  Optional<PrimVar> thread_binding_;
  std::vector<Array<TensorVar>> workspace_stack_;
  int cluster_size_{1};
  int mbarrier_count_{0};
  std::vector<int> mbarrier_arrive_counts_;
  Optional<TensorVar> mbarrier_buffer_;
  Map<TensorVar, PrimExpr> barrier_arrive_updates_;
  Map<String, Any> smem_alignment_map_;
  std::vector<PrimExpr> loop_mbar_phase_stack_;
  bool has_tma_{false};
  bool in_tma_context_{false};
};

} // namespace

namespace transform {
tvm::transform::Pass LowerTileOp() {
  return tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule, tvm::transform::PassContext) {
        VectorizeMemoryScope memory_scope(function);
        return LowerTileOpPass::Apply(std::move(function));
      },
      0, "tl.LowerTileOp");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.LowerTileOp", LowerTileOp);
}
} // namespace transform
} // namespace tvm::tl
