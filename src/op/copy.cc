/*!
 * \file tl/op/copy.cc
 * \brief Define the copy operator, backend dispatch, and shared fallback
 *        lowering helpers.
 */

#include "copy.h"
#include "../target/utils.h"
#include "../transform/common/loop_fusion_utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"
#include "../transform/ptx_async_copy_injector.h"
#include "utils.h"

#include "builtin.h"
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/transform.h>

#include <limits>
#include <sstream>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

Stmt LowerNormalCopy(const CopyNode &op, const LowerArgs &T,
                     arith::Analyzer *analyzer) {
  bool is_cpu_target = T.target->GetTargetDeviceType() == kDLCPU;
  auto simt_loop = op.MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));

  For vectorized_thread_loop;
  auto par_op = ParallelOp(fused_loop);

  if (is_cpu_target || IsLocalBuffer(op.src) || IsLocalBuffer(op.dst)) {
    if (IsLocalBuffer(op.src) && !IsLocalBuffer(op.dst)) {
      // A conflict write only occurs when multiple threads write to the same
      // global address. If any dst_range dimension's min depends on the thread
      // variable, each thread targets a distinct location and there is no
      // conflict.
      bool dst_depends_on_thread = false;
      for (const auto &range : op.dst_range) {
        if (tir::UsesVar(range->min, [&](const VarNode *v) {
              return v == T.thread_var.get();
            })) {
          dst_depends_on_thread = true;
          break;
        }
      }
      if (!dst_depends_on_thread) {
        LOG(WARNING) << "Copy from local buffer `" << op.src->name << "` to "
                     << op.dst.scope() << " buffer `" << op.dst->name
                     << "` may cause conflicted write.";
      }
    }
    vectorized_thread_loop = VectorizeLoop(fused_loop, T.layout_map);
    return vectorized_thread_loop;
  }

  std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                    InferLevel::kFree};
  for (auto level : levels) {
    par_op->InferLayout({T.target,
                         T.thread_bounds,
                         T.layout_map,
                         analyzer,
                         false,
                         T.buffer_remap,
                         {}},
                        level);
  }
  auto loop_layout = par_op->GetLoopLayout();
  return LowerParallelLoop(par_op->GetRoot(), loop_layout, T.thread_var,
                           analyzer, T.layout_map,
                           par_op->GetPredicate(T.thread_var));
}

Stmt LowerCPAsyncCopy(const CopyNode &op, const LowerArgs &T,
                      arith::Analyzer *analyzer) {
  using namespace tvm::transform;

  PassContext pass_ctx = PassContext::Current();
  bool enable_async_copy =
      pass_ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(true)).value();
  bool no_implicit_commit_wait = op.GetNoImplicitAsyncCommitWait();
  bool explicit_async_semantics =
      no_implicit_commit_wait || op.GetIsAsyncCopy();
  if (!enable_async_copy && !explicit_async_semantics) {
    return LowerNormalCopy(op, T, analyzer);
  }

  auto simt_loop = op.MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
  auto par_op = ParallelOp(fused_loop);

  std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                    InferLevel::kFree};
  for (auto level : levels) {
    par_op->InferLayout({T.target,
                         T.thread_bounds,
                         T.layout_map,
                         analyzer,
                         false,
                         T.buffer_remap,
                         {}},
                        level);
  }
  auto loop_layout = par_op->GetLoopLayout();
  Stmt lowered_loop =
      LowerParallelLoop(par_op->GetRoot(), loop_layout, T.thread_var, analyzer,
                        T.layout_map, par_op->GetPredicate(T.thread_var));

  auto inject_result =
      InjectPTXAsyncCopy(lowered_loop, /*enable_auto_async_copy=*/true,
                         /*async_without_async_commit_wait=*/
                         no_implicit_commit_wait || op.GetIsAsyncCopy());
  Stmt cp_async_loop = inject_result.stmt;
  if (!inject_result.injected_ptx_async_copy) {
    LOG(WARNING) << "cp.async rewrite miss for copy src=" << op.src->name
                 << " (scope=" << op.src.scope() << ", dtype=" << op.src->dtype
                 << "), dst=" << op.dst->name << " (scope=" << op.dst.scope()
                 << ", dtype=" << op.dst->dtype
                 << "), no_implicit_async_commit_wait="
                 << no_implicit_commit_wait
                 << ", is_async_copy=" << op.GetIsAsyncCopy();
    if (no_implicit_commit_wait) {
      LOG(WARNING)
          << "Pipeline-managed async copy fallback to normal copy because "
             "cp.async rewrite found no eligible global->shared store.";
      return lowered_loop;
    }
    if (explicit_async_semantics) {
      LOG(FATAL) << "Explicit async copy semantics require cp.async lowering, "
                    "but no eligible global->shared store was rewritten.";
    }
    LOG(WARNING) << "Fallback to normal copy because cp.async rewrite found "
                    "no eligible global->shared store.";
    return LowerNormalCopy(op, T, analyzer);
  }
  if (no_implicit_commit_wait) {
    return cp_async_loop;
  }
  if (op.GetIsAsyncCopy()) {
    Stmt commit_group =
        Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
    return SeqStmt({cp_async_loop, commit_group});
  }
  return cp_async_loop;
}

namespace {

std::vector<CopyImpl> &CopyImplRegistry() {
  static std::vector<CopyImpl> registry;
  return registry;
}

LayoutMap DefaultInferCopyLayout(const CopyNode &op, const LayoutInferArgs &T,
                                 InferLevel level) {
  return op.InferLayoutImpl(T, level);
}

bool DefaultMatchTarget(Target target) { return true; }

Stmt DefaultLowerCopy(const CopyNode &op, const LowerArgs &T,
                      arith::Analyzer *analyzer) {
  if (op.GetIsTmaCopy()) {
    LOG(FATAL) << "T.tma_copy() requires a target-specific copy "
                  "implementation, but no implementation is registered for "
               << T.target->ToDebugString();
  }
  if (op.GetIsAsyncCopy() || op.GetNoImplicitAsyncCommitWait()) {
    LOG(FATAL) << "Async copy requires a target-specific copy implementation, "
                  "but no implementation is registered for "
               << T.target->ToDebugString();
  }
  return LowerNormalCopy(op, T, analyzer);
}

CopyInstructionKind DefaultClassifyCopyInstruction(const CopyNode &op,
                                                   Target target,
                                                   bool in_pipeline,
                                                   arith::Analyzer *analyzer) {
  if (op.GetIsAsyncCopy()) {
    return CopyInstructionKind::kCPAsync;
  }
  return CopyInstructionKind::kSync;
}

CopyPipelineRole DefaultClassifyCopyPipelineRole(const CopyNode &op,
                                                 Target target,
                                                 arith::Analyzer *analyzer) {
  if (op.GetIsAsyncCopy()) {
    return CopyPipelineRole::kCPAsyncProducer;
  }
  return CopyPipelineRole::kConsumer;
}

bool DefaultCanPipelineManageCopyAsync(const CopyNode &op, Target target,
                                       arith::Analyzer *analyzer) {
  return false;
}

bool DefaultIsSyncGlobalToSharedCopyLike(const CopyNode &op, Target target,
                                         arith::Analyzer *analyzer) {
  return false;
}

CopyImpl MakeDefaultCopyImpl() {
  return CopyImpl{
      "default.Copy",
      DefaultMatchTarget,
      std::numeric_limits<int>::min(),
      DefaultInferCopyLayout,
      DefaultLowerCopy,
      DefaultClassifyCopyInstruction,
      DefaultClassifyCopyPipelineRole,
      DefaultCanPipelineManageCopyAsync,
      DefaultIsSyncGlobalToSharedCopyLike,
  };
}

void EnsureDefaultCopyImplRegistered() {
  auto &registry = CopyImplRegistry();
  if (registry.empty()) {
    registry.push_back(MakeDefaultCopyImpl());
  }
}

const CopyImpl &ResolveCopyImpl(Target target) {
  EnsureDefaultCopyImplRegistered();
  const auto &registry = CopyImplRegistry();
  const CopyImpl *best_impl = nullptr;
  int best_priority = std::numeric_limits<int>::min();
  for (const CopyImpl &impl : registry) {
    if (impl.match_target(target) && impl.priority >= best_priority) {
      best_impl = &impl;
      best_priority = impl.priority;
    }
  }
  ICHECK(best_impl != nullptr);
  return *best_impl;
}

LayoutMap InferCopyLayout(const CopyNode &op, const LayoutInferArgs &T,
                          InferLevel level) {
  return ResolveCopyImpl(T.target).infer_layout(op, T, level);
}

Stmt LowerCopyForTarget(const CopyNode &op, const LowerArgs &T,
                        arith::Analyzer *analyzer) {
  return ResolveCopyImpl(T.target).lower(op, T, analyzer);
}

std::vector<Conv2DIm2ColImpl> &Conv2DIm2ColImplRegistry() {
  static std::vector<Conv2DIm2ColImpl> registry;
  return registry;
}

Stmt DefaultLowerConv2DIm2Col(const Conv2DIm2ColOpNode &op, const LowerArgs &T,
                              arith::Analyzer *analyzer) {
  LOG(FATAL) << "Conv2D im2col requires a target-specific implementation, "
                "but no implementation is registered for "
             << T.target->ToDebugString();
  return Stmt();
}

Conv2DIm2ColImpl MakeDefaultConv2DIm2ColImpl() {
  return Conv2DIm2ColImpl{
      "default.Conv2DIm2Col",
      DefaultMatchTarget,
      std::numeric_limits<int>::min(),
      DefaultLowerConv2DIm2Col,
  };
}

void EnsureDefaultConv2DIm2ColImplRegistered() {
  auto &registry = Conv2DIm2ColImplRegistry();
  if (registry.empty()) {
    registry.push_back(MakeDefaultConv2DIm2ColImpl());
  }
}

const Conv2DIm2ColImpl &ResolveConv2DIm2ColImpl(Target target) {
  EnsureDefaultConv2DIm2ColImplRegistered();
  const auto &registry = Conv2DIm2ColImplRegistry();
  const Conv2DIm2ColImpl *best_impl = nullptr;
  int best_priority = std::numeric_limits<int>::min();
  for (const Conv2DIm2ColImpl &impl : registry) {
    if (impl.match_target(target) && impl.priority >= best_priority) {
      best_impl = &impl;
      best_priority = impl.priority;
    }
  }
  ICHECK(best_impl != nullptr);
  return *best_impl;
}

Stmt LowerConv2DIm2ColForTarget(const Conv2DIm2ColOpNode &op,
                                const LowerArgs &T, arith::Analyzer *analyzer) {
  return ResolveConv2DIm2ColImpl(T.target).lower(op, T, analyzer);
}

} // namespace

void RegisterCopyImpl(CopyImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.infer_layout != nullptr);
  ICHECK(impl.lower != nullptr);
  ICHECK(impl.classify_instruction != nullptr);
  ICHECK(impl.classify_pipeline_role != nullptr);
  ICHECK(impl.can_pipeline_managed_async != nullptr);
  ICHECK(impl.is_sync_global_to_shared_prefix != nullptr);
  EnsureDefaultCopyImplRegistered();
  CopyImplRegistry().push_back(impl);
}

CopyInstructionKind
ClassifyCopyInstructionForTarget(const CopyNode &op, Target target,
                                 bool in_pipeline, arith::Analyzer *analyzer) {
  return ResolveCopyImpl(target).classify_instruction(op, target, in_pipeline,
                                                      analyzer);
}

CopyPipelineRole ClassifyCopyPipelineRoleForTarget(const CopyNode &op,
                                                   Target target,
                                                   arith::Analyzer *analyzer) {
  return ResolveCopyImpl(target).classify_pipeline_role(op, target, analyzer);
}

bool CanPipelineManageCopyAsyncForTarget(const CopyNode &op, Target target,
                                         arith::Analyzer *analyzer) {
  return ResolveCopyImpl(target).can_pipeline_managed_async(op, target,
                                                            analyzer);
}

bool IsSyncGlobalToSharedCopyLikeForTarget(const CopyNode &op, Target target,
                                           arith::Analyzer *analyzer) {
  return ResolveCopyImpl(target).is_sync_global_to_shared_prefix(op, target,
                                                                 analyzer);
}

void RegisterConv2DIm2ColImpl(Conv2DIm2ColImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.lower != nullptr);
  EnsureDefaultConv2DIm2ColImplRegistered();
  Conv2DIm2ColImplRegistry().push_back(impl);
}

// Constructs a Copy operator node from call arguments and annotations.
// args[0]: source region, args[1]: destination region
// annotations: Map containing coalesced_width, disable_tma, eviction_policy,
// etc.
Copy::Copy(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<CopyNode> node = tvm::ffi::make_object<CopyNode>();
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->src = src_access.region->buffer;
  node->dst = dst_access.region->buffer;
  node->src_range = src_access.region->region;
  node->dst_range = dst_access.region->region;
  node->SetAccessRegions({src_access, dst_access});
  // Copy annotations from the Call node
  node->annotations = annotations;
  data_ = std::move(node);
}

// Creates a shallow clone of this CopyNode.
TileOperator CopyNode::Clone() const {
  auto op = tvm::ffi::make_object<CopyNode>(*this);
  if (par_op_.defined()) {
    op->par_op_ = Downcast<ParallelOp>(par_op_->Clone());
  }
  return Copy(op);
}

// Creates iterator variables for dimensions with extent > 1.
Array<IterVar> CopyNode::MakeIterVars() const {
  // Choose the range set from the lowest-level memory scope between src and
  // dst. Scope levels: global < shared/shared.dyn/shared.tmem < local.fragment
  // (fragment)
  auto scope_level = [](const Buffer &b) -> int {
    String s = b.scope();
    if (s == "local.fragment" || s == "local")
      return 2;
    if (s == "shared" || s == "shared.dyn" || s == "shared.tmem")
      return 1;
    // default to global level for unknown scopes
    return 0;
  };

  int src_level = scope_level(src);
  int dst_level = scope_level(dst);
  bool base_is_src = (src_level >= dst_level);
  const Array<Range> &base_ranges = base_is_src ? src_range : dst_range;

  // Sanity check: when switching away from the original (src_range),
  // ensure the chosen base ranges are not provably smaller than the original
  // per dimension. This guards against generating undersized loop domains.
  // Improved logic: use two pointers to traverse both base_ranges and
  // src_range, skipping dimensions with extent == 1. The number of non-1
  // extents must match.
  arith::Analyzer analyzer;

  size_t base_dim = 0, src_dim = 0;
  while (base_dim < base_ranges.size() && src_dim < src_range.size()) {
    // Skip base extents that are 1
    while (base_dim < base_ranges.size() &&
           is_one(base_ranges[base_dim]->extent)) {
      ++base_dim;
    }
    // Skip src extents that are 1
    while (src_dim < src_range.size() && is_one(src_range[src_dim]->extent)) {
      ++src_dim;
    }
    // Both indices now at non-1, or at end
    if (base_dim < base_ranges.size() && src_dim < src_range.size()) {
      PrimExpr base_ext = base_ranges[base_dim]->extent;
      PrimExpr src_ext = src_range[src_dim]->extent;
      // Only fail if base extent is provably smaller than src extent
      if (analyzer.CanProve(base_ext < src_ext)) {
        std::ostringstream oss;
        oss << "Selected loop range is smaller than original src range at "
               "matched non-1 dimension: "
            << "base(extent=" << base_ext
            << ", scope=" << (base_is_src ? src.scope() : dst.scope())
            << ", min=" << base_ranges[base_dim]->min
            << ", base_dim=" << base_dim << ") < src(extent=" << src_ext
            << ", min=" << src_range[src_dim]->min << ", src_dim=" << src_dim
            << ", scope=" << src.scope() << ") for src=" << src->name
            << ", dst=" << dst->name << "\n";
        oss << "src buffer: " << src->name << ", scope=" << src.scope() << "\n";
        oss << "dst buffer: " << dst->name << ", scope=" << dst.scope() << "\n";
        oss << "base_ranges[" << base_dim
            << "]: min=" << base_ranges[base_dim]->min
            << ", extent=" << base_ext << "\n";
        oss << "src_ranges[" << src_dim << "]: min=" << src_range[src_dim]->min
            << ", extent=" << src_ext << "\n";
        LOG(FATAL) << oss.str();
      }
      ++base_dim;
      ++src_dim;
    }
  }

  // Any remaining unmatched dimensions in either range must all have extent ==
  // 1
  while (base_dim < base_ranges.size()) {
    ICHECK(is_one(base_ranges[base_dim]->extent))
        << "base_ranges has extra non-1 extent at dim " << base_dim;
    ++base_dim;
  }
  while (src_dim < src_range.size()) {
    ICHECK(is_one(src_range[src_dim]->extent))
        << "src_range has extra non-1 extent at dim " << src_dim;
    ++src_dim;
  }

  Array<IterVar> loop_vars;
  size_t idx = 0;
  for (size_t i = 0; i < base_ranges.size(); i++) {
    if (is_one(base_ranges[i]->extent))
      continue;
    Var var = Var(std::string{char('i' + idx)}, base_ranges[i]->extent->dtype);
    idx++;
    loop_vars.push_back(
        {Range(0, base_ranges[i]->extent), var, IterVarType::kDataPar});
  }
  return loop_vars;
}

// Generates index expressions for accessing src (src_dst=0) or dst (src_dst=1)
// buffers.
Array<PrimExpr> CopyNode::MakeIndices(const Array<IterVar> &ivs,
                                      int src_dst) const {
  Array<PrimExpr> indices;
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent))
      indices.push_back(ranges[i]->min);
    else {
      indices.push_back(ranges[i]->min + ivs[idx]->var);
      idx++;
    }
  }
  ICHECK(idx == ivs.size())
      << "idx = " << idx << ", ivs.size() = " << ivs.size()
      << "src name = " << src->name << ", dst name = " << dst->name;
  return indices;
}

// Builds a boundary predicate for memory accesses.
// Returns a conjunction of bounds checks, or empty PrimExpr if all checks pass.
PrimExpr CopyNode::MakePredicate(arith::Analyzer *analyzer,
                                 const Array<IterVar> &ivs,
                                 Array<PrimExpr> extents, int src_dst) const {
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;

  Array<PrimExpr> cond_list;
  ICHECK(extents.size() == ranges.size()) << extents << " " << ranges;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent))
      continue;
    PrimExpr cond = ranges[i]->min + ivs[idx]->var < extents[i];
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    cond = ranges[i]->min + ivs[idx]->var >= 0;
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    idx++;
  }
  if (cond_list.empty())
    return {};
  else {
    PrimExpr cond = cond_list[0];
    for (size_t i = 1; i < cond_list.size(); i++)
      cond = And(cond, cond_list[i]);
    return cond;
  }
}

// Constructs a SIMT-style nested loop that implements the copy.
For CopyNode::MakeSIMTLoop(arith::Analyzer *analyzer) const {
  Array<IterVar> loop_vars = MakeIterVars();
  bool is_scalar = loop_vars.empty();

  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);
  ICHECK(loop_vars.size() <= src_range.size())
      << "loop_vars.size() = " << loop_vars.size()
      << ", src_range.size() = " << src_range.size() << ", src = " << src->name
      << ", dst = " << dst->name;

  ICHECK(loop_vars.size() <= dst_range.size())
      << "loop_vars.size() = " << loop_vars.size()
      << ", dst_range.size() = " << dst_range.size() << ", src = " << src->name
      << ", dst = " << dst->name;

  Array<PrimExpr> src_indices = MakeIndices(loop_vars, 0);
  Array<PrimExpr> dst_indices = MakeIndices(loop_vars, 1);

  PrimExpr src_predicate = MakePredicate(analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate = MakePredicate(analyzer, loop_vars, dst->shape, 1);

  PrimExpr value = BufferLoad(src, src_indices);
  if (src->dtype != dst->dtype)
    value = Cast(dst->dtype, value);
  if (src_predicate.defined())
    value = if_then_else(src_predicate, value, make_zero(dst->dtype));

  Stmt body = BufferStore(dst, value, dst_indices);
  if (dst_predicate.defined())
    body = IfThenElse(dst_predicate, body);
  if (is_scalar) {
    return For(Var("i"), 0, 1, ForKind::kSerial, body);
  }

  for (int i = loop_vars.size() - 1; i >= 0; i--) {
    Map<String, ObjectRef> loop_annotations;

    // Only attach the parallel related annotations on the outermost loop (i ==
    // 0)
    if (i == 0) {
      if (annotations.count(attr::kCoalescedWidth)) {
        loop_annotations.Set(attr::kCoalescedWidth,
                             annotations.Get(attr::kCoalescedWidth).value());
      }
      if (annotations.count(attr::kParallelLoopLayout)) {
        loop_annotations.Set(
            attr::kParallelLoopLayout,
            annotations.Get(attr::kParallelLoopLayout).value());
      }
    }

    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body, std::nullopt, loop_annotations);
  }
  return Downcast<For>(body);
}

// Computes a linearized shared-memory layout for TMA transfers.
// Maps [i, j] -> [i // 256, j // 256, i % 256, j % 256]
Layout CopyNode::ComputeLinearLayout(const Buffer &shared_tensor) const {
  Array<PrimExpr> input_size = shared_tensor->shape;
  Array<PrimExpr> forward_vars;
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_vars.push_back(InputPlaceholder(i));
  }
  // [i, j] -> [i // 256, j // 256, i % 256, j % 256]
  Array<PrimExpr> forward_index;
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_index.push_back(FloorDiv(forward_vars[i], 256));
  }
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_index.push_back(FloorMod(forward_vars[i], 256));
  }
  return Layout(input_size, forward_index);
}

LayoutMap CopyNode::InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const {
  return InferCopyLayout(*this, T, level);
}

// Infers memory layouts for the default target-independent copy path.
LayoutMap CopyNode::InferLayoutImpl(const LayoutInferArgs &T,
                                    InferLevel level) const {
  if (GetIsTmaCopy()) {
    LOG(FATAL) << "T.tma_copy() requires a target-specific copy "
                  "implementation, but no implementation is registered for "
               << T.target->ToDebugString();
  }
  if (GetIsAsyncCopy() || GetNoImplicitAsyncCommitWait()) {
    LOG(FATAL) << "Async copy requires a target-specific copy implementation, "
                  "but no implementation is registered for "
               << T.target->ToDebugString();
  }
  return InferSIMTLayout(T, level);
}

LayoutMap CopyNode::InferSIMTLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  if (!par_op_.defined()) {
    arith::Analyzer analyzer;
    par_op_ = ParallelOp(MakeSIMTLoop(&analyzer));
  }
  return par_op_->InferLayout(T, level);
}
// Lowers the copy operation by dispatching to the selected target
// implementation.
Stmt CopyNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  return LowerCopyForTarget(*this, T, analyzer);
}

// Constructs a Conv2DIm2ColOp node from call arguments.
// args: src, dst, nhw_step, c_step, kernel, stride, dilation, padding,
// eviction_policy
Conv2DIm2ColOp::Conv2DIm2ColOp(Array<PrimExpr> args,
                               Map<String, ObjectRef> annotations) {
  ObjectPtr<Conv2DIm2ColOpNode> node =
      tvm::ffi::make_object<Conv2DIm2ColOpNode>();
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->srcRegion_ = src_access.region;
  node->dstRegion_ = dst_access.region;
  node->SetAccessRegions({src_access, dst_access});
  node->src_ = node->srcRegion_->buffer;
  node->dst_ = node->dstRegion_->buffer;
  node->nhw_step_ = args[2];
  node->c_step_ = args[3];
  node->kernel_ = args[4].as<IntImm>().value()->value;
  node->stride_ = args[5].as<IntImm>().value()->value;
  node->dilation_ = args[6].as<IntImm>().value()->value;
  node->padding_ = args[7].as<IntImm>().value()->value;
  node->eviction_policy_ = args[8].as<IntImm>().value()->value;
  node->annotations_ = annotations;
  data_ = std::move(node);
}

// Creates a shallow copy of this Conv2DIm2ColOpNode.
TileOperator Conv2DIm2ColOpNode::Clone() const {
  auto op = tvm::ffi::make_object<Conv2DIm2ColOpNode>(*this);
  return Conv2DIm2ColOp(op);
}

Stmt Conv2DIm2ColOpNode::Lower(const LowerArgs &T,
                               arith::Analyzer *analyzer) const {
  return LowerConv2DIm2ColForTarget(*this, T, analyzer);
}

void CopyNode::CollectFragmentLayouts(const PrimExpr &expr,
                                      const Map<Var, PrimExpr> &let_var_to_expr,
                                      const LayoutMap &existing_layouts,
                                      PrimExpr thread_extent,
                                      Range thread_bounds,
                                      Map<Buffer, Layout> &result_map) const {
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (auto bl = node.as<BufferLoadNode>()) {
      if (IsFragmentBuffer(bl->buffer) && !existing_layouts.count(bl->buffer) &&
          !result_map.count(bl->buffer)) {
        auto f = Fragment::FullyReplicated(bl->buffer->shape, thread_extent);
        result_map.Set(bl->buffer, f->BindThreadRange(thread_bounds));
      }
    } else if (auto var_node = node.as<VarNode>()) {
      auto var = tvm::ffi::GetRef<Var>(var_node);
      if (let_var_to_expr.count(var)) {
        CollectFragmentLayouts(let_var_to_expr[var], let_var_to_expr,
                               existing_layouts, thread_extent, thread_bounds,
                               result_map);
      }
    }
  });
}

// Register the Copy operation with TVM's TIR system
// This makes the copy operation available for use in TVM programs
// - Takes 5 inputs: src_buffer, dst_buffer, coalesced_width, disable_tma,
// eviction_policy
// - Marked as opaque since it has side effects (memory writes)
TIR_REGISTER_TL_TILE_OP(Copy, copy)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.async_copy")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "async_copy")
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               Map<String, ObjectRef> ann = annotations;
                               ann.Set("is_async_copy",
                                       IntImm(DataType::Int(32), 1));
                               return Copy(args, ann);
                             })
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// Register the tma_copy operation — same as copy but forces TMA path
// and emits only expect_tx + tma_load (no wait).
TVM_REGISTER_OP("tl.tileop.tma_copy")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "tma_copy")
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               Map<String, ObjectRef> ann = annotations;
                               ann.Set("is_tma_copy",
                                       IntImm(DataType::Int(32), 1));
                               return Copy(args, ann);
                             })
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// Layout inference hook - returns empty map (no layout suggestions).
LayoutMap Conv2DIm2ColOpNode::InferLayout(const LayoutInferArgs &T,
                                          InferLevel level) const {
  return {};
}

// Register the Conv2DIm2Col operation with TVM's TIR system
// This operation performs im2col transformation for 2D convolutions using TMA
// - Takes 9 inputs: src_buffer, dst_buffer, nhw_step, c_step, kernel, stride,
// dilation, padding, eviction_policy
// - Marked as opaque since it has side effects (memory writes)
TIR_REGISTER_TL_TILE_OP(Conv2DIm2ColOp, c2d_im2col)
    .set_num_inputs(9)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  CopyNode::RegisterReflection();
  Conv2DIm2ColOpNode::RegisterReflection();
}
} // namespace tl
} // namespace tvm
