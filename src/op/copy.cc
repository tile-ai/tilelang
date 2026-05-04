/*!
 * \file tl/op/copy.cc
 * \brief Define copy operator for various memory transfer strategies (Normal,
 *        Bulk/TMA, LDSM/STSM) and lowering logic for GPU code generation.
 *
 * implementing memory copy operations that can target CPUs or GPUs with
 * optimization for different instructions like bulk copy, matrix load/store,
 * and Hopper's new TMA (Tensor Memory Accelerator).
 */

#include "copy.h"
#include "../layout/tcgen05_layout.h"
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
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

/// Build a TMA leader-thread condition using tl_shuffle_elect.
/// \param thread_extent The number of threads in the current group
///        (e.g., full block extent for non-WS, producer_extent for WS).
///        The elected thread will be the first lane of the first warp in
///        the group.
static PrimExpr MakeTmaLeaderCondition(PrimExpr thread_extent) {
  return Call(DataType::Bool(), tl_shuffle_elect(), {std::move(thread_extent)});
}

static int64_t TMABytesFromElements(int64_t elements, DataType dtype) {
  return (elements * dtype.bits() + 7) / 8;
}

static PrimExpr TMABytesFromElements(PrimExpr elements, DataType dtype) {
  PrimExpr elements_i64 = cast(DataType::Int(64), elements);
  int bits = dtype.bits();
  if (bits % 8 == 0) {
    return elements_i64 * IntImm(DataType::Int(64), bits / 8);
  }
  return FloorDiv(elements_i64 * IntImm(DataType::Int(64), bits) +
                      IntImm(DataType::Int(64), 7),
                  IntImm(DataType::Int(64), 8));
}

static PrimExpr TMABitsFromElements(PrimExpr elements, DataType dtype) {
  return cast(DataType::Int(64), elements) *
         IntImm(DataType::Int(64), dtype.bits());
}

static int64_t TMAElementsForBytes(int64_t bytes, DataType dtype) {
  ICHECK_EQ((bytes * 8) % dtype.bits(), 0)
      << bytes << " bytes cannot be represented as whole elements of " << dtype;
  return bytes * 8 / dtype.bits();
}

PrimExpr GetCopyMbarPhaseExpr(const Map<String, ObjectRef> &annotations,
                              const LowerArgs &T) {
  PrimExpr phase = T.mbar_phase_expr;
  if (auto explicit_phase = GetAnnotatedMbarPhaseExpr(annotations)) {
    phase = explicit_phase.value();
  }
  return phase;
}

} // namespace

Stmt LowerNormalCopy(const CopyNode &op, const LowerArgs &T,
                     arith::Analyzer *analyzer) {
  bool is_cpu_target = T.target->GetTargetDeviceType() == kDLCPU;
  auto simt_loop = CopyLoweringAccess::MakeSIMTLoop(op, analyzer);
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

  auto simt_loop = CopyLoweringAccess::MakeSIMTLoop(op, analyzer);
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
  return CopyLoweringAccess::InferLayoutImpl(op, T, level);
}

CopyInst DefaultSelectCopyInst(const CopyNode &op, Target target,
                               const LayoutMap &layout_map,
                               arith::Analyzer *analyzer, bool buffer_oob) {
  if (op.GetIsTmaCopy()) {
    LOG(FATAL) << "T.tma_copy() requires a target-specific copy "
                  "implementation, but no implementation is registered for "
               << target->ToDebugString();
  }
  if (op.GetIsAsyncCopy() || op.GetNoImplicitAsyncCommitWait()) {
    LOG(FATAL) << "Async copy requires a target-specific copy implementation, "
                  "but no implementation is registered for "
               << target->ToDebugString();
  }
  return CopyInst::kNormal;
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

CopyImpl MakeDefaultCopyImpl() {
  return CopyImpl{
      "default.Copy",
      DefaultMatchTarget,
      std::numeric_limits<int>::min(),
      DefaultInferCopyLayout,
      DefaultSelectCopyInst,
      DefaultLowerCopy,
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

CopyInst SelectCopyInstForTarget(const CopyNode &op, Target target,
                                 const LayoutMap &layout_map,
                                 arith::Analyzer *analyzer, bool buffer_oob) {
  return ResolveCopyImpl(target).select_inst(op, target, layout_map, analyzer,
                                             buffer_oob);
}

Stmt LowerCopyForTarget(const CopyNode &op, const LowerArgs &T,
                        arith::Analyzer *analyzer) {
  return ResolveCopyImpl(T.target).lower(op, T, analyzer);
}

} // namespace

void RegisterCopyImpl(CopyImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.infer_layout != nullptr);
  ICHECK(impl.select_inst != nullptr);
  ICHECK(impl.lower != nullptr);
  EnsureDefaultCopyImplRegistered();
  CopyImplRegistry().push_back(impl);
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

// Infers memory layouts for this Copy operation based on target and copy
// instruction.
LayoutMap CopyNode::InferLayoutImpl(const LayoutInferArgs &T,
                                    InferLevel level) const {
  auto target = T.target;
  CopyInst copy_inst;
  if (GetIsAsyncCopy()) {
    // Layout inference does not require a full cp.async legality proof (which
    // depends on final vectorization decisions). Keep the op as CPAsync for
    // inference, and enforce legality during lowering.
    if (!TargetHasAsyncCopy(target)) {
      LOG(FATAL) << "T.async_copy is only supported on targets with cp.async "
                    "support (SM80+). Got target="
                 << target;
    }
    if (!IsGlobalBuffer(src) || !IsSharedBuffer(dst)) {
      LOG(FATAL)
          << "T.async_copy only supports global->shared/shared.dyn copies. "
          << "Got src=" << src->name << " (scope=" << src.scope()
          << "), dst=" << dst->name << " (scope=" << dst.scope() << ").";
    }
    if (src->dtype != dst->dtype) {
      LOG(FATAL) << "T.async_copy requires equal byte-addressable dtypes. "
                 << "Got src dtype=" << src->dtype
                 << ", dst dtype=" << dst->dtype << ".";
    }
    copy_inst = CopyInst::kCPAsync;
  } else {
    copy_inst = GetCopyInst(target, T.layout_map, T.analyzer, T.buffer_oob);
  }

  // If user annotated a loop layout on T.copy, enforce SIMT (normal) copy.
  // Parallel-loop layout only applies to SIMT-style loops we generate here;
  // other copy instructions (TMA/LDSM/STSM/TMem) are incompatible.
  if (annotations.count(attr::kParallelLoopLayout)) {
    if (copy_inst != CopyInst::kNormal && copy_inst != CopyInst::kCPAsync) {
      std::ostringstream oss;
      oss << "T.copy loop layout annotation requires SIMT copy; got "
          << CopyInstToString(copy_inst) << " for src=" << src->name
          << ", dst=" << dst->name
          << ". Remove loop_layout or change copy pattern.";
      LOG(FATAL) << oss.str();
    }
  }

  // Handle tensor memory (tmem) layout inference for both load and store
  if (copy_inst == CopyInst::kTMemLoad || copy_inst == CopyInst::kTMemStore) {
    // TODO (mzw) Add support for tcgen05.cp in CUDA tmem lowering.
    LayoutMap results;
    bool is_tmem_load = (copy_inst == CopyInst::kTMemLoad);
    Buffer tmem_buf = is_tmem_load ? src : dst;
    Buffer reg_buf = is_tmem_load ? dst : src;

    if (!T.layout_map.count(reg_buf) && T.layout_map.count(tmem_buf)) {
      Layout tmem_layout = T.layout_map[tmem_buf];
      Array<IterVar> logical_coords = MakeIterVars();
      Array<PrimExpr> logical_coords_var = {logical_coords[0]->var,
                                            logical_coords[1]->var};
      Array<PrimExpr> phy_indices = tmem_layout->Forward(logical_coords_var);

      // Tmem physical coord range analysis
      auto analyzer = std::make_shared<arith::Analyzer>();
      for (const auto &iv : logical_coords)
        analyzer->Bind(iv->var, iv->dom);
      arith::ConstIntBound phy_row_bounds =
          analyzer->const_int_bound(phy_indices[0]);
      arith::ConstIntBound phy_col_bounds =
          analyzer->const_int_bound(phy_indices[1]);
      Range row_dom = Range((int)(phy_row_bounds->min_value),
                            (int)(phy_row_bounds->max_value + 1));
      Range col_dom = Range((int)(phy_col_bounds->min_value),
                            (int)(phy_col_bounds->max_value + 1));

      constexpr int WARP_SIZE = 32;
      constexpr int WARPGROUP_SIZE = 4 * WARP_SIZE;
      ICHECK(is_const_int(T.thread_bounds->extent))
          << "Tensor memory copy requires thread_bounds->extent (num_threads) "
             "to be constant integers";
      int num_threads = *as_const_int(T.thread_bounds->extent);
      ICHECK(num_threads % WARPGROUP_SIZE == 0)
          << "Tensor memory copy requires thread bounds to be aligned to "
             "warpgroups, but found "
          << "thread range = " << T.thread_bounds;

      for (int num_useful_wgs = num_threads / WARPGROUP_SIZE;
           num_useful_wgs >= 1; --num_useful_wgs) {
        int num_useful_threads = num_useful_wgs * WARPGROUP_SIZE;
        Tcgen05Meta meta = getTcgen05MetaLd_32dp32b();
        auto [is_success, tmem_coord2frag, num_chunks_each_wg] =
            expandTcgen05Layout(
                meta, phy_col_bounds->max_value - phy_col_bounds->min_value + 1,
                num_useful_threads, row_dom, col_dom);
        if (!is_success) {
          continue;
        }
        Fragment logical_coord2frag =
            Fragment(logical_coords, tmem_coord2frag->Forward(phy_indices),
                     tmem_coord2frag->ForwardThread(phy_indices, std::nullopt),
                     make_itervar("rep", 1));
        results.Set(reg_buf,
                    logical_coord2frag->BindThreadRange(T.thread_bounds));
        break;
      }
    }

    return results;
  }

  if (copy_inst == CopyInst::kBulkLoad || copy_inst == CopyInst::kBulkStore ||
      copy_inst == CopyInst::kBulkLoad1D ||
      copy_inst == CopyInst::kBulkStore1D) {
    // if can apply swizzling, we skip layout inference
    // for bulk load/store, we can directly apply the layout of normal copy
    // This must be a global/shared layout, so we can skip the parallel op
    // layout inference (parallel layout inference only annotate the loop layout
    // and the register layout).
    Map<Buffer, Layout> result_map;

    bool is_tma_1d = copy_inst == CopyInst::kBulkLoad1D ||
                     copy_inst == CopyInst::kBulkStore1D;
    bool is_load =
        copy_inst == CopyInst::kBulkLoad || copy_inst == CopyInst::kBulkLoad1D;
    bool is_store = copy_inst == CopyInst::kBulkStore ||
                    copy_inst == CopyInst::kBulkStore1D;
    auto global_tensor = is_load ? src : dst;
    auto shared_tensor = is_load ? dst : src;
    auto shared_range = is_load ? dst_range : src_range;

    if (is_tma_1d && shared_range.size() == 1) {
      // 1D TMA Store with single dimension can not be swizzled
      // But 1D TMA can also have multiple dimensions when the last
      // dimension is continuous.
      return result_map;
    }

    // Collect fragment buffers from indices and mark them as fully replicated
    // For Bulk Load/Store, fragment buffers used as indices should be
    // replicated across all threads
    PrimExpr thread_extent = T.thread_bounds->extent;
    for (const auto &range : src_range) {
      CollectFragmentLayouts(range->min, T.let_var_to_expr, T.layout_map,
                             thread_extent, T.thread_bounds, result_map);
      CollectFragmentLayouts(range->extent, T.let_var_to_expr, T.layout_map,
                             thread_extent, T.thread_bounds, result_map);
    }
    for (const auto &range : dst_range) {
      CollectFragmentLayouts(range->min, T.let_var_to_expr, T.layout_map,
                             thread_extent, T.thread_bounds, result_map);
      CollectFragmentLayouts(range->extent, T.let_var_to_expr, T.layout_map,
                             thread_extent, T.thread_bounds, result_map);
    }

    // check shared layout is non-swizzle
    // skip layout inference if shared layout is already annotated
    if (level == InferLevel::kFree && !T.layout_map.count(shared_tensor)) {
      if (is_store) {
        // For BulkStore, we should perform swizzle if possible.
        // TMA Store is always 1d like, we can directly use the last two
        // dimensions to analysis swizzling.
        int dim = shared_tensor->shape.size();
        const int64_t mat_stride = *as_const_int(shared_tensor->shape[dim - 2]);
        const int64_t mat_continuous =
            *as_const_int(shared_tensor->shape[dim - 1]);
        Layout swizzle_layout_2d = makeGemmABLayoutHopper(
            mat_stride, mat_continuous, mat_continuous,
            shared_tensor->dtype.bits(), /*k_inner=*/true);
        // If makeGemmABLayoutHopper returns a linear layout, fallback to
        // ComputeLinearLayout which handles arbitrary tensor shapes correctly.
        if (StructuralEqual()(
                swizzle_layout_2d,
                makeLinearLayout(Array<PrimExpr>{Integer(mat_stride),
                                                 Integer(mat_continuous)}))) {
          result_map.Set(shared_tensor, ComputeLinearLayout(shared_tensor));
        } else {
          result_map.Set(shared_tensor, ExpandLayoutToMatchBuffer(
                                            swizzle_layout_2d, shared_tensor));
        }
      } else if (level == InferLevel::kFree) {
        // create a new layout map for tma linear layout
        Layout linear_layout = ComputeLinearLayout(shared_tensor);
        result_map.Set(shared_tensor, linear_layout);
      }
    }
    return result_map;
  }

  // for LDSM/STSM, the layout was deduced from register layout
  // so we can directly apply the layout of normal copy
  // Use parallel op to infer the layout
  if (!par_op_.defined()) {
    arith::Analyzer analyzer;
    par_op_ = ParallelOp((MakeSIMTLoop(&analyzer)));
  }
  auto layout_map = par_op_->InferLayout(T, level);
  return layout_map;
}
// Shared stride validation for TMA bulk load/store.
bool CopyNode::CheckGlobalStrides(const Buffer &buffer,
                                  arith::Analyzer *analyzer) {
  Array<PrimExpr> strides = buffer->strides;
  if (strides.empty()) {
    PrimExpr stride = 1;
    strides.resize(buffer->shape.size());
    for (int i = static_cast<int>(buffer->shape.size()) - 1; i >= 0; --i) {
      strides.Set(i, stride);
      stride *= buffer->shape[i];
    }
  }

  if (!strides.empty() &&
      analyzer->CanProve(strides[strides.size() - 1] != 1,
                         arith::ProofStrength::kSymbolicBound)) {
    LOG(WARNING) << "TMA bulk copy requires contiguous innermost global stride"
                 << ", but got " << strides[strides.size() - 1]
                 << " for buffer " << buffer->name
                 << ", fallback to normal copy.";
    return false;
  }

  for (size_t i = 0; i + 1 < strides.size(); ++i) {
    PrimExpr stride_bytes = TMABytesFromElements(strides[i], buffer->dtype);
    if (analyzer->CanProve(
            FloorMod(stride_bytes, IntImm(DataType::Int(64), 16)) != 0,
            arith::ProofStrength::kSymbolicBound)) {
      LOG(WARNING) << "TMA bulk copy cannot support a global stride of "
                   << stride_bytes << " for buffer " << buffer->name
                   << ", fallback to normal copy.";
      return false;
    }
    if (const int64_t *stride =
            as_const_int(analyzer->Simplify(stride_bytes))) {
      if (*stride >= (int64_t{1} << 40)) {
        LOG(WARNING) << "TMA bulk copy cannot support a global stride of "
                     << stride_bytes << " for buffer " << buffer->name
                     << ", fallback to normal copy.";
        return false;
      }
    }
  }
  return true;
}

// Checks if this copy can be lowered to a Bulk Load (TMA) instruction.
// Requires: TMA support, global->shared scope, matching dtypes.
bool CopyNode::CheckBulkLoad(Target target, arith::Analyzer *analyzer,
                             bool check_last_dim) const {
  // 1. arch must have bulk copy support
  if (!TargetHasBulkCopy(target))
    return false;
  // 2. src and dst must be global and shared
  if (src.scope() != "global" ||
      (dst.scope() != "shared.dyn" && dst.scope() != "shared"))
    return false;
  // 3. check shape.
  // last dim of src * dtype.bits() must be a multiple of 16
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
  // now we check src (gmem) as tma box dim is deduced from src
  if (check_last_dim &&
      analyzer->CanProve(
          FloorMod(TMABitsFromElements(src_range[src_range.size() - 1]->extent,
                                       src->dtype),
                   IntImm(DataType::Int(64), 128)) != 0,
          arith::ProofStrength::kSymbolicBound)) {
    LOG(WARNING)
        << "src range must have last dim multiple of 16 for tma bulk load "
        << src->name << " range " << src_range[src_range.size() - 1]->extent
        << " * " << src->dtype.bits() << " bits % 128 != 0";
    return false;
  }

  // 4. src and dst must have the same dtype
  if (src->dtype != dst->dtype) {
    LOG(WARNING) << "src and dst must have the same dtype for tma load "
                 << src->name << " vs. " << dst->name << " dtype " << src->dtype
                 << " vs. " << dst->dtype << " will be fallback to normal copy";
    return false;
  }
  if (!CheckGlobalStrides(src, analyzer))
    return false;
  return true;
}

bool CopyNode::CheckBulkCopy1D(const Buffer &global_tensor,
                               const Buffer &shared_tensor,
                               const Array<Range> &global_range,
                               const Array<Range> &shared_range,
                               const LayoutMap &layout_map,
                               arith::Analyzer *analyzer) const {

  // Step 1: check shared is contiguous (linear layout is also contiguous)
  bool shared_is_contiguous = true;
  if (layout_map.count(shared_tensor)) {
    // Check if the layout is linear
    Layout existing =
        layout_map.Get(shared_tensor).value().as<Layout>().value();
    Layout linear_layout = makeLinearLayout(shared_tensor->shape);
    shared_is_contiguous = StructuralEqual()(existing, linear_layout);
  }
  // Step 2: check global is contiguous
  bool global_is_contiguous = true;
  bool global_not_full_dim_encounter = false;
  for (int i = global_range.size() - 1; i >= 0; i--) {
    if (!global_not_full_dim_encounter) {
      if (!analyzer->CanProve(global_range[i]->extent ==
                                      global_tensor->shape[i] &&
                                  global_range[i]->min == 0,
                              arith::ProofStrength::kSymbolicBound)) {
        global_not_full_dim_encounter = true;
      }
    } else {
      if (!analyzer->CanProve(global_range[i]->extent == 1,
                              arith::ProofStrength::kSymbolicBound)) {
        global_is_contiguous = false;
        break;
      }
    }
  }

  // Step 3: check element match and no OOB
  PrimExpr shared_elements = 1;
  for (size_t i = 0; i < shared_range.size(); i++) {
    shared_elements *= shared_range[i]->extent;
  }
  PrimExpr global_elements = 1;
  for (size_t i = 0; i < global_range.size(); i++) {
    global_elements *= global_range[i]->extent;
  }
  bool element_match =
      analyzer->CanProveEqual(shared_elements, global_elements);

  return (shared_is_contiguous && global_is_contiguous && element_match);
}

bool CopyNode::CheckBulkLoad1D(Target target, const LayoutMap &layout_map,
                               arith::Analyzer *analyzer) const {
  if (!CheckBulkLoad(target, analyzer, false))
    return false;
  auto global_tensor = src;
  auto shared_tensor = dst;
  auto global_range = src_range;
  auto shared_range = dst_range;
  return CheckBulkCopy1D(global_tensor, shared_tensor, global_range,
                         shared_range, layout_map, analyzer);
}

bool CopyNode::CheckBulkStore1D(Target target, const LayoutMap &layout_map,
                                arith::Analyzer *analyzer) const {
  if (!CheckBulkStore(target, analyzer, false))
    return false;
  auto shared_tensor = src;
  auto global_tensor = dst;
  auto shared_range = src_range;
  auto global_range = dst_range;
  return CheckBulkCopy1D(global_tensor, shared_tensor, global_range,
                         shared_range, layout_map, analyzer);
}

// Checks if this copy can be lowered to a Bulk Store (TMA) instruction.
// Requires: TMA support, shared->global scope, matching dtypes.
bool CopyNode::CheckBulkStore(Target target, arith::Analyzer *analyzer,
                              bool check_last_dim) const {
  // 1. arch must have bulk copy support
  if (!TargetHasBulkCopy(target))
    return false;
  // 2. src and dst must be shared.dyn and local.fragment
  if ((src.scope() != "shared.dyn" && src.scope() != "shared") ||
      dst.scope() != "global")
    return false;
  // 3. check shape.
  // last dim of dst * dtype.bits() must be a multiple of 16
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
  // now we check dst (gmem) as tma box dim is deduced from dst
  if (check_last_dim &&
      analyzer->CanProve(
          FloorMod(TMABitsFromElements(dst_range[dst_range.size() - 1]->extent,
                                       dst->dtype),
                   IntImm(DataType::Int(64), 128)) != 0,
          arith::ProofStrength::kSymbolicBound)) {
    LOG(WARNING)
        << "dst range must have last dim multiple of 16 for tma bulk store "
        << dst->name << " range " << dst_range[dst_range.size() - 1]->extent
        << " * " << dst->dtype.bits() << " bits % 128 != 0";
    return false;
  }
  // 4. src and dst must have the same dtype
  if (src->dtype != dst->dtype) {
    LOG(WARNING) << "src and dst must have the same dtype for tma store "
                 << src->name << " vs. " << dst->name << " dtype " << src->dtype
                 << " vs. " << dst->dtype << " will be fallback to normal copy";
    return false;
  }
  if (!CheckGlobalStrides(dst, analyzer))
    return false;
  return true;
}

// Checks if copy can use CUDA's Load Matrix (LDSM) instruction.
// Requires: LDMATRIX support, shared->fragment scope.
bool CopyNode::CheckLDSMCopy(Target target) const {
  return TargetHasLdmatrix(target) && IsSharedBuffer(src) &&
         IsFragmentBuffer(dst);
}

// Checks if copy can use CUDA's Store Matrix (STSM) instruction.
// Requires: STMATRIX support, fragment->shared scope.
bool CopyNode::CheckSTSMCopy(Target target) const {
  return TargetHasStmatrix(target) && IsFragmentBuffer(src) &&
         IsSharedBuffer(dst);
}

// Checks if copy can use tensor memory load (tcgen05.ld).
// Requires: tmem support, shared.tmem->fragment scope.
bool CopyNode::CheckTMemLoad(Target target) const {
  return TargetHasTmem(target) && src.scope() == "shared.tmem" &&
         IsFragmentBuffer(dst);
}

// Checks if copy can use tensor memory store (tcgen05.st).
// Requires: tmem support, fragment->shared.tmem scope.
bool CopyNode::CheckTMemStore(Target target) const {
  return TargetHasTmem(target) && IsFragmentBuffer(src) &&
         dst.scope() == "shared.tmem";
}

// Checks if copy can use cp.async global->shared path.
// Requirements:
// - target has async copy capability
// - source is global and destination is shared/shared.dyn
// - source/destination dtypes match
// - vectorized copy width (bytes) is one of {4, 8, 16}
// - if OOB guards are required, only a *uniform* (scalar) source predicate
//   is supported (dst must be in-bounds)
bool CopyNode::CheckCPAsyncCopyPreconditions() const {
  if (!IsGlobalBuffer(src) || !IsSharedBuffer(dst)) {
    return false;
  }
  if (src->dtype != dst->dtype) {
    return false;
  }
  return true;
}

bool CopyNode::CheckPipelineManagedCPAsyncCopy() const {
  return !GetIsTmaCopy() && !GetIsAsyncCopy() &&
         CheckCPAsyncCopyPreconditions();
}

bool CopyNode::CheckPipelineManagedCPAsyncCopy(
    Target target, arith::Analyzer *analyzer) const {
  return CheckPipelineManagedCPAsyncCopy() &&
         CheckCPAsyncCopy(target, LayoutMap(), analyzer);
}

bool CopyNode::CheckCPAsyncCopy(Target target, const LayoutMap &layout_map,
                                arith::Analyzer *analyzer) const {
  if (!TargetHasAsyncCopy(target)) {
    return false;
  }
  if (!CheckCPAsyncCopyPreconditions()) {
    return false;
  }
  // Skip vectorize size check here because, during the Infer Layout stage,
  // the layout is not stable and the vectorized size cannot be determined.
  return true;
}

// Selects the most specific copy instruction for the given target and buffers.
// Priority: BulkLoad1D, BulkStore1D, BulkLoad, BulkStore, LDSM, STSM,
// TMemLoad, TMemStore, CPAsync, Normal.
CopyInst CopyNode::GetCopyInst(Target target, const LayoutMap &layout_map,
                               arith::Analyzer *analyzer,
                               bool buffer_oob) const {
  return SelectCopyInstForTarget(*this, target, layout_map, analyzer,
                                 buffer_oob);
}

// Lowers the copy operation to PTX code by dispatching to specialized lowering
// functions.
Stmt CopyNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  return LowerCopyForTarget(*this, T, analyzer);
}

// Encodes the TMA descriptor into an array of PrimExpr for
// create_tma_descriptor().
Array<PrimExpr> TMADesc::EncodeCallArgs() const {
  Array<PrimExpr> args;
  args.reserve(rank * 4 + 7);

  args.push_back(data_type);
  args.push_back(static_cast<int>(rank));
  args.push_back(global_addr);
  for (auto e : global_shape)
    args.push_back(e);
  for (auto e : global_stride)
    args.push_back(e);
  for (auto e : smem_box)
    args.push_back(e);
  for (auto e : smem_stride)
    args.push_back(e);
  args.push_back(interleave);
  args.push_back(swizzle);
  args.push_back(l2_promotion);
  args.push_back(oob_fill);

  return args;
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

// Lowers Conv2D im2col into a TMA-backed PTX sequence for Hopper.
Stmt Conv2DIm2ColOpNode::Lower(const LowerArgs &T,
                               arith::Analyzer *analyzer) const {
  ICHECK(TargetIsHopper(T.target));
  ICHECK(IsGlobalBuffer(src_) && IsSharedBuffer(dst_));
  ICHECK(src_->shape.size() == 4);
  ICHECK(src_->dtype == dst_->dtype);

  // Use dstRegion_ to derive tile dimensions and shared memory offset.
  // dstRegion_ always has the correct ranges regardless of whether MVB
  // added a leading stage dimension to the buffer — the last two ranges
  // give the tile (pixel, channel) extents and mins.
  size_t ndim = dstRegion_->region.size();
  ICHECK(ndim >= 2) << "im2col dstRegion must have at least 2 dims";
  Layout shared_layout;
  if (T.layout_map.count(dst_)) {
    shared_layout = T.layout_map[dst_];
  }

  TMAIm2ColDesc desc;
  desc.rank = src_->shape.size();
  desc.data_type = to_CUtensorMapDataType(src_->dtype);
  desc.global_addr = src_->data;
  desc.global_shape = ReverseArray(src_->shape);

  if (!src_->strides.empty()) {
    desc.global_stride = ReverseArray(src_->strides);
  } else {
    // Create stride from shape
    PrimExpr stride = 1;
    desc.global_stride.reserve(desc.rank);
    for (size_t i = 0; i < desc.rank; i++) {
      desc.global_stride.push_back(stride);
      stride *= desc.global_shape[i];
    }
  }
  // The first stride element should be 1
  ICHECK(is_one(desc.global_stride[0])) << desc.global_stride;
  // Make global stride in bytes
  desc.global_stride = desc.global_stride.Map(
      [&](PrimExpr e) { return TMABytesFromElements(e, src_->dtype); });
  desc.elem_stride = {1, stride_, stride_, 1};
  desc.lower_corner = {-padding_, -padding_};
  desc.upper_corner = {-padding_, -padding_};
  desc.smem_box_pixel =
      Downcast<IntImm>(dstRegion_->region[ndim - 2]->extent)->value;
  desc.smem_box_channel =
      Downcast<IntImm>(dstRegion_->region[ndim - 1]->extent)->value;
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);
  if (!shared_layout.defined()) {
    desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
  } else {
    ICHECK(shared_layout->InputDim() >= 2) << "Cannot detect TMA layout.";
    if (StructuralEqual()(shared_layout, makeQuarterBankSwizzleLayout(dst_))) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B);
    } else if (StructuralEqual()(shared_layout,
                                 makeHalfBankSwizzleLayout(dst_))) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B);
    } else if (StructuralEqual()(shared_layout,
                                 makeFullBankSwizzleLayout(dst_))) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B);
    } else {
      LOG(FATAL) << "Cannot detect TMA layout.";
    }
  }

  Call create_desc = Call(DataType::Handle(), create_tma_im2col_descriptor(),
                          desc.EncodeCallArgs());

  Array<PrimExpr> global_coords; // c, w, h, n
  Array<PrimExpr> image_offset;  // w, h
  global_coords.reserve(desc.rank);

  ICHECK(analyzer->CanProveEqual(
      FloorMod(desc.global_shape[0], desc.smem_box_channel), 0))
      << "Currently can only support divisible channel case";

  global_coords.push_back(
      FloorMod(c_step_ * desc.smem_box_channel, desc.global_shape[0]));
  image_offset.push_back(
      dilation_ *
      FloorMod(FloorDiv(c_step_ * desc.smem_box_channel, desc.global_shape[0]),
               kernel_));
  image_offset.push_back(dilation_ * FloorDiv(c_step_ * desc.smem_box_channel,
                                              desc.global_shape[0] * kernel_));

  PrimExpr h_dim =
      FloorDiv(src_->shape[1] + 2 * padding_ - (kernel_ - 1) * dilation_ - 1,
               stride_) +
      1;
  PrimExpr w_dim =
      FloorDiv(src_->shape[2] + 2 * padding_ - (kernel_ - 1) * dilation_ - 1,
               stride_) +
      1;
  global_coords.push_back(
      stride_ * FloorMod(nhw_step_ * desc.smem_box_pixel, w_dim) - padding_);
  global_coords.push_back(
      stride_ *
          FloorMod(FloorDiv(nhw_step_ * desc.smem_box_pixel, w_dim), h_dim) -
      padding_);
  global_coords.push_back(
      FloorDiv(nhw_step_ * desc.smem_box_pixel, w_dim * h_dim));

  // Allocate mbarrier(s) for TMA im2col load synchronization,
  // matching the protocol used by regular TMA loads.
  // If a barrier was provided by the WS pass (via annotation), use it directly.
  int barrier_base_id = -1;
  PrimExpr mbar_handle;
  if (auto user_barrier = annotations_.Get("barrier")) {
    // WS pass provided a barrier: use it without allocating a new one.
    mbar_handle = Downcast<PrimExpr>(user_barrier.value());
    barrier_base_id = 0;
  } else if (T.AllocMBarrier) {
    // Allocate a single barrier slot; pipeline buffer versioning expands it
    // per stage when needed.
    barrier_base_id = T.AllocMBarrier(1);
    PrimExpr mbar_idx = IntImm(DataType::Int(32), barrier_base_id);
    mbar_handle = BufferLoad(T.mbarrier_buffer->value(), {mbar_idx});
  }

  Array<PrimExpr> args;
  args.reserve(desc.rank * 2 + 2);
  args.push_back(create_desc);
  args.push_back(barrier_base_id >= 0 ? mbar_handle : PrimExpr(0));
  auto dst_buffer = T.buffer_remap.count(dst_) ? T.buffer_remap[dst_] : dst_;
  // Compute flat element offset from dstRegion_ mins and buffer strides.
  // For a plain 2D buffer this is 0; for a versioned 3D buffer this
  // resolves to stage_idx * pixel * channel — no special-casing needed.
  PrimExpr flat_offset = IntImm(DataType::Int(32), 0);
  {
    PrimExpr stride = IntImm(DataType::Int(32), 1);
    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
      flat_offset = flat_offset + dstRegion_->region[i]->min * stride;
      stride = stride * dst_->shape[i];
    }
  }
  PrimExpr tile_elems =
      IntImm(DataType::Int(32), desc.smem_box_pixel * desc.smem_box_channel);
  PrimExpr shared_addr = dst_buffer.access_ptr(
      /*access_mask=*/2, /*dtype=*/DataType::Handle(), /*content_lanes=*/1,
      /*offset=*/flat_offset, /*extent=*/tile_elems);
  args.push_back(shared_addr);
  for (auto coord : global_coords)
    args.push_back(coord);
  for (auto offset : image_offset)
    args.push_back(offset);
  args.push_back(this->eviction_policy_);
  Stmt tma_copy_stmt =
      Evaluate(Call(DataType::Handle(), tma_load_im2col(), args));

  if (barrier_base_id >= 0) {
    bool ws_barrier = annotations_.Get("barrier").has_value();
    // Total bytes transferred by im2col TMA copy
    PrimExpr total_bytes = TMABytesFromElements(
        IntImm(DataType::Int(32), desc.smem_box_pixel * desc.smem_box_channel),
        dst_->dtype);

    Stmt barrier_before_tma_stmt = Evaluate(Call(
        DataType::Handle(), mbarrier_expect_tx(), {mbar_handle, total_bytes}));

    if (ws_barrier) {
      // External barrier (WS pass or InjectSoftwarePipeline).
      // Build: expect_tx + tma_load [+ arrive if emit_arrive is set].
      Array<Stmt> producer_seq{barrier_before_tma_stmt, tma_copy_stmt};
      if (auto emit_arrive_val = annotations_.Get("emit_arrive")) {
        if (Downcast<IntImm>(emit_arrive_val.value())->value != 0) {
          producer_seq.push_back(
              Evaluate(Call(DataType::Handle(), builtin::ptx_arrive_barrier(),
                            {mbar_handle})));
        }
      }
      Stmt producer =
          IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent),
                     SeqStmt(producer_seq));
      return producer;
    }

    Stmt barrier_after_tma_stmt = Evaluate(
        Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {mbar_handle}));

    // Thread-gated block: expect_tx + tma_load_im2col + arrive
    Stmt producer = IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent),
                               SeqStmt({barrier_before_tma_stmt, tma_copy_stmt,
                                        barrier_after_tma_stmt}));

    // Emit producer + wait pair for pipeline/WS passes.
    Stmt wait_stmt =
        Evaluate(Call(DataType::Handle(), mbarrier_wait_parity(),
                      {mbar_handle, GetCopyMbarPhaseExpr(annotations_, T)}));

    return SeqStmt({producer, wait_stmt});
  }

  Stmt tma_copy = IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent),
                             tma_copy_stmt);
  return tma_copy;
}

// Encodes the TMA im2col descriptor for create_tma_im2col_descriptor().
Array<PrimExpr> TMAIm2ColDesc::EncodeCallArgs() const {
  Array<PrimExpr> args;
  args.reserve(rank * 5 + 5);

  args.push_back(data_type);
  args.push_back(static_cast<int>(rank));
  args.push_back(global_addr);
  for (auto e : global_shape)
    args.push_back(e);
  for (auto e : global_stride)
    args.push_back(e);
  for (auto e : elem_stride)
    args.push_back(e);
  for (auto e : lower_corner)
    args.push_back(e);
  for (auto e : upper_corner)
    args.push_back(e);
  args.push_back(smem_box_pixel);
  args.push_back(smem_box_channel);
  args.push_back(interleave);
  args.push_back(swizzle);
  args.push_back(l2_promotion);
  args.push_back(oob_fill);

  return args;
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
