/*!
 * \file tl/op/copy.cc
 * \brief Define the copy operator, backend dispatch, and shared normal-copy
 *        lowering helpers.
 */

#include "copy.h"
#include "../transform/common/loop_fusion_utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"
#include "ir/analyzer.h"
#include "span_utils.h"
#include "support/check.h"
#include "utils.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/runtime/logging.h>

#include "builtin.h"
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/attr.h>

#include <limits>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;
using namespace ffi;

Stmt LowerNormalCopy(const CopyNode &op, const LowerArgs &lower_args,
                     sym::AnalyzerObj *analyzer) {
  bool is_cpu_target = lower_args.target->GetTargetDeviceType() == kDLCPU;
  auto simt_loop = op.MakeSIMTLoop(analyzer);
  // Explicit layouts address the original copy axes. Keep that domain when
  // lowering, as InferSIMTLayout does; fusing it would leave a multi-axis
  // layout attached to a single loop without remapping its coordinates.
  auto fused_loop = op.annotations.count(attr::kParallelLoopLayout)
                        ? simt_loop
                        : ParallelLoopFuser::Fuse(simt_loop).as_or_throw<For>();

  auto par_op = ParallelOp(fused_loop, op.annotations);

  if (is_cpu_target || IsLocalBuffer(op.src) || IsLocalBuffer(op.dst)) {
    if (IsLocalBuffer(op.src) && !IsLocalBuffer(op.dst)) {
      // A conflict write only occurs when multiple threads write to the same
      // global address. If any dst_range dimension's min depends on the thread
      // index, each thread targets a distinct location and there is no
      // conflict. The thread index is an expression: collect the variables it
      // uses by identity (the real threadIdx.x Var on GPU; none when it is a
      // constant, e.g. 0 on CPU).
      std::unordered_set<const VarObj *> thread_index_vars;
      ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
          lower_args.thread_index,
          [&](const VarObj *v) -> ffi::Expected<ffi::WalkResult> {
            thread_index_vars.insert(v);
            return ffi::WalkResult::Skip();
          });
      bool dst_depends_on_thread = false;
      for (const auto &range : op.dst_range) {
        if (ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
                range->begin,
                [&](const VarObj *v) -> ffi::Expected<ffi::WalkResult> {
                  return (thread_index_vars.count(v) != 0)
                             ? ffi::WalkResult::Interrupt(
                                   ffi::VisitInterrupt(ffi::GetRef<Var>(v)))
                             : ffi::WalkResult::Skip();
                })
                .has_value()) {
          dst_depends_on_thread = true;
          break;
        }
      }
      if (!dst_depends_on_thread) {
        DLOG(WARNING) << "Copy from local buffer `" << op.src->name << "` to "
                      << op.dst.scope() << " buffer `" << op.dst->name
                      << "` may cause conflicted write.";
      }
    }
    auto vectorized_thread_loop =
        VectorizeLoop(fused_loop, lower_args.layout_map, -1, lower_args.target);
    return vectorized_thread_loop;
  }

  std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                    InferLevel::kFree};
  for (auto level : levels) {
    par_op->InferLayout({lower_args.target,
                         lower_args.thread_bounds,
                         lower_args.layout_map,
                         analyzer,
                         lower_args.buffer_remap,
                         {}},
                        level);
  }
  auto loop_layout = par_op->GetLoopLayout();
  return LowerParallelLoop(
      par_op->GetRoot(), loop_layout, lower_args.thread_index, analyzer,
      lower_args.layout_map, par_op->GetPredicate(lower_args.thread_index),
      /*parallel_loop=*/true, par_op->LoopLayoutRequiresPaddingGuard(),
      lower_args.target);
}

namespace {

TileOperator ApplyCopyBlockAnnotations(TileOperator tile_op,
                                       BlockAnnotations block_annotations) {
  Copy copy = (tile_op).template as_or_throw<Copy>();

  // Safe because this handler is invoked immediately after TLOpBuilder creates
  // a fresh CopyNode, before the node escapes ParseOperator.
  auto *node = const_cast<CopyNode *>(copy.operator->());
  ICHECK(node != nullptr);

  node->src_oob_safe_value = std::nullopt;
  auto safe_value_map_obj = block_annotations.Get(attr::kSafeValueMap);
  if (!safe_value_map_obj) {
    return copy;
  }

  auto safe_value_map = NormalizeSafeValueMap(*safe_value_map_obj);
  auto it = safe_value_map.find(node->src);
  if (it != safe_value_map.end()) {
    node->src_oob_safe_value = (*it).second;
  }
  return copy;
}

std::vector<CopyImpl> &CopyImplRegistry() {
  static std::vector<CopyImpl> registry;
  return registry;
}

const CopyImpl &ResolveCopyImpl(Target target) {
  const auto &registry = CopyImplRegistry();
  const CopyImpl *best_impl = nullptr;
  int best_priority = std::numeric_limits<int>::min();
  for (const CopyImpl &impl : registry) {
    if (impl.match_target(target) && impl.priority >= best_priority) {
      best_impl = &impl;
      best_priority = impl.priority;
    }
  }
  ICHECK(best_impl != nullptr)
      << "tl.copy requires a target-specific implementation, but no copy "
         "implementation is registered for "
      << target->str();
  return *best_impl;
}

LayoutMap InferCopyLayout(const CopyNode &op,
                          const LayoutInferArgs &layout_args,
                          InferLevel level) {
  return ResolveCopyImpl(layout_args.target)
      .infer_layout(op, layout_args, level);
}

Stmt LowerCopyForTarget(const CopyNode &op, const LowerArgs &lower_args,
                        sym::AnalyzerObj *analyzer) {
  return ResolveCopyImpl(lower_args.target).lower(op, lower_args, analyzer);
}

std::vector<Im2ColImpl> &Im2ColImplRegistry() {
  static std::vector<Im2ColImpl> registry;
  return registry;
}

const Im2ColImpl &ResolveIm2ColImpl(Target target) {
  const auto &registry = Im2ColImplRegistry();
  const Im2ColImpl *best_impl = nullptr;
  int best_priority = std::numeric_limits<int>::min();
  for (const Im2ColImpl &impl : registry) {
    if (impl.match_target(target) && impl.priority >= best_priority) {
      best_impl = &impl;
      best_priority = impl.priority;
    }
  }
  ICHECK(best_impl != nullptr)
      << "Im2Col requires a target-specific implementation, but no "
         "implementation is registered for "
      << target->str();
  return *best_impl;
}

Stmt LowerIm2ColForTarget(const Im2ColOpNode &op, const LowerArgs &lower_args,
                          sym::AnalyzerObj *analyzer) {
  return ResolveIm2ColImpl(lower_args.target).lower(op, lower_args, analyzer);
}

bool MatchAnyIm2ColTarget(Target /*target*/) { return true; }

Stmt LowerIm2ColSIMT(const Im2ColOpNode &op, const LowerArgs &lower_args,
                     sym::AnalyzerObj *analyzer) {
  const TensorVar &src = op.src_;
  const TensorVar &dst = op.dst_;
  const TensorRegion &dst_region = op.dstRegion_;

  ICHECK(src.shape().size() == 4);
  ICHECK(src.dtype() == dst.dtype());

  size_t ndim = dst_region->region.size();
  ICHECK(ndim >= 2) << "im2col dstRegion must have at least 2 dims";

  PrimExpr block_m = dst_region->region[ndim - 2]->extent;
  PrimExpr block_k = dst_region->region[ndim - 1]->extent;
  PrimVar i("i", block_m.ty());
  PrimVar j("j", block_k.ty());
  BindAnalyzer(analyzer, i, Range::FromMinExtent(IntImm(i.ty(), 0), block_m));
  BindAnalyzer(analyzer, j, Range::FromMinExtent(IntImm(j.ty(), 0), block_k));

  PrimExpr h_dim = src.shape()[1];
  PrimExpr w_dim = src.shape()[2];
  PrimExpr c_dim = src.shape()[3];
  PrimExpr out_h =
      floordiv(h_dim + 2 * op.padding_ - (op.kernel_ - 1) * op.dilation_ - 1,
               op.stride_) +
      1;
  PrimExpr out_w =
      floordiv(w_dim + 2 * op.padding_ - (op.kernel_ - 1) * op.dilation_ - 1,
               op.stride_) +
      1;
  PrimExpr out_hw = out_h * out_w;

  PrimExpr k = op.c_step_ * block_k + j;
  PrimExpr m = op.nhw_step_ * block_m + i;
  PrimExpr access_h = floordiv(floormod(m, out_hw), out_w) * op.stride_ +
                      floordiv(k, op.kernel_ * c_dim) * op.dilation_ -
                      op.padding_;
  PrimExpr access_w = floormod(m, out_w) * op.stride_ +
                      floormod(floordiv(k, c_dim), op.kernel_) * op.dilation_ -
                      op.padding_;

  PrimExpr in_bound = And(And(access_h >= IntImm(access_h.ty(), 0),
                              access_w >= IntImm(access_w.ty(), 0)),
                          And(access_h < h_dim, access_w < w_dim));

  PrimExpr value = TensorLoad(
      src, {floordiv(m, out_hw), access_h, access_w, floormod(k, c_dim)});
  value = if_then_else(in_bound, value, MakeSafeValueConstant(dst.dtype(), 0));

  Array<PrimExpr> dst_indices;
  dst_indices.reserve(ndim);
  for (size_t dim = 0; dim < ndim; ++dim) {
    const Range &range = dst_region->region[dim];
    if (dim == ndim - 2) {
      dst_indices.push_back(range->begin + i);
    } else if (dim == ndim - 1) {
      dst_indices.push_back(range->begin + j);
    } else {
      dst_indices.push_back(range->begin);
    }
  }

  Stmt body = TensorStore(dst, dst_indices, value);
  body = For(j, IntImm(j.ty(), 0), block_k, IntImm(j.ty(), 1),
             ForKind::kParallel, SeqStmt(body));
  body = For(i, IntImm(i.ty(), 0), block_m, IntImm(i.ty(), 1),
             ForKind::kParallel, SeqStmt(body));

  auto fused_loop =
      (ParallelLoopFuser::Fuse((body).template as_or_throw<For>()))
          .template as_or_throw<For>();
  auto par_op = ParallelOp(fused_loop, op.annotations_);
  std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                    InferLevel::kFree};
  for (auto level : levels) {
    par_op->InferLayout({lower_args.target,
                         lower_args.thread_bounds,
                         lower_args.layout_map,
                         analyzer,
                         lower_args.buffer_remap,
                         {}},
                        level);
  }
  auto loop_layout = par_op->GetLoopLayout();
  return LowerParallelLoop(
      par_op->GetRoot(), loop_layout, lower_args.thread_index, analyzer,
      lower_args.layout_map, par_op->GetPredicate(lower_args.thread_index),
      /*parallel_loop=*/true, par_op->LoopLayoutRequiresPaddingGuard(),
      lower_args.target);
}

bool RegisterDefaultIm2Col() {
  RegisterIm2ColImpl(Im2ColImpl{
      "default.Im2Col",
      MatchAnyIm2ColTarget,
      0,
      LowerIm2ColSIMT,
  });
  return true;
}

const bool default_im2col_registered = RegisterDefaultIm2Col();

} // namespace

void RegisterCopyImpl(CopyImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.infer_layout != nullptr);
  ICHECK(impl.lower != nullptr);
  CopyImplRegistry().push_back(impl);
}

void RegisterIm2ColImpl(Im2ColImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.lower != nullptr);
  Im2ColImplRegistry().push_back(impl);
}

// Constructs a Copy operator node from call arguments and annotations.
// args[0]: source region, args[1]: destination region
// annotations: Map containing common SIMT hints and backend-specific metadata.
Copy::Copy(Array<Expr> args, Map<String, Any> annotations) {
  ICHECK_EQ(args.size(), 2U);
  ObjectPtr<CopyNode> node = make_object<CopyNode>();
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->src = RegionTensor(src_access.region);
  node->dst = RegionTensor(dst_access.region);
  node->src_range = src_access.region->region;
  node->dst_range = dst_access.region->region;
  node->SetAccessRegions({src_access, dst_access});
  // Copy annotations from the Call node
  node->annotations = annotations;
  if (auto dst_block = node->annotations.Get("dst_block")) {
    if (auto value = dst_block.value().as<PrimExpr>()) {
      if (!IsConstInt(value.value(), -1))
        node->dst_block = value.value();
    } else {
      int64_t block_id = GetIntegerAnnotation(dst_block.value());
      if (block_id != -1)
        node->dst_block = IntImm(PrimType::Int(32), block_id);
    }
  }
  data_ = std::move(node);
}

// Creates a shallow clone of this CopyNode.
TileOperator CopyNode::Clone() const {
  auto op = make_object<CopyNode>(*this);
  if (par_op_.defined()) {
    op->par_op_ = (par_op_->Clone()).template as_or_throw<ParallelOp>();
  }
  return Copy(op);
}

// Creates iterator variables for dimensions with extent > 1.
Array<LayoutIterVar> CopyNode::MakeIterVars() const {
  // Choose the range set from the lowest-level memory scope between src and
  // dst. Scope levels: global < shared/shared.dyn/shared.tmem < local.fragment
  // (fragment)
  auto scope_level = [](const TensorVar &b) -> int {
    String s = b.scope().name();
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
  auto analyzer = CreateTileLangAnalyzer();

  size_t base_dim = 0, src_dim = 0;
  while (base_dim < base_ranges.size() && src_dim < src_range.size()) {
    // Skip base extents that are 1
    while (base_dim < base_ranges.size() &&
           IsOne(base_ranges[base_dim]->extent)) {
      ++base_dim;
    }
    // Skip src extents that are 1
    while (src_dim < src_range.size() && IsOne(src_range[src_dim]->extent)) {
      ++src_dim;
    }
    // Both indices now at non-1, or at end
    if (base_dim < base_ranges.size() && src_dim < src_range.size()) {
      PrimExpr base_ext = base_ranges[base_dim]->extent;
      PrimExpr src_ext = src_range[src_dim]->extent;
      // Only fail if base extent is provably smaller than src extent
      if (analyzer->CanProve(base_ext < src_ext)) {
        std::ostringstream oss;
        oss << "Selected loop range is smaller than original src range at "
               "matched non-1 dimension: "
            << "base(extent=" << base_ext
            << ", scope=" << (base_is_src ? src.scope() : dst.scope())
            << ", min=" << base_ranges[base_dim]->begin
            << ", base_dim=" << base_dim << ") < src(extent=" << src_ext
            << ", min=" << src_range[src_dim]->begin << ", src_dim=" << src_dim
            << ", scope=" << src.scope() << ") for src=" << src->name
            << ", dst=" << dst->name << "\n";
        oss << "src buffer: " << src->name << ", scope=" << src.scope() << "\n";
        oss << "dst buffer: " << dst->name << ", scope=" << dst.scope() << "\n";
        oss << "base_ranges[" << base_dim
            << "]: min=" << base_ranges[base_dim]->begin
            << ", extent=" << base_ext << "\n";
        oss << "src_ranges[" << src_dim
            << "]: min=" << src_range[src_dim]->begin << ", extent=" << src_ext
            << "\n";
        LOG(FATAL) << oss.str() << SpanHintSuffix({dst->loc, src->loc});
      }
      ++base_dim;
      ++src_dim;
    }
  }

  // Any remaining unmatched dimensions in either range must all have extent ==
  // 1
  while (base_dim < base_ranges.size()) {
    ICHECK(IsOne(base_ranges[base_dim]->extent))
        << "base_ranges has extra non-1 extent at dim " << base_dim;
    ++base_dim;
  }
  while (src_dim < src_range.size()) {
    ICHECK(IsOne(src_range[src_dim]->extent))
        << "src_range has extra non-1 extent at dim " << src_dim;
    ++src_dim;
  }

  Array<LayoutIterVar> loop_vars;
  size_t idx = 0;
  for (size_t i = 0; i < base_ranges.size(); i++) {
    if (IsOne(base_ranges[i]->extent))
      continue;
    PrimVar var(std::string{char('i' + idx)}, base_ranges[i]->extent.ty());
    idx++;
    loop_vars.push_back(
        {Range(IntImm(var.ty(), 0), base_ranges[i]->extent), var});
  }
  return loop_vars;
}

// Generates index expressions for accessing src (src_dst=0) or dst (src_dst=1)
// buffers.
Array<PrimExpr> CopyNode::MakeIndices(const Array<LayoutIterVar> &ivs,
                                      int src_dst) const {
  Array<PrimExpr> indices;
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (IsOne(ranges[i]->extent))
      indices.push_back(ranges[i]->begin);
    else {
      indices.push_back(ranges[i]->begin + ivs[idx]->var);
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
Optional<PrimExpr> CopyNode::MakePredicate(sym::AnalyzerObj *analyzer,
                                           const Array<LayoutIterVar> &ivs,
                                           Array<PrimExpr> extents,
                                           int src_dst) const {
  auto indices = MakeIndices(ivs, src_dst);
  ICHECK_EQ(extents.size(), indices.size());
  Optional<PrimExpr> predicate;
  for (size_t i = 0; i < indices.size(); ++i) {
    // Unit-extent slices still need their origin checked against the tensor.
    for (const PrimExpr &condition :
         {indices[i] >= 0, indices[i] < extents[i]}) {
      if (!analyzer->CanProve(condition, sym::ProofStrength::kSymbolicBound)) {
        predicate = predicate.has_value() ? And(predicate.value(), condition)
                                          : condition;
      }
    }
  }
  return predicate;
}

// Constructs a SIMT-style nested loop that implements the copy.
For CopyNode::MakeSIMTLoop(sym::AnalyzerObj *analyzer) const {
  auto scoped_analyzer =
      analyzer ? analyzer->Clone() : CreateTileLangAnalyzer();
  analyzer = scoped_analyzer.get();
  if (IsFP4UnpackLoad(src, dst)) {
    LOG(FATAL) << "SIMT copy from packed global float4_e2m1fn to unpacked "
               << "shared float4_e2m1_unpacked is not supported; use "
               << "T.tma_copy() for FP4 unpack loads (src=" << src->name
               << ", dst=" << dst->name << ").";
  }

  Array<LayoutIterVar> loop_vars = MakeIterVars();
  bool is_scalar = loop_vars.empty();

  for (const auto &iv : loop_vars)
    BindAnalyzer(analyzer, iv->var, iv->dom);
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

  Optional<PrimExpr> src_predicate =
      MakePredicate(analyzer, loop_vars, src.shape(), 0);
  Optional<PrimExpr> dst_predicate =
      MakePredicate(analyzer, loop_vars, dst.shape(), 1);

  PrimExpr value = TensorLoad(src, src_indices);
  if (src.dtype() != dst.dtype())
    value = Cast(dst.dtype(), value);
  if (src_predicate.has_value()) {
    PrimExpr safe_value = src_oob_safe_value.has_value()
                              ? src_oob_safe_value.value()
                              : MakeSafeValueConstant(src.dtype(), 0);
    if (safe_value.ty() != dst.dtype())
      safe_value = Cast(dst.dtype(), safe_value);
    value = if_then_else(src_predicate.value(), value,
                         analyzer->Simplify(safe_value));
  }

  Stmt body = TensorStore(dst, dst_indices, value);
  if (dst_predicate.has_value())
    body = If(dst_predicate.value(), SeqStmt(body));
  if (is_scalar) {
    return For(PrimVar("copy_scalar"), 0, 1, 1, ForKind::kDefault,
               SeqStmt(body));
  }

  for (int i = loop_vars.size() - 1; i >= 0; i--) {
    Map<String, Any> loop_annotations;
    // Dynamic layout hints remain analysis inputs of ParallelOp. Only an
    // integer coalescing policy is copied into the native loop metadata.
    if (i == 0 && annotations.count(attr::kCoalescedWidth)) {
      loop_annotations.Set(
          attr::kCoalescedWidth,
          GetIntegerAnnotation(annotations[attr::kCoalescedWidth]));
    }
    const auto &var = loop_vars[i]->var;
    body = For(var, IntImm(var.ty(), 0), loop_vars[i]->dom->extent,
               IntImm(var.ty(), 1), ForKind::kParallel, SeqStmt(body),
               loop_annotations);
  }
  return (body).template as_or_throw<For>();
}

LayoutMap CopyNode::InferLayout(const LayoutInferArgs &layout_args,
                                InferLevel level) const {
  return InferCopyLayout(*this, layout_args, level);
}

LayoutMap CopyNode::InferSIMTLayout(const LayoutInferArgs &layout_args,
                                    InferLevel level) const {
  if (!par_op_.defined()) {
    auto analyzer = CreateTileLangAnalyzer();
    par_op_ = ParallelOp(MakeSIMTLoop(analyzer.get()), annotations);
  }
  return par_op_->InferLayout(layout_args, level);
}
// Lowers the copy operation by dispatching to the selected target
// implementation.
Stmt CopyNode::Lower(const LowerArgs &lower_args,
                     sym::AnalyzerObj *analyzer) const {
  return LowerCopyForTarget(*this, lower_args, analyzer);
}

// Constructs an Im2ColOp node from call arguments.
// args: src, dst, nhw_step, c_step, kernel, stride, dilation, padding.
// The CUDA-only eviction_policy hint rides in the annotations map.
Im2ColOp::Im2ColOp(Array<Expr> args, Map<String, Any> annotations) {
  ICHECK_EQ(args.size(), 8U);
  ObjectPtr<Im2ColOpNode> node = make_object<Im2ColOpNode>();
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->srcRegion_ = src_access.region;
  node->dstRegion_ = dst_access.region;
  node->SetAccessRegions({src_access, dst_access});
  node->src_ = RegionTensor(node->srcRegion_);
  node->dst_ = RegionTensor(node->dstRegion_);
  node->nhw_step_ = args[2].as_or_throw<PrimExpr>();
  node->c_step_ = args[3].as_or_throw<PrimExpr>();
  node->kernel_ = static_cast<int>(GetIntegerArgument(args[4]));
  node->stride_ = static_cast<int>(GetIntegerArgument(args[5]));
  node->dilation_ = static_cast<int>(GetIntegerArgument(args[6]));
  node->padding_ = static_cast<int>(GetIntegerArgument(args[7]));
  if (auto val = annotations.Get("eviction_policy")) {
    node->eviction_policy_ =
        static_cast<int>(GetIntegerAnnotation(val.value()));
  }
  node->annotations_ = annotations;
  data_ = std::move(node);
}

// Creates a shallow copy of this Im2ColOpNode.
TileOperator Im2ColOpNode::Clone() const {
  auto op = make_object<Im2ColOpNode>(*this);
  return Im2ColOp(op);
}

Stmt Im2ColOpNode::Lower(const LowerArgs &lower_args,
                         sym::AnalyzerObj *analyzer) const {
  return LowerIm2ColForTarget(*this, lower_args, analyzer);
}

const Op &Copy::Get() {
  static const Op op = Op::Get("tl.tileop.copy");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  auto builder = [](Array<Expr> args, Map<String, Any> hints) {
    return Copy(std::move(args), std::move(hints));
  };
  RegisterTileOperator("tl.tileop.copy", builder,
                       OpBlockAnnotationHandlerFunc(ApplyCopyBlockAnnotations));
  RegisterTileOperator(
      "tl.tileop.async_copy",
      [](Array<Expr> args, Map<String, Any> hints) {
        hints.Set("is_async_copy", true);
        return Copy(std::move(args), std::move(hints));
      },
      OpBlockAnnotationHandlerFunc(ApplyCopyBlockAnnotations));
  RegisterTileOperator(
      "tl.tileop.tma_copy",
      [](Array<Expr> args, Map<String, Any> hints) {
        hints.Set("is_tma_copy", true);
        return Copy(std::move(args), std::move(hints));
      },
      OpBlockAnnotationHandlerFunc(ApplyCopyBlockAnnotations));
}

// Layout inference hook - returns empty map (no layout suggestions).
LayoutMap Im2ColOpNode::InferLayout(const LayoutInferArgs &layout_args,
                                    InferLevel level) const {
  return {};
}

TIR_REGISTER_TL_TILE_OP(Im2ColOp, im2col)

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterTileOperator("tl.tileop.c2d_im2col",
                       [](Array<Expr> args, Map<String, Any> hints) {
                         return Im2ColOp(std::move(args), std::move(hints));
                       });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  CopyNode::RegisterReflection();
  Im2ColOpNode::RegisterReflection();
}
} // namespace tl
} // namespace tvm
