#include "compat/analyzer.h"
/*!
 * \file tl/op/copy.cc
 * \brief Define the copy operator, backend dispatch, and shared normal-copy
 *        lowering helpers.
 */

#include "../transform/common/loop_fusion_utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"
#include "compat/visit.h"
#include "copy.h"
#include "op/prim.h"
#include "span_utils.h"
#include "support/check.h"
#include "utils.h"
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
using namespace ffi;

Stmt LowerNormalCopy(const CopyNode &op, const LowerArgs &lower_args,
                     sym::Analyzer analyzer) {
  bool is_cpu_target = lower_args.target->GetTargetDeviceType() == kDLCPU;
  auto simt_loop = op.MakeSIMTLoop(analyzer);
  auto fused_loop = ParallelLoopFuser::Fuse(simt_loop).as_or_throw<For>();
  auto par_op = ParallelOp(fused_loop);

  if (is_cpu_target || IsLocalBuffer(op.src) || IsLocalBuffer(op.dst)) {
    if (IsLocalBuffer(op.src) && !IsLocalBuffer(op.dst)) {
      // A conflict write only occurs when multiple threads write to the same
      // global address. If any dst_range dimension's min depends on the thread
      // index, each thread targets a distinct location and there is no
      // conflict. The thread index is an expression: collect the variables it
      // uses by identity (the real threadIdx.x Var on GPU; none when it is a
      // constant, e.g. 0 on CPU).
      std::unordered_set<const VarNode *> thread_index_vars;
      PostOrderVisit(lower_args.thread_index, [&](const ObjectRef &object) {
        if (const auto *variable = object.as<VarNode>()) {
          thread_index_vars.insert(variable);
        }
      });
      bool dst_depends_on_thread = false;
      for (const auto &range : op.dst_range) {
        PostOrderVisit(range->begin, [&](const ObjectRef &object) {
          if (const auto *variable = object.as<VarNode>()) {
            dst_depends_on_thread |= thread_index_vars.count(variable) != 0;
          }
        });
        if (dst_depends_on_thread) {
          break;
        }
      }
      if (!dst_depends_on_thread) {
        DLOG(WARNING) << "Copy from local buffer `" << op.src->name << "` to "
                      << op.dst.scope().name() << " buffer `" << op.dst->name
                      << "` may cause conflicted write.";
      }
    }
    if (lower_args.target->kind->name == "c") {
      return fused_loop;
    }
    return VectorizeLoop(fused_loop, analyzer, lower_args.layout_map);
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
      lower_args.target->kind->name != "c");
}

namespace {

TileOperator ApplyCopyBlockAnnotations(TileOperator tile_op,
                                       BlockAnnotations block_annotations) {
  Copy copy = tile_op.as_or_throw<Copy>();

  // Safe because this handler is invoked immediately after TLOpBuilder creates
  // a fresh CopyNode, before the node escapes ParseOperator.
  auto *node = const_cast<CopyNode *>(copy.operator->());
  ICHECK(node != nullptr);

  node->src_oob_safe_value = std::nullopt;
  auto safe_value_map_obj = block_annotations.Get(attr::kSafeValueMap);
  if (!safe_value_map_obj) {
    return copy;
  }

  auto safe_value_map =
      safe_value_map_obj.value().as_or_throw<Map<Var, PrimExpr>>();
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
      << target;
  return *best_impl;
}

LayoutMap InferCopyLayout(const CopyNode &op,
                          const LayoutInferArgs &layout_args,
                          InferLevel level) {
  return ResolveCopyImpl(layout_args.target)
      .infer_layout(op, layout_args, level);
}

Stmt LowerCopyForTarget(const CopyNode &op, const LowerArgs &lower_args,
                        sym::Analyzer analyzer) {
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
      << target;
  return *best_impl;
}

Stmt LowerIm2ColForTarget(const Im2ColOpNode &op, const LowerArgs &lower_args,
                          sym::Analyzer analyzer) {
  return ResolveIm2ColImpl(lower_args.target).lower(op, lower_args, analyzer);
}

bool MatchAnyIm2ColTarget(Target /*target*/) { return true; }

Stmt LowerIm2ColSIMT(const Im2ColOpNode &op, const LowerArgs &lower_args,
                     sym::Analyzer analyzer) {
  const TensorVar &src = op.src_;
  const TensorVar &dst = op.dst_;
  const TensorRegion &dst_region = op.dstRegion_;

  ICHECK(src.shape().size() == 4);
  ICHECK(src.dtype() == dst.dtype());

  size_t ndim = dst_region->region.size();
  ICHECK(ndim >= 2) << "im2col dstRegion must have at least 2 dims";

  PrimExpr block_m = dst_region->region[ndim - 2]->extent;
  PrimExpr block_k = dst_region->region[ndim - 1]->extent;
  PrimVar row("im2col_row", block_m.ty());
  PrimVar column("im2col_column", block_k.ty());
  tl::arith::Bind(analyzer, row,
                  Range::FromMinExtent(tl::MakeConst(row.ty(), 0), block_m));
  tl::arith::Bind(analyzer, column,
                  Range::FromMinExtent(tl::MakeConst(column.ty(), 0), block_k));

  const auto &src_ranges = op.srcRegion_->region;
  PrimExpr h_dim = src_ranges[1]->extent;
  PrimExpr w_dim = src_ranges[2]->extent;
  PrimExpr c_dim = src_ranges[3]->extent;
  PrimExpr out_h =
      FloorDiv(h_dim + tl::MakeConst(h_dim.ty(), 2) * op.padding_ -
                   (tl::MakeConst(h_dim.ty(), op.kernel_) - 1) * op.dilation_ -
                   1,
               op.stride_) +
      1;
  PrimExpr out_w =
      FloorDiv(w_dim + tl::MakeConst(w_dim.ty(), 2) * op.padding_ -
                   (tl::MakeConst(w_dim.ty(), op.kernel_) - 1) * op.dilation_ -
                   1,
               op.stride_) +
      1;
  PrimExpr out_hw = Max(out_h, 0) * Max(out_w, 0);
  PrimExpr safe_hw = Max(out_hw, 1);
  PrimExpr safe_width = Max(out_w, 1);
  PrimExpr safe_channels = Max(c_dim, 1);

  PrimExpr channel_step = op.c_step_ * block_k + column;
  PrimExpr position = op.nhw_step_ * block_m + row;
  PrimExpr access_h =
      FloorDiv(FloorMod(position, safe_hw), safe_width) * op.stride_ +
      FloorDiv(channel_step, op.kernel_ * safe_channels) * op.dilation_ -
      op.padding_;
  PrimExpr access_w =
      FloorMod(position, safe_width) * op.stride_ +
      FloorMod(FloorDiv(channel_step, safe_channels), op.kernel_) *
          op.dilation_ -
      op.padding_;

  PrimExpr in_bound = And(And(access_h >= tl::MakeConst(access_h.ty(), 0),
                              access_w >= tl::MakeConst(access_w.ty(), 0)),
                          And(access_h < h_dim, access_w < w_dim));

  in_bound = in_bound && position >= 0 &&
             position < src_ranges[0]->extent * out_hw && channel_step >= 0 &&
             channel_step < c_dim * op.kernel_ * op.kernel_;
  Array<PrimExpr> src_indices{
      src_ranges[0]->begin + FloorDiv(position, safe_hw),
      src_ranges[1]->begin + access_h, src_ranges[2]->begin + access_w,
      src_ranges[3]->begin + FloorMod(channel_step, safe_channels)};
  for (size_t axis = 0; axis < src_indices.size(); ++axis) {
    in_bound = in_bound && src_indices[axis] >= 0 &&
               src_indices[axis] < src.shape()[axis];
  }
  PrimExpr value = TensorLoad(src, src_indices);
  value = if_then_else(in_bound, value, tl::MakeConst(dst.dtype(), 0));

  Array<PrimExpr> dst_indices;
  dst_indices.reserve(ndim);
  for (size_t dim = 0; dim < ndim; ++dim) {
    const Range &range = dst_region->region[dim];
    if (dim == ndim - 2) {
      dst_indices.push_back(range->begin + row);
    } else if (dim == ndim - 1) {
      dst_indices.push_back(range->begin + column);
    } else {
      dst_indices.push_back(range->begin);
    }
  }

  Stmt body = TensorStore(dst, dst_indices, value);
  PrimExpr dst_bound = tl::MakeConst(PrimType::Bool(), true);
  for (size_t axis = 0; axis < ndim; ++axis) {
    dst_bound = dst_bound && dst_indices[axis] >= 0 &&
                dst_indices[axis] < dst.shape()[axis];
  }
  if (!tl::arith::CanProve(analyzer, dst_bound)) {
    body = If(tl::arith::Simplify(analyzer, dst_bound), SeqStmt(body));
  }
  body = For(column, tl::MakeConst(column.ty(), 0), block_k,
             tl::MakeConst(column.ty(), 1), ForKind::kParallel, SeqStmt(body));
  body = For(row, tl::MakeConst(row.ty(), 0), block_m,
             tl::MakeConst(row.ty(), 1), ForKind::kParallel, SeqStmt(body));

  auto fused_loop = ParallelLoopFuser::Fuse(body).as_or_throw<For>();
  if (lower_args.target->GetTargetDeviceType() == kDLCPU) {
    if (lower_args.target->kind->name == "c") {
      return fused_loop;
    }
    return VectorizeLoop(fused_loop, analyzer, lower_args.layout_map);
  }
  auto par_op = ParallelOp(fused_loop);
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
      lower_args.target->kind->name != "c");
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
Copy::Copy(Array<Expr> args, BlockAnnotations annotations) {
  TVM_FFI_CHECK_EQ(args.size(), 2, ValueError)
      << "Copy expects source and destination";
  ObjectPtr<CopyNode> node = make_object<CopyNode>(ffi::UnsafeInit{});
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->src = src_access.region->source.as_or_throw<TensorVar>();
  node->dst = dst_access.region->source.as_or_throw<TensorVar>();
  node->src_range = src_access.region->region;
  node->dst_range = dst_access.region->region;
  node->SetAccessRegions({src_access, dst_access});
  sym::Analyzer analyzer;
  for (const auto &ranges : {node->src_range, node->dst_range}) {
    for (const Range &range : ranges) {
      TVM_FFI_CHECK(!tl::arith::CanProve(analyzer, range->extent < 0),
                    ValueError)
          << "Copy regions must have nonnegative extents";
    }
  }
  // Copy annotations from the Call node
  node->annotations = annotations;
  if (auto dst_block = node->annotations.Get("dst_block")) {
    auto constant = GetIntAnnotation(*dst_block);
    auto block = constant ? PrimExpr(IntImm::Int32(*constant))
                          : dst_block.value().as_or_throw<PrimExpr>();
    node->annotations.Set("dst_block", block);
    if (GetConstInt(block) != std::optional<int64_t>(-1)) {
      node->dst_block = block;
    }
  }
  data_ = std::move(node);
}

// Creates a shallow clone of this CopyNode.
TileOperator CopyNode::Clone() const {
  auto op = make_object<CopyNode>(*this);
  if (par_op_.defined()) {
    op->par_op_ = par_op_->Clone().as_or_throw<ParallelOp>();
  }
  return Copy(op);
}

// Creates iterator variables for dimensions with extent > 1.
Array<LayoutAxis> CopyNode::MakeIterVars() const {
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
  sym::Analyzer analyzer;

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
      if (tl::arith::CanProve(analyzer, base_ext < src_ext)) {
        std::ostringstream oss;
        oss << "Selected loop range is smaller than original src range at "
               "matched non-1 dimension: "
            << "base(extent=" << base_ext << ", scope="
            << (base_is_src ? src.scope().name() : dst.scope().name())
            << ", min=" << base_ranges[base_dim]->begin
            << ", base_dim=" << base_dim << ") < src(extent=" << src_ext
            << ", min=" << src_range[src_dim]->begin << ", src_dim=" << src_dim
            << ", scope=" << src.scope().name() << ") for src=" << src->name
            << ", dst=" << dst->name << "\n";
        oss << "src buffer: " << src->name << ", scope=" << src.scope().name()
            << "\n";
        oss << "dst buffer: " << dst->name << ", scope=" << dst.scope().name()
            << "\n";
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

  Array<LayoutAxis> loop_vars;
  size_t idx = 0;
  for (size_t i = 0; i < base_ranges.size(); i++) {
    if (IsOne(base_ranges[i]->extent))
      continue;
    PrimVar var("copy_axis_" + std::to_string(idx),
                base_ranges[i]->extent.ty());
    idx++;
    loop_vars.push_back(
        LayoutAxis(Range::FromMinExtent(tl::MakeConst(var.ty(), 0),
                                        base_ranges[i]->extent),
                   var));
  }
  return loop_vars;
}

// Generates index expressions for accessing src (src_dst=0) or dst (src_dst=1)
// buffers.
Array<PrimExpr> CopyNode::MakeIndices(const Array<LayoutAxis> &ivs,
                                      int src_dst) const {
  Array<PrimExpr> indices;
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (IsOne(ranges[i]->extent))
      indices.push_back(ranges[i]->begin);
    else {
      TVM_FFI_CHECK(idx < ivs.size(), ValueError)
          << "Copy regions must have matching non-unit dimensions";
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
Optional<PrimExpr> CopyNode::MakePredicate(sym::Analyzer analyzer,
                                           const Array<LayoutAxis> &ivs,
                                           Array<PrimExpr> extents,
                                           int src_dst) const {
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;

  Array<PrimExpr> cond_list;
  ICHECK(extents.size() == ranges.size()) << extents << " " << ranges;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    PrimExpr index = ranges[i]->begin;
    if (!IsOne(ranges[i]->extent)) {
      PrimExpr in_region = ivs[idx]->var < ranges[i]->extent;
      if (!tl::arith::CanProve(analyzer, in_region,
                               sym::ProofStrength::kSymbolicBound)) {
        cond_list.push_back(in_region);
      }
      index = index + ivs[idx++]->var;
    }
    PrimExpr cond = index < extents[i];
    if (!tl::arith::CanProve(analyzer, cond,
                             sym::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    cond = index >= 0;
    if (!tl::arith::CanProve(analyzer, cond,
                             sym::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
  }
  if (cond_list.empty())
    return std::nullopt;
  else {
    PrimExpr cond = cond_list[0];
    for (size_t i = 1; i < cond_list.size(); i++)
      cond = And(cond, cond_list[i]);
    return cond;
  }
}

// Constructs a SIMT-style nested loop that implements the copy.
For CopyNode::MakeSIMTLoop(sym::Analyzer analyzer) const {
  if (IsFP4UnpackLoad(src, dst)) {
    LOG(FATAL) << "SIMT copy from packed global float4_e2m1fn to unpacked "
               << "shared float4_e2m1_unpacked is not supported; use "
               << "T.tma_copy() for FP4 unpack loads (src=" << src->name
               << ", dst=" << dst->name << ").";
  }

  Array<LayoutAxis> loop_vars = MakeIterVars();
  bool is_scalar = loop_vars.empty();

  for (const auto &iv : loop_vars)
    tl::arith::Bind(analyzer, iv->var, iv->dom);
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

  auto src_predicate = MakePredicate(analyzer, loop_vars, src.shape(), 0);
  auto dst_predicate = MakePredicate(analyzer, loop_vars, dst.shape(), 1);

  PrimExpr value = TensorLoad(src, src_indices);
  if (src.dtype() != dst.dtype())
    value = Cast(dst.dtype(), value);
  if (src_predicate.has_value()) {
    PrimExpr safe_value = tl::MakeConst(src.dtype(), 0);
    if (src_oob_safe_value.has_value()) {
      safe_value = src_oob_safe_value.value();
    }
    if (safe_value.ty() != dst.dtype())
      safe_value = Cast(dst.dtype(), safe_value);
    value = if_then_else(src_predicate.value(), value,
                         tl::arith::Simplify(analyzer, safe_value));
  }

  Stmt body = TensorStore(dst, dst_indices, value, dst->loc);
  if (dst_predicate.has_value())
    body = If(dst_predicate.value(), SeqStmt(body));
  if (is_scalar) {
    loop_vars.push_back(LayoutAxis(Range(0, 1), PrimVar("copy_scalar")));
  }

  for (int i = loop_vars.size() - 1; i >= 0; i--) {
    BlockAnnotations loop_annotations;

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

    PrimType type = loop_vars[i]->var.ty();
    body = For(loop_vars[i]->var, tl::MakeConst(type, 0),
               loop_vars[i]->dom->extent, tl::MakeConst(type, 1),
               ForKind::kParallel, SeqStmt(body), loop_annotations);
  }
  return body.as_or_throw<For>();
}

LayoutMap CopyNode::InferLayout(const LayoutInferArgs &layout_args,
                                InferLevel level) const {
  return InferCopyLayout(*this, layout_args, level);
}

LayoutMap CopyNode::InferSIMTLayout(const LayoutInferArgs &layout_args,
                                    InferLevel level) const {
  if (!par_op_.defined()) {
    sym::Analyzer analyzer;
    par_op_ = ParallelOp(MakeSIMTLoop(analyzer));
  }
  return par_op_->InferLayout(layout_args, level);
}
// Lowers the copy operation by dispatching to the selected target
// implementation.
Stmt CopyNode::Lower(const LowerArgs &lower_args,
                     sym::Analyzer analyzer) const {
  return LowerCopyForTarget(*this, lower_args, analyzer);
}

// Constructs an Im2ColOp node from call arguments.
// args: src, dst, nhw_step, c_step, kernel, stride, dilation, padding.
// The CUDA-only eviction_policy hint rides in the annotations map.
Im2ColOp::Im2ColOp(Array<Expr> args, BlockAnnotations annotations) {
  TVM_FFI_CHECK_EQ(args.size(), 8, ValueError)
      << "Im2Col expects eight arguments";
  ObjectPtr<Im2ColOpNode> node = make_object<Im2ColOpNode>(ffi::UnsafeInit{});
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->srcRegion_ = src_access.region;
  node->dstRegion_ = dst_access.region;
  node->SetAccessRegions({src_access, dst_access});
  node->src_ = node->srcRegion_->source.as_or_throw<TensorVar>();
  node->dst_ = node->dstRegion_->source.as_or_throw<TensorVar>();
  node->nhw_step_ = args[2].as_or_throw<PrimExpr>();
  node->c_step_ = args[3].as_or_throw<PrimExpr>();
  auto parameter = [&](size_t index, int minimum, const char *name) {
    auto value = GetConstInt(args[index]);
    TVM_FFI_CHECK(value && *value >= minimum &&
                      *value <= std::numeric_limits<int>::max(),
                  ValueError)
        << "Im2Col " << name << " must be a constant integer >= " << minimum;
    return static_cast<int>(*value);
  };
  node->kernel_ = parameter(4, 1, "kernel");
  node->stride_ = parameter(5, 1, "stride");
  node->dilation_ = parameter(6, 1, "dilation");
  node->padding_ = parameter(7, 0, "padding");
  TVM_FFI_CHECK_EQ(node->src_.shape().size(), 4, ValueError)
      << "Im2Col expects an NHWC source";
  TVM_FFI_CHECK_GE(node->dst_.shape().size(), 2, ValueError)
      << "Im2Col expects a matrix destination";
  TVM_FFI_CHECK(node->src_.dtype() == node->dst_.dtype(), ValueError)
      << "Im2Col source and destination element types must match";
  sym::Analyzer analyzer;
  for (const TensorRegion &region : {node->srcRegion_, node->dstRegion_}) {
    for (const Range &range : region->region) {
      TVM_FFI_CHECK(!tl::arith::CanProve(analyzer, range->extent < 0),
                    ValueError)
          << "Im2Col regions must have nonnegative extents";
    }
  }
  for (size_t axis = 0; axis + 2 < node->dstRegion_->region.size(); ++axis) {
    TVM_FFI_CHECK(IsOne(node->dstRegion_->region[axis]->extent), ValueError)
        << "Im2Col leading destination dimensions must have extent one";
  }
  if (auto val = annotations.Get("eviction_policy")) {
    auto policy = GetConstInt(val.value().as_or_throw<PrimExpr>());
    TVM_FFI_CHECK(policy && *policy >= 0 && *policy <= 2, ValueError)
        << "Im2Col eviction_policy must be 0, 1, or 2";
    node->eviction_policy_ = *policy;
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
                         sym::Analyzer analyzer) const {
  return LowerIm2ColForTarget(*this, lower_args, analyzer);
}

// Register the Copy operation with TVM's TIR system
// This makes the copy operation available for use in TVM programs
// - Takes 2 inputs (src_buffer, dst_buffer); options ride in annotations.
// - Marked as opaque since it has side effects (memory writes)
TIR_REGISTER_TL_TILE_OP(Copy, copy, 2)

// Layout inference hook - returns empty map (no layout suggestions).
LayoutMap Im2ColOpNode::InferLayout(const LayoutInferArgs &layout_args,
                                    InferLevel level) const {
  return {};
}

// Register the Im2Col operation with TVM's TIR system
// This operation performs im2col transformation for 2D convolutions using a
// target-specific lowering.
// - Takes 8 inputs: src_buffer, dst_buffer, nhw_step, c_step, kernel, stride,
// dilation, padding; the CUDA eviction_policy hint rides in annotations
// - Marked as opaque since it has side effects (memory writes)
TIR_REGISTER_TL_TILE_OP(Im2ColOp, im2col, 8)

// Deprecated compatibility alias. Prefer tl.tileop.im2col; this alias is
// scheduled for removal in TileLang 0.14.0.
TVM_FFI_STATIC_INIT_BLOCK() {
  CopyNode::RegisterReflection();
  Im2ColOpNode::RegisterReflection();
  RegisterTileOperator<Copy, 2>("tl.tileop.async_copy");
  RegisterTileOperator<Copy, 2>("tl.tileop.tma_copy");
  RegisterTileOperator<Im2ColOp, 8>("tl.tileop.c2d_im2col");
  for (const String &name :
       {String("copy"), String("async_copy"), String("tma_copy")}) {
    OpDef("tl.tileop." + name)
        .set_attr<OpBlockAnnotationHandlerFunc>(kTLOpBlockAnnotationHandler,
                                                ApplyCopyBlockAnnotations);
  }
  for (const String &name : {String("async_copy"), String("tma_copy")}) {
    OpDef("tl.tileop." + name)
        .set_attr<OpBuilderFunc>(
            "TLOpBuilder",
            [name](Array<Expr> args, BlockAnnotations annotations) {
              annotations.Set("is_" + name, IntImm(PrimType::Int(32), 1));
              return Copy(std::move(args), std::move(annotations));
            },
            true);
  }
}
} // namespace tl
} // namespace tvm
