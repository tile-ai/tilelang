/*!
 * \file tl/op/copy.cc
 * \brief Define the copy operator, backend dispatch, and shared normal-copy
 *        lowering helpers.
 */

#include "copy.h"
#include "../transform/common/loop_fusion_utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"
#include "support/check.h"
#include "utils.h"
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>

#include "builtin.h"
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include <limits>
#include <sstream>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

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
        if (tirx::UsesVar(range->min, [&](const VarNode *v) {
              return v == T.thread_var.get();
            })) {
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
                           par_op->GetPredicate(T.thread_var),
                           /*parallel_loop=*/true, /*should_vectorize=*/true,
                           par_op->LoopLayoutRequiresPaddingGuard());
}

namespace {

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

LayoutMap InferCopyLayout(const CopyNode &op, const LayoutInferArgs &T,
                          InferLevel level) {
  return ResolveCopyImpl(T.target).infer_layout(op, T, level);
}

Stmt LowerCopyForTarget(const CopyNode &op, const LowerArgs &T,
                        arith::Analyzer *analyzer) {
  return ResolveCopyImpl(T.target).lower(op, T, analyzer);
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

Stmt LowerIm2ColForTarget(const Im2ColOpNode &op, const LowerArgs &T,
                          arith::Analyzer *analyzer) {
  return ResolveIm2ColImpl(T.target).lower(op, T, analyzer);
}

bool MatchAnyIm2ColTarget(Target /*target*/) { return true; }

Stmt LowerIm2ColSIMT(const Im2ColOpNode &op, const LowerArgs &T,
                     arith::Analyzer *analyzer) {
  const Buffer &src = op.src_;
  const Buffer &dst = op.dst_;
  const BufferRegion &dst_region = op.dstRegion_;

  ICHECK(src->shape.size() == 4);
  ICHECK(src->dtype == dst->dtype);

  size_t ndim = dst_region->region.size();
  ICHECK(ndim >= 2) << "im2col dstRegion must have at least 2 dims";

  PrimExpr block_m = dst_region->region[ndim - 2]->extent;
  PrimExpr block_k = dst_region->region[ndim - 1]->extent;
  Var i("i", block_m.dtype());
  Var j("j", block_k.dtype());
  analyzer->Bind(i, Range::FromMinExtent(make_zero(i.dtype()), block_m));
  analyzer->Bind(j, Range::FromMinExtent(make_zero(j.dtype()), block_k));

  PrimExpr h_dim = src->shape[1];
  PrimExpr w_dim = src->shape[2];
  PrimExpr c_dim = src->shape[3];
  PrimExpr out_h =
      FloorDiv(h_dim + 2 * op.padding_ - (op.kernel_ - 1) * op.dilation_ - 1,
               op.stride_) +
      1;
  PrimExpr out_w =
      FloorDiv(w_dim + 2 * op.padding_ - (op.kernel_ - 1) * op.dilation_ - 1,
               op.stride_) +
      1;
  PrimExpr out_hw = out_h * out_w;

  PrimExpr k = op.c_step_ * block_k + j;
  PrimExpr m = op.nhw_step_ * block_m + i;
  PrimExpr access_h = FloorDiv(FloorMod(m, out_hw), out_w) * op.stride_ +
                      FloorDiv(k, op.kernel_ * c_dim) * op.dilation_ -
                      op.padding_;
  PrimExpr access_w = FloorMod(m, out_w) * op.stride_ +
                      FloorMod(FloorDiv(k, c_dim), op.kernel_) * op.dilation_ -
                      op.padding_;

  PrimExpr in_bound = And(And(access_h >= make_zero(access_h.dtype()),
                              access_w >= make_zero(access_w.dtype())),
                          And(access_h < h_dim, access_w < w_dim));

  PrimExpr value = BufferLoad(
      src, {FloorDiv(m, out_hw), access_h, access_w, FloorMod(k, c_dim)});
  value = if_then_else(in_bound, value, make_zero(dst->dtype));

  Array<PrimExpr> dst_indices;
  dst_indices.reserve(ndim);
  for (size_t dim = 0; dim < ndim; ++dim) {
    const Range &range = dst_region->region[dim];
    if (dim == ndim - 2) {
      dst_indices.push_back(range->min + i);
    } else if (dim == ndim - 1) {
      dst_indices.push_back(range->min + j);
    } else {
      dst_indices.push_back(range->min);
    }
  }

  Stmt body = BufferStore(dst, value, dst_indices);
  body = For(j, make_zero(j.dtype()), block_k, ForKind::kParallel, body);
  body = For(i, make_zero(i.dtype()), block_m, ForKind::kParallel, body);

  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(Downcast<For>(body)));
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
  return LowerParallelLoop(par_op->GetRoot(), loop_layout, T.thread_var,
                           analyzer, T.layout_map,
                           par_op->GetPredicate(T.thread_var),
                           /*parallel_loop=*/true, /*should_vectorize=*/true,
                           par_op->LoopLayoutRequiresPaddingGuard());
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
Copy::Copy(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<CopyNode> node = make_object<CopyNode>();
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->src = src_access.region->buffer;
  node->dst = dst_access.region->buffer;
  node->src_range = src_access.region->region;
  node->dst_range = dst_access.region->region;
  node->SetAccessRegions({src_access, dst_access});
  // Copy annotations from the Call node
  node->annotations = annotations;
  if (auto dst_block = node->annotations.Get("dst_block")) {
    if (auto int_imm = dst_block->as<IntImmNode>()) {
      if (int_imm->value != -1) {
        node->dst_block = Integer(int_imm->value);
      }
    } else {
      node->dst_block = Downcast<PrimExpr>(dst_block.value());
    }
  }
  data_ = std::move(node);
}

// Creates a shallow clone of this CopyNode.
TileOperator CopyNode::Clone() const {
  auto op = make_object<CopyNode>(*this);
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
  if (IsFP4UnpackLoad(src, dst)) {
    LOG(FATAL) << "SIMT copy from packed global float4_e2m1fn to unpacked "
               << "shared float4_e2m1_unpacked is not supported; use "
               << "T.tma_copy() for FP4 unpack loads (src=" << src->name
               << ", dst=" << dst->name << ").";
  }

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

LayoutMap CopyNode::InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const {
  return InferCopyLayout(*this, T, level);
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

// Constructs an Im2ColOp node from call arguments.
// args: src, dst, nhw_step, c_step, kernel, stride, dilation, padding,
// eviction_policy
Im2ColOp::Im2ColOp(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<Im2ColOpNode> node = make_object<Im2ColOpNode>();
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

// Creates a shallow copy of this Im2ColOpNode.
TileOperator Im2ColOpNode::Clone() const {
  auto op = make_object<Im2ColOpNode>(*this);
  return Im2ColOp(op);
}

Stmt Im2ColOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  return LowerIm2ColForTarget(*this, T, analyzer);
}

// Register the Copy operation with TVM's TIR system
// This makes the copy operation available for use in TVM programs
// - Takes 5 inputs: src_buffer, dst_buffer, and annotation-driven options.
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
LayoutMap Im2ColOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  return {};
}

// Register the Im2Col operation with TVM's TIR system
// This operation performs im2col transformation for 2D convolutions using a
// target-specific lowering.
// - Takes 9 inputs: src_buffer, dst_buffer, nhw_step, c_step, kernel, stride,
// dilation, padding, eviction_policy
// - Marked as opaque since it has side effects (memory writes)
TIR_REGISTER_TL_TILE_OP(Im2ColOp, im2col)
    .set_num_inputs(9)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// Deprecated compatibility alias. Prefer tl.tileop.im2col; this alias is
// scheduled for removal in TileLang 0.14.0.
TVM_REGISTER_OP("tl.tileop.c2d_im2col")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "c2d_im2col")
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               return Im2ColOp(args, annotations);
                             })
    .set_num_inputs(9)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  CopyNode::RegisterReflection();
  Im2ColOpNode::RegisterReflection();
}
} // namespace tl
} // namespace tvm
