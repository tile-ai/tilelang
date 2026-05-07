/*!
 * \file tl/op/multimem.cc
 * \brief Unified multimem operator implementation.
 *
 * Reuses CopyNode's ParallelOp + InferLayout + VectorizeLoop pipeline,
 * then post-processes to replace mcast buffer accesses with multimem
 * instructions.
 */

#include "multimem.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include <sstream>

#include "../transform/common/loop_fusion_utils.h"
#include "../transform/common/loop_parallel_transform_utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"
#include "multimem_rewriter.h"
#include "operator.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

// === MultimemOp Constructor ===
// args[0]: src region (tl.region call), args[1]: dst region, args[2]: mode,
// args[3]: reduce_op
MultimemOp::MultimemOp(Array<PrimExpr> args,
                       Map<String, ObjectRef> annotations) {
  ObjectPtr<MultimemOpNode> node = tvm::ffi::make_object<MultimemOpNode>();

  // Parse buffer regions using same utility as CopyNode
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto region = NormalizeToBufferRegion(args[i]);
    rgs[i] = region->region;
    bf[i] = region->buffer;
  }
  node->src = bf[0];
  node->dst = bf[1];
  node->src_range = rgs[0];
  node->dst_range = rgs[1];

  node->mode = static_cast<MultimemMode>(args[2].as<IntImm>().value()->value);
  node->reduce_op = args[3].as<IntImm>().value()->value;

  // Validate buffer scopes based on mode:
  //   ld_reduce: src=global(mcast), dst=local.fragment
  //   st:        src=local.fragment, dst=global(mcast)
  //   red:       src=local.fragment, dst=global(mcast)
  String src_scope = node->src.scope();
  String dst_scope = node->dst.scope();
  switch (node->mode) {
  case MultimemMode::kLdReduce:
    ICHECK(src_scope == "global")
        << "multimem_ld_reduce: src must be global (multicast) buffer, got '"
        << src_scope << "' for buffer '" << node->src->name << "'";
    ICHECK(dst_scope == "local.fragment")
        << "multimem_ld_reduce: dst must be local.fragment buffer, got '"
        << dst_scope << "' for buffer '" << node->dst->name << "'";
    break;
  case MultimemMode::kSt:
    ICHECK(src_scope == "local.fragment")
        << "multimem_st: src must be local.fragment buffer, got '" << src_scope
        << "' for buffer '" << node->src->name << "'";
    ICHECK(dst_scope == "global")
        << "multimem_st: dst must be global (multicast) buffer, got '"
        << dst_scope << "' for buffer '" << node->dst->name << "'";
    break;
  case MultimemMode::kRed:
    ICHECK(src_scope == "local.fragment")
        << "multimem_red: src must be local.fragment buffer, got '" << src_scope
        << "' for buffer '" << node->src->name << "'";
    ICHECK(dst_scope == "global")
        << "multimem_red: dst must be global (multicast) buffer, got '"
        << dst_scope << "' for buffer '" << node->dst->name << "'";
    break;
  case MultimemMode::kTmaStore:
  case MultimemMode::kTmaRedStore:
    ICHECK(src_scope == "shared" || src_scope == "shared.dyn")
        << "multimem_tma_store: src must be shared memory, got '" << src_scope
        << "' for buffer '" << node->src->name << "'";
    ICHECK(dst_scope == "global")
        << "multimem_tma_store: dst must be global (multicast) buffer, got '"
        << dst_scope << "' for buffer '" << node->dst->name << "'";
    break;
  }

  data_ = std::move(node);
}

// === GetCoalescedWidth ===
// 128-bit per multimem instruction => width = 128 / dtype_bits
int MultimemOpNode::GetCoalescedWidth() const {
  int bits = src->dtype.bits();
  return 128 / bits; // f32->4, f16->8, bf16->8
}

// === MakeIterVars ===
// Creates loop iteration variables from ranges (skipping dims with extent==1)
Array<IterVar> MultimemOpNode::MakeIterVars() const {
  // Use the range with the higher scope level as basis (same logic as CopyNode)
  auto scope_level = [](const Buffer &b) -> int {
    String s = b.scope();
    if (s == "local.fragment" || s == "local")
      return 2;
    if (s == "shared" || s == "shared.dyn" || s == "shared.tmem")
      return 1;
    return 0;
  };

  int src_level = scope_level(src);
  int dst_level = scope_level(dst);
  bool base_is_src = (src_level >= dst_level);
  const Array<Range> &base_ranges = base_is_src ? src_range : dst_range;

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

// === MakeIndices ===
Array<PrimExpr> MultimemOpNode::MakeIndices(const Array<IterVar> &ivs,
                                            int src_dst) const {
  Array<PrimExpr> indices;
  const Array<Range> &ranges = src_dst == 0 ? src_range : dst_range;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent))
      indices.push_back(ranges[i]->min);
    else {
      indices.push_back(ranges[i]->min + ivs[idx]->var);
      idx++;
    }
  }
  return indices;
}

// === MakePredicate ===
PrimExpr MultimemOpNode::MakePredicate(arith::Analyzer *analyzer,
                                       const Array<IterVar> &ivs,
                                       Array<PrimExpr> extents,
                                       int src_dst) const {
  const Array<Range> &ranges = src_dst == 0 ? src_range : dst_range;
  Array<PrimExpr> cond_list;
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
  PrimExpr result = cond_list[0];
  for (size_t i = 1; i < cond_list.size(); i++)
    result = And(result, cond_list[i]);
  return result;
}

// === MakeSIMTLoop ===
// Creates the element-wise parallel loop: for (i,j): dst[i,j] = src[i,j]
// with coalesced_width annotation to limit vectorization to 128 bits.
For MultimemOpNode::MakeSIMTLoop(arith::Analyzer *analyzer) const {
  Array<IterVar> loop_vars = MakeIterVars();
  bool is_scalar = loop_vars.empty();

  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);

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

  int coalesced_width = GetCoalescedWidth();
  for (int i = loop_vars.size() - 1; i >= 0; i--) {
    Map<String, ObjectRef> loop_annotations;
    loop_annotations.Set(attr::kCoalescedWidth,
                         IntImm(DataType::Int(32), coalesced_width));
    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body, std::nullopt, loop_annotations);
  }
  return Downcast<For>(body);
}

// === InferLayout ===
// Delegates to ParallelOp for layout inference (same as
// CopyNode::LowerNormalCopy)
LayoutMap MultimemOpNode::InferLayout(const LayoutInferArgs &T,
                                      InferLevel level) const {
  // Multimem ops always go through the ParallelOp path during Lower.
  // No standalone layout inference needed.
  return {};
}

// === Lower ===
// The main lowering path: MakeSIMTLoop -> ParallelOp pipeline ->
// MultimemRewriter
Stmt MultimemOpNode::Lower(const LowerArgs &T,
                           arith::Analyzer *analyzer) const {
  if (mode == MultimemMode::kTmaStore || mode == MultimemMode::kTmaRedStore) {
    return LowerBulkCopy(T, analyzer);
  }

  // Step 1-2: Create SIMT loop and fuse/transform
  auto simt_loop = MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
  auto transformed_loop =
      Downcast<For>(ParallelLoopTransformer::Substitute(fused_loop));

  // Step 3: Create ParallelOp and run InferLayout at multiple levels
  auto par_op = ParallelOp(transformed_loop);

  std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                    InferLevel::kFree};
  for (auto level : levels) {
    par_op->InferLayout({T.target,
                         T.thread_bounds,
                         T.layout_map,
                         analyzer,
                         false,
                         T.buffer_remap,
                         {},
                         false},
                        level);
  }

  // Step 4: Lower the parallel loop (PartitionLoop + VectorizeLoop)
  auto loop_layout = par_op->GetLoopLayout();
  Stmt result =
      LowerParallelLoop(par_op->GetRoot(), loop_layout, T.thread_var, analyzer,
                        {}, par_op->GetPredicate(T.thread_var));

  // Step 5: Post-process — replace mcast buffer accesses with multimem
  // call_extern
  Buffer mcast_buf = (mode == MultimemMode::kLdReduce) ? src : dst;
  // Remap the mcast buffer if needed
  if (T.buffer_remap.count(mcast_buf)) {
    mcast_buf = T.buffer_remap[mcast_buf];
  }
  result = MultimemRewriter(mcast_buf, mode, reduce_op).Rewrite(result);
  return result;
}

// === LowerBulkCopy ===
// CTA-collective bulk async store from shared to multicast global.
// Reuses the 1D address computation pattern from CopyNode::LowerBulkCopy1D,
// but emits multimem.cp.async.bulk or multimem.cp.reduce.async.bulk PTX.
Stmt MultimemOpNode::LowerBulkCopy(const LowerArgs &T,
                                   arith::Analyzer *analyzer) const {
  bool is_reduce = (mode == MultimemMode::kTmaRedStore);
  // Both modes: src=shared, dst=mcast_global
  auto &shared_tensor = src;
  auto &global_tensor = dst;
  auto &shared_range = src_range;
  auto &global_range = dst_range;

  // Compute total elements
  PrimExpr shared_elements = 1;
  for (size_t i = 0; i < shared_range.size(); i++) {
    shared_elements *= shared_range[i]->extent;
  }
  PrimExpr elements = analyzer->Simplify(shared_elements);
  PrimExpr size_bytes = elements * shared_tensor->dtype.bytes();

  // 16-byte alignment check (at compile time if constant)
  if (auto *imm = size_bytes.as<IntImmNode>()) {
    ICHECK(imm->value % 16 == 0)
        << "multimem_tma_store: transfer size must be 16-byte aligned, got "
        << imm->value;
  }

  // Compute flat shared offset
  std::vector<PrimExpr> shared_strides;
  PrimExpr sh_stride = 1;
  for (int i = static_cast<int>(shared_tensor->shape.size()) - 1; i >= 0; --i) {
    shared_strides.insert(shared_strides.begin(), sh_stride);
    sh_stride *= shared_tensor->shape[i];
  }
  PrimExpr shared_offset = 0;
  for (size_t i = 0; i < shared_range.size(); i++) {
    shared_offset += shared_range[i]->min * shared_strides[i];
  }

  // Compute flat global offset
  std::vector<PrimExpr> global_strides;
  PrimExpr gl_stride = 1;
  for (int i = static_cast<int>(global_tensor->shape.size()) - 1; i >= 0; --i) {
    global_strides.insert(global_strides.begin(), gl_stride);
    gl_stride *= global_tensor->shape[i];
  }
  PrimExpr global_offset = 0;
  for (size_t i = 0; i < global_range.size(); i++) {
    global_offset += global_range[i]->min * global_strides[i];
  }

  // Build address_of(BufferLoad(buffer, {flat_offset}))
  auto make_addr = [](const Buffer &buf, PrimExpr flat_idx) -> PrimExpr {
    return Call(DataType::Handle(), builtin::address_of(),
                {BufferLoad(buf, {flat_idx})});
  };
  PrimExpr smem_addr = make_addr(shared_tensor, shared_offset);
  PrimExpr mcast_addr = make_addr(global_tensor, global_offset);

  // Build function name based on mode and dtype
  std::string func_name;
  if (is_reduce) {
    func_name = "tl::multimem::cp_reduce_async_bulk_";
    switch (reduce_op) {
    case 0:
      func_name += "add_";
      break;
    case 1:
      func_name += "min_";
      break;
    case 2:
      func_name += "max_";
      break;
    default:
      LOG(FATAL) << "Invalid reduce_op: " << reduce_op;
    }
    func_name += shared_tensor->dtype.is_float16()    ? "f16"
                 : shared_tensor->dtype.is_bfloat16() ? "bf16"
                                                      : "f32";
  } else {
    func_name = "tl::multimem::cp_async_bulk";
  }

  Array<PrimExpr> extern_args;
  extern_args.push_back(StringImm(func_name));
  extern_args.push_back(mcast_addr);
  extern_args.push_back(smem_addr);
  extern_args.push_back(size_bytes);

  Stmt bulk_copy =
      Evaluate(Call(DataType::Handle(), builtin::call_extern(), extern_args));

  // Gate with tid == 0 (single thread per CTA emits the PTX)
  bulk_copy = IfThenElse(EQ(T.thread_var, T.thread_bounds->min), bulk_copy);
  return bulk_copy;
}

// === Clone ===
TileOperator MultimemOpNode::Clone() const {
  auto node = tvm::ffi::make_object<MultimemOpNode>(*this);
  return MultimemOp(node);
}

// === Registration ===
TIR_REGISTER_TL_TILE_OP(MultimemOp, multimem)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm
