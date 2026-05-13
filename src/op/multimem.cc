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

namespace {

std::string MultimemDTypeToTag(DataType dtype) {
  if (dtype.is_float() && dtype.bits() == 32)
    return "float";
  if (dtype.is_float16())
    return "half_t";
  if (dtype.is_bfloat16())
    return "bfloat16_t";
  LOG(FATAL) << "Unsupported dtype for multimem: " << dtype;
  return "";
}

std::string MultimemReduceOpToTag(int reduce_op) {
  switch (reduce_op) {
  case 0:
    return "tl::multimem::ReduceOp::ADD";
  case 1:
    return "tl::multimem::ReduceOp::MIN";
  case 2:
    return "tl::multimem::ReduceOp::MAX";
  default:
    LOG(FATAL) << "Invalid reduce_op: " << reduce_op;
    return "";
  }
}

std::string MultimemFuncName(MultimemMode mode, int reduce_op, int lanes,
                             DataType dtype) {
  std::stringstream ss;
  switch (mode) {
  case MultimemMode::kLdReduce:
    ss << "tl::multimem::LdReduceV" << lanes << "<"
       << MultimemReduceOpToTag(reduce_op) << ", "
       << MultimemDTypeToTag(dtype) << ">::run";
    break;
  case MultimemMode::kSt:
    ss << "tl::multimem::StV" << lanes << "<" << MultimemDTypeToTag(dtype)
       << ">::run";
    break;
  case MultimemMode::kRed:
    ss << "tl::multimem::RedV" << lanes << "<"
       << MultimemReduceOpToTag(reduce_op) << ", "
       << MultimemDTypeToTag(dtype) << ">::run";
    break;
  default:
    LOG(FATAL) << "Unsupported multimem mode for vector instruction: "
               << static_cast<int>(mode);
  }
  return ss.str();
}

PrimExpr MakeAddressOf(const Buffer &buffer, const Array<PrimExpr> &indices) {
  return Call(DataType::Handle(), builtin::address_of(),
              {BufferLoad(buffer, indices)});
}

PrimExpr ProductExtent(const Array<Range> &ranges, size_t begin,
                       size_t end) {
  PrimExpr result = 1;
  for (size_t i = begin; i < end; ++i) {
    result = result * ranges[i]->extent;
  }
  return result;
}

PrimExpr FlattenIndices(const Array<PrimExpr> &indices,
                        const Array<PrimExpr> &shape,
                        arith::Analyzer *analyzer) {
  ICHECK_EQ(indices.size(), shape.size());
  PrimExpr flat = 0;
  PrimExpr stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    flat = flat + indices[i] * stride;
    stride = stride * shape[i];
  }
  return analyzer->Simplify(flat);
}

Array<PrimExpr> UnflattenIndex(PrimExpr flat, const Array<PrimExpr> &shape,
                               arith::Analyzer *analyzer) {
  Array<PrimExpr> indices;
  PrimExpr remaining = flat;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    indices.insert(indices.begin(),
                   analyzer->Simplify(floormod(remaining, shape[i])));
    remaining = analyzer->Simplify(floordiv(remaining, shape[i]));
  }
  return indices;
}

} // namespace

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
  // Match the native multimem instruction width used by the element type.
  // fp32 has vector forms; fp16/bf16 use packed x2 instructions.
  return bits == 32 ? 4 : 2; // f32->V4, f16/bf16->V2 packed x2 ops
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
  if (mode == MultimemMode::kTmaStore || mode == MultimemMode::kTmaRedStore) {
    return {};
  }
  if (IsPacked16BitMultimem()) {
    if (mode == MultimemMode::kLdReduce) {
      return {};
    }
    Buffer local_buf = (mode == MultimemMode::kLdReduce) ? dst : src;
    Buffer remapped_local = local_buf;
    if (T.buffer_remap.count(remapped_local)) {
      remapped_local = T.buffer_remap[remapped_local];
    }
    if (T.layout_map.count(remapped_local)) {
      return {};
    }

    PrimExpr numel = 1;
    const Array<Range> &local_range =
        (mode == MultimemMode::kLdReduce) ? dst_range : src_range;
    for (const auto &range : local_range) {
      numel = numel * range->extent;
    }
    ICHECK(T.analyzer != nullptr);
    PrimExpr thread_extent = T.thread_bounds->extent;
    PrimExpr pair_width = IntImm(DataType::Int(32), 2);
    PrimExpr replicate_extent = T.analyzer->Simplify(
        floordiv(numel + thread_extent * pair_width - 1,
                 thread_extent * pair_width) *
        pair_width);
    Array<PrimExpr> logical_indices;
    for (size_t i = 0; i < remapped_local->shape.size(); ++i) {
      logical_indices.push_back(InputPlaceholder(i));
    }
    PrimExpr logical =
        FlattenIndices(logical_indices, remapped_local->shape, T.analyzer);
    PrimExpr pair_id = floordiv(logical, pair_width);
    PrimExpr local_offset = T.analyzer->Simplify(
        FloorMod(logical, pair_width) +
        pair_width * floordiv(pair_id, thread_extent));
    PrimExpr thread = T.analyzer->Simplify(FloorMod(pair_id, thread_extent));

    Fragment fragment =
        Fragment(remapped_local->shape, {local_offset}, thread,
                 replicate_extent, std::nullopt)
            ->BindThreadRange(T.thread_bounds);
    LayoutMap result;
    result.Set(remapped_local, fragment);
    return result;
  }
  arith::Analyzer analyzer;
  auto par_op = ParallelOp(MakeTransformedSIMTLoop(&analyzer));
  return par_op->InferLayout(T, level);
}

// === Lower ===
// The main lowering path: MakeSIMTLoop -> ParallelOp pipeline ->
// MultimemRewriter
Stmt MultimemOpNode::Lower(const LowerArgs &T,
                           arith::Analyzer *analyzer) const {
  if (mode == MultimemMode::kTmaStore || mode == MultimemMode::kTmaRedStore) {
    return LowerBulkCopy(T, analyzer);
  }
  if (IsPacked16BitMultimem()) {
    return LowerPacked16Bit(T, analyzer);
  }

  // Step 1-2: Create SIMT loop and fuse/transform
  auto transformed_loop = MakeTransformedSIMTLoop(analyzer);

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

For MultimemOpNode::MakeTransformedSIMTLoop(
    arith::Analyzer *analyzer) const {
  auto simt_loop = MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
  return Downcast<For>(ParallelLoopTransformer::Substitute(fused_loop));
}

bool MultimemOpNode::IsPacked16BitMultimem() const {
  if (mode == MultimemMode::kTmaStore || mode == MultimemMode::kTmaRedStore) {
    return false;
  }
  return (src->dtype.is_float16() || src->dtype.is_bfloat16()) &&
         src->dtype == dst->dtype;
}

Stmt MultimemOpNode::LowerPacked16Bit(const LowerArgs &T,
                                      arith::Analyzer *analyzer) const {
  Buffer local_buf = (mode == MultimemMode::kLdReduce) ? dst : src;
  Buffer mcast_buf = (mode == MultimemMode::kLdReduce) ? src : dst;
  Array<Range> local_range =
      (mode == MultimemMode::kLdReduce) ? dst_range : src_range;
  Array<Range> mcast_range =
      (mode == MultimemMode::kLdReduce) ? src_range : dst_range;

  ICHECK_EQ(local_range.size(), mcast_range.size())
      << "multimem packed x2 lowering expects matching region rank";
  ICHECK(!local_range.empty())
      << "multimem packed x2 lowering expects a non-scalar region";
  const size_t ndim = local_range.size();
  ICHECK_EQ(local_buf->shape.size(), ndim)
      << "multimem packed x2 lowering expects the local region rank to match "
         "the buffer rank";
  ICHECK_EQ(mcast_buf->shape.size(), ndim)
      << "multimem packed x2 lowering expects the multicast region rank to "
         "match the buffer rank";

  for (size_t i = 0; i < ndim; ++i) {
    ICHECK(analyzer->CanProve(local_range[i]->extent == mcast_range[i]->extent,
                              arith::ProofStrength::kSymbolicBound))
        << "multimem packed x2 lowering expects matching region extents";
  }

  const PrimExpr last_extent = analyzer->Simplify(local_range[ndim - 1]->extent);
  if (auto *imm = last_extent.as<IntImmNode>()) {
    ICHECK_EQ(imm->value % 2, 0)
        << "multimem packed x2 lowering requires the last dimension extent to "
           "be divisible by 2";
  }

  Buffer remapped_local = local_buf;
  Buffer remapped_mcast = mcast_buf;
  if (T.buffer_remap.count(remapped_local)) {
    remapped_local = T.buffer_remap[remapped_local];
  }
  if (T.buffer_remap.count(remapped_mcast)) {
    remapped_mcast = T.buffer_remap[remapped_mcast];
  }

  PrimExpr numel = ProductExtent(local_range, 0, ndim);
  PrimExpr leading_elements = ProductExtent(local_range, 0, ndim - 1);
  PrimExpr pairs_per_row = analyzer->Simplify(floordiv(last_extent, 2));
  PrimExpr total_pairs = analyzer->Simplify(leading_elements * pairs_per_row);
  PrimExpr thread_extent = T.thread_bounds->extent;
  PrimExpr thread_offset = T.thread_var - T.thread_bounds->min;
  PrimExpr trip_count =
      analyzer->Simplify(floordiv(total_pairs + thread_extent - 1,
                                  thread_extent));

  Var loop_var("multimem_pair_iter", DataType::Int(32));
  PrimExpr pair_id =
      analyzer->Simplify(loop_var * thread_extent + thread_offset);
  PrimExpr linear_leading = analyzer->Simplify(floordiv(pair_id, pairs_per_row));
  PrimExpr last_pair =
      analyzer->Simplify(pair_id - linear_leading * pairs_per_row);
  PrimExpr local_offset = analyzer->Simplify(loop_var * 2);

  auto make_indices = [&](const Array<Range> &ranges,
                          const Buffer &buffer) -> Array<PrimExpr> {
    Array<PrimExpr> indices;
    PrimExpr remaining = linear_leading;
    for (size_t i = 0; i + 1 < ndim; ++i) {
      PrimExpr stride = ProductExtent(ranges, i + 1, ndim - 1);
      PrimExpr coord = analyzer->Simplify(floordiv(remaining, stride));
      remaining = analyzer->Simplify(remaining - coord * stride);
      indices.push_back(analyzer->Simplify(ranges[i]->min + coord));
    }
    indices.push_back(
        analyzer->Simplify(ranges[ndim - 1]->min + last_pair * 2));
    ICHECK_EQ(indices.size(), buffer->shape.size());
    return indices;
  };

  Array<PrimExpr> local_indices =
      UnflattenIndex(local_offset, remapped_local->shape, analyzer);
  Array<PrimExpr> mcast_indices = make_indices(mcast_range, remapped_mcast);

  Array<PrimExpr> args;
  args.push_back(
      StringImm(MultimemFuncName(mode, reduce_op, 2, local_buf->dtype)));
  if (mode == MultimemMode::kLdReduce) {
    args.push_back(MakeAddressOf(remapped_local, local_indices));
    args.push_back(MakeAddressOf(remapped_mcast, mcast_indices));
  } else {
    args.push_back(MakeAddressOf(remapped_mcast, mcast_indices));
    args.push_back(MakeAddressOf(remapped_local, local_indices));
  }
  Stmt body =
      Evaluate(Call(DataType::Handle(), builtin::call_extern(), args));
  body = IfThenElse(pair_id < total_pairs, body);
  return For(loop_var, 0, trip_count, ForKind::kSerial, body);
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
