/*!
 * \file tl/cuda/op/copy.cc
 * \brief CUDA implementation for tl.copy lowering.
 */

#include "op/copy.h"
#include "support/check.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>

#include "cuda/op/copy.h"
#include "cuda/target_utils.h"
#include "cuda/transform/ptx_async_copy_injector.h"
#include "layout/cute_layout.h"
#include "layout/tcgen05_layout.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <cctype>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

PrimExpr MakeTmaLeaderCondition(PrimExpr thread_extent) {
  return Call(DataType::Bool(), tl_shuffle_elect(), {std::move(thread_extent)});
}

int TMAPayloadElementBits(DataType dtype) {
  // IR elements are 8-bit unpacked SMEM slots, but TMA/mbarrier transactions
  // count the packed FP4 payload loaded from global memory.
  if (dtype.is_float4_e2m1_unpacked()) {
    return 4;
  }
  return dtype.bits();
}

PrimExpr TMABytesFromElements(PrimExpr elements, int bits) {
  PrimExpr elements_i64 = cast(DataType::Int(64), elements);
  if (bits % 8 == 0) {
    return elements_i64 * IntImm(DataType::Int(64), bits / 8);
  }
  return FloorDiv(elements_i64 * IntImm(DataType::Int(64), bits) +
                      IntImm(DataType::Int(64), 7),
                  IntImm(DataType::Int(64), 8));
}

int64_t TMABytesFromElements(int64_t elements, int bits) {
  return (elements * bits + 7) / 8;
}

PrimExpr TMABytesFromElements(PrimExpr elements, DataType dtype) {
  return TMABytesFromElements(elements, dtype.bits());
}

int64_t TMABytesFromElements(int64_t elements, DataType dtype) {
  return TMABytesFromElements(elements, dtype.bits());
}

PrimExpr TMAGlobalBytesFromElements(PrimExpr elements, DataType dtype) {
  return TMABytesFromElements(elements, TMAPayloadElementBits(dtype));
}

int64_t TMAGlobalBytesFromElements(int64_t elements, DataType dtype) {
  return TMABytesFromElements(elements, TMAPayloadElementBits(dtype));
}

PrimExpr TMATransactionBytesFromElements(PrimExpr elements, DataType dtype) {
  return TMABytesFromElements(elements, TMAPayloadElementBits(dtype));
}

int64_t TMATransactionBytesFromElements(int64_t elements, DataType dtype) {
  return TMABytesFromElements(elements, TMAPayloadElementBits(dtype));
}

int64_t TMAElementsForBytes(int64_t bytes, DataType dtype) {
  ICHECK_EQ((bytes * 8) % dtype.bits(), 0)
      << bytes << " bytes cannot be represented as whole elements of " << dtype;
  return bytes * 8 / dtype.bits();
}

PrimExpr GetCopyMbarPhaseExpr(const Map<String, ObjectRef> &annotations,
                              const LowerArgs &lower_args) {
  PrimExpr phase = lower_args.mbar_phase_expr;
  if (auto explicit_phase = GetAnnotatedMbarPhaseExpr(annotations)) {
    phase = explicit_phase.value();
  }
  return phase;
}

std::string SanitizeIdentifierPart(const std::string &name) {
  std::string sanitized;
  sanitized.reserve(name.size());
  for (unsigned char ch : name) {
    sanitized.push_back(std::isalnum(ch) || ch == '_' ? static_cast<char>(ch)
                                                      : '_');
  }
  if (sanitized.empty()) {
    sanitized = "buffer";
  }
  if (std::isdigit(static_cast<unsigned char>(sanitized.front()))) {
    sanitized.insert(sanitized.begin(), '_');
  }
  return sanitized;
}

std::string MakeCopyMBarrierName(const Buffer &src, const Buffer &dst) {
  return SanitizeIdentifierPart(src->name) + "_to_" +
         SanitizeIdentifierPart(dst->name) + "_mbarrier";
}

bool GetBoolAnnotation(const CopyNode &op, const char *key) {
  if (auto val = op.annotations.Get(key)) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value != 0;
    }
  }
  return false;
}

bool GetDisableTMA(const CopyNode &op) {
  return GetBoolAnnotation(op, "disable_tma");
}

bool GetIsTmaCopy(const CopyNode &op) {
  return GetBoolAnnotation(op, "is_tma_copy");
}

int TensorMapDataTypeForTMA(DataType global_dtype, DataType shared_dtype) {
  // 16U4_ALIGN16B: f8f6f4 / mxf8f6f4 unpacked FP4 SMEM
  // (float_e2m1_unpacksmem_t). 16U4_ALIGN8B: packed FP4 for mxf4 / mxf4nvf4
  // (float_e2m1_t).
  constexpr int kTensorMapDataType16U4Align16B = 14;
  if (shared_dtype.is_float4_e2m1_unpacked()) {
    ICHECK(global_dtype.is_float4_e2m1fn())
        << "FP4 packed global tensor required for unpacked shared TMA copy";
    return kTensorMapDataType16U4Align16B;
  }
  return to_CUtensorMapDataType(global_dtype);
}

int GetEvictionPolicy(const CopyNode &op) {
  if (auto val = op.annotations.Get("eviction_policy")) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value;
    }
  }
  return 0; // default: evict_normal
}

int64_t GetClusterMask(const CopyNode &op) {
  if (auto val = op.annotations.Get("cluster_mask")) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value;
    }
  }
  return 0;
}

int MinRankInClusterMask(int64_t cluster_mask) {
  ICHECK_GT(cluster_mask, 0);
  uint64_t mask = static_cast<uint64_t>(cluster_mask);
  int rank = 0;
  while ((mask & 1U) == 0U) {
    mask >>= 1;
    ++rank;
  }
  return rank;
}

Optional<PrimExpr> GetBarrier(const CopyNode &op) {
  if (auto val = op.annotations.Get("barrier")) {
    if (val->as<tirx::BufferLoadNode>()) {
      return Downcast<PrimExpr>(val.value());
    }
  }
  return Optional<PrimExpr>();
}

bool GetIsAsyncCopy(const CopyNode &op) {
  if (GetBoolAnnotation(op, "is_async_copy")) {
    return true;
  }
  // Backward-compatibility with historical annotation key.
  return GetBoolAnnotation(op, "force_cp_async");
}

bool GetNoImplicitAsyncCommitWait(const CopyNode &op) {
  return GetBoolAnnotation(op, attr::kAsyncCopyNoImplicitCommitWait);
}

PrimExpr GetLeaderScopeThreads(const CopyNode &op,
                               const LowerArgs &lower_args) {
  if (auto val = op.annotations.Get("leader_scope_threads")) {
    auto int_val = val->as<IntImmNode>();
    ICHECK(int_val) << "T.tma_copy leader_scope_threads annotation must be an "
                       "integer constant.";
    ICHECK_GT(int_val->value, 0)
        << "T.tma_copy leader_scope_threads must be positive.";
    ICHECK_EQ(int_val->value % 32, 0)
        << "T.tma_copy leader_scope_threads must be a multiple of warp size "
           "(32).";
    return IntImm(DataType::Int(32), int_val->value);
  }
  return lower_args.thread_bounds->extent;
}

bool IsContiguousRegion(const Buffer &buf, const Array<Range> &ranges,
                        arith::Analyzer *analyzer) {
  ICHECK_EQ(buf->shape.size(), ranges.size())
      << "IsContiguousRegion: buffer/range rank mismatch for " << buf->name;

  int n = static_cast<int>(ranges.size());
  int pivot = -1;
  for (int i = 0; i < n; ++i) {
    if (!analyzer->CanProveEqual(ranges[i]->extent, 1)) {
      pivot = i;
      break;
    }
  }
  if (pivot == -1) {
    return true;
  }

  for (int i = 0; i < pivot; ++i) {
    ICHECK(analyzer->CanProveEqual(ranges[i]->extent, 1))
        << "IsContiguousRegion: dim " << i << " precedes pivot " << pivot
        << " but has non-unit extent " << ranges[i]->extent << " for buffer "
        << buf->name;
  }

  for (int i = pivot + 1; i < n; ++i) {
    if (!analyzer->CanProveEqual(ranges[i]->min, 0) ||
        !analyzer->CanProveEqual(ranges[i]->extent, buf->shape[i])) {
      return false;
    }
  }
  return true;
}

std::pair<Array<Stmt>, PrimExpr>
MakeTMARows(const Buffer &src, const Array<Range> &src_ranges,
            const Buffer &dst, const Array<Range> &dst_ranges,
            PrimExpr dst_block, PrimExpr barrier_load,
            arith::Analyzer *analyzer) {
  int n = static_cast<int>(src_ranges.size());

  auto linear_off = [](const Buffer &buf,
                       const Array<Range> &ranges) -> PrimExpr {
    int r = static_cast<int>(ranges.size());
    PrimExpr off = 0, stride = 1;
    for (int i = r - 1; i >= 0; --i) {
      off = off + ranges[i]->min * stride;
      if (i > 0) {
        stride = stride * buf->shape[i];
      }
    }
    return off;
  };

  if (IsContiguousRegion(src, src_ranges, analyzer) &&
      IsContiguousRegion(dst, dst_ranges, analyzer)) {
    PrimExpr total_elems = 1;
    for (const auto &r : src_ranges) {
      total_elems = total_elems * r->extent;
    }
    PrimExpr size_bytes =
        cast(DataType::UInt(32), TMABytesFromElements(total_elems, src->dtype));
    PrimExpr src_ptr = src.access_ptr(1, DataType::Handle(), 1,
                                      linear_off(src, src_ranges), total_elems);
    PrimExpr dst_ptr = dst.access_ptr(2, DataType::Handle(), 1,
                                      linear_off(dst, dst_ranges), total_elems);
    Stmt call =
        Evaluate(Call(DataType::Handle(), tma_store_cluster(),
                      {dst_ptr, src_ptr, dst_block, size_bytes, barrier_load}));
    return {{call}, IntImm(DataType::Int(32), 1)};
  }

  int split_dim = -1;
  for (int d = 0; d < n; ++d) {
    if (!analyzer->CanProveEqual(src_ranges[d]->extent, 1)) {
      split_dim = d;
      break;
    }
  }
  ICHECK(split_dim >= 0)
      << "MakeTMARows: all dimensions are trivial yet region is not "
         "contiguous";

  PrimExpr extent = src_ranges[split_dim]->extent;
  const auto *ext_imm = extent.as<IntImmNode>();

  if (ext_imm) {
    Array<Stmt> all_stmts;
    PrimExpr total = IntImm(DataType::Int(32), 0);
    for (int64_t k = 0; k < ext_imm->value; ++k) {
      Array<Range> new_src = src_ranges;
      Array<Range> new_dst = dst_ranges;
      PrimExpr kexpr = IntImm(DataType::Int(32), k);
      new_src.Set(split_dim,
                  Range::FromMinExtent(src_ranges[split_dim]->min + kexpr, 1));
      new_dst.Set(split_dim,
                  Range::FromMinExtent(dst_ranges[split_dim]->min + kexpr, 1));
      auto [stmts, cnt] = MakeTMARows(src, new_src, dst, new_dst, dst_block,
                                      barrier_load, analyzer);
      for (const auto &s : stmts) {
        all_stmts.push_back(s);
      }
      total = total + cnt;
    }
    return {all_stmts, total};
  }

  Var k("k_tma_row", DataType::Int(32));
  Array<Range> body_src = src_ranges;
  Array<Range> body_dst = dst_ranges;
  body_src.Set(split_dim,
               Range::FromMinExtent(src_ranges[split_dim]->min + k, 1));
  body_dst.Set(split_dim,
               Range::FromMinExtent(dst_ranges[split_dim]->min + k, 1));
  auto [body_stmts, body_cnt] = MakeTMARows(src, body_src, dst, body_dst,
                                            dst_block, barrier_load, analyzer);
  Stmt body = body_stmts.size() == 1 ? body_stmts[0]
                                     : static_cast<Stmt>(SeqStmt(body_stmts));
  Stmt for_loop =
      For(k, IntImm(DataType::Int(32), 0), extent, ForKind::kSerial, body);
  return {{for_loop}, extent * body_cnt};
}

} // namespace

namespace cuda {

// The TMA unit applies the descriptor's swizzle pattern relative to the
// shared-memory base address, so the base must sit on a swizzle-pattern
// repeat boundary or the data lands with a shifted phase (silently wrong
// results, no fault). Report the requirement implied by the chosen
// CU_TENSOR_MAP_SWIZZLE_* mode so MergeSharedMemoryAllocations can align the
// buffer accordingly.
static void RequireTMASmemAlignment(const LowerArgs &lower_args,
                                    const Buffer &shared_tensor,
                                    int cu_tensor_map_swizzle) {
  if (!lower_args.require_smem_alignment)
    return;
  // CU_TENSOR_MAP_SWIZZLE_* values equal the SwizzleMode canonical ordinals.
  SwizzleMode mode = SwizzleMode::FromOrdinal(cu_tensor_map_swizzle);
  lower_args.require_smem_alignment(shared_tensor->data, mode.SmemAlignment());
}

struct TMAIm2ColDesc {
  size_t rank;
  int data_type;
  Array<PrimExpr> global_shape;
  Array<PrimExpr> global_stride;
  Array<PrimExpr> elem_stride;
  Array<PrimExpr> lower_corner;
  Array<PrimExpr> upper_corner;
  PrimExpr global_addr;
  int smem_box_pixel;
  int smem_box_channel;
  int swizzle;
  int interleave;
  int oob_fill;
  int l2_promotion;

  Array<PrimExpr> EncodeCallArgs() const {
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
};

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level);

  static Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer);

private:
  static Layout ComputeLinearLayout(const Buffer &shared_tensor);

  static void CollectFragmentLayouts(const PrimExpr &expr,
                                     const Map<Var, PrimExpr> &bind_var_to_expr,
                                     const LayoutMap &existing_layouts,
                                     PrimExpr thread_extent,
                                     Range thread_bounds,
                                     Map<Buffer, Layout> &result_map);

  static CopyInst SelectInst(const CopyNode &op, Target target,
                             const LayoutMap &layout_map,
                             arith::Analyzer *analyzer, bool buffer_oob);

  static void CheckParallelLoopLayout(const CopyNode &op, CopyInst copy_inst);

  static LayoutMap InferTMemLayout(const CopyNode &op,
                                   const LayoutInferArgs &layout_args,
                                   CopyInst copy_inst);

  static LayoutMap InferBulkLayout(const CopyNode &op,
                                   const LayoutInferArgs &layout_args,
                                   InferLevel level, CopyInst copy_inst);

  static Stmt LowerNormal(const CopyNode &op, const LowerArgs &lower_args,
                          arith::Analyzer *analyzer);

  static Stmt LowerCluster(const CopyNode &op, const LowerArgs &lower_args,
                           arith::Analyzer *analyzer);

  static Stmt LowerCPAsync(const CopyNode &op, const LowerArgs &lower_args,
                           arith::Analyzer *analyzer);

  static Stmt LowerLDSM(const CopyNode &op, const LowerArgs &lower_args,
                        arith::Analyzer *analyzer, CopyInst copy_inst);

  static Stmt LowerTmem(const CopyNode &op, const LowerArgs &lower_args,
                        arith::Analyzer *analyzer);

  static Stmt LowerBulk(const CopyNode &op, const LowerArgs &lower_args,
                        arith::Analyzer *analyzer, CopyInst copy_inst);

  static Stmt LowerBulkGather4(const CopyNode &op, const LowerArgs &lower_args,
                               arith::Analyzer *analyzer, CopyInst copy_inst);

  static Stmt LowerBulk1D(const CopyNode &op, const LowerArgs &lower_args,
                          arith::Analyzer *analyzer, CopyInst copy_inst);
};

struct Im2Col {
  static Stmt Lower(const Im2ColOpNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer);
};

Layout Copy::ComputeLinearLayout(const Buffer &shared_tensor) {
  Array<PrimExpr> input_size = shared_tensor->shape;
  Array<PrimExpr> forward_vars;
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_vars.push_back(InputPlaceholder(i));
  }

  Array<PrimExpr> forward_index;
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_index.push_back(FloorDiv(forward_vars[i], 256));
  }
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_index.push_back(FloorMod(forward_vars[i], 256));
  }
  return Layout(input_size, forward_index);
}

void Copy::CollectFragmentLayouts(const PrimExpr &expr,
                                  const Map<Var, PrimExpr> &bind_var_to_expr,
                                  const LayoutMap &existing_layouts,
                                  PrimExpr thread_extent, Range thread_bounds,
                                  Map<Buffer, Layout> &result_map) {
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (auto bl = node.as<BufferLoadNode>()) {
      if (IsFragmentBuffer(bl->buffer) && !existing_layouts.count(bl->buffer) &&
          !result_map.count(bl->buffer)) {
        auto f = Fragment::FullyReplicated(bl->buffer->shape, thread_extent);
        result_map.Set(bl->buffer, f->BindThreadRange(thread_bounds));
      }
    } else if (auto var_node = node.as<VarNode>()) {
      auto var = GetRef<Var>(var_node);
      if (bind_var_to_expr.count(var)) {
        CollectFragmentLayouts(bind_var_to_expr[var], bind_var_to_expr,
                               existing_layouts, thread_extent, thread_bounds,
                               result_map);
      }
    }
  });
}

LayoutMap Copy::InferLayout(const CopyNode &op,
                            const LayoutInferArgs &layout_args,
                            InferLevel level) {
  CopyInst copy_inst =
      SelectInst(op, layout_args.target, layout_args.layout_map,
                 layout_args.analyzer, layout_args.buffer_oob);
  CheckParallelLoopLayout(op, copy_inst);

  if (copy_inst == CopyInst::kTMemLoad || copy_inst == CopyInst::kTMemStore) {
    return InferTMemLayout(op, layout_args, copy_inst);
  }
  if (copy_inst == CopyInst::kBulkLoad || copy_inst == CopyInst::kBulkStore ||
      copy_inst == CopyInst::kBulkLoad1D ||
      copy_inst == CopyInst::kBulkStore1D) {
    return InferBulkLayout(op, layout_args, level, copy_inst);
  }

  // For normal/cp.async/LDSM/STSM, layout inference follows the generated
  // SIMT loop. CUDA-specific explicit layout cases are handled above.
  return op.InferSIMTLayout(layout_args, level);
}

void Copy::CheckParallelLoopLayout(const CopyNode &op, CopyInst copy_inst) {
  if (!op.annotations.count(attr::kParallelLoopLayout)) {
    return;
  }
  if (copy_inst == CopyInst::kNormal || copy_inst == CopyInst::kCPAsync) {
    return;
  }

  std::ostringstream oss;
  oss << "T.copy loop layout annotation requires SIMT copy; got "
      << CopyInstToString(copy_inst) << " for src=" << op.src->name
      << ", dst=" << op.dst->name
      << ". Remove loop_layout or change copy pattern.";
  LOG(FATAL) << oss.str();
}

LayoutMap Copy::InferTMemLayout(const CopyNode &op,
                                const LayoutInferArgs &layout_args,
                                CopyInst copy_inst) {
  // TODO (mzw) Add support for tcgen05.cp in CUDA tmem lowering.
  LayoutMap results;
  bool is_tmem_load = copy_inst == CopyInst::kTMemLoad;
  Buffer tmem_buf = is_tmem_load ? op.src : op.dst;
  Buffer reg_buf = is_tmem_load ? op.dst : op.src;

  if (!layout_args.layout_map.count(reg_buf) &&
      layout_args.layout_map.count(tmem_buf)) {
    Layout tmem_layout = layout_args.layout_map[tmem_buf];
    Array<IterVar> logical_coords = op.MakeIterVars();
    Array<PrimExpr> logical_coords_var = {logical_coords[0]->var,
                                          logical_coords[1]->var};
    Array<PrimExpr> phy_indices = tmem_layout->Forward(logical_coords_var);

    arith::Analyzer analyzer;
    for (const auto &iv : logical_coords) {
      analyzer.Bind(iv->var, iv->dom);
    }
    arith::ConstIntBound phy_row_bounds =
        analyzer.const_int_bound(phy_indices[0]);
    arith::ConstIntBound phy_col_bounds =
        analyzer.const_int_bound(phy_indices[1]);
    Range row_dom = Range(static_cast<int>(phy_row_bounds->min_value),
                          static_cast<int>(phy_row_bounds->max_value + 1));
    Range col_dom = Range(static_cast<int>(phy_col_bounds->min_value),
                          static_cast<int>(phy_col_bounds->max_value + 1));

    constexpr int WARP_SIZE = 32;
    constexpr int WARPGROUP_SIZE = 4 * WARP_SIZE;
    ICHECK(is_const_int(layout_args.thread_bounds->extent))
        << "Tensor memory copy requires thread_bounds->extent (num_threads) "
           "to be constant integers";
    int num_threads = *as_const_int(layout_args.thread_bounds->extent);
    ICHECK(num_threads % WARPGROUP_SIZE == 0)
        << "Tensor memory copy requires thread bounds to be aligned to "
           "warpgroups, but found "
        << "thread range = " << layout_args.thread_bounds;

    for (int num_useful_wgs = num_threads / WARPGROUP_SIZE; num_useful_wgs >= 1;
         --num_useful_wgs) {
      int num_useful_threads = num_useful_wgs * WARPGROUP_SIZE;
      Tcgen05Meta meta = GetTcgen05MetaLd32Dp32B();
      auto [is_success, tmem_coord2frag, num_chunks_each_wg] =
          ExpandTcgen05Layout(
              meta, phy_col_bounds->max_value - phy_col_bounds->min_value + 1,
              num_useful_threads, row_dom, col_dom);
      (void)num_chunks_each_wg;
      if (!is_success) {
        continue;
      }
      Fragment logical_coord2frag =
          Fragment(logical_coords, tmem_coord2frag->Forward(phy_indices),
                   tmem_coord2frag->ForwardThread(phy_indices, std::nullopt),
                   MakeIterVar("rep", 1));
      results.Set(reg_buf, logical_coord2frag->BindThreadRange(
                               layout_args.thread_bounds));
      break;
    }
  }

  return results;
}

LayoutMap Copy::InferBulkLayout(const CopyNode &op,
                                const LayoutInferArgs &layout_args,
                                InferLevel level, CopyInst copy_inst) {
  Map<Buffer, Layout> result_map;

  bool is_tma_1d =
      copy_inst == CopyInst::kBulkLoad1D || copy_inst == CopyInst::kBulkStore1D;
  bool is_load =
      copy_inst == CopyInst::kBulkLoad || copy_inst == CopyInst::kBulkLoad1D;
  bool is_store =
      copy_inst == CopyInst::kBulkStore || copy_inst == CopyInst::kBulkStore1D;
  Buffer shared_tensor = is_load ? op.dst : op.src;
  Array<Range> shared_range = is_load ? op.dst_range : op.src_range;

  if (is_tma_1d && shared_range.size() == 1) {
    // 1D TMA Store with single dimension can not be swizzled. 1D TMA can also
    // have multiple dimensions when the last dimension is continuous.
    return result_map;
  }

  // Fragment buffers used as TMA indices should be replicated on all threads.
  PrimExpr thread_extent = layout_args.thread_bounds->extent;
  for (const auto &range : op.src_range) {
    CollectFragmentLayouts(range->min, layout_args.bind_var_to_expr,
                           layout_args.layout_map, thread_extent,
                           layout_args.thread_bounds, result_map);
    CollectFragmentLayouts(range->extent, layout_args.bind_var_to_expr,
                           layout_args.layout_map, thread_extent,
                           layout_args.thread_bounds, result_map);
  }
  for (const auto &range : op.dst_range) {
    CollectFragmentLayouts(range->min, layout_args.bind_var_to_expr,
                           layout_args.layout_map, thread_extent,
                           layout_args.thread_bounds, result_map);
    CollectFragmentLayouts(range->extent, layout_args.bind_var_to_expr,
                           layout_args.layout_map, thread_extent,
                           layout_args.thread_bounds, result_map);
  }

  if (is_tma_1d) {
    // 1D TMA requires contiguous shared memory. Do not infer a swizzled shared
    // layout here, otherwise final instruction selection may fall back to
    // descriptor-based multidimensional TMA.
    return result_map;
  }

  if (level == InferLevel::kFree &&
      !layout_args.layout_map.count(shared_tensor)) {
    if (is_store) {
      // For BulkStore, infer a swizzled shared-memory layout when possible.
      int dim = shared_tensor->shape.size();
      const int64_t mat_stride = *as_const_int(shared_tensor->shape[dim - 2]);
      const int64_t mat_continuous =
          *as_const_int(shared_tensor->shape[dim - 1]);
      Layout swizzle_layout_2d =
          MakeGemmABLayoutHopper(mat_stride, mat_continuous, mat_continuous,
                                 shared_tensor->dtype.bits(),
                                 /*k_inner=*/true);
      if (StructuralEqual()(swizzle_layout_2d, MakeLinearLayout(Array<PrimExpr>{
                                                   Integer(mat_stride),
                                                   Integer(mat_continuous)}))) {
        result_map.Set(shared_tensor, ComputeLinearLayout(shared_tensor));
      } else {
        result_map.Set(shared_tensor, ExpandLayoutToMatchBuffer(
                                          swizzle_layout_2d, shared_tensor));
      }
    } else {
      result_map.Set(shared_tensor, ComputeLinearLayout(shared_tensor));
    }
  }

  return result_map;
}

CopyInst Copy::SelectInst(const CopyNode &op, Target target,
                          const LayoutMap &layout_map,
                          arith::Analyzer *analyzer, bool buffer_oob) {
  CopyAnalysisContext ctx;
  ctx.target = target;
  ctx.layout_map = &layout_map;
  ctx.analyzer = analyzer;
  ctx.buffer_oob = buffer_oob;
  ctx.emit_diagnostics = true;
  auto result = SelectCopyInstForLowering(op, ctx);
  ICHECK(result.supported) << result.reason;
  return result.inst;
}

Stmt Copy::Lower(const CopyNode &op, const LowerArgs &lower_args,
                 arith::Analyzer *analyzer) {
  auto copy_inst = SelectInst(op, lower_args.target, lower_args.layout_map,
                              analyzer, /*buffer_oob=*/false);
  if (op.dst_block.defined()) {
    ICHECK(TargetHasBulkCopy(lower_args.target))
        << "T.copy with dst_block requires cluster-copy support (CUDA SM90+). "
        << "Got target=" << lower_args.target;
    return LowerCluster(op, lower_args, analyzer);
  }
  if (copy_inst == CopyInst::kTMemLoad || copy_inst == CopyInst::kTMemStore) {
    auto tmem_copy = LowerTmem(op, lower_args, analyzer);
    ICHECK(tmem_copy.defined()) << "Failed to lower tensor memory copy";
    return tmem_copy;
  } else if (copy_inst == CopyInst::kBulkLoad1D ||
             copy_inst == CopyInst::kBulkStore1D) {
    auto bulk_copy = LowerBulk1D(op, lower_args, analyzer, copy_inst);
    ICHECK(bulk_copy.defined()) << "Failed to lower bulk load 1d";
    return bulk_copy;
  } else if (copy_inst == CopyInst::kBulkLoad ||
             copy_inst == CopyInst::kBulkStore) {
    auto bulk_copy = LowerBulk(op, lower_args, analyzer, copy_inst);
    ICHECK(bulk_copy.defined()) << "Failed to lower bulk load/store";
    return bulk_copy;
  } else if (copy_inst == CopyInst::kBulkLoadGather4 ||
             copy_inst == CopyInst::kBulkStoreScatter4) {
    auto bulk_copy = LowerBulkGather4(op, lower_args, analyzer, copy_inst);
    ICHECK(bulk_copy.defined()) << "Failed to lower tma gather4/scatter4";
    return bulk_copy;
  } else if (copy_inst == CopyInst::kLDSM || copy_inst == CopyInst::kSTSM) {
    auto ldsm_copy = LowerLDSM(op, lower_args, analyzer, copy_inst);
    ICHECK(ldsm_copy.defined()) << "Failed to lower ptx matrix copy";
    return ldsm_copy;
  } else if (copy_inst == CopyInst::kCPAsync) {
    auto cp_async_copy = LowerCPAsync(op, lower_args, analyzer);
    ICHECK(cp_async_copy.defined()) << "Failed to lower cp.async copy";
    return cp_async_copy;
  } else if (copy_inst == CopyInst::kNormal) {
    return LowerNormal(op, lower_args, analyzer);
  } else {
    LOG(FATAL) << "Unsupported copy inst " << static_cast<int>(copy_inst);
  }
}

Stmt Copy::LowerCPAsync(const CopyNode &op, const LowerArgs &lower_args,
                        arith::Analyzer *analyzer) {
  using namespace tvm::transform;

  PassContext pass_ctx = PassContext::Current();
  bool enable_async_copy =
      pass_ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(true)).value();
  bool no_implicit_commit_wait = GetNoImplicitAsyncCommitWait(op);
  bool explicit_async_semantics = no_implicit_commit_wait || GetIsAsyncCopy(op);
  if (!enable_async_copy && !explicit_async_semantics) {
    return LowerNormal(op, lower_args, analyzer);
  }

  auto simt_loop = op.MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
  auto par_op = ParallelOp(fused_loop);

  std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                    InferLevel::kFree};
  for (auto level : levels) {
    par_op->InferLayout({lower_args.target,
                         lower_args.thread_bounds,
                         lower_args.layout_map,
                         analyzer,
                         false,
                         lower_args.buffer_remap,
                         {}},
                        level);
  }
  auto loop_layout = par_op->GetLoopLayout();
  Stmt lowered_loop = LowerParallelLoop(
      par_op->GetRoot(), loop_layout, lower_args.thread_var, analyzer,
      lower_args.layout_map, par_op->GetPredicate(lower_args.thread_var),
      /*parallel_loop=*/true, /*should_vectorize=*/true,
      par_op->LoopLayoutRequiresPaddingGuard());

  auto inject_result =
      InjectPTXAsyncCopy(lowered_loop, /*enable_auto_async_copy=*/true,
                         /*async_without_async_commit_wait=*/
                         no_implicit_commit_wait || GetIsAsyncCopy(op));
  Stmt cp_async_loop = inject_result.stmt;
  if (!inject_result.injected_ptx_async_copy) {
    DLOG(WARNING) << "cp.async rewrite miss for copy src=" << op.src->name
                  << " (scope=" << op.src.scope() << ", dtype=" << op.src->dtype
                  << "), dst=" << op.dst->name << " (scope=" << op.dst.scope()
                  << ", dtype=" << op.dst->dtype
                  << "), no_implicit_async_commit_wait="
                  << no_implicit_commit_wait
                  << ", is_async_copy=" << GetIsAsyncCopy(op);
    if (no_implicit_commit_wait) {
      DLOG(WARNING)
          << "Pipeline-managed async copy fallback to normal copy because "
             "cp.async rewrite found no eligible global->shared store.";
      return lowered_loop;
    }
    if (explicit_async_semantics) {
      LOG(FATAL) << "Explicit async copy semantics require cp.async lowering, "
                    "but no eligible global->shared store was rewritten.";
    }
    DLOG(WARNING) << "Fallback to normal copy because cp.async rewrite found "
                     "no eligible global->shared store.";
    return LowerNormal(op, lower_args, analyzer);
  }
  if (no_implicit_commit_wait) {
    return cp_async_loop;
  }
  if (GetIsAsyncCopy(op)) {
    Stmt commit_group =
        Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
    return SeqStmt({cp_async_loop, commit_group});
  }
  return cp_async_loop;
}

Stmt Copy::LowerNormal(const CopyNode &op, const LowerArgs &lower_args,
                       arith::Analyzer *analyzer) {
  return tl::LowerNormalCopy(op, lower_args, analyzer);
}

Stmt Copy::LowerCluster(const CopyNode &op, const LowerArgs &lower_args,
                        arith::Analyzer *analyzer) {
  const Buffer &src = op.src;
  const Buffer &dst = op.dst;
  const Array<Range> &src_range = op.src_range;
  const Array<Range> &dst_range = op.dst_range;
  ICHECK(op.dst_block.defined());
  ICHECK(src.scope() == "shared" || src.scope() == "shared.dyn");
  ICHECK(dst.scope() == "shared" || dst.scope() == "shared.dyn");

  if (auto barrier_opt = GetBarrier(op)) {
    bool src_contiguous = IsContiguousRegion(src, src_range, analyzer);
    bool dst_contiguous = IsContiguousRegion(dst, dst_range, analyzer);

    PrimExpr src_elements = 1;
    for (auto r : src_range) {
      src_elements = src_elements * r->extent;
    }
    PrimExpr dst_elements = 1;
    for (auto r : dst_range) {
      dst_elements = dst_elements * r->extent;
    }
    bool element_match = analyzer->CanProveEqual(src_elements, dst_elements);

    if (src_contiguous && dst_contiguous && element_match) {
      PrimExpr barrier_load = barrier_opt.value();

      auto compute_linear_offset = [](const Buffer &buf,
                                      const Array<Range> &ranges) -> PrimExpr {
        PrimExpr offset = 0;
        PrimExpr stride = 1;
        for (int i = static_cast<int>(ranges.size()) - 1; i >= 0; --i) {
          offset = offset + ranges[i]->min * stride;
          if (i > 0) {
            stride = stride * buf->shape[i];
          }
        }
        return offset;
      };

      PrimExpr dst_offset = compute_linear_offset(dst, dst_range);
      PrimExpr src_offset = compute_linear_offset(src, src_range);
      PrimExpr total_elements = 1;
      for (auto r : src_range) {
        total_elements = total_elements * r->extent;
      }
      PrimExpr size_bytes = cast(
          DataType::UInt(32), TMABytesFromElements(total_elements, src->dtype));

      PrimExpr dst_ptr =
          dst.access_ptr(2, DataType::Handle(), 1, dst_offset, total_elements);
      PrimExpr src_ptr =
          src.access_ptr(1, DataType::Handle(), 1, src_offset, total_elements);

      Stmt bulk_copy = Evaluate(Call(
          DataType::Handle(), tma_store_cluster(),
          {dst_ptr, src_ptr, op.dst_block.value(), size_bytes, barrier_load}));

      return IfThenElse(
          EQ(lower_args.thread_var, lower_args.thread_bounds->min), bulk_copy);
    }

    bool same_shape = (src_range.size() == dst_range.size());
    for (size_t d = 0; d < src_range.size() && same_shape; ++d) {
      if (!analyzer->CanProveEqual(src_range[d]->extent,
                                   dst_range[d]->extent)) {
        same_shape = false;
      }
    }

    if (element_match && same_shape) {
      PrimExpr barrier_load = barrier_opt.value();
      const auto *barrier_buf_load = barrier_load.as<tirx::BufferLoadNode>();
      ICHECK(barrier_buf_load)
          << "LowerCluster: expected BufferLoad for barrier annotation";
      Var barrier_data_var = barrier_buf_load->buffer->data;

      auto [tma_stmts, n_rows] =
          MakeTMARows(src, src_range, dst, dst_range, op.dst_block.value(),
                      barrier_load, analyzer);

      if (lower_args.update_barrier_arrive) {
        lower_args.update_barrier_arrive(barrier_data_var, n_rows);
      }

      Stmt seq = (tma_stmts.size() == 1)
                     ? tma_stmts[0]
                     : static_cast<Stmt>(SeqStmt(tma_stmts));
      return IfThenElse(
          EQ(lower_args.thread_var, lower_args.thread_bounds->min), seq);
    }

    LOG(WARNING)
        << "Falling back to element-wise cluster copy: bulk cluster paths "
           "require matching element counts and same per-dim extents between "
           "src and dst. src="
        << src->name << ", dst=" << dst->name;
  }

  auto simt_loop = op.MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));

  std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                    InferLevel::kFree};
  auto par_op = ParallelOp(fused_loop);
  for (auto level : levels) {
    par_op->InferLayout({lower_args.target,
                         lower_args.thread_bounds,
                         lower_args.layout_map,
                         analyzer,
                         false,
                         lower_args.buffer_remap,
                         {}},
                        level);
  }
  auto loop_layout = par_op->GetLoopLayout();
  auto thread_loop = PartitionLoop(par_op->GetRoot(), lower_args.thread_var,
                                   analyzer, loop_layout);
  auto vectorized_thread_loop =
      VectorizeLoop(thread_loop, lower_args.layout_map, /*vectorize_hint=*/1);

  class ClusterCopyReplacer : public StmtExprMutator {
  public:
    ClusterCopyReplacer(const Buffer &dst, PrimExpr dst_block,
                        const Buffer &target_dst, Optional<Layout> dst_layout)
        : dst_(dst), dst_block_(dst_block), target_dst_(target_dst),
          dst_layout_(dst_layout) {}

    Stmt VisitStmt_(const BufferStoreNode *op) final {
      if (op->buffer.same_as(dst_)) {
        Array<PrimExpr> physical_indices = op->indices;
        if (!target_dst_.same_as(dst_) && dst_layout_.defined()) {
          physical_indices = dst_layout_.value()->Forward(op->indices);
        }

        PrimExpr linearized_index = physical_indices[0];
        if (physical_indices.size() > 1) {
          PrimExpr multiplier = 1;
          linearized_index = 0;
          for (int i = static_cast<int>(physical_indices.size()) - 1; i >= 0;
               --i) {
            linearized_index =
                linearized_index + physical_indices[i] * multiplier;
            if (i > 0) {
              multiplier = multiplier * target_dst_->shape[i];
            }
          }
        }

        Buffer target_buffer = target_dst_;
        if (target_dst_.same_as(dst_)) {
          target_buffer = op->buffer;
        }

        PrimExpr total_elems = 1;
        for (const PrimExpr &s : target_buffer->shape) {
          total_elems = total_elems * s;
        }

        Stmt remote_store =
            Evaluate(Call(DataType::Handle(), ptx_cluster_store(),
                          {target_buffer.access_ptr(2), op->value, dst_block_,
                           linearized_index}));

        return IfThenElse(linearized_index < total_elems, remote_store, Stmt());
      }
      return StmtExprMutator::VisitStmt_(op);
    }

  private:
    const Buffer &dst_;
    PrimExpr dst_block_;
    const Buffer &target_dst_;
    Optional<Layout> dst_layout_;
  };

  Buffer target_dst = dst;
  if (lower_args.buffer_remap.count(dst)) {
    target_dst = lower_args.buffer_remap[dst];
  }

  Optional<Layout> dst_layout = std::nullopt;
  if (lower_args.layout_map.count(dst)) {
    dst_layout = lower_args.layout_map[dst];
  }

  Stmt simt_copy = ClusterCopyReplacer(dst, op.dst_block.value(), target_dst,
                                       dst_layout)(vectorized_thread_loop);

  if (auto barrier_opt = GetBarrier(op)) {
    Stmt sync = Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                              {StringImm("shared")}));
    Stmt arrive =
        Evaluate(Call(DataType::Handle(), ptx_arrive_cluster_barrier(),
                      {barrier_opt.value(), op.dst_block.value()}));
    Stmt guarded_arrive = IfThenElse(
        EQ(lower_args.thread_var, lower_args.thread_bounds->min), arrive);
    return SeqStmt({simt_copy, sync, guarded_arrive});
  }
  return simt_copy;
}

Stmt Copy::LowerLDSM(const CopyNode &op, const LowerArgs &lower_args,
                     arith::Analyzer *analyzer, CopyInst copy_inst) {
  const Buffer &src = op.src;
  const Buffer &dst = op.dst;
  const Array<Range> &src_range = op.src_range;
  const Array<Range> &dst_range = op.dst_range;

  ICHECK(copy_inst == CopyInst::kLDSM || copy_inst == CopyInst::kSTSM)
      << "Invalid copy inst " << static_cast<int>(copy_inst);
  bool is_ldmatrix = copy_inst == CopyInst::kLDSM;

  Array<IterVar> loop_vars = op.MakeIterVars();
  if (loop_vars.size() < 2) {
    return LowerNormal(op, lower_args, analyzer);
  }
  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);
  PrimExpr src_predicate = op.MakePredicate(analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate = op.MakePredicate(analyzer, loop_vars, dst->shape, 1);
  if (src_predicate.defined() || dst_predicate.defined()) {
    return LowerNormal(op, lower_args, analyzer);
  }

  Buffer shared_tensor = is_ldmatrix ? src : dst;
  Buffer local_tensor = is_ldmatrix ? dst : src;
  Array<Range> local_region = is_ldmatrix ? src_range : dst_range;
  bool is_full_range = true;
  for (size_t i = 0; i < local_region.size(); i++) {
    if (!analyzer->CanProveEqual(local_region[i]->extent,
                                 local_tensor->shape[i])) {
      is_full_range = false;
      break;
    }
  }
  if (!is_full_range) {
    return LowerNormal(op, lower_args, analyzer);
  }

  Array<PrimExpr> local_indices =
      op.MakeIndices(loop_vars, is_ldmatrix ? 1 : 0);
  Fragment local_layout =
      Downcast<Fragment>(lower_args.layout_map[local_tensor]);
  Array<PrimExpr> local_indices_transformed =
      local_layout->Forward(local_indices);
  local_tensor = lower_args.buffer_remap[local_tensor];
  if (local_layout->OutputDim() != 1) {
    return LowerNormal(op, lower_args, analyzer);
  }

  Array<PrimExpr> shared_indices =
      op.MakeIndices(loop_vars, is_ldmatrix ? 0 : 1);
  bool is_transposed;
  IterVar col_var = loop_vars[loop_vars.size() - 1];
  IterVar row_var = loop_vars[loop_vars.size() - 2];
  PrimExpr local_layout_thread_map =
      FloorMod(local_layout->ForwardThread(local_indices, std::nullopt), 32);
  PrimExpr matrix_8x8_thread_map = MakeGemmFragment8x8()->ForwardThread(
      {FloorMod(row_var, 8), FloorMod(col_var, 8)}, std::nullopt);
  PrimExpr matrix_8x8_thread_map_trans =
      MakeGemmFragment8x8Transposed()->ForwardThread(
          {FloorMod(row_var, 8), FloorMod(col_var, 8)}, std::nullopt);
  PrimExpr local_indices_flattened =
      local_tensor.OffsetOf(local_indices_transformed).back();
  if (analyzer->CanProveEqual(matrix_8x8_thread_map, local_layout_thread_map) &&
      IndicesCanVectorize(local_indices_flattened, col_var->var,
                          col_var->dom->extent, 2, analyzer)) {
    is_transposed = false;
  } else if (analyzer->CanProveEqual(matrix_8x8_thread_map_trans,
                                     local_layout_thread_map) &&
             IndicesCanVectorize(local_indices_flattened, row_var->var,
                                 row_var->dom->extent, 2, analyzer)) {
    is_transposed = true;
  } else {
    return LowerNormal(op, lower_args, analyzer);
  }

  const bool use_m16n8_stmatrix =
      !is_ldmatrix && is_transposed &&
      TargetHasStmatrix(lower_args.target, /*is_m16n8=*/true) &&
      shared_tensor->dtype.bits() == 8;
  const int shared_elem_bytes = use_m16n8_stmatrix ? 1 : 2;
  if (shared_tensor->dtype.bytes() != shared_elem_bytes) {
    return LowerNormal(op, lower_args, analyzer);
  }
  const int elems_per_reg = 4 / shared_elem_bytes;
  PrimExpr flattened_indice = shared_tensor.OffsetOf(shared_indices).back();
  if (!IndicesCanVectorize(flattened_indice, loop_vars.back()->var,
                           loop_vars.back()->dom->extent,
                           use_m16n8_stmatrix ? 4 : 8, analyzer)) {
    return LowerNormal(op, lower_args, analyzer);
  }

  for (size_t i = 0; i < dst_range.size(); i++) {
    if (!is_zero(dst_range[i]->min) ||
        !analyzer->CanProveEqual(dst_range[i]->extent, dst->shape[i]))
      return LowerNormal(op, lower_args, analyzer);
  }

  PrimExpr extent = local_tensor->shape[0];
  int num = 1;
  if (analyzer->CanProveEqual(FloorMod(extent, elems_per_reg * 4), 0))
    num = 4;
  else if (analyzer->CanProveEqual(FloorMod(extent, elems_per_reg * 2), 0))
    num = 2;

  Array<PrimExpr> args;
  const Op &copy_op = is_ldmatrix ? tl::ptx_ldmatrix() : tl::ptx_stmatrix();
  args.push_back(static_cast<int>(is_transposed));
  args.push_back(num);

  Var local_iter("i");
  Layout inv = local_layout->Inverse();
  Array<PrimExpr> shared_coords;
  PrimExpr warp = FloorDiv(lower_args.thread_var, 32) * 32;
  if (!is_transposed) {
    auto local_index = analyzer->Simplify(
        local_iter * elems_per_reg * num +
        elems_per_reg * FloorMod(FloorDiv(lower_args.thread_var, 8), num));
    auto thread_index =
        analyzer->Simplify(warp + FloorMod(lower_args.thread_var, 8) * 4);
    shared_coords = inv->Forward({local_index, thread_index});
  } else {
    auto local_index = analyzer->Simplify(
        local_iter * elems_per_reg * num +
        elems_per_reg * FloorMod(FloorDiv(lower_args.thread_var, 8), num) +
        FloorMod(lower_args.thread_var, elems_per_reg));
    auto thread_index = analyzer->Simplify(
        warp + FloorDiv(FloorMod(lower_args.thread_var, 8), elems_per_reg));
    shared_coords = inv->Forward({local_index, thread_index});
  }
  shared_coords.pop_back();
  PrimExpr shared_addr = Call(
      DataType::Handle(), tl::access_ptr(),
      {BufferLoad(shared_tensor, shared_coords), PrimExpr(elems_per_reg * num),
       make_const(DataType::Int(32), is_ldmatrix ? 1 : 2)});
  args.push_back(shared_addr);

  if (is_ldmatrix) {
    if (local_tensor->dtype != shared_tensor->dtype) {
      return LowerNormal(op, lower_args, analyzer);
    }
    PrimExpr local_addr =
        Call(DataType::Handle(), tl::access_ptr(),
             {BufferLoad(local_tensor, {local_iter * 2 * num}),
              PrimExpr(2 * num), make_const(DataType::Int(32), 2)});
    args.push_back(local_addr);
  } else {
    for (int i = 0; i < num; i++) {
      Array<PrimExpr> values;
      for (int j = 0; j < elems_per_reg; ++j) {
        PrimExpr value =
            BufferLoad(local_tensor, {local_iter * elems_per_reg * num +
                                      elems_per_reg * i + j});
        if (local_tensor->dtype != shared_tensor->dtype) {
          value = Cast(shared_tensor->dtype, value);
        }
        values.push_back(value);
      }
      PrimExpr value_packed = use_m16n8_stmatrix
                                  ? Call(DataType::Int(32), pack_b8x4(), values)
                                  : Call(DataType::Int(32), pack_b16(), values);
      args.push_back(value_packed);
    }
    args.push_back(StringImm(use_m16n8_stmatrix ? "m16n8" : "m8n8"));
  }

  auto body = Evaluate(Call(DataType::Handle(), copy_op, args));
  For for_node = For(local_iter, 0, FloorDiv(extent, elems_per_reg * num),
                     ForKind::kSerial, body);
  for_node = PragmaUnrollLoop(for_node);
  auto range = lower_args.thread_bounds;
  if (range.defined()) {
    auto thread_var = lower_args.thread_var;
    auto thread_var_with_offset = thread_var - range->min;
    for_node.CopyOnWrite()->body =
        Substitute(for_node->body, {{thread_var, thread_var_with_offset}});
  }
  return for_node;
}

Stmt Copy::LowerTmem(const CopyNode &op, const LowerArgs &lower_args,
                     arith::Analyzer *analyzer) {
  const Buffer &src = op.src;
  const Buffer &dst = op.dst;

  if (src.scope() != "shared.tmem" && dst.scope() != "shared.tmem") {
    return Stmt();
  }
  ICHECK(TargetHasTmem(lower_args.target))
      << "Target " << lower_args.target->str()
      << " does not support tensor memory copy";

  bool is_ld = false;
  bool is_st = false;
  bool is_cp = false;
  bool src_needs_pack = 16 == src->dtype.bits();
  bool dst_needs_unpack = 16 == dst->dtype.bits();

  if (src.scope() == "shared.tmem" && IsFragmentBuffer(dst)) {
    is_ld = true;
  } else if (IsFragmentBuffer(src) && dst.scope() == "shared.tmem") {
    is_st = true;
  } else if (src.scope() == "shared.dyn" && dst.scope() == "shared.tmem") {
    is_cp = true;
  } else {
    LOG(FATAL) << "Unsupported tensor memory copy: "
               << "src scope = " << src.scope()
               << ", dst scope = " << dst.scope();
  }
  ICHECK(!is_cp)
      << "Copy from shared memory to tensor memory is not supported yet";

  Array<IterVar> loop_vars = op.MakeIterVars();
  ICHECK(loop_vars.size() == 2) << "Only support 2D tensor memory copy, got "
                                << loop_vars.size() << " dimensions";
  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);
  PrimExpr src_predicate = op.MakePredicate(analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate = op.MakePredicate(analyzer, loop_vars, dst->shape, 1);
  ICHECK(!src_predicate.defined() && !dst_predicate.defined())
      << "Tensor memory copy does not support predicates, got " << src_predicate
      << " and " << dst_predicate;
  ICHECK(is_const_int(loop_vars[0]->dom->min) &&
         is_const_int(loop_vars[0]->dom->extent) &&
         is_const_int(loop_vars[1]->dom->min) &&
         is_const_int(loop_vars[1]->dom->extent))
      << "Tensor memory copy requires loop bounds to be constant integers";
  int64_t logical_row_min = *as_const_int(loop_vars[0]->dom->min);
  int64_t logical_col_min = *as_const_int(loop_vars[1]->dom->min);

  constexpr int WARP_SIZE = 32;
  constexpr int WARPGROUP_SIZE = 4 * WARP_SIZE;
  ICHECK(is_const_int(lower_args.thread_bounds->extent))
      << "Tensor memory copy requires thread_bounds->extent (num_threads) to "
         "be constant integers";
  int num_threads = *as_const_int(lower_args.thread_bounds->extent);
  ICHECK(analyzer->CanProveEqual(
             FloorMod(lower_args.thread_bounds->min, WARPGROUP_SIZE), 0) &&
         num_threads % WARPGROUP_SIZE == 0)
      << "Tensor memory copy requires thread bounds to be aligned to "
         "warpgroups, but found "
      << "thread range = " << lower_args.thread_bounds;

  Buffer tmem_buf = is_ld ? src : dst;
  Buffer reg_buf = is_ld ? dst : src;
  int tmem_side = is_ld ? 0 : 1;
  bool needs_pack_unpack = is_ld ? src_needs_pack : dst_needs_unpack;

  ICHECK(lower_args.layout_map.count(tmem_buf))
      << "Tmem buffer " << tmem_buf->name
      << " does not have a layout specified";
  ICHECK(lower_args.layout_map.count(reg_buf))
      << "Register buffer " << reg_buf->name
      << " does not have a layout specified";
  Layout tmem_layout = lower_args.layout_map[tmem_buf];
  Fragment reg_layout = Downcast<Fragment>(lower_args.layout_map[reg_buf]);

  Array<PrimExpr> logical_indices = op.MakeIndices(loop_vars, tmem_side);
  Array<PrimExpr> phy_indices = tmem_layout->Forward(logical_indices);

  arith::ConstIntBound phy_row_bounds =
      analyzer->const_int_bound(phy_indices[0]);
  arith::ConstIntBound phy_col_bounds =
      analyzer->const_int_bound(phy_indices[1]);
  int tmem_phy_row_min = phy_row_bounds->min_value;
  int tmem_phy_row_max = phy_row_bounds->max_value;
  int tmem_phy_col_min = phy_col_bounds->min_value;
  int tmem_phy_col_max = phy_col_bounds->max_value;
  int tmem_phy_col_extent = tmem_phy_col_max - tmem_phy_col_min + 1;
  Range row_dom = Range(tmem_phy_row_min, tmem_phy_row_max + 1);
  Range col_dom = Range(tmem_phy_col_min, tmem_phy_col_max + 1);

  bool have_succeeded = false;
  Stmt body;

  auto try_tcgen05_instruction = [&](Tcgen05Meta meta) {
    if (have_succeeded) {
      return;
    }
    if (tmem_phy_row_min != 0 || tmem_phy_row_max != 127) {
      return;
    }
    if (tmem_phy_col_min % meta.width != 0 ||
        (tmem_phy_col_max + 1) % meta.width != 0) {
      return;
    }

    for (int num_useful_wgs = num_threads / WARPGROUP_SIZE; num_useful_wgs >= 1;
         num_useful_wgs--) {
      int num_useful_threads = num_useful_wgs * WARPGROUP_SIZE;
      auto [is_success, target_frag, num_chunks_each_wg] = ExpandTcgen05Layout(
          meta, tmem_phy_col_extent, num_useful_threads, row_dom, col_dom);
      if (!is_success) {
        continue;
      }

      PrimExpr target_thread =
          target_frag->ForwardThread(phy_indices, std::nullopt);
      PrimExpr reg_thread =
          reg_layout->ForwardThread(logical_indices, std::nullopt);
      if (!analyzer->CanProveEqual(target_thread, reg_thread)) {
        continue;
      }
      PrimExpr target_reg = target_frag->Forward(phy_indices)[0];
      PrimExpr reg_val = reg_layout->Forward(logical_indices)[0];
      if (!analyzer->CanProveEqual(target_reg, reg_val)) {
        continue;
      }

      bool use_pack_unpack_modifier = is_ld ? needs_pack_unpack : false;
      int effective_chunks =
          needs_pack_unpack ? num_chunks_each_wg / 2 : num_chunks_each_wg;
      PrimExpr relative_wg_idx =
          FloorDiv(Sub(lower_args.thread_var, lower_args.thread_bounds->min),
                   WARPGROUP_SIZE);
      PrimExpr col_offset =
          num_useful_threads == WARPGROUP_SIZE
              ? PrimExpr(0)
              : relative_wg_idx * (effective_chunks * meta.width);
      have_succeeded = true;
      Array<PrimExpr> args;
      Stmt call;
      if (is_ld) {
        args.push_back(IntImm(DataType::Int(32), meta.width * 32));
        args.push_back(IntImm(DataType::Int(32), effective_chunks));
        args.push_back(Bool(use_pack_unpack_modifier));
        args.push_back(
            BufferLoad(tmem_buf, {(int)logical_row_min, (int)logical_col_min}));
        args.push_back(col_offset);
        args.push_back(reg_buf.access_ptr(/*access_mask=*/2, DataType::Handle(),
                                          /*content_lanes=*/1, /*offset=*/0,
                                          PrimExpr(tmem_phy_col_extent)));
        call = Evaluate(Call(DataType::Handle(), tcgen05_ld(), args));
      } else {
        args.push_back(IntImm(DataType::Int(32), meta.width * 32));
        args.push_back(IntImm(DataType::Int(32), effective_chunks));
        args.push_back(Bool(use_pack_unpack_modifier));
        args.push_back(
            BufferLoad(tmem_buf, {(int)logical_row_min, (int)logical_col_min}));
        args.push_back(col_offset);
        args.push_back(reg_buf.access_ptr(/*access_mask=*/1, DataType::Handle(),
                                          /*content_lanes=*/1, /*offset=*/0,
                                          PrimExpr(tmem_phy_col_extent)));
        call = Evaluate(Call(DataType::Handle(), tcgen05_st(), args));
      }
      if (num_useful_threads != num_threads) {
        body =
            IfThenElse(lower_args.thread_var <
                           lower_args.thread_bounds->min + num_useful_threads,
                       call, Stmt());
      } else {
        body = call;
      }
      break;
    }
  };

  if (is_ld) {
    try_tcgen05_instruction(GetTcgen05MetaLd32Dp32B());
    try_tcgen05_instruction(GetTcgen05MetaLd32Dp64B());
    try_tcgen05_instruction(GetTcgen05MetaLd32Dp128B());
    try_tcgen05_instruction(GetTcgen05MetaLd32Dp256B());
  } else {
    try_tcgen05_instruction(GetTcgen05MetaSt32Dp32B());
    try_tcgen05_instruction(GetTcgen05MetaSt32Dp64B());
    try_tcgen05_instruction(GetTcgen05MetaSt32Dp128B());
    try_tcgen05_instruction(GetTcgen05MetaSt32Dp256B());
  }

  ICHECK(have_succeeded) << "Failed to find a suitable instruction for tcgen05."
                         << (is_ld ? "ld" : "st") << ". Check your layout.";

  return body;
}

namespace {

Array<PrimExpr> makeRowMajorStrides(const Array<PrimExpr> &shape) {
  Array<PrimExpr> strides(shape.size(), PrimExpr{1});
  PrimExpr stride = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    strides.Set(i, stride);
    stride *= shape[i];
  }
  return strides;
}

Array<PrimExpr> makeColumnMajorStrides(const Array<PrimExpr> &shape) {
  Array<PrimExpr> strides;
  PrimExpr stride = 1;
  for (int64_t i = 0; i < shape.size(); i++) {
    strides.push_back(stride);
    stride *= shape[i];
  }
  return strides;
}

struct BulkCopyTile {
  // Shape of the copied tile.
  Array<int64_t> tile_shape;
  // tile -> logical shared.
  cute::Layout tile_to_shared_logical;
  // tile -> logical global, using ScaledBasis for modes.
  cute::Layout tile_to_global_mode;
  // logical shared offset.
  cute::IntTuple shared_logical_offset;
  // logical global coords.
  cute::IntTuple global_logical_coords;
};

// The shared tile and the global tile should be of the exact same size. And we
// derive the mappings of the tile to the logical shared and global tensors.
BulkCopyTile ComputeBulkCopyTile(const Array<PrimExpr> &shared_shape,
                                 const Array<Range> &shared_range,
                                 const Array<Range> &global_range) {
  // The tile should have constant shape.
  Array<int64_t> shared_range_extents =
      shared_range.Map([](const Range &range) {
        auto s = as_const_int(range->extent);
        ICHECK(s) << "extent of shared_range: " << range << " is not constant";
        return *s;
      });
  Array<int64_t> global_range_extents =
      global_range.Map([](const Range &range) {
        auto s = as_const_int(range->extent);
        ICHECK(s) << "extent of global_range: " << range << " is not constant";
        return *s;
      });

  // Find out the corresponding shared modes and global modes.
  Array<int64_t> tile_shape;
  Array<cute::IntTuple> tile_to_shared_mode_stride, tile_to_global_mode_stride;
  int64_t shared_cur = 0, global_cur = 0;
  while (shared_cur < shared_range_extents.size() &&
         global_cur < global_range_extents.size()) {
    int64_t s_extent = shared_range_extents[shared_cur];
    int64_t g_extent = global_range_extents[global_cur];
    if (s_extent <= 1) {
      shared_cur++;
      continue;
    }
    if (g_extent <= 1) {
      global_cur++;
      continue;
    }
    ICHECK_EQ(s_extent, g_extent)
        << "Shared tile and global tile shape mismatch: tile_shape_shared["
        << shared_cur << "] = " << s_extent << " != " << g_extent
        << " = tile_shape_global[" << global_cur << "]";
    tile_shape.push_back(s_extent);
    tile_to_shared_mode_stride.push_back(cute::E({shared_cur}));
    tile_to_global_mode_stride.push_back(cute::E({global_cur}));
    shared_cur++;
    global_cur++;
  }
  for (; shared_cur < shared_range_extents.size(); shared_cur++) {
    ICHECK_EQ(shared_range_extents[shared_cur], 1)
        << "Shared tile and global tile shape mismatch: "
        << shared_range_extents << " vs " << global_range_extents;
  }
  for (; global_cur < global_range_extents.size(); global_cur++) {
    ICHECK_EQ(global_range_extents[global_cur], 1)
        << "Shared tile and global tile shape mismatch: "
        << shared_range_extents << " vs " << global_range_extents;
  }

  // tile -> logical shared mode
  auto tile_to_shared_mode =
      cute::Layout(tile_shape, std::move(tile_to_shared_mode_stride));
  // logical shared mode -> logical shared
  auto shared_mode_to_shared_logical =
      cute::MakeColumnMajorLayout(shared_shape);
  // tile -> logical shared mode -> logical shared
  auto tile_to_shared_logical =
      cute::Composition(shared_mode_to_shared_logical, tile_to_shared_mode);

  auto shared_coords =
      shared_range.Map([](const Range &r) { return cute::IntTuple(r->min); });
  auto shared_logical_offset =
      shared_mode_to_shared_logical(cute::IntTupleTuple(shared_coords));

  auto global_logical_coords =
      global_range.Map([](const Range &r) { return cute::IntTuple(r->min); });

  return {
      tile_shape,
      std::move(tile_to_shared_logical),
      cute::Layout(tile_shape, std::move(tile_to_global_mode_stride)),
      std::move(shared_logical_offset),
      cute::IntTupleTuple(std::move(global_logical_coords)),
  };
}

} // namespace

Stmt Copy::LowerBulk(const CopyNode &op, const LowerArgs &lower_args,
                     arith::Analyzer *analyzer, CopyInst copy_inst) {
  const Buffer &src = op.src;
  const Buffer &dst = op.dst;
  const Array<Range> &src_range = op.src_range;
  const Array<Range> &dst_range = op.dst_range;
  const Map<String, ObjectRef> &annotations = op.annotations;

  ICHECK(copy_inst == CopyInst::kBulkLoad || copy_inst == CopyInst::kBulkStore)
      << "Invalid copy inst " << static_cast<int>(copy_inst);
  bool is_load = copy_inst == CopyInst::kBulkLoad;
  Buffer global_tensor = is_load ? src : dst;
  Buffer shared_tensor = is_load ? dst : src;
  Array<Range> global_range = is_load ? src_range : dst_range;
  Array<Range> shared_range = is_load ? dst_range : src_range;

  auto fallback_to_normal = [&](const char *reason) -> Stmt {
    if (GetIsTmaCopy(op)) {
      LOG(FATAL) << "T.tma_copy() cannot fall back to normal copy in "
                 << "LowerBulk: " << reason << ", src=" << src->name
                 << ", dst=" << dst->name;
    }
    return LowerNormal(op, lower_args, analyzer);
  };

  if (lower_args.layout_map.count(global_tensor)) {
    DLOG(WARNING) << "TMA bulk copy cannot support a non-swizzled global "
                     "layout, fallback to normal copy.";
    return fallback_to_normal("non-swizzled global layout");
  }

  Array<PrimExpr> global_shape = global_tensor->shape;
  Array<PrimExpr> global_stride;
  if (!global_tensor->strides.empty()) {
    global_stride = global_tensor->strides;
  } else {
    global_stride = makeRowMajorStrides(global_shape);
  }

  // Check if tile shapes match, and compute the tile layouts.
  // E.g., tile_shape = (64,512)
  //       tile_to_shared_logical = (64,512):(1,64)
  //       tile_to_global_mode = (64,512):(1@1,1@2)
  auto [tile_shape, tile_to_shared_logical, tile_to_global_mode,
        shared_logical_offset, global_logical_coords] =
      ComputeBulkCopyTile(shared_tensor->shape, shared_range, global_range);

  TMADesc desc;
  ICHECK(
      IsValidTMADtypePair(is_load, global_tensor->dtype, shared_tensor->dtype))
      << "Copy between buffer " << global_tensor->name << " and "
      << shared_tensor->name << " with incompatible data type "
      << global_tensor->dtype << " and " << shared_tensor->dtype;

  desc.data_type =
      TensorMapDataTypeForTMA(global_tensor->dtype, shared_tensor->dtype);
  desc.global_addr = global_tensor->data;
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);

  Array<PrimExpr> shared_shape = shared_tensor->shape;
  // logical shared -> physical shared, in TileLang convention
  Layout shared_layout;
  if (lower_args.layout_map.count(shared_tensor)) {
    shared_layout = lower_args.layout_map.at(shared_tensor);
    ICHECK(lower_args.buffer_remap.count(shared_tensor))
        << "shared_tensor: " << shared_tensor->name
        << " not found in buffer_remap";
    shared_tensor = lower_args.buffer_remap.at(shared_tensor);
  } else {
    shared_layout = MakeLinearLayout(shared_shape);
  }

  // Convert the TileLang layout to a possibly swizzled CuTe ComposedLayout.
  // This computes the swizzle from an arbitrary TileLang layout.
  // E.g., composed = Sw<3,3,3> o 0 o (64,64,8):(64,1,4096)
  auto composed = cute::ComposedLayoutFromTileLang(shared_layout);
  if (!composed.defined()) {
    DLOG(WARNING) << "Shared layout for src: " << src->name
                  << ", dst: " << dst->name
                  << " is not a CuTe swizzle over an affine layout, fallback "
                     "to normal copy";
    return fallback_to_normal("undecodable shared swizzle layout");
  }
  // Recast element-space layout into byte-address space.
  // Because CuTe swizzle are based on byte addresses.
  int elem_bits = shared_tensor->dtype.bits();
  // E.g., composed_bytes = Sw<3,4,3> o 0 o (64,64,8):(128,2,8192)
  auto composed_bytes = composed.value().Recast(elem_bits, /*new_bits=*/8);
  const auto *sw = composed_bytes->swizzle.get();
  int b_bits = sw->b_bits, m_base = sw->m_base, s_shift = sw->s_shift;
  if (!sw->IsSwizzled()) {
    desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
  } else if (b_bits == 1 && m_base == 4 && s_shift == 3) {
    desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B);
  } else if (b_bits == 2 && m_base == 4 && s_shift == 3) {
    desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B);
  } else if (b_bits == 3 && m_base == 4 && s_shift == 3) {
    desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B);
  } else {
    DLOG(WARNING) << "Shared swizzle Sw<" << b_bits << "," << m_base << ","
                  << s_shift << "> for src: " << src->name
                  << ", dst: " << dst->name
                  << " is not a TMA swizzle atom, fallback to normal copy";
    return fallback_to_normal("non-TMA shared swizzle layout");
  }

  // The TMA unit applies the descriptor's swizzle pattern relative to the
  // shared-memory base address, so the base must sit on a swizzle-pattern
  // repeat boundary or the data lands with a shifted phase (silently wrong
  // results, no fault). Report the requirement implied by the chosen swizzle
  // mode so MergeSharedMemoryAllocations can align the buffer accordingly.
  RequireTMASmemAlignment(lower_args, shared_tensor, desc.swizzle);

  // logical shared -> physical shared (without swizzle)
  // E.g., smem_plain = (64,64,8):(64,1,4096)
  auto smem_plain = composed.value()->layout;

  // tile -> logical shared -> physical shared (without siwzzle)
  // E.g., tile_to_smem_plain = (64,64,8):(64,1,4096)
  auto tile_to_smem_plain =
      cute::Composition(smem_plain, tile_to_shared_logical);

  // physical shared (without swizzle) -> tile
  // E.g., smem_plain_to_tile = (64,64,8):(64,1,4096)
  auto smem_plain_to_tile = cute::RightInverse(tile_to_smem_plain);
  // right_inverse only inverts the bijective stride chain, so a gapped or
  // non-injective SMEM layout would silently cover fewer elements than the
  // tile.
  int64_t inv_size = cute::AsConst(cute::Size(smem_plain_to_tile));
  int64_t tile_size = cute::AsConst(cute::Size(tile_to_smem_plain));
  ICHECK_EQ(inv_size, tile_size)
      << "plain SMEM layout is not a bijection (gapped or non-injective): "
      << "RightInverse covers " << inv_size << " of " << tile_size
      << " elements; try to use annotate_layout to make your SMEM tile "
         "contiguous";

  // Each TMA box dim is at most 256.
  const int64_t max_box_dim = 256;

  // physical shared (without swizzle) -> tile -> logical global mode
  // E.g, physical_shared_to_global_mode = (64,64,8):(1@2,1@1,64@2)
  auto physical_shared_to_global_mode = cute::Coalesce(
      cute::Composition(tile_to_global_mode, smem_plain_to_tile), max_box_dim);

  // Truncate, because we do not want to start in the middle of a global mode.
  // E.g., smem_rank = 2
  int64_t smem_rank = 0;
  while (smem_rank < cute::Rank(physical_shared_to_global_mode)) {
    auto st =
        cute::BasisValue(physical_shared_to_global_mode->stride[smem_rank]);
    if (!cute::IsConst(st) || AsConst(st) != 1) {
      break;
    }
    smem_rank++;
  }

  if (smem_rank == 0) {
    DLOG(WARNING) << "TMA bulk copy requires a contiguous innermost stride for "
                     "SMEM, fallback to normal copy.";
    return fallback_to_normal("non-contiguous SMEM innermost stride");
  }

  // physical shared (without swizzle) -> logical global mode (no > 1 stride)
  // Ideally the shape of this is the exact box_dim. But we have
  // several hardware constraints.
  // E.g., tile_gbasis = (64, 64):(1@2, 1@1)
  auto tile_gbasis = cute::Take(physical_shared_to_global_mode, smem_rank);

  // logical global mode -> TMA mode
  Array<cute::IntTuple> gmode_to_tma_mode_shape(global_shape.size(),
                                                cute::IntTuple(int64_t(1)));
  Array<cute::IntTuple> gmode_to_tma_mode_stride(global_shape.size(),
                                                 cute::E({}));
  // TMA mode -> logical global mode
  std::vector<int64_t> tma_mode_to_gmode(global_shape.size());
  // Some of the modes are not included in tile_gbasis and we need to track
  // the modes, and later add the missed modes.
  std::vector<bool> visited_global_modes(global_shape.size(), false);
  // We have some hardware restrictions on tile_gbasis. But we can shrink it to
  // meet some requirements if we need to. This is the modified tile_gbasis.
  Array<cute::IntTuple> box_shape, box_stride;
  for (int64_t i = 0; i < smem_rank; i++) {
    int64_t cap = max_box_dim;
    bool is_swizzle_inner = (i == 0 && sw->IsSwizzled());
    if (is_swizzle_inner) {
      int64_t span_bits = sw->Granularity() * 8;
      cap = std::max<int64_t>(1, span_bits / elem_bits);
    }
    int64_t box_dim = cute::AsConst(tile_gbasis->shape[i]);
    if (box_dim > cap) {
      // This exceeds the hardware constraint. But we can make it work by
      // spliting a mode.
      // Only the slowest box mode has leftover the rest loop can absorb;
      // shrinking a faster mode would gap the contiguous shared prefix.
      int64_t inner = -1;
      if (i == smem_rank - 1) {
        for (int64_t d = cap; d > 1; --d) {
          if (box_dim % d == 0) {
            inner = d;
            break;
          }
        }
      }
      if (inner != -1) {
        box_dim = inner;
      }
    }
    if (is_swizzle_inner) {
      ICHECK_EQ(box_dim, cap)
          << "The innermost box dim of a BulkCopy is not the swizzle "
             "granularity: "
          << "shared_layout = " << shared_layout << ", "
          << "tile_to_smem_plain = " << tile_to_smem_plain << ", "
          << "physical_shared_to_global_mode = "
          << physical_shared_to_global_mode << ", "
          << "tile_gbasis = " << tile_gbasis << ". "
          << "Currently the automatically generated swizzling do not violate "
             "this constraint. If you are annotating the layout, please make "
             "sure the contiguous SMEM tile mode has size of your swizzle "
             "granularity.";
    } else {
      if (box_dim > cap) {
        DLOG(WARNING)
            << "TMA box dim " << box_dim << " (mode " << i
            << ") exceeds the cap " << cap
            << " and cannot be cleanly split, fallback to normal copy";
        return fallback_to_normal("box dim exceeds cap");
      }
    }
    // The innermost box dim must be a whole 16-byte multiple: TMA transfers the
    // innermost box as 16-byte vectors.
    if (i == 0 &&
        TMABytesFromElements(box_dim, shared_tensor->dtype) % 16 != 0) {
      DLOG(WARNING) << "TMA innermost box dim " << box_dim << " (="
                    << TMABytesFromElements(box_dim, shared_tensor->dtype)
                    << " bytes) is not 16-byte aligned for src: " << src->name
                    << ", dst: " << dst->name << ", fallback to normal copy";
      return fallback_to_normal("inner box not 16-byte aligned");
    }
    box_shape.push_back(cute::IntTuple(box_dim));
    box_stride.push_back(tile_gbasis->stride[i]);

    // Collect the global mode information.
    auto basis = cute::BasisPath(tile_gbasis->stride[i]);
    ICHECK(basis.size() == 1);
    auto mode = basis[0];
    ICHECK(!visited_global_modes[mode]);
    visited_global_modes[mode] = true;
    gmode_to_tma_mode_stride.Set(mode, cute::E({i}));
    tma_mode_to_gmode[i] = mode;
  }

  // Adopt the validated/shrunk box dims (a no-op when nothing was shrunk).
  tile_gbasis = cute::Layout(cute::IntTupleTuple(std::move(box_shape)),
                             cute::IntTupleTuple(std::move(box_stride)));

  // The TMA rank with all the modes.
  // E.g., tma_rank = 3
  int64_t tma_rank = smem_rank;
  // Basically tile_gbasis, but with all the global modes.
  auto tma_gbasis_shape = cute::TupleFields(cute::Wrap(tile_gbasis->shape));
  auto tma_gbasis_stride = cute::TupleFields(cute::Wrap(tile_gbasis->stride));
  for (int64_t i = static_cast<int64_t>(global_shape.size()) - 1; i >= 0; i--) {
    if (visited_global_modes[i])
      continue;
    gmode_to_tma_mode_stride.Set(i, cute::E({tma_rank}));
    tma_mode_to_gmode[tma_rank] = i;
    tma_gbasis_shape.push_back(1);
    tma_gbasis_stride.push_back(cute::E({i}));
    tma_rank++;
  }

  ICHECK_EQ(tma_rank, global_shape.size());
  desc.rank = tma_rank;
  ICHECK(desc.rank >= 1 && desc.rank <= 5) << desc.rank;

  // logical global mode -> global mode in TMA's view
  // E.g., (1,1,1):(1@2,1@1,1@0)
  auto gmode_to_tma_mode = cute::Layout(std::move(gmode_to_tma_mode_shape),
                                        std::move(gmode_to_tma_mode_stride));

  // physical shared (without swizzle) -> logical global mode, all global modes
  // E.g., tma_gbasis = (64,64.1):(1@2,1@1,1@0)
  auto tma_gbasis =
      cute::Layout(std::move(tma_gbasis_shape), std::move(tma_gbasis_stride));
  ICHECK_EQ(cute::Rank(tma_gbasis), tma_rank);

  // The size in the box.
  // E.g., box_size = 4096
  const int64_t box_size =
      cute::AsConst(cute::Size(tile_gbasis)); // also tma_gbasis
  // The size out of the box.
  // E.g., rest_size = 8
  const int64_t rest_size =
      cute::AsConst(cute::Size(tile_to_shared_logical)) / box_size;

  // Rest index -> physical shared
  // E.g., 8:4096
  auto rest_to_smem = cute::Coalesce(cute::Layout(rest_size, box_size));
  // Rest index -> physical shared -> logical global mode
  // E.g., 8:64@2
  auto rest_to_gmode =
      cute::Composition(physical_shared_to_global_mode, rest_to_smem);
  // Rest index -> logical global mode -> global mode in TMA's view
  // E.g., 8:64@0
  auto rest_to_tma_mode =
      cute::Coalesce(cute::Composition(gmode_to_tma_mode, rest_to_gmode));

  desc.global_shape = Array<PrimExpr>();
  desc.global_stride = Array<PrimExpr>();
  desc.smem_box = Array<PrimExpr>();
  desc.smem_stride = Array<PrimExpr>(tma_rank, PrimExpr(1));

  for (int64_t i = 0; i < tma_rank; i++) {
    desc.smem_box.push_back(
        cute::AsConstOrPrimExpr(tma_gbasis->shape[i], DataType::Int(32)));

    desc.global_shape.push_back(global_shape[tma_mode_to_gmode[i]]);

    PrimExpr elem_stride = global_stride[tma_mode_to_gmode[i]];
    if (i == 0 && !is_one(elem_stride)) {
      DLOG(WARNING)
          << "TMA innermost global stride " << elem_stride
          << " != 1 element for src: " << src->name << ", dst: " << dst->name
          << " (transposed/non-contiguous box), fallback to normal copy";
      return fallback_to_normal("non-contiguous innermost TMA box");
    }
    PrimExpr byte_stride =
        TMAGlobalBytesFromElements(elem_stride, global_tensor->dtype);
    if (i >= 1) {
      if (auto s = as_const_int(byte_stride)) {
        if (*s % 16 != 0 || *s >= (1LL << 40)) {
          DLOG(WARNING) << "TMA global stride " << byte_stride
                        << " unsupported for src: " << src->name
                        << ", dst: " << dst->name
                        << ", fallback to normal copy";
          return fallback_to_normal("unsupported global stride");
        }
      }
    }
    desc.global_stride.push_back(byte_stride);
  }

  Call create_descriptor =
      Call(DataType::Handle(), create_tma_descriptor(), desc.EncodeCallArgs());

  int64_t cluster_mask = GetClusterMask(op);
  bool use_multicast = is_load && (cluster_mask > 0);

  int barrier_base_id = -1;
  PrimExpr mbar_handle;
  bool is_cluster_barrier = false;
  if (is_load) {
    if (auto user_barrier = annotations.Get("barrier")) {
      mbar_handle = Downcast<PrimExpr>(user_barrier.value());
      barrier_base_id = 0;
      if (auto bl = mbar_handle.as<BufferLoadNode>()) {
        is_cluster_barrier = bl->buffer.scope() == "shared.cluster_barrier";
      }
    } else if (GetIsTmaCopy(op)) {
      LOG(FATAL) << "T.tma_copy() requires a barrier argument. "
                 << "Use T.tma_copy(src, dst, barrier=mbar[idx]).";
    } else if (lower_args.alloc_mbarrier) {
      barrier_base_id =
          lower_args.alloc_mbarrier(1, MakeCopyMBarrierName(op.src, op.dst));
      PrimExpr mbar_idx = IntImm(DataType::Int(32), barrier_base_id);
      mbar_handle = BufferLoad(lower_args.mbarrier_buffer->value(), {mbar_idx});
    }
  }

  Array<PrimExpr> args;
  args.reserve(desc.rank + 4);
  args.push_back(create_descriptor);
  if (is_load)
    args.push_back(barrier_base_id >= 0 ? mbar_handle : PrimExpr(0));
  auto tma_op = is_load ? tma_load() : tma_store();

  Stmt tma_copy;

  PrimExpr total_elements = IntImm(DataType::Int(32), box_size);

  auto build_multicast_args = [&](const Array<PrimExpr> &regular_args) {
    Array<PrimExpr> mc_args;
    mc_args.reserve(regular_args.size() + 1);
    mc_args.push_back(regular_args[0]); // descriptor
    mc_args.push_back(regular_args[1]); // mbarrier
    mc_args.push_back(regular_args[2]); // shared memory pointer
    mc_args.push_back(IntImm(DataType::Int(32), cluster_mask));
    for (size_t i = 3; i < regular_args.size(); ++i) {
      mc_args.push_back(regular_args[i]);
    }
    return mc_args;
  };

  // physical shared offset
  cute::IntTuple shared_offset = smem_plain(shared_logical_offset);
  auto make_shared_offset = [&](std::optional<PrimExpr> rest_idx) {
    cute::IntTuple off = shared_offset;
    if (rest_idx.has_value())
      off += rest_to_smem(*rest_idx);
    return cute::AsConstOrPrimExpr(off, DataType::Int(32));
  };

  // TMA coords
  cute::IntTuple tma_coords;
  {
    Array<cute::IntTuple> modes;
    modes.reserve(tma_rank);
    for (int64_t i = 0; i < tma_rank; i++)
      modes.push_back(global_logical_coords[tma_mode_to_gmode[i]]);
    tma_coords = cute::IntTupleTuple(std::move(modes));
  }
  auto make_tma_coords = [&](std::optional<PrimExpr> rest_idx) {
    cute::IntTuple coords = tma_coords;
    if (rest_idx.has_value())
      coords += rest_to_tma_mode(*rest_idx);
    Array<PrimExpr> out;
    out.reserve(tma_rank);
    for (int64_t i = 0; i < tma_rank; i++)
      out.push_back(cute::AsConstOrPrimExpr(coords[i], DataType::Int(32)));
    return out;
  };

  if (rest_size > 1) {
    Var loop_var("i", DataType::Int(32));
    int loop_extent = rest_size;

    PrimExpr shared_addr =
        shared_tensor.access_ptr(is_load ? 2 : 1, DataType::Handle(), 1,
                                 make_shared_offset(loop_var), total_elements);
    args.push_back(shared_addr);
    for (auto coord : make_tma_coords(loop_var))
      args.push_back(coord);
    int need_reduce = 0;
    if (!is_load)
      args.push_back(need_reduce);
    args.push_back(GetEvictionPolicy(op));
    Map<String, ObjectRef> ann_loop;
    if (is_cluster_barrier && TargetIsSm100(lower_args.target) && is_load) {
      ann_loop.Set("use_2cta", IntImm(DataType::Int(32), 1));
    }
    tma_copy = For(loop_var, 0, loop_extent, ForKind::kUnrolled,
                   Evaluate(Call(DataType::Handle(), tma_op, args, ann_loop)));

    if (use_multicast) {
      Array<PrimExpr> mc_args = build_multicast_args(args);
      Stmt multicast_copy = For(
          loop_var, 0, loop_extent, ForKind::kUnrolled,
          Evaluate(Call(DataType::Handle(), tma_load_multicast(), mc_args)));

      int min_cta_rank = MinRankInClusterMask(cluster_mask);
      PrimExpr block_rank =
          Call(DataType::Int(32), block_rank_in_cluster(), {});
      PrimExpr mask_imm = IntImm(DataType::Int(32), cluster_mask);
      PrimExpr not_in_mask = EQ(bitwise_and(right_shift(mask_imm, block_rank),
                                            IntImm(DataType::Int(32), 1)),
                                IntImm(DataType::Int(32), 0));
      Stmt regular_or_noop = IfThenElse(not_in_mask, tma_copy, std::nullopt);
      tma_copy =
          IfThenElse(EQ(block_rank, IntImm(DataType::Int(32), min_cta_rank)),
                     multicast_copy, regular_or_noop);
    }
  } else {
    PrimExpr shared_addr = shared_tensor.access_ptr(
        is_load ? 2 : 1, DataType::Handle(), 1,
        make_shared_offset(std::nullopt), total_elements);
    args.push_back(shared_addr);
    for (auto coord : make_tma_coords(std::nullopt))
      args.push_back(coord);
    int need_reduce = 0;
    if (!is_load)
      args.push_back(need_reduce);
    args.push_back(GetEvictionPolicy(op));
    Map<String, ObjectRef> ann;
    if (TargetIsSm100(lower_args.target) && is_load &&
        (annotations.find("use_2cta") != annotations.end() ||
         is_cluster_barrier)) {
      ann.Set("use_2cta", IntImm(DataType::Int(32), 1));
    }
    tma_copy = Evaluate(Call(DataType::Handle(), tma_op, args, ann));

    if (use_multicast) {
      Array<PrimExpr> mc_args = build_multicast_args(args);
      Stmt multicast_copy =
          Evaluate(Call(DataType::Handle(), tma_load_multicast(), mc_args));

      int min_cta_rank = MinRankInClusterMask(cluster_mask);
      PrimExpr block_rank =
          Call(DataType::Int(32), block_rank_in_cluster(), {});
      PrimExpr mask_imm = IntImm(DataType::Int(32), cluster_mask);
      PrimExpr not_in_mask = EQ(bitwise_and(right_shift(mask_imm, block_rank),
                                            IntImm(DataType::Int(32), 1)),
                                IntImm(DataType::Int(32), 0));
      Stmt regular_or_noop = IfThenElse(not_in_mask, tma_copy, std::nullopt);
      tma_copy =
          IfThenElse(EQ(block_rank, IntImm(DataType::Int(32), min_cta_rank)),
                     multicast_copy, regular_or_noop);
    }
  }

  if (!is_load) {
    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(tma_copy);
    seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_arrive(), {})));
    if (!GetIsTmaCopy(op)) {
      seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_wait(),
                                  {IntImm(DataType::Int(32), 0), Bool(true)})));
    }
    tma_copy = SeqStmt(std::move(seq));
  }

  if (is_load && barrier_base_id >= 0) {
    PrimExpr total_bytes;
    if (rest_size > 1) {
      int loop_extent = rest_size;
      total_bytes = TMATransactionBytesFromElements(
          total_elements * loop_extent, shared_tensor->dtype);
    } else {
      total_bytes =
          TMATransactionBytesFromElements(total_elements, shared_tensor->dtype);
    }

    Stmt barrier_before_tma_stmt;
    Optional<Stmt> barrier_after_tma_stmt = std::nullopt;
    if (GetIsTmaCopy(op)) {
      if (is_cluster_barrier) {
        PrimExpr cluster_total_bytes =
            total_bytes * IntImm(DataType::Int(32), lower_args.cluster_size);
        Stmt expect_stmt =
            Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                          {mbar_handle, cluster_total_bytes}));
        PrimExpr rank = Call(DataType::Int(32), block_rank_in_cluster(), {});
        barrier_before_tma_stmt =
            IfThenElse(EQ(rank, IntImm(DataType::Int(32), 0)), expect_stmt);
      } else {
        barrier_before_tma_stmt =
            Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                          {mbar_handle, total_bytes}));
      }
      if (auto emit_arrive_val = annotations.Get("emit_arrive")) {
        if (Downcast<IntImm>(emit_arrive_val.value())->value != 0) {
          barrier_after_tma_stmt =
              Evaluate(Call(DataType::Handle(), builtin::ptx_arrive_barrier(),
                            {mbar_handle}));
        }
      }
    } else {
      barrier_before_tma_stmt =
          Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                        {mbar_handle, total_bytes}));
      barrier_after_tma_stmt = Evaluate(Call(
          DataType::Handle(), builtin::ptx_arrive_barrier(), {mbar_handle}));
    }

    Array<Stmt> producer_seq{barrier_before_tma_stmt, tma_copy};
    if (barrier_after_tma_stmt.defined()) {
      producer_seq.push_back(barrier_after_tma_stmt.value());
    }

    Stmt producer = IfThenElse(
        MakeTmaLeaderCondition(GetLeaderScopeThreads(op, lower_args)),
        SeqStmt(producer_seq));

    if (GetIsTmaCopy(op)) {
      return producer;
    }

    Stmt wait_stmt = Evaluate(
        Call(DataType::Handle(), mbarrier_wait_parity(),
             {mbar_handle, GetCopyMbarPhaseExpr(annotations, lower_args)}));

    return SeqStmt({producer, wait_stmt});
  }

  tma_copy = IfThenElse(
      MakeTmaLeaderCondition(GetLeaderScopeThreads(op, lower_args)), tma_copy);

  return tma_copy;
}

namespace {

Array<PrimExpr> GetGather4Rows(const CopyNode &op) {
  if (auto val = op.annotations.Get("gather4_rows")) {
    return Downcast<Array<PrimExpr>>(val.value());
  }
  return {};
}

PrimExpr GetGather4Col(const CopyNode &op) {
  if (auto val = op.annotations.Get("gather4_col")) {
    return Downcast<PrimExpr>(val.value());
  }
  return PrimExpr();
}

} // namespace

Stmt Copy::LowerBulkGather4(const CopyNode &op, const LowerArgs &lower_args,
                            arith::Analyzer *analyzer, CopyInst copy_inst) {
  ICHECK(copy_inst == CopyInst::kBulkLoadGather4 ||
         copy_inst == CopyInst::kBulkStoreScatter4);
  bool is_load = copy_inst == CopyInst::kBulkLoadGather4;

  Buffer global_tensor = is_load ? op.src : op.dst;
  Buffer shared_tensor = is_load ? op.dst : op.src;
  Buffer shared_tensor_unmapped = shared_tensor;

  ICHECK_EQ(global_tensor->shape.size(), 2u);
  ICHECK_EQ(shared_tensor->shape.size(), 2u);
  auto shared_lead = as_const_int(shared_tensor->shape[0]);
  ICHECK(shared_lead != nullptr && *shared_lead == 4)
      << "tma_gather4/scatter4 shared tile leading dim must be 4, got "
      << shared_tensor->shape[0];
  ICHECK_EQ(global_tensor->dtype, shared_tensor->dtype);

  Array<PrimExpr> rows = GetGather4Rows(op);
  PrimExpr col = GetGather4Col(op);
  ICHECK_EQ(rows.size(), 4u);
  ICHECK(col.defined());

  TMADesc desc;
  desc.rank = 2;
  desc.data_type = to_CUtensorMapDataType(global_tensor->dtype);
  desc.global_addr = global_tensor->data;
  desc.global_shape = ReverseArray(global_tensor->shape);

  if (!global_tensor->strides.empty()) {
    desc.global_stride = ReverseArray(global_tensor->strides);
  } else {
    PrimExpr stride = 1;
    desc.global_stride.reserve(2);
    for (size_t i = 0; i < global_tensor->shape.size(); ++i) {
      desc.global_stride.push_back(stride);
      stride *= global_tensor->shape[global_tensor->shape.size() - 1 - i];
    }
  }
  ICHECK(is_one(desc.global_stride[0]))
      << "tma_gather4/scatter4 requires unit innermost global stride, got "
      << desc.global_stride;
  desc.global_stride = desc.global_stride.Map([&](PrimExpr e) {
    return TMAGlobalBytesFromElements(e, global_tensor->dtype);
  });
  for (size_t i = 1; i < desc.global_stride.size(); ++i) {
    if (auto stride = desc.global_stride[i].as<IntImmNode>()) {
      ICHECK(stride->value % 16 == 0 && stride->value < (1LL << 40))
          << "tma_gather4/scatter4 global stride[" << i
          << "] = " << stride->value
          << " bytes must be 16-byte aligned and < 2^40";
    }
  }

  // The descriptor's row box-dim must be 1, not 4. The four-row pack is
  // implicit in the cp.async.bulk.tensor.tile::gather4 PTX, which takes 4
  // row coordinates and materializes them into 4 logical rows of the shared
  // tile. Setting box[1]=4 here would describe a contiguous 4-row strip; the
  // gather4 unrolling would then read OOB → CUDA_ERROR_ILLEGAL_INSTRUCTION.
  PrimExpr K_box = shared_tensor->shape[1];
  if (auto k = as_const_int(K_box)) {
    int64_t k_bytes = TMABytesFromElements(*k, shared_tensor->dtype);
    ICHECK(k_bytes % 16 == 0)
        << "tma_gather4/scatter4 K_box * dtype.bytes() = " << k_bytes
        << " must be 16-byte aligned";
  }
  desc.smem_box = {K_box, IntImm(DataType::Int(32), 1)};
  desc.smem_stride = {IntImm(DataType::Int(32), 1),
                      IntImm(DataType::Int(32), 1)};
  desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  Layout shared_layout;
  if (lower_args.layout_map.count(shared_tensor)) {
    shared_layout = lower_args.layout_map.at(shared_tensor);
    ICHECK(lower_args.buffer_remap.count(shared_tensor));
    shared_tensor = lower_args.buffer_remap.at(shared_tensor);
  }
  desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
  if (shared_layout.defined() && shared_layout->InputDim() >= 2) {
    SwizzleMode mode = DetectSwizzleMode(shared_layout, shared_tensor_unmapped);
    if (mode == SwizzleMode::Swizzle32B()) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B);
    } else if (mode == SwizzleMode::Swizzle64B()) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B);
    } else if (mode == SwizzleMode::Swizzle128B()) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B);
    }
  }
  if (auto k = as_const_int(K_box)) {
    int64_t k_bytes = TMABytesFromElements(*k, shared_tensor->dtype);
    int max_bytes = 0;
    if (desc.swizzle == static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B))
      max_bytes = 32;
    else if (desc.swizzle == static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B))
      max_bytes = 64;
    else if (desc.swizzle == static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B))
      max_bytes = 128;
    if (max_bytes > 0) {
      ICHECK(k_bytes <= max_bytes)
          << "tma_gather4/scatter4 K_box * dtype.bytes() = " << k_bytes
          << " exceeds " << max_bytes << "B swizzle limit";
    }
  }
  RequireTMASmemAlignment(lower_args, shared_tensor, desc.swizzle);

  Call create_descriptor =
      Call(DataType::Handle(), create_tma_descriptor(), desc.EncodeCallArgs());

  PrimExpr total_elements = 4 * K_box;
  PrimExpr smem_addr =
      shared_tensor.access_ptr(is_load ? 2 : 1, DataType::Handle(), 1,
                               IntImm(DataType::Int(32), 0), total_elements);

  Array<PrimExpr> args;
  args.push_back(create_descriptor);
  if (is_load) {
    auto user_barrier = op.annotations.Get("barrier");
    ICHECK(user_barrier.has_value())
        << "tma_gather4 requires a 'barrier' annotation";
    args.push_back(Downcast<PrimExpr>(user_barrier.value()));
  }
  args.push_back(smem_addr);
  args.push_back(col);
  for (auto r : rows)
    args.push_back(r);
  args.push_back(IntImm(DataType::Int(32), GetEvictionPolicy(op)));

  // Fire-and-forget: caller manages mbarrier_expect_tx / wait (loads) and
  // tma_store_arrive / wait (stores), and the leader-thread guard.
  auto tl_op = is_load ? tma_load_gather4() : tma_store_scatter4();
  return Evaluate(Call(DataType::Handle(), tl_op, args));
}

Stmt Copy::LowerBulk1D(const CopyNode &op, const LowerArgs &lower_args,
                       arith::Analyzer *analyzer, CopyInst copy_inst) {
  const Buffer &src = op.src;
  const Buffer &dst = op.dst;
  const Array<Range> &src_range = op.src_range;
  const Array<Range> &dst_range = op.dst_range;
  const Map<String, ObjectRef> &annotations = op.annotations;

  ICHECK(copy_inst == CopyInst::kBulkLoad1D ||
         copy_inst == CopyInst::kBulkStore1D);

  int64_t cluster_mask = GetClusterMask(op);
  ICHECK(cluster_mask == 0)
      << "cluster_mask=0x" << std::hex << cluster_mask
      << " requires descriptor-based TMA (kBulkLoad); the 1D bulk-copy path "
         "does not support multicast. src="
      << src->name << " (scope=" << src.scope() << "), dst=" << dst->name
      << " (scope=" << dst.scope() << ").";

  bool is_load = copy_inst == CopyInst::kBulkLoad1D;
  auto shared_range = is_load ? dst_range : src_range;
  auto global_range = is_load ? src_range : dst_range;
  auto shared_tensor = is_load ? dst : src;
  auto global_tensor = is_load ? src : dst;

  PrimExpr shared_elements = 1;
  for (size_t i = 0; i < shared_range.size(); i++) {
    shared_elements *= shared_range[i]->extent;
  }

  std::vector<PrimExpr> shared_strides;
  PrimExpr shared_stride = 1;
  for (size_t i = 0; i < shared_tensor->shape.size(); i++) {
    auto s = shared_tensor->shape[shared_tensor->shape.size() - i - 1];
    shared_strides.insert(shared_strides.begin(), shared_stride);
    shared_stride *= s;
  }

  Array<PrimExpr> shared_indices;
  for (auto r : shared_range)
    shared_indices.push_back(r->min);

  Array<PrimExpr> global_indices;
  for (auto r : global_range) {
    global_indices.push_back(r->min);
  }
  std::vector<PrimExpr> global_strides;
  PrimExpr global_stride = 1;
  for (size_t i = 0; i < global_tensor->shape.size(); i++) {
    auto s = global_tensor->shape[global_tensor->shape.size() - i - 1];
    global_strides.insert(global_strides.begin(), global_stride);
    global_stride *= s;
  }

  PrimExpr global_offset = 0;
  for (size_t i = 0; i < global_indices.size(); i++) {
    global_offset += global_indices[i] * global_strides[i];
  }

  PrimExpr shared_offset = 0;
  for (size_t i = 0; i < shared_indices.size(); i++) {
    shared_offset += shared_indices[i] * shared_strides[i];
  }

  PrimExpr elements = analyzer->Simplify(shared_elements);
  PrimExpr shared_addr = shared_tensor.access_ptr(
      is_load ? 2 : 1, DataType::Handle(), 1, shared_offset, elements);
  PrimExpr global_addr = global_tensor.access_ptr(
      is_load ? 1 : 2, DataType::Handle(), 1, global_offset, elements);

  int barrier_base_id = -1;
  PrimExpr mbar_handle;
  if (is_load) {
    if (auto user_barrier = annotations.Get("barrier")) {
      mbar_handle = Downcast<PrimExpr>(user_barrier.value());
      barrier_base_id = 0;
    } else if (GetIsTmaCopy(op)) {
      LOG(FATAL) << "T.tma_copy() requires a barrier argument. "
                 << "Use T.tma_copy(src, dst, barrier=mbar[idx]).";
    } else if (lower_args.alloc_mbarrier) {
      barrier_base_id =
          lower_args.alloc_mbarrier(1, MakeCopyMBarrierName(op.src, op.dst));
      PrimExpr mbar_idx = IntImm(DataType::Int(32), barrier_base_id);
      mbar_handle = BufferLoad(lower_args.mbarrier_buffer->value(), {mbar_idx});
    }
  }

  Stmt tma_copy;
  PrimExpr total_bytes =
      TMATransactionBytesFromElements(elements, shared_tensor->dtype);
  if (is_load) {
    PrimExpr mbar_arg = barrier_base_id >= 0 ? mbar_handle : PrimExpr(0);
    tma_copy = Evaluate(Call(DataType::Handle(), tma_load(),
                             {shared_addr, global_addr, mbar_arg, total_bytes,
                              GetEvictionPolicy(op)}));
  } else {
    int need_reduce = 0;
    tma_copy = Evaluate(Call(DataType::Handle(), tma_store(),
                             {global_addr, shared_addr, total_bytes,
                              need_reduce, GetEvictionPolicy(op)}));
  }

  if (!is_load) {
    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(tma_copy);
    seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_arrive(), {})));
    if (!GetIsTmaCopy(op)) {
      seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_wait(),
                                  {IntImm(DataType::Int(32), 0), Bool(true)})));
    }
    tma_copy = SeqStmt(std::move(seq));
  }

  if (is_load && barrier_base_id >= 0) {
    Stmt barrier_before_tma_stmt;
    Optional<Stmt> barrier_after_tma_stmt = std::nullopt;
    if (GetIsTmaCopy(op)) {
      barrier_before_tma_stmt =
          Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                        {mbar_handle, total_bytes}));
    } else {
      barrier_before_tma_stmt =
          Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                        {mbar_handle, total_bytes}));
      barrier_after_tma_stmt = Evaluate(Call(
          DataType::Handle(), builtin::ptx_arrive_barrier(), {mbar_handle}));
    }

    Array<Stmt> producer_seq{barrier_before_tma_stmt, tma_copy};
    if (barrier_after_tma_stmt.defined()) {
      producer_seq.push_back(barrier_after_tma_stmt.value());
    }

    Stmt producer = IfThenElse(
        MakeTmaLeaderCondition(GetLeaderScopeThreads(op, lower_args)),
        SeqStmt(producer_seq));

    if (GetIsTmaCopy(op)) {
      return producer;
    }

    Stmt wait_stmt = Evaluate(
        Call(DataType::Handle(), mbarrier_wait_parity(),
             {mbar_handle, GetCopyMbarPhaseExpr(annotations, lower_args)}));

    return SeqStmt({producer, wait_stmt});
  }

  tma_copy = IfThenElse(
      MakeTmaLeaderCondition(GetLeaderScopeThreads(op, lower_args)), tma_copy);
  return tma_copy;
}

Stmt Im2Col::Lower(const Im2ColOpNode &op, const LowerArgs &lower_args,
                   arith::Analyzer *analyzer) {
  const BufferRegion &dst_region = op.dstRegion_;
  const Buffer &src = op.src_;
  const Buffer &dst = op.dst_;

  ICHECK(TargetIsHopper(lower_args.target));
  ICHECK(IsGlobalBuffer(src) && IsSharedBuffer(dst));
  ICHECK(src->shape.size() == 4);
  ICHECK(src->dtype == dst->dtype);

  size_t ndim = dst_region->region.size();
  ICHECK(ndim >= 2) << "im2col dstRegion must have at least 2 dims";
  Layout shared_layout;
  if (lower_args.layout_map.count(dst)) {
    shared_layout = lower_args.layout_map[dst];
  }

  TMAIm2ColDesc desc;
  desc.rank = src->shape.size();
  desc.data_type = to_CUtensorMapDataType(src->dtype);
  desc.global_addr = src->data;
  desc.global_shape = ReverseArray(src->shape);

  if (!src->strides.empty()) {
    desc.global_stride = ReverseArray(src->strides);
  } else {
    PrimExpr stride = 1;
    desc.global_stride.reserve(desc.rank);
    for (size_t i = 0; i < desc.rank; i++) {
      desc.global_stride.push_back(stride);
      stride *= desc.global_shape[i];
    }
  }
  ICHECK(is_one(desc.global_stride[0])) << desc.global_stride;
  desc.global_stride = desc.global_stride.Map(
      [&](PrimExpr e) { return TMAGlobalBytesFromElements(e, src->dtype); });
  desc.elem_stride = {1, op.stride_, op.stride_, 1};
  desc.lower_corner = {-op.padding_, -op.padding_};
  desc.upper_corner = {-op.padding_, -op.padding_};
  desc.smem_box_pixel =
      Downcast<IntImm>(dst_region->region[ndim - 2]->extent)->value;
  desc.smem_box_channel =
      Downcast<IntImm>(dst_region->region[ndim - 1]->extent)->value;
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);
  if (!shared_layout.defined()) {
    desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
  } else {
    ICHECK(shared_layout->InputDim() >= 2) << "Cannot detect TMA layout.";
    if (StructuralEqual()(shared_layout, MakeQuarterBankSwizzleLayout(dst))) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B);
    } else if (StructuralEqual()(shared_layout,
                                 MakeHalfBankSwizzleLayout(dst))) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B);
    } else if (StructuralEqual()(shared_layout,
                                 MakeFullBankSwizzleLayout(dst))) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B);
    } else {
      LOG(FATAL) << "Cannot detect TMA layout.";
    }
  }
  RequireTMASmemAlignment(
      lower_args,
      lower_args.buffer_remap.count(dst) ? lower_args.buffer_remap[dst] : dst,
      desc.swizzle);

  Call create_desc = Call(DataType::Handle(), create_tma_im2col_descriptor(),
                          desc.EncodeCallArgs());

  Array<PrimExpr> global_coords;
  Array<PrimExpr> image_offset;
  global_coords.reserve(desc.rank);

  ICHECK(analyzer->CanProveEqual(
      FloorMod(desc.global_shape[0], desc.smem_box_channel), 0))
      << "Currently can only support divisible channel case";

  global_coords.push_back(
      FloorMod(op.c_step_ * desc.smem_box_channel, desc.global_shape[0]));
  image_offset.push_back(op.dilation_ *
                         FloorMod(FloorDiv(op.c_step_ * desc.smem_box_channel,
                                           desc.global_shape[0]),
                                  op.kernel_));
  image_offset.push_back(op.dilation_ *
                         FloorDiv(op.c_step_ * desc.smem_box_channel,
                                  desc.global_shape[0] * op.kernel_));

  PrimExpr h_dim = FloorDiv(src->shape[1] + 2 * op.padding_ -
                                (op.kernel_ - 1) * op.dilation_ - 1,
                            op.stride_) +
                   1;
  PrimExpr w_dim = FloorDiv(src->shape[2] + 2 * op.padding_ -
                                (op.kernel_ - 1) * op.dilation_ - 1,
                            op.stride_) +
                   1;
  global_coords.push_back(
      op.stride_ * FloorMod(op.nhw_step_ * desc.smem_box_pixel, w_dim) -
      op.padding_);
  global_coords.push_back(
      op.stride_ *
          FloorMod(FloorDiv(op.nhw_step_ * desc.smem_box_pixel, w_dim), h_dim) -
      op.padding_);
  global_coords.push_back(
      FloorDiv(op.nhw_step_ * desc.smem_box_pixel, w_dim * h_dim));

  int barrier_base_id = -1;
  PrimExpr mbar_handle;
  if (auto user_barrier = op.annotations_.Get("barrier")) {
    mbar_handle = Downcast<PrimExpr>(user_barrier.value());
    barrier_base_id = 0;
  } else if (lower_args.alloc_mbarrier) {
    barrier_base_id =
        lower_args.alloc_mbarrier(1, MakeCopyMBarrierName(op.src_, op.dst_));
    PrimExpr mbar_idx = IntImm(DataType::Int(32), barrier_base_id);
    mbar_handle = BufferLoad(lower_args.mbarrier_buffer->value(), {mbar_idx});
  }

  Array<PrimExpr> args;
  args.reserve(desc.rank * 2 + 2);
  args.push_back(create_desc);
  args.push_back(barrier_base_id >= 0 ? mbar_handle : PrimExpr(0));
  Buffer dst_buffer =
      lower_args.buffer_remap.count(dst) ? lower_args.buffer_remap[dst] : dst;
  PrimExpr flat_offset = IntImm(DataType::Int(32), 0);
  {
    PrimExpr stride = IntImm(DataType::Int(32), 1);
    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
      flat_offset = flat_offset + dst_region->region[i]->min * stride;
      stride = stride * dst->shape[i];
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
  args.push_back(op.eviction_policy_);
  Stmt tma_copy_stmt =
      Evaluate(Call(DataType::Handle(), tma_load_im2col(), args));

  if (barrier_base_id >= 0) {
    bool ws_barrier = op.annotations_.Get("barrier").has_value();
    PrimExpr total_bytes = TMABytesFromElements(
        IntImm(DataType::Int(32), desc.smem_box_pixel * desc.smem_box_channel),
        dst->dtype);

    Stmt barrier_before_tma_stmt = Evaluate(Call(
        DataType::Handle(), mbarrier_expect_tx(), {mbar_handle, total_bytes}));

    if (ws_barrier) {
      Array<Stmt> producer_seq{barrier_before_tma_stmt, tma_copy_stmt};
      if (auto emit_arrive_val = op.annotations_.Get("emit_arrive")) {
        if (Downcast<IntImm>(emit_arrive_val.value())->value != 0) {
          producer_seq.push_back(
              Evaluate(Call(DataType::Handle(), builtin::ptx_arrive_barrier(),
                            {mbar_handle})));
        }
      }
      return IfThenElse(
          MakeTmaLeaderCondition(lower_args.thread_bounds->extent),
          SeqStmt(producer_seq));
    }

    Stmt barrier_after_tma_stmt = Evaluate(
        Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {mbar_handle}));

    Stmt producer =
        IfThenElse(MakeTmaLeaderCondition(lower_args.thread_bounds->extent),
                   SeqStmt({barrier_before_tma_stmt, tma_copy_stmt,
                            barrier_after_tma_stmt}));

    Stmt wait_stmt = Evaluate(
        Call(DataType::Handle(), mbarrier_wait_parity(),
             {mbar_handle, GetCopyMbarPhaseExpr(op.annotations_, lower_args)}));

    return SeqStmt({producer, wait_stmt});
  }

  return IfThenElse(MakeTmaLeaderCondition(lower_args.thread_bounds->extent),
                    tma_copy_stmt);
}

} // namespace cuda

namespace {

bool MatchCudaCopyTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaCopy() {
  RegisterCopyImpl(CopyImpl{
      "cuda.Copy",
      MatchCudaCopyTarget,
      100,
      cuda::Copy::InferLayout,
      cuda::Copy::Lower,
  });
  return true;
}

const bool cuda_copy_registered = RegisterCudaCopy();

bool RegisterCudaIm2Col() {
  RegisterIm2ColImpl(Im2ColImpl{
      "cuda.Im2Col",
      [](Target target) {
        return MatchCudaCopyTarget(target) && TargetIsHopper(target);
      },
      100,
      cuda::Im2Col::Lower,
  });
  return true;
}

const bool cuda_im2col_registered = RegisterCudaIm2Col();

} // namespace

} // namespace tl
} // namespace tvm
