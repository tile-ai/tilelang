/*!
 * \file tl/cuda/op/copy.cc
 * \brief CUDA implementation for tl.copy lowering.
 */

#include "op/copy.h"
#include "support/check.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>

#include "cuda/op/builtin.h"
#include "cuda/op/copy.h"
#include "cuda/op/tma_layout.h"
#include "cuda/target_utils.h"
#include "cuda/transform/ptx_async_copy_injector.h"
#include "layout/cute_layout.h"
#include "layout/tcgen05_layout.h"
#include "op/utils.h"
#include "span_utils.h"
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
#include <optional>
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

bool IsProvably16ByteMultiple(const PrimExpr &bytes,
                              arith::Analyzer *analyzer) {
  return analyzer->CanProveEqual(FloorMod(bytes, IntImm(bytes.dtype(), 16)),
                                 IntImm(bytes.dtype(), 0));
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

std::optional<std::pair<Array<Stmt>, PrimExpr>>
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
    PrimExpr total_bytes = TMABytesFromElements(total_elems, src->dtype);
    if (!IsProvably16ByteMultiple(total_bytes, analyzer)) {
      return std::nullopt;
    }
    PrimExpr size_bytes = cast(DataType::UInt(32), total_bytes);
    PrimExpr src_ptr = src.access_ptr(1, DataType::Handle(), 1,
                                      linear_off(src, src_ranges), total_elems);
    PrimExpr dst_ptr = dst.access_ptr(2, DataType::Handle(), 1,
                                      linear_off(dst, dst_ranges), total_elems);
    Stmt call =
        Evaluate(Call(DataType::Handle(), tma_store_cluster(),
                      {dst_ptr, src_ptr, dst_block, size_bytes, barrier_load}));
    return std::make_pair(Array<Stmt>{call},
                          PrimExpr(IntImm(DataType::Int(32), 1)));
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
      auto sub = MakeTMARows(src, new_src, dst, new_dst, dst_block,
                             barrier_load, analyzer);
      if (!sub.has_value()) {
        return std::nullopt;
      }
      auto &[stmts, cnt] = sub.value();
      for (const auto &s : stmts) {
        all_stmts.push_back(s);
      }
      total = total + cnt;
    }
    return std::make_pair(all_stmts, total);
  }

  Var k("k_tma_row", DataType::Int(32));
  Array<Range> body_src = src_ranges;
  Array<Range> body_dst = dst_ranges;
  body_src.Set(split_dim,
               Range::FromMinExtent(src_ranges[split_dim]->min + k, 1));
  body_dst.Set(split_dim,
               Range::FromMinExtent(dst_ranges[split_dim]->min + k, 1));
  auto body_result = MakeTMARows(src, body_src, dst, body_dst, dst_block,
                                 barrier_load, analyzer);
  if (!body_result.has_value()) {
    return std::nullopt;
  }
  auto &[body_stmts, body_cnt] = body_result.value();
  Stmt body = body_stmts.size() == 1 ? body_stmts[0]
                                     : static_cast<Stmt>(SeqStmt(body_stmts));
  Stmt for_loop =
      For(k, IntImm(DataType::Int(32), 0), extent, ForKind::kSerial, body);
  return std::make_pair(Array<Stmt>{for_loop}, extent * body_cnt);
}

// Recast a (datapath, value-column) tile to b32 columns (CuTe upcast on the
// column axis): b32 column = value column / values_per_b32, the sub-slot
// position dropping onto a stride-0 mode.
cute::Layout TmemTileToB32Columns(const cute::Layout &tile,
                                  int64_t values_per_b32) {
  cute::Layout value_to_b32 =
      cute::MakeLayout({cute::Layout(1, cute::E({0})),
                        cute::Layout(cute::IntTupleTuple({values_per_b32, 1}),
                                     cute::IntTupleTuple({0, cute::E({1})}))});
  return cute::Composition(value_to_b32, tile);
}

// Whether a 16-bit fragment stores one value per b32 word (the low half:
// the MMA accumulator format, PTX "Packing format for matrix D in Tensor
// Memory") and so needs the tcgen05 pack::16b/unpack::16b modifiers.  Its
// columns are then half-filled: size < occupied datapaths * column cosize.
// Decided on the WHOLE fragment (storage is a property of the buffer);
// counting only occupied datapaths keeps Layout F's datapath gaps from
// looking like holes.
bool TmemFragmentNeedsPack16b(const cute::Layout &fragment) {
  int64_t datapaths = cute::AsConst(
      cute::Size(cute::Filter(cute::Composition(kDpOnly, fragment))));
  int64_t columns =
      cute::AsConst(cute::Cosize(cute::Composition(kColOnly, fragment)));
  return cute::AsConst(cute::Size(fragment)) < datapaths * columns;
}

} // namespace

namespace cuda {

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
  static void CollectFragmentLayouts(const PrimExpr &expr,
                                     const Map<Var, PrimExpr> &bind_var_to_expr,
                                     const LayoutMap &existing_layouts,
                                     PrimExpr thread_extent,
                                     Range thread_bounds,
                                     Map<Buffer, Layout> &result_map);

  static CopyInst SelectInst(const CopyNode &op, Target target,
                             const LayoutMap &layout_map,
                             arith::Analyzer *analyzer);

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
  CopyInst copy_inst = SelectInst(op, layout_args.target,
                                  layout_args.layout_map, layout_args.analyzer);
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

// Infer the register Fragment of a TMEM<->fragment copy from the TMEM
// buffer's layout, so the copy lowers to tcgen05.ld/st (LowerTmem below).
//
// Running example (a split epilogue reading one accumulator in two halves,
// 128 threads):
//   C_tmem  = alloc_tmem((128, 128), f32), fragment (128,128):(1@0,1@1)
//   C_local = alloc_fragment((128, 128), f32)
//   T.copy(C_tmem[:, 0:64],   C_local[:, 0:64])    <- this op
//   T.copy(C_tmem[:, 64:128], C_local[:, 64:128])  <- later op, same result
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

    // tmem_frag: logical buffer coord -> physical TMEM (datapath@0,
    //   column@1).
    //   E.g. (128,128):(1@0,1@1).
    Optional<cute::Layout> tmem_frag =
        cute::LayoutFromTileLangHierarchical(tmem_layout);
    ICHECK(tmem_frag.defined())
        << "TMEM layout of " << tmem_buf->name
        << " is not decodable by the CuTe analyzer: " << tmem_layout;

    // cute::Restrict splits the fragment into Region origin + tile: the
    // (possibly dynamic) origin only moves the base address.
    // tile: region-local coord -> (datapath, column) step from the origin.
    //   E.g. origin = (0,0), tile = (128,64):(1@0,1@1); the second copy
    //   differs only in origin = (0,64).
    const Array<Range> &tmem_ranges =
        is_tmem_load ? op.src_range : op.dst_range;
    const Array<Range> &reg_ranges = is_tmem_load ? op.dst_range : op.src_range;
    cute::Layout tile = cute::Restrict(tmem_frag.value(), tmem_ranges).get<1>();

    // A pack::16b tile is planned in b32 columns (shapes untouched: b32
    // column j holds exactly value j); a packed-pair tile keeps value
    // columns and moves whole columns with the plain instruction.
    int64_t values_per_b32 = 32 / (tmem_buf->dtype.bits());
    bool pack16b_tile =
        values_per_b32 > 1 && TmemFragmentNeedsPack16b(tmem_frag.value());
    if (pack16b_tile)
      tile = TmemTileToB32Columns(tile, values_per_b32);

    // The register buffer is a grid of such slices (a split epilogue issues
    // one copy per slice), each appending its own registers.
    // E.g. nrep = (1, 2): two column halves.
    size_t rank = reg_ranges.size();
    Array<int64_t> nrep;
    int64_t total_reps = 1;
    for (size_t dim = 0; dim < rank; ++dim) {
      const auto *buf_ext = as_const_int(reg_buf->shape[dim]);
      const auto *reg_ext = as_const_int(reg_ranges[dim]->extent);
      ICHECK(buf_ext && reg_ext && *buf_ext % *reg_ext == 0);
      nrep.push_back(*buf_ext / *reg_ext);
      total_reps *= nrep.back();
    }

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
      // local_frag: region-local coord -> (thread@0, value@1).
      //   E.g. (128,64):(1@0,1@1) for 32dp32b, 128 threads.
      int64_t values_per_column = pack16b_tile ? 1 : values_per_b32;
      Tcgen05CopyPlan expanded =
          ExpandTcgen05Layout(GetTcgen05MetaLd32Dp32B(), tile,
                              num_useful_threads, values_per_column);
      // A Layout F tile occupies only the low 16 datapaths per
      // sub-partition, which no full-width atom can tile; fall back to the
      // narrowest half-subpartition atom.
      if (!expanded.defined())
        expanded = ExpandTcgen05Layout(GetTcgen05MetaLd16Dp64B(), tile,
                                       num_useful_threads, values_per_column);
      if (!expanded.defined()) {
        continue;
      }
      cute::Layout local_frag = expanded->fragment;
      int64_t vals_per_slice =
          cute::AsConst(cute::Size(local_frag)) / num_useful_threads;

      // Repeat the slice along the value vector to cover the whole register
      // buffer (row-major slice grid): each further slice appends one
      // slice's worth of registers per thread, so both halves of a split
      // copy infer the same full-buffer Fragment.  Unit-extent region dims
      // carry only their repetition mode.
      // rep_grid: per-dim slice index -> value@1.
      //   E.g. (1,2):(128@1,64@1) -- the second half's 64 registers per
      //   thread follow the first's.
      cute::Layout rep_grid = cute::Composition(
          cute::Layout(total_reps, vals_per_slice * cute::E({1})),
          cute::MakeRowMajorLayout(nrep));
      Array<cute::Layout> modes;
      int64_t tile_idx = 0;
      for (size_t dim = 0; dim < rank; ++dim) {
        if (is_one(reg_ranges[dim]->extent)) {
          modes.push_back(rep_grid[dim]);
        } else {
          modes.push_back(
              cute::MakeLayout({local_frag[tile_idx++], rep_grid[dim]}));
        }
      }
      ICHECK_EQ(tile_idx, cute::Rank(tile))
          << "TMEM copy tile rank does not match the register loop rank";

      // logical_coord2frag: full buffer coord -> (thread@0, value@1).
      Fragment logical_coord2frag = FragmentToTileLang(cute::MakeLayout(modes));
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
    if (is_store && shared_tensor->shape.size() >= 2) {
      // For BulkStore, infer a swizzled shared-memory layout when possible.
      // Rank-1 shared buffers cannot be swizzled (there is no stride dim to
      // index), so they take the linear-layout branch below. A rank-1 store
      // can reach this multi-dim arm when its region is not provably in bounds,
      // making the raw 1D TMA path unavailable. Without the rank guard,
      // reading shape[dim - 2] would fault (see issue #2529).
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
        result_map.Set(shared_tensor,
                       MakeTmaLinearLayout(shared_tensor->shape, shared_range));
      } else {
        result_map.Set(shared_tensor, ExpandLayoutToMatchBuffer(
                                          swizzle_layout_2d, shared_tensor));
      }
    } else {
      result_map.Set(shared_tensor,
                     MakeTmaLinearLayout(shared_tensor->shape, shared_range));
    }
  }

  return result_map;
}

CopyInst Copy::SelectInst(const CopyNode &op, Target target,
                          const LayoutMap &layout_map,
                          arith::Analyzer *analyzer) {
  CopyAnalysisContext ctx;
  ctx.target = target;
  ctx.layout_map = &layout_map;
  ctx.analyzer = analyzer;
  ctx.emit_diagnostics = true;
  auto result = SelectCopyInstForLowering(op, ctx);
  ICHECK(result.supported) << result.reason
                           << SpanHintSuffix({op.dst->span, op.src->span});
  return result.inst;
}

Stmt Copy::Lower(const CopyNode &op, const LowerArgs &lower_args,
                 arith::Analyzer *analyzer) {
  auto copy_inst =
      SelectInst(op, lower_args.target, lower_args.layout_map, analyzer);
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
                         lower_args.buffer_remap,
                         {}},
                        level);
  }
  auto loop_layout = par_op->GetLoopLayout();
  Stmt lowered_loop = LowerParallelLoop(
      par_op->GetRoot(), loop_layout, lower_args.thread_index, analyzer,
      lower_args.layout_map, par_op->GetPredicate(lower_args.thread_index),
      /*parallel_loop=*/true, /*should_vectorize=*/true,
      par_op->LoopLayoutRequiresPaddingGuard());

  auto inject_result =
      InjectPTXAsyncCopy(lowered_loop, /*async_without_async_commit_wait=*/
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
      PrimExpr total_bytes = TMABytesFromElements(total_elements, src->dtype);

      if (IsProvably16ByteMultiple(total_bytes, analyzer)) {
        PrimExpr size_bytes = cast(DataType::UInt(32), total_bytes);

        PrimExpr dst_ptr = dst.access_ptr(2, DataType::Handle(), 1, dst_offset,
                                          total_elements);
        PrimExpr src_ptr = src.access_ptr(1, DataType::Handle(), 1, src_offset,
                                          total_elements);

        Stmt bulk_copy = Evaluate(Call(DataType::Handle(), tma_store_cluster(),
                                       {dst_ptr, src_ptr, op.dst_block.value(),
                                        size_bytes, barrier_load}));

        return IfThenElse(
            EQ(lower_args.thread_index, lower_args.thread_bounds->min),
            bulk_copy);
      }
      LOG(WARNING) << "Cluster bulk copy size " << total_bytes
                   << " bytes is not a multiple of 16 as required by "
                      "cp.async.bulk; falling back to element-wise cluster "
                      "copy. src="
                   << src->name << ", dst=" << dst->name;
    } else {
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

        auto tma_rows =
            MakeTMARows(src, src_range, dst, dst_range, op.dst_block.value(),
                        barrier_load, analyzer);

        if (tma_rows.has_value()) {
          auto &[tma_stmts, n_rows] = tma_rows.value();

          if (lower_args.update_barrier_arrive) {
            lower_args.update_barrier_arrive(barrier_data_var, n_rows);
          }

          Stmt seq = (tma_stmts.size() == 1)
                         ? tma_stmts[0]
                         : static_cast<Stmt>(SeqStmt(tma_stmts));
          return IfThenElse(
              EQ(lower_args.thread_index, lower_args.thread_bounds->min), seq);
        }
        LOG(WARNING)
            << "Cluster bulk copy row size is not a multiple of 16 bytes as "
               "required by cp.async.bulk; falling back to element-wise "
               "cluster copy. src="
            << src->name << ", dst=" << dst->name;
      } else {
        LOG(WARNING)
            << "Falling back to element-wise cluster copy: bulk cluster paths "
               "require matching element counts and same per-dim extents "
               "between src and dst. src="
            << src->name << ", dst=" << dst->name;
      }
    }
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
                         lower_args.buffer_remap,
                         {}},
                        level);
  }
  auto loop_layout = par_op->GetLoopLayout();
  auto thread_loop = PartitionLoop(par_op->GetRoot(), lower_args.thread_index,
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
        EQ(lower_args.thread_index, lower_args.thread_bounds->min), arrive);
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
  // Normalize the logical thread index against the thread range once, and
  // build the copy indices from the normalized expression directly.
  PrimExpr norm_thread_index = lower_args.thread_index;
  if (lower_args.thread_bounds.defined()) {
    norm_thread_index = norm_thread_index - lower_args.thread_bounds->min;
  }
  PrimExpr warp = FloorDiv(norm_thread_index, 32) * 32;
  if (!is_transposed) {
    auto local_index = analyzer->Simplify(
        local_iter * elems_per_reg * num +
        elems_per_reg * FloorMod(FloorDiv(norm_thread_index, 8), num));
    auto thread_index =
        analyzer->Simplify(warp + FloorMod(norm_thread_index, 8) * 4);
    shared_coords = inv->Forward({local_index, thread_index});
  } else {
    auto local_index = analyzer->Simplify(
        local_iter * elems_per_reg * num +
        elems_per_reg * FloorMod(FloorDiv(norm_thread_index, 8), num) +
        FloorMod(norm_thread_index, elems_per_reg));
    auto thread_index = analyzer->Simplify(
        warp + FloorDiv(FloorMod(norm_thread_index, 8), elems_per_reg));
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
  return for_node;
}

namespace {

// Unpack a (datapath, column) coordinate IntTuple into two scalar
// expressions; adding the rank-2 zero first materializes an axis the layout
// never touches as a plain zero.
Array<PrimExpr> TmemCoordExprs(const cute::IntTuple &coord) {
  DataType dtype = DataType::Int(32);
  cute::IntTuple normalized = coord + cute::IntTupleTuple({0, 0});
  Array<cute::IntTuple> fields = cute::TupleFields(normalized);
  ICHECK_EQ(fields.size(), 2U)
      << "TMEM coordinate must be (datapath, column), got " << coord;
  return {cute::AsConstOrPrimExpr(fields[0], dtype),
          cute::AsConstOrPrimExpr(fields[1], dtype)};
}

} // namespace

// Lower a TMEM<->fragment copy to tcgen05.ld/st calls.
//
// Running example (the SECOND half of InferTMemLayout's split epilogue, so
// the Region origin is nonzero):
//   T.copy(C_tmem[:, 64:128], C_local[:, 64:128])
//
// The instruction pattern is static and validated against the register
// Fragment; the (possibly dynamic) Region origin only becomes the
// tcgen05_ld/st base-address token (datapath<<16 | b32column in
// LowerSharedTmem).
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
    LOG(FATAL) << "Unsupported tensor memory copy: " << "src scope = "
               << src.scope() << ", dst scope = " << dst.scope();
  }
  ICHECK(!is_cp)
      << "Copy from shared memory to tensor memory is not supported yet";

  Array<IterVar> loop_vars = op.MakeIterVars();
  ICHECK_GE(loop_vars.size(), 2U)
      << "Tensor memory copy needs at least the two matrix dimensions, got "
      << loop_vars.size();
  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);
  PrimExpr src_predicate = op.MakePredicate(analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate = op.MakePredicate(analyzer, loop_vars, dst->shape, 1);
  ICHECK(!src_predicate.defined() && !dst_predicate.defined())
      << "Tensor memory copy does not support predicates, got " << src_predicate
      << " and " << dst_predicate;
  for (const auto &iv : loop_vars) {
    ICHECK(is_const_int(iv->dom->min) && is_const_int(iv->dom->extent))
        << "Tensor memory copy requires loop bounds to be constant integers";
  }
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

  Array<PrimExpr> tmem_logical_indices = op.MakeIndices(loop_vars, tmem_side);
  Array<PrimExpr> reg_logical_indices =
      op.MakeIndices(loop_vars, 1 - tmem_side);

  // tmem_frag: logical buffer coord -> physical TMEM (datapath@0, column@1).
  Optional<cute::Layout> tmem_frag =
      cute::LayoutFromTileLangHierarchical(tmem_layout);
  ICHECK(tmem_frag.defined())
      << "TMEM layout of " << tmem_buf->name
      << " is not decodable by the CuTe analyzer: " << tmem_layout;

  // As in InferTMemLayout, split the fragment into Region origin + tile.
  // tile: region-local coord -> (datapath, column) step from the origin.
  //   E.g. phy_origin = (0,64), tile = (128,64):(1@0,1@1).
  const Array<Range> &tmem_ranges =
      tmem_side == 0 ? op.src_range : op.dst_range;
  auto restricted = cute::Restrict(tmem_frag.value(), tmem_ranges);
  Array<PrimExpr> phy_origin = TmemCoordExprs(restricted.get<0>());
  cute::Layout tile = restricted.get<1>();
  ICHECK_EQ(cute::Rank(tile), static_cast<int64_t>(loop_vars.size()))
      << "TMEM copy tile rank does not match the copy loop rank";

  // 16-bit values come in two layouts (TmemFragmentNeedsPack16b): pack::16b
  // plans on the b32-upcast tile (one register gathers two adjacent
  // columns' low halves); packed pairs plan in value columns (one register
  // per column).  Either way one register covers two values, so the
  // wrapper's N is half the planned chunk count.
  int64_t values_per_b32 = 32 / (tmem_buf->dtype.bits());
  bool pack16b =
      values_per_b32 > 1 && TmemFragmentNeedsPack16b(tmem_frag.value());
  if (pack16b) {
    tile = TmemTileToB32Columns(tile, values_per_b32);
    // The width-alignment checks below run in the tile's (b32) units; a
    // pack::16b origin is always storage-slot aligned.
    ICHECK(analyzer->CanProveEqual(
        FloorMod(phy_origin[1], IntImm(DataType::Int(32), values_per_b32)), 0))
        << "A pack::16b TMEM origin must start on a b32 column";
    phy_origin.Set(
        1, analyzer->Simplify(FloorDiv(
               phy_origin[1], IntImm(DataType::Int(32), values_per_b32))));
  }

  size_t tmem_rank = tmem_buf->shape.size();

  // The address token below is the Region's logical origin; LowerTileOp
  // materializes the buffer's layout over it.
  Map<Var, PrimExpr> loop_to_origin;
  for (const auto &iv : loop_vars) {
    loop_to_origin.Set(iv->var, iv->dom->min);
  }
  Array<PrimExpr> tmem_origin_indices =
      tmem_logical_indices.Map([&](const PrimExpr &index) {
        return analyzer->Simplify(Substitute(index, loop_to_origin));
      });
  if (needs_pack_unpack && !pack16b) {
    ICHECK(analyzer->CanProveEqual(
        FloorMod(tmem_origin_indices[tmem_rank - 1], 2), 0))
        << "A sliced packed 16-bit TMEM copy must start at an even logical "
           "column";
  }

  bool have_succeeded = false;
  Stmt body;

  Array<PrimExpr> loop_exprs;
  for (const auto &iv : loop_vars) {
    loop_exprs.push_back(iv->var);
  }

  auto try_tcgen05_instruction = [&](Tcgen05Meta meta) {
    if (have_succeeded) {
      return;
    }
    int width = static_cast<int>(Tcgen05AtomWidth(meta));
    if (!analyzer->CanProveEqual(phy_origin[0], 0) ||
        !analyzer->CanProveEqual(FloorMod(phy_origin[1], width), 0)) {
      return;
    }

    for (int num_useful_wgs = num_threads / WARPGROUP_SIZE; num_useful_wgs >= 1;
         num_useful_wgs--) {
      int num_useful_threads = num_useful_wgs * WARPGROUP_SIZE;
      // plan: region-local loop coord -> (thread@0, value@1).
      Tcgen05CopyPlan plan = ExpandTcgen05Layout(meta, tile, num_useful_threads,
                                                 pack16b ? 1 : values_per_b32);
      if (!plan.defined()) {
        continue;
      }
      int num_chunks_each_wg = static_cast<int>(plan->num_chunks_each_wg);

      // target_frag: loop coord -> (thread@0, value@1), the single
      //   conversion to a TileLang Fragment.
      //   E.g. (128,64):(1@0,1@1).
      Fragment target_frag = FragmentToTileLang(plan->fragment);

      // The pattern must agree with the register Fragment up to the
      // Region's (possibly dynamic) base register, which only offsets the
      // register pointer below.
      // E.g. reg_val == 64+j with reg_origin = 64: registers 64..127.
      PrimExpr target_thread =
          target_frag->ForwardThread(loop_exprs, std::nullopt);
      PrimExpr reg_thread =
          reg_layout->ForwardThread(reg_logical_indices, std::nullopt);
      if (!analyzer->CanProveEqual(target_thread, reg_thread)) {
        continue;
      }
      PrimExpr target_reg = target_frag->Forward(loop_exprs)[0];
      PrimExpr reg_val = reg_layout->Forward(reg_logical_indices)[0];
      PrimExpr reg_origin =
          analyzer->Simplify(Substitute(reg_val, loop_to_origin));
      if (!analyzer->CanProveEqual(target_reg, reg_val - reg_origin)) {
        continue;
      }

      bool use_pack_unpack_modifier = pack16b;
      // One register covers two 16-bit values, so N halves.
      int effective_chunks =
          needs_pack_unpack ? num_chunks_each_wg / 2 : num_chunks_each_wg;
      PrimExpr relative_wg_idx =
          FloorDiv(Sub(lower_args.thread_index, lower_args.thread_bounds->min),
                   WARPGROUP_SIZE);
      // Column offsets are raw b32 columns: pack::16b already counts b32
      // columns, a packed-pair count halves (two value columns per b32).
      int chunk_cols =
          pack16b ? num_chunks_each_wg * width : effective_chunks * width;
      PrimExpr col_offset = num_useful_threads == WARPGROUP_SIZE
                                ? PrimExpr(0)
                                : relative_wg_idx * chunk_cols;
      int64_t vals_per_issue = plan->vals_per_issue;
      have_succeeded = true;

      // One tcgen05 call per issue: issue r starts at the region-local
      // coords idx2crd(rest_domain(r)) and appends vals_per_issue registers
      // per thread.
      // E.g. the running example (one issue) emits
      //   tl::tcgen05_ld_32dp32bNx<64, false>(C_tmem[0, 64], 0,
      //                                       &C_local[64]);
      Array<Stmt> calls;
      Buffer orig_reg_buf = is_ld ? op.dst : op.src;
      for (int64_t issue = 0; issue < plan->num_issues; ++issue) {
        int64_t flat = cute::AsConst(plan->rest_domain(issue));
        Map<Var, PrimExpr> loop_to_issue;
        for (const auto &iv : loop_vars) {
          int64_t extent = *as_const_int(iv->dom->extent);
          loop_to_issue.Set(iv->var, iv->dom->min +
                                         IntImm(iv->var->dtype, flat % extent));
          flat /= extent;
        }
        Array<PrimExpr> issue_tmem_indices =
            tmem_logical_indices.Map([&](const PrimExpr &index) {
              return analyzer->Simplify(Substitute(index, loop_to_issue));
            });
        // The access_ptr offset is a LOGICAL linear offset (row-major over
        // the buffer shape); LowerTileOp later maps it through the Fragment
        // to the per-thread register index.
        PrimExpr issue_reg_offset = IntImm(DataType::Int(32), 0);
        for (size_t d = 0; d < reg_logical_indices.size(); ++d) {
          issue_reg_offset = issue_reg_offset * orig_reg_buf->shape[d] +
                             Substitute(reg_logical_indices[d], loop_to_issue);
        }
        issue_reg_offset = analyzer->Simplify(issue_reg_offset);
        Array<PrimExpr> args;
        args.push_back(IntImm(DataType::Int(32), width * 32));
        args.push_back(IntImm(DataType::Int(32), effective_chunks));
        args.push_back(Bool(use_pack_unpack_modifier));
        args.push_back(BufferLoad(tmem_buf, issue_tmem_indices));
        args.push_back(col_offset);
        args.push_back(IntImm(DataType::Int(32),
                              static_cast<int>(plan->datapaths_per_warp)));
        args.push_back(reg_buf.access_ptr(
            /*access_mask=*/is_ld ? 2 : 1, DataType::Handle(),
            /*content_lanes=*/1, /*offset=*/issue_reg_offset,
            PrimExpr(static_cast<int>(vals_per_issue))));
        calls.push_back(Evaluate(Call(
            DataType::Handle(), is_ld ? tcgen05_ld() : tcgen05_st(), args)));
      }
      Stmt call = calls.size() == 1 ? calls[0] : SeqStmt(calls);
      if (num_useful_threads != num_threads) {
        body =
            IfThenElse(lower_args.thread_index <
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
    try_tcgen05_instruction(GetTcgen05MetaLd16Dp64B());
    try_tcgen05_instruction(GetTcgen05MetaLd16Dp128B());
    try_tcgen05_instruction(GetTcgen05MetaLd16Dp256B());
  } else {
    try_tcgen05_instruction(GetTcgen05MetaSt32Dp32B());
    try_tcgen05_instruction(GetTcgen05MetaSt16Dp64B());
    try_tcgen05_instruction(GetTcgen05MetaSt16Dp128B());
    try_tcgen05_instruction(GetTcgen05MetaSt16Dp256B());
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

PrimExpr TMABulkCopyPlan::SharedOffset(std::optional<PrimExpr> rest_idx) const {
  cute::IntTuple off = shared_offset;
  if (rest_idx.has_value())
    off += rest_to_smem(*rest_idx);
  return cute::AsConstOrPrimExpr(off, DataType::Int(32));
}

Array<PrimExpr>
TMABulkCopyPlan::TmaCoords(std::optional<PrimExpr> rest_idx) const {
  cute::IntTuple coords = tma_coords;
  if (rest_idx.has_value())
    coords += rest_to_tma_mode(*rest_idx);
  Array<PrimExpr> out;
  out.reserve(desc.rank);
  for (int64_t i = 0; i < static_cast<int64_t>(desc.rank); i++)
    out.push_back(cute::AsConstOrPrimExpr(coords[i], DataType::Int(32)));
  return out;
}

Stmt TMABulkCopyPlan::EmitInstructions(
    const std::function<Stmt(std::optional<PrimExpr>)> &make_copy) const {
  if (rest_size > 1) {
    Var loop_var("i", DataType::Int(32));
    int loop_extent = rest_size;
    return For(loop_var, 0, loop_extent, ForKind::kUnrolled,
               make_copy(loop_var));
  }
  return make_copy(std::nullopt);
}

TMABulkCopyAnalysis
AnalyzeTMABulkCopy(const LowerArgs &lower_args, const Buffer &global_tensor,
                   Buffer shared_tensor, const Array<Range> &global_range,
                   const Array<Range> &shared_range,
                   const std::vector<int64_t> &box_dim_caps) {
  auto unsupported = [&](const std::string &reason) -> TMABulkCopyAnalysis {
    DLOG(WARNING) << "TMA bulk copy between " << global_tensor->name << " and "
                  << shared_tensor->name << " unsupported: " << reason;
    return {std::nullopt, reason};
  };

  if (lower_args.layout_map.count(global_tensor)) {
    return unsupported("global tensor " + std::string(global_tensor->name) +
                       " has a non-trivial layout");
  }

  Array<PrimExpr> global_shape = global_tensor->shape;
  ICHECK(box_dim_caps.empty() || box_dim_caps.size() == global_shape.size());
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

  TMABulkCopyPlan plan;
  TMADesc &desc = plan.desc;
  desc.global_addr = global_tensor->data;

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

  // Convert the TileLang layout to a TensorMap-compatible CuTe layout.
  TMASharedLayoutAnalysis layout_analysis =
      AnalyzeTMASharedLayout(shared_layout, shared_tensor->dtype);
  if (!layout_analysis.encoding.has_value()) {
    return unsupported("shared layout cannot be encoded for TMA: " +
                       layout_analysis.reason);
  }
  const TMASharedLayoutEncoding &layout_encoding =
      layout_analysis.encoding.value();
  // The box loop below may widen the mode; desc.swizzle and the alignment
  // requirement are recorded once the mode is final.
  SwizzleMode swizzle_mode = layout_encoding.swizzle_mode;
  int elem_bits = shared_tensor->dtype.bits();

  // logical shared -> physical shared (without swizzle)
  // E.g., smem_plain = (64,64,8):(64,1,4096)
  auto smem_plain = layout_encoding.composed->layout;

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

  // physical shared (without swizzle) -> tile -> logical global mode
  // E.g, physical_shared_to_global_mode = (64,64,8):(1@2,1@1,64@2)
  // Merged runs may not exceed the largest per-dim cap; the box loop below
  // enforces the exact cap of whichever dim a run belongs to.
  int64_t coalesce_cap = kTmaMaxBoxDim;
  for (int64_t c : box_dim_caps)
    coalesce_cap = std::max(coalesce_cap, c);
  auto physical_shared_to_global_mode = cute::Coalesce(
      cute::Composition(tile_to_global_mode, smem_plain_to_tile), coalesce_cap);

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
    return unsupported("non-contiguous SMEM innermost stride");
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
  int64_t box_elems = 1; // running product of the accepted box dims
  for (int64_t i = 0; i < smem_rank; i++) {
    // Collect the global mode information.
    auto basis = cute::BasisPath(tile_gbasis->stride[i]);
    ICHECK(basis.size() == 1);
    auto mode = basis[0];
    ICHECK(!visited_global_modes[mode]);

    int64_t box_dim = cute::AsConst(tile_gbasis->shape[i]);
    int64_t cap = box_dim_caps.empty() ? kTmaMaxBoxDim : box_dim_caps[mode];
    if (i == 0 && !swizzle_mode.IsNone()) {
      // A TensorMap requires the innermost box bytes to fit within the
      // swizzle span. Recovery reports the narrowest pattern observable in
      // the buffer; a wider mode describes the same layout iff its extra XOR
      // source bits lie beyond the buffer (then it also recovers the same
      // plain layout, so the pairing above stays valid). Widen one mode at a
      // time while the contiguous run wants more than the span; the recovery
      // proof gates each step. b_bits is recast-invariant, so the byte-space
      // width binds the element-space recovery directly.
      int64_t run_bytes =
          TMABytesFromElements(std::min(box_dim, cap), shared_tensor->dtype);
      while (swizzle_mode.ByteWidth() < 128 &&
             run_bytes > swizzle_mode.ByteWidth()) {
        int b_bits = swizzle_mode.BBits() + 1;
        if (!cute::ComposedLayoutFromTileLang(shared_layout, b_bits)
                 .defined()) {
          break;
        }
        swizzle_mode = SwizzleMode::FromBBits(b_bits);
      }
      int64_t span_elems =
          std::max<int64_t>(1, swizzle_mode.ByteWidth() * 8 / elem_bits);
      cap = std::min(cap, span_elems);
    }
    bool truncate_box = false;
    if (box_dim > cap) {
      // Cap the mode at its largest clean divisor. The leftover and every
      // slower mode fall to the rest loop: the capped mode becomes the box's
      // outermost, so the shared prefix stays contiguous. Rest instructions
      // step the shared tile by the box size, so the divisor must also keep
      // the box a whole kTmaSmemAddrAlign multiple.
      int64_t inner = -1;
      for (int64_t d = cap; d > 1; --d) {
        if (box_dim % d == 0 &&
            TMABytesFromElements(box_elems * d, shared_tensor->dtype) %
                    kTmaSmemAddrAlign ==
                0) {
          inner = d;
          break;
        }
      }
      if (inner == -1) {
        std::ostringstream oss;
        oss << "TMA box dim " << box_dim << " (mode " << i
            << ") exceeds the cap " << cap << " and cannot be cleanly split";
        return unsupported(oss.str());
      }
      box_dim = inner;
      truncate_box = true;
    }
    // The innermost box dim must be a whole 16-byte multiple: TMA transfers the
    // innermost box as 16-byte vectors.
    if (i == 0 &&
        TMABytesFromElements(box_dim, shared_tensor->dtype) % 16 != 0) {
      std::ostringstream oss;
      oss << "TMA innermost box dim " << box_dim
          << " (=" << TMABytesFromElements(box_dim, shared_tensor->dtype)
          << " bytes) is not a whole 16-byte multiple";
      return unsupported(oss.str());
    }
    box_shape.push_back(cute::IntTuple(box_dim));
    box_stride.push_back(tile_gbasis->stride[i]);
    visited_global_modes[mode] = true;
    gmode_to_tma_mode_stride.Set(mode, cute::E({i}));
    tma_mode_to_gmode[i] = mode;
    if (truncate_box)
      break;
  }

  desc.swizzle = swizzle_mode.CanonicalOrdinal();
  // The TMA unit applies the descriptor's swizzle pattern to absolute
  // shared-memory addresses, so the buffer base must sit on a swizzle-pattern
  // repeat boundary or the data lands with a shifted phase (silently wrong
  // results, no fault). Report the requirement implied by the chosen swizzle
  // mode so MergeSharedMemoryAllocations can align the buffer accordingly.
  // With an aligned base, per-instruction phases are exactly the ones the
  // recovered global swizzle encodes, so rest instructions may start at any
  // in-buffer position.
  RequireTMASmemAlignment(lower_args, shared_tensor, swizzle_mode);

  // Adopt the validated/shrunk box dims (a no-op when nothing was shrunk).
  smem_rank = static_cast<int64_t>(box_shape.size());
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
  plan.box_size = cute::AsConst(cute::Size(tile_gbasis)); // also tma_gbasis
  // The size out of the box.
  // E.g., rest_size = 8
  plan.rest_size =
      cute::AsConst(cute::Size(tile_to_shared_logical)) / plan.box_size;

  // Rest instructions step the shared tile by the box size; the divisor
  // search above keeps split boxes aligned, but a box formed from unpaired
  // slower modes may still violate the instruction address alignment.
  int64_t box_bytes = TMABytesFromElements(plan.box_size, shared_tensor->dtype);
  if (plan.rest_size > 1 && box_bytes % kTmaSmemAddrAlign != 0) {
    std::ostringstream oss;
    oss << "TMA rest instructions step the shared tile by " << box_bytes
        << " bytes, which is not a whole " << kTmaSmemAddrAlign
        << "-byte multiple";
    return unsupported(oss.str());
  }

  // Physical shared offset of the tile base.
  plan.shared_offset = smem_plain(shared_logical_offset);

  // Rest index -> physical shared
  // E.g., 8:4096
  plan.rest_to_smem =
      cute::Coalesce(cute::Layout(plan.rest_size, plan.box_size));
  // Rest index -> physical shared -> logical global mode
  // E.g., 8:64@2
  auto rest_to_gmode =
      cute::Composition(physical_shared_to_global_mode, plan.rest_to_smem);
  // Rest index -> logical global mode -> global mode in TMA's view
  // E.g., 8:64@0
  plan.rest_to_tma_mode =
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
      std::ostringstream oss;
      oss << "TMA innermost global stride " << elem_stride
          << " != 1 element (transposed/non-contiguous box)";
      return unsupported(oss.str());
    }
    PrimExpr byte_stride =
        TMAGlobalBytesFromElements(elem_stride, global_tensor->dtype);
    if (i >= 1) {
      if (auto s = as_const_int(byte_stride)) {
        if (*s % 16 != 0 || *s >= (1LL << 40)) {
          std::ostringstream oss;
          oss << "TMA global stride " << byte_stride
              << " must be a 16-byte multiple below 2^40";
          return unsupported(oss.str());
        }
      }
    }
    desc.global_stride.push_back(byte_stride);
  }

  // TMA coords
  {
    Array<cute::IntTuple> modes;
    modes.reserve(tma_rank);
    for (int64_t i = 0; i < tma_rank; i++)
      modes.push_back(global_logical_coords[tma_mode_to_gmode[i]]);
    plan.tma_coords = cute::IntTupleTuple(std::move(modes));
  }

  plan.shared_tensor = shared_tensor;
  return {std::move(plan), ""};
}

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

  auto fallback_to_normal = [&](const std::string &reason) -> Stmt {
    if (GetIsTmaCopy(op)) {
      LOG(FATAL) << "T.tma_copy() cannot fall back to normal copy in "
                 << "LowerBulk: " << reason << ", src=" << src->name
                 << ", dst=" << dst->name;
    }
    return LowerNormal(op, lower_args, analyzer);
  };

  ICHECK(
      IsValidTMADtypePair(is_load, global_tensor->dtype, shared_tensor->dtype))
      << "Copy between buffer " << global_tensor->name << " and "
      << shared_tensor->name << " with incompatible data type "
      << global_tensor->dtype << " and " << shared_tensor->dtype;

  TMABulkCopyAnalysis bulk_analysis = AnalyzeTMABulkCopy(
      lower_args, global_tensor, shared_tensor, global_range, shared_range);
  if (!bulk_analysis.plan.has_value()) {
    DLOG(WARNING) << "TMA bulk copy unsupported for src: " << src->name
                  << ", dst: " << dst->name << ": " << bulk_analysis.reason
                  << ", fallback to normal copy";
    return fallback_to_normal(bulk_analysis.reason);
  }
  const TMABulkCopyPlan &plan = bulk_analysis.plan.value();

  TMADesc desc = plan.desc;
  desc.data_type =
      TensorMapDataTypeForTMA(global_tensor->dtype, shared_tensor->dtype);
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);
  shared_tensor = plan.shared_tensor;

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
                 << "Use T.tma_copy(src, dst, barrier=mbar[idx])."
                 << SpanHintSuffix({op.dst->span, op.src->span});
    } else if (lower_args.alloc_mbarrier) {
      barrier_base_id =
          lower_args.alloc_mbarrier(1, MakeCopyMBarrierName(op.src, op.dst));
      PrimExpr mbar_idx = IntImm(DataType::Int(32), barrier_base_id);
      mbar_handle = BufferLoad(lower_args.mbarrier_buffer->value(), {mbar_idx});
    }
  }

  auto tma_op = is_load ? tma_load() : tma_store();
  PrimExpr total_elements = IntImm(DataType::Int(32), plan.box_size);

  auto make_args = [&](std::optional<PrimExpr> rest_idx) {
    Array<PrimExpr> args;
    args.reserve(desc.rank + 4);
    args.push_back(create_descriptor);
    if (is_load)
      args.push_back(barrier_base_id >= 0 ? mbar_handle : PrimExpr(0));
    args.push_back(shared_tensor.access_ptr(is_load ? 2 : 1, DataType::Handle(),
                                            1, plan.SharedOffset(rest_idx),
                                            total_elements));
    for (auto coord : plan.TmaCoords(rest_idx))
      args.push_back(coord);
    if (!is_load)
      args.push_back(0); // need_reduce
    args.push_back(GetEvictionPolicy(op));
    return args;
  };

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

  // The leading CTA in the mask issues the multicast; CTAs outside the mask
  // replay the regular copy.
  auto wrap_multicast = [&](Stmt regular, Stmt multicast) {
    int min_cta_rank = MinRankInClusterMask(cluster_mask);
    PrimExpr block_rank = Call(DataType::Int(32), block_rank_in_cluster(), {});
    PrimExpr mask_imm = IntImm(DataType::Int(32), cluster_mask);
    PrimExpr not_in_mask = EQ(bitwise_and(right_shift(mask_imm, block_rank),
                                          IntImm(DataType::Int(32), 1)),
                              IntImm(DataType::Int(32), 0));
    Stmt regular_or_noop = IfThenElse(not_in_mask, regular, std::nullopt);
    return IfThenElse(EQ(block_rank, IntImm(DataType::Int(32), min_cta_rank)),
                      multicast, regular_or_noop);
  };

  Stmt tma_copy;
  if (plan.rest_size > 1) {
    Var loop_var("i", DataType::Int(32));
    int loop_extent = plan.rest_size;
    Array<PrimExpr> args = make_args(loop_var);
    Map<String, ObjectRef> ann;
    if (is_cluster_barrier && TargetIsSm100(lower_args.target) && is_load) {
      ann.Set("use_2cta", IntImm(DataType::Int(32), 1));
    }
    tma_copy = For(loop_var, 0, loop_extent, ForKind::kUnrolled,
                   Evaluate(Call(DataType::Handle(), tma_op, args, ann)));
    if (use_multicast) {
      tma_copy = wrap_multicast(
          tma_copy, For(loop_var, 0, loop_extent, ForKind::kUnrolled,
                        Evaluate(Call(DataType::Handle(), tma_load_multicast(),
                                      build_multicast_args(args)))));
    }
  } else {
    Array<PrimExpr> args = make_args(std::nullopt);
    Map<String, ObjectRef> ann;
    if (TargetIsSm100(lower_args.target) && is_load &&
        (annotations.find("use_2cta") != annotations.end() ||
         is_cluster_barrier)) {
      ann.Set("use_2cta", IntImm(DataType::Int(32), 1));
    }
    tma_copy = Evaluate(Call(DataType::Handle(), tma_op, args, ann));
    if (use_multicast) {
      tma_copy = wrap_multicast(
          tma_copy, Evaluate(Call(DataType::Handle(), tma_load_multicast(),
                                  build_multicast_args(args))));
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
    if (plan.rest_size > 1) {
      int loop_extent = plan.rest_size;
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

  // Same decomposition as LowerBulk, over a virtual 4-row view of the global
  // tensor: the gathered rows act as global mode 0 whose coordinates are
  // given per instruction, and columns start at `col`.
  PrimExpr K = shared_tensor->shape[1];
  TMABulkCopyAnalysis bulk_analysis = AnalyzeTMABulkCopy(
      lower_args, global_tensor, shared_tensor,
      /*global_range=*/
      {Range::FromMinExtent(0, 4), Range::FromMinExtent(col, K)},
      /*shared_range=*/
      {Range::FromMinExtent(0, 4), Range::FromMinExtent(0, K)});
  ICHECK(bulk_analysis.plan.has_value())
      << "tma_gather4/scatter4 cannot lower the shared tile of "
      << shared_tensor->name << ": " << bulk_analysis.reason;
  const TMABulkCopyPlan &plan = bulk_analysis.plan.value();

  TMADesc desc = plan.desc;
  desc.data_type = to_CUtensorMapDataType(global_tensor->dtype);
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);

  auto row_box = as_const_int(desc.smem_box[1]);
  ICHECK(row_box != nullptr && *row_box == 4)
      << "tma_gather4/scatter4 requires the 4 gathered rows to sit right "
         "above the contiguous row chunk in shared memory, got box "
      << desc.smem_box;
  // The descriptor's row box-dim must be 1, not 4. The four-row pack is
  // implicit in the cp.async.bulk.tensor.tile::gather4 PTX, which takes 4
  // row coordinates and materializes them into 4 logical rows of the shared
  // tile. Setting box[1]=4 here would describe a contiguous 4-row strip; the
  // gather4 unrolling would then read OOB -> CUDA_ERROR_ILLEGAL_INSTRUCTION.
  desc.smem_box.Set(1, IntImm(DataType::Int(32), 1));

  Call create_descriptor =
      Call(DataType::Handle(), create_tma_descriptor(), desc.EncodeCallArgs());

  PrimExpr user_barrier;
  if (is_load) {
    auto barrier = op.annotations.Get("barrier");
    ICHECK(barrier.has_value())
        << "tma_gather4 requires a 'barrier' annotation";
    user_barrier = Downcast<PrimExpr>(barrier.value());
  }

  PrimExpr total_elements = IntImm(DataType::Int(32), plan.box_size);
  Buffer shared_physical = plan.shared_tensor;
  auto tl_op = is_load ? tma_load_gather4() : tma_store_scatter4();
  auto make_copy = [&](std::optional<PrimExpr> rest_idx) {
    Array<PrimExpr> args;
    args.push_back(create_descriptor);
    if (is_load)
      args.push_back(user_barrier);
    args.push_back(shared_physical.access_ptr(
        is_load ? 2 : 1, DataType::Handle(), 1, plan.SharedOffset(rest_idx),
        total_elements));
    args.push_back(plan.TmaCoords(rest_idx)[0]); // column coordinate
    for (auto r : rows)
      args.push_back(r);
    args.push_back(IntImm(DataType::Int(32), GetEvictionPolicy(op)));
    return Evaluate(Call(DataType::Handle(), tl_op, args));
  };

  // Fire-and-forget: caller manages mbarrier_expect_tx / wait (loads) and
  // tma_store_arrive / wait (stores), and the leader-thread guard.
  return plan.EmitInstructions(make_copy);
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
  if (global_tensor->strides.empty()) {
    PrimExpr global_stride = 1;
    for (size_t i = 0; i < global_tensor->shape.size(); i++) {
      auto s = global_tensor->shape[global_tensor->shape.size() - i - 1];
      global_strides.insert(global_strides.begin(), global_stride);
      global_stride *= s;
    }
  } else {
    global_strides.assign(global_tensor->strides.begin(),
                          global_tensor->strides.end());
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
                 << "Use T.tma_copy(src, dst, barrier=mbar[idx])."
                 << SpanHintSuffix({op.dst->span, op.src->span});
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

  // Safety net: instruction selection (CheckBulkCopy1D) must not route a
  // provably misaligned transfer here. Symbolic sizes cannot be proven
  // either way and are allowed through, matching the selection logic.
  ICHECK(!analyzer->CanProve(
      FloorMod(total_bytes, make_const(total_bytes.dtype(), 16)) !=
          make_const(total_bytes.dtype(), 0),
      arith::ProofStrength::kSymbolicBound))
      << "BulkCopy1D requires total_bytes to be 16-byte aligned, but got "
         "invalid size.\n"
      << "  Total_bytes = " << total_bytes << "\n";

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

  TMAIm2ColDesc desc;
  desc.rank = src->shape.size();
  desc.data_type = to_CUtensorMapDataType(src->dtype);
  desc.global_addr = src->data;
  desc.global_shape = ReverseArray(src->shape);
  desc.global_stride = ReverseArray(
      src->strides.empty() ? makeRowMajorStrides(src->shape) : src->strides);
  ICHECK(is_one(desc.global_stride[0])) << desc.global_stride;
  desc.global_stride = desc.global_stride.Map(
      [&](PrimExpr e) { return TMAGlobalBytesFromElements(e, src->dtype); });
  desc.elem_stride = {1, op.stride_, op.stride_, 1};
  desc.lower_corner = {-op.padding_, -op.padding_};
  desc.upper_corner = {-op.padding_, -op.padding_};
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);

  PrimExpr h_dim = FloorDiv(src->shape[1] + 2 * op.padding_ -
                                (op.kernel_ - 1) * op.dilation_ - 1,
                            op.stride_) +
                   1;
  PrimExpr w_dim = FloorDiv(src->shape[2] + 2 * op.padding_ -
                                (op.kernel_ - 1) * op.dilation_ - 1,
                            op.stride_) +
                   1;

  // Same decomposition as LowerBulk, over a virtual (pixels, channels) view
  // of the global tensor: the composed shared layout decides the box and the
  // per-instruction coordinate steps; only the descriptor encoding is
  // im2col-specific.
  PrimExpr channels = src->shape[3];
  PrimExpr box_pixel = dst_region->region[ndim - 2]->extent;
  PrimExpr box_channel = dst_region->region[ndim - 1]->extent;
  Buffer gmem_view(src->data, src->dtype,
                   {src->shape[0] * h_dim * w_dim, channels}, {channels, 1},
                   PrimExpr(), "im2col_gmem_view", src->data_alignment,
                   src->offset_factor, src->buffer_type);
  TMABulkCopyAnalysis bulk_analysis = AnalyzeTMABulkCopy(
      lower_args, gmem_view, dst,
      /*global_range=*/
      {Range::FromMinExtent(op.nhw_step_ * box_pixel, box_pixel),
       Range::FromMinExtent(op.c_step_ * box_channel, box_channel)},
      /*shared_range=*/dst_region->region,
      /*box_dim_caps=*/{kTmaMaxIm2ColPixelsPerColumn, kTmaMaxBoxDim});
  ICHECK(bulk_analysis.plan.has_value())
      << "im2col cannot lower the shared tile of " << dst->name << ": "
      << bulk_analysis.reason;
  const TMABulkCopyPlan &plan = bulk_analysis.plan.value();

  desc.smem_box_channel = Downcast<IntImm>(plan.desc.smem_box[0])->value;
  desc.smem_box_pixel = Downcast<IntImm>(plan.desc.smem_box[1])->value;
  desc.swizzle = plan.desc.swizzle;

  Call create_desc = Call(DataType::Handle(), create_tma_im2col_descriptor(),
                          desc.EncodeCallArgs());

  // Channel steps stay inside one box_channel window (box_channel divides the
  // channel count), so the filter-tap decomposition is per-copy constant.
  ICHECK(analyzer->CanProveEqual(FloorMod(channels, box_channel), 0))
      << "Currently can only support divisible channel case";

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

  PrimExpr tile_elems = IntImm(DataType::Int(32), plan.box_size);
  Buffer shared_physical = plan.shared_tensor;
  auto make_copy = [&](std::optional<PrimExpr> rest_idx) {
    Array<PrimExpr> coords = plan.TmaCoords(rest_idx);
    PrimExpr channel_pos = coords[0];
    PrimExpr pixel_pos = coords[1];

    Array<PrimExpr> args;
    args.reserve(desc.rank * 2 + 2);
    args.push_back(create_desc);
    args.push_back(barrier_base_id >= 0 ? mbar_handle : PrimExpr(0));
    args.push_back(shared_physical.access_ptr(
        /*access_mask=*/2, /*dtype=*/DataType::Handle(), /*content_lanes=*/1,
        /*offset=*/plan.SharedOffset(rest_idx), /*extent=*/tile_elems));
    args.push_back(FloorMod(channel_pos, channels));
    args.push_back(op.stride_ * FloorMod(pixel_pos, w_dim) - op.padding_);
    args.push_back(op.stride_ * FloorMod(FloorDiv(pixel_pos, w_dim), h_dim) -
                   op.padding_);
    args.push_back(FloorDiv(pixel_pos, w_dim * h_dim));
    args.push_back(op.dilation_ *
                   FloorMod(FloorDiv(channel_pos, channels), op.kernel_));
    args.push_back(op.dilation_ * FloorDiv(channel_pos, channels * op.kernel_));
    args.push_back(op.eviction_policy_);
    return Evaluate(Call(DataType::Handle(), tma_load_im2col(), args));
  };

  Stmt tma_copy_stmt = plan.EmitInstructions(make_copy);

  if (barrier_base_id >= 0) {
    bool ws_barrier = op.annotations_.Get("barrier").has_value();
    PrimExpr total_bytes = TMABytesFromElements(
        IntImm(DataType::Int(32), plan.box_size * plan.rest_size), dst->dtype);

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
