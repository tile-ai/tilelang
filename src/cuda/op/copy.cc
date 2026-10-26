#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "op/gpu.h"
#include "op/prim.h"
/*!
 * \file tl/cuda/op/copy.cc
 * \brief CUDA implementation for tl.copy lowering.
 */

#include "op/copy.h"
#include "compat/visit.h"
#include "cuda/stubs/cuda.h"
#include "support/check.h"
#include "transform/common/mbarrier.h"
#include <tvm/ffi/extra/structural_equal.h>
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

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/gpu.h>
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

Expr MakeFlatAccessPtr(const TensorVar &tensor, int mask,
                       PrimExpr offset = PrimExpr(0),
                       Optional<PrimExpr> elements = std::nullopt) {
  Array<PrimExpr> indices;
  PrimExpr remaining = offset;
  PrimExpr extent = tl::MakeConst(offset.ty(), 1);
  for (int axis = static_cast<int>(tensor.shape().size()) - 1; axis >= 0;
       --axis) {
    indices.insert(indices.begin(),
                   axis == 0 ? remaining
                             : floormod(remaining, tensor.shape()[axis]));
    remaining = floordiv(remaining, tensor.shape()[axis]);
    extent = extent * tensor.shape()[axis];
  }
  return Call(std::nullopt, access_ptr(),
              {TensorLoad(tensor, indices), elements.value_or(extent - offset),
               PrimExpr(mask)});
}

PrimExpr MakeTmaLeaderCondition(PrimExpr thread_extent) {
  return Call(PrimType::Bool(), tl_shuffle_elect(), {std::move(thread_extent)})
      .as_or_throw<PrimExpr>();
}

int TMAPayloadElementBits(PrimType dtype) {
  // IR elements are 8-bit unpacked SMEM slots, but TMA/mbarrier transactions
  // count the packed FP4 payload loaded from global memory.
  if (IsFP4UnpackedType(dtype)) {
    return 4;
  }
  return dtype.bits();
}

PrimExpr TMABytesFromElements(PrimExpr elements, int bits) {
  PrimExpr elements_i64 = Cast(PrimType::Int(64), elements);
  if (bits % 8 == 0) {
    return elements_i64 * IntImm(PrimType::Int(64), bits / 8);
  }
  return FloorDiv(elements_i64 * IntImm(PrimType::Int(64), bits) +
                      IntImm(PrimType::Int(64), 7),
                  IntImm(PrimType::Int(64), 8));
}

int64_t TMABytesFromElements(int64_t elements, int bits) {
  return (elements * bits + 7) / 8;
}

PrimExpr TMABytesFromElements(PrimExpr elements, PrimType dtype) {
  return TMABytesFromElements(elements, dtype.bits());
}

int64_t TMABytesFromElements(int64_t elements, PrimType dtype) {
  return TMABytesFromElements(elements, dtype.bits());
}

PrimExpr TMAGlobalBytesFromElements(PrimExpr elements, PrimType dtype) {
  return TMABytesFromElements(elements, TMAPayloadElementBits(dtype));
}

int64_t TMAGlobalBytesFromElements(int64_t elements, PrimType dtype) {
  return TMABytesFromElements(elements, TMAPayloadElementBits(dtype));
}

PrimExpr TMATransactionBytesFromElements(PrimExpr elements, PrimType dtype) {
  return TMABytesFromElements(elements, TMAPayloadElementBits(dtype));
}

int64_t TMATransactionBytesFromElements(int64_t elements, PrimType dtype) {
  return TMABytesFromElements(elements, TMAPayloadElementBits(dtype));
}

int64_t TMAElementsForBytes(int64_t bytes, PrimType dtype) {
  ICHECK_EQ((bytes * 8) % dtype.bits(), 0)
      << bytes << " bytes cannot be represented as whole elements of " << dtype;
  return bytes * 8 / dtype.bits();
}

bool IsProvably16ByteMultiple(const PrimExpr &bytes, sym::Analyzer analyzer) {
  return tl::arith::CanProveEqual(
      analyzer, FloorMod(bytes, IntImm(bytes.ty(), 16)), IntImm(bytes.ty(), 0));
}

PrimExpr GetCopyMbarPhaseExpr(const BlockAnnotations &annotations,
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

std::string MakeCopyMBarrierName(const TensorVar &src, const TensorVar &dst) {
  return SanitizeIdentifierPart(src->name) + "_to_" +
         SanitizeIdentifierPart(dst->name) + "_mbarrier";
}

bool GetBoolAnnotation(const CopyNode &op, const char *key) {
  if (auto val = op.annotations.Get(key)) {
    auto integer = GetIntAnnotation(val.value());
    TVM_FFI_CHECK(integer.has_value(), TypeError)
        << "Copy annotation `" << key << "` must be boolean";
    return *integer != 0;
  }
  return false;
}

bool GetDisableTMA(const CopyNode &op) {
  return GetBoolAnnotation(op, "disable_tma");
}

bool GetIsTmaCopy(const CopyNode &op) {
  return GetBoolAnnotation(op, "is_tma_copy");
}

int TensorMapDataTypeForTMA(PrimType global_dtype, PrimType shared_dtype) {
  // 16U4_ALIGN16B: f8f6f4 / mxf8f6f4 unpacked FP4 SMEM
  // (float_e2m1_unpacksmem_t). 16U4_ALIGN8B: packed FP4 for mxf4 / mxf4nvf4
  // (float_e2m1_t).
  constexpr int kTensorMapPrimType16U4Align16B = 14;
  if (IsFP4UnpackedType(shared_dtype)) {
    ICHECK(global_dtype.MatchesElementType(kDLFloat4_e2m1fn, 4))
        << "FP4 packed global tensor required for unpacked shared TMA copy";
    return kTensorMapPrimType16U4Align16B;
  }
  return to_CUtensorMapDataType(global_dtype);
}

int GetEvictionPolicy(const CopyNode &op) {
  if (auto val = op.annotations.Get("eviction_policy")) {
    auto policy = GetIntAnnotation(*val);
    TVM_FFI_CHECK(policy.has_value() && *policy >= 0 && *policy <= 2,
                  ValueError)
        << "Copy eviction_policy must be an integer in [0, 2]";
    return static_cast<int>(*policy);
  }
  return 0; // default: evict_normal
}

int64_t GetClusterMask(const CopyNode &op) {
  if (auto val = op.annotations.Get("cluster_mask")) {
    auto mask = GetIntAnnotation(*val);
    TVM_FFI_CHECK(mask.has_value() && *mask >= 0 && *mask <= 65535, ValueError)
        << "Copy cluster_mask must be an unsigned 16-bit integer";
    return *mask;
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
    if (val->as<TensorLoadNode>()) {
      return (val.value()).as_or_throw<PrimExpr>();
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
    auto int_val = GetIntAnnotation(*val);
    ICHECK(int_val) << "T.tma_copy leader_scope_threads annotation must be an "
                       "integer constant.";
    ICHECK_GT(*int_val, 0)
        << "T.tma_copy leader_scope_threads must be positive.";
    ICHECK_EQ(*int_val % 32, 0)
        << "T.tma_copy leader_scope_threads must be a multiple of warp size "
           "(32).";
    return IntImm(PrimType::Int(32), *int_val);
  }
  return lower_args.thread_bounds->extent;
}

bool IsContiguousRegion(const TensorVar &buf, const Array<Range> &ranges,
                        sym::Analyzer analyzer) {
  ICHECK_EQ(buf.shape().size(), ranges.size())
      << "IsContiguousRegion: buffer/range rank mismatch for " << buf->name;

  int n = static_cast<int>(ranges.size());
  int pivot = -1;
  for (int i = 0; i < n; ++i) {
    if (!tl::arith::CanProveEqual(analyzer, ranges[i]->extent, 1)) {
      pivot = i;
      break;
    }
  }
  if (pivot == -1) {
    return true;
  }

  for (int i = 0; i < pivot; ++i) {
    ICHECK(tl::arith::CanProveEqual(analyzer, ranges[i]->extent, 1))
        << "IsContiguousRegion: dim " << i << " precedes pivot " << pivot
        << " but has non-unit extent " << ranges[i]->extent << " for buffer "
        << buf->name;
  }

  for (int i = pivot + 1; i < n; ++i) {
    if (!tl::arith::CanProveEqual(analyzer, ranges[i]->begin, 0) ||
        !tl::arith::CanProveEqual(analyzer, ranges[i]->extent,
                                  buf.shape()[i])) {
      return false;
    }
  }
  return true;
}

std::optional<std::pair<Array<Stmt>, PrimExpr>>
MakeTMARows(const TensorVar &src, const Array<Range> &src_ranges,
            const TensorVar &dst, const Array<Range> &dst_ranges,
            PrimExpr dst_block, PrimExpr barrier_load, sym::Analyzer analyzer) {
  int n = static_cast<int>(src_ranges.size());

  auto linear_off = [](const TensorVar &buf,
                       const Array<Range> &ranges) -> PrimExpr {
    int r = static_cast<int>(ranges.size());
    PrimExpr off = 0, stride = 1;
    for (int i = r - 1; i >= 0; --i) {
      off = off + ranges[i]->begin * stride;
      if (i > 0) {
        stride = stride * buf.shape()[i];
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
    PrimExpr total_bytes = TMABytesFromElements(total_elems, src.dtype());
    if (!IsProvably16ByteMultiple(total_bytes, analyzer)) {
      return std::nullopt;
    }
    PrimExpr size_bytes = Cast(PrimType::UInt(32), total_bytes);
    Expr src_ptr =
        MakeFlatAccessPtr(src, 1, linear_off(src, src_ranges), total_elems);
    Expr dst_ptr =
        MakeFlatAccessPtr(dst, 2, linear_off(dst, dst_ranges), total_elems);
    Stmt call =
        Evaluate(Call(TupleType(Array<Type>{}), tma_store_cluster(),
                      {dst_ptr, src_ptr, dst_block, size_bytes, barrier_load}));
    return std::make_pair(Array<Stmt>{call},
                          PrimExpr(IntImm(PrimType::Int(32), 1)));
  }

  int split_dim = -1;
  for (int d = 0; d < n; ++d) {
    if (!tl::arith::CanProveEqual(analyzer, src_ranges[d]->extent, 1)) {
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
    PrimExpr total = IntImm(PrimType::Int(32), 0);
    for (int64_t k = 0; k < ext_imm->value; ++k) {
      Array<Range> new_src = src_ranges;
      Array<Range> new_dst = dst_ranges;
      PrimExpr kexpr = IntImm(PrimType::Int(32), k);
      new_src.Set(split_dim, Range::FromMinExtent(
                                 src_ranges[split_dim]->begin + kexpr, 1));
      new_dst.Set(split_dim, Range::FromMinExtent(
                                 dst_ranges[split_dim]->begin + kexpr, 1));
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

  PrimVar k("k_tma_row", PrimType::Int(32));
  Array<Range> body_src = src_ranges;
  Array<Range> body_dst = dst_ranges;
  body_src.Set(split_dim,
               Range::FromMinExtent(src_ranges[split_dim]->begin + k, 1));
  body_dst.Set(split_dim,
               Range::FromMinExtent(dst_ranges[split_dim]->begin + k, 1));
  auto body_result = MakeTMARows(src, body_src, dst, body_dst, dst_block,
                                 barrier_load, analyzer);
  if (!body_result.has_value()) {
    return std::nullopt;
  }
  auto &[body_stmts, body_cnt] = body_result.value();
  Stmt body = body_stmts.size() == 1 ? body_stmts[0]
                                     : static_cast<Stmt>(SeqStmt(body_stmts));
  Stmt for_loop = For(k, 0, extent, 1, ForKind::kDefault, SeqStmt(body));
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
  Expr global_addr{ffi::UnsafeInit{}};
  int smem_box_pixel;
  int smem_box_channel;
  int swizzle;
  int interleave;
  int oob_fill;
  int l2_promotion;

  Array<Expr> EncodeCallArgs() const {
    Array<Expr> args;
    args.reserve(rank * 5 + 5);

    args.push_back(PrimExpr(data_type));
    args.push_back(PrimExpr(static_cast<int>(rank)));
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
    args.push_back(PrimExpr(smem_box_pixel));
    args.push_back(PrimExpr(smem_box_channel));
    args.push_back(PrimExpr(interleave));
    args.push_back(PrimExpr(swizzle));
    args.push_back(PrimExpr(l2_promotion));
    args.push_back(PrimExpr(oob_fill));

    return args;
  }
};

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level);

  static Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
                    sym::Analyzer analyzer);

private:
  static void CollectFragmentLayouts(const Expr &expr,
                                     const Map<Var, Expr> &bind_var_to_expr,
                                     const LayoutMap &existing_layouts,
                                     PrimExpr thread_extent,
                                     Range thread_bounds,
                                     Map<TensorVar, Layout> &result_map);

  static CopyInst SelectInst(const CopyNode &op, Target target,
                             const LayoutMap &layout_map,
                             sym::Analyzer analyzer);

  static void CheckParallelLoopLayout(const CopyNode &op, CopyInst copy_inst);

  static LayoutMap InferTMemLayout(const CopyNode &op,
                                   const LayoutInferArgs &layout_args,
                                   CopyInst copy_inst);

  static LayoutMap InferBulkLayout(const CopyNode &op,
                                   const LayoutInferArgs &layout_args,
                                   InferLevel level, CopyInst copy_inst);

  static Stmt LowerNormal(const CopyNode &op, const LowerArgs &lower_args,
                          sym::Analyzer analyzer);

  static Stmt LowerCluster(const CopyNode &op, const LowerArgs &lower_args,
                           sym::Analyzer analyzer);

  static Stmt LowerCPAsync(const CopyNode &op, const LowerArgs &lower_args,
                           sym::Analyzer analyzer);

  static Stmt LowerLDSM(const CopyNode &op, const LowerArgs &lower_args,
                        sym::Analyzer analyzer, CopyInst copy_inst);

  static Stmt LowerTmem(const CopyNode &op, const LowerArgs &lower_args,
                        sym::Analyzer analyzer);

  static Stmt LowerBulk(const CopyNode &op, const LowerArgs &lower_args,
                        sym::Analyzer analyzer, CopyInst copy_inst);

  static Stmt LowerBulkGather4(const CopyNode &op, const LowerArgs &lower_args,
                               sym::Analyzer analyzer, CopyInst copy_inst);

  static Stmt LowerBulk1D(const CopyNode &op, const LowerArgs &lower_args,
                          sym::Analyzer analyzer, CopyInst copy_inst);
};

struct Im2Col {
  static Stmt Lower(const Im2ColOpNode &op, const LowerArgs &lower_args,
                    sym::Analyzer analyzer);
};

void Copy::CollectFragmentLayouts(const Expr &expr,
                                  const Map<Var, Expr> &bind_var_to_expr,
                                  const LayoutMap &existing_layouts,
                                  PrimExpr thread_extent, Range thread_bounds,
                                  Map<TensorVar, Layout> &result_map) {
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (auto bl = node.as<TensorLoadNode>()) {
      TensorVar tensor = bl->source.as_or_throw<TensorVar>();
      if (IsFragmentBuffer(tensor) && !existing_layouts.count(tensor) &&
          !result_map.count(tensor)) {
        auto fragment =
            Fragment::FullyReplicated(tensor.shape(), thread_extent);
        result_map.Set(tensor, fragment->BindThreadRange(thread_bounds));
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
  TensorVar tmem_buf = is_tmem_load ? op.src : op.dst;
  TensorVar reg_buf = is_tmem_load ? op.dst : op.src;

  if (!layout_args.layout_map.count(reg_buf) &&
      layout_args.layout_map.count(tmem_buf)) {
    Layout tmem_layout = layout_args.layout_map[tmem_buf];

    // tmem_frag: logical buffer coord -> physical TMEM (datapath@0,
    //   column@1).
    //   E.g. (128,128):(1@0,1@1).
    Optional<cute::Layout> tmem_frag =
        cute::LayoutFromTileLangHierarchical(tmem_layout);
    ICHECK(tmem_frag.has_value())
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
    int64_t values_per_b32 = 32 / (tmem_buf.dtype().bits());
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
      auto buf_ext = GetConstInt(reg_buf.shape()[dim]);
      auto reg_ext = GetConstInt(reg_ranges[dim]->extent);
      ICHECK(buf_ext && reg_ext && *buf_ext % *reg_ext == 0);
      nrep.push_back(*buf_ext / *reg_ext);
      total_reps *= nrep.back();
    }

    constexpr int WARP_SIZE = 32;
    constexpr int WARPGROUP_SIZE = 4 * WARP_SIZE;
    ICHECK(GetConstInt(layout_args.thread_bounds->extent))
        << "Tensor memory copy requires thread_bounds->extent (num_threads) "
           "to be constant integers";
    int num_threads = *GetConstInt(layout_args.thread_bounds->extent);
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
        if (IsOne(reg_ranges[dim]->extent)) {
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
  Map<TensorVar, Layout> result_map;

  bool is_tma_1d =
      copy_inst == CopyInst::kBulkLoad1D || copy_inst == CopyInst::kBulkStore1D;
  bool is_load =
      copy_inst == CopyInst::kBulkLoad || copy_inst == CopyInst::kBulkLoad1D;
  bool is_store =
      copy_inst == CopyInst::kBulkStore || copy_inst == CopyInst::kBulkStore1D;
  TensorVar shared_tensor = is_load ? op.dst : op.src;
  Array<Range> shared_range = is_load ? op.dst_range : op.src_range;

  if (is_tma_1d && shared_range.size() == 1) {
    // 1D TMA Store with single dimension can not be swizzled. 1D TMA can also
    // have multiple dimensions when the last dimension is continuous.
    return result_map;
  }

  // Fragment buffers used as TMA indices should be replicated on all threads.
  PrimExpr thread_extent = layout_args.thread_bounds->extent;
  for (const auto &range : op.src_range) {
    CollectFragmentLayouts(range->begin, layout_args.bind_var_to_expr,
                           layout_args.layout_map, thread_extent,
                           layout_args.thread_bounds, result_map);
    CollectFragmentLayouts(range->extent, layout_args.bind_var_to_expr,
                           layout_args.layout_map, thread_extent,
                           layout_args.thread_bounds, result_map);
  }
  for (const auto &range : op.dst_range) {
    CollectFragmentLayouts(range->begin, layout_args.bind_var_to_expr,
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
    if (is_store && shared_tensor.shape().size() >= 2) {
      // For BulkStore, infer a swizzled shared-memory layout when possible.
      // Rank-1 shared buffers cannot be swizzled (there is no stride dim to
      // index), so they take the linear-layout branch below. A rank-1 store
      // can reach this multi-dim arm when its region is not provably in bounds,
      // making the raw 1D TMA path unavailable. Without the rank guard,
      // reading shape[dim - 2] would fault (see issue #2529).
      int dim = shared_tensor.shape().size();
      const int64_t mat_stride = *GetConstInt(shared_tensor.shape()[dim - 2]);
      const int64_t mat_continuous =
          *GetConstInt(shared_tensor.shape()[dim - 1]);
      Layout swizzle_layout_2d =
          MakeGemmABLayoutHopper(mat_stride, mat_continuous, mat_continuous,
                                 shared_tensor.dtype().bits(),
                                 /*k_inner=*/true);
      if (StructuralEqual()(
              swizzle_layout_2d,
              MakeLinearLayout(swizzle_layout_2d->InputShape()))) {
        result_map.Set(shared_tensor,
                       MakeTmaLinearLayout(shared_tensor.shape(), shared_range,
                                           shared_tensor.dtype()));
      } else {
        result_map.Set(shared_tensor, ExpandLayoutToMatchBuffer(
                                          swizzle_layout_2d, shared_tensor));
      }
    } else {
      result_map.Set(shared_tensor,
                     MakeTmaLinearLayout(shared_tensor.shape(), shared_range,
                                         shared_tensor.dtype()));
    }
  }

  return result_map;
}

CopyInst Copy::SelectInst(const CopyNode &op, Target target,
                          const LayoutMap &layout_map, sym::Analyzer analyzer) {
  CopyAnalysisContext ctx(target);
  ctx.layout_map = &layout_map;
  ctx.analyzer = analyzer;
  ctx.emit_diagnostics = true;
  auto result = SelectCopyInstForLowering(op, ctx);
  ICHECK(result.supported) << result.reason
                           << SpanHintSuffix({op.dst->loc, op.src->loc});
  return result.inst;
}

Stmt Copy::Lower(const CopyNode &op, const LowerArgs &lower_args,
                 sym::Analyzer analyzer) {
  auto copy_inst =
      SelectInst(op, lower_args.target, lower_args.layout_map, analyzer);
  if (op.dst_block.has_value()) {
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
                        sym::Analyzer analyzer) {
  using namespace tvm::transform;

  PassContext pass_ctx = PassContext::Current();
  bool enable_async_copy =
      pass_ctx->GetConfig<bool>(kEnableAsyncCopy).value_or(true);
  bool no_implicit_commit_wait = GetNoImplicitAsyncCommitWait(op);
  bool explicit_async_semantics = no_implicit_commit_wait || GetIsAsyncCopy(op);
  if (!enable_async_copy && !explicit_async_semantics) {
    return LowerNormal(op, lower_args, analyzer);
  }

  auto simt_loop = op.MakeSIMTLoop(analyzer);
  auto fused_loop = (ParallelLoopFuser::Fuse(simt_loop)).as_or_throw<For>();
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
      /*parallel_loop=*/true, par_op->LoopLayoutRequiresPaddingGuard());

  auto inject_result = InjectPTXAsyncCopy(
      lowered_loop, /*async_without_async_commit_wait=*/
      no_implicit_commit_wait || GetIsAsyncCopy(op), analyzer);
  Stmt cp_async_loop = inject_result.stmt;
  if (!inject_result.injected_ptx_async_copy) {
    DLOG(WARNING) << "cp.async rewrite miss for copy src=" << op.src->name
                  << " (scope=" << op.src.scope().name()
                  << ", dtype=" << op.src.dtype() << "), dst=" << op.dst->name
                  << " (scope=" << op.dst.scope().name()
                  << ", dtype=" << op.dst.dtype()
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
                    "but no eligible global->shared store was rewritten.\n"
                 << lowered_loop;
    }
    DLOG(WARNING) << "Fallback to normal copy because cp.async rewrite found "
                     "no eligible global->shared store.";
    return LowerNormal(op, lower_args, analyzer);
  }
  if (no_implicit_commit_wait) {
    return cp_async_loop;
  }
  if (GetIsAsyncCopy(op)) {
    Stmt commit_group = Evaluate(
        Call(PrimType::Void(), Op::Get("tirx.ptx.cp_async_commit_group"),
             {StringImm("async"), StringImm("commit_group"), StringImm("")}));
    return SeqStmt({cp_async_loop, commit_group});
  }
  return cp_async_loop;
}

Stmt Copy::LowerNormal(const CopyNode &op, const LowerArgs &lower_args,
                       sym::Analyzer analyzer) {
  return tl::LowerNormalCopy(op, lower_args, analyzer);
}

Stmt Copy::LowerCluster(const CopyNode &op, const LowerArgs &lower_args,
                        sym::Analyzer analyzer) {
  const TensorVar &src = op.src;
  const TensorVar &dst = op.dst;
  const Array<Range> &src_range = op.src_range;
  const Array<Range> &dst_range = op.dst_range;
  ICHECK(op.dst_block.has_value());
  ICHECK(IsSharedBuffer(src));
  ICHECK(IsSharedBuffer(dst));

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
    bool element_match =
        tl::arith::CanProveEqual(analyzer, src_elements, dst_elements);

    if (src_contiguous && dst_contiguous && element_match) {
      PrimExpr barrier_load = barrier_opt.value();

      auto compute_linear_offset = [](const TensorVar &buf,
                                      const Array<Range> &ranges) -> PrimExpr {
        PrimExpr offset = 0;
        PrimExpr stride = 1;
        for (int i = static_cast<int>(ranges.size()) - 1; i >= 0; --i) {
          offset = offset + ranges[i]->begin * stride;
          if (i > 0) {
            stride = stride * buf.shape()[i];
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
      PrimExpr total_bytes = tl::arith::Simplify(
          analyzer, TMABytesFromElements(total_elements, src.dtype()));

      if (IsProvably16ByteMultiple(total_bytes, analyzer)) {
        PrimExpr size_bytes = Cast(PrimType::UInt(32), total_bytes);

        Expr dst_ptr = MakeFlatAccessPtr(dst, 2, dst_offset, total_elements);
        Expr src_ptr = MakeFlatAccessPtr(src, 1, src_offset, total_elements);

        Stmt bulk_copy =
            Evaluate(Call(TupleType(Array<Type>{}), tma_store_cluster(),
                          {dst_ptr, src_ptr, op.dst_block.value(), size_bytes,
                           barrier_load}));

        return If(EQ(lower_args.thread_index, lower_args.thread_bounds->begin),
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
        if (!tl::arith::CanProveEqual(analyzer, src_range[d]->extent,
                                      dst_range[d]->extent)) {
          same_shape = false;
        }
      }

      if (element_match && same_shape) {
        PrimExpr barrier_load = barrier_opt.value();
        const auto *barrier_buf_load = barrier_load.as<TensorLoadNode>();
        ICHECK(barrier_buf_load)
            << "LowerCluster: expected BufferLoad for barrier annotation";
        Var barrier_data_var =
            barrier_buf_load->source.as_or_throw<TensorVar>();

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
          return If(
              EQ(lower_args.thread_index, lower_args.thread_bounds->begin),
              seq);
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
  auto fused_loop = (ParallelLoopFuser::Fuse(simt_loop)).as_or_throw<For>();

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

  class ClusterCopyReplacer : public tvm::tl::StmtExprMutator {
  public:
    ClusterCopyReplacer(const TensorVar &dst, PrimExpr dst_block,
                        const TensorVar &target_dst,
                        Optional<Layout> dst_layout)
        : dst_(dst), dst_block_(dst_block), target_dst_(target_dst),
          dst_layout_(dst_layout) {}

    UnchangedOr<Stmt> Mutate_(const TensorStoreNode *op,
                              InplaceMode mode) final {
      if (op->dest.same_as(dst_)) {
        Array<PrimExpr> physical_indices = op->indices;
        if (!target_dst_.same_as(dst_) && dst_layout_.has_value()) {
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
              multiplier = multiplier * target_dst_.shape()[i];
            }
          }
        }

        TensorVar target_buffer = target_dst_;
        if (target_dst_.same_as(dst_)) {
          target_buffer = op->dest.as_or_throw<TensorVar>();
        }

        PrimExpr total_elems = 1;
        for (const PrimExpr &s : target_buffer.shape()) {
          total_elems = total_elems * s;
        }

        Stmt remote_store =
            Evaluate(Call(TupleType(Array<Type>{}), ptx_cluster_store(),
                          {MakeFlatAccessPtr(target_buffer, 2), op->value,
                           dst_block_, linearized_index}));

        return If(linearized_index < total_elems, remote_store, std::nullopt);
      }
      return tvm::tl::StmtExprMutator::Mutate_(op, mode);
    }

  private:
    const TensorVar &dst_;
    PrimExpr dst_block_;
    const TensorVar &target_dst_;
    Optional<Layout> dst_layout_;
  };

  TensorVar target_dst = dst;
  if (lower_args.buffer_remap.count(dst)) {
    target_dst = lower_args.buffer_remap[dst];
  }

  Optional<Layout> dst_layout = std::nullopt;
  if (lower_args.layout_map.count(dst)) {
    dst_layout = lower_args.layout_map[dst];
  }

  auto replacer = make_object<ClusterCopyReplacer>(dst, op.dst_block.value(),
                                                   target_dst, dst_layout);
  Stmt simt_copy = replacer->Mutate(vectorized_thread_loop)
                       .ValueOrUnchanged(vectorized_thread_loop);

  if (auto barrier_opt = GetBarrier(op)) {
    Stmt sync = Evaluate(
        Call(PrimType::Int(32), tl::StorageSyncOp(), {StringImm("shared")}));
    Stmt arrive =
        Evaluate(Call(TupleType(Array<Type>{}), ptx_arrive_cluster_barrier(),
                      {barrier_opt.value(), op.dst_block.value()}));
    Stmt guarded_arrive = If(
        EQ(lower_args.thread_index, lower_args.thread_bounds->begin), arrive);
    return SeqStmt({simt_copy, sync, guarded_arrive});
  }
  return simt_copy;
}

Stmt Copy::LowerLDSM(const CopyNode &op, const LowerArgs &lower_args,
                     sym::Analyzer analyzer, CopyInst copy_inst) {
  const TensorVar &src = op.src;
  const TensorVar &dst = op.dst;
  const Array<Range> &src_range = op.src_range;
  const Array<Range> &dst_range = op.dst_range;

  ICHECK(copy_inst == CopyInst::kLDSM || copy_inst == CopyInst::kSTSM)
      << "Invalid copy inst " << static_cast<int>(copy_inst);
  bool is_ldmatrix = copy_inst == CopyInst::kLDSM;

  Array<LayoutAxis> loop_vars = op.MakeIterVars();
  if (loop_vars.size() < 2) {
    return LowerNormal(op, lower_args, analyzer);
  }
  for (const auto &iv : loop_vars)
    tl::arith::Bind(analyzer, iv->var, iv->dom);
  auto src_predicate = op.MakePredicate(analyzer, loop_vars, src.shape(), 0);
  auto dst_predicate = op.MakePredicate(analyzer, loop_vars, dst.shape(), 1);
  if (src_predicate.has_value() || dst_predicate.has_value()) {
    return LowerNormal(op, lower_args, analyzer);
  }

  TensorVar shared_tensor = is_ldmatrix ? src : dst;
  TensorVar local_tensor = is_ldmatrix ? dst : src;
  const Array<Range> &shared_range = is_ldmatrix ? src_range : dst_range;
  const Array<Range> &fragment_range = is_ldmatrix ? dst_range : src_range;
  if (shared_range.size() != fragment_range.size()) {
    return LowerNormal(op, lower_args, analyzer);
  }
  // The inverse layout below covers the complete fragment. The shared side
  // may be a subregion, but its extents must match the fragment.
  for (size_t i = 0; i < fragment_range.size(); i++) {
    if (!IsZero(fragment_range[i]->begin) ||
        !tl::arith::CanProveEqual(analyzer, fragment_range[i]->extent,
                                  local_tensor.shape()[i]) ||
        !tl::arith::CanProveEqual(analyzer, shared_range[i]->extent,
                                  fragment_range[i]->extent)) {
      return LowerNormal(op, lower_args, analyzer);
    }
  }

  Array<PrimExpr> local_indices =
      op.MakeIndices(loop_vars, is_ldmatrix ? 1 : 0);
  Fragment local_layout =
      (lower_args.layout_map[local_tensor]).as_or_throw<Fragment>();
  Array<PrimExpr> local_indices_transformed =
      local_layout->Forward(local_indices);
  local_tensor = lower_args.buffer_remap[local_tensor];
  if (local_layout->OutputDim() != 1) {
    return LowerNormal(op, lower_args, analyzer);
  }

  Array<PrimExpr> shared_indices =
      op.MakeIndices(loop_vars, is_ldmatrix ? 0 : 1);
  bool is_transposed;
  LayoutAxis col_var = loop_vars[loop_vars.size() - 1];
  LayoutAxis row_var = loop_vars[loop_vars.size() - 2];
  PrimExpr local_layout_thread_map =
      FloorMod(local_layout->ForwardThread(local_indices, std::nullopt), 32);
  PrimExpr matrix_8x8_thread_map = MakeGemmFragment8x8()->ForwardThread(
      {FloorMod(row_var->var, 8), FloorMod(col_var->var, 8)}, std::nullopt);
  PrimExpr matrix_8x8_thread_map_trans =
      MakeGemmFragment8x8Transposed()->ForwardThread(
          {FloorMod(row_var->var, 8), FloorMod(col_var->var, 8)}, std::nullopt);
  PrimExpr local_indices_flattened =
      local_tensor.ty()->ElemOffset(local_indices_transformed).back();
  if (tl::arith::CanProveEqual(analyzer, matrix_8x8_thread_map,
                               local_layout_thread_map) &&
      IndicesCanVectorize(local_indices_flattened, col_var->var,
                          col_var->dom->extent, 2, analyzer)) {
    is_transposed = false;
  } else if (tl::arith::CanProveEqual(analyzer, matrix_8x8_thread_map_trans,
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
      shared_tensor.dtype().bits() == 8;
  const int shared_elem_bytes = use_m16n8_stmatrix ? 1 : 2;
  if (shared_tensor.dtype().StorageBytes() != shared_elem_bytes) {
    return LowerNormal(op, lower_args, analyzer);
  }
  const int elems_per_reg = 4 / shared_elem_bytes;
  PrimExpr flattened_indice =
      shared_tensor.ty()->ElemOffset(shared_indices).back();
  if (!IndicesCanVectorize(flattened_indice, col_var->var, col_var->dom->extent,
                           use_m16n8_stmatrix ? 16 : 8, analyzer)) {
    return LowerNormal(op, lower_args, analyzer);
  }

  if (use_m16n8_stmatrix) {
    // Each register packs two adjacent rows at column c, followed by the same
    // two rows at c + 8. These four elements must be consecutive in the
    // fragment.
    PrimVar packed_var("packed", row_var->var.ty());
    PrimExpr packed_index =
        Substitute(local_indices_flattened,
                   {{row_var->var,
                     FloorDiv(row_var->var, 2) * 2 + FloorMod(packed_var, 2)},
                    {col_var->var, FloorDiv(col_var->var, 16) * 16 +
                                       FloorMod(col_var->var, 8) +
                                       FloorDiv(packed_var, 2) * 8}});
    if (!IndicesCanVectorize(packed_index, packed_var,
                             tl::MakeConst(packed_var.ty(), 4), 4, analyzer,
                             /*allow_broadcast=*/false)) {
      return LowerNormal(op, lower_args, analyzer);
    }
  }

  PrimExpr extent = local_tensor.shape()[0];
  int num = 1;
  if (tl::arith::CanProveEqual(analyzer, FloorMod(extent, elems_per_reg * 4),
                               0))
    num = 4;
  else if (tl::arith::CanProveEqual(analyzer,
                                    FloorMod(extent, elems_per_reg * 2), 0))
    num = 2;

  Array<Expr> args;
  const Op &copy_op = is_ldmatrix ? tl::ptx_ldmatrix() : tl::ptx_stmatrix();
  args.push_back(PrimExpr(static_cast<int>(is_transposed)));
  args.push_back(PrimExpr(num));

  PrimVar local_iter("i");
  Layout inv = local_layout->Inverse();
  Array<PrimExpr> shared_coords;
  // Normalize the logical thread index against the thread range once, and
  // build the copy indices from the normalized expression directly.
  PrimExpr norm_thread_index = lower_args.thread_index;
  if (lower_args.thread_bounds.defined()) {
    norm_thread_index = norm_thread_index - lower_args.thread_bounds->begin;
  }
  PrimExpr warp = FloorDiv(norm_thread_index, 32) * 32;
  if (!is_transposed) {
    auto local_index = tl::arith::Simplify(
        analyzer,
        local_iter * elems_per_reg * num +
            elems_per_reg * FloorMod(FloorDiv(norm_thread_index, 8), num));
    auto thread_index = tl::arith::Simplify(
        analyzer, warp + FloorMod(norm_thread_index, 8) * 4);
    shared_coords = inv->Forward({local_index, thread_index});
  } else {
    auto local_index = tl::arith::Simplify(
        analyzer,
        local_iter * elems_per_reg * num +
            elems_per_reg * FloorMod(FloorDiv(norm_thread_index, 8), num) +
            FloorMod(norm_thread_index, 2));
    auto thread_index = tl::arith::Simplify(
        analyzer, warp + FloorDiv(FloorMod(norm_thread_index, 8), 2));
    shared_coords = inv->Forward({local_index, thread_index});
  }
  shared_coords.pop_back();
  // The fragment inverse is relative to the copy region, not the shared
  // buffer. Translate before the shared layout remaps these coordinates.
  for (size_t i = 0; i < shared_coords.size(); i++)
    shared_coords.Set(
        i, tl::arith::Simplify(analyzer,
                               shared_coords[i] + shared_range[i]->begin));
  Expr shared_addr = Call(
      std::nullopt, tl::access_ptr(),
      {TensorLoad(shared_tensor, shared_coords), PrimExpr(elems_per_reg * num),
       tl::MakeConst(PrimType::Int(32), is_ldmatrix ? 1 : 2)});
  args.push_back(shared_addr);

  if (is_ldmatrix) {
    if (local_tensor.dtype() != shared_tensor.dtype()) {
      return LowerNormal(op, lower_args, analyzer);
    }
    Expr local_addr =
        Call(std::nullopt, tl::access_ptr(),
             {TensorLoad(local_tensor, {local_iter * 2 * num}),
              PrimExpr(2 * num), tl::MakeConst(PrimType::Int(32), 2)});
    args.push_back(local_addr);
  } else {
    for (int i = 0; i < num; i++) {
      Array<PrimExpr> values;
      for (int j = 0; j < elems_per_reg; ++j) {
        PrimExpr value =
            TensorLoad(local_tensor, {local_iter * elems_per_reg * num +
                                      elems_per_reg * i + j});
        if (local_tensor.dtype() != shared_tensor.dtype()) {
          value = Cast(shared_tensor.dtype(), value);
        }
        values.push_back(value);
      }
      Expr value_packed = use_m16n8_stmatrix
                              ? Call(PrimType::Int(32), pack_b8x4(), values)
                              : Call(PrimType::Int(32), pack_b16(), values);
      args.push_back(value_packed);
    }
    args.push_back(StringImm(use_m16n8_stmatrix ? "m16n8" : "m8n8"));
  }

  auto body = Evaluate(Call(TupleType(Array<Type>{}), copy_op, args));
  For for_node = For(local_iter, 0, FloorDiv(extent, elems_per_reg * num), 1,
                     ForKind::kDefault, SeqStmt(body));
  for_node = PragmaUnrollLoop(for_node);
  return for_node;
}

namespace {

// Unpack a (datapath, column) coordinate IntTuple into two scalar
// expressions; adding the rank-2 zero first materializes an axis the layout
// never touches as a plain zero.
Array<PrimExpr> TmemCoordExprs(const cute::IntTuple &coord) {
  PrimType dtype = PrimType::Int(32);
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
                     sym::Analyzer analyzer) {
  const TensorVar &src = op.src;
  const TensorVar &dst = op.dst;

  if (!IsTmemBuffer(src) && !IsTmemBuffer(dst)) {
    return Stmt(ffi::UnsafeInit{});
  }
  ICHECK(TargetHasTmem(lower_args.target))
      << "Target " << lower_args.target->str()
      << " does not support tensor memory copy";

  bool is_ld = false;
  bool is_st = false;
  bool is_cp = false;
  bool src_needs_pack = 16 == src.dtype().bits();
  bool dst_needs_unpack = 16 == dst.dtype().bits();

  if (IsTmemBuffer(src) && IsFragmentBuffer(dst)) {
    is_ld = true;
  } else if (IsFragmentBuffer(src) && IsTmemBuffer(dst)) {
    is_st = true;
  } else if (IsSharedBuffer(src) && IsTmemBuffer(dst)) {
    is_cp = true;
  } else {
    LOG(FATAL) << "Unsupported tensor memory copy: " << "src scope = "
               << src.scope().name() << ", dst scope = " << dst.scope().name();
  }
  ICHECK(!is_cp)
      << "Copy from shared memory to tensor memory is not supported yet";

  Array<LayoutAxis> loop_vars = op.MakeIterVars();
  ICHECK_GE(loop_vars.size(), 2U)
      << "Tensor memory copy needs at least the two matrix dimensions, got "
      << loop_vars.size();
  for (const auto &iv : loop_vars)
    tl::arith::Bind(analyzer, iv->var, iv->dom);
  auto src_predicate = op.MakePredicate(analyzer, loop_vars, src.shape(), 0);
  auto dst_predicate = op.MakePredicate(analyzer, loop_vars, dst.shape(), 1);
  ICHECK(!src_predicate.has_value() && !dst_predicate.has_value())
      << "Tensor memory copy does not support predicates, got " << src_predicate
      << " and " << dst_predicate;
  for (const auto &iv : loop_vars) {
    ICHECK(GetConstInt(iv->dom->begin) && GetConstInt(iv->dom->extent))
        << "Tensor memory copy requires loop bounds to be constant integers";
  }
  constexpr int WARP_SIZE = 32;
  constexpr int WARPGROUP_SIZE = 4 * WARP_SIZE;
  ICHECK(GetConstInt(lower_args.thread_bounds->extent))
      << "Tensor memory copy requires thread_bounds->extent (num_threads) to "
         "be constant integers";
  int num_threads = *GetConstInt(lower_args.thread_bounds->extent);
  ICHECK(tl::arith::CanProveEqual(
             analyzer,
             FloorMod(lower_args.thread_bounds->begin, WARPGROUP_SIZE), 0) &&
         num_threads % WARPGROUP_SIZE == 0)
      << "Tensor memory copy requires thread bounds to be aligned to "
         "warpgroups, but found "
      << "thread range = " << lower_args.thread_bounds;

  TensorVar tmem_buf = is_ld ? src : dst;
  TensorVar reg_buf = is_ld ? dst : src;
  int tmem_side = is_ld ? 0 : 1;
  bool needs_pack_unpack = is_ld ? src_needs_pack : dst_needs_unpack;

  ICHECK(lower_args.layout_map.count(tmem_buf))
      << "Tmem buffer " << tmem_buf->name
      << " does not have a layout specified";
  ICHECK(lower_args.layout_map.count(reg_buf))
      << "Register buffer " << reg_buf->name
      << " does not have a layout specified";
  Layout tmem_layout = lower_args.layout_map[tmem_buf];
  Fragment reg_layout =
      (lower_args.layout_map[reg_buf]).as_or_throw<Fragment>();

  Array<PrimExpr> tmem_logical_indices = op.MakeIndices(loop_vars, tmem_side);
  Array<PrimExpr> reg_logical_indices =
      op.MakeIndices(loop_vars, 1 - tmem_side);

  // tmem_frag: logical buffer coord -> physical TMEM (datapath@0, column@1).
  Optional<cute::Layout> tmem_frag =
      cute::LayoutFromTileLangHierarchical(tmem_layout);
  ICHECK(tmem_frag.has_value())
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
  int64_t values_per_b32 = 32 / (tmem_buf.dtype().bits());
  bool pack16b =
      values_per_b32 > 1 && TmemFragmentNeedsPack16b(tmem_frag.value());
  if (pack16b) {
    tile = TmemTileToB32Columns(tile, values_per_b32);
    // The width-alignment checks below run in the tile's (b32) units; a
    // pack::16b origin is always storage-slot aligned.
    ICHECK(tl::arith::CanProveEqual(
        analyzer,
        floormod(phy_origin[1], IntImm(PrimType::Int(32), values_per_b32)), 0))
        << "A pack::16b TMEM origin must start on a b32 column";
    phy_origin.Set(
        1, tl::arith::Simplify(
               analyzer, floordiv(phy_origin[1],
                                  IntImm(PrimType::Int(32), values_per_b32))));
  }

  size_t tmem_rank = tmem_buf.shape().size();

  // The address token below is the Region's logical origin; LowerTileOp
  // materializes the buffer's layout over it.
  Map<PrimVar, PrimExpr> loop_to_origin;
  for (const auto &iv : loop_vars) {
    loop_to_origin.Set(iv->var, iv->dom->begin);
  }
  Array<PrimExpr> tmem_origin_indices =
      tmem_logical_indices.Map([&](const PrimExpr &index) {
        return tl::arith::Simplify(analyzer, Substitute(index, loop_to_origin));
      });
  if (needs_pack_unpack && !pack16b) {
    ICHECK(tl::arith::CanProveEqual(
        analyzer, floormod(tmem_origin_indices[tmem_rank - 1], 2), 0))
        << "A sliced packed 16-bit TMEM copy must start at an even logical "
           "column";
  }

  bool have_succeeded = false;
  Stmt body{ffi::UnsafeInit{}};

  Array<PrimExpr> loop_exprs;
  for (const auto &iv : loop_vars) {
    loop_exprs.push_back(iv->var);
  }

  auto try_tcgen05_instruction = [&](Tcgen05Meta meta) {
    if (have_succeeded) {
      return;
    }
    int width = static_cast<int>(Tcgen05AtomWidth(meta));
    if (!tl::arith::CanProveEqual(analyzer, phy_origin[0], 0) ||
        !tl::arith::CanProveEqual(analyzer, floormod(phy_origin[1], width),
                                  0)) {
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
      if (!tl::arith::CanProveEqual(analyzer, target_thread, reg_thread)) {
        continue;
      }
      PrimExpr target_reg = target_frag->Forward(loop_exprs)[0];
      PrimExpr reg_val = reg_layout->Forward(reg_logical_indices)[0];
      PrimExpr reg_origin =
          tl::arith::Simplify(analyzer, Substitute(reg_val, loop_to_origin));
      if (!tl::arith::CanProveEqual(analyzer, target_reg,
                                    reg_val - reg_origin)) {
        continue;
      }

      bool use_pack_unpack_modifier = pack16b;
      // One register covers two 16-bit values, so N halves.
      int effective_chunks =
          needs_pack_unpack ? num_chunks_each_wg / 2 : num_chunks_each_wg;
      PrimExpr relative_wg_idx = FloorDiv(
          Sub(lower_args.thread_index, lower_args.thread_bounds->begin),
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
      TensorVar orig_reg_buf = is_ld ? op.dst : op.src;
      for (int64_t issue = 0; issue < plan->num_issues; ++issue) {
        int64_t flat = cute::AsConst(plan->rest_domain(issue));
        Map<PrimVar, PrimExpr> loop_to_issue;
        for (const auto &iv : loop_vars) {
          int64_t extent = *GetConstInt(iv->dom->extent);
          loop_to_issue.Set(iv->var, iv->dom->begin +
                                         IntImm(iv->var.ty(), flat % extent));
          flat /= extent;
        }
        Array<PrimExpr> issue_tmem_indices =
            tmem_logical_indices.Map([&](const PrimExpr &index) {
              return tl::arith::Simplify(analyzer,
                                         Substitute(index, loop_to_issue));
            });
        // The access_ptr offset is a LOGICAL linear offset (row-major over
        // the buffer shape); LowerTileOp later maps it through the Fragment
        // to the per-thread register index.
        PrimExpr issue_reg_offset = IntImm(PrimType::Int(32), 0);
        for (size_t d = 0; d < reg_logical_indices.size(); ++d) {
          issue_reg_offset = issue_reg_offset * orig_reg_buf.shape()[d] +
                             Substitute(reg_logical_indices[d], loop_to_issue);
        }
        issue_reg_offset = tl::arith::Simplify(analyzer, issue_reg_offset);
        Array<Expr> args;
        args.push_back(IntImm(PrimType::Int(32), width * 32));
        args.push_back(IntImm(PrimType::Int(32), effective_chunks));
        args.push_back(
            tl::MakeConst(PrimType::Bool(), use_pack_unpack_modifier));
        args.push_back(TensorLoad(tmem_buf, issue_tmem_indices));
        args.push_back(col_offset);
        args.push_back(IntImm(PrimType::Int(32),
                              static_cast<int>(plan->datapaths_per_warp)));
        args.push_back(
            MakeFlatAccessPtr(reg_buf, is_ld ? 2 : 1, issue_reg_offset,
                              PrimExpr(static_cast<int>(vals_per_issue))));
        calls.push_back(
            Evaluate(Call(TupleType(Array<Type>{}),
                          is_ld ? tcgen05_ld() : tcgen05_st(), args)));
      }
      Stmt call = calls.size() == 1 ? calls[0] : SeqStmt(calls);
      if (num_useful_threads != num_threads) {
        body = If(lower_args.thread_index <
                      lower_args.thread_bounds->begin + num_useful_threads,
                  call, std::nullopt);
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

} // namespace

Stmt Copy::LowerBulk(const CopyNode &op, const LowerArgs &lower_args,
                     sym::Analyzer analyzer, CopyInst copy_inst) {
  const TensorVar &src = op.src;
  const TensorVar &dst = op.dst;
  const Array<Range> &src_range = op.src_range;
  const Array<Range> &dst_range = op.dst_range;
  const BlockAnnotations &annotations = op.annotations;

  ICHECK(copy_inst == CopyInst::kBulkLoad || copy_inst == CopyInst::kBulkStore)
      << "Invalid copy inst " << static_cast<int>(copy_inst);
  bool is_load = copy_inst == CopyInst::kBulkLoad;
  TensorVar global_tensor = is_load ? src : dst;
  TensorVar shared_tensor = is_load ? dst : src;
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

  ICHECK(IsValidTMADtypePair(is_load, global_tensor.dtype(),
                             shared_tensor.dtype()))
      << "Copy between buffer " << global_tensor->name << " and "
      << shared_tensor->name << " with incompatible data type "
      << global_tensor.dtype() << " and " << shared_tensor.dtype();

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
      TensorMapDataTypeForTMA(global_tensor.dtype(), shared_tensor.dtype());
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);
  shared_tensor = plan.shared_tensor;

  Call create_descriptor = Call(PtrType(tirx::TensorMapType()),
                                create_tma_descriptor(), desc.EncodeCallArgs());

  int64_t cluster_mask = GetClusterMask(op);
  bool use_multicast = is_load && (cluster_mask > 0);

  int barrier_base_id = -1;
  PrimExpr mbar_handle{ffi::UnsafeInit{}};
  bool is_cluster_barrier = false;
  if (is_load) {
    if (auto user_barrier = annotations.Get("barrier")) {
      mbar_handle = (user_barrier.value()).as_or_throw<PrimExpr>();
      barrier_base_id = 0;
      if (auto bl = mbar_handle.as<TensorLoadNode>()) {
        is_cluster_barrier =
            bl->source.as_or_throw<TensorVar>().scope().name() ==
            "shared.cluster_barrier";
      }
    } else if (GetIsTmaCopy(op)) {
      LOG(FATAL) << "T.tma_copy() requires a barrier argument. "
                 << "Use T.tma_copy(src, dst, barrier=mbar[idx])."
                 << SpanHintSuffix({op.dst->loc, op.src->loc});
    } else if (lower_args.alloc_mbarrier) {
      barrier_base_id =
          lower_args.alloc_mbarrier(1, MakeCopyMBarrierName(op.src, op.dst));
      PrimExpr mbar_idx = IntImm(PrimType::Int(32), barrier_base_id);
      mbar_handle = TensorLoad(lower_args.mbarrier_buffer->value(), {mbar_idx});
    }
  }

  auto tma_op = is_load ? tma_load() : tma_store();
  PrimExpr total_elements = IntImm(PrimType::Int(32), plan.box_size);

  auto make_args = [&](std::optional<PrimExpr> rest_idx) {
    Array<Expr> args;
    args.reserve(desc.rank + 4);
    args.push_back(create_descriptor);
    if (is_load)
      args.push_back(barrier_base_id >= 0 ? mbar_handle : PrimExpr(0));
    args.push_back(plan.SharedPointer(rest_idx, is_load ? 2 : 1));
    for (auto coord : plan.TmaCoords(rest_idx))
      args.push_back(coord);
    if (!is_load)
      args.push_back(PrimExpr(0)); // need_reduce
    args.push_back(PrimExpr(GetEvictionPolicy(op)));
    return args;
  };

  auto build_multicast_args = [&](const Array<Expr> &regular_args) {
    Array<Expr> mc_args;
    mc_args.reserve(regular_args.size() + 1);
    mc_args.push_back(regular_args[0]); // descriptor
    mc_args.push_back(regular_args[1]); // mbarrier
    mc_args.push_back(regular_args[2]); // shared memory pointer
    mc_args.push_back(IntImm(PrimType::Int(32), cluster_mask));
    for (size_t i = 3; i < regular_args.size(); ++i) {
      mc_args.push_back(regular_args[i]);
    }
    return mc_args;
  };

  // The leading CTA in the mask issues the multicast; CTAs outside the mask
  // replay the regular copy.
  auto wrap_multicast = [&](Stmt regular, Stmt multicast) {
    int min_cta_rank = MinRankInClusterMask(cluster_mask);
    PrimExpr block_rank = Call(PrimType::Int(32), block_rank_in_cluster(), {})
                              .as_or_throw<PrimExpr>();
    PrimExpr mask_imm = IntImm(PrimType::Int(32), cluster_mask);
    PrimExpr not_in_mask = EQ(bitwise_and(right_shift(mask_imm, block_rank),
                                          IntImm(PrimType::Int(32), 1)),
                              IntImm(PrimType::Int(32), 0));
    Stmt regular_or_noop = If(not_in_mask, regular, std::nullopt);
    return If(EQ(block_rank, IntImm(PrimType::Int(32), min_cta_rank)),
              SeqStmt(multicast), SeqStmt(regular_or_noop));
  };

  Stmt tma_copy{ffi::UnsafeInit{}};
  if (plan.rest_size > 1) {
    PrimVar loop_var("i", PrimType::Int(32));
    int loop_extent = plan.rest_size;
    Array<Expr> args = make_args(loop_var);
    BlockAnnotations ann;
    if (is_cluster_barrier && TargetIsSm100(lower_args.target) && is_load) {
      ann.Set("use_2cta", IntImm(PrimType::Int(32), 1));
    }
    tma_copy = For(
        loop_var, 0, loop_extent, 1, ForKind::kUnrolled,
        Evaluate(Call(TupleType(Array<Type>{}), tma_op, args, DictAttrs(ann))));
    if (use_multicast) {
      tma_copy = wrap_multicast(
          tma_copy,
          For(loop_var, 0, loop_extent, 1, ForKind::kUnrolled,
              Evaluate(Call(TupleType(Array<Type>{}), tma_load_multicast(),
                            build_multicast_args(args)))));
    }
  } else {
    Array<Expr> args = make_args(std::nullopt);
    BlockAnnotations ann;
    if (TargetIsSm100(lower_args.target) && is_load &&
        (annotations.find("use_2cta") != annotations.end() ||
         is_cluster_barrier)) {
      ann.Set("use_2cta", IntImm(PrimType::Int(32), 1));
    }
    tma_copy =
        Evaluate(Call(TupleType(Array<Type>{}), tma_op, args, DictAttrs(ann)));
    if (use_multicast) {
      tma_copy =
          wrap_multicast(tma_copy, Evaluate(Call(TupleType(Array<Type>{}),
                                                 tma_load_multicast(),
                                                 build_multicast_args(args))));
    }
  }

  if (!is_load) {
    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(tma_copy);
    seq.push_back(
        Evaluate(Call(TupleType(Array<Type>{}), tma_store_arrive(), {})));
    if (!GetIsTmaCopy(op)) {
      seq.push_back(Evaluate(Call(TupleType(Array<Type>{}), tma_store_wait(),
                                  {IntImm(PrimType::Int(32), 0),
                                   tl::MakeConst(PrimType::Bool(), true)})));
    }
    tma_copy = SeqStmt(std::move(seq));
  }

  if (is_load && barrier_base_id >= 0) {
    PrimExpr total_bytes{ffi::UnsafeInit{}};
    if (plan.rest_size > 1) {
      int loop_extent = plan.rest_size;
      total_bytes = TMATransactionBytesFromElements(
          total_elements * loop_extent, shared_tensor.dtype());
    } else {
      total_bytes = TMATransactionBytesFromElements(total_elements,
                                                    shared_tensor.dtype());
    }

    Stmt barrier_before_tma_stmt{ffi::UnsafeInit{}};
    Optional<Stmt> barrier_after_tma_stmt = std::nullopt;
    if (GetIsTmaCopy(op)) {
      if (is_cluster_barrier) {
        PrimExpr cluster_total_bytes =
            total_bytes * IntImm(PrimType::Int(32), lower_args.cluster_size);
        Stmt expect_stmt =
            Evaluate(Call(TupleType(Array<Type>{}), mbarrier_expect_tx(),
                          {mbar_handle, cluster_total_bytes}));
        PrimExpr rank = Call(PrimType::Int(32), block_rank_in_cluster(), {})
                            .as_or_throw<PrimExpr>();
        barrier_before_tma_stmt =
            If(EQ(rank, IntImm(PrimType::Int(32), 0)), expect_stmt);
      } else {
        barrier_before_tma_stmt =
            Evaluate(Call(TupleType(Array<Type>{}), mbarrier_expect_tx(),
                          {mbar_handle, total_bytes}));
      }
      if (auto emit_arrive_val = annotations.Get("emit_arrive")) {
        if ((emit_arrive_val.value()).as_or_throw<IntImm>()->value != 0) {
          barrier_after_tma_stmt = Evaluate(
              Call(TupleType(Array<Type>{}), mbarrier_arrive(), {mbar_handle}));
        }
      }
    } else {
      barrier_before_tma_stmt =
          Evaluate(Call(TupleType(Array<Type>{}), mbarrier_expect_tx(),
                        {mbar_handle, total_bytes}));
      barrier_after_tma_stmt = Evaluate(
          Call(TupleType(Array<Type>{}), mbarrier_arrive(), {mbar_handle}));
    }

    Array<Stmt> producer_seq{barrier_before_tma_stmt, tma_copy};
    if (barrier_after_tma_stmt.has_value()) {
      producer_seq.push_back(barrier_after_tma_stmt.value());
    }

    Stmt producer =
        If(MakeTmaLeaderCondition(GetLeaderScopeThreads(op, lower_args)),
           SeqStmt(producer_seq));

    if (GetIsTmaCopy(op)) {
      return producer;
    }

    Stmt wait_stmt = Evaluate(
        Call(TupleType(Array<Type>{}), mbarrier_wait_parity(),
             {mbar_handle, GetCopyMbarPhaseExpr(annotations, lower_args)}));

    return SeqStmt({producer, wait_stmt});
  }

  tma_copy = If(MakeTmaLeaderCondition(GetLeaderScopeThreads(op, lower_args)),
                tma_copy);

  return tma_copy;
}

namespace {

Array<PrimExpr> GetGather4Rows(const CopyNode &op) {
  if (auto val = op.annotations.Get("gather4_rows")) {
    return (val.value()).as_or_throw<Array<PrimExpr>>();
  }
  return {};
}

PrimExpr GetGather4Col(const CopyNode &op) {
  if (auto val = op.annotations.Get("gather4_col")) {
    return (val.value()).as_or_throw<PrimExpr>();
  }
  return PrimExpr(ffi::UnsafeInit{});
}

} // namespace

Stmt Copy::LowerBulkGather4(const CopyNode &op, const LowerArgs &lower_args,
                            sym::Analyzer analyzer, CopyInst copy_inst) {
  ICHECK(copy_inst == CopyInst::kBulkLoadGather4 ||
         copy_inst == CopyInst::kBulkStoreScatter4);
  bool is_load = copy_inst == CopyInst::kBulkLoadGather4;

  TensorVar global_tensor = is_load ? op.src : op.dst;
  TensorVar shared_tensor = is_load ? op.dst : op.src;

  ICHECK_EQ(global_tensor.shape().size(), 2u);
  ICHECK_EQ(shared_tensor.shape().size(), 2u);
  auto shared_lead = GetConstInt(shared_tensor.shape()[0]);
  ICHECK(shared_lead.has_value() && *shared_lead == 4)
      << "tma_gather4/scatter4 shared tile leading dim must be 4, got "
      << shared_tensor.shape()[0];
  ICHECK_EQ(global_tensor.dtype(), shared_tensor.dtype());

  Array<PrimExpr> rows = GetGather4Rows(op);
  PrimExpr col = GetGather4Col(op);
  ICHECK_EQ(rows.size(), 4u);
  ICHECK(col.defined());

  // Same decomposition as LowerBulk, over a virtual 4-row view of the global
  // tensor: the gathered rows act as global mode 0 whose coordinates are
  // given per instruction, and columns start at `col`.
  PrimExpr K = shared_tensor.shape()[1];
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
  desc.data_type = to_CUtensorMapDataType(global_tensor.dtype());
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);

  auto row_box = GetConstInt(desc.smem_box[1]);
  ICHECK(row_box.has_value() && *row_box == 4)
      << "tma_gather4/scatter4 requires the 4 gathered rows to sit right "
         "above the contiguous row chunk in shared memory, got box "
      << desc.smem_box;
  // The descriptor's row box-dim must be 1, not 4. The four-row pack is
  // implicit in the cp.async.bulk.tensor.tile::gather4 PTX, which takes 4
  // row coordinates and materializes them into 4 logical rows of the shared
  // tile. Setting box[1]=4 here would describe a contiguous 4-row strip; the
  // gather4 unrolling would then read OOB -> CUDA_ERROR_ILLEGAL_INSTRUCTION.
  desc.smem_box.Set(1, IntImm(PrimType::Int(32), 1));

  Call create_descriptor = Call(PtrType(tirx::TensorMapType()),
                                create_tma_descriptor(), desc.EncodeCallArgs());

  PrimExpr user_barrier{ffi::UnsafeInit{}};
  if (is_load) {
    auto barrier = op.annotations.Get("barrier");
    ICHECK(barrier.has_value())
        << "tma_gather4 requires a 'barrier' annotation";
    user_barrier = (barrier.value()).as_or_throw<PrimExpr>();
  }

  PrimExpr total_elements = IntImm(PrimType::Int(32), plan.box_size);
  TensorVar shared_physical = plan.shared_tensor;
  auto tl_op = is_load ? tma_load_gather4() : tma_store_scatter4();
  auto make_copy = [&](std::optional<PrimExpr> rest_idx) {
    Array<Expr> args;
    args.push_back(create_descriptor);
    if (is_load)
      args.push_back(user_barrier);
    args.push_back(plan.SharedPointer(rest_idx, is_load ? 2 : 1));
    args.push_back(plan.TmaCoords(rest_idx)[0]); // column coordinate
    for (auto r : rows)
      args.push_back(r);
    args.push_back(IntImm(PrimType::Int(32), GetEvictionPolicy(op)));
    return Evaluate(Call(TupleType(Array<Type>{}), tl_op, args));
  };

  // Fire-and-forget: caller manages mbarrier_expect_tx / wait (loads) and
  // tma_store_arrive / wait (stores), and the leader-thread guard.
  return plan.EmitInstructions(make_copy);
}

Stmt Copy::LowerBulk1D(const CopyNode &op, const LowerArgs &lower_args,
                       sym::Analyzer analyzer, CopyInst copy_inst) {
  const TensorVar &src = op.src;
  const TensorVar &dst = op.dst;
  const Array<Range> &src_range = op.src_range;
  const Array<Range> &dst_range = op.dst_range;
  const BlockAnnotations &annotations = op.annotations;

  ICHECK(copy_inst == CopyInst::kBulkLoad1D ||
         copy_inst == CopyInst::kBulkStore1D);

  int64_t cluster_mask = GetClusterMask(op);
  ICHECK(cluster_mask == 0)
      << "cluster_mask=0x" << std::hex << cluster_mask
      << " requires descriptor-based TMA (kBulkLoad); the 1D bulk-copy path "
         "does not support multicast. src="
      << src->name << " (scope=" << src.scope().name() << "), dst=" << dst->name
      << " (scope=" << dst.scope().name() << ").";

  bool is_load = copy_inst == CopyInst::kBulkLoad1D;
  auto shared_range = is_load ? dst_range : src_range;
  auto global_range = is_load ? src_range : dst_range;
  auto shared_tensor = is_load ? dst : src;
  auto global_tensor = is_load ? src : dst;

  PrimExpr shared_elements = 1;
  for (size_t i = 0; i < shared_range.size(); i++) {
    shared_elements *= shared_range[i]->extent;
  }

  PrimExpr elements = tl::arith::Simplify(analyzer, shared_elements);
  Expr shared_addr = MakeAccessPtrFromRegion(
      TensorRegion(shared_tensor, shared_range, TensorRegionType()),
      is_load ? kAccessWrite : kAccessRead);
  Expr global_addr = MakeAccessPtrFromRegion(
      TensorRegion(global_tensor, global_range, TensorRegionType()),
      is_load ? kAccessRead : kAccessWrite);

  int barrier_base_id = -1;
  PrimExpr mbar_handle{ffi::UnsafeInit{}};
  if (is_load) {
    if (auto user_barrier = annotations.Get("barrier")) {
      mbar_handle = (user_barrier.value()).as_or_throw<PrimExpr>();
      barrier_base_id = 0;
    } else if (GetIsTmaCopy(op)) {
      LOG(FATAL) << "T.tma_copy() requires a barrier argument. "
                 << "Use T.tma_copy(src, dst, barrier=mbar[idx])."
                 << SpanHintSuffix({op.dst->loc, op.src->loc});
    } else if (lower_args.alloc_mbarrier) {
      barrier_base_id =
          lower_args.alloc_mbarrier(1, MakeCopyMBarrierName(op.src, op.dst));
      PrimExpr mbar_idx = IntImm(PrimType::Int(32), barrier_base_id);
      mbar_handle = TensorLoad(lower_args.mbarrier_buffer->value(), {mbar_idx});
    }
  }

  Stmt tma_copy{ffi::UnsafeInit{}};
  PrimExpr total_bytes =
      TMATransactionBytesFromElements(elements, shared_tensor.dtype());

  // Safety net: instruction selection (CheckBulkCopy1D) must not route a
  // provably misaligned transfer here. Symbolic sizes cannot be proven
  // either way and are allowed through, matching the selection logic.
  ICHECK(!tl::arith::CanProve(
      analyzer,
      FloorMod(total_bytes, tl::MakeConst(total_bytes.ty(), 16)) !=
          tl::MakeConst(total_bytes.ty(), 0),
      sym::ProofStrength::kSymbolicBound))
      << "BulkCopy1D requires total_bytes to be 16-byte aligned, but got "
         "invalid size.\n"
      << "  Total_bytes = " << total_bytes << "\n";

  if (is_load) {
    PrimExpr mbar_arg = barrier_base_id >= 0 ? mbar_handle : PrimExpr(0);
    tma_copy = Evaluate(Call(TupleType(Array<Type>{}), tma_load(),
                             {shared_addr, global_addr, mbar_arg, total_bytes,
                              PrimExpr(GetEvictionPolicy(op))}));
  } else {
    int need_reduce = 0;
    tma_copy = Evaluate(
        Call(TupleType(Array<Type>{}), tma_store(),
             {global_addr, shared_addr, total_bytes, PrimExpr(need_reduce),
              PrimExpr(GetEvictionPolicy(op))}));
  }

  if (!is_load) {
    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(tma_copy);
    seq.push_back(
        Evaluate(Call(TupleType(Array<Type>{}), tma_store_arrive(), {})));
    if (!GetIsTmaCopy(op)) {
      seq.push_back(Evaluate(Call(TupleType(Array<Type>{}), tma_store_wait(),
                                  {IntImm(PrimType::Int(32), 0),
                                   tl::MakeConst(PrimType::Bool(), true)})));
    }
    tma_copy = SeqStmt(std::move(seq));
  }

  if (is_load && barrier_base_id >= 0) {
    Stmt barrier_before_tma_stmt{ffi::UnsafeInit{}};
    Optional<Stmt> barrier_after_tma_stmt = std::nullopt;
    if (GetIsTmaCopy(op)) {
      barrier_before_tma_stmt =
          Evaluate(Call(TupleType(Array<Type>{}), mbarrier_expect_tx(),
                        {mbar_handle, total_bytes}));
      if (auto emit_arrive_val = annotations.Get("emit_arrive")) {
        if ((emit_arrive_val.value()).as_or_throw<IntImm>()->value != 0) {
          barrier_after_tma_stmt = Evaluate(
              Call(TupleType(Array<Type>{}), mbarrier_arrive(), {mbar_handle}));
        }
      }
    } else {
      barrier_before_tma_stmt =
          Evaluate(Call(TupleType(Array<Type>{}), mbarrier_expect_tx(),
                        {mbar_handle, total_bytes}));
      barrier_after_tma_stmt = Evaluate(
          Call(TupleType(Array<Type>{}), mbarrier_arrive(), {mbar_handle}));
    }

    Array<Stmt> producer_seq{barrier_before_tma_stmt, tma_copy};
    if (barrier_after_tma_stmt.has_value()) {
      producer_seq.push_back(barrier_after_tma_stmt.value());
    }

    Stmt producer =
        If(MakeTmaLeaderCondition(GetLeaderScopeThreads(op, lower_args)),
           SeqStmt(producer_seq));

    if (GetIsTmaCopy(op)) {
      return producer;
    }

    Stmt wait_stmt = Evaluate(
        Call(TupleType(Array<Type>{}), mbarrier_wait_parity(),
             {mbar_handle, GetCopyMbarPhaseExpr(annotations, lower_args)}));

    return SeqStmt({producer, wait_stmt});
  }

  tma_copy = If(MakeTmaLeaderCondition(GetLeaderScopeThreads(op, lower_args)),
                tma_copy);
  return tma_copy;
}

Stmt Im2Col::Lower(const Im2ColOpNode &op, const LowerArgs &lower_args,
                   sym::Analyzer analyzer) {
  const TensorRegion &dst_region = op.dstRegion_;
  const TensorVar &src = op.src_;
  const TensorVar &dst = op.dst_;

  ICHECK(TargetIsHopper(lower_args.target));
  ICHECK(IsGlobalBuffer(src) && IsSharedBuffer(dst));
  ICHECK(src.shape().size() == 4);
  ICHECK(src.dtype() == dst.dtype());

  size_t ndim = dst_region->region.size();
  ICHECK(ndim >= 2) << "im2col dstRegion must have at least 2 dims";

  TMAIm2ColDesc desc;
  desc.rank = src.shape().size();
  desc.data_type = to_CUtensorMapDataType(src.dtype());
  desc.global_addr = src.data();
  desc.global_shape = ReverseArray(src.shape());
  desc.global_stride =
      ReverseArray(MatchShapeStridePattern(src.ty()).value().get<1>());
  ICHECK(IsOne(desc.global_stride[0])) << desc.global_stride;
  desc.global_stride = desc.global_stride.Map(
      [&](PrimExpr e) { return TMAGlobalBytesFromElements(e, src.dtype()); });
  for (size_t axis = 1; axis < desc.global_stride.size(); ++axis) {
    auto stride =
        GetConstInt(tl::arith::Simplify(analyzer, desc.global_stride[axis]));
    TVM_FFI_CHECK(!stride || *stride % 16 == 0, ValueError)
        << "im2col cannot lower " << src->name
        << ": global strides must be a 16-byte multiple";
  }
  desc.elem_stride = {1, op.stride_, op.stride_, 1};
  desc.lower_corner = {-op.padding_, -op.padding_};
  desc.upper_corner = {-op.padding_, -op.padding_};
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);

  PrimExpr h_dim = FloorDiv(src.shape()[1] + 2 * op.padding_ -
                                (op.kernel_ - 1) * op.dilation_ - 1,
                            op.stride_) +
                   1;
  PrimExpr w_dim = FloorDiv(src.shape()[2] + 2 * op.padding_ -
                                (op.kernel_ - 1) * op.dilation_ - 1,
                            op.stride_) +
                   1;

  // Same decomposition as LowerBulk, over a virtual (pixels, channels) view
  // of the global tensor: the composed shared layout decides the box and the
  // per-instruction coordinate steps; only the descriptor encoding is
  // im2col-specific.
  PrimExpr channels = src.shape()[3];
  PrimExpr box_pixel = dst_region->region[ndim - 2]->extent;
  PrimExpr box_channel = dst_region->region[ndim - 1]->extent;
  TensorVar gmem_view("im2col_gmem_view",
                      TensorType({src.shape()[0] * h_dim * w_dim, channels},
                                 src.dtype(), src.scope(),
                                 src.ty()->data_alignment));
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

  desc.smem_box_channel = GetConstInt(plan.desc.smem_box[0]).value();
  desc.smem_box_pixel = GetConstInt(plan.desc.smem_box[1]).value();
  desc.swizzle = plan.desc.swizzle;

  Call create_desc =
      Call(PtrType(tirx::TensorMapType()), create_tma_im2col_descriptor(),
           desc.EncodeCallArgs());

  // Channel steps stay inside one box_channel window (box_channel divides the
  // channel count), so the filter-tap decomposition is per-copy constant.
  ICHECK(tl::arith::CanProveEqual(analyzer, FloorMod(channels, box_channel), 0))
      << "Currently can only support divisible channel case";

  int barrier_base_id = -1;
  PrimExpr mbar_handle{ffi::UnsafeInit{}};
  if (auto user_barrier = op.annotations_.Get("barrier")) {
    mbar_handle = (user_barrier.value()).as_or_throw<PrimExpr>();
    barrier_base_id = 0;
  } else if (lower_args.alloc_mbarrier) {
    barrier_base_id =
        lower_args.alloc_mbarrier(1, MakeCopyMBarrierName(op.src_, op.dst_));
    PrimExpr mbar_idx = IntImm(PrimType::Int(32), barrier_base_id);
    mbar_handle = TensorLoad(lower_args.mbarrier_buffer->value(), {mbar_idx});
  }

  PrimExpr tile_elems = IntImm(PrimType::Int(32), plan.box_size);
  TensorVar shared_physical = plan.shared_tensor;
  auto make_copy = [&](std::optional<PrimExpr> rest_idx) {
    Array<PrimExpr> coords = plan.TmaCoords(rest_idx);
    PrimExpr channel_pos = coords[0];
    PrimExpr pixel_pos = coords[1];

    Array<Expr> args;
    args.reserve(desc.rank * 2 + 2);
    args.push_back(create_desc);
    args.push_back(barrier_base_id >= 0 ? mbar_handle : PrimExpr(0));
    args.push_back(plan.SharedPointer(rest_idx, kAccessWrite));
    args.push_back(FloorMod(channel_pos, channels));
    args.push_back(op.stride_ * FloorMod(pixel_pos, w_dim) - op.padding_);
    args.push_back(op.stride_ * FloorMod(FloorDiv(pixel_pos, w_dim), h_dim) -
                   op.padding_);
    args.push_back(FloorDiv(pixel_pos, w_dim * h_dim));
    args.push_back(op.dilation_ *
                   FloorMod(FloorDiv(channel_pos, channels), op.kernel_));
    args.push_back(op.dilation_ * FloorDiv(channel_pos, channels * op.kernel_));
    args.push_back(PrimExpr(op.eviction_policy_));
    return Evaluate(Call(TupleType(Array<Type>{}), tma_load_im2col(), args));
  };

  Stmt tma_copy_stmt = plan.EmitInstructions(make_copy);

  if (barrier_base_id >= 0) {
    bool ws_barrier = op.annotations_.Get("barrier").has_value();
    PrimExpr total_bytes = TMABytesFromElements(
        IntImm(PrimType::Int(32), plan.box_size * plan.rest_size), dst.dtype());

    Stmt barrier_before_tma_stmt =
        Evaluate(Call(TupleType(Array<Type>{}), mbarrier_expect_tx(),
                      {mbar_handle, total_bytes}));

    if (ws_barrier) {
      Array<Stmt> producer_seq{barrier_before_tma_stmt, tma_copy_stmt};
      if (auto emit_arrive_val = op.annotations_.Get("emit_arrive")) {
        if ((emit_arrive_val.value()).as_or_throw<IntImm>()->value != 0) {
          producer_seq.push_back(Evaluate(Call(
              TupleType(Array<Type>{}), mbarrier_arrive(), {mbar_handle})));
        }
      }
      return If(MakeTmaLeaderCondition(lower_args.thread_bounds->extent),
                SeqStmt(producer_seq));
    }

    Stmt barrier_after_tma_stmt = Evaluate(
        Call(TupleType(Array<Type>{}), mbarrier_arrive(), {mbar_handle}));

    Stmt producer = If(MakeTmaLeaderCondition(lower_args.thread_bounds->extent),
                       SeqStmt({barrier_before_tma_stmt, tma_copy_stmt,
                                barrier_after_tma_stmt}));

    Stmt wait_stmt = Evaluate(
        Call(TupleType(Array<Type>{}), mbarrier_wait_parity(),
             {mbar_handle, GetCopyMbarPhaseExpr(op.annotations_, lower_args)}));

    return SeqStmt({producer, wait_stmt});
  }

  return If(MakeTmaLeaderCondition(lower_args.thread_bounds->extent),
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
