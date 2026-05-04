/*!
 * \file tl/backend/cuda/op/copy.cc
 * \brief CUDA implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "layout/tcgen05_layout.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "target/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"
#include "transform/ptx_async_copy_injector.h"

#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

#include <sstream>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

PrimExpr MakeTmaLeaderCondition(PrimExpr thread_extent) {
  return Call(DataType::Bool(), tl_shuffle_elect(), {std::move(thread_extent)});
}

PrimExpr TMABytesFromElements(PrimExpr elements, DataType dtype) {
  PrimExpr elements_i64 = cast(DataType::Int(64), elements);
  int bits = dtype.bits();
  if (bits % 8 == 0) {
    return elements_i64 * IntImm(DataType::Int(64), bits / 8);
  }
  return FloorDiv(elements_i64 * IntImm(DataType::Int(64), bits) +
                      IntImm(DataType::Int(64), 7),
                  IntImm(DataType::Int(64), 8));
}

int64_t TMABytesFromElements(int64_t elements, DataType dtype) {
  return (elements * dtype.bits() + 7) / 8;
}

int64_t TMAElementsForBytes(int64_t bytes, DataType dtype) {
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

namespace cuda {

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                               InferLevel level);

  static CopyInst SelectInst(const CopyNode &op, Target target,
                             const LayoutMap &layout_map,
                             arith::Analyzer *analyzer, bool buffer_oob);

  static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer);

private:
  static void CheckParallelLoopLayout(const CopyNode &op, CopyInst copy_inst);

  static LayoutMap InferTMemLayout(const CopyNode &op, const LayoutInferArgs &T,
                                   CopyInst copy_inst);

  static LayoutMap InferBulkLayout(const CopyNode &op, const LayoutInferArgs &T,
                                   InferLevel level, CopyInst copy_inst);

  static Stmt LowerNormal(const CopyNode &op, const LowerArgs &T,
                          arith::Analyzer *analyzer);

  static Stmt LowerCPAsync(const CopyNode &op, const LowerArgs &T,
                           arith::Analyzer *analyzer);

  static Stmt LowerLDSM(const CopyNode &op, const LowerArgs &T,
                        arith::Analyzer *analyzer, CopyInst copy_inst);

  static Stmt LowerTmem(const CopyNode &op, const LowerArgs &T,
                        arith::Analyzer *analyzer);

  static Stmt LowerBulk(const CopyNode &op, const LowerArgs &T,
                        arith::Analyzer *analyzer, CopyInst copy_inst);

  static Stmt LowerBulk1D(const CopyNode &op, const LowerArgs &T,
                          arith::Analyzer *analyzer, CopyInst copy_inst);
};

LayoutMap Copy::InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                            InferLevel level) {
  CopyInst copy_inst =
      SelectInst(op, T.target, T.layout_map, T.analyzer, T.buffer_oob);
  CheckParallelLoopLayout(op, copy_inst);

  if (copy_inst == CopyInst::kTMemLoad || copy_inst == CopyInst::kTMemStore) {
    return InferTMemLayout(op, T, copy_inst);
  }
  if (copy_inst == CopyInst::kBulkLoad || copy_inst == CopyInst::kBulkStore ||
      copy_inst == CopyInst::kBulkLoad1D ||
      copy_inst == CopyInst::kBulkStore1D) {
    return InferBulkLayout(op, T, level, copy_inst);
  }

  // For normal/cp.async/LDSM/STSM, layout inference follows the generated
  // SIMT loop. CUDA-specific explicit layout cases are handled above.
  return CopyLoweringAccess::InferSIMTLayout(op, T, level);
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

LayoutMap Copy::InferTMemLayout(const CopyNode &op, const LayoutInferArgs &T,
                                CopyInst copy_inst) {
  // TODO (mzw) Add support for tcgen05.cp in CUDA tmem lowering.
  LayoutMap results;
  bool is_tmem_load = copy_inst == CopyInst::kTMemLoad;
  Buffer tmem_buf = is_tmem_load ? op.src : op.dst;
  Buffer reg_buf = is_tmem_load ? op.dst : op.src;

  if (!T.layout_map.count(reg_buf) && T.layout_map.count(tmem_buf)) {
    Layout tmem_layout = T.layout_map[tmem_buf];
    Array<IterVar> logical_coords = CopyLoweringAccess::MakeIterVars(op);
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
    ICHECK(is_const_int(T.thread_bounds->extent))
        << "Tensor memory copy requires thread_bounds->extent (num_threads) "
           "to be constant integers";
    int num_threads = *as_const_int(T.thread_bounds->extent);
    ICHECK(num_threads % WARPGROUP_SIZE == 0)
        << "Tensor memory copy requires thread bounds to be aligned to "
           "warpgroups, but found "
        << "thread range = " << T.thread_bounds;

    for (int num_useful_wgs = num_threads / WARPGROUP_SIZE; num_useful_wgs >= 1;
         --num_useful_wgs) {
      int num_useful_threads = num_useful_wgs * WARPGROUP_SIZE;
      Tcgen05Meta meta = getTcgen05MetaLd_32dp32b();
      auto [is_success, tmem_coord2frag, num_chunks_each_wg] =
          expandTcgen05Layout(
              meta, phy_col_bounds->max_value - phy_col_bounds->min_value + 1,
              num_useful_threads, row_dom, col_dom);
      (void)num_chunks_each_wg;
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

LayoutMap Copy::InferBulkLayout(const CopyNode &op, const LayoutInferArgs &T,
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
  PrimExpr thread_extent = T.thread_bounds->extent;
  for (const auto &range : op.src_range) {
    CopyLoweringAccess::CollectFragmentLayouts(
        op, range->min, T.let_var_to_expr, T.layout_map, thread_extent,
        T.thread_bounds, result_map);
    CopyLoweringAccess::CollectFragmentLayouts(
        op, range->extent, T.let_var_to_expr, T.layout_map, thread_extent,
        T.thread_bounds, result_map);
  }
  for (const auto &range : op.dst_range) {
    CopyLoweringAccess::CollectFragmentLayouts(
        op, range->min, T.let_var_to_expr, T.layout_map, thread_extent,
        T.thread_bounds, result_map);
    CopyLoweringAccess::CollectFragmentLayouts(
        op, range->extent, T.let_var_to_expr, T.layout_map, thread_extent,
        T.thread_bounds, result_map);
  }

  if (level == InferLevel::kFree && !T.layout_map.count(shared_tensor)) {
    if (is_store) {
      // For BulkStore, infer a swizzled shared-memory layout when possible.
      int dim = shared_tensor->shape.size();
      const int64_t mat_stride = *as_const_int(shared_tensor->shape[dim - 2]);
      const int64_t mat_continuous =
          *as_const_int(shared_tensor->shape[dim - 1]);
      Layout swizzle_layout_2d =
          makeGemmABLayoutHopper(mat_stride, mat_continuous, mat_continuous,
                                 shared_tensor->dtype.bits(),
                                 /*k_inner=*/true);
      if (StructuralEqual()(swizzle_layout_2d, makeLinearLayout(Array<PrimExpr>{
                                                   Integer(mat_stride),
                                                   Integer(mat_continuous)}))) {
        result_map.Set(shared_tensor, CopyLoweringAccess::ComputeLinearLayout(
                                          op, shared_tensor));
      } else {
        result_map.Set(shared_tensor, ExpandLayoutToMatchBuffer(
                                          swizzle_layout_2d, shared_tensor));
      }
    } else {
      result_map.Set(shared_tensor, CopyLoweringAccess::ComputeLinearLayout(
                                        op, shared_tensor));
    }
  }

  return result_map;
}

CopyInst Copy::SelectInst(const CopyNode &op, Target target,
                          const LayoutMap &layout_map,
                          arith::Analyzer *analyzer, bool buffer_oob) {
  if (op.GetIsTmaCopy()) {
    bool is_cutedsl = TargetIsCuTeDSL(target);
    if (!is_cutedsl && !buffer_oob &&
        op.CheckBulkLoad1D(target, layout_map, analyzer)) {
      return CopyInst::kBulkLoad1D;
    } else if (!is_cutedsl && !buffer_oob &&
               op.CheckBulkStore1D(target, layout_map, analyzer)) {
      return CopyInst::kBulkStore1D;
    } else if (op.CheckBulkLoad(target, analyzer)) {
      return CopyInst::kBulkLoad;
    } else if (op.CheckBulkStore(target, analyzer)) {
      return CopyInst::kBulkStore;
    } else {
      LOG(FATAL) << "T.tma_copy() requires TMA-capable target and "
                    "global<->shared copy pattern, but TMA is not available "
                    "for src="
                 << op.src->name << ", dst=" << op.dst->name;
    }
  }

  bool is_async_copy = op.GetIsAsyncCopy();
  bool no_implicit_commit_wait = op.GetNoImplicitAsyncCommitWait();

  if (is_async_copy || no_implicit_commit_wait) {
    bool cp_async_supported = op.CheckCPAsyncCopy(target, layout_map, analyzer);
    ICHECK(cp_async_supported)
        << "Explicit async copy semantics require cp.async lowering, but "
           "constraints were not satisfied. Got src="
        << op.src->name << " (scope=" << op.src.scope()
        << ", dtype=" << op.src->dtype << "), dst=" << op.dst->name
        << " (scope=" << op.dst.scope() << ", dtype=" << op.dst->dtype << ").";
    return CopyInst::kCPAsync;
  }

  if (!op.GetDisableTMA() &&
      !tvm::transform::PassContext::Current()
           ->GetConfig<Bool>(kDisableTMALower, Bool(false))
           .value()) {
    bool is_cutedsl = TargetIsCuTeDSL(target);
    if (!is_cutedsl && !buffer_oob &&
        op.CheckBulkStore1D(target, layout_map, analyzer)) {
      return CopyInst::kBulkStore1D;
    } else if (op.CheckBulkStore(target, analyzer)) {
      return CopyInst::kBulkStore;
    }
  }

  if (op.CheckLDSMCopy(target)) {
    return CopyInst::kLDSM;
  } else if (op.CheckSTSMCopy(target)) {
    return CopyInst::kSTSM;
  } else if (op.CheckTMemLoad(target)) {
    return CopyInst::kTMemLoad;
  } else if (op.CheckTMemStore(target)) {
    return CopyInst::kTMemStore;
  } else {
    return CopyInst::kNormal;
  }
}

Stmt Copy::Lower(const CopyNode &op, const LowerArgs &T,
                 arith::Analyzer *analyzer) {
  auto copy_inst =
      SelectInst(op, T.target, T.layout_map, analyzer, /*buffer_oob=*/false);
  if (copy_inst == CopyInst::kTMemLoad || copy_inst == CopyInst::kTMemStore) {
    auto tmem_copy = LowerTmem(op, T, analyzer);
    ICHECK(tmem_copy.defined()) << "Failed to lower tensor memory copy";
    return tmem_copy;
  } else if (copy_inst == CopyInst::kBulkLoad1D ||
             copy_inst == CopyInst::kBulkStore1D) {
    auto bulk_copy = LowerBulk1D(op, T, analyzer, copy_inst);
    ICHECK(bulk_copy.defined()) << "Failed to lower bulk load 1d";
    return bulk_copy;
  } else if (copy_inst == CopyInst::kBulkLoad ||
             copy_inst == CopyInst::kBulkStore) {
    auto bulk_copy = LowerBulk(op, T, analyzer, copy_inst);
    ICHECK(bulk_copy.defined()) << "Failed to lower bulk load/store";
    return bulk_copy;
  } else if (copy_inst == CopyInst::kLDSM || copy_inst == CopyInst::kSTSM) {
    auto ldsm_copy = LowerLDSM(op, T, analyzer, copy_inst);
    ICHECK(ldsm_copy.defined()) << "Failed to lower ptx matrix copy";
    return ldsm_copy;
  } else if (copy_inst == CopyInst::kCPAsync) {
    auto cp_async_copy = LowerCPAsync(op, T, analyzer);
    ICHECK(cp_async_copy.defined()) << "Failed to lower cp.async copy";
    return cp_async_copy;
  } else if (copy_inst == CopyInst::kNormal) {
    return LowerNormal(op, T, analyzer);
  } else {
    LOG(FATAL) << "Unsupported copy inst " << static_cast<int>(copy_inst);
  }
}

Stmt Copy::LowerCPAsync(const CopyNode &op, const LowerArgs &T,
                        arith::Analyzer *analyzer) {
  return tl::LowerCPAsyncCopy(op, T, analyzer);
}

Stmt Copy::LowerNormal(const CopyNode &op, const LowerArgs &T,
                       arith::Analyzer *analyzer) {
  return tl::LowerNormalCopy(op, T, analyzer);
}

Stmt Copy::LowerLDSM(const CopyNode &op, const LowerArgs &T,
                     arith::Analyzer *analyzer, CopyInst copy_inst) {
  const Buffer &src = op.src;
  const Buffer &dst = op.dst;
  const Array<Range> &src_range = op.src_range;
  const Array<Range> &dst_range = op.dst_range;

  ICHECK(copy_inst == CopyInst::kLDSM || copy_inst == CopyInst::kSTSM)
      << "Invalid copy inst " << static_cast<int>(copy_inst);
  bool is_ldmatrix = copy_inst == CopyInst::kLDSM;

  Array<IterVar> loop_vars = CopyLoweringAccess::MakeIterVars(op);
  if (loop_vars.size() < 2) {
    return LowerNormal(op, T, analyzer);
  }
  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);
  PrimExpr src_predicate =
      CopyLoweringAccess::MakePredicate(op, analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate =
      CopyLoweringAccess::MakePredicate(op, analyzer, loop_vars, dst->shape, 1);
  if (src_predicate.defined() || dst_predicate.defined()) {
    return LowerNormal(op, T, analyzer);
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
    return LowerNormal(op, T, analyzer);
  }

  Array<PrimExpr> local_indices =
      CopyLoweringAccess::MakeIndices(op, loop_vars, is_ldmatrix ? 1 : 0);
  Fragment local_layout = Downcast<Fragment>(T.layout_map[local_tensor]);
  Array<PrimExpr> local_indices_transformed =
      local_layout->Forward(local_indices);
  local_tensor = T.buffer_remap[local_tensor];
  if (local_layout->OutputDim() != 1) {
    return LowerNormal(op, T, analyzer);
  }

  Array<PrimExpr> shared_indices =
      CopyLoweringAccess::MakeIndices(op, loop_vars, is_ldmatrix ? 0 : 1);
  bool is_transposed;
  IterVar col_var = loop_vars[loop_vars.size() - 1];
  IterVar row_var = loop_vars[loop_vars.size() - 2];
  PrimExpr local_layout_thread_map =
      FloorMod(local_layout->ForwardThread(local_indices, std::nullopt), 32);
  PrimExpr matrix_8x8_thread_map = makeGemmFragment8x8()->ForwardThread(
      {FloorMod(row_var, 8), FloorMod(col_var, 8)}, std::nullopt);
  PrimExpr matrix_8x8_thread_map_trans =
      makeGemmFragment8x8Transposed()->ForwardThread(
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
    return LowerNormal(op, T, analyzer);
  }
  if (shared_tensor->dtype.bytes() != 2) {
    return LowerNormal(op, T, analyzer);
  }
  PrimExpr flattened_indice = shared_tensor.OffsetOf(shared_indices).back();
  if (!IndicesCanVectorize(flattened_indice, loop_vars.back()->var,
                           loop_vars.back()->dom->extent, 8, analyzer)) {
    return LowerNormal(op, T, analyzer);
  }

  for (size_t i = 0; i < dst_range.size(); i++) {
    if (!is_zero(dst_range[i]->min) ||
        !analyzer->CanProveEqual(dst_range[i]->extent, dst->shape[i]))
      return LowerNormal(op, T, analyzer);
  }

  PrimExpr extent = local_tensor->shape[0];
  int num = 1;
  if (analyzer->CanProveEqual(FloorMod(extent, 8), 0))
    num = 4;
  else if (analyzer->CanProveEqual(FloorMod(extent, 4), 0))
    num = 2;

  Array<PrimExpr> args;
  const Op &copy_op = is_ldmatrix ? tl::ptx_ldmatrix() : tl::ptx_stmatrix();
  args.push_back(static_cast<int>(is_transposed));
  args.push_back(num);

  Var local_iter("i");
  Layout inv = local_layout->Inverse();
  Array<PrimExpr> shared_coords;
  PrimExpr warp = FloorDiv(T.thread_var, 32) * 32;
  if (!is_transposed) {
    auto local_index = analyzer->Simplify(
        local_iter * 2 * num + 2 * FloorMod(FloorDiv(T.thread_var, 8), num));
    auto thread_index =
        analyzer->Simplify(warp + FloorMod(T.thread_var, 8) * 4);
    shared_coords = inv->Forward({local_index, thread_index});
  } else {
    auto local_index = analyzer->Simplify(
        local_iter * 2 * num + 2 * FloorMod(FloorDiv(T.thread_var, 8), num) +
        FloorMod(T.thread_var, 2));
    auto thread_index =
        analyzer->Simplify(warp + FloorDiv(FloorMod(T.thread_var, 8), 2));
    shared_coords = inv->Forward({local_index, thread_index});
  }
  shared_coords.pop_back();
  PrimExpr shared_addr =
      Call(DataType::Handle(), tl::access_ptr(),
           {BufferLoad(shared_tensor, shared_coords), PrimExpr(2 * num),
            make_const(DataType::Int(32), is_ldmatrix ? 1 : 2)});
  args.push_back(shared_addr);

  if (is_ldmatrix) {
    if (local_tensor->dtype != shared_tensor->dtype) {
      return LowerNormal(op, T, analyzer);
    }
    PrimExpr local_addr =
        Call(DataType::Handle(), tl::access_ptr(),
             {BufferLoad(local_tensor, {local_iter * 2 * num}),
              PrimExpr(2 * num), make_const(DataType::Int(32), 2)});
    args.push_back(local_addr);
  } else {
    for (int i = 0; i < num; i++) {
      PrimExpr value0 =
          BufferLoad(local_tensor, {local_iter * 2 * num + 2 * i});
      PrimExpr value1 =
          BufferLoad(local_tensor, {local_iter * 2 * num + 2 * i + 1});
      if (local_tensor->dtype != shared_tensor->dtype) {
        value0 = Cast(shared_tensor->dtype, value0);
        value1 = Cast(shared_tensor->dtype, value1);
      }
      PrimExpr value_packed =
          Call(DataType::Int(32), pack_b16(), {value0, value1});
      args.push_back(value_packed);
    }
  }

  auto body = Evaluate(Call(DataType::Handle(), copy_op, args));
  For for_node =
      For(local_iter, 0, FloorDiv(extent, 2 * num), ForKind::kSerial, body);
  for_node = PragmaUnrollLoop(for_node);
  auto range = T.thread_bounds;
  if (range.defined()) {
    auto thread_var = T.thread_var;
    auto thread_var_with_offset = thread_var - range->min;
    for_node.CopyOnWrite()->body =
        Substitute(for_node->body, {{thread_var, thread_var_with_offset}});
  }
  return for_node;
}

Stmt Copy::LowerTmem(const CopyNode &op, const LowerArgs &T,
                     arith::Analyzer *analyzer) {
  const Buffer &src = op.src;
  const Buffer &dst = op.dst;

  if (src.scope() != "shared.tmem" && dst.scope() != "shared.tmem") {
    return Stmt();
  }
  ICHECK(TargetHasTmem(T.target)) << "Target " << T.target->ToDebugString()
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

  Array<IterVar> loop_vars = CopyLoweringAccess::MakeIterVars(op);
  ICHECK(loop_vars.size() == 2) << "Only support 2D tensor memory copy, got "
                                << loop_vars.size() << " dimensions";
  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);
  PrimExpr src_predicate =
      CopyLoweringAccess::MakePredicate(op, analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate =
      CopyLoweringAccess::MakePredicate(op, analyzer, loop_vars, dst->shape, 1);
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
  ICHECK(is_const_int(T.thread_bounds->extent))
      << "Tensor memory copy requires thread_bounds->extent (num_threads) to "
         "be constant integers";
  int num_threads = *as_const_int(T.thread_bounds->extent);
  ICHECK(analyzer->CanProveEqual(FloorMod(T.thread_bounds->min, WARPGROUP_SIZE),
                                 0) &&
         num_threads % WARPGROUP_SIZE == 0)
      << "Tensor memory copy requires thread bounds to be aligned to "
         "warpgroups, but found "
      << "thread range = " << T.thread_bounds;

  Buffer tmem_buf = is_ld ? src : dst;
  Buffer reg_buf = is_ld ? dst : src;
  int tmem_side = is_ld ? 0 : 1;
  bool needs_pack_unpack = is_ld ? src_needs_pack : dst_needs_unpack;

  ICHECK(T.layout_map.count(tmem_buf)) << "Tmem buffer " << tmem_buf->name
                                       << " does not have a layout specified";
  ICHECK(T.layout_map.count(reg_buf)) << "Register buffer " << reg_buf->name
                                      << " does not have a layout specified";
  Layout tmem_layout = T.layout_map[tmem_buf];
  Fragment reg_layout = Downcast<Fragment>(T.layout_map[reg_buf]);

  Array<PrimExpr> logical_indices =
      CopyLoweringAccess::MakeIndices(op, loop_vars, tmem_side);
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
      auto [is_success, target_frag, num_chunks_each_wg] = expandTcgen05Layout(
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
          FloorDiv(Sub(T.thread_var, T.thread_bounds->min), WARPGROUP_SIZE);
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
            IfThenElse(T.thread_var < T.thread_bounds->min + num_useful_threads,
                       call, Stmt());
      } else {
        body = call;
      }
      break;
    }
  };

  if (is_ld) {
    try_tcgen05_instruction(getTcgen05MetaLd_32dp32b());
    try_tcgen05_instruction(getTcgen05MetaLd_32dp64b());
    try_tcgen05_instruction(getTcgen05MetaLd_32dp128b());
    try_tcgen05_instruction(getTcgen05MetaLd_32dp256b());
  } else {
    try_tcgen05_instruction(getTcgen05MetaSt_32dp32b());
    try_tcgen05_instruction(getTcgen05MetaSt_32dp64b());
    try_tcgen05_instruction(getTcgen05MetaSt_32dp128b());
    try_tcgen05_instruction(getTcgen05MetaSt_32dp256b());
  }

  ICHECK(have_succeeded) << "Failed to find a suitable instruction for tcgen05."
                         << (is_ld ? "ld" : "st") << ". Check your layout.";

  return body;
}

Stmt Copy::LowerBulk(const CopyNode &op, const LowerArgs &T,
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
  Buffer shared_tensor_unmapped = shared_tensor;
  Array<Range> global_range = is_load ? src_range : dst_range;
  Array<Range> shared_range = is_load ? dst_range : src_range;

  if (T.layout_map.count(global_tensor)) {
    LOG(WARNING) << "TMA bulk copy cannot support a non-swizzled global "
                    "layout, fallback to normal copy.";
    return LowerNormal(op, T, analyzer);
  }

  auto linear_layout =
      CopyLoweringAccess::ComputeLinearLayout(op, shared_tensor);

  Array<PrimExpr> shared_indices;
  for (auto r : shared_range)
    shared_indices.push_back(r->min);
  std::vector<PrimExpr> shared_strides;
  PrimExpr shared_stride = 1;
  for (size_t i = 0; i < shared_tensor->shape.size(); i++) {
    auto s = shared_tensor->shape[shared_tensor->shape.size() - i - 1];
    shared_strides.insert(shared_strides.begin(), shared_stride);
    shared_stride *= s;
  }

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

  ICHECK(shared_strides.size() == shared_indices.size())
      << "shared_strides.size() != shared_indices.size()"
      << shared_strides.size() << " " << shared_indices.size();
  PrimExpr shared_offset = 0;
  for (size_t i = 0; i < shared_indices.size(); i++) {
    shared_offset += shared_indices[i] * shared_strides[i];
  }
  PrimExpr global_offset = 0;
  for (size_t i = 0; i < global_indices.size(); i++) {
    global_offset += global_indices[i] * global_strides[i];
  }

  TMADesc desc;
  desc.rank = global_tensor->shape.size();
  ICHECK(desc.rank >= 1 && desc.rank <= 5) << desc.rank;

  ICHECK(global_tensor->dtype == shared_tensor->dtype)
      << "Copy between buffer " << global_tensor->name << " and "
      << shared_tensor->name << " with different data type "
      << global_tensor->dtype << " and " << shared_tensor->dtype;

  desc.data_type = to_CUtensorMapDataType(global_tensor->dtype);
  desc.global_addr = global_tensor->data;
  desc.global_shape = ReverseArray(global_tensor->shape);
  Array<PrimExpr> global_coords =
      ReverseArray(global_range.Map([](Range r) { return r->min; }));
  if (!global_tensor->strides.empty()) {
    desc.global_stride = ReverseArray(global_tensor->strides);
  } else {
    PrimExpr stride = 1;
    desc.global_stride.reserve(desc.rank);
    for (size_t i = 0; i < desc.rank; i++) {
      desc.global_stride.push_back(stride);
      stride *= desc.global_shape[i];
    }
  }
  ICHECK(is_one(desc.global_stride[0])) << desc.global_stride;
  desc.global_stride = desc.global_stride.Map([&](PrimExpr e) {
    return TMABytesFromElements(e, global_tensor->dtype);
  });
  for (size_t i{1}; i < desc.global_stride.size(); i++) {
    auto stride = desc.global_stride[i].as<IntImmNode>();
    if (stride != nullptr) {
      if (stride->value % 16 != 0 || stride->value >= (1ULL << 40)) {
        LOG(WARNING) << "TMA bulk copy cannot support a global stride of "
                     << desc.global_stride[i] << ", fallback to normal copy.";
        return LowerNormal(op, T, analyzer);
      }
    }
  }

  auto s_range_idx = 0;
  for (size_t i = 0; i < global_range.size(); i++) {
    auto g_range = global_range[i];
    if (is_one(g_range->extent)) {
      continue;
    }
    while (is_one(shared_range[s_range_idx]->extent) &&
           s_range_idx < shared_range.size()) {
      s_range_idx++;
    }
    if (s_range_idx >= shared_range.size()) {
      LOG(FATAL) << "TMA bulk copy cannot support a global range of "
                 << global_range << ", shared_range " << shared_range;
    }
    auto s_range = shared_range[s_range_idx];
    s_range_idx++;

    ICHECK(StructuralEqual()(g_range->extent, s_range->extent))
        << global_tensor->name << "[" << i << "] is illegal, "
        << global_tensor->name << "[" << i << "] = " << g_range->extent << ", "
        << shared_tensor->name << "[" << s_range_idx
        << "] = " << s_range->extent;
  }
  desc.smem_box =
      ReverseArray(global_range.Map([](Range r) { return r->extent; }));

  desc.smem_stride = Array<PrimExpr>(desc.rank, PrimExpr(1));
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);

  Layout shared_layout;
  if (T.layout_map.count(shared_tensor)) {
    shared_layout = T.layout_map.at(shared_tensor);
    ICHECK(T.buffer_remap.count(shared_tensor))
        << "shared_tensor: " << shared_tensor->name
        << " not found in buffer_remap";
    shared_tensor = T.buffer_remap.at(shared_tensor);
  }
  if (!shared_layout.defined()) {
    desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
  } else if (StructuralEqual()(shared_layout, linear_layout)) {
    desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
  } else {
    if (shared_layout->InputDim() < 2) {
      LOG(WARNING) << "TMA bulk copy cannot support shared layout with input "
                   << "dimension " << shared_layout->InputDim()
                   << ", fallback to normal copy.";
      return LowerNormal(op, T, analyzer);
    }
    const int ndim = static_cast<int>(shared_layout->InputDim());
    auto stride = as_const_int(shared_layout->InputShape()[ndim - 2]);
    auto continuous = as_const_int(shared_layout->InputShape()[ndim - 1]);
    ICHECK(stride != nullptr && continuous != nullptr);
    SwizzleMode swizzle_mode =
        DetectSwizzleMode(shared_layout, shared_tensor_unmapped);
    if (swizzle_mode == SwizzleMode::kQuarter) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B);
    } else if (swizzle_mode == SwizzleMode::kHalf) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B);
    } else if (swizzle_mode == SwizzleMode::kFull) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B);
    } else if (StructuralEqual()(
                   shared_layout,
                   makeGemmABLayoutPadded(*stride, *continuous,
                                          shared_tensor->dtype.bits()))) {
      LOG(WARNING) << "Bulk copy cannot support a padded layout for src: "
                   << src->name << ", dst: " << dst->name
                   << ", fallback to normal copy";
      return LowerNormal(op, T, analyzer);
    } else {
      LOG(WARNING) << "Came across unsupported swizzle layout for src: "
                   << src->name << ", dst: " << dst->name
                   << ", fallback to normal copy";
      return LowerNormal(op, T, analyzer);
    }
  }

  auto inner_box_dim = as_const_int(desc.smem_box[0]);
  if (inner_box_dim == nullptr) {
    LOG(WARNING) << "inner_box_dim " << desc.smem_box[0]
                 << " can only be a constant integer for TMA bulk copy, "
                    "fallback to normal copy";
    return LowerNormal(op, T, analyzer);
  }
  int instruction_dim = *inner_box_dim;
  if (desc.swizzle == static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B)) {
    instruction_dim = TMAElementsForBytes(64, src->dtype);
  } else if (desc.swizzle == static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B)) {
    instruction_dim = TMAElementsForBytes(128, src->dtype);
  }
  if (instruction_dim > 256) {
    ICHECK((*inner_box_dim) % 256 == 0)
        << "inner_box_dim: " << *inner_box_dim << " is not divisible by 256";
    instruction_dim = 256;
  }
  ICHECK((*inner_box_dim) % instruction_dim == 0)
      << "inner_box_dim: " << *inner_box_dim
      << " is not divisible by instruction_dim: " << instruction_dim;
  desc.smem_box.Set(0, PrimExpr(instruction_dim));

  int inner_box_dim_ =
      TMABytesFromElements(instruction_dim, shared_tensor->dtype);

  struct SwizzleCheck {
    int swizzle;
    int max_dim;
  };
  static const std::vector<SwizzleCheck> swizzle_checks = {
      {static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B), 32},
      {static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B), 64},
      {static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B), 128},
  };
  for (const auto &check : swizzle_checks) {
    if (desc.swizzle == check.swizzle && inner_box_dim_ > check.max_dim) {
      LOG(WARNING) << "TMA bulk copy cannot support a swizzled global layout "
                      "with inner_box_dim_ > "
                   << check.max_dim << ", will be fallback to normal copy";
      return LowerNormal(op, T, analyzer);
    }
  }

  Call create_descriptor =
      Call(DataType::Handle(), create_tma_descriptor(), desc.EncodeCallArgs());

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
    } else if (op.GetIsTmaCopy()) {
      LOG(FATAL) << "T.tma_copy() requires a barrier argument. "
                 << "Use T.tma_copy(src, dst, barrier=mbar[idx]).";
    } else if (T.AllocMBarrier) {
      barrier_base_id = T.AllocMBarrier(1);
      PrimExpr mbar_idx = IntImm(DataType::Int(32), barrier_base_id);
      mbar_handle = BufferLoad(T.mbarrier_buffer->value(), {mbar_idx});
    }
  }

  Array<PrimExpr> args;
  args.reserve(desc.rank + 4);
  args.push_back(create_descriptor);
  if (is_load)
    args.push_back(barrier_base_id >= 0 ? mbar_handle : PrimExpr(0));
  auto tma_op = is_load ? tma_load() : tma_store();

  Stmt tma_copy;
  PrimExpr total_elements = 1;
  for (auto e : desc.smem_box)
    total_elements *= e;

  if ((*inner_box_dim) != instruction_dim) {
    Var loop_var("i");
    int loop_extent = (*inner_box_dim) / instruction_dim;

    PrimExpr shared_addr = shared_tensor.access_ptr(
        is_load ? 2 : 1, DataType::Handle(), 1,
        shared_offset + total_elements * loop_var, total_elements);
    args.push_back(shared_addr);
    global_coords.Set(0, global_coords[0] + instruction_dim * loop_var);
    for (auto coord : global_coords)
      args.push_back(coord);
    int need_reduce = 0;
    if (!is_load)
      args.push_back(need_reduce);
    args.push_back(op.GetEvictionPolicy());
    Map<String, ObjectRef> ann_loop;
    if (is_cluster_barrier && TargetIsSm100(T.target) && is_load) {
      ann_loop.Set("use_2cta", IntImm(DataType::Int(32), 1));
    }
    tma_copy = For(loop_var, 0, loop_extent, ForKind::kUnrolled,
                   Evaluate(Call(DataType::Handle(), tma_op, args, ann_loop)));
  } else {
    PrimExpr shared_addr = shared_tensor.access_ptr(
        is_load ? 2 : 1, DataType::Handle(), 1, shared_offset, total_elements);
    args.push_back(shared_addr);
    for (auto coord : global_coords)
      args.push_back(coord);
    int need_reduce = 0;
    if (!is_load)
      args.push_back(need_reduce);
    args.push_back(op.GetEvictionPolicy());
    Map<String, ObjectRef> ann;
    if (TargetIsSm100(T.target) && is_load &&
        (annotations.find("use_2cta") != annotations.end() ||
         is_cluster_barrier)) {
      ann.Set("use_2cta", IntImm(DataType::Int(32), 1));
    }
    tma_copy = Evaluate(Call(DataType::Handle(), tma_op, args, ann));
  }

  if (!is_load) {
    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(tma_copy);
    seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_arrive(), {})));
    if (!op.GetIsTmaCopy()) {
      seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_wait(),
                                  {IntImm(DataType::Int(32), 0)})));
    }
    tma_copy = SeqStmt(std::move(seq));
  }

  if (is_load && barrier_base_id >= 0) {
    PrimExpr total_bytes;
    if ((*inner_box_dim) != instruction_dim) {
      int loop_extent = (*inner_box_dim) / instruction_dim;
      total_bytes = TMABytesFromElements(total_elements * loop_extent,
                                         shared_tensor->dtype);
    } else {
      total_bytes = TMABytesFromElements(total_elements, shared_tensor->dtype);
    }

    Stmt barrier_before_tma_stmt;
    Optional<Stmt> barrier_after_tma_stmt = std::nullopt;
    if (op.GetIsTmaCopy()) {
      if (is_cluster_barrier) {
        PrimExpr cluster_total_bytes =
            total_bytes * IntImm(DataType::Int(32), T.cluster_size);
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

    Stmt producer = IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent),
                               SeqStmt(producer_seq));

    if (op.GetIsTmaCopy()) {
      return producer;
    }

    Stmt wait_stmt =
        Evaluate(Call(DataType::Handle(), mbarrier_wait_parity(),
                      {mbar_handle, GetCopyMbarPhaseExpr(annotations, T)}));

    return SeqStmt({producer, wait_stmt});
  }

  tma_copy =
      IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent), tma_copy);

  return tma_copy;
}

Stmt Copy::LowerBulk1D(const CopyNode &op, const LowerArgs &T,
                       arith::Analyzer *analyzer, CopyInst copy_inst) {
  const Buffer &src = op.src;
  const Buffer &dst = op.dst;
  const Array<Range> &src_range = op.src_range;
  const Array<Range> &dst_range = op.dst_range;
  const Map<String, ObjectRef> &annotations = op.annotations;

  ICHECK(copy_inst == CopyInst::kBulkLoad1D ||
         copy_inst == CopyInst::kBulkStore1D);

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
    } else if (op.GetIsTmaCopy()) {
      LOG(FATAL) << "T.tma_copy() requires a barrier argument. "
                 << "Use T.tma_copy(src, dst, barrier=mbar[idx]).";
    } else if (T.AllocMBarrier) {
      barrier_base_id = T.AllocMBarrier(1);
      PrimExpr mbar_idx = IntImm(DataType::Int(32), barrier_base_id);
      mbar_handle = BufferLoad(T.mbarrier_buffer->value(), {mbar_idx});
    }
  }

  Stmt tma_copy;
  PrimExpr total_bytes = TMABytesFromElements(elements, shared_tensor->dtype);
  if (is_load) {
    PrimExpr mbar_arg = barrier_base_id >= 0 ? mbar_handle : PrimExpr(0);
    tma_copy = Evaluate(Call(DataType::Handle(), tma_load(),
                             {shared_addr, global_addr, mbar_arg, total_bytes,
                              op.GetEvictionPolicy()}));
  } else {
    int need_reduce = 0;
    tma_copy = Evaluate(Call(DataType::Handle(), tma_store(),
                             {global_addr, shared_addr, total_bytes,
                              need_reduce, op.GetEvictionPolicy()}));
  }

  if (!is_load) {
    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(tma_copy);
    seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_arrive(), {})));
    if (!op.GetIsTmaCopy()) {
      seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_wait(),
                                  {IntImm(DataType::Int(32), 0)})));
    }
    tma_copy = SeqStmt(std::move(seq));
  }

  if (is_load && barrier_base_id >= 0) {
    Stmt barrier_before_tma_stmt;
    Optional<Stmt> barrier_after_tma_stmt = std::nullopt;
    if (op.GetIsTmaCopy()) {
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

    Stmt producer = IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent),
                               SeqStmt(producer_seq));

    if (op.GetIsTmaCopy()) {
      return producer;
    }

    Stmt wait_stmt =
        Evaluate(Call(DataType::Handle(), mbarrier_wait_parity(),
                      {mbar_handle, GetCopyMbarPhaseExpr(annotations, T)}));

    return SeqStmt({producer, wait_stmt});
  }

  tma_copy =
      IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent), tma_copy);
  return tma_copy;
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
      cuda::Copy::SelectInst,
      cuda::Copy::Lower,
  });
  return true;
}

const bool cuda_copy_registered = RegisterCudaCopy();

} // namespace

} // namespace tl
} // namespace tvm
