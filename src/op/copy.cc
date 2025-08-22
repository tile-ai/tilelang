/*!
 * \file tl/op/copy.cc
 * \brief Define copy operator for various memory transfer strategies (Normal,
 *        Bulk/TMA, LDSM/STSM) and lowering logic for GPU code generation.
 *
 * This module is part of TVM TensorIR's Tensor Layout (TL) operations,
 * implementing memory copy operations that can target CPUs or GPUs with
 * optimization for different instructions like bulk copy, matrix load/store,
 * and Hopper's new TMA (Tensor Memory Accelerator).
 */

#include "copy.h"
#include "../target/utils.h"
#include "../transform/common/loop_fusion_utils.h"
#include "../transform/common/loop_parallel_transform_utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"

#include "../target/cuda.h"
#include "../target/utils.h"
#include "builtin.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Helper to map TVM's DataType to CUDA's CUtensorMapDataType enum value.
 * This function converts TVM data types to CUDA tensor map data types for TMA
 * operations.
 */
static int to_CUtensorMapDataType(DataType dtype) {
  CUtensorMapDataType tp;
  if (dtype.is_float()) {
    switch (dtype.bits()) {
    case 64:
      tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
      break;
    case 32:
      tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
      break;
    case 16:
      tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
      break;
    case 8:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else if (dtype.is_bfloat16()) {
    tp = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if (dtype.is_float8_e4m3() || dtype.is_float8_e5m2()) {
    tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if (dtype.is_int()) {
    switch (dtype.bits()) {
    case 64:
      tp = CU_TENSOR_MAP_DATA_TYPE_INT64;
      break;
    case 32:
      tp = CU_TENSOR_MAP_DATA_TYPE_INT32;
      break;
    case 16:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT16;
      break;
    case 8:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else if (dtype.is_uint()) {
    switch (dtype.bits()) {
    case 64:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT64;
      break;
    case 32:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT32;
      break;
    case 16:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT16;
      break;
    case 8:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else {
    ICHECK(0) << dtype;
  }
  return static_cast<int>(tp);
}

/*!
 * \brief Utility function to reverse an array.
 * This is commonly used to convert between row-major and column-major layouts.
 */
template <typename T> static Array<T> ReverseArray(Array<T> array) {
  return Array<T>{array.rbegin(), array.rend()};
}

/*!
 * \brief Constructor for Copy operator.
 * \param args Array of PrimExpr representing the arguments of the copy
 * operation. \param vmap BufferMap mapping original buffer names to new buffer
 * names.
 */
Copy::Copy(Array<PrimExpr> args, BufferMap vmap) : args_(args) {
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto expr = args[i];
    auto call = expr.as<CallNode>();
    ICHECK(call);
    auto region = RegionOp(call->args, vmap);
    rgs[i] = region.GetRanges();
    bf[i] = region.GetBuffer();
  }
  std::tie(this->src, this->dst) = std::tie(bf[0], bf[1]);
  std::tie(this->src_range, this->dst_range) = std::tie(rgs[0], rgs[1]);
  if (args.size() >= 3) {
    auto coalesced_width = Downcast<IntImm>(args[2]);
    if (coalesced_width->value > 0) {
      this->coalesced_width = coalesced_width;
    }
  }
  if (args.size() >= 4) {
    this->disable_tma = Downcast<Bool>(args[3]);
  }
  if (args.size() >= 5) {
    this->eviction_policy = args[4].as<IntImmNode>()->value;
  }
}

/*!
 * \brief Create iterator variables for the copy operation.
 * This function creates iteration variables for dimensions that have extent
 * > 1. \return Array of IterVar representing the iterator variables for the
 * copy operation.
 */
Array<IterVar> Copy::MakeIterVars() const {
  Array<IterVar> loop_vars;
  size_t idx = 0;
  for (size_t i = 0; i < src_range.size(); i++) {
    if (is_one(src_range[i]->extent))
      continue;
    Var var = Var(std::string{char('i' + idx)}, src_range[i]->extent->dtype);
    idx++;
    loop_vars.push_back(
        {Range(0, src_range[i]->extent), var, IterVarType::kDataPar});
  }
  return loop_vars;
}

/*!
 * \brief Create indices for the copy operation.
 * This function generates the actual index expressions for accessing source or
 * destination buffers. For dimensions with extent=1, it uses the range minimum;
 * for others, it adds the iteration variable. \param ivs Array of IterVar
 * returned by MakeIterVars(). \param src_dst 0 for src_indices, 1 for
 * dst_indices. \return Array of PrimExpr representing the indices for the copy
 * operation.
 */
Array<PrimExpr> Copy::MakeIndices(const Array<IterVar> &ivs,
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

/*!
 * \brief Create predicate for the copy operation.
 * This function generates boundary checks to ensure memory access safety.
 * It creates conditions like (min + iv) < extent and (min + iv) >= 0 for each
 * dimension. \param analyzer Arithmetic analyzer for simplification. \param ivs
 * Array of IterVar. \param extents Array of PrimExpr representing the extents
 * of the copy operation. \param src_dst 0 for src_indices, 1 for dst_indices.
 * \return PrimExpr representing the predicate for the copy operation.
 */
PrimExpr Copy::MakePredicate(arith::Analyzer *analyzer,
                             const Array<IterVar> &ivs, Array<PrimExpr> extents,
                             int src_dst) const {
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

/*!
 * \brief Create SIMT loop for the copy operation.
 * This function generates a single-threaded loop structure for the copy
 * operation. It handles scalar copies (single element) and multi-dimensional
 * copies with nested loops. \param analyzer Arithmetic analyzer for
 * simplification. \return For representing the SIMT loop for the copy
 * operation.
 */
For Copy::MakeSIMTLoop(arith::Analyzer *analyzer) const {
  Array<IterVar> loop_vars = MakeIterVars();
  bool is_scalar = loop_vars.size() == 0;
  if (is_scalar) {
    return For(Var("i"), 0, 1, ForKind::kSerial,
               BufferStore(dst, BufferLoad(src, {0}), {0}));
  }

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
  for (int i = loop_vars.size() - 1; i >= 0; i--) {
    Map<String, ObjectRef> annotations = {};
    if (coalesced_width.defined()) {
      annotations.Set("coalesced_width", coalesced_width);
    }
    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body, std::nullopt, annotations);
  }
  return Downcast<For>(body);
}

/*!
 * \brief Compute linear layout for TMA copy.
 * This function creates a linear layout transformation for shared memory in TMA
 * operations. It transforms multi-dimensional indices into a linear address
 * using a 256-element block pattern. The transformation follows: [i, j] ->
 * [i//256, j//256, i%256, j%256] \param shared_tensor Buffer representing the
 * shared tensor. \return Layout representing the linear layout for the TMA
 * copy.
 */
Layout Copy::ComputeLinearLayout(const Buffer &shared_tensor) const {
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

/*!
 * \brief Infer layout for the copy operation.
 * This function determines the optimal memory layout for the copy operation
 * based on the target architecture. For bulk load/store operations, it may
 * apply swizzling layouts for better performance. For LDSM/STSM operations, it
 * uses register layout inference from the underlying parallel op. \param T
 * LayoutInferArgs containing target and layout map. \param level InferLevel
 * indicating the level of layout inference. \return LayoutMap containing the
 * inferred layout.
 */
LayoutMap Copy::InferLayout(const LayoutInferArgs &T, InferLevel level) {
  auto target = T.target;
  using namespace tvm::transform;
  PassContext pass_ctx = PassContext::Current();
  bool disable_tma_lower =
      pass_ctx->GetConfig<bool>(kDisableTMALower, false).value();
  auto copy_inst = GetCopyInst(target, disable_tma_lower || disable_tma);
  if (copy_inst == CopyInst::kBulkLoad || copy_inst == CopyInst::kBulkStore) {
    // if can apply swizzling, we skip layout inference
    // for bulk load/store, we can directly apply the layout of normal copy
    // This must be a global/shared layout, so we can skip the parallel op
    // layout inference (parallel layout inference only annotate the loop layout
    // and the register layout).
    bool is_load = copy_inst == CopyInst::kBulkLoad;
    Buffer global_tensor = is_load ? src : dst;
    Buffer shared_tensor = is_load ? dst : src;
    // check shared layout is non-swizzle
    // skip layout inference if shared layout is already annotated
    if (level == InferLevel::kFree && !T.layout_map.count(shared_tensor)) {
      // create a new layout map for tma linear layout
      Layout linear_layout = ComputeLinearLayout(shared_tensor);
      return Map<Buffer, Layout>({{shared_tensor, linear_layout}});
    }
  }

  // for LDSM/STSM, the layout was deduced from register layout
  // so we can directly apply the layout of normal copy
  // Use parallel op to infer the layout
  if (!par_op_) {
    arith::Analyzer analyzer;
    par_op_ = std::make_unique<ParallelOp>(MakeSIMTLoop(&analyzer));
  }
  return par_op_->InferLayout(T, level);
}

/*!
 * \brief Check if the copy operation is a bulk load.
 * This function verifies if the copy operation can be implemented using CUDA's
 * Bulk Load instruction. Requirements include: target supports bulk copy,
 * source is global memory, destination is shared.dyn, and both buffers have the
 * same data type. \param target Target device. \return True if the copy
 * operation is a bulk load, false otherwise.
 */
bool Copy::CheckBulkLoad(Target target) const {
  // 1. arch must have bulk copy support
  if (!TargetHasBulkCopy(target))
    return false;
  // 2. src and dst must be global and shared
  if (src.scope() != "global" ||
      (dst.scope() != "shared.dyn" && dst.scope() != "shared"))
    return false;
  // 3. check shape.
  // TODO(lei): validate if we can utilize tma under this shape.
  // 4. src and dst must have the same dtype
  if (src->dtype != dst->dtype) {
    LOG(WARNING) << "src and dst must have the same dtype for tma load "
                 << src->name << " vs. " << dst->name << " dtype " << src->dtype
                 << " vs. " << dst->dtype << " will be fallback to normal copy";
    return false;
  }
  return true;
}

/*!
 * \brief Check if the copy operation is a bulk store.
 * This function verifies if the copy operation can be implemented using CUDA's
 * Bulk Store instruction. Requirements include: target supports bulk copy,
 * source is shared.dyn, destination is global memory, and both buffers have the
 * same data type. \param target Target device. \return True if the copy
 * operation is a bulk store, false otherwise.
 */
bool Copy::CheckBulkStore(Target target) const {
  // 1. arch must have bulk copy support
  if (!TargetHasBulkCopy(target))
    return false;
  // 2. src and dst must be shared.dyn and local.fragment
  if ((src.scope() != "shared.dyn" && src.scope() != "shared") ||
      dst.scope() != "global")
    return false;
  // 3. check shape.
  // TODO(lei): validate if we can utilize tma under this shape.
  // 4. src and dst must have the same dtype
  if (src->dtype != dst->dtype) {
    LOG(WARNING) << "src and dst must have the same dtype for tma store "
                 << src->name << " vs. " << dst->name << " dtype " << src->dtype
                 << " vs. " << dst->dtype << " will be fallback to normal copy";
    return false;
  }
  return true;
}

/*!
 * \brief Check if the copy operation is a LDSM copy.
 * This function verifies if the copy operation can be implemented using CUDA's
 * Load Matrix (LDSM) instruction. Requirements include: target supports
 * LDMATRIX, source is shared.dyn, destination is local.fragment. \param target
 * Target device. \return True if the copy operation is a LDSM copy, false
 * otherwise.
 */
bool Copy::CheckLDSMCopy(Target target) const {
  return TargetHasLdmatrix(target) &&
         (src.scope() == "shared.dyn" || src.scope() == "shared") &&
         dst.scope() == "local.fragment";
}

/*!
 * \brief Check if the copy operation is a STSM copy.
 * This function verifies if the copy operation can be implemented using CUDA's
 * Store Matrix (STSM) instruction. Requirements include: target supports
 * STMATRIX, source is local.fragment, destination is shared.dyn. \param target
 * Target device. \return True if the copy operation is a STSM copy, false
 * otherwise.
 */
bool Copy::CheckSTSMCopy(Target target) const {
  return TargetHasStmatrix(target) && src.scope() == "local.fragment" &&
         (dst.scope() == "shared.dyn" || dst.scope() == "shared");
}

/*!
 * \brief Get the copy instruction type.
 * This function determines the most appropriate copy instruction based on the
 * target architecture and buffer memory scopes. It checks for specialized
 * instructions (TMA, LDSM, STSM) in order of preference, falling back to normal
 * copy if no specialized instruction is applicable. \param target Target
 * device. \return CopyInst representing the copy instruction type.
 */
Copy::CopyInst Copy::GetCopyInst(Target target, bool disable_tma_lower) const {
  // disable_tma_lower is from pass_configs
  // when tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER is True,
  // we will not use tma for bulk load/store
  if (!disable_tma_lower && CheckBulkLoad(target)) {
    return CopyInst::kBulkLoad;
  } else if (!disable_tma_lower && CheckBulkStore(target)) {
    return CopyInst::kBulkStore;
  } else if (CheckLDSMCopy(target)) {
    return CopyInst::kLDSM;
  } else if (CheckSTSMCopy(target)) {
    return CopyInst::kSTSM;
  } else {
    return CopyInst::kNormal;
  }
}

/*!
 * \brief Lower the copy operation to PTX code.
 * This function converts the high-level copy operation into low-level PTX
 * instructions. It dispatches to specialized lowering functions based on the
 * determined copy instruction type:
 * - Bulk Load/Store: Uses Tensor Memory Accelerator (TMA) instructions
 * - LDSM/STSM: Uses matrix load/store instructions for tensor cores
 * - Normal: Uses standard load/store operations with loop transformations
 * \param T LowerArgs containing target and layout map.
 * \param analyzer Arithmetic analyzer for simplification.
 * \return Stmt representing the PTX code for the copy operation.
 */
Stmt Copy::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  Target target = T.target;
  using namespace tvm::transform;
  PassContext pass_ctx = PassContext::Current();
  bool disable_tma_lower =
      pass_ctx->GetConfig<bool>(kDisableTMALower, false).value();
  auto copy_inst = GetCopyInst(target, disable_tma_lower || disable_tma);
  if (copy_inst == CopyInst::kBulkLoad || copy_inst == CopyInst::kBulkStore) {
    auto bulk_copy = LowerBulkCopy(T, analyzer, copy_inst);
    ICHECK(bulk_copy.defined()) << "Failed to lower bulk copy";
    return bulk_copy;
  } else if (copy_inst == CopyInst::kLDSM || copy_inst == CopyInst::kSTSM) {
    auto ldsm_copy = LowerLDSMCopy(T, analyzer, copy_inst);
    ICHECK(ldsm_copy.defined()) << "Failed to lower ptx matrix copy";
    return ldsm_copy;
  } else if (copy_inst == CopyInst::kNormal) {
    return LowerNormalCopy(T, analyzer);
  } else {
    LOG(FATAL) << "Unsupported copy inst " << static_cast<int>(copy_inst);
  }
}

/*!
 * \brief Lower the copy operation to a normal copy.
 * This function generates standard load/store operations for targets that don't
 * support specialized copy instructions. It applies loop fusion,
 * parallelization, and vectorization transformations to optimize performance on
 * both CPU and GPU targets. \param T LowerArgs containing target and layout
 * map. \param analyzer Arithmetic analyzer for simplification. \return Stmt
 * representing the normal copy code.
 */
Stmt Copy::LowerNormalCopy(const LowerArgs &T,
                           arith::Analyzer *analyzer) const {
  bool is_cpu_target = T.target->GetTargetDeviceType() == kDLCPU;
  auto simt_loop = MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));

  auto transformed_loop =
      Downcast<For>(ParallelLoopTransformer::Substitute(fused_loop));

  For vectorized_thread_loop;
  auto par_op = std::make_unique<ParallelOp>(transformed_loop);

  if (is_cpu_target) {
    vectorized_thread_loop = VectorizeLoop(transformed_loop);
  } else {
    std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                      InferLevel::kFree};
    for (auto level : levels) {
      par_op->InferLayout(
          {T.target, T.thread_bounds, T.layout_map, T.buffer_remap}, level);
    }
    auto loop_layout = par_op->GetLoopLayout();
    auto thread_var = T.thread_var;
    auto thread_loop =
        PartitionLoop(par_op->GetRoot(), T.thread_var, analyzer, loop_layout);
    vectorized_thread_loop = VectorizeLoop(thread_loop);
  }

  if (par_op->GetPredicate(T.thread_var).defined()) {
    return IfThenElse(par_op->GetPredicate(T.thread_var).value(),
                      vectorized_thread_loop);
  }
  return vectorized_thread_loop;
}

/*!
 * \brief Lower the copy operation to LDSM/STSM copy.
 * This function generates PTX code for matrix load/store operations
 * (LDSM/STSM). It handles 8x8 fragment layout validation, shared memory stride
 * checking, and generates optimized matrix transfer instructions for tensor
 * cores. Falls back to normal copy if layout constraints are not satisfied.
 * \param T LowerArgs containing target and layout map.
 * \param analyzer Arithmetic analyzer for simplification.
 * \param copy_inst CopyInst representing the copy instruction type.
 * \return Stmt representing the LDSM/STSM copy code.
 */
Stmt Copy::LowerLDSMCopy(const LowerArgs &T, arith::Analyzer *analyzer,
                         CopyInst copy_inst) const {
  ICHECK(copy_inst == CopyInst::kLDSM || copy_inst == CopyInst::kSTSM)
      << "Invalid copy inst " << static_cast<int>(copy_inst);
  bool is_ldmatrix = copy_inst == CopyInst::kLDSM;

  // Check no predicates
  Array<IterVar> loop_vars = MakeIterVars();
  if (loop_vars.size() < 2) {
    // cannot support 1-d case
    return LowerNormalCopy(T, analyzer);
  }
  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);
  PrimExpr src_predicate = MakePredicate(analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate = MakePredicate(analyzer, loop_vars, dst->shape, 1);
  if (src_predicate.defined() || dst_predicate.defined()) {
    // stmatrix and ldmatrix can only support no predicate
    return LowerNormalCopy(T, analyzer);
  }

  Buffer shared_tensor = is_ldmatrix ? src : dst;
  Buffer local_tensor = is_ldmatrix ? dst : src;

  Array<PrimExpr> local_indices = MakeIndices(loop_vars, is_ldmatrix ? 1 : 0);
  Fragment local_layout = Downcast<Fragment>(T.layout_map[local_tensor]);
  Array<PrimExpr> local_indices_transformed =
      local_layout->Forward(local_indices);
  local_tensor = T.buffer_remap[local_tensor];
  // currently only support 1-d case
  if (local_layout->OutputDim() != 1) {
    // TMA ldmatrix/stmatrix cannot support non-1-d layout, will be fallback to
    // normal copy
    return LowerNormalCopy(T, analyzer);
  }

  Array<PrimExpr> shared_indices = MakeIndices(loop_vars, is_ldmatrix ? 0 : 1);
  Array<PrimExpr> shared_indices_transformed = shared_indices;
  Layout shared_layout;
  if (T.buffer_remap.count(shared_tensor)) {
    shared_layout = T.layout_map[shared_tensor];
    shared_tensor = T.buffer_remap[shared_tensor];
    shared_indices_transformed = shared_layout->Forward(shared_indices);
  }

  // Check local_layout follows 8x8 layout
  // LDSM/STSM instructions require 8x8 matrix fragment layout
  // This matches the warp-level matrix multiplication pattern used in tensor
  // cores We check both normal and transposed layouts to support different
  // access patterns
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
      IndiceCanVectorize(local_indices_flattened, col_var->var,
                         col_var->dom->extent, 2, analyzer)) {
    is_transposed = false;
  } else if (analyzer->CanProveEqual(matrix_8x8_thread_map_trans,
                                     local_layout_thread_map) &&
             IndiceCanVectorize(local_indices_flattened, row_var->var,
                                row_var->dom->extent, 2, analyzer)) {
    is_transposed = true;
  } else {
    // TMA ldmatrix/stmatrix cannot support non-8x8 layout, will be fallback to
    // normal copy
    return LowerNormalCopy(T, analyzer);
  }
  // Check shared_layout is 16 bytes continuous
  // LDSM/STSM instructions require 16-byte aligned data (half-precision floats)
  // This is a hardware constraint for matrix load/store operations
  if (shared_tensor->dtype.bytes() != 2) {
    // TMA ldmatrix/stmatrix cannot support non-16 bytes continuous layout, will
    // be fallback to normal copy
    return LowerNormalCopy(T, analyzer);
  }
  PrimExpr flattened_indice =
      shared_tensor.OffsetOf(shared_indices_transformed).back();
  if (!IndiceCanVectorize(flattened_indice, loop_vars.back()->var,
                          loop_vars.back()->dom->extent, 8, analyzer)) {
    // TMA ldmatrix/stmatrix cannot support non-16 bytes continuous layout, will
    // be fallback to normal copy
    return LowerNormalCopy(T, analyzer);
  }

  // Can only support local_range to be a full range
  for (size_t i = 0; i < dst_range.size(); i++) {
    if (!is_zero(dst_range[i]->min) ||
        !analyzer->CanProveEqual(dst_range[i]->extent, dst->shape[i]))
      // TMA ldmatrix/stmatrix cannot support non-full range, will be fallback
      // to normal copy
      return LowerNormalCopy(T, analyzer);
  }

  // Do the lowering here, try vectorized ldmatrix/stmatrix by 4/2/1
  PrimExpr extent = local_tensor->shape[0];
  int num = 1;
  if (analyzer->CanProveEqual(FloorMod(extent, 8), 0))
    num = 4;
  else if (analyzer->CanProveEqual(FloorMod(extent, 4), 0))
    num = 2;

  Array<PrimExpr> args;
  const Op &op = is_ldmatrix ? tl::ptx_ldmatrix() : tl::ptx_stmatrix();
  args.push_back(static_cast<int>(is_transposed));
  args.push_back(num);

  // Create shared address with regard to local address
  // if not transpose
  // coords = Inverse(base + 2 * (thread / 8) % num, warp + (thread % 8) * 4))
  // if transpose
  // coords = Inverse(base + 2 * (thread / 8) % num + thread % 2, warp + thread
  // % 8 / 2)
  Var local_iter("i");
  Layout inv = local_layout->Inverse();
  Array<PrimExpr> shared_coords;
  PrimExpr warp = FloorDiv(T.thread_var, 32) * 32;
  if (!is_transposed)
    shared_coords = inv->Forward(
        {local_iter * 2 * num + 2 * FloorMod(FloorDiv(T.thread_var, 8), num),
         warp + FloorMod(T.thread_var, 8) * 4});
  else
    shared_coords = inv->Forward(
        {local_iter * 2 * num + 2 * FloorMod(FloorDiv(T.thread_var, 8), num) +
             FloorMod(T.thread_var, 2),
         warp + FloorDiv(FloorMod(T.thread_var, 8), 2)});
  shared_coords.pop_back(); // remove rep
  if (shared_layout.defined())
    shared_coords = shared_layout->Forward(shared_coords);
  PrimExpr shared_addr = shared_tensor.access_ptr(
      is_ldmatrix ? 1 : 2, DataType::Handle(), 1,
      shared_tensor.OffsetOf(shared_coords).back(), PrimExpr(2 * num));
  args.push_back(shared_addr);

  if (is_ldmatrix) {
    // Can only support same dtype for ldmatrx
    if (local_tensor->dtype != shared_tensor->dtype) {
      // TMA ldmatrix cannot support different dtype, will be fallback to normal
      // copy
      return LowerNormalCopy(T, analyzer);
    }
    PrimExpr local_addr = local_tensor.access_ptr(
        2, DataType::Handle(), 1, local_iter * 2 * num, PrimExpr(2 * num));
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

  auto body = Evaluate(Call(DataType::Handle(), op, args));
  For for_node =
      For(local_iter, 0, FloorDiv(extent, 2 * num), ForKind::kSerial, body);
  for_node = LoopPragmaUnroll(for_node);
  auto range = T.thread_bounds;
  if (range.defined()) {
    auto thread_var = T.thread_var;
    auto thread_var_with_offset = thread_var - range->min;
    for_node.CopyOnWrite()->body =
        Substitute(for_node->body, {{thread_var, thread_var_with_offset}});
  }
  return for_node;
}

/*!
 * \brief Lower the copy operation to bulk copy using TMA.
 * This function generates PTX code for Tensor Memory Accelerator (TMA) bulk
 * copy operations. It creates TMA descriptors, handles shared memory layout
 * detection (including swizzling), and generates optimized bulk load/store
 * instructions for Hopper architecture. Falls back to normal copy if layout or
 * shape constraints are not satisfied. \param T LowerArgs containing target and
 * layout map. \param analyzer Arithmetic analyzer for simplification. \param
 * copy_inst CopyInst representing the copy instruction type. \return Stmt
 * representing the bulk copy code.
 */
Stmt Copy::LowerBulkCopy(const LowerArgs &T, arith::Analyzer *analyzer,
                         CopyInst copy_inst) const {
  ICHECK(copy_inst == CopyInst::kBulkLoad || copy_inst == CopyInst::kBulkStore)
      << "Invalid copy inst " << static_cast<int>(copy_inst);
  bool is_load = copy_inst == CopyInst::kBulkLoad;
  Buffer global_tensor = is_load ? src : dst;
  Buffer shared_tensor = is_load ? dst : src;
  Array<Range> global_range = is_load ? src_range : dst_range;
  Array<Range> shared_range = is_load ? dst_range : src_range;
  // TMA bulk copy cannot support a non-swizzled global layout, will be fallback
  // to normal copy
  if (T.layout_map.count(global_tensor)) {
    LOG(WARNING) << "TMA bulk copy cannot support a non-swizzled global "
                    "layout, fallback to normal copy.";
    return LowerNormalCopy(T, analyzer);
  }

  // linear layout must be computed before remapping
  auto linear_layout = ComputeLinearLayout(shared_tensor);

  Array<PrimExpr> indices;
  for (auto r : shared_range)
    indices.push_back(r->min);
  std::vector<PrimExpr> strides;
  PrimExpr stride = 1;
  for (size_t i = 0; i < shared_tensor->shape.size(); i++) {
    auto s = shared_tensor->shape[shared_tensor->shape.size() - i - 1];
    strides.insert(strides.begin(), stride);
    stride *= s;
  }

  ICHECK(strides.size() == indices.size())
      << "strides.size() != indices.size()" << strides.size() << " "
      << indices.size();
  PrimExpr offset = 0;
  for (size_t i = 0; i < indices.size(); i++) {
    offset += indices[i] * strides[i];
  }
  Layout shared_layout;
  if (T.layout_map.count(shared_tensor)) {
    shared_layout = T.layout_map[shared_tensor];
    shared_tensor = T.buffer_remap[shared_tensor];
  }

  TMADesc desc;
  // Verify copy rank
  desc.rank = global_tensor->shape.size();
  ICHECK(desc.rank >= 1 && desc.rank <= 5) << desc.rank;

  // Verify datatype
  ICHECK(global_tensor->dtype == shared_tensor->dtype)
      << "Copy between buffer " << global_tensor->name << " and "
      << shared_tensor->name << " with different data type "
      << global_tensor->dtype << " and " << shared_tensor->dtype;

  desc.data_type = to_CUtensorMapDataType(global_tensor->dtype);

  // Global Tensor Shape and Stride
  desc.global_addr = global_tensor->data;
  desc.global_shape = ReverseArray(global_tensor->shape);
  Array<PrimExpr> global_coords =
      ReverseArray(global_range.Map([](Range r) { return r->min; }));
  if (!global_tensor->strides.empty()) {
    desc.global_stride = ReverseArray(global_tensor->strides);
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
  desc.global_stride = desc.global_stride.Map([&](PrimExpr e) {
    return cast(DataType::Int(64), e) * global_tensor->dtype.bytes();
  });
  for (size_t i{1}; i < desc.global_stride.size(); i++) {
    auto stride = desc.global_stride[i].as<IntImmNode>();
    if (stride != nullptr) {
      // otherwise, the stride is symbolic, we need to check in future with
      // assumptions
      if (stride->value % 16 != 0 || stride->value >= (1ULL << 40)) {
        LOG(WARNING) << "TMA bulk copy cannot support a global stride of "
                     << desc.global_stride[i] << ", fallback to normal copy.";
        return LowerNormalCopy(T, analyzer);
      }
    }
  }

  // Smem Box
  // check smem range and global range is legal
  auto s_range_idx = 0;
  for (size_t i = 0; i < global_range.size(); i++) {
    auto g_range = global_range[i];
    if (is_one(g_range->extent)) {
      continue;
    }
    // skip one range if it is 1
    // in case of global range is [128, 64], while shared range is [1, 128, 64]
    // A_shared[0, :, :].
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
  // L2 & OOB
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  // Detect smem layout
  // Shared memory swizzling is crucial for TMA performance
  // It determines how data is arranged in shared memory banks to minimize bank
  // conflicts Different swizzle patterns (32B, 64B, 128B) offer different
  // trade-offs between access efficiency and memory usage
  desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);
  if (!shared_layout.defined()) {
    desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
  } else if (StructuralEqual()(shared_layout, linear_layout)) {
    desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
  } else {
    ICHECK(shared_layout->InputDim() == 2) << "Cannot detect TMA layout.";
    auto stride = as_const_int(shared_layout->InputShape()[0]);
    auto continuous = as_const_int(shared_layout->InputShape()[1]);
    ICHECK(stride != nullptr && continuous != nullptr);
    // We also need to check if the shape satisfies the following doc:
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
    if (StructuralEqual()(shared_layout, makeQuarterBankSwizzleLayout(
                                             *stride, *continuous,
                                             shared_tensor->dtype.bits()))) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B);
    } else if (StructuralEqual()(
                   shared_layout,
                   makeHalfBankSwizzleLayout(*stride, *continuous,
                                             shared_tensor->dtype.bits()))) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B);
    } else if (StructuralEqual()(
                   shared_layout,
                   makeFullBankSwizzleLayout(*stride, *continuous,
                                             shared_tensor->dtype.bits()))) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B);
    } else if (StructuralEqual()(
                   shared_layout,
                   makeGemmABLayoutPadded(*stride, *continuous,
                                          shared_tensor->dtype.bits()))) {
      LOG(WARNING) << "Bulk copy cannot support a padded layout for src: "
                   << src->name << ", dst: " << dst->name
                   << ", fallback to normal copy";
      return LowerNormalCopy(T, analyzer);
    } else {
      LOG(WARNING) << "Came across unsupported swizzle layout for src: "
                   << src->name << ", dst: " << dst->name
                   << ", fallback to normal copy";
      return LowerNormalCopy(T, analyzer);
    }
  }

  auto inner_box_dim = as_const_int(desc.smem_box[0]);
  ICHECK(inner_box_dim != nullptr);
  int instruction_dim = *inner_box_dim;
  if (desc.swizzle == static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B)) {
    instruction_dim = 64 / src->dtype.bytes();
  } else if (desc.swizzle == static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B)) {
    instruction_dim = 128 / src->dtype.bytes();
  }
  if (instruction_dim > 256) {
    // smem_box dim must be in [0, 256]
    // if is 512, we need to split the copy into two parts
    ICHECK((*inner_box_dim) % 256 == 0)
        << "inner_box_dim: " << *inner_box_dim << " is not divisible by 256";
    instruction_dim = 256;
  }
  ICHECK((*inner_box_dim) % instruction_dim == 0)
      << "inner_box_dim: " << *inner_box_dim
      << " is not divisible by instruction_dim: " << instruction_dim;
  desc.smem_box.Set(0, PrimExpr(instruction_dim));

  int inner_box_dim_ = instruction_dim * shared_tensor->dtype.bytes();

  // Check inner_box_dim_ for each swizzle type in a cleaner way
  struct SwizzleCheck {
    int swizzle;
    int max_dim;
  };
  static const SwizzleCheck swizzle_checks[] = {
      {static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B), 32},
      {static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B), 64},
      {static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B), 128},
  };
  for (const auto &check : swizzle_checks) {
    if (desc.swizzle == check.swizzle && inner_box_dim_ > check.max_dim) {
      LOG(WARNING) << "TMA bulk copy cannot support a swizzled global layout "
                      "with inner_box_dim_ > "
                   << check.max_dim << ", will be fallback to normal copy";
      return LowerNormalCopy(T, analyzer);
    }
  }

  Call create_descriptor =
      Call(DataType::Handle(), create_tma_descriptor(), desc.EncodeCallArgs());

  Array<PrimExpr> args;
  args.reserve(desc.rank + 4);
  args.push_back(create_descriptor);
  if (is_load)
    args.push_back(0); // mbarrier id placeholder
  auto op = is_load ? tma_load() : tma_store();

  Stmt tma_copy;
  PrimExpr total_elements = 1;
  for (auto e : desc.smem_box)
    total_elements *= e;

  if ((*inner_box_dim) != instruction_dim) {
    Var loop_var("i");
    int loop_extent = (*inner_box_dim) / instruction_dim;

    PrimExpr shared_addr = shared_tensor.access_ptr(
        is_load ? 2 : 1, DataType::Handle(), 1,
        offset + total_elements * loop_var, total_elements);
    args.push_back(shared_addr);
    global_coords.Set(0, global_coords[0] + instruction_dim * loop_var);
    for (auto coord : global_coords)
      args.push_back(coord);
    args.push_back(this->eviction_policy);
    tma_copy = For(loop_var, 0, loop_extent, ForKind::kUnrolled,
                   Evaluate(Call(DataType::Handle(), op, args)));
  } else {
    PrimExpr shared_addr = shared_tensor.access_ptr(
        is_load ? 2 : 1, DataType::Handle(), 1, offset, total_elements);
    args.push_back(shared_addr);
    for (auto coord : global_coords)
      args.push_back(coord);
    args.push_back(this->eviction_policy);
    tma_copy = Evaluate(Call(DataType::Handle(), op, args));
  }
  tma_copy = IfThenElse(EQ(T.thread_var, T.thread_bounds->min), tma_copy);

  return tma_copy;
}

/*!
 * \brief Encode the TMA descriptor into an array of PrimExpr.
 * This function serializes the TMA descriptor fields into a format suitable for
 * passing to the create_tma_descriptor() builtin function. The encoding follows
 * the expected argument order for the TMA descriptor creation.
 * \return Array of PrimExpr representing the encoded TMA descriptor.
 */
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

/*!
 * \brief Constructor for Conv2DIm2ColOp.
 * This operation performs im2col transformation for 2D convolution on GPU using
 * TMA. It extracts patches from the input tensor and rearranges them for
 * efficient matrix multiplication. \param args Array of PrimExpr representing
 * the arguments of the Conv2DIm2ColOp. \param vmap BufferMap mapping original
 * buffer names to new buffer names.
 */
Conv2DIm2ColOp::Conv2DIm2ColOp(Array<PrimExpr> args, BufferMap vmap) {
  src = vmap[GetVarFromAccessPtr(args[0])];
  dst = vmap[GetVarFromAccessPtr(args[1])];
  nhw_step = args[2];
  c_step = args[3];
  kernel = args[4].as<IntImm>().value()->value;
  stride = args[5].as<IntImm>().value()->value;
  dilation = args[6].as<IntImm>().value()->value;
  padding = args[7].as<IntImm>().value()->value;
  eviction_policy = args[8].as<IntImm>().value()->value;
}

/*!
 * \brief Lower the Conv2DIm2ColOp to PTX code.
 * This function generates optimized im2col transformation using TMA
 * instructions. It creates a TMA descriptor for the im2col operation, handling
 * convolution parameters like kernel size, stride, padding, and dilation. The
 * operation is optimized for Hopper architecture with support for different
 * shared memory layouts. \param T LowerArgs containing target and layout map.
 * \param analyzer Arithmetic analyzer for simplification.
 * \return Stmt representing the PTX code for the Conv2DIm2ColOp.
 */
Stmt Conv2DIm2ColOp::Lower(const LowerArgs &T,
                           arith::Analyzer *analyzer) const {
  ICHECK(TargetIsHopper(T.target));
  ICHECK(src.scope() == "global" &&
         (dst.scope() == "shared.dyn" || dst.scope() == "shared"));
  ICHECK(src->shape.size() == 4);
  ICHECK(dst->shape.size() == 2);
  ICHECK(src->dtype == dst->dtype);
  Layout shared_layout;
  if (T.layout_map.count(dst)) {
    shared_layout = T.layout_map[dst];
  }

  TMAIm2ColDesc desc;
  desc.rank = src->shape.size();
  desc.data_type = to_CUtensorMapDataType(src->dtype);
  desc.global_addr = src->data;
  desc.global_shape = ReverseArray(src->shape);

  if (!src->strides.empty()) {
    desc.global_stride = ReverseArray(src->strides);
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
  desc.global_stride = desc.global_stride.Map([&](PrimExpr e) {
    return cast(DataType::Int(64), e) * src->dtype.bytes();
  });
  desc.elem_stride = {1, stride, stride, 1};
  desc.lower_corner = {-padding, -padding};
  desc.upper_corner = {-padding, -padding};
  desc.smem_box_pixel = Downcast<IntImm>(dst->shape[0])->value;
  desc.smem_box_channel = Downcast<IntImm>(dst->shape[1])->value;
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);
  if (!shared_layout.defined()) {
    desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
  } else {
    ICHECK(shared_layout->InputDim() == 2) << "Cannot detect TMA layout.";
    auto stride = as_const_int(shared_layout->InputShape()[0]);
    auto continuous = as_const_int(shared_layout->InputShape()[1]);
    ICHECK(stride != nullptr && continuous != nullptr);

    if (StructuralEqual()(shared_layout,
                          makeQuarterBankSwizzleLayout(*stride, *continuous,
                                                       dst->dtype.bits()))) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B);
    } else if (StructuralEqual()(shared_layout, makeHalfBankSwizzleLayout(
                                                    *stride, *continuous,
                                                    dst->dtype.bits()))) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B);
    } else if (StructuralEqual()(shared_layout, makeFullBankSwizzleLayout(
                                                    *stride, *continuous,
                                                    dst->dtype.bits()))) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B);
    } else {
      ICHECK(0) << "Cannot detect TMA layout.";
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
      FloorMod(c_step * desc.smem_box_channel, desc.global_shape[0]));
  image_offset.push_back(
      dilation *
      FloorMod(FloorDiv(c_step * desc.smem_box_channel, desc.global_shape[0]),
               kernel));
  image_offset.push_back(dilation * FloorDiv(c_step * desc.smem_box_channel,
                                             desc.global_shape[0] * kernel));

  PrimExpr h_dim =
      FloorDiv(src->shape[1] + 2 * padding - (kernel - 1) * dilation - 1,
               stride) +
      1;
  PrimExpr w_dim =
      FloorDiv(src->shape[2] + 2 * padding - (kernel - 1) * dilation - 1,
               stride) +
      1;
  global_coords.push_back(
      stride * FloorMod(nhw_step * desc.smem_box_pixel, w_dim) - padding);
  global_coords.push_back(
      stride *
          FloorMod(FloorDiv(nhw_step * desc.smem_box_pixel, w_dim), h_dim) -
      padding);
  global_coords.push_back(
      FloorDiv(nhw_step * desc.smem_box_pixel, w_dim * h_dim));

  Array<PrimExpr> args;
  args.reserve(desc.rank * 2 + 2);
  args.push_back(create_desc);
  args.push_back(0); // mbar placeholder
  auto dst_buffer = T.buffer_remap.count(dst) ? T.buffer_remap[dst] : dst;
  auto shared_addr = dst_buffer.access_ptr(2);
  args.push_back(shared_addr);
  for (auto coord : global_coords)
    args.push_back(coord);
  for (auto offset : image_offset)
    args.push_back(offset);
  args.push_back(this->eviction_policy);
  Stmt tma_copy =
      IfThenElse(EQ(T.thread_var, T.thread_bounds->min),
                 Evaluate(Call(DataType::Handle(), tma_load_im2col(), args)));
  return tma_copy;
}

/*!
 * \brief Encode the TMA im2col descriptor into an array of PrimExpr.
 * This function serializes the TMA im2col descriptor fields for passing to the
 * create_tma_im2col_descriptor() builtin function. It includes
 * convolution-specific parameters like kernel size, stride, padding, and
 * dilation in addition to standard tensor descriptor fields. \return Array of
 * PrimExpr representing the encoded TMA im2col descriptor.
 */
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

// Register the Copy operation with TVM's TIR system
// This makes the copy operation available for use in TVM programs
// - Takes 4 inputs: src_buffer, dst_buffer, coalesced_width, disable_tma
// - Marked as opaque since it has side effects (memory writes)
TIR_REGISTER_TL_OP(Copy, copy)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// Register the Conv2DIm2Col operation with TVM's TIR system
// This operation performs im2col transformation for 2D convolutions using TMA
// - Takes 9 inputs: src_buffer, dst_buffer, nhw_step, c_step, kernel, stride,
// dilation, padding, eviction_policy
// - Marked as opaque since it has side effects (memory writes)
TIR_REGISTER_TL_OP(Conv2DIm2ColOp, c2d_im2col)
    .set_num_inputs(9)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm