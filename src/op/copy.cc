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
#include "../layout/tcgen05_layout.h"
#include "../target/utils.h"
#include "../transform/common/loop_fusion_utils.h"
#include "../transform/common/loop_parallel_transform_utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"
#include "region.h"

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
 * \brief Construct a Copy operator node from call arguments and a buffer map.
 *
 * This constructor parses the first two entries of `args` as Call nodes
 * describing source and destination Regions (via RegionOp), extracts their
 * Buffers and Ranges, and stores them on the newly created CopyNode. It also
 * reads optional arguments:
 * - args[2] (IntImm): coalesced width (stored only if > 0),
 * - args[3] (Bool): disable TMA lowering flag,
 * - args[4] (IntImm): eviction policy.
 *
 * Preconditions:
 * - `args` must contain at least two Call-compatible PrimExpr entries
 * describing regions; an ICHECK will fail if they are not CallNodes.
 *
 * @param args Array of PrimExpr where:
 *   - args[0] is the source Region call,
 *   - args[1] is the destination Region call,
 *   - optional args[2..4] are coalesced width, disable_tma, and eviction
 * policy.
 * @param vmap BufferMap used to resolve RegionOp buffers and ranges.
 */
Copy::Copy(Array<PrimExpr> args, BufferMap vmap) {
  ObjectPtr<CopyNode> node = tvm::ffi::make_object<CopyNode>();
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto expr = args[i];
    auto call = expr.as<CallNode>();
    ICHECK(call);
    auto region = RegionOp(call->args, vmap);
    rgs[i] = region->GetRanges();
    bf[i] = region->GetBuffer();
  }
  std::tie(node->src, node->dst) = std::tie(bf[0], bf[1]);
  std::tie(node->src_range, node->dst_range) = std::tie(rgs[0], rgs[1]);
  if (args.size() >= 3) {
    auto coalesced_width = Downcast<IntImm>(args[2]);
    if (coalesced_width->value > 0) {
      node->coalesced_width = coalesced_width;
    }
  }
  if (args.size() >= 4) {
    node->disable_tma = Downcast<Bool>(args[3]);
  }
  if (args.size() >= 5) {
    node->eviction_policy = args[4].as<IntImmNode>()->value;
  }
  data_ = std::move(node);
}

/**
 * @brief Create a shallow clone of this CopyNode as a TileOperator.
 *
 * Produces a new CopyNode object copy-constructed from this node. If a parallel
 * sub-operation (par_op_) is present, the sub-operation is cloned as well and
 * attached to the new node. The returned value is a TileOperator wrapper
 * around the newly created node.
 *
 * @return TileOperator A TileOperator owning the cloned CopyNode.
 */
TileOperator CopyNode::Clone() const {
  auto op = tvm::ffi::make_object<CopyNode>(*this);
  if (par_op_.defined()) {
    op->par_op_ = Downcast<ParallelOp>(par_op_->Clone());
  }
  return Copy(op);
}

/*!
 * \brief Create iterator variables for the copy operation.
 * This function creates iteration variables for dimensions that have extent
 * > 1. \return Array of IterVar representing the iterator variables for the
 * copy operation.
 */
Array<IterVar> CopyNode::MakeIterVars() const {
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
 * \brief Create s for the copy operation.
 * This function generates the actual index expressions for accessing source or
 * destination buffers. For dimensions with extent=1, it uses the range minimum;
 * for others, it adds the iteration variable. \param ivs Array of IterVar
 * returned by MakeIterVars(). \param src_dst 0 for src_indices, 1 for
 * dst_indices. \return Array of PrimExpr representing the indices for the copy
 * operation.
 */
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

/**
 * @brief Build a boundary predicate that guards memory accesses for the copy.
 *
 * Constructs a conjunction of per-dimension bounds checks (e.g. `min + iv <
 * extent` and `min + iv >= 0`) for every dynamic dimension involved in the
 * copy. Uses the provided arithmetic analyzer to elide checks that can be
 * proven statically.
 *
 * The function ICHECKs that the supplied `extents` align with the operator's
 * recorded ranges for the selected side (source when `src_dst == 0`,
 * destination when `src_dst == 1`).
 *
 * @param ivs IterVars corresponding to the varying dimensions of the copy. Each
 *   IterVar maps to a non-unit extent dimension in the stored ranges.
 * @param extents Extents of the tensor being accessed (must match the number of
 *   ranges); used as the upper bounds for generated checks.
 * @param src_dst Selects which side's ranges to use: `0` for source, `1` for
 *   destination.
 * @return PrimExpr A conjunction of necessary bounds checks, or an empty
 * `PrimExpr` (null) if all checks are provably true and no predicate is
 * required.
 */
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

/**
 * @brief Construct a SIMT-style nested loop that implements the copy.
 *
 * Builds a loop nest that performs element-wise loads from the source buffer
 * and stores into the destination buffer. For a scalar copy (no varying
 * iteration dimensions) this returns a single serial loop executing one
 * store. For multi-dimensional copies it:
 * - creates data-parallel loops (Parallel For) for each varying dimension,
 * - binds the resulting iteration variables to the provided arithmetic
 *   analyzer for simplification,
 * - computes source and destination index expressions,
 * - applies per-buffer boundary predicates (if needed) to mask out-of-range
 *   accesses,
 * - inserts a cast when src and dst dtypes differ,
 * - applies an optional `coalesced_width` annotation to generated parallel
 *   loops when present.
 *
 * @param analyzer Analyzer used to simplify and bind loop variable domains.
 * @return For A nested For statement representing the generated SIMT loop nest.
 */
For CopyNode::MakeSIMTLoop(arith::Analyzer *analyzer) const {
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
    Map<String, ObjectRef> annotations = {};
    if (coalesced_width.defined()) {
      annotations.Set("coalesced_width", coalesced_width);
    }
    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body, std::nullopt, annotations);
  }
  return Downcast<For>(body);
}

/**
 * @brief Compute a linearized shared-memory layout used for TMA transfers.
 *
 * Creates a Layout that maps an N-D shared tensor into a 1-D-like ordering
 * suitable for TMA by blocking each dimension into 256-element tiles and
 * splitting each original index into a quotient and remainder. Effectively
 * transforms each index i_k into two coordinates: floor(i_k / 256) and
 * i_k % 256, producing an ordering equivalent to concatenating all quotients
 * followed by all remainders.
 *
 * @param shared_tensor The shared-memory buffer whose shape defines the input
 *        dimensions for the layout inference.
 * @return Layout A Layout describing the linearized ordering for the TMA copy.
 */
Layout CopyNode::ComputeLinearLayout(const Buffer &shared_tensor) const {
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

/**
 * @brief Infer memory layouts for this Copy operation.
 *
 * Determines an appropriate LayoutMap for the copy based on the target and
 * enabled lowering paths. For TMA-capable targets when the chosen copy
 * instruction is BulkLoad or BulkStore, this may produce a linearized shared
 * memory layout suitable for TMA transfers (only when inference is invoked at
 * InferLevel::kFree and no layout for the shared buffer is already annotated).
 * For other cases (including LDSM/STSM and the normal copy path), layout
 * inference is delegated to the SIMT parallel operation produced by
 * MakeSIMTLoop().
 *
 * This method may read PassContext configuration (kDisableTMALower) and may
 * lazily construct and cache the parallel operation in par_op_ as a side
 * effect.
 *
 * @param T LayoutInferArgs containing target and the current layout map.
 * @param level The inference level controlling how aggressive/layouts may be
 *              proposed.
 * @return LayoutMap mapping buffers to inferred layouts (may be empty if no
 *         additional layouts are suggested).
 */
LayoutMap CopyNode::InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const {
  auto target = T.target;
  using namespace tvm::transform;
  PassContext pass_ctx = PassContext::Current();
  bool disable_tma_lower =
      pass_ctx->GetConfig<Bool>(kDisableTMALower, Bool(false)).value();
  auto copy_inst = GetCopyInst(target, disable_tma_lower || disable_tma,
                               T.layout_map, T.analyzer, T.buffer_oob);

  // Handle tensor memory (tmem) layout inference
  if (copy_inst == CopyInst::kTMemLoad || copy_inst == CopyInst::kTMemStore) {
    // Tensor memory copy
    // TODO (mzw) Add support for tcgen05.st/cp (in conj. with LowerTmemCopy)
    ICHECK(copy_inst == CopyInst::kTMemLoad)
        << "Only support tensor memory copy from shared.tmem to local.fragment "
           "currently";
    LayoutMap results;
    if (!T.layout_map.count(dst) && T.layout_map.count(src)) {
      // Use the default layout (32dp32b) if not specified
      // NOTE (mzw) We will check the layout in LowerTmemCopy(), so don't
      // worry for tmem-incompatible layout
      Layout src_layout = T.layout_map[src];
      Array<IterVar> logical_coords = MakeIterVars();
      Array<PrimExpr> logical_coords_var = {logical_coords[0]->var,
                                            logical_coords[1]->var};
      Array<PrimExpr> phy_indices = src_layout->Forward(logical_coords_var);

      // Tmem physical coord range analysis
      auto analyzer = std::make_shared<arith::Analyzer>();
      for (const auto &iv : logical_coords)
        analyzer->Bind(iv->var, iv->dom);
      arith::ConstIntBound phy_row_bounds =
          analyzer->const_int_bound(phy_indices[0]);
      arith::ConstIntBound phy_col_bounds =
          analyzer->const_int_bound(phy_indices[1]);
      Range row_dom = Range((int)(phy_row_bounds->min_value),
                            (int)(phy_row_bounds->max_value + 1));
      Range col_dom = Range((int)(phy_col_bounds->min_value),
                            (int)(phy_col_bounds->max_value + 1));

      constexpr int WARP_SIZE = 32; // Set to 32 since only sm100 is supported
      constexpr int WARPGROUP_SIZE = 4 * WARP_SIZE;
      ICHECK(is_const_int(T.thread_bounds->extent))
          << "Tensor memory copy requires thread_bounds->extent (num_threads) "
             "to be constant integers";
      int num_threads = *as_const_int(T.thread_bounds->extent);
      ICHECK(num_threads % WARPGROUP_SIZE == 0)
          << "Tensor memory copy requires thread bounds to be aligned to "
             "warpgroups, but found "
          << "thread range = " << T.thread_bounds;

      for (int num_useful_wgs = num_threads / WARPGROUP_SIZE;
           num_useful_wgs >= 1; --num_useful_wgs) {
        int num_useful_threads = num_useful_wgs * WARPGROUP_SIZE;
        Tcgen05Meta meta = getTcgen05Meta_32dp32b();
        auto [is_success, tmem_coord2frag, num_chunks_each_wg] =
            expandTcgen05Layout(
                meta, phy_col_bounds->max_value - phy_col_bounds->min_value + 1,
                num_useful_threads, row_dom, col_dom);
        if (!is_success) {
          continue;
        }
        Fragment logical_coord2frag =
            Fragment(logical_coords, tmem_coord2frag->Forward(phy_indices),
                     tmem_coord2frag->ForwardThread(phy_indices, std::nullopt),
                     make_itervar("rep", 1));
        results.Set(dst, logical_coord2frag->BindThreadRange(T.thread_bounds));
        break;
      }
    }
    return results;
  }

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
  if (!par_op_.defined()) {
    arith::Analyzer analyzer;
    par_op_ = ParallelOp((MakeSIMTLoop(&analyzer)));
  }
  return par_op_->InferLayout(T, level);
}
/**
 * @brief Determine whether this CopyNode can be lowered to a Bulk Load (TMA)
 * instruction.
 *
 * The function returns true when all of the following hold:
 * - the target architecture advertises bulk-copy/TMA support;
 * - the source buffer resides in global memory;
 * - the destination buffer resides in shared memory (either "shared" or
 * "shared.dyn");
 * - the source and destination have the same element data type.
 *
 * If the source and destination dtypes differ, a warning is logged and the
 * function returns false (the caller is expected to fall back to a normal
 * copy).
 *
 * @param target The compilation target to query for bulk-copy support.
 * @return true if the copy can be implemented as a Bulk Load (TMA); false
 * otherwise.
 */
bool CopyNode::CheckBulkLoad(Target target, arith::Analyzer *analyzer,
                             bool check_last_dim) const {
  // 1. arch must have bulk copy support
  if (!TargetHasBulkCopy(target))
    return false;
  // 2. src and dst must be global and shared
  if (src.scope() != "global" ||
      (dst.scope() != "shared.dyn" && dst.scope() != "shared"))
    return false;
  // 3. check shape.
  // last dim of src * dtype.bits() must be a multiple of 16
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
  // now we check src (gmem) as tma box dim is deduced from src
  if (check_last_dim &&
      analyzer->CanProve(
          FloorMod(src_range[src_range.size() - 1]->extent * src->dtype.bytes(),
                   16) != 0,
          arith::ProofStrength::kSymbolicBound)) {
    LOG(WARNING)
        << "src range must have last dim multiple of 16 for tma bulk load "
        << src->name << " range " << src_range[src_range.size() - 1]->extent
        << " * " << src->dtype.bytes() << " % 16 != 0";
    return false;
  }

  // 4. src and dst must have the same dtype
  if (src->dtype != dst->dtype) {
    LOG(WARNING) << "src and dst must have the same dtype for tma load "
                 << src->name << " vs. " << dst->name << " dtype " << src->dtype
                 << " vs. " << dst->dtype << " will be fallback to normal copy";
    return false;
  }
  return true;
}

bool CopyNode::CheckBulkCopy1D(const Buffer &global_tensor,
                               const Buffer &shared_tensor,
                               const Array<Range> &global_range,
                               const Array<Range> &shared_range,
                               const LayoutMap &layout_map,
                               arith::Analyzer *analyzer) const {

  // Step 1: check shared is contiguous
  bool shared_is_contiguous = true;
  if (layout_map.count(shared_tensor)) {
    shared_is_contiguous = false;
  }
  // Step 2: check global is contiguous
  bool global_is_contiguous = true;
  bool global_not_full_dim_encounter = false;
  for (int i = global_range.size() - 1; i >= 0; i--) {
    if (!global_not_full_dim_encounter) {
      if (!analyzer->CanProve(global_range[i]->extent ==
                                      global_tensor->shape[i] &&
                                  global_range[i]->min == 0,
                              arith::ProofStrength::kSymbolicBound)) {
        global_not_full_dim_encounter = true;
      }
    } else {
      if (!analyzer->CanProve(global_range[i]->extent == 1,
                              arith::ProofStrength::kSymbolicBound)) {
        global_is_contiguous = false;
        break;
      }
    }
  }

  // Step 3: check element match and no OOB
  PrimExpr shared_elements = 1;
  for (size_t i = 0; i < shared_range.size(); i++) {
    shared_elements *= shared_range[i]->extent;
  }
  PrimExpr global_elements = 1;
  for (size_t i = 0; i < global_range.size(); i++) {
    global_elements *= global_range[i]->extent;
  }
  bool element_match =
      analyzer->CanProveEqual(shared_elements, global_elements);

  return (shared_is_contiguous && global_is_contiguous && element_match);
}

bool CopyNode::CheckBulkLoad1D(Target target, const LayoutMap &layout_map,
                               arith::Analyzer *analyzer) const {
  if (!CheckBulkLoad(target, analyzer, false))
    return false;
  auto global_tensor = src;
  auto shared_tensor = dst;
  auto global_range = src_range;
  auto shared_range = dst_range;
  return CheckBulkCopy1D(global_tensor, shared_tensor, global_range,
                         shared_range, layout_map, analyzer);
}

bool CopyNode::CheckBulkStore1D(Target target, const LayoutMap &layout_map,
                                arith::Analyzer *analyzer) const {
  if (!CheckBulkStore(target, analyzer, false))
    return false;
  auto shared_tensor = src;
  auto global_tensor = dst;
  auto shared_range = src_range;
  auto global_range = dst_range;
  return CheckBulkCopy1D(global_tensor, shared_tensor, global_range,
                         shared_range, layout_map, analyzer);
}

/**
 * @brief Determine if this CopyNode can be lowered to a CUDA BulkStore (TMA
 * store).
 *
 * Checks whether the target supports bulk copy, the source buffer is in shared
 * memory (shared or shared.dyn), the destination buffer is in global memory,
 * and both buffers have the same element data type. If the data types differ,
 * a warning is logged and false is returned.
 *
 * @param target Target device/architecture to check for bulk-copy support.
 * @return true if all conditions for a BulkStore are met; false otherwise.
 */
bool CopyNode::CheckBulkStore(Target target, arith::Analyzer *analyzer,
                              bool check_last_dim) const {
  // 1. arch must have bulk copy support
  if (!TargetHasBulkCopy(target))
    return false;
  // 2. src and dst must be shared.dyn and local.fragment
  if ((src.scope() != "shared.dyn" && src.scope() != "shared") ||
      dst.scope() != "global")
    return false;
  // 3. check shape.
  // last dim of dst * dtype.bits() must be a multiple of 16
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
  // now we check dst (gmem) as tma box dim is deduced from dst
  if (check_last_dim &&
      analyzer->CanProve(
          FloorMod(dst_range[dst_range.size() - 1]->extent * dst->dtype.bytes(),
                   16) != 0,
          arith::ProofStrength::kSymbolicBound)) {
    LOG(WARNING)
        << "dst range must have last dim multiple of 16 for tma bulk store "
        << dst->name << " range " << dst_range[dst_range.size() - 1]->extent
        << " * " << dst->dtype.bytes() << " % 16 != 0";
    return false;
  }
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
bool CopyNode::CheckLDSMCopy(Target target) const {
  return TargetHasLdmatrix(target) &&
         (src.scope() == "shared.dyn" || src.scope() == "shared") &&
         dst.scope() == "local.fragment";
}

/**
 * @brief Determine whether this copy can use the STMATRIX store (STSM) path.
 *
 * Returns true when the target supports STMATRIX and the source buffer is in
 * the `local.fragment` scope while the destination buffer is in shared memory
 * (`shared` or `shared.dyn`).
 *
 * @param target The compilation target to query for STMATRIX support.
 * @return true if the copy may be lowered to an STSM instruction; false
 * otherwise.
 */
bool CopyNode::CheckSTSMCopy(Target target) const {
  return TargetHasStmatrix(target) && src.scope() == "local.fragment" &&
         (dst.scope() == "shared.dyn" || dst.scope() == "shared");
}

/**
 * @brief Determine whether this copy can use tensor memory load (tcgen05.ld).
 *
 * Returns true when the target supports tensor memory and the source buffer is
 * in `shared.tmem` scope while the destination buffer is in `local.fragment`.
 *
 * @param target The compilation target to query for tensor memory support.
 * @return true if the copy may be lowered to a tcgen05.ld instruction; false
 * otherwise.
 */
bool CopyNode::CheckTMemLoad(Target target) const {
  return TargetHasTmem(target) && src.scope() == "shared.tmem" &&
         dst.scope() == "local.fragment";
}

/**
 * @brief Determine whether this copy can use tensor memory store (tcgen05.st).
 *
 * Returns true when the target supports tensor memory and the source buffer is
 * in `local.fragment` scope while the destination buffer is in `shared.tmem`.
 *
 * @param target The compilation target to query for tensor memory support.
 * @return true if the copy may be lowered to a tcgen05.st instruction; false
 * otherwise.
 */
bool CopyNode::CheckTMemStore(Target target) const {
  return TargetHasTmem(target) && src.scope() == "local.fragment" &&
         dst.scope() == "shared.tmem";
}

/**
 * @brief Selects the most specific copy instruction supported for the given
 * target and buffers.
 *
 * Determines which specialized copy lowering to use (TMA bulk load/store, LDSM,
 * STSM, TMem load/store) based on target capabilities and the memory scopes of
 * the source/destination buffers. If TMA lowering is disabled via the flag,
 * BulkLoad/BulkStore are not selected. The selection priority is: TMemLoad,
 * TMemStore, BulkLoad1D, BulkStore1D, BulkLoad, BulkStore, LDSM, STSM, then
 * Normal (fallback).
 *
 * @param target The compilation target used to query hardware capabilities.
 * @param disable_tma_lower If true, prevents selecting TMA-based bulk
 * load/store instructions.
 * @return CopyInst The chosen copy instruction enum value.
 */
CopyInst CopyNode::GetCopyInst(Target target, bool disable_tma_lower,
                               const LayoutMap &layout_map,
                               arith::Analyzer *analyzer,
                               bool buffer_oob = false) const {
  // disable_tma_lower is from pass_configs
  // when tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER is True,
  // we will not use tma for bulk load/store

  // Check tensor memory operations first (highest priority for SM100/Blackwell)
  // 1d tma access can not support out of bound access
  if (!disable_tma_lower && !buffer_oob &&
      CheckBulkLoad1D(target, layout_map, analyzer)) {
    return CopyInst::kBulkLoad1D;
  } else if (!disable_tma_lower && !buffer_oob &&
             CheckBulkStore1D(target, layout_map, analyzer)) {
    return CopyInst::kBulkStore1D;
  } else if (!disable_tma_lower && CheckBulkLoad(target, analyzer)) {
    return CopyInst::kBulkLoad;
  } else if (!disable_tma_lower && CheckBulkStore(target, analyzer)) {
    return CopyInst::kBulkStore;
  } else if (CheckLDSMCopy(target)) {
    return CopyInst::kLDSM;
  } else if (CheckSTSMCopy(target)) {
    return CopyInst::kSTSM;
  } else if (CheckTMemLoad(target)) {
    return CopyInst::kTMemLoad;
  } else if (CheckTMemStore(target)) {
    return CopyInst::kTMemStore;
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
Stmt CopyNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  Target target = T.target;

  using namespace tvm::transform;
  PassContext pass_ctx = PassContext::Current();
  bool disable_tma_lower =
      pass_ctx->GetConfig<Bool>(kDisableTMALower, Bool(false)).value();
  auto copy_inst = GetCopyInst(target, disable_tma_lower || disable_tma,
                               T.layout_map, analyzer);
  if (copy_inst == CopyInst::kTMemLoad || copy_inst == CopyInst::kTMemStore) {
    auto tmem_copy = LowerTmemCopy(T, analyzer);
    ICHECK(tmem_copy.defined()) << "Failed to lower tensor memory copy";
    return tmem_copy;
  } else if (copy_inst == CopyInst::kBulkLoad1D ||
             copy_inst == CopyInst::kBulkStore1D) {
    auto bulk_copy = LowerBulkCopy1D(T, analyzer, copy_inst);
    ICHECK(bulk_copy.defined()) << "Failed to lower bulk load 1d";
    return bulk_copy;
  } else if (copy_inst == CopyInst::kBulkLoad ||
             copy_inst == CopyInst::kBulkStore) {
    auto bulk_copy = LowerBulkCopy(T, analyzer, copy_inst);
    ICHECK(bulk_copy.defined()) << "Failed to lower bulk load/store";
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

/**
 * @brief Lower the copy operator using the generic (non-specialized) path.
 *
 * Generates standard load/store code paths for targets that cannot or should
 * not use specialized copy instructions (TMA, LDSM/STSM). Builds a SIMT loop,
 * fuses and transforms parallel loops, infers and applies loop layouts on GPU
 * targets, partitions by thread, and applies vectorization appropriate to the
 * device (CPU or GPU). If a thread-level predicate is required, the resulting
 * body is guarded with an IfThenElse.
 *
 * @param T Lowering context including the target, thread bounds, thread var,
 *          layout map, and buffer remapping used during layout inference and
 *          loop partitioning.
 * @param analyzer Arithmetic analyzer used to simplify and reason about bounds
 *                 during loop partitioning and predicate construction.
 * @return Stmt Lowered statement representing the transformed, vectorized
 *              normal-copy loop (possibly wrapped in a predicate).
 */
Stmt CopyNode::LowerNormalCopy(const LowerArgs &T,
                               arith::Analyzer *analyzer) const {
  bool is_cpu_target = T.target->GetTargetDeviceType() == kDLCPU;
  auto simt_loop = MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));

  auto transformed_loop =
      Downcast<For>(ParallelLoopTransformer::Substitute(fused_loop));

  For vectorized_thread_loop;
  auto par_op = ParallelOp(transformed_loop);

  if (is_cpu_target) {
    vectorized_thread_loop = VectorizeLoop(transformed_loop);
  } else {
    std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                      InferLevel::kFree};
    for (auto level : levels) {
      par_op->InferLayout({T.target, T.thread_bounds, T.layout_map, analyzer,
                           false, T.buffer_remap},
                          level);
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

/**
 * @brief Lower a Copy operator to LDSM/STSM (warp-level 8x8 matrix)
 * instructions.
 *
 * Lowers a CopyNode into PTX matrix load/store (LDSM/STSM) sequences when the
 * access/layouts meet the hardware constraints required by warp-level 8x8
 * fragment transfers (thread-mapped 8x8 fragment layout, 16-byte contiguous
 * shared memory accesses, full-range local tiles, matching dtypes for loads,
 * and no access predicates). If these conditions are not met the function
 * falls back to lowering via LowerNormalCopy().
 *
 * The routine validates layout/thread-mapping compatibility (including support
 * for transposed fragment layouts), determines vectorization factor (4/2/1)
 * based on extent alignment, computes shared/local addresses, emits the
 * appropriate ptx_ldmatrix/ptx_stmatrix call(s), and wraps them in a small
 * loop that may be unrolled and adjusted for thread-bounds offsets.
 *
 * @param T Lowering context (target, layout/ buffer remaps, thread/ bounds).
 * @param analyzer Arithmetic analyzer used to simplify and prove bounds.
 * @param copy_inst Must be either CopyInst::kLDSM or CopyInst::kSTSM to select
 *                  matrix-load vs matrix-store lowering.
 * @return Stmt A statement implementing the LDSM/STSM lowering, or the result
 *              of LowerNormalCopy(...) when constraints require fallback.
 */
Stmt CopyNode::LowerLDSMCopy(const LowerArgs &T, arith::Analyzer *analyzer,
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

/**
 * @brief Lower tensor memory copy operations (tcgen05.ld/st/cp).
 *
 * Handles copy operations involving shared.tmem buffers (tensor memory on
 * SM100/Blackwell). Supports three types of tensor memory copies:
 * - tcgen05.ld: tensor memory -> register (local.fragment)
 * - tcgen05.st: register (local.fragment) -> tensor memory
 * - tcgen05.cp: shared memory -> tensor memory
 *
 * The function validates buffer scopes, extracts 2D loop structure, performs
 * layout compatibility checks, selects an appropriate TCGEN05 instruction
 * variant based on data width and thread count, and emits the corresponding PTX
 * intrinsic call.
 *
 * Currently only tcgen05.ld is fully supported; st/cp will trigger an ICHECK
 * failure.
 *
 * @param T Lowering context (target, thread bounds, layout maps, buffer
 * remaps).
 * @param analyzer Arithmetic analyzer for proving bounds and simplifying
 * expressions.
 * @return Stmt The lowered tensor memory copy statement, or an empty Stmt if
 * this copy does not involve tensor memory.
 */
Stmt CopyNode::LowerTmemCopy(const LowerArgs &T,
                             arith::Analyzer *analyzer) const {
  if (src.scope() != "shared.tmem" && dst.scope() != "shared.tmem") {
    return Stmt();
  }
  ICHECK(TargetHasTmem(T.target)) << "Target " << T.target->ToDebugString()
                                  << " does not support tensor memory copy";

  // Decide copy type
  bool is_ld = false; // tcgen05.ld (tensor memory -> register)
  bool is_st = false; // tcgen05.st (register -> tensor memory)
  bool is_cp = false; // tcgen05.cp (shared memory -> tensor memory)
  if (src.scope() == "shared.tmem" && dst.scope() == "local.fragment") {
    is_ld = true;
  } else if (src.scope() == "local.fragment" && dst.scope() == "shared.tmem") {
    is_st = true;
  } else if (src.scope() == "shared.dyn" && dst.scope() == "shared.tmem") {
    is_cp = true;
  } else {
    ICHECK(0) << "Unsupported tensor memory copy: "
              << "src scope = " << src.scope()
              << ", dst scope = " << dst.scope();
  }
  // Currently tcgen05.cp is not supported
  // TODO (mzw) Support tcgen05.cp
  ICHECK(!is_cp)
      << "Copy from shared memory to tensor memory is not supported yet";
  // Currently tcgen05.st is not supported
  // TODO (mzw) Support tcgen05.st
  ICHECK(!is_st) << "Copy from register to tensor memory is not supported yet";

  // Extract loop variables and ranges
  Array<IterVar> loop_vars = MakeIterVars();
  ICHECK(loop_vars.size() == 2) << "Only support 2D tensor memory copy, got "
                                << loop_vars.size() << " dimensions";
  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);
  PrimExpr src_predicate = MakePredicate(analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate = MakePredicate(analyzer, loop_vars, dst->shape, 1);
  ICHECK(!src_predicate.defined() && !dst_predicate.defined())
      << "Tensor memory copy does not support predicates, got " << src_predicate
      << " and " << dst_predicate;
  ICHECK(is_const_int(loop_vars[0]->dom->min) &&
         is_const_int(loop_vars[0]->dom->extent) &&
         is_const_int(loop_vars[1]->dom->min) &&
         is_const_int(loop_vars[1]->dom->extent))
      << "Tensor memory copy requires loop bounds to be constant integers";
  int64_t logical_row_min = *as_const_int(loop_vars[0]->dom->min);
  int64_t logical_row_extent = *as_const_int(loop_vars[0]->dom->extent);
  int64_t logical_col_min = *as_const_int(loop_vars[1]->dom->min);
  int64_t logical_col_extent = *as_const_int(loop_vars[1]->dom->extent);

  // Extract thread bounds
  constexpr int WARP_SIZE = 32; // Set to 32 since only sm100 is supported
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

  // TODO (mzw) Buffer remap for shared.dyn when is_cp is true?

  // Retrieve layout
  ICHECK(T.layout_map.count(src))
      << "Source buffer " << src->name << " does not have a layout specified";
  ICHECK(T.layout_map.count(dst)) << "Destination buffer " << dst->name
                                  << " does not have a layout specified";
  Layout src_layout = T.layout_map[src];
  Fragment dst_layout = Downcast<Fragment>(T.layout_map[dst]);

  // Check layout
  Array<PrimExpr> logical_indices = MakeIndices(loop_vars, 0);
  Array<PrimExpr> phy_indices =
      src_layout->Forward(logical_indices); // "phy" for "physical"

  // Analyse the range of tmem_phy_row and tmem_phy_col
  arith::ConstIntBound phy_row_bounds =
      analyzer->const_int_bound(phy_indices[0]);
  arith::ConstIntBound phy_col_bounds =
      analyzer->const_int_bound(phy_indices[1]);
  int tmem_phy_row_min = phy_row_bounds->min_value;
  int tmem_phy_row_max = phy_row_bounds->max_value;
  int tmem_phy_col_min = phy_col_bounds->min_value;
  int tmem_phy_col_max = phy_col_bounds->max_value;
  int tmem_phy_row_extent = tmem_phy_row_max - tmem_phy_row_min + 1;
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
      PrimExpr dst_thread =
          dst_layout->ForwardThread(logical_indices, std::nullopt);
      if (!analyzer->CanProveEqual(target_thread, dst_thread)) {
        continue;
      }
      PrimExpr target_reg = target_frag->Forward(phy_indices)[0];
      PrimExpr dst_reg = dst_layout->Forward(logical_indices)[0];
      if (!analyzer->CanProveEqual(target_reg, dst_reg)) {
        continue;
      }

      // All checks passed, we can use this instruction
      PrimExpr relative_wg_idx =
          FloorDiv(Sub(T.thread_var, T.thread_bounds->min), WARPGROUP_SIZE);
      PrimExpr col_offset =
          num_useful_threads == WARPGROUP_SIZE
              ? PrimExpr(0)
              : relative_wg_idx * (num_chunks_each_wg * meta.width);
      have_succeeded = true;
      Array<PrimExpr> args;
      args.push_back(StringImm(meta.intrinsics_name + "<" +
                               std::to_string(num_chunks_each_wg) + ">"));
      args.push_back(
          BufferLoad(src, {(int)logical_row_min,
                           (int)logical_col_min})); // Will be translated later
                                                    // in lower_shared_tmem pass
      args.push_back(col_offset);
      args.push_back(dst.access_ptr(2, DataType::Handle(), 1, 0,
                                    PrimExpr(tmem_phy_col_extent)));

      Stmt call =
          Evaluate(Call(DataType::Handle(), builtin::call_extern(), args));
      if (num_useful_threads != num_threads) {
        body =
            IfThenElse(T.thread_var < T.thread_bounds->min + num_useful_threads,
                       call, // No-op for unused threads
                       Stmt());
      } else {
        body = call;
      }
      break;
    }
  };

  try_tcgen05_instruction(getTcgen05Meta_32dp32b());
  try_tcgen05_instruction(getTcgen05Meta_32dp64b());
  try_tcgen05_instruction(getTcgen05Meta_32dp128b());
  try_tcgen05_instruction(getTcgen05Meta_32dp256b());

  ICHECK(have_succeeded) << "Failed to find a suitable instruction for "
                            "tcgen05.ld. Check your layout.";

  return body;
}

/**
 * @brief Lower a Copy operator to a bulk TMA (Tensor Memory Accelerator)
 * transfer.
 *
 * Lowers the copy to an optimized TMA load or store when the target and buffer
 * layouts permit. Constructs a TMADesc, detects shared-memory
 * swizzle/interleave patterns, encodes global shape/stride/SMEM parameters, and
 * emits either a 1D TMA transfer (when global/shared are contiguous and element
 * counts match, currently only for loads) or a full multi-dimensional TMA call.
 * The emitted statement is guarded so only the thread with min thread id
 * executes the TMA.
 *
 * If preconditions are not satisfied (unsupported swizzle, stride/size limits,
 * mismatched element counts, OOB risks, or other hardware constraints), this
 * function falls back to LowerNormalCopy.
 *
 * @param T LowerArgs containing target information, thread/bounds variables,
 *          and layout/ buffer remap information used for descriptor
 * construction.
 * @param analyzer Analyzer used to prove shapes/contiguity/equality
 * constraints.
 * @param copy_inst Indicates whether to emit a BulkLoad (TMA load) or BulkStore
 *                  (TMA store). Must be CopyInst::kBulkLoad or kBulkStore.
 * @return Stmt A TIR statement performing the bulk TMA copy (or the result of
 *         LowerNormalCopy when falling back).
 */
Stmt CopyNode::LowerBulkCopy(const LowerArgs &T, arith::Analyzer *analyzer,
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
  // TODO(lei): find a much smarter way to deduce smem box dim
  // instead of using global_range
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
        shared_offset + total_elements * loop_var, total_elements);
    args.push_back(shared_addr);
    global_coords.Set(0, global_coords[0] + instruction_dim * loop_var);
    for (auto coord : global_coords)
      args.push_back(coord);
    int need_reduce = 0;
    if (!is_load)
      args.push_back(need_reduce);
    args.push_back(this->eviction_policy);
    tma_copy = For(loop_var, 0, loop_extent, ForKind::kUnrolled,
                   Evaluate(Call(DataType::Handle(), op, args)));
  } else {
    PrimExpr shared_addr = shared_tensor.access_ptr(
        is_load ? 2 : 1, DataType::Handle(), 1, shared_offset, total_elements);
    args.push_back(shared_addr);
    for (auto coord : global_coords)
      args.push_back(coord);
    int need_reduce = 0;
    if (!is_load)
      args.push_back(need_reduce);
    args.push_back(this->eviction_policy);
    tma_copy = Evaluate(Call(DataType::Handle(), op, args));
  }
  tma_copy = IfThenElse(EQ(T.thread_var, T.thread_bounds->min), tma_copy);

  return tma_copy;
}

Stmt CopyNode::LowerBulkCopy1D(const LowerArgs &T, arith::Analyzer *analyzer,
                               CopyInst copy_inst) const {
  ICHECK(copy_inst == CopyInst::kBulkLoad1D ||
         copy_inst == CopyInst::kBulkStore1D);

  // Add 1D TMA copy when the global and shared memory is contiguous
  // Check if shared_tensor->name is present in T.buffer_var_gemm
  // (Array<PrimExpr>) to avoid use 1D TMA copy for swizzled layout
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
  Stmt tma_copy;
  if (is_load) {
    // the zero is a placeholder for mbarrier ids
    tma_copy = Evaluate(
        Call(DataType::Handle(), tma_load(),
             {shared_addr, global_addr, 0,
              elements * shared_tensor->dtype.bytes(), this->eviction_policy}));
  } else {
    int need_reduce = 0;
    tma_copy = Evaluate(
        Call(DataType::Handle(), tma_store(),
             {global_addr, shared_addr, elements * shared_tensor->dtype.bytes(),
              need_reduce, this->eviction_policy}));
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

/**
 * @brief Construct a Conv2DIm2ColOp node.
 *
 * Initializes a Conv2DIm2ColOpNode from raw TL-call arguments and a buffer map.
 * The constructor extracts source and destination Buffers from vmap and reads
 * convolution parameters encoded in args:
 * - args[0]: source tensor access pointer
 * - args[1]: destination tensor access pointer
 * - args[2]: nhw_step (PrimExpr)
 * - args[3]: c_step (PrimExpr)
 * - args[4]: kernel (IntImm)
 * - args[5]: stride (IntImm)
 * - args[6]: dilation (IntImm)
 * - args[7]: padding (IntImm)
 * - args[8]: eviction_policy (IntImm)
 *
 * The created node stores these values (src, dst, nhw_step, c_step, kernel,
 * stride, dilation, padding, eviction_policy) for later lowering to TMA-based
 * GPU intrinsics.
 *
 * @param args Array of PrimExpr TL-call arguments (see list above).
 * @param vmap Mapping from original buffer variables to actual Buffer objects.
 */
Conv2DIm2ColOp::Conv2DIm2ColOp(Array<PrimExpr> args, BufferMap vmap) {
  ObjectPtr<Conv2DIm2ColOpNode> node =
      tvm::ffi::make_object<Conv2DIm2ColOpNode>();
  node->src = vmap[GetVarFromAccessPtr(args[0])];
  node->dst = vmap[GetVarFromAccessPtr(args[1])];
  node->nhw_step = args[2];
  node->c_step = args[3];
  node->kernel = args[4].as<IntImm>().value()->value;
  node->stride = args[5].as<IntImm>().value()->value;
  node->dilation = args[6].as<IntImm>().value()->value;
  node->padding = args[7].as<IntImm>().value()->value;
  node->eviction_policy = args[8].as<IntImm>().value()->value;
  data_ = std::move(node);
}

/**
 * @brief Create a shallow copy of this Conv2DIm2ColOpNode wrapped as a
 * TileOperator.
 *
 * Produces a new Conv2DIm2ColOp that owns a freshly allocated
 * Conv2DIm2ColOpNode initialized from this node (member-wise copy). This is
 * used to duplicate the operator node for compiler passes that require
 * independent operator instances.
 *
 * @return TileOperator A TileOperator containing the cloned Conv2DIm2ColOpNode.
 */
TileOperator Conv2DIm2ColOpNode::Clone() const {
  auto op = tvm::ffi::make_object<Conv2DIm2ColOpNode>(*this);
  return Conv2DIm2ColOp(op);
}

/**
 * @brief Lower Conv2D im2col into a TMA-backed PTX sequence for Hopper.
 *
 * Constructs a TMA im2col descriptor from the Conv2DIm2ColOp parameters
 * (kernel, stride, dilation, padding, channel/image tiling, dtype and shapes),
 * emits a call to create the im2col descriptor, and returns a statement that
 * invokes the corresponding tma_load_im2col builtin guarded to a single
 * thread. The lowering assumes the destination resides in shared memory and the
 * source in global memory and uses the provided layout information (when
 * available) to select the appropriate shared-memory swizzle.
 *
 * Preconditions (checked with ICHECK):
 * - Target is Hopper.
 * - src.scope() == "global" and dst.scope() is "shared.dyn" or "shared".
 * - src->shape has rank 4 and dst->shape has rank 2.
 * - src and dst have the same dtype.
 * - When a shared layout is supplied it must match a recognized TMA swizzle
 *   pattern (32B/64B/128B) or an ICHECK will fail.
 *
 * @param T Lowering context (target, layout map, thread_var, thread_bounds,
 *          buffer remapping, etc.). Used to fetch target/layout and to emit a
 *          thread-guarded TMA call.
 * @param analyzer Arithmetic analyzer used to prove divisibility and simplify
 *                 expressions required by descriptor construction.
 * @return Stmt A TIR statement that performs a tma_load_im2col call wrapped in
 *              a thread-min guard (IfThenElse). The returned statement is ready
 *              to be inserted into the lowered TIR.
 */
Stmt Conv2DIm2ColOpNode::Lower(const LowerArgs &T,
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
// - Takes 5 inputs: src_buffer, dst_buffer, coalesced_width, disable_tma,
// eviction_policy
// - Marked as opaque since it has side effects (memory writes)
TIR_REGISTER_TL_OP(Copy, copy)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

/**
 * @brief Layout inference hook for Conv2DIm2ColOpNode.
 *
 * This operator does not provide any layout inference; the function
 * intentionally returns an empty LayoutMap to indicate no layout suggestions.
 *
 * @param T Context for layout inference (ignored).
 * @param level Inference level (ignored).
 * @return LayoutMap An empty map.
 */
LayoutMap Conv2DIm2ColOpNode::InferLayout(const LayoutInferArgs &T,
                                          InferLevel level) const {
  return {};
}

// Register the Conv2DIm2Col operation with TVM's TIR system
// This operation performs im2col transformation for 2D convolutions using TMA
// - Takes 9 inputs: src_buffer, dst_buffer, nhw_step, c_step, kernel, stride,
// dilation, padding, eviction_policy
// - Marked as opaque since it has side effects (memory writes)
TIR_REGISTER_TL_OP(Conv2DIm2ColOp, c2d_im2col)
    .set_num_inputs(9)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  CopyNode::RegisterReflection();
  Conv2DIm2ColOpNode::RegisterReflection();
}
} // namespace tl
} // namespace tvm
