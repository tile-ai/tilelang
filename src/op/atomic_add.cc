/*!
 * \file tl/op/atomic_add.cc
 *
 * Define element-wise operators.
 */

#include "./atomic_add.h"
#include "./copy.h"
#include "utils.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../layout/layout.h"
#include "../target/utils.h"
#include "../transform/atomicadd_vectorize.h"
#include "../transform/common/loop_fusion_utils.h"
#include "../transform/loop_partition.h"
#include "builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

/**
 * @brief Construct an AtomicAdd operator from call arguments and annotations.
 *
 * Builds the internal AtomicAddNode, extracts the source and destination
 * regions and their backing Buffers from the first two region-style expressions
 * in `args` (BufferLoad/BufferRegion), and stores them along with their
 * ranges. Annotations are copied directly from the Call node.
 *
 * @param args Call-style PrimExprs where:
 *             - args[0] is the source region call,
 *             - args[1] is the destination region call.
 * @param annotations Map containing optional keys:
 *             - "use_tma": whether to use TMA for memory operations
 *             - "memory_order": memory order for atomic operations
 * Notes:
 * - The constructor checks that args[0] and args[1] are region-compatible.
 * - The constructed node is stored in this->data_.
 */
AtomicAdd::AtomicAdd(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<AtomicAddNode> node = tvm::ffi::make_object<AtomicAddNode>();
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto region = NormalizeToBufferRegion(args[i]);
    rgs[i] = region->region;
    bf[i] = region->buffer;
  }
  std::tie(node->src, node->dst) = std::tie(bf[0], bf[1]);
  std::tie(node->src_range, node->dst_range) = std::tie(rgs[0], rgs[1]);
  // Copy annotations from the Call node
  node->annotations = annotations;
  data_ = std::move(node);
}

/**
 * @brief Create a deep copy of this AtomicAdd node wrapped as a TileOperator.
 *
 * Produces a new AtomicAddNode object copied from this node. If this node has
 * an associated ParallelOp (par_op_), the parallel op is cloned and attached to
 * the new node so the cloned operator preserves parallelization state.
 *
 * @return TileOperator A TileOperator owning the cloned AtomicAddNode.
 */
TileOperator AtomicAddNode::Clone() const {
  auto op = tvm::ffi::make_object<AtomicAddNode>(*this);
  if (par_op_.defined()) {
    op->par_op_ = Downcast<ParallelOp>(par_op_->Clone());
  }
  return AtomicAdd(op);
}

/**
 * @brief Create data-parallel iteration variables for non-singleton dimensions
 * of the source.
 *
 * Constructs an Array of IterVar corresponding to each dimension in `src_range`
 * whose extent is not equal to 1. Each IterVar has domain Range(0, extent), a
 * Var named sequentially ("i", "j", "k", ...) with the same dtype as the
 * extent, and type IterVarType::kDataPar. The ordering of returned itervars
 * matches the order of dimensions in `src_range`.
 *
 * @return Array<IterVar> Iteration variables for all non-singleton extents in
 * `src_range`.
 */
Array<IterVar> AtomicAddNode::MakeIterVars() const {
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

// ivs: itervars returned by MakeIterVars()
/**
 * @brief Build index expressions for either source or destination from loop
 * iter vars.
 *
 * Given a list of iteration variables that correspond to the non-singleton
 * extents of the selected region (source when src_dst == 0, destination when
 * src_dst == 1), return an array of index expressions matching the full rank of
 * that region. For dimensions with extent == 1, the corresponding index is the
 * range's minimum; otherwise the index is `min + ivar`.
 *
 * @param ivs Iteration variables in order for all non-singleton dimensions of
 * the chosen region.
 * @param src_dst Selects which region to index: 0 for source (src_range), 1 for
 * destination (dst_range).
 * @return Array<PrimExpr> Index expressions for every dimension of the selected
 * region, in original dimension order.
 *
 * @note The function checks that the number of provided iter vars equals the
 * number of non-singleton extents; it will abort (ICHECK) if they differ.
 */
Array<PrimExpr> AtomicAddNode::MakeIndices(const Array<IterVar> &ivs,
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

std::pair<Array<PrimExpr>, PrimExpr>
AtomicAddNode::ReturnIndicesAndSize(int src_dst) const {
  Array<PrimExpr> indices;
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;
  PrimExpr size = 1;
  for (size_t i = 0; i < ranges.size(); i++) {
    indices.push_back(ranges[i]->min);
    size *= ranges[i]->extent;
  }
  return {indices, size};
}

/**
 * @brief Build a combined bound-check predicate for indexed access.
 *
 * Constructs an AND'd predicate ensuring each non-singleton index (derived from
 * `ivs`) stays within [0, extent) for the selected operand (source when
 * `src_dst==0`, destination otherwise). For each non-unit Range in the chosen
 * range list this produces two conditions:
 *   - range.min + iv >= 0
 *   - range.min + iv < extent
 *
 * Conditions that the analyzer can prove (with symbolic bounds) are omitted.
 * If no uncertain conditions remain, an empty PrimExpr is returned.
 *
 * Note: the function ICHECKs that `extents.size()` equals the number of ranges
 * for the selected operand.
 *
 * @param ivs Iteration variables corresponding to non-singleton extents (order
 *            matches the non-unit ranges of the chosen operand).
 * @param extents Per-dimension upper bounds to check against; must have the
 *                same size as the selected range list.
 * @param src_dst Selects which ranges to validate: 0 => `src_range`, else
 *                `dst_range`.
 * @return PrimExpr A conjunction of remaining (non-provable) bounds checks, or
 *         an empty PrimExpr when no checks are required.
 */
PrimExpr AtomicAddNode::MakePredicate(arith::Analyzer *analyzer,
                                      const Array<IterVar> &ivs,
                                      Array<PrimExpr> extents,
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

/**
 * @brief Build a SIMT-style loop nest that performs element-wise atomic
 * additions from src to dst.
 *
 * Constructs a nested loop (parallelized per iter var) that loads a value from
 * the source buffer, optionally casts it to the destination dtype, and performs
 * an extern atomic add into the destination buffer address. For scalar
 * (zero-dimensional) operations a trivial serial For with a single BufferStore
 * is returned.
 *
 * The method:
 * - Creates iter vars for all non-singleton extents and binds them into the
 * provided analyzer.
 * - Validates loop variable counts against src/dst ranges (ICHECK on mismatch).
 * - Computes indexed accesses and emits optional bound predicates;
 * out-of-bounds accesses are masked to zero when predicates are uncertain.
 * - Emits an extern `call_extern("AtomicAdd", address_of(dst_value),
 * src_value)` call wrapped in an Evaluate statement.
 * - Wraps the body with a parallel For at each loop level. If `coalesced_width`
 * is defined it is attached as the "coalesced_width" annotation on each loop.
 *
 * Note: This function mutates the analyzer binding state by binding loop
 * variables and may fail via ICHECK if internal assumptions about shapes are
 * violated.
 *
 * @return A nested For loop (parallel loops) implementing the atomic-add
 * kernel. For scalar cases a serial For of extent 1 is returned.
 */
For AtomicAddNode::MakeSIMTLoop(arith::Analyzer *analyzer) const {
  Array<IterVar> loop_vars = MakeIterVars();
  bool is_scalar = loop_vars.empty();
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

  Array<PrimExpr> new_args;

  // Optional bounds predicates for src and dst
  PrimExpr src_predicate = MakePredicate(analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate = MakePredicate(analyzer, loop_vars, dst->shape, 1);

  // Load source value and cast to dst dtype if needed
  PrimExpr src_value = BufferLoad(src, src_indices);
  if (src->dtype != dst->dtype)
    src_value = Cast(dst->dtype, src_value);

  // Build a pointer to destination element using tvm_access_ptr
  PrimExpr dst_ptr = Call(DataType::Handle(), builtin::address_of(),
                          {BufferLoad(dst, dst_indices)});

  new_args.push_back(dst_ptr);
  new_args.push_back(src_value);
  new_args.push_back(GetMemoryOrder());

  // erase use_tma from annotations
  auto annotations = this->annotations;
  annotations.erase("use_tma");
  Call atomicadd_call =
      tvm::tir::Call(dst->dtype, atomicadd_elem_op(), new_args, annotations);

  Stmt body = tvm::tir::Evaluate(atomicadd_call);

  for (int i = loop_vars.size() - 1; i >= 0; i--) {
    Map<String, ObjectRef> loop_annotations;
    if (i == 0) {
      if (annotations.count(attr::kCoalescedWidth)) {
        loop_annotations.Set(attr::kCoalescedWidth,
                             annotations.Get(attr::kCoalescedWidth).value());
      }
    }

    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body, std::nullopt, loop_annotations);
  }
  return Downcast<For>(body);
}

/**
 * @brief Compute linear layout for shared tensor (used in TMA atomic add).
 *
 * Creates a tiled layout that splits each dimension into blocks of 256
 * elements. The layout maps [i, j, ...] to [i // 256, j // 256, ..., i % 256, j
 * % 256, ...].
 *
 * @param shared_tensor The shared memory buffer to compute layout for.
 * @return Layout A tiled linear layout for the buffer.
 */
Layout AtomicAddNode::ComputeLinearLayout(const Buffer &shared_tensor) const {
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
 * @brief Infer and return the layout map for the atomic add operator.
 *
 * For TMA atomic add operations (when use_tma=True):
 *   - src is always shared memory, dst is always global memory
 *   - Automatically applies swizzle layout to the shared memory buffer when
 *     the operation is not 1D, improving memory access efficiency
 *
 * For non-TMA atomic add operations:
 *   - Returns empty layout map (no layout inference needed)
 *
 * @param T Layout inference inputs, including an optional mapping of buffers to
 * layouts.
 * @param level Inference strictness level.
 * @return LayoutMap The inferred layout mapping for buffers used by this
 * operator.
 */
LayoutMap AtomicAddNode::InferLayout(const LayoutInferArgs &T,
                                     InferLevel level) const {
  // Handle TMA atomic add layout inference
  if (GetUseTMA()) {
    Map<Buffer, Layout> result_map;

    // For TMA atomic add: src is shared memory, dst is global memory
    Buffer shared_tensor = src;
    Array<Range> shared_range = src_range;

    // Check if this is 1D TMA
    bool is_tma_1d = shared_range.size() == 1;

    if (is_tma_1d) {
      // 1D TMA atomic add with single dimension cannot be swizzled
      return result_map;
    }

    // For non-1D TMA atomic add, apply swizzle layout if possible
    if (level == InferLevel::kFree && !T.layout_map.count(shared_tensor)) {
      // TMA atomic add is similar to TMA Store - we should perform swizzle if
      // possible Use the last two dimensions to analyze swizzling
      int dim = shared_tensor->shape.size();
      const int64_t mat_stride = *as_const_int(shared_tensor->shape[dim - 2]);
      const int64_t mat_continuous =
          *as_const_int(shared_tensor->shape[dim - 1]);
      Layout swizzle_layout =
          makeGemmABLayoutHopper(mat_stride, mat_continuous, mat_continuous,
                                 shared_tensor->dtype.bits(), /*k_inner=*/true);
      // If makeGemmABLayoutHopper returns a linear layout, fallback to
      // ComputeLinearLayout which handles arbitrary tensor shapes correctly.
      if (StructuralEqual()(swizzle_layout, makeLinearLayout(Array<PrimExpr>{
                                                Integer(mat_stride),
                                                Integer(mat_continuous)}))) {
        result_map.Set(shared_tensor, ComputeLinearLayout(shared_tensor));
      } else {
        result_map.Set(shared_tensor, swizzle_layout);
      }
    }

    return result_map;
  }

  // For non-TMA atomic add, check that src and dst have the same layout if both
  // are fragments
  if (IsFragmentBuffer(src) && IsFragmentBuffer(dst)) {
    if (T.layout_map.count(src) && T.layout_map.count(dst)) {
      Layout src_layout = T.layout_map.at(src);
      Layout dst_layout = T.layout_map.at(dst);
      ICHECK(StructuralEqual()(src_layout, dst_layout))
          << "AtomicAdd requires src and dst to have the same layout, but got "
          << "src layout: " << src_layout << ", dst layout: " << dst_layout
          << " for src buffer: " << src->name << ", dst buffer: " << dst->name;
    }
  }
  return {};
}

/**
 * @brief Lower the atomic-add top-level operator into a parallel, vectorized
 * TIR loop.
 *
 * Constructs a SIMT-style loop for the atomic-add, fuses parallel loops, runs
 * layout inference at multiple levels, partitions the root loop by the provided
 * thread variable, vectorizes the thread loop, and returns the final
 * (optionally predicate-guarded) statement.
 *
 * The lowering pipeline:
 *  - Build the SIMT loop via MakeSIMTLoop.
 *  - Fuse parallel loops into a single For and wrap as a ParallelOp.
 *  - Run layout inference at kCommon, kStrict, and kFree levels using fields
 * from `T`.
 *  - Obtain the loop layout, partition the root loop with PartitionLoop by
 * `T.thread_var`.
 *  - Vectorize the partitioned thread loop via VectorizeLoop.
 *  - If the ParallelOp produced a predicate for `T.thread_var`, return an
 * IfThenElse that guards the vectorized loop with that predicate; otherwise
 * return the vectorized loop.
 *
 * @param T Lowering context whose fields are used:
 *   - T.target: target architecture for layout inference and lowering
 * decisions.
 *   - T.thread_var: the Var used to partition the outer loop for thread-level
 * parallelism.
 *   - T.thread_bounds: bounds associated with the thread dimension (used during
 * partitioning).
 *   - T.layout_map, T.buffer_remap: layout and buffer remapping inputs used
 * during InferLayout.
 * @param analyzer Analyzer used for symbolic reasoning during partitioning and
 * folding (omitted from detailed param docs as a common analysis utility).
 * @return Stmt A lowered TIR statement representing the parallelized and
 * vectorized atomic-add.
 */
Stmt AtomicAddNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  Target target = T.target;
  if (GetUseTMA()) {
    // For AtomicAdd with TMA: src is shared memory, dst is global memory
    // Use cp.reduce.async.bulk.tensor instruction with tensor descriptor
    Buffer shared_tensor = src;
    Buffer global_tensor = dst;
    Array<Range> shared_range = src_range;
    Array<Range> global_range = dst_range;

    // Build TMADesc for the global tensor
    TMADesc desc;
    desc.rank = global_tensor->shape.size();
    ICHECK(desc.rank >= 1 && desc.rank <= 5)
        << "TMA reduce only supports 1-5 dimensions, got " << desc.rank;

    // Data type must match
    ICHECK(global_tensor->dtype == shared_tensor->dtype)
        << "AtomicAdd between buffer " << shared_tensor->name << " and "
        << global_tensor->name << " with different data type "
        << shared_tensor->dtype << " and " << global_tensor->dtype;

    desc.data_type = to_CUtensorMapDataType(global_tensor->dtype);

    // Global tensor shape and stride
    desc.global_addr = global_tensor->data;
    desc.global_shape = ReverseArray(global_tensor->shape);
    Array<PrimExpr> global_coords =
        ReverseArray(global_range.Map([](Range r) { return r->min; }));

    if (!global_tensor->strides.empty()) {
      desc.global_stride = ReverseArray(global_tensor->strides);
    } else {
      // Create stride from shape (row-major)
      PrimExpr stride = 1;
      desc.global_stride.reserve(desc.rank);
      for (size_t i = 0; i < desc.rank; i++) {
        desc.global_stride.push_back(stride);
        stride *= desc.global_shape[i];
      }
    }
    // Make global stride in bytes
    desc.global_stride = desc.global_stride.Map([&](PrimExpr e) {
      return cast(DataType::Int(64), e) * global_tensor->dtype.bytes();
    });

    // Shared memory box (copy extent)
    desc.smem_box =
        ReverseArray(global_range.Map([](Range r) { return r->extent; }));
    desc.smem_stride = Array<PrimExpr>(desc.rank, PrimExpr(1));

    // L2 & OOB settings
    desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
    desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // Detect smem layout for swizzle (similar to copy.cc)
    // linear layout must be computed before remapping
    auto linear_layout = makeLinearLayout(shared_tensor->shape);
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
        LOG(WARNING) << "AtomicAdd TMA cannot support a padded layout for src: "
                     << src->name << ", dst: " << dst->name;
        desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
      } else {
        LOG(WARNING) << "AtomicAdd TMA unsupported swizzle layout for src: "
                     << src->name << ", dst: " << dst->name;
        desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
      }
    }

    // Adjust instruction_dim based on swizzle type (similar to copy.cc)
    auto inner_box_dim = as_const_int(desc.smem_box[0]);
    ICHECK(inner_box_dim != nullptr)
        << "inner_box_dim must be a constant integer for TMA atomic add";
    int instruction_dim = *inner_box_dim;
    if (desc.swizzle == static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B)) {
      instruction_dim = 64 / shared_tensor->dtype.bytes();
    } else if (desc.swizzle == static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B)) {
      instruction_dim = 128 / shared_tensor->dtype.bytes();
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

    int inner_box_dim_ = instruction_dim * shared_tensor->dtype.bytes();
    // Check inner_box_dim_ for each swizzle type
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
        LOG(WARNING) << "AtomicAdd TMA cannot support swizzled layout with "
                        "inner_box_dim_ > "
                     << check.max_dim;
      }
    }

    // Compute shared memory offset
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
    PrimExpr shared_offset = 0;
    for (size_t i = 0; i < shared_indices.size(); i++) {
      shared_offset += shared_indices[i] * shared_strides[i];
    }

    // Create TMA descriptor
    Call create_descriptor = Call(DataType::Handle(), create_tma_descriptor(),
                                  desc.EncodeCallArgs());

    // Compute total elements for access_ptr
    PrimExpr total_elements = 1;
    for (auto e : desc.smem_box)
      total_elements *= e;

    // erase use_tma from annotations
    auto op_annotations = this->annotations;
    op_annotations.erase("use_tma");

    Stmt tma_reduce;
    if ((*inner_box_dim) != instruction_dim) {
      // Need to split the operation into multiple TMA calls
      Var loop_var("i");
      int loop_extent = (*inner_box_dim) / instruction_dim;

      Array<PrimExpr> args;
      args.reserve(desc.rank + 4);
      args.push_back(create_descriptor);
      PrimExpr shared_addr = shared_tensor.access_ptr(
          1, DataType::Handle(), 1, shared_offset + total_elements * loop_var,
          total_elements);
      args.push_back(shared_addr);
      Array<PrimExpr> loop_global_coords = global_coords;
      loop_global_coords.Set(0, global_coords[0] + instruction_dim * loop_var);
      for (auto coord : loop_global_coords)
        args.push_back(coord);
      int need_reduce = 1;
      args.push_back(need_reduce);
      int eviction_policy = 0;
      args.push_back(eviction_policy);
      tma_reduce = For(loop_var, 0, loop_extent, ForKind::kUnrolled,
                       Evaluate(Call(DataType::Handle(), tma_store(), args,
                                     op_annotations)));
    } else {
      Array<PrimExpr> args;
      args.reserve(desc.rank + 4);
      args.push_back(create_descriptor);
      PrimExpr shared_addr = shared_tensor.access_ptr(
          1, DataType::Handle(), 1, shared_offset, total_elements);
      args.push_back(shared_addr);
      for (auto coord : global_coords)
        args.push_back(coord);
      int need_reduce = 1;
      args.push_back(need_reduce);
      int eviction_policy = 0;
      args.push_back(eviction_policy);
      tma_reduce =
          Evaluate(Call(DataType::Handle(), tma_store(), args, op_annotations));
    }

    return IfThenElse(EQ(T.thread_var, T.thread_bounds->min), tma_reduce);
  }
  auto simt_loop = MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));

  auto GetArchInt = [&](const Target &tgt) -> int {
    int arch_int = 0;
    if (auto s = tgt->GetAttr<String>("arch")) {
      std::string arch = s.value();
      if (arch.rfind("sm_", 0) == 0)
        arch_int = std::stoi(arch.substr(3));
    }
    return arch_int;
  };

  struct AtomicLoopNestCollector : tir::StmtExprVisitor {
    Array<IterVar> loop_vars;
    Map<Buffer, Array<PrimExpr>> indice_map;
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> writes;
    arith::Analyzer analyzer;

    void Run(const Stmt &s) { StmtExprVisitor::VisitStmt(s); }

    void VisitStmt_(const ForNode *op) final {
      if (op->kind == ForKind::kParallel) {
        loop_vars.push_back(IterVar(Range(op->min, op->extent), op->loop_var,
                                    IterVarType::kDataPar));
      }
      analyzer.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const BufferStoreNode *op) final {
      if (IsFragmentBuffer(op->buffer)) {
        indice_map.Set(op->buffer, op->indices);
        writes.insert(op->buffer);
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const BufferLoadNode *op) final {
      if (IsFragmentBuffer(op->buffer)) {
        indice_map.Set(op->buffer, op->indices);
      }
      StmtExprVisitor::VisitExpr_(op);
    }
  };

  auto ComputeLoopLayoutFromBuffer =
      [&](const Buffer &buf, const Array<PrimExpr> &indices,
          const LayoutMap &layout_map, const Range &thread_bounds,
          const Array<IterVar> &loop_vars) -> Fragment {
    Fragment src = layout_map[buf].as<Fragment>().value();
    Var rep;
    auto rep_iter =
        IterVar(Range(0, src->ReplicateExtent()), rep, IterVarType::kDataPar);
    PrimExpr fth = src->ForwardThread(indices, rep);
    fth = analyzer->Simplify(fth);
    Fragment out = Fragment(loop_vars, /*forward_index=*/{}, fth, rep_iter)
                       ->BindThreadRange(thread_bounds);
    return out;
  };

  struct AtomicInferResult {
    Fragment loop_layout;
    Optional<PrimExpr> predicate;
  };

  auto AtomicAddInferLayout =
      [&](const For &loop, const LayoutInferArgs &args) -> AtomicInferResult {
    AtomicLoopNestCollector C;
    C.Run(loop);
    Optional<Buffer> read_src;
    int best_rank = -1;
    for (auto kv : C.indice_map) {
      const Buffer &buf = kv.first;
      if (!IsFragmentBuffer(buf))
        continue;
      if (!args.layout_map.count(buf))
        continue;
      int rank = static_cast<int>(kv.second.size());
      if (rank > best_rank) {
        best_rank = rank;
        read_src = buf;
      }
    }
    AtomicAddVectorizePlanner planner;
    int sm = GetArchInt(target);
    auto plan = planner.Plan(loop, sm);
    int vec = std::max(plan.vector_size, 1);
    if (auto cw = loop->annotations.Get(attr::kCoalescedWidth)) {
      if (const auto *imm = cw->as<IntImmNode>()) {
        int expected = imm->value;
        ICHECK_GT(expected, 0);
        ICHECK(vec % expected == 0)
            << "vector_size " << vec << " not divisible by coalesced_width "
            << expected;
        vec = expected;
      } else {
        LOG(FATAL) << "coalesced_width should be IntImmNode.";
      }
    }
    PrimExpr total = 1;
    for (Stmt s = loop; s.as<For>().has_value(); s = s.as<For>().value()->body)
      total = total * s.as<For>().value()->extent;
    PrimExpr denom = args.thread_bounds->extent * vec;
    while (!analyzer->CanProve(floormod(total, denom) == 0) && vec > 1) {
      vec >>= 1;
      denom = args.thread_bounds->extent * vec;
    }
    if (vec < 1)
      vec = 1;
    Fragment loop_layout;
    if (read_src) {
      loop_layout = ComputeLoopLayoutFromBuffer(
          read_src.value(), C.indice_map[read_src.value()], args.layout_map,
          args.thread_bounds, C.loop_vars);
    } else {
      const For &remapped = loop;
      loop_layout = PlanLoopPartition(remapped, vec, args.thread_bounds);
    }

    Optional<PrimExpr> pred;
    if (plan.dynamic && plan.condition.defined()) {
      pred = plan.condition;
    }
    DLOG(INFO) << "[AtomicAddInferLayout] vec=" << vec
               << " loop_layout=" << loop_layout->DebugOutput();
    return {loop_layout, pred};
  };

  auto ret =
      AtomicAddInferLayout(fused_loop, {T.target, T.thread_bounds, T.layout_map,
                                        analyzer, false, T.buffer_remap});
  Fragment loop_layout = ret.loop_layout;
  auto thread_loop =
      PartitionLoop(fused_loop, T.thread_var, analyzer, loop_layout);
  auto vectorized_thread_loop =
      VectorizeAtomicAdd(thread_loop, GetArchInt(target));
  return vectorized_thread_loop;
}

TIR_REGISTER_TL_TILE_OP(AtomicAdd, atomicadd)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { AtomicAddNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
