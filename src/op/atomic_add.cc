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
  ICHECK(args.size() >= 2)
      << "AtomicAdd expects at least 2 arguments (src, dst), got "
      << args.size();
  ObjectPtr<AtomicAddNode> node = tvm::ffi::make_object<AtomicAddNode>();

  if (IsBufferLikeExpr(args[0])) {
    auto region = NormalizeToBufferRegion(args[0]);
    node->src = region->buffer;
    node->src_range = region->region;
  } else {
    node->src_value = args[0];
  }

  auto region = NormalizeToBufferRegion(args[1]);
  node->dst = region->buffer;
  node->dst_range = region->region;

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

const Op &AtomicAddNode::GetElemOp() const { return atomic_add_elem_op(); }

/**
 * @brief Get vectorization length based on dst dtype and target SM version.
 *
 * Returns:
 *   - 2 for float16/bfloat16
 *   - 4 for float32 on SM >= 90
 *   - 1 for all other cases
 *
 * @param target The target architecture to check SM version.
 * @return int The vectorization length.
 */
int AtomicAddNode::GetVectorizeLength(Target target) const {
  DataType dtype = dst->dtype;
  if (dtype.is_float16() || dtype.is_bfloat16()) {
    return 2;
  }
  if (dtype.is_float() && dtype.bits() == 32 &&
      TargetHasSMVersionGE(target, 90)) {
    return 4;
  }
  return 1;
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
 * - Emits an extern `call_intrin(op.Op.get("tl.atomic_add_elem_op"),
 * address_of(dst_value), src_value), annotations)` call wrapped in an Evaluate
 * statement.
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
  ICHECK(!loop_vars.empty()) << "MakeIterVars in AtomicOp should not return "
                                "empty vars (at least 1 var)";

  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);

  ICHECK(loop_vars.size() <= dst_range.size())
      << "loop_vars.size() = " << loop_vars.size()
      << ", dst_range.size() = " << dst_range.size() << ", dst = " << dst->name;

  Array<PrimExpr> dst_indices = MakeIndices(loop_vars, 1);
  Array<PrimExpr> new_args;

  // Optional bounds predicates for src and dst
  PrimExpr dst_predicate = MakePredicate(analyzer, loop_vars, dst->shape, 1);

  // Src arg to be passed to the Call atomic operation
  PrimExpr src_value_arg;

  // If src is a Buffer
  if (!src_value.defined()) {
    ICHECK(loop_vars.size() <= src_range.size())
        << "loop_vars.size() = " << loop_vars.size()
        << ", src_range.size() = " << src_range.size()
        << ", src = " << src->name << ", dst = " << dst->name;

    Array<PrimExpr> src_indices = MakeIndices(loop_vars, 0);
    PrimExpr src_predicate = MakePredicate(analyzer, loop_vars, src->shape, 0);
    // Load source value
    src_value_arg = BufferLoad(src, src_indices);
  } else {
    src_value_arg = src_value;
  }
  // Cast to dst dtype if needed
  if (src_value_arg->dtype != dst->dtype)
    src_value_arg = Cast(dst->dtype, src_value_arg);

  // Build an access pointer to the destination element (rw).
  DataType idx_dtype =
      dst_indices.empty() ? DataType::Int(32) : dst_indices[0].dtype();
  PrimExpr dst_ptr =
      Call(DataType::Handle(), tl::access_ptr(),
           {BufferLoad(dst, dst_indices), make_const(idx_dtype, 1),
            make_const(DataType::Int(32), 3)});

  new_args.push_back(dst_ptr);
  new_args.push_back(src_value_arg);
  new_args.push_back(GetMemoryOrder());

  // erase use_tma from annotations
  auto annotations = this->annotations;
  annotations.erase("use_tma");
  Call atomicadd_call =
      tvm::tir::Call(dst->dtype, atomic_add_elem_op(), new_args, annotations);

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
      Layout swizzle_layout_2d =
          makeGemmABLayoutHopper(mat_stride, mat_continuous, mat_continuous,
                                 shared_tensor->dtype.bits(), /*k_inner=*/true);
      // If makeGemmABLayoutHopper returns a linear layout, fallback to
      // ComputeLinearLayout which handles arbitrary tensor shapes correctly.
      if (StructuralEqual()(swizzle_layout_2d, makeLinearLayout(Array<PrimExpr>{
                                                   Integer(mat_stride),
                                                   Integer(mat_continuous)}))) {
        result_map.Set(shared_tensor, ComputeLinearLayout(shared_tensor));
      } else {
        result_map.Set(shared_tensor, ExpandLayoutToMatchBuffer(
                                          swizzle_layout_2d, shared_tensor));
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
    Buffer shared_tensor_unmapped = shared_tensor;
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
      ICHECK(shared_layout->InputDim() >= 2) << "Cannot detect TMA layout.";
      const int ndim = static_cast<int>(shared_layout->InputDim());
      auto stride = as_const_int(shared_layout->InputShape()[ndim - 2]);
      auto continuous = as_const_int(shared_layout->InputShape()[ndim - 1]);
      ICHECK(stride != nullptr && continuous != nullptr);
      if (StructuralEqual()(shared_layout, makeQuarterBankSwizzleLayout(
                                               shared_tensor_unmapped))) {
        desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B);
      } else if (StructuralEqual()(
                     shared_layout,
                     makeHalfBankSwizzleLayout(shared_tensor_unmapped))) {
        desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B);
      } else if (StructuralEqual()(
                     shared_layout,
                     makeFullBankSwizzleLayout(shared_tensor_unmapped))) {
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

    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(tma_reduce);
    seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_arrive(), {})));
    seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_wait(), {})));
    return IfThenElse(EQ(T.thread_var, T.thread_bounds->min),
                      SeqStmt(std::move(seq)));
  }
  auto simt_loop = MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
  auto par_op = ParallelOp(fused_loop);
  std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                    InferLevel::kFree};
  // 1.give par_op a recommended vectorize size. (only works for free layout
  // inference).
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
  auto lowered_loop =
      LowerParallelLoop(fused_loop, loop_layout, T.thread_var, analyzer,
                        T.layout_map, par_op->GetPredicate(T.thread_var));
  return lowered_loop;
}

TIR_REGISTER_TL_TILE_OP(AtomicAdd, atomicadd)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { AtomicAddNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
