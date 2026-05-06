/*!
 * \file tl/op/atomic_add.cc
 *
 * Define element-wise operators.
 */

#include "./atomic_add.h"
#include "utils.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../transform/common/loop_fusion_utils.h"
#include "../transform/loop_partition.h"
#include "builtin.h"

#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

std::vector<AtomicAddImpl> &AtomicAddImplRegistry() {
  static std::vector<AtomicAddImpl> registry;
  return registry;
}

const AtomicAddImpl &ResolveAtomicAddImpl(Target target) {
  const auto &registry = AtomicAddImplRegistry();
  const AtomicAddImpl *matched_impl = nullptr;
  for (const AtomicAddImpl &impl : registry) {
    if (impl.match_target(target)) {
      ICHECK(matched_impl == nullptr)
          << "tl.atomic_add found multiple target-specific implementations for "
          << target->ToDebugString() << ": " << matched_impl->name << " and "
          << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.atomic_add requires a target-specific implementation, but no "
         "atomic_add implementation is registered for "
      << target->ToDebugString();
  return *matched_impl;
}

} // namespace

void RegisterAtomicAddImpl(AtomicAddImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.infer_layout != nullptr);
  ICHECK(impl.lower != nullptr);
  AtomicAddImplRegistry().push_back(impl);
}

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
  std::vector<AccessRegion> access_regions;

  if (IsBufferLikeExpr(args[0])) {
    auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
    node->src = src_access.region->buffer;
    node->src_range = src_access.region->region;
    access_regions.push_back(std::move(src_access));
  } else {
    node->src_value = args[0];
  }

  auto dst_access = NormalizeToAccessRegion(args[1], kAccessReadWrite);
  dst_access.access_mask = kAccessReadWrite;
  node->dst = dst_access.region->buffer;
  node->dst_range = dst_access.region->region;
  access_regions.push_back(std::move(dst_access));
  node->SetAccessRegions(std::move(access_regions));

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
 * @brief Infer and return the layout map for the target-independent SIMT atomic
 * add path.
 */
LayoutMap AtomicAddNode::InferSIMTLayout(const LayoutInferArgs &T,
                                         InferLevel level) const {
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

LayoutMap AtomicAddNode::InferLayout(const LayoutInferArgs &T,
                                     InferLevel level) const {
  return ResolveAtomicAddImpl(T.target).infer_layout(*this, T, level);
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
Stmt AtomicAddNode::LowerSIMT(const LowerArgs &T,
                              arith::Analyzer *analyzer) const {
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

Stmt AtomicAddNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  return ResolveAtomicAddImpl(T.target).lower(*this, T, analyzer);
}

TIR_REGISTER_TL_TILE_OP(AtomicAdd, atomicadd)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { AtomicAddNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
