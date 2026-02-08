/*!
 * \file op/parallel.cc
 * \brief Define Parallel for operator
 */

#include "parallel.h"

#include <algorithm>
#include <tvm/tir/op.h>

#include "../layout/layout.h"
#include "arith/int_operator.h"

#include "../layout/utils.h"
#include "../target/utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

class IfBufferRemapLoopGenerator : public StmtExprMutator {
public:
  static For run(Stmt stmt, Map<Buffer, Buffer> buffer_remap,
                 Map<Buffer, Layout> layout_map) {
    IfBufferRemapLoopGenerator generator(buffer_remap, layout_map);
    return Downcast<For>(generator(std::move(stmt)));
  }

private:
  IfBufferRemapLoopGenerator(Map<Buffer, Buffer> buffer_remap,
                             Map<Buffer, Layout> layout_map)
      : buffer_remap_(buffer_remap), layout_map_(layout_map) {}

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));

    if (buffer_remap_.count(load->buffer)) {
      auto new_indices = layout_map_[load->buffer]->Forward(load->indices);
      auto new_buffer = buffer_remap_[load->buffer];

      return BufferLoad(new_buffer, new_indices);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (buffer_remap_.count(store->buffer)) {
      auto new_indices = layout_map_[store->buffer]->Forward(store->indices);
      auto new_buffer = buffer_remap_[store->buffer];
      return BufferStore(new_buffer, store->value, new_indices);
    }
    return store;
  }

  Map<Buffer, Buffer> buffer_remap_;
  Map<Buffer, Layout> layout_map_;
};

} // anonymous namespace

/**
 * @brief Handle a parallel For node during traversal, collecting loop metadata.
 *
 * Visits a parallel loop, asserts the loop is parallel, records a data-parallel
 * IterVar for the loop, binds the loop variable range into the analyzer scope,
 * and extracts any reducer information from the loop's annotations into the
 * visitor's reducer_info_map_. Continues traversal into the loop body.
 */
void ParallelLoopNestVisitor::VisitStmt_(const ForNode *op) {
  if (op->kind == ForKind::kParallel)
    p->loop_vars_.push_back(IterVar(Range(op->min, op->extent), op->loop_var,
                                    IterVarType::kDataPar));
  else
    p->inner_vars_.Set(op->loop_var,
                       IterVar(Range(op->min, op->extent), op->loop_var,
                               IterVarType::kOrdered));
  p->analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
  auto reducer_info_map =
      op->annotations.Get(attr::kReducerInfo)->as<Map<Var, ReducerInfo>>();
  if (reducer_info_map) {
    for (auto &&[buffer, info] : reducer_info_map.value())
      p->reducer_info_map_.Set(buffer, info);
  }
  StmtExprVisitor::VisitStmt_(op);
}

void ParallelLoopNestVisitor::VisitStmt_(const BufferStoreNode *op) {
  if (IsFragmentBuffer(op->buffer)) {
    p->RecordBufferAccess(op->buffer, op->indices, /*is_write=*/true);
  }
  StmtExprVisitor::VisitStmt_(op);
}

void ParallelLoopNestVisitor::VisitExpr_(const BufferLoadNode *op) {
  if (IsFragmentBuffer(op->buffer)) {
    p->RecordBufferAccess(op->buffer, op->indices, /*is_write=*/false);
  }
  StmtExprVisitor::VisitExpr_(op);
}

ParallelOpNode::ParallelOpNode(For root) : root_(root), V(this) {
  V.VisitStmt(root);
  // Cache any annotated layout/predicate on the outermost loop.
  using namespace attr;
  if (root_->annotations.count(kParallelLoopLayout)) {
    annotated_layout_unbound_ =
        Downcast<Fragment>(root_->annotations.Get(kParallelLoopLayout).value());
  }
  if (root_->annotations.count(kParallelLoopPredicate)) {
    annotated_predicate_ = Downcast<PrimExpr>(
        root_->annotations.Get(kParallelLoopPredicate).value());
  }
  // Collect cross-thread access info and buffer store info.
  PostOrderVisit(root_, [&](const ObjectRef &obj) {
    if (const auto *store = obj.as<BufferStoreNode>()) {
      auto buffer = store->buffer;
      if (IsSharedBuffer(buffer) || IsGlobalBuffer(buffer)) {
        has_cross_thread_access_ = true;
        store_shared_global_buffers_.emplace_back(buffer);
      } else if (IsFragmentBuffer(buffer)) {
        store_fragment_buffers_.emplace_back(buffer);
      }
    } else if (const auto *load = obj.as<BufferLoadNode>()) {
      if (IsSharedBuffer(load->buffer) || IsGlobalBuffer(load->buffer)) {
        has_cross_thread_access_ = true;
      }
    }
  });
}

TileOperator ParallelOpNode::Clone() const {
  auto op = tvm::ffi::make_object<ParallelOpNode>(*this);
  return ParallelOp(op);
}

void ParallelOpNode::ExpandLetBindings(
    const Map<Var, PrimExpr> &let_var_to_expr) {
  if (let_var_to_expr.empty())
    return;

  // Helper function to recursively find BufferLoads through let bindings
  std::function<void(const PrimExpr &)> expand = [&](const PrimExpr &expr) {
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (auto bl = node.as<BufferLoadNode>()) {
        if (IsFragmentBuffer(bl->buffer)) {
          RecordBufferAccess(bl->buffer, bl->indices, /*is_write=*/false);
        }
      } else if (auto var_node = node.as<VarNode>()) {
        auto var = tvm::ffi::GetRef<Var>(var_node);
        if (let_var_to_expr.count(var)) {
          expand(let_var_to_expr[var]);
        }
      }
    });
  };

  // Only expand let bindings that are used in root_
  // First, collect all vars used in root_
  std::unordered_set<const VarNode *> used_vars;
  PostOrderVisit(root_, [&](const ObjectRef &node) {
    if (auto var_node = node.as<VarNode>()) {
      used_vars.insert(var_node);
    }
  });

  // Only expand let bindings for vars that are actually used in root_
  for (const auto &[var, expr] : let_var_to_expr) {
    if (used_vars.count(var.get())) {
      expand(expr);
    }
  }
}

void ParallelOpNode::RecordBufferAccess(const Buffer &buffer,
                                        const Array<PrimExpr> &indices,
                                        bool is_write) {
  auto it = indice_map_.find(buffer);
  if (it != indice_map_.end()) {
    ICHECK(StructuralEqual()(it->second.indices, indices))
        << buffer << ": " << indices << " and " << it->second.indices;
  } else {
    BufferAccessInfo info;
    info.indices = indices;
    it = indice_map_.emplace(buffer, std::move(info)).first;
  }
  if (is_write) {
    it->second.is_write = true;
  } else {
    it->second.is_read = true;
  }
}

const ParallelOpNode::BufferAccessInfo &
ParallelOpNode::GetAccessInfo(const Buffer &buffer) const {
  auto it = indice_map_.find(buffer);
  ICHECK(it != indice_map_.end())
      << "Missing access info for buffer " << buffer;
  return it->second;
}

bool ParallelOpNode::IsBufferCompletelyReplicated(
    const Buffer &buffer, const LayoutMap &layout_map) const {
  if (!IsFragmentBuffer(buffer))
    return false;
  auto frag = layout_map[buffer].as<Fragment>().value();
  // buffer indices should be IntImm
  for (const auto &index : GetAccessInfo(buffer).indices) {
    if (!index.as<IntImmNode>()) {
      return false;
    } else if (index.as<IntImmNode>()->value != 0) {
      LOG(FATAL) << "buffer " << buffer << " is not completed replicated";
    }
  }
  return frag->IsCompletedReplicated();
}

Stmt ParallelOpNode::Lower(const LowerArgs &T,
                           arith::Analyzer *analyzer) const {
  return root_;
}

// (annotations parsed in ctor; adoption happens in InferLayout)

bool ParallelOpNode::IsCommonAccessIndice(const Buffer &buffer) const {
  auto common_indice = loop_vars_.Map([](const auto &iv) { return iv->var; });
  return StructuralEqual()(GetAccessInfo(buffer).indices, common_indice);
}

/*! \brief Infer the layout for parallel operations based on different inference
 * levels
 *
 * The inference level controls how aggressively we try to infer and optimize
 * layouts:
 * - kStrict (2): Most conservative level. Only allows explicitly defined
 * layouts. Returns empty layout map if loop_layout_ is not already defined.
 *                Used when exact layout control is required.
 *
 * - kCommon (1): Intermediate level between strict and free.
 *                Allows common layout patterns while maintaining some
 * constraints.
 *
 * - kFree (0):   Most permissive level. Allows maximum optimization freedom.
 *                Will attempt layout inference even without source buffers.
 *                Can generate new layouts based on vectorization and thread
 * bounds. Used when maximum performance optimization is desired.
 */
LayoutMap ParallelOpNode::InferLayout(const LayoutInferArgs &T,
                                      InferLevel level) const {
  if (loop_layout_inferred_)
    return {};

  // Expand let bindings to find fragment buffer accesses
  if (!T.let_var_to_expr.empty()) {
    const_cast<ParallelOpNode *>(this)->ExpandLetBindings(T.let_var_to_expr);
  }

  if (level == InferLevel::kStrict) {
    LayoutMap results;
    // Deduce buffers that should be complicated replicated.
    // For example:
    // for i in T.Parallel(m):
    //   fragment[0] = x[i]
    // then fragment[0] must be replicated on all threads.
    for (const auto &[buffer, access] : indice_map_) {
      if (T.layout_map.count(buffer)) {
        continue;
      }
      if (!IsFragmentBuffer(buffer))
        continue;

      // Check if all indices are zero
      bool all_indices_zero = true;
      for (const auto &index : access.indices) {
        if (const auto *imm = index.as<IntImmNode>()) {
          if (imm->value != 0) {
            all_indices_zero = false;
            LOG(FATAL)
                << "Fragment buffer access with non-zero index [" << imm->value
                << "] is not supported. "
                << "Only fragment[0] access is allowed within T.Parallel loop.";
          }
        } else {
          // Non-constant index, not all zero
          all_indices_zero = false;
        }
      }

      // Only set layout if all indices are zero
      if (all_indices_zero) {
        Array<IterVar> forward_vars;
        for (const auto &s : buffer->shape) {
          forward_vars.push_back(
              IterVar(Range(0, s), Var(), IterVarType::kDataPar));
        }
        Var rep;
        auto rep_iter =
            IterVar({0, T.thread_bounds->extent}, rep, IterVarType::kDataPar);

        // Use default fragment indexing (single output dim) to
        // stay consistent with other ops (e.g., ReduceOp), and
        // bind the thread range for comparability.
        const PrimExpr &forward_thread = rep;
        auto frag = Fragment(forward_vars, /*forward_index=*/{}, forward_thread,
                             rep_iter)
                        ->BindThreadRange(T.thread_bounds);
        results.Set(buffer, frag);
      }
    }
    return results;
  }

  // Collect fragment buffers with const index and all fragment_buffers
  std::vector<Buffer> const_index_fragment_buffer, fragment_buffers;
  for (const auto &[buffer, access] : indice_map_) {
    if (!IsFragmentBuffer(buffer))
      continue;
    fragment_buffers.push_back(buffer);

    bool is_const_index = true;
    for (const auto &index : access.indices) {
      if (!index.as<IntImmNode>()) {
        is_const_index = false;
        break;
      }
    }
    if (is_const_index) {
      const_index_fragment_buffer.push_back(buffer);
    }
  }

  // Determine if common layout propagation should be applied.
  // If there are fragment buffers with non-constant indices, we need to
  // propagate the common layout pattern to ensure consistency across all
  // fragments. Example cases:
  //   - Need propagation: frag_a[0] = T.min(frag_a[0], frag_b[i])
  //     (const index frag_a interacts with non-const index frag_b)
  //   - No propagation needed: shared_a[i] = frag_a[0]
  //     (const index frag_a with non-fragment buffer)

  bool allow_layout_propgate =
      const_index_fragment_buffer.empty() ||
      (fragment_buffers.size() > const_index_fragment_buffer.size());

  // Step 1: try to infer loop's partition from a source fragment
  Buffer source_buffer, read_source_buffer;
  Buffer replicated_write_buffer; // Backup: fully replicated write buffer

  for (const auto &[buffer, access] : indice_map_) {
    if (T.layout_map.count(buffer)) {
      // skip reducers with rep=ALL
      if (auto info = reducer_info_map_.Get(buffer->data);
          info && info.value()->rep == ReducerRepType::ALL)
        continue;

      auto frag = T.layout_map[buffer].as<Fragment>().value();
      bool is_fully_replicated =
          IsBufferCompletelyReplicated(buffer, T.layout_map);

      if (access.is_write) {
        source_buffer = buffer;
      } else {
        // Keep the buffer with largest number of indices
        // (which means the inference based on that buffer is more accurate)
        // as read_source_buffer to get more accurate layout
        // if the buffer is completed replicated, we don't need to infer the
        // layout from this buffer.
        if ((!read_source_buffer.defined() ||
             access.indices.size() >
                 GetAccessInfo(read_source_buffer).indices.size())) {
          read_source_buffer = buffer;
        }
        // If the buffer is not replicated and shape is equal to the
        // source_buffer, use it as source_buffer because the layout inference
        // is more accurate
        if (is_one(frag->ReplicateExtent()) && !source_buffer.defined()) {
          source_buffer = buffer;
        }
      }
    }
  }
  // moved to ComputeLoopLayoutFromBuffer

  // Try to infer loop layout from buffers in order of preference only if we
  // don't already have a layout (e.g., from annotations):
  // 1. Annotated loop layout
  // 2. Non-replicated write buffer (most reliable)
  // 3. Non-replicated read buffer
  // 4. Fully replicated write buffer (backup, may cause issues)
  // 5. Free inference mode (no source buffer)
  if (!loop_layout_.defined() && annotated_layout_unbound_.defined()) {
    loop_layout_ =
        annotated_layout_unbound_.value()->BindThreadRange(T.thread_bounds);
    if (annotated_predicate_.defined()) {
      predicate_ = annotated_predicate_.value();
    }
  } else if (!loop_layout_.defined() && source_buffer.defined() &&
             allow_layout_propgate) {
    loop_layout_ = ComputeLoopLayoutFromBuffer(source_buffer, T);
  } else if (!loop_layout_.defined() && level == InferLevel::kFree) {
    // For free layout inference
    // In free inference, try two mechanisms and prefer the one that
    // minimizes replication while remaining compatible:
    // 1) compute_loop_layout_from_buffer (always correct but may
    // over-replicate) 2) PlanLoopPartition (often smaller replication)
    Fragment candidate_from_buffer;
    Fragment candidate_from_plan;

    if (read_source_buffer.defined() && allow_layout_propgate) {
      candidate_from_buffer =
          ComputeLoopLayoutFromBuffer(read_source_buffer, T);
    }

    // try to infer loop layout with two mechanisms and choose the best one
    {
      candidate_from_plan = ComputePlanCandidate(T);
    }

    // Choose the best candidate:
    if (candidate_from_buffer.defined() && candidate_from_plan.defined()) {
      loop_layout_ =
          ChooseBestCandidate(candidate_from_buffer, candidate_from_plan, T);
    } else if (candidate_from_plan.defined()) {
      loop_layout_ = candidate_from_plan;
      DLOG(INFO) << "[FreeInfer] only PlanLoopPartition available, choose it.";
    } else if (candidate_from_buffer.defined()) {
      loop_layout_ = candidate_from_buffer;
      DLOG(INFO)
          << "[FreeInfer] only compute_from_buffer available, choose it.";
    }
  } else if (!loop_layout_.defined()) {
    // In non-free mode without a source buffer, if we don't have any layout
    // yet (e.g., no annotation), we have nothing to infer here.
    return {};
  }

  // check loop_layout_ is injective
  auto injective_res = loop_layout_->DetectInjective();
  if (!injective_res->errors.empty()) {
    std::ostringstream oss;
    oss << "Loop layout is not injective: " << loop_layout_->DebugOutput()
        << '\n'
        << "  errors: " << injective_res->errors << '\n'
        << "  loop AST: " << root_;
    throw LoopLayoutInjectiveException(oss.str());
  }

  PrimExpr loop_thread_extent = loop_layout_->ThreadExtent();

  auto block_size = T.thread_bounds->extent;
  if (loop_layout_.defined()) {
    if (loop_layout_->ThreadRange().defined()) {
      auto thread_range = loop_layout_->ThreadRange();
      block_size = thread_range->extent;
      AddPredicate(GE(InputPlaceholder(0), thread_range->min));
      AddPredicate(
          LT(InputPlaceholder(0), thread_range->min + thread_range->extent));
    }
  }

  if (!analyzer_.CanProveEqual(loop_thread_extent, block_size)) {
    AddPredicate(
        LT(InputPlaceholder(0), loop_thread_extent + T.thread_bounds->min));
  }

  // Step 2: Check that the loop's partition can correctly align with all source
  // fragment, and infer layout only when it's not yet layout-ed.
  ValidateCandidateAgainstFragments(loop_layout_, T, /*throw_on_error=*/true,
                                    /*check_forward_index=*/false,
                                    source_buffer);

  // Step 3: Build replication guards
  BuildReplicationGuardsIfNeeded(
      T, store_shared_global_buffers_, store_fragment_buffers_,
      has_cross_thread_access_, const_index_fragment_buffer);

  // Step 4: Collect buffer fragments
  LayoutMap results;
  for (const auto &[buffer, access] : indice_map_) {
    if (!T.layout_map.count(buffer)) {
      auto dst_layout =
          CompleteBufferFragment(buffer)->BindThreadRange(T.thread_bounds);
      results.Set(buffer, dst_layout);
    }
  }
  loop_layout_inferred_ = true;
  return results;
}

Optional<PrimExpr> ParallelOpNode::GetPredicate(Var thread_var) const {
  if (predicate_.defined()) {
    return Substitute(predicate_.value(), {{InputPlaceholder(0), thread_var}});
  } else {
    return std::nullopt;
  }
}

Fragment ParallelOpNode::CompleteBufferFragment(const Buffer &buffer) const {
  ICHECK(loop_layout_.defined());
  if (IsCommonAccessIndice(buffer)) {
    return loop_layout_;
  }
  // Prefer a simple path: if original 2D indices form a bijective map, invert
  // them directly and avoid introducing a synthetic replicate dimension.
  {
    auto res2d =
        arith::DetectIterMap(GetAccessInfo(buffer).indices, ToVMap(loop_vars_),
                             1, arith::IterMapLevel::Bijective,
                             const_cast<arith::Analyzer *>(&analyzer_));
    if (res2d->errors.empty()) {
      Layout ind_inv2d =
          Layout(loop_vars_, GetAccessInfo(buffer).indices)->Inverse();
      PrimExpr indice_rep_extent = 1;
      PrimExpr loop_rep_extent = loop_layout_->ReplicateExtent();
      PrimExpr dest_buffer_rep_extent = indice_rep_extent * loop_rep_extent;
      Array<PrimExpr> fwd2;
      for (size_t i = 0; i < buffer->shape.size(); i++) {
        fwd2.push_back(InputPlaceholder(i));
      }
      PrimExpr thd_b2 =
          loop_layout_->ForwardThread(ind_inv2d->Forward(fwd2), std::nullopt);
      return Fragment(buffer->shape, {}, thd_b2, dest_buffer_rep_extent,
                      std::nullopt)
          ->CondenseReplicateVar();
    }
  }
  // Otherwise, infer an extra flattened iterator that captures truly-unused
  // pieces of the loop space (if any), then try inversion with it.
  PrimExpr rep_b = MakeFlattenedExpression(DivideUnusedIterators(
      GetAccessInfo(buffer).indices, loop_vars_, &analyzer_));
  auto bijective_indice = GetAccessInfo(buffer).indices;
  bijective_indice.push_back(rep_b);
  Layout ind_inv = Layout(loop_vars_, bijective_indice)->Inverse();

  PrimExpr indice_rep_extent =
      ind_inv->InputShape().back(); // this is the size of rep_b
  PrimExpr loop_rep_extent = loop_layout_->ReplicateExtent();
  PrimExpr dest_buffer_rep_extent = indice_rep_extent * loop_rep_extent;
  Array<PrimExpr> fwd;
  for (size_t i = 0; i < buffer->shape.size(); i++) {
    fwd.push_back(InputPlaceholder(i));
  }
  fwd.push_back(FloorMod(ReplicationPlaceholder(), indice_rep_extent));
  PrimExpr thd_b = loop_layout_->ForwardThread(
      ind_inv->Forward(fwd),
      FloorDiv(ReplicationPlaceholder(), indice_rep_extent));
  return Fragment(buffer->shape, {}, thd_b, dest_buffer_rep_extent,
                  std::nullopt)
      ->CondenseReplicateVar();
}

TVM_FFI_STATIC_INIT_BLOCK() { ParallelOpNode::RegisterReflection(); }

bool ParallelOpNode::ValidateCandidateAgainstFragments(
    const Fragment &candidate, const LayoutInferArgs &T, bool throw_on_error,
    bool check_forward_index, const Buffer &source_buffer) const {
  auto vars =
      loop_vars_.Map([](const IterVar &iv) { return PrimExpr(iv->var); });
  for (const auto &[buffer, access] : indice_map_) {
    if (!T.layout_map.count(buffer))
      continue;
    if (auto info = reducer_info_map_.Get(buffer->data);
        info && info.value()->rep == ReducerRepType::ALL)
      continue;
    auto fragment = T.layout_map[buffer].as<Fragment>().value();
    std::ostringstream oss;
    bool success = true;
    if (access.is_read &&
        !ProveFragmentContains(candidate, fragment, vars, access.indices,
                               analyzer_, check_forward_index)) {
      if (throw_on_error) {
        oss << "Layout infer conflict between " << buffer << " and "
            << source_buffer << " in T.Parallel loop:" << '\n'
            << "    loop " << candidate->DebugOutput() << '\n'
            << "    fragment " << fragment->DebugOutput() << '\n';
      }
      success = false;
    }
    if (access.is_write &&
        !ProveFragmentContains(fragment, candidate, access.indices, vars,
                               analyzer_, check_forward_index)) {
      if (throw_on_error) {
        oss << "Layout infer conflict between " << buffer << " and "
            << source_buffer << " in T.Parallel loop:" << '\n'
            << "    loop " << candidate->DebugOutput() << '\n'
            << "    fragment " << fragment->DebugOutput() << '\n';
      }
      success = false;
    }
    if (!success) {
      if (throw_on_error) {
        throw LayoutConflictException(oss.str());
      }
      return false;
    }
  }
  return true;
}

Fragment
ParallelOpNode::ComputeLoopLayoutFromBuffer(const Buffer &buffer,
                                            const LayoutInferArgs &T) const {
  Fragment src_layout = T.layout_map[buffer].as<Fragment>().value();
  DLOG(INFO) << "[compute_loop_layout_from_buffer] infer from buffer `"
             << buffer << "` of layout " << src_layout->DebugOutput() << '\n';

  Fragment result;

  if (IsCommonAccessIndice(buffer)) {
    result = src_layout;
  } else {
    Var rep("_rep");
    auto rep_iter =
        IterVar({0, src_layout->ReplicateExtent()}, rep, IterVarType::kDataPar);
    PrimExpr loop_var_to_thread =
        src_layout->ForwardThread(GetAccessInfo(buffer).indices, rep);
    loop_var_to_thread = analyzer_.Simplify(loop_var_to_thread);
    PostOrderVisit(loop_var_to_thread, [&](const ObjectRef &objref) {
      if (auto opt_var = objref.as<Var>();
          opt_var && inner_vars_.count(*opt_var)) {
        std::ostringstream oss;
        oss << "loop_var_to_thread = " << loop_var_to_thread
            << "contains inner var" << *opt_var;
        throw LayoutConflictException(oss.str());
      }
    });

    try {
      result = Fragment(loop_vars_, {}, loop_var_to_thread, rep_iter)
                   ->BindThreadRange(T.thread_bounds);
    } catch (const tvm::runtime::Error &err) {
      std::ostringstream msg;
      msg << "Layout inference for buffer `" << buffer->name
          << "` failed inside `T.parallel` loop.";

      msg << "\nUnderlying TVM error: " << err.what();
      msg << "\nProblematic loop AST:\n " << root_;
      msg << "\nHint: ensure the loop extent divides the thread binding or "
             "adjust the fragment mapping.";
      LOG(FATAL) << msg.str();
    }
  }
  DLOG(INFO) << "[compute_loop_layout_from_buffer] ... and get "
             << result->DebugOutput() << '\n';
  // Lei: This is a tradeoff, disable it for now.
  // // Try DeReplicate first to reduce replication if possible.
  // Fragment dereplicated_layout = candidate_from_buffer->DeReplicate();
  // if (ValidateCandidateAgainstFragments(
  //         dereplicated_layout, T, /*throw_on_error=*/false,
  //         /*check_forward_index=*/false,
  //         /*source_buffer=*/read_source_buffer)) {
  //   candidate_from_buffer = dereplicated_layout;
  // }
  return result;
}

Fragment ParallelOpNode::ComputePlanCandidate(const LayoutInferArgs &T) const {
  // Vectorize Size must be aware of the buffer_remap
  // As the pass will do post processing to the layout
  auto maybe_remapped_root_ =
      IfBufferRemapLoopGenerator::run(root_, T.buffer_remap, T.layout_map);
  int vector_size =
      GetVectorizeSize(maybe_remapped_root_, T.analyzer, T.layout_map);
  DLOG(INFO) << "[PlanLoopPartition] vector_size = " << vector_size << '\n';

  PrimExpr loop_total_size = 1;
  for (Stmt l = root_; l.as<For>().has_value(); l = l.as<For>().value()->body)
    loop_total_size = loop_total_size * l.as<For>().value()->extent;
  DLOG(INFO) << "[PlanLoopPartition] loop_total_size = " << loop_total_size
             << '\n';
  while (!analyzer_.CanProve(floormod(loop_total_size, T.thread_bounds->extent *
                                                           vector_size) == 0) &&
         vector_size > 1)
    vector_size /= 2;
  DLOG(INFO) << "[PlanLoopPartition] after adjust: vector_size = "
             << vector_size << '\n';

  // Check if coalesced_width is defined
  if (auto coalesced_width = root_->annotations.Get(attr::kCoalescedWidth)) {
    if (const auto *imm = coalesced_width->as<IntImmNode>()) {
      int expected = imm->value;
      // Verify that vector_size is divisible by expected
      if (vector_size % expected != 0) {
        LOG(FATAL) << "Vector size " << vector_size
                   << " is not divisible by coalesced width " << expected;
      }
      vector_size = expected;
    } else {
      LOG(FATAL) << "coalesced_width should be an IntImmNode.";
    }
  }
  DLOG(INFO) << "[PlanLoopPartition] root_ = " << root_
             << " ############# vector_size = " << vector_size
             << ", thread_bounds = " << T.thread_bounds << '\n';
  auto plan = PlanLoopPartition(root_, vector_size, T.thread_bounds);
  DLOG(INFO) << "[PlanLoopPartition] candidate = " << plan->DebugOutput()
             << '\n';
  return plan;
}

void ParallelOpNode::BuildReplicationGuardsIfNeeded(
    const LayoutInferArgs &T,
    const std::vector<Buffer> &store_shared_global_buffers,
    const std::vector<Buffer> &store_fragment_buffers,
    bool has_cross_thread_access,
    const std::vector<Buffer> &const_index_fragment_buffer) const {
  if (is_one(loop_layout_->ReplicateExtent()))
    return;
  if (!has_cross_thread_access)
    return;

  if (!store_fragment_buffers.empty()) {
    bool replicate_is_from_dynamic_index_fragment = false;
    for (const auto &fragment : store_fragment_buffers) {
      if (!T.layout_map.count(fragment)) {
        continue;
      }

      auto fragment_layout = T.layout_map[fragment].as<Fragment>().value();
      if (is_one(fragment_layout->ReplicateExtent()))
        continue;

      if (analyzer_.CanProveEqual(fragment_layout->ReplicateExtent(),
                                  loop_layout_->ReplicateExtent()))
        continue;
      if (std::find(const_index_fragment_buffer.begin(),
                    const_index_fragment_buffer.end(),
                    fragment) == const_index_fragment_buffer.end()) {
        replicate_is_from_dynamic_index_fragment = true;
      }
    }

    if (!replicate_is_from_dynamic_index_fragment)
      return;

    ICHECK(store_shared_global_buffers.empty())
        << "Invalid layout: cannot have both fragment and shared store buffers "
           "in replicated loop layout.";
    return;
  } else {
    auto inv = loop_layout_->Inverse();
    Array<PrimExpr> fwd;
    for (size_t i = 0; i < loop_layout_->OutputDim(); i++)
      fwd.push_back(0);
    fwd.push_back(InputPlaceholder(0) - T.thread_bounds->min);
    auto rep = inv->Forward(fwd).back();
    AddPredicate(EQ(rep, 0));
  }
}
Fragment
ParallelOpNode::ChooseBestCandidate(const Fragment &candidate_from_buffer,
                                    const Fragment &candidate_from_plan,
                                    const LayoutInferArgs &T) const {
  // Strategy overview:
  // 1) Validate each candidate against all known source fragments. If only one
  //    is compatible, choose it immediately.
  // 2) If both are compatible, compare their containment relation:
  //      - If buffer-based contains plan-based, prefer plan (usually smaller
  //      rep).
  //      - If plan-based contains buffer-based, prefer buffer.
  // 3) If neither contains the other, prefer the one with provably smaller or
  //    equal replication extent; otherwise fall back to buffer-based candidate.
  // Note: Final global validation happens after selection elsewhere.
  auto vars =
      loop_vars_.Map([](const IterVar &iv) { return PrimExpr(iv->var); });
  auto contains = [&](const Fragment &big, const Fragment &small) {
    // contains(A, B) means: for any loop index, the threads that access
    // B's elements are a subset of those that access A's elements.
    return ProveFragmentContains(small, big, vars, vars, analyzer_);
  };

  bool buf_ok = ValidateCandidateAgainstFragments(candidate_from_buffer, T);
  bool plan_ok = ValidateCandidateAgainstFragments(candidate_from_plan, T);

  if (buf_ok && !plan_ok) {
    DLOG(INFO)
        << "[FreeInfer] prefer compute_from_buffer (only valid candidate).";
    return candidate_from_buffer;
  }
  if (plan_ok && !buf_ok) {
    DLOG(INFO)
        << "[FreeInfer] prefer PlanLoopPartition (only valid candidate).";
    return candidate_from_plan;
  }
  if (!(buf_ok && plan_ok)) {
    // Both invalid here; let the caller continue to final validation/throw.
    // Returning buffer-based candidate keeps behavior deterministic.
    return candidate_from_buffer; // arbitrary; caller will catch later
  }

  bool buf_contains_plan = contains(candidate_from_buffer, candidate_from_plan);
  bool plan_contains_buf = contains(candidate_from_plan, candidate_from_buffer);

  auto rep_buf = candidate_from_buffer->ReplicateExtent();
  auto rep_plan = candidate_from_plan->ReplicateExtent();

  // Prefer the contained candidate (tends to minimize replication while
  // respecting access coverage):
  if (buf_contains_plan && !plan_contains_buf) {
    return candidate_from_plan;
  }
  if (plan_contains_buf && !buf_contains_plan) {
    return candidate_from_buffer;
  }
  // Neither strictly contains the other; prefer the one with smaller/equal rep.
  if (analyzer_.CanProve(rep_plan <= rep_buf)) {
    return candidate_from_plan;
  }
  // Safe fallback: buffer-based candidate is always correct.
  return candidate_from_buffer;
}

} // namespace tl
} // namespace tvm
