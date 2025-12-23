/*!
 * \file op/parallel.cc
 * \brief Define Parallel for operator
 */

#include "parallel.h"

#include <algorithm>
#include <tvm/tir/op.h>

#include "../layout/utils.h"
#include "../target/utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace attr {
/*! \brief Mark that how the loop is vectorized. */
constexpr const char *coalesced_width = "coalesced_width";
} // namespace attr

// ProveFragmentContains checks whether the threads that access elements of a
// smaller fragment (small_frag) are a subset of the threads that access
// elements of a larger fragment (large_frag) for any given loop index. This
// function ensures that if the small fragment's layout corresponds to the loop
// itself, accessing the large fragment's elements is valid. Additionally, if
// small is updated to large, the originally valid access remains valid. The
// proof is performed by:
//
// 1. Defining a variable `rep_small` to represent the replicate index of the
//    small fragment that is being checked.
// 2. Using the `small_frag_indices` and `rep_small` to derive the thread
// accessing
//    the element in the small fragment.
// 3. Using `large_frag_indices` to derive the physical index of the large
// fragment
//    along with the thread information, and then feeding these into the inverse
//    of the large fragment to obtain the logical index and replicate index.
// 4. Verifying the mapping by checking whether the computed thread using the
// inverse
//    layout corresponds to the original thread calculated for the small
//    fragment. If they don't match, this indicates that the inverse layout's
//    domain does not include the thread and thus the access is invalid.
bool ProveFragmentContains(Fragment small_frag, Fragment large_frag,
                           Array<PrimExpr> small_frag_indices,
                           Array<PrimExpr> large_frag_indices,
                           arith::Analyzer &analyzer_) {
  Var rep_small("__checking_frag_contains_rep");
  analyzer_.Bind(rep_small,
                 Range(IntImm(small_frag->ReplicateExtent()->dtype, 0),
                       small_frag->ReplicateExtent()),
                 true); // Bind the replicate extent of small_frag.
  // Derive thread for small_frag.
  auto thread = small_frag->ForwardThread(small_frag_indices, rep_small);

  // Get physical index and thread for large_frag.
  auto large_frag_physical_and_thread = large_frag->Forward(large_frag_indices);
  // Add small_frag's thread to the large fragment's thread info.
  large_frag_physical_and_thread.push_back(thread);
  // Get the inverse of the large fragment.
  auto inv_large_frag = large_frag->Inverse();
  // Compute logical index and replicate index using inverse layout.
  auto inv_large_frag_logical_and_rep =
      inv_large_frag->Forward(large_frag_physical_and_thread);

  // Extract replicate index from the result.
  auto inv_large_frag_rep =
      inv_large_frag_logical_and_rep[inv_large_frag_logical_and_rep.size() - 1];

  // Calculate thread based on the logical index and replicate index.
  auto check_thread =
      large_frag->ForwardThread(large_frag_indices, inv_large_frag_rep);

  // Simplify the difference between the threads.
  auto diff = analyzer_.Simplify(thread - check_thread);
  // If the difference is zero, the threads match and the access is valid.
  return is_zero(diff);
}

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
    if (p->indice_map_.find(op->buffer) != p->indice_map_.end()) {
      ICHECK(StructuralEqual()(p->indice_map_.at(op->buffer), op->indices))
          << op->buffer << ": " << op->indices << " and "
          << p->indice_map_.at(op->buffer);
    } else {
      p->indice_map_.Set(op->buffer, op->indices);
    }
    p->buffer_is_write_.insert(op->buffer);
  }
  StmtExprVisitor::VisitStmt_(op);
}

void ParallelLoopNestVisitor::VisitExpr_(const BufferLoadNode *op) {
  if (IsFragmentBuffer(op->buffer)) {
    if (p->indice_map_.find(op->buffer) != p->indice_map_.end()) {
      ICHECK(StructuralEqual()(p->indice_map_.at(op->buffer), op->indices))
          << op->buffer << ": " << op->indices << " and "
          << p->indice_map_.at(op->buffer);
    } else {
      p->indice_map_.Set(op->buffer, op->indices);
    }
  }
  StmtExprVisitor::VisitExpr_(op);
}

ParallelOpNode::ParallelOpNode(For root) : root_(root), V(this) {
  V.VisitStmt(root);
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
        if (IsFragmentBuffer(bl->buffer) && !indice_map_.count(bl->buffer)) {
          LOG(INFO) << "ExpandLetBindings: set buffer " << bl->buffer
                    << " with indices " << bl->indices;
          indice_map_.Set(bl->buffer, bl->indices);
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

Stmt ParallelOpNode::Lower(const LowerArgs &T,
                           arith::Analyzer *analyzer) const {
  return root_;
}

bool ParallelOpNode::IsCommonAccessIndice(const Buffer &buffer) const {
  auto common_indice = loop_vars_.Map([](const auto &iv) { return iv->var; });
  return StructuralEqual()(indice_map_[buffer], common_indice);
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
  if (loop_layout_.defined())
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
    for (const auto &[buffer, indices] : indice_map_) {
      if (T.layout_map.count(buffer)) {
        continue;
      }
      if (!IsFragmentBuffer(buffer))
        continue;

      // Check if all indices are zero
      bool all_indices_zero = true;
      for (const auto &index : indices) {
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
  auto buffer_is_completed_replicated = [&](const Buffer &buffer) {
    if (!IsFragmentBuffer(buffer))
      return false;
    auto frag = T.layout_map[buffer].as<Fragment>().value();
    // buffer indices should be IntImm
    for (const auto &index : indice_map_[buffer]) {
      if (!index.as<IntImmNode>()) {
        return false;
      } else if (index.as<IntImmNode>()->value != 0) {
        LOG(FATAL) << "buffer " << buffer << " is not completed replicated";
      }
    }
    return frag->IsCompletedReplicated();
  };
  // Collect fragment buffers with const index and all fragment_buffers
  std::vector<Buffer> const_index_fragment_buffer, fragment_buffers;
  for (const auto &[buffer, indices] : indice_map_) {
    if (!IsFragmentBuffer(buffer))
      continue;
    fragment_buffers.push_back(buffer);

    bool is_const_index = true;
    for (const auto &index : indices) {
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

  for (const auto &[buffer, indices] : indice_map_) {
    if (T.layout_map.count(buffer)) {
      // skip reducers with rep=ALL
      if (auto info = reducer_info_map_.Get(buffer->data);
          info && info.value()->rep == ReducerRepType::ALL)
        continue;

      auto frag = T.layout_map[buffer].as<Fragment>().value();
      bool is_fully_replicated = buffer_is_completed_replicated(buffer);

      if (buffer_is_write_.count(buffer)) {
        source_buffer = buffer;
      } else {
        // Keep the buffer with largest number of indices
        // (which means the inference based on that buffer is more accurate)
        // as read_source_buffer to get more accurate layout
        // if the buffer is completed replicated, we don't need to infer the
        // layout from this buffer.
        if ((!read_source_buffer.defined() ||
             indice_map_[buffer].size() >
                 indice_map_[read_source_buffer].size())) {
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
  auto compute_loop_layout_from_buffer = [&](const Buffer &buffer) {
    Fragment src_layout = T.layout_map[buffer].as<Fragment>().value();
    DLOG(INFO) << "[compute_loop_layout_from_buffer] infer from buffer `"
               << buffer << "` of layout " << src_layout->DebugOutput() << '\n';

    Fragment result;
    if (IsCommonAccessIndice(buffer)) {
      result = src_layout;
    } else {
      Var rep;
      auto rep_iter = IterVar({0, src_layout->ReplicateExtent()}, rep,
                              IterVarType::kDataPar);
      PrimExpr loop_var_to_thread =
          src_layout->ForwardThread(indice_map_[buffer], rep);
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
    return result;
  };

  // Try to infer loop layout from buffers in order of preference:
  // 1. Non-replicated write buffer (most reliable)
  // 2. Non-replicated read buffer
  // 3. Fully replicated write buffer (backup, may cause issues)
  // 4. Free inference mode (no source buffer)

  if (source_buffer.defined() && allow_layout_propgate) {
    loop_layout_ = compute_loop_layout_from_buffer(source_buffer);
  } else if (level == InferLevel::kFree) {
    // For free layout inference
    // If replication exists and buffer has cross-thread shared memory access,
    // add predicate
    bool has_cross_thread_access = false;
    PostOrderVisit(root_, [&](const ObjectRef &obj) {
      if (const auto *store = obj.as<BufferStoreNode>()) {
        // check if scope is shared or global
        if (store->buffer.scope() == "shared" ||
            store->buffer.scope() == "shared.dyn" ||
            store->buffer.scope() == "global") {
          has_cross_thread_access = true;
        }
      } else if (const auto *load = obj.as<BufferLoadNode>()) {
        // check if scope is shared or global
        if (load->buffer.scope() == "shared" ||
            load->buffer.scope() == "shared.dyn" ||
            load->buffer.scope() == "global") {
          has_cross_thread_access = true;
        }
      }
    });

    // check if loop body contains a "pure" buffer store (i.e., direct
    // assignment, not compound update)
    std::vector<Buffer> store_shared_global_buffers, store_fragment_buffers;
    // Buffers that scope is above fragments.
    // global, shared, shared.dyn
    // which can be used to analysis replicate case
    PostOrderVisit(root_, [&](const ObjectRef &obj) {
      if (const auto *store = obj.as<BufferStoreNode>()) {
        auto buffer = store->buffer;
        if (buffer.scope() == "shared" || buffer.scope() == "shared.dyn" ||
            buffer.scope() == "global") {
          store_shared_global_buffers.emplace_back(buffer);
        } else if (IsFragmentBuffer(buffer)) {
          store_fragment_buffers.emplace_back(buffer);
        }
      }
    });
    if (read_source_buffer.defined() && allow_layout_propgate) {
      loop_layout_ = compute_loop_layout_from_buffer(read_source_buffer);
    }

    if (!loop_layout_.defined()) {
      // No source buffer available, use free mode inference
      // Vectorize Size must be aware of the buffer_remap
      // As the pass will do post processing to the layout
      auto maybe_remapped_root_ =
          IfBufferRemapLoopGenerator::run(root_, T.buffer_remap, T.layout_map);
      int vector_size = GetVectorizeSize(maybe_remapped_root_, T.analyzer);
      DLOG(INFO) << "[PlanLoopPartition] vector_size = " << vector_size << '\n';

      PrimExpr loop_total_size = 1;
      for (Stmt l = root_; l.as<For>().has_value();
           l = l.as<For>().value()->body)
        loop_total_size = loop_total_size * l.as<For>().value()->extent;
      DLOG(INFO) << "[PlanLoopPartition] loop_total_size = " << loop_total_size
                 << '\n';
      while (!analyzer_.CanProve(
                 floormod(loop_total_size,
                          T.thread_bounds->extent * vector_size) == 0) &&
             vector_size > 1)
        vector_size /= 2;
      DLOG(INFO) << "[PlanLoopPartition] after adjust: vector_size = "
                 << vector_size << '\n';

      // Check if coalesced_width is defined
      if (auto coalesced_width =
              root_->annotations.Get(tl::attr::coalesced_width)) {
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
      loop_layout_ = PlanLoopPartition(root_, vector_size, T.thread_bounds);
      DLOG(INFO) << "[PlanLoopPartition] loop_layout_ = "
                 << loop_layout_->DebugOutput() << '\n';
    }

    // Lambda that guards replicated accesses:
    // - When a loop layout replicates a fragment buffer (rep > 1), each thread
    //   observes the same fragment elements. Blindly storing to shared/global
    //   memory in that case would add the same value multiple times.
    // - We therefore restrict the store so that only the replica with rep == 0
    //   performs the update (e.g. global[i] += fragment[i] only fires once).
    // Trigger conditions for this guard:
    // 1) There are cross-thread stores targeting shared/global memory (no
    //    fragment stores in this branch; atomic_add and similar remain TODO).
    // 2) The loop layout replicate extent is greater than 1, inferred from the
    //    thread bounds captured in the layout.

    [this, &store_shared_global_buffers, &store_fragment_buffers,
     &has_cross_thread_access, &const_index_fragment_buffer, &T]() {
      if (is_one(loop_layout_->ReplicateExtent()))
        return;
      if (!has_cross_thread_access)
        return;

      if (!store_fragment_buffers.empty()) {
        // Iterate replicated fragment stores: when the fragment index is a
        // constant (e.g. fragment[0]), every thread touches the same slot, so
        // the rep == 0 predicate is unnecessary. Example: for i in
        // T.Parallel(...):
        //   shared[i] = ...
        //   fragment[0] = ...
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
            << "Invalid layout: cannot have both fragment and shared store "
               "buffers "
               "in replicated loop layout.";
        return;
      } else {
        // Now, store is global or shared
        // or T.call_extern or T.call_intrin ...
        auto inv = loop_layout_->Inverse();
        Array<PrimExpr> fwd;
        for (size_t i = 0; i < loop_layout_->OutputDim(); i++)
          fwd.push_back(0);
        fwd.push_back(InputPlaceholder(0) - T.thread_bounds->min);
        auto rep = inv->Forward(fwd).back();
        AddPredicate(EQ(rep, 0));
      }
    }();
  } else {
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
  // fragment, and infer layout only when it's not yet layout-ed
  LayoutMap results;
  for (const auto &[buffer, _] : indice_map_) {
    if (T.layout_map.count(buffer)) {
      auto fragment = T.layout_map[buffer].as<Fragment>().value();
      auto vars =
          loop_vars_.Map([](const IterVar &iv) { return PrimExpr(iv->var); });
      if (!ProveFragmentContains(loop_layout_, fragment, vars,
                                 indice_map_[buffer], analyzer_)) {
        std::ostringstream oss;
        oss << "Layout infer conflict between " << buffer << " and "
            << source_buffer << " in T.Parallel loop:" << '\n'
            << "    loop " << loop_layout_->DebugOutput() << '\n'
            << "    fragment " << fragment->DebugOutput() << '\n';
        throw LayoutConflictException(oss.str());
      }
    } else {
      auto dst_layout =
          CompleteBufferFragment(buffer)->BindThreadRange(T.thread_bounds);
      results.Set(buffer, dst_layout);
    }
  }
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
        arith::DetectIterMap(indice_map_[buffer], ToVMap(loop_vars_), 1,
                             arith::IterMapLevel::Bijective,
                             const_cast<arith::Analyzer *>(&analyzer_));
    if (res2d->errors.empty()) {
      Layout ind_inv2d = Layout(loop_vars_, indice_map_[buffer])->Inverse();
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
  PrimExpr rep_b = MakeFlattenedExpression(
      DivideUnusedIterators(indice_map_[buffer], loop_vars_, &analyzer_));
  auto bijective_indice = indice_map_[buffer];
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

} // namespace tl
} // namespace tvm
