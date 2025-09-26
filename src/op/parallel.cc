/*!
 * \file op/parallel.cc
 * \brief Define Parallel for operator
 */

#include "parallel.h"

#include <tvm/tir/op.h>

#include "../layout/utils.h"
#include "../target/utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"

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
  if (op->buffer.scope() == "local.fragment") {
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
  if (op->buffer.scope() == "local.fragment") {
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
  auto op = make_object<ParallelOpNode>(*this);
  return ParallelOp(op);
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
  if (level == InferLevel::kStrict)
    return {};

  // Step 1: try to infer loop's partition from a source fragment
  Buffer source_buffer, read_source_buffer;
  for (const auto &[buffer, indices] : indice_map_) {
    if (T.layout_map.count(buffer)) {
      // skip reducers with rep=ALL
      if (auto info = reducer_info_map_.Get(buffer->data);
          info && info.value()->rep == ReducerRepType::ALL)
        continue;

      auto frag = T.layout_map[buffer].as<Fragment>().value();
      if (buffer_is_write_.count(buffer)) {
        source_buffer = buffer;
      } else {
        // Keep the buffer with largest number of indices
        // (which means the inference based on that buffer is more accurate)
        // as read_source_buffer to get more accurate layout
        if (!read_source_buffer.defined() ||
            indice_map_[buffer].size() >
                indice_map_[read_source_buffer].size()) {
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
      result = Fragment(loop_vars_, {}, loop_var_to_thread, rep_iter)
                   ->BindThreadRange(T.thread_bounds);
    }
    DLOG(INFO) << "[compute_loop_layout_from_buffer] ... and get "
               << result->DebugOutput() << '\n';
    return result;
  };
  if (source_buffer.defined()) {
    loop_layout_ = compute_loop_layout_from_buffer(source_buffer);
  } else if (level == InferLevel::kFree) {
    if (read_source_buffer.defined()) {
      loop_layout_ = compute_loop_layout_from_buffer(read_source_buffer);
      // // Loop don't need to be replicated.
      // if (!is_one(loop_layout_->ReplicateExtent()))
      //   loop_layout_ = loop_layout_->DeReplicate();

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
      bool has_pure_buffer_store = false;
      PostOrderVisit(root_, [&](const ObjectRef &obj) {
        if (const auto *store = obj.as<BufferStoreNode>()) {
          // Check if the value is a direct load from another buffer (i.e., b[i]
          // = a[i])
          if (const auto *load = store->value.as<BufferLoadNode>()) {
            has_pure_buffer_store = true;
          }
        }
      });

      if (!is_one(loop_layout_->ReplicateExtent()) && has_cross_thread_access &&
          !has_pure_buffer_store) {
        auto inv = loop_layout_->Inverse();
        Array<PrimExpr> fwd;
        for (size_t i = 0; i < loop_layout_->OutputDim(); i++)
          fwd.push_back(0);
        fwd.push_back(InputPlaceholder(0) - T.thread_bounds->min);
        auto rep = inv->Forward(fwd).back();
        AddPredicate(EQ(rep, 0));
      }
    } else {
      // Vectorize Size must be aware of the buffer_remap
      // As the pass will do post processing to the layout
      auto maybe_remapped_root_ =
          IfBufferRemapLoopGenerator::run(root_, T.buffer_remap, T.layout_map);
      int vector_size = GetVectorizeSize(maybe_remapped_root_);

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
  } else {
    return {};
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

TVM_FFI_STATIC_INIT_BLOCK({ ParallelOpNode::RegisterReflection(); });

} // namespace tl
} // namespace tvm
