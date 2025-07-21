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

void ParallelLoopNestVisitor::VisitStmt_(const ForNode *op) {
  ICHECK(op->kind == ForKind::kParallel);
  p->loop_vars_.push_back(
      IterVar(Range(op->min, op->extent), op->loop_var, IterVarType::kDataPar));
  p->analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
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

ParallelOp::ParallelOp(For root) : root_(root), V(this) { V.VisitStmt(root); }

bool ParallelOp::IsCommonAccessIndice(const Buffer &buffer) const {
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
LayoutMap ParallelOp::InferLayout(const LayoutInferArgs &T, InferLevel level) {
  if (loop_layout_.defined())
    return {};
  if (level == InferLevel::kStrict)
    return {};

  // Step 1: try to infer loop's partition from a source fragment
  Buffer source_buffer, read_source_buffer;
  for (const auto &[buffer, indices] : indice_map_) {
    if (T.layout_map.count(buffer)) {
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
    if (IsCommonAccessIndice(buffer)) {
      return src_layout;
    } else {
      Var rep;
      auto rep_iter = IterVar({0, src_layout->ReplicateExtent()}, rep,
                              IterVarType::kDataPar);
      PrimExpr loop_var_to_thread =
          src_layout->ForwardThread(indice_map_[buffer], rep);
      return Fragment(loop_vars_, {}, loop_var_to_thread, rep_iter)
          ->BindThreadRange(T.thread_bounds);
    }
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

      // Check if coalesced_width is defined
      if (auto coalesced_width =
              root_->annotations.Get(tl::attr::coalesced_width)) {
        if (const auto *imm = coalesced_width.as<IntImmNode>()) {
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
      loop_layout_ = PlanLoopPartition(root_, vector_size, T.thread_bounds);
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
  // fragment
  for (const auto &[buffer, _] : indice_map_) {
    if (T.layout_map.count(buffer)) {
      auto fragment = T.layout_map[buffer].as<Fragment>().value();
      // TODO: Add thread checks for replicated cases
      // need to wildcard match the rhs with lhs
      if (!is_one(loop_layout_->ReplicateExtent()) ||
          !is_one(fragment->ReplicateExtent()))
        continue;
      auto vars =
          loop_vars_.Map([](const IterVar &iv) { return PrimExpr(iv->var); });
      auto lhs = loop_layout_->ForwardThread(vars, NullOpt);
      auto rhs = fragment->ForwardThread(indice_map_[buffer], NullOpt);
      auto diff = analyzer_.Simplify(lhs - rhs);
      ICHECK(is_zero(diff))
          << "Layout infer conflict for " << buffer << " " << source_buffer
          << "\nLHS = " << lhs << "\nRHS = " << rhs;
    }
  }
  // Step 3: Infer other fragment's layout from the loop's partition
  LayoutMap results;
  for (const auto &[buffer, _] : indice_map_) {
    if (!T.layout_map.count(buffer)) {
      results.Set(buffer, CompleteBufferFragment(buffer)->BindThreadRange(
                              T.thread_bounds));
    }

    // Layout infer conflict for local.fragment can not be handled here
    // because the source_buffer is not always available
    // (zhengju) do not modify strict layout even if it is conflict with the
    // dst layout. This will not influence the result because the strict
    // layout is usually with rep = 1 Since the real layout map is
    // controlled by layout_inference.cc, we should add this check there
    if (buffer.scope() == "local.fragment" && source_buffer.defined() &&
        source_buffer.scope() == "local.fragment") {
      if (T.layout_map.count(buffer)) {
        const FragmentNode *src_layout =
            T.layout_map[buffer].as<Fragment>().get();
        Fragment dst_layout_fragment =
            CompleteBufferFragment(buffer)->BindThreadRange(T.thread_bounds);
        const FragmentNode *dst_layout =
            dst_layout_fragment.as<Fragment>().get();
        if (as_const_int(dst_layout->ReplicateExtent()) &&
            as_const_int(src_layout->ReplicateExtent()) &&
            (*as_const_int(dst_layout->ReplicateExtent()) >
             *as_const_int(src_layout->ReplicateExtent()))) {
          results.Set(buffer, dst_layout_fragment);
          continue;
        }
        if (src_layout && dst_layout) {
          ICHECK(src_layout->IsEqual(dst_layout, true))
              << "Layout may conflict with ParallelOp for buffer " << buffer
              << " vs. " << source_buffer << "\nError body begin:\n"
              << GetRoot()->body << "\nError body end"
              << "\nLHS = " << src_layout->DebugOutput()
              << "\nRHS = " << dst_layout->DebugOutput()
              << "\nYou may need to use a shared memory to transform the "
                 "layout";
        }
      }
    }
  }
  return results;
}

Optional<PrimExpr> ParallelOp::GetPredicate(Var thread_var) const {
  if (predicate_.defined()) {
    return Substitute(predicate_.value(), {{InputPlaceholder(0), thread_var}});
  } else {
    return NullOpt;
  }
}

Fragment ParallelOp::CompleteBufferFragment(const Buffer &buffer) {
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
  return Fragment(buffer->shape, {}, thd_b, dest_buffer_rep_extent, NullOpt)
      ->CondenseReplicateVar();
}

} // namespace tl
} // namespace tvm
