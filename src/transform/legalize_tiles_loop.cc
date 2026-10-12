#include <algorithm>
#include <unordered_map>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../layout/layout.h"
#include "../support/ffi_aliases.h"
#include "../tileview/tileview.h"
#include "../tileview/tileview_planner.h"
#include "common/attr.h"
#include "common/constr_visitor.h"

namespace tvm {
namespace tl {

using namespace tir;

/* ============================================================
 * TileView Collector
 *
 * Collect block-level:
 *   block.annotations["tileview_map"]
 *     : Map<Var, TileView>
 * ============================================================ */
class TileViewCollector : public StmtExprVisitor {
public:
  using TileViewMap =
      std::unordered_map<Var, TileView, ObjectPtrHash, ObjectPtrEqual>;

  static TileViewMap Collect(const PrimFunc &f) {
    TileViewCollector collector;
    collector(f->body);
    return std::move(collector.tileviews_);
  }

private:
  void VisitStmt_(const BlockNode *block) final {
    auto it = block->annotations.find(attr::kTileViewMap);
    if (it != block->annotations.end()) {
      auto tv_map = Downcast<Map<Var, TileView>>((*it).second);
      for (const auto &kv : tv_map) {
        auto res = tileviews_.emplace(kv.first, kv.second);
        ICHECK(res.second) << "Duplicate TileView for buffer " << kv.first;
      }
    }
    StmtExprVisitor::VisitStmt_(block);
  }

private:
  TileViewMap tileviews_;
};

class NestedTilesScopeDetector : public StmtExprVisitor {
public:
  static bool Exists(const Stmt &stmt) {
    NestedTilesScopeDetector detector;
    detector(stmt);
    return detector.found_;
  }

private:
  void VisitStmt_(const ForNode *loop) final {
    if (loop->annotations.count(attr::kTileDomain)) {
      found_ = true;
      return;
    }
    StmtExprVisitor::VisitStmt_(loop);
  }

  bool found_{false};
};

class TilesParallelWriteChecker : public ConstrVisitor {
public:
  static void Check(const For &scope_root) {
    TilesParallelWriteChecker checker;
    checker(scope_root);
  }

  static bool IsParallelTileLoop(const ForNode *op) {
    auto it = op->annotations.find(attr::tile_level_loop);
    if (it == op->annotations.end()) {
      return false;
    }
    if (const auto *imm = (*it).second.as<IntImmNode>()) {
      return imm->value != 0;
    }
    return true;
  }

private:
  bool CanProveExprEqual(const PrimExpr &lhs, const PrimExpr &rhs,
                         arith::Analyzer *analyzer) const {
    PrimExpr lhs_simpl = analyzer->Simplify(lhs);
    PrimExpr rhs_simpl = analyzer->Simplify(rhs);

    if (StructuralEqual()(lhs_simpl, rhs_simpl)) {
      return true;
    }
    if (analyzer->CanProve(lhs_simpl == rhs_simpl)) {
      return true;
    }

    const Object *lhs_node = lhs_simpl.get();
    const Object *rhs_node = rhs_simpl.get();
    if (lhs_node == nullptr || rhs_node == nullptr ||
        lhs_node->type_index() != rhs_node->type_index()) {
      return false;
    }

    if (const auto *lhs_load = lhs_simpl.as<BufferLoadNode>()) {
      const auto *rhs_load = rhs_simpl.as<BufferLoadNode>();
      ICHECK(rhs_load);
      if (!lhs_load->buffer.same_as(rhs_load->buffer) ||
          lhs_load->indices.size() != rhs_load->indices.size()) {
        return false;
      }
      for (size_t i = 0; i < lhs_load->indices.size(); ++i) {
        if (!analyzer->CanProve(lhs_load->indices[i] == rhs_load->indices[i])) {
          return false;
        }
      }
      return true;
    }

    if (const auto *lhs_add = lhs_simpl.as<AddNode>()) {
      const auto *rhs_add = rhs_simpl.as<AddNode>();
      ICHECK(rhs_add);
      return CanProveExprEqual(lhs_add->a, rhs_add->a, analyzer) &&
             CanProveExprEqual(lhs_add->b, rhs_add->b, analyzer);
    }
    if (const auto *lhs_sub = lhs_simpl.as<SubNode>()) {
      const auto *rhs_sub = rhs_simpl.as<SubNode>();
      ICHECK(rhs_sub);
      return CanProveExprEqual(lhs_sub->a, rhs_sub->a, analyzer) &&
             CanProveExprEqual(lhs_sub->b, rhs_sub->b, analyzer);
    }
    if (const auto *lhs_mul = lhs_simpl.as<MulNode>()) {
      const auto *rhs_mul = rhs_simpl.as<MulNode>();
      ICHECK(rhs_mul);
      return CanProveExprEqual(lhs_mul->a, rhs_mul->a, analyzer) &&
             CanProveExprEqual(lhs_mul->b, rhs_mul->b, analyzer);
    }
    if (const auto *lhs_div = lhs_simpl.as<DivNode>()) {
      const auto *rhs_div = rhs_simpl.as<DivNode>();
      ICHECK(rhs_div);
      return CanProveExprEqual(lhs_div->a, rhs_div->a, analyzer) &&
             CanProveExprEqual(lhs_div->b, rhs_div->b, analyzer);
    }
    if (const auto *lhs_mod = lhs_simpl.as<ModNode>()) {
      const auto *rhs_mod = rhs_simpl.as<ModNode>();
      ICHECK(rhs_mod);
      return CanProveExprEqual(lhs_mod->a, rhs_mod->a, analyzer) &&
             CanProveExprEqual(lhs_mod->b, rhs_mod->b, analyzer);
    }
    if (const auto *lhs_floordiv = lhs_simpl.as<FloorDivNode>()) {
      const auto *rhs_floordiv = rhs_simpl.as<FloorDivNode>();
      ICHECK(rhs_floordiv);
      return CanProveExprEqual(lhs_floordiv->a, rhs_floordiv->a, analyzer) &&
             CanProveExprEqual(lhs_floordiv->b, rhs_floordiv->b, analyzer);
    }
    if (const auto *lhs_floormod = lhs_simpl.as<FloorModNode>()) {
      const auto *rhs_floormod = rhs_simpl.as<FloorModNode>();
      ICHECK(rhs_floormod);
      return CanProveExprEqual(lhs_floormod->a, rhs_floormod->a, analyzer) &&
             CanProveExprEqual(lhs_floormod->b, rhs_floormod->b, analyzer);
    }
    if (const auto *lhs_min = lhs_simpl.as<MinNode>()) {
      const auto *rhs_min = rhs_simpl.as<MinNode>();
      ICHECK(rhs_min);
      return CanProveExprEqual(lhs_min->a, rhs_min->a, analyzer) &&
             CanProveExprEqual(lhs_min->b, rhs_min->b, analyzer);
    }
    if (const auto *lhs_max = lhs_simpl.as<MaxNode>()) {
      const auto *rhs_max = rhs_simpl.as<MaxNode>();
      ICHECK(rhs_max);
      return CanProveExprEqual(lhs_max->a, rhs_max->a, analyzer) &&
             CanProveExprEqual(lhs_max->b, rhs_max->b, analyzer);
    }
    if (const auto *lhs_cast = lhs_simpl.as<CastNode>()) {
      const auto *rhs_cast = rhs_simpl.as<CastNode>();
      ICHECK(rhs_cast);
      return lhs_cast->dtype == rhs_cast->dtype &&
             CanProveExprEqual(lhs_cast->value, rhs_cast->value, analyzer);
    }
    if (const auto *lhs_not = lhs_simpl.as<NotNode>()) {
      const auto *rhs_not = rhs_simpl.as<NotNode>();
      ICHECK(rhs_not);
      return CanProveExprEqual(lhs_not->a, rhs_not->a, analyzer);
    }
    if (const auto *lhs_and = lhs_simpl.as<AndNode>()) {
      const auto *rhs_and = rhs_simpl.as<AndNode>();
      ICHECK(rhs_and);
      return CanProveExprEqual(lhs_and->a, rhs_and->a, analyzer) &&
             CanProveExprEqual(lhs_and->b, rhs_and->b, analyzer);
    }
    if (const auto *lhs_or = lhs_simpl.as<OrNode>()) {
      const auto *rhs_or = rhs_simpl.as<OrNode>();
      ICHECK(rhs_or);
      return CanProveExprEqual(lhs_or->a, rhs_or->a, analyzer) &&
             CanProveExprEqual(lhs_or->b, rhs_or->b, analyzer);
    }
    if (const auto *lhs_eq = lhs_simpl.as<EQNode>()) {
      const auto *rhs_eq = rhs_simpl.as<EQNode>();
      ICHECK(rhs_eq);
      return CanProveExprEqual(lhs_eq->a, rhs_eq->a, analyzer) &&
             CanProveExprEqual(lhs_eq->b, rhs_eq->b, analyzer);
    }
    if (const auto *lhs_ne = lhs_simpl.as<NENode>()) {
      const auto *rhs_ne = rhs_simpl.as<NENode>();
      ICHECK(rhs_ne);
      return CanProveExprEqual(lhs_ne->a, rhs_ne->a, analyzer) &&
             CanProveExprEqual(lhs_ne->b, rhs_ne->b, analyzer);
    }
    if (const auto *lhs_lt = lhs_simpl.as<LTNode>()) {
      const auto *rhs_lt = rhs_simpl.as<LTNode>();
      ICHECK(rhs_lt);
      return CanProveExprEqual(lhs_lt->a, rhs_lt->a, analyzer) &&
             CanProveExprEqual(lhs_lt->b, rhs_lt->b, analyzer);
    }
    if (const auto *lhs_le = lhs_simpl.as<LENode>()) {
      const auto *rhs_le = rhs_simpl.as<LENode>();
      ICHECK(rhs_le);
      return CanProveExprEqual(lhs_le->a, rhs_le->a, analyzer) &&
             CanProveExprEqual(lhs_le->b, rhs_le->b, analyzer);
    }
    if (const auto *lhs_gt = lhs_simpl.as<GTNode>()) {
      const auto *rhs_gt = rhs_simpl.as<GTNode>();
      ICHECK(rhs_gt);
      return CanProveExprEqual(lhs_gt->a, rhs_gt->a, analyzer) &&
             CanProveExprEqual(lhs_gt->b, rhs_gt->b, analyzer);
    }
    if (const auto *lhs_ge = lhs_simpl.as<GENode>()) {
      const auto *rhs_ge = rhs_simpl.as<GENode>();
      ICHECK(rhs_ge);
      return CanProveExprEqual(lhs_ge->a, rhs_ge->a, analyzer) &&
             CanProveExprEqual(lhs_ge->b, rhs_ge->b, analyzer);
    }
    if (const auto *lhs_select = lhs_simpl.as<SelectNode>()) {
      const auto *rhs_select = rhs_simpl.as<SelectNode>();
      ICHECK(rhs_select);
      return CanProveExprEqual(lhs_select->condition, rhs_select->condition,
                               analyzer) &&
             CanProveExprEqual(lhs_select->true_value, rhs_select->true_value,
                               analyzer) &&
             CanProveExprEqual(lhs_select->false_value, rhs_select->false_value,
                               analyzer);
    }

    return false;
  }

  void VisitStmt_(const ForNode *op) final {
    bool is_tile_loop = op->annotations.count(attr::tile_level_loop);
    bool is_parallel_tile_loop = IsParallelTileLoop(op);

    if (is_tile_loop && is_parallel_tile_loop) {
      parallel_tile_loop_vars_.push_back(op->loop_var);
      ConstrVisitor::VisitStmt_(op);
      parallel_tile_loop_vars_.pop_back();
      return;
    }

    ConstrVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (parallel_tile_loop_vars_.empty() || op->buffer.scope() == "local.var" ||
        op->buffer.scope() == "local") {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }

    ConstrSet cset{constr_stack_};
    ffi::Map<Var, PrimExpr> subs = MakeIterationSubstitution();
    cset.Extend(cset.Substitute(subs));

    PrimExpr same_destination = Bool(true);
    for (const auto &idx : op->indices) {
      same_destination =
          tir::And(same_destination, idx == tir::Substitute(idx, subs));
    }

    PrimExpr distinct_iteration = Bool(false);
    for (const auto &var : parallel_tile_loop_vars_) {
      distinct_iteration = tir::Or(distinct_iteration,
                                   var != Downcast<Var>(subs.Get(var).value()));
    }

    PrimExpr first_predicate =
        op->predicate.defined() ? op->predicate.value() : Bool(true);
    PrimExpr second_predicate = tir::Substitute(first_predicate, subs);

    PrimExpr conflict_guard =
        tir::And(tir::And(same_destination, distinct_iteration),
                 tir::And(first_predicate, second_predicate));
    PrimExpr other_value = tir::Substitute(op->value, subs);

    arith::Analyzer guard_analyzer;
    cset.Populate(guard_analyzer);

    arith::Analyzer equal_analyzer;
    cset.Populate(equal_analyzer);
    equal_analyzer.EnterConstraint(conflict_guard);

    if (!guard_analyzer.z3_prover.CanProve(tir::Not(conflict_guard)) &&
        !CanProveExprEqual(op->value, other_value, &equal_analyzer)) {
      LOG(FATAL)
          << "Implicit reduction in T.Tiles is not supported. Distinct "
             "parallel tile iterations can write different values to the same "
             "destination element in BufferStore to '"
          << op->buffer->name
          << "'. Please use explicit T.atomic_add or dedicated reduction "
             "loops.";
    }

    StmtExprVisitor::VisitStmt_(op);
  }

  ffi::Map<Var, PrimExpr> MakeIterationSubstitution() const {
    ffi::Map<Var, PrimExpr> subs;
    for (const auto &var : parallel_tile_loop_vars_) {
      subs.Set(var, Var(var->name_hint + "<OTHER>", var.dtype()));
    }
    for (const auto &constr : constr_stack_) {
      if (!constr.var.defined()) {
        continue;
      }
      if (subs.count(constr.var)) {
        continue;
      }
      subs.Set(constr.var,
               Var(constr.var->name_hint + "<OTHER>", constr.var.dtype()));
    }
    return subs;
  }

  std::vector<Var> parallel_tile_loop_vars_;
};

/* ============================================================
 * LegalizeTilesLoopRewriter
 *
 * Rewrite tile-level For loops using a TileView plan solved by
 * the shared tileview planner.
 * ============================================================ */
class LegalizeTilesLoopRewriter : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc f) {
    LegalizeTilesLoopRewriter rewriter;
    rewriter.tileviews_ = TileViewCollector::Collect(f);
    rewriter.tile_processor_config_ =
        GetSunmmioTileProcessorConfig(f->GetAttr<Target>("target"));

    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  static Stmt StripTileLoopAnnotations(const ForNode *root) {
    class Stripper : public StmtExprMutator {
      Stmt VisitStmt_(const ForNode *loop) final {
        For rewritten = Downcast<For>(StmtExprMutator::VisitStmt_(loop));
        auto *n = rewritten.CopyOnWrite();
        n->annotations.erase(attr::tile_level_loop);
        n->annotations.erase(attr::kTileDomain);
        n->annotations.erase(attr::kTileLoopStage);
        n->annotations.erase(attr::tile_tile_size);
        n->annotations.erase(attr::tile_execution_domain_axes);
        n->annotations.erase(attr::tile_execution_axis);
        return rewritten;
      }
    } stripper;
    return stripper(ffi::GetRef<For>(root));
  }

  static void AddBufferIfMissing(std::vector<Buffer> *buffers,
                                 const Buffer &buffer) {
    auto it = std::find_if(
        buffers->begin(), buffers->end(),
        [&](const Buffer &candidate) { return candidate.same_as(buffer); });
    if (it == buffers->end()) {
      buffers->push_back(buffer);
    }
  }

  static std::vector<const ForNode *>
  CollectTileLoopChain(const ForNode *root) {
    std::vector<const ForNode *> loops;
    const ForNode *current = root;
    while (current != nullptr &&
           current->annotations.count(attr::tile_level_loop)) {
      loops.push_back(current);
      const auto *next = current->body.as<ForNode>();
      if (next == nullptr || !next->annotations.count(attr::tile_level_loop)) {
        break;
      }
      current = next;
    }
    return loops;
  }

  /* ---- Collect buffer and layout info from Block nodes ---- */
  Stmt VisitStmt_(const BlockNode *block) final {
    // Collect layout_map from block annotation
    if (block->annotations.count(attr::kLayoutMap)) {
      auto layout_map_obj = block->annotations.Get(attr::kLayoutMap).value();
      if (auto layout_map = layout_map_obj.as<Map<Buffer, Layout>>()) {
        for (const auto &[buffer, layout] : layout_map.value()) {
          layout_map_.Set(buffer, layout);
        }
      } else if (auto layout_map = layout_map_obj.as<Map<Var, Layout>>()) {
        std::vector<Buffer> block_buffers;
        for (const Buffer &buffer : block->alloc_buffers) {
          AddBufferIfMissing(&block_buffers, buffer);
        }
        for (const BufferRegion &region : block->reads) {
          AddBufferIfMissing(&block_buffers, region->buffer);
        }
        for (const BufferRegion &region : block->writes) {
          AddBufferIfMissing(&block_buffers, region->buffer);
        }
        for (const MatchBufferRegion &match_buffer : block->match_buffers) {
          AddBufferIfMissing(&block_buffers, match_buffer->buffer);
        }

        for (const auto &[buffer_var, layout] : layout_map.value()) {
          bool found = false;
          for (const Buffer &buffer : block_buffers) {
            if (buffer->data.same_as(buffer_var)) {
              layout_map_.Set(buffer, layout);
              found = true;
            }
          }
          ICHECK(found)
              << "layout_map annotation references unknown buffer var "
              << buffer_var << ".";
        }
      } else {
        LOG(FATAL)
            << "Unsupported layout_map annotation type in LegalizeTilesLoop.";
      }
    }

    return StmtExprMutator::VisitStmt_(block);
  }

  Stmt VisitStmt_(const ForNode *loop) final {
    // Only rewrite tile-level loops
    if (!loop->annotations.count(attr::tile_level_loop)) {
      return StmtExprMutator::VisitStmt_(loop);
    }

    // ---- Stage Check (Idempotency) ----
    int stage = static_cast<int>(TileLoopStage::kInitial);

    auto stage_it = loop->annotations.find(attr::kTileLoopStage);
    if (stage_it != loop->annotations.end()) {
      stage = Downcast<Integer>((*stage_it).second)->value;
    }

    if (stage >= static_cast<int>(TileLoopStage::kLegalized)) {
      // Already legalized -> skip
      LOG(INFO) << "[Legalize tiles loop] tile loop stage is: " << stage
                << ", so skip. ";
      return StmtExprMutator::VisitStmt_(loop);
    }

    bool starts_scope = loop->annotations.count(attr::kTileDomain);
    if (starts_scope) {
      ICHECK(!in_active_scope_) << "Nested T.Tiles scopes are not supported.";

      Array<PrimExpr> domain =
          Downcast<Array<PrimExpr>>(loop->annotations.at(attr::kTileDomain));
      ICHECK(!NestedTilesScopeDetector::Exists(loop->body))
          << "Nested T.Tiles scopes are not supported.";
      if (TilesParallelWriteChecker::IsParallelTileLoop(loop)) {
        TilesParallelWriteChecker::Check(ffi::GetRef<For>(loop));
      }
      auto scope_loops = CollectTileLoopChain(loop);
      ICHECK_GE(scope_loops.size(), domain.size())
          << "T.Tiles scope loop rank does not cover the declared domain rank.";
      scope_loops.resize(domain.size());
      auto accesses = CollectBufferAccesses(loop->body);
      auto plan = TryPlanTileViewsForTilesScope(domain, scope_loops, accesses,
                                                tileviews_, layout_map_,
                                                tile_processor_config_);
      if (!plan.has_value()) {
        LOG(WARNING) << "T.Tiles domain " << domain
                     << " cannot infer a legal TileView and is falling back "
                        "to scalar serial loops using tile.pick/tile.set. This "
                        "may severely degrade performance.";
        return StripTileLoopAnnotations(loop);
      }
      active_tileview_plan_ = plan.value();
      if (active_tileview_plan_.requires_aligned_1d_bridge) {
        LOG(WARNING)
            << "T.Tiles domain " << domain
            << " uses a logical 1D tile smaller than the native RSRAM "
               "alignment. Codegen will load an aligned carrier and pick, "
               "extract, or insert the logical tile. This may degrade "
               "performance.";
      }
      if (active_tileview_plan_.requires_small_2d_carrier) {
        LOG(WARNING)
            << "T.Tiles domain " << domain
            << " uses a logical small 2D tile carried by one register-sized "
               "RSRAM tile. Codegen will extract and insert the logical "
               "slice through the carrier.";
      }
      if (active_tileview_plan_.execution_domain_axes.size() < domain.size()) {
        LOG(WARNING) << "T.Tiles domain " << domain
                     << " cannot infer a full-rank TileView and is falling "
                        "back to serial outer axes plus 1D tile execution. "
                        "This may degrade performance.";
      }
      active_scope_depth_ = 0;
      in_active_scope_ = true;
    }

    ICHECK(in_active_scope_)
        << "Tile loop encountered without an active T.Tiles domain root.";

    // Enter tile loop (depth == tile dimension)
    int dim = active_scope_depth_++;
    Stmt new_body = VisitStmt(loop->body);
    active_scope_depth_--;

    const TileViewPlan &plan = active_tileview_plan_;
    const TileView &tv = plan.execution_tileview;
    Array<PrimExpr> tiled_shape = tv->TiledBufferShape();

    ICHECK(dim < static_cast<int>(tiled_shape.size()))
        << "Tile loop depth exceeds tiled buffer rank";

    // Rewrite loop
    For new_for = ffi::GetRef<For>(loop);
    auto *n = new_for.CopyOnWrite();
    n->extent = tiled_shape[dim];
    n->body = new_body;

    // Attach normalized loop annotations
    n->annotations.Set(attr::tile_tile_size, tv->TileShape());
    n->annotations.Set(attr::kTileLoopStage,
                       Integer(static_cast<int>(TileLoopStage::kLegalized)));
    // ---- Determine whether this logical domain loop carries an execution axis
    // If plan.execution_domain_axes[k] == dim, then:
    //   - this loop carries execution axis k
    //   - tile.tile_size[k] applies to this logical domain axis
    // Example: execution_domain_axes=[1,0] means tile_size[0] belongs to the
    // second logical loop axis, and tile_size[1] belongs to the first.
    auto axis_it = std::find(plan.execution_domain_axes.begin(),
                             plan.execution_domain_axes.end(), dim);
    bool is_tile_execution = axis_it != plan.execution_domain_axes.end();

    if (is_tile_execution) {
      n->annotations.Set(attr::tile_execution_axis,
                         Integer(static_cast<int>(std::distance(
                             plan.execution_domain_axes.begin(), axis_it))));
    }

    if (starts_scope) {
      // Scope-level execution axis -> logical domain axis mapping.
      Array<PrimExpr> execution_domain_axes;
      for (int axis : plan.execution_domain_axes) {
        execution_domain_axes.push_back(Integer(axis));
      }
      n->annotations.Set(attr::tile_execution_domain_axes,
                         execution_domain_axes);
    }

    if (starts_scope) {
      in_active_scope_ = false;
      active_tileview_plan_ = TileViewPlan();
      active_scope_depth_ = 0;
    }
    return new_for;
  }

private:
  TileViewMap tileviews_;
  Map<Buffer, Layout> layout_map_;
  SunmmioTileProcessorConfig tile_processor_config_{
      GetSunmmioTileProcessorConfig(ffi::Optional<Target>())};
  bool in_active_scope_{false};
  TileViewPlan active_tileview_plan_;
  int active_scope_depth_{0};
};

/* ============================================================
 * Pass Registration
 * ============================================================ */
using namespace tir::transform;

tvm::transform::Pass LegalizeTilesLoop() {
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    return LegalizeTilesLoopRewriter::Rewrite(std::move(f));
  };

  return CreatePrimFuncPass(pass_func,
                            /*opt_level=*/0, "tl.LegalizeTilesLoop", {});
}

/* ============================================================
 * FFI Registration
 * ============================================================ */
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("tl.transform.LegalizeTilesLoop",
                                        LegalizeTilesLoop);
}

} // namespace tl
} // namespace tvm
