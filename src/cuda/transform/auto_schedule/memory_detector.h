#pragma once
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "op/builtin.h"
#include "op/operator.h"
#include "op/utils.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Array;
using ffi::GetRef;

// Collect direct buffer regions and scalar-variable accesses from a statement.
// Buffer indices are relaxed over surrounding loop domains, and repeated
// accesses to the same logical buffer are unioned dimension by dimension.
// Adapted from BlockReadWriteDetector in TVM.
class MemoryAccessDetector : public StmtExprVisitor {
public:
  MemoryAccessDetector() = default;

  // Analyze a statement and collect read/write regions
  void Analyze(const Stmt &stmt) {
    read_buffers_.clear();
    write_buffers_.clear();
    read_regions_.clear();
    write_regions_.clear();
    dom_map_.clear();
    hint_map_.clear();
    pending_conditions_.clear();
    let_bindings_.clear();
    read_vars_.clear();
    write_vars_.clear();
    operator()(stmt);
  }

  void Analyze(const PrimExpr &expr) {
    read_buffers_.clear();
    write_buffers_.clear();
    read_regions_.clear();
    write_regions_.clear();
    dom_map_.clear();
    hint_map_.clear();
    pending_conditions_.clear();
    let_bindings_.clear();
    read_vars_.clear();
    write_vars_.clear();
    operator()(expr);
  }

  // Return collected read regions
  std::vector<BufferRegion> GetReadRegions() const {
    return CollectRegions(read_buffers_, read_regions_);
  }

  // Return collected write regions
  std::vector<BufferRegion> GetWriteRegions() const {
    return CollectRegions(write_buffers_, write_regions_);
  }

  // Return all variables that are read from
  std::vector<Var> GetReadVars() const { return read_vars_; }
  // Return all variables that are written to
  std::vector<Var> GetWriteVars() const { return write_vars_; }

private:
  /*! \brief Iteration range for loop_vars */
  std::unordered_map<const VarNode *, arith::IntSet> dom_map_;
  /*! \brief Extra iteration range hint for free vars */
  std::unordered_map<const VarNode *, arith::IntSet> hint_map_;
  /*! \brief Unresolved conditions within current scope. */
  std::vector<PrimExpr> pending_conditions_;
  /*! \brief The buffers that the current block reads */
  std::vector<Buffer> read_buffers_;
  /*! \brief The buffers that the current block writes */
  std::vector<Buffer> write_buffers_;
  /*! \brief The read regions of the current block */
  std::vector<std::vector<tvm::arith::IntSet>> read_regions_;
  /*! \brief The write regions of the current block */
  std::vector<std::vector<tvm::arith::IntSet>> write_regions_;
  /*!\ brief Internal analyzer. */
  arith::Analyzer ana_;
  /*! \brief let bindings inside the block */
  std::unordered_map<const VarNode *, PrimExpr> let_bindings_;

  /*! \brief The set of variables that are read in the current block.  */
  std::vector<Var> read_vars_;
  /*! \brief The set of variables that are written in the current block.  */
  std::vector<Var> write_vars_;

  /*!
   * \brief Update read/write buffers and regions with provided buffer and
   * region
   */
  void Update(std::vector<Buffer> *buffers,
              std::vector<std::vector<arith::IntSet>> *regions, Buffer buffer,
              std::vector<arith::IntSet> region) {
    UpdateReadVar(buffer->data);
    // Check if buffer already exists
    for (size_t i = 0; i < buffers->size(); ++i) {
      if ((*buffers)[i].same_as(buffer)) {
        // Merge regions
        ICHECK_EQ((*regions)[i].size(), region.size());
        for (size_t j = 0; j < region.size(); ++j) {
          (*regions)[i][j] = arith::Union({(*regions)[i][j], region[j]});
        }
        return;
      }
    }
    // New buffer
    buffers->push_back(buffer);
    regions->push_back(region);
  }

  /*!
   * \brief Update the set of read variables with the given variable
   * \param var The variable to add to the set of read variables
   */
  void UpdateReadVar(const Var &var) {
    for (const auto &v : read_vars_) {
      if (v.same_as(var))
        return;
    }
    read_vars_.push_back(var);
  }

  /*!
   * \brief Update the set of write variables with the given variable
   * \param var The variable to add to the set of write variables
   */
  void UpdateWriteVar(const Var &var) {
    for (const auto &v : write_vars_) {
      if (v.same_as(var))
        return;
    }
    write_vars_.push_back(var);
  }

  /*!
   * \brief Record one normalized access region: relax every range over the
   * loop domain, charge the index computations' variable reads, and file
   * the region under the mask's sides.
   */
  void ProcessAccessRegion(const AccessRegion &access) {
    const Buffer &buffer = access.region->buffer;
    std::vector<arith::IntSet> int_sets;
    int_sets.reserve(access.region->region.size());
    for (const auto &range : access.region->region) {
      int_sets.push_back(RelaxAccessIndex(range->min, range->extent));
      VisitExpr(range->min);
      VisitExpr(range->extent);
    }
    if (access.access_mask & kAccessRead)
      Update(&read_buffers_, &read_regions_, buffer, int_sets);
    if (access.access_mask & kAccessWrite)
      Update(&write_buffers_, &write_regions_, buffer, int_sets);
  }

  /*!
   * \brief Process a buffer-like argument (BufferRegion, BufferLoad, or
   * tl.region bridge) of a tile op.
   * \param is_read The access side when the argument carries no mask itself
   */
  void ProcessBufferRegion(const PrimExpr &arg, bool is_read) {
    ProcessAccessRegion(
        NormalizeToAccessRegion(arg, is_read ? kAccessRead : kAccessWrite));
  }

  /*! \brief Helper function to collect access regions. */
  std::vector<BufferRegion> CollectRegions(
      const std::vector<Buffer> &buffers,
      const std::vector<std::vector<tvm::arith::IntSet>> &regions) const {
    std::vector<BufferRegion> result;
    result.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
      const Buffer &buffer = buffers[i];
      const std::vector<arith::IntSet> &int_sets = regions[i];
      Region region;
      size_t ndim = buffer->shape.size();
      size_t region_ndim = int_sets.size();

      // Assert that region dimension equals buffer dimension
      ICHECK_EQ(region_ndim, ndim) << "Region dimension " << region_ndim
                                   << " must equal buffer dimension " << ndim;

      region.reserve(ndim);
      for (size_t j = 0; j < ndim; ++j) {
        const tvm::arith::IntSet &int_set = int_sets[j];
        region.push_back(
            int_set.CoverRange(Range::FromMinExtent(0, buffer->shape[j])));
      }

      result.push_back(BufferRegion(buffer, region));
    }
    return result;
  }

  /*! \brief Resolve Let bindings in an expression to a fixpoint. */
  PrimExpr ResolveLets(const PrimExpr &e) {
    PrimExpr current = e;
    PrimExpr remapped = Substitute(current, let_bindings_);
    while (!remapped.same_as(current)) {
      current = remapped;
      remapped = Substitute(current, let_bindings_);
    }
    return current;
  }

  /*! \brief Relax the half-open range [base, base + extent) over the loop
   * domain, resolving Let bindings first. The default extent of 1 gives the
   * single-point relaxation of one index. */
  arith::IntSet RelaxAccessIndex(const PrimExpr &index,
                                 PrimExpr extent = IntImm(DataType::Int(32),
                                                          1)) {
    PrimExpr base = ResolveLets(index);
    arith::IntSet lo_set =
        arith::EvalSet(arith::IntSet::Vector(base), dom_map_);
    const auto *ext_imm = extent.as<IntImmNode>();
    if (ext_imm && ext_imm->value <= 1) {
      return lo_set;
    }
    PrimExpr hi = base + ResolveLets(extent) - make_const(base.dtype(), 1);
    arith::IntSet hi_set = arith::EvalSet(arith::IntSet::Vector(hi), dom_map_);
    return arith::IntSet::Interval(lo_set.min(), hi_set.max());
  }

  void VisitStmt_(const ForNode *op) override {
    Range range = Range::FromMinExtent(op->min, op->extent);
    dom_map_[op->loop_var.get()] = arith::IntSet::FromRange(range);
    StmtExprVisitor::VisitStmt_(op);
    dom_map_.erase(op->loop_var.get());
  }

  void VisitStmt_(const IfThenElseNode *op) override {
    VisitExpr(op->condition);
    {
      // Visit then branch
      // Simplified: we don't handle conditional bounds for now
      StmtExprVisitor::VisitStmt(op->then_case);
    }
    if (op->else_case) {
      // Visit else branch
      StmtExprVisitor::VisitStmt(op->else_case.value());
    }
  }

  void VisitStmt_(const BindNode *op) override {
    // The value contributes this task's direct reads; the bound variable is a
    // scalar write. The scheduler follows these direct def-use links on demand
    // when it computes the transitive Let dependency closure.
    VisitExpr(op->value);
    let_bindings_[op->var.get()] = op->value;
    UpdateWriteVar(op->var);
  }

  void VisitExpr_(const VarNode *op) override {
    UpdateReadVar(tvm::ffi::GetRef<Var>(op));
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) override {
    std::vector<arith::IntSet> relaxed_region;
    size_t num_indices = op->indices.size();
    size_t buffer_ndim = op->buffer->shape.size();

    // Assert that indices count equals buffer dimension
    ICHECK_EQ(num_indices, buffer_ndim)
        << "BufferLoad indices count " << num_indices
        << " must equal buffer dimension " << buffer_ndim;

    for (PrimExpr index : op->indices) {
      relaxed_region.push_back(RelaxAccessIndex(index));
    }
    Update(&read_buffers_, &read_regions_, op->buffer, relaxed_region);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) override {
    std::vector<arith::IntSet> relaxed_region;
    size_t num_indices = op->indices.size();
    size_t buffer_ndim = op->buffer->shape.size();

    // Assert that indices count equals buffer dimension
    ICHECK_EQ(num_indices, buffer_ndim)
        << "BufferStore indices count " << num_indices
        << " must equal buffer dimension " << buffer_ndim;

    for (PrimExpr index : op->indices) {
      relaxed_region.push_back(RelaxAccessIndex(index));
    }
    Update(&write_buffers_, &write_regions_, op->buffer, relaxed_region);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) override {
    static const auto reduce_op = Op::Get("tl.tileop.reduce");
    static const auto tl_access_ptr_op = Op::Get("tl.access_ptr");

    // A tl.region(...) bridge: decode through the shared normalizer.
    if (op->op.same_as(tl::region())) {
      ProcessAccessRegion(NormalizeToAccessRegion(GetRef<PrimExpr>(op)));
      return;
    }

    // Check for tl.tileop.reduce call
    if (op->op.same_as(reduce_op)) {
      // Handle tl.tileop.reduce call for memory access analysis
      // args[0] = input buffer region (read)
      // args[1] = output buffer region (write)
      // args[2] = reduce_type (string)
      // args[3] = dim (int)
      // args[4] = clear (bool)
      if (op->args.size() >= 2) {
        // Process first argument as read region
        ProcessBufferRegion(op->args[0], true); // is_read = true
        // Process second argument as write region
        ProcessBufferRegion(op->args[1], false); // is_read = false
      }
      return;
    }

    // Handle other calls (e.g., builtin::tvm_access_ptr)
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      // Simplified: skip for now
      StmtExprVisitor::VisitExpr_(op);
      return;
    }

    // Handle TileLang `tl.access_ptr(BufferLoad(buf, mins), extent, rw_mask)`.
    // The access is a contiguous run of `extent` elements in row-major layout
    // starting at the base index `mins`. Reconstruct the per-dimension region
    // by distributing the flat extent across dimensions from innermost outward,
    // rather than conservatively taking the whole buffer for every dimension.
    if (op->op.same_as(tl_access_ptr_op)) {
      if (op->args.size() >= 3) {
        const auto *buffer_load = op->args[0].as<BufferLoadNode>();
        const auto *mask_int = op->args[2].as<IntImmNode>();
        if (buffer_load && mask_int) {
          Buffer buffer = buffer_load->buffer;
          int rw_mask = mask_int->value;
          size_t ndim = buffer->shape.size();
          std::vector<arith::IntSet> region(ndim);

          const auto *ext_imm = op->args[1].as<IntImmNode>();
          if (ext_imm && buffer_load->indices.size() == ndim) {
            // Distribute a known flat extent (in elements) across dims,
            // innermost-first. `rem` is the number of elements still to be
            // covered at the current dimension; each dim consumes as many of
            // its `shape[i]` indices as the run spans, carrying the rounded-up
            // remainder to the next (outer) dim.
            int64_t rem = ext_imm->value;
            for (size_t k = ndim; k-- > 0;) {
              PrimExpr base = buffer_load->indices[k];
              const auto *dim_imm = buffer->shape[k].as<IntImmNode>();
              if (rem <= 1) {
                region[k] = RelaxAccessIndex(base);
              } else if (dim_imm) {
                int64_t d = dim_imm->value;
                if (rem >= d) {
                  // Run spills past this dim: cover it whole and carry up.
                  region[k] = arith::IntSet::FromRange(
                      Range::FromMinExtent(0, buffer->shape[k]));
                  rem = (rem + d - 1) / d;
                } else {
                  region[k] = RelaxAccessIndex(base, IntImm(base.dtype(), rem));
                  rem = 1;
                }
              } else {
                // Non-constant dim size: be conservative for this and any
                // remaining outer dims.
                region[k] = arith::IntSet::FromRange(
                    Range::FromMinExtent(0, buffer->shape[k]));
                rem = 1;
              }
            }
          } else {
            // Non-constant extent or arity mismatch: fall back to whole buffer.
            for (size_t k = 0; k < ndim; ++k) {
              region[k] = arith::IntSet::FromRange(
                  Range::FromMinExtent(0, buffer->shape[k]));
            }
          }

          if (rw_mask & 1) {
            Update(&read_buffers_, &read_regions_, buffer, region);
          }
          if (rw_mask & 2) {
            Update(&write_buffers_, &write_regions_, buffer, region);
          }
          for (const PrimExpr &index : buffer_load->indices) {
            VisitExpr(index);
          }
          VisitExpr(op->args[1]);
          return;
        }
      }
      StmtExprVisitor::VisitExpr_(op);
      return;
    }

    // Query each TileLang operator's exact access regions so GEMM, copy, fill,
    // scan, atomic and future tile ops retain region sensitivity without one
    // special case per operation.
    if (TileOperator tile_op = ParseOperator(GetRef<Call>(op));
        tile_op.defined()) {
      AccessRegions accesses = tile_op->GetAccessRegions();
      auto add_regions = [&](const Array<BufferRegion> &regions, bool is_read) {
        for (const BufferRegion &buffer_region : regions) {
          std::vector<arith::IntSet> int_sets;
          int_sets.reserve(buffer_region->region.size());
          for (const Range &range : buffer_region->region) {
            int_sets.push_back(RelaxAccessIndex(range->min, range->extent));
            VisitExpr(range->min);
            VisitExpr(range->extent);
          }
          if (is_read) {
            Update(&read_buffers_, &read_regions_, buffer_region->buffer,
                   int_sets);
          } else {
            Update(&write_buffers_, &write_regions_, buffer_region->buffer,
                   int_sets);
          }
        }
      };
      add_regions(accesses.reads, true);
      add_regions(accesses.writes, false);
      return;
    }

    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const SBlockNode *op) override {
    // TIRX represents many TileLang regions as direct SBlock nodes.  Match the
    // old TIR behavior for BlockNode by visiting the body so task-level buffer
    // dependencies are preserved.
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const SBlockRealizeNode *op) override {
    // Don't visit child blocks recursively
  }
};

} // namespace tl
} // namespace tvm
