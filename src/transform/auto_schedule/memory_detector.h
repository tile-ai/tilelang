#pragma once

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

// MemoryAccessDetector: detect read/write regions in statements
// Adapted from BlockReadWriteDetector in TVM
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
    operator()(stmt);
  }

  // Return collected read regions
  std::vector<BufferRegion> GetReadRegions() const {
    return CollectRegions(read_buffers_, read_regions_);
  }

  // Return collected write regions
  std::vector<BufferRegion> GetWriteRegions() const {
    return CollectRegions(write_buffers_, write_regions_);
  }

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

  /*!
   * \brief Update read/write buffers and regions with provided buffer and
   * region
   */
  void Update(std::vector<Buffer> *buffers,
              std::vector<std::vector<arith::IntSet>> *regions, Buffer buffer,
              std::vector<arith::IntSet> region) {
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
   * \brief Process a buffer region argument from reduce operation
   * \param arg The argument which could be BufferRegion, BufferLoad, or
   * tl.tileop.region call
   * \param is_read Whether this is a read (true) or write (false) access
   */
  void ProcessBufferRegion(const PrimExpr &arg, bool is_read) {
    // Check if it's a BufferRegion
    if (const auto *buffer_region = arg.as<BufferRegionNode>()) {
      Buffer buffer = buffer_region->buffer;
      const Region &region = buffer_region->region;
      std::vector<arith::IntSet> int_sets;
      int_sets.reserve(region.size());
      for (const auto &range : region) {
        // Create IntSet for range [min, min + extent)
        int_sets.push_back(arith::IntSet::FromRange(
            Range::FromMinExtent(range->min, range->extent)));
      }
      if (is_read) {
        Update(&read_buffers_, &read_regions_, buffer, int_sets);
      } else {
        Update(&write_buffers_, &write_regions_, buffer, int_sets);
      }
      return;
    }

    // Check if it's a BufferLoad
    if (const auto *buffer_load = arg.as<BufferLoadNode>()) {
      Buffer buffer = buffer_load->buffer;
      std::vector<arith::IntSet> int_sets;
      int_sets.reserve(buffer_load->indices.size());
      for (PrimExpr index : buffer_load->indices) {
        // Create IntSet for single point
        int_sets.push_back(RelaxAccessIndex(index));
      }
      if (is_read) {
        Update(&read_buffers_, &read_regions_, buffer, int_sets);
      } else {
        Update(&write_buffers_, &write_regions_, buffer, int_sets);
      }
      return;
    }

    // Check if it's a tl.tileop.region call (should already be handled by
    // VisitExpr_) but we can still process it recursively
    if (const auto *call = arg.as<CallNode>()) {
      static const auto region_op = Op::Get("tl.tileop.region");
      if (call->op.same_as(region_op)) {
        // Recursively visit this call to handle it
        VisitExpr_(call);
        return;
      }
    }

    // If we reach here, the argument type is not supported
    LOG(WARNING) << "Unsupported argument type in tl.tileop.reduce: "
                 << arg->GetTypeKey();
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

  /*! \brief Helper function to relax the buffer indices */
  arith::IntSet RelaxAccessIndex(const PrimExpr &index) {
    PrimExpr current = index;
    PrimExpr remapped = Substitute(current, let_bindings_);
    while (!remapped.same_as(current)) {
      current = remapped;
      remapped = Substitute(current, let_bindings_);
    }
    return arith::EvalSet(arith::IntSet::Vector(current), dom_map_);
  }

  void operator()(const Stmt &stmt) { StmtExprVisitor::operator()(stmt); }

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

  void VisitStmt_(const LetStmtNode *op) override {
    let_bindings_[op->var.get()] = op->value;
    StmtExprVisitor::VisitStmt_(op);
    let_bindings_.erase(op->var.get());
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
    static const auto region_op = Op::Get("tl.tileop.region");
    static const auto reduce_op = Op::Get("tl.tileop.reduce");

    // Check for tl.tileop.region call
    if (op->op.same_as(region_op)) {
      // Handle tl.tileop.region call for memory access analysis
      // args[0] = buffer (BufferLoad), args[1] = access_type (1: read, 2:
      // write, 3: read/write) args[2..] = extents
      if (op->args.size() >= 2) {
        // Extract access type
        const auto *access_int = op->args[1].as<IntImmNode>();
        ICHECK(access_int);
        int access_type = access_int->value;

        // Extract buffer from BufferLoad
        if (const auto *buffer_load = op->args[0].as<BufferLoadNode>()) {
          Buffer buffer = buffer_load->buffer;
          std::vector<arith::IntSet> relaxed_region;

          // Assert that BufferLoad accesses a single element (no Ramp indices)
          for (size_t i = 0; i < buffer_load->indices.size(); ++i) {
            const PrimExpr &index = buffer_load->indices[i];
            // Check if index is a Ramp (vector access)
            if (index.as<RampNode>()) {
              LOG(FATAL) << "BufferLoad in tl.tileop.region should access a "
                            "single element, "
                         << "but found Ramp index at dimension " << i;
            }
          }

          // Use provided extents if available, otherwise use buffer load
          // indices
          size_t num_indices = buffer_load->indices.size();
          size_t buffer_ndim = buffer->shape.size();

          // Assert that indices count equals buffer dimension
          ICHECK_EQ(num_indices, buffer_ndim)
              << "BufferLoad indices count " << num_indices
              << " must equal buffer dimension " << buffer_ndim;

          if (op->args.size() > 2) {
            // args[2..] are extents for the region
            // Number of extents provided
            size_t num_extents = op->args.size() - 2;

            // Assert that extents count equals indices count
            ICHECK_EQ(num_extents, num_indices)
                << "Extents count " << num_extents
                << " must equal indices count " << num_indices;

            relaxed_region.reserve(num_indices);
            for (size_t i = 0; i < num_indices; ++i) {
              PrimExpr min = buffer_load->indices[i];
              PrimExpr extent = op->args[2 + i];

              // Create IntSet for range [min, min + extent)
              relaxed_region.push_back(
                  arith::IntSet::FromRange(Range::FromMinExtent(min, extent)));
            }
          } else {
            // No extents provided: each dimension is a single point at the
            // index
            for (PrimExpr index : buffer_load->indices) {
              // Create IntSet for single point
              relaxed_region.push_back(RelaxAccessIndex(index));
            }
          }

          // Add to appropriate list based on access type
          if (access_type == 1 || access_type == 3) { // read or read/write
            Update(&read_buffers_, &read_regions_, buffer, relaxed_region);
          }
          if (access_type == 2 || access_type == 3) { // write or read/write
            Update(&write_buffers_, &write_regions_, buffer, relaxed_region);
          }
        } else {
          LOG(FATAL)
              << "First argument of tl.tileop.region should be a BufferLoad";
        }
      }
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

    StmtExprVisitor::VisitExpr_(op);
  }

  // Skip block-specific handling for now
  void VisitStmt_(const BlockRealizeNode *op) override {
    // Don't visit child blocks recursively
  }
};

} // namespace tl
} // namespace tvm
