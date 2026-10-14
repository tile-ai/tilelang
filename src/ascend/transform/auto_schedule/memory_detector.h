#pragma once
#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ascend/op/ascend_mte_plan.h"
#include "ascend/op/builtin.h"
#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "support/check.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;

// MemoryAccessDetector: detect read/write regions in statements
// Adapted from BlockReadWriteDetector in TVM
class MemoryAccessDetector : public StmtExprVisitor {
public:
  // Task/synchronization access semantics stay logical. Physical footprints
  // are requested only by buffer-lifetime analysis and never stored in Task.
  enum class AccessMode { kLogical, kPhysical };
  explicit MemoryAccessDetector(AccessMode mode = AccessMode::kLogical)
      : mode_(mode) {}

  // Analyze a statement and collect read/write regions
  void Analyze(const Stmt &stmt) {
    read_buffers_.clear();
    write_buffers_.clear();
    read_regions_.clear();
    write_regions_.clear();
    dom_map_.clear();
    hint_map_.clear();
    pending_conditions_.clear();
    unconditional_write_regions_.clear();
    loop_domains_.clear();
    conditional_depth_ = 0;
    non_unit_step_loop_depth_ = 0;
    has_loop_break_ = false;
    opaque_pointer_context_ = false;
    let_bindings_.clear();
    read_vars_.clear();
    write_vars_.clear();
    opaque_access_vars_.clear();
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
    unconditional_write_regions_.clear();
    loop_domains_.clear();
    conditional_depth_ = 0;
    non_unit_step_loop_depth_ = 0;
    has_loop_break_ = false;
    opaque_pointer_context_ = false;
    let_bindings_.clear();
    read_vars_.clear();
    write_vars_.clear();
    opaque_access_vars_.clear();
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
  // Return storage Vars used through accesses whose precise regions are not
  // represented by GetReadRegions()/GetWriteRegions().
  std::vector<Var> GetOpaqueAccessVars() const { return opaque_access_vars_; }
  // Return each provably dense, unconditional write separately. Unlike
  // GetWriteRegions(), these regions are not merged into a may-write hull.
  std::vector<BufferRegion> GetMustWriteRegions() const {
    // A loop_break may truncate any loop domain represented by this atomic
    // task. Keep its may-write regions for dependencies, but do not use any
    // write in the task as a complete definition for lifetime analysis.
    if (has_loop_break_)
      return {};
    return unconditional_write_regions_;
  }

private:
  /*! \brief Iteration range for loop_vars */
  std::unordered_map<const VarNode *, arith::IntSet> dom_map_;
  /*! \brief Extra iteration range hint for free vars */
  std::unordered_map<const VarNode *, arith::IntSet> hint_map_;
  /*! \brief Unresolved conditions within current scope. */
  std::vector<PrimExpr> pending_conditions_;
  /*! \brief Individual dense writes that execute whenever the task does. */
  std::vector<BufferRegion> unconditional_write_regions_;
  /*! \brief Loop domains active at the current statement. */
  std::vector<std::pair<Var, Range>> loop_domains_;
  /*! \brief Current nesting depth of control flow that may skip its body. */
  size_t conditional_depth_{0};
  /*! \brief Current nesting depth of loops with a non-unit step. */
  size_t non_unit_step_loop_depth_{0};
  /*! \brief Whether this task contains a loop_break intrinsic. */
  bool has_loop_break_{false};
  // Pointer consumers whose physical footprint is not encoded by access_ptr.
  bool opaque_pointer_context_{false};
  /*! \brief The buffers that the current block reads */
  std::vector<Buffer> read_buffers_;
  /*! \brief The buffers that the current block writes */
  std::vector<Buffer> write_buffers_;
  /*! \brief The read regions of the current block */
  std::vector<std::vector<tvm::arith::IntSet>> read_regions_;
  /*! \brief The write regions of the current block */
  std::vector<std::vector<tvm::arith::IntSet>> write_regions_;
  AccessMode mode_;
  /*!\ brief Internal analyzer. */
  arith::Analyzer ana_;
  /*! \brief let bindings inside the block */
  std::unordered_map<const VarNode *, PrimExpr> let_bindings_;

  /*! \brief The set of variables that are read in the current block.  */
  std::vector<Var> read_vars_;
  /*! \brief The set of variables that are written in the current block.  */
  std::vector<Var> write_vars_;
  /*! \brief Buffer storage Vars with an access of unknown region. */
  std::vector<Var> opaque_access_vars_;

  /*!
   * \brief Update read/write buffers and regions with provided buffer and
   * region
   */
  void Update(std::vector<Buffer> *buffers,
              std::vector<std::vector<arith::IntSet>> *regions, Buffer buffer,
              std::vector<arith::IntSet> region) {
    UpdateReadVar(buffer->data);
    // Equal logical indices do not prove equal physical addresses when layout
    // metadata reads mutable state. Lowering uses these expressions at
    // accesses, so retain their reads as well as excluding the viewed storage
    // from reuse.
    auto visit_mutable_layout = [&](const PrimExpr &expr) {
      if (mode_ == AccessMode::kPhysical &&
          SideEffect(expr) > CallEffectKind::kPure) {
        UpdateOpaqueAccessVar(buffer->data);
        VisitExpr(expr);
      }
    };
    for (const PrimExpr &extent : buffer->shape)
      visit_mutable_layout(extent);
    for (const PrimExpr &stride : buffer->strides)
      visit_mutable_layout(stride);
    visit_mutable_layout(buffer->elem_offset);
    MergeRegion(buffers, regions, buffer, region);
  }

  static void MergeRegion(std::vector<Buffer> *buffers,
                          std::vector<std::vector<arith::IntSet>> *regions,
                          const Buffer &buffer,
                          const std::vector<arith::IntSet> &region) {
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

  BufferRegion
  MakeBufferRegion(const Buffer &buffer,
                   const std::vector<arith::IntSet> &int_sets) const {
    Region region;
    size_t ndim = buffer->shape.size();
    ICHECK_EQ(int_sets.size(), ndim) << "Region dimension " << int_sets.size()
                                     << " must equal buffer dimension " << ndim;
    region.reserve(ndim);
    for (size_t i = 0; i < ndim; ++i) {
      region.push_back(
          int_sets[i].CoverRange(Range::FromMinExtent(0, buffer->shape[i])));
    }
    return BufferRegion(buffer, std::move(region));
  }

  void UpdateWrite(Buffer buffer, std::vector<arith::IntSet> region,
                   const std::vector<PrimExpr> &mins,
                   const std::vector<PrimExpr> &extents,
                   bool must_execute = true) {
    if (mode_ == AccessMode::kPhysical && conditional_depth_ == 0 &&
        must_execute && IsDenseWriteRegion(mins, extents, region))
      unconditional_write_regions_.push_back(MakeBufferRegion(buffer, region));
    Update(&write_buffers_, &write_regions_, std::move(buffer),
           std::move(region));
  }

  // Return true only when the rectangular hull produced by relaxing this
  // access is exactly covered. In particular, reject strided writes such as
  // a[2 * i] and correlated multi-dimensional writes such as a[i, i].
  bool IsDenseWriteRegion(const std::vector<PrimExpr> &mins,
                          const std::vector<PrimExpr> &extents,
                          const std::vector<arith::IntSet> &region) {
    ICHECK_EQ(mins.size(), extents.size());
    ICHECK_EQ(mins.size(), region.size());
    // CoverRange falls back to the whole allocation for an unbounded set.
    // That is a may-write bound, never evidence that every element is written.
    for (const arith::IntSet &axis : region) {
      if (!axis.HasLowerBound() || !axis.HasUpperBound())
        return false;
    }
    // loop_domains_ stores a continuous interval for each loop variable. It
    // therefore cannot prove dense coverage for a stepped loop without first
    // rewriting the loop variable through a dense logical iteration ordinal.
    if (non_unit_step_loop_depth_ != 0)
      return false;

    auto uses_active_loop = [&](const PrimExpr &expr) {
      return UsesVar(expr, [&](const VarNode *node) {
        Var var = ffi::GetRef<Var>(node);
        for (const auto &[loop_var, _] : loop_domains_) {
          if (var.same_as(loop_var))
            return true;
        }
        return false;
      });
    };

    ffi::Map<Var, Range> input_iters;
    for (const auto &[loop_var, loop_domain] : loop_domains_) {
      PrimExpr min = ResolveLets(loop_domain->min);
      PrimExpr extent = ResolveLets(loop_domain->extent);
      // Dependent loop domains do not describe a Cartesian iteration space to
      // DetectIterMap. Zero-trip loops also cannot establish a definition.
      if (uses_active_loop(min) || uses_active_loop(extent) ||
          !ana_.CanProve(extent > 0)) {
        return false;
      }
      input_iters.Set(loop_var, Range::FromMinExtent(min, extent));
    }

    ffi::Array<PrimExpr> expanded_indices;
    expanded_indices.reserve(mins.size());
    for (size_t i = 0; i < mins.size(); ++i) {
      PrimExpr min = ResolveLets(mins[i]);
      PrimExpr extent = ResolveLets(extents[i]);
      if (uses_active_loop(extent) || !ana_.CanProve(extent > 0))
        return false;
      Var offset("must_write_offset", min.dtype());
      input_iters.Set(offset,
                      Range::FromMinExtent(make_const(min.dtype(), 0), extent));
      expanded_indices.push_back(min + offset);
    }

    arith::IterMapResult iter_map =
        arith::DetectIterMap(expanded_indices, input_iters, Bool(true),
                             arith::IterMapLevel::Surjective, &ana_);
    if (iter_map->indices.size() != expanded_indices.size() ||
        !ana_.CanProve(!iter_map->padding_predicate)) {
      return false;
    }
    for (size_t i = 0; i < iter_map->indices.size(); ++i) {
      const arith::IterSumExpr &index = iter_map->indices[i];
      PrimExpr min = index->base;
      PrimExpr max = index->base;
      if (!index->args.empty()) {
        if (index->args.size() != 1)
          return false;
        const arith::IterSplitExpr &split = index->args[0];
        PrimExpr one = make_const(split->scale.dtype(), 1);
        if (ana_.CanProveEqual(split->scale, one)) {
          max = max + split->extent - one;
        } else if (ana_.CanProveEqual(split->scale, -one)) {
          min = min - split->extent + one;
        } else {
          return false;
        }
      }
      // Surjectivity proves coverage of the iterator image, not an arbitrary
      // EvalSet overestimate. A runtime index (even modulo the buffer size)
      // writes one element, rather than all elements in its possible range.
      if (!ana_.CanProveEqual(region[i].min(), min) ||
          !ana_.CanProveEqual(region[i].max(), max)) {
        return false;
      }
    }
    return true;
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

  void UpdateOpaqueAccessVar(const Var &var) {
    if (mode_ != AccessMode::kPhysical)
      return;
    for (const Var &existing : opaque_access_vars_) {
      if (existing.same_as(var))
        return;
    }
    opaque_access_vars_.push_back(var);
  }

  /*!
   * \brief Process a buffer region argument from reduce operation
   * \param arg The argument which could be BufferRegion, BufferLoad, or
   * tl.region call
   * \param is_read Whether this is a read (true) or write (false) access
   */
  void RecordBufferRegion(const BufferRegion &buffer_region, bool is_read) {
    Buffer buffer = buffer_region->buffer;
    const Region &region = buffer_region->region;
    std::vector<arith::IntSet> int_sets;
    std::vector<PrimExpr> mins;
    std::vector<PrimExpr> extents;
    int_sets.reserve(region.size());
    mins.reserve(region.size());
    extents.reserve(region.size());
    for (const Range &range : region) {
      int_sets.push_back(RelaxAccessIndex(range->min, range->extent));
      mins.push_back(range->min);
      extents.push_back(range->extent);
    }
    if (is_read) {
      Update(&read_buffers_, &read_regions_, buffer, int_sets);
    } else {
      UpdateWrite(buffer, std::move(int_sets), mins, extents);
    }
  }

  BufferRegion ExpandPaddedCopyWriteRegion(const AscendCopy &copy,
                                           BufferRegion write) {
    if (copy->data_select == 0 || !IsGlobalBuffer(copy->src) ||
        !IsSharedBuffer(copy->dst) || write->region.empty()) {
      return write;
    }

    // DMA rows follow the coalesced source/destination layouts, not the last
    // logical buffer axis. Use the same coalescing and unit-axis normalization
    // as PlanMTECopy, then select the row boundary from the strided side.
    arith::Analyzer analyzer;
    auto src_range = NormalizeEmptyUnitAxesForMTE(copy->src_range, &analyzer);
    auto dst_range = NormalizeEmptyUnitAxesForMTE(copy->dst_range, &analyzer);
    int src_bits = copy->src->dtype.bits() * copy->src->dtype.lanes();
    int elem_bits = write->buffer->dtype.bits() * write->buffer->dtype.lanes();
    auto src = StridedLayout::FromBufferRange(copy->src, src_range, src_bits)
                   .Coalesce(&analyzer);
    auto dst = StridedLayout::FromBufferRange(copy->dst, dst_range, elem_bits)
                   .Coalesce(&analyzer);
    auto total_elems = [&](const StridedLayout &layout) {
      PrimExpr total = Integer(1);
      for (const Mode &mode : layout.modes)
        total = total * mode.size;
      return total;
    };
    if (src_bits != elem_bits || src.modes.empty() || dst.modes.empty() ||
        src.modes.size() > 2 || dst.modes.size() > 2 ||
        !analyzer.CanProveEqual(total_elems(src), total_elems(dst)) ||
        (src.modes.size() == 2 && dst.modes.size() == 2 &&
         (!analyzer.CanProveEqual(src.modes[0].size, dst.modes[0].size) ||
          !analyzer.CanProveEqual(src.modes[1].size, dst.modes[1].size)))) {
      UpdateOpaqueAccessVar(write->buffer->data);
      return write;
    }
    PrimExpr row_elems = analyzer.Simplify(
        src.modes.size() == 2 ? src.modes[0].size : dst.modes[0].size);
    const auto *row_extent = row_elems.as<IntImmNode>();
    if (row_extent == nullptr) {
      UpdateOpaqueAccessVar(write->buffer->data);
      return write;
    }

    if (elem_bits <= 0 || 256 % elem_bits != 0 || row_extent->value <= 0 ||
        row_extent->value >
            (std::numeric_limits<int64_t>::max() - 255) / elem_bits) {
      UpdateOpaqueAccessVar(write->buffer->data);
      return write;
    }
    int64_t row_bits = row_extent->value * elem_bits;
    if (row_bits % 8 != 0) {
      UpdateOpaqueAccessVar(write->buffer->data);
      return write;
    }
    int64_t padded_extent = ((row_bits + 255) / 256) * 256 / elem_bits;
    if (padded_extent == row_extent->value)
      return write;

    Region write_region = write->region;
    const Range &row_range = write_region.back();
    PrimExpr last_stride = write->buffer->strides.empty()
                               ? Integer(1)
                               : write->buffer->strides.back();
    // Expanding the last logical axis is exact only when it is the descriptor's
    // contiguous row. In a view with stride=16, for example, row_elems=1 and
    // its seven padded floats occupy gaps between view elements, not extra
    // logical rows. Those physical lanes have no region in this view.
    if (!analyzer.CanProveEqual(last_stride, Integer(1)) ||
        !analyzer.CanProveEqual(row_range->extent, row_elems)) {
      UpdateOpaqueAccessVar(write->buffer->data);
      return write;
    }

    PrimExpr physical_extent = IntImm(row_range->extent.dtype(), padded_extent);
    // Lifetime analysis may reject reuse, but must not add an InsertSync
    // failure. Keep a finite footprint when the offset is symbolic.
    if (analyzer.CanProve(row_range->min + physical_extent >
                          write->buffer->shape.back()))
      UpdateOpaqueAccessVar(write->buffer->data);
    write_region.Set(write_region.size() - 1,
                     Range::FromMinExtent(row_range->min, physical_extent));
    return BufferRegion(write->buffer, std::move(write_region));
  }

  void ProcessBufferRegion(const PrimExpr &arg, bool is_read) {
    // Check if it's a BufferRegion
    if (const auto *buffer_region = arg.as<BufferRegionNode>()) {
      RecordBufferRegion(ffi::GetRef<BufferRegion>(buffer_region), is_read);
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
        std::vector<PrimExpr> extents(buffer_load->indices.size(),
                                      make_const(DataType::Int(32), 1));
        UpdateWrite(buffer, std::move(int_sets),
                    std::vector<PrimExpr>(buffer_load->indices.begin(),
                                          buffer_load->indices.end()),
                    extents);
      }
      return;
    }

    // Check if it's a tl.region call (should already be handled by
    // VisitExpr_) but we can still process it recursively
    if (const auto *call = arg.as<CallNode>()) {
      static const Op &region_op = region();
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
      result.push_back(MakeBufferRegion(buffers[i], regions[i]));
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
    loop_domains_.emplace_back(op->loop_var, range);
    PrimExpr step = op->step.has_value() ? ResolveLets(op->step.value())
                                         : make_const(op->loop_var.dtype(), 1);
    bool has_non_unit_step =
        !ana_.CanProveEqual(step, make_const(step.dtype(), 1));
    if (has_non_unit_step)
      ++non_unit_step_loop_depth_;
    StmtExprVisitor::VisitStmt_(op);
    if (has_non_unit_step)
      --non_unit_step_loop_depth_;
    loop_domains_.pop_back();
    dom_map_.erase(op->loop_var.get());
  }

  void VisitStmt_(const IfThenElseNode *op) override {
    VisitExpr(op->condition);
    ++conditional_depth_;
    // The ordinary write regions remain may-write unions. Writes visited while
    // this depth is nonzero are excluded from the separate must-write list.
    StmtExprVisitor::VisitStmt(op->then_case);
    if (op->else_case) {
      // Visit else branch
      StmtExprVisitor::VisitStmt(op->else_case.value());
    }
    --conditional_depth_;
  }

  void VisitStmt_(const WhileNode *op) override {
    VisitExpr(op->condition);
    ++conditional_depth_;
    StmtExprVisitor::VisitStmt(op->body);
    --conditional_depth_;
  }

  void VisitStmt_(const BindNode *op) override {
    bool previous_context = opaque_pointer_context_;
    opaque_pointer_context_ |= op->var.dtype().is_handle();
    VisitExpr(op->value);
    opaque_pointer_context_ = previous_context;
    let_bindings_[op->var.get()] = op->value;
    UpdateWriteVar(op->var);
  }

  void VisitStmt_(const AttrStmtNode *op) override {
    if (op->attr_key == tirx::attr::tilelang_assume ||
        op->attr_key == attr::kAssumeRequiresRuntimeCheck) {
      if (const auto *expr = op->node.as<PrimExprNode>())
        VisitExpr(ffi::GetRef<PrimExpr>(expr));
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const VarNode *op) override {
    // Bare storage pointers have no bounded footprint. A structured access in
    // the same task must not hide their escape to an external consumer.
    if (op->dtype.is_handle())
      UpdateOpaqueAccessVar(ffi::GetRef<Var>(op));
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
    std::vector<PrimExpr> mins;
    std::vector<PrimExpr> extents;
    size_t num_indices = op->indices.size();
    size_t buffer_ndim = op->buffer->shape.size();

    // Assert that indices count equals buffer dimension
    ICHECK_EQ(num_indices, buffer_ndim)
        << "BufferStore indices count " << num_indices
        << " must equal buffer dimension " << buffer_ndim;

    relaxed_region.reserve(num_indices);
    mins.reserve(num_indices);
    extents.reserve(num_indices);
    for (PrimExpr index : op->indices) {
      relaxed_region.push_back(RelaxAccessIndex(index));
      mins.push_back(index);
      extents.push_back(make_const(index.dtype(), 1));
    }
    bool must_execute = !op->predicate.defined() ||
                        (op->predicate.value().dtype().is_scalar() &&
                         ana_.CanProve(op->predicate.value()));
    UpdateWrite(op->buffer, std::move(relaxed_region), mins, extents,
                must_execute);
    // Storing an access_ptr in a handle carrier lets later consumers advance
    // or reinterpret it beyond the original declared region.
    bool previous_context = opaque_pointer_context_;
    opaque_pointer_context_ |= op->value.dtype().is_handle();
    StmtExprVisitor::VisitStmt_(op);
    opaque_pointer_context_ = previous_context;
  }

  // Calculate a physical byte span without letting overflow prove coverage.
  static std::optional<int64_t> PhysicalSpan(int64_t repeats, int64_t stride,
                                             int64_t tail) {
    if (repeats < 0 || stride < 0 || tail < 0 ||
        (stride != 0 &&
         repeats > (std::numeric_limits<int64_t>::max() - tail) / stride))
      return std::nullopt;
    return repeats * stride + tail;
  }

  void VisitPhysicalPointer(const PrimExpr &pointer,
                            std::optional<int64_t> bytes, int mask) {
    if (const auto *call = pointer.as<CallNode>();
        call && call->op.same_as(tl::access_ptr()) && call->args.size() >= 3) {
      if (const auto *load = call->args[0].as<BufferLoadNode>()) {
        DataType dtype = load->buffer->dtype;
        PrimExpr extent = ana_.Simplify(ResolveLets(call->args[1]));
        const auto *declared_extent = extent.as<IntImmNode>();
        const auto *declared_mask = call->args[2].as<IntImmNode>();
        if (bytes && *bytes > 0 && dtype.is_scalar() && dtype.bits() >= 8 &&
            dtype.bits() % 8 == 0 && declared_extent && declared_mask) {
          int64_t elements = (*bytes - 1) / (dtype.bits() / 8) + 1;
          // Keep the producer's conservative region, including padding. A
          // smaller physical span must not narrow its logical-view contract.
          if (elements <= std::numeric_limits<int32_t>::max() &&
              declared_extent->value >= elements &&
              (declared_mask->value & mask) == mask) {
            VisitExpr(pointer);
            return;
          }
        }
        UpdateOpaqueAccessVar(load->buffer->data);
      }
    }
    bool previous_context = opaque_pointer_context_;
    opaque_pointer_context_ = true;
    VisitExpr(pointer);
    opaque_pointer_context_ = previous_context;
  }

  void VisitNd2NzCall(const CallNode *op, bool scatter) {
    std::optional<int64_t> source_bytes;
    std::optional<int64_t> destination_bytes;
    if (op->args.size() == 6) {
      const auto *rows = op->args[2].as<IntImmNode>();
      const auto *cols = op->args[3].as<IntImmNode>();
      auto element_bytes = [&](size_t index) -> int64_t {
        const auto *dtype = op->args[index].as<StringImmNode>();
        if (dtype && dtype->value == "float")
          return 4;
        if (dtype && (dtype->value == "half" || dtype->value == "bfloat16_t"))
          return 2;
        return 0;
      };
      if (rows && cols && rows->value > 0 && cols->value > 0 &&
          rows->value < std::numeric_limits<int64_t>::max()) {
        int64_t dst_bytes = element_bytes(scatter ? 4 : 5);
        if (scatter) {
          int64_t src_bytes = element_bytes(5);
          auto source_row = PhysicalSpan(cols->value, src_bytes, 0);
          auto destination_row = PhysicalSpan(cols->value, dst_bytes, 0);
          if (src_bytes && dst_bytes && source_row && destination_row) {
            // The scatter template loads whole 256-byte vectors, including
            // the last partial vector of the last source row.
            int64_t vectors = (*source_row - 1) / 256 + 1;
            if (auto last_row = PhysicalSpan(vectors, 256, 0))
              source_bytes =
                  PhysicalSpan(rows->value - 1, *source_row, *last_row);
            destination_bytes =
                PhysicalSpan(rows->value + 1, *destination_row, 0);
          }
        } else {
          const auto *full_rows = op->args[4].as<IntImmNode>();
          if (dst_bytes && full_rows && full_rows->value >= rows->value &&
              cols->value % (32 / dst_bytes) == 0) {
            int64_t bursts = cols->value / (32 / dst_bytes);
            // Match EmitNd2NzPostCopy_: 32-byte DMA blocks, one source-gap
            // row, and full_rows - rows destination-gap rows per burst.
            if (auto blocks =
                    PhysicalSpan(bursts - 1, rows->value + 1, rows->value))
              source_bytes = PhysicalSpan(*blocks, 32, 0);
            if (auto blocks =
                    PhysicalSpan(bursts - 1, full_rows->value, rows->value))
              destination_bytes = PhysicalSpan(*blocks, 32, 0);
          }
        }
      }
    }
    for (size_t i = 0; i < op->args.size(); ++i) {
      if (i == (scatter ? 0U : 1U))
        VisitPhysicalPointer(op->args[i], source_bytes, 1);
      else if (i == (scatter ? 1U : 0U))
        VisitPhysicalPointer(op->args[i], destination_bytes, 2);
      else
        VisitExpr(op->args[i]);
    }
    for (const auto &kv : op->annotations) {
      if (auto expr = kv.second.as<PrimExpr>())
        VisitExpr(expr.value());
    }
  }

  std::vector<arith::IntSet> LogicalPointerRegion(const BufferLoad &load,
                                                  const PrimExpr &extent) {
    const Buffer &buffer = load->buffer;
    size_t ndim = buffer->shape.size();
    std::vector<arith::IntSet> region(ndim);

    const auto *ext_imm = extent.as<IntImmNode>();
    if (ext_imm && load->indices.size() == ndim) {
      // Distribute a known flat extent (in elements) across dims,
      // innermost-first. `rem` is the number of elements still to be
      // covered at the current dimension; each dim consumes as many of
      // its `shape[i]` indices as the run spans, carrying the rounded-up
      // remainder to the next (outer) dim.
      int64_t rem = ext_imm->value;
      for (size_t k = ndim; k-- > 0;) {
        PrimExpr base = load->indices[k];
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
        region[k] =
            arith::IntSet::FromRange(Range::FromMinExtent(0, buffer->shape[k]));
      }
    }
    return region;
  }

  std::vector<arith::IntSet> PhysicalPointerRegion(const BufferLoad &load,
                                                   const PrimExpr &extent) {
    const Buffer &buffer = load->buffer;
    size_t ndim = buffer->shape.size();
    std::vector<arith::IntSet> region(ndim);

    bool reconstructed = false;
    const auto *ext_imm = extent.as<IntImmNode>();
    // Explicit strides, including dense T.view layouts, stay opaque
    // for multi-element runs until their physical image is proved.
    if (ext_imm && ext_imm->value > 0 && load->indices.size() == ndim &&
        (ext_imm->value == 1 || buffer->strides.empty())) {
      int64_t remaining = ext_imm->value;
      reconstructed = true;
      for (size_t k = ndim; k-- > 0;) {
        PrimExpr base = load->indices[k];
        if (remaining == 1) {
          region[k] = RelaxAccessIndex(base);
          if (!region[k].HasLowerBound() || !region[k].HasUpperBound()) {
            reconstructed = false;
            break;
          }
          continue;
        }

        const int64_t *dimension = as_const_int(buffer->shape[k]);
        arith::IntSet base_set = RelaxAccessIndex(base);
        const int64_t *base_min = as_const_int(base_set.min());
        const int64_t *base_max = as_const_int(base_set.max());
        if (dimension == nullptr || *dimension <= 0 || base_min == nullptr ||
            base_max == nullptr || *base_min < 0 || *base_max >= *dimension ||
            *base_max > std::numeric_limits<int64_t>::max() - remaining) {
          reconstructed = false;
          break;
        }

        int64_t end_exclusive = *base_max + remaining;
        int64_t span =
            end_exclusive / *dimension + (end_exclusive % *dimension != 0);
        if (span == 1) {
          region[k] = RelaxAccessIndex(base, IntImm(base.dtype(), remaining));
          remaining = 1;
        } else {
          // Wrapping this dimension can touch both its high and low
          // coordinates, so its rectangular hull is the whole axis.
          region[k] = arith::IntSet::FromRange(
              Range::FromMinExtent(0, buffer->shape[k]));
          remaining = span;
        }
      }
      // A remaining carry escapes the outermost logical dimension, even
      // if the backing allocation has room for the physical access.
      reconstructed &= remaining == 1;
    }
    if (!reconstructed) {
      UpdateOpaqueAccessVar(buffer->data);
      // The storage is opaque: this placeholder cannot prove reuse and
      // is never passed to synchronization dependency analysis.
      for (size_t k = 0; k < ndim; ++k) {
        region[k] =
            arith::IntSet::FromRange(Range::FromMinExtent(0, buffer->shape[k]));
      }
    }
    return region;
  }

  void VisitExpr_(const CallNode *op) override {
    static const Op &region_op = region();
    static const auto reduce_op = Op::Get("tl.tileop.reduce");
    static const auto tl_access_ptr_op = Op::Get("tl.access_ptr");

    if (mode_ == AccessMode::kPhysical &&
        (op->op.same_as(tl::ascend_nd2nz_scatter()) ||
         op->op.same_as(tl::ascend_nd2nz_post_copy()))) {
      // These are statement intrinsics even when legacy producers use a
      // handle return type. Check their declared pointer spans against the
      // actual DMA parameters; a base-element extent is not a sound footprint.
      VisitNd2NzCall(op, op->op.same_as(tl::ascend_nd2nz_scatter()));
      return;
    }

    if (op->op.as<OpNode>()) {
      // vld2(addr, dist[, off]) places its optional address operand third.
      // Keep calls with additional addressing operands conservative as well.
      if (op->op.same_as(tl::simd_vgatherb()) ||
          op->op.same_as(tl::simd_vgather2()) ||
          op->op.same_as(tl::simd_vscatter()) ||
          op->op.same_as(tl::simd_vsstb()) ||
          (op->op.same_as(tl::simd_vld2()) && op->args.size() >= 3)) {
        // SIMD gather/scatter and offset-bearing vld2 carry separate
        // addressing operands outside access_ptr. The
        // pointer's compact extent cannot establish physical lifetime coverage.
        bool previous_context = opaque_pointer_context_;
        opaque_pointer_context_ = true;
        StmtExprVisitor::VisitExpr_(op);
        opaque_pointer_context_ = previous_context;
        return;
      }
    }

    if (op->op.same_as(tl::loop_break()))
      has_loop_break_ = true;

    if (mode_ == AccessMode::kPhysical && IsAscendCopyCall(op)) {
      // Visit the original operands first so symbolic indices and the logical
      // source/destination regions retain their normal access accounting.
      StmtExprVisitor::VisitExpr_(op);
      AscendCopy copy(op->args, op->annotations);
      for (const BufferRegion &write : copy->GetAccessRegions().writes)
        RecordBufferRegion(ExpandPaddedCopyWriteRegion(copy, write), false);
      return;
    }

    // Check for tl.region call
    if (op->op.same_as(region_op)) {
      // Handle tl.region call for memory access analysis
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
              LOG(FATAL) << "BufferLoad in tl.region should access a "
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
              relaxed_region.push_back(RelaxAccessIndex(min, extent));
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
            std::vector<PrimExpr> extents;
            extents.reserve(num_indices);
            if (op->args.size() > 2) {
              for (size_t i = 0; i < num_indices; ++i)
                extents.push_back(op->args[2 + i]);
            } else {
              for (const PrimExpr &index : buffer_load->indices)
                extents.push_back(make_const(index.dtype(), 1));
            }
            UpdateWrite(buffer, std::move(relaxed_region),
                        std::vector<PrimExpr>(buffer_load->indices.begin(),
                                              buffer_load->indices.end()),
                        extents);
          }

          for (const auto &index : buffer_load->indices) {
            VisitExpr(index);
          }
          for (size_t i = 2; i < op->args.size(); ++i) {
            VisitExpr(op->args[i]);
          }
        } else {
          LOG(FATAL) << "First argument of tl.region should be a BufferLoad";
        }
      }
      return;
    }

    // Check for tl.tileop.reduce call
    if (op->op.same_as(reduce_op)) {
      // Handle tl.tileop.reduce call for memory access analysis
      // args[0] = input buffer region (read)
      // args[1] = output buffer region (write, and read when clear=false)
      // args[2] = reduce_type (string)
      // args[3] = dim (int)
      // args[4] = clear (bool)
      if (op->args.size() >= 2) {
        // Process first argument as read region
        ProcessBufferRegion(op->args[0], true); // is_read = true
        // Process second argument as write region
        ProcessBufferRegion(op->args[1], false); // is_read = false
        // The frontend historically encodes the destination region as `w`
        // even when clear=false. In that mode the reduction accumulates onto
        // the old destination value, so preserve the read side explicitly.
        bool reads_destination = true;
        if (op->args.size() >= 5) {
          if (const auto *clear = op->args[4].as<IntImmNode>())
            reads_destination = clear->value == 0;
        }
        if (mode_ == AccessMode::kPhysical && reads_destination)
          RecordBufferRegion(NormalizeToBufferRegion(op->args[1]), true);
      }
      return;
    }

    if (op->op.same_as(builtin::address_of())) {
      // The base element does not describe the footprint of a pointer passed
      // to an external consumer: address_of(a[0]) may also access a[1]. Keep
      // the storage opaque even when this task has other precise regions.
      if (!op->args.empty()) {
        if (const auto *load = op->args[0].as<BufferLoadNode>())
          UpdateOpaqueAccessVar(load->buffer->data);
      }
      // Still visit the base/index expressions for ordinary dependencies.
      StmtExprVisitor::VisitExpr_(op);
      return;
    }

    // Handle other calls (e.g., builtin::tvm_access_ptr)
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      // This flat pointer has no logical Buffer handle from which this
      // detector can reconstruct a sound region. Mark the storage explicitly
      // so a structured access in the same task cannot hide this opaque use.
      if (op->args.size() >= 2) {
        if (const auto *storage = op->args[1].as<VarNode>())
          UpdateOpaqueAccessVar(ffi::GetRef<Var>(storage));
      }
      StmtExprVisitor::VisitExpr_(op);
      return;
    }

    // Handle TileLang `tl.access_ptr(BufferLoad(buf, mins), extent, rw_mask)`.
    // The access is a contiguous run of `extent` elements in row-major layout
    // starting at the base index `mins`. Reconstruct a conservative rectangular
    // hull by carrying both the extent and each dimension's starting offset. If
    // the physical footprint cannot be represented, exclude the storage from
    // alias proofs. A whole logical view can omit physical storage elements.
    if (op->op.same_as(tl_access_ptr_op)) {
      if (op->args.size() >= 3) {
        const auto *buffer_load = op->args[0].as<BufferLoadNode>();
        const auto *mask_int = op->args[2].as<IntImmNode>();
        if (buffer_load && mask_int) {
          Buffer buffer = buffer_load->buffer;
          if (opaque_pointer_context_)
            UpdateOpaqueAccessVar(buffer->data);
          int rw_mask = mask_int->value;
          BufferLoad load = ffi::GetRef<BufferLoad>(buffer_load);
          std::vector<arith::IntSet> region =
              mode_ == AccessMode::kLogical
                  ? LogicalPointerRegion(load, op->args[1])
                  : PhysicalPointerRegion(load, op->args[1]);

          if (rw_mask & 1) {
            Update(&read_buffers_, &read_regions_, buffer, region);
          }
          if (rw_mask & 2) {
            // access_ptr describes the range an enclosing intrinsic may write,
            // not a guarantee that every element is written. Predicated SIMD
            // stores are a common counterexample, so keep this as may-write.
            Update(&write_buffers_, &write_regions_, buffer, std::move(region));
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

    // A generic pointer-returning call may advance or reinterpret its input
    // pointer even when no Bind or handle buffer carries the result. The
    // bounded access_ptr and address_of cases above keep their own semantics.
    bool previous_context = opaque_pointer_context_;
    opaque_pointer_context_ |= op->dtype.is_handle();
    StmtExprVisitor::VisitExpr_(op);
    opaque_pointer_context_ = previous_context;
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

} // namespace ascend
} // namespace tl
} // namespace tvm
