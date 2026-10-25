#pragma once
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/sym/iter_affine_map.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./ir_structure.h"
#include "ascend/layout/ascend_layouts.h"
#include "ascend/op/ascend_mte_plan.h"
#include "ascend/op/builtin.h"
#include "ascend/op/copy.h"
#include "ascend/op/oob_padding.h"
#include "ascend/op/utils.h"
#include "ascend/transform/native_visitor.h"
#include "compat/visit.h"
#include "op/builtin.h"
#include "op/gemm.h"
#include "op/gemm_blockscaled.h"
#include "op/operator.h"
#include "op/utils.h"
#include "support/check.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;

inline bool
ScheduleExprUsesVar(const PrimExpr &expression,
                    const std::function<bool(const VarNode *)> &predicate) {
  for (const Var &variable : tvm::UndefinedVars(expression))
    if (predicate(variable.get()))
      return true;
  return false;
}

inline bool HasContiguousTaskLayout(const TensorVar &tensor,
                                    sym::Analyzer analyzer) {
  auto pattern = tirx::MatchShapeStridePattern(tensor.ty());
  if (!pattern)
    return false;
  PrimExpr stride = IntImm::Int32(1);
  const auto &strides = pattern.value().get<1>();
  for (size_t axis = tensor.shape().size(); axis > 0; --axis) {
    if (!analyzer->CanProveEqual(strides[axis - 1], stride))
      return false;
    stride = stride * tensor.shape()[axis - 1];
  }
  return true;
}

inline void MergeL0Pitch(Map<Var, Optional<PrimExpr>> *pitches,
                         const Var &storage, Optional<PrimExpr> pitch) {
  auto it = pitches->find(storage);
  if (it != pitches->end()) {
    sym::Analyzer analyzer;
    if (!(*it).second.has_value() || !pitch.has_value() ||
        !analyzer->CanProveEqual((*it).second.value(), pitch.value()))
      pitch = std::nullopt;
  }
  pitches->Set(storage, pitch);
}

inline Map<Var, PrimExpr> GetCallL0Pitches(const Call &call,
                                           const LayoutMap &layouts) {
  Map<Var, PrimExpr> result;
  auto record = [&](const TensorRegion &region, size_t mn_axis,
                    Optional<PrimExpr> l0_pitch = std::nullopt) {
    PrimExpr pitch =
        l0_pitch.value_or(floordiv(region->region[mn_axis]->extent + 15, 16));
    result.Set(region->source.as_or_throw<TensorVar>(), pitch);
  };
  if (IsAscendCopyCall(call.get())) {
    AscendCopy copy(call->args, AscendCallAnnotations(call.get()));
    if (IsL1Buffer(copy->src) &&
        (IsL0ABuffer(copy->dst) || IsL0BBuffer(copy->dst))) {
      auto layout = layouts.Get(copy->dst);
      AscendFractalLayoutInfo info;
      if (layout.has_value() &&
          TryExtractAscendFractalLayout(layout.value(), copy->dst, &info) &&
          IsAscendMajorKind(info.kind))
        record(NativeTensorRegion(copy->dst, copy->dst_range),
               copy->dst_range.size() - 2 +
                   (info.row16_axis == LogicalAxis::kCol),
               copy->l0_pitch);
    } else if (IsL0CBuffer(copy->src)) {
      record(NativeTensorRegion(copy->src, copy->src_range),
             copy->src_range.size() - 2, copy->l0_pitch);
    }
  } else if ((call->op.same_as(Gemm::Get()) ||
              call->op.same_as(GemmBlockScaled::Get()))) {
    Gemm gemm = ParseOperator(call).as_or_throw<Gemm>();
    if (IsL0ABuffer(gemm->a_) && IsL0BBuffer(gemm->b_)) {
      record(gemm->aRegion_, gemm->a_.shape().size() - (gemm->transA_ ? 1 : 2));
      record(gemm->bRegion_, gemm->b_.shape().size() - (gemm->transB_ ? 2 : 1));
      record(gemm->cRegion_, gemm->c_.shape().size() - 2);
    }
  }
  return result;
}

// MemoryAccessDetector: detect read/write regions in statements
// Adapted from BlockReadWriteDetector in TVM
class MemoryAccessDetectorImpl : public AnalysisVisitor {
public:
  // Lifetime analysis separately requests may/must-write coverage.
  enum class AccessMode { kLogical, kPhysical };
  explicit MemoryAccessDetectorImpl(AccessMode mode = AccessMode::kLogical,
                                    LayoutMap layouts = {})
      : layouts_(std::move(layouts)), mode_(mode) {}

  // Analyze a statement and collect read/write regions
  void Analyze(const Stmt &stmt) {
    l0_pitches_.clear();
    operation_pitches_.clear();
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

  void Analyze(const Expr &expr) {
    l0_pitches_.clear();
    operation_pitches_.clear();
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
  std::vector<TensorRegion> GetReadRegions() const {
    return CollectRegions(read_buffers_, read_regions_);
  }

  // Return collected write regions
  std::vector<TensorRegion> GetWriteRegions() const {
    return CollectRegions(write_buffers_, write_regions_);
  }

  const Map<Var, Optional<PrimExpr>> &GetL0Pitches() const {
    return l0_pitches_;
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
  std::vector<TensorRegion> GetMustWriteRegions() const {
    // A loop_break may truncate any loop domain represented by this atomic
    // task. Keep its may-write regions for dependencies, but do not use any
    // write in the task as a complete definition for lifetime analysis.
    if (has_loop_break_)
      return {};
    return unconditional_write_regions_;
  }

private:
  LayoutMap layouts_;
  Map<Var, PrimExpr> operation_pitches_;
  Map<Var, Optional<PrimExpr>> l0_pitches_;

  /*! \brief Iteration range for loop_vars */
  ffi::Map<Var, sym::IntSet> dom_map_;
  /*! \brief Extra iteration range hint for free vars */
  ffi::Map<Var, sym::IntSet> hint_map_;
  /*! \brief Unresolved conditions within current scope. */
  std::vector<PrimExpr> pending_conditions_;
  /*! \brief Individual dense writes that execute whenever the task does. */
  std::vector<TensorRegion> unconditional_write_regions_;
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
  std::vector<TensorVar> read_buffers_;
  /*! \brief The buffers that the current block writes */
  std::vector<TensorVar> write_buffers_;
  /*! \brief The read regions of the current block */
  std::vector<std::vector<tvm::sym::IntSet>> read_regions_;
  /*! \brief The write regions of the current block */
  std::vector<std::vector<tvm::sym::IntSet>> write_regions_;
  AccessMode mode_;
  /*!\ brief Internal analyzer. */
  sym::Analyzer ana_;
  /*! \brief let bindings inside the block */
  ffi::Map<PrimVar, PrimExpr> let_bindings_;

  /*! \brief The set of variables that are read in the current block.  */
  std::vector<Var> read_vars_;
  /*! \brief The set of variables that are written in the current block.  */
  std::vector<Var> write_vars_;
  /*! \brief TensorVar storage Vars with an access of unknown region. */
  std::vector<Var> opaque_access_vars_;

  /*!
   * \brief Update read/write buffers and regions with provided buffer and
   * region
   */
  void Update(std::vector<TensorVar> *buffers,
              std::vector<std::vector<sym::IntSet>> *regions, TensorVar buffer,
              std::vector<sym::IntSet> region) {
    UpdateReadVar(buffer);
    // Equal logical indices do not prove equal physical addresses when layout
    // metadata reads mutable state. Lowering uses these expressions at
    // accesses, so retain their reads as well as excluding the viewed storage
    // from reuse.
    auto visit_mutable_layout = [&](const PrimExpr &expr) {
      if (mode_ == AccessMode::kPhysical &&
          SideEffect(expr) > CallEffectKind::kPure) {
        UpdateOpaqueAccessVar(buffer);
        VisitExpr(expr);
      }
    };
    for (const PrimExpr &extent : buffer.shape())
      visit_mutable_layout(extent);
    if (auto geometry = tirx::MatchShapeStridePattern(buffer.ty())) {
      for (const PrimExpr &stride : geometry.value().get<1>())
        visit_mutable_layout(stride);
      visit_mutable_layout(geometry.value().get<2>());
    } else if (mode_ == AccessMode::kPhysical && buffer.ty()->layout) {
      auto indices = buffer.shape().Map(
          [](const PrimExpr &extent) { return MakeConst(extent.ty(), 0); });
      for (const PrimExpr &offset : buffer.ty()->ElemOffset(indices))
        visit_mutable_layout(offset);
    }
    if (mode_ == AccessMode::kLogical &&
        (IsL0ABuffer(buffer) || IsL0BBuffer(buffer) || IsL0CBuffer(buffer))) {
      Optional<PrimExpr> pitch = operation_pitches_.Get(buffer);
      if (pitch.has_value()) {
        PrimExpr value = ana_->Simplify(ResolveLets(pitch.value()));
        if (SideEffect(value) > CallEffectKind::kPure ||
            ScheduleExprUsesVar(value, [&](const VarNode *var) {
              return dom_map_.count(ffi::GetRef<Var>(var)) != 0;
            }))
          pitch = std::nullopt;
        else
          pitch = value;
      }
      MergeL0Pitch(&l0_pitches_, buffer, pitch);
    }
    MergeRegion(buffers, regions, buffer, region);
  }

  static void MergeRegion(std::vector<TensorVar> *buffers,
                          std::vector<std::vector<sym::IntSet>> *regions,
                          const TensorVar &buffer,
                          const std::vector<sym::IntSet> &region) {
    // Check if buffer already exists
    for (size_t i = 0; i < buffers->size(); ++i) {
      if ((*buffers)[i].same_as(buffer)) {
        // Merge regions
        ICHECK_EQ((*regions)[i].size(), region.size());
        for (size_t j = 0; j < region.size(); ++j) {
          (*regions)[i][j] = sym::Union({(*regions)[i][j], region[j]});
        }
        return;
      }
    }
    // New buffer
    buffers->push_back(buffer);
    regions->push_back(region);
  }

  TensorRegion
  MakeTensorRegion(const TensorVar &buffer,
                   const std::vector<sym::IntSet> &int_sets) const {
    ffi::Array<Range> region;
    size_t ndim = buffer.shape().size();
    ICHECK_EQ(int_sets.size(), ndim)
        << "ffi::Array<Range> dimension " << int_sets.size()
        << " must equal buffer dimension " << ndim;
    region.reserve(ndim);
    for (size_t i = 0; i < ndim; ++i) {
      region.push_back(
          int_sets[i]
              .CoverRange(Range::FromMinExtent(PrimExpr(0), buffer.shape()[i]))
              .value());
    }
    return NativeTensorRegion(buffer, std::move(region));
  }

  void UpdateWrite(TensorVar buffer, std::vector<sym::IntSet> region,
                   const std::vector<PrimExpr> &mins,
                   const std::vector<PrimExpr> &extents,
                   bool must_execute = true) {
    if (mode_ == AccessMode::kPhysical && conditional_depth_ == 0 &&
        must_execute && IsDenseWriteRegion(mins, extents, region))
      unconditional_write_regions_.push_back(MakeTensorRegion(buffer, region));
    Update(&write_buffers_, &write_regions_, std::move(buffer),
           std::move(region));
  }

  // Return true only when the rectangular hull produced by relaxing this
  // access is exactly covered. In particular, reject strided writes such as
  // a[2 * i] and correlated multi-dimensional writes such as a[i, i].
  bool IsDenseWriteRegion(const std::vector<PrimExpr> &mins,
                          const std::vector<PrimExpr> &extents,
                          const std::vector<sym::IntSet> &region) {
    ICHECK_EQ(mins.size(), extents.size());
    ICHECK_EQ(mins.size(), region.size());
    // CoverRange falls back to the whole allocation for an unbounded set.
    // That is a may-write bound, never evidence that every element is written.
    for (const sym::IntSet &axis : region) {
      if (!axis.HasLowerBound() || !axis.HasUpperBound())
        return false;
    }
    // loop_domains_ stores a continuous interval for each loop variable. It
    // therefore cannot prove dense coverage for a stepped loop without first
    // rewriting the loop variable through a dense logical iteration ordinal.
    if (non_unit_step_loop_depth_ != 0)
      return false;

    auto uses_active_loop = [&](const PrimExpr &expr) {
      return ScheduleExprUsesVar(expr, [&](const VarNode *node) {
        Var var = ffi::GetRef<Var>(node);
        for (const auto &[loop_var, _] : loop_domains_) {
          if (var.same_as(loop_var))
            return true;
        }
        return false;
      });
    };

    ffi::Map<PrimVar, Range> input_iters;
    for (const auto &[loop_var, loop_domain] : loop_domains_) {
      PrimExpr min = ResolveLets(loop_domain->begin);
      PrimExpr extent = ResolveLets(loop_domain->extent);
      // Dependent loop domains do not describe a Cartesian iteration space to
      // DetectIterMap. Zero-trip loops also cannot establish a definition.
      if (uses_active_loop(min) || uses_active_loop(extent) ||
          !ana_->CanProve(extent > 0)) {
        return false;
      }
      input_iters.Set(loop_var.as_or_throw<PrimVar>(),
                      Range::FromMinExtent(min, extent));
    }

    ffi::Array<PrimExpr> expanded_indices;
    expanded_indices.reserve(mins.size());
    for (size_t i = 0; i < mins.size(); ++i) {
      PrimExpr min = ResolveLets(mins[i]);
      PrimExpr extent = ResolveLets(extents[i]);
      if (uses_active_loop(extent) || !ana_->CanProve(extent > 0))
        return false;
      PrimVar offset("must_write_offset", min.ty());
      input_iters.Set(offset,
                      Range::FromMinExtent(MakeConst(min.ty(), 0), extent));
      expanded_indices.push_back(min + offset);
    }

    sym::IterMapResult iter_map =
        sym::DetectIterMap(expanded_indices, input_iters, IntImm::Bool(true),
                           sym::IterMapLevel::Surjective, ana_);
    if (iter_map->indices.size() != expanded_indices.size() ||
        !iter_map->padding_predicate ||
        !ana_->CanProve(!iter_map->padding_predicate.value())) {
      return false;
    }
    for (size_t i = 0; i < iter_map->indices.size(); ++i) {
      const sym::IterSumExpr &index = iter_map->indices[i];
      PrimExpr min = index->base;
      PrimExpr max = index->base;
      if (!index->args.empty()) {
        if (index->args.size() != 1)
          return false;
        const sym::IterSplitExpr &split = index->args[0];
        PrimExpr one = MakeConst(split->scale.ty(), 1);
        if (ana_->CanProveEqual(split->scale, one)) {
          max = max + split->extent - one;
        } else if (ana_->CanProveEqual(split->scale, -one)) {
          min = min - split->extent + one;
        } else {
          return false;
        }
      }
      // Surjectivity proves coverage of the iterator image, not an arbitrary
      // EvalSet overestimate. A runtime index (even modulo the buffer size)
      // writes one element, rather than all elements in its possible range.
      if (!ana_->CanProveEqual(region[i].min(), min) ||
          !ana_->CanProveEqual(region[i].max(), max)) {
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
    if (mode_ == AccessMode::kLogical)
      return;
    for (const Var &existing : opaque_access_vars_) {
      if (existing.same_as(var))
        return;
    }
    opaque_access_vars_.push_back(var);
  }

  /*!
   * \brief Process a buffer region argument from reduce operation
   * \param arg The argument which could be TensorRegion, TensorLoad, or
   * tl.region call
   * \param is_read Whether this is a read (true) or write (false) access
   */
  void RecordTensorRegion(const TensorRegion &buffer_region, bool is_read) {
    TensorVar buffer = buffer_region->source.as_or_throw<TensorVar>();
    const ffi::Array<Range> &region = buffer_region->region;
    std::vector<sym::IntSet> int_sets;
    std::vector<PrimExpr> mins;
    std::vector<PrimExpr> extents;
    int_sets.reserve(region.size());
    mins.reserve(region.size());
    extents.reserve(region.size());
    for (const Range &range : region) {
      int_sets.push_back(RelaxAccessIndex(range->begin, range->extent));
      mins.push_back(range->begin);
      extents.push_back(range->extent);
    }
    if (is_read) {
      Update(&read_buffers_, &read_regions_, buffer, int_sets);
    } else {
      UpdateWrite(buffer, std::move(int_sets), mins, extents);
    }
  }

  // Compact storage is opaque to allocation reuse. Preserve its selected
  // plane as may-access without treating it as a complete definition.
  void RecordL0MayAccess(const TensorRegion &region, bool is_read,
                         bool compact) {
    if (!compact)
      return;
    UpdateOpaqueAccessVar(region->source.as_or_throw<TensorVar>());
    std::vector<sym::IntSet> ranges;
    for (size_t i = 0; i < region->region.size(); ++i) {
      const Range &range = region->region[i];
      ranges.push_back(
          i + 2 < region->region.size()
              ? RelaxAccessIndex(range->begin, range->extent)
              : sym::IntSet::FromRange(Range::FromMinExtent(
                    0, region->source.as_or_throw<TensorVar>().shape()[i])));
    }
    if (is_read)
      Update(&read_buffers_, &read_regions_,
             region->source.as_or_throw<TensorVar>(), std::move(ranges));
    else
      Update(&write_buffers_, &write_regions_,
             region->source.as_or_throw<TensorVar>(), std::move(ranges));
  }

  TensorRegion ExpandPaddedCopyWriteRegion(const AscendCopy &copy,
                                           TensorRegion write) {
    if (copy->data_select == 0 || !IsGlobalBuffer(copy->src) ||
        !IsSharedBuffer(copy->dst) || write->region.empty()) {
      return write;
    }

    // DMA rows follow the coalesced source/destination layouts, not the last
    // logical buffer axis. Use the same coalescing and unit-axis normalization
    // as PlanMTECopy, then select the row boundary from the strided side.
    sym::Analyzer analyzer;
    auto src_range = NormalizeEmptyUnitAxesForMTE(copy->src_range, analyzer);
    auto dst_range = NormalizeEmptyUnitAxesForMTE(copy->dst_range, analyzer);
    int src_bits = copy->src.dtype().bits() * copy->src.dtype().lanes();
    int elem_bits = write->source.as_or_throw<TensorVar>().dtype().bits() *
                    write->source.as_or_throw<TensorVar>().dtype().lanes();
    auto src = StridedLayout::FromBufferRange(copy->src, src_range, src_bits)
                   .Coalesce(analyzer);
    auto dst = StridedLayout::FromBufferRange(copy->dst, dst_range, elem_bits)
                   .Coalesce(analyzer);
    auto total_elems = [&](const StridedLayout &layout) {
      PrimExpr total = IntImm::Int32(1);
      for (const Mode &mode : layout.modes)
        total = total * mode.size;
      return total;
    };
    if (src_bits != elem_bits || src.modes.empty() || dst.modes.empty() ||
        src.modes.size() > 2 || dst.modes.size() > 2 ||
        !analyzer->CanProveEqual(total_elems(src), total_elems(dst)) ||
        (src.modes.size() == 2 && dst.modes.size() == 2 &&
         (!analyzer->CanProveEqual(src.modes[0].size, dst.modes[0].size) ||
          !analyzer->CanProveEqual(src.modes[1].size, dst.modes[1].size)))) {
      UpdateOpaqueAccessVar(write->source.as_or_throw<TensorVar>());
      return write;
    }
    PrimExpr row_elems = analyzer->Simplify(
        src.modes.size() == 2 ? src.modes[0].size : dst.modes[0].size);
    const auto *row_extent = row_elems.as<IntImmNode>();
    if (row_extent == nullptr) {
      UpdateOpaqueAccessVar(write->source.as_or_throw<TensorVar>());
      return write;
    }

    if (elem_bits <= 0 || 256 % elem_bits != 0 ||
        row_extent->value.as<int64_t>().value() <= 0 ||
        row_extent->value.as<int64_t>().value() >
            (std::numeric_limits<int64_t>::max() - 255) / elem_bits) {
      UpdateOpaqueAccessVar(write->source.as_or_throw<TensorVar>());
      return write;
    }
    int64_t row_bits = row_extent->value.as<int64_t>().value() * elem_bits;
    if (row_bits % 8 != 0) {
      UpdateOpaqueAccessVar(write->source.as_or_throw<TensorVar>());
      return write;
    }
    int64_t padded_extent = ((row_bits + 255) / 256) * 256 / elem_bits;
    if (padded_extent == row_extent->value.as<int64_t>().value())
      return write;

    ffi::Array<Range> write_region = write->region;
    const Range &row_range = write_region.back();
    auto geometry = tirx::MatchShapeStridePattern(
        write->source.as_or_throw<TensorVar>().ty());
    if (!geometry || geometry.value().get<1>().empty()) {
      UpdateOpaqueAccessVar(write->source.as_or_throw<TensorVar>());
      return write;
    }
    PrimExpr last_stride = geometry.value().get<1>().back();
    // Expanding the last logical axis is exact only when it is the descriptor's
    // contiguous row. In a view with stride=16, for example, row_elems=1 and
    // its seven padded floats occupy gaps between view elements, not extra
    // logical rows. Those physical lanes have no region in this view.
    if (!analyzer->CanProveEqual(last_stride, IntImm::Int32(1)) ||
        !analyzer->CanProveEqual(row_range->extent, row_elems)) {
      UpdateOpaqueAccessVar(write->source.as_or_throw<TensorVar>());
      return write;
    }

    PrimExpr physical_extent = IntImm(row_range->extent.ty(), padded_extent);
    // Lifetime analysis may reject reuse, but must not add an InsertSync
    // failure. Keep a finite footprint when the offset is symbolic.
    if (analyzer->CanProve(
            row_range->begin + physical_extent >
            write->source.as_or_throw<TensorVar>().shape().back()))
      UpdateOpaqueAccessVar(write->source.as_or_throw<TensorVar>());
    write_region.Set(write_region.size() - 1,
                     Range::FromMinExtent(row_range->begin, physical_extent));
    return NativeTensorRegion(write->source.as_or_throw<TensorVar>(),
                              std::move(write_region));
  }

  /*! \brief Helper function to collect access regions. */
  std::vector<TensorRegion> CollectRegions(
      const std::vector<TensorVar> &buffers,
      const std::vector<std::vector<tvm::sym::IntSet>> &regions) const {
    std::vector<TensorRegion> result;
    result.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
      result.push_back(MakeTensorRegion(buffers[i], regions[i]));
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
  sym::IntSet RelaxAccessIndex(const PrimExpr &index,
                               PrimExpr extent = IntImm(PrimType::Int(32), 1)) {
    PrimExpr base = ResolveLets(index);
    sym::IntSet lo_set = sym::EvalSet(sym::IntSet::Vector(base), dom_map_);
    const auto *ext_imm = extent.as<IntImmNode>();
    if (ext_imm && ext_imm->value.as<int64_t>().value() <= 1) {
      return lo_set;
    }
    PrimExpr hi = base + ResolveLets(extent) - MakeConst(base.ty(), 1);
    sym::IntSet hi_set = sym::EvalSet(sym::IntSet::Vector(hi), dom_map_);
    return sym::IntSet::Interval(lo_set.min(), hi_set.max());
  }

  void VisitStmt_(const ForNode *op) override {
    Range range = ScheduleLoopRange(op);
    dom_map_.Set(op->var, sym::IntSet::FromRange(range));
    loop_domains_.emplace_back(op->var, range);
    PrimExpr step = ResolveLets(op->step);
    bool non_unit = !ana_->CanProve(step == 1 || step == -1);
    if (non_unit)
      ++non_unit_step_loop_depth_;
    AnalysisVisitor::VisitStmt_(op);
    if (non_unit)
      --non_unit_step_loop_depth_;
    loop_domains_.pop_back();
    dom_map_.erase(op->var);
  }

  void VisitStmt_(const IfNode *op) override {
    VisitExpr(op->condition);
    ++conditional_depth_;
    // The ordinary write regions remain may-write unions. Writes visited while
    // this depth is nonzero are excluded from the separate must-write list.
    AnalysisVisitor::VisitStmt(op->then_case);
    if (op->else_case) {
      // Visit else branch
      AnalysisVisitor::VisitStmt(op->else_case.value());
    }
    --conditional_depth_;
  }

  void VisitStmt_(const WhileNode *op) override {
    VisitExpr(op->condition);
    ++conditional_depth_;
    AnalysisVisitor::VisitStmt(op->body);
    --conditional_depth_;
  }

  void VisitStmt_(const BindNode *op) override {
    bool previous_context = opaque_pointer_context_;
    opaque_pointer_context_ |= op->var->ty.as<PtrTypeNode>() != nullptr;
    VisitExpr(op->value);
    opaque_pointer_context_ = previous_context;
    if (auto variable = op->var.as<PrimVar>()) {
      if (auto value = op->value.as<PrimExpr>())
        let_bindings_.Set(variable.value(), value.value());
    }
    UpdateWriteVar(op->var);
  }

  void VisitStmt_(const RegionStmtNode *op) override {
    auto key = ScheduleMarkerKey(op);
    if (!key.empty()) {
      if (IsScheduleGuardAttribute(key) || key == kScheduleCondition) {
        if (auto expression = ScheduleMarkerPayload(op).as<Expr>())
          VisitExpr(expression.value());
      }
      if (key == kScheduleCondition)
        ++conditional_depth_;
      VisitStmt(UnwrapScheduleMarker(op));
      if (key == kScheduleCondition)
        --conditional_depth_;
      return;
    }
    auto old_domains = dom_map_;
    size_t old_loops = loop_domains_.size();
    if ((op->op.same_as(tirx::launch_thread_op()) ||
         op->op.same_as(Op::Get("tl.launch_grid"))) &&
        op->args.size() == 2 && op->body_params.size() == 1) {
      auto variable = op->body_params[0].as_or_throw<PrimVar>();
      auto extent = op->args[1].as_or_throw<PrimExpr>();
      Range range = Range::FromMinExtent(MakeConst(extent.ty(), 0), extent);
      dom_map_.Set(variable, sym::IntSet::FromRange(range));
      loop_domains_.emplace_back(variable, range);
    }
    for (const Expr &argument : op->args)
      VisitExpr(argument);
    VisitStmt(op->body);
    dom_map_ = old_domains;
    loop_domains_.erase(loop_domains_.begin() + old_loops, loop_domains_.end());
    for (const Var &result : op->result_vars)
      UpdateWriteVar(result);
  }

  void VisitStmt_(const BreakNode *) override { has_loop_break_ = true; }
  void VisitStmt_(const ContinueNode *) override { has_loop_break_ = true; }
  void VisitStmt_(const ReturnNode *op) override {
    has_loop_break_ = true;
    AnalysisVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const VarNode *op) override {
    // Bare storage pointers have no bounded footprint. A structured access in
    // the same task must not hide their escape to an external consumer.
    if (op->ty.as<PtrTypeNode>() != nullptr ||
        op->ty.as<TensorTypeNode>() != nullptr)
      UpdateOpaqueAccessVar(ffi::GetRef<Var>(op));
    UpdateReadVar(tvm::ffi::GetRef<Var>(op));
    AnalysisVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const TensorLoadNode *op) override {
    std::vector<sym::IntSet> relaxed_region;
    size_t num_indices = op->indices.size();
    size_t buffer_ndim = op->source.as_or_throw<TensorVar>().shape().size();

    // Assert that indices count equals buffer dimension
    ICHECK_EQ(num_indices, buffer_ndim)
        << "TensorLoad indices count " << num_indices
        << " must equal buffer dimension " << buffer_ndim;

    for (PrimExpr index : op->indices) {
      relaxed_region.push_back(RelaxAccessIndex(index));
    }
    Update(&read_buffers_, &read_regions_, op->source.as_or_throw<TensorVar>(),
           relaxed_region);
    Visit(op->source->ty);
    for (const PrimExpr &index : op->indices)
      VisitExpr(index);
  }

  void VisitStmt_(const TensorStoreNode *op) override {
    std::vector<sym::IntSet> relaxed_region;
    std::vector<PrimExpr> mins;
    std::vector<PrimExpr> extents;
    size_t num_indices = op->indices.size();
    size_t buffer_ndim = op->dest.as_or_throw<TensorVar>().shape().size();

    // Assert that indices count equals buffer dimension
    ICHECK_EQ(num_indices, buffer_ndim)
        << "TensorStore indices count " << num_indices
        << " must equal buffer dimension " << buffer_ndim;

    relaxed_region.reserve(num_indices);
    mins.reserve(num_indices);
    extents.reserve(num_indices);
    for (PrimExpr index : op->indices) {
      relaxed_region.push_back(RelaxAccessIndex(index));
      mins.push_back(index);
      extents.push_back(MakeConst(index.ty(), 1));
    }
    bool must_execute = true;
    UpdateWrite(op->dest.as_or_throw<TensorVar>(), std::move(relaxed_region),
                mins, extents, must_execute);
    Visit(op->dest->ty);
    for (const PrimExpr &index : op->indices)
      VisitExpr(index);
    VisitExpr(op->value);
  }

  void VisitTensorDeclarationPointer(const Expr &pointer) {
    if (auto variable = pointer.as<Var>()) {
      UpdateReadVar(variable.value());
      Visit(variable.value()->ty);
      return;
    }
    if (const auto *call = pointer.as<CallNode>()) {
      if (call->op.same_as(tirx::tensor_data_ptr_op()) &&
          call->args.size() == 1) {
        VisitTensorDeclarationPointer(call->args[0]);
        return;
      }
      if (call->op.same_as(tirx::ptr_byte_offset_op()) &&
          call->args.size() == 2) {
        VisitTensorDeclarationPointer(call->args[0]);
        VisitExpr(call->args[1]);
        return;
      }
      if (call->op.same_as(tirx::address_of_op()) && call->args.size() == 1) {
        if (const auto *load = call->args[0].as<TensorLoadNode>()) {
          VisitTensorDeclarationPointer(load->source);
          for (const PrimExpr &index : load->indices)
            VisitExpr(index);
          return;
        }
      }
    }
    VisitExpr(pointer);
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

  void VisitPhysicalPointer(const Expr &pointer, std::optional<int64_t> bytes,
                            int mask) {
    if (const auto *call = pointer.as<CallNode>();
        call && call->op.same_as(tl::access_ptr()) && call->args.size() >= 3) {
      if (const auto *load = call->args[0].as<TensorLoadNode>()) {
        PrimType dtype = load->source.as_or_throw<TensorVar>().dtype();
        PrimExpr extent =
            ana_->Simplify(ResolveLets(call->args[1].as_or_throw<PrimExpr>()));
        const auto *declared_extent = extent.as<IntImmNode>();
        const auto *declared_mask = call->args[2].as<IntImmNode>();
        if (bytes && *bytes > 0 && dtype.IsScalar() && dtype.bits() >= 8 &&
            dtype.bits() % 8 == 0 && declared_extent && declared_mask) {
          int64_t elements = (*bytes - 1) / (dtype.bits() / 8) + 1;
          // Keep the producer's conservative region, including padding. A
          // smaller physical span must not narrow its logical-view contract.
          if (elements <= std::numeric_limits<int32_t>::max() &&
              declared_extent->value.as<int64_t>().value() >= elements &&
              (declared_mask->value.as<int64_t>().value() & mask) == mask) {
            VisitExpr(pointer);
            return;
          }
        }
        UpdateOpaqueAccessVar(load->source.as_or_throw<TensorVar>());
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
      if (rows && cols && rows->value.as<int64_t>().value() > 0 &&
          cols->value.as<int64_t>().value() > 0 &&
          rows->value.as<int64_t>().value() <
              std::numeric_limits<int64_t>::max()) {
        int64_t dst_bytes = element_bytes(scatter ? 4 : 5);
        if (scatter) {
          int64_t src_bytes = element_bytes(5);
          auto source_row =
              PhysicalSpan(cols->value.as<int64_t>().value(), src_bytes, 0);
          auto destination_row =
              PhysicalSpan(cols->value.as<int64_t>().value(), dst_bytes, 0);
          if (src_bytes && dst_bytes && source_row && destination_row) {
            // The scatter template loads whole 256-byte vectors, including
            // the last partial vector of the last source row.
            int64_t vectors = (*source_row - 1) / 256 + 1;
            if (auto last_row = PhysicalSpan(vectors, 256, 0))
              source_bytes = PhysicalSpan(rows->value.as<int64_t>().value() - 1,
                                          *source_row, *last_row);
            destination_bytes = PhysicalSpan(
                rows->value.as<int64_t>().value() + 1, *destination_row, 0);
          }
        } else {
          const auto *full_rows = op->args[4].as<IntImmNode>();
          if (dst_bytes && full_rows &&
              full_rows->value.as<int64_t>().value() >=
                  rows->value.as<int64_t>().value() &&
              cols->value.as<int64_t>().value() % (32 / dst_bytes) == 0) {
            int64_t bursts =
                cols->value.as<int64_t>().value() / (32 / dst_bytes);
            // Match EmitNd2NzPostCopy_: 32-byte DMA blocks, one source-gap
            // row, and full_rows - rows destination-gap rows per burst.
            if (auto blocks = PhysicalSpan(
                    bursts - 1, rows->value.as<int64_t>().value() + 1,
                    rows->value.as<int64_t>().value()))
              source_bytes = PhysicalSpan(*blocks, 32, 0);
            if (auto blocks = PhysicalSpan(
                    bursts - 1, full_rows->value.as<int64_t>().value(),
                    rows->value.as<int64_t>().value()))
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
    for (const auto &kv : AscendCallAnnotations(op)) {
      if (auto expr = kv.second.as<PrimExpr>())
        VisitExpr(expr.value());
    }
  }

  std::vector<sym::IntSet> LogicalPointerRegion(const TensorLoad &load,
                                                const PrimExpr &extent) {
    const TensorVar &buffer = load->source.as_or_throw<TensorVar>();
    size_t ndim = buffer.shape().size();
    std::vector<sym::IntSet> region(ndim, sym::IntSet::Nothing());

    const auto *ext_imm = extent.as<IntImmNode>();
    if (ext_imm && load->indices.size() == ndim) {
      // Distribute a known flat extent (in elements) across dims,
      // innermost-first. `rem` is the number of elements still to be
      // covered at the current dimension; each dim consumes as many of
      // its `shape[i]` indices as the run spans, carrying the rounded-up
      // remainder to the next (outer) dim.
      int64_t rem = ext_imm->value.as<int64_t>().value();
      for (size_t k = ndim; k-- > 0;) {
        PrimExpr base = load->indices[k];
        const auto *dim_imm = buffer.shape()[k].as<IntImmNode>();
        if (rem <= 1) {
          region[k] = RelaxAccessIndex(base);
        } else if (dim_imm) {
          int64_t d = dim_imm->value.as<int64_t>().value();
          if (rem >= d) {
            // Run spills past this dim: cover it whole and carry up.
            region[k] = sym::IntSet::FromRange(
                Range::FromMinExtent(PrimExpr(0), buffer.shape()[k]));
            rem = (rem + d - 1) / d;
          } else {
            region[k] = RelaxAccessIndex(base, IntImm(base.ty(), rem));
            rem = 1;
          }
        } else {
          // Non-constant dim size: be conservative for this and any
          // remaining outer dims.
          region[k] = sym::IntSet::FromRange(
              Range::FromMinExtent(PrimExpr(0), buffer.shape()[k]));
          rem = 1;
        }
      }
    } else {
      // Non-constant extent or arity mismatch: fall back to whole buffer.
      for (size_t k = 0; k < ndim; ++k) {
        region[k] = sym::IntSet::FromRange(
            Range::FromMinExtent(PrimExpr(0), buffer.shape()[k]));
      }
    }
    return region;
  }

  std::vector<sym::IntSet> PhysicalPointerRegion(const TensorLoad &load,
                                                 const PrimExpr &extent) {
    const TensorVar &buffer = load->source.as_or_throw<TensorVar>();
    size_t ndim = buffer.shape().size();
    std::vector<sym::IntSet> region(ndim, sym::IntSet::Nothing());

    bool reconstructed = false;
    const auto *ext_imm = extent.as<IntImmNode>();
    // Explicit strides, including dense T.view layouts, stay opaque
    // for multi-element runs until their physical image is proved.
    if (ext_imm && ext_imm->value.as<int64_t>().value() > 0 &&
        load->indices.size() == ndim &&
        (ext_imm->value.as<int64_t>().value() == 1 ||
         HasContiguousTaskLayout(buffer, ana_))) {
      int64_t remaining = ext_imm->value.as<int64_t>().value();
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

        auto dimension = GetConstInt(buffer.shape()[k]);
        sym::IntSet base_set = RelaxAccessIndex(base);
        auto base_min = GetConstInt(base_set.min());
        auto base_max = GetConstInt(base_set.max());
        if (!dimension || *dimension <= 0 || !base_min || !base_max ||
            *base_min < 0 || *base_max >= *dimension ||
            *base_max > std::numeric_limits<int64_t>::max() - remaining) {
          reconstructed = false;
          break;
        }

        int64_t end_exclusive = *base_max + remaining;
        int64_t span =
            end_exclusive / *dimension + (end_exclusive % *dimension != 0);
        if (span == 1) {
          region[k] = RelaxAccessIndex(base, IntImm(base.ty(), remaining));
          remaining = 1;
        } else {
          // Wrapping this dimension can touch both its high and low
          // coordinates, so its rectangular hull is the whole axis.
          region[k] = sym::IntSet::FromRange(
              Range::FromMinExtent(PrimExpr(0), buffer.shape()[k]));
          remaining = span;
        }
      }
      // A remaining carry escapes the outermost logical dimension, even
      // if the backing allocation has room for the physical access.
      reconstructed &= remaining == 1;
    }
    if (!reconstructed) {
      UpdateOpaqueAccessVar(buffer);
      // The storage is opaque: this placeholder cannot prove reuse and
      // is never passed to synchronization dependency analysis.
      for (size_t k = 0; k < ndim; ++k) {
        region[k] = sym::IntSet::FromRange(
            Range::FromMinExtent(PrimExpr(0), buffer.shape()[k]));
      }
    }
    return region;
  }

  void VisitExpr_(const CallNode *op) override {
    static const Op &region_op = region();
    static const auto tl_access_ptr_op = Op::Get("tl.access_ptr");

    if (op->op.same_as(Op::Get("tl.layout_hint"))) {
      for (const Expr &argument : op->args)
        VisitTensorDeclarationPointer(argument);
      return;
    }

    if (op->op.same_as(tirx::alloc_tensor_op()) ||
        op->op.same_as(tirx::decl_tensor_op())) {
      Visit(op->ty);
      for (const Expr &argument : op->args) {
        if (op->op.same_as(tirx::decl_tensor_op()))
          VisitTensorDeclarationPointer(argument);
        else
          VisitExpr(argument);
      }
      return;
    }

    if (mode_ == AccessMode::kLogical &&
        op->op.same_as(tl::ascend_nd2nz_post_copy()) && op->args.size() == 6) {
      const auto *pointer = op->args[0].as<CallNode>();
      const auto *load = pointer && pointer->op.same_as(tl_access_ptr_op) &&
                                 pointer->args.size() == 3
                             ? pointer->args[0].as<TensorLoadNode>()
                             : nullptr;
      if (load && load->indices.size() >= 2 &&
          load->indices.size() ==
              load->source.as_or_throw<TensorVar>().shape().size()) {
        // The DMA parameters retain the rectangular destination even when
        // rows is dynamic. A flat access_ptr extent would lose the two AIVs'
        // disjoint row prefixes and create a spurious cross-AIV WAW edge.
        Array<Range> ranges;
        size_t ndim = load->indices.size();
        for (size_t i = 0; i < ndim; ++i) {
          PrimExpr extent = i + 2 == ndim ? op->args[2].as_or_throw<PrimExpr>()
                            : i + 1 == ndim
                                ? op->args[3].as_or_throw<PrimExpr>()
                                : IntImm::Int32(1);
          ranges.push_back(Range::FromMinExtent(load->indices[i], extent));
          VisitExpr(load->indices[i]);
        }
        RecordTensorRegion(
            NativeTensorRegion(load->source.as_or_throw<TensorVar>(), ranges),
            false);
        for (size_t i = 1; i < op->args.size(); ++i)
          VisitExpr(op->args[i]);
        for (const auto &[key, value] : AscendCallAnnotations(op))
          if (auto expr = value.as<PrimExpr>())
            VisitExpr(expr.value());
        return;
      }
    }

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
        AnalysisVisitor::VisitExpr_(op);
        opaque_pointer_context_ = previous_context;
        return;
      }
    }

    if (op->op.same_as(tl::loop_break()))
      has_loop_break_ = true;

    if (auto operation = ParseOperator(ffi::GetRef<Call>(op));
        operation.defined()) {
      auto saved_pitches = operation_pitches_;
      if (mode_ == AccessMode::kLogical)
        operation_pitches_ = GetCallL0Pitches(ffi::GetRef<Call>(op), layouts_);
      ffi::Optional<AscendCopy> copy;
      ffi::Optional<Gemm> gemm;
      if (IsAscendCopyCall(op))
        copy = AscendCopy(op->args, AscendCallAnnotations(op));
      if (op->op.same_as(Gemm::Get()) || op->op.same_as(GemmBlockScaled::Get()))
        gemm = operation.as_or_throw<Gemm>();
      auto record = [&](TensorRegion region, bool read) {
        if (mode_ == AccessMode::kPhysical && copy && !read)
          region = ExpandPaddedCopyWriteRegion(copy.value(), region);
        RecordTensorRegion(region, read);
        for (const Range &range : region->region) {
          VisitExpr(range->begin);
          VisitExpr(range->extent);
        }
        if (mode_ != AccessMode::kPhysical)
          return;
        auto tensor = region->source.as_or_throw<TensorVar>();
        bool l0 =
            IsL0ABuffer(tensor) || IsL0BBuffer(tensor) || IsL0CBuffer(tensor);
        if (!l0 || region->region.size() < 2)
          return;
        bool compact = copy && copy.value()->l0_pitch.has_value();
        if (copy) {
          for (size_t axis = region->region.size() - 2;
               axis < region->region.size(); ++axis)
            compact |= !ana_->CanProveEqual(region->region[axis]->extent,
                                            tensor.shape()[axis]);
        } else if (gemm) {
          size_t axis = tensor.shape().size() - 2;
          if (tensor.same_as(gemm.value()->a_))
            axis += gemm.value()->transA_;
          if (tensor.same_as(gemm.value()->b_))
            axis += !gemm.value()->transB_;
          compact |= !ana_->CanProveEqual(
              floordiv(region->region[axis]->extent + 15, 16),
              floordiv(tensor.shape()[axis] + 15, 16));
        }
        RecordL0MayAccess(region, read, compact);
      };
      auto accesses = operation->GetAccessRegions();
      for (const TensorRegion &region : accesses.reads)
        record(region, true);
      for (const TensorRegion &region : accesses.writes)
        record(region, false);
      for (const Expr &argument : op->args) {
        if (argument.as<TensorRegionNode>() || argument.as<TensorVar>())
          continue;
        if (const auto *descriptor = argument.as<CallNode>();
            descriptor && descriptor->op.same_as(region_op))
          continue;
        VisitExpr(argument);
      }
      for (const auto &[key, value] : AscendCallAnnotations(op))
        if (auto expression = value.as<Expr>())
          VisitExpr(expression.value());
      operation_pitches_ = saved_pitches;
      return;
    }

    if (op->op.same_as(region_op)) {
      auto access = NormalizeToAccessRegion(ffi::GetRef<Call>(op));
      if (access.access_mask & kAccessRead)
        RecordTensorRegion(access.region, true);
      if (access.access_mask & kAccessWrite)
        RecordTensorRegion(access.region, false);
      for (const Range &range : access.region->region) {
        VisitExpr(range->begin);
        VisitExpr(range->extent);
      }
      return;
    }

    if (op->op.same_as(tirx::address_of_op()) ||
        op->op.same_as(tirx::tensor_data_ptr_op())) {
      if (auto tensor = op->args[0].as<TensorVar>())
        UpdateOpaqueAccessVar(tensor.value());
      if (const auto *load = op->args[0].as<TensorLoadNode>())
        UpdateOpaqueAccessVar(load->source.as_or_throw<TensorVar>());
      AnalysisVisitor::VisitExpr_(op);
      return;
    }

    // Handle TileLang `tl.access_ptr(TensorLoad(buf, mins), extent, rw_mask)`.
    // The access is a contiguous run of `extent` elements in row-major layout
    // starting at the base index `mins`. Reconstruct a conservative rectangular
    // hull by carrying both the extent and each dimension's starting offset. If
    // the physical footprint cannot be represented, exclude the storage from
    // alias proofs. A whole logical view can omit physical storage elements.
    if (op->op.same_as(tl_access_ptr_op)) {
      if (op->args.size() >= 3) {
        const auto *buffer_load = op->args[0].as<TensorLoadNode>();
        if (buffer_load) {
          TensorVar buffer = buffer_load->source.as_or_throw<TensorVar>();
          if (opaque_pointer_context_)
            UpdateOpaqueAccessVar(buffer);
          int rw_mask =
              GetConservativeAccessMask(op->args[2].as_or_throw<PrimExpr>());
          TensorLoad load = ffi::GetRef<TensorLoad>(buffer_load);
          std::vector<sym::IntSet> region =
              mode_ == AccessMode::kLogical
                  ? LogicalPointerRegion(load,
                                         op->args[1].as_or_throw<PrimExpr>())
                  : PhysicalPointerRegion(load,
                                          op->args[1].as_or_throw<PrimExpr>());

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
          VisitExpr(op->args[2]);
          return;
        }
      }
      AnalysisVisitor::VisitExpr_(op);
      return;
    }

    // A generic pointer-returning call may advance or reinterpret its input
    // pointer even when no Bind or handle buffer carries the result. The
    // bounded access_ptr and address_of cases above keep their own semantics.
    bool previous_context = opaque_pointer_context_;
    opaque_pointer_context_ |= op->ty.as<PtrTypeNode>() != nullptr;
    if (op->op.same_as(tirx::reinterpret_op()))
      for (const Expr &argument : op->args)
        opaque_pointer_context_ |= argument->ty.as<PtrTypeNode>() != nullptr;
    AnalysisVisitor::VisitExpr_(op);
    opaque_pointer_context_ = previous_context;
  }
};

class MemoryAccessDetector {
public:
  using AccessMode = MemoryAccessDetectorImpl::AccessMode;
  explicit MemoryAccessDetector(AccessMode mode = AccessMode::kLogical,
                                LayoutMap layouts = {})
      : detector_(ffi::make_object<MemoryAccessDetectorImpl>(
            mode, std::move(layouts))) {}

  void Analyze(const Stmt &statement) { detector_->Analyze(statement); }
  void Analyze(const Expr &expression) { detector_->Analyze(expression); }
  std::vector<TensorRegion> GetReadRegions() const {
    return detector_->GetReadRegions();
  }
  std::vector<TensorRegion> GetWriteRegions() const {
    return detector_->GetWriteRegions();
  }
  const Map<Var, Optional<PrimExpr>> &GetL0Pitches() const {
    return detector_->GetL0Pitches();
  }
  std::vector<Var> GetReadVars() const { return detector_->GetReadVars(); }
  std::vector<Var> GetWriteVars() const { return detector_->GetWriteVars(); }
  std::vector<Var> GetOpaqueAccessVars() const {
    return detector_->GetOpaqueAccessVars();
  }
  std::vector<TensorRegion> GetMustWriteRegions() const {
    return detector_->GetMustWriteRegions();
  }

private:
  ffi::ObjectPtr<MemoryAccessDetectorImpl> detector_;
};

} // namespace ascend
} // namespace tl
} // namespace tvm
