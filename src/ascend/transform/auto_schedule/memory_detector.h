#include "ascend/transform/source_scopes.h"
#include "ir/annotations.h"
#include "ir/loop_math.h"
#include "layout/expr_utils.h"
#include "op/builtin.h"
#include "transform/common/constr_visitor.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/op/region.h>
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#pragma once

#include "ascend/analysis/storage.h"
#include "ascend/layout/ascend_layouts.h"
#include "ascend/op/ascend_mte_plan.h"
#include "ascend/op/builtin.h"
#include "ascend/op/copy.h"
#include "ascend/op/oob_padding.h"
#include "ascend/op/utils.h"
#include "ir_structure.h"
#include "op/gemm.h"
#include "op/gemm_blockscaled.h"
#include "op/operator.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <tvm/sym/int_set.h>
#include <tvm/sym/iter_affine_map.h>
#include <tvm/tirx/stmt_functor.h>
#include <unordered_map>
#include <vector>

namespace tvm::tl::ascend {

using namespace tirx;
using namespace prim;

inline int64_t TaskImmediateValue(const IntImmObj *integer) {
  TVM_FFI_CHECK(integer != nullptr, ValueError)
      << "Expected a constant footprint integer";
  auto value = integer->value.as<int64_t>();
  TVM_FFI_CHECK(value.has_value(), ValueError)
      << "Task footprint integer is not representable as int64";
  return value.value();
}

inline ScheduleUnitMetadata TaskCallAnnotations(const CallObj *call) {
  if (!call->attrs)
    return {};
  auto attrs = call->attrs.value().as_or_throw<DictAttrs>();
  if (attrs.GetAttr<ffi::Array<ffi::Any>>("annotation_schema"))
    return DecodeTileOpArguments(call).annotations;
  ScheduleUnitMetadata values;
  for (const auto &[key, value] : attrs->dict)
    if (key != "tl.source_annotation_keys")
      values.Set(key, value);
  return values;
}

inline ffi::Array<Expr> TaskCallOperands(const CallObj *call) {
  if (call->attrs) {
    auto attrs = call->attrs.value().as_or_throw<DictAttrs>();
    if (attrs.GetAttr<ffi::Array<ffi::Any>>("annotation_schema"))
      return DecodeTileOpArguments(call).operands;
  }
  return call->args;
}

inline void MergeL0Pitch(Map<Var, Optional<PrimExpr>> *pitches,
                         const Var &storage, Optional<PrimExpr> pitch) {
  auto it = pitches->find(storage);
  if (it != pitches->end()) {
    sym::Analyzer analyzer;
    if (!(*it).second || !pitch ||
        !analyzer->CanProveEqual(*(*it).second, *pitch))
      pitch = std::nullopt;
  }
  pitches->Set(storage, pitch);
}

inline Map<Var, PrimExpr> GetCallL0Pitches(const Call &call,
                                           const LayoutMap &layouts,
                                           const StorageProvenance &storage) {
  Map<Var, PrimExpr> result;
  auto record = [&](const TensorRegion &region, size_t mn_axis,
                    Optional<PrimExpr> l0_pitch = std::nullopt) {
    PrimExpr pitch =
        l0_pitch.value_or(FloorDiv(region->region[mn_axis]->extent + 15, 16));
    result.Set(storage.Identity(RegionTensor(region)), pitch);
  };
  if (IsAscendCopyCall(call.get())) {
    auto decoded = DecodeTileOpArguments(call.get());
    AscendCopy copy(decoded.operands, decoded.annotations);
    if (IsL1Buffer(copy->src) &&
        (IsL0ABuffer(copy->dst) || IsL0BBuffer(copy->dst))) {
      auto layout = FindLayoutForBuffer(layouts, copy->dst);
      AscendFractalLayoutInfo info;
      if (layout && TryExtractAscendFractalLayout(*layout, copy->dst, &info) &&
          IsAscendMajorKind(info.kind))
        record(TensorRegion(copy->dst, copy->dst_range, TensorRegionType()),
               copy->dst_range.size() - 2 +
                   (info.row16_axis == LogicalAxis::kCol),
               copy->l0_pitch);
    } else if (IsL0CBuffer(copy->src)) {
      record(TensorRegion(copy->src, copy->src_range, TensorRegionType()),
             copy->src_range.size() - 2, copy->l0_pitch);
    }
  } else if (call->op.same_as(Gemm::Get()) ||
             call->op.same_as(GemmBlockScaled::Get())) {
    Gemm gemm = ParseOperator(call).as_or_throw<Gemm>();
    if (IsL0ABuffer(gemm->a_) && IsL0BBuffer(gemm->b_)) {
      record(gemm->aRegion_, gemm->a_.shape().size() - (gemm->transA_ ? 1 : 2));
      record(gemm->bRegion_, gemm->b_.shape().size() - (gemm->transB_ ? 2 : 1));
      record(gemm->cRegion_, gemm->c_.shape().size() - 2);
    }
  }
  return result;
}

class MemoryAccessDetector : public tirx::StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  using Result = ffi::Optional<ffi::VisitInterrupt>;
  enum class AccessMode { kLogical, kPhysical };
  explicit MemoryAccessDetector(
      AccessMode mode = AccessMode::kLogical, LayoutMap layouts = {},
      StorageProvenance storage = MakeAscendStorageProvenance(),
      ConstrSet context = {})
      : layouts_(std::move(layouts)), mode_(mode), storage_(std::move(storage)),
        context_(std::move(context)) {}

  void Analyze(const Stmt &stmt) {
    Reset();
    storage_.Collect(stmt);
    Visit(stmt);
  }
  void Analyze(const Expr &expr) {
    Reset();
    Visit(expr);
  }
  Var StorageIdentity(const TensorVar &tensor) const {
    return storage_.Identity(tensor);
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

  /*! \brief Unresolved conditions within current scope. */

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
  std::vector<std::vector<sym::IntSet>> read_regions_;
  /*! \brief The write regions of the current block */
  std::vector<std::vector<sym::IntSet>> write_regions_;
  AccessMode mode_;
  /*!\ brief Internal analyzer. */
  sym::Analyzer ana_;
  StorageProvenance storage_ = MakeAscendStorageProvenance();
  ConstrSet context_;
  std::unique_ptr<ConstrAnalyzerContext> proof_context_;
  /*! \brief let bindings inside the block */
  ffi::Map<Var, PrimExpr> let_bindings_;

  /*! \brief The set of variables that are read in the current block.  */
  std::vector<Var> read_vars_;
  /*! \brief The set of variables that are written in the current block.  */
  std::vector<Var> write_vars_;
  /*! \brief TensorVar storage Vars with an access of unknown region. */
  std::vector<Var> opaque_access_vars_;

  void Reset() {
    proof_context_.reset();
    ana_ = sym::Analyzer();
    proof_context_ = context_.Populate(ana_);
    l0_pitches_.clear();
    operation_pitches_.clear();
    read_buffers_.clear();
    write_buffers_.clear();
    read_regions_.clear();
    write_regions_.clear();
    dom_map_.clear();
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
  }

  void VisitExpr(const Expr &value) { Visit(value); }
  void VisitStmt(const Stmt &value) { Visit(value); }

  ffi::Array<PrimExpr> TensorStrides(const TensorVar &tensor) const {
    auto pattern = MatchShapeStridePattern(tensor.ty());
    TVM_FFI_CHECK(pattern.has_value(), ValueError)
        << "Expected a strided native tensor for this MTE footprint";
    return pattern.value().get<1>();
  }

  /*!
   * \brief Update read/write buffers and regions with provided buffer and
   * region
   */
  void Update(std::vector<TensorVar> *buffers,
              std::vector<std::vector<sym::IntSet>> *regions, TensorVar buffer,
              std::vector<sym::IntSet> region) {
    UpdateReadVar(buffer);
    UpdateReadVar(StorageIdentity(buffer));
    // Equal logical indices do not prove equal physical addresses when layout
    // metadata reads mutable state. Lowering uses these expressions at
    // accesses, so retain their reads as well as excluding the viewed storage
    // from reuse.
    auto visit_mutable_layout = [&](const PrimExpr &expr) {
      if (mode_ == AccessMode::kPhysical &&
          SideEffect(expr) > CallEffectKind::kPure) {
        UpdateOpaqueAccessVar(StorageIdentity(buffer));
        VisitExpr(expr);
      }
    };
    for (const PrimExpr &extent : buffer.shape())
      visit_mutable_layout(extent);
    if (auto pattern = MatchShapeStridePattern(buffer.ty())) {
      for (const PrimExpr &stride : pattern.value().get<1>())
        visit_mutable_layout(stride);
      visit_mutable_layout(pattern.value().get<2>());
    } else if (mode_ == AccessMode::kPhysical) {
      UpdateOpaqueAccessVar(StorageIdentity(buffer));
    }
    auto address = storage_.Address(buffer);
    if (!address.exact)
      UpdateOpaqueAccessVar(StorageIdentity(buffer));
    visit_mutable_layout(address.byte_offset);
    if (mode_ == AccessMode::kLogical &&
        (IsL0ABuffer(buffer) || IsL0BBuffer(buffer) || IsL0CBuffer(buffer))) {
      Optional<PrimExpr> pitch =
          operation_pitches_.Get(StorageIdentity(buffer));
      if (pitch.has_value()) {
        PrimExpr value = ana_->Simplify(ResolveLets(pitch.value()));
        if (SideEffect(value) > CallEffectKind::kPure ||
            ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
                value,
                [&](const VarObj *var) -> ffi::Expected<ffi::WalkResult> {
                  return (dom_map_.count(ffi::GetRef<Var>(var)) != 0)
                             ? ffi::WalkResult::Interrupt(
                                   ffi::VisitInterrupt(ffi::GetRef<Var>(var)))
                             : ffi::WalkResult::Skip();
                })
                .has_value())
          pitch = std::nullopt;
        else
          pitch = value;
      }
      MergeL0Pitch(&l0_pitches_, StorageIdentity(buffer), pitch);
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
    Region region;
    size_t ndim = buffer.shape().size();
    ICHECK_EQ(int_sets.size(), ndim) << "Region dimension " << int_sets.size()
                                     << " must equal buffer dimension " << ndim;
    region.reserve(ndim);
    for (size_t i = 0; i < ndim; ++i) {
      region.push_back(
          int_sets[i]
              .CoverRange(Range::FromMinExtent(0, buffer.shape()[i]))
              .value());
    }
    return TensorRegion(buffer, std::move(region), TensorRegionType());
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
      return ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
                 expr,
                 [&](const VarObj *node) -> ffi::Expected<ffi::WalkResult> {
                   Var var = ffi::GetRef<Var>(node);
                   for (const auto &[loop_var, _] : loop_domains_) {
                     if (var.same_as(loop_var))
                       return (true) ? ffi::WalkResult::Interrupt(
                                           ffi::VisitInterrupt(
                                               ffi::GetRef<Var>(node)))
                                     : ffi::WalkResult::Skip();
                   }
                   return ffi::WalkResult::Skip();
                 })
          .has_value();
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
        !iter_map->padding_predicate.has_value() ||
        !ana_->CanProve(Not(iter_map->padding_predicate.value()))) {
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
    TensorVar buffer = RegionTensor(buffer_region);
    const Region &region = buffer_region->region;
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
    UpdateOpaqueAccessVar(StorageIdentity(RegionTensor(region)));
    std::vector<sym::IntSet> ranges;
    for (size_t i = 0; i < region->region.size(); ++i) {
      const Range &range = region->region[i];
      ranges.push_back(i + 2 < region->region.size()
                           ? RelaxAccessIndex(range->begin, range->extent)
                           : sym::IntSet::FromRange(Range::FromMinExtent(
                                 0, RegionTensor(region).shape()[i])));
    }
    if (is_read)
      Update(&read_buffers_, &read_regions_, RegionTensor(region),
             std::move(ranges));
    else
      Update(&write_buffers_, &write_regions_, RegionTensor(region),
             std::move(ranges));
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
    auto src_range =
        NormalizeEmptyUnitAxesForMTE(copy->src_range, analyzer.get());
    auto dst_range =
        NormalizeEmptyUnitAxesForMTE(copy->dst_range, analyzer.get());
    int src_bits = copy->src.dtype().bits() * copy->src.dtype().lanes();
    int elem_bits = RegionTensor(write).dtype().bits() *
                    RegionTensor(write).dtype().lanes();
    auto src = StridedLayout::FromBufferRange(copy->src, src_range, src_bits)
                   .Coalesce(analyzer.get());
    auto dst = StridedLayout::FromBufferRange(copy->dst, dst_range, elem_bits)
                   .Coalesce(analyzer.get());
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
      UpdateOpaqueAccessVar(StorageIdentity(RegionTensor(write)));
      return write;
    }
    PrimExpr row_elems = analyzer->Simplify(
        src.modes.size() == 2 ? src.modes[0].size : dst.modes[0].size);
    const auto *row_extent = row_elems.as<IntImmObj>();
    if (row_extent == nullptr) {
      UpdateOpaqueAccessVar(StorageIdentity(RegionTensor(write)));
      return write;
    }

    if (elem_bits <= 0 || 256 % elem_bits != 0 ||
        TaskImmediateValue(row_extent) <= 0 ||
        TaskImmediateValue(row_extent) >
            (std::numeric_limits<int64_t>::max() - 255) / elem_bits) {
      UpdateOpaqueAccessVar(StorageIdentity(RegionTensor(write)));
      return write;
    }
    int64_t row_bits = TaskImmediateValue(row_extent) * elem_bits;
    if (row_bits % 8 != 0) {
      UpdateOpaqueAccessVar(StorageIdentity(RegionTensor(write)));
      return write;
    }
    int64_t padded_extent = ((row_bits + 255) / 256) * 256 / elem_bits;
    if (padded_extent == TaskImmediateValue(row_extent))
      return write;

    Region write_region = write->region;
    const Range &row_range = write_region.back();
    PrimExpr last_stride = !RegionTensor(write).ty()->layout.has_value()
                               ? IntImm::Int32(1)
                               : TensorStrides(RegionTensor(write)).back();
    // Expanding the last logical axis is exact only when it is the descriptor's
    // contiguous row. In a view with stride=16, for example, row_elems=1 and
    // its seven padded floats occupy gaps between view elements, not extra
    // logical rows. Those physical lanes have no region in this view.
    if (!analyzer->CanProveEqual(last_stride, IntImm::Int32(1)) ||
        !analyzer->CanProveEqual(row_range->extent, row_elems)) {
      UpdateOpaqueAccessVar(StorageIdentity(RegionTensor(write)));
      return write;
    }

    PrimExpr physical_extent = IntImm(row_range->extent.ty(), padded_extent);
    // Lifetime analysis may reject reuse, but must not add an InsertSync
    // failure. Keep a finite footprint when the offset is symbolic.
    if (analyzer->CanProve(row_range->begin + physical_extent >
                           RegionTensor(write).shape().back()))
      UpdateOpaqueAccessVar(StorageIdentity(RegionTensor(write)));
    write_region.Set(write_region.size() - 1,
                     Range::FromMinExtent(row_range->begin, physical_extent));
    return TensorRegion(RegionTensor(write), std::move(write_region),
                        TensorRegionType());
  }

  /*! \brief Helper function to collect access regions. */
  std::vector<TensorRegion>
  CollectRegions(const std::vector<TensorVar> &buffers,
                 const std::vector<std::vector<sym::IntSet>> &regions) const {
    std::vector<TensorRegion> result;
    result.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
      result.push_back(MakeTensorRegion(buffers[i], regions[i]));
    }
    return result;
  }

  /*! \brief Resolve Let bindings in an expression to a fixpoint. */
  PrimExpr ResolveLets(const PrimExpr &expression) {
    PrimExpr current = expression;
    std::unordered_set<const ExprObj *> seen;
    for (size_t count = 0; count <= let_bindings_.size(); ++count) {
      PrimExpr remapped = SubstituteConstraintExpr(current, let_bindings_);
      if (remapped.same_as(current))
        return current;
      current = std::move(remapped);
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
    const auto *ext_imm = extent.as<IntImmObj>();
    if (ext_imm && TaskImmediateValue(ext_imm) <= 1) {
      return lo_set;
    }
    PrimExpr hi = base + ResolveLets(extent) - MakeConst(base.ty(), 1);
    sym::IntSet hi_set = sym::EvalSet(sym::IntSet::Vector(hi), dom_map_);
    return sym::IntSet::Interval(lo_set.min(), hi_set.max());
  }

  Result Visit_(const SeqStmtObj *op) final {
    auto bindings = let_bindings_;
    for (const Stmt &statement : op->seq)
      Visit(statement);
    let_bindings_ = std::move(bindings);
    return std::nullopt;
  }

  Result Visit_(const ForObj *op) final {
    Visit(op->begin);
    Visit(op->end);
    Visit(op->step);
    PrimExpr begin = ResolveLets(op->begin), end = ResolveLets(op->end);
    PrimExpr step = ResolveLets(op->step);
    if (SideEffect(begin) <= CallEffectKind::kPure &&
        SideEffect(end) <= CallEffectKind::kPure &&
        ana_->CanProve(begin >= end))
      return std::nullopt;
    auto constant = TryConstInt(ana_->Simplify(step));
    Range range(begin, end);
    auto domains = dom_map_;
    // Entering negative/unknown-step loops can visit values below begin until
    // a Break. They have no finite descending range under the raw For test.
    bool known_range = CanUseLoopBounds(begin, end, step, ana_.get());
    dom_map_.Set(op->var, known_range ? sym::IntSet::FromRange(range)
                                      : sym::IntSet::Everything());
    loop_domains_.emplace_back(op->var, range);
    bool stepped = !known_range || !constant || *constant != 1;
    if (stepped)
      ++non_unit_step_loop_depth_;
    Visit(op->body);
    if (stepped)
      --non_unit_step_loop_depth_;
    loop_domains_.pop_back();
    dom_map_ = std::move(domains);
    return std::nullopt;
  }

  Result Visit_(const IfObj *op) final {
    Visit(op->condition);
    ++conditional_depth_;
    Visit(op->then_case);
    if (op->else_case)
      Visit(*op->else_case);
    --conditional_depth_;
    return std::nullopt;
  }

  Result Visit_(const BreakObj *) final {
    has_loop_break_ = true;
    return std::nullopt;
  }

  Result Visit_(const WhileObj *op) final {
    Visit(op->condition);
    ++conditional_depth_;
    Visit(op->body);
    --conditional_depth_;
    return std::nullopt;
  }

  Result Visit_(const BindObj *op) final {
    Expr value = op->value;
    if (const auto *call = value.as<CallObj>(); IsScheduledBind(call))
      value = ReadScheduledBinding(call).value;
    if (value->ty.as<PtrTypeObj>())
      VisitAddressInputs(value);
    else
      Visit(value);
    if (auto primitive = value.as<PrimExpr>()) {
      auto freshen = ffi::make_object<FreshenMutableReads>();
      let_bindings_.Set(op->var, (*freshen)(ResolveLets(*primitive)));
    }
    UpdateWriteVar(op->var);
    return std::nullopt;
  }

  Result Visit_(const RegionStmtObj *op) final {
    if (IsAscendBlock(op) || IsSemanticTask(op) || IsScheduleUnit(op) ||
        IsStageScope(op))
      return Visit(op->body);
    if (auto attribute = ReadSourceAttribute(op)) {
      if (IsScheduleGuardAttribute(attribute->key)) {
        if (auto value = attribute->node.try_cast<PrimExpr>())
          Visit(*value);
      }
      return Visit(op->body);
    }
    if (op->op.same_as(tirx::launch_thread_op())) {
      // Launch bounds are evaluated before the thread's body parameter is
      // defined. Projected subcores have distinct parameters, so their
      // accesses must be relaxed over each launch's own domain.
      Visit(op->attrs);
      for (const Expr &arg : op->args)
        Visit(arg);
      PrimExpr extent = ResolveLets(op->args[1].as_or_throw<PrimExpr>());
      bool known_range = SideEffect(extent) <= CallEffectKind::kPure;
      if (known_range && ana_->CanProve(extent <= 0))
        return std::nullopt;
      Var thread = op->body_params[0];
      Range range = Range::FromMinExtent(MakeConst(extent.ty(), 0), extent);
      auto domains = dom_map_;
      dom_map_.Set(thread, known_range ? sym::IntSet::FromRange(range)
                                       : sym::IntSet::Everything());
      // The same domains also guard must-write coverage: a symbolic empty
      // launch, strided writes and correlated axes cannot define a dense hull.
      loop_domains_.emplace_back(thread, range);
      if (!known_range)
        ++conditional_depth_;
      Visit(op->body);
      if (!known_range)
        --conditional_depth_;
      loop_domains_.pop_back();
      dom_map_ = std::move(domains);
      return std::nullopt;
    }
    return StmtExprVisitor::Visit_(op);
  }

  Result Visit_(const VarObj *op) final {
    Var variable = ffi::GetRef<Var>(op);
    if (op->ty.as<PtrTypeObj>()) {
      auto address = storage_.Address(variable);
      if (auto root = address.root.as<Var>()) {
        UpdateOpaqueAccessVar(root.value());
        UpdateReadVar(root.value());
      } else {
        UpdateOpaqueAccessVar(variable);
      }
    }
    UpdateReadVar(variable);
    return std::nullopt;
  }

  Result Visit_(const TensorRegionObj *op) final {
    // An address operand does not itself read or write its referenced memory.
    // Its consumer's registered access contract supplies that information.
    UpdateReadVar(op->source.as_or_throw<Var>());
    for (const Range &range : op->region) {
      Visit(range->begin);
      Visit(range->extent);
    }
    return std::nullopt;
  }

  Result Visit_(const TensorLoadObj *op) final {
    TensorVar tensor = op->source.as_or_throw<TensorVar>();
    TVM_FFI_CHECK_EQ(op->indices.size(), tensor.shape().size(), ValueError);
    std::vector<sym::IntSet> region;
    for (const auto &index : op->indices) {
      region.push_back(RelaxAccessIndex(index));
      Visit(index);
    }
    Update(&read_buffers_, &read_regions_, tensor, std::move(region));
    return std::nullopt;
  }

  Result Visit_(const TensorStoreObj *op) final {
    TensorVar tensor = op->dest.as_or_throw<TensorVar>();
    TVM_FFI_CHECK_EQ(op->indices.size(), tensor.shape().size(), ValueError);
    std::vector<sym::IntSet> region;
    std::vector<PrimExpr> mins, extents;
    for (const auto &index : op->indices) {
      region.push_back(RelaxAccessIndex(index));
      mins.push_back(index);
      extents.push_back(MakeConst(index.ty(), 1));
      Visit(index);
    }
    UpdateWrite(tensor, std::move(region), mins, extents);
    bool old_context = opaque_pointer_context_;
    opaque_pointer_context_ |= op->value->ty.as<PtrTypeObj>() != nullptr;
    Visit(op->value);
    opaque_pointer_context_ = old_context;
    return std::nullopt;
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
    if (const auto *call = pointer.as<CallObj>();
        call && call->op.same_as(tl::access_ptr()) && call->args.size() >= 3) {
      if (const auto *load = call->args[0].as<TensorLoadObj>()) {
        PrimType dtype = load->source.as_or_throw<TensorVar>().dtype();
        PrimExpr extent =
            ana_->Simplify(ResolveLets(call->args[1].as_or_throw<PrimExpr>()));
        const auto *declared_extent = extent.as<IntImmObj>();
        const auto *declared_mask = call->args[2].as<IntImmObj>();
        if (bytes && *bytes > 0 && dtype.lanes() == 1 && dtype.bits() >= 8 &&
            dtype.bits() % 8 == 0 && declared_extent && declared_mask) {
          int64_t elements = (*bytes - 1) / (dtype.bits() / 8) + 1;
          // Keep the producer's conservative region, including padding. A
          // smaller physical span must not narrow its logical-view contract.
          if (elements <= std::numeric_limits<int32_t>::max() &&
              TaskImmediateValue(declared_extent) >= elements &&
              (TaskImmediateValue(declared_mask) & mask) == mask) {
            VisitExpr(pointer);
            return;
          }
        }
        UpdateOpaqueAccessVar(
            StorageIdentity(load->source.as_or_throw<TensorVar>()));
      }
    }
    bool previous_context = opaque_pointer_context_;
    opaque_pointer_context_ = true;
    VisitExpr(pointer);
    opaque_pointer_context_ = previous_context;
  }

  bool HasNd2NzLogicalRegion(const CallObj *op, size_t index) const {
    auto annotation = TaskCallAnnotations(op).Get("nd2nz_logical_regions");
    const auto *mask =
        annotation ? annotation.value().as<IntImmObj>() : nullptr;
    return mask && (TaskImmediateValue(mask) & (1 << index));
  }

  void VisitNZPhysicalPointer(const CallObj *op, size_t index,
                              std::optional<int64_t> bytes, int mask) {
    const auto arguments = TaskCallOperands(op);
    const Expr &pointer = arguments[index];
    const auto *call = pointer.as<CallObj>();
    const auto *load =
        call && call->op.same_as(tl::access_ptr()) && call->args.size() == 3
            ? call->args[0].as<TensorLoadObj>()
            : nullptr;
    if (load && load->source.as_or_throw<TensorVar>().shape().size() >= 2) {
      size_t ndim = load->source.as_or_throw<TensorVar>().shape().size();
      if (arguments.size() == 7 && bytes && *bytes > 0 &&
          HasNd2NzLogicalRegion(op, index) && load->indices.size() == ndim) {
        // InsertNd2Nz proved that these logical axes map to the NZ DMA
        // blocks. Its access_ptr extent is a physical span including gaps;
        // interpreting it as a row-major run can omit rows of a column slice.
        std::vector<sym::IntSet> region;
        for (size_t i = 0; i < ndim; ++i) {
          PrimExpr extent = i + 2 == ndim ? arguments[2].as_or_throw<PrimExpr>()
                            : i + 1 == ndim
                                ? arguments[3].as_or_throw<PrimExpr>()
                                : IntImm::Int32(1);
          region.push_back(RelaxAccessIndex(load->indices[i], extent));
          VisitExpr(load->indices[i]);
        }
        if (mask & 1)
          Update(&read_buffers_, &read_regions_,
                 load->source.as_or_throw<TensorVar>(), region);
        if (mask & 2)
          // Keep pointer writes as may-write regions, excluding DMA gaps.
          Update(&write_buffers_, &write_regions_,
                 load->source.as_or_throw<TensorVar>(), region);
        VisitExpr(call->args[1]);
        return;
      }
      // An explicit pitch does not establish how physical blocks map back
      // to this multidimensional view. Retain flat pointer span handling,
      // but exclude this storage from reuse without a layout proof.
      UpdateOpaqueAccessVar(
          StorageIdentity(load->source.as_or_throw<TensorVar>()));
    }
    VisitPhysicalPointer(pointer, bytes, mask);
  }

  void VisitNd2NzCall(const CallObj *op, bool scatter) {
    const auto arguments = TaskCallOperands(op);
    std::optional<int64_t> source_bytes;
    std::optional<int64_t> destination_bytes;
    if (arguments.size() == 7) {
      const auto *rows = arguments[2].as<IntImmObj>();
      const auto *cols = arguments[3].as<IntImmObj>();
      auto element_bytes = [&](size_t index) -> int64_t {
        const auto *dtype = arguments[index].as<StringImmObj>();
        if (dtype && dtype->value == "float")
          return 4;
        if (dtype && (dtype->value == "half" || dtype->value == "bfloat16_t"))
          return 2;
        return 0;
      };
      if (rows && cols && TaskImmediateValue(rows) > 0 &&
          TaskImmediateValue(cols) > 0 &&
          TaskImmediateValue(rows) < std::numeric_limits<int64_t>::max()) {
        int64_t dst_bytes = element_bytes(6);
        if (scatter) {
          int64_t src_bytes = element_bytes(5);
          auto source_row =
              PhysicalSpan(TaskImmediateValue(cols), src_bytes, 0);
          auto destination_row =
              PhysicalSpan(TaskImmediateValue(cols), dst_bytes, 0);
          if (src_bytes && dst_bytes && source_row && destination_row) {
            // The scatter template loads whole 256-byte vectors, including
            // the last partial vector of the last source row.
            int64_t vectors = (*source_row - 1) / 256 + 1;
            if (auto last_row = PhysicalSpan(vectors, 256, 0))
              source_bytes = PhysicalSpan(TaskImmediateValue(rows) - 1,
                                          *source_row, *last_row);
            const auto *pitch = arguments[4].as<IntImmObj>();
            if (pitch && TaskImmediateValue(cols) % (32 / dst_bytes) == 0) {
              int64_t bursts = TaskImmediateValue(cols) / (32 / dst_bytes);
              if (auto blocks =
                      PhysicalSpan(bursts - 1, TaskImmediateValue(pitch),
                                   TaskImmediateValue(rows)))
                destination_bytes = PhysicalSpan(*blocks, 32, 0);
            }
          }
        } else {
          const auto *src_pitch = arguments[4].as<IntImmObj>();
          const auto *dst_pitch = arguments[5].as<IntImmObj>();
          if (dst_bytes && src_pitch && dst_pitch &&
              TaskImmediateValue(src_pitch) >= TaskImmediateValue(rows) &&
              TaskImmediateValue(dst_pitch) >= TaskImmediateValue(rows) &&
              TaskImmediateValue(cols) % (32 / dst_bytes) == 0) {
            int64_t bursts = TaskImmediateValue(cols) / (32 / dst_bytes);
            // Match lowering: both pitches are explicit 32-byte block strides.
            if (auto blocks =
                    PhysicalSpan(bursts - 1, TaskImmediateValue(src_pitch),
                                 TaskImmediateValue(rows)))
              source_bytes = PhysicalSpan(*blocks, 32, 0);
            if (auto blocks =
                    PhysicalSpan(bursts - 1, TaskImmediateValue(dst_pitch),
                                 TaskImmediateValue(rows)))
              destination_bytes = PhysicalSpan(*blocks, 32, 0);
          }
        }
      }
    }
    for (size_t i = 0; i < arguments.size(); ++i) {
      if (scatter && i == 0)
        VisitPhysicalPointer(arguments[i], source_bytes, 1);
      else if (!scatter && i == 1)
        VisitNZPhysicalPointer(op, i, source_bytes, 1);
      else if (i == (scatter ? 1U : 0U))
        VisitNZPhysicalPointer(op, i, destination_bytes, 2);
      else
        VisitExpr(arguments[i]);
    }
    for (const auto &kv : TaskCallAnnotations(op)) {
      if (auto expr = kv.second.as<PrimExpr>())
        VisitExpr(expr.value());
    }
  }

  std::vector<sym::IntSet> LogicalPointerRegion(const TensorLoad &load,
                                                const PrimExpr &extent) {
    TensorVar buffer = load->source.as_or_throw<TensorVar>();
    size_t ndim = buffer.shape().size();
    std::vector<sym::IntSet> region(ndim, sym::IntSet::Nothing());

    const auto *ext_imm = extent.as<IntImmObj>();
    if (ext_imm && load->indices.size() == ndim) {
      // Distribute a known flat extent (in elements) across dims,
      // innermost-first. `rem` is the number of elements still to be
      // covered at the current dimension; each dim consumes as many of
      // its `shape[i]` indices as the run spans, carrying the rounded-up
      // remainder to the next (outer) dim.
      int64_t rem = TaskImmediateValue(ext_imm);
      for (size_t k = ndim; k-- > 0;) {
        PrimExpr base = load->indices[k];
        const auto *dim_imm = buffer.shape()[k].as<IntImmObj>();
        if (rem <= 1) {
          region[k] = RelaxAccessIndex(base);
        } else if (dim_imm) {
          int64_t d = TaskImmediateValue(dim_imm);
          if (rem >= d) {
            // Run spills past this dim: cover it whole and carry up.
            region[k] = sym::IntSet::FromRange(
                Range::FromMinExtent(0, buffer.shape()[k]));
            rem = (rem + d - 1) / d;
          } else {
            region[k] = RelaxAccessIndex(base, IntImm(base.ty(), rem));
            rem = 1;
          }
        } else {
          // Non-constant dim size: be conservative for this and any
          // remaining outer dims.
          region[k] = sym::IntSet::FromRange(
              Range::FromMinExtent(0, buffer.shape()[k]));
          rem = 1;
        }
      }
    } else {
      // Non-constant extent or arity mismatch: fall back to whole buffer.
      for (size_t k = 0; k < ndim; ++k) {
        region[k] =
            sym::IntSet::FromRange(Range::FromMinExtent(0, buffer.shape()[k]));
      }
    }
    return region;
  }

  std::vector<sym::IntSet> PhysicalPointerRegion(const TensorLoad &load,
                                                 const PrimExpr &extent) {
    TensorVar buffer = load->source.as_or_throw<TensorVar>();
    size_t ndim = buffer.shape().size();
    std::vector<sym::IntSet> region(ndim, sym::IntSet::Nothing());

    bool reconstructed = false;
    const auto *ext_imm = extent.as<IntImmObj>();
    // Explicit strides, including dense T.view layouts, stay opaque
    // for multi-element runs until their physical image is proved.
    if (ext_imm && TaskImmediateValue(ext_imm) > 0 &&
        load->indices.size() == ndim &&
        (TaskImmediateValue(ext_imm) == 1 ||
         !buffer.ty()->layout.has_value())) {
      int64_t remaining = TaskImmediateValue(ext_imm);
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

        auto dimension = TryConstInt(buffer.shape()[k]);
        sym::IntSet base_set = RelaxAccessIndex(base);
        auto base_min = TryConstInt(base_set.min());
        auto base_max = TryConstInt(base_set.max());
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
              Range::FromMinExtent(0, buffer.shape()[k]));
          remaining = span;
        }
      }
      // A remaining carry escapes the outermost logical dimension, even
      // if the backing allocation has room for the physical access.
      reconstructed &= remaining == 1;
    }
    if (!reconstructed) {
      UpdateOpaqueAccessVar(StorageIdentity(buffer));
      // The storage is opaque: this placeholder cannot prove reuse and
      // is never passed to synchronization dependency analysis.
      for (size_t k = 0; k < ndim; ++k) {
        region[k] =
            sym::IntSet::FromRange(Range::FromMinExtent(0, buffer.shape()[k]));
      }
    }
    return region;
  }

  Result Visit_(const CallObj *op) final {
    if (IsScheduledBind(op))
      return Visit(ReadScheduledBinding(op).value);
    if (IsAnnotationCall(op))
      return std::nullopt;
    if (op->op.same_as(tl::loop_break()))
      has_loop_break_ = true;
    if (op->op.same_as(tirx::alloc_tensor_op()))
      return Visit(op->ty);
    if (op->op.same_as(tirx::decl_tensor_op())) {
      // Creating a typed view records address provenance, not a memory read.
      for (const Expr &arg : op->args)
        VisitAddressInputs(arg);
      Visit(op->ty);
      return std::nullopt;
    }
    if (op->op.same_as(tirx::masked_load_op()) ||
        op->op.same_as(tirx::masked_store_op())) {
      const bool write = op->op.same_as(tirx::masked_store_op());
      const size_t first_index = write ? 2 : 1;
      TensorVar tensor = op->args[0].as_or_throw<TensorVar>();
      TVM_FFI_CHECK_EQ(op->args.size(), first_index + tensor.shape().size() + 1,
                       ValueError);
      std::vector<sym::IntSet> region;
      for (size_t i = first_index; i + 1 < op->args.size(); ++i) {
        auto index = op->args[i].as_or_throw<PrimExpr>();
        region.push_back(RelaxAccessIndex(index));
        Visit(index);
      }
      // The mask can disable arbitrary lanes. Its rectangular may-access
      // image establishes ownership, but never a complete initialization.
      if (write) {
        Update(&write_buffers_, &write_regions_, tensor, std::move(region));
        Visit(op->args[1]);
      } else {
        Update(&read_buffers_, &read_regions_, tensor, std::move(region));
      }
      Visit(op->args.back());
      return std::nullopt;
    }
    if (op->op.same_as(tl::ptr_access())) {
      TVM_FFI_CHECK_EQ(op->args.size(), 4U, ValueError);
      int mask = TaskImmediateValue(op->args[3].as<IntImmObj>());
      VisitAddressInputs(op->args[0]);
      Visit(op->args[1]);
      Visit(op->args[2]);
      if (mask == 0)
        return std::nullopt;
      auto address = storage_.Address(op->args[0]);
      if (auto root = address.root.as<Var>()) {
        UpdateOpaqueAccessVar(*root);
        if (auto tensor = root->as<TensorVar>()) {
          // A raw byte-offset pointer need not follow the root's logical
          // strides, origin, or element width. Preserve its canonical storage
          // and access direction for owner discovery, while the opaque entry
          // prevents this placeholder from proving disjointness or reuse.
          std::vector<sym::IntSet> region;
          for (const auto &extent : tensor->shape())
            region.push_back(
                sym::IntSet::FromRange(Range::FromMinExtent(0, extent)));
          if (mask & 1)
            Update(&read_buffers_, &read_regions_, *tensor, region);
          if (mask & 2)
            Update(&write_buffers_, &write_regions_, *tensor, region);
        }
      } else {
        // Retain any input-storage escapes in an unrecognized pointer call.
        // No fabricated tensor root can make such an address versionable.
        Visit(op->args[0]);
      }
      return std::nullopt;
    }
    if (auto tile = ParseOperator(ffi::GetRef<Call>(op)); tile.defined()) {
      auto old_pitches = operation_pitches_;
      operation_pitches_ =
          GetCallL0Pitches(ffi::GetRef<Call>(op), layouts_, storage_);
      auto accesses = tile->GetAccessRegions();
      for (const TensorRegion &region : accesses.reads)
        RecordTensorRegion(region, true);
      for (const TensorRegion &region : accesses.writes) {
        TensorRegion write = region;
        if (mode_ == AccessMode::kPhysical && IsAscendCopyCall(op)) {
          auto decoded = DecodeTileOpArguments(op);
          write = ExpandPaddedCopyWriteRegion(
              AscendCopy(decoded.operands, decoded.annotations), write);
        }
        RecordTensorRegion(write, false);
      }
      if (mode_ == AccessMode::kPhysical && IsAscendCopyCall(op)) {
        auto decoded = DecodeTileOpArguments(op);
        AscendCopy copy(decoded.operands, decoded.annotations);
        auto record = [&](const TensorRegion &region, bool read) {
          TensorVar tensor = RegionTensor(region);
          if (!(IsL0ABuffer(tensor) || IsL0BBuffer(tensor) ||
                IsL0CBuffer(tensor)))
            return;
          bool compact = copy->l0_pitch.has_value();
          if (region->region.size() < 2)
            compact = true;
          else
            for (size_t axis = region->region.size() - 2;
                 axis < region->region.size(); ++axis)
              compact |= !ana_->CanProveEqual(region->region[axis]->extent,
                                              tensor.shape()[axis]);
          RecordL0MayAccess(region, read, compact);
        };
        for (const TensorRegion &region : accesses.reads)
          record(region, true);
        for (const TensorRegion &region : accesses.writes)
          record(region, false);
      }
      if (mode_ == AccessMode::kPhysical &&
          (op->op.same_as(Gemm::Get()) ||
           op->op.same_as(GemmBlockScaled::Get()))) {
        Gemm gemm = tile.as_or_throw<Gemm>();
        if (IsL0ABuffer(gemm->a_) && IsL0BBuffer(gemm->b_)) {
          auto record = [&](const TensorRegion &region, bool read,
                            size_t axis) {
            bool compact = !ana_->CanProveEqual(
                FloorDiv(region->region[axis]->extent + 15, 16),
                FloorDiv(RegionTensor(region).shape()[axis] + 15, 16));
            RecordL0MayAccess(region, read, compact);
          };
          record(gemm->aRegion_, true,
                 gemm->a_.shape().size() - (gemm->transA_ ? 1 : 2));
          record(gemm->bRegion_, true,
                 gemm->b_.shape().size() - (gemm->transB_ ? 2 : 1));
          record(gemm->cRegion_, false, gemm->c_.shape().size() - 2);
          if (!IsOne(gemm->clearAccum_))
            record(gemm->cRegion_, true, gemm->c_.shape().size() - 2);
        }
      }
      for (const Expr &operand : DecodeTileOpArguments(op).operands)
        Visit(operand);
      operation_pitches_ = std::move(old_pitches);
      return std::nullopt;
    }
    if (op->op.same_as(tl::ascend_nd2nz_scatter()) ||
        op->op.same_as(tl::ascend_nd2nz_post_copy())) {
      if (mode_ == AccessMode::kPhysical) {
        VisitNd2NzCall(op, op->op.same_as(tl::ascend_nd2nz_scatter()));
        return std::nullopt;
      }
      VisitLogicalNd2NzCall(op);
      return std::nullopt;
    }
    if (op->op.same_as(tl::access_ptr())) {
      TVM_FFI_CHECK_EQ(op->args.size(), 3U, ValueError);
      auto load = op->args[0].as_or_throw<TensorLoad>();
      TensorVar tensor = load->source.as_or_throw<TensorVar>();
      int mask = TaskImmediateValue(op->args[2].as<IntImmObj>());
      if (opaque_pointer_context_)
        UpdateOpaqueAccessVar(StorageIdentity(tensor));
      PrimExpr extent = op->args[1].as_or_throw<PrimExpr>();
      auto region = mode_ == AccessMode::kLogical
                        ? LogicalPointerRegion(load, extent)
                        : PhysicalPointerRegion(load, extent);
      if (mask & 1)
        Update(&read_buffers_, &read_regions_, tensor, region);
      if (mask & 2)
        Update(&write_buffers_, &write_regions_, tensor, region);
      for (const auto &index : load->indices)
        Visit(index);
      Visit(extent);
      return std::nullopt;
    }
    if (op->op.same_as(tirx::address_of_op())) {
      // address_of names only the base element. Its consumer may read past
      // it, even if this task also has a precise read from the same storage.
      if (!op->args.empty())
        if (const auto *load = op->args[0].as<TensorLoadObj>())
          UpdateOpaqueAccessVar(
              StorageIdentity(load->source.as_or_throw<TensorVar>()));
      return StmtExprVisitor::Visit_(op);
    }
    if (op->op.same_as(tirx::tensor_data_ptr_op())) {
      TensorVar tensor = op->args[0].as_or_throw<TensorVar>();
      UpdateOpaqueAccessVar(StorageIdentity(tensor));
      UpdateReadVar(tensor);
      return std::nullopt;
    }
    bool old_context = opaque_pointer_context_;
    opaque_pointer_context_ |=
        op->ty.as<PtrTypeObj>() != nullptr ||
        // Native handle carriers store pointer bits in a primitive tensor.
        // Reinterpreting the address to that integer must retain the escape,
        // since a later pointer consumer can exceed access_ptr's extent.
        (op->op.same_as(tirx::reinterpret_op()) && !op->args.empty() &&
         op->args[0]->ty.as<PtrTypeObj>() != nullptr) ||
        op->op.same_as(tl::simd_vgatherb()) ||
        op->op.same_as(tl::simd_vgather2()) ||
        op->op.same_as(tl::simd_vscatter()) ||
        op->op.same_as(tl::simd_vsstb()) ||
        (op->op.same_as(tl::simd_vld2()) && op->args.size() >= 3);
    auto result = StmtExprVisitor::Visit_(op);
    opaque_pointer_context_ = old_context;
    return result;
  }

  void VisitAddressInputs(const Expr &expression) {
    if (auto variable = expression.as<Var>()) {
      UpdateReadVar(variable.value());
      return;
    }
    if (const auto *call = expression.as<CallObj>()) {
      if (call->op.same_as(tirx::tensor_data_ptr_op())) {
        UpdateReadVar(call->args[0].as_or_throw<Var>());
        return;
      }
      if (call->op.same_as(tirx::reinterpret_op()) ||
          call->op.same_as(tirx::ptr_byte_offset_op())) {
        VisitAddressInputs(call->args[0]);
        for (size_t i = 1; i < call->args.size(); ++i)
          Visit(call->args[i]);
        return;
      }
    }
    Visit(expression);
  }

  void VisitLogicalNd2NzCall(const CallObj *op) {
    const auto arguments = TaskCallOperands(op);
    bool scatter = op->op.same_as(tl::ascend_nd2nz_scatter());
    for (size_t arg = 0; arg < arguments.size(); ++arg) {
      const auto *pointer = arg < 2 ? arguments[arg].as<CallObj>() : nullptr;
      const auto *load = pointer && pointer->op.same_as(tl::access_ptr()) &&
                                 pointer->args.size() == 3
                             ? pointer->args[0].as<TensorLoadObj>()
                             : nullptr;
      if (!load) {
        Visit(arguments[arg]);
        continue;
      }
      TensorVar tensor = load->source.as_or_throw<TensorVar>();
      bool read = arg == (scatter ? 0U : 1U);
      bool logical = arguments.size() == 7 && load->indices.size() >= 2 &&
                     load->indices.size() == tensor.shape().size() &&
                     ((read && scatter) || HasNd2NzLogicalRegion(op, arg));
      Array<Range> ranges;
      for (size_t axis = 0; axis < tensor.shape().size(); ++axis) {
        PrimExpr begin = logical ? load->indices[axis]
                                 : MakeConst(tensor.shape()[axis].ty(), 0);
        PrimExpr extent = !logical ? tensor.shape()[axis]
                          : axis + 2 == tensor.shape().size()
                              ? arguments[2].as_or_throw<PrimExpr>()
                          : axis + 1 == tensor.shape().size()
                              ? arguments[3].as_or_throw<PrimExpr>()
                              : MakeConst(begin.ty(), 1);
        ranges.push_back(Range::FromMinExtent(begin, extent));
      }
      RecordTensorRegion(TensorRegion(tensor, ranges, TensorRegionType()),
                         read);
      for (const auto &index : load->indices)
        Visit(index);
    }
  }
};

} // namespace tvm::tl::ascend
