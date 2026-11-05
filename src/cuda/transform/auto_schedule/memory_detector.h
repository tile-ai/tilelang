/*!
 * \file auto_schedule/memory_detector.h
 * \brief Native memory and scalar dependencies used by warp scheduling.
 */
#pragma once

#include <tvm/ir/analysis.h>
#include <tvm/sym/int_set.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>

#include <memory>
#include <vector>

#include "op/native_utils.h"
#include "op/operator.h"
#include "op/utils.h"
#include "transform/common/pipeline_access.h"

namespace tvm::tl::cuda {

using namespace tirx;
using namespace prim;
using namespace ffi;

// The public holder is an ordinary C++ value; traversal lives in an owned
// native visitor. Each Analyze starts with fresh lexical facts. In particular,
// mutable loads in a scalar Bind never become permanent symbolic rewrites.
class MemoryAccessDetector {
  class Collector : public StmtExprVisitor {
  public:
    using StmtExprVisitor::Visit;
    using StmtExprVisitor::Visit_;
    using Result = Optional<VisitInterrupt>;

    explicit Collector(const PipelineMemoryAnalysis *memory)
        : memory_(memory) {}

    std::vector<TensorRegion> Reads() const {
      return Regions(read_tensors_, read_regions_);
    }
    std::vector<TensorRegion> Writes() const {
      return Regions(write_tensors_, write_regions_);
    }
    const std::vector<Var> &ReadVars() const { return read_vars_; }
    const std::vector<Var> &WriteVars() const { return write_vars_; }
    bool HasUnknownAccess() const { return unknown_access_; }

  private:
    using Domain = Map<Var, sym::IntSet>;
    using RegionsByTensor = std::vector<std::vector<sym::IntSet>>;

    static void AddVar(std::vector<Var> *values, const Var &var) {
      for (const auto &old : *values)
        if (old.same_as(var))
          return;
      values->push_back(var);
    }

    void Update(std::vector<TensorVar> *tensors, RegionsByTensor *regions,
                TensorVar tensor, std::vector<sym::IntSet> bounds) {
      AddVar(&read_vars_, tensor);
      if (memory_) {
        auto root = memory_->Root(tensor);
        if (!root) {
          unknown_access_ = true;
        } else if (!root.value().same_as(tensor)) {
          if (auto canonical = root.value().as<TensorVar>()) {
            // Unlike old Buffer.data, a native view has its own strides,
            // offset and element type. Unlike coordinates are never compared.
            tensor = *canonical;
            bounds.clear();
            for (const auto &dim : tensor.shape())
              bounds.push_back(
                  sym::IntSet::FromMinExtent(IntImm(dim.ty(), 0), dim));
          } else {
            unknown_access_ = true;
          }
        }
      }
      for (size_t i = 0; i < tensors->size(); ++i) {
        if (!(*tensors)[i].same_as(tensor))
          continue;
        TVM_FFI_CHECK_EQ((*regions)[i].size(), bounds.size(), ValueError);
        for (size_t j = 0; j < bounds.size(); ++j)
          (*regions)[i][j] = sym::Union({(*regions)[i][j], bounds[j]});
        return;
      }
      tensors->push_back(tensor);
      regions->push_back(std::move(bounds));
    }

    std::vector<TensorRegion> Regions(const std::vector<TensorVar> &tensors,
                                      const RegionsByTensor &bounds) const {
      std::vector<TensorRegion> result;
      for (size_t i = 0; i < tensors.size(); ++i) {
        Array<Range> ranges;
        TVM_FFI_CHECK_EQ(tensors[i].shape().size(), bounds[i].size(),
                         ValueError);
        for (size_t j = 0; j < bounds[i].size(); ++j) {
          auto dim = tensors[i].shape()[j];
          ranges.push_back(
              bounds[i][j]
                  .CoverRange(Range::FromMinExtent(IntImm(dim.ty(), 0), dim))
                  .value());
        }
        result.push_back(TensorRegion(tensors[i], ranges, TensorRegionType()));
      }
      return result;
    }

    PrimExpr Resolve(const PrimExpr &value) const {
      // Stored replacements were resolved at their definition sites, so one
      // simultaneous substitution suffices and cannot loop on a self-binding.
      return SubstituteLayoutVars(value, let_bindings_);
    }

    sym::IntSet Relax(const PrimExpr &index,
                      Optional<PrimExpr> extent = std::nullopt) const {
      auto base = Resolve(index);
      auto lo = sym::EvalSet(sym::IntSet::Vector(base), domains_);
      if (!extent || IsOne(*extent))
        return lo;
      auto count = Resolve(*extent);
      if (auto literal = TryConstInt(count); literal && *literal <= 0)
        return sym::IntSet::Nothing();
      auto hi = sym::EvalSet(
          sym::IntSet::Vector(base + count - IntImm(base.ty(), 1)), domains_);
      return sym::IntSet::Interval(lo.min(), hi.max());
    }

    void Access(const TensorRegion &region, int mask) {
      std::vector<sym::IntSet> bounds;
      for (const Range &range : region->region) {
        bounds.push_back(Relax(range->begin, range->extent));
        Visit(range->begin);
        Visit(range->extent);
      }
      TensorVar tensor = RegionTensor(region);
      if (mask & kAccessRead)
        Update(&read_tensors_, &read_regions_, tensor, bounds);
      if (mask & kAccessWrite)
        Update(&write_tensors_, &write_regions_, tensor, bounds);
    }

    Result Visit_(const SeqStmtObj *op) final {
      auto saved = let_bindings_;
      for (const auto &statement : op->seq)
        Visit(statement);
      let_bindings_ = std::move(saved);
      return std::nullopt;
    }

    Result Visit_(const ForObj *op) final {
      Visit(op->begin);
      Visit(op->end);
      Visit(op->step);
      auto saved = domains_;
      auto step = TryConstInt(op->step);
      if (step && *step > 0) {
        domains_.Set(op->var, sym::IntSet::FromRange(
                                  Range(Resolve(op->begin), Resolve(op->end))));
      } else if (step && *step < 0) {
        domains_.Set(op->var,
                     sym::IntSet::FromRange(Range(
                         Resolve(op->end) + IntImm(op->end.ty(), 1),
                         Resolve(op->begin) + IntImm(op->begin.ty(), 1))));
      } else {
        domains_.Set(op->var, sym::IntSet::Everything());
      }
      Visit(op->body);
      domains_ = std::move(saved);
      return std::nullopt;
    }

    Result Visit_(const IfObj *op) final {
      Visit(op->condition);
      Visit(op->then_case);
      if (op->else_case)
        Visit(*op->else_case);
      return std::nullopt;
    }

    Result Visit_(const RegionStmtObj *op) final {
      // Metadata references are compiler operands, not runtime reads of the
      // annotated buffers. Region arguments with real execution semantics
      // (thread extents etc.) are still charged to the containing operation.
      if (op->op->name != "tl.attr")
        for (const auto &argument : op->args)
          Visit(argument);
      auto saved = domains_;
      if (op->op.same_as(tirx::launch_thread_op())) {
        auto extent = op->args[1].as_or_throw<PrimExpr>();
        domains_.Set(op->body_params[0], sym::IntSet::FromMinExtent(
                                             IntImm(extent.ty(), 0), extent));
      }
      Visit(op->body);
      domains_ = std::move(saved);
      return std::nullopt;
    }

    Result Visit_(const BindObj *op) final {
      Visit(op->value);
      Visit(op->var->ty);
      AddVar(&write_vars_, op->var);
      if (auto primitive = op->value.as<PrimExpr>();
          primitive && SideEffect(op->value) <= CallEffectKind::kPure)
        let_bindings_.Set(op->var, Resolve(*primitive));
      return std::nullopt;
    }

    Result Visit_(const VarObj *op) final {
      AddVar(&read_vars_, GetRef<Var>(op));
      return std::nullopt;
    }

    Result Visit_(const TensorLoadObj *op) final {
      Array<Range> ranges;
      for (const PrimExpr &index : op->indices)
        ranges.push_back(Range::FromMinExtent(index, IntImm(index.ty(), 1)));
      Access(TensorRegion(op->source, ranges, TensorRegionType(), op->loc),
             kAccessRead);
      return std::nullopt;
    }

    Result Visit_(const TensorStoreObj *op) final {
      Array<Range> ranges;
      for (const PrimExpr &index : op->indices)
        ranges.push_back(Range::FromMinExtent(index, IntImm(index.ty(), 1)));
      Access(TensorRegion(op->dest, ranges, TensorRegionType(), op->loc),
             kAccessWrite);
      return Visit(op->value);
    }

    void Address(const Expr &base) {
      if (const auto *load = base.as<TensorLoadObj>()) {
        Visit(load->source);
        for (const auto &index : load->indices)
          Visit(index);
      } else if (const auto *region = base.as<TensorRegionObj>()) {
        Visit(region->source);
        for (const auto &range : region->region) {
          Visit(range->begin);
          Visit(range->extent);
        }
      } else {
        Visit(base);
      }
    }

    Optional<TensorVar> PointerTensor(const Expr &base) const {
      if (memory_)
        return memory_->Tensor(base);
      if (auto tensor = base.as<TensorVar>())
        return tensor;
      if (const auto *load = base.as<TensorLoadObj>())
        return load->source.as<TensorVar>();
      if (const auto *region = base.as<TensorRegionObj>())
        return region->source.as<TensorVar>();
      if (const auto *call = base.as<CallObj>()) {
        if ((call->op.same_as(tirx::tensor_data_ptr_op()) ||
             call->op.same_as(tirx::ptr_byte_offset_op()) ||
             call->op.same_as(tirx::reinterpret_op()) ||
             call->op.same_as(tirx::address_of_op())) &&
            !call->args.empty())
          return PointerTensor(call->args[0]);
      }
      return std::nullopt;
    }

    void RawAccess(const Expr &pointer, int mask) {
      if (auto tensor = PointerTensor(pointer))
        Access(FullTensorRegion(*tensor), mask);
      else
        unknown_access_ = true;
      Address(pointer);
    }

    Result Visit_(const CallObj *op) final {
      const auto *name = op->op.as<OpObj>();
      if (name && name->name == "tl.annotation")
        return std::nullopt;
      if (op->op.same_as(tirx::alloc_tensor_op()) ||
          op->op.same_as(tirx::decl_tensor_op())) {
        for (const auto &arg : op->args)
          Address(arg);
        return std::nullopt;
      }
      if (op->op.same_as(tl::region())) {
        auto access = NormalizeToAccessRegion(GetRef<Expr>(op));
        Access(access.region, access.access_mask);
        return std::nullopt;
      }
      if (TileOperator tile = ParseOperator(GetRef<Call>(op)); tile.defined()) {
        auto accesses = tile->GetAccessRegions();
        for (const auto &region : accesses.reads)
          Access(region, kAccessRead);
        for (const auto &region : accesses.writes)
          Access(region, kAccessWrite);
        // Non-region scalar operands are runtime inputs as well (GEMM wait,
        // copy bounds, reduce index, predicates). Metadata Tuple is not.
        for (const auto &arg : DecodeTileOpArguments(op).operands) {
          if (!arg.as<TensorRegionObj>() && !arg.as<TensorLoadObj>())
            Visit(arg);
        }
        return std::nullopt;
      }
      if (name && name->name == "tl.access_ptr") {
        auto mask =
            GetConservativeAccessMask(op->args[2].as_or_throw<PrimExpr>());
        auto tensor = PointerTensor(op->args[0]);
        const auto *load = op->args[0].as<TensorLoadObj>();
        if (tensor && load && tensor.value().shape().size() == 1 &&
            !tensor.value().ty()->layout) {
          auto region = TensorRegion(
              *tensor,
              {Range::FromMinExtent(load->indices[0],
                                    op->args[1].as_or_throw<PrimExpr>())},
              TensorRegionType(), op->loc);
          Access(region, mask);
          Address(op->args[0]);
        } else {
          // A flat physical run may cross rows, padding or swizzles. Taking
          // the full tensor avoids the old under-approximation at row ends.
          RawAccess(op->args[0], mask);
        }
        Visit(op->args[1]);
        Visit(op->args[2]);
        return std::nullopt;
      }
      if (name && name->name == "tl.ptr_access") {
        RawAccess(op->args[0], GetConservativeAccessMask(
                                   op->args[3].as_or_throw<PrimExpr>()));
        for (size_t i = 1; i < op->args.size(); ++i)
          Visit(op->args[i]);
        return std::nullopt;
      }
      if (op->op.same_as(tirx::address_of_op())) {
        Address(op->args[0]);
        return std::nullopt;
      }
      if (op->op.same_as(tirx::tensor_data_ptr_op())) {
        Visit(op->args[0]);
        return std::nullopt;
      }
      auto effects = Op::GetAttrMap(tvm::op_attr::kCallEffectKind);
      auto effect = name && effects.count(GetRef<Op>(name))
                        ? effects[GetRef<Op>(name)]
                        : CallEffectKind::kOpaque;
      if (effect >= CallEffectKind::kReadState) {
        for (const auto &argument : op->args) {
          if (!argument->ty.as<PtrTypeObj>() &&
              !argument->ty.as<TensorTypeObj>())
            continue;
          const auto *descriptor = argument.as<CallObj>();
          const auto *desc_name =
              descriptor ? descriptor->op.as<OpObj>() : nullptr;
          if (desc_name && (desc_name->name == "tl.access_ptr" ||
                            desc_name->name == "tl.ptr_access"))
            continue;
          RawAccess(argument, effect == CallEffectKind::kReadState
                                  ? kAccessRead
                                  : kAccessReadWrite);
        }
      }
      return StmtExprVisitor::Visit_(op);
    }

    const PipelineMemoryAnalysis *memory_;
    Domain domains_;
    Map<Var, PrimExpr> let_bindings_;
    std::vector<TensorVar> read_tensors_, write_tensors_;
    RegionsByTensor read_regions_, write_regions_;
    std::vector<Var> read_vars_, write_vars_;
    bool unknown_access_{false};
  };

public:
  explicit MemoryAccessDetector(const PipelineMemoryAnalysis *memory = nullptr)
      : memory_(memory) {}
  void Analyze(const Stmt &stmt) {
    collector_ = make_object<Collector>(memory_);
    collector_->Visit(stmt);
  }
  void Analyze(const Expr &expr) {
    collector_ = make_object<Collector>(memory_);
    collector_->Visit(expr);
  }
  std::vector<TensorRegion> GetReadRegions() const {
    return collector_->Reads();
  }
  std::vector<TensorRegion> GetWriteRegions() const {
    return collector_->Writes();
  }
  std::vector<Var> GetReadVars() const { return collector_->ReadVars(); }
  std::vector<Var> GetWriteVars() const { return collector_->WriteVars(); }
  bool HasUnknownAccess() const { return collector_->HasUnknownAccess(); }

private:
  const PipelineMemoryAnalysis *memory_;
  ObjectPtr<Collector> collector_;
};

} // namespace tvm::tl::cuda
