#ifndef TILELANG_TRANSFORM_COMMON_ITERATION_PRIVATE_H_
#define TILELANG_TRANSFORM_COMMON_ITERATION_PRIVATE_H_

#include "compat/expr.h"
#include "memory_access.h"

#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/memory.h>

#include <optional>
#include <unordered_set>
#include <vector>

namespace tvm::tl {

class IterationPrivateStorage : public tirx::StmtExprVisitor {
public:
  explicit IterationPrivateStorage(tirx::TensorVar tensor,
                                   MemoryAliases aliases = {},
                                   bool allow_loop_carried = false)
      : tensor_(tensor), aliases_(std::move(aliases)),
        allow_loop_carried_(allow_loop_carried) {}

  bool Check(const SeqStmt &body) {
    Visit(body);
    return safe_;
  }

private:
  ffi::Optional<ffi::VisitInterrupt> Visit_(const BreakNode *) final {
    safe_ = false;
    return std::nullopt;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const ContinueNode *) final {
    safe_ = false;
    return std::nullopt;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const ReturnNode *) final {
    safe_ = false;
    return std::nullopt;
  }

  bool IsStorage(const Expr &storage) const {
    MemoryLocation location = aliases_.Resolve(storage);
    return location.allocation && location.allocation.value().same_as(tensor_);
  }

  std::optional<std::pair<PrimExpr, PrimExpr>>
  Footprint(const Expr &storage, const ffi::Array<PrimExpr> &indices) {
    auto tensor = storage.as<tirx::TensorVar>();
    if (!tensor || !IsStorage(storage))
      return std::nullopt;
    ffi::Array<PrimExpr> coordinates;
    ffi::Optional<PrimVar> lane;
    int64_t lanes = 1;
    for (const PrimExpr &index : indices) {
      if (const auto *ramp = index.as<prim::RampNode>()) {
        auto count = GetConstInt(ramp->lanes);
        if (!count || *count <= 0 || (lane && *count != lanes))
          return std::nullopt;
        if (!lane)
          lane = PrimVar("private_lane", ramp->base.ty());
        lanes = *count;
        coordinates.push_back(ramp->base + lane.value() * ramp->stride);
      } else if (index.ty().IsScalar()) {
        coordinates.push_back(index);
      } else {
        return std::nullopt;
      }
    }
    MemoryLocation location = aliases_.Element(storage, coordinates);
    if (!location.precise)
      return std::nullopt;
    PrimExpr element_bytes =
        IntImm::Int64(tensor.value().dtype().StorageBytes());
    PrimExpr offset = location.offset;
    if (lane) {
      offset =
          Substitute(offset, {{lane.value(), MakeConst(lane.value().ty(), 0)}});
      if (!analyzer_->CanProveEqual(
              location.offset,
              offset + Cast(PrimType::Int(64), lane.value()) * element_bytes))
        return std::nullopt;
    }
    return std::pair{offset, element_bytes * IntImm::Int64(lanes)};
  }

  bool Matches(const Expr &storage, const ffi::Array<PrimExpr> &indices) {
    if (complete_)
      return true;
    auto access = Footprint(storage, indices);
    if (!access)
      return false;
    for (const auto &written : written_) {
      if (analyzer_->CanProve(written.first <= access->first) &&
          analyzer_->CanProve(access->first + access->second <=
                              written.first + written.second))
        return true;
    }
    return false;
  }

  bool Covers(const Stmt &statement, std::vector<const ForNode *> loops = {}) {
    if (const auto *sequence = statement.as<SeqStmtNode>()) {
      for (const Stmt &child : sequence->seq) {
        if (Covers(child, loops))
          return true;
      }
      return false;
    }
    if (const auto *region = statement.as<RegionStmtNode>()) {
      return (region->op->name == "tirx.scope" ||
              region->op->name == "tl.scope") &&
             Covers(region->body, loops);
    }
    if (const auto *loop = statement.as<ForNode>()) {
      if (!IsZero(loop->begin) || !IsOne(loop->step) ||
          !analyzer_->CanProve(loop->end > 0))
        return false;
      loops.push_back(loop);
      return Covers(loop->body, loops);
    }
    const auto *store = statement.as<TensorStoreNode>();
    if (!store || !IsStorage(store->dest)) {
      return false;
    }
    int lanes = tensor_.dtype().lanes();
    if ((lanes & (lanes - 1)) != 0)
      return false;
    auto access = Footprint(store->dest, store->indices);
    if (!access)
      return false;
    PrimExpr linear = IntImm::Int64(0);
    PrimExpr count = IntImm::Int64(1);
    for (const ForNode *loop : loops) {
      linear = linear * Cast(PrimType::Int(64), loop->end) +
               Cast(PrimType::Int(64), loop->var);
      count = count * Cast(PrimType::Int(64), loop->end);
    }
    return analyzer_->CanProveEqual(access->first, linear * access->second) &&
           analyzer_->CanProveEqual(count * access->second,
                                    GetBufferStorageSizeBytes(tensor_));
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const TensorLoadNode *node) final {
    if (IsStorage(node->source) && !Matches(node->source, node->indices))
      safe_ = false;
    for (const PrimExpr &index : node->indices)
      Visit(index);
    return std::nullopt;
  }

  ffi::Optional<ffi::VisitInterrupt>
  Visit_(const TensorRegionNode *node) final {
    if (IsStorage(node->source) && !complete_)
      safe_ = false;
    for (const Range &range : node->region) {
      Visit(range->begin);
      Visit(range->extent);
    }
    return std::nullopt;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const TensorStoreNode *node) final {
    Visit(node->value);
    for (const PrimExpr &index : node->indices)
      Visit(index);
    if (IsStorage(node->dest)) {
      if (auto footprint = Footprint(node->dest, node->indices))
        written_.push_back(*footprint);
      complete_ |= Covers(ffi::GetRef<Stmt>(node));
    }
    return std::nullopt;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const VarNode *node) final {
    if (IsStorage(ffi::GetRef<Var>(node)))
      safe_ = false;
    return std::nullopt;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const BindNode *node) final {
    aliases_.Bind(node);
    const auto *allocation = node->value.as<CallNode>();
    if (allocation && allocation->op.same_as(tirx::alloc_tensor_op()) &&
        node->var.same_as(tensor_)) {
      written_.clear();
      complete_ = false;
      return Visit(node->value);
    }
    if ((node->var->ty.as<PtrTypeNode>() ||
         node->var->ty.as<tirx::TensorTypeNode>()) &&
        SideEffect(node->value) == CallEffectKind::kPure &&
        IsStorage(node->var))
      return std::nullopt;
    return StmtExprVisitor::Visit_(node);
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const ForNode *node) final {
    Visit(node->begin);
    Visit(node->end);
    Visit(node->step);
    if (allow_loop_carried_) {
      bool references_storage = false;
      PostOrderVisit(node->body, [&](const ffi::ObjectRef &object) {
        if (auto variable = object.as<Var>())
          references_storage |= IsStorage(*variable);
      });
      if (!references_storage)
        return std::nullopt;
      auto begin = GetConstInt(node->begin);
      auto end = GetConstInt(node->end);
      if (begin && end && IsOne(node->step) && *begin >= 0 && *end >= *begin &&
          *end - *begin <= remaining_iterations_) {
        for (int64_t value = *begin; value < *end && safe_; ++value) {
          if (remaining_iterations_-- <= 0) {
            safe_ = false;
            break;
          }
          Visit(Substitute(node->body,
                           {{node->var, MakeConst(node->var.ty(), value)}}));
        }
        return std::nullopt;
      }
    }
    auto previous = written_;
    bool initialized = complete_;
    Visit(node->body);
    written_ = std::move(previous);
    complete_ = initialized || Covers(ffi::GetRef<Stmt>(node));
    return std::nullopt;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const WhileNode *node) final {
    Visit(node->condition);
    auto previous = written_;
    bool initialized = complete_;
    Visit(node->body);
    written_ = std::move(previous);
    complete_ = initialized;
    return std::nullopt;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const IfNode *node) final {
    Visit(node->condition);
    if (allow_loop_carried_) {
      if (auto condition = GetConstInt(analyzer_->Simplify(node->condition))) {
        if (*condition)
          Visit(node->then_case);
        else if (node->else_case)
          Visit(*node->else_case);
        return std::nullopt;
      }
    }
    auto previous = written_;
    bool initialized = complete_;
    Visit(node->then_case);
    bool then_complete = complete_;
    written_ = previous;
    complete_ = initialized;
    if (node->else_case)
      Visit(*node->else_case);
    complete_ &= then_complete;
    written_ = std::move(previous);
    return std::nullopt;
  }

  tirx::TensorVar tensor_;
  MemoryAliases aliases_;
  sym::Analyzer analyzer_;
  bool complete_ = false;
  bool safe_ = true;
  bool allow_loop_carried_ = false;
  int64_t remaining_iterations_ = 4096;
  std::vector<std::pair<PrimExpr, PrimExpr>> written_;
};

inline bool IsIterationPrivateStorage(const tirx::TensorVar &tensor,
                                      const SeqStmt &body,
                                      MemoryAliases aliases = {}) {
  return ffi::make_object<IterationPrivateStorage>(tensor, std::move(aliases))
      ->Check(body);
}

inline bool HasInitializedStorageReads(const tirx::TensorVar &tensor,
                                       const SeqStmt &body,
                                       MemoryAliases aliases = {}) {
  return ffi::make_object<IterationPrivateStorage>(tensor, std::move(aliases),
                                                   true)
      ->Check(body);
}

} // namespace tvm::tl

#endif
