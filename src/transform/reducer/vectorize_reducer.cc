/*!
 * \file tl/transform/reducer/vectorize_reducer.cc
 * \brief Reducer-aware vectorization after physical layout materialization.
 */

#include "vectorize_reducer.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/tirx/stmt_functor.h>

#include "backend/common/op/reduce.h"
#include "op/deferred_reducer.h"
#include "op/reduce.h"
#include "transform/loop_vectorize.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

bool IsPackedReduceType(const ReduceType &type) {
  return type->IsSum() || type->IsAbsSum() || type->IsMax() || type->IsMin() ||
         type->IsAbsMax();
}

bool ContainsReducerUpdate(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &object) {
    if (const auto *attr_stmt = object.as<AttrStmtNode>()) {
      found |= attr_stmt->attr_key == attr::kReducerUpdate;
    }
  });
  return found;
}

bool ContainsNestedLoop(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &object) {
    found |= object.as<ForNode>() != nullptr;
  });
  return found;
}

class ReducerUpdateMarkerRemover : public StmtExprMutator {
private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == attr::kReducerUpdate) {
      return VisitStmt(op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

struct DirectReducerUpdate {
  AttrStmt marker;
  BufferStore store;
  ReduceType type;
};

std::optional<DirectReducerUpdate> MatchDirectReducerUpdate(const Stmt &stmt) {
  const auto *marker = stmt.as<AttrStmtNode>();
  if (marker == nullptr || marker->attr_key != attr::kReducerUpdate) {
    return std::nullopt;
  }
  Optional<ReduceType> type = marker->node.as<ReduceType>();
  const auto *store = marker->body.as<BufferStoreNode>();
  if (!type.defined() || store == nullptr) {
    return std::nullopt;
  }
  return DirectReducerUpdate{GetRef<AttrStmt>(marker),
                             GetRef<BufferStore>(store), type.value()};
}

int GetPackedVectorWidth(const For &loop, const Target &target) {
  int width = 2;
  bool found = false;
  bool supported = true;
  PostOrderVisit(loop->body, [&](const ObjectRef &object) {
    const auto *marker = object.as<AttrStmtNode>();
    if (marker == nullptr || marker->attr_key != attr::kReducerUpdate) {
      return;
    }
    found = true;
    Optional<ReduceType> type = marker->node.as<ReduceType>();
    const auto *store = marker->body.as<BufferStoreNode>();
    if (!type.defined() || store == nullptr ||
        !IsPackedReduceType(type.value())) {
      supported = false;
      return;
    }
    width = std::min(width, backend::reduce::GetTargetPreferredVectorizedSize(
                                store->buffer->dtype, target));
  });
  return found && supported ? width : 1;
}

bool ReducerTargetsCanVectorize(const For &loop, int vector_width) {
  bool found = false;
  bool can_vectorize = true;
  PostOrderVisit(loop->body, [&](const ObjectRef &object) {
    const auto *marker = object.as<AttrStmtNode>();
    if (marker == nullptr || marker->attr_key != attr::kReducerUpdate) {
      return;
    }
    found = true;
    const auto *store = marker->body.as<BufferStoreNode>();
    if (store == nullptr) {
      can_vectorize = false;
      return;
    }
    Stmt target_store =
        BufferStore(store->buffer, make_zero(store->buffer->dtype),
                    store->indices, store->predicate, store->span);
    For target_loop(loop->loop_var, loop->min, loop->extent, ForKind::kSerial,
                    std::move(target_store), std::nullopt, {}, loop->step,
                    loop->span);
    arith::Analyzer analyzer;
    int target_width = GetVectorizeSize(target_loop, &analyzer);
    can_vectorize &=
        target_width >= vector_width && target_width % vector_width == 0;
  });
  return found && can_vectorize;
}

class ReducerUpdateVectorizer : public StmtExprMutator {
public:
  explicit ReducerUpdateVectorizer(Target target)
      : target_(std::move(target)) {}

private:
  Stmt VisitStmt_(const SBlockNode *op) final {
    allocation_stack_.emplace_back();
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    std::vector<Buffer> allocations = std::move(allocation_stack_.back());
    allocation_stack_.pop_back();
    if (!allocations.empty()) {
      SBlockNode *block_ptr = block.CopyOnWrite();
      Array<Buffer> buffers = block->alloc_buffers;
      for (Buffer &buffer : allocations) {
        buffers.push_back(std::move(buffer));
      }
      block_ptr->alloc_buffers = std::move(buffers);
    }
    return block;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (!ContainsReducerUpdate(loop->body) || ContainsNestedLoop(loop->body)) {
      return loop;
    }

    const int packed_width = GetPackedVectorWidth(loop, target_);
    if (packed_width <= 1) {
      return loop;
    }

    arith::Analyzer analyzer;
    const int ordinary_width = GetVectorizeSize(loop, &analyzer);
    if (ordinary_width >= packed_width && ordinary_width % packed_width == 0 &&
        ReducerTargetsCanVectorize(loop, packed_width)) {
      For vectorized = VectorizeLoop(loop, &analyzer, {}, packed_width);
      return ReducerUpdateMarkerRemover()(std::move(vectorized));
    }

    if (std::optional<Stmt> packed = TryPackLocalReduction(loop)) {
      return packed.value();
    }
    return loop;
  }

  std::optional<Stmt> TryPackLocalReduction(const For &loop) {
    if (allocation_stack_.empty() || !is_zero(loop->min) ||
        loop->kind == ForKind::kVectorized || loop->thread_binding.defined() ||
        !loop->annotations.empty()) {
      return std::nullopt;
    }
    if (loop->step.defined() && !is_one(loop->step.value())) {
      return std::nullopt;
    }
    const int64_t *extent = as_const_int(loop->extent);
    if (extent == nullptr || *extent < 2 || *extent % 2 != 0) {
      return std::nullopt;
    }

    std::optional<DirectReducerUpdate> update =
        MatchDirectReducerUpdate(loop->body);
    if (!update.has_value() || !IsPackedReduceType(update->type) ||
        update->store->predicate.defined() ||
        backend::reduce::GetTargetPreferredVectorizedSize(
            update->store->buffer->dtype, target_) != 2) {
      return std::nullopt;
    }

    arith::Analyzer analyzer;
    analyzer.Bind(loop->loop_var,
                  Range::FromMinExtent(loop->min, loop->extent));
    for (const PrimExpr &index : update->store->indices) {
      if (!CanProveIndependent(index, loop->loop_var, &analyzer)) {
        return std::nullopt;
      }
    }

    const auto *contribution_load = update->marker->value.as<BufferLoadNode>();
    if (contribution_load == nullptr ||
        contribution_load->predicate.defined() ||
        contribution_load->indices.empty()) {
      return std::nullopt;
    }
    for (size_t i = 0; i + 1 < contribution_load->indices.size(); ++i) {
      if (!CanProveIndependent(contribution_load->indices[i], loop->loop_var,
                               &analyzer)) {
        return std::nullopt;
      }
    }
    const PrimExpr &contiguous_index = contribution_load->indices.back();
    if (!backend::reduce::CanUsePackedRamp(contiguous_index, loop->loop_var, 2,
                                           &analyzer)) {
      return std::nullopt;
    }

    const DataType scalar_dtype = update->store->buffer->dtype;
    const DataType vector_dtype = scalar_dtype.with_lanes(2);
    Buffer packed_partial =
        decl_buffer({Integer(1)}, vector_dtype,
                    update->store->buffer->name + "_vector_partial_" +
                        std::to_string(vector_partial_counter_++),
                    "local");

    Var packed_var(loop->loop_var->name_hint + "_packed",
                   loop->loop_var.dtype());
    Array<PrimExpr> vector_indices;
    vector_indices.reserve(contribution_load->indices.size());
    for (size_t i = 0; i + 1 < contribution_load->indices.size(); ++i) {
      vector_indices.push_back(
          Substitute(contribution_load->indices[i],
                     {{loop->loop_var, packed_var * Integer(2)}}));
    }
    PrimExpr ramp_base = analyzer.Simplify(Substitute(
        contiguous_index, {{loop->loop_var, packed_var * Integer(2)}}));
    vector_indices.push_back(Ramp(ramp_base, Integer(1), 2));
    PrimExpr vector_contribution =
        BufferLoad(contribution_load->buffer, std::move(vector_indices));

    PrimExpr vector_accumulator = BufferLoad(packed_partial, {Integer(0)});
    std::optional<PrimExpr> packed_combine = TryMakePackedReduceCombine(
        update->type, vector_accumulator, vector_contribution);
    if (!packed_combine.has_value()) {
      return std::nullopt;
    }

    Stmt initialize = BufferStore(
        packed_partial,
        Broadcast(MakeReduceIdentity(update->type, scalar_dtype), 2),
        {Integer(0)});
    Stmt packed_update =
        BufferStore(packed_partial, packed_combine.value(), {Integer(0)});
    Stmt packed_loop =
        For(packed_var, Integer(0), Integer(*extent / 2), loop->kind,
            std::move(packed_update), loop->thread_binding, loop->annotations,
            std::nullopt, loop->span);

    PrimExpr packed_result = BufferLoad(packed_partial, {Integer(0)});
    PrimExpr horizontal = MakeReduceCombine(
        update->type, Shuffle::ExtractElement(packed_result, 0),
        Shuffle::ExtractElement(packed_result, 1));
    PrimExpr scalar_accumulator =
        BufferLoad(update->store->buffer, update->store->indices);
    Stmt finish = BufferStore(
        update->store->buffer,
        MakeReduceCombine(update->type, scalar_accumulator, horizontal),
        update->store->indices, update->store->predicate, update->store->span);

    allocation_stack_.back().push_back(packed_partial);
    return SeqStmt(
        {std::move(initialize), std::move(packed_loop), std::move(finish)});
  }

  Target target_;
  int vector_partial_counter_{0};
  std::vector<std::vector<Buffer>> allocation_stack_;
};

} // namespace

Stmt VectorizeReducerUpdates(Stmt stmt, Target target) {
  if (!ContainsReducerUpdate(stmt)) {
    return stmt;
  }
  stmt = ReducerUpdateVectorizer(std::move(target))(std::move(stmt));
  return ReducerUpdateMarkerRemover()(std::move(stmt));
}

} // namespace tl
} // namespace tvm
