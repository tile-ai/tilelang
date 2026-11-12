/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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

/*!
 * \file flatten_buffer.cc
 */

#include "../../3rdparty/tvm/src/tirx/ir/data_type_rewriter.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "common/index_predicate_rewriter.h"
#include "support/check.h"
#include "tir/transforms/ir_utils.h"
#include <tvm/arith/iter_affine_map.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/cast.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

/*!
 * \brief Transform multi-dimension BufferLoad/BufferStore into device-supported
 * dimension for the TIR not contains opaque block.
 */
class BufferFlattener : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Flatten(PrimFunc func) {
    arith::Analyzer ana;
    auto pass = BufferFlattener(&ana);
    if (auto init_map =
            func->attrs.GetAttr<Map<Var, PrimExpr>>(tl::attr::kLocalVarInit)) {
      pass.local_var_init_map_ = init_map.value();
    }
    auto writer = func.CopyOnWrite();
    pass.MarkBufferMapShapes(func);
    Stmt rewritten = pass.VisitStmt(func->body);
    writer->body = pass.index_predicate_rewriter_.Rewrite(rewritten);
    // The buffers in func->buffer_map are deliberately left
    // unflattened, as they are used for validation of user-provided
    // arguments.  The flattened buffers used in the updated
    // function body alias the argument buffers.
    return func;
  }

private:
  using IRMutatorWithAnalyzer::VisitExpr;
  using IRMutatorWithAnalyzer::VisitExpr_;
  using IRMutatorWithAnalyzer::VisitStmt;
  using IRMutatorWithAnalyzer::VisitStmt_;

  explicit BufferFlattener(arith::Analyzer *ana) : IRMutatorWithAnalyzer(ana) {}

  Stmt VisitStmt_(const SBlockNode *op) final {
    ICHECK_EQ(op->match_buffers.size(), 0)
        << "Unexpected MatchBufferRegion found during "
           "tir.transform.FlattenBuffer.  "
        << "All MatchBufferRegion should be removed in "
           "tir.transform.LowerMatchBuffer.";

    SBlock block = GetRef<SBlock>(op);

    Array<Buffer> alloc_buffers = op->alloc_buffers;
    alloc_buffers.MutateByApply(
        [this](const Buffer &buf) { return GetFlattenedBuffer(buf); });
    if (!alloc_buffers.same_as(op->alloc_buffers)) {
      block.CopyOnWrite()->alloc_buffers = alloc_buffers;
    }

    Array<BufferRegion> reads = op->reads;
    reads.MutateByApply([this](BufferRegion region) {
      return MutateBufferRegion(std::move(region));
    });
    if (!reads.same_as(op->reads)) {
      block.CopyOnWrite()->reads = reads;
    }

    Array<BufferRegion> writes = op->writes;
    writes.MutateByApply([this](BufferRegion region) {
      return MutateBufferRegion(std::move(region));
    });
    if (!writes.same_as(op->writes)) {
      block.CopyOnWrite()->writes = writes;
    }

    return StmtExprMutator::VisitStmt_(block.get());
  }

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    auto node = Downcast<AllocBuffer>(StmtExprMutator::VisitStmt_(op));

    auto new_buf = GetFlattenedBuffer(node->buffer);
    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (new_buf->dtype == DataType::Bool()) {
      auto writer = new_buf.CopyOnWrite();
      writer->dtype = DataType::Int(8);
    }
    if (!node->buffer.same_as(new_buf)) {
      node.CopyOnWrite()->buffer = new_buf;
    }

    if (!local_var_init_map_.empty()) {
      auto init_it = local_var_init_map_.find(node->buffer->data);
      if (init_it != local_var_init_map_.end()) {
        const PrimExpr &init = (*init_it).second;
        node.CopyOnWrite()->annotations.Set(tl::attr::kLocalVarInit, init);
      }
    }

    return std::move(node);
  }

  Stmt VisitStmt_(const DeclBufferNode *op) final {
    auto node = Downcast<DeclBuffer>(StmtExprMutator::VisitStmt_(op));

    auto new_buf = GetFlattenedBuffer(node->buffer);
    if (!node->buffer.same_as(new_buf)) {
      node.CopyOnWrite()->buffer = new_buf;
    }

    return std::move(node);
  }

  Buffer GetFlattenedBuffer(const Buffer &buf) {
    auto it = buffer_remap_.find(buf);
    if (it != buffer_remap_.end()) {
      return it->second;
    }
    auto flattened = buf.GetFlattenedBuffer();
    auto writer = flattened.CopyOnWrite();

    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (flattened->dtype == DataType::Bool()) {
      writer->dtype = DataType::Int(8);
    }
    // canonicalize shape
    for (size_t i = 0; i < flattened->shape.size(); ++i) {
      const PrimExpr &shape = flattened->shape[i];
      // Preserve recorded access nodes until their guards have been rewritten.
      // The simplifier can clone loads while canonicalizing this metadata.
      if (!index_predicate_rewriter_.HasPromotedAccess(shape)) {
        writer->shape.Set(i, analyzer_->canonical_simplify(shape));
      }
    }
    // Flattened indices already include buf->elem_offset (see
    // VisitBufferAccess). Zero elem_offset so later passes (e.g.
    // Buffer::access_ptr) use index as the sole offset and do not add
    // buffer->elem_offset again.
    writer->elem_offset = make_const(flattened->DefaultIndexType(), 0);

    buffer_remap_[buf] = flattened;
    return flattened;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    bool store_returns_bool = (op->value.dtype() == DataType::Bool());
    IndexPredicateRewriter::Promoter promoter;
    store = VisitBufferAccess(store, promoter);

    // Handle casts from the value's dtype to the dtype of the
    // backing array.
    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (store_returns_bool) {
      ICHECK_EQ(store->buffer->dtype, DataType::Int(8))
          << "Expected int8 backing array for boolean tensor";
      auto writer = store.CopyOnWrite();
      writer->value = tvm::cast(DataType::Int(8), store->value);
    }
    index_predicate_rewriter_.Record(store, promoter);
    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    bool load_returns_bool = (op->dtype == DataType::Bool());
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    IndexPredicateRewriter::Promoter promoter;
    load = VisitBufferAccess(load, promoter);
    // Handle casts from dtype of the backing array to value's dtype.
    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (load_returns_bool && !under_address_of) {
      ICHECK_EQ(load->buffer->dtype, DataType::Int(8))
          << "Expected int8 backing array for boolean tensor";
      load.CopyOnWrite()->dtype = DataType::Int(8);
      index_predicate_rewriter_.Record(load, promoter);
      return tvm::cast(DataType::Bool(), load);
    } else {
      index_predicate_rewriter_.Record(load, promoter);
      return std::move(load);
    }
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::address_of())) {
      under_address_of = true;
      auto result = StmtExprMutator::VisitExpr_(op);
      under_address_of = false;
      return result;
    }
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (call->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_GE(call->args.size(), 4)
          << "tvm_access_ptr must have offset and extent arguments";
      IndexPredicateRewriter::Promoter promoter;
      PrimExpr original_offset = call->args[2];
      PrimExpr offset =
          index_predicate_rewriter_.PrepareIndex(original_offset, promoter);
      if (NeedsInt64Promotion(offset)) {
        PrimExpr promoted = promoter(offset);
        promoter.RecordExtent(original_offset, promoted, call->args[3]);
        promoter.RecordExtent(offset, promoted, call->args[3]);
        offset = promoted;
      }
      if (!offset.same_as(call->args[2])) {
        call.CopyOnWrite()->args.Set(2, offset);
      }
      index_predicate_rewriter_.Record(call, promoter);
    }
    return std::move(call);
  }

  bool NeedsInt64Promotion(const PrimExpr &index) {
    DataType dtype = index->dtype;
    if (!dtype.is_int() || dtype.bits() >= 64) {
      return false;
    }

    auto int_bound = analyzer_->const_int_bound(index);
    int64_t max_value = int_bound->max_value;
    int64_t min_value = int_bound->min_value;
    const int64_t type_max = (1LL << (dtype.bits() - 1));
    const int64_t type_min = -(1LL << (dtype.bits() - 1));
    return max_value >= (type_max - 1) || min_value < type_min;
  }

  Array<PrimExpr>
  GetSimplifiedElemOffset(const Buffer &buffer, const Array<PrimExpr> &indices,
                          IndexPredicateRewriter::Promoter &promoter) {
    auto prepared_indices = indices.Map([&](const PrimExpr &index) {
      return index_predicate_rewriter_.PrepareIndex(index, promoter);
    });
    auto flattened_indices = buffer->ElemOffset(prepared_indices);
    Array<PrimExpr> safe_indices;
    for (const auto &original_index : flattened_indices) {
      PrimExpr index =
          index_predicate_rewriter_.PrepareIndex(original_index, promoter);
      if (NeedsInt64Promotion(index)) {
        PrimExpr promoted = promoter(index);
        promoter.RecordIndex(original_index, promoted);
        promoter.RecordIndex(index, promoted);
        safe_indices.push_back(promoted);
      } else {
        safe_indices.push_back(index);
      }
    }
    return this->IterMapSimplifyWithContext(safe_indices, false);
  }

  template <typename Node>
  Node VisitBufferAccess(Node node,
                         IndexPredicateRewriter::Promoter &promoter) {
    ICHECK(node->buffer.defined());
    auto flattened_indices =
        GetSimplifiedElemOffset(node->buffer, node->indices, promoter);
    Buffer flattened_buffer = GetFlattenedBuffer(node->buffer);

    auto writer = node.CopyOnWrite();
    writer->buffer = flattened_buffer;
    writer->indices = flattened_indices;
    return node;
  }

  BufferRegion MutateBufferRegion(BufferRegion region) {
    Buffer orig_buf = region->buffer;
    Buffer flattened_buf = GetFlattenedBuffer(orig_buf);
    if (flattened_buf.same_as(orig_buf)) {
      return region;
    }

    Array<PrimExpr> min_values;
    Array<PrimExpr> max_values;
    for (const auto &range : region->region) {
      min_values.push_back(range->min);
      max_values.push_back(range->min + range->extent - 1);
    }

    IndexPredicateRewriter::Promoter promoter;
    Array<PrimExpr> flattened_min =
        GetSimplifiedElemOffset(orig_buf, min_values, promoter);
    Array<PrimExpr> flattened_max =
        GetSimplifiedElemOffset(orig_buf, max_values, promoter);

    Array<Range> flattened_ranges;
    ICHECK_EQ(flattened_min.size(), flattened_max.size());
    for (size_t i = 0; i < flattened_min.size(); i++) {
      flattened_ranges.push_back(Range(flattened_min[i], flattened_max[i] + 1));
    }

    return BufferRegion(flattened_buf, flattened_ranges);
  }

  /*! \brief Whether the current buffer is under address_of */
  bool under_address_of = false;
  /*! \brief Map of buffers being remapped. */
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffer_remap_;

  /*! \brief The updated external buffer map. */
  Map<Var, Buffer> updated_extern_buffer_map_;

  /*! \brief Local var initializers preserved from block annotations. */
  Map<Var, PrimExpr> local_var_init_map_;

  IndexPredicateRewriter index_predicate_rewriter_;
};

PrimFunc FlattenBufferRewriter(PrimFunc f) {
  return BufferFlattener::Flatten(std::move(f));
}

using namespace tirx::transform;
tvm::transform::Pass FlattenBuffer() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return FlattenBufferRewriter(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.FlattenBuffer", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.FlattenBuffer", FlattenBuffer);
}

} // namespace tl
} // namespace tvm
