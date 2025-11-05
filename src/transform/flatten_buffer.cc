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

#include "arith/ir_mutator_with_analyzer.h"
#include "tir/transforms/ir_utils.h"
#include <tvm/arith/iter_affine_map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/data_type_rewriter.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

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
    writer->body = pass.VisitStmt(func->body);
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

  class Int64Promoter : public tir::IndexDataTypeRewriter {
  public:
    using Parent = IndexDataTypeRewriter;

    PrimExpr VisitExpr_(const VarNode *op) final {
      if (op->dtype.is_int() && op->dtype.bits() < 64) {
        return cast(DataType::Int(64), tvm::ffi::GetRef<Var>(op));
      }
      return tvm::ffi::GetRef<PrimExpr>(op);
    }

    PrimExpr VisitExpr_(const IntImmNode *op) final {
      if (op->dtype.is_int() && op->dtype.bits() < 64) {
        return IntImm(DataType::Int(64), op->value);
      }
      return tvm::ffi::GetRef<PrimExpr>(op);
    }

    PrimExpr VisitExpr_(const CastNode *op) final {
      if (op->dtype.is_int() && op->dtype.bits() < 64) {
        return cast(DataType::Int(64), op->value);
      }
      return tvm::ffi::GetRef<PrimExpr>(op);
    }

    Stmt VisitStmt_(const BufferStoreNode *op) final {
      // Force indices to be int64
      auto node = Downcast<BufferStore>(Parent::VisitStmt_(op));
      return std::move(node);
    }

    PrimExpr VisitExpr_(const BufferLoadNode *op) final {
      auto node = Downcast<BufferLoad>(Parent::VisitExpr_(op));
      return std::move(node);
    }
  };

  explicit BufferFlattener(arith::Analyzer *ana) : IRMutatorWithAnalyzer(ana) {}

  Stmt VisitStmt_(const BlockNode *op) final {
    ICHECK_EQ(op->match_buffers.size(), 0)
        << "Unexpected MatchBufferRegion found during "
           "tir.transform.FlattenBuffer.  "
        << "All MatchBufferRegion should be removed in "
           "tir.transform.LowerMatchBuffer.";

    Block block = tvm::ffi::GetRef<Block>(op);

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

  Stmt VisitStmt_(const AllocateNode *op) final {
    // Determine the flattened extents first, before stripping of
    // DeclBuffer.
    auto new_extents = [&]() -> Array<PrimExpr> {
      if (op->extents.size() == 1) {
        // No flattening required for buffers that are already flat
        return op->extents;
      }

      if (auto *decl_buffer = op->body.as<DeclBufferNode>()) {
        // N-d buffer, use the DeclBuffer inside to determine how it
        // should be flattened.
        auto &buffer = decl_buffer->buffer;
        bool matching_buffer = [&]() {
          if (!decl_buffer->buffer->data.same_as(op->buffer_var)) {
            return false;
          }
          if (op->dtype != buffer->dtype) {
            return false;
          }
          if (op->extents.size() != buffer->shape.size()) {
            return false;
          }
          ExprDeepEqual expr_equal;
          for (size_t i = 0; i < op->extents.size(); i++) {
            if (!expr_equal(op->extents[i], buffer->shape[i])) {
              return false;
            }
          }
          return true;
        }();

        if (matching_buffer) {
          Buffer flattened = GetFlattenedBuffer(buffer);
          return flattened->shape;
        } else {
          ICHECK(decl_buffer->buffer->axis_separators.empty())
              << "DeclBuffer node doesn't match Allocate extents, but also "
                 "shouldn't be "
                 "flattened to 1-d physical memory";
        }
      }

      // Fallback, this is an allocation without a matching DeclBuffer
      PrimExpr flat_extent = 1;
      for (const auto &dim : op->extents) {
        flat_extent *= dim;
      }
      return {flat_extent};
    }();

    Allocate alloc = Downcast<Allocate>(StmtExprMutator::VisitStmt_(op));

    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (alloc->dtype == DataType::Bool()) {
      alloc.CopyOnWrite()->dtype = DataType::Int(8);
    }

    if (!new_extents.same_as(alloc->extents)) {
      alloc.CopyOnWrite()->extents = new_extents;
    }
    if (!local_var_init_map_.empty()) {
      auto init_it = local_var_init_map_.find(alloc->buffer_var);
      if (init_it != local_var_init_map_.end()) {
        const PrimExpr &init = (*init_it).second;
        alloc.CopyOnWrite()->annotations.Set(tl::attr::kLocalVarInit, init);
      }
    }

    return std::move(alloc);
  }

  Stmt VisitStmt_(const DeclBufferNode *op) final {
    // TODO(rfc-70): Update the DeclBuffer node instead of
    // stripping it out.  Stripping it out in the current
    // implementation as not all lowering passes support
    // DeclBuffer.
    return VisitStmt(op->body);
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
      writer->shape.Set(i, analyzer_->canonical_simplify(flattened->shape[i]));
    }

    buffer_remap_[buf] = flattened;
    return flattened;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    bool store_returns_bool = (op->value.dtype() == DataType::Bool());
    store = VisitBufferAccess(store);

    // Handle casts from the value's dtype to the dtype of the
    // backing array.
    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (store_returns_bool) {
      ICHECK_EQ(store->buffer->dtype, DataType::Int(8))
          << "Expected int8 backing array for boolean tensor";
      auto writer = store.CopyOnWrite();
      writer->value = tvm::cast(DataType::Int(8), store->value);
      return std::move(store);
    }
    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    bool load_returns_bool = (op->dtype == DataType::Bool());
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    load = VisitBufferAccess(load);
    // Handle casts from dtype of the backing array to value's dtype.
    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (load_returns_bool && !under_address_of) {
      ICHECK_EQ(load->buffer->dtype, DataType::Int(8))
          << "Expected int8 backing array for boolean tensor";
      load.CopyOnWrite()->dtype = DataType::Int(8);
      return tvm::cast(DataType::Bool(), load);
    } else {
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
    return StmtExprMutator::VisitExpr_(op);
  }

  Array<PrimExpr> GetSimplifiedElemOffset(const Buffer &buffer,
                                          const Array<PrimExpr> &indices) {
    auto flattened_indices = buffer->ElemOffset(indices);
    Array<PrimExpr> safe_indices;
    for (auto index : flattened_indices) {
      auto int_bound = analyzer_->const_int_bound(index);
      DataType dtype = index->dtype;
      if (dtype.is_int() && dtype.bits() < 64) {
        int64_t max_value = int_bound->max_value;
        int64_t min_value = int_bound->min_value;
        const int64_t type_max = (1LL << (dtype.bits() - 1));
        const int64_t type_min = -(1LL << (dtype.bits() - 1));

        if (max_value >= (type_max - 1) || min_value < type_min) {
          Int64Promoter promoter;
          for (auto &index : flattened_indices) {
            safe_indices.push_back(promoter(index));
          }
        } else {
          safe_indices.push_back(index);
        }
      } else {
        safe_indices.push_back(index);
      }
    }
    return this->IterMapSimplifyWithContext(safe_indices, false);
  }

  template <typename Node> Node VisitBufferAccess(Node node) {
    ICHECK(node->buffer.defined());
    auto flattened_indices =
        GetSimplifiedElemOffset(node->buffer, node->indices);
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

    Array<PrimExpr> flattened_min =
        GetSimplifiedElemOffset(orig_buf, min_values);
    Array<PrimExpr> flattened_max =
        GetSimplifiedElemOffset(orig_buf, max_values);

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
};

PrimFunc FlattenBufferRewriter(PrimFunc f) {
  return BufferFlattener::Flatten(std::move(f));
}

using namespace tir::transform;
tvm::transform::Pass FlattenBuffer() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return FlattenBufferRewriter(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.FlattenBuffer", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.FlattenBuffer", FlattenBuffer);
}

} // namespace tl
} // namespace tvm
