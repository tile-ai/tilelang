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
 * \file lower_opaque_block.cc
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <utility>

#include "../op/builtin.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::attr;
/*!
 * \brief Remove Block to ensure that the TIR can not be scheduled again.
 */
class OpaqueBlockLower : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc f) {
    auto fptr = f.CopyOnWrite();
    OpaqueBlockLower lower;
    if (auto existing =
            fptr->attrs.GetAttr<Map<Var, PrimExpr>>(tl::attr::kLocalVarInit)) {
      lower.local_var_init_map_ = existing.value();
    }
    lower.storage_align_ = CollectStorageAlignAnnotation(fptr->body);
    fptr->body = lower(std::move(fptr->body));
    if (!lower.local_var_init_map_.empty()) {
      f = WithAttr(std::move(f), tl::attr::kLocalVarInit,
                   lower.local_var_init_map_);
    }
    return f;
  }

private:
  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    // We have convert blocks into opaque blocks in previous passes.
    ICHECK(op->iter_values.empty())
        << "Non-opaque blocks are not allowed in FlattenBuffer. Please "
           "call pass ConvertBlocksToOpaque before.";
    // Step 1. Visit the body
    Block new_block = Downcast<Block>(this->VisitStmt(op->block));
    PrimExpr predicate = this->VisitExpr(op->predicate);
    // Step 2. Transform the `predicate` to if-then-else
    Stmt body = new_block->body;
    if (!is_one(predicate)) {
      body = IfThenElse(predicate, std::move(body));
    }
    // Step 3. Handle annotations, block annotations are not preserved by
    // default.
    std::vector<std::pair<std::string, PrimExpr>> pragma_attrs;
    HandleAnnotations(new_block->annotations, &pragma_attrs, /*is_block=*/true,
                      new_block->alloc_buffers);

    // Step 4. Handle allocations in reverse order
    for (size_t i = new_block->alloc_buffers.size(); i > 0; --i) {
      const Buffer &buffer = new_block->alloc_buffers[i - 1];
      Array<PrimExpr> allocation_shape = GetBufferAllocationShape(buffer);
      body = DeclBuffer(buffer, std::move(body));
      Map<String, ffi::Any> allocate_annotations;
      auto it = storage_align_.find(buffer->data);
      if (it != storage_align_.end()) {
        StorageAlignAnnotation allocate_aligns;
        for (auto tuple : it->second) {
          tuple.Set<0>(-1);
          allocate_aligns.push_back(tuple);
        }
        allocate_annotations.Set(tir::attr::buffer_dim_align, allocate_aligns);
      }
      auto init_it = local_var_init_map_.find(buffer->data);
      if (init_it != local_var_init_map_.end()) {
        const PrimExpr &init = (*init_it).second;
        allocate_annotations.Set(tl::attr::kLocalVarInit, init);
      }
      body = Allocate(buffer->data, buffer->dtype, allocation_shape,
                      const_true(), std::move(body), allocate_annotations);
    }
    // Step 5. Insert attribute statements converted from pragmas
    for (auto it = pragma_attrs.rbegin(); it != pragma_attrs.rend(); ++it) {
      body = AttrStmt(Integer(0), it->first, it->second, std::move(body));
    }
    return body;
  }
  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    if (block->annotations.count("stmt_group")) {
      return block->body;
    }
    return block;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Step 1. Update unit loop info.
    PrimExpr min = this->VisitExpr(op->min);
    PrimExpr extent = this->VisitExpr(op->extent);
    if (is_one(extent) && IsEffectivelyEmptyAnnotation(op->annotations)) {
      // handling unit loop
      unit_loop_vars_[op->loop_var] = min;
    }
    // Step 2. Visit recursively
    Stmt body = this->VisitStmt(op->body);
    // Step 3. Handle annotations
    std::vector<std::pair<std::string, PrimExpr>> pragma_attrs;
    Map<String, ffi::Any> new_annotations =
        HandleAnnotations(op->annotations, &pragma_attrs, /*is_block=*/false);
    // Step 4. Create new For loop accordingly
    if (op->kind == ForKind::kThreadBinding) {
      // Case 1. Thread binding
      ICHECK(op->thread_binding.defined());
      String thread_tag = op->thread_binding.value()->thread_tag;
      body = MakeLaunchThread(min, extent, op->loop_var, thread_tag, body);
    } else if (is_one(extent) &&
               IsEffectivelyEmptyAnnotation(op->annotations)) {
      // Case 2. Unit loop
      return body;
    } else {
      // Case 3. An ordinary loop
      body = For(op->loop_var, std::move(min), std::move(extent), op->kind,
                 std::move(body), std::nullopt, new_annotations);
    }
    // Step 5. Insert nested attrs
    for (auto it = pragma_attrs.rbegin(); it != pragma_attrs.rend(); ++it) {
      body = AttrStmt(op->loop_var, it->first, it->second, std::move(body));
    }
    return body;
  }

  // Treat annotations as empty if they are truly empty or contain only
  // the unroll hint `pragma_unroll_explicit`. This allows unit-length
  // loops produced by unroll pragmas to be simplified away.
  bool
  IsEffectivelyEmptyAnnotation(const Map<String, ffi::Any> &annotations) const {
    if (annotations.empty()) {
      return true;
    }
    if (annotations.size() == 1) {
      auto it = annotations.find(tir::attr::pragma_unroll_explicit);
      if (it != annotations.end()) {
        return true;
      }
    }
    return false;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = tvm::ffi::GetRef<Var>(op);
    auto it = unit_loop_vars_.find(var);
    if (it == unit_loop_vars_.end()) {
      return var;

    } else {
      PrimExpr expr = it->second;
      if (expr.dtype() != var.dtype()) {
        expr = tvm::cast(var.dtype(), std::move(expr));
      }
      return expr;
    }
  }

  static Stmt MakeLaunchThread(PrimExpr min, PrimExpr extent, Var var,
                               const String &thread_tag, Stmt body) {
    IterVar iter_var(/*dom=*/Range::FromMinExtent(std::move(min), extent),
                     /*var=*/std::move(var),
                     /*iter_type=*/IterVarType::kThreadIndex,
                     /*thread_tag=*/thread_tag);
    String attr_key = (thread_tag == "vthread" || thread_tag == "vthread.x" ||
                       thread_tag == "vthread.y" || thread_tag == "vthread.z")
                          ? tir::attr::virtual_thread
                          : tir::attr::thread_extent;
    return AttrStmt(/*node=*/std::move(iter_var),
                    /*attr_key=*/std::move(attr_key),
                    /*value=*/std::move(extent),
                    /*body=*/std::move(body));
  }

  /*! \brief Convert attr value from annotation map into PrimExpr. */
  PrimExpr ConvertAttrValue(const String &key, const Any &obj) {
    if (obj == nullptr) {
      return PrimExpr();
    } else if (auto expr = obj.try_cast<PrimExpr>()) {
      return expr.value();
    } else if (auto str = obj.try_cast<String>()) {
      return std::move(StringImm(str.value()));
    } else {
      LOG(FATAL) << "Illegal attribute of key " << key << ", value type "
                 << obj.GetTypeKey() << " not supported";
      return PrimExpr();
    }
  }

  /*!
   * \brief Helper to handle annotation dict.
   * (1) if the attr key is prefixed by `pragma_`, move to ordered kv list. They
   * are lowered to `AttrStmt` by legacy TE schedule convention.
   * (2) the non-pragma loop annotations are preserved
   * (3) the non-pragma block annotations are dropped
   * \return New annotation dict with preserved keys. Also update pragma attr
   * pairs ordered by key.
   */
  Map<String, ffi::Any>
  HandleAnnotations(const Map<String, ffi::Any> &annotations,
                    std::vector<std::pair<std::string, PrimExpr>> *pragma_attrs,
                    bool is_block,
                    const Array<Buffer> &alloc_buffers = Array<Buffer>()) {
    Map<String, ffi::Any> preserved_annotations;
    pragma_attrs->clear();
    for (const auto &kv : annotations) {
      const String &key = kv.first;
      if (tir::attr::IsPragmaKey(key)) {
        pragma_attrs->emplace_back(key, ConvertAttrValue(key, kv.second));
      } else if (key == tl::attr::kLocalVarInit) {
        if (auto local_init_map = kv.second.try_cast<Map<Var, PrimExpr>>()) {
          for (const auto &pair : local_init_map.value()) {
            local_var_init_map_.Set(pair.first, pair.second);
          }
        } else if (auto init_expr = kv.second.try_cast<PrimExpr>()) {
          ICHECK(is_block) << "`" << tl::attr::kLocalVarInit
                           << "` on non-block annotations is not supported";
          Buffer target = ResolveLocalVarBuffer(alloc_buffers);
          if (!target.defined()) {
            LOG(WARNING) << "Failed to resolve buffer for `"
                         << tl::attr::kLocalVarInit << "` annotation";
            continue;
          }
          local_var_init_map_.Set(target->data, init_expr.value());
        } else {
          LOG(FATAL) << "Expected `" << tl::attr::kLocalVarInit
                     << "` to be a PrimExpr or Map<Var, PrimExpr>, but got "
                     << kv.second.GetTypeKey();
        }
      } else if (!is_block) {
        // the loop annotation is preserved
        preserved_annotations.Set(key, kv.second);
      }
    }
    std::sort(
        pragma_attrs->begin(), pragma_attrs->end(),
        [](const auto &p1, const auto &p2) { return p1.first < p2.first; });
    return preserved_annotations;
  }

  Buffer ResolveLocalVarBuffer(const Array<Buffer> &alloc_buffers) const {
    for (const Buffer &buffer : alloc_buffers) {
      std::string scope = buffer.scope();
      if (scope.find("local.var") != std::string::npos) {
        return buffer;
      }
    }
    if (!alloc_buffers.empty()) {
      return alloc_buffers.back();
    }
    return Buffer();
  }

  /*! \brief Record the loop_var and loop start value of unit loops, whose
   * extent is one. */
  std::unordered_map<Var, PrimExpr> unit_loop_vars_;

  /*! \brief Attr keys to preserve into loop annotations. */
  std::unordered_set<std::string> preserved_annotations_;

  /*! \brief The map from buffer var to its storage alignment information. */
  std::unordered_map<Var, StorageAlignAnnotation> storage_align_;

  /*! \brief Local var initializers collected from block annotations. */
  Map<Var, PrimExpr> local_var_init_map_;
};

PrimFunc TLLowerOpaqueBlock(PrimFunc f) {
  return OpaqueBlockLower::Rewrite(std::move(f));
}

tir::transform::Pass LowerOpaqueBlock() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return TLLowerOpaqueBlock(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerOpaqueBlock", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerOpaqueBlock", LowerOpaqueBlock);
}

} // namespace tl
} // namespace tvm
