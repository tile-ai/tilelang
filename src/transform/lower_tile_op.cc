/*!
 * \file lower_tile_op.cc
 * \brief Lower the tile op for further codegen.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>
#include <unordered_map>
#include <vector>

#include "../layout/layout.h"
#include "../layout/utils.h"
#include "../op/builtin.h"
#include "../op/gemm.h"
#include "../op/gemm_sp.h"
#include "../op/operator.h"
#include "../op/utils.h"
#include "../target/utils.h"

#include "arith/ir_mutator_with_analyzer.h"
#include "layout_reducer.h"
#include "loop_partition.h"

namespace tvm {
namespace tl {

using namespace tir;

static Buffer makeBufferWithLayout(const Buffer &buffer, const Layout &layout,
                                   Map<Var, Var> &var_remap) {
  const auto *ptr_type =
      TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
  Type new_type;
  // convert fragments to normal local buffer
  if (ptr_type->storage_scope == "local.fragment") {
    new_type = PointerType(ptr_type->element_type, "local");
  } else {
    new_type = buffer->data->type_annotation;
  }
  Var new_var;
  if (ptr_type->storage_scope == "global") {
    new_var = buffer->data;
  } else {
    if (var_remap.count(buffer->data)) {
      new_var = var_remap[buffer->data];
    } else {
      new_var = Var(buffer->data->name_hint, new_type);
      var_remap.Set(buffer->data, new_var);
    }
  }
  Array<PrimExpr> layout_shape = layout->OutputShape();
  Array<PrimExpr> output_shape = layout_shape;
  if (ptr_type->storage_scope == "shared" ||
      ptr_type->storage_scope == "shared.dyn") {
    int replicate_extent = 1;
    Array<PrimExpr> buffer_shape = buffer->shape;
    int buffer_extent = 1;
    int layout_extent = 1;
    for (size_t i = 0; i < buffer_shape.size(); i++) {
      auto shape = buffer_shape[i].as<IntImmNode>();
      buffer_extent *= shape->value;
    }
    for (size_t i = 0; i < layout_shape.size(); i++) {
      auto shape = layout_shape[i].as<IntImmNode>();
      ICHECK(shape) << "Layout output shape must be constant integer, but got: "
                    << layout_shape[i];
      layout_extent *= shape->value;
    }
    replicate_extent = buffer_extent / layout_extent;
    if (replicate_extent > 1) {
      output_shape.insert(output_shape.begin(), replicate_extent);
    }
  }
  return Buffer(new_var, buffer->dtype, output_shape, {}, buffer->elem_offset,
                buffer->name, buffer->data_alignment, buffer->offset_factor,
                buffer->buffer_type);
}

// The function `makeBufferWithLayout` creates a new Buffer object based on the
// given buffer and layout. It handles remapping of buffer variables, adjusts
// the storage scope if needed (e.g., from "local.fragment" to "local"), and
// computes the output shape according to the layout. For shared memory buffers,
// it also handles replication if the buffer's extent is larger than the
// layout's extent.
class LayoutRemapRewriter : public arith::IRMutatorWithAnalyzer {
public:
  static Stmt Substitute(Stmt stmt, Map<Buffer, Layout> layout_remap) {
    arith::Analyzer analyzer;
    LayoutRemapRewriter substituter(&analyzer);
    substituter.layout_remap_ = std::move(layout_remap);
    return substituter.VisitStmt(stmt);
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Stmt VisitStmt_(const BlockNode *op) final {
    auto block = Downcast<Block>(arith::IRMutatorWithAnalyzer::VisitStmt_(op));
    if (op->annotations.count(attr::kLayoutMap)) {
      block.CopyOnWrite()->annotations.Set(attr::kLayoutMap, layout_remap_);
    }
    return block;
  }

  Map<Buffer, Layout> layout_remap_;
};

/*!
 * \brief A class that rewrites buffer references in a statement based on a
 * given buffer remapping.
 *
 * This class is used to update buffer references in a statement after buffer
 * transformations have been applied. It specifically handles the remapping of
 * padding annotations.
 */
class RemapBufferRewriter : public arith::IRMutatorWithAnalyzer {
public:
  /*!
   * \brief Substitute buffer references in a statement based on a given buffer
   * remapping. \param stmt The statement to rewrite. \param buffer_remap A map
   * from old buffers to new buffers. \return The rewritten statement.
   */
  static Stmt Substitute(const Stmt &stmt, Map<Buffer, Buffer> buffer_remap) {
    arith::Analyzer analyzer;
    RemapBufferRewriter substituter(&analyzer);
    substituter.buffer_remap_ = std::move(buffer_remap);
    return substituter.VisitStmt(stmt);
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Stmt VisitStmt_(const BlockNode *op) final {
    if (op->annotations.count(attr::kSafeValueMap)) {
      return RewritePaddingMap(op);
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  /*!
   * \brief Rewrite the padding map annotation of a block.
   * \param op The block node to rewrite.
   * \return The rewritten block.
   */
  Stmt RewritePaddingMap(const BlockNode *op) {
    auto safe_value_map = op->annotations.Get(attr::kSafeValueMap);
    if (!safe_value_map) {
      LOG(FATAL) << "Padding map annotation is missing";
    }

    Map<Var, Var> var_remap = CreateVarRemap();
    Map<Var, PrimExpr> new_safe_value_map = RemapPaddingMap(
        Downcast<Map<Var, PrimExpr>>(safe_value_map.value()), var_remap);

    auto block = Downcast<Block>(IRMutatorWithAnalyzer::VisitStmt_(op));
    auto block_ptr = block.CopyOnWrite();
    block_ptr->annotations.Set(attr::kSafeValueMap, new_safe_value_map);
    return block;
  }

  /*!
   * \brief Create a mapping from old variables to new variables based on buffer
   * remapping. \return A map from old variables to new variables.
   */
  Map<Var, Var> CreateVarRemap() const {
    Map<Var, Var> var_remap;
    for (const auto &[buffer, buffer_remap] : buffer_remap_) {
      var_remap.Set(buffer->data, buffer_remap->data);
    }
    return var_remap;
  }

  /*!
   * \brief Remap the padding map using the variable remapping.
   * \param safe_value_map The original padding map.
   * \param var_remap The variable remapping.
   * \return The remapped padding map.
   */
  Map<Var, PrimExpr> RemapPaddingMap(const Map<Var, PrimExpr> &safe_value_map,
                                     const Map<Var, Var> &var_remap) const {
    Map<Var, PrimExpr> new_safe_value_map;
    for (const auto &[var, padding] : safe_value_map) {
      if (var_remap.count(var)) {
        new_safe_value_map.Set(var_remap.at(var), padding);
      } else {
        new_safe_value_map.Set(var, padding);
      }
    }
    return new_safe_value_map;
  }

  Map<Buffer, Buffer> buffer_remap_;
};

class LowerTileOpPass : arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    LowerTileOpPass substituter(&analyzer);
    // Trace the buffer map for tvm_access_ptr
    // Insert both handle var and data var as keys for lookup
    for (const auto &[param_var, buffer] : f->buffer_map) {
      substituter.buffer_map_.insert(
          {param_var, buffer}); // handle key (e.g., dQ_handle)
      substituter.buffer_map_.insert(
          {buffer->data, buffer}); // data key (e.g., dQ)
    }
    for (const auto &[_, buffer] : f->buffer_map) {
      substituter.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined()) << "LowerTileOpPass: Require the target attribute";
    substituter.target_ = target.value();
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    fptr->body =
        RemapBufferRewriter::Substitute(fptr->body, substituter.buffer_remap_);
    fptr->body =
        LayoutRemapRewriter::Substitute(fptr->body, substituter.layout_remap_);
    tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
    Optional<Bool> opt_disable_tma_lower =
        ctxt->GetConfig(kDisableTMALower, Optional<Bool>());

    if (!opt_disable_tma_lower.value_or(Bool(false))) {
      // @lei: this is a workaround, as if we don't disable tma lower,
      // cp async lowering won't be generated.
      ctxt->config.Set(kDisableTMALower, Bool(!substituter.has_tma_));
    }
    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Stmt VisitStmt_(const BlockNode *op) final {
    // Record the mapping from buffer data var to buffer for later lookup
    for (auto buffer : op->alloc_buffers) {
      buffer_map_.insert({buffer->data, buffer});
    }
    for (auto match_buffer : op->match_buffers) {
      buffer_map_.insert({match_buffer->buffer->data, match_buffer->buffer});
    }
    for (auto buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    Map<Var, Layout> vmap;
    if (op->annotations.count(attr::kLayoutMap)) {
      auto layout_map = op->annotations.at(attr::kLayoutMap)
                            .as<Map<Buffer, Layout>>()
                            .value();
      for (auto [buffer, layout] : layout_map) {
        buffer_remap_.Set(buffer,
                          makeBufferWithLayout(buffer, layout, var_remap_));
        layout_map_.Set(buffer, layout);
      }
    }
    // Begin a new workspace collection frame for this block scope
    workspace_stack_.emplace_back();

    auto block = Downcast<Block>(arith::IRMutatorWithAnalyzer::VisitStmt_(op));
    auto block_ptr = block.CopyOnWrite();
    for (size_t i = 0; i < block->alloc_buffers.size(); i++) {
      auto buffer = block->alloc_buffers[i];
      if (buffer_remap_.count(buffer)) {
        block_ptr->alloc_buffers.Set(i, buffer_remap_[buffer]);
      }
    }
    // Attach any workspaces requested within this block to its alloc_buffers
    if (!workspace_stack_.empty()) {
      for (const auto &buffer : workspace_stack_.back()) {
        block_ptr->alloc_buffers.push_back(buffer);
      }
      workspace_stack_.pop_back();
    }
    return block;
  }

  int CheckAndGetBufferRowSize(const Buffer &buffer) {
    CHECK(buffer->shape.size() >= 2)
        << "The dimension of Buffer \"" << buffer->name << "\" with shape "
        << buffer->shape << " should be at least 2";

    auto dim = buffer->shape.size();
    auto buffer_row_size = buffer->shape[dim - 1].as<IntImmNode>()->value;
    return buffer_row_size;
  }

  struct AccessPtrResult {
    PrimExpr expr;
    bool rewritten{false};
  };

  AccessPtrResult
  HandleAccessPtrAndOffset(const PrimExpr &access_ptr,
                           const Optional<PrimExpr> &offset = std::nullopt,
                           DataType dtype = DataType::Int(32)) {
    AccessPtrResult result{access_ptr, false};
    // The 2th arg of T.tvm_access_ptr call is offset, we set it to 0 and
    // accumulate it to smem_offset
    CHECK(access_ptr->IsInstance<CallNode>())
        << "Invalid access ptr for permuted layout: " << access_ptr;
    auto access_ptr_call = Downcast<Call>(access_ptr);
    if (access_ptr_call->op.same_as(builtin::tvm_access_ptr())) {
      // tvm_access_ptr format: (dtype, data, offset, extent, rw_mask)
      auto buffer_var = Downcast<Var>(access_ptr_call->args[1]);

      // Find original buffer from buffer_map_ using buffer_var
      auto it = buffer_map_.find(buffer_var);
      if (it == buffer_map_.end()) {
        // If not found, buffer_var might be a new var after remap
        // Do reverse lookup in var_remap_
        for (const auto &[old_var, new_var] : var_remap_) {
          if (new_var.same_as(buffer_var)) {
            it = buffer_map_.find(old_var);
            break;
          }
        }
      }

      if (it == buffer_map_.end()) {
        return result; // Buffer not found, no transformation needed
      }

      Buffer original_buffer = it->second;

      // Check if this buffer has a layout
      if (!layout_map_.count(original_buffer)) {
        return result; // No layout, no transformation needed
      }

      Layout layout = layout_map_[original_buffer];
      Buffer new_buffer = buffer_remap_[original_buffer];

      // In TMA context, swizzle is encoded in TMA descriptor parameters
      // rather than in memory indices, so we only update buffer data
      // without recomputing indices.
      if (in_tma_context_) {
        Array<PrimExpr> new_args = access_ptr_call->args;
        new_args.Set(1, new_buffer->data); // Only replace data var
        layout_remap_.Set(new_buffer, layout);
        result.rewritten = true;
        result.expr =
            Call(access_ptr_call->dtype, access_ptr_call->op, new_args,
                 access_ptr_call->annotations, access_ptr_call->span);
        return result;
      }

      // Get the offset from tvm_access_ptr args[2]
      PrimExpr elem_offset = access_ptr_call->args[2];
      if (offset.defined()) {
        elem_offset = elem_offset + offset.value();
      }
      // Get original and new buffer shapes
      Array<PrimExpr> old_shape = original_buffer->shape;
      Array<PrimExpr> new_shape = new_buffer->shape;
      // Convert linear offset to multi-dimensional indices
      Array<PrimExpr> multi_dim_indices;
      PrimExpr remaining_offset = elem_offset;
      for (int i = static_cast<int>(old_shape.size()) - 1; i >= 0; --i) {
        multi_dim_indices.insert(
            multi_dim_indices.begin(),
            analyzer_->Simplify(floormod(remaining_offset, old_shape[i])));
        remaining_offset = floordiv(remaining_offset, old_shape[i]);
      }
      // Apply layout transformation
      auto forward_indices = layout->Forward(multi_dim_indices);
      PrimExpr new_offset = 0;
      PrimExpr stride_offset = 1;
      for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i) {
        new_offset += forward_indices[i] * stride_offset;
        stride_offset *= new_shape[i];
      }
      // Verify that access is within a single tile
      ICHECK(is_zero(analyzer_->Simplify(remaining_offset)))
          << "Access offset exceeds tile bounds, remaining_offset: "
          << remaining_offset;
      new_offset = analyzer_->Simplify(new_offset);
      Array<PrimExpr> new_indices;
      layout_remap_.Set(new_buffer, layout);

      // Build new tvm_access_ptr call with new buffer and offset
      Array<PrimExpr> new_args = access_ptr_call->args;
      new_args.Set(1, new_buffer->data); // Replace data var
      new_args.Set(2, new_offset);       // Replace offset
      result.rewritten = true;
      result.expr = Call(access_ptr_call->dtype, access_ptr_call->op, new_args,
                         access_ptr_call->annotations, access_ptr_call->span);
      return result;
    } else if (access_ptr_call->op.same_as(builtin::address_of())) {
      Optional<PrimExpr> resolved = ResolveBufferLoad(access_ptr_call->args[0]);
      ICHECK(resolved.defined())
          << "Invalid access op for permuted layout: " << access_ptr;
      PrimExpr load_expr = resolved.value();
      if (!load_expr.same_as(access_ptr_call->args[0])) {
        auto node = access_ptr_call.CopyOnWrite();
        node->args.Set(0, load_expr);
        access_ptr_call =
            Call(access_ptr_call->dtype, access_ptr_call->op, {load_expr},
                 access_ptr_call->annotations, access_ptr_call->span);
      }
      BufferLoad load = Downcast<BufferLoad>(access_ptr_call->args[0]);
      Array<PrimExpr> indices = load->indices;
      Array<PrimExpr> old_shape = load->buffer->shape;

      CHECK_EQ(indices.size(), old_shape.size())
          << "Indices size and shape size must match for general N-dimensional "
             "buffer "
          << "but got indices size: " << indices.size()
          << " and shape size: " << old_shape.size();

      Buffer remap_key = FindRemapBuffer(load->buffer).value_or(load->buffer);
      Optional<Layout> layout = FindLayout(remap_key);
      if (!layout.defined() || !buffer_map_.count(remap_key->data)) {
        return result;
      }
      auto new_buffer = buffer_remap_.count(remap_key)
                            ? buffer_remap_[remap_key]
                            : load->buffer;
      auto new_shape = new_buffer->shape;

      // In TMA context, swizzle is encoded in TMA descriptor parameters
      // rather than in memory indices, so we only update buffer data
      // without recomputing indices.
      if (in_tma_context_) {
        Array<PrimExpr> new_args = {BufferLoad(new_buffer, indices)};
        if (buffer_remap_.count(remap_key)) {
          layout_remap_.Set(new_buffer, layout.value());
        }
        result.rewritten = true;
        result.expr =
            Call(access_ptr_call->dtype, access_ptr_call->op, new_args,
                 access_ptr_call->annotations, access_ptr_call->span);
        return result;
      }

      PrimExpr elem_offset = 0;
      PrimExpr stride = 1;

      for (int i = static_cast<int>(old_shape.size()) - 1; i >= 0; --i) {
        elem_offset += indices[i] * stride;
        stride *= old_shape[i];
      }

      PrimExpr smem_offset =
          elem_offset + (offset.defined() ? offset.value() : 0);

      auto buffer_map_iter = buffer_map_.find(Downcast<Var>(remap_key->data));

      int buffer_row_size = CheckAndGetBufferRowSize(buffer_map_iter->second);
      (void)buffer_row_size;

      // Convert offset to target-dimension, reindex it and convert it back
      Array<PrimExpr> multi_dim_indices;
      PrimExpr remaining_offset = smem_offset;

      for (int i = static_cast<int>(old_shape.size()) - 1; i >= 0; --i) {
        multi_dim_indices.insert(multi_dim_indices.begin(),
                                 floormod(remaining_offset, old_shape[i]));
        remaining_offset = floordiv(remaining_offset, old_shape[i]);
      }

      auto forward_indices = layout.value()->Forward(multi_dim_indices);
      PrimExpr new_offset = 0;
      PrimExpr stride_offset = 1;
      for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i) {
        new_offset += forward_indices[i] * stride_offset;
        stride_offset *= new_shape[i];
      }
      new_offset = analyzer_->Simplify(new_offset);

      Array<PrimExpr> new_indices;
      for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i) {
        new_indices.insert(new_indices.begin(),
                           floormod(new_offset, new_shape[i]));
        new_offset = floordiv(new_offset, new_shape[i]);
      }

      Array<PrimExpr> new_args = {BufferLoad(new_buffer, new_indices)};
      if (buffer_remap_.count(remap_key)) {
        layout_remap_.Set(new_buffer, layout.value());
      }
      result.rewritten = true;
      result.expr = Call(access_ptr_call->dtype, access_ptr_call->op, new_args,
                         access_ptr_call->annotations, access_ptr_call->span);
      return result;
    } else {
      LOG(FATAL) << "Invalid access op for permuted layout: " << access_ptr;
    }

    return result;
  }

  Optional<PrimExpr> ResolveBufferLoad(const PrimExpr &expr) const {
    if (expr->IsInstance<BufferLoadNode>()) {
      return expr;
    }
    if (const auto *var_node = expr.as<VarNode>()) {
      Var var = tvm::ffi::GetRef<Var>(var_node);
      auto it = let_bindings_.find(var);
      if (it != let_bindings_.end()) {
        return it->second;
      }
    }
    return Optional<PrimExpr>();
  }

  Optional<Buffer> FindRemapBuffer(const Buffer &buffer) const {
    if (buffer_remap_.count(buffer)) {
      return buffer;
    }
    auto it = buffer_map_.find(buffer->data);
    if (it != buffer_map_.end() && buffer_remap_.count(it->second)) {
      return it->second;
    }
    for (const auto &kv : buffer_remap_) {
      if (kv.first->data.same_as(buffer->data)) {
        return kv.first;
      }
      if (kv.first->name == buffer->name) {
        return kv.first;
      }
    }
    return Optional<Buffer>();
  }

  Optional<Layout> FindLayout(const Buffer &buffer) const {
    if (layout_map_.count(buffer)) {
      return layout_map_[buffer];
    }
    auto it = buffer_map_.find(buffer->data);
    if (it != buffer_map_.end() && layout_map_.count(it->second)) {
      return layout_map_[it->second];
    }
    for (const auto &kv : layout_map_) {
      if (kv.first->data.same_as(buffer->data)) {
        return kv.second;
      }
      if (kv.first->name == buffer->name) {
        return kv.second;
      }
    }
    return Optional<Layout>();
  }

  PrimExpr VisitExpr_(const tir::CallNode *op) final {
    if (op->op.same_as(tl::tma_load()) ||
        op->op.same_as(tl::tma_load_im2col()) ||
        op->op.same_as(tl::tma_store())) {
      // skip tma related calls, as they were transformed implicitly.
      has_tma_ = true;
      in_tma_context_ = true;
      auto call = Downcast<Call>(IRMutatorWithAnalyzer::VisitExpr_(op));
      in_tma_context_ = false;
      return call;
    }

    if (is_ptx_) {
      return Downcast<Call>(op);
    }

    // Handle ptx_ldmatrix
    if (op->op.same_as(builtin::ptx_ldmatrix())) {
      is_ptx_ = true;
      auto call = Downcast<Call>(IRMutatorWithAnalyzer::VisitExpr_(op));
      is_ptx_ = false;
      // form: T.ptx_ldmatrix(..., smem_ptr, smem_offset)
      // smem_ptr: T.tvm_access_ptr(ptype, data, offset, extent, rw_mask)
      // or T.address_of(buffer, offset)
      PrimExpr access_ptr = call->args[5];
      PrimExpr smem_offset = call->args[6];
      Call access_ptr_call = Downcast<Call>(access_ptr);

      // Handle both tvm_access_ptr and address_of
      if (access_ptr_call->op.same_as(builtin::tvm_access_ptr())) {
        auto new_access_ptr =
            HandleAccessPtrAndOffset(access_ptr, smem_offset, call->dtype);
        if (new_access_ptr.rewritten) {
          auto new_call = call.CopyOnWrite();
          new_call->args.Set(5, new_access_ptr.expr);
          new_call->args.Set(6, IntImm(smem_offset->dtype, 0));
        }
      } else if (access_ptr_call->op.same_as(builtin::address_of())) {
        Optional<PrimExpr> resolved =
            ResolveBufferLoad(access_ptr_call->args[0]);
        ICHECK(resolved.defined())
            << "Invalid address_of argument for permuted layout: "
            << access_ptr_call->args[0];
        PrimExpr load_expr = resolved.value();
        if (!load_expr.same_as(access_ptr_call->args[0])) {
          auto call_node = call.CopyOnWrite();
          call_node->args.Set(
              5, Call(access_ptr_call->dtype, access_ptr_call->op, {load_expr},
                      access_ptr_call->annotations, access_ptr_call->span));
          access_ptr_call = Downcast<Call>(call->args[5]);
          access_ptr = call->args[5];
        }
        auto new_access_ptr =
            HandleAccessPtrAndOffset(access_ptr, smem_offset, call->dtype);
        if (new_access_ptr.rewritten) {
          auto new_call = call.CopyOnWrite();
          new_call->args.Set(5, new_access_ptr.expr);
          new_call->args.Set(6, IntImm(smem_offset->dtype, 0));
        }
      } else {
        LOG(FATAL) << "Invalid access ptr for permuted layout: " << access_ptr;
      }
      return call;
    }

    if (op->op.same_as(tl::ptx_ldmatrix())) {
      is_ptx_ = true;
      auto call = Downcast<Call>(IRMutatorWithAnalyzer::VisitExpr_(op));
      is_ptx_ = false;
      // form: T.ptx_ldmatrix(..., smem_ptr, smem_offset)
      // smem_ptr: T.tvm_access_ptr(ptype, data, offset, extent, rw_mask)
      // or T.address_of(buffer, offset)
      PrimExpr access_ptr = call->args[2];
      Call access_ptr_call = Downcast<Call>(access_ptr);

      // Handle both tvm_access_ptr and address_of
      if (access_ptr_call->op.same_as(builtin::tvm_access_ptr())) {
        auto new_access_ptr =
            HandleAccessPtrAndOffset(access_ptr, std::nullopt, call->dtype);
        if (new_access_ptr.rewritten) {
          auto new_call = call.CopyOnWrite();
          new_call->args.Set(2, new_access_ptr.expr);
        }
      } else if (access_ptr_call->op.same_as(builtin::address_of())) {
        Optional<PrimExpr> resolved =
            ResolveBufferLoad(access_ptr_call->args[0]);
        ICHECK(resolved.defined())
            << "Invalid address_of argument for permuted layout: "
            << access_ptr_call->args[0];
        PrimExpr load_expr = resolved.value();
        if (!load_expr.same_as(access_ptr_call->args[0])) {
          auto call_node = call.CopyOnWrite();
          call_node->args.Set(
              2, Call(access_ptr_call->dtype, access_ptr_call->op, {load_expr},
                      access_ptr_call->annotations, access_ptr_call->span));
          access_ptr_call = Downcast<Call>(call->args[2]);
          access_ptr = call->args[2];
        }
        auto new_access_ptr =
            HandleAccessPtrAndOffset(access_ptr, std::nullopt, call->dtype);
        if (new_access_ptr.rewritten) {
          auto new_call = call.CopyOnWrite();
          new_call->args.Set(2, new_access_ptr.expr);
        }
      } else {
        LOG(FATAL) << "Invalid access ptr for permuted layout: " << access_ptr;
      }
      return call;
    }

    // Handle tl::ptx_stmatrix
    if (op->op.same_as(tl::ptx_stmatrix())) {
      is_ptx_ = true;
      auto call = Downcast<Call>(IRMutatorWithAnalyzer::VisitExpr_(op));
      is_ptx_ = false;
      // form: T.ptx_stmatrix(trans, num, smem_ptr, value0, value1, ...)
      // smem_ptr: T.tvm_access_ptr(ptype, data, offset, extent, rw_mask)
      // or T.address_of(buffer, offset)
      PrimExpr access_ptr = call->args[2];
      Call access_ptr_call = Downcast<Call>(access_ptr);

      // Handle both tvm_access_ptr and address_of
      if (access_ptr_call->op.same_as(builtin::tvm_access_ptr())) {
        auto new_access_ptr =
            HandleAccessPtrAndOffset(access_ptr, std::nullopt, call->dtype);
        if (new_access_ptr.rewritten) {
          auto new_call = call.CopyOnWrite();
          new_call->args.Set(2, new_access_ptr.expr);
        }
      } else if (access_ptr_call->op.same_as(builtin::address_of())) {
        Optional<PrimExpr> resolved =
            ResolveBufferLoad(access_ptr_call->args[0]);
        ICHECK(resolved.defined())
            << "Invalid address_of argument for permuted layout: "
            << access_ptr_call->args[0];
        PrimExpr load_expr = resolved.value();
        if (!load_expr.same_as(access_ptr_call->args[0])) {
          auto call_node = call.CopyOnWrite();
          call_node->args.Set(
              2, Call(access_ptr_call->dtype, access_ptr_call->op, {load_expr},
                      access_ptr_call->annotations, access_ptr_call->span));
          access_ptr_call = Downcast<Call>(call->args[2]);
          access_ptr = call->args[2];
        }
        auto new_access_ptr =
            HandleAccessPtrAndOffset(access_ptr, std::nullopt, call->dtype);
        if (new_access_ptr.rewritten) {
          auto new_call = call.CopyOnWrite();
          new_call->args.Set(2, new_access_ptr.expr);
        }
      } else {
        LOG(FATAL) << "Invalid access ptr for permuted layout: " << access_ptr;
      }
      return call;
    }

    // Handle mma_store
    if (op->op.same_as(builtin::mma_store())) {
      is_ptx_ = true;
      auto call = Downcast<Call>(IRMutatorWithAnalyzer::VisitExpr_(op));
      is_ptx_ = false;
      // because we will directly store result to Buffer instead of calling
      // mma_store now
      auto access_ptr = call->args[2];
      auto new_access_ptr =
          HandleAccessPtrAndOffset(access_ptr, std::nullopt, call->dtype);
      if (new_access_ptr.rewritten) {
        auto new_call = call.CopyOnWrite();
        new_call->args.Set(2, new_access_ptr.expr);
      }
      return call;
    }

    // Handle standalone tvm_access_ptr calls with layout transformation
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      auto call = Downcast<Call>(IRMutatorWithAnalyzer::VisitExpr_(op));
      auto new_access_ptr =
          HandleAccessPtrAndOffset(call, std::nullopt, call->dtype);
      if (new_access_ptr.rewritten) {
        return new_access_ptr.expr;
      }
      return call;
    }

    // Default: visit normally
    auto call = Downcast<Call>(IRMutatorWithAnalyzer::VisitExpr_(op));
    return call;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto load = Downcast<BufferLoad>(IRMutatorWithAnalyzer::VisitExpr_(op));
    if (is_ptx_) {
      return load;
    }
    auto buffer = load->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_indices = layout_map_[buffer]->Forward(load->indices);
      auto new_buffer = buffer_remap_[load->buffer];
      layout_remap_.Set(new_buffer, layout_map_[load->buffer]);
      return BufferLoad(new_buffer, new_indices);
    } else if (var_remap_.count(buffer->data)) {
      auto new_buffer = Buffer(
          var_remap_[buffer->data], buffer->dtype, buffer->shape,
          buffer->strides, buffer->elem_offset, buffer->name,
          buffer->data_alignment, buffer->offset_factor, buffer->buffer_type);
      return BufferLoad(new_buffer, load->indices);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(IRMutatorWithAnalyzer::VisitStmt_(op));
    auto buffer = store->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_indices = layout_map_[buffer]->Forward(store->indices);
      auto new_buffer = buffer_remap_[store->buffer];
      layout_remap_.Set(new_buffer, layout_map_[store->buffer]);
      return BufferStore(new_buffer, store->value, new_indices);
    } else if (var_remap_.count(buffer->data)) {
      auto new_buffer = Buffer(
          var_remap_[buffer->data], buffer->dtype, buffer->shape,
          buffer->strides, buffer->elem_offset, buffer->name,
          buffer->data_alignment, buffer->offset_factor, buffer->buffer_type);
      return BufferStore(new_buffer, store->value, store->indices);
    }
    return store;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    auto var = Downcast<Var>(IRMutatorWithAnalyzer::VisitExpr_(op));
    if (buffer_data_to_buffer_.count(var)) {
      auto buffer = buffer_data_to_buffer_[var];
      if (buffer_remap_.count(buffer))
        return buffer_remap_[buffer]->data;
    }
    return var;
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    PrimExpr value = this->VisitExpr(op->value);
    bool recorded = false;
    if (value->IsInstance<BufferLoadNode>()) {
      let_bindings_[op->var] = value;
      recorded = true;
    }
    if (SideEffect(value) <= CallEffectKind::kPure) {
      analyzer_->Bind(op->var, value);
    }
    Stmt body = this->VisitStmt(op->body);
    if (recorded) {
      let_bindings_.erase(op->var);
    }
    if (value.same_as(op->value) && body.same_as(op->body)) {
      return tvm::ffi::GetRef<Stmt>(op);
    } else {
      auto n = this->CopyOnWrite(op);
      n->value = value;
      n->body = body;
      return Stmt(n);
    }
  }

  /**
   * @brief Handle an Evaluate node, lowering a detected tile operator to TIR.
   *
   * This visit implementation detects whether the Evaluate node represents a
   * tile operator invocation (via ParseOperator). If no tile operator is found
   * or the call targets a global function, the node is delegated to the base
   * visitor.
   *
   * When a tile operator is present, the method:
   * - Builds a workspace-allocation callback that creates a dynamic shared
   * buffer named "workspace" (storage scope "shared.dyn") and returns its write
   *   access pointer.
   * - Determines thread bounds for lowering from the analyzer's constant-int
   *   information for thread_var_; if unavailable, a default range [0,1) is
   * used.
   * - Invokes tile_op->Lower(...) with LowerArgs containing target, thread
   *   bounds, thread variable, the workspace callback, layout and buffer remap
   *   maps, and the list of GEMM-involved buffer vars; the analyzer is passed
   *   through for use during lowering.
   *
   * The lowered statement returned by the operator is then visited by the base
   * IRMutatorWithAnalyzer and that result is returned.
   *
   * @return Stmt The (possibly transformed) statement after lowering or base
   * visitor processing.
   */
  Stmt VisitStmt_(const EvaluateNode *op) final {
    const CallNode *call = op->value.as<CallNode>();
    // Do not analysis the call node to the global function.
    if (call && call->op.as<GlobalVarNode>())
      return Downcast<Evaluate>(IRMutatorWithAnalyzer::VisitStmt_(op));

    auto tile_op = ParseOperator(tvm::ffi::GetRef<Stmt>(op));
    if (!tile_op.defined())
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    AddWorkspaceCallback callback = [this](int num_elem, DataType dtype) {
      auto workspace =
          decl_buffer({PrimExpr(num_elem)}, dtype, "workspace", "shared.dyn");
      // Record workspace under the innermost block scope so its lifetime
      // covers the statements that requested it and does not sink into
      // subsequently created inner blocks (e.g., GEMM macro blocks).
      if (!workspace_stack_.empty()) {
        workspace_stack_.back().push_back(workspace);
      } else {
        // Fallback: create a temporary frame (should be rare)
        workspace_stack_.emplace_back(Array<Buffer>{workspace});
      }
      return workspace.access_ptr(2); // write
    };

    Range thread_bounds;

    if (analyzer_->const_int_bound.IsBound(thread_var_->var)) {
      auto const_int_bound = analyzer_->const_int_bound(thread_var_);
      auto min_value = const_int_bound->min_value;
      auto max_value = const_int_bound->max_value;
      auto extent = max_value + 1 - min_value;
      thread_bounds =
          Range::FromMinExtent(IntImm(thread_var_->var.dtype(), min_value),
                               IntImm(thread_var_->var.dtype(), extent));
    } else {
      thread_bounds = Range::FromMinExtent(0, 1);
    }

    // Convert let_bindings_ to Map<Var, PrimExpr> for LowerArgs
    Map<Var, PrimExpr> let_var_to_expr;
    for (const auto &[var, expr] : let_bindings_) {
      let_var_to_expr.Set(var, expr);
    }

    auto lowered = tile_op->Lower(
        LowerArgs{target_, thread_bounds, thread_var_->var, callback,
                  layout_map_, buffer_remap_, let_var_to_expr},
        analyzer_);
    return IRMutatorWithAnalyzer::VisitStmt(lowered);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      ICHECK_NE(iv->thread_tag.length(), 0U);
      if (iv->thread_tag == "threadIdx.x") {
        thread_var_ = iv;
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_block_size_ = iv->dom->extent.as<IntImmNode>()->value;
      }
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  /**
   * @brief Handle a Parallel For node, lowering it based on the layout
   * annotation.
   *
   * This method checks if the For node has a parallel_loop_layout annotation.
   * If the For node is a parallel loop (ForKind::kParallel):
   * - It must have the parallel_loop_layout annotation, otherwise an error is
   *   raised.
   * - The loop is partitioned and vectorized based on the annotated layout.
   * - If a predicate annotation exists, the loop is wrapped with an IfThenElse.
   *
   * Special handling for reducers and local buffers:
   * - If the loop stores into local buffers, thread partitioning is skipped.
   * - If the loop only manipulates local buffers, thread partitioning is
   * skipped.
   * - If reducers are present, vectorization is skipped.
   * - Vectorization is only applied if non-local buffers or vectorizable casts
   *   are present.
   *
   * @return Stmt The lowered statement.
   */
  Stmt VisitStmt_(const ForNode *op) final {
    // Extract reducer info from annotations
    Map<Var, ReducerInfo> reducer_info;
    if (op->annotations.count(attr::kReducerInfo)) {
      reducer_info = op->annotations.Get(attr::kReducerInfo)
                         ->as<Map<Var, ReducerInfo>>()
                         .value();
    }

    // First visit the body
    For for_node = Downcast<For>(arith::IRMutatorWithAnalyzer::VisitStmt_(op));

    // Only process parallel loops
    if (op->kind != ForKind::kParallel) {
      return for_node;
    }

    // For nested parallel loops, the annotation is placed on the outermost
    // loop. Inner parallel loops without annotation should be skipped here –
    // they will be processed as part of the outer loop's partitioning.
    // Rationale: inner loops cannot govern their outer loops; the outermost
    // loop is the correct place to carry layout so we can rewrite the whole
    // nested region in one place.
    if (!op->annotations.count(attr::kParallelLoopLayout)) {
      return for_node;
    }

    auto loop_layout = Downcast<Fragment>(
        op->annotations.Get(attr::kParallelLoopLayout).value());

    // Get predicate if it exists
    Optional<PrimExpr> predicate;
    if (op->annotations.count(attr::kParallelLoopPredicate)) {
      predicate = Downcast<PrimExpr>(
          op->annotations.Get(attr::kParallelLoopPredicate).value());
    }

    auto root = tvm::ffi::GetRef<For>(op);

    // Check if the loop stores into local buffers.
    // For example:
    //   for i in T.Parallel(1024):
    //     A_local[i] = A_global[i]
    // Here, A_local is a register-local buffer held independently by each
    // thread, so explicit thread binding is not required.
    bool store_into_local = false;
    PostOrderVisit(root, [&](const ObjectRef &obj) {
      if (const auto *store = obj.as<BufferStoreNode>()) {
        if (IsLocalBuffer(store->buffer)) {
          store_into_local = true;
        }
      }
    });

    // Check if the loop only manipulates "local" buffers.
    // for i in T.Parallel(1024):
    //     A_local[i] = B_local[i]
    // This indicates register usage and justifies skipping thread binding.
    bool local_register_only = true;
    PostOrderVisit(root, [&](const ObjectRef &obj) {
      if (const auto *store = obj.as<BufferStoreNode>()) {
        if (!IsLocalBuffer(store->buffer)) {
          local_register_only = false;
        }
      } else if (const auto *load = obj.as<BufferLoadNode>()) {
        if (!IsLocalBuffer(load->buffer)) {
          local_register_only = false;
        }
      }
    });

    // Determine if this is a true parallel loop requiring thread partitioning.
    // Skip partitioning for loops that only operate on local/register buffers.
    bool parallel_loop = !local_register_only && !store_into_local;

    // Check if there are non-local buffer accesses (for vectorization decision)
    bool has_non_local = false;
    PostOrderVisit(for_node->body, [&](const ObjectRef &obj) {
      if (const auto *load = obj.as<BufferLoadNode>()) {
        if (!IsLocalBuffer(load->buffer) && !IsFragmentBuffer(load->buffer)) {
          has_non_local = true;
        }
      } else if (const auto *store = obj.as<BufferStoreNode>()) {
        if (!IsLocalBuffer(store->buffer) && !IsFragmentBuffer(store->buffer)) {
          has_non_local = true;
        }
      }
    });

    // Check if reducers are present in the loop body
    // Workaround: if reducer is presented, don't vectorize loop
    // Best solution should be isolate reduction axis out of vectorization
    //
    // Note: reducer_info stores original buffer data vars, but after visiting
    // the body, buffers may have been remapped via var_remap_. We need to find
    // the original var to check against reducer_info.
    bool has_reducer = false;
    PostOrderVisit(for_node->body, [&](const ObjectRef &obj) {
      if (!has_reducer) {
        if (const auto *store = obj.as<BufferStoreNode>()) {
          Var data_var = store->buffer->data;
          // Find the original var if it was remapped
          // var_remap_ maps old_var -> new_var, so we need reverse lookup
          Var original_var = data_var;
          for (const auto &[old_var, new_var] : var_remap_) {
            if (new_var.same_as(data_var)) {
              original_var = old_var;
              break;
            }
          }
          has_reducer = reducer_info.count(original_var) != 0;
        }
      }
    });

    // Check if vectorizable cast operations exist
    bool has_cast_operations = false;
    PostOrderVisit(for_node->body, [&](const ObjectRef &obj) {
      if (const auto *cast = obj.as<CastNode>()) {
        DataType from_ty = cast->value.dtype();
        DataType target_ty = cast->dtype;
        if (IsCudaVectorizableCast(from_ty, target_ty) &&
            TargetIsCuda(Target::Current())) {
          has_cast_operations = true;
        }
      }
    });

    // Decide whether to vectorize:
    // - Only if there are non-local buffers or vectorizable casts
    // - AND no reducers are present
    bool should_vectorize =
        (has_non_local || has_cast_operations) && !has_reducer;

    // Lower the parallel loop using the common function
    return LowerParallelLoop(for_node, loop_layout, thread_var_->var, analyzer_,
                             layout_map_, predicate, parallel_loop,
                             should_vectorize);
  }

  Target target_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Layout> layout_map_;
  Map<Buffer, Layout> layout_remap_;
  Map<Buffer, Buffer> buffer_remap_;
  // This is a workaround for cpu backend,
  // we need to define a thread_var for the serial loop.
  IterVar thread_var_ = IterVar(Range::FromMinExtent(0, 1), Var("v_thread"),
                                IterVarType::kDataPar);
  size_t thread_block_size_ = 0;
  // Stack of per-Block workspace buffers gathered while visiting children
  std::vector<Array<Buffer>> workspace_stack_;
  // For ptx Node, we need to remap the buffer and indices
  // By access CallNode instead of BufferLoad Node.
  bool is_ptx_{false};
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>
      let_bindings_;
  // Mapping from data Var of a Buffer to Buffer, for lookup
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
  Map<Var, Var> var_remap_;
  bool has_tma_{false};
  // Flag to indicate we are inside a TMA context (tma_load, tma_load_im2col,
  // tma_store). When true, HandleAccessPtrAndOffset only updates buffer data
  // without recomputing indices, since swizzle is encoded in TMA descriptor
  // parameters rather than in memory indices.
  bool in_tma_context_{false};
};

namespace transform {

using namespace tir::transform;

tvm::transform::Pass LowerTileOp() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return LowerTileOpPass::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerTileOp", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerTileOp", LowerTileOp);
}
} // namespace transform

} // namespace tl
} // namespace tvm
