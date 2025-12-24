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

#include "arith/ir_mutator_with_analyzer.h"
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
    substituter.buffer_map_.insert(f->buffer_map.begin(), f->buffer_map.end());
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
      LOG(FATAL) << "Transformation for tvm_access_ptr is not implemented yet";
    } else if (access_ptr_call->op.same_as(builtin::address_of())) {
      Optional<PrimExpr> resolved = ResolveBufferLoad(access_ptr_call->args[0]);
      ICHECK(resolved.defined())
          << "Invalid access op for permuted layout: " << access_ptr;
      PrimExpr load_expr = resolved.value();
      if (!load_expr.same_as(access_ptr_call->args[0])) {
        auto node = access_ptr_call.CopyOnWrite();
        node->args.Set(0, load_expr);
        access_ptr_call = Call(access_ptr_call->dtype, access_ptr_call->op,
                               {load_expr}, access_ptr_call->span);
      }
      BufferLoad load = Downcast<BufferLoad>(access_ptr_call->args[0]);
      Array<PrimExpr> indices = load->indices;
      Array<PrimExpr> old_shape = load->buffer->shape;

      CHECK_EQ(indices.size(), old_shape.size())
          << "Indices size and shape size must match for general N-dimensional "
             "buffer "
          << "but got indices size: " << indices.size()
          << " and shape size: " << old_shape.size();

      PrimExpr elem_offset = 0;
      PrimExpr stride = 1;

      for (int i = static_cast<int>(old_shape.size()) - 1; i >= 0; --i) {
        elem_offset += indices[i] * stride;
        stride *= old_shape[i];
      }

      PrimExpr smem_offset =
          elem_offset + (offset.defined() ? offset.value() : 0);

      Buffer remap_key = FindRemapBuffer(load->buffer).value_or(load->buffer);
      Optional<Layout> layout = FindLayout(remap_key);
      if (!layout.defined() || !buffer_map_.count(remap_key->data)) {
        return result;
      }
      auto new_buffer = buffer_remap_.count(remap_key)
                            ? buffer_remap_[remap_key]
                            : load->buffer;
      auto new_shape = new_buffer->shape;

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
                         access_ptr_call->span);
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
    if ((!has_tma_) && (op->op.same_as(tl::tma_load()) ||
                        op->op.same_as(tl::tma_load_im2col()) ||
                        op->op.same_as(tl::tma_store()))) {
      has_tma_ = true;
    }
    Array<RelaxExpr> ptx_instructions = {builtin::ptx_ldmatrix(),
                                         builtin::mma_store()};

    if (std::find(ptx_instructions.begin(), ptx_instructions.end(), op->op) ==
        ptx_instructions.end()) {
      auto call = Downcast<Call>(IRMutatorWithAnalyzer::VisitExpr_(op));
      return call;
    } else {
      is_ptx_ = true;
    }
    // Rewrite from/to shared or shared.dyn to/from local
    auto call = Downcast<Call>(IRMutatorWithAnalyzer::VisitExpr_(op));
    if (call->op.same_as(builtin::ptx_ldmatrix())) {
      // form: T.ptx_ldmatrix(..., smem_ptr, smem_offset)
      // smem_ptr: T.tvm_access_ptr(ptype, data, offset, extent, rw_mask)
      // or T.address_of(buffer, offset)
      PrimExpr access_ptr = call->args[5];
      PrimExpr smem_offset = call->args[6];
      Call address_of_call = Downcast<Call>(access_ptr);
      if (!address_of_call->op.same_as(builtin::address_of())) {
        LOG(FATAL) << "Invalid access ptr for permuted layout: " << access_ptr;
      }
      Optional<PrimExpr> resolved = ResolveBufferLoad(address_of_call->args[0]);
      ICHECK(resolved.defined())
          << "Invalid address_of argument for permuted layout: "
          << address_of_call->args[0];
      PrimExpr load_expr = resolved.value();
      if (!load_expr.same_as(address_of_call->args[0])) {
        auto call_node = call.CopyOnWrite();
        call_node->args.Set(5, Call(address_of_call->dtype, address_of_call->op,
                                    {load_expr}, address_of_call->span));
        address_of_call = Downcast<Call>(call->args[5]);
        access_ptr = call->args[5];
      }
      BufferLoad load = Downcast<BufferLoad>(address_of_call->args[0]);
      auto new_access_ptr =
          HandleAccessPtrAndOffset(access_ptr, smem_offset, call->dtype);
      if (new_access_ptr.rewritten) {
        auto new_call = call.CopyOnWrite();
        new_call->args.Set(5, new_access_ptr.expr);
        new_call->args.Set(6, IntImm(smem_offset->dtype, 0));
      }
    } else if (call->op.same_as(builtin::mma_store())) {
      // because we will directly store result to Buffer instead of calling
      // mma_store now
      auto access_ptr = call->args[2];
      auto new_access_ptr =
          HandleAccessPtrAndOffset(access_ptr, std::nullopt, call->dtype);
      if (new_access_ptr.rewritten) {
        auto new_call = call.CopyOnWrite();
        new_call->args.Set(2, new_access_ptr.expr);
      }
    } else {
      LOG(FATAL) << "Invalid call node: " << call;
    }
    is_ptx_ = false;
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
