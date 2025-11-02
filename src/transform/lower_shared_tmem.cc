/*!
 *  \file lower_shared_tmem.cc
 *  \brief Convert shared.tmem buffers to plain shared + ptx init, and do
 *         coordinate translation (from logical address to physical address)
 */
#include "../op/builtin.h"
#include "../target/utils.h"
#include "tvm/ir/type.h"
#include "tvm/tir/builtin.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

class SharedTmemRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt body) {
    SharedTmemRewriter rewriter;
    return rewriter(body);
  }

private:
  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = tvm::ffi::GetRef<Block>(op);
    Array<Buffer> alloc_buffers = op->alloc_buffers;
    if (op->annotations.count(attr::kLayoutMap)) {
      auto layout_map = op->annotations.Get(attr::kLayoutMap);
      ICHECK(layout_map) << "layout map is not defined";
      layout_map_ = layout_map->as<Map<Buffer, Layout>>().value();
    }

    // Record the mapping from buffer data var to buffer for later lookup
    for (auto buffer : alloc_buffers) {
      buffer_map_.insert({buffer->data, buffer});
    }
    for (auto match_buffer : op->match_buffers) {
      buffer_map_.insert({match_buffer->buffer->data, match_buffer->buffer});
    }

    Array<Buffer> tmem_buffers;

    for (const auto &[data, buffer] : buffer_map_) {
      const auto *ptr_type =
          buffer->data->type_annotation.as<PointerTypeNode>();
      auto storage_scope = ptr_type->storage_scope;
      ICHECK(ptr_type) << "Buffer Var's type annotation must be of PointerType";
      if (storage_scope == "shared.tmem") {
        tmem_buffers.push_back(buffer);
      }
    }

    if (tmem_buffers.empty()) {
      return StmtExprMutator::VisitStmt_(op);
    }

    ICHECK(thread_var_.defined()) << "thread_var_ is not defined";

    for (auto buffer : tmem_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }

    /*
    Transform the tmem buffers to new allocations
    transform:
        tmem_buf0 = T.alloc_buffer((128, 128,), "uint64",
    scope="shared.tmem")
        tmem_buf1 = T.alloc_buffer((128, 128,), "uint64",
    scope="shared.tmem")

    into:
        tmem_buf0 = T.alloc_buffer((1,), "uint64", scope="shared.tmem_addr")
        tmem_buf1 = T.alloc_buffer((1,), "uint64", scope="shared.tmem_addr")

        if tx == 0:
          T.ptx_init_tensor_memory(tmem_buf0[0], 128)
          T.ptx_init_tensor_memory(tmem_buf1[0], 128)
    */
    // 1. create new data vars
    Array<Var> new_data_vars;
    for (auto buffer : tmem_buffers) {
      auto data = buffer->data;
      if (var_remap_.count(data))
        continue;
      auto new_data =
          Var(data->name_hint, PointerType(PrimType(tmem_dtype_), "shared"));
      var_remap_.Set(data, new_data);
      new_data_vars.push_back(new_data);
    }

    // 2. create new buffers
    Array<Buffer> new_buffers;
    for (auto buffer : tmem_buffers) {
      auto data = buffer->data;
      ICHECK(var_remap_.find(data) != var_remap_.end())
          << "data not found in var_remap_";
      auto new_data = var_remap_.at(data);
      auto new_buffer = Buffer(new_data, tmem_dtype_, Array<PrimExpr>({1}),
                               Array<PrimExpr>({1}), PrimExpr(0), buffer->name,
                               buffer->data_alignment, buffer->offset_factor,
                               buffer->buffer_type);
      new_buffers.push_back(new_buffer);
      buffer_remap_.Set(buffer, new_buffer);
      buffer_data_to_buffer_.Set(new_data, new_buffer);
    }

    // remove the tmem buffers
    alloc_buffers.MutateByApply([this](Buffer buf) {
      if (buffer_remap_.find(buf) != buffer_remap_.end()) {
        return buffer_remap_.at(buf);
      }
      return buf;
    });
    if (!alloc_buffers.same_as(op->alloc_buffers)) {
      block.CopyOnWrite()->alloc_buffers = alloc_buffers;
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }

    // 3. create init & dealloc calls for new buffers
    std::vector<Stmt> init_mtmem_calls_;
    std::vector<Stmt> dealloc_tmem_calls_;
    for (auto buffer : tmem_buffers) {
      auto data = buffer->data;
      auto old_buffer = buffer_data_to_buffer_.at(data);
      auto new_buffer = buffer_remap_.at(old_buffer);

      // Tmem physical coord range analysis
      ICHECK(old_buffer->shape.size() == 2);

      auto analyzer = std::make_shared<arith::Analyzer>();
      arith::ConstIntBound phy_col_bounds =
          analyzer->const_int_bound(old_buffer->shape[1]);
      int num_cols_required = phy_col_bounds->max_value;
      ICHECK(num_cols_required <= 512)
          << "The number of columns required for tmem buffer "
          << old_buffer->name << " is " << num_cols_required
          << ", which exceeds the maximum of 512 columns";

      int num_cols_allocated = 32; // Align num_cols_allocated to power of 2
      for (; num_cols_allocated < num_cols_required; num_cols_allocated *= 2)
        ;

      auto new_buffer_access = new_buffer.access_ptr(1, DataType::Handle(), 1,
                                                     PrimExpr(0), PrimExpr(1));
      auto alloc_call = Call(DataType::Handle(), tl::ptx_init_tensor_memory(),
                             {new_buffer_access, PrimExpr(num_cols_allocated)});
      init_mtmem_calls_.push_back(Evaluate(alloc_call));
      auto dealloc_call =
          Call(DataType::Handle(), tl::ptx_deallocate_tensor_memory(),
               {new_buffer_access, PrimExpr(num_cols_allocated)});
      dealloc_tmem_calls_.push_back(Evaluate(dealloc_call));
    }
    auto compare_by_buffer_name = [&](const Stmt &a, const Stmt &b) {
      auto call_a = a.as<EvaluateNode>()->value.as<CallNode>();
      auto call_b = b.as<EvaluateNode>()->value.as<CallNode>();
      auto num_cols_a = call_a->args[1].as<IntImmNode>()->value;
      auto num_cols_b = call_b->args[1].as<IntImmNode>()->value;
      return num_cols_a > num_cols_b;
    };
    std::sort(init_mtmem_calls_.begin(), init_mtmem_calls_.end(),
              compare_by_buffer_name);

    Array<Stmt> new_body;
    auto target = Target::Current();
    auto warp_size = TargetGetWarpSize(target);
    auto thread_var_div_warp_size =
        FloorDiv(thread_var_->var, IntImm(thread_var_->var->dtype, warp_size));
    new_body.push_back(IfThenElse(EQ(thread_var_div_warp_size, 0),
                                  init_mtmem_calls_.size() > 1
                                      ? SeqStmt(init_mtmem_calls_)
                                      : init_mtmem_calls_.back(),
                                  Stmt()));
    new_body.push_back(
        Evaluate(Call(DataType::Handle(), builtin::tvm_storage_sync(),
                      {StringImm("shared")})));
    new_body.push_back(block->body);
    new_body.push_back(IfThenElse(EQ(thread_var_div_warp_size, 0),
                                  dealloc_tmem_calls_.size() > 1
                                      ? SeqStmt(dealloc_tmem_calls_)
                                      : dealloc_tmem_calls_.back(),
                                  Stmt()));

    auto block_ptr = block.CopyOnWrite();
    block_ptr->annotations.erase(attr::kLayoutMap);
    block_ptr->body = SeqStmt(new_body);

    return StmtExprMutator::VisitStmt_(block.get());
  }

  PrimExpr GetTmemOffset(const Buffer &buffer, const Array<PrimExpr> &indices) {
    ICHECK(buffer->shape.size() == 2);
    ICHECK(indices.size() == 2);
    ICHECK(layout_map_.defined());
    ICHECK(layout_map_.count(buffer))
        << "The layout of tmem buffer " << buffer->name
        << " is not defined in the layout map";
    auto layout = layout_map_[buffer];
    ICHECK(layout.defined());
    Array<PrimExpr> tmem_phy_coords = layout->Forward(indices);
    PrimExpr result =
        tmem_phy_coords[0] << 16 |
        tmem_phy_coords
            [1]; // https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-addressing
    return result;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    // Translate tmem[logical_row, logical_col] to tmem[0] + tmem_offset
    // Where
    // - (logical_row, logical_col) is the logical address in the tmem buffer
    // - tmem[0] is the base address allocated for the tmem buffer
    // - tmem_offset = tmem_phy_coords[0]<<16 | tmem_phy_coords[1]
    //   where tmem_phy_coords = layout.Forward(logical_row, logical_col)
    //   is the physical address in the tmem buffer
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto buffer = load->buffer;
    auto indices = load->indices;

    if (buffer_remap_.count(buffer)) {
      auto new_buffer = buffer_remap_[load->buffer];
      return BufferLoad(new_buffer, {0}) + GetTmemOffset(buffer, indices);
    } else if (var_remap_.count(buffer->data)) {
      auto new_buffer = Buffer(
          var_remap_[buffer->data], tmem_dtype_, buffer->shape, buffer->strides,
          buffer->elem_offset, buffer->name, buffer->data_alignment,
          buffer->offset_factor, buffer->buffer_type);
      return BufferLoad(new_buffer, {0}) + GetTmemOffset(buffer, indices);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto buffer = store->buffer;
    ICHECK(buffer.scope() != "shared.tmem")
        << "We should never directly store data into tmem!";
    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5U);
      Var buffer_data = Downcast<Var>(op->args[1]);
      if (!var_remap_.count(buffer_data)) {
        return StmtExprMutator::VisitExpr_(op);
      }
      Var new_data = var_remap_[buffer_data];
      return Call(
          op->dtype, op->op,
          {op->args[0], new_data, op->args[2], op->args[3], op->args[4]});
    }
    auto expr = StmtExprMutator::VisitExpr_(op);
    return expr;
  }
  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = tvm::ffi::GetRef<Var>(op);
    if (var_remap_.count(var)) {
      return var_remap_[var];
    }
    return var;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_var_ = iv;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // Datatypes for tmem
  const DataType tmem_dtype_ = DataType::UInt(32);
  // This is a workaround for cpu backend,
  // we need to define a thread_var for the serial loop.
  IterVar thread_var_;
  Map<Var, Var> var_remap_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Buffer> buffer_remap_;
  // Mapping from data Var of a Buffer to Buffer, for lookup
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
  Map<Buffer, Layout> layout_map_;
};

PrimFunc LowerSharedTmem(PrimFunc f) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target.defined()) << "LowerSharedTmem: Require the target attribute";
  SharedTmemRewriter rewriter;
  f.CopyOnWrite()->body = rewriter.Rewrite(f->body);
  return f;
}

namespace transform {
using namespace tir::transform;

tvm::transform::Pass LowerSharedTmem() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return tl::LowerSharedTmem(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerSharedTmem", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerSharedTmem", LowerSharedTmem);
}

} // namespace transform
} // namespace tl
} // namespace tvm
