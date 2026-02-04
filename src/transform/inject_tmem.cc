/*!
 * \file inject_tmem.cc
 * \brief Inject Tensor Memory allocation, usage and destruction
 *
 * This pass detects operations that implicitly utilize Tensor Memory on Blackwell and above
 * architectures, automatically allocates, uses and destroys Tensor Memory.
 *
 * Key behaviors:
 * - Detects gemm operations where C is a local.fragment buffer
 * - Allocates a shared.tmem buffer for the accumulator
 * - Replaces the gemm C operand with the tmem buffer
 * - Inserts a copy from tmem to the original fragment after the loop
 */

// todo: liveness analysis to reuse TMEM as much as possible
// todo: check compatibility of TMEM with multi buffers
// todo: consider ts tcgen5mma

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/arith/analyzer.h>

#include <variant>

#include "../op/copy.h"
#include "../op/gemm.h"
#include "../op/gemm_py.h"
#include "../op/operator.h"
#include "tvm/tir/expr.h"


namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

using GemmNodeVariant = std::variant<const GemmNode *, const GemmPyNode *>;

std::optional<GemmNodeVariant> TryParseGemmNode(const TileOperator &tile_op) {
  if (auto gemm = tile_op.as<GemmNode>()) {
    return gemm;
  } else if (auto gemm_py = tile_op.as<GemmPyNode>()) {
    return gemm_py;
  }
  return std::nullopt;
}

/*!
 * \brief Check if a buffer is a fragment buffer (local.fragment scope)
 */
static bool IsFragmentBufferLocal(const Buffer &buffer) {
  const auto *ptr_type = buffer->data->type_annotation.as<PointerTypeNode>();
  if (ptr_type == nullptr) return false;
  return ptr_type->storage_scope == "local.fragment";
}

/*!
 * \brief Check if a buffer is already a tmem buffer (shared.tmem scope)
 */
static bool IsTmemBuffer(const Buffer &buffer) {
  const auto *ptr_type = buffer->data->type_annotation.as<PointerTypeNode>();
  if (ptr_type == nullptr) return false;
  return ptr_type->storage_scope == "shared.tmem";
}

/*!
 * \brief Create a Call node for a Copy tile operation
 */
static Call MakeCopyCall(const Buffer &src, const Buffer &dst) {
  // Create source and destination buffer regions using FullRegion
  BufferRegion src_region = BufferRegion::FullRegion(src);
  BufferRegion dst_region = BufferRegion::FullRegion(dst);

  // Create the copy call using the Copy tile op
  // Convert BufferRegion to PrimExpr for the call arguments
  Array<PrimExpr> args;
  args.push_back(src_region->ToPrimExpr());
  args.push_back(dst_region->ToPrimExpr());

  return Call(DataType::Handle(), Copy::Get(), args);
}

static bool IsFullRegion(const BufferRegion &region) {
    const Buffer &buffer = region->buffer;
    const Array<Range> &ranges = region->region;
    for (size_t i = 0; i < ranges.size(); i++) {
      const Range &range = ranges[i];
      if (!is_zero(range->min))
        return false;
      if (!StructuralEqual()(range->extent, buffer->shape[i]))
        return false;
    }
    return true;
}


/*!
 * \brief Information about a gemm operation that needs tmem injection
 */
struct GemmTmemInfo {
  Buffer frag_buffer;   // Original fragment buffer
  Buffer tmem_buffer;   // New tmem buffer to create
  int m, n;             // Dimensions for the tmem buffer
  DataType dtype;       // Data type of the buffer
};

/*!
 * \brief First pass: Collect all gemm operations that need tmem injection
 */
class GemmCollector : public StmtExprVisitor {
public:
  // Map from fragment buffer to the tmem buffer that should replace it
  std::unordered_map<const BufferNode *, GemmTmemInfo> gemm_infos;
  // Collected fragment buffers that are used as gemm C operands
  std::unordered_set<const BufferNode *> fragment_c_buffers;

  void VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      TileOperator tile_op = ParseOperator(tvm::ffi::GetRef<Call>(call));
      // Check if the tile operator is a Gemm or GemmPy
      auto gemm_node = TryParseGemmNode(tile_op);
      if (gemm_node.has_value()) {
        CollectGemmInfo(gemm_node.value());
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

private:
  void CollectGemmInfo(const GemmNodeVariant &gemm_node) {
    std::visit([&](auto &gemm_node) {
        Buffer c_buffer = gemm_node->c_;
    
        // Check if C is already a tmem buffer (nothing to do)
        if (IsTmemBuffer(c_buffer)) {
          LOG(INFO) << "C is already a tmem buffer, nothing to do";
          return;
        }
        ICHECK(IsFragmentBufferLocal(c_buffer));
        BufferRegion c_region = gemm_node->cRegion_;
        ICHECK(IsFullRegion(c_region)) << "Currently only support full region for C";
        // todo: support c as partial region
        fragment_c_buffers.insert(c_buffer.get());

        // todo: check tcgen5 instructions is selected

        // todo: fix_v1: requires 2d raises error
    
        // Only add if not already tracked
        if (gemm_infos.find(c_buffer.get()) == gemm_infos.end()) {
          GemmTmemInfo info;
          info.frag_buffer = c_buffer;
          info.m = gemm_node->m_;
          info.n = gemm_node->n_;
          info.dtype = c_buffer->dtype;
    
          // Create the tmem buffer
          // Use the same name with _tmem suffix
          std::string tmem_name = c_buffer->name + "_tmem";  // todo: name mangle to avoid conflict
          Var tmem_data(tmem_name,
                        PointerType(PrimType(info.dtype), "shared.tmem"));
          info.tmem_buffer =
              Buffer(tmem_data, info.dtype, {info.m, info.n}, {}, PrimExpr(0),
                     tmem_name, c_buffer->data_alignment, c_buffer->offset_factor,
                     c_buffer->buffer_type);
    
          gemm_infos[c_buffer.get()] = info;
        }
    }, gemm_node);
  }
};

/*!
 * \brief Second pass: Transform the IR to inject tmem
 */
class TmemInjector : public StmtExprMutator {
public:
  explicit TmemInjector(
      const std::unordered_map<const BufferNode *, GemmTmemInfo> &gemm_infos)
      : gemm_infos_(gemm_infos) {}

private:
  Stmt VisitStmt_(const BlockNode *op) final {
    // Check if this block has any buffers that need tmem injection
    Array<Buffer> new_alloc_buffers;
    bool need_tmem_allocation = false;

    for (const auto &buffer : op->alloc_buffers) {
      new_alloc_buffers.push_back(buffer);
      auto it = gemm_infos_.find(buffer.get());
      if (it != gemm_infos_.end()) {
        // Add the corresponding tmem buffer allocation
        new_alloc_buffers.push_back(it->second.tmem_buffer);
        frag_to_tmem_[buffer.get()] = it->second.tmem_buffer;
        tmem_to_frag_[it->second.tmem_buffer.get()] = buffer;
        need_tmem_allocation = true;
      }
    }

    // Visit the block body
    Stmt new_body = VisitStmt(op->body);

    // If we need to allocate tmem, update the allocations in the block
    if (need_tmem_allocation) {
      Block new_block = tvm::ffi::GetRef<Block>(op);
      auto block_ptr = new_block.CopyOnWrite();
      block_ptr->alloc_buffers = new_alloc_buffers;
      block_ptr->body = new_body;
      return new_block;
    }

    // Otherwise, if body changed, create a new block
    if (!new_body.same_as(op->body)) {
      Block new_block = tvm::ffi::GetRef<Block>(op);
      auto block_ptr = new_block.CopyOnWrite();
      block_ptr->body = new_body;
      return new_block;
    }

    return tvm::ffi::GetRef<Block>(op);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Check if this is a pipelined loop that contains gemm with tmem injection
    bool is_pipelined = op->annotations.count("software_pipeline_stage") > 0 ||
                        op->annotations.count("num_stages") > 0;

    Stmt new_body = VisitStmt(op->body);

    // Check if we have pending copy insertions for this loop level
    if (is_pipelined && !pending_copies_.empty()) {
      // Create the modified for loop
      For new_for = tvm::ffi::GetRef<For>(op);
      if (!new_body.same_as(op->body)) {
        auto for_ptr = new_for.CopyOnWrite();
        for_ptr->body = new_body;
      }

      // Insert the copies after the loop
      Array<Stmt> stmts;
      stmts.push_back(new_for);

      for (const auto &copy_info : pending_copies_) {
        const Buffer &tmem_buf = copy_info.first;
        const Buffer &frag_buf = copy_info.second;

        // Create copy from tmem to fragment
        Call copy_call = MakeCopyCall(tmem_buf, frag_buf);
        stmts.push_back(Evaluate(copy_call));
      }

      pending_copies_.clear();
      return SeqStmt(stmts);
    }

    if (!new_body.same_as(op->body)) {
      For new_for = tvm::ffi::GetRef<For>(op);
      auto for_ptr = new_for.CopyOnWrite();
      for_ptr->body = new_body;
      return new_for;
    }
    return tvm::ffi::GetRef<For>(op);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      TileOperator tile_op = ParseOperator(tvm::ffi::GetRef<Call>(call));
      // Check if the tile operator is a Gemm or GemmPy
      auto gemm_node = TryParseGemmNode(tile_op);
      if (gemm_node.has_value()) {
        return TransformGemm(gemm_node.value(), call);
      };
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt TransformGemm(const GemmNodeVariant &gemm_node, const CallNode *call) {
    return std::visit([&, call](auto &node) -> Stmt {
        Buffer c_buffer = node->c_;
        auto it = frag_to_tmem_.find(c_buffer.get());
        if (it == frag_to_tmem_.end()) {
          // Not tracked for tmem injection, return original call
          return Evaluate(tvm::ffi::GetRef<Call>(call));
        }
        const Buffer &tmem_buffer = it->second;

        // Schedule a copy to be inserted after the pipelined loop
        pending_copies_.push_back({tmem_buffer, c_buffer});

        // Rebuild the gemm call with the tmem buffer
        // args[2] is the C buffer region
        Array<PrimExpr> new_args;
        for (size_t i = 0; i < call->args.size(); i++) {
            if (i == 2) {
                // Replace the C region with tmem buffer region
                BufferRegion old_region = node->cRegion_;
                BufferRegion new_region(tmem_buffer, old_region->region);
                new_args.push_back(new_region->ToPrimExpr());
            } else {
                new_args.push_back(call->args[i]);
            }
        }

        Call new_call(call->dtype, call->op, new_args, call->annotations);
        return Evaluate(new_call);
    }, gemm_node);
  }

  const std::unordered_map<const BufferNode *, GemmTmemInfo> &gemm_infos_;
  std::unordered_map<const BufferNode *, Buffer> frag_to_tmem_;
  std::unordered_map<const BufferNode *, Buffer> tmem_to_frag_;
  // Pending copies to insert after the current pipelined loop
  // Pair of (tmem_buffer, frag_buffer)
  std::vector<std::pair<Buffer, Buffer>> pending_copies_;
};

tvm::transform::Pass InjectTmem() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    LOG(INFO) << "InjectTmem pass started!";
    // First pass: collect gemm operations that need tmem injection
    GemmCollector collector;
    collector(f->body);
    LOG(INFO) << "Collector collected: " << collector.gemm_infos.size();
    // If no gemm needs injection, return unchanged
    if (collector.gemm_infos.empty()) {
      return f;
    }

    // Second pass: transform the IR
    TmemInjector injector(collector.gemm_infos);
    auto *n = f.CopyOnWrite();
    n->body = injector(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectTmem", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectTmem", InjectTmem);
};

}  // namespace tl
}  // namespace tvm