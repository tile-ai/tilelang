/*!
 * \file lower_l2_persistent_annotation.cc
 * \brief Lower L2 persistent annotation
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "../runtime/runtime.h"

namespace tvm {
namespace tl {

namespace attr {
// BlockAttr, Containing the layout for all the buffers in the block
constexpr const char *kL2RatioMap = "l2_hit_ratio_map";
constexpr const char *kL2PersistentMap = "l2_persistent_map";
} // namespace attr

using namespace tir;

class LowerL2Persistent : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    PrimFuncNode *fptr = f.CopyOnWrite();
    LowerL2Persistent substituter;
    // Trace the buffer map for tvm_access_ptr
    substituter.buffer_map_.insert(f->buffer_map.begin(), f->buffer_map.end());
    for (const auto &[_, buffer] : f->buffer_map) {
      substituter.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    fptr->body = substituter.VisitStmt(f->body);
    Map<String, Array<PrimExpr>> init_l2_persistent_map;
    for (auto [buffer, hit_ratio] : substituter.hit_ratio_map_) {
      Array<PrimExpr> l2_persistent_arguments;
      // Argument 0: hit ratio
      // Argument 1: size in bytes
      l2_persistent_arguments.push_back(hit_ratio);
      PrimExpr size_in_bytes = IntImm(DataType::Int(64), buffer->dtype.bytes());
      for (auto dim : buffer->shape) {
        size_in_bytes = size_in_bytes * dim;
      }
      l2_persistent_arguments.push_back(size_in_bytes);
      init_l2_persistent_map.Set(buffer->name, l2_persistent_arguments);
    }
    if (!init_l2_persistent_map.empty()) {
      f = WithAttr(std::move(f), attr::kL2PersistentMap,
                   init_l2_persistent_map);
    }
    return f;
  }

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

    if (op->annotations.count(attr::kL2RatioMap)) {
      auto hit_ratio_map = op->annotations.at(attr::kL2RatioMap)
                               .as<Map<Var, FloatImm>>()
                               .value();
      for (auto [buffer_var, hit_ratio] : hit_ratio_map) {
        Buffer buffer = buffer_data_to_buffer_.at(buffer_var);
        hit_ratio_map_.Set(buffer, hit_ratio);
      }
    }
    auto block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    auto block_ptr = block.CopyOnWrite();
    block_ptr->annotations.erase(attr::kL2RatioMap);
    return block;
  }

private:
  // Mapping from data Var of a Buffer to Buffer, for lookup
  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
  Map<Buffer, FloatImm> hit_ratio_map_;
  LowerL2Persistent() = default;
};

using namespace tir::transform;

tvm::transform::Pass LowerL2Persistent() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return LowerL2Persistent::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerL2Persistent", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerL2Persistent", LowerL2Persistent);
}

} // namespace tl
} // namespace tvm
