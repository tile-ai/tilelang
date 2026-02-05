/*!
 * \file lower_ldg_stg.cc
 * \brief Lower Ramp-based global memory load/store to ldg/stg intrinsics
 *
 * This pass transforms vectorized global memory loads and stores (using Ramp
 * indices) into explicit ldg32/64/128/256 and stg32/64/128/256 intrinsics for
 * better codegen.
 *
 * Key behaviors:
 * 1. Converts Ramp-based global BufferLoad to ldg intrinsics
 * 2. Converts Ramp-based global BufferStore to stg intrinsics
 * 3. Supports predicated loads (if_then_else with else=0)
 * 4. Supports predicated stores (if_then_else with empty then case)
 * 5. Skips loads in async scope (will be lowered to cp.async)
 * 6. Only enabled for CUDA targets
 *
 * Pass configurations:
 * - tl.enable_lower_ldgstg: Enable non-predicated ldg/stg lowering (default:
 * OFF)
 * - tl.enable_lower_ldgstg_predicated: Enable predicated ldg/stg lowering
 * (default: OFF)
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "tir/ir/buffer_common.h"

namespace tvm {
namespace tl {

using namespace tir;

class LowerLDGSTGRewriter : public StmtExprMutator {
public:
  explicit LowerLDGSTGRewriter(bool enable_non_predicated,
                               bool enable_predicated)
      : enable_non_predicated_(enable_non_predicated),
        enable_predicated_(enable_predicated) {}

  Stmt VisitStmt_(const AttrStmtNode *attr) final {
    if (attr->attr_key == tir::attr::async_scope) {
      // Mark that we're inside an async scope
      bool old_in_async = in_async_scope_;
      in_async_scope_ = true;
      auto body = this->VisitStmt(attr->body);
      in_async_scope_ = old_in_async;
      if (body.same_as(attr->body)) {
        return tvm::ffi::GetRef<Stmt>(attr);
      }
      return AttrStmt(attr->node, attr->attr_key, attr->value, body);
    }
    return StmtExprMutator::VisitStmt_(attr);
  }

  Stmt VisitStmt_(const BufferStoreNode *store) final {
    // Skip if non-predicated lowering is disabled
    if (!enable_non_predicated_) {
      return StmtExprMutator::VisitStmt_(store);
    }

    // Only handle global memory stores
    if (store->buffer.scope() != "global") {
      return StmtExprMutator::VisitStmt_(store);
    }

    // Assume buffer has been flattened by FlattenBuffer pass
    ICHECK(store->indices.size() == 1)
        << "Expected flattened buffer with single index, but got "
        << store->indices.size() << " indices for buffer "
        << store->buffer->name;

    // Check if this is a Ramp-based store (vectorized)
    if (store->indices[0]->IsInstance<RampNode>()) {
      auto ramp = store->indices[0].as<RampNode>();
      // Check if stride is 1 (contiguous access)
      if (auto stride_imm = ramp->stride.as<IntImmNode>()) {
        if (stride_imm->value == 1) {
          // Get lanes from the index dtype
          int lanes = store->indices[0]->dtype.lanes();
          // Use bits() to correctly handle sub-byte types like float4_e2m1fn
          int total_bits = lanes * store->buffer->dtype.bits();

          // Check for supported vector widths (32/64/128/256 bits)
          if (total_bits == 32 || total_bits == 64 || total_bits == 128 ||
              total_bits == 256) {
            return LowerToSTG(store, ramp->base, total_bits);
          }
        }
      }
    } else {
      // Single element store (non-Ramp)
      int bits = store->buffer->dtype.bits();
      if (bits == 32 || bits == 64 || bits == 128 || bits == 256) {
        return LowerToSTG(store, store->indices[0], bits);
      }
    }

    // Check if store value is an if_then_else with empty then case (predicated
    // store) This pattern appears as: if (pred) { store } which gets lowered to
    // BufferStore with if_then_else in the IfThenElse statement handling
    return StmtExprMutator::VisitStmt_(store);
  }

  Stmt VisitStmt_(const IfThenElseNode *if_stmt) final {
    // Skip if predicated lowering is disabled
    if (!enable_predicated_) {
      return StmtExprMutator::VisitStmt_(if_stmt);
    }

    // Check for predicated store pattern:
    // if (pred) { } else { BufferStore(...) }
    // This represents a store that only happens when pred is false
    // We convert this to stg with predicate = !pred

    // Actually, the more common pattern for predicated store is:
    // if (pred) { BufferStore(...) }
    // which means store only when pred is true
    if (!if_stmt->else_case.defined()) {
      // Check if then_case is a single BufferStore to global memory with Ramp
      if (auto seq = if_stmt->then_case.as<SeqStmtNode>()) {
        // Multiple statements, not a simple predicated store
        return StmtExprMutator::VisitStmt_(if_stmt);
      }

      if (auto store = if_stmt->then_case.as<BufferStoreNode>()) {
        // Only handle global memory stores
        if (store->buffer.scope() == "global") {
          // Assume buffer has been flattened by FlattenBuffer pass
          ICHECK(store->indices.size() == 1)
              << "Expected flattened buffer with single index, but got "
              << store->indices.size() << " indices for buffer "
              << store->buffer->name;

          // Check if this is a Ramp-based store (vectorized)
          if (store->indices[0]->IsInstance<RampNode>()) {
            auto ramp = store->indices[0].as<RampNode>();
            // Check if stride is 1 (contiguous access)
            if (auto stride_imm = ramp->stride.as<IntImmNode>()) {
              if (stride_imm->value == 1) {
                // Get lanes from the index dtype
                int lanes = store->indices[0]->dtype.lanes();
                // Use bits() to correctly handle sub-byte types like
                // float4_e2m1fn
                int total_bits = lanes * store->buffer->dtype.bits();

                // Check for supported vector widths (32/64/128/256 bits)
                if (total_bits == 32 || total_bits == 64 || total_bits == 128 ||
                    total_bits == 256) {
                  return LowerToSTGPredicated(store, ramp->base, total_bits,
                                              if_stmt->condition);
                }
              }
            }
          } else {
            // Single element predicated store (non-Ramp)
            int bits = store->buffer->dtype.bits();
            if (bits == 32 || bits == 64 || bits == 128 || bits == 256) {
              return LowerToSTGPredicated(store, store->indices[0], bits,
                                          if_stmt->condition);
            }
          }
        }
      }
    }

    return StmtExprMutator::VisitStmt_(if_stmt);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *load) final {
    // Skip loads in async scope (will be lowered to cp.async)
    if (in_async_scope_) {
      return StmtExprMutator::VisitExpr_(load);
    }

    // Only handle global memory loads
    if (load->buffer.scope() != "global") {
      return StmtExprMutator::VisitExpr_(load);
    }

    // Check if we're in a predicated context (from IfThenElse store pattern)
    // In this case, we need to use predicated load regardless of
    // enable_non_predicated_
    bool use_predicated = current_predicate_.defined();

    // Skip if non-predicated lowering is disabled and we're not in predicated
    // context
    if (!enable_non_predicated_ && !use_predicated) {
      return StmtExprMutator::VisitExpr_(load);
    }

    // Assume buffer has been flattened by FlattenBuffer pass
    ICHECK(load->indices.size() == 1)
        << "Expected flattened buffer with single index, but got "
        << load->indices.size() << " indices for buffer " << load->buffer->name;

    // Check if this is a Ramp-based load (vectorized)
    if (load->indices[0]->IsInstance<RampNode>()) {
      auto ramp = load->indices[0].as<RampNode>();
      // Check if stride is 1 (contiguous access)
      if (auto stride_imm = ramp->stride.as<IntImmNode>()) {
        if (stride_imm->value == 1) {
          // Get lanes from the index dtype
          int lanes = load->indices[0]->dtype.lanes();
          // Use bits() to correctly handle sub-byte types like float4_e2m1fn
          int total_bits = lanes * load->buffer->dtype.bits();

          // Check for supported vector widths (32/64/128/256 bits)
          if (total_bits == 32 || total_bits == 64 || total_bits == 128 ||
              total_bits == 256) {
            if (use_predicated) {
              return LowerToLDGPredicated(load, ramp->base, total_bits,
                                          current_predicate_.value());
            }
            return LowerToLDG(load, ramp->base, total_bits);
          }
        }
      }
    } else {
      // Single element load (non-Ramp)
      int bits = load->buffer->dtype.bits();
      if (bits == 32 || bits == 64 || bits == 128 || bits == 256) {
        if (use_predicated) {
          return LowerToLDGPredicated(load, load->indices[0], bits,
                                      current_predicate_.value());
        }
        return LowerToLDG(load, load->indices[0], bits);
      }
    }

    return StmtExprMutator::VisitExpr_(load);
  }

  PrimExpr VisitExpr_(const CallNode *call) final {
    // Skip if predicated lowering is disabled
    if (!enable_predicated_) {
      return StmtExprMutator::VisitExpr_(call);
    }

    // Check for if_then_else pattern for predicated loads
    if (call->op.same_as(builtin::if_then_else()) && call->args.size() == 3) {
      // Skip if in async scope
      if (in_async_scope_) {
        return StmtExprMutator::VisitExpr_(call);
      }

      PrimExpr condition = call->args[0];
      PrimExpr then_value = call->args[1];
      PrimExpr else_value = call->args[2];

      // Check if else value is zero (required for predicated ldg)
      bool else_is_zero = false;
      if (auto bcast = else_value.as<BroadcastNode>()) {
        if (auto f = bcast->value.as<FloatImmNode>()) {
          else_is_zero = (f->value == 0.0f);
        } else if (auto i = bcast->value.as<IntImmNode>()) {
          else_is_zero = (i->value == 0);
        }
      } else if (auto f = else_value.as<FloatImmNode>()) {
        else_is_zero = (f->value == 0.0f);
      } else if (auto i = else_value.as<IntImmNode>()) {
        else_is_zero = (i->value == 0);
      }

      if (else_is_zero) {
        // Check if then_value is a BufferLoad from global memory with Ramp
        if (auto load = then_value.as<BufferLoadNode>()) {
          if (load->buffer.scope() == "global") {
            // Assume buffer has been flattened by FlattenBuffer pass
            ICHECK(load->indices.size() == 1)
                << "Expected flattened buffer with single index, but got "
                << load->indices.size() << " indices for buffer "
                << load->buffer->name;

            if (load->indices[0]->IsInstance<RampNode>()) {
              auto ramp = load->indices[0].as<RampNode>();
              if (auto stride_imm = ramp->stride.as<IntImmNode>()) {
                if (stride_imm->value == 1) {
                  // Get lanes from the index dtype
                  int lanes = load->indices[0]->dtype.lanes();
                  // Use bits() to correctly handle sub-byte types like
                  // float4_e2m1fn
                  int total_bits = lanes * load->buffer->dtype.bits();

                  // Check for supported vector widths (32/64/128/256 bits)
                  if (total_bits == 32 || total_bits == 64 ||
                      total_bits == 128 || total_bits == 256) {
                    return LowerToLDGPredicated(load, ramp->base, total_bits,
                                                condition);
                  }
                }
              }
            } else {
              // Single element predicated load (non-Ramp)
              int bits = load->buffer->dtype.bits();
              if (bits == 32 || bits == 64 || bits == 128 || bits == 256) {
                return LowerToLDGPredicated(load, load->indices[0], bits,
                                            condition);
              }
            }
          }
        }
      }
    }

    return StmtExprMutator::VisitExpr_(call);
  }

private:
  bool in_async_scope_{false};
  bool enable_non_predicated_{false};
  bool enable_predicated_{true};
  Optional<PrimExpr>
      current_predicate_; // Track predicate context for nested loads

  // Create access pointer for the buffer at given base offset
  PrimExpr CreateAccessPtr(const Buffer &buffer, const PrimExpr &base,
                           int access_mask) {
    // access_mask: 1 = read, 2 = write
    return buffer.access_ptr(access_mask, DataType::Handle(), 1, base);
  }

  // Lower a BufferLoad to ldg intrinsic
  PrimExpr LowerToLDG(const BufferLoadNode *load, const PrimExpr &base,
                      int bits) {
    PrimExpr ptr = CreateAccessPtr(load->buffer, base, 1);

    DataType ret_dtype;
    Op ldg_op;
    if (bits == 32) {
      ret_dtype = DataType::UInt(32);
      ldg_op = ldg32();
    } else if (bits == 64) {
      ret_dtype = DataType::UInt(32, 2);
      ldg_op = ldg64();
    } else if (bits == 128) {
      ret_dtype = DataType::UInt(32, 4);
      ldg_op = ldg128();
    } else if (bits == 256) {
      ret_dtype = DataType::UInt(32, 8);
      ldg_op = ldg256();
    } else {
      LOG(FATAL) << "Unsupported bit width for ldg: " << bits;
      return PrimExpr();
    }

    // Create ldg call
    PrimExpr ldg_result = Call(ret_dtype, ldg_op, {ptr});

    // Reinterpret to the original dtype if needed
    if (load->dtype != ret_dtype) {
      return Call(load->dtype, builtin::reinterpret(), {ldg_result});
    }
    return ldg_result;
  }

  // Lower a predicated BufferLoad to ldg intrinsic
  PrimExpr LowerToLDGPredicated(const BufferLoadNode *load,
                                const PrimExpr &base, int bits,
                                const PrimExpr &predicate) {
    PrimExpr ptr = CreateAccessPtr(load->buffer, base, 1);

    DataType ret_dtype;
    Op ldg_op;
    if (bits == 32) {
      ret_dtype = DataType::UInt(32);
      ldg_op = ldg32();
    } else if (bits == 64) {
      ret_dtype = DataType::UInt(32, 2);
      ldg_op = ldg64();
    } else if (bits == 128) {
      ret_dtype = DataType::UInt(32, 4);
      ldg_op = ldg128();
    } else if (bits == 256) {
      ret_dtype = DataType::UInt(32, 8);
      ldg_op = ldg256();
    } else {
      LOG(FATAL) << "Unsupported bit width for ldg: " << bits;
      return PrimExpr();
    }

    // Create predicated ldg call
    PrimExpr ldg_result = Call(ret_dtype, ldg_op, {ptr, predicate});

    // Reinterpret to the original dtype if needed
    if (load->dtype != ret_dtype) {
      return Call(load->dtype, builtin::reinterpret(), {ldg_result});
    }
    return ldg_result;
  }

  // Lower a BufferStore to stg intrinsic
  Stmt LowerToSTG(const BufferStoreNode *store, const PrimExpr &base,
                  int bits) {
    PrimExpr ptr = CreateAccessPtr(store->buffer, base, 2);

    // Get the value to store
    PrimExpr value = this->VisitExpr(store->value);

    // Reinterpret value to uint32xN if needed
    DataType store_dtype;
    const Op *stg_op;
    switch (bits) {
    case 32:
      store_dtype = DataType::UInt(32);
      stg_op = &stg32();
      break;
    case 64:
      store_dtype = DataType::UInt(32, 2);
      stg_op = &stg64();
      break;
    case 128:
      store_dtype = DataType::UInt(32, 4);
      stg_op = &stg128();
      break;
    case 256:
      store_dtype = DataType::UInt(32, 8);
      stg_op = &stg256();
      break;
    default:
      LOG(FATAL) << "Unsupported bit width for stg: " << bits;
      return Stmt();
    }

    // Reinterpret value if dtype doesn't match
    if (value.dtype() != store_dtype) {
      value = Call(store_dtype, builtin::reinterpret(), {value});
    }

    // Create stg call
    return Evaluate(Call(DataType::Handle(), *stg_op, {ptr, value}));
  }

  // Lower a predicated BufferStore to stg intrinsic
  Stmt LowerToSTGPredicated(const BufferStoreNode *store, const PrimExpr &base,
                            int bits, const PrimExpr &predicate) {
    PrimExpr ptr = CreateAccessPtr(store->buffer, base, 2);

    // Set predicate context so that nested loads also use predicated version
    Optional<PrimExpr> old_predicate = current_predicate_;
    current_predicate_ = predicate;

    // Get the value to store (loads inside will use predicated version)
    PrimExpr value = this->VisitExpr(store->value);

    // Restore old predicate context
    current_predicate_ = old_predicate;

    // Reinterpret value to uint32xN if needed
    DataType store_dtype;
    const Op *stg_op;
    switch (bits) {
    case 32:
      store_dtype = DataType::UInt(32);
      stg_op = &stg32();
      break;
    case 64:
      store_dtype = DataType::UInt(32, 2);
      stg_op = &stg64();
      break;
    case 128:
      store_dtype = DataType::UInt(32, 4);
      stg_op = &stg128();
      break;
    case 256:
      store_dtype = DataType::UInt(32, 8);
      stg_op = &stg256();
      break;
    default:
      LOG(FATAL) << "Unsupported bit width for stg: " << bits;
      return Stmt();
    }

    // Reinterpret value if dtype doesn't match
    if (value.dtype() != store_dtype) {
      value = Call(store_dtype, builtin::reinterpret(), {value});
    }

    // Create predicated stg call
    return Evaluate(Call(DataType::Handle(), *stg_op, {ptr, value, predicate}));
  }
};

using namespace tir::transform;

tvm::transform::Pass LowerLDGSTG() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    // Check if target is CUDA
    auto target_opt = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target_opt.defined()) {
      // No target bound, skip this pass
      return f;
    }
    Target target = target_opt.value();
    if (target->kind->name != "cuda") {
      // Not a CUDA target, skip
      return f;
    }

    // Check if target has "cutedsl" key - skip for CuTeDSL backend
    for (const auto &key : target->keys) {
      if (key == "cutedsl") {
        return f;
      }
    }

    // Read pass configurations
    // Non-predicated ldg/stg: default OFF
    bool enable_non_predicated =
        ctx->GetConfig<Bool>(kEnableLowerLDGSTG, Bool(false)).value();
    // Predicated ldg/stg: default OFF
    bool enable_predicated =
        ctx->GetConfig<Bool>(kEnableLowerLDGSTGPredicated, Bool(false)).value();

    // If both are disabled, skip the pass entirely
    if (!enable_non_predicated && !enable_predicated) {
      return f;
    }

    auto *n = f.CopyOnWrite();
    n->body =
        LowerLDGSTGRewriter(enable_non_predicated, enable_predicated)(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerLDGSTG", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerLDGSTG", LowerLDGSTG);
}

} // namespace tl
} // namespace tvm
