// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "../op/builtin.h"
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/data_type_rewriter.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;
class ConfigIndexBitwidthRewriter : public IndexDataTypeRewriter {
public:
  using Parent = IndexDataTypeRewriter;
  ConfigIndexBitwidthRewriter(int index_bitwidth, bool auto_check)
      : _index_bitwidth_(index_bitwidth), _auto_check_(auto_check) {}

  Stmt operator()(Stmt s) { return VisitStmt(s); }

protected:
  using Parent::VisitExpr_;
  using Parent::VisitStmt_;

  PrimExpr VisitExpr_(const VarNode *op) final {
    if (op->dtype.is_int()) {
      DataType new_dtype = op->dtype;
      if (_auto_check_) {
        new_dtype = (op->dtype.bits() < 64) ? DataType::Int(64) : op->dtype;
      } else if (op->dtype.bits() < _index_bitwidth_) {
        new_dtype = DataType::Int(_index_bitwidth_);
      }

      if (new_dtype != op->dtype && !var_remap_.count(op)) {
        var_remap_[op] = Var(op->name_hint, new_dtype);
      }
    }
    return Parent::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const IntImmNode *op) final {
    if (is_enabled_ && op->dtype.is_int()) {
      if (_auto_check_) {
        int64_t value = op->value;
        int required_bits = value > INT32_MAX || value < INT32_MIN ? 64 : 32;
        return IntImm(DataType::Int(required_bits), value);
      } else if (op->dtype.bits() < _index_bitwidth_) {
        return IntImm(DataType::Int(_index_bitwidth_), op->value);
      }
    }
    return GetRef<PrimExpr>(op);
  }

  PrimExpr VisitExpr_(const CastNode *op) final {
    if (is_enabled_ && op->dtype.is_int() && op->dtype.bits() < 64) {
      PrimExpr value = VisitExpr(op->value);
      return Cast(DataType::Int(_index_bitwidth_), value);
    }
    return Parent::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    // Force indices to be int64
    bool is_enabled = is_enabled_;
    is_enabled_ = true;
    auto node = Downcast<BufferStore>(Parent::VisitStmt_(op));
    is_enabled_ = is_enabled;
    return std::move(node);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    // Force indices to be int64
    bool is_enabled = is_enabled_;
    is_enabled_ = true;
    auto node = Downcast<BufferLoad>(Parent::VisitExpr_(op));
    is_enabled_ = is_enabled;
    return std::move(node);
  }

  int _index_bitwidth_;
  bool _auto_check_;
};

tvm::transform::Pass ConfigIndexBitwidth() {
  using namespace tir::transform;
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto *n = f.CopyOnWrite();
    tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
    Optional<Integer> opt_config_index_bitwidth =
        ctxt->GetConfig(kConfigIndexBitwidth, Optional<Integer>());

    bool auto_check = !opt_config_index_bitwidth.defined();
    int config_index_bitwidth =
        auto_check ? 64 : opt_config_index_bitwidth.value()->value;

    n->body = ConfigIndexBitwidthRewriter(config_index_bitwidth,
                                          auto_check)(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ConfigIndexBitwidth", {});
}

TVM_REGISTER_GLOBAL("tl.transform.ConfigIndexBitwidth")
    .set_body_typed(ConfigIndexBitwidth);

} // namespace tl
} // namespace tvm
