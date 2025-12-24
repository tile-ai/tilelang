#include "../op/builtin.h"
#include "arith/ir_mutator_with_analyzer.h"
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/data_type_rewriter.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;
using namespace arith;
class ConfigIndexBitwidthRewriter : public IndexDataTypeRewriter {
public:
  using Parent = IndexDataTypeRewriter;
  ConfigIndexBitwidthRewriter(int index_bitwidth)
      : _index_bitwidth_(index_bitwidth) {}

  Stmt operator()(const Stmt &s) { return VisitStmt(s); }

protected:
  using Parent::VisitExpr_;
  using Parent::VisitStmt_;

  PrimExpr VisitExpr_(const VarNode *op) final {
    if (op->dtype.is_int() && op->dtype.bits() < 64) {
      DataType new_dtype = DataType::Int(64);
      if (!var_remap_.count(op)) {
        var_remap_[op] = Var(op->name_hint, new_dtype);
      }
    }
    return Parent::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const IntImmNode *op) final {
    if (is_enabled_ && op->dtype.is_int() && op->dtype.bits() < 64) {
      return IntImm(DataType::Int(_index_bitwidth_), op->value);
    }
    return tvm::ffi::GetRef<PrimExpr>(op);
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
};

class IndexLegalizer : public IRMutatorWithAnalyzer {

public:
  static Stmt Rewrite(const Stmt &stmt) {
    Analyzer ana;
    auto pass = IndexLegalizer(&ana);
    return pass.VisitStmt(stmt);
  }

private:
  explicit IndexLegalizer(arith::Analyzer *ana) : IRMutatorWithAnalyzer(ana) {}

  class Int64Promoter : public IndexDataTypeRewriter {
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

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto buffer_store =
        Downcast<BufferStore>(IRMutatorWithAnalyzer::VisitStmt_(op));
    auto indices = buffer_store->indices;
    Array<PrimExpr> new_indices;
    for (auto index : indices) {
      if (index->dtype.is_int() && index->dtype.bits() < 64) {
        auto int_bound = analyzer_->const_int_bound(index);
        if (int_bound->max_value >= (1LL << (index->dtype.bits() - 1)) - 1 ||
            int_bound->min_value < -(1LL << (index->dtype.bits() - 1))) {
          Int64Promoter promoter;
          index = promoter(index);
          new_indices.push_back(index);
          continue;
        }
      }
      new_indices.push_back(index);
    }
    buffer_store.CopyOnWrite()->indices = new_indices;
    return std::move(buffer_store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto buffer_load =
        Downcast<BufferLoad>(IRMutatorWithAnalyzer::VisitExpr_(op));
    auto indices = buffer_load->indices;
    Array<PrimExpr> new_indices;
    for (auto index : indices) {
      if (index->dtype.is_int() && index->dtype.bits() < 64) {
        auto int_bound = analyzer_->const_int_bound(index);
        if (int_bound->max_value >= (1LL << (index->dtype.bits() - 1)) - 1 ||
            int_bound->min_value < -(1LL << (index->dtype.bits() - 1))) {
          Int64Promoter promoter;
          index = promoter(index);
          new_indices.push_back(index);
          continue;
        }
      }
      new_indices.push_back(index);
    }
    buffer_load.CopyOnWrite()->indices = new_indices;
    return std::move(buffer_load);
  }
};

tvm::transform::Pass ConfigIndexBitwidth() {
  using namespace tir::transform;
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto *n = f.CopyOnWrite();
    // Get pass config `tl.config_index_bitwidth`
    tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
    Optional<Integer> opt_config_index_bitwidth =
        ctxt->GetConfig(kConfigIndexBitwidth, Optional<Integer>());
    if (opt_config_index_bitwidth.defined()) {
      int config_index_bitwidth = opt_config_index_bitwidth.value()->value;
      n->body = ConfigIndexBitwidthRewriter(config_index_bitwidth)(n->body);
    }
    // Legalize out-of-bound indices to be int64
    n->body = IndexLegalizer::Rewrite(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ConfigIndexBitwidth", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ConfigIndexBitwidth",
                        ConfigIndexBitwidth);
}

} // namespace tl
} // namespace tvm
