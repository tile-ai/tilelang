// Test-only bridge to the compiler's fragment containment contract.
#include "layout/utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/expr.h>

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def(
      "tl.testing.ProveFragmentContains",
      [](tvm::tl::Fragment small, tvm::tl::Fragment large, int small_min,
         int large_min, int small_index, int large_index, bool check_index,
         int small_extent, int large_extent) {
        if (small_min >= 0) {
          small = small->BindThreadRange(
              tvm::Range::FromMinExtent(small_min, small_extent));
        }
        if (large_min >= 0) {
          large = large->BindThreadRange(
              tvm::Range::FromMinExtent(large_min, large_extent));
        }
        tvm::arith::Analyzer analyzer;
        tvm::tirx::Var logical_index("logical_index", tvm::DataType::Int(32));
        analyzer.Bind(logical_index, tvm::Range(0, 4));
        tvm::ffi::Array<tvm::PrimExpr> small_indices{
            small_index < 0 ? tvm::PrimExpr(logical_index)
                            : tvm::IntImm(tvm::DataType::Int(32), small_index)};
        tvm::ffi::Array<tvm::PrimExpr> large_indices{
            large_index < 0 ? tvm::PrimExpr(logical_index)
                            : tvm::IntImm(tvm::DataType::Int(32), large_index)};
        return tvm::tl::ProveFragmentContains(
            small, large, small_indices, large_indices, analyzer, check_index);
      });
}
