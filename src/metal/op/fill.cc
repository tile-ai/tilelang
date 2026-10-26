#include "backend/common/op/fill.h"
#include "backend/common/target_utils.h"
#include "metal/op/utils.h"
#include "op/prim.h"

#include <tvm/backend/metal/op/cooperative_tensor.h>

namespace tvm::tl::metal {

struct Fill {
  static Stmt Lower(const FillNode &operation, const LowerArgs &arguments,
                    sym::Analyzer analyzer) {
    bool cooperative = IsCooperativeTensorBuffer(operation.dst);
    if (!cooperative && !IsSIMDGroupBuffer(operation.dst)) {
      return backend::Fill::Lower(operation, arguments, analyzer);
    }
    int64_t elements = 1;
    for (const Range &range : operation.region) {
      auto extent = GetConstInt(range->extent);
      TVM_FFI_CHECK(extent && *extent > 0 && elements <= INT64_MAX / *extent,
                    ValueError)
          << "Metal matrix fill requires positive constant extents";
      elements *= *extent;
    }
    int rows = cooperative ? 16 : 8;
    int columns = cooperative ? 32 : 8;
    int matrix_elements = rows * columns;
    TVM_FFI_CHECK(elements % matrix_elements == 0, ValueError)
        << "Metal matrix fill requires a multiple of " << matrix_elements
        << " elements";
    auto intrinsic = cooperative
                         ? ::tvm::backend::metal::cooperative_tensor_fill_op()
                         : Op::Get("tirx.metal.make_filled_simdgroup_matrix");
    ffi::Array<Stmt> statements;
    for (int64_t index = 0; index < elements / matrix_elements; ++index) {
      statements.push_back(
          Evaluate(Call(PrimType::Void(), intrinsic,
                        {operation.dst.data(), IntImm::Int32(index),
                         tl::CastValue(operation.value, operation.dst.dtype()),
                         IntImm::Int32(rows), IntImm::Int32(columns)})));
    }
    return SeqStmt(statements);
  }
};

namespace {

const bool registered = []() {
  RegisterFillImpl({"metal.Fill", TargetIsMetal, Fill::Lower});
  return true;
}();

} // namespace
} // namespace tvm::tl::metal
