#ifndef TILELANG_TRANSFORM_LOWER_TILE_OP_H_
#define TILELANG_TRANSFORM_LOWER_TILE_OP_H_

#include "op/operator.h"

#include <tvm/tirx/function.h>

#include <functional>

namespace tvm::tl {

struct LowerTileOpOptions {
  std::function<TileOperator(const TileOperator &, sym::Analyzer)>
      rewrite_operator;
  std::function<Expr(const Call &, sym::Analyzer)> rewrite_access_pointer;
};

TILELANG_DLL tirx::Function LowerTileOperators(tirx::Function function,
                                               LowerTileOpOptions options = {});

} // namespace tvm::tl

#endif
