#include "codegen_sunmmio.h"

#include "../../op/utils.h"
#include "../../transform/common/attr.h"
#include "../sunmmio_utils.h"
#include "sunmmio_mlir_builder.h"
#include "sunmmio_mlir_context.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <optional>
#include <set>
#include <sstream>
#include <string>

#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/ir/op.h>
#include <tvm/node/structural_equal.h>
#include <tvm/runtime/logging.h>

namespace tvm {
namespace codegen {

namespace {

using namespace tir;

bool IsSunmmioLocalVarBuffer(const tir::Buffer &buffer) {
  if (!buffer.defined()) {
    return false;
  }
  return buffer.scope() == "local.var";
}

struct TilesScopeInfo {
  const ForNode *root{nullptr};
  ffi::Array<PrimExpr> domain_shape;
  std::vector<SunMMIOValue> domain_values;
  std::vector<const ForNode *> domain_loops;
  std::vector<const ForNode *> execution_loops;
  const ForNode *interior_axis0_loop{nullptr};
  const ForNode *interior_axis1_loop{nullptr};
  std::vector<int> execution_domain_axes;
  std::vector<int64_t> tile_shape;
  Stmt tile_block_body;
  PrimExpr tail_predicate;
  Stmt full_tile_body;
  Stmt tail_tile_body;
  Stmt full_tile_block_body;
  Stmt tail_tile_block_body;
  const ForNode *tail_interior_axis0_loop{nullptr};
  const ForNode *tail_interior_axis1_loop{nullptr};
  bool is_reduce_scope{false};
};

struct TileBlockState {
  const TilesScopeInfo *scope{nullptr};
  SunmmioMlirContext *mlir_ctx{nullptr};
  std::unordered_map<const VarNode *, SunMMIOValue> let_values;
  std::unordered_map<const VarNode *, int64_t> let_unsqueeze_axes;
  std::unordered_map<const BufferNode *, SunMMIOValue> register_tile_values;
  std::unordered_map<const BufferNode *, SunMMIOType> register_tile_types;
  std::unordered_map<const BufferNode *, int64_t> register_unsqueeze_axes;
  std::unordered_map<const BufferNode *, SunMMIOValue> local_tile_values;
  std::unordered_map<const BufferNode *, int64_t> local_unit_tile_axes;
  std::unordered_map<std::string, SunMMIOValue> tile_view_cache;
  std::unordered_map<std::string, SunMMIOValue> current_tile_values;
  std::optional<SunMMIOValue> tile_mask;
  std::optional<PrimExpr> active_tail_store_predicate;
  const ForNode *interior_axis0_loop{nullptr};
  const ForNode *interior_axis1_loop{nullptr};
};

struct TileAccessInfo {
  Buffer buffer;
  int tile_rank{0};
  std::vector<int64_t> tile_shape;
  std::vector<int> tile_axes;
  std::vector<SunMMIOValue> partition_indices;
  std::vector<int64_t> tiled_dims;
  bool promoted_unit_tile_view{false};
  int64_t unsqueeze_axis{-1};
  bool requires_aligned_1d_load{false};
  int64_t aligned_load_bytes{0};
  int64_t aligned_load_elems{0};
};

struct Aligned1DAddressInfo {
  SunMMIOValue offset_elems;
  std::vector<SunMMIOValue> partition_indices;
};

struct TiledIndexMatch {
  int64_t offset{0};
  bool uses_execution_index{false};
  PrimExpr partition_index;
};

struct TailMaskInfo {
  SunMMIOValue valid_rows;
  SunMMIOValue valid_cols;
  SunMMIOValue row_tail_cond;
  SunMMIOValue col_tail_cond;
  SunMMIOType mask_type;
};

std::optional<int> GetExecutionAxisAnnotation(const ForNode *loop) {
  if (loop == nullptr) {
    return std::nullopt;
  }
  auto axis_it = loop->annotations.find(tl::attr::tile_execution_axis);
  if (axis_it == loop->annotations.end()) {
    return std::nullopt;
  }
  return static_cast<int>(Downcast<Integer>((*axis_it).second)->value);
}

std::vector<int64_t>
ParseStaticIntArray(const ffi::Map<ffi::String, ffi::Any> &annotations,
                    const char *key) {
  auto it = annotations.find(key);
  ICHECK(it != annotations.end()) << "Missing tile annotation `" << key << "`";
  ffi::Array<PrimExpr> values = Downcast<ffi::Array<PrimExpr>>((*it).second);
  std::vector<int64_t> result;
  result.reserve(values.size());
  for (const PrimExpr &value : values) {
    const auto *imm = value.as<IntImmNode>();
    ICHECK(imm) << "Tile annotation `" << key << "` must be static IntImm";
    result.push_back(static_cast<int64_t>(imm->value));
  }
  return result;
}

std::optional<int64_t> TryGetIntImm(const PrimExpr &expr) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    return static_cast<int64_t>(imm->value);
  }
  return std::nullopt;
}

bool ContainsFloorDivOrMod(const PrimExpr &expr) {
  bool found = false;
  tir::PostOrderVisit(expr, [&](const ObjectRef &obj) {
    found = found || obj.as<FloorDivNode>() != nullptr ||
            obj.as<FloorModNode>() != nullptr;
  });
  return found;
}

bool ContainsAnyVar(const PrimExpr &expr,
                    const std::vector<const VarNode *> &vars) {
  bool found = false;
  tir::PostOrderVisit(expr, [&](const ObjectRef &obj) {
    if (found) {
      return;
    }
    if (const auto *var = obj.as<VarNode>()) {
      found = std::find(vars.begin(), vars.end(), var) != vars.end();
    }
  });
  return found;
}

std::optional<PrimExpr> TryRewritePositiveFloorDivTailCompare(
    const PrimExpr &condition,
    const std::vector<const VarNode *> &interior_vars) {
  auto rewrite = [&](const PrimExpr &lhs,
                     const PrimExpr &rhs) -> std::optional<PrimExpr> {
    auto rhs_imm = TryGetIntImm(rhs);
    if (!rhs_imm.has_value()) {
      return std::nullopt;
    }

    std::vector<PrimExpr> terms;
    std::function<void(const PrimExpr &)> flatten_add =
        [&](const PrimExpr &expr) {
          if (const auto *add = expr.as<AddNode>()) {
            flatten_add(add->a);
            flatten_add(add->b);
            return;
          }
          terms.push_back(expr);
        };
    flatten_add(lhs);

    std::optional<PrimExpr> div_numerator;
    int64_t divisor = 0;
    PrimExpr linear_part;
    for (const PrimExpr &term : terms) {
      if (const auto *div = term.as<FloorDivNode>()) {
        if (div_numerator.has_value()) {
          return std::nullopt;
        }
        auto imm_divisor = TryGetIntImm(div->b);
        if (!imm_divisor.has_value() || imm_divisor.value() <= 0 ||
            ContainsFloorDivOrMod(div->a) ||
            !ContainsAnyVar(div->a, interior_vars)) {
          return std::nullopt;
        }
        div_numerator = div->a;
        divisor = imm_divisor.value();
        continue;
      }
      if (ContainsFloorDivOrMod(term)) {
        return std::nullopt;
      }
      if (ContainsAnyVar(term, interior_vars)) {
        return std::nullopt;
      }
      linear_part = linear_part.defined() ? linear_part + term : term;
    }

    if (!div_numerator.has_value()) {
      return std::nullopt;
    }

    DataType dtype = div_numerator.value().dtype();
    PrimExpr divisor_expr = IntImm(dtype, divisor);
    PrimExpr scaled_lhs = (linear_part.defined() ? linear_part * divisor_expr
                                                 : PrimExpr(IntImm(dtype, 0))) +
                          div_numerator.value();
    PrimExpr scaled_rhs = IntImm(dtype, rhs_imm.value() * divisor);
    return scaled_lhs < scaled_rhs;
  };

  if (const auto *lt = condition.as<LTNode>()) {
    return rewrite(lt->a, lt->b);
  }
  return std::nullopt;
}

std::optional<PrimExpr>
TryRewriteAffineInteriorLT(const PrimExpr &condition,
                           const std::vector<const VarNode *> &interior_vars) {
  const auto *lt = condition.as<LTNode>();
  if (lt == nullptr || interior_vars.empty() ||
      ContainsAnyVar(lt->b, interior_vars)) {
    return std::nullopt;
  }

  Array<Var> vars;
  vars.reserve(interior_vars.size());
  for (const VarNode *var : interior_vars) {
    vars.push_back(ffi::GetRef<Var>(var));
  }

  arith::Analyzer analyzer;
  Array<PrimExpr> coeffs = arith::DetectLinearEquation(lt->a, vars);
  if (coeffs.empty() || coeffs.size() != vars.size() + 1) {
    return std::nullopt;
  }

  int active_interior_vars = 0;
  for (size_t i = 0; i < vars.size(); ++i) {
    PrimExpr coeff = analyzer.Simplify(coeffs[i]);
    if (analyzer.CanProve(coeff == make_zero(coeff.dtype()))) {
      continue;
    }
    if (!analyzer.CanProve(coeff == make_const(coeff.dtype(), 1))) {
      return std::nullopt;
    }
    ++active_interior_vars;
  }
  if (active_interior_vars != 1) {
    return std::nullopt;
  }

  PrimExpr scalar_base = analyzer.Simplify(coeffs.back());
  // Keep affine tile offsets out of i16 mask arithmetic, which SUVM cannot
  // lower, by moving the scalar base to the other side of the comparison.
  PrimExpr tile_part = analyzer.Simplify(lt->a - scalar_base);
  PrimExpr adjusted_rhs = analyzer.Simplify(lt->b - scalar_base);
  return tile_part < adjusted_rhs;
}

std::vector<const ForNode *> CollectLinearForChain(const ForNode *root) {
  std::vector<const ForNode *> loops;
  const ForNode *current = root;
  while (current != nullptr) {
    loops.push_back(current);
    current = current->body.as<ForNode>();
  }
  return loops;
}

SunmmioMlirContext *
TryGetMlirContext(std::unique_ptr<SunMMIOBuilder> &builder) {
  auto *suvm_builder = dynamic_cast<SuvmSunmmioBuilder *>(builder.get());
  if (!suvm_builder) {
    return nullptr;
  }
  return &suvm_builder->Context();
}

SunMMIOType MakeTileType(DataType dtype, const std::vector<int64_t> &shape) {
  SunMMIOType type;
  type.kind = SunMMIOType::Kind::kTile;
  type.dtype = CanonicalizeSuvmDType(dtype).with_lanes(1);
  type.lanes = 1;
  for (int64_t dim : shape) {
    type.shape.push_back(IntImm(DataType::Int(32), dim));
  }
  return type;
}

SunMMIOType MakeTileViewType(DataType dtype,
                             const std::vector<int64_t> &shape) {
  SunMMIOType type;
  type.kind = SunMMIOType::Kind::kTileView;
  type.dtype = CanonicalizeSuvmDType(dtype).with_lanes(1);
  type.lanes = 1;
  for (int64_t dim : shape) {
    type.shape.push_back(IntImm(DataType::Int(32), dim));
  }
  return type;
}

std::pair<const ForNode *, const ForNode *>
FindInteriorLoops(const Stmt &stmt) {
  if (const auto *loop = stmt.as<ForNode>()) {
    auto axis_it = loop->annotations.find(tl::attr::tile_interior_axis);
    if (axis_it != loop->annotations.end()) {
      int axis = Downcast<Integer>((*axis_it).second)->value;
      if (axis == 0) {
        const ForNode *axis1 = nullptr;
        if (const auto *inner = loop->body.as<ForNode>()) {
          auto inner_axis_it =
              inner->annotations.find(tl::attr::tile_interior_axis);
          if (inner_axis_it != inner->annotations.end() &&
              Downcast<Integer>((*inner_axis_it).second)->value == 1) {
            axis1 = inner;
          }
        }
        return {loop, axis1};
      }
    }
  }

  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &s : seq->seq) {
      auto found = FindInteriorLoops(s);
      if (found.first != nullptr) {
        return found;
      }
    }
    return {nullptr, nullptr};
  }

  if (const auto *ifs = stmt.as<IfThenElseNode>()) {
    auto found = FindInteriorLoops(ifs->then_case);
    if (found.first != nullptr) {
      return found;
    }
    if (ifs->else_case.defined()) {
      return FindInteriorLoops(ifs->else_case.value());
    }
  }

  return {nullptr, nullptr};
}

bool IsTileLike(const SunMMIOValue &value) {
  return value.type.kind == SunMMIOType::Kind::kTile;
}

bool IsScalarLike(const SunMMIOValue &value) {
  return value.type.kind == SunMMIOType::Kind::kScalar ||
         value.type.kind == SunMMIOType::Kind::kIndex;
}

bool IsRsramScope(const std::string &scope) {
  return scope == "shared.rsram" || scope == "rsram";
}

bool ContainsVectorCoreInTileReduce(const Stmt &stmt) {
  if (!stmt.defined()) {
    return false;
  }
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *call = eval->value.as<CallNode>()) {
      const auto *op_node = call->op.as<OpNode>();
      return op_node && op_node->name == "tl.vector_core_in_tile_reduce";
    }
    return false;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &s : seq->seq) {
      if (ContainsVectorCoreInTileReduce(s)) {
        return true;
      }
    }
    return false;
  }
  if (const auto *ifs = stmt.as<IfThenElseNode>()) {
    if (ContainsVectorCoreInTileReduce(ifs->then_case)) {
      return true;
    }
    return ifs->else_case.defined() &&
           ContainsVectorCoreInTileReduce(ifs->else_case.value());
  }
  if (const auto *loop = stmt.as<ForNode>()) {
    return ContainsVectorCoreInTileReduce(loop->body);
  }
  if (const auto *alloc = stmt.as<AllocateNode>()) {
    return ContainsVectorCoreInTileReduce(alloc->body);
  }
  if (const auto *decl = stmt.as<DeclBufferNode>()) {
    return ContainsVectorCoreInTileReduce(decl->body);
  }
  if (const auto *let = stmt.as<LetStmtNode>()) {
    return ContainsVectorCoreInTileReduce(let->body);
  }
  return false;
}

bool IsReduceLikeTileBody(const Stmt &stmt) {
  if (ContainsVectorCoreInTileReduce(stmt)) {
    return true;
  }
  const auto *seq = stmt.as<SeqStmtNode>();
  if (!seq) {
    return false;
  }
  bool has_guard = false;
  bool has_interior = false;
  for (const Stmt &s : seq->seq) {
    has_guard = has_guard || s.as<IfThenElseNode>() != nullptr;
    has_interior = has_interior || FindInteriorLoops(s).first != nullptr;
  }
  return has_guard && has_interior;
}

std::vector<int64_t> ExtractStaticShape(const SunMMIOType &type) {
  std::vector<int64_t> shape;
  shape.reserve(type.shape.size());
  for (const PrimExpr &dim : type.shape) {
    const auto *imm = dim.as<IntImmNode>();
    ICHECK(imm)
        << "Tiles lowering currently requires static tile/memtensor shape";
    shape.push_back(static_cast<int64_t>(imm->value));
  }
  return shape;
}

std::vector<int64_t> ExtractStaticPrimExprs(llvm::ArrayRef<PrimExpr> exprs,
                                            const char *what) {
  std::vector<int64_t> values;
  values.reserve(exprs.size());
  for (const PrimExpr &expr : exprs) {
    const auto *imm = expr.as<IntImmNode>();
    ICHECK(imm) << what << " must be static for aligned 1D access";
    values.push_back(static_cast<int64_t>(imm->value));
  }
  return values;
}

bool StaticShapesEqual(const SunMMIOType &a, const SunMMIOType &b) {
  return ExtractStaticShape(a) == ExtractStaticShape(b);
}

bool CanBroadcastShapeTo(const std::vector<int64_t> &src_shape,
                         const std::vector<int64_t> &dst_shape) {
  if (src_shape.size() != dst_shape.size()) {
    return false;
  }
  for (size_t i = 0; i < src_shape.size(); ++i) {
    if (src_shape[i] != dst_shape[i] && src_shape[i] != 1) {
      return false;
    }
  }
  return true;
}

std::optional<int64_t> GetStaticLoopExtent(const ForNode *loop) {
  if (loop == nullptr) {
    return std::nullopt;
  }
  const auto *imm = loop->extent.as<IntImmNode>();
  if (!imm) {
    return std::nullopt;
  }
  return static_cast<int64_t>(imm->value);
}

std::optional<int> GetInteriorAxisAnnotation(const ForNode *loop) {
  if (loop == nullptr) {
    return std::nullopt;
  }
  auto axis_it = loop->annotations.find(tl::attr::tile_interior_axis);
  if (axis_it == loop->annotations.end()) {
    return std::nullopt;
  }
  return static_cast<int>(Downcast<Integer>((*axis_it).second)->value);
}

std::optional<TiledIndexMatch>
MatchTiledIndex(const PrimExpr &index, const Var &exec, const Var &interior,
                int64_t tile_extent, bool allow_standalone_interior,
                arith::Analyzer *analyzer) {
  auto matches_integer_var = [](const PrimExpr &expr, const Var &var) {
    PrimExpr current = expr;
    while (const auto *cast = current.as<CastNode>()) {
      DataType source_dtype = cast->value.dtype();
      if ((!cast->dtype.is_int() && !cast->dtype.is_uint()) ||
          (!source_dtype.is_int() && !source_dtype.is_uint()) ||
          cast->dtype.bits() < source_dtype.bits()) {
        break;
      }
      current = cast->value;
    }
    return current.same_as(var);
  };

  if (matches_integer_var(index, interior)) {
    if (!allow_standalone_interior) {
      return std::nullopt;
    }
    return TiledIndexMatch{0, false, PrimExpr(IntImm(index.dtype(), 0))};
  }

  std::vector<PrimExpr> terms;
  std::function<void(const PrimExpr &)> flatten_add =
      [&](const PrimExpr &expr) {
        if (const auto *add = expr.as<AddNode>()) {
          flatten_add(add->a);
          flatten_add(add->b);
          return;
        }
        terms.push_back(expr);
      };
  flatten_add(index);

  bool seen_interior = false;
  bool seen_exec = false;
  int64_t const_offset = 0;
  PrimExpr dynamic_base;

  auto match_exec_mul = [&](const PrimExpr &expr) -> bool {
    // Simplification folds exec * 1 + interior to bare exec when the logical
    // tile extent is one and the interior coordinate is always zero.
    if (tile_extent == 1 && matches_integer_var(expr, exec)) {
      return true;
    }
    const auto *mul = expr.as<MulNode>();
    if (!mul) {
      return false;
    }
    auto matches = [&](const PrimExpr &var_term,
                       const PrimExpr &imm_term) -> bool {
      if (!matches_integer_var(var_term, exec)) {
        return false;
      }
      const auto *imm = imm_term.as<IntImmNode>();
      return imm && static_cast<int64_t>(imm->value) == tile_extent;
    };
    return matches(mul->a, mul->b) || matches(mul->b, mul->a);
  };

  for (const PrimExpr &term : terms) {
    if (matches_integer_var(term, interior)) {
      if (seen_interior) {
        return std::nullopt;
      }
      seen_interior = true;
      continue;
    }
    if (match_exec_mul(term)) {
      if (seen_exec) {
        return std::nullopt;
      }
      seen_exec = true;
      continue;
    }
    if (const auto *imm = term.as<IntImmNode>()) {
      const_offset += static_cast<int64_t>(imm->value);
      continue;
    }
    dynamic_base = dynamic_base.defined() ? dynamic_base + term : term;
  }

  bool has_unit_exec_without_interior = tile_extent == 1 && seen_exec;
  if ((!seen_interior && !has_unit_exec_without_interior) ||
      const_offset % tile_extent != 0) {
    return std::nullopt;
  }

  PrimExpr partition_index =
      PrimExpr(IntImm(index.dtype(), const_offset / tile_extent));
  if (dynamic_base.defined()) {
    PrimExpr extent = PrimExpr(IntImm(index.dtype(), tile_extent));
    PrimExpr remainder = analyzer->Simplify(floormod(dynamic_base, extent));
    if (!analyzer->CanProve(remainder == PrimExpr(IntImm(index.dtype(), 0)))) {
      return std::nullopt;
    }
    PrimExpr dynamic_partition =
        analyzer->Simplify(floordiv(dynamic_base, extent));
    partition_index = analyzer->Simplify(partition_index + dynamic_partition);
  }

  if (seen_exec) {
    partition_index = analyzer->Simplify(partition_index + exec);
    return TiledIndexMatch{const_offset / tile_extent, true, partition_index};
  }
  if (allow_standalone_interior) {
    return TiledIndexMatch{const_offset / tile_extent, false, partition_index};
  }
  return std::nullopt;
}

} // namespace

bool CodeGenTileLangSunMMIO::TryLowerTilesScope(const tir::ForNode *op) {
  if (!op->annotations.count(tl::attr::kTileDomain)) {
    return false;
  }

  TilesScopeInfo scope;
  scope.root = op;
  scope.domain_shape =
      Downcast<ffi::Array<PrimExpr>>(op->annotations.at(tl::attr::kTileDomain));
  {
    std::vector<int64_t> parsed_axes = ParseStaticIntArray(
        op->annotations, tl::attr::tile_execution_domain_axes);
    scope.execution_domain_axes.reserve(parsed_axes.size());
    for (int64_t axis : parsed_axes) {
      scope.execution_domain_axes.push_back(static_cast<int>(axis));
    }
  }
  scope.tile_shape =
      ParseStaticIntArray(op->annotations, tl::attr::tile_tile_size);
  ICHECK_EQ(scope.execution_domain_axes.size(), scope.tile_shape.size())
      << "tile.execution_domain_axes and tile.tile_size rank mismatch";

  std::vector<const ForNode *> chain = CollectLinearForChain(op);
  ICHECK(!chain.empty()) << "Tiles scope root must be a loop";
  const size_t domain_rank = scope.domain_shape.size();
  const size_t shared_prefix_depth = std::min(chain.size(), domain_rank);
  const bool has_partial_execution_prefix = shared_prefix_depth < domain_rank;
  for (size_t i = 0; i < shared_prefix_depth; ++i) {
    scope.domain_loops.push_back(chain[i]);
  }
  scope.execution_loops.assign(scope.execution_domain_axes.size(), nullptr);
  for (const ForNode *loop : scope.domain_loops) {
    auto axis_it = loop->annotations.find(tl::attr::tile_execution_axis);
    if (axis_it == loop->annotations.end()) {
      continue;
    }
    int exec_axis = Downcast<Integer>((*axis_it).second)->value;
    ICHECK_GE(exec_axis, 0);
    ICHECK_LT(static_cast<size_t>(exec_axis), scope.execution_loops.size())
        << "tile.execution_axis is out of range";
    scope.execution_loops[static_cast<size_t>(exec_axis)] = loop;
  }
  if (has_partial_execution_prefix) {
    bool has_shared_execution_loop = false;
    for (const ForNode *loop : scope.execution_loops) {
      has_shared_execution_loop = has_shared_execution_loop || loop != nullptr;
    }
    ICHECK(has_shared_execution_loop)
        << "Partial Tiles scope must expose at least one shared execution loop";
  } else {
    for (const ForNode *loop : scope.execution_loops) {
      ICHECK(loop != nullptr)
          << "Tiles scope is missing an execution loop for one tile axis";
    }
  }

  Stmt tile_scope_stmt = scope.domain_loops.back()->body;
  scope.is_reduce_scope = IsReduceLikeTileBody(tile_scope_stmt);
  if (has_partial_execution_prefix) {
    auto loops = FindInteriorLoops(tile_scope_stmt);
    scope.interior_axis0_loop = loops.first;
    scope.interior_axis1_loop = loops.second;
    scope.tile_block_body = tile_scope_stmt;
  } else if (scope.is_reduce_scope) {
    auto loops = FindInteriorLoops(tile_scope_stmt);
    scope.interior_axis0_loop = loops.first;
    scope.interior_axis1_loop = loops.second;
    ICHECK(scope.interior_axis0_loop != nullptr)
        << "Reduce tiles scope is missing interior axis 0 loop";
    scope.tile_block_body = tile_scope_stmt;
  } else if (const auto *ifs = tile_scope_stmt.as<IfThenElseNode>()) {
    // This wrapper is decomposed into the full-tile and tail-tile paths below,
    // so it never enters lower_stmt itself.
    MarkVisitedNodeType(ifs->GetTypeKey());
    scope.tail_predicate = ifs->condition;
    scope.full_tile_body = ifs->then_case;
    scope.tail_tile_body =
        ifs->else_case.defined() ? ifs->else_case.value() : Stmt();
    auto full_loops = FindInteriorLoops(scope.full_tile_body);
    scope.interior_axis0_loop = full_loops.first;
    scope.interior_axis1_loop = full_loops.second;
    ICHECK(scope.interior_axis0_loop != nullptr)
        << "Tiles full-tile branch is missing interior axis 0 loop";
    scope.full_tile_block_body = scope.full_tile_body;
    auto tail_loops = FindInteriorLoops(scope.tail_tile_body);
    scope.tail_interior_axis0_loop = tail_loops.first;
    scope.tail_interior_axis1_loop = tail_loops.second;
    ICHECK(scope.tail_interior_axis0_loop != nullptr)
        << "Tiles tail-tile branch is missing interior axis 0 loop";
    scope.tail_tile_block_body = scope.tail_tile_body;
    scope.tile_block_body = scope.full_tile_block_body;
  } else {
    auto loops = FindInteriorLoops(tile_scope_stmt);
    scope.interior_axis0_loop = loops.first;
    scope.interior_axis1_loop = loops.second;
    ICHECK(scope.interior_axis0_loop != nullptr)
        << "Tiles scope is missing interior axis 0 loop";
    scope.tile_block_body = tile_scope_stmt;
  }

  SunmmioMlirContext *mlir_ctx = TryGetMlirContext(builder_);
  ICHECK(mlir_ctx != nullptr)
      << "Tiles lowering currently expects SuvmSunmmioBuilder";

  const tl::SunmmioTileProcessorConfig tile_processor_config =
      tl::GetSunmmioTileProcessorConfig(target_);
  auto populate_aligned_1d_access = [&](TileAccessInfo *access) {
    const DataType dtype = CanonicalizeSuvmDType(access->buffer->dtype);
    const int64_t align_bytes =
        static_cast<int64_t>(tile_processor_config.rsram_align_bytes);
    ICHECK_GT(align_bytes, 0)
        << "Sunmmio RSRAM alignment must be a positive byte count";
    const int64_t align_elems =
        static_cast<int64_t>(tl::GetSunmmioRsramAlignmentElems(
            tile_processor_config.rsram_align_bytes, dtype));
    ICHECK_GT(align_elems, 0)
        << "Sunmmio RSRAM alignment must cover at least one element";

    access->aligned_load_bytes = align_bytes;
    access->aligned_load_elems = align_elems;
    access->requires_aligned_1d_load =
        access->tile_shape[0] < access->aligned_load_elems;
  };

  auto analyze_access = [&](const Buffer &buffer,
                            const ffi::Array<PrimExpr> &indices,
                            TileBlockState *state) -> TileAccessInfo {
    TileAccessInfo access;
    access.buffer = buffer;
    const BufferBinding &binding = LookupBuffer(buffer);

    std::vector<int64_t> memtensor_shape =
        ExtractStaticShape(binding.buffer_type);
    access.partition_indices.reserve(memtensor_shape.size());
    access.tiled_dims.clear();

    arith::Analyzer analyzer;
    std::vector<int> logical_tile_axes(indices.size(), -1);
    std::vector<int64_t> logical_tile_shapes(indices.size(), -1);
    std::vector<PrimExpr> logical_partition_indices(indices.size());
    for (int dim = 0; dim < static_cast<int>(indices.size()); ++dim) {
      MarkVisitedExprRoot(indices[dim]);
      for (int axis = 0; axis < static_cast<int>(scope.tile_shape.size());
           ++axis) {
        const ForNode *exec_loop = scope.execution_loops[axis];

        std::vector<const ForNode *> candidate_interior_loops;
        auto push_candidate = [&](const ForNode *loop) {
          if (loop == nullptr) {
            return;
          }
          if (std::find(candidate_interior_loops.begin(),
                        candidate_interior_loops.end(),
                        loop) == candidate_interior_loops.end()) {
            candidate_interior_loops.push_back(loop);
          }
        };
        const ForNode *primary_loop =
            axis == 0 ? state->interior_axis0_loop : state->interior_axis1_loop;
        bool primary_owns_axis = false;
        if (primary_loop != nullptr) {
          auto extent = GetStaticLoopExtent(primary_loop);
          // Tile-loop fusion can place non-reduction regions with different
          // logical tile extents under one execution shell.  Reduction helper
          // loops also reuse tile.interior_axis, but retain the original
          // shell-shape matching rules.
          if (!extent || *extent == scope.tile_shape[axis] ||
              exec_loop == nullptr || !scope.is_reduce_scope) {
            push_candidate(primary_loop);
            primary_owns_axis = true;
          }
        }
        if (!primary_owns_axis) {
          for (const ForNode *loop :
               {state->interior_axis0_loop, state->interior_axis1_loop}) {
            auto extent = GetStaticLoopExtent(loop);
            if (extent && *extent == scope.tile_shape[axis]) {
              push_candidate(loop);
            }
          }
        }

        for (const ForNode *interior_loop : candidate_interior_loops) {
          std::optional<TiledIndexMatch> match;
          std::optional<int64_t> interior_extent =
              GetStaticLoopExtent(interior_loop);
          int64_t matched_tile_extent =
              interior_extent.value_or(scope.tile_shape[axis]);
          if (exec_loop != nullptr) {
            match =
                MatchTiledIndex(indices[dim], exec_loop->loop_var,
                                interior_loop->loop_var, matched_tile_extent,
                                /*allow_standalone_interior=*/true, &analyzer);
          } else if (indices[dim].same_as(interior_loop->loop_var)) {
            match = TiledIndexMatch{0, false,
                                    PrimExpr(IntImm(indices[dim].dtype(), 0))};
          }
          if (match) {
            // MatchTiledIndex consumes the original affine index tree directly;
            // the simplified partition index no longer contains every input
            // node.
            MarkVisitedExprTree(indices[dim]);
            logical_tile_axes[dim] = axis;
            logical_tile_shapes[dim] = matched_tile_extent;
            logical_partition_indices[dim] = match->partition_index;
            break;
          }
        }
        if (logical_tile_axes[dim] >= 0) {
          break;
        }
      }
    }

    for (int dim = 0; dim < static_cast<int>(memtensor_shape.size()); ++dim) {
      if (dim < static_cast<int>(logical_tile_axes.size()) &&
          logical_tile_axes[dim] >= 0) {
        int axis = logical_tile_axes[dim];
        access.tiled_dims.push_back(dim);
        access.tile_shape.push_back(logical_tile_shapes[dim] > 0
                                        ? logical_tile_shapes[dim]
                                        : scope.tile_shape[axis]);
        access.tile_axes.push_back(axis);
        access.partition_indices.push_back(
            EvalExpr(logical_partition_indices[dim]));
      } else {
        if (dim < static_cast<int>(indices.size())) {
          access.partition_indices.push_back(EvalExpr(indices[dim]));
        } else {
          access.partition_indices.push_back(builder_->ConstantInt(
              NewValueName(), 0,
              SunMMIOType{SunMMIOType::Kind::kScalar, DataType::Int(32), 1, {}},
              DataType::Int(32)));
        }
      }
    }

    auto force_it = state->local_unit_tile_axes.find(buffer.get());
    if (force_it != state->local_unit_tile_axes.end() &&
        access.tile_shape.size() == 1 && access.tile_axes.size() == 1) {
      // Final in-tile reduce writeback consumes a local 2D unit tile, e.g.
      // !suvm.tile<1x32>.  The TIR target access is often rank-1 after the
      // reduced dimension disappears, so promote the target view to a matching
      // unit tile_view and keep load/compute/store entirely in 2D tile form.
      int unit_axis = static_cast<int>(force_it->second);
      int existing_axis = access.tile_axes[0];
      if (unit_axis != existing_axis) {
        auto is_tiled_dim = [&](int64_t dim) {
          return std::find(access.tiled_dims.begin(), access.tiled_dims.end(),
                           dim) != access.tiled_dims.end();
        };
        std::optional<int64_t> unit_dim;
        for (int64_t dim = 0;
             dim < static_cast<int64_t>(memtensor_shape.size()); ++dim) {
          if (!is_tiled_dim(dim)) {
            unit_dim = dim;
            break;
          }
        }
        if (unit_dim.has_value()) {
          int64_t existing_dim = access.tiled_dims[0];
          // Keep tile_view dimensions in memtensor/layout order.  For reduce
          // axis 1 this intentionally creates a row-major 1xN view instead of
          // a Nx1 view with tiled_dims=[data_dim, unit_dim]; RHS unit-vector
          // tiles are re-oriented later with squeeze/unsqueeze when needed.
          if (*unit_dim < existing_dim) {
            access.tiled_dims.insert(access.tiled_dims.begin(), *unit_dim);
            access.tile_axes.insert(access.tile_axes.begin(), unit_axis);
            access.tile_shape.insert(access.tile_shape.begin(), 1);
          } else {
            access.tiled_dims.push_back(*unit_dim);
            access.tile_axes.push_back(unit_axis);
            access.tile_shape.push_back(1);
          }
          access.promoted_unit_tile_view = true;
        }
      }
    }

    access.tile_rank = static_cast<int>(access.tile_shape.size());
    ICHECK(access.tile_rank == 1 || access.tile_rank == 2)
        << "Clean v4 tiles lowering currently only supports 1D or 2D tile "
           "accesses inside T.Tiles";
    if (access.tile_rank == 1) {
      ICHECK_EQ(access.tile_axes.size(), 1U);
      access.unsqueeze_axis = access.tile_axes[0] == 0 ? 1 : 0;
      if (IsRsramScope(binding.buffer_type.memory_scope)) {
        populate_aligned_1d_access(&access);
      }
    }
    return access;
  };

  auto cached_value_matches_access = [&](const SunMMIOValue &value,
                                         const TileAccessInfo &access) {
    if (!IsTileLike(value)) {
      return false;
    }
    if (access.requires_aligned_1d_load) {
      return ExtractStaticShape(value.type) ==
             std::vector<int64_t>{access.aligned_load_elems};
    }
    return ExtractStaticShape(value.type) == access.tile_shape;
  };

  auto append_value_key = [](std::ostringstream &os,
                             const SunMMIOValue &value) {
    os << value.value << ":" << static_cast<int>(value.type.kind) << ":"
       << static_cast<int>(value.dtype.code()) << ":"
       << static_cast<int>(value.dtype.bits()) << ":"
       << static_cast<int>(value.dtype.lanes());
  };

  auto make_tile_cache_key =
      [&](const TileAccessInfo &access,
          const std::optional<Aligned1DAddressInfo> &aligned_address =
              std::nullopt) {
        std::ostringstream os;
        os << access.buffer.get();
        os << "|shape=";
        const std::vector<int64_t> shape =
            aligned_address.has_value()
                ? std::vector<int64_t>{access.aligned_load_elems}
                : access.tile_shape;
        for (int64_t dim : shape) {
          os << dim << ",";
        }
        os << "|dims=";
        for (int64_t dim : access.tiled_dims) {
          os << dim << ",";
        }
        os << "|idx=";
        const std::vector<SunMMIOValue> &indices =
            aligned_address.has_value() ? aligned_address->partition_indices
                                        : access.partition_indices;
        for (const SunMMIOValue &index : indices) {
          append_value_key(os, index);
          os << ",";
        }
        return os.str();
      };

  auto make_current_value_name = [&](const Buffer &buffer,
                                     const std::string &cache_key) {
    std::ostringstream os;
    os << "__tile_current_" << buffer->name << "_"
       << std::hash<std::string>{}(cache_key);
    return os.str();
  };

  auto erase_current_values_for_buffer = [&](TileBlockState *state,
                                             const BufferNode *buffer) {
    std::string prefix;
    {
      std::ostringstream os;
      os << buffer << "|";
      prefix = os.str();
    }
    for (auto it = state->current_tile_values.begin();
         it != state->current_tile_values.end();) {
      if (it->first.rfind(prefix, 0) == 0) {
        it = state->current_tile_values.erase(it);
      } else {
        ++it;
      }
    }
  };

  auto get_or_create_tile_view = [&](const TileAccessInfo &access,
                                     TileBlockState *state) -> SunMMIOValue {
    bool bypass_cache = access.promoted_unit_tile_view;
    std::string cache_key = make_tile_cache_key(access);
    if (!bypass_cache) {
      auto it = state->tile_view_cache.find(cache_key);
      if (it != state->tile_view_cache.end()) {
        if (ExtractStaticShape(it->second.type) == access.tile_shape) {
          return it->second;
        }
      }
    }
    const BufferBinding &binding = LookupBuffer(access.buffer);
    SunMMIOValue memtensor{
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1),
        binding.handle, binding.buffer_type};
    SunMMIOType view_type =
        MakeTileViewType(access.buffer->dtype, access.tile_shape);
    SunMMIOValue view = builder_->GetPartitionedTileView(
        NewValueName(), memtensor, access.partition_indices, access.tiled_dims,
        view_type, CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));
    if (!bypass_cache) {
      state->tile_view_cache.emplace(cache_key, view);
    }
    return view;
  };

  auto make_tile_view_from_region = [&](const BufferRegion &region,
                                        TileBlockState *state) -> SunMMIOValue {
    (void)state;
    const Buffer &buffer = region->buffer;
    const BufferBinding &binding = LookupBuffer(buffer);
    SunMMIOValue memtensor{CanonicalizeSuvmDType(buffer->dtype).with_lanes(1),
                           binding.handle, binding.buffer_type};

    std::vector<SunMMIOValue> indices;
    indices.reserve(region->region.size());
    std::vector<int64_t> tiled_dims;
    std::vector<int64_t> tile_shape;
    for (int64_t dim = 0; dim < static_cast<int64_t>(region->region.size());
         ++dim) {
      const Range &range = region->region[dim];
      indices.push_back(EnsureIndex(EvalExpr(range->min)));
      const auto *extent_imm = range->extent.as<IntImmNode>();
      ICHECK(extent_imm) << "Tile region extent must be IntImm";
      MarkVisitedExprRoot(range->extent);
      if (extent_imm->value != 1) {
        tiled_dims.push_back(dim);
        tile_shape.push_back(static_cast<int64_t>(extent_imm->value));
      }
    }
    if (tile_shape.empty()) {
      ICHECK(!region->region.empty())
          << "Tile region lowering expects at least one region dimension";
      // A fully scalar region is still represented as a one-element tile so it
      // can be the destination of tile.reduce from a 1D source tile.
      tiled_dims.push_back(0);
      tile_shape.push_back(1);
    }
    ICHECK(tile_shape.size() == 1 || tile_shape.size() == 2)
        << "Tile region lowering expects one or two non-unit extents";

    SunMMIOType view_type = MakeTileViewType(buffer->dtype, tile_shape);
    return builder_->GetPartitionedTileView(
        NewValueName(), memtensor, indices, tiled_dims, view_type,
        CanonicalizeSuvmDType(buffer->dtype).with_lanes(1));
  };

  auto make_tile_type_from_region = [&](const BufferRegion &region) {
    const Buffer &buffer = region->buffer;
    std::vector<int64_t> tile_shape;
    tile_shape.reserve(region->region.size());
    for (const Range &range : region->region) {
      const auto *extent_imm = range->extent.as<IntImmNode>();
      ICHECK(extent_imm) << "Register tile region extent must be IntImm";
      MarkVisitedExprRoot(range->extent);
      if (extent_imm->value != 1) {
        tile_shape.push_back(static_cast<int64_t>(extent_imm->value));
      }
    }
    if (tile_shape.empty()) {
      tile_shape.push_back(1);
    }
    ICHECK(tile_shape.size() == 1 || tile_shape.size() == 2)
        << "Register tile region expects one or two non-unit extents";
    return MakeTileType(buffer->dtype, tile_shape);
  };

  auto make_reduce_register_tile_type_from_buffer = [&](const Buffer &buffer) {
    std::vector<int64_t> tile_shape;
    tile_shape.reserve(buffer->shape.size());
    std::optional<int64_t> single_non_unit_dim;
    for (int64_t dim = 0; dim < static_cast<int64_t>(buffer->shape.size());
         ++dim) {
      const PrimExpr &extent = buffer->shape[static_cast<size_t>(dim)];
      const auto *extent_imm = extent.as<IntImmNode>();
      ICHECK(extent_imm) << "Reduce register temp shape must be static";
      if (extent_imm->value != 1) {
        tile_shape.push_back(static_cast<int64_t>(extent_imm->value));
        single_non_unit_dim = single_non_unit_dim.has_value()
                                  ? std::optional<int64_t>()
                                  : std::optional<int64_t>(dim);
      }
    }
    if (tile_shape.empty()) {
      tile_shape.push_back(1);
    }
    ICHECK(tile_shape.size() == 1 || tile_shape.size() == 2)
        << "Reduce register temp expects one or two non-unit extents";
    SunMMIOType tile_type = MakeTileType(buffer->dtype, tile_shape);
    return std::make_pair(tile_type, single_non_unit_dim);
  };

  auto make_register_value_name = [&](const Buffer &buffer) {
    return "__tile_reg_" + buffer->name;
  };

  auto make_register_tile_value = [&](const Buffer &buffer,
                                      const SunMMIOType &type) {
    return SunMMIOValue{type.dtype, make_register_value_name(buffer), type};
  };

  auto make_local_value_name = [&](const Buffer &buffer) {
    return "__tile_local_" + buffer->name;
  };

  auto note_register_unsqueeze_axis = [&](TileBlockState *state,
                                          const Buffer &buffer, int64_t axis) {
    if (!IsSunmmioReduceRegisterTempBuffer(buffer)) {
      return;
    }
    auto existing = state->register_unsqueeze_axes.find(buffer.get());
    if (existing != state->register_unsqueeze_axes.end() &&
        existing->second != axis) {
      LOG(FATAL) << "Reduce register temp " << buffer->name
                 << " is reused with incompatible unit axes: existing axis "
                 << existing->second << ", new axis " << axis;
    }
    // vector_core_in_tile_reduce squeezes the reduced axis when its destination
    // is a 1D register tile. Later BufferLoad users must insert the inverse
    // unsqueeze on the same axis to recover the expected 2D tile shape.
    state->register_unsqueeze_axes[buffer.get()] = axis;
  };

  auto collect_tile_live_out_values = [&](TileBlockState *state) {
    std::vector<SunMMIOValue> live_out_values;
    std::set<std::string> seen;
    auto add_value = [&](const SunMMIOValue &value) {
      if (!IsTileLike(value) || value.value.empty()) {
        return;
      }
      if (state->mlir_ctx == nullptr ||
          !state->mlir_ctx->LookupMLIRValue(value.value)) {
        return;
      }
      if (seen.insert(value.value).second) {
        live_out_values.push_back(value);
      }
    };
    for (const auto &kv : state->register_tile_values) {
      add_value(kv.second);
    }
    for (const auto &kv : state->local_tile_values) {
      add_value(kv.second);
    }
    for (const auto &kv : state->current_tile_values) {
      add_value(kv.second);
    }
    std::sort(live_out_values.begin(), live_out_values.end(),
              [](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
                return lhs.value < rhs.value;
              });
    return live_out_values;
  };

  auto discover_reduce_register_temps = [&](const Stmt &stmt,
                                            TileBlockState *state) {
    std::function<void(const Stmt &)> visit_stmt;
    std::function<void(const PrimExpr &)> visit_expr;
    auto register_buffer = [&](const Buffer &buffer,
                               const ffi::Array<PrimExpr> &indices) {
      if (!IsSunmmioReduceLoopCarriedTempBuffer(buffer) ||
          state->register_tile_types.count(buffer.get())) {
        return;
      }
      (void)indices;
      auto [tile_type, single_non_unit_dim] =
          make_reduce_register_tile_type_from_buffer(buffer);
      state->register_tile_types[buffer.get()] = tile_type;
      state->register_tile_values[buffer.get()] =
          make_register_tile_value(buffer, tile_type);
      if (ExtractStaticShape(tile_type).size() == 1 &&
          single_non_unit_dim.has_value() &&
          !state->register_unsqueeze_axes.count(buffer.get())) {
        state->register_unsqueeze_axes[buffer.get()] =
            single_non_unit_dim.value() == 0 ? 1 : 0;
      }
    };

    visit_expr = [&](const PrimExpr &expr) {
      if (!expr.defined()) {
        return;
      }
      if (const auto *load = expr.as<BufferLoadNode>()) {
        register_buffer(load->buffer, load->indices);
        return;
      }
      if (const auto *call = expr.as<CallNode>()) {
        const auto *op_node = call->op.as<OpNode>();
        if (op_node && op_node->name == "tl.vector_core_in_tile_reduce" &&
            call->args.size() >= 3) {
          BufferRegion dst_region = tl::NormalizeToBufferRegion(call->args[1]);
          BufferRegion src_region = tl::NormalizeToBufferRegion(call->args[2]);
          if (call->args.size() >= 4) {
            const auto *axis_imm = call->args[3].as<IntImmNode>();
            ICHECK(axis_imm)
                << "tl.vector_core_in_tile_reduce axis must be IntImm";
            note_register_unsqueeze_axis(state, dst_region->buffer,
                                         static_cast<int64_t>(axis_imm->value));
          }
          if (IsSunmmioReduceLoopCarriedTempBuffer(src_region->buffer) &&
              !state->register_tile_types.count(src_region->buffer.get())) {
            SunMMIOType src_type = make_tile_type_from_region(src_region);
            state->register_tile_types[src_region->buffer.get()] = src_type;
            state->register_tile_values[src_region->buffer.get()] =
                make_register_tile_value(src_region->buffer, src_type);
          }
          if (IsSunmmioReduceLoopCarriedTempBuffer(dst_region->buffer) &&
              !state->register_tile_types.count(dst_region->buffer.get())) {
            SunMMIOType dst_type = make_tile_type_from_region(dst_region);
            state->register_tile_types[dst_region->buffer.get()] = dst_type;
            state->register_tile_values[dst_region->buffer.get()] =
                make_register_tile_value(dst_region->buffer, dst_type);
          }
          return;
        }
      }
      tir::PostOrderVisit(expr, [&](const ObjectRef &obj) {
        if (const auto *load = obj.as<BufferLoadNode>()) {
          register_buffer(load->buffer, load->indices);
        }
      });
    };

    visit_stmt = [&](const Stmt &s) {
      if (!s.defined()) {
        return;
      }
      if (const auto *seq = s.as<SeqStmtNode>()) {
        for (const Stmt &child : seq->seq) {
          visit_stmt(child);
        }
        return;
      }
      if (const auto *ifs = s.as<IfThenElseNode>()) {
        visit_expr(ifs->condition);
        visit_stmt(ifs->then_case);
        if (ifs->else_case.defined()) {
          visit_stmt(ifs->else_case.value());
        }
        return;
      }
      if (const auto *loop = s.as<ForNode>()) {
        visit_expr(loop->min);
        visit_expr(loop->extent);
        visit_stmt(loop->body);
        return;
      }
      if (const auto *let = s.as<LetStmtNode>()) {
        visit_expr(let->value);
        visit_stmt(let->body);
        return;
      }
      if (const auto *alloc = s.as<AllocateNode>()) {
        visit_expr(alloc->condition);
        visit_stmt(alloc->body);
        return;
      }
      if (const auto *decl = s.as<DeclBufferNode>()) {
        visit_stmt(decl->body);
        return;
      }
      if (const auto *store = s.as<BufferStoreNode>()) {
        register_buffer(store->buffer, store->indices);
        visit_expr(store->value);
        return;
      }
      if (const auto *eval = s.as<EvaluateNode>()) {
        visit_expr(eval->value);
      }
    };

    visit_stmt(stmt);
  };

  auto initialize_reduce_register_temps = [&](TileBlockState *state) {
    for (const auto &kv : state->register_tile_types) {
      const BufferNode *buffer_node = kv.first;
      const SunMMIOType &tile_type = kv.second;
      SunMMIOType scalar_type{
          SunMMIOType::Kind::kScalar, tile_type.dtype, 1, {}};
      SunMMIOValue zero =
          tile_type.dtype.is_float() || tile_type.dtype.is_bfloat16()
              ? builder_->ConstantFloat(NewValueName(), "0.0", scalar_type,
                                        tile_type.dtype)
              : builder_->ConstantInt(NewValueName(), 0, scalar_type,
                                      tile_type.dtype);
      SunMMIOValue filled =
          builder_->TileFill(NewValueName(), zero, tile_type, tile_type.dtype);
      state->register_tile_values[buffer_node] = builder_->BindValueAlias(
          make_register_value_name(ffi::GetRef<Buffer>(buffer_node)), filled);
    }
  };

  std::function<SunMMIOValue(const PrimExpr &, TileBlockState *,
                             std::optional<DataType>)>
      lower_expr;
  std::function<void(const Stmt &, TileBlockState *)> lower_stmt;
  std::function<void(const Stmt &, TileBlockState *)> lower_reduce_stmt;
  std::function<void(const CallNode *, TileBlockState *)>
      lower_vector_core_in_tile_reduce;

  auto find_local_unit_axis_in_expr =
      [&](const PrimExpr &expr,
          TileBlockState *state) -> std::optional<int64_t> {
    std::optional<int64_t> axis;
    if (!expr.defined()) {
      return axis;
    }
    tir::PostOrderVisit(expr, [&](const ObjectRef &obj) {
      if (axis.has_value()) {
        return;
      }
      const auto *load = obj.as<BufferLoadNode>();
      if (!load) {
        return;
      }
      if (!state->local_tile_values.count(load->buffer.get())) {
        return;
      }
      auto axis_it = state->local_unit_tile_axes.find(load->buffer.get());
      if (axis_it != state->local_unit_tile_axes.end()) {
        axis = axis_it->second;
      }
    });
    return axis;
  };

  auto choose_result_dtype = [&](DataType expr_dtype,
                                 std::optional<DataType> preferred_dtype) {
    DataType dtype = preferred_dtype.value_or(expr_dtype);
    return CanonicalizeSuvmDType(dtype).with_lanes(1);
  };

  auto canonical_integer_preferred_dtype =
      [](std::optional<DataType> dtype) -> std::optional<DataType> {
    if (!dtype.has_value()) {
      return std::nullopt;
    }
    DataType canonical = CanonicalizeSuvmDType(dtype.value()).with_lanes(1);
    if (!canonical.is_int()) {
      return std::nullopt;
    }
    return canonical;
  };

  auto is_float_like_dtype = [](DataType dtype) {
    return dtype.is_float() || dtype.is_bfloat16();
  };

  auto mask_index_dtype_for_value_dtype = [](DataType value_dtype) {
    DataType dtype = CanonicalizeSuvmDType(value_dtype).with_lanes(1);
    if (dtype.is_bfloat16() || (dtype.is_int() && dtype.bits() == 16)) {
      return DataType::Int(16);
    }
    if ((dtype.is_float() && dtype.bits() == 32) ||
        (dtype.is_int() && dtype.bits() == 32)) {
      return DataType::Int(32);
    }
    LOG(FATAL) << "Unsupported masked tile dtype for SunMMIO mask index "
                  "lowering: "
               << dtype;
    TVM_FFI_UNREACHABLE();
    return DataType::Int(32);
  };

  auto arithmetic_flavor_for_dtype = [](DataType dtype) {
    if (dtype.is_float() || dtype.is_bfloat16()) {
      return ArithmeticFlavor::kFloat;
    }
    if (dtype.is_uint()) {
      return ArithmeticFlavor::kUnsignedInt;
    }
    if (dtype.is_bool()) {
      return ArithmeticFlavor::kBool;
    }
    return ArithmeticFlavor::kSignedInt;
  };

  auto is_tile_compare_operand = [](const SunMMIOValue &lhs,
                                    const SunMMIOValue &rhs) {
    return IsTileLike(lhs) || IsTileLike(rhs);
  };

  auto supports_mixed_precision_binary = [](BinaryOp op) {
    // The SUVM dialect currently allows mixed element types for tile.mulf.
    // Other tile float binary ops carry AllElementTypesMatch and must be kept
    // type-homogeneous before hitting the verifier.
    return op == BinaryOp::kMul;
  };

  auto supports_mixed_precision_unary = [](TileUnaryOp op) {
    // These vfpwln-family ops may choose an output precision independently of
    // the input precision.  abs/neg and f32-only rounding ops do not.
    return op == TileUnaryOp::kExp || op == TileUnaryOp::kLn ||
           op == TileUnaryOp::kRecip || op == TileUnaryOp::kRsqrt;
  };

  auto cast_value_to_dtype = [&](const SunMMIOValue &value, DataType dtype) {
    DataType dst_dtype = CanonicalizeSuvmDType(dtype).with_lanes(1);
    if (value.dtype == dst_dtype) {
      return value;
    }
    if (IsTileLike(value)) {
      SunMMIOType dst_type =
          MakeTileType(dst_dtype, ExtractStaticShape(value.type));
      return builder_->Cast(NewValueName(), value, dst_type, dst_dtype);
    }
    SunMMIOType dst_type{SunMMIOType::Kind::kScalar, dst_dtype, 1, {}};
    return builder_->Cast(NewValueName(), value, dst_type, dst_dtype);
  };

  auto unit_axis_for_2d_shape =
      [](const std::vector<int64_t> &shape) -> std::optional<int64_t> {
    if (shape.size() != 2) {
      return std::nullopt;
    }
    if (shape[0] == 1) {
      return 0;
    }
    if (shape[1] == 1) {
      return 1;
    }
    return std::nullopt;
  };

  auto unit_vector_extent = [](const std::vector<int64_t> &shape,
                               int64_t unit_axis) -> int64_t {
    return unit_axis == 0 ? shape[1] : shape[0];
  };

  auto shape_to_string = [](const std::vector<int64_t> &shape) {
    std::ostringstream os;
    os << "<";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i != 0) {
        os << "x";
      }
      os << shape[i];
    }
    os << ">";
    return os.str();
  };

  auto checked_tile_unsqueeze = [&](const SunMMIOValue &value,
                                    const SunMMIOType &dst_type, int64_t axis,
                                    DataType dtype, const char *context) {
    ICHECK(IsTileLike(value)) << context << " expects a tile input";
    std::vector<int64_t> src_shape = ExtractStaticShape(value.type);
    std::vector<int64_t> dst_shape = ExtractStaticShape(dst_type);
    ICHECK_GE(axis, 0) << context << " has negative unsqueeze axis";
    ICHECK_LE(axis, static_cast<int64_t>(src_shape.size()))
        << context << " unsqueeze axis " << axis << " is out of range for "
        << shape_to_string(src_shape);
    ICHECK_EQ(dst_shape.size(), src_shape.size() + 1)
        << context << " unsqueeze rank mismatch: src "
        << shape_to_string(src_shape) << ", dst " << shape_to_string(dst_shape);
    std::vector<int64_t> expected = src_shape;
    expected.insert(expected.begin() + axis, 1);
    ICHECK(expected == dst_shape)
        << context << " invalid tile.unsqueeze axis " << axis << ": src "
        << shape_to_string(src_shape) << ", expected dst "
        << shape_to_string(expected) << ", actual dst "
        << shape_to_string(dst_shape);
    return builder_->TileUnsqueeze(NewValueName(), value, dst_type, axis,
                                   dtype);
  };

  auto checked_tile_squeeze = [&](const SunMMIOValue &value,
                                  const SunMMIOType &dst_type, int64_t axis,
                                  DataType dtype, const char *context) {
    ICHECK(IsTileLike(value)) << context << " expects a tile input";
    std::vector<int64_t> src_shape = ExtractStaticShape(value.type);
    std::vector<int64_t> dst_shape = ExtractStaticShape(dst_type);
    ICHECK_GE(axis, 0) << context << " has negative squeeze axis";
    ICHECK_LT(axis, static_cast<int64_t>(src_shape.size()))
        << context << " squeeze axis " << axis << " is out of range for "
        << shape_to_string(src_shape);
    ICHECK_EQ(src_shape[static_cast<size_t>(axis)], 1)
        << context << " cannot squeeze non-unit axis " << axis << " from "
        << shape_to_string(src_shape);
    ICHECK_EQ(src_shape.size(), dst_shape.size() + 1)
        << context << " squeeze rank mismatch: src "
        << shape_to_string(src_shape) << ", dst " << shape_to_string(dst_shape);
    std::vector<int64_t> expected = src_shape;
    expected.erase(expected.begin() + axis);
    ICHECK(expected == dst_shape)
        << context << " invalid tile.squeeze axis " << axis << ": src "
        << shape_to_string(src_shape) << ", expected dst "
        << shape_to_string(expected) << ", actual dst "
        << shape_to_string(dst_shape);
    return builder_->TileSqueeze(NewValueName(), value, dst_type, axis, dtype);
  };

  auto reorient_unit_tile_to_shape =
      [&](const SunMMIOValue &value,
          const std::vector<int64_t> &dst_shape) -> SunMMIOValue {
    if (!IsTileLike(value)) {
      return value;
    }
    std::vector<int64_t> src_shape = ExtractStaticShape(value.type);
    if (src_shape == dst_shape) {
      return value;
    }

    if (src_shape.size() == 1 && dst_shape.size() == 2) {
      std::optional<int64_t> dst_unit_axis = unit_axis_for_2d_shape(dst_shape);
      if (dst_unit_axis.has_value() &&
          src_shape[0] == unit_vector_extent(dst_shape, *dst_unit_axis)) {
        SunMMIOType dst_type = MakeTileType(value.dtype, dst_shape);
        return checked_tile_unsqueeze(value, dst_type, *dst_unit_axis,
                                      value.dtype,
                                      "reorient rank-1 tile to unit tile");
      }
      if (src_shape[0] == dst_shape[0]) {
        SunMMIOType dst_type = MakeTileType(value.dtype, {dst_shape[0], 1});
        return checked_tile_unsqueeze(value, dst_type, 1, value.dtype,
                                      "reorient rank-1 tile to column tile");
      }
      if (src_shape[0] == dst_shape[1]) {
        SunMMIOType dst_type = MakeTileType(value.dtype, {1, dst_shape[1]});
        return checked_tile_unsqueeze(value, dst_type, 0, value.dtype,
                                      "reorient rank-1 tile to row tile");
      }
      return value;
    }

    if (src_shape.size() == 2 && dst_shape.size() == 1) {
      std::optional<int64_t> src_unit_axis = unit_axis_for_2d_shape(src_shape);
      if (src_unit_axis.has_value() &&
          unit_vector_extent(src_shape, *src_unit_axis) == dst_shape[0]) {
        SunMMIOType dst_type = MakeTileType(value.dtype, dst_shape);
        return checked_tile_squeeze(value, dst_type, *src_unit_axis,
                                    value.dtype,
                                    "reorient unit tile to rank-1 tile");
      }
      return value;
    }

    if (src_shape.size() == 2 && dst_shape.size() == 2) {
      std::optional<int64_t> src_unit_axis = unit_axis_for_2d_shape(src_shape);
      std::optional<int64_t> dst_unit_axis = unit_axis_for_2d_shape(dst_shape);
      // A 2D unit tile already carries row/column orientation.  Do not
      // transpose it just because its non-unit extent matches another target
      // dimension; broadcasting can expand it to a full 2D tile while
      // preserving the source orientation.
      if (src_unit_axis.has_value() && !dst_unit_axis.has_value()) {
        return value;
      }
      if (src_unit_axis.has_value() && dst_unit_axis.has_value() &&
          unit_vector_extent(src_shape, *src_unit_axis) ==
              unit_vector_extent(dst_shape, *dst_unit_axis)) {
        std::vector<int64_t> squeezed_shape{
            unit_vector_extent(src_shape, *src_unit_axis)};
        SunMMIOType squeezed_type = MakeTileType(value.dtype, squeezed_shape);
        SunMMIOValue squeezed = checked_tile_squeeze(
            value, squeezed_type, *src_unit_axis, value.dtype,
            "reorient unit tile through rank-1 tile");
        SunMMIOType dst_type = MakeTileType(value.dtype, dst_shape);
        return checked_tile_unsqueeze(squeezed, dst_type, *dst_unit_axis,
                                      value.dtype,
                                      "reorient rank-1 tile to target unit "
                                      "tile");
      }
    }
    return value;
  };

  auto cast_tile_dtype_preserving_shape = [&](const SunMMIOValue &tile,
                                              DataType dst_dtype) {
    ICHECK(IsTileLike(tile))
        << "cast_tile_dtype_preserving_shape expects a tile value";
    DataType canonical_dst_dtype =
        CanonicalizeSuvmDType(dst_dtype).with_lanes(1);
    if (tile.dtype == canonical_dst_dtype) {
      return tile;
    }
    SunMMIOType dst_type =
        MakeTileType(canonical_dst_dtype, ExtractStaticShape(tile.type));
    return builder_->Cast(NewValueName(), tile, dst_type, canonical_dst_dtype);
  };

  std::function<SunMMIOValue(const SunMMIOValue &,
                             const std::vector<int64_t> &)>
      broadcast_tile_to_shape;
  std::function<SunMMIOValue(const SunMMIOValue &,
                             const std::vector<int64_t> &)>
      orient_tile_operand_to_shape;

  auto normalize_for_store = [&](const TileAccessInfo &access,
                                 const SunMMIOValue &value) -> SunMMIOValue {
    DataType dst_dtype =
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1);
    SunMMIOType dst_tile_type = MakeTileType(dst_dtype, access.tile_shape);
    if (value.type.kind == SunMMIOType::Kind::kTile) {
      SunMMIOValue tile = value;
      tile = reorient_unit_tile_to_shape(tile, access.tile_shape);
      if (ExtractStaticShape(tile.type) != access.tile_shape) {
        tile = broadcast_tile_to_shape(tile, access.tile_shape);
      }
      ICHECK(StaticShapesEqual(tile.type, dst_tile_type))
          << "Tiles store normalization cannot normalize RHS shape";
      return cast_tile_dtype_preserving_shape(tile, dst_dtype);
    }
    ICHECK(IsScalarLike(value))
        << "Tiles store normalization only supports scalar or tile values";
    SunMMIOValue scalar = value;
    if (scalar.type.kind != SunMMIOType::Kind::kScalar ||
        scalar.dtype != dst_dtype) {
      SunMMIOType scalar_type{SunMMIOType::Kind::kScalar, dst_dtype, 1, {}};
      scalar = builder_->Cast(NewValueName(), scalar, scalar_type, dst_dtype);
    }
    return builder_->TileFill(NewValueName(), scalar, dst_tile_type, dst_dtype);
  };

  auto normalize_for_aligned_1d_store = [&](const TileAccessInfo &access,
                                            const SunMMIOValue &value) {
    DataType dst_dtype =
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1);
    std::vector<int64_t> vector_shape = access.tile_shape;

    if (IsTileLike(value)) {
      SunMMIOValue tile = value;
      std::vector<int64_t> shape = ExtractStaticShape(tile.type);
      if (shape != vector_shape) {
        tile = reorient_unit_tile_to_shape(tile, vector_shape);
        shape = ExtractStaticShape(tile.type);
      }
      ICHECK(shape == vector_shape)
          << "Aligned 1D tile store cannot normalize RHS shape: src "
          << shape_to_string(shape) << ", expected "
          << shape_to_string(vector_shape);
      SunMMIOType dst_tile_type =
          MakeTileType(access.buffer->dtype, vector_shape);
      if (tile.dtype == dst_dtype &&
          StaticShapesEqual(tile.type, dst_tile_type)) {
        return tile;
      }
      return builder_->Cast(NewValueName(), tile, dst_tile_type, dst_dtype);
    }

    ICHECK(IsScalarLike(value))
        << "Aligned 1D tile store normalization only supports scalar or tile "
           "values";
    SunMMIOValue scalar = value;
    if (scalar.type.kind != SunMMIOType::Kind::kScalar ||
        scalar.dtype != dst_dtype) {
      SunMMIOType scalar_type{SunMMIOType::Kind::kScalar, dst_dtype, 1, {}};
      scalar = builder_->Cast(NewValueName(), scalar, scalar_type, dst_dtype);
    }
    return builder_->TileFill(NewValueName(), scalar,
                              MakeTileType(access.buffer->dtype, vector_shape),
                              dst_dtype);
  };

  auto make_index_const = [&](int64_t value) {
    return builder_->ConstantInt(
        NewValueName(), value,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
  };

  auto get_const_index_value =
      [&](const SunMMIOValue &value) -> std::optional<int64_t> {
    mlir::Value mlir_value = mlir_ctx->LookupMLIRValue(value.value);
    if (!mlir_value) {
      return std::nullopt;
    }
    if (auto cst = mlir::getConstantIntValue(mlir_value)) {
      return static_cast<int64_t>(*cst);
    }
    return std::nullopt;
  };

  auto add_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
    auto lhs_cst = get_const_index_value(lhs);
    auto rhs_cst = get_const_index_value(rhs);
    if (lhs_cst.has_value() && rhs_cst.has_value()) {
      return make_index_const(*lhs_cst + *rhs_cst);
    }
    return builder_->Binary(
        NewValueName(), BinaryOp::kAdd, ArithmeticFlavor::kIndex, lhs, rhs,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
  };

  auto sub_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
    auto lhs_cst = get_const_index_value(lhs);
    auto rhs_cst = get_const_index_value(rhs);
    if (lhs_cst.has_value() && rhs_cst.has_value()) {
      return make_index_const(*lhs_cst - *rhs_cst);
    }
    return builder_->Binary(
        NewValueName(), BinaryOp::kSub, ArithmeticFlavor::kIndex, lhs, rhs,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
  };

  auto mul_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
    auto lhs_cst = get_const_index_value(lhs);
    auto rhs_cst = get_const_index_value(rhs);
    if (lhs_cst.has_value() && rhs_cst.has_value()) {
      return make_index_const(*lhs_cst * *rhs_cst);
    }
    return builder_->Binary(
        NewValueName(), BinaryOp::kMul, ArithmeticFlavor::kIndex, lhs, rhs,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
  };

  auto min_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
    auto lhs_cst = get_const_index_value(lhs);
    auto rhs_cst = get_const_index_value(rhs);
    if (lhs_cst.has_value() && rhs_cst.has_value()) {
      return make_index_const(std::min(*lhs_cst, *rhs_cst));
    }
    return builder_->Binary(
        NewValueName(), BinaryOp::kMin, ArithmeticFlavor::kIndex, lhs, rhs,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
  };

  auto div_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
    auto lhs_cst = get_const_index_value(lhs);
    auto rhs_cst = get_const_index_value(rhs);
    if (lhs_cst.has_value() && rhs_cst.has_value()) {
      ICHECK_NE(*rhs_cst, 0) << "index division by zero in aligned 1D lowering";
      return make_index_const(*lhs_cst / *rhs_cst);
    }
    return builder_->Binary(
        NewValueName(), BinaryOp::kDiv, ArithmeticFlavor::kIndex, lhs, rhs,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
  };

  auto mod_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
    auto lhs_cst = get_const_index_value(lhs);
    auto rhs_cst = get_const_index_value(rhs);
    if (lhs_cst.has_value() && rhs_cst.has_value()) {
      ICHECK_NE(*rhs_cst, 0) << "index modulo by zero in aligned 1D lowering";
      return make_index_const(*lhs_cst % *rhs_cst);
    }
    return builder_->Binary(
        NewValueName(), BinaryOp::kMod, ArithmeticFlavor::kIndex, lhs, rhs,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
  };

  auto materialize_domain_values = [&](TilesScopeInfo *tile_scope) {
    tile_scope->domain_values.clear();
    tile_scope->domain_values.reserve(tile_scope->domain_shape.size());
    for (const PrimExpr &extent : tile_scope->domain_shape) {
      tile_scope->domain_values.push_back(EnsureIndex(EvalExpr(extent)));
    }
  };

  auto domain_value = [&](int domain_axis) -> const SunMMIOValue & {
    ICHECK_GE(domain_axis, 0);
    ICHECK_LT(static_cast<size_t>(domain_axis), scope.domain_values.size())
        << "Tile domain axis is missing its materialized value";
    return scope.domain_values[static_cast<size_t>(domain_axis)];
  };

  auto ceildiv_index = [&](const SunMMIOValue &value, int64_t divisor) {
    ICHECK_GT(divisor, 0) << "Tile loop extent divisor must be positive";
    SunMMIOValue divisor_value = make_index_const(divisor);
    return div_index(add_index(value, make_index_const(divisor - 1)),
                     divisor_value);
  };

  auto materialized_loop_extent =
      [&](const ForNode *loop) -> std::optional<SunMMIOValue> {
    auto loop_it =
        std::find(scope.domain_loops.begin(), scope.domain_loops.end(), loop);
    if (loop_it == scope.domain_loops.end()) {
      return std::nullopt;
    }
    size_t loop_axis =
        static_cast<size_t>(std::distance(scope.domain_loops.begin(), loop_it));
    ICHECK_LT(loop_axis, scope.domain_values.size());
    auto execution_axis = GetExecutionAxisAnnotation(loop);
    if (!execution_axis.has_value()) {
      // Tiles lowering replaces the complete source extent with the
      // materialized domain value instead of recursively lowering it.
      MarkVisitedExprTree(loop->extent);
      return scope.domain_values[loop_axis];
    }
    ICHECK_GE(execution_axis.value(), 0);
    ICHECK_LT(static_cast<size_t>(execution_axis.value()),
              scope.execution_domain_axes.size());
    ICHECK_LT(static_cast<size_t>(execution_axis.value()),
              scope.tile_shape.size());
    int domain_axis = scope.execution_domain_axes[execution_axis.value()];
    SunMMIOValue extent = ceildiv_index(
        domain_value(domain_axis), scope.tile_shape[execution_axis.value()]);
    // The synthesized ceildiv is the lowering result for the complete source
    // extent, whose child expressions therefore do not enter EvalExpr.
    MarkVisitedExprTree(loop->extent);
    return extent;
  };

  auto build_full_tile_condition = [&]() {
    SunMMIOType bool_ty{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
    std::optional<SunMMIOValue> result;
    for (size_t execution_axis = 0;
         execution_axis < scope.execution_loops.size(); ++execution_axis) {
      const ForNode *loop = scope.execution_loops[execution_axis];
      ICHECK(loop != nullptr)
          << "Full-tile condition is missing an execution loop";
      ICHECK_LT(execution_axis, scope.execution_domain_axes.size());
      ICHECK_LT(execution_axis, scope.tile_shape.size());
      SunMMIOValue exec_index = EnsureIndex(EvalExpr(loop->loop_var));
      SunMMIOValue tile_extent =
          make_index_const(scope.tile_shape[execution_axis]);
      SunMMIOValue tile_end =
          mul_index(add_index(exec_index, make_index_const(1)), tile_extent);
      SunMMIOValue axis_is_full = builder_->Compare(
          NewValueName(), CompareOp::kLE, CompareDomain::kSignedInt, tile_end,
          domain_value(scope.execution_domain_axes[execution_axis]),
          tile_end.type);
      axis_is_full = EnsureType(axis_is_full, bool_ty, DataType::Bool());
      result =
          result.has_value()
              ? std::optional<SunMMIOValue>(builder_->Binary(
                    NewValueName(), BinaryOp::kAnd, ArithmeticFlavor::kBool,
                    result.value(), axis_is_full, bool_ty, DataType::Bool()))
              : std::optional<SunMMIOValue>(axis_is_full);
    }
    ICHECK(result.has_value())
        << "Full-tile condition requires at least one execution axis";
    // The full-tile predicate is rebuilt from domain metadata, replacing the
    // complete source predicate rather than recursively lowering its children.
    MarkVisitedExprTree(scope.tail_predicate);
    return result.value();
  };

  auto compute_aligned_1d_address =
      [&](const TileAccessInfo &access,
          const SunMMIOType &memtensor_type) -> Aligned1DAddressInfo {
    ICHECK(access.requires_aligned_1d_load);
    ICHECK_EQ(access.tiled_dims.size(), 1U)
        << "Aligned 1D access expects exactly one tiled dimension";

    std::vector<int64_t> memtensor_shape = ExtractStaticShape(memtensor_type);
    ICHECK_EQ(access.partition_indices.size(), memtensor_shape.size())
        << "Aligned 1D access expects one partition index per memtensor "
           "dimension";
    int64_t tiled_dim = access.tiled_dims[0];
    ICHECK_LT(tiled_dim, static_cast<int64_t>(memtensor_shape.size()));

    std::vector<int64_t> layout_shape =
        memtensor_type.layout_hshape.empty()
            ? memtensor_shape
            : ExtractStaticPrimExprs(memtensor_type.layout_hshape,
                                     "layout shape");
    std::vector<int64_t> strides =
        memtensor_type.layout_hstride.empty()
            ? std::vector<int64_t>{}
            : ExtractStaticPrimExprs(memtensor_type.layout_hstride,
                                     "layout stride");
    if (strides.empty()) {
      strides.assign(layout_shape.size(), 1);
      for (int dim = static_cast<int>(memtensor_shape.size()) - 2; dim >= 0;
           --dim) {
        strides[static_cast<size_t>(dim)] =
            strides[static_cast<size_t>(dim + 1)] *
            layout_shape[static_cast<size_t>(dim + 1)];
      }
    }
    ICHECK_EQ(strides.size(), layout_shape.size())
        << "Aligned 1D access expects one stride per layout mode";
    bool is_flat_layout = true;
    if (!memtensor_type.layout_dim_levels.empty()) {
      for (uint8_t level : memtensor_type.layout_dim_levels) {
        is_flat_layout = is_flat_layout && level == 1;
      }
    }
    is_flat_layout =
        is_flat_layout && layout_shape.size() == memtensor_shape.size();

    std::vector<size_t> logical_mode_offsets(memtensor_shape.size());
    if (is_flat_layout) {
      for (size_t dim = 0; dim < memtensor_shape.size(); ++dim) {
        logical_mode_offsets[dim] = dim;
      }
    } else {
      ICHECK_EQ(memtensor_type.layout_dim_levels.size(), memtensor_shape.size())
          << "Aligned 1D hierarchical access expects one level count per "
             "logical memtensor dimension";
      size_t mode_offset = 0;
      for (size_t dim = 0; dim < memtensor_shape.size(); ++dim) {
        int levels = memtensor_type.layout_dim_levels[dim];
        ICHECK_GT(levels, 0);
        logical_mode_offsets[dim] = mode_offset;
        mode_offset += static_cast<size_t>(levels);
        ICHECK_LE(mode_offset, layout_shape.size());
      }
      ICHECK_EQ(mode_offset, layout_shape.size());
    }

    // Fast path for row-major padded layouts and row-contiguous hierarchical
    // layouts such as ZZ.  A hierarchical carrier stays inside the innermost
    // tiled mode; every other mode begins at a carrier-aligned address.
    size_t tiled_mode = logical_mode_offsets[static_cast<size_t>(tiled_dim)];
    bool carrier_fits_tiled_mode =
        is_flat_layout ||
        layout_shape[tiled_mode] % access.aligned_load_elems == 0;
    if (strides[tiled_mode] == 1 && carrier_fits_tiled_mode &&
        access.aligned_load_elems % access.tile_shape[0] == 0) {
      bool outer_strides_are_aligned = true;
      for (size_t mode = 0; mode < strides.size(); ++mode) {
        if (mode == tiled_mode) {
          continue;
        }
        outer_strides_are_aligned =
            outer_strides_are_aligned &&
            strides[mode] % access.aligned_load_elems == 0;
      }
      if (outer_strides_are_aligned) {
        SunMMIOValue tile_partition = EnsureIndex(
            access.partition_indices[static_cast<size_t>(tiled_dim)]);
        SunMMIOValue tile_base_elem =
            mul_index(tile_partition, make_index_const(access.tile_shape[0]));
        SunMMIOValue aligned_elems =
            make_index_const(access.aligned_load_elems);
        std::vector<SunMMIOValue> aligned_partition_indices =
            access.partition_indices;
        aligned_partition_indices[static_cast<size_t>(tiled_dim)] =
            div_index(tile_base_elem, aligned_elems);
        SunMMIOValue offset_elems = mod_index(tile_base_elem, aligned_elems);
        return Aligned1DAddressInfo{offset_elems, aligned_partition_indices};
      }
    }

    ICHECK(is_flat_layout)
        << "Aligned 1D hierarchical access requires a carrier that fits in "
           "the stride-1 innermost tiled mode with aligned outer mode strides";

    SunMMIOValue linear_elem = make_index_const(0);
    for (int64_t dim = 0; dim < static_cast<int64_t>(memtensor_shape.size());
         ++dim) {
      SunMMIOValue dim_index =
          EnsureIndex(access.partition_indices[static_cast<size_t>(dim)]);
      if (dim == tiled_dim) {
        dim_index =
            mul_index(dim_index, make_index_const(access.tile_shape[0]));
      }
      SunMMIOValue dim_offset = mul_index(
          dim_index, make_index_const(strides[static_cast<size_t>(dim)]));
      linear_elem = add_index(linear_elem, dim_offset);
    }

    int64_t dtype_bytes = static_cast<int64_t>(
        CanonicalizeSuvmDType(access.buffer->dtype).bytes());
    SunMMIOValue elem_size = make_index_const(dtype_bytes);
    SunMMIOValue aligned_bytes = make_index_const(access.aligned_load_bytes);
    SunMMIOValue aligned_elems = make_index_const(access.aligned_load_elems);
    SunMMIOValue base_bytes = mul_index(linear_elem, elem_size);
    SunMMIOValue region_index = div_index(base_bytes, aligned_bytes);
    SunMMIOValue offset_bytes = mod_index(base_bytes, aligned_bytes);
    SunMMIOValue offset_elems = div_index(offset_bytes, elem_size);

    std::vector<SunMMIOValue> aligned_partition_indices(memtensor_shape.size(),
                                                        make_index_const(0));
    SunMMIOValue remaining = mul_index(region_index, aligned_elems);
    for (int64_t dim = 0; dim < static_cast<int64_t>(memtensor_shape.size());
         ++dim) {
      int64_t stride = strides[static_cast<size_t>(dim)];
      ICHECK_GT(stride, 0) << "Aligned 1D access expects positive strides";
      SunMMIOValue dim_index = div_index(remaining, make_index_const(stride));
      SunMMIOValue dim_contrib = mul_index(dim_index, make_index_const(stride));
      remaining = builder_->Binary(
          NewValueName(), BinaryOp::kSub, ArithmeticFlavor::kIndex, remaining,
          dim_contrib,
          SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
          DataType::Int(32));
      if (dim == tiled_dim) {
        aligned_partition_indices[static_cast<size_t>(dim)] =
            div_index(dim_index, aligned_elems);
      } else {
        aligned_partition_indices[static_cast<size_t>(dim)] = dim_index;
      }
    }
    return Aligned1DAddressInfo{offset_elems, aligned_partition_indices};
  };

  auto load_aligned_1d_tile = [&](const TileAccessInfo &access,
                                  TileBlockState *state) -> SunMMIOValue {
    ICHECK(access.requires_aligned_1d_load);
    ICHECK_EQ(access.tiled_dims.size(), 1U)
        << "Aligned 1D tile load expects exactly one tiled dimension";

    const BufferBinding &binding = LookupBuffer(access.buffer);
    SunMMIOValue memtensor{
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1),
        binding.handle, binding.buffer_type};
    Aligned1DAddressInfo aligned_address =
        compute_aligned_1d_address(access, binding.buffer_type);
    std::string cache_key = make_tile_cache_key(access, aligned_address);

    SunMMIOType aligned_view_type =
        MakeTileViewType(access.buffer->dtype, {access.aligned_load_elems});
    SunMMIOValue aligned_view = builder_->GetPartitionedTileView(
        NewValueName(), memtensor, aligned_address.partition_indices,
        access.tiled_dims, aligned_view_type,
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));
    SunMMIOType aligned_tile_type =
        MakeTileType(access.buffer->dtype, {access.aligned_load_elems});
    SunMMIOValue aligned_tile;
    auto current_it = state->current_tile_values.find(cache_key);
    if (current_it != state->current_tile_values.end() &&
        cached_value_matches_access(current_it->second, access) &&
        state->mlir_ctx != nullptr &&
        state->mlir_ctx->LookupMLIRValue(current_it->second.value)) {
      aligned_tile = current_it->second;
    } else {
      aligned_tile = builder_->TileLoad(
          NewValueName(), aligned_view, aligned_tile_type, std::nullopt,
          std::nullopt,
          CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));
      state->current_tile_values[cache_key] = builder_->BindValueAlias(
          make_current_value_name(access.buffer, cache_key), aligned_tile);
    }

    DataType value_dtype =
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1);
    if (access.tile_shape[0] == 1 && SupportsSuvmTilePickDType(value_dtype)) {
      SunMMIOType scalar_type{SunMMIOType::Kind::kScalar, value_dtype, 1, {}};
      return builder_->TilePick(NewValueName(), aligned_tile,
                                {aligned_address.offset_elems}, scalar_type,
                                value_dtype);
    }

    std::vector<SunMMIOValue> slice_offsets{aligned_address.offset_elems};
    SunMMIOType sliced_tile_type =
        MakeTileType(access.buffer->dtype, access.tile_shape);
    SunMMIOValue sliced_tile = builder_->TileSlice(
        NewValueName(), aligned_tile, slice_offsets, sliced_tile_type,
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));
    return sliced_tile;
  };

  auto store_aligned_1d_tile =
      [&](const TileAccessInfo &access, const SunMMIOValue &value,
          const std::optional<SunMMIOValue> &store_mask,
          TileBlockState *state) -> SunMMIOValue {
    ICHECK(access.requires_aligned_1d_load);
    ICHECK_EQ(access.tiled_dims.size(), 1U)
        << "Aligned 1D tile store expects exactly one tiled dimension";

    const BufferBinding &binding = LookupBuffer(access.buffer);
    SunMMIOValue memtensor{
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1),
        binding.handle, binding.buffer_type};
    Aligned1DAddressInfo aligned_address =
        compute_aligned_1d_address(access, binding.buffer_type);
    std::string cache_key = make_tile_cache_key(access, aligned_address);

    SunMMIOType aligned_view_type =
        MakeTileViewType(access.buffer->dtype, {access.aligned_load_elems});
    SunMMIOValue aligned_view = builder_->GetPartitionedTileView(
        NewValueName(), memtensor, aligned_address.partition_indices,
        access.tiled_dims, aligned_view_type,
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));
    SunMMIOType aligned_tile_type =
        MakeTileType(access.buffer->dtype, {access.aligned_load_elems});
    SunMMIOValue aligned_tile;
    auto current_it = state->current_tile_values.find(cache_key);
    if (current_it != state->current_tile_values.end() &&
        cached_value_matches_access(current_it->second, access) &&
        state->mlir_ctx != nullptr &&
        state->mlir_ctx->LookupMLIRValue(current_it->second.value)) {
      aligned_tile = current_it->second;
    } else {
      aligned_tile = builder_->TileLoad(
          NewValueName(), aligned_view, aligned_tile_type, std::nullopt,
          std::nullopt,
          CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));
    }
    std::vector<int64_t> slice_shape = access.tile_shape;
    SunMMIOType slice_type = MakeTileType(access.buffer->dtype, slice_shape);

    SunMMIOValue src_slice = value;
    if (src_slice.type.shape.size() != 1) {
      src_slice = reorient_unit_tile_to_shape(src_slice, slice_shape);
    }
    ICHECK(StaticShapesEqual(src_slice.type, slice_type))
        << "Aligned 1D store source must be rank-1";

    std::vector<SunMMIOValue> slice_offsets{aligned_address.offset_elems};

    if (store_mask.has_value()) {
      SunMMIOValue mask = store_mask.value();
      if (IsTileLike(mask)) {
        mask = reorient_unit_tile_to_shape(mask, slice_shape);
        if (ExtractStaticShape(mask.type) != slice_shape) {
          mask = broadcast_tile_to_shape(mask, slice_shape);
        }
        ICHECK(StaticShapesEqual(mask.type,
                                 MakeTileType(DataType::Bool(), slice_shape)))
            << "Aligned 1D store predicate cannot normalize mask shape";
      } else {
        SunMMIOType bool_scalar_type{
            SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
        mask = EnsureType(mask, bool_scalar_type, DataType::Bool());
        mask = builder_->TileFill(NewValueName(), mask,
                                  MakeTileType(DataType::Bool(), slice_shape),
                                  DataType::Bool());
      }
      SunMMIOValue old_slice = builder_->TileSlice(
          NewValueName(), aligned_tile, slice_offsets, slice_type,
          CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));
      src_slice = builder_->TileSelect(
          NewValueName(), mask, src_slice, old_slice, slice_type,
          CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));
    }

    SunMMIOValue merged_tile = builder_->TileInsertSlice(
        NewValueName(), aligned_tile, src_slice, slice_offsets,
        aligned_tile_type,
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));
    builder_->TileStore(merged_tile, aligned_view, std::nullopt);
    return builder_->BindValueAlias(
        make_current_value_name(access.buffer, cache_key), merged_tile);
  };

  auto make_tile_access_from_region =
      [&](const BufferRegion &region,
          const std::optional<std::vector<int>> &tile_axes_override =
              std::nullopt) -> TileAccessInfo {
    TileAccessInfo access;
    access.buffer = region->buffer;
    const BufferBinding &binding = LookupBuffer(access.buffer);
    std::vector<int64_t> memtensor_shape =
        ExtractStaticShape(binding.buffer_type);
    access.partition_indices.reserve(memtensor_shape.size());

    size_t tile_axis_index = 0;
    for (int64_t dim = 0; dim < static_cast<int64_t>(memtensor_shape.size());
         ++dim) {
      SunMMIOValue min = dim < static_cast<int64_t>(region->region.size())
                             ? EnsureIndex(EvalExpr(region->region[dim]->min))
                             : make_index_const(0);
      int64_t extent = 1;
      if (dim < static_cast<int64_t>(region->region.size())) {
        const auto *extent_imm = region->region[dim]->extent.as<IntImmNode>();
        ICHECK(extent_imm) << "Tile region extent must be IntImm";
        MarkVisitedExprRoot(region->region[dim]->extent);
        extent = static_cast<int64_t>(extent_imm->value);
      }

      if (extent != 1) {
        access.tiled_dims.push_back(dim);
        access.tile_shape.push_back(extent);
        int axis = tile_axes_override.has_value()
                       ? tile_axes_override->at(tile_axis_index)
                       : static_cast<int>(tile_axis_index);
        access.tile_axes.push_back(axis);
        access.partition_indices.push_back(
            div_index(min, make_index_const(extent)));
        ++tile_axis_index;
      } else {
        access.partition_indices.push_back(min);
      }
    }

    if (access.tile_shape.empty()) {
      ICHECK(!memtensor_shape.empty())
          << "Tile region lowering expects at least one memtensor dimension";
      access.tiled_dims.push_back(0);
      access.tile_shape.push_back(1);
      access.tile_axes.push_back(0);
      if (access.partition_indices.empty()) {
        access.partition_indices.push_back(make_index_const(0));
      }
    }
    if (tile_axes_override.has_value()) {
      ICHECK_EQ(tile_axes_override->size(), access.tile_axes.size())
          << "Tile region axis override must match tile rank";
    }

    access.tile_rank = static_cast<int>(access.tile_shape.size());
    ICHECK(access.tile_rank == 1 || access.tile_rank == 2)
        << "Tile region lowering expects one or two non-unit extents";
    if (access.tile_rank == 1) {
      ICHECK_EQ(access.tile_axes.size(), 1U);
      access.unsqueeze_axis = access.tile_axes[0] == 0 ? 1 : 0;
      if (IsRsramScope(binding.buffer_type.memory_scope)) {
        populate_aligned_1d_access(&access);
      }
    }
    return access;
  };

  auto build_tail_mask_info = [&](TileBlockState *state) -> TailMaskInfo {
    (void)state;
    ICHECK(scope.tail_predicate.defined());

    auto make_index_const = [&](int64_t value) {
      return builder_->ConstantInt(
          NewValueName(), value,
          SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
          DataType::Int(32));
    };

    auto sub_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
      return builder_->Binary(
          NewValueName(), BinaryOp::kSub, ArithmeticFlavor::kIndex, lhs, rhs,
          SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
          DataType::Int(32));
    };

    auto min_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
      return builder_->Binary(
          NewValueName(), BinaryOp::kMin, ArithmeticFlavor::kIndex, lhs, rhs,
          SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
          DataType::Int(32));
    };

    SunMMIOValue exec_i = EvalExpr(scope.execution_loops[0]->loop_var);
    SunMMIOValue exec_j = EvalExpr(scope.execution_loops[1]->loop_var);
    SunMMIOValue tile_m = make_index_const(scope.tile_shape[0]);
    SunMMIOValue tile_n = make_index_const(scope.tile_shape[1]);
    const SunMMIOValue &domain_m = domain_value(scope.execution_domain_axes[0]);
    const SunMMIOValue &domain_n = domain_value(scope.execution_domain_axes[1]);

    SunMMIOValue valid_rows = min_index(
        tile_m,
        sub_index(domain_m,
                  builder_->Binary(
                      NewValueName(), BinaryOp::kMul, ArithmeticFlavor::kIndex,
                      exec_i, tile_m,
                      SunMMIOType{
                          SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
                      DataType::Int(32))));
    SunMMIOValue valid_cols = min_index(
        tile_n,
        sub_index(domain_n,
                  builder_->Binary(
                      NewValueName(), BinaryOp::kMul, ArithmeticFlavor::kIndex,
                      exec_j, tile_n,
                      SunMMIOType{
                          SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
                      DataType::Int(32))));

    SunMMIOType bool_ty{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
    SunMMIOValue row_tail_cond = builder_->Compare(
        NewValueName(), CompareOp::kLT, CompareDomain::kSignedInt, valid_rows,
        tile_m, valid_rows.type);
    row_tail_cond = EnsureType(row_tail_cond, bool_ty, DataType::Bool());
    SunMMIOValue col_tail_cond = builder_->Compare(
        NewValueName(), CompareOp::kLT, CompareDomain::kSignedInt, valid_cols,
        tile_n, valid_cols.type);
    col_tail_cond = EnsureType(col_tail_cond, bool_ty, DataType::Bool());

    SunMMIOType mask_type;
    mask_type.kind = SunMMIOType::Kind::kTile;
    mask_type.dtype = DataType::Bool();
    mask_type.lanes = 1;
    for (int64_t dim : scope.tile_shape) {
      mask_type.shape.push_back(IntImm(DataType::Int(32), dim));
    }
    return TailMaskInfo{valid_rows, valid_cols, row_tail_cond, col_tail_cond,
                        mask_type};
  };

  auto can_prove_expr_equal = [](const PrimExpr &lhs, const PrimExpr &rhs) {
    arith::Analyzer analyzer;
    PrimExpr lhs_simpl = analyzer.Simplify(lhs);
    PrimExpr rhs_simpl = analyzer.Simplify(rhs);
    if (StructuralEqual()(lhs_simpl, rhs_simpl)) {
      return true;
    }
    return analyzer.CanProve(lhs_simpl == rhs_simpl);
  };

  auto cast_prim_expr = [](const PrimExpr &expr, DataType dtype) -> PrimExpr {
    if (expr.dtype() == dtype) {
      return expr;
    }
    return Cast(dtype, expr);
  };

  auto is_canonical_tail_axis_predicate = [&](const PrimExpr &predicate,
                                              TileBlockState *state, int axis) {
    if (axis < 0 || axis >= static_cast<int>(scope.tile_shape.size()) ||
        axis >= static_cast<int>(scope.execution_loops.size()) ||
        axis >= static_cast<int>(scope.execution_domain_axes.size())) {
      return false;
    }
    if (state->interior_axis0_loop == nullptr ||
        (scope.tile_shape.size() > 1 &&
         state->interior_axis1_loop == nullptr)) {
      return false;
    }
    const auto *lt = predicate.as<LTNode>();
    if (!lt) {
      return false;
    }

    const ForNode *exec_loop = scope.execution_loops[axis];
    const ForNode *interior_loop =
        axis == 0 ? state->interior_axis0_loop : state->interior_axis1_loop;
    if (exec_loop == nullptr || interior_loop == nullptr) {
      return false;
    }

    auto normalize_exec_var = [&](const PrimExpr &expr) {
      std::vector<const VarNode *> vars;
      tir::PostOrderVisit(expr, [&](const ObjectRef &obj) {
        const auto *var = obj.as<VarNode>();
        if (var == nullptr || var == exec_loop->loop_var.get() ||
            var == interior_loop->loop_var.get()) {
          return;
        }
        if (std::find(vars.begin(), vars.end(), var) == vars.end()) {
          vars.push_back(var);
        }
      });
      if (vars.size() != 1) {
        return expr;
      }
      PrimExpr candidate = tir::Substitute(
          expr, {{ffi::GetRef<Var>(vars[0]), exec_loop->loop_var}});
      arith::Analyzer check_analyzer;
      Array<Var> check_vars{exec_loop->loop_var, interior_loop->loop_var};
      Array<PrimExpr> check_coeffs =
          arith::DetectLinearEquation(candidate, check_vars);
      if (check_coeffs.empty() || check_coeffs.size() != 3U) {
        return expr;
      }
      PrimExpr exec_coeff = check_analyzer.Simplify(check_coeffs[0]);
      PrimExpr interior_coeff = check_analyzer.Simplify(check_coeffs[1]);
      if (!check_analyzer.CanProve(
              exec_coeff ==
              make_const(exec_coeff.dtype(),
                         static_cast<int64_t>(scope.tile_shape[axis])))) {
        return expr;
      }
      if (!check_analyzer.CanProve(interior_coeff ==
                                   make_const(interior_coeff.dtype(), 1))) {
        return expr;
      }
      return candidate;
    };

    PrimExpr lhs = normalize_exec_var(lt->a);
    arith::Analyzer analyzer;
    Array<Var> vars{exec_loop->loop_var, interior_loop->loop_var};
    Array<PrimExpr> coeffs = arith::DetectLinearEquation(lhs, vars);
    if (coeffs.empty()) {
      return false;
    }
    ICHECK_EQ(coeffs.size(), 3U);
    PrimExpr exec_coeff = analyzer.Simplify(coeffs[0]);
    PrimExpr interior_coeff = analyzer.Simplify(coeffs[1]);
    PrimExpr base = analyzer.Simplify(coeffs[2]);
    if (!analyzer.CanProve(
            exec_coeff ==
            make_const(exec_coeff.dtype(),
                       static_cast<int64_t>(scope.tile_shape[axis])))) {
      return false;
    }
    if (!analyzer.CanProve(interior_coeff ==
                           make_const(interior_coeff.dtype(), 1))) {
      return false;
    }
    if (!analyzer.CanProve(base == make_zero(base.dtype()))) {
      return false;
    }

    int domain_axis = scope.execution_domain_axes[axis];
    if (domain_axis < 0 ||
        domain_axis >= static_cast<int>(scope.domain_shape.size())) {
      return false;
    }
    PrimExpr expected_rhs =
        cast_prim_expr(scope.domain_shape[domain_axis], lt->b.dtype());
    return can_prove_expr_equal(lt->b, expected_rhs);
  };

  auto is_canonical_tail_load_predicate = [&](const PrimExpr &predicate,
                                              TileBlockState *state,
                                              const TileAccessInfo &access) {
    if (access.tile_rank == 1) {
      ICHECK_EQ(access.tile_axes.size(), 1U);
      int axis = access.tile_axes[0];
      if (axis < 0 || axis >= static_cast<int>(scope.tile_shape.size()) ||
          access.tile_shape.empty() ||
          access.tile_shape[0] != scope.tile_shape[axis]) {
        return false;
      }
      return is_canonical_tail_axis_predicate(predicate, state, axis);
    }

    if (access.tile_shape != scope.tile_shape) {
      return false;
    }

    if (scope.tile_shape.size() == 1) {
      return is_canonical_tail_axis_predicate(predicate, state, 0);
    }

    if (scope.is_reduce_scope) {
      return false;
    }

    if (!state->tile_mask.has_value() || !scope.tail_predicate.defined() ||
        scope.tile_shape.size() != 2) {
      return false;
    }

    std::array<bool, 2> matched_axes{false, false};
    std::function<bool(const PrimExpr &)> collect_axis_predicate =
        [&](const PrimExpr &expr) -> bool {
      if (const auto *and_op = expr.as<AndNode>()) {
        return collect_axis_predicate(and_op->a) &&
               collect_axis_predicate(and_op->b);
      }
      for (int axis = 0; axis < 2; ++axis) {
        if (is_canonical_tail_axis_predicate(expr, state, axis)) {
          matched_axes[axis] = true;
          return true;
        }
      }
      return false;
    };

    if (!collect_axis_predicate(predicate)) {
      return false;
    }
    return matched_axes[0] && matched_axes[1];
  };

  auto match_canonical_rank2_predicate_axes = [&](const PrimExpr &predicate,
                                                  TileBlockState *state) {
    std::array<bool, 2> matched_axes{false, false};
    if (scope.tile_shape.size() != 2) {
      return matched_axes;
    }
    std::function<bool(const PrimExpr &)> collect =
        [&](const PrimExpr &expr) -> bool {
      if (const auto *and_op = expr.as<AndNode>()) {
        return collect(and_op->a) && collect(and_op->b);
      }
      for (int axis = 0; axis < 2; ++axis) {
        if (is_canonical_tail_axis_predicate(expr, state, axis)) {
          matched_axes[axis] = true;
          return true;
        }
      }
      return false;
    };
    if (!collect(predicate)) {
      return std::array<bool, 2>{false, false};
    }
    return matched_axes;
  };

  auto build_canonical_rank2_predicate_mask =
      [&](const PrimExpr &predicate, TileBlockState *state,
          const TileAccessInfo &access,
          DataType mask_index_dtype) -> std::optional<SunMMIOValue> {
    if (access.tile_rank != 2 || scope.tile_shape.size() != 2 ||
        access.tile_shape.size() != 2) {
      return std::nullopt;
    }
    std::array<bool, 2> matched_axes =
        match_canonical_rank2_predicate_axes(predicate, state);
    if (!matched_axes[0] && !matched_axes[1]) {
      return std::nullopt;
    }

    SunMMIOType mask_type = MakeTileType(DataType::Bool(), access.tile_shape);
    std::optional<SunMMIOValue> mask;
    for (int axis = 0; axis < 2; ++axis) {
      if (!matched_axes[axis]) {
        continue;
      }
      int domain_axis = scope.execution_domain_axes[axis];
      SunMMIOValue scope_tile_extent = make_index_const(scope.tile_shape[axis]);
      SunMMIOValue access_tile_extent =
          make_index_const(access.tile_shape[axis]);
      const SunMMIOValue &domain_extent = domain_value(domain_axis);
      SunMMIOValue exec_index =
          EnsureIndex(EvalExpr(scope.execution_loops[axis]->loop_var));
      SunMMIOValue valid_extent = min_index(
          access_tile_extent,
          sub_index(domain_extent, mul_index(exec_index, scope_tile_extent)));
      SunMMIOValue axis_mask = builder_->TileAxisMask(
          NewValueName(), axis, valid_extent, mask_type, mask_index_dtype);
      mask = mask.has_value()
                 ? std::optional<SunMMIOValue>(builder_->TileMaskAnd(
                       NewValueName(), mask.value(), axis_mask, mask_type))
                 : std::optional<SunMMIOValue>(axis_mask);
    }
    return mask;
  };

  auto build_canonical_tail_mask = [&](const TileAccessInfo &access,
                                       DataType mask_index_dtype) {
    ICHECK_EQ(access.tile_rank, 1)
        << "Canonical single-axis tail mask expects a rank-1 tile access";
    ICHECK_EQ(access.tile_axes.size(), 1U);
    int axis = access.tile_axes[0];
    ICHECK_GE(axis, 0);
    ICHECK_LT(axis, static_cast<int>(scope.tile_shape.size()));
    ICHECK_LT(axis, static_cast<int>(scope.execution_domain_axes.size()));
    ICHECK_LT(axis, static_cast<int>(scope.execution_loops.size()));
    ICHECK(scope.execution_loops[axis] != nullptr)
        << "Canonical single-axis tail mask is missing the execution loop";
    int domain_axis = scope.execution_domain_axes[axis];
    ICHECK_GE(domain_axis, 0);
    ICHECK_LT(domain_axis, static_cast<int>(scope.domain_shape.size()));

    SunMMIOValue tile_extent = make_index_const(scope.tile_shape[axis]);
    const SunMMIOValue &domain_extent = domain_value(domain_axis);
    SunMMIOValue exec_index =
        EnsureIndex(EvalExpr(scope.execution_loops[axis]->loop_var));
    SunMMIOValue valid_lanes =
        min_index(tile_extent,
                  sub_index(domain_extent, mul_index(exec_index, tile_extent)));

    SunMMIOType mask_type = MakeTileType(DataType::Bool(), access.tile_shape);
    return builder_->TileAxisMask(NewValueName(), 0, valid_lanes, mask_type,
                                  mask_index_dtype);
  };

  auto mx_scale_e8m0_prefix_extent =
      [&](DataType value_dtype, const Buffer &buffer,
          const TileAccessInfo &access,
          bool predicate_defined) -> std::optional<int64_t> {
    DataType dtype = CanonicalizeSuvmDType(value_dtype).with_lanes(1);
    if (!dtype.is_float8_e8m0fnu() || access.requires_aligned_1d_load ||
        access.tile_shape.size() != 1 || access.tile_shape[0] != 64 ||
        access.tiled_dims.size() != 1 || !predicate_defined) {
      return std::nullopt;
    }

    int64_t tiled_dim = access.tiled_dims[0];
    if (tiled_dim < 0 ||
        tiled_dim >= static_cast<int64_t>(buffer->shape.size())) {
      return std::nullopt;
    }
    const auto *extent =
        buffer->shape[static_cast<size_t>(tiled_dim)].as<IntImmNode>();
    if (!extent || extent->value != 32) {
      return std::nullopt;
    }
    return static_cast<int64_t>(extent->value);
  };
  auto mx_scale_e8m0_prefix_load_extent =
      [&](const BufferLoadNode *load,
          const TileAccessInfo &access) -> std::optional<int64_t> {
    return mx_scale_e8m0_prefix_extent(load->buffer->dtype, load->buffer,
                                       access, load->predicate.defined());
  };
  auto mx_scale_e8m0_prefix_store_extent =
      [&](const BufferStoreNode *store,
          const TileAccessInfo &access) -> std::optional<int64_t> {
    return mx_scale_e8m0_prefix_extent(store->buffer->dtype, store->buffer,
                                       access, store->predicate.defined());
  };

  auto merge_broadcast_shapes = [&](const std::vector<int64_t> &lhs,
                                    const std::vector<int64_t> &rhs) {
    ICHECK_EQ(lhs.size(), rhs.size())
        << "Tile expression broadcast currently expects matching ranks";
    std::vector<int64_t> result;
    result.reserve(lhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (lhs[i] == rhs[i]) {
        result.push_back(lhs[i]);
      } else if (lhs[i] == 1) {
        result.push_back(rhs[i]);
      } else if (rhs[i] == 1) {
        result.push_back(lhs[i]);
      } else {
        LOG(FATAL) << "Incompatible tile expression broadcast shapes";
      }
    }
    return result;
  };

  auto tile_result_shape = [&](const SunMMIOValue &lhs,
                               const SunMMIOValue &rhs) {
    if (IsTileLike(lhs) && IsTileLike(rhs)) {
      std::vector<int64_t> lhs_shape = ExtractStaticShape(lhs.type);
      std::vector<int64_t> rhs_shape = ExtractStaticShape(rhs.type);
      if (lhs_shape.size() == rhs_shape.size()) {
        return merge_broadcast_shapes(lhs_shape, rhs_shape);
      }
      if (lhs_shape.size() == 1 && rhs_shape.size() == 2) {
        ICHECK(lhs_shape[0] == rhs_shape[0] || lhs_shape[0] == rhs_shape[1])
            << "Rank-1 tile expression operand cannot be oriented to rank-2 "
               "result";
        return rhs_shape;
      }
      if (lhs_shape.size() == 2 && rhs_shape.size() == 1) {
        ICHECK(rhs_shape[0] == lhs_shape[0] || rhs_shape[0] == lhs_shape[1])
            << "Rank-1 tile expression operand cannot be oriented to rank-2 "
               "result";
        return lhs_shape;
      }
      LOG(FATAL) << "Tile expression broadcast currently supports only rank-1 "
                    "and rank-2 operands";
    }
    if (IsTileLike(lhs)) {
      return ExtractStaticShape(lhs.type);
    }
    if (IsTileLike(rhs)) {
      return ExtractStaticShape(rhs.type);
    }
    return scope.tile_shape;
  };

  auto merge_optional_tile_shapes =
      [&](const std::optional<std::vector<int64_t>> &lhs,
          const std::optional<std::vector<int64_t>> &rhs)
      -> std::optional<std::vector<int64_t>> {
    if (lhs.has_value() && rhs.has_value()) {
      const std::vector<int64_t> &lhs_shape = lhs.value();
      const std::vector<int64_t> &rhs_shape = rhs.value();
      if (lhs_shape.size() == rhs_shape.size()) {
        return merge_broadcast_shapes(lhs_shape, rhs_shape);
      }
      if (lhs_shape.size() == 1 && rhs_shape.size() == 2) {
        ICHECK(lhs_shape[0] == rhs_shape[0] || lhs_shape[0] == rhs_shape[1])
            << "Rank-1 tile expression operand cannot be oriented to rank-2 "
               "result";
        return rhs_shape;
      }
      if (lhs_shape.size() == 2 && rhs_shape.size() == 1) {
        ICHECK(rhs_shape[0] == lhs_shape[0] || rhs_shape[0] == lhs_shape[1])
            << "Rank-1 tile expression operand cannot be oriented to rank-2 "
               "result";
        return lhs_shape;
      }
      LOG(FATAL) << "Tile expression broadcast currently supports only rank-1 "
                    "and rank-2 operands";
    }
    if (lhs.has_value()) {
      return lhs;
    }
    return rhs;
  };

  std::function<std::optional<std::vector<int64_t>>(const PrimExpr &,
                                                    TileBlockState *)>
      infer_tile_expr_shape;
  std::function<std::optional<int64_t>(const PrimExpr &, TileBlockState *)>
      infer_tile_expr_unsqueeze_axis;

  broadcast_tile_to_shape =
      [&](const SunMMIOValue &value,
          const std::vector<int64_t> &dst_shape) -> SunMMIOValue {
    if (!IsTileLike(value)) {
      return value;
    }
    SunMMIOValue tile = value;
    std::vector<int64_t> src_shape = ExtractStaticShape(tile.type);
    if (src_shape == dst_shape) {
      return tile;
    }
    if (src_shape.size() != dst_shape.size()) {
      tile = reorient_unit_tile_to_shape(tile, dst_shape);
      src_shape = ExtractStaticShape(tile.type);
      if (src_shape == dst_shape) {
        return tile;
      }
    }
    ICHECK_EQ(src_shape.size(), dst_shape.size())
        << "Tile broadcast expects rank-compatible shapes";
    ICHECK(CanBroadcastShapeTo(src_shape, dst_shape))
        << "Tile value with shape " << shape_to_string(src_shape)
        << " is not broadcastable to target shape "
        << shape_to_string(dst_shape);
    ICHECK(!tile.dtype.is_bool())
        << "Bool tile predicates must be lowered to the target shape directly; "
           "cannot broadcast mask from "
        << shape_to_string(src_shape) << " to " << shape_to_string(dst_shape);
    SunMMIOType dst_type = MakeTileType(tile.dtype, dst_shape);
    return builder_->TileBroadcast(NewValueName(), tile, dst_type, tile.dtype);
  };

  orient_tile_operand_to_shape =
      [&](const SunMMIOValue &value,
          const std::vector<int64_t> &result_shape) -> SunMMIOValue {
    if (!IsTileLike(value)) {
      return value;
    }
    SunMMIOValue tile = value;
    std::vector<int64_t> src_shape = ExtractStaticShape(tile.type);
    if (src_shape.size() == result_shape.size()) {
      return tile;
    }
    tile = reorient_unit_tile_to_shape(tile, result_shape);
    ICHECK_EQ(ExtractStaticShape(tile.type).size(), result_shape.size())
        << "Tile binary operand rank cannot be normalized to result rank";
    return tile;
  };

  auto lower_interior_loop_var =
      [&](const VarNode *var, TileBlockState *state,
          std::optional<DataType> preferred_index_dtype)
      -> std::optional<SunMMIOValue> {
    auto try_axis = [&](const ForNode *loop,
                        int64_t axis) -> std::optional<SunMMIOValue> {
      if (loop == nullptr || loop->loop_var.get() != var) {
        return std::nullopt;
      }
      const auto *min_imm = loop->min.as<IntImmNode>();
      ICHECK(min_imm && min_imm->value == 0)
          << "Tile interior loop var lowering expects zero-based loops";
      std::optional<int64_t> extent = GetStaticLoopExtent(loop);
      ICHECK(extent.has_value())
          << "Tile interior loop var lowering expects static extent";
      DataType dtype =
          CanonicalizeSuvmDType(loop->loop_var.dtype()).with_lanes(1);
      if (auto preferred =
              canonical_integer_preferred_dtype(preferred_index_dtype)) {
        dtype = preferred.value();
      }
      SunMMIOValue range = builder_->TileRange(
          NewValueName(), MakeTileType(dtype, {*extent}), dtype);
      if (scope.tile_shape.size() == 1) {
        return range;
      }
      ICHECK_EQ(scope.tile_shape.size(), 2U)
          << "Tile interior loop var lowering expects 1D or 2D tiles";
      std::vector<int64_t> unit_shape = axis == 0
                                            ? std::vector<int64_t>{*extent, 1}
                                            : std::vector<int64_t>{1, *extent};
      int64_t unsqueeze_axis = axis == 0 ? 1 : 0;
      return checked_tile_unsqueeze(range, MakeTileType(dtype, unit_shape),
                                    unsqueeze_axis, dtype,
                                    "tile interior loop var");
    };

    if (auto value = try_axis(state->interior_axis0_loop, 0)) {
      return value;
    }
    if (auto value = try_axis(state->interior_axis1_loop, 1)) {
      return value;
    }
    return std::nullopt;
  };

  infer_tile_expr_shape =
      [&](const PrimExpr &expr,
          TileBlockState *state) -> std::optional<std::vector<int64_t>> {
    auto infer_binary_shape =
        [&](const PrimExpr &lhs,
            const PrimExpr &rhs) -> std::optional<std::vector<int64_t>> {
      return merge_optional_tile_shapes(infer_tile_expr_shape(lhs, state),
                                        infer_tile_expr_shape(rhs, state));
    };
    if (const auto *var = expr.as<VarNode>()) {
      auto let_it = state->let_values.find(var);
      if (let_it != state->let_values.end() && IsTileLike(let_it->second)) {
        return ExtractStaticShape(let_it->second.type);
      }
      auto try_axis = [&](const ForNode *loop,
                          int64_t axis) -> std::optional<std::vector<int64_t>> {
        if (loop == nullptr || loop->loop_var.get() != var) {
          return std::nullopt;
        }
        std::optional<int64_t> extent = GetStaticLoopExtent(loop);
        ICHECK(extent.has_value())
            << "Tile expression shape inference expects static interior loops";
        if (scope.tile_shape.size() == 1) {
          return std::vector<int64_t>{*extent};
        }
        ICHECK_EQ(scope.tile_shape.size(), 2U)
            << "Tile expression shape inference expects 1D or 2D tiles";
        return axis == 0 ? std::vector<int64_t>{*extent, 1}
                         : std::vector<int64_t>{1, *extent};
      };
      if (auto shape = try_axis(state->interior_axis0_loop, 0)) {
        return shape;
      }
      if (auto shape = try_axis(state->interior_axis1_loop, 1)) {
        return shape;
      }
      return std::nullopt;
    }
    if (const auto *let = expr.as<LetNode>()) {
      return infer_tile_expr_shape(let->body, state);
    }
    if (const auto *load = expr.as<BufferLoadNode>()) {
      auto local_it = state->local_tile_values.find(load->buffer.get());
      if (local_it != state->local_tile_values.end() &&
          IsTileLike(local_it->second)) {
        return ExtractStaticShape(local_it->second.type);
      }
      auto reg_it = state->register_tile_values.find(load->buffer.get());
      if (reg_it != state->register_tile_values.end() &&
          IsTileLike(reg_it->second)) {
        return ExtractStaticShape(reg_it->second.type);
      }
      TileAccessInfo access =
          analyze_access(load->buffer, load->indices, state);
      if (access.requires_aligned_1d_load) {
        return access.tile_shape;
      }
      std::string cache_key = make_tile_cache_key(access);
      auto current_it = state->current_tile_values.find(cache_key);
      if (current_it != state->current_tile_values.end() &&
          IsTileLike(current_it->second)) {
        return ExtractStaticShape(current_it->second.type);
      }
      return std::nullopt;
    }
    if (const auto *cast = expr.as<CastNode>()) {
      return infer_tile_expr_shape(cast->value, state);
    }
    if (const auto *add = expr.as<AddNode>()) {
      return infer_binary_shape(add->a, add->b);
    }
    if (const auto *sub = expr.as<SubNode>()) {
      return infer_binary_shape(sub->a, sub->b);
    }
    if (const auto *mul = expr.as<MulNode>()) {
      return infer_binary_shape(mul->a, mul->b);
    }
    if (const auto *div = expr.as<DivNode>()) {
      return infer_binary_shape(div->a, div->b);
    }
    if (const auto *div = expr.as<FloorDivNode>()) {
      return infer_binary_shape(div->a, div->b);
    }
    if (const auto *mod = expr.as<ModNode>()) {
      return infer_binary_shape(mod->a, mod->b);
    }
    if (const auto *mod = expr.as<FloorModNode>()) {
      return infer_binary_shape(mod->a, mod->b);
    }
    if (const auto *min = expr.as<MinNode>()) {
      return infer_binary_shape(min->a, min->b);
    }
    if (const auto *max = expr.as<MaxNode>()) {
      return infer_binary_shape(max->a, max->b);
    }
    if (const auto *eq = expr.as<EQNode>()) {
      return infer_binary_shape(eq->a, eq->b);
    }
    if (const auto *ne = expr.as<NENode>()) {
      return infer_binary_shape(ne->a, ne->b);
    }
    if (const auto *lt = expr.as<LTNode>()) {
      return infer_binary_shape(lt->a, lt->b);
    }
    if (const auto *le = expr.as<LENode>()) {
      return infer_binary_shape(le->a, le->b);
    }
    if (const auto *gt = expr.as<GTNode>()) {
      return infer_binary_shape(gt->a, gt->b);
    }
    if (const auto *ge = expr.as<GENode>()) {
      return infer_binary_shape(ge->a, ge->b);
    }
    if (const auto *and_op = expr.as<AndNode>()) {
      return infer_binary_shape(and_op->a, and_op->b);
    }
    if (const auto *or_op = expr.as<OrNode>()) {
      return infer_binary_shape(or_op->a, or_op->b);
    }
    if (const auto *select = expr.as<SelectNode>()) {
      return merge_optional_tile_shapes(
          infer_tile_expr_shape(select->condition, state),
          merge_optional_tile_shapes(
              infer_tile_expr_shape(select->true_value, state),
              infer_tile_expr_shape(select->false_value, state)));
    }
    if (const auto *call = expr.as<CallNode>()) {
      const auto *op_node = call->op.as<OpNode>();
      if (op_node && call->args.size() >= 3 &&
          op_node->name == "tir.if_then_else") {
        return merge_optional_tile_shapes(
            infer_tile_expr_shape(call->args[0], state),
            merge_optional_tile_shapes(
                infer_tile_expr_shape(call->args[1], state),
                infer_tile_expr_shape(call->args[2], state)));
      }
      if (op_node && call->args.size() == 1) {
        return infer_tile_expr_shape(call->args[0], state);
      }
    }
    return std::nullopt;
  };

  infer_tile_expr_unsqueeze_axis =
      [&](const PrimExpr &expr,
          TileBlockState *state) -> std::optional<int64_t> {
    std::optional<int64_t> inferred_axis;
    bool has_conflict = false;
    auto merge_axis = [&](std::optional<int64_t> axis) {
      if (!axis.has_value() || has_conflict) {
        return;
      }
      ICHECK(axis.value() == 0 || axis.value() == 1)
          << "Rank-1 tile orientation expects an unsqueeze axis of 0 or 1";
      if (inferred_axis.has_value() && inferred_axis.value() != axis.value()) {
        has_conflict = true;
        inferred_axis.reset();
        return;
      }
      inferred_axis = axis;
    };

    tir::PostOrderVisit(expr, [&](const ObjectRef &obj) {
      if (has_conflict) {
        return;
      }
      if (const auto *var = obj.as<VarNode>()) {
        auto axis_it = state->let_unsqueeze_axes.find(var);
        if (axis_it != state->let_unsqueeze_axes.end()) {
          merge_axis(axis_it->second);
        }
        return;
      }
      const auto *load = obj.as<BufferLoadNode>();
      if (!load) {
        return;
      }

      auto local_it = state->local_tile_values.find(load->buffer.get());
      if (local_it != state->local_tile_values.end()) {
        if (ExtractStaticShape(local_it->second.type).size() == 1) {
          auto axis_it = state->local_unit_tile_axes.find(load->buffer.get());
          if (axis_it != state->local_unit_tile_axes.end()) {
            merge_axis(axis_it->second);
          }
        }
        return;
      }
      auto register_it = state->register_tile_values.find(load->buffer.get());
      if (register_it != state->register_tile_values.end()) {
        if (ExtractStaticShape(register_it->second.type).size() == 1) {
          auto axis_it =
              state->register_unsqueeze_axes.find(load->buffer.get());
          if (axis_it != state->register_unsqueeze_axes.end()) {
            merge_axis(axis_it->second);
          }
        }
        return;
      }

      TileAccessInfo access =
          analyze_access(load->buffer, load->indices, state);
      if (access.tile_rank == 1 && access.tile_shape[0] > 1) {
        merge_axis(access.unsqueeze_axis);
      }
    });
    return has_conflict ? std::nullopt : inferred_axis;
  };

  lower_expr = [&](const PrimExpr &expr, TileBlockState *state,
                   std::optional<DataType> preferred_dtype) -> SunMMIOValue {
    MarkVisitedExprRoot(expr);
    std::function<SunMMIOValue(const PrimExpr &, const std::vector<int64_t> &,
                               std::optional<DataType>)>
        lower_bool_expr_to_shape;
    auto rewrite_mask_condition = [&](const PrimExpr &condition) {
      PrimExpr rewritten_condition = condition;
      std::vector<const VarNode *> interior_vars;
      if (state->interior_axis0_loop != nullptr) {
        interior_vars.push_back(state->interior_axis0_loop->loop_var.get());
      }
      if (state->interior_axis1_loop != nullptr) {
        interior_vars.push_back(state->interior_axis1_loop->loop_var.get());
      }
      if (auto rewritten =
              TryRewriteAffineInteriorLT(rewritten_condition, interior_vars)) {
        MarkVisitedExprTree(rewritten_condition);
        rewritten_condition = rewritten.value();
      }
      if (ContainsFloorDivOrMod(rewritten_condition)) {
        if (auto rewritten = TryRewritePositiveFloorDivTailCompare(
                rewritten_condition, interior_vars)) {
          MarkVisitedExprTree(rewritten_condition);
          rewritten_condition = rewritten.value();
        }
      }
      return rewritten_condition;
    };
    auto orient_expr_tile_operand =
        [&](const SunMMIOValue &value, const PrimExpr &source_expr,
            const std::vector<int64_t> &result_shape) -> SunMMIOValue {
      if (!IsTileLike(value)) {
        return value;
      }
      std::vector<int64_t> src_shape = ExtractStaticShape(value.type);
      if (src_shape.size() == 1 && result_shape.size() == 2) {
        if (auto axis = infer_tile_expr_unsqueeze_axis(source_expr, state)) {
          std::vector<int64_t> unit_shape =
              axis.value() == 1 ? std::vector<int64_t>{src_shape[0], 1}
                                : std::vector<int64_t>{1, src_shape[0]};
          if (CanBroadcastShapeTo(unit_shape, result_shape)) {
            return checked_tile_unsqueeze(
                value, MakeTileType(value.dtype, unit_shape), axis.value(),
                value.dtype, "orient rank-1 expression at rank-2 consumer");
          }
        }
      }
      return orient_tile_operand_to_shape(value, result_shape);
    };
    auto emit_select = [&](const PrimExpr &condition,
                           const PrimExpr &true_value_expr,
                           const PrimExpr &false_value_expr, DataType dtype) {
      MarkVisitedExprRoot(condition);
      PrimExpr condition_to_lower = rewrite_mask_condition(condition);
      SunMMIOValue true_value =
          lower_expr(true_value_expr, state, preferred_dtype);
      SunMMIOValue false_value =
          lower_expr(false_value_expr, state, preferred_dtype);
      std::optional<std::vector<int64_t>> value_shape =
          merge_optional_tile_shapes(
              IsTileLike(true_value) ? std::optional<std::vector<int64_t>>(
                                           ExtractStaticShape(true_value.type))
                                     : std::nullopt,
              IsTileLike(false_value)
                  ? std::optional<std::vector<int64_t>>(
                        ExtractStaticShape(false_value.type))
                  : std::nullopt);
      DataType mask_index_dtype = mask_index_dtype_for_value_dtype(dtype);
      SunMMIOValue cond =
          value_shape.has_value()
              ? lower_bool_expr_to_shape(condition_to_lower,
                                         value_shape.value(), mask_index_dtype)
              : lower_expr(condition_to_lower, state, mask_index_dtype);
      if (IsTileLike(cond) || IsTileLike(true_value) ||
          IsTileLike(false_value)) {
        std::vector<int64_t> result_shape;
        if (IsTileLike(true_value) && IsTileLike(false_value)) {
          result_shape = tile_result_shape(true_value, false_value);
        } else if (IsTileLike(true_value)) {
          result_shape = ExtractStaticShape(true_value.type);
        } else if (IsTileLike(false_value)) {
          result_shape = ExtractStaticShape(false_value.type);
        } else {
          result_shape = ExtractStaticShape(cond.type);
        }
        if (IsTileLike(cond)) {
          cond =
              orient_expr_tile_operand(cond, condition_to_lower, result_shape);
          std::vector<int64_t> cond_shape = ExtractStaticShape(cond.type);
          if (cond_shape != result_shape) {
            result_shape = merge_broadcast_shapes(cond_shape, result_shape);
          }
        }
        DataType result_dtype = choose_result_dtype(dtype, preferred_dtype);
        SunMMIOType scalar_type{
            SunMMIOType::Kind::kScalar, result_dtype, 1, {}};
        if (IsTileLike(true_value)) {
          true_value = orient_expr_tile_operand(true_value, true_value_expr,
                                                result_shape);
          true_value = broadcast_tile_to_shape(
              cast_value_to_dtype(true_value, result_dtype), result_shape);
        } else {
          true_value = EnsureType(true_value, scalar_type, result_dtype);
          true_value = builder_->TileFill(
              NewValueName(), true_value,
              MakeTileType(result_dtype, result_shape), result_dtype);
        }
        if (IsTileLike(false_value)) {
          false_value = orient_expr_tile_operand(false_value, false_value_expr,
                                                 result_shape);
          false_value = broadcast_tile_to_shape(
              cast_value_to_dtype(false_value, result_dtype), result_shape);
        } else {
          false_value = EnsureType(false_value, scalar_type, result_dtype);
          false_value = builder_->TileFill(
              NewValueName(), false_value,
              MakeTileType(result_dtype, result_shape), result_dtype);
        }
        if (IsTileLike(cond)) {
          cond = broadcast_tile_to_shape(cond, result_shape);
        } else {
          SunMMIOType cond_scalar_type{
              SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
          cond = EnsureType(cond, cond_scalar_type, DataType::Bool());
          cond = builder_->TileFill(
              NewValueName(), cond,
              MakeTileType(DataType::Bool(), result_shape), DataType::Bool());
        }
        SunMMIOType result_type = MakeTileType(result_dtype, result_shape);
        return builder_->TileSelect(NewValueName(), cond, true_value,
                                    false_value, result_type, result_dtype);
      }
      DataType result_dtype = choose_result_dtype(dtype, preferred_dtype);
      SunMMIOType result_type{SunMMIOType::Kind::kScalar, result_dtype, 1, {}};
      return builder_->Select(NewValueName(), cond, true_value, false_value,
                              result_type, result_dtype);
    };

    if (const auto *var = expr.as<VarNode>()) {
      auto it = state->let_values.find(var);
      if (it != state->let_values.end()) {
        return it->second;
      }
      if (auto interior_value =
              lower_interior_loop_var(var, state, preferred_dtype)) {
        return *interior_value;
      }
      return LookupVar(var);
    }
    if (const auto *let = expr.as<LetNode>()) {
      SunMMIOValue value = lower_expr(let->value, state, preferred_dtype);
      TileBlockState let_state = *state;
      let_state.let_values[let->var.get()] = value;
      if (auto axis = infer_tile_expr_unsqueeze_axis(let->value, state)) {
        let_state.let_unsqueeze_axes[let->var.get()] = axis.value();
      }
      return lower_expr(let->body, &let_state, preferred_dtype);
    }
    if (const auto *load = expr.as<BufferLoadNode>()) {
      if (IsSunmmioLocalVarBuffer(load->buffer)) {
        return EmitLocalVarLoad(load->buffer, load->indices);
      }
      auto local_it = state->local_tile_values.find(load->buffer.get());
      if (local_it != state->local_tile_values.end()) {
        return local_it->second;
      }
      auto reg_it = state->register_tile_values.find(load->buffer.get());
      if (reg_it != state->register_tile_values.end()) {
        SunMMIOValue value = reg_it->second;
        if (IsTileLike(value) && value.type.shape.size() == 1 &&
            state->interior_axis0_loop != nullptr &&
            state->interior_axis1_loop == nullptr) {
          int64_t unsqueeze_axis = 0;
          auto axis_it =
              state->register_unsqueeze_axes.find(load->buffer.get());
          if (axis_it != state->register_unsqueeze_axes.end()) {
            unsqueeze_axis = axis_it->second;
          }
          ICHECK(unsqueeze_axis == 0 || unsqueeze_axis == 1)
              << "1D register tile can only be unsqueezed back to a 2D tile";
          int64_t extent = ExtractStaticShape(value.type)[0];
          std::vector<int64_t> unsqueezed_shape =
              unsqueeze_axis == 0 ? std::vector<int64_t>{1, extent}
                                  : std::vector<int64_t>{extent, 1};
          SunMMIOType unsqueezed_type =
              MakeTileType(value.dtype, unsqueezed_shape);
          value =
              checked_tile_unsqueeze(value, unsqueezed_type, unsqueeze_axis,
                                     value.dtype, "reduce register tile load");
        }
        return value;
      }
      TileAccessInfo access =
          analyze_access(load->buffer, load->indices, state);
      std::optional<int64_t> mx_scale_valid_elems =
          mx_scale_e8m0_prefix_load_extent(load, access);
      std::string cache_key = make_tile_cache_key(access);
      if (!access.promoted_unit_tile_view && !access.requires_aligned_1d_load) {
        auto it = state->current_tile_values.find(cache_key);
        if (it != state->current_tile_values.end() &&
            cached_value_matches_access(it->second, access) &&
            state->mlir_ctx != nullptr &&
            state->mlir_ctx->LookupMLIRValue(it->second.value)) {
          return it->second;
        }
      }
      SunMMIOValue tile;
      if (access.requires_aligned_1d_load) {
        tile = load_aligned_1d_tile(access, state);
      } else {
        SunMMIOValue view = get_or_create_tile_view(access, state);
        SunMMIOType tile_type =
            MakeTileType(load->buffer->dtype, access.tile_shape);
        std::optional<SunMMIOValue> load_mask;
        std::optional<SunMMIOValue> load_maskedoff;
        bool skip_load_predicate = false;
        if (load->predicate.defined()) {
          skip_load_predicate =
              mx_scale_valid_elems.has_value() ||
              (state->active_tail_store_predicate.has_value() &&
               can_prove_expr_equal(
                   load->predicate.value(),
                   state->active_tail_store_predicate.value())) ||
              is_canonical_tail_load_predicate(load->predicate.value(), state,
                                               access);
        }
        if (load->predicate.defined() && !skip_load_predicate) {
          DataType mask_index_dtype =
              mask_index_dtype_for_value_dtype(load->buffer->dtype);
          std::optional<SunMMIOValue> canonical_mask =
              build_canonical_rank2_predicate_mask(
                  load->predicate.value(), state, access, mask_index_dtype);
          SunMMIOValue lowered_mask =
              canonical_mask.has_value()
                  ? canonical_mask.value()
                  : lower_bool_expr_to_shape(load->predicate.value(),
                                             access.tile_shape,
                                             mask_index_dtype);
          if (!IsTileLike(lowered_mask)) {
            SunMMIOType bool_scalar_type{
                SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
            lowered_mask =
                EnsureType(lowered_mask, bool_scalar_type, DataType::Bool());
            lowered_mask = builder_->TileFill(
                NewValueName(), lowered_mask,
                MakeTileType(DataType::Bool(), access.tile_shape),
                DataType::Bool());
          }
          DataType value_dtype =
              CanonicalizeSuvmDType(load->buffer->dtype).with_lanes(1);
          SunMMIOType scalar_type{
              SunMMIOType::Kind::kScalar, value_dtype, 1, {}};
          SunMMIOValue zero =
              value_dtype.is_float() || value_dtype.is_bfloat16()
                  ? builder_->ConstantFloat(NewValueName(), "0.0", scalar_type,
                                            value_dtype)
                  : builder_->ConstantInt(NewValueName(), 0, scalar_type,
                                          value_dtype);
          load_maskedoff =
              builder_->TileFill(NewValueName(), zero, tile_type, value_dtype);
          load_mask = lowered_mask;
        }
        // Always load the full padded tile.  Tail stores preserve old
        // destination values explicitly with tile.select, and predicated loads
        // are represented as load + select so this path does not depend on
        // masked tile.load dialect semantics.
        tile = builder_->TileLoad(
            NewValueName(), view, tile_type, std::nullopt, std::nullopt,
            CanonicalizeSuvmDType(load->buffer->dtype).with_lanes(1));
        if (load_mask.has_value()) {
          tile = builder_->TileSelect(
              NewValueName(), load_mask.value(), tile, load_maskedoff.value(),
              tile_type,
              CanonicalizeSuvmDType(load->buffer->dtype).with_lanes(1));
        }
      }
      if (!access.promoted_unit_tile_view && !access.requires_aligned_1d_load) {
        state->current_tile_values[cache_key] = builder_->BindValueAlias(
            make_current_value_name(load->buffer, cache_key), tile);
      }
      return tile;
    }
    if (const auto *imm = expr.as<IntImmNode>()) {
      DataType dtype = CanonicalizeSuvmDType(imm->dtype);
      SunMMIOType scalar_type{SunMMIOType::Kind::kScalar, dtype, 1, {}};
      return builder_->ConstantInt(NewValueName(), imm->value, scalar_type,
                                   dtype.with_lanes(1));
    }
    if (const auto *imm = expr.as<FloatImmNode>()) {
      DataType dtype = CanonicalizeSuvmDType(imm->dtype);
      SunMMIOType scalar_type{SunMMIOType::Kind::kScalar, dtype, 1, {}};
      std::ostringstream os;
      os << std::setprecision(17) << imm->value;
      return builder_->ConstantFloat(NewValueName(), os.str(), scalar_type,
                                     dtype.with_lanes(1));
    }
    auto emit_binary = [&](BinaryOp op, const PrimExpr &lhs_expr,
                           const PrimExpr &rhs_expr, DataType dtype) {
      bool supports_mixed_precision = supports_mixed_precision_binary(op);
      std::optional<DataType> integer_preferred =
          canonical_integer_preferred_dtype(preferred_dtype);
      DataType expr_result_dtype = CanonicalizeSuvmDType(dtype).with_lanes(1);
      bool use_integer_preferred =
          integer_preferred.has_value() && expr_result_dtype.is_int();
      std::optional<DataType> operand_preferred_dtype =
          (supports_mixed_precision || use_integer_preferred) ? preferred_dtype
                                                              : std::nullopt;
      SunMMIOValue lhs = lower_expr(lhs_expr, state, operand_preferred_dtype);
      SunMMIOValue rhs = lower_expr(rhs_expr, state, operand_preferred_dtype);
      DataType result_dtype =
          use_integer_preferred
              ? integer_preferred.value()
              : (supports_mixed_precision
                     ? choose_result_dtype(dtype, preferred_dtype)
                     : expr_result_dtype);
      if (IsScalarLike(lhs) && IsScalarLike(rhs)) {
        SunMMIOType result_type{
            SunMMIOType::Kind::kScalar, result_dtype, 1, {}};
        lhs = EnsureType(lhs, result_type, result_dtype);
        rhs = EnsureType(rhs, result_type, result_dtype);
        return builder_->Binary(NewValueName(), op,
                                arithmetic_flavor_for_dtype(result_dtype), lhs,
                                rhs, result_type, result_dtype);
      }
      std::vector<int64_t> result_shape = tile_result_shape(lhs, rhs);
      lhs = orient_expr_tile_operand(lhs, lhs_expr, result_shape);
      rhs = orient_expr_tile_operand(rhs, rhs_expr, result_shape);
      SunMMIOType tile_type = MakeTileType(result_dtype, result_shape);
      auto broadcast_scalar_to_tile = [&](SunMMIOValue value) {
        if (IsTileLike(value)) {
          return value;
        }
        SunMMIOType scalar_type{
            SunMMIOType::Kind::kScalar, result_dtype, 1, {}};
        value = EnsureType(value, scalar_type, result_dtype);
        return builder_->TileFill(NewValueName(), value, tile_type,
                                  result_dtype);
      };
      if (supports_mixed_precision) {
        if (!IsTileLike(lhs) && !is_float_like_dtype(lhs.dtype)) {
          lhs = cast_value_to_dtype(lhs, result_dtype);
        }
        if (!IsTileLike(rhs) && !is_float_like_dtype(rhs.dtype)) {
          rhs = cast_value_to_dtype(rhs, result_dtype);
        }
      } else {
        lhs = cast_value_to_dtype(lhs, result_dtype);
        rhs = cast_value_to_dtype(rhs, result_dtype);
      }
      return builder_->Binary(NewValueName(), op,
                              arithmetic_flavor_for_dtype(result_dtype), lhs,
                              rhs, tile_type, result_dtype);
    };
    auto emit_compare_to_shape =
        [&](CompareOp op, const PrimExpr &lhs_expr, const PrimExpr &rhs_expr,
            const std::optional<std::vector<int64_t>> &forced_shape,
            std::optional<DataType> operand_preferred_dtype) {
          SunMMIOValue lhs =
              lower_expr(lhs_expr, state, operand_preferred_dtype);
          SunMMIOValue rhs =
              lower_expr(rhs_expr, state, operand_preferred_dtype);
          if (!is_tile_compare_operand(lhs, rhs) && !forced_shape.has_value()) {
            SunMMIOType operand_type = lhs.type;
            rhs = EnsureType(rhs, operand_type, lhs.dtype);
            return builder_->Compare(NewValueName(), op,
                                     GetCompareDomain(lhs.dtype), lhs, rhs,
                                     operand_type);
          }
          if (!is_tile_compare_operand(lhs, rhs)) {
            SunMMIOType operand_type = lhs.type;
            rhs = EnsureType(rhs, operand_type, lhs.dtype);
            return builder_->Compare(NewValueName(), op,
                                     GetCompareDomain(lhs.dtype), lhs, rhs,
                                     operand_type);
          }
          std::vector<int64_t> result_shape = forced_shape.has_value()
                                                  ? forced_shape.value()
                                                  : tile_result_shape(lhs, rhs);
          SunMMIOType operand_type = MakeTileType(
              IsTileLike(lhs) ? lhs.dtype : rhs.dtype, result_shape);
          DataType operand_dtype = operand_type.dtype;
          lhs = orient_expr_tile_operand(lhs, lhs_expr, result_shape);
          rhs = orient_expr_tile_operand(rhs, rhs_expr, result_shape);
          if (forced_shape.has_value()) {
            auto operand_shape = [&](const SunMMIOValue &value) {
              if (IsTileLike(value)) {
                return ExtractStaticShape(value.type);
              }
              return std::vector<int64_t>(result_shape.size(), 1);
            };
            std::vector<int64_t> inferred_shape =
                merge_broadcast_shapes(operand_shape(lhs), operand_shape(rhs));
            if (inferred_shape != result_shape) {
              if (IsTileLike(lhs) &&
                  ExtractStaticShape(lhs.type) != result_shape) {
                lhs = broadcast_tile_to_shape(lhs, result_shape);
              }
              if (IsTileLike(rhs) &&
                  ExtractStaticShape(rhs.type) != result_shape) {
                rhs = broadcast_tile_to_shape(rhs, result_shape);
              }
            }
          }
          if (!IsTileLike(lhs) && lhs.dtype != operand_dtype) {
            SunMMIOType scalar_type{
                SunMMIOType::Kind::kScalar, operand_dtype, 1, {}};
            lhs =
                builder_->Cast(NewValueName(), lhs, scalar_type, operand_dtype);
          }
          if (!IsTileLike(rhs) && rhs.dtype != operand_dtype) {
            SunMMIOType scalar_type{
                SunMMIOType::Kind::kScalar, operand_dtype, 1, {}};
            rhs =
                builder_->Cast(NewValueName(), rhs, scalar_type, operand_dtype);
          }
          return builder_->Compare(NewValueName(), op,
                                   GetCompareDomain(operand_dtype), lhs, rhs,
                                   operand_type);
        };
    auto emit_compare = [&](CompareOp op, const PrimExpr &lhs_expr,
                            const PrimExpr &rhs_expr) {
      return emit_compare_to_shape(
          op, lhs_expr, rhs_expr, std::nullopt,
          canonical_integer_preferred_dtype(preferred_dtype));
    };
    auto try_emit_compare_to_shape =
        [&](const PrimExpr &compare_expr,
            const std::vector<int64_t> &target_shape,
            std::optional<DataType> operand_preferred_dtype)
        -> std::optional<SunMMIOValue> {
      if (const auto *eq = compare_expr.as<EQNode>()) {
        return emit_compare_to_shape(CompareOp::kEQ, eq->a, eq->b, target_shape,
                                     operand_preferred_dtype);
      }
      if (const auto *ne = compare_expr.as<NENode>()) {
        return emit_compare_to_shape(CompareOp::kNE, ne->a, ne->b, target_shape,
                                     operand_preferred_dtype);
      }
      if (const auto *lt = compare_expr.as<LTNode>()) {
        return emit_compare_to_shape(CompareOp::kLT, lt->a, lt->b, target_shape,
                                     operand_preferred_dtype);
      }
      if (const auto *le = compare_expr.as<LENode>()) {
        return emit_compare_to_shape(CompareOp::kLE, le->a, le->b, target_shape,
                                     operand_preferred_dtype);
      }
      if (const auto *gt = compare_expr.as<GTNode>()) {
        return emit_compare_to_shape(CompareOp::kGT, gt->a, gt->b, target_shape,
                                     operand_preferred_dtype);
      }
      if (const auto *ge = compare_expr.as<GENode>()) {
        return emit_compare_to_shape(CompareOp::kGE, ge->a, ge->b, target_shape,
                                     operand_preferred_dtype);
      }
      return std::nullopt;
    };
    auto ensure_logical_scalar = [&](SunMMIOValue value) {
      SunMMIOType bool_type{
          SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
      return EnsureType(value, bool_type, DataType::Bool());
    };
    auto adapt_logical_tile_operand =
        [&](const SunMMIOValue &value, const std::vector<int64_t> &target_shape,
            const PrimExpr &source_expr) {
          if (!IsTileLike(value)) {
            return ensure_logical_scalar(value);
          }
          SunMMIOValue tile =
              orient_expr_tile_operand(value, source_expr, target_shape);
          if (ExtractStaticShape(tile.type) == target_shape) {
            return tile;
          }
          if (tile.dtype.is_bool()) {
            UnsupportedExpr(
                source_expr.get(),
                "Cannot materialize bool tile broadcast with "
                "suvm.tile.broadcast; lower the mask producer to the "
                "target shape instead");
          }
          return broadcast_tile_to_shape(tile, target_shape);
        };
    auto emit_logical_values = [&](BinaryOp op, const SunMMIOValue &lhs_value,
                                   const SunMMIOValue &rhs_value,
                                   const std::vector<int64_t> &result_shape,
                                   const PrimExpr &lhs_expr,
                                   const PrimExpr &rhs_expr) {
      SunMMIOValue lhs =
          adapt_logical_tile_operand(lhs_value, result_shape, lhs_expr);
      SunMMIOValue rhs =
          adapt_logical_tile_operand(rhs_value, result_shape, rhs_expr);
      if (IsScalarLike(lhs) && IsScalarLike(rhs)) {
        SunMMIOType result_type{
            SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
        lhs = EnsureType(lhs, result_type, DataType::Bool());
        rhs = EnsureType(rhs, result_type, DataType::Bool());
        return builder_->Binary(NewValueName(), op, ArithmeticFlavor::kBool,
                                lhs, rhs, result_type, DataType::Bool());
      }
      SunMMIOType result_type = MakeTileType(DataType::Bool(), result_shape);
      return builder_->Binary(NewValueName(), op, ArithmeticFlavor::kBool, lhs,
                              rhs, result_type, DataType::Bool());
    };
    lower_bool_expr_to_shape =
        [&](const PrimExpr &bool_expr, const std::vector<int64_t> &target_shape,
            std::optional<DataType> mask_index_dtype) -> SunMMIOValue {
      MarkVisitedExprRoot(bool_expr);
      PrimExpr bool_expr_to_lower = rewrite_mask_condition(bool_expr);
      std::optional<DataType> integer_preferred =
          canonical_integer_preferred_dtype(mask_index_dtype);
      if (auto compare = try_emit_compare_to_shape(
              bool_expr_to_lower, target_shape, integer_preferred)) {
        return compare.value();
      }
      if (const auto *and_op = bool_expr_to_lower.as<AndNode>()) {
        SunMMIOValue lhs = lower_bool_expr_to_shape(and_op->a, target_shape,
                                                    integer_preferred);
        SunMMIOValue rhs = lower_bool_expr_to_shape(and_op->b, target_shape,
                                                    integer_preferred);
        return emit_logical_values(BinaryOp::kAnd, lhs, rhs, target_shape,
                                   and_op->a, and_op->b);
      }
      if (const auto *or_op = bool_expr_to_lower.as<OrNode>()) {
        SunMMIOValue lhs =
            lower_bool_expr_to_shape(or_op->a, target_shape, integer_preferred);
        SunMMIOValue rhs =
            lower_bool_expr_to_shape(or_op->b, target_shape, integer_preferred);
        return emit_logical_values(BinaryOp::kOr, lhs, rhs, target_shape,
                                   or_op->a, or_op->b);
      }
      SunMMIOValue value =
          lower_expr(bool_expr_to_lower, state, integer_preferred);
      return adapt_logical_tile_operand(value, target_shape,
                                        bool_expr_to_lower);
    };
    auto emit_logical_binary = [&](BinaryOp op, const PrimExpr &lhs_expr,
                                   const PrimExpr &rhs_expr, DataType dtype) {
      auto result_shape =
          merge_optional_tile_shapes(infer_tile_expr_shape(lhs_expr, state),
                                     infer_tile_expr_shape(rhs_expr, state));
      if (result_shape.has_value()) {
        std::optional<DataType> integer_preferred =
            canonical_integer_preferred_dtype(preferred_dtype);
        SunMMIOValue lhs = lower_bool_expr_to_shape(
            lhs_expr, result_shape.value(), integer_preferred);
        SunMMIOValue rhs = lower_bool_expr_to_shape(
            rhs_expr, result_shape.value(), integer_preferred);
        return emit_logical_values(op, lhs, rhs, result_shape.value(), lhs_expr,
                                   rhs_expr);
      }

      std::optional<DataType> integer_preferred =
          canonical_integer_preferred_dtype(preferred_dtype);
      SunMMIOValue lhs = lower_expr(lhs_expr, state, integer_preferred);
      SunMMIOValue rhs = lower_expr(rhs_expr, state, integer_preferred);
      if (IsScalarLike(lhs) && IsScalarLike(rhs)) {
        SunMMIOType result_type{SunMMIOType::Kind::kScalar,
                                CanonicalizeSuvmDType(dtype).with_lanes(1),
                                1,
                                {}};
        lhs = EnsureType(lhs, result_type, result_type.dtype);
        rhs = EnsureType(rhs, result_type, result_type.dtype);
        return builder_->Binary(NewValueName(), op,
                                arithmetic_flavor_for_dtype(result_type.dtype),
                                lhs, rhs, result_type, result_type.dtype);
      }
      std::vector<int64_t> fallback_shape = tile_result_shape(lhs, rhs);
      return emit_logical_values(op, lhs, rhs, fallback_shape, lhs_expr,
                                 rhs_expr);
    };
    auto emit_unary = [&](TileUnaryOp op, const PrimExpr &arg, DataType dtype,
                          bool force_f32 = false) {
      bool supports_mixed_precision =
          !force_f32 && supports_mixed_precision_unary(op);
      SunMMIOValue data =
          lower_expr(arg, state,
                     force_f32 ? std::optional<DataType>(DataType::Float(32))
                               : std::nullopt);
      if (!IsTileLike(data)) {
        UnsupportedExpr(
            expr.get(),
            "Clean v4 tiles lowering currently only supports tile-valued unary "
            "math inside T.Tiles");
      }
      DataType result_dtype =
          force_f32 ? DataType::Float(32)
                    : (supports_mixed_precision
                           ? choose_result_dtype(dtype, preferred_dtype)
                           : CanonicalizeSuvmDType(dtype).with_lanes(1));
      if ((!supports_mixed_precision || force_f32) &&
          data.dtype != result_dtype) {
        SunMMIOType f32_type =
            MakeTileType(result_dtype, ExtractStaticShape(data.type));
        data = builder_->Cast(NewValueName(), data, f32_type, result_dtype);
      }
      SunMMIOType result_type =
          MakeTileType(result_dtype, ExtractStaticShape(data.type));
      return builder_->Unary(NewValueName(), op, data, result_type,
                             result_dtype);
    };
    if (const auto *add = expr.as<AddNode>()) {
      return emit_binary(BinaryOp::kAdd, add->a, add->b, add->dtype);
    }
    if (const auto *sub = expr.as<SubNode>()) {
      const auto *zero = sub->a.as<FloatImmNode>();
      if (zero && zero->value == 0.0) {
        return emit_unary(TileUnaryOp::kNeg, sub->b, sub->dtype);
      }
      return emit_binary(BinaryOp::kSub, sub->a, sub->b, sub->dtype);
    }
    if (const auto *mul = expr.as<MulNode>()) {
      return emit_binary(BinaryOp::kMul, mul->a, mul->b, mul->dtype);
    }
    if (const auto *div = expr.as<DivNode>()) {
      return emit_binary(BinaryOp::kDiv, div->a, div->b, div->dtype);
    }
    if (const auto *div = expr.as<FloorDivNode>()) {
      return emit_binary(BinaryOp::kDiv, div->a, div->b, div->dtype);
    }
    if (const auto *mod = expr.as<ModNode>()) {
      return emit_binary(BinaryOp::kMod, mod->a, mod->b, mod->dtype);
    }
    if (const auto *mod = expr.as<FloorModNode>()) {
      return emit_binary(BinaryOp::kMod, mod->a, mod->b, mod->dtype);
    }
    if (const auto *min = expr.as<MinNode>()) {
      return emit_binary(BinaryOp::kMin, min->a, min->b, min->dtype);
    }
    if (const auto *max = expr.as<MaxNode>()) {
      return emit_binary(BinaryOp::kMax, max->a, max->b, max->dtype);
    }
    if (const auto *and_op = expr.as<AndNode>()) {
      return emit_logical_binary(BinaryOp::kAnd, and_op->a, and_op->b,
                                 and_op->dtype);
    }
    if (const auto *or_op = expr.as<OrNode>()) {
      return emit_logical_binary(BinaryOp::kOr, or_op->a, or_op->b,
                                 or_op->dtype);
    }
    if (const auto *eq = expr.as<EQNode>()) {
      return emit_compare(CompareOp::kEQ, eq->a, eq->b);
    }
    if (const auto *ne = expr.as<NENode>()) {
      return emit_compare(CompareOp::kNE, ne->a, ne->b);
    }
    if (const auto *lt = expr.as<LTNode>()) {
      return emit_compare(CompareOp::kLT, lt->a, lt->b);
    }
    if (const auto *le = expr.as<LENode>()) {
      return emit_compare(CompareOp::kLE, le->a, le->b);
    }
    if (const auto *gt = expr.as<GTNode>()) {
      return emit_compare(CompareOp::kGT, gt->a, gt->b);
    }
    if (const auto *ge = expr.as<GENode>()) {
      return emit_compare(CompareOp::kGE, ge->a, ge->b);
    }
    if (const auto *select = expr.as<SelectNode>()) {
      return emit_select(select->condition, select->true_value,
                         select->false_value, select->dtype);
    }
    if (const auto *cast = expr.as<CastNode>()) {
      // An explicit cast applies after its operand has been evaluated.
      SunMMIOValue value = lower_expr(cast->value, state, std::nullopt);
      if (IsTileLike(value)) {
        DataType dst_dtype = CanonicalizeSuvmDType(cast->dtype).with_lanes(1);
        if (value.dtype == dst_dtype) {
          return value;
        }
        SunMMIOType dst_type = MakeTileType(CanonicalizeSuvmDType(cast->dtype),
                                            ExtractStaticShape(value.type));
        return builder_->Cast(NewValueName(), value, dst_type,
                              CanonicalizeSuvmDType(cast->dtype).with_lanes(1));
      }
      SunMMIOType scalar_type{SunMMIOType::Kind::kScalar,
                              CanonicalizeSuvmDType(cast->dtype),
                              1,
                              {}};
      return builder_->Cast(NewValueName(), value, scalar_type,
                            CanonicalizeSuvmDType(cast->dtype).with_lanes(1));
    }
    if (const auto *call = expr.as<CallNode>()) {
      const auto *op_node = call->op.as<OpNode>();
      if (op_node && op_node->name == "tl.infinity") {
        ICHECK_EQ(call->args.size(), 1U)
            << "tl.infinity expects one dtype argument";
        const auto *dtype_arg = call->args[0].as<StringImmNode>();
        ICHECK(dtype_arg) << "tl.infinity dtype must be StringImm";
        MarkVisitedExprRoot(call->args[0]);
        DataType dtype = CanonicalizeSuvmDType(call->dtype).with_lanes(1);
        SunMMIOType scalar_type{SunMMIOType::Kind::kScalar, dtype, 1, {}};
        return builder_->ConstantFloat(NewValueName(), "inf", scalar_type,
                                       dtype);
      }
      if (op_node && call->args.size() >= 3 &&
          op_node->name == "tir.if_then_else") {
        return emit_select(call->args[0], call->args[1], call->args[2],
                           call->dtype);
      }
      if (op_node && call->args.size() == 1 && op_node->name == "tir.exp") {
        return emit_unary(TileUnaryOp::kExp, call->args[0], call->dtype);
      }
      if (op_node && call->args.size() == 1 && op_node->name == "tir.exp2") {
        PrimExpr scaled_arg = call->args[0] * FloatImm(call->args[0].dtype(),
                                                       0.69314718055994530942);
        return emit_unary(TileUnaryOp::kExp, scaled_arg, call->dtype, true);
      }
      if (op_node && call->args.size() == 1 &&
          (op_node->name == "tir.fabs" || op_node->name == "tir.abs")) {
        return emit_unary(TileUnaryOp::kAbs, call->args[0], call->dtype);
      }
      if (op_node && call->args.size() == 1 && op_node->name == "tir.ceil") {
        return emit_unary(TileUnaryOp::kCeil, call->args[0], call->dtype, true);
      }
      if (op_node && call->args.size() == 1 && op_node->name == "tir.floor") {
        return emit_unary(TileUnaryOp::kFloor, call->args[0], call->dtype,
                          true);
      }
      if (op_node && call->args.size() == 1 && op_node->name == "tir.log") {
        return emit_unary(TileUnaryOp::kLn, call->args[0], call->dtype);
      }
      if (op_node && call->args.size() == 1 && op_node->name == "tir.log2") {
        SunMMIOValue ln_value =
            emit_unary(TileUnaryOp::kLn, call->args[0], call->dtype);
        DataType result_dtype = ln_value.dtype;
        SunMMIOType scale_type{SunMMIOType::Kind::kScalar, result_dtype, 1, {}};
        SunMMIOValue log2e = builder_->ConstantFloat(
            NewValueName(), "1.4426950408889634", scale_type, result_dtype);
        SunMMIOType result_type =
            MakeTileType(result_dtype, ExtractStaticShape(ln_value.type));
        return builder_->Binary(NewValueName(), BinaryOp::kMul,
                                ArithmeticFlavor::kFloat, ln_value, log2e,
                                result_type, result_dtype);
      }
      if (op_node && call->args.size() == 1 && op_node->name == "tir.round") {
        return emit_unary(TileUnaryOp::kRound, call->args[0], call->dtype,
                          true);
      }
      if (op_node && call->args.size() == 1 && op_node->name == "tir.rsqrt") {
        return emit_unary(TileUnaryOp::kRsqrt, call->args[0], call->dtype);
      }
      if (op_node && call->args.size() == 1 && op_node->name == "tir.trunc") {
        return emit_unary(TileUnaryOp::kTrunc, call->args[0], call->dtype,
                          true);
      }
      if (op_node && call->args.size() == 2 && op_node->name == "tir.fmod") {
        return emit_binary(BinaryOp::kMod, call->args[0], call->args[1],
                           call->dtype);
      }
      if (op_node && op_node->name == "tl.ieee_frcp") {
        ICHECK_EQ(call->args.size(), 2U)
            << "tl.ieee_frcp expects value and rounding mode";
        const auto *rounding_mode = call->args[1].as<StringImmNode>();
        ICHECK(rounding_mode) << "tl.ieee_frcp rounding mode must be StringImm";
        MarkVisitedExprRoot(call->args[1]);
        return emit_unary(TileUnaryOp::kRecip, call->args[0], call->dtype);
      }
      if (op_node && call->args.size() == 1 &&
          op_node->name == "tl.ieee_frsqrt") {
        return emit_unary(TileUnaryOp::kRsqrt, call->args[0], call->dtype);
      }
    }
    UnsupportedExpr(expr.get(),
                    "Clean v4 tiles lowering currently supports only "
                    "BufferLoad/add/sub/mul/div/mod/min/max/compare/select/"
                    "cast/constants and selected unary math calls");
  };

  lower_stmt = [&](const Stmt &stmt, TileBlockState *state) {
    MarkVisitedNodeType(stmt->GetTypeKey());
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        lower_stmt(s, state);
      }
      return;
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      MarkVisitedExprRoot(loop->min);
      MarkVisitedExprRoot(loop->extent);
      if (loop->step.has_value()) {
        MarkVisitedExprRoot(loop->step.value());
      }
      auto axis = GetInteriorAxisAnnotation(loop);
      if (!axis.has_value()) {
        UnsupportedStmt(loop,
                        "Clean v4 tiles lowering only supports tile.interior "
                        "loops inside T.Tiles bodies");
      }
      TileBlockState loop_state = *state;
      if (axis.value() == 0) {
        loop_state.interior_axis0_loop = loop;
        loop_state.interior_axis1_loop = nullptr;
      } else if (axis.value() == 1) {
        loop_state.interior_axis1_loop = loop;
      } else {
        UnsupportedStmt(loop,
                        "Clean v4 tiles lowering currently supports up to 2D "
                        "interior loops");
      }
      lower_stmt(loop->body, &loop_state);
      state->tile_view_cache = loop_state.tile_view_cache;
      state->current_tile_values = loop_state.current_tile_values;
      state->register_tile_values = loop_state.register_tile_values;
      state->register_unsqueeze_axes = loop_state.register_unsqueeze_axes;
      state->local_tile_values = loop_state.local_tile_values;
      state->local_unit_tile_axes = loop_state.local_unit_tile_axes;
      return;
    }
    if (const auto *ifs = stmt.as<IfThenElseNode>()) {
      SunMMIOType bool_ty{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
      SunMMIOValue cond =
          EnsureType(EvalExpr(ifs->condition), bool_ty, DataType::Bool());
      auto saved_cache = state->current_tile_values;
      auto saved_registers = state->register_tile_values;
      auto saved_locals = state->local_tile_values;
      auto saved_local_axes = state->local_unit_tile_axes;

      builder_->BeginIf(cond, std::vector<SunMMIOValue>{});
      TileBlockState then_state = *state;
      lower_stmt(ifs->then_case, &then_state);
      if (ifs->else_case.defined()) {
        builder_->BeginElse();
        TileBlockState else_state = *state;
        lower_stmt(ifs->else_case.value(), &else_state);
      }
      builder_->EndIf();

      state->current_tile_values = saved_cache;
      state->register_tile_values = saved_registers;
      state->local_tile_values = saved_locals;
      state->local_unit_tile_axes = saved_local_axes;
      return;
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      SunMMIOValue value = lower_expr(let->value, state, std::nullopt);
      TileBlockState let_state = *state;
      let_state.let_values[let->var.get()] = value;
      if (auto axis = infer_tile_expr_unsqueeze_axis(let->value, state)) {
        let_state.let_unsqueeze_axes[let->var.get()] = axis.value();
      }
      lower_stmt(let->body, &let_state);
      state->tile_view_cache = let_state.tile_view_cache;
      state->current_tile_values = let_state.current_tile_values;
      state->register_tile_values = let_state.register_tile_values;
      state->register_unsqueeze_axes = let_state.register_unsqueeze_axes;
      state->local_tile_values = let_state.local_tile_values;
      state->local_unit_tile_axes = let_state.local_unit_tile_axes;
      return;
    }
    if (const auto *alloc = stmt.as<AllocateNode>()) {
      auto buffer_it = buffer_data_to_buffer_.find(alloc->buffer_var.get());
      if (buffer_it != buffer_data_to_buffer_.end() &&
          IsSunmmioReduceRegisterTempBuffer(buffer_it->second)) {
        EnterScope();
        lower_reduce_stmt(alloc->body, state);
        ExitScope();
        return;
      }
      UnsupportedStmt(alloc,
                      "T.Tiles hybrid lowering only supports Allocate for "
                      "reduce register temporaries");
    }
    if (const auto *decl = stmt.as<DeclBufferNode>()) {
      if (IsSunmmioReduceRegisterTempBuffer(decl->buffer)) {
        EnterScope();
        lower_reduce_stmt(decl->body, state);
        ExitScope();
        return;
      }
      UnsupportedStmt(decl,
                      "T.Tiles hybrid lowering only supports DeclBuffer for "
                      "reduce register temporaries");
    }
    if (const auto *store = stmt.as<BufferStoreNode>()) {
      if (IsSunmmioLocalVarBuffer(store->buffer)) {
        EmitLocalVarStore(
            store->buffer, store->indices,
            lower_expr(store->value, state, store->buffer->dtype));
        return;
      }
      auto local_it = state->local_tile_values.find(store->buffer.get());
      if (local_it != state->local_tile_values.end()) {
        SunMMIOType local_type = local_it->second.type;
        SunMMIOValue rhs = lower_expr(store->value, state, local_type.dtype);
        if (!IsTileLike(rhs)) {
          SunMMIOType scalar_type{
              SunMMIOType::Kind::kScalar, local_type.dtype, 1, {}};
          rhs = EnsureType(rhs, scalar_type, local_type.dtype);
          rhs = builder_->TileFill(NewValueName(), rhs, local_type,
                                   local_type.dtype);
        } else if (!StaticShapesEqual(rhs.type, local_type) ||
                   rhs.dtype != local_type.dtype) {
          rhs =
              builder_->Cast(NewValueName(), rhs, local_type, local_type.dtype);
        }
        state->local_tile_values[store->buffer.get()] =
            builder_->BindValueAlias(make_local_value_name(store->buffer), rhs);
        return;
      }
      auto reg_ty_it = state->register_tile_types.find(store->buffer.get());
      if (reg_ty_it != state->register_tile_types.end()) {
        SunMMIOType reg_type = reg_ty_it->second;
        SunMMIOValue rhs = lower_expr(store->value, state, reg_type.dtype);
        if (!IsTileLike(rhs)) {
          SunMMIOType scalar_type{
              SunMMIOType::Kind::kScalar, reg_type.dtype, 1, {}};
          rhs = EnsureType(rhs, scalar_type, reg_type.dtype);
          rhs =
              builder_->TileFill(NewValueName(), rhs, reg_type, reg_type.dtype);
        } else {
          std::vector<int64_t> rhs_shape = ExtractStaticShape(rhs.type);
          std::vector<int64_t> reg_shape = ExtractStaticShape(reg_type);
          if (rhs_shape != reg_shape &&
              rhs_shape.size() == reg_shape.size() + 1) {
            for (int64_t axis = 0;
                 axis < static_cast<int64_t>(rhs_shape.size()); ++axis) {
              if (rhs_shape[axis] != 1) {
                continue;
              }
              std::vector<int64_t> squeezed_shape = rhs_shape;
              squeezed_shape.erase(squeezed_shape.begin() + axis);
              if (squeezed_shape == reg_shape) {
                rhs = checked_tile_squeeze(rhs, reg_type, axis, reg_type.dtype,
                                           "reduce register tile store");
                rhs_shape = reg_shape;
                break;
              }
            }
          }
          ICHECK(rhs_shape == reg_shape)
              << "Reduce register tile store cannot normalize RHS shape";
          if (rhs.dtype != reg_type.dtype) {
            rhs = builder_->Cast(NewValueName(), rhs, reg_type, reg_type.dtype);
          }
        }
        state->register_tile_values[store->buffer.get()] =
            builder_->BindValueAlias(make_register_value_name(store->buffer),
                                     rhs);
        return;
      }
      std::optional<int64_t> forced_unit_axis =
          find_local_unit_axis_in_expr(store->value, state);
      TileAccessInfo natural_access =
          analyze_access(store->buffer, store->indices, state);
      bool use_forced_unit_axis = forced_unit_axis.has_value() &&
                                  !natural_access.requires_aligned_1d_load;
      bool canonical_tail_store_for_rhs =
          store->predicate.defined() &&
          (is_canonical_tail_load_predicate(store->predicate.value(), state,
                                            natural_access) ||
           (!scope.is_reduce_scope && natural_access.tile_rank == 2 &&
            natural_access.tile_shape == scope.tile_shape && [&]() {
              std::array<bool, 2> axes = match_canonical_rank2_predicate_axes(
                  store->predicate.value(), state);
              return axes[0] || axes[1];
            }()));
      std::optional<int64_t> saved_forced_axis;
      bool had_saved_forced_axis = false;
      if (use_forced_unit_axis) {
        // Lower the target load/store in the same 2D unit-tile shape as the
        // local in-tile reduce result, avoiding fake 1D load/squeeze/store.
        auto saved_it = state->local_unit_tile_axes.find(store->buffer.get());
        if (saved_it != state->local_unit_tile_axes.end()) {
          saved_forced_axis = saved_it->second;
          had_saved_forced_axis = true;
        }
        state->local_unit_tile_axes[store->buffer.get()] = *forced_unit_axis;
      }
      TileAccessInfo access =
          use_forced_unit_axis
              ? analyze_access(store->buffer, store->indices, state)
              : natural_access;
      std::optional<PrimExpr> saved_tail_store_predicate =
          state->active_tail_store_predicate;
      if ((state->tile_mask.has_value() || canonical_tail_store_for_rhs) &&
          store->predicate.defined()) {
        state->active_tail_store_predicate = store->predicate.value();
      }
      SunMMIOValue raw_rhs =
          lower_expr(store->value, state,
                     CanonicalizeSuvmDType(store->buffer->dtype).with_lanes(1));
      state->active_tail_store_predicate = saved_tail_store_predicate;
      SunMMIOValue rhs = access.requires_aligned_1d_load
                             ? normalize_for_aligned_1d_store(access, raw_rhs)
                             : normalize_for_store(access, raw_rhs);
      std::string cache_key = make_tile_cache_key(access);
      std::optional<int64_t> mx_scale_valid_elems =
          mx_scale_e8m0_prefix_store_extent(store, access);
      std::optional<DataType> store_mask_index_dtype;
      auto get_store_mask_index_dtype = [&]() -> DataType {
        if (!store_mask_index_dtype.has_value()) {
          store_mask_index_dtype =
              mask_index_dtype_for_value_dtype(store->buffer->dtype);
        }
        return store_mask_index_dtype.value();
      };
      std::optional<SunMMIOValue> mask =
          (access.tile_rank == 2) ? state->tile_mask : std::nullopt;
      bool canonical_tail_store = store->predicate.defined() &&
                                  is_canonical_tail_load_predicate(
                                      store->predicate.value(), state, access);
      if (!mx_scale_valid_elems.has_value() &&
          !access.requires_aligned_1d_load && store->predicate.defined() &&
          !mask.has_value()) {
        if (canonical_tail_store && access.tile_rank == 1) {
          mask =
              build_canonical_tail_mask(access, get_store_mask_index_dtype());
        } else if (auto canonical_mask = build_canonical_rank2_predicate_mask(
                       store->predicate.value(), state, access,
                       get_store_mask_index_dtype())) {
          mask = canonical_mask.value();
        } else {
          mask = lower_expr(store->predicate.value(), state,
                            get_store_mask_index_dtype());
        }
      }
      if (!mx_scale_valid_elems.has_value() &&
          access.requires_aligned_1d_load && store->predicate.defined() &&
          !canonical_tail_store) {
        mask = lower_expr(store->predicate.value(), state,
                          get_store_mask_index_dtype());
      }
      std::optional<SunMMIOValue> dst_view;
      if ((mx_scale_valid_elems.has_value() || mask.has_value()) &&
          !access.requires_aligned_1d_load) {
        dst_view = get_or_create_tile_view(access, state);
        SunMMIOType dst_tile_type =
            MakeTileType(store->buffer->dtype, access.tile_shape);
        SunMMIOValue old_tile = builder_->TileLoad(
            NewValueName(), dst_view.value(), dst_tile_type, std::nullopt,
            std::nullopt,
            CanonicalizeSuvmDType(store->buffer->dtype).with_lanes(1));
        if (mx_scale_valid_elems.has_value()) {
          std::vector<SunMMIOValue> offsets{make_index_const(0)};
          SunMMIOType slice_type =
              MakeTileType(store->buffer->dtype, {*mx_scale_valid_elems});
          SunMMIOValue valid_slice = builder_->TileSlice(
              NewValueName(), rhs, offsets, slice_type,
              CanonicalizeSuvmDType(store->buffer->dtype).with_lanes(1));
          rhs = builder_->TileInsertSlice(
              NewValueName(), old_tile, valid_slice, offsets, dst_tile_type,
              CanonicalizeSuvmDType(store->buffer->dtype).with_lanes(1));
        } else {
          SunMMIOValue store_mask = mask.value();
          if (IsTileLike(store_mask)) {
            store_mask =
                reorient_unit_tile_to_shape(store_mask, access.tile_shape);
            if (ExtractStaticShape(store_mask.type) != access.tile_shape) {
              store_mask =
                  broadcast_tile_to_shape(store_mask, access.tile_shape);
            }
            ICHECK(StaticShapesEqual(
                store_mask.type,
                MakeTileType(DataType::Bool(), access.tile_shape)))
                << "Predicated tile store cannot normalize mask shape";
          } else {
            SunMMIOType bool_scalar_type{
                SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
            store_mask =
                EnsureType(store_mask, bool_scalar_type, DataType::Bool());
            store_mask = builder_->TileFill(
                NewValueName(), store_mask,
                MakeTileType(DataType::Bool(), access.tile_shape),
                DataType::Bool());
          }
          rhs = builder_->TileSelect(
              NewValueName(), store_mask, rhs, old_tile, dst_tile_type,
              CanonicalizeSuvmDType(store->buffer->dtype).with_lanes(1));
        }
      }
      std::optional<SunMMIOValue> updated_aligned_tile;
      erase_current_values_for_buffer(state, store->buffer.get());
      if (access.requires_aligned_1d_load) {
        if (!mask.has_value() && canonical_tail_store) {
          mask =
              build_canonical_tail_mask(access, get_store_mask_index_dtype());
        }
        updated_aligned_tile = store_aligned_1d_tile(access, rhs, mask, state);
      } else {
        if (!dst_view.has_value()) {
          dst_view = get_or_create_tile_view(access, state);
        }
        builder_->TileStore(rhs, dst_view.value(), std::nullopt);
      }
      if (access.promoted_unit_tile_view) {
        erase_current_values_for_buffer(state, store->buffer.get());
      } else if (updated_aligned_tile.has_value()) {
        Aligned1DAddressInfo aligned_address = compute_aligned_1d_address(
            access, LookupBuffer(access.buffer).buffer_type);
        std::string aligned_cache_key =
            make_tile_cache_key(access, aligned_address);
        state->current_tile_values[aligned_cache_key] =
            updated_aligned_tile.value();
      } else {
        state->current_tile_values[cache_key] = builder_->BindValueAlias(
            make_current_value_name(store->buffer, cache_key), rhs);
      }
      if (use_forced_unit_axis) {
        if (had_saved_forced_axis) {
          state->local_unit_tile_axes[store->buffer.get()] = *saved_forced_axis;
        } else {
          state->local_unit_tile_axes.erase(store->buffer.get());
        }
      }
      return;
    }
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (const auto *call = eval->value.as<CallNode>()) {
        const auto *op_node = call->op.as<OpNode>();
        if (op_node && op_node->name == "tl.vector_core_in_tile_reduce") {
          lower_vector_core_in_tile_reduce(call, state);
          return;
        }
        if (op_node && (op_node->name == "tl.barrier_init" ||
                        op_node->name == "tl.barrier_arrive_and_wait" ||
                        op_node->name == "tl.sunmmio_sync")) {
          (void)EvalExpr(eval->value);
          return;
        }
        std::string op_name =
            op_node ? std::string(op_node->name) : std::string("<non-op-call>");
        UnsupportedStmt(eval, "Clean v4 tiles lowering does not support "
                              "Evaluate call op `" +
                                  op_name + "` inside T.Tiles");
      }
    }
    UnsupportedStmt(stmt.get(),
                    "Clean v4 tiles lowering currently supports only "
                    "SeqStmt/token Evaluate/BufferStore");
  };

  lower_vector_core_in_tile_reduce = [&](const CallNode *call,
                                         TileBlockState *state) {
    MarkVisitedExprRoot(ffi::GetRef<PrimExpr>(call));
    ICHECK_EQ(call->args.size(), 4U)
        << "tl.vector_core_in_tile_reduce expects predicate, dst region, src "
           "region, and axis";
    const auto *predicate = call->args[0].as<StringImmNode>();
    ICHECK(predicate)
        << "tl.vector_core_in_tile_reduce predicate must be StringImm";
    MarkVisitedNodeType(predicate->GetTypeKey());
    BufferRegion dst_region = NormalizeRegionTracked(call->args[1]);
    BufferRegion src_region = NormalizeRegionTracked(call->args[2]);
    const auto *axis_imm = call->args[3].as<IntImmNode>();
    ICHECK(axis_imm) << "tl.vector_core_in_tile_reduce axis must be IntImm";
    MarkVisitedNodeType(axis_imm->GetTypeKey());
    int64_t axis = static_cast<int64_t>(axis_imm->value);
    note_register_unsqueeze_axis(state, dst_region->buffer, axis);

    SunMMIOValue src_tile;
    std::vector<int64_t> src_shape;
    auto reg_src_it =
        state->register_tile_values.find(src_region->buffer.get());
    if (reg_src_it != state->register_tile_values.end()) {
      src_tile = reg_src_it->second;
      src_shape = ExtractStaticShape(src_tile.type);
    } else {
      SunMMIOValue src_view = make_tile_view_from_region(src_region, state);
      src_shape = ExtractStaticShape(src_view.type);
      SunMMIOType src_tile_type =
          MakeTileType(src_region->buffer->dtype, src_shape);
      src_tile = builder_->TileLoad(
          NewValueName(), src_view, src_tile_type, std::nullopt, std::nullopt,
          CanonicalizeSuvmDType(src_region->buffer->dtype).with_lanes(1));
    }

    std::vector<int64_t> result_shape = src_shape;
    ICHECK_GE(axis, 0);
    ICHECK_LT(axis, static_cast<int64_t>(result_shape.size()));
    result_shape[axis] = 1;
    SunMMIOType result_tile_type =
        MakeTileType(src_region->buffer->dtype, result_shape);
    SunMMIOValue reduced = builder_->TileReduce(
        NewValueName(), static_cast<std::string>(predicate->value), src_tile,
        result_tile_type, axis,
        CanonicalizeSuvmDType(src_region->buffer->dtype).with_lanes(1));

    if (IsSunmmioReduceLocalTempBuffer(dst_region->buffer)) {
      SunMMIOValue local = builder_->BindValueAlias(
          make_local_value_name(dst_region->buffer), reduced);
      state->local_tile_values[dst_region->buffer.get()] = local;
      state->local_unit_tile_axes[dst_region->buffer.get()] = axis;
      return;
    }

    bool dst_is_register =
        IsSunmmioReduceRegisterTempBuffer(dst_region->buffer) &&
        state->register_tile_types.count(dst_region->buffer.get());
    if (dst_is_register) {
      SunMMIOType dst_tile_type =
          state->register_tile_types.at(dst_region->buffer.get());
      SunMMIOValue rhs = reduced;
      std::vector<int64_t> rhs_shape = ExtractStaticShape(rhs.type);
      std::vector<int64_t> dst_shape = ExtractStaticShape(dst_tile_type);
      if (rhs_shape.size() == dst_shape.size() + 1) {
        for (int64_t axis_to_squeeze = 0;
             axis_to_squeeze < static_cast<int64_t>(rhs_shape.size());
             ++axis_to_squeeze) {
          if (rhs_shape[axis_to_squeeze] != 1) {
            continue;
          }
          std::vector<int64_t> squeezed_shape = rhs_shape;
          squeezed_shape.erase(squeezed_shape.begin() + axis_to_squeeze);
          if (squeezed_shape == dst_shape) {
            rhs = checked_tile_squeeze(
                rhs, dst_tile_type, axis_to_squeeze,
                CanonicalizeSuvmDType(dst_region->buffer->dtype).with_lanes(1),
                "vector_core_in_tile_reduce register destination");
            break;
          }
        }
      }
      if (!StaticShapesEqual(rhs.type, dst_tile_type) ||
          rhs.dtype !=
              CanonicalizeSuvmDType(dst_region->buffer->dtype).with_lanes(1)) {
        rhs = builder_->Cast(
            NewValueName(), rhs, dst_tile_type,
            CanonicalizeSuvmDType(dst_region->buffer->dtype).with_lanes(1));
      }
      state->register_tile_values[dst_region->buffer.get()] =
          builder_->BindValueAlias(make_register_value_name(dst_region->buffer),
                                   rhs);
    } else {
      SunMMIOType dst_tile_type = make_tile_type_from_region(dst_region);
      std::vector<int64_t> dst_shape = ExtractStaticShape(dst_tile_type);
      std::optional<std::vector<int>> dst_tile_axes;
      if (dst_shape.size() == 1 && result_shape.size() > 1) {
        std::vector<int> kept_axes;
        for (int64_t axis_idx = 0;
             axis_idx < static_cast<int64_t>(result_shape.size()); ++axis_idx) {
          if (result_shape[axis_idx] != 1) {
            kept_axes.push_back(static_cast<int>(axis_idx));
          }
        }
        if (kept_axes.size() == 1 &&
            dst_shape[0] == result_shape[static_cast<size_t>(kept_axes[0])]) {
          dst_tile_axes = kept_axes;
        }
      }
      TileAccessInfo dst_access =
          make_tile_access_from_region(dst_region, dst_tile_axes);
      SunMMIOValue rhs =
          dst_access.requires_aligned_1d_load
              ? normalize_for_aligned_1d_store(dst_access, reduced)
              : normalize_for_store(dst_access, reduced);
      erase_current_values_for_buffer(state, dst_region->buffer.get());
      if (dst_access.requires_aligned_1d_load) {
        Aligned1DAddressInfo aligned_address = compute_aligned_1d_address(
            dst_access, LookupBuffer(dst_access.buffer).buffer_type);
        std::string dst_cache_key =
            make_tile_cache_key(dst_access, aligned_address);
        state->current_tile_values[dst_cache_key] =
            store_aligned_1d_tile(dst_access, rhs, std::nullopt, state);
      } else {
        SunMMIOValue dst_view = get_or_create_tile_view(dst_access, state);
        builder_->TileStore(rhs, dst_view, std::nullopt);
        std::string dst_cache_key = make_tile_cache_key(dst_access);
        state->current_tile_values[dst_cache_key] = builder_->BindValueAlias(
            make_current_value_name(dst_region->buffer, dst_cache_key), rhs);
      }
    }
  };

  lower_reduce_stmt = [&](const Stmt &stmt, TileBlockState *state) {
    MarkVisitedNodeType(stmt->GetTypeKey());
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        lower_reduce_stmt(s, state);
      }
      return;
    }
    if (const auto *ifs = stmt.as<IfThenElseNode>()) {
      SunMMIOType bool_ty{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
      SunMMIOValue cond =
          EnsureType(EvalExpr(ifs->condition), bool_ty, DataType::Bool());
      auto saved_cache = state->current_tile_values;
      auto saved_registers = state->register_tile_values;
      auto saved_locals = state->local_tile_values;
      auto saved_local_axes = state->local_unit_tile_axes;
      std::vector<SunMMIOValue> live_out_values =
          collect_tile_live_out_values(state);
      builder_->BeginIf(cond, live_out_values);
      TileBlockState then_state = *state;
      then_state.current_tile_values = saved_cache;
      then_state.register_tile_values = saved_registers;
      then_state.local_tile_values = saved_locals;
      then_state.local_unit_tile_axes = saved_local_axes;
      lower_reduce_stmt(ifs->then_case, &then_state);
      if (ifs->else_case.defined()) {
        builder_->BeginElse();
        TileBlockState else_state = *state;
        else_state.current_tile_values = saved_cache;
        else_state.register_tile_values = saved_registers;
        else_state.local_tile_values = saved_locals;
        else_state.local_unit_tile_axes = saved_local_axes;
        lower_reduce_stmt(ifs->else_case.value(), &else_state);
      }
      builder_->EndIf();
      state->current_tile_values = saved_cache;
      state->register_tile_values = then_state.register_tile_values;
      state->local_tile_values = saved_locals;
      state->local_unit_tile_axes = saved_local_axes;
      return;
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      SunMMIOValue value = lower_expr(let->value, state, std::nullopt);
      TileBlockState let_state = *state;
      let_state.let_values[let->var.get()] = value;
      if (auto axis = infer_tile_expr_unsqueeze_axis(let->value, state)) {
        let_state.let_unsqueeze_axes[let->var.get()] = axis.value();
      }
      lower_reduce_stmt(let->body, &let_state);
      state->tile_view_cache = let_state.tile_view_cache;
      state->current_tile_values = let_state.current_tile_values;
      state->register_tile_values = let_state.register_tile_values;
      state->register_unsqueeze_axes = let_state.register_unsqueeze_axes;
      state->local_tile_values = let_state.local_tile_values;
      state->local_unit_tile_axes = let_state.local_unit_tile_axes;
      return;
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      MarkVisitedExprRoot(loop->min);
      MarkVisitedExprRoot(loop->extent);
      if (loop->step.has_value()) {
        MarkVisitedExprRoot(loop->step.value());
      }
      auto axis = GetInteriorAxisAnnotation(loop);
      if (!axis.has_value()) {
        std::optional<TilesScopeInfo> saved_scope;
        if (loop->annotations.count(tl::attr::kTileDomain)) {
          saved_scope = scope;
          scope = TilesScopeInfo{};
          scope.root = loop;
          scope.domain_shape = Downcast<ffi::Array<PrimExpr>>(
              loop->annotations.at(tl::attr::kTileDomain));
          {
            std::vector<int64_t> parsed_axes = ParseStaticIntArray(
                loop->annotations, tl::attr::tile_execution_domain_axes);
            scope.execution_domain_axes.reserve(parsed_axes.size());
            for (int64_t axis : parsed_axes) {
              scope.execution_domain_axes.push_back(static_cast<int>(axis));
            }
          }
          scope.tile_shape =
              ParseStaticIntArray(loop->annotations, tl::attr::tile_tile_size);
          ICHECK_EQ(scope.execution_domain_axes.size(), scope.tile_shape.size())
              << "Nested tile.execution_domain_axes and tile.tile_size rank "
                 "mismatch";

          std::vector<const ForNode *> chain = CollectLinearForChain(loop);
          ICHECK_GE(chain.size(), scope.domain_shape.size())
              << "Nested Tiles scope loop chain shorter than tile.domain rank";
          for (size_t i = 0; i < scope.domain_shape.size(); ++i) {
            scope.domain_loops.push_back(chain[i]);
          }
          scope.execution_loops.assign(scope.execution_domain_axes.size(),
                                       nullptr);
          for (const ForNode *domain_loop : scope.domain_loops) {
            auto axis_it =
                domain_loop->annotations.find(tl::attr::tile_execution_axis);
            if (axis_it == domain_loop->annotations.end()) {
              continue;
            }
            int exec_axis = Downcast<Integer>((*axis_it).second)->value;
            ICHECK_GE(exec_axis, 0);
            ICHECK_LT(static_cast<size_t>(exec_axis),
                      scope.execution_loops.size())
                << "Nested tile.execution_axis is out of range";
            scope.execution_loops[static_cast<size_t>(exec_axis)] = domain_loop;
          }
          for (const ForNode *exec_loop : scope.execution_loops) {
            ICHECK(exec_loop != nullptr)
                << "Nested Tiles scope is missing an execution loop for one "
                   "tile axis";
          }
          scope.tile_block_body = scope.domain_loops.back()->body;
          scope.is_reduce_scope = IsReduceLikeTileBody(scope.tile_block_body);
          auto loops = FindInteriorLoops(scope.tile_block_body);
          scope.interior_axis0_loop = loops.first;
          scope.interior_axis1_loop = loops.second;
        } else if (auto exec_axis = GetExecutionAxisAnnotation(loop);
                   exec_axis.has_value()) {
          auto loops = FindInteriorLoops(loop->body);
          if (loops.first != nullptr && loops.second != nullptr) {
            auto axis0_extent = GetStaticLoopExtent(loops.first);
            auto axis1_extent = GetStaticLoopExtent(loops.second);
            ICHECK(axis0_extent.has_value() && axis1_extent.has_value())
                << "Hybrid local tile fragment expects static interior "
                   "extents";
            saved_scope = scope;
            TilesScopeInfo parent_scope = scope;
            scope = TilesScopeInfo{};
            scope.root = loop;
            scope.interior_axis0_loop = loops.first;
            scope.interior_axis1_loop = loops.second;
            if (!parent_scope.execution_loops.empty() &&
                exec_axis.value() >= 0 &&
                static_cast<size_t>(exec_axis.value()) <
                    parent_scope.execution_loops.size() &&
                parent_scope.execution_loops[0] != nullptr &&
                parent_scope.domain_loops.size() + 1 ==
                    parent_scope.domain_shape.size()) {
              scope.domain_shape = parent_scope.domain_shape;
              scope.domain_values = parent_scope.domain_values;
              scope.domain_loops = parent_scope.domain_loops;
              scope.domain_loops.push_back(loop);
              scope.execution_loops = parent_scope.execution_loops;
              scope.execution_loops[static_cast<size_t>(exec_axis.value())] =
                  loop;
              scope.execution_domain_axes = parent_scope.execution_domain_axes;
              scope.tile_shape = parent_scope.tile_shape;
            } else {
              ICHECK_EQ(exec_axis.value(), 1)
                  << "Hybrid local tile fragment without a parent execution "
                     "prefix is only supported for execution axis 1";
              scope.domain_shape = ffi::Array<PrimExpr>{
                  Integer(axis0_extent.value()),
                  loop->extent * Integer(axis1_extent.value())};
              scope.domain_loops = {loop};
              scope.execution_loops = {nullptr, loop};
              scope.execution_domain_axes = {0, 1};
              scope.tile_shape = {axis0_extent.value(), axis1_extent.value()};
            }
            scope.tile_block_body = loop->body;
            scope.is_reduce_scope = IsReduceLikeTileBody(scope.tile_block_body);
          }
        }
        if (saved_scope.has_value() && scope.domain_values.empty()) {
          materialize_domain_values(&scope);
        }
        SunMMIOValue min = EnsureIndex(EvalExpr(loop->min));
        std::optional<SunMMIOValue> materialized_extent =
            materialized_loop_extent(loop);
        SunMMIOValue extent = materialized_extent.has_value()
                                  ? materialized_extent.value()
                                  : EnsureIndex(EvalExpr(loop->extent));
        SunMMIOValue step = EmitConstIndex(1);
        SunMMIOValue upper = builder_->Binary(
            NewValueName(), BinaryOp::kAdd, ArithmeticFlavor::kIndex, min,
            extent,
            SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
            DataType::Int(32));
        std::vector<SunMMIOValue> live_out_values =
            collect_tile_live_out_values(state);
        std::string iv = "%" + loop->loop_var->name_hint;
        builder_->BeginFor(iv, min, upper, step, loop->annotations,
                           live_out_values);
        EnterScope();
        BindVar(loop->loop_var,
                SunMMIOValue{
                    loop->loop_var.dtype(), iv,
                    SunMMIOType{
                        SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}}});
        TileBlockState loop_state = *state;
        for (const auto &kv : state->register_tile_types) {
          const BufferNode *buffer = kv.first;
          auto value_it = state->register_tile_values.find(buffer);
          ICHECK(value_it != state->register_tile_values.end());
          loop_state.register_tile_values[buffer] = builder_->BindValueAlias(
              value_it->second.value,
              SunMMIOValue{value_it->second.dtype, value_it->second.value,
                           kv.second});
        }
        auto saved_locals = loop_state.local_tile_values;
        auto saved_local_axes = loop_state.local_unit_tile_axes;
        if (!saved_scope.has_value()) {
          loop_state.local_tile_values.clear();
          loop_state.local_unit_tile_axes.clear();
        }
        lower_reduce_stmt(loop->body, &loop_state);
        if (!saved_scope.has_value()) {
          loop_state.local_tile_values = saved_locals;
          loop_state.local_unit_tile_axes = saved_local_axes;
        }
        ExitScope();
        builder_->EndFor();
        state->register_tile_values = loop_state.register_tile_values;
        state->current_tile_values = loop_state.current_tile_values;
        if (saved_scope.has_value()) {
          state->local_tile_values = loop_state.local_tile_values;
          state->local_unit_tile_axes = loop_state.local_unit_tile_axes;
          scope = saved_scope.value();
        }
        return;
      }
      TileBlockState loop_state = *state;
      if (axis.value() == 0) {
        loop_state.interior_axis0_loop = loop;
        loop_state.interior_axis1_loop = nullptr;
      } else if (axis.value() == 1) {
        loop_state.interior_axis1_loop = loop;
      } else {
        UnsupportedStmt(loop,
                        "Reduce tiles lowering currently supports up to 2D "
                        "interior loops");
      }
      lower_reduce_stmt(loop->body, &loop_state);
      state->tile_view_cache = loop_state.tile_view_cache;
      state->current_tile_values = loop_state.current_tile_values;
      state->register_tile_values = loop_state.register_tile_values;
      state->register_unsqueeze_axes = loop_state.register_unsqueeze_axes;
      state->local_tile_values = loop_state.local_tile_values;
      state->local_unit_tile_axes = loop_state.local_unit_tile_axes;
      return;
    }
    if (const auto *alloc = stmt.as<AllocateNode>()) {
      auto buffer_it = buffer_data_to_buffer_.find(alloc->buffer_var.get());
      if (buffer_it != buffer_data_to_buffer_.end() &&
          IsSunmmioReduceRegisterTempBuffer(buffer_it->second)) {
        EnterScope();
        lower_reduce_stmt(alloc->body, state);
        ExitScope();
        return;
      }
      UnsupportedStmt(alloc,
                      "T.Tiles hybrid lowering only supports Allocate for "
                      "reduce register temporaries");
    }
    if (const auto *decl = stmt.as<DeclBufferNode>()) {
      if (IsSunmmioReduceRegisterTempBuffer(decl->buffer)) {
        EnterScope();
        lower_reduce_stmt(decl->body, state);
        ExitScope();
        return;
      }
      UnsupportedStmt(decl,
                      "T.Tiles hybrid lowering only supports DeclBuffer for "
                      "reduce register temporaries");
    }
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (const auto *call = eval->value.as<CallNode>()) {
        const auto *op_node = call->op.as<OpNode>();
        if (op_node && op_node->name == "tl.vector_core_in_tile_reduce") {
          lower_vector_core_in_tile_reduce(call, state);
          return;
        }
      }
    }
    lower_stmt(stmt, state);
  };

  auto infer_tail_mask_index_dtype = [&](const Stmt &body) {
    auto record_dtype = [&](std::optional<DataType> *slot, DataType value_dtype,
                            const char *context) {
      DataType candidate = mask_index_dtype_for_value_dtype(value_dtype);
      if (slot->has_value() && slot->value() != candidate) {
        LOG(FATAL) << "Tail mask lowering currently requires one mask index "
                      "dtype per tail scope; saw conflicting dtype in "
                   << context;
      }
      *slot = candidate;
    };

    std::optional<DataType> store_mask_index_dtype;
    std::optional<DataType> fallback_mask_index_dtype;
    tir::PostOrderVisit(body, [&](const ObjectRef &obj) {
      if (const auto *store = obj.as<BufferStoreNode>()) {
        if (store->predicate.defined()) {
          record_dtype(&store_mask_index_dtype, store->buffer->dtype,
                       "predicated BufferStore");
        }
        return;
      }
      if (const auto *load = obj.as<BufferLoadNode>()) {
        if (load->predicate.defined()) {
          record_dtype(&fallback_mask_index_dtype, load->buffer->dtype,
                       "predicated BufferLoad");
        }
        return;
      }
      if (const auto *select = obj.as<SelectNode>()) {
        record_dtype(&fallback_mask_index_dtype, select->dtype, "tir.Select");
        return;
      }
      if (const auto *call = obj.as<CallNode>()) {
        const auto *op_node = call->op.as<OpNode>();
        if (op_node && call->args.size() >= 3 &&
            op_node->name == "tir.if_then_else") {
          record_dtype(&fallback_mask_index_dtype, call->dtype,
                       "tir.if_then_else");
        }
      }
    });
    if (store_mask_index_dtype.has_value()) {
      return store_mask_index_dtype.value();
    }
    if (fallback_mask_index_dtype.has_value()) {
      return fallback_mask_index_dtype.value();
    }
    return DataType::Int(32);
  };

  auto emit_tile_stmt = [&](TileBlockState *state) {
    if (has_partial_execution_prefix) {
      lower_reduce_stmt(scope.tile_block_body, state);
      return;
    }
    if (scope.tail_predicate.defined() && scope.full_tile_body.defined() &&
        scope.tail_tile_body.defined()) {
      auto lower_tail_with_mask = [&](const SunMMIOValue &mask) {
        TileBlockState tail_state = *state;
        tail_state.tile_mask = mask;
        tail_state.interior_axis0_loop = scope.tail_interior_axis0_loop;
        tail_state.interior_axis1_loop = scope.tail_interior_axis1_loop;
        lower_stmt(scope.tail_tile_block_body, &tail_state);
      };

      SunMMIOValue cond = build_full_tile_condition();
      builder_->BeginIf(cond, std::vector<SunMMIOValue>{});
      TileBlockState full_state = *state;
      full_state.tile_mask.reset();
      full_state.interior_axis0_loop = scope.interior_axis0_loop;
      full_state.interior_axis1_loop = scope.interior_axis1_loop;
      if (scope.is_reduce_scope || !state->register_tile_values.empty()) {
        lower_reduce_stmt(scope.full_tile_block_body, &full_state);
      } else {
        lower_stmt(scope.full_tile_block_body, &full_state);
      }
      builder_->BeginElse();
      DataType tail_mask_index_dtype =
          infer_tail_mask_index_dtype(scope.tail_tile_block_body);
      if (scope.tile_shape.size() == 1) {
        ICHECK_EQ(scope.execution_loops.size(), 1U);
        ICHECK_EQ(scope.execution_domain_axes.size(), 1U);
        SunMMIOValue tile_extent = make_index_const(scope.tile_shape[0]);
        SunMMIOValue exec_index =
            EnsureIndex(EvalExpr(scope.execution_loops[0]->loop_var));
        SunMMIOValue valid_lanes = min_index(
            tile_extent, sub_index(domain_value(scope.execution_domain_axes[0]),
                                   mul_index(exec_index, tile_extent)));
        SunMMIOType mask_type =
            MakeTileType(DataType::Bool(), scope.tile_shape);
        SunMMIOValue mask = builder_->TileAxisMask(
            NewValueName(), 0, valid_lanes, mask_type, tail_mask_index_dtype);
        lower_tail_with_mask(mask);
        builder_->EndIf();
        return;
      }
      ICHECK_EQ(scope.tile_shape.size(), 2U)
          << "Tail tile lowering supports rank-1 or rank-2 tile scopes";
      TailMaskInfo mask_info = build_tail_mask_info(state);
      builder_->BeginIf(mask_info.row_tail_cond, std::vector<SunMMIOValue>{});
      builder_->BeginIf(mask_info.col_tail_cond, std::vector<SunMMIOValue>{});
      SunMMIOValue rect_mask = builder_->TileRectMask(
          NewValueName(), mask_info.valid_rows, mask_info.valid_cols,
          mask_info.mask_type, tail_mask_index_dtype);
      lower_tail_with_mask(rect_mask);
      builder_->BeginElse();
      SunMMIOValue row_mask =
          builder_->TileAxisMask(NewValueName(), 0, mask_info.valid_rows,
                                 mask_info.mask_type, tail_mask_index_dtype);
      lower_tail_with_mask(row_mask);
      builder_->EndIf();
      builder_->BeginElse();
      SunMMIOValue col_mask =
          builder_->TileAxisMask(NewValueName(), 1, mask_info.valid_cols,
                                 mask_info.mask_type, tail_mask_index_dtype);
      lower_tail_with_mask(col_mask);
      builder_->EndIf();
      builder_->EndIf();
      return;
    }
    if (scope.is_reduce_scope || !state->register_tile_values.empty()) {
      lower_reduce_stmt(scope.tile_block_body, state);
      return;
    }
    lower_stmt(scope.tile_block_body, state);
  };

  std::function<void(size_t, TileBlockState *)> emit_loop_nest;
  emit_loop_nest = [&](size_t loop_index, TileBlockState *state) {
    if (loop_index == scope.domain_loops.size()) {
      emit_tile_stmt(state);
      return;
    }
    const ForNode *loop = scope.domain_loops[loop_index];
    MarkVisitedNodeType(loop->GetTypeKey());
    SunMMIOValue min = EnsureIndex(EvalExpr(loop->min));
    SunMMIOValue extent = materialized_loop_extent(loop).value();
    SunMMIOValue step = EmitConstIndex(1);
    SunMMIOValue upper = builder_->Binary(
        NewValueName(), BinaryOp::kAdd, ArithmeticFlavor::kIndex, min, extent,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
    std::string iv = "%" + loop->loop_var->name_hint;
    std::vector<SunMMIOValue> live_out_values =
        collect_tile_live_out_values(state);
    if (!live_out_values.empty()) {
      builder_->BeginFor(iv, min, upper, step, loop->annotations,
                         live_out_values);
    } else {
      builder_->BeginFor(iv, min, upper, step, loop->annotations,
                         std::vector<SunMMIOValue>{});
    }
    EnterScope();
    BindVar(
        loop->loop_var,
        SunMMIOValue{
            loop->loop_var.dtype(), iv,
            SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}}});
    emit_loop_nest(loop_index + 1, state);
    ExitScope();
    builder_->EndFor();
  };

  TileBlockState state;
  state.scope = &scope;
  state.mlir_ctx = mlir_ctx;
  state.interior_axis0_loop = scope.interior_axis0_loop;
  state.interior_axis1_loop = scope.interior_axis1_loop;
  // Snapshot dynamic domains before any loop so bounds and masks share SSA.
  materialize_domain_values(&scope);
  discover_reduce_register_temps(tile_scope_stmt, &state);
  if (!state.register_tile_types.empty()) {
    initialize_reduce_register_temps(&state);
  }
  emit_loop_nest(0, &state);
  return true;
}

} // namespace codegen
} // namespace tvm
