#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "op/prim.h"
#include "transform/common/data_type_rewriter.h"
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <array>
#include <sstream>

#include "codegen_metal.h"

namespace tvm::tl::codegen_impl {
namespace {

class MetalFunctionCollector : public tvm::tl::StmtExprVisitor {
public:
  std::vector<Call> cooperative_calls;
  std::unordered_map<std::string, PrimExpr> thread_extents;
  bool has_allocation{false};
  bool has_loop_jump{false};

  ffi::Optional<VisitInterrupt> Visit_(const CallNode *node) final {
    if (const auto *operation = node->op.as<OpNode>();
        operation &&
        operation->name.starts_with("tirx.metal.cooperative_tensor_")) {
      cooperative_calls.push_back(ffi::GetRef<Call>(node));
    }
    return tvm::tl::StmtExprVisitor::Visit_(node);
  }

  ffi::Optional<VisitInterrupt> Visit_(const BindNode *node) final {
    if (const auto *call = node->value.as<CallNode>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      has_allocation = true;
    }
    return tvm::tl::StmtExprVisitor::Visit_(node);
  }

  ffi::Optional<VisitInterrupt> Visit_(const BreakNode *node) final {
    has_loop_jump = true;
    return tvm::tl::StmtExprVisitor::Visit_(node);
  }

  ffi::Optional<VisitInterrupt> Visit_(const ContinueNode *node) final {
    has_loop_jump = true;
    return tvm::tl::StmtExprVisitor::Visit_(node);
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtNode *node) final {
    if (node->op.same_as(tirx::launch_thread_op())) {
      std::string tag = node->args[0].as_or_throw<StringImm>()->value;
      if (tag.starts_with("threadIdx."))
        thread_extents.insert_or_assign(tag,
                                        node->args[1].as_or_throw<PrimExpr>());
    }
    return tvm::tl::StmtExprVisitor::Visit_(node);
  }
};

} // namespace

int64_t CodeGenMetal::ConstantIndex(const Expr &value) {
  PrimExpr simplified =
      tl::arith::Simplify(analyzer_, value.as_or_throw<PrimExpr>());
  const auto *constant = simplified.as<IntImmNode>();
  TVM_FFI_CHECK(constant, ValueError)
      << "Metal matrix dimensions and fragment indices must be "
         "constant after unrolling, but got "
      << simplified;
  return static_cast<int64_t>(constant->value);
}

void CodeGenMetal::AnalyzeMetalFunction(const Function &function) {
  cooperative_tensors_.clear();
  cooperative_accumulators_.clear();
  cooperative_inputs_.clear();
  block_indices_[0] = std::nullopt;
  block_indices_[1] = std::nullopt;
  active_mlx_panel_ = 0;
  needs_fragment_lane_vars_ = false;
  auto collector = ffi::make_object<MetalFunctionCollector>();
  collector->Visit(function->body);
  uses_cooperative_tensors_ = !collector->cooperative_calls.empty();
  total_threads_ = collector->thread_extents.empty() ? 0 : 1;
  for (const auto &[tag, extent] : collector->thread_extents) {
    if (const auto *constant = extent.as<IntImmNode>();
        constant && constant->value > 0) {
      total_threads_ *= static_cast<int64_t>(constant->value);
    } else {
      total_threads_ = 0;
      break;
    }
  }
  for (const Call &call : collector->cooperative_calls) {
    std::string name = call->op.as_or_throw<Op>()->name;
    if (name.ends_with("_load") || name.ends_with("_store"))
      needs_fragment_lane_vars_ = true;
    if (name.ends_with("_multiply_accumulate")) {
      cooperative_inputs_.insert(GetMatrixBufferVar(call->args[2]).get());
      cooperative_inputs_.insert(GetMatrixBufferVar(call->args[4]).get());
      if (ConstantIndex(call->args[8]) == 16 &&
          ConstantIndex(call->args[9]) == 32 &&
          ConstantIndex(call->args[10]) == 16 &&
          ConstantIndex(call->args[11]) == 0 &&
          ConstantIndex(call->args[12]) == 0) {
        cooperative_accumulators_.insert_or_assign(
            GetMatrixBufferVar(call->args[0]).get(), call);
        cooperative_accumulators_.insert_or_assign(
            GetMatrixBufferVar(call->args[6]).get(), call);
      }
    }
  }
  for (const VarNode *input : cooperative_inputs_)
    cooperative_accumulators_.erase(input);
  if (uses_cooperative_tensors_ && !emitted_cooperative_header_) {
    header_stream_
        << "#include "
           "<MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
    emitted_cooperative_header_ = true;
  }
}

CodeGenMetal::CooperativeTensorState &
CodeGenMetal::GetCooperativeTensor(const Var &buffer) {
  auto &state = cooperative_tensors_[buffer.get()];
  if (!state.dtype.empty())
    return state;
  PrimType type = PrimType::Void();
  if (const auto *tensor = buffer->ty.as<TensorTypeNode>()) {
    type = tensor->dtype;
  } else if (const auto *pointer = buffer->ty.as<PtrTypeNode>()) {
    type = pointer->element_type.as_or_throw<PrimType>();
  }
  TVM_FFI_CHECK(type == PrimType::Float(16) || type == PrimType::Float(32) ||
                    type == PrimType::BFloat(16),
                TypeError)
      << "Metal cooperative tensors require float16, bfloat16, or float32 "
         "elements";
  std::ostringstream dtype;
  PrintType(type, dtype);
  state.dtype = dtype.str();
  return state;
}

void CodeGenMetal::AllocateCooperativeTensor(const Var &buffer,
                                             size_t element_count) {
  TVM_FFI_CHECK(element_count > 0 && element_count % 64 == 0, ValueError)
      << "Metal cooperative tensor storage must contain a positive multiple of "
         "64 elements";
  auto &state = GetCooperativeTensor(buffer);
  std::string storage = GetVarID(buffer.get());
  auto candidate = cooperative_accumulators_.find(buffer.get());
  if (state.dtype != "float" || candidate == cooperative_accumulators_.end() ||
      element_count % 512 != 0 || element_count > 2048) {
    stream << "thread " << state.dtype << " " << storage << '['
           << element_count / 32 << "];\n";
    return;
  }
  std::string left_dtype =
      GetCooperativeTensor(GetMatrixBufferVar(candidate->second->args[2]))
          .dtype;
  std::string right_dtype =
      GetCooperativeTensor(GetMatrixBufferVar(candidate->second->args[4]))
          .dtype;
  std::string descriptor = name_supply_->FreshName(storage + "_descriptor");
  state.operation = name_supply_->FreshName(storage + "_matmul");
  stream
      << "constexpr auto " << descriptor
      << " = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, "
         "true, "
         "mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);\n";
  PrintIndent();
  stream << "mpp::tensor_ops::matmul2d<" << descriptor
         << ", metal::execution_simdgroup> " << state.operation << ";\n";
  for (size_t tile = 0; tile < element_count / 512; ++tile) {
    std::string name =
        name_supply_->FreshName(storage + "_tile_" + std::to_string(tile));
    state.tiles.push_back(name);
    PrintIndent();
    stream << "auto " << name << " = " << state.operation
           << ".get_destination_cooperative_tensor<decltype(" << state.operation
           << ".get_left_input_cooperative_tensor<" << left_dtype << ", "
           << right_dtype << ", float>()), decltype(" << state.operation
           << ".get_right_input_cooperative_tensor<" << left_dtype << ", "
           << right_dtype << ", float>()), float>();\n";
  }
}

std::string CodeGenMetal::CooperativeElement(const Var &buffer,
                                             const PrimExpr &index,
                                             int elements_per_tile,
                                             int element) {
  auto &state = GetCooperativeTensor(buffer);
  if (state.tiles.empty()) {
    return GetVarID(buffer.get()) + "[(" + PrintExpr(index) + ") * " +
           std::to_string(elements_per_tile) + " + " + std::to_string(element) +
           "]";
  }
  int64_t offset = ConstantIndex(index) * elements_per_tile + element;
  TVM_FFI_CHECK(offset >= 0 &&
                    offset < static_cast<int64_t>(state.tiles.size()) * 16,
                ValueError)
      << "Metal cooperative tensor fragment access is out of bounds";
  return state.tiles[offset / 16] + "[" + std::to_string(offset % 16) + "]";
}

void CodeGenMetal::PrintCooperativeFill(const CallNode *node,
                                        std::ostream &output) {
  Var buffer = GetMatrixBufferVar(node->args[0]);
  PrimExpr index = node->args[1].as_or_throw<PrimExpr>();
  std::string value = PrintExpr(node->args[2]);
  int64_t rows = ConstantIndex(node->args[3]);
  int64_t columns = ConstantIndex(node->args[4]);
  TVM_FFI_CHECK(rows > 0 && columns > 0 && rows * columns % 32 == 0, ValueError)
      << "Metal cooperative tensor fill requires complete SIMD-group fragments";
  int elements = rows * columns / 32;
  auto &state = GetCooperativeTensor(buffer);
  output << "{ ";
  int chunk = state.tiles.empty() ? elements : 16;
  TVM_FFI_CHECK_EQ(elements % chunk, 0, ValueError);
  for (int offset = 0; offset < elements; offset += chunk) {
    std::string first = CooperativeElement(buffer, index, elements, offset);
    output << "_Pragma(\"clang loop unroll(full)\") for (ushort __element = 0; "
              "__element < "
           << chunk << "; ++__element) ((thread " << state.dtype << "*)(&"
           << first << "))[__element] = " << value << "; ";
  }
  output << "}";
}

void CodeGenMetal::PrintCooperativeCopy(const CallNode *node, bool store,
                                        std::ostream &output) {
  Var buffer = GetMatrixBufferVar(node->args[0]);
  PrimExpr index = node->args[1].as_or_throw<PrimExpr>();
  std::string pointer = PrintExpr(node->args[2]);
  std::string stride = PrintExpr(node->args[3]);
  int64_t rows = ConstantIndex(node->args[4]);
  int64_t columns = ConstantIndex(node->args[5]);
  TVM_FFI_CHECK(rows > 0 && columns > 0 && rows % 16 == 0 && columns % 16 == 0,
                ValueError)
      << "Metal cooperative tensor copies require complete 16x16 fragments";
  auto pointer_type = node->args[2]->ty.as_or_throw<PtrType>();
  std::ostringstream address_space;
  PrintStorageScope(pointer_type->scope, address_space);
  std::ostringstream element_type;
  PrintType(pointer_type->element_type, element_type);
  std::string memory_type = element_type.str();
  std::string dtype = GetCooperativeTensor(buffer).dtype;
  std::string qualifier = (store ? "" : "const ") + address_space.str();
  std::string memory = store ? "__dst" : "__src";
  output << "{ " << qualifier << memory_type << "* " << memory << " = ("
         << qualifier << memory_type << "*)" << pointer << "; ";
  int elements = rows * columns / 32;
  int offset = 0;
  for (int row_fragment = 0; row_fragment < rows / 16; ++row_fragment) {
    for (int column_fragment = 0; column_fragment < columns / 16;
         ++column_fragment) {
      output << "{ ushort __row = __base_row + " << row_fragment * 16
             << "; ushort __column = __base_col + " << column_fragment * 16
             << "; ";
      for (int row_half = 0; row_half < 2; ++row_half) {
        std::string tensor_value =
            "*(thread " + dtype + "4*)(&" +
            CooperativeElement(buffer, index, elements, offset + row_half * 4) +
            ")";
        std::string memory_value =
            "*(" + qualifier + memory_type + "4*)(&" + memory + "[(__row + " +
            std::to_string(row_half * 8) + ") * (" + stride + ") + __column])";
        output << (store ? memory_value : tensor_value) << " = ";
        if (memory_type != dtype)
          output << (store ? memory_type : dtype) << "4(";
        output << (store ? tensor_value : memory_value);
        if (memory_type != dtype)
          output << ")";
        output << "; ";
      }
      output << "} ";
      offset += 8;
    }
  }
  output << "}";
}

void CodeGenMetal::PrintCooperativeMultiply(const CallNode *node,
                                            std::ostream &output) {
  int64_t rows = ConstantIndex(node->args[8]);
  int64_t columns = ConstantIndex(node->args[9]);
  int64_t reduction = ConstantIndex(node->args[10]);
  bool transpose_left = ConstantIndex(node->args[11]) != 0;
  bool transpose_right = ConstantIndex(node->args[12]) != 0;
  TVM_FFI_CHECK(rows > 0 && columns > 0 && reduction > 0 && rows % 16 == 0 &&
                    columns % 16 == 0 && reduction % 16 == 0 &&
                    (rows == 32 || columns == 32 || reduction == 32),
                ValueError)
      << "MPP matmul2d requires complete 16x16 fragments and at least one "
         "dimension of 32";
  Var destination = GetMatrixBufferVar(node->args[0]);
  Var left = GetMatrixBufferVar(node->args[2]);
  Var right = GetMatrixBufferVar(node->args[4]);
  Var accumulator = GetMatrixBufferVar(node->args[6]);
  std::string left_dtype = GetCooperativeTensor(left).dtype;
  std::string right_dtype = GetCooperativeTensor(right).dtype;
  std::string result_dtype = GetCooperativeTensor(destination).dtype;
  std::string accumulator_dtype = GetCooperativeTensor(accumulator).dtype;
  TVM_FFI_CHECK_EQ(result_dtype, accumulator_dtype, TypeError);
  auto &result_state = GetCooperativeTensor(destination);
  std::string operation =
      result_state.tiles.empty() ? "__operation" : result_state.operation;
  std::string result = "__ct_c";
  if (!result_state.tiles.empty()) {
    int64_t tile = ConstantIndex(node->args[1]);
    TVM_FFI_CHECK(tile >= 0 &&
                      static_cast<size_t>(tile) < result_state.tiles.size(),
                  ValueError)
        << "Metal cooperative tensor destination is out of bounds";
    result = result_state.tiles[tile];
  }
  output << "{ ";
  if (result_state.tiles.empty()) {
    output
        << "constexpr auto __descriptor = mpp::tensor_ops::matmul2d_descriptor("
        << rows << ", " << columns << ", " << reduction << ", "
        << (transpose_left ? "true" : "false") << ", "
        << (transpose_right ? "true" : "false")
        << ", true, "
           "mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate); "
           "mpp::tensor_ops::matmul2d<__descriptor, "
           "metal::execution_simdgroup> __operation; ";
  }
  output << "auto __ct_a = " << operation
         << ".get_left_input_cooperative_tensor<" << left_dtype << ", "
         << right_dtype << ", " << result_dtype
         << ">(); auto __ct_b = " << operation
         << ".get_right_input_cooperative_tensor<" << left_dtype << ", "
         << right_dtype << ", " << result_dtype << ">(); ";
  if (result_state.tiles.empty())
    output << "auto __ct_c = " << operation
           << ".get_destination_cooperative_tensor<decltype(__ct_a), "
              "decltype(__ct_b), "
           << result_dtype << ">(); ";
  auto copy_input = [&](const Var &buffer, const Expr &index, int elements,
                        const std::string &tensor, const std::string &dtype) {
    std::string first =
        CooperativeElement(buffer, index.as_or_throw<PrimExpr>(), elements, 0);
    output << "_Pragma(\"clang loop unroll(full)\") for (ushort __element = 0; "
              "__element < "
           << elements << "; ++__element) " << tensor
           << "[__element] = ((thread const " << dtype << "*)(&" << first
           << "))[__element]; ";
  };
  copy_input(left, node->args[3], rows * reduction / 32, "__ct_a", left_dtype);
  copy_input(right, node->args[5], reduction * columns / 32, "__ct_b",
             right_dtype);
  bool inplace = !result_state.tiles.empty() &&
                 destination.same_as(accumulator) &&
                 ConstantIndex(node->args[1]) == ConstantIndex(node->args[7]);
  if (!inplace)
    copy_input(accumulator, node->args[7], rows * columns / 32, result,
               accumulator_dtype);
  output << operation << ".run(__ct_a, __ct_b, " << result << "); ";
  if (result_state.tiles.empty()) {
    std::string first =
        CooperativeElement(destination, node->args[1].as_or_throw<PrimExpr>(),
                           rows * columns / 32, 0);
    output << "_Pragma(\"clang loop unroll(full)\") for (ushort __element = 0; "
              "__element < "
           << rows * columns / 32 << "; ++__element) ((thread " << result_dtype
           << "*)(&" << first << "))[__element] = __ct_c[__element]; ";
  }
  output << "}";
}

void CodeGenMetal::Dispatch_(const ForNode *node) {
  const auto *begin = node->begin.as<IntImmNode>();
  const auto *end = node->end.as<IntImmNode>();
  const auto *step = node->step.as<IntImmNode>();
  int64_t count = 0;
  if (begin && end && step && step->value == 1)
    count = static_cast<int64_t>(end->value - begin->value);
  if (uses_cooperative_tensors_ && count > 0 && count <= 4) {
    auto collector = ffi::make_object<MetalFunctionCollector>();
    collector->Visit(node->body);
    if (!collector->has_allocation && !collector->has_loop_jump &&
        !collector->cooperative_calls.empty()) {
      auto saved_ids = var_idmap_;
      auto saved_types = handle_data_type_;
      auto saved_scopes = alloc_storage_scope_;
      auto saved_analyzer = analyzer_->Clone();
      for (int64_t iteration = 0; iteration < count; ++iteration) {
        PrimExpr value =
            tl::MakeConst(node->var.ty(), begin->value + iteration);
        Stmt body = tl::SubstituteWithDataTypeLegalization(
            node->body, [&](const Var &variable) -> ffi::Optional<PrimExpr> {
              if (variable.same_as(node->var))
                return value;
              return std::nullopt;
            });
        PrintIndent();
        stream << "{\n";
        int scope = BeginScope();
        PrintStmt(body);
        EndScope(scope);
        PrintIndent();
        stream << "}\n";
        var_idmap_ = saved_ids;
        handle_data_type_ = saved_types;
        alloc_storage_scope_ = saved_scopes;
        analyzer_ = saved_analyzer->Clone();
      }
      return;
    }
  }
  CodeGenC::Dispatch_(node);
}

bool CodeGenMetal::TryPrintMlxCoordinate(const PrimExpr &value,
                                         std::ostream &output) {
  if (active_mlx_panel_ <= 0 || !block_indices_[0] || !block_indices_[1] ||
      !value.ty().IsScalar() || !value.ty().MatchesCode(kDLInt, kDLUInt))
    return false;
  auto strip_cast = [](PrimExpr expression) {
    while (const auto *cast = expression.as<prim::CastNode>()) {
      PrimType destination = cast->ty.as_or_throw<PrimType>();
      PrimType source = cast->value.ty();
      if (!destination.IsScalar() || !source.IsScalar() ||
          !destination.MatchesCode(kDLInt, kDLUInt) ||
          !source.MatchesCode(kDLInt, kDLUInt) ||
          destination.bits() < source.bits())
        break;
      expression = cast->value;
    }
    return expression;
  };
  auto is_block = [&](const PrimExpr &expression, size_t axis) {
    return strip_cast(expression).same_as(block_indices_[axis].value());
  };
  auto is_constant = [&](const PrimExpr &expression, int64_t expected) {
    auto constant = strip_cast(expression).as<IntImmNode>();
    return constant && constant->value == expected;
  };
  int panel_shift = 0;
  while ((int64_t{1} << panel_shift) < active_mlx_panel_)
    ++panel_shift;
  auto is_logical_x = [&](const PrimExpr &expression) {
    PrimExpr base = strip_cast(expression);
    if (const auto *shift = base.as<prim::RShiftNode>())
      return is_block(shift->a, 0) && is_constant(shift->b, panel_shift);
    if (const auto *division = base.as<prim::FloorDivNode>())
      return is_block(division->a, 0) &&
             is_constant(division->b, active_mlx_panel_);
    if (const auto *division = base.as<prim::DivNode>())
      return is_block(division->a, 0) &&
             is_constant(division->b, active_mlx_panel_);
    return active_mlx_panel_ == 1 && is_block(base, 0);
  };
  auto is_remainder = [&](const PrimExpr &expression) {
    PrimExpr base = strip_cast(expression);
    if (const auto *mask = base.as<prim::BitwiseAndNode>())
      return (is_block(mask->a, 0) &&
              is_constant(mask->b, active_mlx_panel_ - 1)) ||
             (is_block(mask->b, 0) &&
              is_constant(mask->a, active_mlx_panel_ - 1));
    if (const auto *modulo = base.as<prim::FloorModNode>())
      return is_block(modulo->a, 0) &&
             is_constant(modulo->b, active_mlx_panel_);
    if (const auto *modulo = base.as<prim::ModNode>())
      return is_block(modulo->a, 0) &&
             is_constant(modulo->b, active_mlx_panel_);
    return false;
  };
  auto is_panel_row = [&](const PrimExpr &expression) {
    PrimExpr base = strip_cast(expression);
    if (const auto *shift = base.as<prim::LShiftNode>())
      return is_block(shift->a, 1) && is_constant(shift->b, panel_shift);
    if (const auto *product = base.as<prim::MulNode>())
      return (is_block(product->a, 1) &&
              is_constant(product->b, active_mlx_panel_)) ||
             (is_block(product->b, 1) &&
              is_constant(product->a, active_mlx_panel_));
    return active_mlx_panel_ == 1 && is_block(base, 1);
  };
  if (is_logical_x(value)) {
    output << CastFromTo("blockIdx.x", PrimType::UInt(32), value.ty());
    return true;
  }
  PrimExpr block_x = tl::CastValue(*block_indices_[0], value.ty());
  PrimExpr block_y = tl::CastValue(*block_indices_[1], value.ty());
  PrimExpr panel = tl::MakeConst(value.ty(), active_mlx_panel_);
  std::array<PrimExpr, 2> coordinates{prim::FloorDiv(block_x, panel),
                                      block_y * panel +
                                          prim::FloorMod(block_x, panel)};
  for (size_t axis = 0; axis < coordinates.size(); ++axis) {
    if (tl::arith::CanProveEqual(analyzer_, value, coordinates[axis])) {
      output << CastFromTo(axis == 0 ? "blockIdx.x" : "blockIdx.y",
                           PrimType::UInt(32), value.ty());
      return true;
    }
  }
  if (!value.as<prim::AddNode>())
    return false;
  struct Term {
    PrimExpr expression;
    int64_t row_coefficient{0};
    int64_t remainder_coefficient{0};
    bool used{false};
  };
  std::vector<Term> terms;
  std::function<void(const PrimExpr &)> collect =
      [&](const PrimExpr &expression) {
        if (const auto *addition = expression.as<prim::AddNode>()) {
          collect(addition->a);
          collect(addition->b);
        } else {
          terms.push_back(Term{expression});
        }
      };
  collect(value);
  for (auto &term : terms) {
    PrimExpr base = strip_cast(term.expression);
    int64_t coefficient = 1;
    if (const auto *product = base.as<prim::MulNode>()) {
      if (const auto *constant = product->b.as<IntImmNode>()) {
        coefficient = static_cast<int64_t>(constant->value);
        base = strip_cast(product->a);
      } else if (const auto *constant = product->a.as<IntImmNode>()) {
        coefficient = static_cast<int64_t>(constant->value);
        base = strip_cast(product->b);
      }
    }
    if (is_block(base, 1) && coefficient % active_mlx_panel_ == 0)
      term.row_coefficient = coefficient / active_mlx_panel_;
    if (is_panel_row(base))
      term.row_coefficient = coefficient;
    if (is_remainder(base))
      term.remainder_coefficient = coefficient;
  }
  std::vector<int64_t> coefficients;
  for (auto &row : terms) {
    if (!row.row_coefficient || row.used)
      continue;
    for (auto &remainder : terms) {
      if (&remainder != &row && !remainder.used &&
          remainder.remainder_coefficient == row.row_coefficient) {
        coefficients.push_back(row.row_coefficient);
        row.used = true;
        remainder.used = true;
        break;
      }
    }
  }
  if (coefficients.empty())
    return false;
  std::string coordinate =
      CastFromTo("blockIdx.y", PrimType::UInt(32), value.ty());
  output << '(';
  bool first = true;
  for (int64_t coefficient : coefficients) {
    if (!first)
      output << " + ";
    first = false;
    output << coordinate;
    if (coefficient != 1)
      output << " * " << coefficient;
  }
  for (const auto &term : terms) {
    if (!term.used)
      output << " + " << PrintExpr(term.expression);
  }
  output << ')';
  return true;
}

void CodeGenMetal::Dispatch_(const prim::AddNode *node, std::ostream &output) {
  if (!TryPrintMlxCoordinate(ffi::GetRef<PrimExpr>(node), output))
    CodeGenC::Dispatch_(node, output);
}

void CodeGenMetal::Dispatch_(const prim::MulNode *node, std::ostream &output) {
  if (!TryPrintMlxCoordinate(ffi::GetRef<PrimExpr>(node), output))
    CodeGenC::Dispatch_(node, output);
}

void CodeGenMetal::Dispatch_(const prim::DivNode *node, std::ostream &output) {
  if (!TryPrintMlxCoordinate(ffi::GetRef<PrimExpr>(node), output))
    CodeGenC::Dispatch_(node, output);
}

void CodeGenMetal::Dispatch_(const prim::FloorDivNode *node,
                             std::ostream &output) {
  if (!TryPrintMlxCoordinate(ffi::GetRef<PrimExpr>(node), output))
    CodeGenC::Dispatch_(node, output);
}

void CodeGenMetal::Dispatch_(const prim::RShiftNode *node,
                             std::ostream &output) {
  if (!TryPrintMlxCoordinate(ffi::GetRef<PrimExpr>(node), output))
    CodeGenC::Dispatch_(node, output);
}

void CodeGenMetal::Dispatch_(const prim::BitwiseAndNode *node,
                             std::ostream &output) {
  if (!TryPrintMlxCoordinate(ffi::GetRef<PrimExpr>(node), output))
    CodeGenC::Dispatch_(node, output);
}

void CodeGenMetal::Dispatch_(const RegionStmtNode *node) {
  auto annotation = node->attrs->dict.Get("metal.threadblock_swizzle_pattern");
  if (!annotation) {
    CodeGenC::Dispatch_(node);
    return;
  }
  auto pattern = annotation->cast<tvm::Tuple>();
  TVM_FFI_CHECK_EQ(pattern->fields.size(), 2, ValueError);
  std::string name = pattern->fields[0].as_or_throw<StringImm>()->value;
  int64_t panel = ConstantIndex(pattern->fields[1]);
  TVM_FFI_CHECK(panel > 0, ValueError)
      << "Metal swizzle panel size must be positive";
  TVM_FFI_CHECK(block_indices_[0] && block_indices_[1], ValueError)
      << "Metal block swizzling requires a two-dimensional grid";
  int saved_panel = active_mlx_panel_;
  auto saved_ids = var_idmap_;
  PrintIndent();
  stream << "{\n";
  int scope = BeginScope();
  PrintIndent();
  stream << "const uint3 __physical_blockIdx = uint3(blockIdx.x, blockIdx.y, "
         << (thread_work_dim_ == 3 ? "blockIdx.z" : "0u") << ");\n";
  PrintIndent();
  if (name == "rasterization2DMLX") {
    TVM_FFI_CHECK((panel & (panel - 1)) == 0, ValueError)
        << "Metal MLX swizzle panel size must be a power of two";
    int shift = 0;
    while ((int64_t{1} << shift) < panel)
      ++shift;
    stream << "const uint3 blockIdx = uint3(__physical_blockIdx.x >> " << shift
           << ", (__physical_blockIdx.y << " << shift
           << ") + (__physical_blockIdx.x & " << panel - 1
           << "u), __physical_blockIdx.z);\n";
    active_mlx_panel_ = panel;
    for (size_t axis = 0; axis < 2; ++axis)
      var_idmap_[block_indices_[axis].value().get()] = CastFromTo(
          axis == 0 ? "__physical_blockIdx.x" : "__physical_blockIdx.y",
          PrimType::UInt(32), block_indices_[axis].value().ty());
  } else {
    TVM_FFI_CHECK(name == "rasterization2DRow" ||
                      name == "rasterization2DColumn",
                  ValueError)
        << "Unknown Metal threadblock swizzle pattern " << name;
    bool column = name == "rasterization2DColumn";
    const char *major_extent = column ? "__gridDim.y" : "__gridDim.x";
    const char *minor_extent = column ? "__gridDim.x" : "__gridDim.y";
    stream << "const uint __linear = __physical_blockIdx.x + "
              "__physical_blockIdx.y * __gridDim.x;\n";
    PrintIndent();
    stream << "const uint __span = " << panel << "u * " << major_extent
           << ";\n";
    PrintIndent();
    stream << "const uint __panel = __linear / __span;\n";
    PrintIndent();
    stream << "const uint __offset = __linear % __span;\n";
    PrintIndent();
    stream << "const uint __stride = min(" << panel << "u, " << minor_extent
           << " - __panel * " << panel << "u);\n";
    PrintIndent();
    stream << "const uint __major = (__panel & 1u) ? " << major_extent
           << " - 1u - __offset / __stride : __offset / __stride;\n";
    PrintIndent();
    stream << "const uint __minor = __offset % __stride + __panel * " << panel
           << "u;\n";
    PrintIndent();
    stream << "const uint3 blockIdx = uint3("
           << (column ? "__minor, __major" : "__major, __minor")
           << ", __physical_blockIdx.z);\n";
  }
  PrintStmt(node->body);
  EndScope(scope);
  PrintIndent();
  stream << "}\n";
  active_mlx_panel_ = saved_panel;
  var_idmap_ = std::move(saved_ids);
}

} // namespace tvm::tl::codegen_impl
