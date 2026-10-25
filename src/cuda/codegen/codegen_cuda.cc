#include "codegen_cuda.h"

#include "backend/common/codegen/native_codegen.h"
#include "compat/expr.h"
#include "ptx.h"

#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <array>
#include <sstream>

namespace tvm::codegen {
namespace {
class ThreadblockSwizzleLowerer : public tirx::StmtExprMutator {
public:
  explicit ThreadblockSwizzleLowerer(const DictAttrs &attributes) {
    SetCluster(attributes);
  }

private:
  ffi::Optional<PrimVar> blocks_[2];
  PrimExpr extents_[2]{1, 1};
  int64_t cluster_[2]{1, 1};

  void SetCluster(const DictAttrs &attributes) {
    if (auto dimensions =
            attributes.GetAttr<ffi::Array<ffi::Any>>("cluster_dims")) {
      TVM_FFI_CHECK_EQ(dimensions->size(), 3, ValueError);
      for (size_t axis = 0; axis < 2; ++axis) {
        auto extent = tl::GetIntAnnotation(dimensions.value()[axis]);
        TVM_FFI_CHECK(extent && *extent > 0, ValueError)
            << "Swizzle cluster dimensions must be positive constants";
        cluster_[axis] = *extent;
      }
    }
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    auto saved_blocks = std::array{blocks_[0], blocks_[1]};
    auto saved_extents = std::array{extents_[0], extents_[1]};
    auto saved_cluster = std::array{cluster_[0], cluster_[1]};
    SetCluster(node->attrs);
    if (node->op.same_as(tirx::launch_thread_op())) {
      auto tag = node->args[0].as_or_throw<StringImm>()->value;
      for (size_t axis = 0; axis < 2; ++axis) {
        if (tag == (axis == 0 ? "blockIdx.x" : "blockIdx.y")) {
          blocks_[axis] = node->body_params[0].as_or_throw<PrimVar>();
          extents_[axis] = node->args[1].as_or_throw<PrimExpr>();
        }
      }
    }
    RegionStmt region = GetRef<RegionStmt>(node);
    if (auto annotation =
            node->attrs->dict.Get("threadblock_swizzle_pattern")) {
      auto pattern = annotation->cast<Tuple>();
      TVM_FFI_CHECK_EQ(pattern->fields.size(), 2, ValueError);
      auto name = pattern->fields[0].as_or_throw<StringImm>()->value;
      auto panel = tl::GetConstInt(pattern->fields[1]);
      TVM_FFI_CHECK(panel && *panel > 0, ValueError)
          << "Threadblock swizzle panel size must be a positive constant";
      TVM_FFI_CHECK(name == "rasterization2DRow" ||
                        name == "rasterization2DColumn",
                    ValueError)
          << "Unsupported CUDA threadblock swizzle: " << name;
      TVM_FFI_CHECK(blocks_[0] || blocks_[1], ValueError)
          << "Threadblock swizzle requires a CUDA block launch";
      bool column = name == "rasterization2DColumn";
      PrimExpr block_x = prim::cast(
          blocks_[0] ? PrimExpr(*blocks_[0]) : PrimExpr(0), PrimType::Int(64));
      PrimExpr block_y = prim::cast(
          blocks_[1] ? PrimExpr(*blocks_[1]) : PrimExpr(0), PrimType::Int(64));
      PrimExpr cluster_x = IntImm::Int64(cluster_[0]);
      PrimExpr cluster_y = IntImm::Int64(cluster_[1]);
      PrimExpr columns =
          prim::FloorDiv(prim::cast(extents_[0], PrimType::Int(64)), cluster_x);
      PrimExpr rows =
          prim::FloorDiv(prim::cast(extents_[1], PrimType::Int(64)), cluster_y);
      PrimExpr linear = prim::FloorDiv(block_x, cluster_x) +
                        prim::FloorDiv(block_y, cluster_y) * columns;
      PrimExpr width = IntImm::Int64(*panel);
      PrimExpr span = width * (column ? rows : columns);
      PrimExpr panel_index = prim::FloorDiv(linear, span);
      PrimExpr panel_offset = prim::FloorMod(linear, span);
      PrimExpr stride =
          prim::Min(width, (column ? columns : rows) - panel_index * width);
      PrimExpr major = prim::FloorDiv(panel_offset, stride);
      major = prim::Select(prim::FloorMod(panel_index, IntImm::Int64(2)) != 0,
                           (column ? rows : columns) - 1 - major, major);
      PrimExpr minor =
          prim::FloorMod(panel_offset, stride) + panel_index * width;
      std::array<PrimExpr, 2> coordinates{
          (column ? minor : major) * cluster_x +
              prim::FloorMod(block_x, cluster_x),
          (column ? major : minor) * cluster_y +
              prim::FloorMod(block_y, cluster_y)};
      ffi::Map<PrimVar, PrimExpr> replacements;
      ffi::Array<Stmt> statements;
      for (size_t axis = 0; axis < 2; ++axis) {
        if (!blocks_[axis])
          continue;
        PrimVar variable(axis == 0 ? "swizzled_block_x" : "swizzled_block_y",
                         blocks_[axis].value().ty());
        statements.push_back(
            Bind(variable, prim::cast(coordinates[axis], variable.ty())));
        replacements.Set(*blocks_[axis], variable);
      }
      Stmt body = tl::Substitute(node->body, replacements);
      statements.push_back(body);
      auto attributes = node->attrs->dict;
      attributes.erase("threadblock_swizzle_pattern");
      region.CopyOnWrite()->attrs = DictAttrs(attributes);
      region.CopyOnWrite()->body = SeqStmt(statements);
    }
    Stmt result =
        StmtExprMutator::Mutate_(region.get(), mode).ValueOrUnchanged(region);
    for (size_t axis = 0; axis < 2; ++axis) {
      blocks_[axis] = saved_blocks[axis];
      extents_[axis] = saved_extents[axis];
      cluster_[axis] = saved_cluster[axis];
    }
    return result;
  }
};

class CudaTemplateLowerer : public tirx::StmtExprMutator {
public:
  SeqStmt Lower(const SeqStmt &body) {
    SeqStmt result = Mutate(body).ValueOrUnchanged(body);
    if (rng_allocations_.empty())
      return result;
    rng_allocations_.push_back(result);
    return SeqStmt(rng_allocations_, body->loc);
  }

private:
  ffi::Array<Stmt> rng_allocations_;
  ffi::Optional<tirx::TensorVar> rng_state_;
  std::string rng_type_;

  ffi::UnchangedOr<Stmt> Mutate_(const EvaluateNode *node,
                                 ffi::InplaceMode mode) final {
    if (const auto *call = node->value.as<CallNode>()) {
      if (const auto *operation = call->op.as<OpNode>();
          operation && operation->name == "tl.no_set_max_nreg")
        return SeqStmt({});
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

  static PrimExpr LoadWord(const Expr &pointer, int bits) {
    if (auto scalar = pointer.as<PrimExpr>())
      return prim::cast(scalar.value(), PrimType::UInt(bits));
    auto pointer_type = pointer->ty.as_or_throw<PtrType>();
    TVM_FFI_CHECK(pointer_type->element_type == PrimType::UInt(bits), TypeError)
        << "Expected a uint" << bits << " storage pointer";
    return Call(PrimType::UInt(bits), tirx::call_extern_op(),
                {StringImm("tl::load_scalar"), pointer})
        .as_or_throw<PrimExpr>();
  }

  static PrimExpr DescriptorOffset(const Expr &pointer, const Expr &offset,
                                   bool tcgen05) {
    auto pointer_type = pointer->ty.as_or_throw<PtrType>();
    TVM_FFI_CHECK(pointer_type->element_type == PrimType::UInt(64), TypeError)
        << "Shared-memory descriptors require uint64 storage";
    return Call(PrimType::UInt(64), tirx::call_extern_op(),
                {StringImm(tcgen05 ? "tl::offset_descriptor<true>"
                                   : "tl::offset_descriptor<false>"),
                 pointer, offset})
        .as_or_throw<PrimExpr>();
  }

  static PrimExpr TmemOffset(const Expr &pointer, const Expr &offset) {
    return LoadWord(pointer, 32) +
           prim::cast(offset.as_or_throw<PrimExpr>(), PrimType::UInt(32));
  }

  static Expr RegisterPointer(const Expr &pointer, const PrimExpr &offset,
                              tl::codegen::ptx::DataType precision,
                              bool accumulator) {
    auto pointer_type = pointer->ty.as_or_throw<PtrType>();
    auto element_type = pointer_type->element_type.as_or_throw<PrimType>();
    int element_bits = element_type.bits();
    if (precision == tl::codegen::ptx::DataType::kInt4 ||
        precision == tl::codegen::ptx::DataType::kUInt4) {
      element_bits = 4;
    }
    PrimType register_type = PrimType::UInt(32);
    if (precision == tl::codegen::ptx::DataType::kFloat64) {
      register_type = PrimType::Float(64);
    } else if (accumulator &&
               precision == tl::codegen::ptx::DataType::kFloat32) {
      register_type = PrimType::Float(32);
    }
    sym::Analyzer analyzer;
    PrimExpr byte_offset = analyzer->Simplify(prim::FloorDiv(
        offset * IntImm(offset.ty(), element_bits), IntImm(offset.ty(), 8)));
    Expr advanced =
        Call(pointer->ty, tirx::ptr_byte_offset_op(), {pointer, byte_offset});
    return tirx::reinterpret(PtrType(register_type, pointer_type->scope),
                             advanced);
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallNode *node,
                                 ffi::InplaceMode mode) final {
    Call call = StmtExprMutator::Mutate_(node, mode)
                    .ValueOrUnchanged(GetRef<Expr>(node))
                    .as_or_throw<Call>();
    auto operation = call->op.as<Op>();
    if (!operation)
      return call;
    if (operation.value()->name == "tl.named_barrier_arrive") {
      return Call(PrimType::Void(), tirx::call_extern_op(),
                  {StringImm("tl::__named_barrier_arrive"), call->args[0],
                   call->args[1]},
                  call->attrs, {}, call->loc);
    }
    if (operation.value()->name == "tl.ptx_mma_block_scale") {
      TVM_FFI_CHECK_EQ(call->args.size(), 21U, ValueError);
      auto text = [&](size_t index) {
        return std::string(call->args[index].as_or_throw<StringImm>()->value);
      };
      int scale_width =
          call->args[5].as_or_throw<IntImm>()->value.as<int>().value();
      TVM_FFI_CHECK(
          text(0) == "float32" && text(1) == "m16n8k64" && text(2) == "row" &&
              text(3) == "col" && text(4) == "mxf4nvf4" && scale_width == 4 &&
              text(6) == "e2m1" && text(7) == "e2m1" && text(8) == "ue4m3",
          ValueError)
          << "Unsupported ptx_mma_block_scale configuration; expected f32 "
             "m16n8k64 row.col mxf4nvf4 4X e2m1.e2m1 ue4m3";
      Expr accumulator = RegisterPointer(
          call->args[13], call->args[14].as_or_throw<PrimExpr>(),
          tl::codegen::ptx::DataType::kFloat32, true);
      ffi::Array<Expr> arguments{
          StringImm("tl::sm120_mma_sync_blockscaled<"
                    "tl::SM120MmaBlockScaledKind::kMxf4nvf4, 4, "
                    "tl::SM120MmaScaleType::kUE4M3>"),
          accumulator,
          RegisterPointer(call->args[9], call->args[10].as_or_throw<PrimExpr>(),
                          tl::codegen::ptx::DataType::kFloat4_e2m1fn, false),
          RegisterPointer(call->args[11],
                          call->args[12].as_or_throw<PrimExpr>(),
                          tl::codegen::ptx::DataType::kFloat4_e2m1fn, false),
          accumulator,
          LoadWord(call->args[15], 32),
          LoadWord(call->args[16], 32)};
      for (size_t index = 17; index < 21; ++index)
        arguments.push_back(prim::cast(
            call->args[index].as_or_throw<PrimExpr>(), PrimType::UInt(16)));
      return Call(PrimType::Void(), tirx::call_extern_op(), arguments,
                  call->attrs, {}, call->loc);
    }
    if (operation.value()->name == "tl.ptx_mma_sp") {
      TVM_FFI_CHECK_EQ(call->args.size(), 16U, ValueError);
      auto text = [&](size_t index) {
        return std::string(call->args[index].as_or_throw<StringImm>()->value);
      };
      int selector =
          call->args[14].as_or_throw<IntImm>()->value.as<int>().value();
      TVM_FFI_CHECK(selector == 0 || selector == 1, ValueError)
          << "Sparse MMA selector must be 0 or 1";
      bool saturate = call->args[15].as_or_throw<IntImm>()->value != 0;
      std::string assembly = tl::codegen::PrintMMAAssembly(
          text(0), text(1), text(2), text(3), text(4), text(5),
          "(const unsigned char*)left", "0", "(const unsigned char*)right", "0",
          "(unsigned char*)output", "0", "(const unsigned char*)metadata", "0",
          std::to_string(selector), "", true, saturate);
      std::string name = "tl_sparse_mma";
      for (size_t index = 0; index < 6; ++index)
        name += "_" + text(index);
      name += "_" + std::to_string(selector) + "_" + std::to_string(saturate);
      std::string source =
          "__device__ __forceinline__ void " + name +
          "(const void* left, const void* right, void* output, "
          "const void* metadata) {\n" +
          assembly + "\n}\n";
      ffi::Array<Expr> arguments{StringImm(name)};
      for (size_t index = 0; index < 4; ++index) {
        auto precision =
            index == 3 ? tl::codegen::ptx::DataType::kUInt32
                       : tl::codegen::ptx::DTypeFromString(text(index + 3));
        arguments.push_back(
            RegisterPointer(call->args[6 + index * 2],
                            call->args[7 + index * 2].as_or_throw<PrimExpr>(),
                            precision, index == 2));
      }
      arguments.push_back(StringImm(source));
      return Call(call->ty, Op::Get("tirx.cuda.func_call"), arguments,
                  call->attrs, {}, call->loc);
    }
    if (operation.value()->name == "tl.ptx_tcgen05_mma_ss" ||
        operation.value()->name == "tl.ptx_tcgen05_mma_ts") {
      bool shared_left = operation.value()->name == "tl.ptx_tcgen05_mma_ss";
      TVM_FFI_CHECK_EQ(call->args.size(), shared_left ? 15U : 14U, ValueError);
      auto precision = tl::codegen::ptx::DTypeFromString(
          call->args[0].as_or_throw<StringImm>()->value);
      bool weight_stationary =
          shared_left && call->args[13].as_or_throw<IntImm>()->value != 0;
      bool two_ctas =
          call->args[shared_left ? 14 : 13].as_or_throw<IntImm>()->value != 0;
      TVM_FFI_CHECK(!weight_stationary || !two_ctas, ValueError)
          << "Weight-stationary TCGEN05 does not support two-CTA issue";
      std::string name = "tl::tcgen05mma_";
      name += weight_stationary ? "ws_ss" : shared_left ? "ss" : "ts";
      name += "<" + tl::codegen::ptx::DTypeEnumToString(precision);
      if (!weight_stationary)
        name += two_ctas ? ", true" : ", false";
      name += ">";
      ffi::Array<Expr> arguments{
          StringImm(name),
          shared_left ? DescriptorOffset(call->args[1], call->args[2], true)
                      : TmemOffset(call->args[1], call->args[2]),
          DescriptorOffset(call->args[3], call->args[4], true),
          TmemOffset(call->args[5], call->args[6]),
          prim::cast(call->args[8].as_or_throw<PrimExpr>(), PrimType::UInt(32)),
          prim::cast(call->args[7].as_or_throw<PrimExpr>(), PrimType::UInt(32)),
          call->args[9],
          call->args[10],
          call->args[11],
          call->args[12]};
      return Call(call->ty, tirx::call_extern_op(), arguments, call->attrs, {},
                  call->loc);
    }
    if (operation.value()->name == "tl.ptx_tcgen05_mma_blockscaled_ss") {
      TVM_FFI_CHECK_EQ(call->args.size(), 16U, ValueError);
      auto precision = tl::codegen::ptx::DTypeFromString(
          call->args[0].as_or_throw<StringImm>()->value);
      bool two_ctas = call->args[15].as_or_throw<IntImm>()->value != 0;
      TVM_FFI_CHECK(call->args[13].as_or_throw<IntImm>()->value == 0 &&
                        call->args[14].as_or_throw<IntImm>()->value == 0,
                    ValueError)
          << "TCGEN05 blockscaled reserved operands must be zero";
      std::string name = "tl::tcgen05mma_blockscaled_ss<" +
                         tl::codegen::ptx::DTypeEnumToString(precision) +
                         (two_ctas ? ", true>" : ", false>");
      ffi::Array<Expr> arguments{
          StringImm(name),
          DescriptorOffset(call->args[1], call->args[2], true),
          DescriptorOffset(call->args[3], call->args[4], true),
          TmemOffset(call->args[5], call->args[6]),
          prim::cast(call->args[8].as_or_throw<PrimExpr>(), PrimType::UInt(32)),
          prim::cast(call->args[7].as_or_throw<PrimExpr>(), PrimType::UInt(32)),
          TmemOffset(call->args[9], call->args[10]),
          TmemOffset(call->args[11], call->args[12])};
      return Call(call->ty, tirx::call_extern_op(), arguments, call->attrs, {},
                  call->loc);
    }
    if (operation.value()->name == "tl.ptx_wgmma_ss" ||
        operation.value()->name == "tl.ptx_wgmma_rs") {
      bool shared_left = operation.value()->name == "tl.ptx_wgmma_ss";
      TVM_FFI_CHECK_EQ(call->args.size(), shared_left ? 15U : 14U, ValueError);
      size_t shift = shared_left ? 1 : 0;
      auto text = [&](size_t index) {
        return std::string(call->args[index].as_or_throw<StringImm>()->value);
      };
      auto flag = [&](size_t index) {
        return call->args[index].as_or_throw<IntImm>()->value != 0;
      };
      auto [rows, columns, reduction] =
          tl::codegen::ptx::ParseMMAShape(text(0));
      auto lhs_type = tl::codegen::ptx::DTypeFromString(text(2 + shift));
      auto rhs_type = tl::codegen::ptx::DTypeFromString(text(3 + shift));
      auto accum_type = tl::codegen::ptx::DTypeFromString(text(4 + shift));
      std::ostringstream name;
      name << (shared_left ? "tl::wgmma_ss<" : "tl::wgmma_rs<")
           << tl::codegen::ptx::DTypeEnumToString(lhs_type) << ", "
           << tl::codegen::ptx::DTypeEnumToString(rhs_type) << ", "
           << tl::codegen::ptx::DTypeEnumToString(accum_type) << ", " << rows
           << ", " << columns << ", " << reduction << ", "
           << (shared_left && !flag(1) ? "true" : "false") << ", "
           << (!flag(1 + shift) ? "true" : "false") << ", "
           << (flag(12 + shift) ? 1 : -1) << ", " << (flag(13 + shift) ? 1 : -1)
           << ">";
      Expr left = shared_left
                      ? Expr(DescriptorOffset(call->args[5 + shift],
                                              call->args[6 + shift], false))
                      : RegisterPointer(call->args[5],
                                        call->args[6].as_or_throw<PrimExpr>(),
                                        lhs_type, false);
      Expr output = RegisterPointer(
          call->args[9 + shift], call->args[10 + shift].as_or_throw<PrimExpr>(),
          accum_type, true);
      auto output_type = output->ty.as_or_throw<PtrType>();
      output = tirx::reinterpret(
          PtrType(PrimType::UInt(32), output_type->scope), output);
      return Call(call->ty, tirx::call_extern_op(),
                  {StringImm(name.str()), left,
                   DescriptorOffset(call->args[7 + shift],
                                    call->args[8 + shift], false),
                   output, call->args[11 + shift]},
                  call->attrs, {}, call->loc);
    }
    if (operation.value()->name == "tl.pack_b16" ||
        operation.value()->name == "tl.pack_b8x4") {
      int elements = operation.value()->name == "tl.pack_b16" ? 2 : 4;
      int bits = 32 / elements;
      TVM_FFI_CHECK_EQ(call->args.size(), static_cast<size_t>(elements),
                       ValueError);
      PrimExpr packed = IntImm(PrimType::UInt(32), 0);
      for (int index = 0; index < elements; ++index) {
        PrimExpr value = call->args[index].as_or_throw<PrimExpr>();
        TVM_FFI_CHECK(value.ty().IsScalar() && value.ty().bits() == bits,
                      TypeError)
            << "Packing requires scalar " << bits << "-bit operands";
        value = prim::cast(tirx::reinterpret(PrimType::UInt(bits), value),
                           PrimType::UInt(32));
        packed = tvm::bitwise_or(packed, tvm::left_shift(value, index * bits));
      }
      return tirx::reinterpret(call->ty, packed);
    }
    if (operation.value()->name == "tl.rng_init") {
      rng_type_ = call->args[3].as_or_throw<StringImm>()->value;
      TVM_FFI_CHECK(rng_type_ == "curandStatePhilox4_32_10_t" ||
                        rng_type_ == "curandStateMRG32k3a_t" ||
                        rng_type_ == "curandStateXORWOW_t",
                    ValueError)
          << "Unsupported CUDA random generator: " << rng_type_;
      tirx::TensorType state_type({IntImm::Int32(16)}, PrimType::UInt(64),
                                  runtime::MemoryScope("local"), 16);
      tirx::TensorVar state(
          "rng_state_" + std::to_string(rng_allocations_.size()), state_type);
      rng_allocations_.push_back(
          Bind(state, Call(state_type, tirx::alloc_tensor_op(), {},
                           std::nullopt, {state_type})));
      rng_state_ = state;
      return Call(call->ty, tirx::call_extern_op(),
                  {StringImm("tl::random_init<" + rng_type_ + ">"),
                   state.data(), call->args[0], call->args[1], call->args[2]},
                  call->attrs, {}, call->loc);
    }
    if (operation.value()->name == "tl.rng_rand" ||
        operation.value()->name == "tl.rng_rand_float") {
      TVM_FFI_CHECK(rng_state_.has_value(), ValueError)
          << "rng_init must precede random number generation";
      std::string name = "uint";
      if (operation.value()->name == "tl.rng_rand_float") {
        name = call->args[0].as_or_throw<StringImm>()->value;
        TVM_FFI_CHECK(name == "uniform" || name == "normal", ValueError)
            << "Unsupported CUDA random distribution: " << name;
        if (call->ty.as_or_throw<PrimType>().bits() == 64)
          name += "_double";
      }
      return Call(call->ty, tirx::call_extern_op(),
                  {StringImm("tl::random_" + name + "<" + rng_type_ + ">"),
                   rng_state_.value().data()},
                  call->attrs, {}, call->loc);
    }
    if (operation.value()->name == "tl.ptx_ldmatrix") {
      TVM_FFI_CHECK_EQ(call->args.size(), 4U, ValueError);
      int matrices =
          call->args[1].as_or_throw<IntImm>()->value.as<int>().value();
      bool transpose = call->args[0].as_or_throw<IntImm>()->value != 0;
      TVM_FFI_CHECK(matrices == 1 || matrices == 2 || matrices == 4, ValueError)
          << "ldmatrix requires one, two, or four matrices";
      ffi::String name = "tl::ptx_ldmatrix_x" + std::to_string(matrices) +
                         (transpose ? "_trans" : "");
      return Call(call->ty, tirx::call_extern_op(),
                  {StringImm(name), call->args[2], call->args[3]}, call->attrs,
                  {}, call->loc);
    }
    if (operation.value()->name == "tl.ptx_stmatrix") {
      TVM_FFI_CHECK_GE(call->args.size(), 4U, ValueError);
      int matrices =
          call->args[1].as_or_throw<IntImm>()->value.as<int>().value();
      bool transpose = call->args[0].as_or_throw<IntImm>()->value != 0;
      TVM_FFI_CHECK(matrices == 1 || matrices == 2 || matrices == 4, ValueError)
          << "stmatrix requires one, two, or four matrices";
      const auto *shape_argument = call->args.back().as<StringImmNode>();
      ffi::String shape =
          shape_argument ? shape_argument->value : ffi::String("m8n8");
      TVM_FFI_CHECK(shape == "m8n8" || (shape == "m16n8" && transpose),
                    ValueError)
          << "Unsupported stmatrix shape or transpose mode: " << shape;
      size_t argument_count = call->args.size() - (shape_argument ? 1 : 0);
      TVM_FFI_CHECK_EQ(argument_count, static_cast<size_t>(matrices + 3),
                       ValueError);
      ffi::String name = "tl::ptx_stmatrix_" + shape + "_x" +
                         std::to_string(matrices) + (transpose ? "_trans" : "");
      ffi::Array<Expr> arguments{StringImm(name)};
      for (size_t index = 2; index < argument_count; ++index)
        arguments.push_back(call->args[index]);
      return Call(call->ty, tirx::call_extern_op(), arguments, call->attrs, {},
                  call->loc);
    }
    if (operation.value()->name == "tirx.ptx_legacy.mma") {
      TVM_FFI_CHECK_EQ(call->args.size(), 13U, ValueError)
          << "Bitwise MMA requires a dedicated instruction lowering";
      auto text = [&](size_t index) {
        return std::string(call->args[index].as_or_throw<StringImm>()->value);
      };
      auto [rows, columns, reduction] =
          tl::codegen::ptx::ParseMMAShape(text(0));
      auto lhs_type = tl::codegen::ptx::DTypeFromString(text(3));
      auto rhs_type = tl::codegen::ptx::DTypeFromString(text(4));
      auto accum_type = tl::codegen::ptx::DTypeFromString(text(5));
      bool saturate = call->args[12].as_or_throw<IntImm>()->value != 0;
      TVM_FFI_CHECK((text(1) == "row" || text(1) == "col") &&
                        (text(2) == "row" || text(2) == "col"),
                    ValueError)
          << "MMA layouts must be row or col";
      std::ostringstream symbol;
      symbol << "tl::mma_sync<" << tl::codegen::ptx::DTypeEnumToString(lhs_type)
             << ", " << tl::codegen::ptx::DTypeEnumToString(rhs_type) << ", "
             << tl::codegen::ptx::DTypeEnumToString(accum_type) << ", " << rows
             << ", " << columns << ", " << reduction << ", "
             << (text(1) == "col" ? "true" : "false") << ", "
             << (text(2) == "col" ? "true" : "false") << ", "
             << (saturate ? "true" : "false") << ">";
      return Call(
          call->ty, tirx::call_extern_op(),
          {StringImm(symbol.str()),
           RegisterPointer(call->args[10],
                           call->args[11].as_or_throw<PrimExpr>(), accum_type,
                           true),
           RegisterPointer(call->args[6], call->args[7].as_or_throw<PrimExpr>(),
                           lhs_type, false),
           RegisterPointer(call->args[8], call->args[9].as_or_throw<PrimExpr>(),
                           rhs_type, false)},
          call->attrs, {}, call->loc);
    }
    return call;
  }
};
} // namespace

ffi::Module BuildTileLangCUDA(IRModule module, Target target,
                              bool source_only) {
  if (auto validate =
          ffi::Function::GetGlobal("tilelang_callback_cuda_validate")) {
    (*validate)(module);
  }
  if (!source_only &&
      ffi::Function::GetGlobal("tilelang_callback_cuda_compile")) {
    module = WithAttr(std::move(module), attr::kCompile,
                      ffi::String("tl.cuda.compile_source"));
  }
  for (const auto &[global, base] : module->functions) {
    auto function = base.as_or_throw<tirx::Function>();
    if (function->body) {
      auto swizzle =
          ffi::make_object<ThreadblockSwizzleLowerer>(function->attrs);
      auto lowerer = ffi::make_object<CudaTemplateLowerer>();
      SeqStmt body = *function->body;
      body =
          swizzle->Mutate(body).ValueOrUnchanged(body).as_or_throw<SeqStmt>();
      function.CopyOnWrite()->body = lowerer->Lower(body);
      module.CopyOnWrite()->functions.Set(global, function);
    }
  }
  module = tirx::transform::BindTarget(target)(std::move(module));
  module = tirx::transform::LowerIntrin()(std::move(module));
  return tl::BuildNativeCodegen(std::move(module), std::move(target),
                                "tilelang_callback_cuda_postproc", source_only);
}
} // namespace tvm::codegen
