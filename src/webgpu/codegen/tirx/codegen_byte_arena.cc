#include "compat/analyzer.h"
#include "op/prim.h"
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/memory.h>

#include <sstream>

#include "codegen_webgpu.h"

namespace tvm::tl::codegen_impl {

CodeGenWebGPU::TensorView CodeGenWebGPU::ResolveTensorView(const Expr &data) {
  if (const auto *variable = data.as<VarNode>()) {
    auto found = tensor_views_.find(variable);
    if (found != tensor_views_.end())
      return found->second;
    TVM_FFI_CHECK(var_idmap_.count(variable), ValueError)
        << "WebGPU tensor views require a named storage array";
    return {ffi::GetRef<Var>(variable), IntImm::Int32(0), false};
  }
  if (const auto *call = data.as<CallNode>()) {
    if (call->op.same_as(tirx::tensor_data_ptr_op()))
      return ResolveTensorView(call->args[0]);
    if (call->op.same_as(tirx::ptr_byte_offset_op())) {
      TensorView view = ResolveTensorView(call->args[0]);
      PrimExpr offset = call->args[1].as_or_throw<PrimExpr>();
      view.byte_offset = tl::arith::Simplify(
          analyzer_, tl::CastValue(view.byte_offset, offset.ty()) + offset);
      return view;
    }
  }
  TVM_FFI_THROW(ValueError)
      << "WebGPU tensor views require a named storage array and byte offset";
}

PrimExpr CodeGenWebGPU::OffsetTensorIndex(const Var &buffer,
                                          const PrimExpr &index) {
  auto found = tensor_views_.find(buffer.get());
  if (found == tensor_views_.end() || prim::IsZero(found->second.byte_offset))
    return index;
  PrimType dtype = buffer->ty.as_or_throw<TensorType>()->dtype;
  size_t stride = dtype.StorageBytes();
  if (dtype.lanes() == 3)
    stride = dtype.WithLanes(4).StorageBytes();
  if (dtype.MatchesCode(kDLBool))
    stride = 4 * dtype.lanes();
  PrimExpr offset =
      prim::FloorDiv(found->second.byte_offset,
                     tl::MakeConst(found->second.byte_offset.ty(), stride));
  offset = tl::CastValue(offset, index.ty().WithLanes(1));
  if (index.ty().lanes() > 1)
    offset = prim::Broadcast(offset, index.ty().lanes());
  return tl::arith::Simplify(analyzer_, index + offset);
}

std::string CodeGenWebGPU::ByteArenaAddress(const Var &buffer,
                                            const std::string &index,
                                            int lane) {
  const auto &view = tensor_views_.at(buffer.get());
  PrimType dtype = buffer->ty.as_or_throw<TensorType>()->dtype;
  std::string offset = PrintExpr(EnforceU32(view.byte_offset));
  return "((" + offset + ") + (" + index + ") * " +
         std::to_string(dtype.StorageBytes()) + "u + " +
         std::to_string((lane % dtype.lanes()) *
                        dtype.WithLanes(1).StorageBytes()) +
         "u)";
}

void CodeGenWebGPU::PrintByteArenaLoad(const TensorLoadNode *node,
                                       std::ostream &output) {
  TVM_FFI_CHECK_EQ(node->indices.size(), 1, ValueError);
  Var buffer = node->source.as_or_throw<Var>();
  const auto &view = tensor_views_.at(buffer.get());
  PrimType result_type = node->ty.as_or_throw<PrimType>();
  PrimType scalar_type = result_type.WithLanes(1);
  PrimType element_type = buffer->ty.as_or_throw<TensorType>()->dtype;
  PrimExpr index = EnforceU32(node->indices[0]);
  std::string indices = SSAGetID(PrintExpr(index), index.ty());
  TVM_FFI_CHECK(scalar_type.MatchesCode(kDLBool) ||
                    (scalar_type.MatchesCode(kDLInt, kDLUInt) &&
                     (scalar_type.bits() == 8 || scalar_type.bits() == 16 ||
                      scalar_type.bits() == 32)) ||
                    scalar_type.MatchesElementType(kDLFloat, 16) ||
                    scalar_type.MatchesElementType(kDLFloat, 32),
                TypeError)
      << "Unsupported WebGPU byte arena element type " << scalar_type;
  bool packed = result_type == PrimType::Int(8, 4);
  if (!result_type.IsScalar()) {
    if (!packed)
      PrintType(result_type, output);
    output << '(';
  }
  for (int lane = 0; lane < result_type.lanes(); ++lane) {
    if (lane)
      output << (packed ? " | " : ", ");
    if (packed)
      output << "((u32(";
    std::string position = indices;
    if (!index.ty().IsScalar())
      position += "[" + std::to_string(lane / element_type.lanes()) + "]";
    std::string address = ByteArenaAddress(buffer, position, lane);
    std::string word = "atomicLoad(&" + GetVarID(view.storage.get()) + "[" +
                       address + " / 4u])";
    std::string shift = "((" + address + " % 4u) * 8u)";
    if (scalar_type.MatchesElementType(kDLFloat, 16)) {
      enable_fp16_ = true;
      output << "f16(unpack2x16float(" << word << ")[((" << address
             << " % 4u) / 2u)])";
    } else if (scalar_type.MatchesElementType(kDLFloat, 32)) {
      output << "bitcast<f32>(" << word << ")";
    } else if (scalar_type.MatchesCode(kDLBool)) {
      output << "((" << word << " >> " << shift << ") & 255u) != 0u";
    } else if (scalar_type.bits() == 32) {
      if (scalar_type.MatchesCode(kDLInt))
        output << "bitcast<i32>(";
      output << word;
      if (scalar_type.MatchesCode(kDLInt))
        output << ')';
    } else {
      int bits = scalar_type.bits();
      std::string low = "((" + word + " >> " + shift + ") & " +
                        std::to_string((uint32_t{1} << bits) - 1) + "u)";
      if (scalar_type.MatchesCode(kDLInt))
        output << "(bitcast<i32>(" << low << " << " << 32 - bits << "u) >> "
               << 32 - bits << "u)";
      else
        output << low;
    }
    if (packed)
      output << ") & 255u) << " << lane * 8 << "u)";
  }
  if (!result_type.IsScalar())
    output << ')';
}

void CodeGenWebGPU::PrintByteArenaStore(const TensorStoreNode *node) {
  TVM_FFI_CHECK_EQ(node->indices.size(), 1, ValueError);
  Var buffer = node->dest.as_or_throw<Var>();
  const auto &view = tensor_views_.at(buffer.get());
  PrimType value_type = node->value.ty();
  PrimType scalar_type = value_type.WithLanes(1);
  PrimType element_type = buffer->ty.as_or_throw<TensorType>()->dtype;
  TVM_FFI_CHECK(scalar_type == element_type.WithLanes(1), TypeError)
      << "WebGPU byte arena store element types must match";
  TVM_FFI_CHECK(scalar_type.MatchesCode(kDLBool) ||
                    (scalar_type.MatchesCode(kDLInt, kDLUInt) &&
                     (scalar_type.bits() == 8 || scalar_type.bits() == 16 ||
                      scalar_type.bits() == 32)) ||
                    scalar_type.MatchesElementType(kDLFloat, 16) ||
                    scalar_type.MatchesElementType(kDLFloat, 32),
                TypeError)
      << "Unsupported WebGPU byte arena element type " << scalar_type;
  std::string value = SSAGetID(PrintExpr(node->value), value_type);
  PrimExpr index = EnforceU32(node->indices[0]);
  std::string indices = SSAGetID(PrintExpr(index), index.ty());
  for (int lane = 0; lane < value_type.lanes(); ++lane) {
    std::string position = indices;
    if (!index.ty().IsScalar())
      position += "[" + std::to_string(lane / element_type.lanes()) + "]";
    std::string address = ByteArenaAddress(buffer, position, lane);
    std::string pointer =
        "&" + GetVarID(view.storage.get()) + "[" + address + " / 4u]";
    std::ostringstream scalar_stream;
    if (value_type.IsScalar())
      scalar_stream << value;
    else
      PrintVecElemLoad(value, value_type, lane, scalar_stream);
    std::string scalar = scalar_stream.str();
    std::string bits;
    if (scalar_type.MatchesElementType(kDLFloat, 16)) {
      enable_fp16_ = true;
      bits = "pack2x16float(vec2<f32>(f32(" + scalar + "), 0.0f))";
    } else if (scalar_type.MatchesCode(kDLFloat) ||
               scalar_type.MatchesElementType(kDLInt, 32)) {
      bits = "bitcast<u32>(" + scalar + ")";
    } else if (scalar_type.MatchesCode(kDLBool)) {
      bits = "select(0u, 1u, " + scalar + ")";
    } else {
      bits = "u32(" + scalar + ")";
    }
    PrintIndent();
    if (scalar_type.bits() == 32) {
      stream << "atomicStore(" << pointer << ", " << bits << ");\n";
    } else {
      int width = scalar_type.MatchesCode(kDLBool) ? 8 : scalar_type.bits();
      std::string mask = std::to_string((uint32_t{1} << width) - 1) + "u";
      std::string shift = "((" + address + " % 4u) * 8u)";
      stream << "atomicAnd(" << pointer << ", ~(" << mask << " << " << shift
             << "));\n";
      PrintIndent();
      stream << "atomicOr(" << pointer << ", ((" << bits << ") & " << mask
             << ") << " << shift << ");\n";
    }
  }
}

} // namespace tvm::tl::codegen_impl
