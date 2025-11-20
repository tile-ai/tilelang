/*!
 * \file tl/op/remote_copy.cc
 * \brief Remote copy operators.
 *
 */

#include "remote_copy.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include <sstream>

#include "../target/cuda.h"
#include "../target/utils.h"
#include "builtin.h"
#include "distributed.h"
#include "parallel.h"

namespace tvm {
namespace tl {

using namespace tir;

PrimExpr PutOpNode::get_offset(const BufferLoadNode *load) const {
  PrimExpr offset = 0;
  PrimExpr stride = 1;
  auto buffer_shape = load->buffer->shape;
  for (int i = load->indices.size() - 1; i >= 0; i--) {
    offset += load->indices[i] * stride;
    stride *= buffer_shape[i];
  }
  return div(offset * load->dtype.bits(), 8);
}

PrimExpr PutOpNode::MakeAddress(const Buffer &buffer,
                                const Array<PrimExpr> &indices) const {
  return Call(DataType::Handle(), builtin::address_of(),
              {BufferLoad(buffer, indices)});
}

PrimExpr PutOpNode::MakeRemappedAddress(const LowerArgs &T,
                                        const Buffer &buffer,
                                        const Array<PrimExpr> &indices) const {
  Buffer remapped = buffer;
  if (T.buffer_remap.count(buffer)) {
    remapped = T.buffer_remap[buffer];
  }
  return MakeAddress(remapped, indices);
}

PutOp::PutOp(Array<PrimExpr> args, BufferMap vmap) {
  ObjectPtr<PutOpNode> node = make_object<PutOpNode>();
  node->src_addr = args[0];
  node->dst_addr = args[1];
  ICHECK(node->src_addr.as<CallNode>()) << "src_addr must be a call node";
  ICHECK(node->src_addr.as<CallNode>()->op.same_as(builtin::address_of()))
      << "src_addr must be address_of op";
  ICHECK(node->dst_addr.as<CallNode>()) << "dst_addr must be a call node";
  ICHECK(node->dst_addr.as<CallNode>()->op.same_as(builtin::address_of()))
      << "dst_addr must be address_of op";

  const auto *src_load =
      node->src_addr.as<CallNode>()->args[0].as<BufferLoadNode>();
  const auto *dst_load =
      node->dst_addr.as<CallNode>()->args[0].as<BufferLoadNode>();
  ICHECK(src_load && dst_load) << "address_of must wrap BufferLoad nodes";

  node->src_offset = node->get_offset(src_load);
  node->dst_offset = node->get_offset(dst_load);
  node->src_buffer = src_load->buffer;
  node->dst_buffer = dst_load->buffer;
  node->src_indices = src_load->indices;
  node->dst_indices = dst_load->indices;

  node->copy_size = args[2];
  node->dst_pe = args[3];
  node->unroll_factor = args[4].as<IntImm>().value()->value;
  node->scope = args[5].as<StringImm>().value()->value;
  node->is_symmetric = node->dst_pe.defined();
  data_ = std::move(node);
  (void)vmap;
}

Stmt PutOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  (void)analyzer;
  Array<PrimExpr> new_args;
  std::stringstream ss;
  if (scope == "warp") {
    ss << "tl::cp_warp<" << copy_size << ", " << unroll_factor << ">";
  } else if (scope == "block") {
    ss << "tl::cp_block<" << copy_size << ">";
  } else {
    LOG(FATAL) << "Invalid scope: " << scope;
  }

  new_args.push_back(StringImm(ss.str()));
  if (is_symmetric) {
    PrimExpr dst_addr_expr = MakeRemappedAddress(T, dst_buffer, dst_indices);
    PrimExpr local_rank = Call(DataType::Int(64), tl::get_rank(), {});
    PrimExpr local_base_ptr =
        Call(DataType::Handle(), tl::get_remote_base_ptr(), {local_rank});
    PrimExpr offset_to_base =
        Sub(Call(DataType::Handle(), tl::get_uintptr_t(), {dst_addr_expr}),
            local_base_ptr);
    new_args.push_back(
        Call(DataType::Handle(), tl::get_remote_base_ptr(), {dst_pe}) +
        offset_to_base);
  } else {
    new_args.push_back(MakeRemappedAddress(T, dst_buffer, dst_indices));
  }
  new_args.push_back(MakeRemappedAddress(T, src_buffer, src_indices));
  auto put = Call(DataType::Handle(), builtin::call_extern(), new_args);
  return Evaluate(put);
}

LayoutMap PutOpNode::InferLayout(const LayoutInferArgs &T,
                                 InferLevel level) const {
  (void)T;
  (void)level;
  return {};
}

TileOperator PutOpNode::Clone() const {
  auto node = make_object<PutOpNode>(*this);
  return PutOp(node);
}

PrimExpr GetOpNode::get_offset(const BufferLoadNode *load) const {
  PrimExpr offset = 0;
  PrimExpr stride = 1;
  auto buffer_shape = load->buffer->shape;
  for (int i = load->indices.size() - 1; i >= 0; i--) {
    offset += load->indices[i] * stride;
    stride *= buffer_shape[i];
  }
  return div(offset * load->dtype.bits(), 8);
}

PrimExpr GetOpNode::MakeAddress(const Buffer &buffer,
                                const Array<PrimExpr> &indices) const {
  return Call(DataType::Handle(), builtin::address_of(),
              {BufferLoad(buffer, indices)});
}

PrimExpr GetOpNode::MakeRemappedAddress(const LowerArgs &T,
                                        const Buffer &buffer,
                                        const Array<PrimExpr> &indices) const {
  Buffer remapped = buffer;
  if (T.buffer_remap.count(buffer)) {
    remapped = T.buffer_remap[buffer];
  }
  return MakeAddress(remapped, indices);
}

GetOp::GetOp(Array<PrimExpr> args, BufferMap vmap) {
  ObjectPtr<GetOpNode> node = make_object<GetOpNode>();
  node->src_addr = args[0];
  node->dst_addr = args[1];
  ICHECK(node->src_addr.as<CallNode>()) << "src_addr must be a call node";
  ICHECK(node->src_addr.as<CallNode>()->op.same_as(builtin::address_of()))
      << "src_addr must be address_of op";
  ICHECK(node->dst_addr.as<CallNode>()) << "dst_addr must be a call node";
  ICHECK(node->dst_addr.as<CallNode>()->op.same_as(builtin::address_of()))
      << "dst_addr must be address_of op";

  const auto *src_load =
      node->src_addr.as<CallNode>()->args[0].as<BufferLoadNode>();
  const auto *dst_load =
      node->dst_addr.as<CallNode>()->args[0].as<BufferLoadNode>();
  ICHECK(src_load && dst_load) << "address_of must wrap BufferLoad nodes";

  node->src_offset = node->get_offset(src_load);
  node->dst_offset = node->get_offset(dst_load);
  node->src_buffer = src_load->buffer;
  node->dst_buffer = dst_load->buffer;
  node->src_indices = src_load->indices;
  node->dst_indices = dst_load->indices;

  node->copy_size = args[2];
  node->src_pe = args[3];
  node->unroll_factor = args[4].as<IntImm>().value()->value;
  node->scope = args[5].as<StringImm>().value()->value;
  node->is_symmetric = node->src_pe.defined();
  data_ = std::move(node);
  (void)vmap;
}

Stmt GetOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  (void)analyzer;
  Array<PrimExpr> new_args;
  std::stringstream ss;
  if (scope == "warp") {
    ss << "tl::cp_warp<" << copy_size << ", " << unroll_factor << ">";
  } else if (scope == "block") {
    ss << "tl::cp_block<" << copy_size << ">";
  } else {
    LOG(FATAL) << "Invalid scope: " << scope;
  }

  new_args.push_back(StringImm(ss.str()));
  PrimExpr dst_addr_expr = MakeRemappedAddress(T, dst_buffer, dst_indices);
  new_args.push_back(dst_addr_expr); // Always dst first in tl_templates
  if (is_symmetric) {
    PrimExpr src_addr_expr = MakeRemappedAddress(T, src_buffer, src_indices);
    PrimExpr local_rank = Call(DataType::Int(64), tl::get_rank(), {});
    PrimExpr local_base_ptr =
        Call(DataType::Handle(), tl::get_remote_base_ptr(), {local_rank});
    PrimExpr offset_to_base =
        Sub(Call(DataType::Handle(), tl::get_uintptr_t(), {src_addr_expr}),
            local_base_ptr);
    new_args.push_back(
        Call(DataType::Handle(), tl::get_remote_base_ptr(), {src_pe}) +
        offset_to_base);
  } else {
    new_args.push_back(MakeRemappedAddress(T, src_buffer, src_indices));
  }

  auto get = Call(DataType::Handle(), builtin::call_extern(), new_args);
  return Evaluate(get);
}

LayoutMap GetOpNode::InferLayout(const LayoutInferArgs &T,
                                 InferLevel level) const {
  (void)T;
  (void)level;
  return {};
}

TileOperator GetOpNode::Clone() const {
  auto node = make_object<GetOpNode>(*this);
  return GetOp(node);
}

StOp::StOp(Array<PrimExpr> args, BufferMap vmap) {
  ObjectPtr<StOpNode> node = make_object<StOpNode>();
  node->dst = args[0];
  ICHECK(node->dst.as<CallNode>()) << "dst must be a call node";
  ICHECK(node->dst.as<CallNode>()->op.same_as(builtin::address_of()))
      << "dst must be address_of op";

  node->value = args[1];
  node->sem = args[2].as<StringImm>().value()->value;
  node->scope = args[3].as<StringImm>().value()->value;
  node->dst_pe = args[4];
  data_ = std::move(node);
  (void)vmap;
}

bool StOpNode::is_distributed() const {
  return !(dst_pe->IsInstance<IntImmNode>() && dst_pe.as<IntImmNode>()->value == -1);
}

Stmt StOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  (void)analyzer;
  (void)T;
  Array<PrimExpr> new_args;
  std::stringstream ss;
  
  // Build function name: tl::st_<sem>_<scope>
  ss << "tl::st_" << sem << "_" << scope;
  
  new_args.push_back(StringImm(ss.str()));
  if (is_distributed()) {
    PrimExpr local_rank = Call(DataType::Int(64), tl::get_rank(), {});
    PrimExpr local_base_ptr =
        Call(DataType::Handle(), tl::get_remote_base_ptr(), {local_rank});
    PrimExpr offset_to_base =
        Sub(Call(DataType::Handle(), tl::get_uintptr_t(), {dst}),
            local_base_ptr);
    new_args.push_back(
        Call(DataType::Handle(), tl::get_remote_base_ptr(), {dst_pe}) +
        offset_to_base);
  } else {
    new_args.push_back(dst);
  }
  new_args.push_back(value);
  
  auto st = Call(DataType::Handle(), builtin::call_extern(), new_args);
  return Evaluate(st);
}

LayoutMap StOpNode::InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const {
  (void)T;
  (void)level;
  return {};
}

TileOperator StOpNode::Clone() const {
  auto node = make_object<StOpNode>(*this);
  return StOp(node);
}

TIR_REGISTER_TL_OP(GetOp, get)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(StOp, st)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK({ PutOpNode::RegisterReflection(); });
TVM_FFI_STATIC_INIT_BLOCK({ GetOpNode::RegisterReflection(); });
TVM_FFI_STATIC_INIT_BLOCK({ StOpNode::RegisterReflection(); });

} // namespace tl
} // namespace tvm
