/*!
 * \file tl/op/remote_copy.h
 * \brief Remote copy operators for distributed computing.
 */

#ifndef TVM_TL_OP_REMOTE_COPY_H_
#define TVM_TL_OP_REMOTE_COPY_H_

#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>

#include "../layout/layout.h"
#include "operator.h"

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Put operation for remote memory copy (local -> remote).
 */
class PutOpNode : public TileOperatorNode {
public:
  PrimExpr src_addr;           ///< Address of the source buffer (address_of)
  PrimExpr dst_addr;           ///< Address of the destination buffer
  PrimExpr src_offset;         ///< Byte offset within the source buffer
  PrimExpr dst_offset;         ///< Byte offset within the destination buffer
  PrimExpr copy_size;          ///< Number of bytes/elements to copy
  PrimExpr dst_pe;             ///< Destination processing element (optional)
  int unroll_factor;           ///< Unroll factor for warp copies
  Buffer src_buffer;           ///< Source buffer reference
  Buffer dst_buffer;           ///< Destination buffer reference
  Array<PrimExpr> src_indices; ///< Source indices used for address computation
  Array<PrimExpr>
      dst_indices;   ///< Destination indices used for address computation
  std::string scope; ///< Scope: {warp, block}
  bool enable_aggressive_vectorize; ///< Whether to enable aggressive
                                    ///< vectorization

  bool is_distributed() const;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.PutOp", PutOpNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PutOpNode>()
        .def_ro("src_addr", &PutOpNode::src_addr)
        .def_ro("dst_addr", &PutOpNode::dst_addr)
        .def_ro("copy_size", &PutOpNode::copy_size)
        .def_ro("dst_pe", &PutOpNode::dst_pe)
        .def_ro("unroll_factor", &PutOpNode::unroll_factor)
        .def_ro("src_buffer", &PutOpNode::src_buffer)
        .def_ro("dst_buffer", &PutOpNode::dst_buffer)
        .def_ro("src_indices", &PutOpNode::src_indices)
        .def_ro("dst_indices", &PutOpNode::dst_indices)
        .def_ro("scope", &PutOpNode::scope)
        .def_ro("enable_aggressive_vectorize",
                &PutOpNode::enable_aggressive_vectorize);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const override;

  PrimExpr get_offset(const BufferLoadNode *load) const;

private:
  PrimExpr MakeAddress(const Buffer &buffer,
                       const Array<PrimExpr> &indices) const;
  PrimExpr MakeRemappedAddress(const LowerArgs &T, const Buffer &buffer,
                               const Array<PrimExpr> &indices) const;
};

class PutOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PutOp, TileOperator, PutOpNode);
  TVM_DLL PutOp(Array<PrimExpr> args,
                Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

/*!
 * \brief Get operation for remote memory copy (remote -> local).
 */
class GetOpNode : public TileOperatorNode {
public:
  PrimExpr src_addr;           ///< Remote source buffer address
  PrimExpr dst_addr;           ///< Local destination buffer address
  PrimExpr src_offset;         ///< Byte offset within the source buffer
  PrimExpr dst_offset;         ///< Byte offset within the destination buffer
  PrimExpr copy_size;          ///< Number of bytes/elements to copy
  PrimExpr src_pe;             ///< Source processing element (optional)
  int unroll_factor;           ///< Unroll factor for warp copies
  Buffer src_buffer;           ///< Source buffer reference
  Buffer dst_buffer;           ///< Destination buffer reference
  Array<PrimExpr> src_indices; ///< Source indices used for address computation
  Array<PrimExpr>
      dst_indices;   ///< Destination indices used for address computation
  std::string scope; ///< Scope: {warp, block}
  bool enable_aggressive_vectorize; ///< Whether to enable aggressive
                                    ///< vectorization

  bool is_distributed() const;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.GetOp", GetOpNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GetOpNode>()
        .def_ro("src_addr", &GetOpNode::src_addr)
        .def_ro("dst_addr", &GetOpNode::dst_addr)
        .def_ro("copy_size", &GetOpNode::copy_size)
        .def_ro("src_pe", &GetOpNode::src_pe)
        .def_ro("unroll_factor", &GetOpNode::unroll_factor)
        .def_ro("src_buffer", &GetOpNode::src_buffer)
        .def_ro("dst_buffer", &GetOpNode::dst_buffer)
        .def_ro("src_indices", &GetOpNode::src_indices)
        .def_ro("dst_indices", &GetOpNode::dst_indices)
        .def_ro("scope", &GetOpNode::scope)
        .def_ro("enable_aggressive_vectorize",
                &GetOpNode::enable_aggressive_vectorize);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const override;

  PrimExpr get_offset(const BufferLoadNode *load) const;

private:
  PrimExpr MakeAddress(const Buffer &buffer,
                       const Array<PrimExpr> &indices) const;
  PrimExpr MakeRemappedAddress(const LowerArgs &T, const Buffer &buffer,
                               const Array<PrimExpr> &indices) const;
};

class GetOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GetOp, TileOperator, GetOpNode);
  TVM_DLL GetOp(Array<PrimExpr> args,
                Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

/*!
 * \brief Store operation for remote memory (with signaling).
 */
class StOpNode : public TileOperatorNode {
public:
  PrimExpr dst;    ///< Destination address
  PrimExpr value;  ///< Value to store
  PrimExpr dst_pe; ///< Destination processing element (optional)
  int scope;
  int sem;
  int na;

  bool is_distributed() const;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.StOp", StOpNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StOpNode>()
        .def_ro("dst", &StOpNode::dst)
        .def_ro("value", &StOpNode::value)
        .def_ro("dst_pe", &StOpNode::dst_pe)
        .def_ro("scope", &StOpNode::scope)
        .def_ro("sem", &StOpNode::sem)
        .def_ro("na", &StOpNode::na);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const override;
};

class StOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(StOp, TileOperator, StOpNode);
  TVM_DLL StOp(Array<PrimExpr> args,
               Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

/*!
 * \brief Load operation for remote memory (with signaling).
 */
class LdOpNode : public TileOperatorNode {
public:
  PrimExpr src;    ///< Source address
  PrimExpr value;  ///< Value to store
  PrimExpr src_pe; ///< Source PE (optional)
  int scope;
  int sem;
  int na;
  int nc;

  bool is_distributed() const;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.LdOp", LdOpNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LdOpNode>()
        .def_ro("src", &LdOpNode::src)
        .def_ro("value", &LdOpNode::value)
        .def_ro("src_pe", &LdOpNode::src_pe)
        .def_ro("scope", &LdOpNode::scope)
        .def_ro("sem", &LdOpNode::sem)
        .def_ro("na", &LdOpNode::na)
        .def_ro("nc", &LdOpNode::nc);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const override;
};

class LdOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(LdOp, TileOperator, LdOpNode);
  TVM_DLL LdOp(Array<PrimExpr> args,
               Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_REMOTE_COPY_H_
