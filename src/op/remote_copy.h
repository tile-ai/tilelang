/*!
 * \file tl/op/remote_copy.h
 * \brief Remote copy operators.
 *
 */

#ifndef TVM_TL_OP_BULK_COPY_H_
#define TVM_TL_OP_BULK_COPY_H_

#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>

#include "../layout/layout.h"
#include "operator.h"

namespace tvm {
namespace tl {

using namespace tir;

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
  bool enable_aggresive_vectorize; ///< Whether to enable aggressive vectorization, only effctive for warp-scope
  
  bool is_distributed() const;

  static constexpr const char *_type_key = "tl.PutOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(PutOpNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const override;

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
        .def_ro("scope", &PutOpNode::scope);
  }

  bool SEqualReduce(const PutOpNode *other, SEqualReducer equal) const {
    return equal(src_addr, other->src_addr) &&
           equal(dst_addr, other->dst_addr) &&
           equal(src_offset, other->src_offset) &&
           equal(dst_offset, other->dst_offset) &&
           equal(copy_size, other->copy_size) && equal(dst_pe, other->dst_pe) &&
           equal(unroll_factor, other->unroll_factor) &&
           equal(src_buffer, other->src_buffer) &&
           equal(dst_buffer, other->dst_buffer) &&
           equal(src_indices, other->src_indices) &&
           equal(dst_indices, other->dst_indices) && scope == other->scope;
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(src_addr);
    hash_reduce(dst_addr);
    hash_reduce(src_offset);
    hash_reduce(dst_offset);
    hash_reduce(copy_size);
    hash_reduce(dst_pe);
    hash_reduce(unroll_factor);
    hash_reduce(src_buffer);
    hash_reduce(dst_buffer);
    hash_reduce(src_indices);
    hash_reduce(dst_indices);
    hash_reduce(scope);
  }

  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;

  PrimExpr get_offset(const BufferLoadNode *load) const;

private:
  PrimExpr MakeAddress(const Buffer &buffer,
                       const Array<PrimExpr> &indices) const;
  PrimExpr MakeRemappedAddress(const LowerArgs &T, const Buffer &buffer,
                               const Array<PrimExpr> &indices) const;
};

class PutOp : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(PutOp, TileOperator, PutOpNode);
  TVM_DLL PutOp(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

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
  bool enable_aggresive_vectorize; ///< Whether to enable aggressive vectorization, only effctive for warp-scope

  bool is_distributed() const;

  static constexpr const char *_type_key = "tl.GetOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(GetOpNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const override;

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
        .def_ro("scope", &GetOpNode::scope);
  }

  bool SEqualReduce(const GetOpNode *other, SEqualReducer equal) const {
    return equal(src_addr, other->src_addr) &&
           equal(dst_addr, other->dst_addr) &&
           equal(src_offset, other->src_offset) &&
           equal(dst_offset, other->dst_offset) &&
           equal(copy_size, other->copy_size) && equal(src_pe, other->src_pe) &&
           equal(unroll_factor, other->unroll_factor) &&
           equal(src_buffer, other->src_buffer) &&
           equal(dst_buffer, other->dst_buffer) &&
           equal(src_indices, other->src_indices) &&
           equal(dst_indices, other->dst_indices) && scope == other->scope;
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(src_addr);
    hash_reduce(dst_addr);
    hash_reduce(src_offset);
    hash_reduce(dst_offset);
    hash_reduce(copy_size);
    hash_reduce(src_pe);
    hash_reduce(unroll_factor);
    hash_reduce(src_buffer);
    hash_reduce(dst_buffer);
    hash_reduce(src_indices);
    hash_reduce(dst_indices);
    hash_reduce(scope);
  }

  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;

  PrimExpr get_offset(const BufferLoadNode *load) const;

private:
  PrimExpr MakeAddress(const Buffer &buffer,
                       const Array<PrimExpr> &indices) const;
  PrimExpr MakeRemappedAddress(const LowerArgs &T, const Buffer &buffer,
                               const Array<PrimExpr> &indices) const;
};

class GetOp : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(GetOp, TileOperator, GetOpNode);
  TVM_DLL GetOp(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

class StOpNode : public TileOperatorNode {
public:
  PrimExpr dst;           ///< Destination address
  PrimExpr value;         ///< Value to store
  PrimExpr dst_pe;        ///< Destination processing element (optional)
  int scope;
  int sem;
  int na;

  bool is_distributed() const;

  static constexpr const char *_type_key = "tl.StOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(StOpNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const override;

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

  bool SEqualReduce(const StOpNode *other, SEqualReducer equal) const {
    return equal(dst, other->dst) &&
           equal(value, other->value) &&
           equal(dst_pe, other->dst_pe) &&
           scope == other->scope &&
           sem == other->sem &&
           na == other->na;
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dst);
    hash_reduce(value);
    hash_reduce(dst_pe);
    hash_reduce(scope);
    hash_reduce(sem);
    hash_reduce(na);
  }

  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;
};

class StOp : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(StOp, TileOperator, StOpNode);
  TVM_DLL StOp(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

class LdOpNode : public TileOperatorNode {
  public:
    PrimExpr src;           ///< Source address
    PrimExpr value;         ///< Value to store
    PrimExpr src_pe;        ///< Source PE (optional)
    int scope;
    int sem;
    int na;
    int nc;
  
    bool is_distributed() const;
  
    static constexpr const char *_type_key = "tl.LdOp";
    TVM_DECLARE_FINAL_OBJECT_INFO(LdOpNode, TileOperatorNode);
  
    Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
    LayoutMap InferLayout(const LayoutInferArgs &T,
                          InferLevel level) const override;
    static const Op &Get();
    TileOperator Clone() const override;
  
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
  
    bool SEqualReduce(const LdOpNode *other, SEqualReducer equal) const {
      return equal(src, other->src) &&
             equal(value, other->value) &&
             equal(src_pe, other->src_pe) &&
             scope == other->scope &&
             sem == other->sem &&
             na == other->na &&
             nc == other->nc;
    }
  
    void SHashReduce(SHashReducer hash_reduce) const {
      hash_reduce(src);
      hash_reduce(value);
      hash_reduce(src_pe);
      hash_reduce(scope);
      hash_reduce(sem);
      hash_reduce(na);
      hash_reduce(nc);
    }
  
    static constexpr bool _type_has_method_sequal_reduce = true;
    static constexpr bool _type_has_method_shash_reduce = true;
  };
  
class LdOp : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(LdOp, TileOperator, LdOpNode);
  TVM_DLL LdOp(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};
  

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_BULK_COPY_H_
