/*!
 * \file tl/op/reduce.h
 * \brief Reduction operators for tensor computations
 */

#ifndef TVM_TL_OP_REDUCE_H_
#define TVM_TL_OP_REDUCE_H_

#include "operator.h"

namespace tvm {

namespace tl {

using namespace tir;

/// Supported reduction operation types
enum class ReduceTypeEnum : uint8_t {
  kSum,    ///< Sum reduction
  kAbsSum, ///< Absolute sum reduction
  kMax,    ///< Maximum value reduction
  kMin,    ///< Minimum value reduction
  kAbsMax, ///< Maximum absolute value reduction
  kBitAnd, ///< Bitwise and reduction
  kBitOr,  ///< Bitwise or reduction
  kBitXor, ///< Bitwise xor reduction
};

/// Node class representing a reduction type
class ReduceTypeNode : public Object {
public:
  int type{-1}; ///< Internal type identifier
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ReduceType", ReduceTypeNode, Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ReduceTypeNode>().def_ro("type", &ReduceTypeNode::type);
  }

  /// Type checking methods
  bool isSum() const { return type == int(ReduceTypeEnum::kSum); }
  bool isAbsSum() const { return type == int(ReduceTypeEnum::kAbsSum); }
  bool isMax() const { return type == int(ReduceTypeEnum::kMax); }
  bool isMin() const { return type == int(ReduceTypeEnum::kMin); }
  bool isAbsMax() const { return type == int(ReduceTypeEnum::kAbsMax); }
  bool isBitAnd() const { return type == int(ReduceTypeEnum::kBitAnd); }
  bool isBitOr() const { return type == int(ReduceTypeEnum::kBitOr); }
  bool isBitXor() const { return type == int(ReduceTypeEnum::kBitXor); }
};

/// Wrapper class for reduction type with string-based construction
class ReduceType : public ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ReduceType, ObjectRef,
                                             ReduceTypeNode);
  TVM_DLL ReduceType(std::string type) {
    auto node = tvm::ffi::make_object<ReduceTypeNode>();
    if (type == "sum") {
      node->type = int(ReduceTypeEnum::kSum);
    } else if (type == "abssum") {
      node->type = int(ReduceTypeEnum::kAbsSum);
    } else if (type == "max") {
      node->type = int(ReduceTypeEnum::kMax);
    } else if (type == "absmax") {
      node->type = int(ReduceTypeEnum::kAbsMax);
    } else if (type == "min") {
      node->type = int(ReduceTypeEnum::kMin);
    } else if (type == "bitand") {
      node->type = int(ReduceTypeEnum::kBitAnd);
    } else if (type == "bitor") {
      node->type = int(ReduceTypeEnum::kBitOr);
    } else if (type == "bitxor") {
      node->type = int(ReduceTypeEnum::kBitXor);
    } else {
      LOG(FATAL) << "Invalid reduce type: " << type;
    }
    data_ = std::move(node);
  }
};

/// Node class for reduction operations
class ReduceOpNode : public TileOperatorNode {
public:
  tir::Buffer src, dst; ///< Source and destination buffers
  // Optional: keep the original regions used to construct this op
  BufferRegion srcRegion_, dstRegion_;
  int dim;         ///< Dimension to reduce along
  ReduceType type; ///< Type of reduction operation
  bool clear;      ///< Whether to clear destination before reduction

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ReduceOp", ReduceOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ReduceOpNode>()
        .def_ro("src", &ReduceOpNode::src)
        .def_ro("dst", &ReduceOpNode::dst)
        .def_ro("srcRegion", &ReduceOpNode::srcRegion_)
        .def_ro("dstRegion", &ReduceOpNode::dstRegion_)
        .def_ro("dim", &ReduceOpNode::dim)
        .def_ro("type", &ReduceOpNode::type)
        .def_ro("clear", &ReduceOpNode::clear);
  }

  /// Lower the operator to TIR statements
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  /// Infer memory layout for buffers
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const;

private:
  /// Generate initial value for reduction
  PrimExpr MakeInitValue() const;
  /// Generate reduction expression
  PrimExpr MakeReduce(const PrimExpr &a, const PrimExpr &b) const;
  /// Generate codegen reducer string
  std::string MakeCodegenReducer() const;
};

/// Wrapper class for reduction operations
class ReduceOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ReduceOp, TileOperator,
                                             ReduceOpNode);
  TVM_DLL ReduceOp(Array<PrimExpr> args);
  static const Op &Get();
};

/// Node class for cumulative sum operations
class CumSumOpNode : public TileOperatorNode {
public:
  tir::Buffer src, dst; ///< Source and destination buffers
  // Optional: keep the original regions used to construct this op
  BufferRegion srcRegion_, dstRegion_;
  int dim;      ///< Dimension along which to compute cumulative sum
  bool reverse; ///< Whether to compute in reverse order
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.CumSumOp", CumSumOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CumSumOpNode>()
        .def_ro("src", &CumSumOpNode::src)
        .def_ro("dst", &CumSumOpNode::dst)
        .def_ro("srcRegion", &CumSumOpNode::srcRegion_)
        .def_ro("dstRegion", &CumSumOpNode::dstRegion_)
        .def_ro("dim", &CumSumOpNode::dim)
        .def_ro("reverse", &CumSumOpNode::reverse);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const;
};

/// Wrapper class for cumulative sum operations
class CumSumOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(CumSumOp, TileOperator,
                                             CumSumOpNode);
  TVM_DLL CumSumOp(Array<PrimExpr> args);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_REDUCE_H_
