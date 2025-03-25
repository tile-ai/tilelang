// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file swizzle.h
 * \brief Define swizzled layout
 *
 */

#ifndef TVM_TL_LAYOUT_SWIZZLE_H_
#define TVM_TL_LAYOUT_SWIZZLE_H_

#include "layout.h"

namespace tvm {
namespace tl {

/*!
 * \brief Swizzle pattern
 */
class SwizzlePattern {
public:
  SwizzlePattern() = default;
  SwizzlePattern(int bits, int base, int shift);
  PrimExpr swizzle(PrimExpr expr) const;
  int Bits() const { return bits_; }
  int Base() const { return base_; }
  int Shift() const { return shift_; }
  bool operator==(const SwizzlePattern &other) const;

private:
  int bits_;
  int base_;
  int shift_;
};

/*!
 * \brief Layout with swizzle
 */
class SwizzledLayoutNode : public LayoutNode {
public:
  SwizzledLayoutNode() = default;
  SwizzledLayoutNode(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
                     SwizzlePattern pattern);

  Array<PrimExpr> Forward(const Array<PrimExpr> &vars) const final;
  Layout Inverse() const final;
  std::string DebugOutput() const final;
  bool IsEqual(const SwizzledLayoutNode *other, bool skip_index = false) const;
  static constexpr const char *_type_key = "tl.SwizzledLayout";
  bool SEqualReduce(const SwizzledLayoutNode *other, SEqualReducer equal) const;
  void VisitAttrs(tvm::AttrVisitor *v);
  TVM_DECLARE_FINAL_OBJECT_INFO(SwizzledLayoutNode, LayoutNode);

private:
  SwizzlePattern pattern_;
};

/*!
 * \brief SwizzledLayout reference class.
 */
class SwizzledLayout : public Layout {
public:
  TVM_DLL SwizzledLayout(Array<IterVar> forward_var,
                         Array<PrimExpr> forward_index, SwizzlePattern pattern);
  TVM_DLL SwizzledLayout(Array<PrimExpr> input_size,
                         Array<PrimExpr> forward_index, SwizzlePattern pattern);

  TVM_DEFINE_OBJECT_REF_METHODS(SwizzledLayout, Layout, SwizzledLayoutNode);
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_LAYOUT_SWIZZLE_H_