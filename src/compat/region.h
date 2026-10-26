#ifndef TL_COMPAT_REGION_H_
#define TL_COMPAT_REGION_H_

#include <tvm/tirx/op/region.h>

namespace tvm::tl {

inline const Op &scope_op() {
  static const Op operation = Op::Get("tl.scope");
  return operation;
}

} // namespace tvm::tl
#endif
