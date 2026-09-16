#ifndef TVM_TL_ASCEND_OP_REDUCE_H_
#define TVM_TL_ASCEND_OP_REDUCE_H_

#include "support/check.h"

namespace tvm {
namespace tl {
namespace ascend {

inline void CheckAllReduceWidth(int reducing_threads, int scale,
                                const char *op_name) {
  ICHECK_GT(reducing_threads, 0)
      << op_name << ": AllReduce threads must be positive, got "
      << reducing_threads;
  ICHECK_GT(scale, 0) << op_name << ": AllReduce scale must be positive, got "
                      << scale;
  ICHECK_EQ(reducing_threads % scale, 0)
      << op_name << ": AllReduce threads (" << reducing_threads
      << ") must be divisible by scale (" << scale << ")";
}

}
}
}

#endif
