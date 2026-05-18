#ifndef TILELANG_SUPPORT_CHECK_H_
#define TILELANG_SUPPORT_CHECK_H_

#include <tvm/ffi/tvm_ffi.h>

#define CHECK(cond, ErrorKind) TVM_FFI_CHECK(cond, ErrorKind)
#define CHECK_LT(x, y, ErrorKind) TVM_FFI_CHECK_LT(x, y, ErrorKind)
#define CHECK_GT(x, y, ErrorKind) TVM_FFI_CHECK_GT(x, y, ErrorKind)
#define CHECK_LE(x, y, ErrorKind) TVM_FFI_CHECK_LE(x, y, ErrorKind)
#define CHECK_GE(x, y, ErrorKind) TVM_FFI_CHECK_GE(x, y, ErrorKind)
#define CHECK_EQ(x, y, ErrorKind) TVM_FFI_CHECK_EQ(x, y, ErrorKind)
#define CHECK_NE(x, y, ErrorKind) TVM_FFI_CHECK_NE(x, y, ErrorKind)
#define CHECK_NOTNULL(x, ErrorKind) TVM_FFI_CHECK_NOTNULL(x, ErrorKind)

#define ICHECK(x) TVM_FFI_ICHECK(x)
#define ICHECK_LT(x, y) TVM_FFI_ICHECK_LT(x, y)
#define ICHECK_GT(x, y) TVM_FFI_ICHECK_GT(x, y)
#define ICHECK_LE(x, y) TVM_FFI_ICHECK_LE(x, y)
#define ICHECK_GE(x, y) TVM_FFI_ICHECK_GE(x, y)
#define ICHECK_EQ(x, y) TVM_FFI_ICHECK_EQ(x, y)
#define ICHECK_NE(x, y) TVM_FFI_ICHECK_NE(x, y)
#define ICHECK_NOTNULL(x) TVM_FFI_ICHECK_NOTNULL(x)

#define DCHECK(x) TVM_FFI_DCHECK(x)
#define DCHECK_LT(x, y) TVM_FFI_DCHECK_LT(x, y)
#define DCHECK_GT(x, y) TVM_FFI_DCHECK_GT(x, y)
#define DCHECK_LE(x, y) TVM_FFI_DCHECK_LE(x, y)
#define DCHECK_GE(x, y) TVM_FFI_DCHECK_GE(x, y)
#define DCHECK_EQ(x, y) TVM_FFI_DCHECK_EQ(x, y)
#define DCHECK_NE(x, y) TVM_FFI_DCHECK_NE(x, y)
#define DCHECK_NOTNULL(x) TVM_FFI_DCHECK_NOTNULL(x)

#endif // TILELANG_SUPPORT_CHECK_H_
