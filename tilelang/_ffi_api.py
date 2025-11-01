"""FFI APIs for tilelang"""

import tvm_ffi

# TVM_REGISTER_GLOBAL("tl.name").set_body_typed(func);
tvm_ffi.init_ffi_api("tl", __name__)
