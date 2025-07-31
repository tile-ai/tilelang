"""FFI APIs for tilelang"""

import tvm.ffi

# TVM_REGISTER_GLOBAL("tl.name").set_body_typed(func);
tvm.ffi._init_api("tl.transform", __name__)  # pylint: disable=protected-access
