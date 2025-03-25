# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""FFI APIs for tilelang"""

import tvm._ffi

# TVM_REGISTER_GLOBAL("tl.name").set_body_typed(func);
tvm._ffi._init_api("tl.transform", __name__)  # pylint: disable=protected-access
