"""FFI APIs for the CuTe layout-algebra namespace.

The C++ side registers all CuTe-layout functions under the dotted ``tl.cute.*``
namespace (which the flat ``tilelang._ffi_api`` module skips, since its remaining
name still contains a dot). Initializing a module against the ``tl.cute`` prefix
strips it cleanly, surfacing ``coalesce``, ``right_inverse``, ``composition``,
``make_layout`` etc. as plain attributes (``_cute_ffi_api.coalesce(...)``).
"""

import tvm_ffi

tvm_ffi.init_ffi_api("tl.cute", __name__)
