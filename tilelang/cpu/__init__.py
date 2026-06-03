from tilelang.backend.target import register_target_kind

from . import pipeline  # noqa: F401
from . import op  # noqa: F401


register_target_kind("cpu", tvm_kind="llvm", override=True)
register_target_kind("llvm", tvm_kind="llvm", override=True)
register_target_kind("c", tvm_kind="c", override=True)
