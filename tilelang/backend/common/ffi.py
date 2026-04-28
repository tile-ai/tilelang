from __future__ import annotations

from tilelang.backend.base import FFIBuilderRef


C_SOURCE_BUILDER = FFIBuilderRef("target.build.tilelang_c", "source")
LLVM_SOURCE_BUILDER = FFIBuilderRef("target.build.llvm", "source")
