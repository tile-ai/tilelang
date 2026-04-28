from __future__ import annotations

from tilelang.backend.base import FFIBuilderRef


HIP_SOURCE_BUILDER = FFIBuilderRef("target.build.tilelang_hip_without_compile", "source")
HIP_COMPILED_BUILDER = FFIBuilderRef("target.build.tilelang_hip", "compiled")
