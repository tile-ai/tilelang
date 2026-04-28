from __future__ import annotations

from tilelang.backend.base import FFIBuilderRef


CUDA_SOURCE_BUILDER = FFIBuilderRef("target.build.tilelang_cuda_without_compile", "source")
CUDA_COMPILED_BUILDER = FFIBuilderRef("target.build.tilelang_cuda", "compiled")
CUTEDSL_SOURCE_BUILDER = FFIBuilderRef("target.build.tilelang_cutedsl_without_compile", "source")
