from __future__ import annotations

from tilelang.backend.base import FFIBuilderRef


METAL_SOURCE_BUILDER = FFIBuilderRef("target.build.metal", "source")
METAL_COMPILED_BUILDER = FFIBuilderRef("target.build.metal", "compiled")
