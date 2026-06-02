from __future__ import annotations

from tilelang.backend.pass_pipeline.pipeline import PassPipeline, register_pipeline
from tilelang.cpu.pipeline import CPUPassPipelineBody


register_pipeline(PassPipeline("webgpu", CPUPassPipelineBody))
