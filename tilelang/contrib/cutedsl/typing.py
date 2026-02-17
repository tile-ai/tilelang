from cutlass._mlir.dialects import nvvm

type_map = {
    "int8": nvvm.WGMMATypes.s8,
    "int32": nvvm.WGMMATypes.s32,
    "uint8": nvvm.WGMMATypes.u8,
    "float16": nvvm.WGMMATypes.f16,
    "fp16": nvvm.WGMMATypes.f16,
    "bfloat16": nvvm.WGMMATypes.bf16,
    "bf16": nvvm.WGMMATypes.bf16,
    "float32": nvvm.WGMMATypes.f32,
    "fp32": nvvm.WGMMATypes.f32,
    "tf32": nvvm.WGMMATypes.tf32,
    "float8_e4m3": nvvm.WGMMATypes.e4m3,
    "float8_e5m2": nvvm.WGMMATypes.e5m2,
    "float8_e4m3fn": nvvm.WGMMATypes.e4m3,
    "e4m3": nvvm.WGMMATypes.e4m3,
    "e5m2": nvvm.WGMMATypes.e5m2,
}
