"""TileLang's const-correct asynchronous copy lowering."""

import tvm_ffi
from tvm_tirx.backend.cuda.codegen.registry import get_codegen as native_get_codegen
from tvm_tirx.backend.cuda.op import func_call

from .cuda_registry import register_codegen


@register_codegen("s_tir.cp_async_raw", namespace="tirx")
def lower_cp_async(*args, attrs=None):
    if len(args) in (5, 6):
        # Legacy InjectPTXAsyncCopy emission: (dst_ptr, dst_off, src_ptr,
        # src_off, cp_size [, predicate]). Offsets are element indices into
        # the typed buffers (the pass uses index_factor=1 except for the
        # shared.dyn-merged byte-buffer path). Emit a C helper that scales
        # the offset by the buffer element size, then runs cp.async.
        #
        # PTX plain form for both .ca and .cg is just
        # ``cp.async.<v>.shared.global [dst], [src], cp_size;`` — three
        # operands, no trailing src-size / cache-policy.
        dst_ptr_in, dst_offset, src_ptr_in, src_offset, cp_size = args[:5]
        predicate = args[5] if len(args) == 6 else -1
        cp_size_v = int(cp_size)
        ca_or_cg = "cg" if cp_size_v == 16 else "ca"

        # Recover the per-side element dtype from each pointer's type
        # type (Var has ty = PtrType(PrimType(dtype))).
        # InjectPTXAsyncCopy emits offsets in element-units of each side's
        # buffer dtype (dst gets dst_offset * src_elem_size only when dst is a
        # merged shared.dyn byte buffer, in which case dst_elem_dtype is uint8
        # and the resulting scale-by-1 is a no-op).
        def _elem_bytes(ptr):
            ta = getattr(ptr, "ty", None)
            if ta is None or getattr(ta, "element_type", None) is None:
                return 1
            et = ta.element_type
            if not hasattr(et, "dtype"):
                return 1
            bits = tvm_ffi.dtype(str(et.dtype)).bits
            assert bits % 8 == 0, f"non-byte element dtype: {et.dtype}"
            return bits // 8

        dst_elem_bytes = _elem_bytes(dst_ptr_in)
        src_elem_bytes = _elem_bytes(src_ptr_in)
        has_predicate = not (
            (isinstance(predicate, int) and predicate == -1) or (hasattr(predicate, "value") and int(predicate.value) == -1)
        )

        def _scale(n):
            return "" if n == 1 else f" * {n}"

        dst_scale = _scale(dst_elem_bytes)
        src_scale = _scale(src_elem_bytes)
        if has_predicate:
            func_name = f"ptx_cp_async_legacy_pred_{ca_or_cg}_{cp_size_v}_{dst_elem_bytes}_{src_elem_bytes}"
            if cp_size_v == 4:
                zero_fill = '    " @!p st.shared.u32 [%0], {%4};\\n"\n'
            elif cp_size_v == 8:
                zero_fill = '    " @!p st.shared.v2.u32 [%0], {%4, %4};\\n"\n'
            elif cp_size_v == 16:
                zero_fill = '    " @!p st.shared.v4.u32 [%0], {%4, %4, %4, %4};\\n"\n'
            else:
                raise ValueError(f"unsupported legacy predicated cp.async size: {cp_size_v}")
            body = (
                f"  uint8_t* dst_p = (uint8_t*)dst + dst_off{dst_scale};\n"
                f"  const uint8_t* src_p = (const uint8_t*)src + src_off{src_scale};\n"
                "  unsigned int dst_addr = __cvta_generic_to_shared(dst_p);\n"
                "  __asm__ __volatile__(\n"
                '    "{\\n"\n'
                '    " .reg .pred p;\\n"\n'
                '    " setp.eq.u32 p, %3, 1;\\n"\n'
                f'    " @p cp.async.{ca_or_cg}.shared.global'
                ' [%0], [%1], %2;\\n"\n'
                f"{zero_fill}"
                '    "}\\n"\n'
                f'    :: "r"(dst_addr), "l"(src_p), "n"({cp_size_v}), "r"(predicate), "r"(0)\n'
                "  );"
            )
            source_code = (
                f"\n__forceinline__ __device__ void {func_name}"
                "(void* dst, int dst_off, const void* src, int src_off, int predicate) {\n"
                f"{body}\n"
                "}\n"
            )
            return func_call(
                func_name,
                dst_ptr_in,
                dst_offset,
                src_ptr_in,
                src_offset,
                predicate,
                source_code,
            )
        # No predicate — plain cp.async.
        func_name = f"ptx_cp_async_legacy_{ca_or_cg}_{cp_size_v}_{dst_elem_bytes}_{src_elem_bytes}"
        body = (
            f"  uint8_t* dst_p = (uint8_t*)dst + dst_off{dst_scale};\n"
            f"  const uint8_t* src_p = (const uint8_t*)src + src_off{src_scale};\n"
            "  unsigned int dst_addr = __cvta_generic_to_shared(dst_p);\n"
            f'  asm volatile("cp.async.{ca_or_cg}.shared.global'
            ' [%0], [%1], %2;"\n'
            f'    :: "r"(dst_addr), "l"(src_p), "n"({cp_size_v}));'
        )
        source_code = (
            f"\n__forceinline__ __device__ void {func_name}(void* dst, int dst_off, const void* src, int src_off) {{\n{body}\n}}\n"
        )
        return func_call(
            func_name,
            dst_ptr_in,
            dst_offset,
            src_ptr_in,
            src_offset,
            source_code,
        )
    else:
        raise ValueError(f"cp_async_raw codegen expects 5/6 args, got {len(args)}")
    return native_get_codegen("tirx.s_tir.cp_async_raw")(args)
