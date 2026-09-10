"""Example: Using postproc callbacks to modify generated Ascend code.

This example demonstrates how to register a post-processing callback that
intercepts and modifies generated backend source code before compilation. This
is useful for:
- Debugging generated code
- Injecting custom pragmas or directives
- Adding custom headers or comments
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.engine.callback import register_ascend_postproc_callback


def vector_add(N):
    """Simple vector add kernel for callback demonstration."""

    @T.prim_func
    def main(
        A: T.Buffer((N,), "float32"),
        B: T.Buffer((N,), "float32"),
        C: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(1) as _, T.SimtVF(threads=128):
            for i in T.Parallel(N):
                C[i] = A[i] + B[i]

    return main


CUSTOM_MARKER = "// [POSTPROC] Modified by register_ascend_postproc_callback"


@register_ascend_postproc_callback
def my_ascend_family_postproc(code, target):
    """Post-process generated AscendC source."""
    print(f"\n--- Ascend postproc callback invoked (code length: {len(code)} chars) ---")
    return CUSTOM_MARKER + "\n" + code


if __name__ == "__main__":
    N = 1024

    print(f"Compiling vector_add kernel (N={N}) with postproc callback...")
    program = vector_add(N)

    kernel = tilelang.compile(program, target="ascend", out_idx=-1)

    source = kernel.get_kernel_source()

    assert CUSTOM_MARKER in source, (
        "Expected marker in generated source, but it was not found.\nThis means the postproc callback was not invoked."
    )
    print("\n[ok] Postproc callback verification passed.")
