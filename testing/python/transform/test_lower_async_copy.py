"""Test for LowerAsyncCopy pass."""

from tilelang import tvm as tvm
from tvm.script import tir as T
from tilelang.transform.pipeline import LowerAsyncCopy


@T.prim_func
def vectorized_copy_kernel(
    A: T.Buffer((1024, 1024), "float16"),
    B: T.Buffer((1024, 1024), "float16"),
):
    T.func_attr({"target": T.target("cuda")})
    with T.block("root"):
        # Allocate shared memory buffer
        A_shared = T.alloc_buffer((128, 32), "float16", scope="shared")

        # Thread binding
        thread_binding = T.launch_thread("threadIdx.x", 128)

        # Vectorized copy from global to shared
        for i in range(4):
            for vec in T.vectorized(8):
                A_shared[(i * 8 + vec) // 8 * 32 + thread_binding // 4, thread_binding % 4 * 8 + (i * 8 + vec) % 8] = A[
                    (i * 8 + vec) // 8 * 32 + thread_binding // 4, thread_binding % 4 * 8 + (i * 8 + vec) % 8
                ]

        # Some computation
        for i, j in T.grid(128, 32):
            B[i, j] = A_shared[i, j] * T.float16(2.0)


def test_lower_async_copy_basic():
    """Test basic vectorized copy lowering."""
    print("=" * 60)
    print("Original function:")
    print("=" * 60)
    print(vectorized_copy_kernel)

    mod = tvm.IRModule.from_expr(vectorized_copy_kernel)

    # Apply the pass
    with tvm.target.Target("cuda"):
        mod = LowerAsyncCopy(verbose=True)(mod)

    print("\n" + "=" * 60)
    print("After LowerAsyncCopy:")
    print("=" * 60)
    print(mod)


@T.prim_func
def simple_vectorized_copy(
    A: T.Buffer((128,), "float16"),
    B: T.Buffer((128,), "float16"),
):
    """Simple 1D vectorized copy test."""
    T.func_attr({"target": T.target("cuda")})
    with T.block("root"):
        A_shared = T.alloc_buffer((128,), "float16", scope="shared")

        tx = T.launch_thread("threadIdx.x", 16)

        # Each thread copies 8 elements
        for vec in T.vectorized(8):
            A_shared[tx * 8 + vec] = A[tx * 8 + vec]

        # Sync and use
        for i in range(8):
            B[tx * 8 + i] = A_shared[tx * 8 + i]


def test_simple_vectorized_copy():
    """Test simple 1D vectorized copy."""
    print("\n" + "=" * 60)
    print("Simple 1D vectorized copy test:")
    print("=" * 60)
    print(simple_vectorized_copy)

    mod = tvm.IRModule.from_expr(simple_vectorized_copy)

    with tvm.target.Target("cuda"):
        mod = LowerAsyncCopy(verbose=True)(mod)

    print("\n" + "=" * 60)
    print("After LowerAsyncCopy:")
    print("=" * 60)
    print(mod)


if __name__ == "__main__":
    test_simple_vectorized_copy()
    test_lower_async_copy_basic()
