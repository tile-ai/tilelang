# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.testing
from tvm.script import tir as T


class BaseCompare(tilelang.testing.CompareBeforeAfter):
    transform = tilelang.transform.AnnotateDeviceRegions()


class TestAnnotateThreadExtent(BaseCompare):
    """Annotation inserted at the "thread_extent" attribute"""

    def before(A: T.Buffer(16, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        i = T.launch_thread("threadIdx.x", 16)
        A[i] = 0.0

    def expected(A: T.Buffer(16, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        T.attr(T.target("cuda"), "target", 0)
        i = T.launch_thread("threadIdx.x", 16)
        A[i] = 0.0


class TestAnnotateDeviceScope(BaseCompare):
    """Annotation inserted at the "device_scope" attribute"""

    def before(A: T.Buffer(1, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        T.attr(0, "device_scope", 0)
        A[0] = 0.0

    def expected(A: T.Buffer(1, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        T.attr(T.target("cuda"), "target", 0)
        T.attr(0, "device_scope", 0)
        A[0] = 0.0


if __name__ == "__main__":
    tilelang.testing.main()
