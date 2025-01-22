# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing

def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(
        mod["main"], transformed.with_attr("global_symbol", "main"), True
    )


@T.prim_func
def trivial_pipeline(A: T.Buffer((16, 1), "float32"), C: T.Buffer((16, 1), "float32")):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0, 1, annotations={"software_pipeline_stage": [0, 1], "software_pipeline_order": [0, 1]}
        ):
            with T.block():
                T.reads(A[tx, i])
                T.writes(C[tx, i])
                B = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with T.block():
                    T.reads(B[tx, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = B[tx, 0] + T.float32(1)

@T.prim_func
def transformed_trivial_pipeline(
    A: T.Buffer((16, 1), "float32"), C: T.Buffer((16, 1), "float32")
) -> None:
    for tx in T.thread_binding(16, thread="threadIdx.x"):
        with T.block():
            T.reads(A[tx, 0])
            T.writes(C[tx, 0])
            B = T.alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
            with T.block():
                T.reads(A[tx, 0])
                T.writes(B[0, tx, 0])
                B[0, tx, 0] = A[tx, 0] * T.float32(2)
            with T.block():
                T.reads(A[tx, 1:1], B[0:2, tx, 0])
                T.writes(B[1:1, tx, 0], C[tx, 0:0])
                for i in range(0):
                    with T.block(""):
                        T.reads(A[tx, i + 1])
                        T.writes(B[i + 1, tx, 0])
                        B[i + 1, tx, 0] = A[tx, i + 1] * T.float32(2)
                    with T.block(""):
                        T.reads(B[i, tx, 0])
                        T.writes(C[tx, i])
                        C[tx, i] = B[i, tx, 0] + T.float32(1)
            with T.block():
                T.reads(B[0, tx, 0])
                T.writes(C[tx, 0])
                C[tx, 0] = B[0, tx, 0] + T.float32(1)

def test_trivial_pipeline():
    _check(trivial_pipeline, transformed_trivial_pipeline)

if __name__ == "__main__":
    tilelang.testing.main()
