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
"""Each CPU codegen entry prepares its IR once, without requiring LLVM."""

import importlib

from tvm_tirx.target import Target

import tilelang
from tilelang.instrumentation import PassInstrumentationTool, compile_pass_instrumentation


class CodegenObserver(PassInstrumentationTool):
    def __init__(self):
        self.calls = []
        self.results = []

    def run_codegen(self, event, next_call):
        self.calls.append(event)
        result = next_call()
        self.results.append(result)
        return result


def test_cpu_pipeline_prepares_each_codegen_once(monkeypatch):
    import torch

    import tilelang.cpu.language as T
    from tilelang.backend.module import BackendContext

    lower_module = importlib.import_module("tilelang.engine.lower")
    original_device = lower_module._prepare_device_codegen_mod
    original_host = BackendContext.preprocess_host_codegen
    preparations = {"device": 0, "host": 0}

    def device(mod):
        preparations["device"] += 1
        return original_device(mod)

    def host(context, mod):
        preparations["host"] += 1
        return original_host(context, mod)

    monkeypatch.setattr(lower_module, "_prepare_device_codegen_mod", device)
    monkeypatch.setattr(BackendContext, "preprocess_host_codegen", host)

    @T.prim_func
    def plus_one(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1):
            for i in T.Parallel(4):
                B[i] = A[i] + 1

    observer = CodegenObserver()
    target = Target("c")
    with compile_pass_instrumentation(tools=[observer], include_default_tools=False, reuse_existing=False):
        kernel = tilelang.compile(plus_one, out_idx=1, target=target, target_host=target, execution_backend="tvm_ffi")
    assert preparations == {"device": 1, "host": 1}
    assert [(event.name, event.role) for event in observer.calls] == [
        ("target.build.tilelang_c", "device"),
        ("target.build.tilelang_c_host", "host"),
    ]
    inputs = torch.tensor([-2, 0, 9, 123], dtype=torch.int32)
    torch.testing.assert_close(kernel(inputs), inputs + 1, rtol=0, atol=0)
