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
"""Wrapping functions to bridge frameworks with DLPack support to TVM"""

from tvm import runtime


def convert_func(tvm_func, tensor_type, to_dlpack_func):
    """Convert a tvm function into one that accepts a tensor from another
       framework, provided the other framework supports DLPACK

    Parameters
    ----------
    tvm_func: Function
        Built tvm function operating on arrays

    tensor_type: Type
        Type of the tensors of the target framework

    to_dlpack_func: Function
        Function to convert the source tensors to DLPACK
    """
    assert callable(tvm_func)
    import torch

    float8_dtype_map = {
        torch.float8_e4m3fn: "float8_e4m3",
        torch.float8_e4m3fnuz: "float8_e4m3fnuz",
        torch.float8_e5m2: "float8_e5m2",
        torch.float8_e5m2fnuz: "float8_e5m2",
    }

    def adapt_tensor(arg):
        if isinstance(arg, tensor_type):
            if arg.dtype in {torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz}:
                return runtime.from_dlpack(to_dlpack_func(arg.view(torch.int8)))._create_view(arg.shape, dtype=float8_dtype_map[arg.dtype])
            return runtime.from_dlpack(to_dlpack_func(arg))
        return arg

    def _wrapper(*args):
        args = tuple(adapt_tensor(arg) for arg in args)
        return tvm_func(*args)

    return _wrapper
