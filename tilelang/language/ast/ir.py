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
# This file is modified from the original version,
# which is part of the TVM project (https://tvm.apache.org/).
# ruff: noqa
"""Historical builder import path for TileLang's native construction boundary.

The old parser-specific FFI constructors are replaced by the shared typed
builder. User code keeps the language spelling; returned nodes are native IR.
"""

from tilelang.language import _construction
from tilelang.language import _source_scope
from tilelang.language import _tensor_spec
from tilelang.language.tir import common as _common
from tvm_tirx import ir as _ir


from tilelang.language.tir.common import *  # noqa: F401,F403
from tilelang.language.dtypes import bool as boolean
from tilelang.language.dtypes import dtype
from tvm_tirx.target import Target as target

buffer_decl = _tensor_spec.buffer
block_attr = _source_scope.sblock_attr
ptr = _construction.handle
buffer_var = _construction.handle


def var(dtype, name=""):
    return _ir.Var(name, dtype)


__all__ = list(
    dict.fromkeys(
        [
            *_common.__all__,
            "boolean",
            "dtype",
            "ptr",
            "buffer_var",
            "var",
            "buffer_decl",
            "block_attr",
            "target",
        ]
    )
)
