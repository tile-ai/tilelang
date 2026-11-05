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
"""TileLang F1 parser implementation exports.

Private legacy TVMScript visitor hooks are replaced by the owning Python AST
mutator and native Builder. Importing this module does not alter target parser
registries; extensions must use the explicit new interfaces below.
"""

from tilelang.language.eager.ast import BaseBuilder, DSLMutator, IRGenerator, mutate
from tilelang.language.eager.builder import Builder

__all__ = ["BaseBuilder", "Builder", "DSLMutator", "IRGenerator", "mutate"]
