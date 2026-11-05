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
"""TileLang test capability decorators backed by the native provider.

The public decorator, marker and TVM_TEST_TARGETS contracts originate in
TVM's python/tvm/testing/utils.py at 4211874e9e4b8770fd20ad0d98ab80afd1df5028.
This bounded testing facade owns those contracts without importing its compiler.
Compile-only support deliberately does not require a locally available device.
"""

from __future__ import annotations

import importlib
import json
import os

import pytest
import tvm_ffi
from tvm_tirx import runtime


DEFAULT_TEST_TARGETS = [
    "llvm",
    "cuda",
    "nvptx",
    {"kind": "vulkan", "from_device": 0},
    "opencl",
    {"kind": "opencl", "device": "mali"},
    {"kind": "opencl", "device": "intel_graphics"},
    "metal",
    "rocm",
    "hexagon",
]


def _test_targets():
    requested = os.environ.get("TVM_TEST_TARGETS", "").strip()
    if not requested:
        return DEFAULT_TEST_TARGETS
    targets = []
    for item in requested.split(";"):
        item = item.strip()
        if item:
            target = json.loads(item) if item.startswith("{") else item
            if target not in targets:
                targets.append(target)
    return targets


def _device_exists(kind):
    try:
        return bool(tvm_ffi.device("cpu" if kind == "llvm" else kind).exist)
    except (RuntimeError, ValueError, TypeError):
        return False


def _runtime_enabled(kind):
    try:
        return bool(runtime.enabled(kind))
    except (RuntimeError, ValueError):
        return False


class Feature:
    """A named pytest capability with separate compile and execution checks.

    Checks return True when supported, False for the default skip explanation,
    or a string giving the reason. Parents contribute checks and markers.
    ``optional`` retains markers but performs no capability checks.
    """

    _all_features = {}

    def __init__(
        self,
        name,
        long_name=None,
        target_kind_enabled=None,
        compile_time_check=None,
        target_kind_hardware=None,
        run_time_check=None,
        parent_features=None,
    ):
        self.name = name
        self.long_name = long_name or name
        self.target_kind_enabled = target_kind_enabled
        self.compile_time_check = compile_time_check
        self.target_kind_hardware = target_kind_hardware
        self.run_time_check = run_time_check
        self.parent_features = [parent_features] if isinstance(parent_features, str) else list(parent_features or [])
        self._all_features[name] = self

    def _register_marker(self, config):
        config.addinivalue_line("markers", f"{self.name}: Mark a test as using {self.long_name}")

    def _uses_marks(self):
        for parent in self.parent_features:
            yield from self._all_features[parent]._uses_marks()
        yield getattr(pytest.mark, self.name)

    @staticmethod
    def _check_mark(check, reason):
        result = check()
        return pytest.mark.skipif(isinstance(result, str) or not result, reason=result if isinstance(result, str) else reason)

    def _compile_only_marks(self):
        for parent in self.parent_features:
            yield from self._all_features[parent]._compile_only_marks()
        if self.compile_time_check is not None:
            yield self._check_mark(self.compile_time_check, f"Compile-time support for {self.long_name} not present")
        if self.target_kind_enabled is not None:
            kind = self.target_kind_enabled.split()[0]
            kinds = [target["kind"] if isinstance(target, dict) else target.split()[0] for target in _test_targets()]
            yield pytest.mark.skipif(kind not in kinds, reason=f"{kind} tests disabled by TVM_TEST_TARGETS environment variable")

    def _run_only_marks(self):
        for parent in self.parent_features:
            yield from self._all_features[parent]._run_only_marks()
        if self.run_time_check is not None:
            yield self._check_mark(self.run_time_check, f"Run-time support for {self.long_name} not present")
        if self.target_kind_hardware is not None:
            yield pytest.mark.skipif(
                not _device_exists(self.target_kind_hardware), reason=f"No device exists for target {self.target_kind_hardware}"
            )

    def marks(self, support_required="compile-and-run"):
        if support_required not in ("compile-and-run", "compile-only", "optional"):
            raise ValueError(f"Unknown feature support type: {support_required}")
        marks = []
        if support_required == "compile-and-run":
            marks.extend(self._run_only_marks())
        if support_required != "optional":
            marks.extend(self._compile_only_marks())
        marks.extend(self._uses_marks())
        return marks

    def __call__(self, func=None, *, support_required="compile-and-run"):
        if support_required not in ("compile-and-run", "compile-only", "optional"):
            raise ValueError(f"Unknown feature support type: {support_required}")

        def decorate(function):
            for mark in self.marks(support_required):
                function = mark(function)
            return function

        return decorate if func is None else decorate(func)

    @classmethod
    def require(cls, name, support_required="compile-and-run"):
        return cls._all_features[name](support_required=support_required)


requires_gpu = Feature("gpu", run_time_check=lambda: any(_device_exists(kind) for kind in ("cuda", "rocm", "opencl", "metal", "vulkan")))
requires_llvm = Feature(
    "llvm",
    "LLVM",
    target_kind_enabled="llvm",
    compile_time_check=lambda: _runtime_enabled("llvm"),
    run_time_check=lambda: _runtime_enabled("llvm"),
    target_kind_hardware="llvm",
)


def _gpu_feature(kind, name):
    return Feature(
        kind,
        name,
        target_kind_enabled=kind,
        compile_time_check=lambda: _runtime_enabled(kind),
        run_time_check=lambda: _runtime_enabled(kind),
        target_kind_hardware=kind,
        parent_features="gpu",
    )


requires_cuda = _gpu_feature("cuda", "CUDA")
requires_rocm = _gpu_feature("rocm", "ROCm")
requires_metal = _gpu_feature("metal", "Metal")


def _compose(args, decs):
    if not args:
        return decs
    function = args[0]
    for decorator in reversed(decs):
        function = decorator(function)
    return function


def requires_package(*packages):
    def available(package):
        try:
            importlib.import_module(package)
            return True
        except ImportError:
            return False

    marks = [pytest.mark.skipif(not available(package), reason=f"Cannot import '{package}'") for package in packages]

    def decorate(function):
        for mark in marks:
            function = mark(function)
        return function

    return decorate
