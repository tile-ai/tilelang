"""Pytest capability markers backed by standalone TIRx and device runtimes."""

import importlib.util

import pytest
import torch
import tvm_ffi


def _compose(functions, decorators):
    for decorator in decorators:
        functions = [decorator(function) for function in functions]
    return functions[0] if len(functions) == 1 else functions


class Feature:
    _all_features = {}

    def __init__(self, name, description, *, compile_time_check, run_time_check=None):
        self.name = name
        self.description = description
        self.compile_time_check = compile_time_check
        self.run_time_check = run_time_check
        self._all_features[name] = self

    def marks(self, support_required="compile-and-run"):
        if support_required not in ("compile-and-run", "compile-only", "optional"):
            raise ValueError(f"Unknown feature support level: {support_required}")
        marks = [getattr(pytest.mark, self.name)]
        if support_required == "optional":
            return marks
        available = bool(self.compile_time_check())
        if support_required == "compile-and-run" and self.run_time_check is not None:
            available = available and bool(self.run_time_check())
        marks.append(pytest.mark.skipif(not available, reason=f"Requires {self.description} ({support_required})"))
        return marks

    def __call__(self, function=None, *, support_required="compile-and-run"):
        def decorate(function):
            return _compose([function], self.marks(support_required))

        return decorate if function is None else decorate(function)


def _has_codegen(name):
    return tvm_ffi.get_global_func(f"target.build.{name}", allow_missing=True) is not None


requires_cuda = Feature(
    "cuda",
    "CUDA",
    compile_time_check=lambda: _has_codegen("tilelang_cuda"),
    run_time_check=lambda: torch.cuda.is_available() and not torch.version.hip,
)
requires_rocm = Feature(
    "rocm",
    "ROCm",
    compile_time_check=lambda: _has_codegen("tilelang_hip"),
    run_time_check=lambda: torch.cuda.is_available() and bool(torch.version.hip),
)
requires_metal = Feature(
    "metal",
    "Metal",
    compile_time_check=lambda: _has_codegen("tilelang_metal"),
    run_time_check=lambda: torch.backends.mps.is_available(),
)
requires_llvm = Feature("llvm", "LLVM", compile_time_check=lambda: _has_codegen("llvm"))


def requires_package(package):
    try:
        available = importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        available = False
    return pytest.mark.skipif(not available, reason=f"Requires Python package {package}")
