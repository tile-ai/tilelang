# Testing library and framework: pytest
# These tests target memory allocation helpers defined in testing/python/language/test_allocate.py
# We validate buffer scopes, shapes, and dtypes.
import importlib.util
import os
import sys
import types
import pytest

try:
    # Prefer normal import if repository exposes a package path; fall back to path import.
    from tvm.script import tir as T
except ImportError:  # pragma: no cover - if tvm not installed in CI, mark as skip
    T = None

MODULE_FS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "language",
    "test_allocate.py",
)

def _import_by_path(path: str, module_name: str = "allocate_helpers") -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load spec for {module_name} at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

pytestmark = pytest.mark.skipif("T" not in globals() or T is None, reason="tvm is required for these tests")

def load_module():
    # The module under test resides at testing/python/language/test_allocate.py
    assert os.path.exists(MODULE_FS_PATH), f"Expected file not found: {MODULE_FS_PATH}"
    mod = _import_by_path(MODULE_FS_PATH)
    # Sanity: required functions exist
    for name in ("alloc_shared", "alloc_local", "alloc_fragment", "alloc_var", "alloc_barrier"):
        assert hasattr(mod, name), f"Missing function {name} in module"
    return mod

class TestAllocShared:
    def test_default_scope_shared_dyn_non_bool(self):
        mod = load_module()
        buf = mod.alloc_shared((16, 32), "float32")
        assert hasattr(buf, "scope")
        assert buf.scope == "shared.dyn"
        assert buf.dtype == "float32"
        assert len(buf.shape) == 2
        # Shape elements in TIR are PrimExprs; compare to int value when possible
        assert int(buf.shape[0]) == 16
        assert int(buf.shape[1]) == 32

    def test_bool_dtype_forces_shared_scope(self):
        mod = load_module()
        # Even if we pass a custom scope, dtype "bool" should force "shared"
        buf = mod.alloc_shared((8,), "bool", scope="shared.dyn")
        assert buf.scope == "shared"
        assert buf.dtype == "bool"
        assert len(buf.shape) == 1
        assert int(buf.shape[0]) == 8

    def test_custom_scope_respected_for_non_bool(self):
        mod = load_module()
        buf = mod.alloc_shared((4, 4), "int32", scope="shared")
        assert buf.scope == "shared"
        assert buf.dtype == "int32"
        assert len(buf.shape) == 2

class TestAllocLocal:
    def test_default_scope_local(self):
        mod = load_module()
        buf = mod.alloc_local((3, 5, 7), "float16")
        assert buf.scope == "local"
        assert buf.dtype == "float16"
        assert len(buf.shape) == 3
        assert int(buf.shape[2]) == 7

    def test_custom_scope_override(self):
        mod = load_module()
        buf = mod.alloc_local((1,), "int8", scope="local.array")
        assert buf.scope == "local.array"
        assert buf.dtype == "int8"
        assert len(buf.shape) == 1
        assert int(buf.shape[0]) == 1

class TestAllocFragment:
    def test_default_scope_local_fragment(self):
        mod = load_module()
        buf = mod.alloc_fragment((2, 2, 2), "float32")
        assert buf.scope == "local.fragment"
        assert buf.dtype == "float32"
        assert len(buf.shape) == 3

    def test_custom_scope_override(self):
        mod = load_module()
        buf = mod.alloc_fragment((9,), "int16", scope="wmma.matrix_a")
        assert buf.scope == "wmma.matrix_a"
        assert buf.dtype == "int16"
        assert len(buf.shape) == 1
        assert int(buf.shape[0]) == 9

class TestAllocVar:
    def test_default_single_element_var(self):
        mod = load_module()
        buf = mod.alloc_var("float64")
        assert buf.scope == "local.var"
        assert buf.dtype == "float64"
        assert len(buf.shape) == 1
        assert int(buf.shape[0]) == 1

    def test_custom_scope_override(self):
        mod = load_module()
        buf = mod.alloc_var("int32", scope="local.scalar")
        assert buf.scope == "local.scalar"
        assert buf.dtype == "int32"
        assert len(buf.shape) == 1
        assert int(buf.shape[0]) == 1

class TestAllocBarrier:
    @pytest.mark.parametrize("arrive_count", [1, 32, 128])
    def test_barrier_shape_and_scope(self, arrive_count):
        mod = load_module()
        buf = mod.alloc_barrier(arrive_count)
        assert buf.scope == "shared.barrier"
        assert buf.dtype == "uint64"
        assert len(buf.shape) == 1
        assert int(buf.shape[0]) == arrive_count