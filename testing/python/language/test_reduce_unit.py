import sys
import types
import builtins
import contextlib
import importlib
import pytest

# --- Lightweight doubles for tvm.tir and tilelang.language ---

class _FakeBuffer:
    def __init__(self, shape, dtype="float32", scope_name="global"):
        self.shape = list(shape)
        self.dtype = dtype
        self._scope_name = scope_name

    def access_ptr(self, mode):
        # Return a simple token capturing mode for assertion
        return f"ptr[{mode}]"

    def scope(self):
        return self._scope_name


class _FakeOp:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def get(name):
        return _FakeOp(name)


def _fake_call_intrin(rt, op, *args):
    # Return a tuple we can assert against without requiring TVM runtime
    return ("intrin", rt, getattr(op, "name", op), *args)


def install_fake_tvm():
    tvm = types.ModuleType("tvm")
    tir = types.ModuleType("tvm.tir")
    tir.Buffer = _FakeBuffer
    tir.call_intrin = _fake_call_intrin
    tir.op = types.SimpleNamespace(Op=_FakeOp)
    tvm.tir = tir
    sys.modules["tvm"] = tvm
    sys.modules["tvm.tir"] = tir


def install_fake_tilelang_language():
    # Provide stubs for copy, macro, alloc_shared
    lang = types.ModuleType("tilelang.language")

    def copy(src, dst):  # record copies by annotating dst
        dst._copied_from = getattr(src, "_copied_from", "src")
        return ("copy", src, dst)

    def alloc_shared(shape, dtype, scope):
        return _FakeBuffer(shape, dtype=dtype, scope_name=scope)

    def macro(fn=None):
        # Simple decorator passthrough
        def wrapper(f):
            f.__is_macro__ = True
            return f
        return wrapper if fn is None else wrapper(fn)

    lang.copy = copy
    lang.alloc_shared = alloc_shared
    lang.macro = macro
    sys.modules["tilelang.language"] = lang


@contextlib.contextmanager
def fake_env():
    # Install fakes then yield; cleanup on exit
    bak = {k: sys.modules.get(k) for k in ["tvm", "tvm.tir", "tilelang.language"]}
    install_fake_tvm()
    install_fake_tilelang_language()
    try:
        yield
    finally:
        # restore
        for k, v in bak.items():
            if v is None and k in sys.modules:
                del sys.modules[k]
            elif v is not None:
                sys.modules[k] = v


def import_under_test():
    # Import the module under test from the provided file path by using importlib.machinery
    import importlib.util
    import pathlib

    mod_name = "tl_reduce_under_test"
    file_path = pathlib.Path("testing/python/language/test_reduce.py")
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -------------------- Tests --------------------

def test__legalize_dim_positive_and_negative_indices():
    with fake_env():
        m = import_under_test()
        buf = m.tir.Buffer((2, 3, 4))
        # positive index unchanged
        assert m._legalize_dim(buf, 1) == 1  # noqa: B101
        # -1 -> last dim
        assert m._legalize_dim(buf, -1) == 2  # noqa: B101
        # -3 -> first dim
        assert m._legalize_dim(buf, -3) == 0  # noqa: B101


def test_reduce_rejects_invalid_out_shape_with_clear_message():
    with fake_env():
        m = import_under_test()
        buf = m.tir.Buffer((2, 3, 4))
        # out shape must be [2,4] or [2,1,4] for dim=1
        bad_out = m.tir.Buffer((2, 2, 4))
        with pytest.raises(ValueError) as ei:
            m.reduce(buf, bad_out, "sum", 1, True)
        msg = str(ei.value)
        assert "Invalid reduce output shape" in msg  # noqa: B101
        assert str(buf.shape) in msg  # noqa: B101
        assert "dim is 1" in msg  # noqa: B101
        assert str(bad_out.shape) in msg  # noqa: B101
        assert "[2, 4]" in msg and "[2, 1, 4]" in msg  # noqa: B101


@pytest.mark.parametrize("dim,out_shape", [
    (1, (2, 4)),      # squeeze reduced dim
    (1, (2, 1, 4)),   # keep reduced dim with size 1
])
@pytest.mark.parametrize("rtype", ["max", "min", "sum", "abssum", "absmax"])
@pytest.mark.parametrize("clear", [True, False])
def test_reduce_passes_through_to_intrin_with_expected_args(dim, out_shape, rtype, clear):
    with fake_env():
        m = import_under_test()
        X, d, Y = 2, 3, 4
        buf = m.tir.Buffer((X, d, Y))
        out = m.tir.Buffer(out_shape)
        result = m.reduce(buf, out, rtype, dim, clear)
        # Validate call_intrin contract
        assert result[0] == "intrin"  # noqa: B101
        assert result[1] == "handle"  # noqa: B101
        assert result[2] == "tl.reduce"  # noqa: B101
        # Buffer pointers and args
        assert result[3] == "ptr[r]"  # noqa: B101
        assert result[4] == "ptr[w]"  # noqa: B101
        assert result[5] == rtype  # noqa: B101
        assert result[6] == dim  # noqa: B101
        assert result[7] is clear  # noqa: B101


def test_reduce_wrappers_legalize_dim_and_forward_correct_reduce_type(monkeypatch):
    with fake_env():
        m = import_under_test()
        calls = []

        def fake_reduce(buffer, out, reduce_type, dim, clear):
            calls.append((buffer.shape, out.shape, reduce_type, dim, clear))
            return "ok"

        monkeypatch.setattr(m, "reduce", fake_reduce)
        buf = m.tir.Buffer((5, 6, 7))
        # Case: out squeezes last dim
        # Use dim explicit to avoid shape mismatch: for dim=1 on shape (5,6,7), valid outs are (5,7) or (5,1,7)
        out1 = m.tir.Buffer((5, 7))
        out2 = m.tir.Buffer((5, 1, 7))
        assert m.reduce_max(buf, out1, dim=1, clear=True) == "ok"  # noqa: B101
        assert m.reduce_min(buf, out2, dim=1, clear=False) == "ok"  # noqa: B101
        assert m.reduce_sum(buf, out1, dim=-2, clear=True) == "ok"  # noqa: B101
        assert m.reduce_abssum(buf, out2, dim=-2) == "ok"  # noqa: B101
        assert m.reduce_absmax(buf, out1, dim=1, clear=False) == "ok"  # noqa: B101
        # Verify calls (reduce_type and legalized dim)
        assert calls == [  # noqa: B101
            ((5, 6, 7), (5, 7),   "max",   1, True),
            ((5, 6, 7), (5, 1, 7), "min",   1, False),
            ((5, 6, 7), (5, 7),   "sum",   1, True),
            ((5, 6, 7), (5, 1, 7), "abssum", 1, True),
            ((5, 6, 7), (5, 7),   "absmax", 1, False),
        ]


def test_cumsum_dim_bounds_and_negative_index_handling():
    with fake_env():
        m = import_under_test()
        src = m.tir.Buffer((2, 3, 4))
        # out-of-bounds positive
        with pytest.raises(ValueError):
            m.cumsum(src, dim=3)
        with pytest.raises(ValueError):
            m.cumsum(src, dim=-4)
        # Negative wraps
        # Should not raise for dim=-1
        m.cumsum(src, dim=-1)


def test_cumsum_defaults_and_direct_intrin_path():
    with fake_env():
        m = import_under_test()
        src = m.tir.Buffer((2, 3, 4))
        # dst defaults to src when None
        res = m.cumsum(src, dst=None, dim=2, reverse=False)
        assert res[0] == "intrin"  # noqa: B101
        assert res[2] == "tl.cumsum"  # noqa: B101
        # pointers and args
        assert res[3] == "ptr[r]"  # noqa: B101
        assert res[4] == "ptr[w]"  # noqa: B101
        assert res[5:] == (2, False)  # noqa: B101


def test_cumsum_fragment_branch_uses_shared_dyn_and_copy_sequence():
    with fake_env():
        m = import_under_test()
        # Place source in fragment scope to hit cumsum_fragment
        src = m.tir.Buffer((2, 3), scope_name="local.fragment")
        dst = m.tir.Buffer((2, 3))
        m.cumsum(src, dst=dst, dim=0, reverse=True)
        # cumsum_fragment returns None (macro decorated function with inlined intrin),
        # but side effects should have happened: alloc_shared copy->intrin->copy
        assert getattr(dst, "_copied_from", None) == "src"  # noqa: B101
        # We can still check intrin return captured by the fake call (cumsum_fragment does not return it)
        # so just assert no exception and dst got data.


def test_finalize_reducer_invokes_intrin_with_write_ptr():
    with fake_env():
        m = import_under_test()
        reducer = m.tir.Buffer((8,))
        res = m.finalize_reducer(reducer)
        assert res[0] == "intrin"  # noqa: B101
        assert res[2] == "tl.finalize_reducer"  # noqa: B101
        assert res[3] == "ptr[w]"  # noqa: B101