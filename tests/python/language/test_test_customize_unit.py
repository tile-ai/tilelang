# ruff: noqa
import sys
import types
import importlib.util
from pathlib import Path

# --- Helpers to stub external heavy dependencies (pytest-free approach, works with unittest/pytest) ---

class _CallRecorder:
    def __init__(self, name="call"):
        self.name = name
        self.calls = []
        self.return_value = object()

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.return_value

    def last(self):
        return self.calls[-1] if self.calls else None

class _FakeOp:
    def __init__(self):
        self._get = _CallRecorder("op_get")

    def get(self, name):
        return self._get(name)

class _FakeIR:
    def __init__(self):
        self.checks = []

    def assert_structural_equal(self, a, b):
        # record and perform a simple structural check for iterable shapes
        self.checks.append((a, b))
        try:
            if hasattr(a, "__iter__") and hasattr(b, "__iter__"):
                if list(a) != list(b):
                    raise AssertionError
        except AssertionError:
            # If shapes aren't iterables or do not match, just pass. Real TVM handles more.
            pass

# Minimal Buffer/Region/Load/Var stand-ins to satisfy type usage
class FakeVar:
    def __init__(self, name, let_value=None):
        self.name = name
        self._let_value = let_value

class FakeRegionDim:
    def __init__(self, min_, extent):
        self.min = min_
        self.extent = extent

class FakeBufferRegion:
    def __init__(self, buffer, region):
        self.buffer = buffer
        self.region = region  # list of FakeRegionDim

class FakeBuffer:
    def __init__(self, shape, dtype="float32", data=None):
        self.shape = list(shape)
        self.dtype = dtype
        self.data = data if data is not None else object()

class FakeBufferLoad:
    def __init__(self, buffer, indices):
        self.buffer = buffer
        self.indices = indices

# Fake tilelang "language" API
class _FakeTileLangLanguage:
    def __init__(self):
        self._call_intrin = _CallRecorder("call_intrin")
        self._call_extern = _CallRecorder("call_extern")
        self._address_of = _CallRecorder("address_of")
        self._has_let_value = _CallRecorder("has_let_value")
        self._get_let_value = _CallRecorder("get_let_value")
        self._Tensor = _CallRecorder("Tensor")
        # math ops
        self._max = lambda a, b: a if a >= b else b
        self._min = lambda a, b: a if a <= b else b

    # Expose API that module under test expects
    def call_intrin(self, *args, **kwargs): return self._call_intrin(*args, **kwargs)
    def call_extern(self, *args, **kwargs): return self._call_extern(*args, **kwargs)
    def address_of(self, *args, **kwargs): return self._address_of(*args, **kwargs)
    def has_let_value(self, var):
        # Return True iff FakeVar has a let value
        has = isinstance(var, FakeVar) and (var._let_value is not None)
        self._has_let_value(var)
        return has
    def get_let_value(self, var):
        self._get_let_value(var)
        return var._let_value
    def Tensor(self, *args, **kwargs): return self._Tensor(*args, **kwargs)
    def BufferLoad(self, *args, **kwargs): return FakeBufferLoad(*args, **kwargs)
    def max(self, a, b): return self._max(a, b)
    def min(self, a, b): return self._min(a, b)

def _install_stubs():
    # Create tvm.tir and tvm.ir stubs
    tvm_mod = types.ModuleType("tvm")
    tvm_tir = types.ModuleType("tvm.tir")
    tvm_ir = types.ModuleType("tvm.ir")

    # Provide names imported by the module
    tvm_tir.PrimExpr = object  # type placeholder
    tvm_tir.Buffer = FakeBuffer
    tvm_tir.BufferLoad = FakeBufferLoad
    tvm_tir.BufferRegion = FakeBufferRegion
    tvm_tir.Var = FakeVar

    fake_op = _FakeOp()
    tvm_tir.op = fake_op

    fake_ir = _FakeIR()
    tvm_ir.assert_structural_equal = fake_ir.assert_structural_equal

    tvm_mod.tir = tvm_tir
    tvm_mod.ir = tvm_ir

    # tilelang.language stub
    tilelang_pkg = types.ModuleType("tilelang")
    tilelang_language = _FakeTileLangLanguage()
    tilelang_pkg.language = tilelang_language

    # Insert into sys.modules before import
    sys.modules.setdefault("tvm", tvm_mod)
    sys.modules.setdefault("tvm.tir", tvm_tir)
    sys.modules.setdefault("tvm.ir", tvm_ir)
    sys.modules.setdefault("tilelang", tilelang_pkg)
    sys.modules.setdefault("tilelang.language", tilelang_language)

    # Return handles for assertions
    return {
        "ir": fake_ir,
        "op": fake_op,
        "T": tilelang_language,
    }

def _load_module_under_test():
    # Locate file testing/python/language/test_customize.py from repository root
    # Ascend until the path exists.
    here = Path(__file__).resolve()
    target_rel = Path("testing/python/language/test_customize.py")
    for base in [here, *list(here.parents)]:
        cand = base / target_rel
        if cand.exists():
            spec = importlib.util.spec_from_file_location("tl_customize_mod", str(cand))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError()

# Common setup per test module
STUBS = _install_stubs()
MOD = _load_module_under_test()
T = STUBS["T"]
OP = STUBS["op"]
FAKE_IR = STUBS["ir"]

# ------------- Tests for region -----------------
def test_region_calls_intrin_with_correct_access_type_and_extents():
    buf = FakeBuffer(shape=[4, 5], dtype="int32")
    load = FakeBufferLoad(buf, [0, 0])

    # Reset recorder
    T._call_intrin.calls.clear()
    OP._get.calls.clear()

    MOD.region(load, "r", 4, 5)
    # access type mapping: r->1
    assert OP._get.calls[-1][0] == (("tl.region",), {})
    args, _ = T._call_intrin.calls[-1]
    # ("handle", op, bufferLoad, access_type, *extents)
    assert args[0] == "handle"
    assert args[2] is load
    assert args[3] == 1
    assert args[4:] == (4, 5)

def test_region_invalid_access_type_raises_keyerror():
    buf = FakeBuffer([1], "int8")
    load = FakeBufferLoad(buf, [0])
    try:
        MOD.region(load, "invalid", 1)
    except KeyError:
        pass
    else:
        raise AssertionError()

# ------------- Tests for buffer_to_tile_region -----------------
def test_buffer_to_tile_region_computes_full_extents_and_mins():
    buf = FakeBuffer(shape=[2, 3, 4], dtype="float32")
    T._call_intrin.calls.clear()
    OP._get.calls.clear()

    MOD.buffer_to_tile_region(buf, "rw")

    # Should construct BufferLoad with mins = [0,0,0]
    # And map "rw" -> 3
    args, _ = T._call_intrin.calls[-1]
    assert args[0] == "handle"
    assert args[3] == 3
    # extents equal to buffer.shape
    assert list(args[4:]) == [2, 3, 4]

# ------------- Tests for buffer_load_to_tile_region -----------------
def test_buffer_load_to_tile_region_extends_extents_with_leading_ones():
    buf = FakeBuffer([1, 2, 3])
    load = FakeBufferLoad(buf, [0, 1, 2])
    T._call_intrin.calls.clear()
    OP._get.calls.clear()

    # extents shorter than indices -> prepend 1s
    MOD.buffer_load_to_tile_region(load, "w", extents=[5, 6])
    args, _ = T._call_intrin.calls[-1]
    # Access type "w" -> 2
    assert args[3] == 2
    # Since indices len=3 and extents len=2, expect new extents [1,5,6]
    assert list(args[4:]) == [1, 5, 6]

def test_buffer_load_to_tile_region_mismatch_raises():
    buf = FakeBuffer([1, 2])
    load = FakeBufferLoad(buf, [0, 1, 2])
    try:
        MOD.buffer_load_to_tile_region(load, "r", extents=[5, 6, 7, 8])
    except AssertionError as e:
        assert "indices" in str(e)
    else:
        raise AssertionError()

# ------------- Tests for buffer_region_to_tile_region -----------------
def test_buffer_region_to_tile_region_uses_region_extents_and_mins():
    buf = FakeBuffer([10, 10, 10])
    region = [
        FakeRegionDim(2, 4),
        FakeRegionDim(1, 3),
        FakeRegionDim(0, 5),
    ]
    br = FakeBufferRegion(buf, region)
    T._call_intrin.calls.clear()
    OP._get.calls.clear()

    MOD.buffer_region_to_tile_region(br, "r", extents=[4, 3, 5])  # equal length
    args, _ = T._call_intrin.calls[-1]
    assert list(args[4:]) == [4, 3, 5]

def test_buffer_region_to_tile_region_asserts_when_extents_too_long():
    buf = FakeBuffer([10, 10])
    region = [FakeRegionDim(0, 2), FakeRegionDim(0, 2)]
    br = FakeBufferRegion(buf, region)
    try:
        MOD.buffer_region_to_tile_region(br, "rw", extents=[1, 2, 3])
    except AssertionError as e:
        assert "region_extents must be >=" in str(e)
    else:
        raise AssertionError()

# ------------- Tests for atomic_max/min -----------------
def test_atomic_max_with_and_without_memory_order():
    buf = FakeBuffer([1], "int32")
    val = 7
    T._call_extern.calls.clear()

    # Without memory order
    MOD.atomic_max(buf, val)
    args, _ = T._call_extern.calls[-1]
    assert args[:2] == ("handle", "AtomicMax")

    # With memory order (release -> 3)
    MOD.atomic_max(buf, val, memory_order="release")
    args, _ = T._call_extern.calls[-1]
    assert args[-1] == 3

def test_atomic_min_memory_order_mapping():
    buf = FakeBuffer([1], "int32")
    val = 1
    T._call_extern.calls.clear()
    MOD.atomic_min(buf, val, memory_order="acq_rel")
    args, _ = T._call_extern.calls[-1]
    assert args[-1] == 4  # acq_rel -> 4

# ------------- Tests for atomic_add (key logic paths) -----------------
def test_atomic_add_both_extents_unknown_falls_back_to_extern():
    # dst not Buffer/Region/Var; value not Buffer/Region/Var -> extent None both
    class Dummy:
        pass
    dst = Dummy()
    val = 3
    T._call_extern.calls.clear()

    MOD.atomic_add(dst, val)
    args, _ = T._call_extern.calls[-1]
    assert args[1] == "AtomicAdd"

def test_atomic_add_buffers_shape_equal_enforced_by_ir():
    a = FakeBuffer([2, 3])
    b = FakeBuffer([2, 3])
    FAKE_IR.checks.clear()
    T._call_intrin.calls.clear()
    OP._get.calls.clear()

    MOD.atomic_add(a, b)
    # ir.assert_structural_equal called
    assert FAKE_IR.checks and list(FAKE_IR.checks[-1][0]) == [2, 3]
    # Should lower to tl.atomicadd intrin
    assert OP._get.calls[-1][0] == (("tl.atomicadd",), {})

def test_atomic_add_extent_broadcasting_and_region_conversion():
    # dst buffer has shape [4], src var with let value pointing to a BufferRegion with extent [1]
    dst = FakeBuffer([4])
    inner_buf = FakeBuffer([1])
    region = FakeBufferRegion(inner_buf, [FakeRegionDim(0, 1)])
    var = FakeVar("v", let_value=region)

    # Clean call records
    T._call_intrin.calls.clear()
    OP._get.calls.clear()
    T._Tensor.calls = []  # not used here, safety

    MOD.atomic_add(dst, var)
    # The lowering should convert both args to regions and invoke tl.atomicadd
    args, _ = T._call_intrin.calls[-1]
    assert args[1] is not None  # op id handle
    # First arg is value region, second is dst region
    # We can only assert the function name retrieved:
    assert OP._get.calls[-1][0] == (("tl.atomicadd",), {})

def test_atomic_add_with_memory_order_on_fallback_path():
    class Dummy:
        pass
    dst = Dummy()
    val = 1
    T._call_extern.calls.clear()
    MOD.atomic_add(dst, val, memory_order="seq_cst")
    args, _ = T._call_extern.calls[-1]
    # seq_cst -> 5
    assert args[-1] == 5

# ------------- Tests for atomic_addx2/x4 and dp4a -----------------
def test_atomic_addx2_and_addx4_and_dp4a_call_extern_with_addresses():
    buf = FakeBuffer([1])
    val = FakeBuffer([2])
    T._address_of.calls.clear()
    T._call_extern.calls.clear()

    MOD.atomic_addx2(buf, val)
    MOD.atomic_addx4(buf, val)
    MOD.dp4a(buf, buf, buf)

    # Last three extern calls names should match
    names = [call[0][1] for call in T._call_extern.calls[-3:]]
    assert names == ["AtomicAddx2", "AtomicAddx4", "DP4A"]
    # Address-of should be used
    assert len(T._address_of.calls) >= 4

# ------------- Tests for clamp -----------------
def test_clamp_applies_min_then_max_correctly():
    # monkeypatch T.max/min via the stub already
    assert MOD.clamp(5, 0, 10) == 5
    assert MOD.clamp(-3, 0, 10) == 0
    assert MOD.clamp(99, 0, 10) == 10

# ------------- Tests for reshape and view -----------------
def test_reshape_creates_tensor_view_with_shape_and_dtype_and_data():
    src = FakeBuffer([2, 2], dtype="int8", data="DATA")
    T._Tensor.calls.clear()
    MOD.reshape(src, [4])
    args, _ = T._Tensor.calls[-1]
    # T.Tensor(shape, src.dtype, src.data)
    assert args[0] == [4]
    assert args[1] == "int8"
    assert args[2] == "DATA"

def test_view_defaults_and_overrides():
    src = FakeBuffer([3, 3], dtype="float16", data="BUF")
    T._Tensor.calls.clear()
    # defaults (shape=None, dtype=None)
    MOD.view(src)
    args, _ = T._Tensor.calls[-1]
    assert args[0] == [3, 3] and args[1] == "float16" and args[2] == "BUF"

    # change shape only
    T._Tensor.calls.clear()
    MOD.view(src, shape=[9])
    args, _ = T._Tensor.calls[-1]
    assert args[0] == [9] and args[1] == "float16"

    # change dtype only
    T._Tensor.calls.clear()
    MOD.view(src, dtype="int32")
    args, _ = T._Tensor.calls[-1]
    assert args[0] == [3, 3] and args[1] == "int32"

# ------------- Tests for atomic_load/store -----------------
def test_atomic_load_and_store_use_memory_order_map():
    buf = FakeBuffer([1], dtype="int32")
    val = 123
    T._call_extern.calls.clear()
    MOD.atomic_load(buf, memory_order="acquire")
    MOD.atomic_store(buf, val, memory_order="release")
    load_args, _ = T._call_extern.calls[-2]
    store_args, _ = T._call_extern.calls[-1]
    assert load_args[1] == "AtomicLoad" and load_args[-1] == 2  # acquire -> 2
    assert store_args[1] == "AtomicStore" and store_args[-1] == 3  # release -> 3