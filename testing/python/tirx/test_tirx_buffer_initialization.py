import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.ir.transform import PassContext
from tvm_tirx.target import Target

from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_layouts")


def check(body, parameters=(), config=None):
    function = tirx.Function(list(parameters), body, ret_type=ir.TupleType([])).with_attr("target", Target("c"))
    module = ir.IRModule({"main": function})
    with PassContext(config=config or {}):
        result = tvm_ffi.get_global_func("tl.transform.VerifyBufferInit")()(module)
    ir.assert_structural_equal(module, result)


@pytest.mark.parametrize(
    "scope,write_first,expected",
    [
        ("local", False, True),
        ("local", True, False),
        ("local.fragment", False, True),
        ("shared", False, False),
        ("shared", True, False),
        ("global", False, False),
    ],
)
def test_order_sensitive_local_and_cross_thread_shared(scope, write_first, expected, capfd):
    temporary = TensorView.create("temporary", [1], "int32", scope=scope)
    output = TensorView.create("output", [1], "int32")
    write = temporary.store(0, 7)
    read = output.store(0, temporary.load(0))
    check([temporary.allocate(), *([write, read] if write_first else [read, write])], [output.tensor])
    diagnostics = capfd.readouterr().err
    assert ("Buffer read before initialization" in diagnostics) == expected


@pytest.mark.parametrize("scope", ["local", "shared"])
def test_read_modify_write_does_not_initialize_itself(scope, capfd):
    temporary = TensorView.create("temporary", [1], "int32", scope=scope)
    check([temporary.allocate(), temporary.store(0, temporary.load(0) + 1)])
    assert "`temporary` is read before anything writes it" in capfd.readouterr().err


@pytest.mark.parametrize("initialize", [False, True])
def test_alias_initialization_follows_backing_storage(initialize, capfd):
    storage = TensorView.create("storage", [1], "int32", scope="local")
    alias = TensorView.create("alias", [1], "int32", scope="local")
    output = TensorView.create("output", [1], "int32")
    body = [storage.allocate(), alias.declare(storage.data)]
    if initialize:
        body.append(alias.store(0, 3))
    body.append(output.store(0, storage.load(0)))
    check(body, [output.tensor])
    assert ("Buffer read before initialization" in capfd.readouterr().err) != initialize


@pytest.mark.parametrize("mask,expected", [(1, True), (2, False), (3, False)])
def test_opaque_pointer_writes_honor_access_masks(mask, expected, capfd):
    storage = TensorView.create("storage", [1], "int32", scope="local")
    output = TensorView.create("output", [1], "int32")
    pointer = ir.Call("tl.access_ptr", [storage.load(0), 1, mask])
    initialize = ir.Evaluate(ir.Call("tirx.call_extern", [ir.StringImm("maybe_initialize"), pointer], ty="void"))
    check([storage.allocate(), initialize, output.store(0, storage.load(0))], [output.tensor])
    assert ("Buffer read before initialization" in capfd.readouterr().err) == expected


def test_diagnostic_can_be_disabled(capfd):
    storage = TensorView.create("storage", [1], "int32", scope="local")
    check([storage.allocate(), storage.store(0, storage.load(0) + 1)], config={"tl.disable_buffer_init_check": True})
    assert "Buffer read before initialization" not in capfd.readouterr().err
