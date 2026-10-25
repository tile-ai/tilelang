from pathlib import Path

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, sym, tirx
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def operation(name, *arguments):
    return ir.Evaluate(ir.Call(f"tl.{name}", list(arguments), ty="void"))


def lower(body, parameters=(), *, thread=None, threads=128, target=None):
    thread = ir.Var("thread", "int32") if thread is None else thread
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(threads)], [thread], None, body)
    function = tirx.Function(list(parameters), launch, ret_type=ir.TupleType([]))
    function = function.with_attr("global_symbol", "tmem_kernel").with_attr("target", target or Target({"kind": "cuda", "arch": "sm_100a"}))
    module = tvm_ffi.get_global_func("tl.cuda.transform.LowerSharedTmem")()(ir.IRModule({"main": function}))
    assert tirx.analysis.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    return module


def calls(module, name):
    result = []
    stmt_functor.post_order_visit(module, lambda node: result.append(node) if isinstance(node, ir.Call) and node.op.name == name else None)
    return result


@pytest.mark.parametrize("widths,expected", [([32], [32]), ([16, 16], [32]), ([4, 32, 8], [32, 32]), ([256, 128, 128], [256, 128, 128])])
def test_physical_arenas_pack_widest_first(widths, expected):
    tensors = [TensorView.create(f"tmem_{index}", [128, width], "int32", scope="shared.tmem") for index, width in enumerate(widths)]
    module = lower([tensor.allocate() for tensor in tensors])
    assert [int(call.args[1]) for call in calls(module, "tl.ptx_init_tensor_memory")] == expected
    assert [int(call.args[1]) for call in calls(module, "tl.ptx_deallocate_tensor_memory")] == expected
    allocations = calls(module, "tirx.alloc_tensor")
    assert len(allocations) == len(tensors)
    assert all(allocation.ty.scope.name == "shared" for allocation in allocations)


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_address_tokens_use_physical_datapaths_and_b32_columns(index_dtype):
    tensor = TensorView.create("tmem", [128, 64], "float16", scope="shared.tmem")
    registers = TensorView.create("registers", [1], "float16", scope="local")
    module = lower(
        [
            tensor.allocate(),
            registers.allocate(),
            operation(
                "tcgen05_ld",
                ir.const(32),
                ir.const(1),
                ir.const(True),
                tensor.load([ir.const(32, index_dtype), ir.const(4, index_dtype)]),
                ir.const(0),
                registers.data,
            ),
        ]
    )
    address = sym.Analyzer().simplify(calls(module, "tl.tcgen05_ld")[0].args[3])
    expected = (ir.const(32, "uint32") << ir.const(16, "uint32")) | ir.const(2, "uint32")
    ir.assert_structural_equal(address.b, expected)


@pytest.mark.parametrize("through_view", [False, True])
def test_dtype_punned_views_preserve_tmem_address_units_and_ownership(through_view):
    source = TensorView.create("source", [128, 128], "float32", scope="shared.tmem")
    view = TensorView.create("view", [128, 256], "bfloat16", scope="shared.tmem")
    registers = TensorView.create("registers", [1], "bfloat16", scope="local")
    pointer = ir.Call("tirx.reinterpret", [source.data], ty=view.data.ty)
    declarations = [source.allocate(), view.declare(pointer)]
    if through_view:
        nested = TensorView.create("nested", [128, 256], "bfloat16", scope="shared.tmem")
        declarations.append(nested.declare(view.data))
        view = nested
    module = lower(
        [
            *declarations,
            registers.allocate(),
            operation("tcgen05_ld", ir.const(32), ir.const(1), ir.const(True), view.load([32, 128]), ir.const(0), registers.data),
        ]
    )
    address = sym.Analyzer().simplify(calls(module, "tl.tcgen05_ld")[0].args[3])
    expected = (ir.const(32, "uint32") << ir.const(16, "uint32")) | ir.const(64, "uint32")
    ir.assert_structural_equal(address.b, expected)
    assert len(calls(module, "tl.ptx_init_tensor_memory")) == 1
    assert len(calls(module, "tl.ptx_deallocate_tensor_memory")) == 1
    assert not calls(module, "tirx.decl_tensor")


def test_borrowed_tmem_view_cannot_release_its_owner():
    source = TensorView.create("source", [128, 32], "int32", scope="shared.tmem")
    view = TensorView.create("view", [128, 32], "int32", scope="shared.tmem")
    with pytest.raises(ValueError, match="Borrowed TMEM views cannot be deallocated"):
        lower([source.allocate(), view.declare(source.data), operation("deallocate_tmem", view.data)])


def test_conditional_release_uses_a_live_flag():
    tensor = TensorView.create("tmem", [128, 32], "int32", scope="shared.tmem")
    condition = ir.Var("condition", "bool")
    module = lower([tensor.allocate(), ir.If(condition, operation("deallocate_tmem", tensor.data), None)], [condition])
    assert len(calls(module, "tl.ptx_deallocate_tensor_memory")) == 2
    assert any(allocation.ty.scope.name == "local" for allocation in calls(module, "tirx.alloc_tensor"))


@pytest.mark.parametrize("kind", ["if", "while", "bind"])
def test_divergent_allocations_are_rejected(kind):
    thread = ir.Var("thread", "int32")
    tensor = TensorView.create("tmem", [128, 32], "int32", scope="shared.tmem")
    condition = thread < 32
    prefix = []
    if kind == "bind":
        predicate = ir.Var("predicate", "bool")
        prefix = [ir.Bind(predicate, condition)]
        condition = predicate
    body = ir.While(condition, [tensor.allocate(), ir.Break()]) if kind == "while" else ir.If(condition, tensor.allocate(), None)
    with pytest.raises(ValueError, match="block-uniform"):
        lower([*prefix, body], thread=thread)


def test_first_warp_can_return_after_path_complete_release():
    tensor = TensorView.create("tmem", [128, 32], "int32", scope="shared.tmem")
    thread = ir.Var("thread", "int32")
    returned = ir.If(thread < 32, [operation("deallocate_tmem", tensor.data), ir.Return(ir.Tuple([]))], None)
    module = lower([tensor.allocate(), returned], thread=thread)
    assert len(calls(module, "tl.ptx_deallocate_tensor_memory")) == 2
    returns = []
    stmt_functor.post_order_visit(module, lambda node: returns.append(node) if isinstance(node, ir.Return) else None)
    assert len(returns) == 1


def test_an_early_return_requires_path_complete_release():
    tensor = TensorView.create("tmem", [128, 32], "int32", scope="shared.tmem")
    condition = ir.Var("condition", "bool")
    body = [tensor.allocate(), ir.If(condition, ir.Return(ir.Tuple([])), None), operation("deallocate_tmem", tensor.data)]
    with pytest.raises(ValueError, match="Every early-return path"):
        lower(body, [condition])


@pytest.mark.parametrize("predicate", [lambda thread: thread == 0, lambda thread: thread >= 32])
def test_explicit_release_requires_the_complete_first_warp(predicate):
    tensor = TensorView.create("tmem", [128, 32], "int32", scope="shared.tmem")
    thread = ir.Var("thread", "int32")
    with pytest.raises(ValueError, match="complete first warp"):
        lower([tensor.allocate(), ir.If(predicate(thread), operation("deallocate_tmem", tensor.data), None)], thread=thread)


@pytest.mark.parametrize("packed", [False, True])
def test_tmem_store_load_executes_on_blackwell(packed, tilelang_builders):
    device = runtime.cuda(0)
    if not device.exist or int(device.compute_version.split(".")[0]) < 10:
        pytest.skip("Blackwell device unavailable")
    root = Path(__file__).resolve().parents[3]
    target = Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}a",
            "backend_config": {"cuda": {"compiler": "nvcc", "nvcc": [f"-I{root / 'src'}", f"-I{root / '3rdparty/cutlass/include'}"]}},
        },
        host="c",
    )
    thread = ir.Var("thread", "int32")
    index = ir.Var("index", "int32")
    clear_index = ir.Var("clear_index", "int32")
    output_index = ir.Var("output_index", "int32")
    output = TensorView.create("output", [128, 32], "int32")
    registers = TensorView.create("registers", [32], "int32", scope="local")
    tensor = TensorView.create("tmem", [128, 32], "int32", scope="shared.tmem")
    prefix = []
    if packed:
        other = TensorView.create("other", [128, 16], "int32", scope="shared.tmem")
        prefix = [other.allocate()]
    body = [
        *prefix,
        tensor.allocate(),
        registers.allocate(),
        ir.For(index, 0, 32, 1, ir.ForKind.UNROLLED, registers.store(index, thread * 32 + index)),
        operation(
            "tcgen05_st", ir.const(32), ir.const(32), ir.const(False), tensor.load([thread // 32 * 32, 0]), ir.const(0), registers.data
        ),
        ir.For(clear_index, 0, 32, 1, ir.ForKind.UNROLLED, registers.store(clear_index, -1)),
        operation(
            "tcgen05_ld", ir.const(32), ir.const(32), ir.const(False), tensor.load([thread // 32 * 32, 0]), ir.const(0), registers.data
        ),
        ir.For(output_index, 0, 32, 1, ir.ForKind.UNROLLED, output.store([thread, output_index], registers.load(output_index))),
    ]
    module = lower(body, [output.tensor], thread=thread, target=target)
    executable = tirx.compile(module, target=target)
    actual = runtime.empty((128, 32), "int32", device=device)
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.arange(128 * 32, dtype="int32").reshape(128, 32))
