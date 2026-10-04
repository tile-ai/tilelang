import pytest

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.ascend import transform as ascend_transform
from tilelang.backend.target import determine_target
from tvm import IRModule, arith, tirx
from tvm.ir import CallingConv


ASCEND_TARGET = determine_target("ascend", return_object=True)
# num_cores is the required, authoritative fold target; no device query is
# performed by AutoPersistent. Common 950-class hardware exposes 64 AI Vector
# (AIV) cores and 32 AI Cube (AIC) cores, so each test picks the count that
# matches the kernel's actual work: vector-only kernels use 64, a gemm (Cube)
# kernel uses 32.
TEST_VECTOR_CORES = 64
TEST_CUBE_CORES = 32


def _module(func):
    return IRModule({"main": func})


def _device_grid_extent(mod, thread_tag="blockIdx.x"):
    from testing.ascend._ir import nodes as _collect

    device_funcs = [
        func
        for _, func in mod.functions.items()
        if (isinstance(func, tirx.PrimFunc) and func.attrs.get("calling_conv") == CallingConv.DEVICE_KERNEL_LAUNCH)
    ]
    assert len(device_funcs) == 1

    # The 1-D core grid is a thread_extent AttrStmt on the launch iter var, not
    # a PrimFunc-level attribute.
    extents = {}
    for attr in _collect(device_funcs[0], tirx.AttrStmt):
        if attr.attr_key == "thread_extent":
            extents[str(attr.node.thread_tag)] = int(attr.value)
    assert thread_tag in extents
    return extents[thread_tag]


def _physical_launch(mod):
    launch = mod["main"].body
    if isinstance(launch, tirx.SBlockRealize) and launch.block.name_hint == "root":
        launch = launch.block.body
    assert isinstance(launch, tirx.For)
    assert launch.kind == tirx.ForKind.THREAD_BINDING
    assert launch.thread_binding.thread_tag == "blockIdx.x"
    return launch


def _physical_launches(mod):
    launch = mod["main"].body
    if isinstance(launch, tirx.SBlockRealize) and launch.block.name_hint == "root":
        launch = launch.block.body
    if isinstance(launch, tirx.SeqStmt):
        return list(launch.seq)
    assert isinstance(launch, tirx.For)
    return [launch]


def _root_under_launch(launch):
    body = launch.body
    # The launch frame emits tx/ty/tz = tl.launch_thread_idx(...) placeholders
    # between the blockIdx.x loop and tilelang_root; skip them.
    if isinstance(body, tirx.SeqStmt):
        body = body.seq[-1]
    assert isinstance(body, tirx.SBlockRealize)
    assert body.block.name_hint == "tilelang_root"
    return body


def _eval(expr, bindings):
    replacements = {var: tirx.const(value, var.dtype) for var, value in bindings.items()}
    value = arith.Analyzer().simplify(tirx.stmt_functor.substitute(expr, replacements))
    return int(value)


def _inject_root_annotation(mod, key, value):
    """Rebuild `mod` with `key: value` added to the tilelang_root block annotations."""
    body = mod["main"].body
    assert isinstance(body, tirx.SBlockRealize) and body.block.name_hint == "root"
    host = body.block
    launch = host.body
    assert isinstance(launch, tirx.For)
    seq = list(launch.body.seq)
    root = seq[-1]
    assert isinstance(root, tirx.SBlockRealize)
    block = root.block

    new_annotations = dict(block.annotations)
    new_annotations[key] = value
    new_block = tirx.SBlock(
        block.iter_vars,
        block.reads,
        block.writes,
        block.name_hint,
        block.body,
        block.init,
        block.alloc_buffers,
        block.match_buffers,
        new_annotations,
    )
    new_root = tirx.SBlockRealize(root.iter_values, root.predicate, new_block)
    seq[-1] = new_root
    new_launch = tirx.For(
        launch.loop_var,
        launch.min,
        launch.extent,
        launch.kind,
        tirx.SeqStmt(seq),
        launch.thread_binding,
        launch.annotations,
        launch.step,
    )
    new_host = tirx.SBlock(
        host.iter_vars,
        host.reads,
        host.writes,
        host.name_hint,
        new_launch,
        host.init,
        host.alloc_buffers,
        host.match_buffers,
        host.annotations,
    )
    new_body = tirx.SBlockRealize(body.iter_values, body.predicate, new_host)
    return IRModule({"main": mod["main"].with_body(new_body)})


def test_regular_kernel_does_not_opt_in():
    with ASCEND_TARGET:

        @T.prim_func
        def kernel(a: T.Tensor((70,), "int32")):
            with T.Kernel(70) as block:
                a[block] = block

    original = _module(kernel)
    out = ascend_transform.AutoPersistent()(original)
    tvm.ir.assert_structural_equal(out, original)
    assert int(_physical_launch(out).extent) == 70


def test_persistent_kernel_can_be_traced_without_device_or_target():
    @T.prim_func
    def kernel(a: T.Tensor((70,), "int32")):
        with T.PersistentKernel(70, num_cores=TEST_VECTOR_CORES) as block:
            a[block] = block

    root = _root_under_launch(_physical_launch(_module(kernel)))
    # num_cores presence is the opt-in marker for AutoPersistent.
    assert int(root.block.annotations["tilelang.persistent_kernel_num_cores"]) == TEST_VECTOR_CORES


def test_one_dimensional_grid_is_striped_and_guarded():
    with ASCEND_TARGET:

        @T.prim_func
        def kernel(a: T.Tensor((70,), "int32")):
            with T.PersistentKernel(70, num_cores=TEST_VECTOR_CORES) as block:
                a[block] = block

    original_launch = _physical_launch(_module(kernel))
    original_body = _root_under_launch(original_launch).block.body
    out = ascend_transform.AutoPersistent()(_module(kernel))
    launch = _physical_launch(out)
    assert int(launch.extent) == TEST_VECTOR_CORES

    root = _root_under_launch(launch)
    assert "tilelang.persistent_kernel_num_cores" not in root.block.annotations
    wave = root.block.body
    assert isinstance(wave, tirx.For)
    assert int(wave.extent) == 2
    assert isinstance(wave.body, tirx.IfThenElse)

    sequence = wave.body.then_case
    assert isinstance(sequence, tirx.SeqStmt)
    logical_block = sequence.seq[0]
    assert isinstance(logical_block, tirx.Bind)
    assert logical_block.var.same_as(original_launch.loop_var)
    assert sequence.seq[1].same_as(original_body)

    # Wave-major assignment: tasks 0/1 go to cores 0/1, while task 64
    # returns to core 0 in the next wave.
    assert _eval(logical_block.value, {launch.loop_var: 0, wave.loop_var: 0}) == 0
    assert _eval(logical_block.value, {launch.loop_var: 1, wave.loop_var: 0}) == 1
    assert _eval(logical_block.value, {launch.loop_var: 0, wave.loop_var: 1}) == TEST_VECTOR_CORES
    assert _eval(wave.body.condition, {launch.loop_var: 6, wave.loop_var: 1}) == 0
    assert _eval(wave.body.condition, {launch.loop_var: 5, wave.loop_var: 1}) == 1


def test_evenly_divisible_grid_does_not_emit_guard():
    with ASCEND_TARGET:

        @T.prim_func
        def kernel(a: T.Tensor((128,), "int32")):
            with T.PersistentKernel(128, num_cores=TEST_VECTOR_CORES) as block:
                a[block] = block

    out = ascend_transform.AutoPersistent()(_module(kernel))
    launch = _physical_launch(out)
    assert int(launch.extent) == TEST_VECTOR_CORES

    wave = _root_under_launch(launch).block.body
    assert isinstance(wave, tirx.For)
    assert int(wave.extent) == 2
    assert not isinstance(wave.body, tirx.IfThenElse)


def test_cube_kernel_folds_to_its_cube_core_count():
    with ASCEND_TARGET:

        @T.prim_func
        def kernel(a: T.Tensor((40,), "float32")):
            with T.PersistentKernel(40, num_cores=TEST_CUBE_CORES) as _block:
                left = T.alloc_l1((16, 16), "float16")
                right = T.alloc_l1((16, 16), "float16")
                accum = T.alloc_l0c((16, 16), "float32")
                T.gemm(left, right, accum)

    out = ascend_transform.AutoPersistent()(_module(kernel))
    assert int(_physical_launch(out).extent) == TEST_CUBE_CORES


def test_grid_smaller_than_core_count_consumes_launch_metadata():
    with ASCEND_TARGET:

        @T.prim_func
        def kernel(a: T.Tensor((16,), "int32")):
            with T.PersistentKernel(16, num_cores=TEST_VECTOR_CORES, num_stages=2) as block:
                a[block] = block

    out = ascend_transform.AutoPersistent()(_module(kernel))
    launch = _physical_launch(out)
    assert int(launch.extent) == 16
    root = _root_under_launch(launch)
    assert not isinstance(root.block.body, tirx.For)
    assert all(not key.startswith("tilelang.persistent_kernel") for key in root.block.annotations)


def test_multidimensional_persistent_kernel_is_rejected():
    with ASCEND_TARGET, pytest.raises(ValueError, match="only supports a 1-D grid"):

        @T.prim_func
        def _frontend_rejected(a: T.Tensor((5, 7), "int32")):
            with T.PersistentKernel(5, 7, num_cores=TEST_VECTOR_CORES) as (row, col):
                a[row, col] = row * 10 + col


@pytest.mark.parametrize("extent", [0, -1, tirx.IntImm("int32", 0)])
def test_nonpositive_persistent_kernel_extent_is_rejected(extent):
    with ASCEND_TARGET, pytest.raises(ValueError, match="launch extent must be positive"):

        @T.prim_func
        def _invalid(a: T.Tensor((1,), "int32")):
            with T.PersistentKernel(extent, num_cores=TEST_VECTOR_CORES) as block:
                a[block] = block


def test_unsigned_persistent_kernel_extent_is_rejected():
    extent = tirx.IntImm("uint32", 1)
    with ASCEND_TARGET, pytest.raises(ValueError, match="launch extent must have a signed integer dtype"):

        @T.prim_func
        def _invalid(a: T.Tensor((1,), "int32")):
            with T.PersistentKernel(extent, num_cores=TEST_VECTOR_CORES) as block:
                a[block] = block


def test_missing_num_cores_is_rejected():
    with ASCEND_TARGET, pytest.raises(TypeError):

        @T.prim_func
        def _invalid(a: T.Tensor((1,), "int32")):
            with T.PersistentKernel(1) as block:
                a[block] = block


@pytest.mark.parametrize("num_cores", [0, -1, True, 1.5])
def test_invalid_num_cores_is_rejected(num_cores):
    with ASCEND_TARGET, pytest.raises(ValueError, match="num_cores must be a positive integer"):

        @T.prim_func
        def _invalid(a: T.Tensor((1,), "int32")):
            with T.PersistentKernel(1, num_cores=num_cores) as block:
                a[block] = block


def test_explicit_persistent_is_treated_as_an_ordinary_nested_loop():
    with ASCEND_TARGET:

        @T.prim_func
        def kernel(a: T.Tensor((70,), "int32")):
            with T.PersistentKernel(70, num_cores=TEST_VECTOR_CORES) as block:
                for task in T.Persistent([70], 70, block):
                    a[task] = task

    out = ascend_transform.AutoPersistent()(_module(kernel))
    assert int(_physical_launch(out).extent) == TEST_VECTOR_CORES


def test_num_stages_and_annotations_are_forwarded_to_wave_loop():
    with ASCEND_TARGET:

        @T.prim_func
        def kernel(a: T.Tensor((70,), "int32")):
            with T.PersistentKernel(
                70,
                num_cores=TEST_VECTOR_CORES,
                num_stages=3,
                annotations={"enable_offset": True, "test_annotation": 7},
            ) as block:
                a[block] = block

    out = ascend_transform.AutoPersistent()(_module(kernel))
    wave = _root_under_launch(_physical_launch(out)).block.body
    assert isinstance(wave, tirx.For)
    assert int(wave.annotations["num_stages"]) == 3
    assert bool(wave.annotations["enable_offset"])
    assert int(wave.annotations["test_annotation"]) == 7


def test_ascend_pipeline_accepts_persistent_wave_scheduling():
    with ASCEND_TARGET:

        @T.prim_func
        def kernel(a: T.Tensor((128,), "float32"), b: T.Tensor((128,), "float32")):
            with T.PersistentKernel(
                128,
                num_cores=TEST_VECTOR_CORES,
                num_stages=2,
                annotations={"enable_offset": True},
            ) as block:
                value = T.alloc_shared((1,), "float32")
                T.copy(a[block : block + 1], value)
                T.copy(value, b[block : block + 1])

    artifact = tilelang.lower(kernel, target=ASCEND_TARGET, enable_device_compile=False)
    assert _device_grid_extent(artifact.device_mod) == TEST_VECTOR_CORES


def test_mixed_kernel_does_not_opt_in():
    with ASCEND_TARGET:

        @T.prim_func
        def kernel(a: T.Tensor((80,), "int32")):
            with T.MixedKernel(80) as (block, sid):
                a[block] = sid

    original = _module(kernel)
    out = ascend_transform.AutoPersistent()(original)
    tvm.ir.assert_structural_equal(out, original)
    assert int(_physical_launch(out).extent) == 80


def test_sequential_launches_fold_only_persistent_kernel():
    with ASCEND_TARGET:

        @T.prim_func
        def kernel(a: T.Tensor((150,), "int32")):
            with T.Kernel(80) as block:
                a[block] = block
            with T.PersistentKernel(70, num_cores=TEST_VECTOR_CORES) as block:
                a[block] = block * 2

    out = ascend_transform.AutoPersistent()(_module(kernel))
    launches = _physical_launches(out)
    assert len(launches) == 2
    # The regular T.Kernel launch is left unchanged.
    assert int(launches[0].extent) == 80
    # The T.PersistentKernel launch is folded onto num_cores.
    assert int(launches[1].extent) == TEST_VECTOR_CORES


def test_grid_variable_in_block_annotation_is_rejected():
    with ASCEND_TARGET:

        @T.prim_func
        def kernel(a: T.Tensor((70,), "int32")):
            with T.PersistentKernel(70, num_cores=TEST_VECTOR_CORES) as block:
                a[block] = block

    mod = _module(kernel)
    grid_var = _physical_launch(mod).loop_var

    # A launch grid variable must not leak into tilelang_root annotations: the
    # logical loop is removed during folding, so this reference would otherwise
    # become undefined. RootMetadataUsesLaunchVars must reject it.
    mod = _inject_root_annotation(mod, "test_grid_expr", grid_var + 1)

    with pytest.raises(tvm.error.InternalError, match="grid variable"):
        ascend_transform.AutoPersistent()(mod)


def test_scalar_alloc_var_is_reinitialized_per_task():
    with ASCEND_TARGET:

        @T.prim_func
        def kernel(a: T.Tensor((128,), "int32")):
            with T.PersistentKernel(128, num_cores=TEST_VECTOR_CORES) as block:
                v = T.alloc_var("int32")
                v = v + 1
                a[block] = v

    out = ascend_transform.AutoPersistent()(_module(kernel))
    launch = _physical_launch(out)
    wave = _root_under_launch(launch).block.body
    assert isinstance(wave, tirx.For)
    assert int(wave.extent) == 2
    # Evenly divisible grid: no tail guard, so the wave body is the SeqStmt
    # directly. Its first statement must be the per-task zeroing of the scalar
    # local variable, matching the explicit init=0 form.
    assert not isinstance(wave.body, tirx.IfThenElse)
    seq = wave.body
    assert isinstance(seq, tirx.SeqStmt)
    zeroing = seq.seq[0]
    assert isinstance(zeroing, tirx.BufferStore)
    assert int(zeroing.value) == 0


if __name__ == "__main__":
    tilelang.testing.main()
