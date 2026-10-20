"""Device acceptance for VMI memory-access descriptors.

Every kernel runs through the real ``tilelang.compile(target="pto")`` pipeline
with the automatic scheduler enabled (the generated PTODSL must contain the
MTE sync flags; no test disables scheduling).  Inputs are built on CPU with
torch, copied to the NPU, and compared against an independent CPU reference
after ``torch.npu.synchronize()``.

Critical dependency paths execute 20 fresh trials each with independently
prepared inputs and a fresh output tensor, and also assert the
generated-source order and MTE sync pair:

* a DMA that writes only the non-zero ``[64,128)`` region feeding
  ``vload(work[64], size=64)``;
* ``vstore(..., work[64])`` feeding a DMA that reads only ``[64,128)``;
* a padded view whose low group is written by DMA and read by a static
  negative group stride, with both groups initialized for the reference.

The remaining cases are one device-numerics representative per access mode:
block_stride, FP4 raw packed bytes (buffer and explicit-pointer forms), a
packed (PK) narrowing store, an explicit ``access_ptr`` plus extra offset, and
gather/scatter with non-zero bases (direct and explicit-pointer forms).  The
device numerical reference is exact for copy-like kernels and an independent
CPU computation (torch cast reference) for narrowing.

A runtime group stride has no proven physical envelope, so those kernels are
rejected at compile time and their diagnostics live in the structural suite;
they are not claimed as device-passed here.

Non-VMI lane counts such as ``size=80`` cannot be constructed by the PTODSL
VMI API; their descriptor/backend evidence lives in the structural suite
(``test_pto_vmi_memory_access.py``) and is not claimed as device-passed.
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["TILELANG_DISABLE_DATA_RACE_CHECK"] = "1"

import torch_npu  # noqa: E402,F401  (must import before tilelang/ptodsl)
import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402
from tvm import tirx  # noqa: E402

F16 = "float16"
F32 = "float32"
FP4 = "float4_e2m1fn"


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------
def _require_npu_runtime():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("torch.npu is unavailable in the current environment")
    try:
        source = torch.arange(8, dtype=torch.float32)
        copied = source.to(torch.device("npu"))
        torch.npu.synchronize()
        assert torch.equal(copied.cpu(), source)
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"NPU runtime is unavailable in the current environment: {err}")


@pytest.fixture(scope="module")
def npu():
    _require_npu_runtime()
    return torch.device("npu")


def _compile(func):
    kernel = tilelang.compile(func, target="pto", out_idx=-1)
    source = kernel.get_kernel_source()
    assert "pto.set_flag(" in source and "pto.wait_flag(" in source, (
        "automatic scheduling must stay enabled (no MTE sync in the generated source):\n" + source
    )
    return kernel, source


def _vpto_text(kernel_source):
    """Compile the generated PTODSL through PTOAS and return the VPTO text."""
    import importlib.util
    import re
    import subprocess
    import tempfile
    import uuid

    decorator = re.search(r'@pto\.jit\(name="([^"]+)"', kernel_source)
    name = decorator.group(1) if decorator is not None else "main_kernel"
    work = tempfile.mkdtemp(prefix="vmi-runtime-vpto-", dir=os.environ.get("TMPDIR") or None)
    source_path = os.path.join(work, "kernel.ptodsl.py")
    pto_path = os.path.join(work, "kernel.pto")
    vpto_path = os.path.join(work, "kernel.vpto.mlir")
    with open(source_path, "w", encoding="utf-8") as file:
        file.write(kernel_source)
    spec = importlib.util.spec_from_file_location("vmi_runtime_" + uuid.uuid4().hex, source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pto_text = getattr(module, name).compile().mlir_text()
    with open(pto_path, "w", encoding="utf-8") as file:
        file.write(pto_text)
    result = subprocess.run(
        ["ptoas", "--pto-arch=a5", "--pto-backend=vpto", "--emit-vpto", pto_path, "-o", vpto_path],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, f"PTOAS rejected the generated kernel:\n{result.stderr}"
    with open(vpto_path, encoding="utf-8") as file:
        return file.read()


def _assert_addptr_pointer_use(source, definition_fragment, call_fragment):
    lines = [line.strip() for line in source.splitlines()]
    definitions = [line for line in lines if "= pto.addptr(" in line and definition_fragment in line]
    assert len(definitions) == 1, f"expected one pointer definition with {definition_fragment!r}: {definitions}"
    name = definitions[0].split("=")[0].strip()
    assert call_fragment.format(name=name) in source, source


def _to_device(value, device):
    return value.to(device) if isinstance(value, torch.Tensor) else value


def _run_trials(kernel, make_case, device, trials):
    """Run *trials* independent cases; return the per-trial outputs."""
    outputs = []
    for trial in range(trials):
        inputs, expected = make_case(trial)
        device_inputs = [_to_device(value, device) for value in inputs]
        output = kernel(*device_inputs)
        torch.npu.synchronize()
        got = output.cpu()
        assert got.dtype == expected.dtype, f"trial {trial}: {got.dtype} != {expected.dtype}"
        assert torch.equal(got, expected), f"trial {trial}: numeric mismatch\n got={got.tolist()[:16]}\n exp={expected.tolist()[:16]}"
        outputs.append(got)
    return outputs


def _line_index(lines, predicate, start=0, what="line"):
    for index in range(start, len(lines)):
        if predicate(lines[index]):
            return index
    raise AssertionError(f"missing {what} from line {start}:\n" + "\n".join(lines))


def _assert_mte2_to_v_then_load(lines, dma_predicate, descriptor_needle):
    dma = _line_index(lines, dma_predicate, what="producer MTE2 DMA")
    set_index = _line_index(lines, lambda line: 'set_flag("MTE2", "V"' in line, dma, "MTE2->V set_flag")
    wait_index = _line_index(lines, lambda line: 'wait_flag("MTE2", "V"' in line, set_index, "MTE2->V wait_flag")
    vload = _line_index(lines, lambda line: "vmi.vload" in line, wait_index, "vmi.vload")
    assert dma < set_index < wait_index < vload
    assert descriptor_needle in lines[vload], lines[vload]


def _assert_store_then_v_to_mte3(lines, dma_predicate, store_needle):
    store = _line_index(lines, lambda line: "vmi.vstore" in line, what="vmi.vstore")
    set_index = _line_index(lines, lambda line: 'set_flag("V", "MTE3"' in line, store, "V->MTE3 set_flag")
    wait_index = _line_index(lines, lambda line: 'wait_flag("V", "MTE3"' in line, set_index, "V->MTE3 wait_flag")
    dma = _line_index(lines, dma_predicate, wait_index, "consumer MTE3 DMA")
    assert store < set_index < wait_index < dma
    assert store_needle in lines[store], lines[store]


def _float_inputs(shape, trial, base=0.0, step=1.0):
    generator = torch.Generator().manual_seed(1000 + trial)
    values = torch.randn(shape, generator=generator)
    return values * step + (base + trial * 37.0)


# ---------------------------------------------------------------------------
# Critical path 1: DMA writing only [64,128) -> vload(work[64])
# ---------------------------------------------------------------------------
def _dma_load_kernel():
    @T.prim_func
    def main(A: T.Tensor((64,), F32), S: T.Tensor((64,), F32), C: T.Tensor((64,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((128,), F32)
            T.copy(S, work[64:128])  # sentinel first
            T.copy(A, work[64:128])  # producer DMA: only [64,128)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                v = T.vmi.vload(work[64], size=64)
                T.vmi.vstore(v, work[0], mask)
            T.copy(work[0:64], C)

    return main


def test_dma_to_nonzero_load_dependency_20_trials(npu):
    kernel, source = _compile(_dma_load_kernel())
    lines = source.splitlines()
    _assert_mte2_to_v_then_load(
        lines,
        lambda line: "mte_gm_ub" in line and "addptr" in line and ", 64)" in line,
        "), 0, size=64)",
    )

    def make_case(trial):
        a = _float_inputs((64,), trial)
        sentinel = torch.full((64,), -777.0 - trial)
        return [a, sentinel], a

    _run_trials(kernel, make_case, npu, trials=20)


# ---------------------------------------------------------------------------
# Critical path 2: vstore(..., work[64]) -> DMA reading only [64,128)
# ---------------------------------------------------------------------------
def _store_dma_kernel():
    @T.prim_func
    def main(A: T.Tensor((64,), F32), S: T.Tensor((64,), F32), C: T.Tensor((64,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((128,), F32)
            T.copy(S, work[64:128])  # sentinel before the store
            T.copy(A, work[0:64])
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                v = T.vmi.vload(work[0], size=64)
                T.vmi.vstore(v, work[64], mask)
            T.copy(work[64:128], C)  # consumer DMA: only [64,128)

    return main


def test_nonzero_store_to_dma_dependency_20_trials(npu):
    kernel, source = _compile(_store_dma_kernel())
    lines = source.splitlines()
    _assert_store_then_v_to_mte3(
        lines,
        lambda line: "mte_ub_gm" in line and "addptr" in line and ", 64)" in line,
        "), 0, mask)",
    )

    def make_case(trial):
        a = _float_inputs((64,), trial)
        sentinel = torch.full((64,), -888.0 - trial)
        return [a, sentinel], a

    _run_trials(kernel, make_case, npu, trials=20)


# ---------------------------------------------------------------------------
# Block stride: stride 1/2, single and multi VL
# ---------------------------------------------------------------------------
def _block_expected(a, base, lanes, block_stride, vector=64, block=8):
    chunks = (lanes + vector - 1) // vector
    values = []
    for chunk in range(chunks):
        for slot in range(vector // block):
            start = base + chunk * (vector // block) * block_stride * block + slot * block_stride * block
            values.extend(a[start : start + block].tolist())
    return torch.tensor(values[:lanes], dtype=a.dtype)


def _block_load_kernel(size, block_stride):
    @T.prim_func
    def main(A: T.Tensor((768,), F32), C: T.Tensor((size,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((768,), F32)
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(size, size=size)
                v = T.vmi.vload(work[64], size=size, block_stride=block_stride)
                T.vmi.vstore(v, work[512], mask)
            T.copy(work[512 : 512 + size], C)

    return main


@pytest.mark.parametrize(
    ("size", "block_stride"),
    [(64, 1), (64, 2), (128, 2)],
    ids=["single-vl-s1", "single-vl-s2", "multi-vl-s2"],
)
def test_block_stride_load(npu, size, block_stride):
    kernel, _ = _compile(_block_load_kernel(size, block_stride))

    def make_case(trial):
        a = _float_inputs((768,), trial)
        return [a], _block_expected(a, 64, size, block_stride)

    _run_trials(kernel, make_case, npu, trials=3)


# ---------------------------------------------------------------------------
# FP4 raw packed bytes: non-zero logical start and explicit pointer offset
# ---------------------------------------------------------------------------
def _fp4_view_kernel(explicit_pointer):
    @T.prim_func
    def main(B: T.Tensor((128,), "uint8"), Z: T.Tensor((128,), "uint8"), C: T.Tensor((128,), "uint8")):
        with T.Kernel(1):
            raw_in = T.alloc_shared((128,), "uint8")
            raw_out = T.alloc_shared((128,), "uint8")
            T.copy(B, raw_in)
            T.copy(Z, raw_out)
            with T.SimdVF():
                f4_in = tirx.decl_buffer((256,), FP4, data=raw_in.data, scope="shared.dyn")
                f4_out = tirx.decl_buffer((256,), FP4, data=raw_out.data, scope="shared.dyn")
                mask = T.vmi.create_mask(64, size=64)
                if explicit_pointer:
                    ptr = T.access_ptr(f4_in[0], "r", extent=256)
                    v = T.vmi.vload(ptr, 64, size=128)
                else:
                    v = T.vmi.vload(f4_in[64], size=128)
                T.vmi.vstore(v, f4_out[128], mask)
            T.copy(raw_out, C)

    return main


@pytest.mark.parametrize("explicit_pointer", [False, True], ids=["bufferload", "ptr-offset"])
def test_fp4_nonzero_logical_start_raw_bytes(npu, explicit_pointer):
    kernel, source = _compile(_fp4_view_kernel(explicit_pointer))
    if explicit_pointer:
        # Logical extra offset 64 is packed once to 32 bytes.  The pointer
        # variable name is not assumed: resolve the actual definition/use.
        import re

        match = re.search(r"v = pto\.vmi\.vload\((\w+), 32, size=\(128 // 2\)\)", source)
        assert match is not None, source
        pointer_name = match.group(1)
        assert re.search(rf"\b{pointer_name} = pto\.(?:addptr|castptr)\(", source), source
    else:
        assert "addptr" in source and "), 32)" in source, source

    def make_case(trial):
        generator = torch.Generator().manual_seed(2000 + trial)
        byte_values = torch.randint(0, 256, (128,), generator=generator, dtype=torch.int64).to(torch.uint8)
        zeros = torch.zeros(128, dtype=torch.uint8)
        expected = zeros.clone()
        expected[64:128] = byte_values[32:96]
        return [byte_values, zeros], expected

    _run_trials(kernel, make_case, npu, trials=3)


# ---------------------------------------------------------------------------
# Packed (PK) store: logical target extent and exact float16 reference
# ---------------------------------------------------------------------------
def _pk_b32_kernel():
    @T.prim_func
    def main(A: T.Tensor((128,), F32), S: T.Tensor((128,), F16), C: T.Tensor((128,), F16)):
        with T.Kernel(1):
            src = T.alloc_shared((128,), F32)
            out = T.alloc_shared((192,), F16)
            T.copy(A, src)
            T.copy(S, out[64:192])
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                v = T.vmi.vload(src[0], size=128)
                narrowed = T.vmi.vcvt(v, to_dtype=F16, rounding="R", saturate="SAT")
                T.vmi.vstore(narrowed, out[64], mask)
            T.copy(out[64:192], C)

    return main


def test_pk_b32_f16_store(npu):
    kernel, source = _compile(_pk_b32_kernel())
    vpto = _vpto_text(source)
    assert vpto.count('dist = "PK_B32"') == 2, vpto

    def make_case(trial):
        a = ((torch.arange(128, dtype=torch.float32) - 64.0) * 0.25) + trial
        expected = a.to(torch.float16)
        sentinel = torch.full((128,), -3.0, dtype=torch.float16)
        return [a, sentinel], expected

    _run_trials(kernel, make_case, npu, trials=3)


# ---------------------------------------------------------------------------
# Explicit pointer
# ---------------------------------------------------------------------------
def _explicit_pointer_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((64,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), F32)
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                ptr = T.access_ptr(work[32], "r", extent=128)
                v = T.vmi.vload(ptr, 16, size=64)
                T.vmi.vstore(v, work[128], mask)
            T.copy(work[128:192], C)

    return main


def test_explicit_pointer_base32_extra16_reads_48(npu):
    kernel, source = _compile(_explicit_pointer_kernel())
    _assert_addptr_pointer_use(source, ", 32)", "vmi.vload({name}, 16, size=64)")

    def make_case(trial):
        a = _float_inputs((256,), trial)
        return [a], a[48:112]

    _run_trials(kernel, make_case, npu, trials=5)


# ---------------------------------------------------------------------------
# Padded static negative group stride: both groups initialized
# ---------------------------------------------------------------------------
def _padded_group_kernel():
    @T.prim_func
    def main(L: T.Tensor((64,), F32), H: T.Tensor((64,), F32), C: T.Tensor((128,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((512,), F32)
            out = T.alloc_shared((128,), F32)
            view = T.StridedTensor((2, 128), (256, 1), F32, data=work.data, scope="shared.dyn")
            T.copy(L, view[0, 0:64])  # low group, second group of the load
            T.copy(H, view[1, 0:64])  # high group, first group of the load
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                v = T.vmi.vload(view[1, 0], size=128, group=2, stride=-256)
                T.vmi.vstore(v, out[0], mask)
            T.copy(out, C)

    return main


def _assert_padded_group_same_view(func):
    """Both input DMAs and the VMI read must name the same view Buffer object."""
    calls = []
    tirx.stmt_functor.post_order_visit(
        func.body,
        lambda node: calls.append(node) if isinstance(node, tirx.Call) else None,
    )
    vload_buffers = []
    dma_destinations = []
    for call in calls:
        name = getattr(call.op, "name", "")
        if name == "tl.vmi.vload":
            source = call.args[0]
            assert isinstance(source, tirx.Call) and getattr(source.op, "name", "") == "tl.access_ptr", source
            vload_buffers.append(source.args[0].buffer)
        elif name == "tl.tileop.ascend_copy":
            dma_destinations.append(call.args[1].args[0])
    assert len(vload_buffers) == 1, vload_buffers
    view = vload_buffers[0]
    same_view = [destination for destination in dma_destinations if destination.buffer.same_as(view)]
    assert len(same_view) == 2, f"expected two DMAs into the VMI view, got {len(same_view)} of {len(dma_destinations)}"
    indices = sorted(tuple(int(index) for index in destination.indices) for destination in same_view)
    assert indices == [(0, 0), (1, 0)], indices
    assert tuple(int(dim) for dim in view.shape) == (2, 128), view.shape
    assert tuple(int(stride) for stride in view.strides) == (256, 1), view.strides


def test_padded_group_negative_stride_20_trials(npu):
    func = _padded_group_kernel()
    _assert_padded_group_same_view(func)
    kernel, source = _compile(func)
    lines = source.splitlines()
    # The low-group DMA is the only GM->UB copy without an address adjustment;
    # the descriptor starts at the view origin and the instruction offset 256
    # is counted exactly once.
    _assert_mte2_to_v_then_load(
        lines,
        lambda line: "mte_gm_ub" in line and "addptr" not in line,
        "), 256, group=2, size=128, stride=-256)",
    )
    vload_line = next(line for line in lines if "vmi.vload" in line)
    assert "addptr" not in vload_line, vload_line

    def make_case(trial):
        low = _float_inputs((64,), trial, base=100.0, step=0.5)
        high = _float_inputs((64,), trial, base=-250.0, step=0.25)
        return [low, high], torch.cat([high, low])

    _run_trials(kernel, make_case, npu, trials=20)


# ---------------------------------------------------------------------------
# Gather / scatter with non-zero direct and explicit-pointer bases
# ---------------------------------------------------------------------------
def _gather_kernel(explicit_pointer):
    @T.prim_func
    def main(A: T.Tensor((256,), F32), I: T.Tensor((64,), "uint32"), C: T.Tensor((64,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), F32)
            idx = T.alloc_shared((64,), "uint32")
            out = T.alloc_shared((64,), F32)
            T.copy(A, work)
            T.copy(I, idx)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                offsets = T.vmi.vload(idx[0], size=64)
                if explicit_pointer:
                    ptr = T.access_ptr(work[64], "r", extent=128)
                    v = T.vmi.vgather(ptr, offsets, mask)
                else:
                    v = T.vmi.vgather(work[64], offsets, mask)
                T.vmi.vstore(v, out[0], mask)
            T.copy(out, C)

    return main


@pytest.mark.parametrize("explicit_pointer", [False, True], ids=["direct", "ptr"])
def test_gather_nonzero_base(npu, explicit_pointer):
    kernel, _ = _compile(_gather_kernel(explicit_pointer))

    def make_case(trial):
        a = _float_inputs((256,), trial)
        generator = torch.Generator().manual_seed(3000 + trial)
        offsets = torch.randperm(128, generator=generator)[:64].to(torch.uint32)
        expected = a[64 + offsets.to(torch.int64)]
        return [a, offsets], expected

    _run_trials(kernel, make_case, npu, trials=3)


def _scatter_kernel(explicit_pointer):
    @T.prim_func
    def main(A: T.Tensor((64,), F32), S: T.Tensor((256,), F32), I: T.Tensor((64,), "uint32"), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            src = T.alloc_shared((64,), F32)
            work = T.alloc_shared((256,), F32)
            idx = T.alloc_shared((64,), "uint32")
            T.copy(A, src)
            T.copy(S, work)
            T.copy(I, idx)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                offsets = T.vmi.vload(idx[0], size=64)
                v = T.vmi.vload(src[0], size=64)
                if explicit_pointer:
                    ptr = T.access_ptr(work[64], "w", extent=128)
                    T.vmi.vscatter(v, ptr, offsets, mask)
                else:
                    T.vmi.vscatter(v, work[64], offsets, mask)
            T.copy(work, C)

    return main


@pytest.mark.parametrize("explicit_pointer", [False, True], ids=["direct", "ptr"])
def test_scatter_nonzero_base(npu, explicit_pointer):
    kernel, _ = _compile(_scatter_kernel(explicit_pointer))

    def make_case(trial):
        a = _float_inputs((64,), trial)
        sentinel = torch.full((256,), -123.0 - trial)
        generator = torch.Generator().manual_seed(4000 + trial)
        offsets = torch.randperm(128, generator=generator)[:64].to(torch.uint32)
        expected = sentinel.clone()
        expected[64 + offsets.to(torch.int64)] = a
        return [a, sentinel, offsets], expected

    _run_trials(kernel, make_case, npu, trials=3)
