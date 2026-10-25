import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tvm_tirx import ir, sym

from tilelang._compat.stmt_functor import post_order_visit


def _get_dynamic_shared_memory_bytes(artifact):
    requirements = []

    def collect(node):
        if isinstance(node, ir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tirx.call_ffi_kernel":
            fields = list(node.attrs.launch_fields)
            if "dynamic_smem_bytes" in fields:
                index = 1 + int(node.attrs.num_kernel_args) + fields.index("dynamic_smem_bytes")
                requirements.append(node.args[index])
            elif "tirx.use_dyn_shared_memory" in node.attrs.launch_params:
                parameters = list(node.attrs.launch_params)
                index = len(node.args) - len(parameters) + parameters.index("tirx.use_dyn_shared_memory")
                requirements.append(node.args[index])

    post_order_visit(artifact.host_mod, collect)
    assert len(requirements) == 1
    return int(sym.Analyzer().simplify(requirements[0]))


def _get_shared_alias_offsets(artifact):
    offsets = {}

    def collect(node):
        if not isinstance(node, ir.Bind) or not isinstance(node.value, ir.Call):
            return
        if node.value.op.name != "tirx.decl_tensor":
            return
        values = []
        post_order_visit(
            node.value,
            lambda value: values.append(value.args[1])
            if isinstance(value, ir.Call) and isinstance(value.op, ir.Op) and value.op.name == "tirx.ptr_byte_offset"
            else None,
        )
        if values:
            assert len(values) == 1
            offsets[node.var.name] = int(sym.Analyzer().simplify(values[0]))

    post_order_visit(artifact.device_mod, collect)
    return offsets


@tilelang.testing.requires_cuda
def test_dynamic_shared_memory_merge_emits_named_aliases():
    @T.prim_func
    def kernel(
        A: T.Tensor((32,), T.float16),
        B: T.Tensor((32,), T.float16),
        C: T.Tensor((32,), T.float16),
    ):
        with T.Kernel(1, threads=32):
            A_shared = T.alloc_shared((32,), T.float16)
            B_shared = T.alloc_shared((32,), T.float16)
            A_shared[0] = A[0]
            B_shared[0] = B[0]
            T.tvm_storage_sync("shared")
            C[0] = A_shared[0] + B_shared[0]

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source

    assert "extern __shared__" in source
    assert _get_shared_alias_offsets(artifact) == {"A_shared": 0, "B_shared": 64}
    assert "A_shared" in source
    assert "B_shared" in source
    assert "((half_t*)buf_dyn_shmem)" not in source
    assert _get_dynamic_shared_memory_bytes(artifact) == 128


@tilelang.testing.requires_cuda
def test_static_and_local_fp4_allocations_use_packed_extents():
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((127,), T.float4_e2m1fn, scope="shared")
            A_local = T.alloc_local((127,), T.float4_e2m1fn)
            A_shared[126] = T.cast(0.0, T.float4_e2m1fn)
            A_local[126] = A_shared[126]

    source = tilelang.lower(kernel, target="cuda").kernel_source

    assert "__nv_fp4_e2m1 A_shared_ptr[64];" in source
    assert "__nv_fp4_e2m1 A_local_ptr[64];" in source


@tilelang.testing.requires_cuda
def test_single_dynamic_fp4_allocation_uses_packed_size():
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((127,), T.float4_e2m1fn)
            A_shared[126] = T.cast(0.0, T.float4_e2m1fn)

    artifact = tilelang.lower(kernel, target="cuda")

    assert "extern __shared__" in artifact.kernel_source
    assert "__nv_fp4_e2m1 A_shared_ptr[];" in artifact.kernel_source
    assert _get_dynamic_shared_memory_bytes(artifact) == 64


@tilelang.testing.requires_cuda
def test_merged_dynamic_fp4_allocations_use_packed_sizes_and_offsets():
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((127,), T.float4_e2m1fn)
            B_shared = T.alloc_shared((127,), T.float4_e2m1fn)
            A_shared[126] = T.cast(0.0, T.float4_e2m1fn)
            B_shared[126] = T.cast(1.0, T.float4_e2m1fn)
            T.tvm_storage_sync("shared")
            A_shared[0] = B_shared[126]

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source

    assert "extern __shared__" in source
    assert "A_shared" in source and "B_shared" in source
    assert _get_shared_alias_offsets(artifact) == {"A_shared": 0, "B_shared": 64}
    assert _get_dynamic_shared_memory_bytes(artifact) == 128


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "dtype,storage_type",
    [(T.int4, "int8_t"), (T.dtype("uint4"), "uint8_t")],
)
def test_static_packed_int4_allocation_uses_packed_extent(dtype, storage_type):
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((127,), dtype, scope="shared")
            A_shared[126] = T.cast(1, dtype)

    source = tilelang.lower(kernel, target="cuda").kernel_source

    assert f"{storage_type} A_shared_ptr[64];" in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [T.int4, T.dtype("uint4")])
def test_single_dynamic_packed_int4_allocation_uses_packed_size(dtype):
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((127,), dtype)
            A_shared[126] = T.cast(1, dtype)

    artifact = tilelang.lower(kernel, target="cuda")

    assert _get_dynamic_shared_memory_bytes(artifact) == 64


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [T.int4, T.dtype("uint4")])
def test_merged_dynamic_packed_int4_allocations_use_packed_sizes_and_offsets(dtype):
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((127,), dtype)
            B_shared = T.alloc_shared((127,), dtype)
            A_shared[126] = T.cast(0, dtype)
            B_shared[126] = T.cast(1, dtype)
            T.tvm_storage_sync("shared")
            A_shared[0] = B_shared[126]

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source

    assert "A_shared" in source and "B_shared" in source
    assert _get_shared_alias_offsets(artifact) == {"A_shared": 0, "B_shared": 64}
    assert _get_dynamic_shared_memory_bytes(artifact) == 128


if __name__ == "__main__":
    tilelang.testing.main()
