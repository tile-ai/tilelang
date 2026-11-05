import importlib.util
import sys
import pytest
import torch
import tilelang.testing

cute = pytest.importorskip("cutlass.cute")
driver = pytest.importorskip("cuda.bindings.driver")

pytestmark = tilelang.testing.requires_cuda.marks()


@pytest.mark.parametrize("dtype,itemsize", [("Float16", 2), ("BFloat16", 2), ("Float32", 4), ("Int32", 4)])
@pytest.mark.parametrize("lanes", [1, 2, 8])
def test_physically_unaligned_copy(dtype, itemsize, lanes, tmp_path):
    text = f"""import cutlass
import cutlass.cute as cute
from cuda.bindings import driver
import tilelang.contrib.cutedsl as tl

@cute.kernel
def copy_bytes(a: cute.Tensor,b: cute.Tensor):
    tid,_,_=cute.arch.thread_idx()
    if tid==0:
        src=cute.make_tensor(tl.recast_ptr(a.iterator+1,dtype=cutlass.{dtype}),cute.make_layout(({lanes},)))
        dst=cute.make_tensor(tl.recast_ptr(b.iterator+1,dtype=cutlass.{dtype}),cute.make_layout(({lanes},)))
        value=tl.load_tensor(src)
        tl.store_tensor(dst,value)

@cute.jit
def launch(a: cute.Tensor,b: cute.Tensor,stream:driver.CUstream):
    copy_bytes(a,b).launch(grid=(1,1,1),block=(32,1,1),stream=stream)
"""
    path = tmp_path / "kernel.py"
    path.write_text(text)
    name = "_tl_byte_" + tmp_path.name.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    nbytes = itemsize * lanes
    src_cpu = torch.tensor([(i * 37 + 13) % 256 for i in range(nbytes + 2)], dtype=torch.uint8)
    src = src_cpu.cuda()
    dst = torch.full((nbytes + 2,), 0xA5, dtype=torch.uint8, device="cuda")
    assert (src.data_ptr() + 1) % itemsize == 1
    ca, cb = cute.runtime.from_dlpack(src, assumed_align=1), cute.runtime.from_dlpack(dst, assumed_align=1)
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(module.launch, ca, cb, stream)
    expected = torch.full((nbytes + 2,), 0xA5, dtype=torch.uint8)
    expected[1 : 1 + nbytes] = src_cpu[1 : 1 + nbytes]
    for _ in range(3):
        compiled(ca, cb, stream)
        torch.cuda.synchronize()
        torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(src.cpu(), src_cpu, rtol=0, atol=0)
