import torch
import tilelang.testing
from tilelang.distributed.utils import create_mapped_tensor


@tilelang.testing.requires_cuda
def test_create_mapped_tensor():
    shape = (1024, 1024)
    dtype = torch.float32
    host_tensor, device_tensor = create_mapped_tensor(shape, dtype)

    # test meta-data
    assert device_tensor.device.type == "cuda"
    assert device_tensor.shape == shape, f"{device_tensor.shape=}"
    assert device_tensor.dtype == dtype, f"{device_tensor.dtype=}"
    assert torch.equal(host_tensor, device_tensor.cpu()), f"{host_tensor=}, {device_tensor=}"

    # test modification
    device_tensor.random_()
    assert torch.equal(host_tensor, device_tensor.cpu()), f"{host_tensor=}, {device_tensor=}"

    print("All checks passed for create_mapped_tensor. ✅")


if __name__ == "__main__":
    tilelang.testing.main()
