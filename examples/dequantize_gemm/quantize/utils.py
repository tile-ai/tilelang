def gen_quant4(k, n, groupsize=-1):
    import torch
    import torch.nn as nn

    maxq = 2**4
    w = torch.randn((k, n), dtype=torch.half, device="cpu")

    original_w = w.clone()

    if groupsize == -1:
        groupsize = k

    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))

    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq

    # Quantize.
    w = torch.round(w / s).int()

    # Unsigned storage.
    w += (maxq) // 2

    w = torch.clamp(w, 0, maxq)

    # Dequantize.
    ref = (w - (maxq) // 2).half() * s

    if groupsize != -1:

        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, n)).contiguous()
            return w

        ref = reshape(ref)
        w = reshape(w)

    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(k, n, bias=False)
    linear.weight.data = ref.t()

    return original_w, linear, s, (w - (maxq) // 2)


def general_compress(lowprecision_weight, source_bits=4, storage_dtype=None):
    import torch

    if storage_dtype is None:
        storage_dtype = torch.int8
    elems_per_byte = 8 // source_bits
    if lowprecision_weight.dtype == torch.float16:
        lowprecision_weight = lowprecision_weight.to(torch.int8)
    int8_weight = torch.zeros(
        (*lowprecision_weight.shape[:-1], lowprecision_weight.shape[-1] // elems_per_byte),
        dtype=torch.int8,
        device=lowprecision_weight.device,
    )
    for j in range(lowprecision_weight.shape[-1] // elems_per_byte):
        for k in range(elems_per_byte):
            int8_weight[..., j] |= (lowprecision_weight[..., j * elems_per_byte + k] << (source_bits * k)).to(torch.int8)

    return int8_weight.to(storage_dtype)


# interleave weight numpy implementation
def interleave_weight(qweight, nbits=4, target_dtype="float16"):
    """Interleave the weight to the target data type.

    Args:
        qweight (_type_): _description_
        nbits (int, optional): _description_. Defaults to 4.
        target_dtype (str, optional): _description_. Defaults to "float16".

    Returns:
        _type_: _description_

    Example:
        qweight = torch.randint(0, 127, (10, 10), dtype=torch.int8).cuda()
        interleave_weight(qweight, 4, "float16")
    """
    import torch

    assert target_dtype in ["float16", "int8"]
    # reinterpret the data type of qweight to int32
    qweight = qweight.view(torch.int32)
    new_qweight = torch.zeros_like(qweight)
    bits_stride = 8 if target_dtype == "int8" else 16
    mask = (1 << nbits) - 1  # for 4bit the val is 0x0000000f
    num_groups = 32 // bits_stride
    elems_per_group = bits_stride // nbits
    for i in range(num_groups):
        for j in range(elems_per_group):
            offset = i * elems_per_group + j
            shift = (offset % num_groups) * bits_stride + (offset // num_groups) * nbits
            new_qweight |= ((qweight >> (nbits * offset)) & mask) << shift

    if nbits == 1 and target_dtype == "int8":
        # special handling for 1b interleave
        n16_weight = new_qweight & torch.int32(0xF0F00F0F)
        n16_weight |= ((new_qweight & torch.int32(0x000000F0)) >> 4) << 16
        n16_weight |= ((new_qweight & torch.int32(0x0000F000)) >> 12) << 24
        n16_weight |= ((new_qweight & torch.int32(0x000F0000)) >> 16) << 4
        n16_weight |= ((new_qweight & torch.int32(0x0F000000)) >> 24) << 12
        return n16_weight.view(torch.int8)
    elif nbits == 2 and target_dtype == "float16":
        n8_weight = new_qweight & torch.int32(0xFF0000FF)
        n8_weight |= ((new_qweight & torch.int32(0x0000FF00)) >> 8) << 16
        n8_weight |= ((new_qweight & torch.int32(0x00FF0000)) >> 16) << 8
        return n8_weight.view(torch.int8)
    elif nbits == 1 and target_dtype == "float16":
        n8_weight = new_qweight & torch.int32(0xF000000F)
        n8_weight |= ((new_qweight & torch.int32(0x000000F0)) >> 4) << 8
        n8_weight |= ((new_qweight & torch.int32(0x00000F00)) >> 8) << 16
        n8_weight |= ((new_qweight & torch.int32(0x0000F000)) >> 12) << 24
        n8_weight |= ((new_qweight & torch.int32(0x000F0000)) >> 16) << 4
        n8_weight |= ((new_qweight & torch.int32(0x00F00000)) >> 20) << 12
        n8_weight |= ((new_qweight & torch.int32(0x0F000000)) >> 24) << 20
        return n8_weight.view(torch.int8)

    return new_qweight.view(torch.int8)
