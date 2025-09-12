import torch


def torch_convert_bit_twiddling(tensor):
    """
    Convert a 2-D uint8 tensor into a bfloat16 tensor by decoding pairs of input bytes with a bit-twiddling scheme.
    This is a parallel implementation using torch operators.
    
    This function expects `tensor` to be a 2-D torch.Tensor of dtype `torch.uint8`. Each output element is produced by combining two input bytes and extracting a bf16-like 16-bit pattern according to one of four positional bit layouts (pos 0..3). The result is scaled by 2**126 to adjust the exponent bias and returned as dtype `torch.bfloat16`.
    
    Parameters:
        tensor (torch.Tensor): 2-D input tensor with dtype `torch.uint8`. Shape (N, K).
    
    Returns:
        torch.Tensor: New tensor of dtype `torch.bfloat16` with shape (N, K*2), where each input column pair produces two bf16 output columns.
    
    Raises:
        AssertionError: If any byte inputs used for a conversion are not dtype `torch.uint8`.
    """
    assert tensor.dim() == 2 and tensor.dtype == torch.uint8
    N, K = tensor.shape
    assert K % 2 == 0, "Number of columns must be even"

    # Combine pairs of uint8 values into uint32 for safe bitwise ops on CUDA
    val0 = tensor[:, 0::2].to(torch.int32)
    val1 = tensor[:, 1::2].to(torch.int32)
    val_concat = (val0 << 8) | val1  # (N, K//2), uint32

    # Expand to match output shape where each pair generates 4 values
    val_concat_expanded = val_concat.repeat_interleave(4, dim=1)  # (N, K//2*4)

    # Positional encoding for bit-twiddling logic
    pos = torch.arange(K * 2, device=tensor.device) % 4  # (K*2,)

    # Bit masks for decoding (as uint32 for CUDA compatibility)
    mask = 0b1000000111000000
    mask1 = 0b1000000000000000
    mask2 = 0b0000000110000000
    mask3 = 0b0000000001000000

    # Calculate results for all 4 positions in parallel
    res0 = val_concat_expanded & mask
    res1 = (val_concat_expanded << 3) & mask
    res2 = (val_concat_expanded << 6) & mask
    res3 = ((val_concat_expanded << 1) & mask1) | ((val_concat_expanded >> 3) & mask2) | ((val_concat_expanded >> 7) & mask3)

    # Select the correct result based on position
    bf16 = torch.where(
        pos == 0, res0, torch.where(pos == 1, res1, torch.where(pos == 2, res2, res3))
    )

    # Convert to uint16 for .view(torch.bfloat16)
    bf16_uint16 = (bf16 & 0xFFFF).to(torch.uint16)
    bf16_bf16 = bf16_uint16.view(torch.bfloat16)
    
    # Avoid integer overflow by using a float32 multiplier for the exponent scaling
    bf16_new = bf16_bf16 * (2.0 ** 126)

    return bf16_new


def torch_convert(tensor, scale_size=None, Scale=None):
    """
    Decode a 2D uint8 tensor into a 2D bfloat16 tensor by expanding each byte into two bf16 values using a 4-bit (nibble) encoding.
    
    Each input byte holds two 4-bit encoded values (low and high nibble). For each nibble this function derives sign/scale bits, a 3-bit exponent fragment and a 1-bit mantissa fragment, assembles a 16-bit bf16 pattern, and returns the resulting tensor with shape (N, K*2) and dtype torch.bfloat16 on the same device as the input.
    
    Parameters:
        tensor (torch.Tensor): 2D tensor of dtype torch.uint8 and shape (N, K). Each byte contains two encoded 4-bit entries that become two bf16 values.
        scale_size (int, optional): If provided, controls how elements of the optional Scale tensor are indexed. When supplied, per-output-element scaling is applied to the exponent using Scale.
        Scale (torch.Tensor, optional): A 2D tensor used to supply per-element integer scale adjustments to the exponent. If scale_size is provided, the scale used for output element (i, j) is Scale[i][j // scale_size].
    
    Returns:
        torch.Tensor: A new tensor of shape (N, K*2) and dtype torch.bfloat16 containing the decoded bf16 values.
    """

    def _convert(val, pos, scale=None):
        assert val.dtype == torch.uint8
        # val = val.view(torch.int8)
        mask = (1 << 4) - 1
        f4 = ((val >> (pos * 4)) & mask).to(torch.int16)
        s = f4 >> 3
        e_f4 = (f4 & 6) >> 1
        e_f16 = e_f4 + 126
        if scale is not None:
            e_f16 = min(e_f16 + scale, (1 << 8) - 1)
        m_f4 = f4 & 1
        m_f16 = m_f4
        val_f16 = (((e_f16 | (s << 8)) << 7) | (m_f16 << 6)) & 0xFFFF
        lower_16_bits = (val_f16 & 0xFFFF).to(torch.uint16)
        return lower_16_bits.view(torch.bfloat16)

    N = tensor.shape[0]
    K = tensor.shape[1]
    new_tensor = torch.empty(N, K * 2, dtype=torch.bfloat16, device=tensor.device)
    for i in range(new_tensor.shape[0]):
        for j in range(new_tensor.shape[1]):
            if scale_size is not None:
                new_tensor[i][j] = _convert(tensor[i][j // 2], j % 2, Scale[i][j // scale_size])
            else:
                new_tensor[i][j] = _convert(tensor[i][j // 2], j % 2)
    return new_tensor


def print_bit(name, val):
    """
    Print the 32-bit binary representation of a CPU scalar extracted from a PyTorch tensor.
    
    Converts `val` to CPU, reads its Python scalar with `.item()`, formats it as a 32-bit binary string, and prints it prefixed by `name`.
    
    Parameters:
        name (str): Label printed before the binary representation.
        val (torch.Tensor): A scalar PyTorch tensor (numeric) whose 32-bit binary representation will be shown.
    """
    val_cpu = val.cpu().item()
    binary_repr = f'{val_cpu:032b}'
    print(name, binary_repr)


if __name__ == "__main__":
    import time

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, K = 256, 256  # Small shape for test, must be even K
    tensor = torch.randint(0, 256, (N, K), dtype=torch.uint8, device=device)

    # Time torch_convert_bit_twiddling
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    out1 = torch_convert_bit_twiddling(tensor)
    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.time()
    print(f"torch_convert_bit_twiddling time: {t1 - t0:.6f} seconds")

    # Time torch_convert_bit_twiddling_parallel
    torch.cuda.synchronize() if device == "cuda" else None
    t2 = time.time()
    out2 = torch_convert_bit_twiddling_parallel(tensor)
    torch.cuda.synchronize() if device == "cuda" else None
    t3 = time.time()
    print(f"torch_convert_bit_twiddling_parallel time: {t3 - t2:.6f} seconds")

    # Use torch.allclose for bfloat16, allow small tolerance
    assert out1.shape == out2.shape, f"Shape mismatch: {out1.shape} vs {out2.shape}"
    if not torch.allclose(out1, out2, atol=1e-2, rtol=1e-2):
        print("out1:", out1)
        print("out2:", out2)
        diff = (out1 - out2).abs()
        print("max diff:", diff.max())
        raise AssertionError("torch_convert_bit_twiddling and torch_convert_bit_twiddling_parallel outputs differ!")

    print("Test passed: torch_convert_bit_twiddling and torch_convert_bit_twiddling_parallel produce the same results.")
