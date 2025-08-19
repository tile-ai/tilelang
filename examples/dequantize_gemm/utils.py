import torch


def torch_convert_bit_twiddling(tensor):
    """
    Convert a 2-D uint8 tensor into a bfloat16 tensor by decoding pairs of input bytes with a bit-twiddling scheme.
    
    This function expects `tensor` to be a 2-D torch.Tensor of dtype `torch.uint8`. Each output element is produced by combining two input bytes and extracting a bf16-like 16-bit pattern according to one of four positional bit layouts (pos 0..3). The result is scaled by 2**126 to adjust the exponent bias and returned as dtype `torch.bfloat16`.
    
    Parameters:
        tensor (torch.Tensor): 2-D input tensor with dtype `torch.uint8`. Shape (N, K).
    
    Returns:
        torch.Tensor: New tensor of dtype `torch.bfloat16` with shape (N, K*2), where each input column pair produces two bf16 output columns.
    
    Raises:
        AssertionError: If any byte inputs used for a conversion are not dtype `torch.uint8`.
    """

    def _convert(val0, val1, pos) -> torch.bfloat16:
        assert val0.dtype == torch.uint8
        assert val1.dtype == torch.uint8
        val0 = val0.view(torch.uint8)
        val1 = val1.view(torch.uint8)
        val_concat = (val0.item() << 8) | val1.item()
        mask = 0b1000000111000000
        if pos == 0:
            bf16 = val_concat & mask
        elif pos == 1:
            bf16 = (val_concat << 3) & mask
        elif pos == 2:
            bf16 = (val_concat << 6) & mask
        elif pos == 3:
            mask1 = 0b1000000000000000
            mask2 = 0b0000000110000000
            mask3 = 0b0000000001000000
            bf16 = ((val_concat << 1) & mask1) | ((val_concat >> 3) & mask2) | (
                (val_concat >> 7) & mask3)
        bf16_new = torch.tensor([bf16], dtype=torch.uint16, device=val0.device).view(torch.bfloat16)
        # Add bias for change from fp4 to bf16
        bf16_new = bf16_new.item() * (2**126)
        return bf16_new

    N = tensor.shape[0]
    K = tensor.shape[1]
    new_tensor = torch.empty(N, K * 2, dtype=torch.bfloat16, device=tensor.device)
    for i in range(new_tensor.shape[0]):
        for j in range(new_tensor.shape[1]):
            new_tensor[i][j] = _convert(tensor[i][j // 4 * 2], tensor[i][j // 4 * 2 + 1], j % 4)
    return new_tensor


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
