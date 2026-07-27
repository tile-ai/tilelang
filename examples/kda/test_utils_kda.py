import torch

RCP_LN2 = 1.4426950216


def print_red_warning(message):
    print(f"\033[31mWARNING: {message}\033[0m")


def calc_sim(x, y, name="tensor"):
    x, y = x.data.double(), y.data.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        print_red_warning(f"{name} all zero")
        return 1
    sim = 2 * (x * y).sum() / denominator
    return sim


def assert_similar(x, y, eps=1e-8, name="tensor", data="", raise_assert=True):
    x_mask = torch.isfinite(x)
    y_mask = torch.isfinite(y)
    if not torch.all(x_mask == y_mask):
        print_red_warning(f"{name} Error: isfinite mask mismatch")
        if raise_assert:
            raise AssertionError
    if not torch.isclose(x.masked_fill(x_mask, 0), y.masked_fill(y_mask, 0), rtol=0, atol=0, equal_nan=True).all():
        print_red_warning(f"{name} Error: nonfinite value mismatch")
        if raise_assert:
            raise AssertionError
    x = x.masked_fill(~x_mask, 0)
    y = y.masked_fill(~y_mask, 0)
    sim = calc_sim(x, y, name)
    diff = 1.0 - sim
    if not (0 <= diff <= eps):
        print_red_warning(f"{name} Error: {diff}")
        if raise_assert:
            raise AssertionError
    else:
        print(f"{name} {data} passed")


def compare_tensors(name, x, y, atol=1e-5, rtol=1e-5):
    import numpy as np
    import torch

    diff = (x - y).abs()

    # ========= Max Absolute Error =========
    max_abs_err = diff.max().item()
    abs_flat_idx = diff.argmax()
    abs_idx = list(np.unravel_index(abs_flat_idx.cpu().numpy(), diff.shape))

    # ========= Relative Error (NaN-safe) =========
    denom = y.abs()
    rel = torch.zeros_like(diff)
    mask = denom > 0
    rel[mask] = diff[mask] / denom[mask]

    max_rel_err = rel.max().item()
    rel_flat_idx = rel.argmax()
    rel_idx = list(np.unravel_index(rel_flat_idx.cpu().numpy(), rel.shape))

    # ========= Cross Error =========
    abs_pos_rel_err = rel[tuple(abs_idx)].item()
    rel_pos_abs_err = diff[tuple(rel_idx)].item()

    # ========= Print =========
    print(f"========== Compare: {name} ==========")

    print(f"Max absolute error : {max_abs_err:.6e}")
    print(f"  at index         : {abs_idx}")
    print(f"  x[{abs_idx}] = {x[tuple(abs_idx)].item():.6e}")
    print(f"  y[{abs_idx}] = {y[tuple(abs_idx)].item():.6e}")
    print(f"  relative error   : {abs_pos_rel_err:.6e}")

    print(f"\nMax relative error : {max_rel_err:.6e}")
    print(f"  at index         : {rel_idx}")
    print(f"  x[{rel_idx}] = {x[tuple(rel_idx)].item():.6e}")
    print(f"  y[{rel_idx}] = {y[tuple(rel_idx)].item():.6e}")
    print(f"  absolute error   : {rel_pos_abs_err:.6e}")

    print("=====================================\n")


def do_bench(fn, *args, warmup=20, rep=10, **kwargs):
    """
    Do benchmark for a function.
    """
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    for _ in range(warmup):
        fn(*args, **kwargs)

    torch.cuda.synchronize()
    for i in range(rep):
        start_event[i].record()
        fn(*args, **kwargs)
        end_event[i].record()
    torch.cuda.synchronize()

    # Record clocks
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
    )
    print(times)
    return times.mean().item()
