import torch


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
