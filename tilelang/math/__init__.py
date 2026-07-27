def next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b
