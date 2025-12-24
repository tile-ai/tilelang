"""Reproduce: scalar parameter type mismatch (int/bool)."""

from common import build_scalar_check_kernel


def main():
    fn = build_scalar_check_kernel(target="cuda")

    # Wrong types
    fn(1.0, True)  # x should be int -> Expect arg[0] to be int
    fn(1, 2.5)  # flag should be bool -> Expect arg[1] to be boolean


if __name__ == "__main__":
    main()
