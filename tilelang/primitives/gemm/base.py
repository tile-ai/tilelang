# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import IntEnum
from dataclasses import dataclass

from typing import Optional
from tvm import tir


class GemmWarpPolicy(IntEnum):
    """
    Enumeration for GEMM Warp Partitioning Policies.
    """

    Square = 0  # Balance warps evenly in a "square" aspect ratio.
    FullRow = 1  # Assign all warps to rows.
    FullCol = 2  # Assign all warps to columns.

    def is_square(self) -> bool:
        """
        Check if the policy is a square partitioning.

        Returns:
            bool: True if the policy is square, False otherwise.
        """
        return self == GemmWarpPolicy.Square

    def is_full_row(self) -> bool:
        """
        Check if the policy is a full row partitioning.

        Returns:
            bool: True if the policy is full row, False otherwise.
        """
        return self == GemmWarpPolicy.FullRow

    def is_full_col(self) -> bool:
        """
        Check if the policy is a full column partitioning.

        Returns:
            bool: True if the policy is full column, False otherwise.
        """
        return self == GemmWarpPolicy.FullCol

    @staticmethod
    def to_prime_factors(num):
        """
        Compute the prime factorization of a given number.

        Args:
            num (int): The number to factorize.

        Returns:
            list: A list of prime factors of the number.
        """
        factors = []
        i = 2
        # Find all prime factors up to the square root of the number.
        while i * i <= num:
            while num % i == 0:  # Check divisibility by `i`.
                factors.append(i)
                num //= i
            i += 1
        # If the remaining number is greater than 1, it's a prime factor.
        if num > 1:
            factors.append(num)
        return factors

    def compute_warp_partition(self, M, N, num_warps):
        """
        Compute the warp partition (m_warp, n_warp) based on the given policy.

        Args:
            M (int): The number of rows in the GEMM workload.
            N (int): The number of columns in the GEMM workload.
            num_warps (int): The total number of warps available.

        Returns:
            tuple: A tuple (m_warp, n_warp) representing the partitioning of warps.

        Raises:
            ValueError: If the policy is invalid or the partitioning fails.
            AssertionError: If M or N is not divisible by the required factor for FullRow or FullCol policies.
        """
        m_warp = 1  # Initial warp count for rows.
        n_warp = 1  # Initial warp count for columns.

        if self.is_full_row():
            # FullRow policy: Allocate all warps to rows.
            m_warp = num_warps
            assert (M % num_warps == 0), "M must be divisible by num_warps for FullRow policy"

        elif self.is_full_col():
            # FullCol policy: Allocate all warps to columns.
            n_warp = num_warps
            assert (N % num_warps == 0), "N must be divisible by num_warps for FullCol policy"

        elif self.is_square():
            # Square policy: Try to balance warps across rows and columns.
            factors = self.to_prime_factors(num_warps)
            for factor in factors:
                M_divisible = (M % (factor * m_warp)) == 0
                N_divisible = (N % (factor * n_warp)) == 0

                # Assign the factor to either m_warp or n_warp based on divisibility and aspect ratio.
                if M_divisible and N_divisible:
                    # Prefer to assign to rows if M is larger, otherwise to columns.
                    if M / m_warp >= N / n_warp:
                        m_warp *= factor
                    else:
                        n_warp *= factor
                elif M_divisible:
                    m_warp *= factor
                elif N_divisible:
                    n_warp *= factor
                else:
                    # If no divisibility condition is met, raise an error.
                    raise ValueError(
                        f"Cannot compute warp partition for shape {M} x {N} with num_warps {num_warps}"
                    )
        else:
            # Raise an error for unknown policies.
            raise ValueError(f"Unknown GemmWarpPolicy: {self}")

        return m_warp, n_warp

    @classmethod
    def from_warp_partition(cls, m_warp: int, n_warp: int) -> 'GemmWarpPolicy':
        """
        Determine the warp policy based on the given warp partitioning.

        Args:
            m_warp (int): Number of warps in the row dimension
            n_warp (int): Number of warps in the column dimension

        Returns:
            GemmWarpPolicy: The corresponding warp policy

        Examples:
            >>> GemmWarpPolicy.from_block_row_cols(4, 1)  # All warps in rows
            GemmWarpPolicy.FullRow
            >>> GemmWarpPolicy.from_block_row_cols(1, 4)  # All warps in columns
            GemmWarpPolicy.FullCol
            >>> GemmWarpPolicy.from_block_row_cols(2, 2)  # Balanced distribution
            GemmWarpPolicy.Square
        """
        if n_warp == 1 and m_warp > 1:
            return cls.FullRow
        elif m_warp == 1 and n_warp > 1:
            return cls.FullCol
        else:
            return cls.Square


@dataclass
class GemmBaseParams:
    # OP Related Config
    A: tir.Buffer
    B: tir.Buffer
    C: tir.Buffer

    transpose_A: bool = False
    transpose_B: bool = False
    block_row_warps: Optional[int] = None
    block_col_warps: Optional[int] = None
    warp_row_tiles: Optional[int] = None
    warp_col_tiles: Optional[int] = None
    chunk: Optional[int] = None
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    k_pack: int = 1

    def get_warp_size(self) -> int:
        # must rewrite to 64 if the target
        # is cdna mfma
        return 32

    def params_as_dict(self):
        return {
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "transpose_A": self.transpose_A,
            "transpose_B": self.transpose_B,
            "block_row_warps": self.block_row_warps,
            "block_col_warps": self.block_col_warps,
            "warp_row_tiles": self.warp_row_tiles,
            "warp_col_tiles": self.warp_col_tiles,
            "chunk": self.chunk,
            "policy": self.policy,
            "k_pack": self.k_pack,
        }

    def infer_block_partition(self, threads: Optional[int]) -> None:
        """
        Infer and set block partition parameters (e.g., block_row_warps,
        block_col_warps, warp_row_tiles, warp_col_tiles, chunk) based on the
        shape of A and B. If these parameters are not already specified, the
        method will attempt to infer them automatically based on the given
        `threads`.

        Parameters
        ----------
        threads : Optional[int]
            The total number of threads in a block. Must be provided
            if any block partition parameter is not already set.

        Raises
        ------
        AssertionError
            If `threads` is None but any block partition parameter is missing,
            or if A and B have inconsistent shapes for GEMM.
        """

        warp_size = self.get_warp_size()
        A, B = self.A, self.B
        transpose_A, transpose_B = self.transpose_A, self.transpose_B
        block_row_warps, block_col_warps = (
            self.block_row_warps,
            self.block_col_warps,
        )
        warp_row_tiles, warp_col_tiles = (
            self.warp_row_tiles,
            self.warp_col_tiles,
        )
        policy = self.policy

        # The field `chunk` is not declared in GemmBaseParams by default.
        # We infer it based on the K dimension of matrices.
        # Initialize chunk from `self` if it exists; otherwise we infer it.
        chunk = getattr(self, "chunk", None)

        # Determine whether block partition parameters need to be inferred
        require_infer = (
            block_row_warps is None or block_col_warps is None or warp_row_tiles is None or
            warp_col_tiles is None or chunk is None)

        A_shape, B_shape = A.shape, B.shape

        if require_infer:
            assert (threads is not None), "threads must be provided for auto inference"
            # Auto-inference only supports 2D matrix multiplication
            assert (
                len(A_shape) == 2 and len(B_shape) == 2
            ), f"Only support 2D matrix multiplication, got {len(A_shape)}D and {len(B_shape)}D"

            # Analyze A/B shapes
            AM = A_shape[1] if transpose_A else A_shape[0]  # M dimension
            BN = B_shape[0] if transpose_B else B_shape[1]  # N dimension
            AK = A_shape[0] if transpose_A else A_shape[1]  # K dimension
            BK = B_shape[1] if transpose_B else B_shape[0]  # K dimension
            assert AK == BK, "A and B shape mismatch"

            block_M = int(AM)
            block_N = int(BN)
            num_warps = threads // warp_size

            # Infer block partition using a user-specified policy
            block_row_warps, block_col_warps = policy.compute_warp_partition(
                block_M, block_N, num_warps)
            warp_row_tiles = block_M // block_row_warps
            warp_col_tiles = block_N // block_col_warps
            chunk = int(AK)

        # rewrite the values
        self.block_row_warps = block_row_warps
        self.block_col_warps = block_col_warps
        self.warp_row_tiles = warp_row_tiles
        self.warp_col_tiles = warp_col_tiles
        self.chunk = chunk

    @property
    def class_attributes(self):
        return self.params_as_dict()

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        fields = self.class_attributes
        field_str = ", ".join(f"{key}={value!r}" for key, value in fields.items())
        return f"{cls_name}({field_str})"
